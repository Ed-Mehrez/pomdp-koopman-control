"""Formal hybrid transfer benchmark with anti-tautology audit.

Frozen controller spec from hybrid_residual_controller.py (commit 735b3fa).
This script adds diagnostic baselines and reporting only — no changes to the
controller, features, kernel, bandwidth, stencil, prior, or env.

Controllers evaluated on held-out cells:
  1. risk_neutral_optimal
  2. bbg_numerical
  3. bbg_global_best_stencil  (diagnostic: single best perturbation from training data)
  4. hybrid_bbg_residual      (the candidate)
  5. hybrid_permuted_labels   (diagnostic: shuffled-reward null)

Pre-registered outcome buckets:
  A — State-dependent positive  (hybrid > BBG and hybrid > global_best_stencil)
  B — Global-tightening positive (hybrid > BBG but hybrid ~ global_best_stencil)
  C — Pilot false positive      (hybrid ~ BBG or permuted null is similar)

Usage:
    python finance/experiments/option_mm_hybrid_formal.py [--pilot]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.bbg_solver import make_bbg_numerical  # noqa: E402
from applications.option_mm.beliefs import EWMAVarianceFilter  # noqa: E402
from applications.option_mm.controllers import make_risk_neutral_optimal  # noqa: E402
from applications.option_mm.env import (  # noqa: E402
    FillModelSpec,
    HestonParams,
    OptionMMAction,
    OptionMMState,
    OptionMarketMakingEnv,
)
from applications.option_mm.hybrid_residual_controller import (  # noqa: E402
    BBGQuoteLookup,
    HybridTrainingBuffer,
    _apply_perturbation,
    _crra_utility_arrow_pratt,
    collect_hybrid_training_data_multi_cell,
    make_hybrid_residual_controller,
)
from applications.option_mm.local_kernel_controller import (  # noqa: E402
    KernelRewardModel,
    extract_state_features,
    median_bandwidth,
)
from applications.option_mm.metrics import (  # noqa: E402
    EpisodeSummary,
    PosteriorSummary,
    UtilitySpec,
    aggregate_episode_summaries,
    crra_utility,
    paired_ce_posterior,
    summarize_episode,
)


# ---------------------------------------------------------------------------
# Cells (frozen from hybrid_transfer.py)
# ---------------------------------------------------------------------------


def make_training_cells() -> list[HestonParams]:
    cells = []
    for kappa in (1.5, 3.0):
        for xi in (0.3, 0.7):
            for rho in (-0.5, -0.9):
                cells.append(HestonParams(kappa=kappa, xi=xi, rho=rho))
    return cells


def make_test_cells() -> list[HestonParams]:
    return [
        HestonParams(kappa=2.0, xi=0.5, rho=-0.7),
        HestonParams(kappa=2.25, xi=0.4, rho=-0.6),
        HestonParams(kappa=1.75, xi=0.6, rho=-0.8),
    ]


# ---------------------------------------------------------------------------
# Config (frozen defaults match hybrid_transfer.py)
# ---------------------------------------------------------------------------


# Frozen stencil from hybrid_residual_controller.py default args.
DEFAULT_STENCIL_WIDTH = (-0.03, -0.015, 0.0, 0.015, 0.03)
DEFAULT_STENCIL_SKEW = (-0.015, -0.005, 0.0, 0.005, 0.015)


@dataclass(frozen=True)
class FormalConfig:
    seed_sequence_entropy: int = 20260410
    n_train_episodes_per_cell: int = 100
    n_test_episodes_per_cell: int = 50
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    gamma_ce: float = 2.0
    max_inventory: int = 10
    ewma_half_life_days: float = 5.0
    noise_width: float = 0.02
    noise_skew: float = 0.01
    exploration_rng_seed: int = 42
    ridge_alpha: float = 1e-3
    max_training_samples: int = 10_000


# ---------------------------------------------------------------------------
# 1.1 Global-best-stencil baseline
# ---------------------------------------------------------------------------


def compute_global_best_perturbation(
    buffer: HybridTrainingBuffer,
) -> tuple[float, float, float]:
    """Find the stencil point with highest mean training reward.

    Returns (best_delta_width, best_delta_skew, best_mean_reward).
    """
    _, perturbations, rewards = buffer.as_arrays()
    # Group by unique perturbation
    rounded = np.round(perturbations, decimals=6)
    unique_keys = {}  # (dw, ds) -> list of rewards
    for i in range(rounded.shape[0]):
        key = (float(rounded[i, 0]), float(rounded[i, 1]))
        unique_keys.setdefault(key, []).append(rewards[i])

    best_key = None
    best_mean = -np.inf
    for key, rews in unique_keys.items():
        m = float(np.mean(rews))
        if m > best_mean:
            best_mean = m
            best_key = key

    return best_key[0], best_key[1], best_mean


def make_global_best_stencil_controller(
    env: OptionMarketMakingEnv,
    initial_state: OptionMMState,
    delta_width: float,
    delta_skew: float,
    gamma_ce: float = 2.0,
    max_inventory: int = 10,
):
    """BBG prior + a single fixed perturbation applied at every step."""
    gamma_local = _crra_utility_arrow_pratt(gamma_ce)(initial_state.wealth)
    bbg = BBGQuoteLookup(
        env, initial_state, gamma=gamma_local, max_inventory=max_inventory,
    )

    def controller(state: OptionMMState, history=None) -> OptionMMAction:
        del history
        bd, ad = bbg.distances(state)
        return _apply_perturbation(state, bd, ad, delta_width, delta_skew)

    return controller


# ---------------------------------------------------------------------------
# 1.2 Instrumented hybrid controller (records perturbation choices)
# ---------------------------------------------------------------------------


def make_instrumented_hybrid_controller(
    env: OptionMarketMakingEnv,
    model: KernelRewardModel,
    initial_state: OptionMMState,
    *,
    gamma_ce: float = 2.0,
    max_inventory: int = 10,
    ewma_half_life_days: float = 5.0,
    stencil_width: tuple[float, ...] = DEFAULT_STENCIL_WIDTH,
    stencil_skew: tuple[float, ...] = DEFAULT_STENCIL_SKEW,
):
    """Like make_hybrid_residual_controller but records every perturbation choice."""
    gamma_local = _crra_utility_arrow_pratt(gamma_ce)(initial_state.wealth)
    bbg = BBGQuoteLookup(
        env, initial_state, gamma=gamma_local, max_inventory=max_inventory,
    )
    ewma = EWMAVarianceFilter(half_life_days=ewma_half_life_days)
    ewma.reset(initial_variance=initial_state.variance, initial_spot=initial_state.spot)

    stencil = np.array(
        [[w, s] for w in stencil_width for s in stencil_skew], dtype=float,
    )
    n_candidates = stencil.shape[0]
    choices: list[tuple[float, float]] = []

    def controller(state: OptionMMState, history=None) -> OptionMMAction:
        del history
        if state.step_index > 0:
            ewma.update(state.spot)
        v_hat = ewma.variance
        feat = extract_state_features(state, env, v_hat)
        bd, ad = bbg.distances(state)

        feat_tiled = np.tile(feat, (n_candidates, 1))
        pred = model.predict(feat_tiled, stencil)
        best_idx = int(np.argmax(pred))
        dw = float(stencil[best_idx, 0])
        ds = float(stencil[best_idx, 1])
        choices.append((round(dw, 6), round(ds, 6)))
        return _apply_perturbation(state, bd, ad, dw, ds)

    return controller, choices


# ---------------------------------------------------------------------------
# Episode runner (extended for all strategies)
# ---------------------------------------------------------------------------


def run_episode(
    strategy_name: str,
    seed: int,
    heston: HestonParams,
    config: FormalConfig,
    utility: UtilitySpec,
    *,
    residual_model: KernelRewardModel | None = None,
    global_best_dw: float = 0.0,
    global_best_ds: float = 0.0,
    choices_accumulator: list | None = None,
) -> tuple[list, list]:
    env = OptionMarketMakingEnv(
        heston=heston,
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()

    local_choices = None
    if strategy_name == "risk_neutral_optimal":
        ctrl = make_risk_neutral_optimal(env)
    elif strategy_name == "bbg_numerical":
        gamma = utility.arrow_pratt(state.wealth)
        ctrl = make_bbg_numerical(env, state, gamma=gamma, max_inventory=config.max_inventory)
    elif strategy_name == "bbg_global_best_stencil":
        ctrl = make_global_best_stencil_controller(
            env, state, global_best_dw, global_best_ds,
            gamma_ce=config.gamma_ce, max_inventory=config.max_inventory,
        )
    elif strategy_name in ("hybrid_bbg_residual", "hybrid_permuted_labels"):
        assert residual_model is not None
        if choices_accumulator is not None:
            ctrl, local_choices = make_instrumented_hybrid_controller(
                env, residual_model, state,
                gamma_ce=config.gamma_ce, max_inventory=config.max_inventory,
                ewma_half_life_days=config.ewma_half_life_days,
            )
        else:
            ctrl = make_hybrid_residual_controller(
                env, residual_model, state,
                gamma_ce=config.gamma_ce, max_inventory=config.max_inventory,
                ewma_half_life_days=config.ewma_half_life_days,
            )
    else:
        raise ValueError(f"unknown strategy: {strategy_name}")

    states = [state]
    infos = []
    while not state.done:
        action = ctrl(state)
        state, _, _, info = env.step(action)
        states.append(state)
        infos.append(info)

    if local_choices is not None and choices_accumulator is not None:
        choices_accumulator.extend(local_choices)

    return states, infos


def run_strategy_on_cell(
    strategy_name: str,
    seeds: list[int],
    heston: HestonParams,
    config: FormalConfig,
    utility: UtilitySpec,
    **kwargs,
) -> list[EpisodeSummary]:
    summaries = []
    for seed in seeds:
        states, infos = run_episode(
            strategy_name, seed, heston, config, utility, **kwargs,
        )
        summaries.append(summarize_episode(
            states=states, infos=infos, inventory_limit=config.max_inventory,
        ))
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def paired_post(
    a: list[EpisodeSummary],
    b: list[EpisodeSummary],
    utility: UtilitySpec,
) -> PosteriorSummary:
    wa = [s.terminal_wealth for s in a]
    wb = [s.terminal_wealth for s in b]
    return paired_ce_posterior(wa, wb, utility=utility, method="delta")


def cell_label(h: HestonParams) -> str:
    return f"kappa={h.kappa}, xi={h.xi}, rho={h.rho}"


def print_perturbation_usage(choices: list[tuple[float, float]], label: str) -> None:
    """Print perturbation-usage diagnostic for the hybrid controller."""
    n = len(choices)
    if n == 0:
        print(f"  {label}: no choices recorded")
        return
    dws = [c[0] for c in choices]
    dss = [c[1] for c in choices]
    zero_frac = sum(1 for c in choices if c == (0.0, 0.0)) / n
    counts = Counter(choices)
    top5 = counts.most_common(5)
    print(f"  {label} perturbation usage (N={n}):")
    print(f"    mean Δwidth = {np.mean(dws):.6f}")
    print(f"    mean Δskew  = {np.mean(dss):.6f}")
    print(f"    fraction at (0,0) = {zero_frac:.3f}")
    print(f"    top 5 stencil points:")
    for pt, cnt in top5:
        print(f"      ({pt[0]:+.4f}, {pt[1]:+.4f})  freq={cnt}/{n} ({cnt/n:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formal hybrid transfer + audit")
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot (20 train, 20 test per cell)")
    args = parser.parse_args(argv)

    config = FormalConfig()
    if args.pilot:
        config = FormalConfig(
            n_train_episodes_per_cell=20,
            n_test_episodes_per_cell=20,
        )

    utility = crra_utility(config.gamma_ce)
    train_cells = make_training_cells()
    test_cells = make_test_cells()

    n_train = config.n_train_episodes_per_cell
    n_test = config.n_test_episodes_per_cell
    ss = np.random.SeedSequence(config.seed_sequence_entropy)
    child_seeds = ss.spawn(n_train + n_test)
    all_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
    train_seeds = all_ints[:n_train]
    test_seeds = all_ints[n_train:]

    out_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        out_lines.append(msg)

    log("=" * 70)
    log("  FORMAL: Hybrid BBG + Residual — Anti-Tautology Audit")
    log("=" * 70)
    log(f"\nTraining cells: {len(train_cells)} (2x2x2 grid)")
    log(f"Test cells:     {len(test_cells)} (interpolated)")
    log(f"Train eps/cell: {n_train}  |  Test eps/cell: {n_test}")
    log(f"Utility: CRRA(gamma={config.gamma_ce})")

    # ===== Training =====
    t0 = time.time()
    log("\n--- Training ---")
    buffer = collect_hybrid_training_data_multi_cell(
        cell_params=train_cells,
        seeds_per_cell=tuple(train_seeds),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        gamma_ce=config.gamma_ce,
        max_inventory=config.max_inventory,
        ewma_half_life_days=config.ewma_half_life_days,
        noise_width=config.noise_width,
        noise_skew=config.noise_skew,
        exploration_rng_seed=config.exploration_rng_seed,
    )
    features, perturbations, rewards = buffer.as_arrays()
    if features.shape[0] > config.max_training_samples:
        rng = np.random.default_rng(config.exploration_rng_seed + 1)
        idx = rng.choice(features.shape[0], size=config.max_training_samples, replace=False)
        features, perturbations, rewards = features[idx], perturbations[idx], rewards[idx]

    bw = median_bandwidth(features, perturbations)
    model = KernelRewardModel(bandwidth=bw, ridge_alpha=config.ridge_alpha)
    model.fit(features, perturbations, rewards)
    log(f"  Data: {buffer.size} tuples, fitted on {features.shape[0]} (bw={bw:.3f})")
    log(f"  Time: {time.time() - t0:.1f}s")

    # 1.1 Global-best-stencil
    log("\n--- 1.1 Global-Best-Stencil Baseline ---")
    gbw, gbs, gbr = compute_global_best_perturbation(buffer)
    log(f"  Best perturbation from training: Δwidth={gbw:+.4f}, Δskew={gbs:+.4f}")
    log(f"  Mean training reward at that point: {gbr:.4f}")

    # 1.3 Permuted-label null
    log("\n--- 1.3 Permuted-Label Null ---")
    perm_rng = np.random.default_rng(config.exploration_rng_seed + 99)
    rewards_perm = rewards.copy()
    perm_rng.shuffle(rewards_perm)
    model_perm = KernelRewardModel(bandwidth=bw, ridge_alpha=config.ridge_alpha)
    model_perm.fit(features, perturbations, rewards_perm)
    log("  Permuted model fitted (same bandwidth/ridge, shuffled rewards)")

    # ===== Evaluation on held-out cells =====
    log("\n" + "=" * 70)
    log("  HELD-OUT CELLS — Formal Evaluation")
    log("=" * 70)

    all_strategies = [
        "risk_neutral_optimal", "bbg_numerical",
        "bbg_global_best_stencil", "hybrid_bbg_residual",
        "hybrid_permuted_labels",
    ]

    for cell in test_cells:
        log(f"\n  --- TEST CELL: {cell_label(cell)} ---")
        cell_results: dict[str, list[EpisodeSummary]] = {}
        cell_choices: list[tuple[float, float]] = []

        for strat in all_strategies:
            kwargs = {}
            if strat == "bbg_global_best_stencil":
                kwargs["global_best_dw"] = gbw
                kwargs["global_best_ds"] = gbs
            elif strat == "hybrid_bbg_residual":
                kwargs["residual_model"] = model
                kwargs["choices_accumulator"] = cell_choices
            elif strat == "hybrid_permuted_labels":
                kwargs["residual_model"] = model_perm

            cell_results[strat] = run_strategy_on_cell(
                strat, test_seeds, cell, config, utility, **kwargs,
            )

        # Absolute CE table
        log(f"\n  {'Strategy':<35s} {'CE':>12s} {'Spread Cap':>12s}")
        log("  " + "-" * 61)
        for name, summaries in cell_results.items():
            agg = aggregate_episode_summaries(summaries)
            u_vals = utility.u(np.array([s.terminal_wealth for s in summaries]))
            ce = utility.ce(float(np.mean(u_vals)))
            log(f"  {name:<35s} {ce:>12.3f} {agg.gross_spread_capture_mean:>12.3f}")

        # Required paired posteriors
        bbg = cell_results["bbg_numerical"]
        contrasts = [
            ("hybrid_bbg_residual", "bbg_numerical"),
            ("bbg_global_best_stencil", "bbg_numerical"),
            ("hybrid_bbg_residual", "bbg_global_best_stencil"),
            ("hybrid_permuted_labels", "bbg_numerical"),
        ]
        log(f"\n  {'Contrast':<55s} {'Mean':>8s} {'sd_post':>8s} {'P(>0)':>8s}")
        log("  " + "-" * 81)
        for na, nb in contrasts:
            p = paired_post(cell_results[na], cell_results[nb], utility)
            label = f"{na} - {nb}"
            log(f"  {label:<55s} {p.mean:>8.3f} {p.sd_post:>8.3f} {p.p_positive:>8.5f}")

        # 1.2 Perturbation-usage diagnostic
        log("")
        print_perturbation_usage(cell_choices, cell_label(cell))
        # Also capture in out_lines
        n_ch = len(cell_choices)
        if n_ch > 0:
            dws = [c[0] for c in cell_choices]
            dss = [c[1] for c in cell_choices]
            zero_frac = sum(1 for c in cell_choices if c == (0.0, 0.0)) / n_ch
            counts = Counter(cell_choices)
            top5 = counts.most_common(5)
            out_lines.append(f"  {cell_label(cell)} perturbation usage (N={n_ch}):")
            out_lines.append(f"    mean Δwidth = {np.mean(dws):.6f}")
            out_lines.append(f"    mean Δskew  = {np.mean(dss):.6f}")
            out_lines.append(f"    fraction at (0,0) = {zero_frac:.3f}")
            out_lines.append(f"    top 5 stencil points:")
            for pt, cnt in top5:
                out_lines.append(f"      ({pt[0]:+.4f}, {pt[1]:+.4f})  freq={cnt}/{n_ch} ({cnt/n_ch:.1%})")

    # ===== Save =====
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    results_file = results_dir / f"option_mm_hybrid_{suffix}_{date.today()}.txt"
    with open(results_file, "w") as f:
        f.write("\n".join(out_lines))
    log(f"\nResults saved to {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
