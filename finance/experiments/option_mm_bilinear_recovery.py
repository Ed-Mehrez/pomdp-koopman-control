"""Prior-free bilinear / CQ-KRONIC recovery benchmark.

Can a data-driven local bilinear controller, with NO BBG prior inside the
policy, recover or narrow the gap to the BBG benchmark?

Controllers:
  1. risk_neutral_optimal
  2. bbg_numerical (benchmark only, NOT inside the candidate)
  3. linear_inventory_skew (first-order diagnostic)
  4. global_fixed_action (anti-tautology baseline)
  5. local_bilinear_sdre (the candidate — prior-free)
  6. permuted_null (shuffled-target diagnostic)

Primary gates:
  Q1: local_bilinear > linear_inventory_skew (P >= 0.95)
  Q2: gap_closure = (bilinear - linear) / (bbg - linear)
  Q3: local_bilinear - bbg (descriptive)

Usage:
    python finance/experiments/option_mm_bilinear_recovery.py [--pilot]
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
from applications.option_mm.controllers import (  # noqa: E402
    make_linear_inventory_skew,
    make_risk_neutral_optimal,
)
from applications.option_mm.env import (  # noqa: E402
    FillModelSpec,
    HestonParams,
    OptionMMAction,
    OptionMMState,
    OptionMarketMakingEnv,
)
from applications.option_mm.inventory_variance import oracle_heston_estimator  # noqa: E402
from applications.option_mm.local_bilinear_controller import (  # noqa: E402
    LocalBilinearModel,
    StateNorm,
    collect_bilinear_training_data,
    make_instrumented_bilinear_controller,
    make_local_bilinear_controller,
    normalize_state,
    train_bilinear_model,
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
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BilinearConfig:
    seed_sequence_entropy: int = 20260410
    n_train_episodes: int = 200
    n_test_episodes: int = 50
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    gamma_ce: float = 2.0
    max_inventory: int = 10
    width_range: tuple[float, float] = (0.10, 0.30)
    skew_range: tuple[float, float] = (-0.05, 0.05)
    exploration_rng_seed: int = 42
    ridge: float = 1e-3
    max_training_samples: int = 8_000
    lambda_q: float = 0.0


# ---------------------------------------------------------------------------
# Global fixed-action baseline
# ---------------------------------------------------------------------------


def compute_global_best_fixed_action(
    buffer,
    width_range: tuple[float, float],
    skew_range: tuple[float, float],
    n_bins: int = 10,
) -> tuple[float, float, float]:
    """Find the best constant (w, s) from binned training data."""
    U = np.array(buffer.u_list)
    dW = np.array(buffer.dw_list)
    w_edges = np.linspace(width_range[0], width_range[1], n_bins + 1)
    s_edges = np.linspace(skew_range[0], skew_range[1], n_bins + 1)

    best_w, best_s, best_mean = 0.20, 0.0, -np.inf
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (U[:, 0] >= w_edges[i]) & (U[:, 0] < w_edges[i + 1])
                & (U[:, 1] >= s_edges[j]) & (U[:, 1] < s_edges[j + 1])
            )
            if mask.sum() < 3:
                continue
            m = float(np.mean(dW[mask]))
            if m > best_mean:
                best_mean = m
                best_w = float((w_edges[i] + w_edges[i + 1]) / 2)
                best_s = float((s_edges[j] + s_edges[j + 1]) / 2)
    return best_w, best_s, best_mean


def make_global_fixed_action_controller(
    w: float,
    s: float,
) -> ...:
    """Constant (width, skew) controller — no state dependence."""
    def controller(state: OptionMMState, history=None) -> OptionMMAction:
        del history
        bid_dist = w + s
        ask_dist = w - s
        return OptionMMAction(
            bid_price=max(state.option_mid - bid_dist, 0.0),
            ask_price=state.option_mid + max(ask_dist, 0.0),
            hedge_trade=-state.net_delta,
        )
    return controller


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    strategy_name: str,
    seed: int,
    config: BilinearConfig,
    utility: UtilitySpec,
    *,
    bilinear_model: LocalBilinearModel | None = None,
    fixed_w: float = 0.20,
    fixed_s: float = 0.0,
    choices_accumulator: list | None = None,
) -> tuple[list, list]:
    env = OptionMarketMakingEnv(
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
    elif strategy_name == "linear_inventory_skew":
        estimator = oracle_heston_estimator(env)
        ctrl = make_linear_inventory_skew(env, estimator, utility)
    elif strategy_name == "global_fixed_action":
        ctrl = make_global_fixed_action_controller(fixed_w, fixed_s)
    elif strategy_name in ("local_bilinear_sdre", "permuted_null"):
        assert bilinear_model is not None
        if choices_accumulator is not None:
            ctrl, local_choices = make_instrumented_bilinear_controller(
                env, bilinear_model, state,
                gamma_ce=config.gamma_ce, lambda_q=config.lambda_q,
                width_range=config.width_range, skew_range=config.skew_range,
            )
        else:
            ctrl = make_local_bilinear_controller(
                env, bilinear_model, state,
                gamma_ce=config.gamma_ce, lambda_q=config.lambda_q,
                width_range=config.width_range, skew_range=config.skew_range,
            )
    else:
        raise ValueError(f"unknown: {strategy_name}")

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


def run_strategy(
    name: str,
    seeds: list[int],
    config: BilinearConfig,
    utility: UtilitySpec,
    **kwargs,
) -> list[EpisodeSummary]:
    summaries = []
    for seed in seeds:
        states, infos = run_episode(name, seed, config, utility, **kwargs)
        summaries.append(summarize_episode(
            states=states, infos=infos, inventory_limit=config.max_inventory,
        ))
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def paired_post(a, b, utility):
    wa = [s.terminal_wealth for s in a]
    wb = [s.terminal_wealth for s in b]
    return paired_ce_posterior(wa, wb, utility=utility, method="delta")


def ce_from_summaries(summaries, utility):
    w = np.array([s.terminal_wealth for s in summaries])
    return utility.ce(float(np.mean(utility.u(w))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bilinear recovery benchmark")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)

    config = BilinearConfig()
    if args.pilot:
        config = BilinearConfig(n_train_episodes=50, n_test_episodes=30)

    utility = crra_utility(config.gamma_ce)
    n_train = config.n_train_episodes
    n_test = config.n_test_episodes

    ss = np.random.SeedSequence(config.seed_sequence_entropy)
    child_seeds = ss.spawn(n_train + n_test)
    all_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
    train_seeds = all_ints[:n_train]
    test_seeds = all_ints[n_train:]

    out: list[str] = []

    def log(msg=""):
        print(msg)
        out.append(msg)

    log("=" * 70)
    log("  BILINEAR RECOVERY: Prior-Free CQ-KRONIC Controller")
    log("=" * 70)
    log(f"Train: {n_train} eps  |  Test: {n_test} eps")
    log(f"Utility: CRRA(gamma={config.gamma_ce})")

    # ===== Training =====
    t0 = time.time()
    log("\n--- Training ---")
    buffer = collect_bilinear_training_data(
        seeds=train_seeds,
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        width_range=config.width_range,
        skew_range=config.skew_range,
        exploration_rng_seed=config.exploration_rng_seed,
    )
    X = np.array(buffer.x_list)
    U = np.array(buffer.u_list)
    X_next = np.array(buffer.x_next_list)
    dW = np.array(buffer.dw_list)
    Q_next = np.array(buffer.q_next_list)

    if X.shape[0] > config.max_training_samples:
        rng = np.random.default_rng(config.exploration_rng_seed + 1)
        idx = rng.choice(X.shape[0], size=config.max_training_samples, replace=False)
        X, U, X_next, dW, Q_next = X[idx], U[idx], X_next[idx], dW[idx], Q_next[idx]

    # Bandwidth
    n_sub = min(X.shape[0], 2000)
    idx_sub = np.random.default_rng(0).choice(X.shape[0], size=n_sub, replace=False)
    X_sub = X[idx_sub]
    sq_d = (
        np.sum(X_sub**2, axis=1, keepdims=True)
        + np.sum(X_sub**2, axis=1, keepdims=True).T
        - 2.0 * X_sub @ X_sub.T
    )
    dists = np.sqrt(np.maximum(sq_d[np.triu_indices(n_sub, k=1)], 0.0))
    bw = max(float(np.median(dists)), 0.1)

    model = LocalBilinearModel(X=X, U=U, X_next=X_next, dW=dW, Q_next=Q_next,
                                bandwidth=bw, ridge=config.ridge)
    log(f"  Data: {buffer.size} tuples, fitted on {X.shape[0]} (bw={bw:.3f})")

    # Global fixed-action baseline
    gw, gs, gm = compute_global_best_fixed_action(
        buffer, config.width_range, config.skew_range,
    )
    log(f"  Global best fixed: w={gw:.3f}, s={gs:.3f}, mean_dW={gm:.2f}")

    # Permuted null
    dW_perm = dW.copy()
    np.random.default_rng(config.exploration_rng_seed + 99).shuffle(dW_perm)
    model_perm = LocalBilinearModel(X=X, U=U, X_next=X_next, dW=dW_perm, Q_next=Q_next,
                                     bandwidth=bw, ridge=config.ridge)
    log(f"  Permuted-null model fitted")
    log(f"  Train time: {time.time()-t0:.1f}s")

    # ===== Evaluation =====
    log("\n--- Evaluation ---")
    strategies = [
        "risk_neutral_optimal", "bbg_numerical", "linear_inventory_skew",
        "global_fixed_action", "local_bilinear_sdre", "permuted_null",
    ]
    results: dict[str, list[EpisodeSummary]] = {}
    bilinear_choices: list[tuple[float, float]] = []

    for strat in strategies:
        t1 = time.time()
        kwargs = {}
        if strat == "global_fixed_action":
            kwargs["fixed_w"] = gw
            kwargs["fixed_s"] = gs
        elif strat == "local_bilinear_sdre":
            kwargs["bilinear_model"] = model
            kwargs["choices_accumulator"] = bilinear_choices
        elif strat == "permuted_null":
            kwargs["bilinear_model"] = model_perm

        results[strat] = run_strategy(strat, test_seeds, config, utility, **kwargs)
        log(f"  {strat}: {time.time()-t1:.1f}s")

    # ===== Results =====
    log("\n" + "=" * 70)
    log("  RESULTS")
    log("=" * 70)

    # CE table
    log(f"\n  {'Strategy':<35s} {'CE':>12s} {'Spread Cap':>12s}")
    log("  " + "-" * 61)
    ce_vals = {}
    for name, summaries in results.items():
        agg = aggregate_episode_summaries(summaries)
        ce = ce_from_summaries(summaries, utility)
        ce_vals[name] = ce
        log(f"  {name:<35s} {ce:>12.3f} {agg.gross_spread_capture_mean:>12.3f}")

    # Paired contrasts
    contrasts = [
        ("local_bilinear_sdre", "linear_inventory_skew"),
        ("local_bilinear_sdre", "bbg_numerical"),
        ("local_bilinear_sdre", "risk_neutral_optimal"),
        ("local_bilinear_sdre", "global_fixed_action"),
        ("local_bilinear_sdre", "permuted_null"),
    ]
    log(f"\n  {'Contrast':<55s} {'Mean':>8s} {'sd_post':>8s} {'P(>0)':>8s}")
    log("  " + "-" * 81)
    for na, nb in contrasts:
        p = paired_post(results[na], results[nb], utility)
        log(f"  {na + ' - ' + nb:<55s} {p.mean:>8.3f} {p.sd_post:>8.3f} {p.p_positive:>8.5f}")

    # Gap-closure
    ce_bilinear = ce_vals["local_bilinear_sdre"]
    ce_linear = ce_vals["linear_inventory_skew"]
    ce_bbg = ce_vals["bbg_numerical"]
    gap = ce_bbg - ce_linear
    if abs(gap) > 1e-6:
        gc = (ce_bilinear - ce_linear) / gap
    else:
        gc = float("nan")
    log(f"\n  Gap closure: (bilinear - linear) / (bbg - linear) = {gc:.3f}")
    log(f"    bilinear CE = {ce_bilinear:.3f}")
    log(f"    linear CE   = {ce_linear:.3f}")
    log(f"    bbg CE      = {ce_bbg:.3f}")
    log(f"    gap         = {gap:.3f}")

    # Action-usage diagnostic
    n_ch = len(bilinear_choices)
    if n_ch > 0:
        ws = [c[0] for c in bilinear_choices]
        ss = [c[1] for c in bilinear_choices]
        log(f"\n  Action usage (N={n_ch}):")
        log(f"    mean w = {np.mean(ws):.4f}, std = {np.std(ws):.4f}")
        log(f"    mean s = {np.mean(ss):.5f}, std = {np.std(ss):.5f}")
        log(f"    w range: [{np.min(ws):.4f}, {np.max(ws):.4f}]")
        log(f"    s range: [{np.min(ss):.5f}, {np.max(ss):.5f}]")
        # Check state dependence: correlation of w with inventory
        # (Can't access raw states here, just print spread of w)
        w_unique = len(set(round(w, 4) for w in ws))
        log(f"    unique w values (4dp): {w_unique}")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    results_file = results_dir / f"option_mm_bilinear_recovery_{suffix}_{date.today()}.txt"
    with open(results_file, "w") as f:
        f.write("\n".join(out))
    log(f"\nResults saved to {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
