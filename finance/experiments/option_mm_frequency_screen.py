"""Multi-frequency screening: does higher-frequency OMM create a real residual signal?

Compares daily / hourly / 5-minute control on the same calendar horizon with
the same controller family. Uses the frozen hybrid spec from commit 735b3fa.

SCALING AUDIT FINDING (run before controllers):
  Fill events per episode are ~3 regardless of frequency (annualized Poisson).
  Higher frequency creates more empty quoting decisions, not more fill events.
  Fill rate: daily 14.6%, hourly 2.4%, 5-min 0.19%.
  This is a structural limitation of the current fill model.

Usage:
    python finance/experiments/option_mm_frequency_screen.py [--pilot]
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

# Re-use from formal audit
from option_mm_hybrid_formal import (  # noqa: E402
    compute_global_best_perturbation,
    make_global_best_stencil_controller,
    make_instrumented_hybrid_controller,
)


# ---------------------------------------------------------------------------
# Cells (frozen)
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
# Frequency configs
# ---------------------------------------------------------------------------

# Same calendar horizon (~20 trading days) at different granularity.
# 5-min is capped at 5 trading days to keep runtime reasonable.
FREQUENCIES = {
    "daily":  {"dt": 1 / 252,   "steps": 20,   "label": "daily (20d)"},
    "hourly": {"dt": 1 / 1638,  "steps": 130,  "label": "hourly (20d)"},
    "5min":   {"dt": 1 / 19656, "steps": 390,  "label": "5-min (5d)"},
}

DEFAULT_STENCIL_WIDTH = (-0.03, -0.015, 0.0, 0.015, 0.03)
DEFAULT_STENCIL_SKEW = (-0.015, -0.005, 0.0, 0.005, 0.015)


@dataclass(frozen=True)
class ScreenConfig:
    seed_sequence_entropy: int = 20260410
    n_train_episodes_per_cell: int = 20
    n_test_episodes_per_cell: int = 20
    initial_cash: float = 100_000.0
    gamma_ce: float = 2.0
    max_inventory: int = 10
    ewma_half_life_days: float = 5.0
    noise_width: float = 0.02
    noise_skew: float = 0.01
    exploration_rng_seed: int = 42
    ridge_alpha: float = 1e-3
    max_training_samples: int = 10_000


STRATEGIES = (
    "risk_neutral_optimal",
    "bbg_numerical",
    "bbg_global_best_stencil",
    "hybrid_bbg_residual",
)


# ---------------------------------------------------------------------------
# dt-aware training data collection (local, avoids modifying frozen module)
# ---------------------------------------------------------------------------


def collect_training_data_with_dt(
    cell_params: list[HestonParams],
    seeds: list[int],
    dt: float,
    horizon_steps: int,
    config: ScreenConfig,
) -> HybridTrainingBuffer:
    """Collect hybrid training data at arbitrary dt/horizon_steps."""
    exploration_rng = np.random.default_rng(config.exploration_rng_seed)
    buffer = HybridTrainingBuffer()
    utility_fn = _crra_utility_arrow_pratt(config.gamma_ce)

    for heston in cell_params:
        for seed in seeds:
            env = OptionMarketMakingEnv(
                heston=heston,
                fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
                dt=dt,
                horizon_steps=horizon_steps,
                initial_cash=config.initial_cash,
                seed=seed,
            )
            state = env.reset()
            gamma_local = utility_fn(state.wealth)
            bbg = BBGQuoteLookup(
                env, state, gamma=gamma_local, max_inventory=config.max_inventory,
            )
            ewma = EWMAVarianceFilter(half_life_days=config.ewma_half_life_days)
            ewma.reset(initial_variance=state.variance, initial_spot=state.spot)

            while not state.done:
                v_hat = ewma.variance
                feat = extract_state_features(state, env, v_hat)
                bd, ad = bbg.distances(state)

                dw = exploration_rng.normal(0, config.noise_width)
                ds = exploration_rng.normal(0, config.noise_skew)
                delta_u = np.array([dw, ds])
                action = _apply_perturbation(state, bd, ad, dw, ds)

                state, _, _, info = env.step(action)
                ewma.update(state.spot)
                reward = info.spread_capture - info.option_fees - info.stock_costs
                buffer.add(feat, delta_u, reward)

    return buffer


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    strategy_name: str,
    seed: int,
    heston: HestonParams,
    dt: float,
    horizon_steps: int,
    config: ScreenConfig,
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
        dt=dt,
        horizon_steps=horizon_steps,
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
    elif strategy_name == "hybrid_bbg_residual":
        assert residual_model is not None
        if choices_accumulator is not None:
            ctrl, local_choices = make_instrumented_hybrid_controller(
                env, residual_model, state,
                gamma_ce=config.gamma_ce, max_inventory=config.max_inventory,
                ewma_half_life_days=config.ewma_half_life_days,
            )
        else:
            from applications.option_mm.hybrid_residual_controller import \
                make_hybrid_residual_controller
            ctrl = make_hybrid_residual_controller(
                env, residual_model, state,
                gamma_ce=config.gamma_ce, max_inventory=config.max_inventory,
                ewma_half_life_days=config.ewma_half_life_days,
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


def run_strategy_on_cell(
    strategy_name: str,
    seeds: list[int],
    heston: HestonParams,
    dt: float,
    horizon_steps: int,
    config: ScreenConfig,
    utility: UtilitySpec,
    **kwargs,
) -> list[EpisodeSummary]:
    summaries = []
    for seed in seeds:
        states, infos = run_episode(
            strategy_name, seed, heston, dt, horizon_steps, config, utility,
            **kwargs,
        )
        summaries.append(summarize_episode(
            states=states, infos=infos, inventory_limit=config.max_inventory,
        ))
    return summaries


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def paired_post(a, b, utility):
    wa = [s.terminal_wealth for s in a]
    wb = [s.terminal_wealth for s in b]
    return paired_ce_posterior(wa, wb, utility=utility, method="delta")


def cell_label(h):
    return f"kappa={h.kappa}, xi={h.xi}, rho={h.rho}"


def perturbation_summary(choices):
    n = len(choices)
    if n == 0:
        return "  (no choices)"
    dws = [c[0] for c in choices]
    dss = [c[1] for c in choices]
    zero_frac = sum(1 for c in choices if c == (0.0, 0.0)) / n
    top3 = Counter(choices).most_common(3)
    lines = [
        f"    mean dw={np.mean(dws):+.5f}, mean ds={np.mean(dss):+.5f}, "
        f"frac(0,0)={zero_frac:.3f}",
    ]
    for pt, cnt in top3:
        lines.append(f"    ({pt[0]:+.4f},{pt[1]:+.4f}) freq={cnt}/{n} ({cnt/n:.0%})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-frequency screening")
    parser.add_argument("--pilot", action="store_true",
                        help="Use 10 train/test per cell for quick check")
    args = parser.parse_args(argv)

    config = ScreenConfig()
    if args.pilot:
        config = ScreenConfig(n_train_episodes_per_cell=10, n_test_episodes_per_cell=10)

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

    out: list[str] = []

    def log(msg=""):
        print(msg)
        out.append(msg)

    log("=" * 70)
    log("  FREQUENCY SCREENING: daily / hourly / 5-min")
    log("=" * 70)
    log(f"Train eps/cell: {n_train}  |  Test eps/cell: {n_test}")
    log(f"Utility: CRRA(gamma={config.gamma_ce})")
    log(f"\nScaling audit finding: fills/ep ~3 at ALL frequencies (Poisson annualized).")
    log(f"Fill rate: daily 14.6%, hourly 2.4%, 5-min 0.19%.")

    # ===== Per frequency =====
    # Compact summary table for final comparison
    summary_rows = []

    for freq_name, freq_params in FREQUENCIES.items():
        dt = freq_params["dt"]
        steps = freq_params["steps"]
        freq_label = freq_params["label"]

        log(f"\n{'='*70}")
        log(f"  FREQUENCY: {freq_label}  (dt={dt:.8f}, steps={steps})")
        log(f"{'='*70}")

        # Train
        t0 = time.time()
        log("  Training...")
        buffer = collect_training_data_with_dt(
            cell_params=train_cells,
            seeds=train_seeds,
            dt=dt,
            horizon_steps=steps,
            config=config,
        )
        features, perturbations, rewards = buffer.as_arrays()
        nonzero_frac = float(np.mean(np.abs(rewards) > 1e-10))
        log(f"  Tuples: {buffer.size}, nonzero reward frac: {nonzero_frac:.4f}")

        if features.shape[0] > config.max_training_samples:
            rng = np.random.default_rng(config.exploration_rng_seed + 1)
            idx = rng.choice(features.shape[0], size=config.max_training_samples, replace=False)
            features, perturbations, rewards = features[idx], perturbations[idx], rewards[idx]

        bw = median_bandwidth(features, perturbations)
        model = KernelRewardModel(bandwidth=bw, ridge_alpha=config.ridge_alpha)
        model.fit(features, perturbations, rewards)

        gbw, gbs, gbr = compute_global_best_perturbation(buffer)
        log(f"  Global best stencil: dw={gbw:+.4f}, ds={gbs:+.4f}, mean_rew={gbr:.3f}")
        log(f"  Train time: {time.time()-t0:.1f}s")

        # Evaluate on test cells
        for cell in test_cells:
            log(f"\n  --- {cell_label(cell)} ---")
            cell_results = {}
            cell_choices = []

            for strat in STRATEGIES:
                kwargs = {}
                if strat == "bbg_global_best_stencil":
                    kwargs["global_best_dw"] = gbw
                    kwargs["global_best_ds"] = gbs
                elif strat == "hybrid_bbg_residual":
                    kwargs["residual_model"] = model
                    kwargs["choices_accumulator"] = cell_choices

                cell_results[strat] = run_strategy_on_cell(
                    strat, test_seeds, cell, dt, steps, config, utility, **kwargs,
                )

            # CE table
            bbg_s = cell_results["bbg_numerical"]
            hybrid_s = cell_results["hybrid_bbg_residual"]
            gbs_s = cell_results["bbg_global_best_stencil"]

            u_bbg = utility.u(np.array([s.terminal_wealth for s in bbg_s]))
            u_hyb = utility.u(np.array([s.terminal_wealth for s in hybrid_s]))

            ce_bbg = utility.ce(float(np.mean(u_bbg)))
            ce_hyb = utility.ce(float(np.mean(u_hyb)))

            p_hb = paired_post(hybrid_s, bbg_s, utility)
            p_gb = paired_post(gbs_s, bbg_s, utility)
            p_hg = paired_post(hybrid_s, gbs_s, utility)

            agg_bbg = aggregate_episode_summaries(bbg_s)
            agg_hyb = aggregate_episode_summaries(hybrid_s)

            log(f"  CE: bbg={ce_bbg:.1f}, hybrid={ce_hyb:.1f}")
            log(f"  Spread cap: bbg={agg_bbg.gross_spread_capture_mean:.1f}, "
                f"hybrid={agg_hyb.gross_spread_capture_mean:.1f}")
            log(f"  hybrid-bbg: {p_hb.mean:+.1f} (sd={p_hb.sd_post:.1f}, P={p_hb.p_positive:.3f})")
            log(f"  global-bbg: {p_gb.mean:+.1f} (sd={p_gb.sd_post:.1f}, P={p_gb.p_positive:.3f})")
            log(f"  hybrid-global: {p_hg.mean:+.1f} (sd={p_hg.sd_post:.1f}, P={p_hg.p_positive:.3f})")
            log(perturbation_summary(cell_choices))

            summary_rows.append({
                "freq": freq_name, "cell": cell_label(cell),
                "hybrid_bbg_mean": p_hb.mean, "hybrid_bbg_p": p_hb.p_positive,
                "hybrid_global_mean": p_hg.mean, "hybrid_global_p": p_hg.p_positive,
                "n_choices": len(cell_choices),
                "zero_frac": (sum(1 for c in cell_choices if c == (0.0, 0.0)) / max(len(cell_choices), 1)),
            })

    # ===== Summary table =====
    log(f"\n{'='*70}")
    log("  SUMMARY TABLE")
    log(f"{'='*70}")
    log(f"{'Freq':<8s} {'Cell':<35s} {'hyb-bbg':>8s} {'P(>0)':>6s} {'hyb-gbl':>8s} {'P(>0)':>6s}")
    log("-" * 75)
    for row in summary_rows:
        log(f"{row['freq']:<8s} {row['cell']:<35s} "
            f"{row['hybrid_bbg_mean']:>+8.1f} {row['hybrid_bbg_p']:>6.3f} "
            f"{row['hybrid_global_mean']:>+8.1f} {row['hybrid_global_p']:>6.3f}")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"option_mm_frequency_screen_{date.today()}.txt"
    with open(results_file, "w") as f:
        f.write("\n".join(out))
    log(f"\nResults saved to {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
