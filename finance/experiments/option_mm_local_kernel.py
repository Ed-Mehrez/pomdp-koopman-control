"""Track B experiment: local kernel controller vs BBG baselines.

Trains a model-free kernel controller on a held-out seed set, then
evaluates against risk_neutral_optimal, bbg_numerical, and
linear_inventory_skew(oracle) on a separate test seed set using the
same paired Bayesian posterior comparison as ablation_v2.

Usage:
    python finance/experiments/option_mm_local_kernel.py [--pilot] [--n-train 500] [--n-test 200]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.bbg_solver import make_bbg_numerical  # noqa: E402
from applications.option_mm.beliefs import EWMAVarianceFilter  # noqa: E402
from applications.option_mm.controllers import (  # noqa: E402
    make_linear_inventory_skew,
    make_risk_neutral_optimal,
)
from applications.option_mm.env import (  # noqa: E402
    FillModelSpec,
    OptionMMAction,
    OptionMMState,
    OptionMarketMakingEnv,
)
from applications.option_mm.inventory_variance import oracle_heston_estimator  # noqa: E402
from applications.option_mm.local_kernel_controller import (  # noqa: E402
    collect_training_data,
    make_local_kernel_controller,
    train_local_kernel_model,
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
class LocalKernelExperimentConfig:
    # Seed allocation: training and test are disjoint.
    seed_sequence_entropy: int = 20260410
    n_train_episodes: int = 500
    n_test_episodes: int = 200

    # Env
    horizon_steps: int = 20
    initial_cash: float = 100_000.0

    # Filter
    ewma_half_life_days: float = 5.0

    # Utility
    gamma_ce: float = 2.0

    # Training
    noise_spread: float = 0.03
    noise_skew: float = 0.02
    exploration_rng_seed: int = 42
    ridge_alpha: float = 1e-3
    max_training_samples: int = 8_000

    # Deployment
    spread_range: tuple[float, float] = (0.10, 0.30)
    skew_range: tuple[float, float] = (-0.05, 0.05)
    n_spread_candidates: int = 9
    n_skew_candidates: int = 9

    # BBG numerical baseline
    inventory_limit: int = 10


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

STRATEGIES = (
    "risk_neutral_optimal",
    "bbg_numerical",
    "linear_inventory_skew_oracle",
    "local_kernel",
)


@dataclass(frozen=True)
class StrategyRun:
    name: str
    summaries: list[EpisodeSummary]

    @property
    def terminal_wealth(self) -> np.ndarray:
        return np.asarray(
            [s.terminal_wealth for s in self.summaries], dtype=float
        )


def run_episode(
    strategy_name: str,
    seed: int,
    config: LocalKernelExperimentConfig,
    utility: UtilitySpec,
    kernel_model=None,
) -> tuple[list[OptionMMState], list]:
    """Run one episode for a given strategy."""
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()

    if strategy_name == "risk_neutral_optimal":
        controller = make_risk_neutral_optimal(env)
    elif strategy_name == "bbg_numerical":
        gamma = utility.arrow_pratt(state.wealth)
        controller = make_bbg_numerical(
            env, state, gamma=gamma, max_inventory=config.inventory_limit,
        )
    elif strategy_name == "linear_inventory_skew_oracle":
        estimator = oracle_heston_estimator(env)
        controller = make_linear_inventory_skew(env, estimator, utility)
    elif strategy_name == "local_kernel":
        assert kernel_model is not None, "kernel model required for local_kernel"
        controller = make_local_kernel_controller(
            env,
            kernel_model,
            state,
            ewma_half_life_days=config.ewma_half_life_days,
            spread_range=config.spread_range,
            skew_range=config.skew_range,
            n_spread_candidates=config.n_spread_candidates,
            n_skew_candidates=config.n_skew_candidates,
        )
    else:
        raise ValueError(f"unknown strategy: {strategy_name}")

    states = [state]
    infos = []
    while not state.done:
        action = controller(state)
        state, _, _, info = env.step(action)
        states.append(state)
        infos.append(info)
    return states, infos


def run_strategy(
    strategy_name: str,
    seeds: tuple[int, ...],
    config: LocalKernelExperimentConfig,
    utility: UtilitySpec,
    kernel_model=None,
) -> StrategyRun:
    summaries = []
    for seed in seeds:
        states, infos = run_episode(
            strategy_name, seed, config, utility, kernel_model,
        )
        summaries.append(
            summarize_episode(
                states=states,
                infos=infos,
                inventory_limit=config.inventory_limit,
            )
        )
    return StrategyRun(name=strategy_name, summaries=summaries)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def paired_posterior(
    run_a: StrategyRun,
    run_b: StrategyRun,
    utility: UtilitySpec,
) -> PosteriorSummary:
    return paired_ce_posterior(
        run_a.terminal_wealth.tolist(),
        run_b.terminal_wealth.tolist(),
        utility=utility,
        method="delta",
    )


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_ce_table(runs: dict[str, StrategyRun], utility: UtilitySpec) -> None:
    print(f"{'Strategy':<35s} {'CE':>14s} {'Wealth Mean':>14s} {'Wealth Std':>12s}")
    print("-" * 77)
    for name, run in runs.items():
        agg = aggregate_episode_summaries(run.summaries)
        u_vals = utility.u(run.terminal_wealth)
        ce = utility.ce(float(np.mean(u_vals)))
        print(
            f"{name:<35s} {ce:>14.3f} {agg.terminal_wealth_mean:>14.3f} "
            f"{agg.terminal_wealth_std:>12.3f}"
        )


def print_contrasts(
    runs: dict[str, StrategyRun],
    utility: UtilitySpec,
    reference: str = "risk_neutral_optimal",
) -> None:
    print(f"\n{'Contrast (vs ' + reference + ')':<45s} "
          f"{'Mean':>10s} {'sd_post':>10s} {'P(>0)':>8s}")
    print("-" * 75)
    ref = runs[reference]
    for name, run in runs.items():
        if name == reference:
            continue
        post = paired_posterior(run, ref, utility)
        print(
            f"{name + ' - ' + reference:<45s} "
            f"{post.mean:>10.3f} {post.sd_post:>10.3f} {post.p_positive:>8.5f}"
        )


def print_head_to_head(
    runs: dict[str, StrategyRun],
    utility: UtilitySpec,
    name_a: str,
    name_b: str,
) -> None:
    post = paired_posterior(runs[name_a], runs[name_b], utility)
    label = f"{name_a} - {name_b}"
    print(f"\n  Head-to-head: {label}")
    print(f"    mean = {post.mean:.3f}, sd_post = {post.sd_post:.3f}, "
          f"P(>0) = {post.p_positive:.5f}")
    print(f"    95% CrI = [{post.ci_low:.3f}, {post.ci_high:.3f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track B: local kernel controller")
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot run (50 train, 50 test)")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    args = parser.parse_args(argv)

    config = LocalKernelExperimentConfig()

    if args.pilot:
        n_train = args.n_train or 50
        n_test = args.n_test or 50
    else:
        n_train = args.n_train or config.n_train_episodes
        n_test = args.n_test or config.n_test_episodes

    # Disjoint seed allocation: use SeedSequence to generate unique
    # integer seeds for training and test.
    ss = np.random.SeedSequence(config.seed_sequence_entropy)
    all_child_seeds = ss.spawn(n_train + n_test)
    all_ints = [int(cs.generate_state(1)[0]) for cs in all_child_seeds]
    train_seeds = tuple(all_ints[:n_train])
    test_seeds = tuple(all_ints[n_train:])

    utility = crra_utility(config.gamma_ce)

    # ----- Training phase -----
    print_header("Track B: Local Kernel Controller Experiment")
    print(f"Training episodes: {n_train}")
    print(f"Test episodes:     {n_test}")
    print(f"Utility:           CRRA(gamma={config.gamma_ce})")
    print(f"Horizon:           {config.horizon_steps} steps")
    print()

    t0 = time.time()
    print("Training kernel reward model...")
    kernel_model = train_local_kernel_model(
        training_seeds=train_seeds,
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        ewma_half_life_days=config.ewma_half_life_days,
        noise_spread=config.noise_spread,
        noise_skew=config.noise_skew,
        exploration_rng_seed=config.exploration_rng_seed,
        ridge_alpha=config.ridge_alpha,
        max_training_samples=config.max_training_samples,
    )
    t_train = time.time() - t0
    print(f"  Training completed in {t_train:.1f}s")

    # ----- Evaluation phase -----
    print("\nEvaluating strategies on test seeds...")
    runs: dict[str, StrategyRun] = {}
    for strategy_name in STRATEGIES:
        t1 = time.time()
        runs[strategy_name] = run_strategy(
            strategy_name,
            test_seeds,
            config,
            utility,
            kernel_model=kernel_model if strategy_name == "local_kernel" else None,
        )
        elapsed = time.time() - t1
        print(f"  {strategy_name}: {elapsed:.1f}s")

    # ----- Results -----
    print_header("Absolute CE Table (CRRA gamma=2)")
    print_ce_table(runs, utility)

    print_header("Paired Posterior Contrasts")
    print_contrasts(runs, utility, reference="risk_neutral_optimal")

    # Head-to-head: local_kernel vs each baseline
    print_header("Head-to-Head Comparisons")
    print_head_to_head(runs, utility, "local_kernel", "risk_neutral_optimal")
    print_head_to_head(runs, utility, "local_kernel", "bbg_numerical")
    print_head_to_head(runs, utility, "local_kernel", "linear_inventory_skew_oracle")

    # ----- Aggregate diagnostics -----
    print_header("Aggregate Diagnostics")
    for name, run in runs.items():
        agg = aggregate_episode_summaries(run.summaries)
        print(f"\n  {name}:")
        print(f"    Spread capture (mean): {agg.gross_spread_capture_mean:.3f}")
        print(f"    Adverse selection (mean): {agg.adverse_selection_cost_mean:.3f}")
        print(f"    Net delta RMS (mean): {agg.net_delta_rms_mean:.3f}")
        print(f"    Inventory |q| mean: {agg.inventory_abs_mean:.3f}")

    # ----- Save results -----
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    from datetime import date
    results_file = results_dir / f"option_mm_local_kernel_{date.today()}.txt"
    with open(results_file, "w") as f:
        f.write(f"Track B: Local Kernel Controller Experiment\n")
        f.write(f"Date: {date.today()}\n")
        f.write(f"Training episodes: {n_train}\n")
        f.write(f"Test episodes: {n_test}\n")
        f.write(f"Utility: CRRA(gamma={config.gamma_ce})\n\n")
        for name, run in runs.items():
            agg = aggregate_episode_summaries(run.summaries)
            u_vals = utility.u(run.terminal_wealth)
            ce = utility.ce(float(np.mean(u_vals)))
            f.write(f"{name}: CE={ce:.3f}, wealth_mean={agg.terminal_wealth_mean:.3f}\n")
        f.write("\nPaired contrasts vs risk_neutral_optimal:\n")
        ref = runs["risk_neutral_optimal"]
        for name, run in runs.items():
            if name == "risk_neutral_optimal":
                continue
            post = paired_posterior(run, ref, utility)
            f.write(f"  {name}: mean={post.mean:.3f}, sd_post={post.sd_post:.3f}, "
                    f"P(>0)={post.p_positive:.5f}\n")
    print(f"\nResults saved to {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
