"""Held-out Heston parameter transfer benchmark for the hybrid controller.

Trains a BBG prior + residual controller on a subset of Heston parameter
cells, then evaluates on held-out cells.  This is the make-or-break
experiment for Track B Step 1: if the residual helps on held-out cells,
the data-driven correction transfers across regimes.

Training cells: (kappa, xi, rho) from a 2x2x2 grid.
Test cells: interpolated parameter values (the "center" of the grid).

Usage:
    python finance/experiments/option_mm_hybrid_transfer.py [--pilot]
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
from applications.option_mm.controllers import make_risk_neutral_optimal  # noqa: E402
from applications.option_mm.env import (  # noqa: E402
    FillModelSpec,
    HestonParams,
    OptionMarketMakingEnv,
)
from applications.option_mm.hybrid_residual_controller import (  # noqa: E402
    collect_hybrid_training_data_multi_cell,
    make_hybrid_residual_controller,
)
from applications.option_mm.local_kernel_controller import (  # noqa: E402
    KernelRewardModel,
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
# Heston parameter cells
# ---------------------------------------------------------------------------


def make_training_cells() -> list[HestonParams]:
    """2x2x2 grid of Heston params for training."""
    cells = []
    for kappa in (1.5, 3.0):
        for xi in (0.3, 0.7):
            for rho in (-0.5, -0.9):
                cells.append(HestonParams(kappa=kappa, xi=xi, rho=rho))
    return cells


def make_test_cells() -> list[HestonParams]:
    """Held-out interpolated cells (center of the training grid)."""
    return [
        HestonParams(kappa=2.0, xi=0.5, rho=-0.7),  # default Heston
        HestonParams(kappa=2.25, xi=0.4, rho=-0.6),  # interpolated
        HestonParams(kappa=1.75, xi=0.6, rho=-0.8),  # interpolated
    ]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferConfig:
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


STRATEGIES = ("risk_neutral_optimal", "bbg_numerical", "hybrid_bbg_residual")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    strategy_name: str,
    seed: int,
    heston: HestonParams,
    config: TransferConfig,
    utility: UtilitySpec,
    residual_model: KernelRewardModel | None = None,
) -> tuple[list, list]:
    env = OptionMarketMakingEnv(
        heston=heston,
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
            env, state, gamma=gamma, max_inventory=config.max_inventory,
        )
    elif strategy_name == "hybrid_bbg_residual":
        assert residual_model is not None
        controller = make_hybrid_residual_controller(
            env, residual_model, state,
            gamma_ce=config.gamma_ce,
            max_inventory=config.max_inventory,
            ewma_half_life_days=config.ewma_half_life_days,
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


def run_strategy_on_cell(
    strategy_name: str,
    seeds: list[int],
    heston: HestonParams,
    config: TransferConfig,
    utility: UtilitySpec,
    residual_model: KernelRewardModel | None = None,
) -> list[EpisodeSummary]:
    summaries = []
    for seed in seeds:
        states, infos = run_episode(
            strategy_name, seed, heston, config, utility, residual_model,
        )
        summaries.append(summarize_episode(
            states=states, infos=infos, inventory_limit=config.max_inventory,
        ))
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def paired_posterior(
    summaries_a: list[EpisodeSummary],
    summaries_b: list[EpisodeSummary],
    utility: UtilitySpec,
) -> PosteriorSummary:
    wa = np.array([s.terminal_wealth for s in summaries_a])
    wb = np.array([s.terminal_wealth for s in summaries_b])
    return paired_ce_posterior(wa.tolist(), wb.tolist(), utility=utility, method="delta")


def print_cell_results(
    heston: HestonParams,
    results: dict[str, list[EpisodeSummary]],
    utility: UtilitySpec,
    label: str,
) -> None:
    print(f"\n  {label}: kappa={heston.kappa}, xi={heston.xi}, rho={heston.rho}")
    for name, summaries in results.items():
        agg = aggregate_episode_summaries(summaries)
        u_vals = utility.u(np.array([s.terminal_wealth for s in summaries]))
        ce = utility.ce(float(np.mean(u_vals)))
        print(f"    {name:<30s} CE={ce:>12.3f}  spread_capture={agg.gross_spread_capture_mean:>8.3f}")

    # Headline contrast: hybrid - bbg
    if "hybrid_bbg_residual" in results and "bbg_numerical" in results:
        post = paired_posterior(results["hybrid_bbg_residual"], results["bbg_numerical"], utility)
        print(f"    hybrid - bbg: mean={post.mean:.3f}, sd_post={post.sd_post:.3f}, "
              f"P(>0)={post.p_positive:.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid transfer benchmark")
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot (20 train eps, 20 test eps per cell)")
    args = parser.parse_args(argv)

    config = TransferConfig()
    if args.pilot:
        config = TransferConfig(
            n_train_episodes_per_cell=20,
            n_test_episodes_per_cell=20,
        )

    utility = crra_utility(config.gamma_ce)
    train_cells = make_training_cells()
    test_cells = make_test_cells()

    # Generate seeds
    ss = np.random.SeedSequence(config.seed_sequence_entropy)
    n_train = config.n_train_episodes_per_cell
    n_test = config.n_test_episodes_per_cell
    child_seeds = ss.spawn(n_train + n_test)
    all_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
    train_seeds = all_ints[:n_train]
    test_seeds = all_ints[n_train:]

    print("=" * 60)
    print("  Hybrid BBG + Residual: Held-Out Heston Transfer Benchmark")
    print("=" * 60)
    print(f"\nTraining cells: {len(train_cells)} (2x2x2 grid)")
    print(f"Test cells:     {len(test_cells)} (interpolated)")
    print(f"Train episodes per cell: {n_train}")
    print(f"Test episodes per cell:  {n_test}")
    print(f"Utility: CRRA(gamma={config.gamma_ce})")

    # ----- Training -----
    t0 = time.time()
    print("\nCollecting training data across cells...")
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
    t_train = time.time() - t0
    print(f"  Training data: {buffer.size} tuples from {len(train_cells)} cells")
    print(f"  Model fitted on {features.shape[0]} samples (bandwidth={bw:.3f})")
    print(f"  Training time: {t_train:.1f}s")

    # ----- Evaluation on training cells (sanity check) -----
    print("\n" + "=" * 60)
    print("  In-regime (training cells) — sanity check")
    print("=" * 60)
    for cell in train_cells[:2]:  # just show 2 for brevity
        cell_results = {}
        for strat in STRATEGIES:
            cell_results[strat] = run_strategy_on_cell(
                strat, test_seeds, cell, config, utility,
                residual_model=model if strat == "hybrid_bbg_residual" else None,
            )
        print_cell_results(cell, cell_results, utility, "TRAIN CELL")

    # ----- Evaluation on held-out cells (the real test) -----
    print("\n" + "=" * 60)
    print("  Held-out cells — transfer benchmark")
    print("=" * 60)
    for cell in test_cells:
        cell_results = {}
        for strat in STRATEGIES:
            t1 = time.time()
            cell_results[strat] = run_strategy_on_cell(
                strat, test_seeds, cell, config, utility,
                residual_model=model if strat == "hybrid_bbg_residual" else None,
            )
        print_cell_results(cell, cell_results, utility, "TEST CELL")

    # ----- Save results -----
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    from datetime import date
    results_file = results_dir / f"option_mm_hybrid_transfer_{date.today()}.txt"
    with open(results_file, "w") as f:
        f.write("Hybrid BBG + Residual: Held-Out Heston Transfer Benchmark\n")
        f.write(f"Date: {date.today()}\n")
        f.write(f"Train cells: {len(train_cells)}, Test cells: {len(test_cells)}\n")
        f.write(f"Train eps/cell: {n_train}, Test eps/cell: {n_test}\n\n")
        for cell in test_cells:
            f.write(f"Test cell: kappa={cell.kappa}, xi={cell.xi}, rho={cell.rho}\n")
            cell_results = {}
            for strat in STRATEGIES:
                cell_results[strat] = run_strategy_on_cell(
                    strat, test_seeds, cell, config, utility,
                    residual_model=model if strat == "hybrid_bbg_residual" else None,
                )
            if "hybrid_bbg_residual" in cell_results and "bbg_numerical" in cell_results:
                post = paired_posterior(
                    cell_results["hybrid_bbg_residual"],
                    cell_results["bbg_numerical"],
                    utility,
                )
                f.write(f"  hybrid - bbg: mean={post.mean:.3f}, sd_post={post.sd_post:.3f}, "
                        f"P(>0)={post.p_positive:.5f}\n")
    print(f"\nResults saved to {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
