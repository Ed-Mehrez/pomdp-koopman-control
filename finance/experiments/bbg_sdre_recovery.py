"""SDRE recovery experiment against the BBG benchmark.

Compares:
  1. risk_neutral  (p=0 baseline)
  2. bbg_numerical (solved HJB, ground truth)
  3. sdre_bilinear (Option 2: reduced bilinear coordinates)
  4. sdre_action_pca (Option 1: action-value Hessian eigenvectors)

Primary contrast: sdre_* vs bbg_numerical.
Success criterion: gap reduction toward bbg_numerical, not necessarily beating it.

Usage:
    python finance/experiments/bbg_sdre_recovery.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from applications.option_mm_bbg.spec import BBGBenchmarkConfig
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv
from applications.option_mm_bbg.solver import (
    solve_bbg_value_function,
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
)
from applications.option_mm_bbg.sdre_recovery import (
    SDRERecoveryConfig,
    make_sdre_recovery_controller,
    _compute_rn_distances,
)


# ---------------------------------------------------------------------------
# Metrics (same as benchmark script)
# ---------------------------------------------------------------------------


def cara_ce(wealths: np.ndarray, gamma: float) -> float:
    if gamma == 0.0:
        return float(np.mean(wealths))
    neg_gw = -gamma * wealths
    mx = float(np.max(neg_gw))
    return -(mx + np.log(np.mean(np.exp(neg_gw - mx)))) / gamma


def mean_var_surrogate(wealths: np.ndarray, gamma: float) -> float:
    return float(np.mean(wealths) - 0.5 * gamma * np.var(wealths))


def bootstrap_ce_diff(w_a, w_b, gamma, n_boot=10_000, seed=999):
    rng = np.random.default_rng(seed)
    n = len(w_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = cara_ce(w_a[idx], gamma) - cara_ce(w_b[idx], gamma)
    return {
        "mean": float(np.mean(diffs)),
        "sd_post": float(np.std(diffs)),
        "p_pos": float(np.mean(diffs > 0)),
        "ci_95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
    }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episodes(config, ctrl, label, n_episodes, log_fn):
    wealths, spreads, avg_vega = [], [], []
    t0 = time.time()
    for seed in range(n_episodes):
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        total_spread = 0.0
        vega_abs_sum, n_steps = 0.0, 0
        while not state.done:
            action = ctrl(state)
            state, _, _, info = env.step(action)
            total_spread += info["spread_capture"]
            vega_abs_sum += abs(state.portfolio_vega)
            n_steps += 1
        wealths.append(state.wealth)
        spreads.append(total_spread)
        avg_vega.append(vega_abs_sum / max(n_steps, 1))

    elapsed = time.time() - t0
    w = np.array(wealths)
    gamma = config.control.gamma
    ce = cara_ce(w, gamma)
    mv = mean_var_surrogate(w, gamma)
    log_fn(f"\n  {label} ({elapsed:.1f}s):")
    log_fn(f"    wealth: mean={w.mean():.0f}, std={w.std():.0f}")
    log_fn(f"    spread: mean={np.mean(spreads):.0f}")
    log_fn(f"    |vega|: mean={np.mean(avg_vega):.0f}")
    log_fn(f"    CARA CE: {ce:.0f}")
    log_fn(f"    mean-var: {mv:.0f}")
    return w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    out: list[str] = []
    def log(msg=""):
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma

    log("=" * 70)
    log("  BBG SDRE Recovery Experiment")
    log("=" * 70)
    log(f"  Options: {config.book.n_options}")
    log(f"  Gamma: {gamma}")
    log(f"  Horizon: {config.control.horizon}")

    # --- Solve BBG HJB for reference ---
    log("\nSolving 3D HJB for benchmark...")
    t0 = time.time()
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    log(f"  HJB solved in {time.time() - t0:.1f}s")

    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # --- Build SDRE controllers ---
    rn_dists = _compute_rn_distances(config)
    rank = 3

    log(f"\nTraining SDRE bilinear (rank={rank}, 500 explore episodes)...")
    t0 = time.time()
    sdre_bilinear_ctrl, bilinear_model = make_sdre_recovery_controller(
        config,
        SDRERecoveryConfig(method="bilinear", rank=rank, n_explore_episodes=500),
        rn_distances=rn_dists,
        return_model=True,
    )
    log(f"  Trained in {time.time() - t0:.1f}s")

    # Report explained variance
    ev = bilinear_model.explained_variance()
    log(f"  Singular values (top 8): {bilinear_model.S_r}")
    log(f"  Explained variance (top 8): {ev[:8]}")
    log(f"  Cumulative EV at rank {rank}: {ev[:rank].sum():.4f}")

    log(f"\nTraining SDRE action_pca (rank={rank})...")
    t0 = time.time()
    sdre_pca_ctrl = make_sdre_recovery_controller(
        config,
        SDRERecoveryConfig(method="action_pca", rank=rank, n_explore_episodes=500),
        rn_distances=rn_dists,
    )
    log(f"  Trained in {time.time() - t0:.1f}s")

    log(f"\nTraining SDRE bilinear_2stage (rank={rank}, overspace=10)...")
    t0 = time.time()
    sdre_2stage_ctrl = make_sdre_recovery_controller(
        config,
        SDRERecoveryConfig(method="bilinear_2stage", rank=rank,
                           n_explore_episodes=500, bilinear_overspace=10),
        rn_distances=rn_dists,
    )
    log(f"  Trained in {time.time() - t0:.1f}s")

    # --- Run evaluation ---
    n_episodes = 200
    log(f"\nEvaluating {n_episodes} episodes per controller...")

    w_rn = run_episodes(config, rn_ctrl, "Risk-neutral", n_episodes, log)
    w_bbg = run_episodes(config, bbg_ctrl, "BBG numerical", n_episodes, log)
    w_bil = run_episodes(config, sdre_bilinear_ctrl, f"SDRE bilinear (r={rank})", n_episodes, log)
    w_pca = run_episodes(config, sdre_pca_ctrl, f"SDRE action_pca (r={rank})", n_episodes, log)
    w_2st = run_episodes(config, sdre_2stage_ctrl, f"SDRE bilinear_2stage (r={rank})", n_episodes, log)

    # --- Paired comparisons ---
    log(f"\n{'='*70}")
    log(f"  Paired CARA CE comparisons (gamma={gamma})")
    log(f"{'='*70}")

    comparisons = [
        ("BBG - RN", w_bbg, w_rn),
        ("Bilinear - RN", w_bil, w_rn),
        ("Bilinear - BBG", w_bil, w_bbg),
        ("ActionPCA - RN", w_pca, w_rn),
        ("ActionPCA - BBG", w_pca, w_bbg),
        ("2Stage - RN", w_2st, w_rn),
        ("2Stage - BBG", w_2st, w_bbg),
        ("2Stage - Bilinear", w_2st, w_bil),
    ]

    for label, wa, wb in comparisons:
        boot = bootstrap_ce_diff(wa, wb, gamma)
        log(f"\n  {label}:")
        log(f"    CE diff mean = {boot['mean']:.0f}")
        log(f"    sd_post = {boot['sd_post']:.0f}")
        log(f"    P(>0) = {boot['p_pos']:.4f}")
        log(f"    95% CrI = [{boot['ci_95'][0]:.0f}, {boot['ci_95'][1]:.0f}]")

    # --- Summary table ---
    ce_rn = cara_ce(w_rn, gamma)
    ce_bbg = cara_ce(w_bbg, gamma)
    ce_bil = cara_ce(w_bil, gamma)
    ce_pca = cara_ce(w_pca, gamma)
    ce_2st = cara_ce(w_2st, gamma)

    log(f"\n  === CE Summary ===")
    log(f"  {'Controller':<25s} {'CE':>10s} {'Mean W':>10s} {'Std W':>10s}")
    log(f"  {'Risk-neutral':<25s} {ce_rn:>10.0f} {w_rn.mean():>10.0f} {w_rn.std():>10.0f}")
    log(f"  {'BBG numerical':<25s} {ce_bbg:>10.0f} {w_bbg.mean():>10.0f} {w_bbg.std():>10.0f}")
    log(f"  {'Bilinear':<25s} {ce_bil:>10.0f} {w_bil.mean():>10.0f} {w_bil.std():>10.0f}")
    log(f"  {'ActionPCA':<25s} {ce_pca:>10.0f} {w_pca.mean():>10.0f} {w_pca.std():>10.0f}")
    log(f"  {'Bilinear 2-stage':<25s} {ce_2st:>10.0f} {w_2st.mean():>10.0f} {w_2st.std():>10.0f}")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_sdre_recovery_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
