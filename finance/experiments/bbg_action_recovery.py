"""BBG action-level recovery analysis.

Evaluates whether the learned reduced-action controllers recover BBG's
action geometry, not just its scalar CE.

For held-out states, reports:
  1. Action reconstruction error (MSE, R^2, cosine sim) vs BBG
  2. Reconstruction quality by rank
  3. Rank-vs-recovery curves

Usage:
    python finance/experiments/bbg_action_recovery.py
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
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv, OptionBookMMAction
from applications.option_mm_bbg.solver import (
    solve_bbg_value_function,
    make_bbg_numerical_controller,
)
from applications.option_mm_bbg.sdre_recovery import (
    _compute_rn_distances,
    _ridge_regression,
    extract_state_features,
)

# Import from the equivalence script
from bbg_recovery_equivalence import (
    _state_features_extended,
    collect_bbg_demonstrations,
    fit_demonstration_recovery,
)


def main() -> int:
    out: list[str] = []

    def log(msg=""):
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    rn_dists = _compute_rn_distances(config)
    n_opt = config.book.n_options
    u_baseline = np.concatenate([rn_dists, rn_dists])

    train_seeds = list(range(500))
    test_seeds = list(range(2000, 2200))

    log("=" * 70)
    log("  BBG Action-Level Recovery Analysis")
    log("=" * 70)

    # Solve HJB
    log("\nSolving HJB...")
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)

    # Collect train and test demonstrations
    log("Collecting BBG demonstrations (train)...")
    train_features, train_perturbations = collect_bbg_demonstrations(
        config, bbg_ctrl, rn_dists, train_seeds,
    )
    log(f"  Train: {len(train_features)} pairs")

    log("Collecting BBG demonstrations (test)...")
    test_features, test_perturbations = collect_bbg_demonstrations(
        config, bbg_ctrl, rn_dists, test_seeds,
    )
    log(f"  Test: {len(test_features)} pairs")

    # Compute baseline metrics (no recovery = predict zero perturbation)
    ss_total = np.sum(test_perturbations ** 2)
    test_mean_pert = np.mean(np.abs(test_perturbations))
    log(f"\n  Test perturbation scale: mean|Δu|={test_mean_pert:.6f}")
    log(f"  Test SS_total: {ss_total:.2f}")

    # Rank sweep for demonstration recovery
    log(f"\n{'='*70}")
    log("  Demonstration Recovery: Rank vs Action Reconstruction")
    log(f"{'='*70}")
    log(f"\n  {'Rank':>5s} {'R²':>8s} {'RMSE':>10s} {'CosSim':>8s} {'SV_1':>12s} {'SV_k':>12s}")

    ranks = [1, 2, 3, 5, 8, 10, 15, 20]
    for rank in ranks:
        W_r, U_r, S_r = fit_demonstration_recovery(
            train_features, train_perturbations, rank,
        )
        # Test reconstruction
        pred = test_features @ W_r.T  # (N_test, 40)
        residual = test_perturbations - pred
        ss_res = np.sum(residual ** 2)
        r2 = 1.0 - ss_res / ss_total if ss_total > 1e-30 else 0.0
        rmse = np.sqrt(np.mean(residual ** 2))

        # Cosine similarity per state
        cos_sims = []
        for i in range(len(test_perturbations)):
            a, b = test_perturbations[i], pred[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-10 and nb > 1e-10:
                cos_sims.append(np.dot(a, b) / (na * nb))
        mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

        log(f"  {rank:>5d} {r2:>8.4f} {rmse:>10.6f} {mean_cos:>8.4f} "
            f"{S_r[0]:>12.4f} {S_r[-1]:>12.4f}")

    # Per-option reconstruction at rank 3
    log(f"\n{'='*70}")
    log("  Per-Option Reconstruction at Rank 3")
    log(f"{'='*70}")

    W_r3, _, _ = fit_demonstration_recovery(train_features, train_perturbations, 3)
    pred3 = test_features @ W_r3.T
    strikes = np.array([o.strike for o in config.book.options])
    mats = np.array([o.maturity for o in config.book.options])

    log(f"\n  {'Option':<12s} {'Bid R²':>8s} {'Ask R²':>8s} {'Bid RMSE':>10s} {'Ask RMSE':>10s}")
    for i in range(n_opt):
        # Bid
        ss_tot_b = np.sum(test_perturbations[:, i] ** 2)
        ss_res_b = np.sum((test_perturbations[:, i] - pred3[:, i]) ** 2)
        r2_b = 1.0 - ss_res_b / ss_tot_b if ss_tot_b > 1e-10 else 0.0
        rmse_b = np.sqrt(np.mean((test_perturbations[:, i] - pred3[:, i]) ** 2))
        # Ask
        j = n_opt + i
        ss_tot_a = np.sum(test_perturbations[:, j] ** 2)
        ss_res_a = np.sum((test_perturbations[:, j] - pred3[:, j]) ** 2)
        r2_a = 1.0 - ss_res_a / ss_tot_a if ss_tot_a > 1e-10 else 0.0
        rmse_a = np.sqrt(np.mean((test_perturbations[:, j] - pred3[:, j]) ** 2))

        log(f"  K={strikes[i]:>4.0f} T={mats[i]:>3.1f}  {r2_b:>8.4f} {r2_a:>8.4f} "
            f"{rmse_b:>10.6f} {rmse_a:>10.6f}")

    # Explained variance of BBG action surface
    log(f"\n{'='*70}")
    log("  BBG Action Surface Singular Values")
    log(f"{'='*70}")

    # Full SVD of the BBG action perturbation matrix (no rank truncation)
    W_full = _ridge_regression(train_features, train_perturbations, 1e-3).T
    _, S_full, _ = np.linalg.svd(W_full, full_matrices=False)
    total_var = np.sum(S_full ** 2)
    log(f"\n  {'Dir':>5s} {'SV':>12s} {'Frac':>8s} {'Cumul':>8s}")
    cum = 0.0
    for i, s in enumerate(S_full):
        frac = s ** 2 / total_var
        cum += frac
        log(f"  {i+1:>5d} {s:>12.4f} {frac:>8.4f} {cum:>8.4f}")
        if cum > 0.999:
            break

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_action_recovery_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
