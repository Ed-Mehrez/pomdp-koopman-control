"""Heuristic-subspace alignment validation for the learned action basis.

Computes:
  1. Principal angles between learned subspace and heuristic span
  2. Projection fraction for each heuristic direction
  3. BBG action reconstruction quality in the learned subspace
  4. Rank vs performance vs alignment summary

Usage:
    python finance/experiments/bbg_action_subspace_validation.py
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
    collect_exploration_data,
    BilinearControlModel,
    ActionPCAModel,
    _compute_rn_distances,
)
from applications.option_mm_bbg.heuristic_action_dictionary import (
    build_heuristic_dictionary,
    principal_angles,
    projection_fraction,
)


def main() -> int:
    out: list[str] = []
    def log(msg=""):
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma
    ranks = [1, 2, 3, 5, 8]

    log("=" * 70)
    log("  Heuristic-Subspace Alignment Validation")
    log("=" * 70)

    # Build heuristic dictionary
    heuristics = build_heuristic_dictionary(config)
    log(f"\n  Heuristic directions: {list(heuristics.keys())}")

    # Stack heuristic directions into a matrix
    h_names = list(heuristics.keys())
    H_mat = np.column_stack([heuristics[n] for n in h_names])  # (40, 4)
    log(f"  Heuristic span dimension: {H_mat.shape[1]}")

    # Collect exploration data
    rn_dists = _compute_rn_distances(config)
    sdre_cfg = SDRERecoveryConfig(n_explore_episodes=500)
    log("\nCollecting exploration data...")
    data = collect_exploration_data(config, rn_dists, sdre_cfg)
    log(f"  {len(data.actions)} transitions collected")

    # Fit bilinear model
    bilinear = BilinearControlModel(config, sdre_cfg.ridge_alpha)
    bilinear.fit(data)
    ev = bilinear.explained_variance()

    env_dt = config.control.horizon / 30

    # Build ActionPCA model
    pca = ActionPCAModel(config, sdre_cfg.ridge_alpha)
    pca.fit(bilinear, gamma, config.heston.xi, env_dt)

    # --- Section 1: Per-heuristic projection fractions vs rank ---
    log(f"\n{'='*70}")
    log(f"  Per-heuristic projection fraction")
    log(f"{'='*70}")

    for method_name, method_label in [("bilinear", "BilinearSVD"), ("action_pca", "ActionPCA")]:
        log(f"\n  Method: {method_label}")
        header = f"  {'Rank':>5s}" + "".join(f"  {n:>14s}" for n in h_names)
        log(header)

        for rank in ranks:
            if method_name == "bilinear":
                bilinear.reduce(rank)
                U_r = bilinear.U_r
            else:
                pca.reduce(rank)
                U_r = pca.U_r

            fracs = [projection_fraction(U_r, heuristics[n]) for n in h_names]
            row = f"  {rank:>5d}" + "".join(f"  {f:>14.3f}" for f in fracs)
            log(row)

    # --- Section 2: Principal angles between learned subspace and heuristic span ---
    log(f"\n{'='*70}")
    log(f"  Principal angles (degrees) between learned and heuristic spans")
    log(f"{'='*70}")

    for method_name, method_label in [("bilinear", "BilinearSVD"), ("action_pca", "ActionPCA")]:
        log(f"\n  Method: {method_label}")
        for rank in ranks:
            if method_name == "bilinear":
                bilinear.reduce(rank)
                U_r = bilinear.U_r
            else:
                pca.reduce(rank)
                U_r = pca.U_r

            angles = principal_angles(U_r, H_mat)
            angles_deg = np.degrees(angles)
            n_angles = min(rank, H_mat.shape[1])
            log(f"    rank={rank}: {angles_deg[:n_angles]}")

    # --- Section 3: BBG action reconstruction in learned subspace ---
    log(f"\n{'='*70}")
    log(f"  BBG action reconstruction quality")
    log(f"{'='*70}")

    # Solve HJB for BBG controller
    log("\n  Solving HJB for reference actions...")
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # Collect BBG actions on held-out states
    u_baseline = np.concatenate([rn_dists, rn_dists])
    n_opt = config.book.n_options
    bbg_actions = []
    for seed in range(50):
        env = OptionBookMarketMakingEnv(config, seed=seed + 2000)
        state = env.reset()
        while not state.done:
            a = bbg_ctrl(state)
            u = np.concatenate([a.bid_distances, a.ask_distances])
            # Only use non-censored actions
            if np.all(u < 1e5):
                bbg_actions.append(u - u_baseline)
            state, _, _, _ = env.step(a)

    if bbg_actions:
        bbg_mat = np.array(bbg_actions)  # (M, 40) perturbations from baseline
        log(f"  Collected {len(bbg_actions)} non-censored BBG action vectors")

        for method_name, method_label in [("bilinear", "BilinearSVD"), ("action_pca", "ActionPCA")]:
            log(f"\n  Method: {method_label}")
            log(f"  {'Rank':>5s}  {'Recon R²':>10s}  {'Recon RMSE':>12s}  {'Mean cos':>10s}")

            for rank in ranks:
                if method_name == "bilinear":
                    bilinear.reduce(rank)
                    U_r = bilinear.U_r
                else:
                    pca.reduce(rank)
                    U_r = pca.U_r

                # Project BBG actions onto learned subspace and back
                projected = bbg_mat @ U_r @ U_r.T   # (M, 40)
                residual = bbg_mat - projected

                # R²
                ss_res = np.sum(residual ** 2)
                ss_tot = np.sum(bbg_mat ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

                # RMSE
                rmse = np.sqrt(np.mean(residual ** 2))

                # Mean cosine similarity
                cos_sims = []
                for i in range(len(bbg_mat)):
                    u_orig = bbg_mat[i]
                    u_proj = projected[i]
                    n1, n2 = np.linalg.norm(u_orig), np.linalg.norm(u_proj)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_sims.append(np.dot(u_orig, u_proj) / (n1 * n2))
                mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

                log(f"  {rank:>5d}  {r2:>10.4f}  {rmse:>12.6f}  {mean_cos:>10.4f}")
    else:
        log("  No non-censored BBG actions collected.")

    # --- Section 4: Explained variance summary ---
    log(f"\n{'='*70}")
    log(f"  Bilinear SVD explained variance")
    log(f"{'='*70}")
    for i, e in enumerate(ev[:10]):
        log(f"    Direction {i+1}: {e:.4f}  (cumulative: {ev[:i+1].sum():.4f})")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_action_subspace_validation_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
