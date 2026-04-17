"""Rank-1 and rank-2 interpretability plots for the learned action subspace.

Generates:
  1. Learned action directions as strike x maturity heatmaps (bid/ask sides)
  2. Heuristic alignment comparison after Procrustes rotation
  3. BBG action reconstruction quality at representative states

Requires the formal evaluation data (.npz) from bbg_recovery_formal.py.

Usage:
    python finance/experiments/bbg_rank_interpretability.py
"""

from __future__ import annotations

import sys
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
    projection_fraction,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _reshape_direction(direction, n_opt, strikes, mats, unique_strikes, unique_mats):
    """Reshape a 40D direction into two 2D grids (bid, ask) over strike x mat."""
    bid = direction[:n_opt]
    ask = direction[n_opt:]
    ns, nm = len(unique_strikes), len(unique_mats)
    bid_grid = np.full((ns, nm), np.nan)
    ask_grid = np.full((ns, nm), np.nan)
    for i in range(n_opt):
        si = unique_strikes.index(strikes[i])
        mi = unique_mats.index(mats[i])
        bid_grid[si, mi] = bid[i]
        ask_grid[si, mi] = ask[i]
    return bid_grid, ask_grid


def plot_direction_heatmaps(U_r, label, n_opt, strikes, mats,
                             unique_strikes, unique_mats, save_dir):
    """Plot each learned direction as bid/ask heatmaps."""
    rank = U_r.shape[1]
    for d in range(rank):
        direction = U_r[:, d]
        bid_grid, ask_grid = _reshape_direction(
            direction, n_opt, strikes, mats, unique_strikes, unique_mats,
        )
        vmax = max(np.nanmax(np.abs(bid_grid)), np.nanmax(np.abs(ask_grid)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{label} — Direction {d+1}", fontsize=13)

        im1 = ax1.imshow(bid_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          aspect="auto", origin="lower")
        ax1.set_title("Bid side")
        ax1.set_xlabel("Maturity")
        ax1.set_ylabel("Strike")
        ax1.set_xticks(range(len(unique_mats)))
        ax1.set_xticklabels([f"{m:.1f}" for m in unique_mats])
        ax1.set_yticks(range(len(unique_strikes)))
        ax1.set_yticklabels([f"{s:.0f}" for s in unique_strikes])

        im2 = ax2.imshow(ask_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          aspect="auto", origin="lower")
        ax2.set_title("Ask side")
        ax2.set_xlabel("Maturity")
        ax2.set_xticks(range(len(unique_mats)))
        ax2.set_xticklabels([f"{m:.1f}" for m in unique_mats])
        ax2.set_yticks(range(len(unique_strikes)))
        ax2.set_yticklabels([f"{s:.0f}" for s in unique_strikes])

        fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Direction weight")
        fig.tight_layout()
        fname = save_dir / f"{label}_dir{d+1}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_heuristic_comparison(U_r, label, heuristics, n_opt, strikes, mats,
                               unique_strikes, unique_mats, save_dir):
    """Bar chart of projection fractions for each heuristic direction."""
    h_names = list(heuristics.keys())
    fracs = [projection_fraction(U_r, heuristics[n]) for n in h_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(h_names, fracs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylabel("Projection fraction")
    ax.set_title(f"{label} — Heuristic alignment (rank {U_r.shape[1]})")
    ax.set_ylim(0, 1)
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.02, f"{f:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fname = save_dir / f"{label}_heuristic_alignment.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_bbg_reconstruction(U_r, label, bbg_ctrl, rn_dists, config,
                             n_opt, strikes, mats, unique_strikes,
                             unique_mats, save_dir):
    """Project BBG actions onto learned subspace at a few states."""
    u_baseline = np.concatenate([rn_dists, rn_dists])
    # Collect a few representative BBG actions
    env = OptionBookMarketMakingEnv(config, seed=5000)
    state = env.reset()
    states_to_plot = []
    actions_to_plot = []
    step = 0
    while not state.done:
        a = bbg_ctrl(state)
        u = np.concatenate([a.bid_distances, a.ask_distances])
        if step in [0, 10, 20]:
            states_to_plot.append(state)
            actions_to_plot.append(u)
        state, _, _, _ = env.step(a)
        step += 1

    for i, (st, u_bbg) in enumerate(zip(states_to_plot, actions_to_plot)):
        u_pert = u_bbg - u_baseline
        # Project onto learned subspace
        Q, _ = np.linalg.qr(U_r, mode="reduced")
        u_proj = Q @ (Q.T @ u_pert)
        u_resid = u_pert - u_proj

        # Reshape for plotting
        pert_bid, pert_ask = _reshape_direction(
            u_pert, n_opt, strikes, mats, unique_strikes, unique_mats)
        proj_bid, proj_ask = _reshape_direction(
            u_proj, n_opt, strikes, mats, unique_strikes, unique_mats)
        resid_bid, resid_ask = _reshape_direction(
            u_resid, n_opt, strikes, mats, unique_strikes, unique_mats)

        vmax = np.nanmax(np.abs(np.concatenate([
            pert_bid.ravel(), pert_ask.ravel()])))
        if vmax < 1e-10:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f"{label} — BBG action reconstruction (step {[0,10,20][i]}, "
                     f"V^pi={st.portfolio_vega:.0f})", fontsize=12)

        titles_row = ["BBG perturbation", "Learned projection", "Residual"]
        for col, (bg, ag, title) in enumerate([
            (pert_bid, pert_ask, "BBG perturbation"),
            (proj_bid, proj_ask, "Learned projection"),
            (resid_bid, resid_ask, "Residual"),
        ]):
            axes[0, col].imshow(bg, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                 aspect="auto", origin="lower")
            axes[0, col].set_title(f"{title} (bid)")
            axes[1, col].imshow(ag, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                 aspect="auto", origin="lower")
            axes[1, col].set_title(f"{title} (ask)")
            for row in range(2):
                axes[row, col].set_xticks(range(len(unique_mats)))
                axes[row, col].set_xticklabels([f"{m:.1f}" for m in unique_mats])
                axes[row, col].set_yticks(range(len(unique_strikes)))
                axes[row, col].set_yticklabels([f"{s:.0f}" for s in unique_strikes])

        r2 = 1 - np.sum(u_resid**2) / (np.sum(u_pert**2) + 1e-30)
        fig.text(0.5, 0.01, f"Reconstruction R^2 = {r2:.3f}", ha="center", fontsize=11)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = save_dir / f"{label}_bbg_recon_step{[0,10,20][i]}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  Saved {fname}")


def main() -> int:
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return 1

    config = BBGBenchmarkConfig.paper_default()
    rn_dists = _compute_rn_distances(config)
    n_opt = config.book.n_options
    gamma = config.control.gamma

    strikes = np.array([o.strike for o in config.book.options])
    mats = np.array([o.maturity for o in config.book.options])
    unique_strikes = sorted(set(strikes))
    unique_mats = sorted(set(mats))

    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)

    heuristics = build_heuristic_dictionary(config)

    # Collect exploration data and fit models
    print("Collecting exploration data...")
    sdre_cfg = SDRERecoveryConfig(n_explore_episodes=500)
    data = collect_exploration_data(config, rn_dists, sdre_cfg)
    env_dt = config.control.horizon / 30

    # Solve HJB for BBG reconstruction
    print("Solving HJB for BBG reference actions...")
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)

    for method, rank, label in [
        ("action_pca", 1, "action_pca_r1"),
        ("action_pca", 3, "action_pca_r3"),
        ("bilinear_2stage", 1, "bilinear_2stage_r1"),
        ("bilinear_2stage", 3, "bilinear_2stage_r3"),
    ]:
        print(f"\n--- {label} ---")
        bl = BilinearControlModel(config, sdre_cfg.ridge_alpha)
        bl.fit(data)

        if method == "action_pca":
            mdl = ActionPCAModel(config)
            mdl.fit(bl, gamma, config.heston.xi, env_dt)
            mdl.reduce(rank)
        elif method == "bilinear_2stage":
            overspace = min(10, 2 * n_opt)
            bl.reduce(overspace)
            U_over = bl.U_r.copy()
            c_pen = gamma * config.heston.xi ** 2 / 8.0 * env_dt
            vc_proj = U_over.T @ bl.vega_channel
            rq_proj = U_over.T @ np.diag(bl.rev_quad) @ U_over
            H_over = rq_proj - c_pen * np.outer(vc_proj, vc_proj)
            eigvals, eigvecs = np.linalg.eigh(H_over)
            idx = np.argsort(eigvals)
            k = min(rank, len(eigvals))
            V_inner = eigvecs[:, idx[:k]]
            U_final = U_over @ V_inner
            norms = np.linalg.norm(U_final, axis=0, keepdims=True)
            bl.U_r = U_final / np.maximum(norms, 1e-15)
            mdl = bl

        U_r = mdl.U_r

        # Plot 1: Direction heatmaps
        plot_direction_heatmaps(
            U_r, label, n_opt, strikes, mats,
            unique_strikes, unique_mats, results_dir,
        )

        # Plot 2: Heuristic alignment
        plot_heuristic_comparison(
            U_r, label, heuristics, n_opt, strikes, mats,
            unique_strikes, unique_mats, results_dir,
        )

        # Plot 3: BBG action reconstruction
        plot_bbg_reconstruction(
            U_r, label, bbg_ctrl, rn_dists, config,
            n_opt, strikes, mats, unique_strikes, unique_mats, results_dir,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
