"""Rank sweep for the SDRE recovery controller.

Tests both bilinear (Option 2) and action_pca (Option 1) at ranks
r = 1, 2, 3, 5, 8.  Reports performance, runtime, and action
reconstruction quality for each rank.

Usage:
    python finance/experiments/bbg_sdre_rank_sweep.py
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
    collect_exploration_data,
    BilinearControlModel,
    ActionPCAModel,
    _compute_rn_distances,
    _sdre_solve,
)


def cara_ce(wealths: np.ndarray, gamma: float) -> float:
    if gamma == 0.0:
        return float(np.mean(wealths))
    neg_gw = -gamma * wealths
    mx = float(np.max(neg_gw))
    return -(mx + np.log(np.mean(np.exp(neg_gw - mx)))) / gamma


def mean_var_surrogate(wealths: np.ndarray, gamma: float) -> float:
    return float(np.mean(wealths) - 0.5 * gamma * np.var(wealths))


def run_episodes(config, ctrl, n_episodes=100):
    wealths = []
    for seed in range(n_episodes):
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            state, _, _, _ = env.step(ctrl(state))
        wealths.append(state.wealth)
    return np.array(wealths)


def action_reconstruction_quality(
    config, bbg_ctrl, sdre_ctrl, n_episodes=20
):
    """Cosine similarity between BBG and SDRE actions on held-out states."""
    cos_sims = []
    for seed in range(n_episodes):
        env = OptionBookMarketMakingEnv(config, seed=seed + 1000)
        state = env.reset()
        while not state.done:
            a_bbg = bbg_ctrl(state)
            a_sdre = sdre_ctrl(state)
            u_bbg = np.concatenate([a_bbg.bid_distances, a_bbg.ask_distances])
            u_sdre = np.concatenate([a_sdre.bid_distances, a_sdre.ask_distances])
            # Cosine similarity (ignoring censored options with 1e6 distance)
            mask = (u_bbg < 1e5) & (u_sdre < 1e5)
            if mask.sum() > 0:
                ub = u_bbg[mask]
                us = u_sdre[mask]
                dot = np.dot(ub, us)
                n1 = np.linalg.norm(ub)
                n2 = np.linalg.norm(us)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_sims.append(dot / (n1 * n2))
            state, _, _, _ = env.step(a_bbg)
    return float(np.mean(cos_sims)) if cos_sims else 0.0


def main() -> int:
    out: list[str] = []
    def log(msg=""):
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma
    ranks = [1, 2, 3, 5, 8]
    n_eval = 100

    log("=" * 70)
    log("  SDRE Recovery Rank Sweep")
    log("=" * 70)

    # Solve HJB for reference
    log("\nSolving HJB...")
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    log("Running baselines...")
    w_rn = run_episodes(config, rn_ctrl, n_eval)
    w_bbg = run_episodes(config, bbg_ctrl, n_eval)
    ce_rn = cara_ce(w_rn, gamma)
    ce_bbg = cara_ce(w_bbg, gamma)
    gap = ce_bbg - ce_rn
    log(f"  RN CE:  {ce_rn:.0f}")
    log(f"  BBG CE: {ce_bbg:.0f}")
    log(f"  Gap:    {gap:.0f}")

    # Pre-collect exploration data (shared across ranks)
    rn_dists = _compute_rn_distances(config)
    sdre_base = SDRERecoveryConfig(n_explore_episodes=500)
    log("\nCollecting exploration data (500 episodes)...")
    t0 = time.time()
    data = collect_exploration_data(config, rn_dists, sdre_base)
    log(f"  Collected {len(data.actions)} transitions in {time.time() - t0:.1f}s")

    # Fit bilinear model once
    bilinear = BilinearControlModel(config, sdre_base.ridge_alpha)
    bilinear.fit(data)
    ev = bilinear.explained_variance()
    log(f"  Explained variance: {ev[:10]}")

    env_dt = config.control.horizon / 30

    # Rank sweep
    log(f"\n{'='*70}")
    log(f"  {'Method':<20s} {'Rank':>5s} {'CE':>10s} {'GC%':>8s} {'CosSim':>8s} {'Time':>6s}")
    log(f"{'='*70}")

    for method in ["bilinear", "action_pca"]:
        for rank in ranks:
            t0 = time.time()

            if method == "bilinear":
                bilinear.reduce(rank)
                model = bilinear
            else:
                pca = ActionPCAModel(config, sdre_base.ridge_alpha)
                pca.fit(bilinear, gamma, config.heston.xi, env_dt)
                pca.reduce(rank)
                model = pca
                model.vega_channel = bilinear.vega_channel
                model.rev_linear = bilinear.rev_linear
                model.rev_quad = bilinear.rev_quad

            # Build controller from pre-fitted model
            n_opt = config.book.n_options
            u_baseline = np.concatenate([rn_dists, rn_dists])
            U_r = model.U_r
            vc = model.vega_channel
            rl = model.rev_linear
            rq = model.rev_quad
            xi = config.heston.xi

            max_pert_frac = 0.8

            def _make_ctrl(U_r=U_r, vc=vc, rl=rl, rq=rq):
                def ctrl(state, history=None):
                    from applications.option_mm_bbg.env import OptionBookMMAction
                    a_star = _sdre_solve(
                        state.portfolio_vega, U_r, vc, rl, rq,
                        gamma, xi, env_dt,
                    )
                    u_delta = U_r @ a_star
                    u_delta = np.clip(u_delta, -max_pert_frac * u_baseline,
                                      max_pert_frac * u_baseline)
                    u_full = u_baseline + u_delta
                    return OptionBookMMAction(
                        bid_distances=np.maximum(u_full[:n_opt], 1e-6),
                        ask_distances=np.maximum(u_full[n_opt:], 1e-6),
                        hedge_trade=-state.net_delta,
                    )
                return ctrl

            ctrl = _make_ctrl()
            train_time = time.time() - t0

            w = run_episodes(config, ctrl, n_eval)
            ce = cara_ce(w, gamma)
            gc = (ce - ce_rn) / gap if abs(gap) > 1e-6 else float("nan")
            cos_sim = action_reconstruction_quality(config, bbg_ctrl, ctrl)

            label = f"{method}_r{rank}"
            log(f"  {label:<20s} {rank:>5d} {ce:>10.0f} {gc:>7.1%} {cos_sim:>8.3f} {train_time:>5.1f}s")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_sdre_rank_sweep_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
