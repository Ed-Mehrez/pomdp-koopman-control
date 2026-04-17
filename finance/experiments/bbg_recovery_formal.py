"""Formal recovery evaluation: anti-triviality, larger-N, grid refinement.

Checks:
  1. Simple baselines (global_width, global_width_skew) vs learned controllers
  2. 500-episode test evaluation with CARA CE as headline
  3. BBG grid refinement (medium / fine / finer)
  4. Data for interpretability plots (saved to results/)

Train/test split:
  - Train seeds: 0-99 (for fitting simple baselines)
  - SDRE exploration: env seeds 0-499, explore_seed=42 (separate randomness)
  - Test seeds: 2000-2499 (500 episodes, disjoint from all training)

Usage:
    python finance/experiments/bbg_recovery_formal.py
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
from applications.option_mm_bbg.pricing import bs_call_price, bs_call_vega_sqrt_nu
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
from applications.option_mm_bbg.heuristic_action_dictionary import (
    build_heuristic_dictionary,
    projection_fraction,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cara_ce(wealths: np.ndarray, gamma: float) -> float:
    if gamma == 0.0:
        return float(np.mean(wealths))
    neg_gw = -gamma * wealths
    mx = float(np.max(neg_gw))
    return -(mx + np.log(np.mean(np.exp(neg_gw - mx)))) / gamma


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
        "ci_95": (float(np.percentile(diffs, 2.5)),
                  float(np.percentile(diffs, 97.5))),
    }


# ---------------------------------------------------------------------------
# Simple baseline controllers
# ---------------------------------------------------------------------------


def _vega_weights(config: BBGBenchmarkConfig) -> np.ndarray:
    """Per-option z_i V_i weights, normalized to unit norm."""
    h = config.heston
    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in config.book.options
    ])
    prices = np.array([
        bs_call_price(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in config.book.options
    ])
    trade_sizes = np.array([config.liquidity.trade_size(p) for p in prices])
    w = trade_sizes * vegas
    return w / (np.linalg.norm(w) + 1e-15)


def make_global_width_controller(config, rn_distances, alpha):
    """All quotes scaled: delta_i = rn_dist_i * (1 + alpha)."""
    scaled = np.maximum(rn_distances * (1.0 + alpha), 1e-6)

    def ctrl(state, history=None):
        return OptionBookMMAction(
            bid_distances=scaled.copy(),
            ask_distances=scaled.copy(),
            hedge_trade=-state.net_delta,
        )
    return ctrl


def make_global_width_skew_controller(config, rn_distances, alpha, beta):
    """Width + vega-proportional skew, state-dependent on V^pi.

    bid_i = rn_i * (1 + alpha + beta * vpi_norm * w_i)
    ask_i = rn_i * (1 + alpha - beta * vpi_norm * w_i)

    When V^pi > 0 (long vega), widens bids and tightens asks to reduce exposure.
    """
    w = _vega_weights(config)
    vega_limit = config.control.vega_limit

    def ctrl(state, history=None):
        vpi_norm = state.portfolio_vega / vega_limit
        bid_mult = 1.0 + alpha + beta * vpi_norm * w
        ask_mult = 1.0 + alpha - beta * vpi_norm * w
        return OptionBookMMAction(
            bid_distances=np.maximum(rn_distances * bid_mult, 1e-6),
            ask_distances=np.maximum(rn_distances * ask_mult, 1e-6),
            hedge_trade=-state.net_delta,
        )
    return ctrl


def _eval_controller_seeds(config, ctrl, seeds):
    """Run controller on given seeds, return wealth array."""
    wealths = []
    for seed in seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            state, _, _, _ = env.step(ctrl(state))
        wealths.append(state.wealth)
    return np.array(wealths)


def fit_global_width(config, rn_distances, train_seeds, gamma):
    """Grid search over alpha to maximize CARA CE on train split."""
    alphas = np.linspace(-0.3, 3.0, 25)
    best_ce, best_alpha = -np.inf, 0.0
    for alpha in alphas:
        ctrl = make_global_width_controller(config, rn_distances, alpha)
        w = _eval_controller_seeds(config, ctrl, train_seeds)
        ce = cara_ce(w, gamma)
        if ce > best_ce:
            best_ce, best_alpha = ce, alpha
    return best_alpha, best_ce


def fit_global_width_skew(config, rn_distances, train_seeds, gamma):
    """Grid search over (alpha, beta) to maximize CARA CE on train split."""
    alphas = np.linspace(-0.3, 3.0, 15)
    betas = np.linspace(0.0, 5.0, 11)
    best_ce, best_alpha, best_beta = -np.inf, 0.0, 0.0
    for alpha in alphas:
        for beta in betas:
            ctrl = make_global_width_skew_controller(
                config, rn_distances, alpha, beta,
            )
            w = _eval_controller_seeds(config, ctrl, train_seeds)
            ce = cara_ce(w, gamma)
            if ce > best_ce:
                best_ce, best_alpha, best_beta = ce, alpha, beta
    return best_alpha, best_beta, best_ce


# ---------------------------------------------------------------------------
# SDRE controller builder (shared exploration)
# ---------------------------------------------------------------------------


def build_sdre_controller(config, rn_distances, bilinear_model, method, rank):
    """Build controller from a pre-fitted bilinear model."""
    n_opt = config.book.n_options
    h = config.heston
    env_dt = config.control.horizon / 30
    gamma = config.control.gamma

    if method == "action_pca":
        model = ActionPCAModel(config)
        model.fit(bilinear_model, gamma, h.xi, env_dt)
        model.reduce(rank)
        model.vega_channel = bilinear_model.vega_channel
        model.rev_linear = bilinear_model.rev_linear
        model.rev_quad = bilinear_model.rev_quad
    elif method == "bilinear_2stage":
        overspace = min(10, 2 * n_opt)
        bilinear_model.reduce(overspace)
        U_over = bilinear_model.U_r.copy()

        c_pen = gamma * h.xi ** 2 / 8.0 * env_dt
        vc_proj = U_over.T @ bilinear_model.vega_channel
        rq_proj = U_over.T @ np.diag(bilinear_model.rev_quad) @ U_over
        H_over = rq_proj - c_pen * np.outer(vc_proj, vc_proj)

        eigvals, eigvecs = np.linalg.eigh(H_over)
        idx = np.argsort(eigvals)
        k = min(rank, len(eigvals))
        V_inner = eigvecs[:, idx[:k]]
        U_final = U_over @ V_inner
        norms = np.linalg.norm(U_final, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-15)

        bilinear_model.U_r = U_final / norms
        bilinear_model.S_r = np.abs(eigvals[idx[:k]])
        model = bilinear_model
    else:
        raise ValueError(method)

    u_baseline = np.concatenate([rn_distances, rn_distances])
    U_r = model.U_r
    vc = model.vega_channel
    rl = model.rev_linear
    rq = model.rev_quad
    xi = h.xi
    max_pert = 0.8

    def ctrl(state, history=None):
        a_star = _sdre_solve(
            state.portfolio_vega, U_r, vc, rl, rq,
            gamma, xi, env_dt,
        )
        u_delta = U_r @ a_star
        u_delta = np.clip(u_delta, -max_pert * u_baseline, max_pert * u_baseline)
        u_full = u_baseline + u_delta
        return OptionBookMMAction(
            bid_distances=np.maximum(u_full[:n_opt], 1e-6),
            ask_distances=np.maximum(u_full[n_opt:], 1e-6),
            hedge_trade=-state.net_delta,
        )

    return ctrl, model


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
    rn_dists = _compute_rn_distances(config)
    n_opt = config.book.n_options

    train_seeds = list(range(100))
    test_seeds = list(range(2000, 2500))
    n_test = len(test_seeds)

    log("=" * 70)
    log("  Formal Recovery Evaluation")
    log("=" * 70)
    log(f"  Train seeds: {len(train_seeds)}, Test seeds: {n_test}")
    log(f"  Gamma: {gamma}")

    # ==================================================================
    # CHECK 3: BBG grid refinement
    # ==================================================================
    log(f"\n{'='*70}")
    log("  Check 3: BBG Grid Refinement")
    log(f"{'='*70}")

    grids = {
        "medium": (60, 15, 30),
        "fine":   (120, 20, 40),
        "finer":  (180, 25, 50),
    }

    bbg_ctrls = {}
    for label, (n_t, n_nu, n_vpi) in grids.items():
        t0 = time.time()
        log(f"\n  Solving HJB [{label}] (grid {n_t}x{n_nu}x{n_vpi})...")
        t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
            config, n_nu=n_nu, n_vpi=n_vpi, n_time=n_t,
        )
        log(f"    Solved in {time.time() - t0:.1f}s")
        bbg_ctrls[label] = make_bbg_numerical_controller(
            config, values, t_grid, nu_grid, vpi_grid,
        )

    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # ==================================================================
    # CHECK 1: Simple baselines
    # ==================================================================
    log(f"\n{'='*70}")
    log("  Check 1: Fitting Simple Baselines (train split)")
    log(f"{'='*70}")

    t0 = time.time()
    best_alpha_w, train_ce_w = fit_global_width(
        config, rn_dists, train_seeds, gamma,
    )
    log(f"\n  global_width: alpha={best_alpha_w:.3f}, train CE={train_ce_w:.0f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    best_alpha_ws, best_beta_ws, train_ce_ws = fit_global_width_skew(
        config, rn_dists, train_seeds, gamma,
    )
    log(f"  global_width_skew: alpha={best_alpha_ws:.3f}, beta={best_beta_ws:.3f}, "
        f"train CE={train_ce_ws:.0f} ({time.time()-t0:.1f}s)")

    width_ctrl = make_global_width_controller(config, rn_dists, best_alpha_w)
    skew_ctrl = make_global_width_skew_controller(
        config, rn_dists, best_alpha_ws, best_beta_ws,
    )

    # ==================================================================
    # SDRE controllers (shared exploration data)
    # ==================================================================
    log(f"\n{'='*70}")
    log("  Training SDRE Controllers")
    log(f"{'='*70}")

    sdre_cfg = SDRERecoveryConfig(n_explore_episodes=500)
    t0 = time.time()
    log("\n  Collecting exploration data (500 episodes)...")
    data = collect_exploration_data(config, rn_dists, sdre_cfg)
    log(f"    {len(data.actions)} transitions in {time.time()-t0:.1f}s")

    bilinear = BilinearControlModel(config, sdre_cfg.ridge_alpha)
    bilinear.fit(data)
    ev = bilinear.explained_variance()
    log(f"    EV top-5: {ev[:5]}")

    env_dt = config.control.horizon / 30

    # Build ActionPCA r1, r3
    sdre_controllers = {}
    for method, ranks in [("action_pca", [1, 3]), ("bilinear_2stage", [1, 3])]:
        for rank in ranks:
            # Re-fit bilinear for each 2stage call since reduce() mutates
            bl = BilinearControlModel(config, sdre_cfg.ridge_alpha)
            bl.fit(data)
            label = f"{method}_r{rank}"
            log(f"  Building {label}...")
            ctrl, mdl = build_sdre_controller(config, rn_dists, bl, method, rank)
            sdre_controllers[label] = (ctrl, mdl)

    # ==================================================================
    # CHECK 2: Larger paired evaluation on TEST split
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"  Check 2: Test Evaluation ({n_test} episodes)")
    log(f"{'='*70}")

    all_controllers = {
        "risk_neutral": rn_ctrl,
        "bbg_medium": bbg_ctrls["medium"],
        "bbg_fine": bbg_ctrls["fine"],
        "bbg_finer": bbg_ctrls["finer"],
        "global_width": width_ctrl,
        "global_width_skew": skew_ctrl,
    }
    for label, (ctrl, _) in sdre_controllers.items():
        all_controllers[label] = ctrl

    results = {}
    for label, ctrl in all_controllers.items():
        t0 = time.time()
        w = _eval_controller_seeds(config, ctrl, test_seeds)
        elapsed = time.time() - t0
        ce = cara_ce(w, gamma)
        results[label] = w
        log(f"  {label:<25s} CE={ce:>10.0f}  mean={w.mean():>10.0f}  "
            f"std={w.std():>8.0f}  ({elapsed:.1f}s)")

    # ==================================================================
    # Paired contrasts
    # ==================================================================
    log(f"\n{'='*70}")
    log("  CARA CE Paired Contrasts (bootstrap, 10K resamples)")
    log(f"{'='*70}")

    key_contrasts = [
        # Check 1: Anti-triviality
        ("action_pca_r1 - risk_neutral", "action_pca_r1", "risk_neutral"),
        ("action_pca_r1 - global_width", "action_pca_r1", "global_width"),
        ("action_pca_r1 - global_width_skew", "action_pca_r1", "global_width_skew"),
        ("bilinear_2stage_r3 - global_width", "bilinear_2stage_r3", "global_width"),
        ("bilinear_2stage_r3 - global_width_skew", "bilinear_2stage_r3", "global_width_skew"),
        ("bilinear_2stage_r1 - global_width", "bilinear_2stage_r1", "global_width"),
        # Check 2: vs benchmark
        ("action_pca_r1 - bbg_fine", "action_pca_r1", "bbg_fine"),
        ("bilinear_2stage_r3 - bbg_fine", "bilinear_2stage_r3", "bbg_fine"),
        ("action_pca_r3 - bbg_fine", "action_pca_r3", "bbg_fine"),
        ("bilinear_2stage_r1 - bbg_fine", "bilinear_2stage_r1", "bbg_fine"),
        # Check 3: Grid refinement
        ("action_pca_r1 - bbg_medium", "action_pca_r1", "bbg_medium"),
        ("action_pca_r1 - bbg_finer", "action_pca_r1", "bbg_finer"),
        # Simple baselines vs benchmark
        ("global_width - risk_neutral", "global_width", "risk_neutral"),
        ("global_width_skew - risk_neutral", "global_width_skew", "risk_neutral"),
        ("global_width_skew - bbg_fine", "global_width_skew", "bbg_fine"),
    ]

    log(f"\n  {'Contrast':<42s} {'mean':>8s} {'sd':>8s} {'P>0':>7s} {'95% CrI':>22s}")
    log(f"  {'-'*87}")

    for label, a, b in key_contrasts:
        boot = bootstrap_ce_diff(results[a], results[b], gamma)
        ci = boot["ci_95"]
        log(f"  {label:<42s} {boot['mean']:>8.0f} {boot['sd_post']:>8.0f} "
            f"{boot['p_pos']:>7.4f} [{ci[0]:>9.0f}, {ci[1]:>9.0f}]")

    # ==================================================================
    # CHECK 3 summary: grid refinement table
    # ==================================================================
    log(f"\n{'='*70}")
    log("  Check 3: Grid Refinement Summary")
    log(f"{'='*70}")
    log(f"\n  {'Grid':<10s} {'BBG CE':>10s} {'APr1 CE':>10s} {'APr1-BBG':>10s} {'2Sr3 CE':>10s} {'2Sr3-BBG':>10s}")

    ce_ap1 = cara_ce(results["action_pca_r1"], gamma)
    ce_2s3 = cara_ce(results["bilinear_2stage_r3"], gamma)
    for g in ["medium", "fine", "finer"]:
        ce_bbg = cara_ce(results[f"bbg_{g}"], gamma)
        log(f"  {g:<10s} {ce_bbg:>10.0f} {ce_ap1:>10.0f} {ce_ap1-ce_bbg:>10.0f} "
            f"{ce_2s3:>10.0f} {ce_2s3-ce_bbg:>10.0f}")

    # ==================================================================
    # CHECK 4: Interpretability data
    # ==================================================================
    log(f"\n{'='*70}")
    log("  Check 4: Interpretability")
    log(f"{'='*70}")

    heuristics = build_heuristic_dictionary(config)
    strikes = np.array([o.strike for o in config.book.options])
    mats = np.array([o.maturity for o in config.book.options])
    unique_strikes = sorted(set(strikes))
    unique_mats = sorted(set(mats))

    for sdre_label in ["action_pca_r1", "action_pca_r3",
                        "bilinear_2stage_r1", "bilinear_2stage_r3"]:
        _, mdl = sdre_controllers[sdre_label]
        U_r = mdl.U_r
        rank = U_r.shape[1]
        log(f"\n  {sdre_label}:")

        # Projection fractions
        for h_name, h_dir in heuristics.items():
            pf = projection_fraction(U_r, h_dir)
            log(f"    proj({h_name}): {pf:.4f}")

        # Show each learned direction
        for d in range(rank):
            direction = U_r[:, d]
            bid_part = direction[:n_opt]
            ask_part = direction[n_opt:]

            # Summarize: mean bid shift, mean ask shift
            log(f"    dir {d+1}: bid_mean={bid_part.mean():.4f} ask_mean={ask_part.mean():.4f} "
                f"bid_std={bid_part.std():.4f} ask_std={ask_part.std():.4f}")

    # ==================================================================
    # CHECK 5: Classification
    # ==================================================================
    log(f"\n{'='*70}")
    log("  Check 5: Outcome Classification")
    log(f"{'='*70}")

    # Gather key numbers
    ce_rn = cara_ce(results["risk_neutral"], gamma)
    ce_bbg_fine = cara_ce(results["bbg_fine"], gamma)
    ce_bbg_finer = cara_ce(results["bbg_finer"], gamma)
    ce_gw = cara_ce(results["global_width"], gamma)
    ce_gws = cara_ce(results["global_width_skew"], gamma)
    ce_ap1 = cara_ce(results["action_pca_r1"], gamma)
    ce_2s3 = cara_ce(results["bilinear_2stage_r3"], gamma)

    # Anti-triviality: does learned beat width+skew?
    boot_ap1_gws = bootstrap_ce_diff(results["action_pca_r1"],
                                      results["global_width_skew"], gamma)
    boot_2s3_gws = bootstrap_ce_diff(results["bilinear_2stage_r3"],
                                      results["global_width_skew"], gamma)
    # vs refined benchmark
    boot_ap1_finer = bootstrap_ce_diff(results["action_pca_r1"],
                                        results["bbg_finer"], gamma)
    boot_2s3_finer = bootstrap_ce_diff(results["bilinear_2stage_r3"],
                                        results["bbg_finer"], gamma)

    beats_skew = boot_ap1_gws["p_pos"] > 0.90 or boot_2s3_gws["p_pos"] > 0.90
    beats_finer_bbg = boot_ap1_finer["p_pos"] > 0.90 or boot_2s3_finer["p_pos"] > 0.90

    log(f"\n  CE summary:")
    log(f"    RN:               {ce_rn:.0f}")
    log(f"    BBG fine:         {ce_bbg_fine:.0f}")
    log(f"    BBG finer:        {ce_bbg_finer:.0f}")
    log(f"    global_width:     {ce_gw:.0f}")
    log(f"    global_width_skew:{ce_gws:.0f}")
    log(f"    ActionPCA r1:     {ce_ap1:.0f}")
    log(f"    Bilinear2S r3:    {ce_2s3:.0f}")

    log(f"\n  Anti-triviality (learned vs width+skew):")
    log(f"    APr1 - gws: mean={boot_ap1_gws['mean']:.0f}, P>0={boot_ap1_gws['p_pos']:.4f}")
    log(f"    2Sr3 - gws: mean={boot_2s3_gws['mean']:.0f}, P>0={boot_2s3_gws['p_pos']:.4f}")

    log(f"\n  Grid stability (learned vs finer BBG):")
    log(f"    APr1 - bbg_finer: mean={boot_ap1_finer['mean']:.0f}, P>0={boot_ap1_finer['p_pos']:.4f}")
    log(f"    2Sr3 - bbg_finer: mean={boot_2s3_finer['mean']:.0f}, P>0={boot_2s3_finer['p_pos']:.4f}")

    if beats_finer_bbg and beats_skew:
        outcome = "A"
        desc = ("Benchmark-beating: learned controller beats refined BBG AND "
                "simple baselines on CARA CE.")
    elif beats_skew:
        outcome = "B"
        desc = ("Strong low-rank recovery: learned controller exceeds simple baselines "
                "but advantage over refined BBG is not decisive.")
    else:
        outcome = "C"
        desc = ("Mostly retuning: simple global width/skew baselines explain "
                "most or all of the apparent gain.")

    log(f"\n  *** OUTCOME: {outcome} ***")
    log(f"  {desc}")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_recovery_formal_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")

    # Save interpretability data for plotting
    np.savez(
        results_dir / f"bbg_recovery_formal_data_{date.today()}.npz",
        strikes=strikes, mats=mats,
        unique_strikes=np.array(unique_strikes),
        unique_mats=np.array(unique_mats),
        rn_dists=rn_dists,
        **{f"U_r_{k}": mdl.U_r for k, (_, mdl) in sdre_controllers.items()},
        **{f"wealth_{k}": v for k, v in results.items()},
    )
    log(f"Saved data to {results_dir / f'bbg_recovery_formal_data_{date.today()}.npz'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
