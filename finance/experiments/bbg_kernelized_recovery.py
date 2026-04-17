"""Kernelized state-conditioned reduced-action recovery experiment.

Tests whether kernelizing the state-to-reduced-coordinate map closes
the systematic ~8K CE gap from the non-kernelized SDRE recovery.

Compares:
  - Non-kernelized SDRE (ActionPCA, bilinear_2stage) at ranks 1-3
  - Kernelized recovery (compact state, rich state) at ranks 1-3
  - BBG benchmark, risk-neutral, simple baselines

All comparisons use the same paired ROPE + precision recovery gate.

Usage:
    python finance/experiments/bbg_kernelized_recovery.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

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
from applications.option_mm.metrics import cara_utility, paired_ce_posterior
from applications.option_mm_bbg.sdre_recovery import (
    SDRERecoveryConfig,
    collect_exploration_data,
    BilinearControlModel,
    ActionPCAModel,
    _compute_rn_distances,
    _sdre_solve,
    _ridge_regression,
    extract_state_features,
    make_kernelized_recovery_controller,
)


# ---------------------------------------------------------------------------
# Paired CE posterior recovery gate (metrics-layer)
# ---------------------------------------------------------------------------


def paired_recovery_gate(
    w_cand: np.ndarray,
    w_bbg: np.ndarray,
    gamma: float,
    h: float,
    s_max: float,
    method: str = "delta",
    n_draws: int = 20_000,
    rng=None,
) -> dict:
    """Recovery gate using the metrics-layer paired CE posterior.

    Returns the same dict interface as the old bootstrap helper so
    downstream reporting code stays unchanged.
    """
    from scipy.stats import norm

    utility = cara_utility(gamma)
    post = paired_ce_posterior(
        w_cand, w_bbg, utility=utility, method=method,
        n_draws=n_draws, rng=rng,
    )
    p_rope = float(
        norm.cdf(h, loc=post.mean, scale=post.sd_post)
        - norm.cdf(-h, loc=post.mean, scale=post.sd_post)
    )
    return {
        "mean": post.mean,
        "sd_post": post.sd_post,
        "p_rope": p_rope,
        "ci_95": (post.ci_low, post.ci_high),
        "passes_a": p_rope >= 0.95,
        "passes_b": post.sd_post <= s_max,
        "passes_both": (p_rope >= 0.95) and (post.sd_post <= s_max),
        "p_positive": post.p_positive,
    }


# ---------------------------------------------------------------------------
# Simple anti-triviality baselines
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
    scaled = np.maximum(rn_distances * (1.0 + alpha), 1e-6)

    def ctrl(state, history=None):
        return OptionBookMMAction(
            bid_distances=scaled.copy(),
            ask_distances=scaled.copy(),
            hedge_trade=-state.net_delta,
        )
    return ctrl


def make_global_width_skew_controller(config, rn_distances, alpha, beta):
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


def fit_global_width(config, rn_distances, train_seeds, gamma):
    alphas = np.linspace(-0.3, 3.0, 25)
    best_ce, best_alpha = -np.inf, 0.0
    for alpha in alphas:
        ctrl = make_global_width_controller(config, rn_distances, alpha)
        w = eval_seeds(config, ctrl, train_seeds)
        ce = cara_ce(w, gamma)
        if ce > best_ce:
            best_ce, best_alpha = ce, alpha
    return best_alpha, best_ce


def fit_global_width_skew(config, rn_distances, train_seeds, gamma):
    alphas = np.linspace(-0.3, 3.0, 15)
    betas = np.linspace(0.0, 5.0, 11)
    best_ce, best_alpha, best_beta = -np.inf, 0.0, 0.0
    for alpha in alphas:
        for beta in betas:
            ctrl = make_global_width_skew_controller(config, rn_distances, alpha, beta)
            w = eval_seeds(config, ctrl, train_seeds)
            ce = cara_ce(w, gamma)
            if ce > best_ce:
                best_ce, best_alpha, best_beta = ce, alpha, beta
    return best_alpha, best_beta, best_ce


# ---------------------------------------------------------------------------
# Metrics + recovery gate
# ---------------------------------------------------------------------------


def cara_ce(wealths: np.ndarray, gamma: float) -> float:
    if gamma == 0.0:
        return float(np.mean(wealths))
    neg_gw = -gamma * wealths
    mx = float(np.max(neg_gw))
    return -(mx + np.log(np.mean(np.exp(neg_gw - mx)))) / gamma


def bootstrap_paired_recovery_gate(w_cand, w_bbg, gamma, h, s_max,
                               n_boot=20_000, seed=999):
    """Legacy bootstrap recovery gate kept only for robustness checks."""
    rng = np.random.default_rng(seed)
    n = len(w_cand)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = cara_ce(w_cand[idx], gamma) - cara_ce(w_bbg[idx], gamma)
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs))
    p_rope = float(np.mean(np.abs(diffs) <= h))
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    return {
        "mean": mean_d, "sd_post": sd_d, "p_rope": p_rope, "ci_95": ci,
        "passes_a": p_rope >= 0.95, "passes_b": sd_d <= s_max,
        "passes_both": (p_rope >= 0.95) and (sd_d <= s_max),
    }


def eval_seeds(config, ctrl, seeds):
    wealths = []
    for seed in seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            state, _, _, _ = env.step(ctrl(state))
        wealths.append(state.wealth)
    return np.array(wealths)


# ---------------------------------------------------------------------------
# SDRE non-kernelized controller builder (from BBG exploration)
# ---------------------------------------------------------------------------


def build_sdre_from_bbg(config, rn_distances, bbg_ctrl, method, rank,
                         n_explore=500, ridge_alpha=1e-3):
    """Non-kernelized SDRE from BBG-regime exploration (baseline comparison)."""
    from applications.option_mm_bbg.sdre_recovery import ExplorationData

    n_opt = config.book.n_options
    h = config.heston

    all_sf, all_u, all_vpi_pre, all_vpi_post = [], [], [], []
    all_dinv, all_spread = [], []
    max_dist = 10.0 * np.max(rn_distances)

    for ep in range(n_explore):
        env = OptionBookMarketMakingEnv(config, seed=ep)
        state = env.reset()
        while not state.done:
            sf = extract_state_features(state, config)
            vpi_pre = state.portfolio_vega
            inv_pre = state.option_inventories.copy()
            action = bbg_ctrl(state)
            u = np.concatenate([action.bid_distances, action.ask_distances])
            u = np.minimum(u, max_dist)
            next_state, _, _, info = env.step(action)
            all_sf.append(sf)
            all_u.append(u)
            all_vpi_pre.append(vpi_pre)
            all_vpi_post.append(next_state.portfolio_vega)
            all_dinv.append(next_state.option_inventories - inv_pre)
            all_spread.append(info["spread_capture"])
            state = next_state

    data = ExplorationData(
        state_features=np.array(all_sf), actions=np.array(all_u),
        vpi_pre=np.array(all_vpi_pre), vpi_post=np.array(all_vpi_post),
        inventory_changes=np.array(all_dinv), spread_captures=np.array(all_spread),
    )

    bilinear = BilinearControlModel(config, ridge_alpha)
    bilinear.fit(data)
    env_dt = config.control.horizon / 30
    gamma = config.control.gamma

    if method == "action_pca":
        model = ActionPCAModel(config, ridge_alpha)
        model.fit(bilinear, gamma, h.xi, env_dt)
        model.reduce(rank)
        model.vega_channel = bilinear.vega_channel
        model.rev_linear = bilinear.rev_linear
        model.rev_quad = bilinear.rev_quad
    elif method == "bilinear_2stage":
        overspace = min(10, 2 * n_opt)
        bilinear.reduce(overspace)
        U_over = bilinear.U_r.copy()
        c_pen = gamma * h.xi ** 2 / 8.0 * env_dt
        vc_proj = U_over.T @ bilinear.vega_channel
        rq_proj = U_over.T @ np.diag(bilinear.rev_quad) @ U_over
        H_over = rq_proj - c_pen * np.outer(vc_proj, vc_proj)
        eigvals, eigvecs = np.linalg.eigh(H_over)
        idx = np.argsort(eigvals)
        k = min(rank, len(eigvals))
        V_inner = eigvecs[:, idx[:k]]
        U_final = U_over @ V_inner
        norms = np.linalg.norm(U_final, axis=0, keepdims=True)
        bilinear.U_r = U_final / np.maximum(norms, 1e-15)
        bilinear.S_r = np.abs(eigvals[idx[:k]])
        model = bilinear
    else:
        raise ValueError(method)

    u_baseline = np.concatenate([rn_distances, rn_distances])
    U_r = model.U_r
    vc, rl, rq = model.vega_channel, model.rev_linear, model.rev_quad
    xi = h.xi

    def ctrl(state, history=None):
        a_star = _sdre_solve(state.portfolio_vega, U_r, vc, rl, rq,
                              gamma, xi, env_dt)
        u_delta = U_r @ a_star
        u_delta = np.clip(u_delta, -0.8 * u_baseline, 0.8 * u_baseline)
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

    train_seeds = list(range(500))
    test_seeds = list(range(2000, 4000))
    n_test = len(test_seeds)

    log("=" * 70)
    log("  Kernelized State-Conditioned Recovery Experiment")
    log("=" * 70)
    log(f"  Train: {len(train_seeds)} eps, Test: {n_test} eps")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  KRR backend: {device}")

    # === Solve BBG HJB ===
    log("\nSolving HJB...")
    t0 = time.time()
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    log(f"  Solved in {time.time() - t0:.1f}s")
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # === ROPE calibration (from pilot) ===
    pilot_seeds = list(range(5000, 5500))
    w_rn_p = eval_seeds(config, rn_ctrl, pilot_seeds)
    w_bbg_p = eval_seeds(config, bbg_ctrl, pilot_seeds)
    gap_pilot = cara_ce(w_bbg_p, gamma) - cara_ce(w_rn_p, gamma)
    h = max(abs(gap_pilot) * 0.40, 1000.0)
    s_max = h
    log(f"\n  ROPE: h={h:.0f}, s_max={s_max:.0f} (pilot gap={gap_pilot:.0f})")

    # === Build non-kernelized SDRE baselines ===
    log("\nBuilding non-kernelized SDRE controllers (BBG exploration)...")
    controllers = {"risk_neutral": rn_ctrl, "bbg_numerical": bbg_ctrl}

    # === Simple anti-triviality baselines ===
    log("\nFitting simple anti-triviality baselines...")
    alpha_w, ce_w = fit_global_width(config, rn_dists, train_seeds, gamma)
    controllers["global_width"] = make_global_width_controller(config, rn_dists, alpha_w)
    log(f"  global_width alpha={alpha_w:+.3f} train_CE={ce_w:.0f}")

    alpha_ws, beta_ws, ce_ws = fit_global_width_skew(config, rn_dists, train_seeds, gamma)
    controllers["global_width_skew"] = make_global_width_skew_controller(
        config, rn_dists, alpha_ws, beta_ws
    )
    log(
        f"  global_width_skew alpha={alpha_ws:+.3f} beta={beta_ws:+.3f} "
        f"train_CE={ce_ws:.0f}"
    )

    # Build SDRE + get U_r bases for kernelized versions
    bases = {}  # {label: U_r}
    for method in ["action_pca", "bilinear_2stage"]:
        for rank in [1, 2, 3]:
            label = f"sdre_{method}_r{rank}"
            t0 = time.time()
            ctrl, model = build_sdre_from_bbg(
                config, rn_dists, bbg_ctrl, method, rank,
            )
            controllers[label] = ctrl
            bases[label] = model.U_r.copy()
            log(f"  {label} ({time.time()-t0:.1f}s)")

    # === Build kernelized controllers ===
    log("\nBuilding kernelized recovery controllers...")

    for method in ["action_pca", "bilinear_2stage"]:
        for rank in [1, 2, 3]:
            U_r = bases[f"sdre_{method}_r{rank}"]

            for state_rep in ["compact", "rich"]:
                label = f"kern_{method}_r{rank}_{state_rep}"
                t0 = time.time()
                ctrl = make_kernelized_recovery_controller(
                    config, U_r, bbg_ctrl, rn_dists,
                    train_seeds=train_seeds,
                    state_rep=state_rep,
                    n_subsample=3000,
                    krr_alpha=1e-2,
                    ls_multiplier=1.0,
                    device=device,
                )
                controllers[label] = ctrl
                log(f"  {label} ({time.time()-t0:.1f}s)")

    # === Evaluate all on test split ===
    log(f"\n{'='*70}")
    log(f"  Test Evaluation ({n_test} episodes)")
    log(f"{'='*70}")

    results = {}
    for label, ctrl in controllers.items():
        t0 = time.time()
        w = eval_seeds(config, ctrl, test_seeds)
        elapsed = time.time() - t0
        ce = cara_ce(w, gamma)
        results[label] = w
        log(f"  {label:<40s} CE={ce:>10.0f}  mean={w.mean():>10.0f}  "
            f"std={w.std():>8.0f}  ({elapsed:.1f}s)")

    # === Recovery gates ===
    w_bbg = results["bbg_numerical"]
    ce_bbg = cara_ce(w_bbg, gamma)
    ce_rn = cara_ce(results["risk_neutral"], gamma)

    log(f"\n{'='*70}")
    log(f"  Recovery Gate (h={h:.0f}, s_max={s_max:.0f})")
    log(f"  BBG CE={ce_bbg:.0f}, RN CE={ce_rn:.0f}")
    log(f"{'='*70}")

    candidates = sorted(k for k in results if k not in ("risk_neutral", "bbg_numerical"))

    log(f"\n  {'Candidate':<40s} {'mean':>7s} {'sd':>7s} {'P(R)':>6s} "
        f"{'GA':>4s} {'GB':>4s} {'95% CrI':>22s}")
    log(f"  {'-'*96}")

    for label in candidates:
        g = paired_recovery_gate(results[label], w_bbg, gamma, h, s_max)
        ci = g["ci_95"]
        ga = "Y" if g["passes_a"] else "."
        gb = "Y" if g["passes_b"] else "."
        log(f"  {label:<40s} {g['mean']:>7.0f} {g['sd_post']:>7.0f} "
            f"{g['p_rope']:>6.3f} {ga:>4s} {gb:>4s} "
            f"[{ci[0]:>9.0f}, {ci[1]:>9.0f}]")

    # === Comparison: non-kernelized vs kernelized ===
    log(f"\n{'='*70}")
    log("  Non-kernelized vs Kernelized (same basis, same rank)")
    log(f"{'='*70}")

    for method in ["action_pca", "bilinear_2stage"]:
        for rank in [1, 2, 3]:
            sdre_label = f"sdre_{method}_r{rank}"
            ce_sdre = cara_ce(results[sdre_label], gamma)
            for state_rep in ["compact", "rich"]:
                kern_label = f"kern_{method}_r{rank}_{state_rep}"
                ce_kern = cara_ce(results[kern_label], gamma)
                improvement = ce_kern - ce_sdre
                log(f"  {kern_label:<40s} CE={ce_kern:>8.0f}  "
                    f"vs SDRE {ce_sdre:>8.0f}  Δ={improvement:>+8.0f}")

    # === Anti-triviality check ===
    log(f"\n{'='*70}")
    log("  Anti-triviality Check")
    log(f"{'='*70}")

    kern_labels = [k for k in results if k.startswith("kern_")]

    if kern_labels:
        best_kern = max(kern_labels, key=lambda k: cara_ce(results[k], gamma))
        best_kern_ce = cara_ce(results[best_kern], gamma)
        best_kern_gate = paired_recovery_gate(
            results[best_kern], w_bbg, gamma, h, s_max
        )
        log(f"  Best kernelized controller: {best_kern}  CE={best_kern_ce:.0f}")

        if "global_width_skew" in results:
            gwr = paired_recovery_gate(
                results[best_kern], results["global_width_skew"], gamma, h, s_max
            )
            log(
                "  best_kernelized - global_width_skew: "
                f"mean={gwr['mean']:+.0f} sd_post={gwr['sd_post']:.0f} "
                f"P(ROPE)={gwr['p_rope']:.3f} "
                f"GateA={'PASS' if gwr['passes_a'] else 'FAIL'} "
                f"GateB={'PASS' if gwr['passes_b'] else 'FAIL'} "
                f"95%CrI=[{gwr['ci_95'][0]:.0f}, {gwr['ci_95'][1]:.0f}]"
            )

    # === Summary ===
    log(f"\n{'='*70}")
    log("  Summary")
    log(f"{'='*70}")

    if kern_labels:
        best_kern = max(kern_labels, key=lambda k: cara_ce(results[k], gamma))
        g_best = paired_recovery_gate(results[best_kern], w_bbg, gamma, h, s_max)
        log(f"\n  Best kernelized: {best_kern}")
        log(f"    CE={cara_ce(results[best_kern], gamma):.0f}, "
            f"gap to BBG: {g_best['mean']:+.0f}, P(ROPE)={g_best['p_rope']:.3f}")
        log(f"    Gate A: {'PASS' if g_best['passes_a'] else 'FAIL'}, "
            f"Gate B: {'PASS' if g_best['passes_b'] else 'FAIL'}")

    # Best non-kernelized SDRE for comparison
    sdre_labels = [k for k in results if k.startswith("sdre_")]
    if sdre_labels:
        best_sdre = max(sdre_labels, key=lambda k: cara_ce(results[k], gamma))
        g_sdre = paired_recovery_gate(results[best_sdre], w_bbg, gamma, h, s_max)
        log(f"\n  Best non-kernelized: {best_sdre}")
        log(f"    CE={cara_ce(results[best_sdre], gamma):.0f}, "
            f"gap to BBG: {g_sdre['mean']:+.0f}, P(ROPE)={g_sdre['p_rope']:.3f}")

    any_pass = any(
        paired_recovery_gate(results[k], w_bbg, gamma, h, s_max)["passes_both"]
        for k in kern_labels
    )
    if any_pass and "global_width_skew" in results:
        gwr = paired_recovery_gate(
            results[best_kern], results["global_width_skew"], gamma, h, s_max
        )
        if gwr["passes_both"]:
            log(
                "\n  Classification: state-conditioning was the missing piece "
                "(full recovery gate passed and kernelized beats global_width_skew)."
            )
        else:
            log(
                "\n  Classification: kernelization helps but anti-triviality "
                "anchor still survives."
            )
    else:
        log("\n  No kernelized controller passes the full recovery gate.")
        # Check if gap shrank
        if kern_labels and sdre_labels:
            best_kern_gap = abs(cara_ce(results[best_kern], gamma) - ce_bbg)
            best_sdre_gap = abs(cara_ce(results[best_sdre], gamma) - ce_bbg)
            if best_kern_gap < best_sdre_gap * 0.7:
                log(
                    f"  Gap shrank: {best_sdre_gap:.0f} -> {best_kern_gap:.0f} "
                    f"({1 - best_kern_gap/best_sdre_gap:.0%} reduction)"
                )
                log("  Classification: kernelization helps but not enough.")
            else:
                log(
                    f"  Gap did NOT shrink materially: "
                    f"{best_sdre_gap:.0f} -> {best_kern_gap:.0f}"
                )
                log("  Classification: state-conditioning was not the missing piece.")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_kernelized_recovery_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
