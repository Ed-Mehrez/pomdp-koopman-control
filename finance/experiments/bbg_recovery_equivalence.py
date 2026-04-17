"""BBG recovery / equivalence experiment.

Tests whether a learned reduced-action controller can match BBG performance
to practical accuracy, using a paired ROPE + precision gate.

Recovery gate:
  Gate A (ROPE):      P(|Delta| <= h)  >= 0.95
  Gate B (precision): sd_post(Delta)   <= s_max
where Delta = CE_candidate - CE_BBG (paired bootstrap).

Two controller families:
  1. Demonstration recovery: supervised fit of BBG action surface at low rank
  2. SDRE recovery: bilinear_2stage / ActionPCA from BBG-regime exploration

Train: BBG demonstrations on seeds 0-499
Test:  seeds 2000-3999 (2000 episodes, disjoint)

Usage:
    python finance/experiments/bbg_recovery_equivalence.py
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
    """Recovery gate using the metrics-layer paired CE posterior."""
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
# ROPE + precision threshold calibration
# ---------------------------------------------------------------------------


def calibrate_rope(bbg_rn_gap: float) -> tuple[float, float]:
    """Calibrate h and s_max from the BBG-RN CE gap scale.

    h = 40% of |BBG - RN| gap: recovery within less than half the gap.
    s_max = h: paired posterior must be tight enough that ROPE is not vacuous.

    Rationale:
      - The BBG-RN gap is the natural benchmark scale.
      - h < |gap|/2 ensures "close to BBG" means closer to BBG than to RN.
      - s_max = h means the posterior standard deviation is at most equal to
        the ROPE half-width, so the ROPE claim requires genuine concentration
        around zero, not just a wide posterior.
    """
    h = max(abs(bbg_rn_gap) * 0.40, 1000.0)  # floor at 1K
    s_max = h
    return h, s_max


# ---------------------------------------------------------------------------
# Demonstration recovery controller
# ---------------------------------------------------------------------------


def _state_features_extended(state, config):
    """Extended features: (1, tau, nu_norm, vpi_norm, vpi_norm^2)."""
    sf = extract_state_features(state, config)
    vpi_norm = sf[2]
    return np.array([1.0, sf[0], sf[1], vpi_norm, vpi_norm ** 2])


def collect_bbg_demonstrations(config, bbg_ctrl, rn_distances, seeds):
    """Collect BBG state-action pairs, masking censored options."""
    n_opt = config.book.n_options
    u_baseline = np.concatenate([rn_distances, rn_distances])
    max_dist = 10.0 * np.max(rn_distances)  # clip censored quotes

    all_features, all_perturbations = [], []

    for seed in seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            x = _state_features_extended(state, config)
            action = bbg_ctrl(state)
            u = np.concatenate([action.bid_distances, action.ask_distances])

            # Clip censored options to prevent dominating the fit
            u_clipped = np.minimum(u, max_dist)
            delta_u = u_clipped - u_baseline

            all_features.append(x)
            all_perturbations.append(delta_u)

            state, _, _, _ = env.step(action)

    return np.array(all_features), np.array(all_perturbations)


def fit_demonstration_recovery(features, perturbations, rank, ridge_alpha=1e-3):
    """Fit rank-r linear recovery of BBG action surface.

    Model: delta_u = W @ x_features
    W_r = truncated SVD of W at rank r
    Returns (W_r, U_r, singular_values).
    """
    # Ridge: delta_u = features @ beta^T => beta = (X^T X + aI)^{-1} X^T Y
    beta = _ridge_regression(features, perturbations, ridge_alpha)
    W = beta.T  # (40, d_features)

    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]
    W_r = U_r @ np.diag(S_r) @ Vt_r

    return W_r, U_r, S_r


def make_demo_recovery_controller(config, W_r, rn_distances):
    """Controller from fitted demonstration recovery model."""
    n_opt = config.book.n_options
    u_baseline = np.concatenate([rn_distances, rn_distances])

    def ctrl(state, history=None):
        x = _state_features_extended(state, config)
        delta_u = W_r @ x
        u = u_baseline + delta_u
        return OptionBookMMAction(
            bid_distances=np.maximum(u[:n_opt], 1e-6),
            ask_distances=np.maximum(u[n_opt:], 1e-6),
            hedge_trade=-state.net_delta,
        )
    return ctrl


# ---------------------------------------------------------------------------
# SDRE recovery from BBG-regime exploration
# ---------------------------------------------------------------------------


def build_sdre_from_bbg_exploration(config, rn_distances, bbg_ctrl, method, rank,
                                     n_explore=500, ridge_alpha=1e-3):
    """Fit SDRE controller using BBG-generated trajectories as exploration.

    Instead of random exploration, the bilinear model is fitted on data
    from the BBG operating regime, giving better-calibrated dynamics.
    """
    n_opt = config.book.n_options
    h = config.heston

    # Collect exploration data using BBG controller
    all_sf, all_u, all_vpi_pre, all_vpi_post = [], [], [], []
    all_dinv, all_spread = [], []

    for ep in range(n_explore):
        env = OptionBookMarketMakingEnv(config, seed=ep)
        state = env.reset()
        while not state.done:
            sf = extract_state_features(state, config)
            vpi_pre = state.portfolio_vega
            inv_pre = state.option_inventories.copy()

            action = bbg_ctrl(state)
            u = np.concatenate([action.bid_distances, action.ask_distances])
            # Clip censored options for the regression
            u = np.minimum(u, 10.0 * np.max(rn_distances))

            next_state, _, _, info = env.step(action)

            all_sf.append(sf)
            all_u.append(u)
            all_vpi_pre.append(vpi_pre)
            all_vpi_post.append(next_state.portfolio_vega)
            all_dinv.append(next_state.option_inventories - inv_pre)
            all_spread.append(info["spread_capture"])

            state = next_state

    from applications.option_mm_bbg.sdre_recovery import ExplorationData
    data = ExplorationData(
        state_features=np.array(all_sf),
        actions=np.array(all_u),
        vpi_pre=np.array(all_vpi_pre),
        vpi_post=np.array(all_vpi_post),
        inventory_changes=np.array(all_dinv),
        spread_captures=np.array(all_spread),
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
# Evaluation
# ---------------------------------------------------------------------------


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
    log("  BBG Recovery / Equivalence Experiment")
    log("=" * 70)
    log(f"  Train: {len(train_seeds)} eps, Test: {n_test} eps")
    log(f"  Gamma: {gamma}")

    # === Solve BBG HJB ===
    log("\nSolving 3D HJB (fine grid)...")
    t0 = time.time()
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    log(f"  Solved in {time.time() - t0:.1f}s")
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # === ROPE calibration ===
    log("\nCalibrating ROPE from baseline CEs (500-episode pilot)...")
    pilot_seeds = list(range(5000, 5500))
    w_rn_pilot = eval_seeds(config, rn_ctrl, pilot_seeds)
    w_bbg_pilot = eval_seeds(config, bbg_ctrl, pilot_seeds)
    ce_rn_pilot = cara_ce(w_rn_pilot, gamma)
    ce_bbg_pilot = cara_ce(w_bbg_pilot, gamma)
    gap_pilot = ce_bbg_pilot - ce_rn_pilot
    h, s_max = calibrate_rope(gap_pilot)

    log(f"  RN CE (pilot): {ce_rn_pilot:.0f}")
    log(f"  BBG CE (pilot): {ce_bbg_pilot:.0f}")
    log(f"  BBG-RN gap: {gap_pilot:.0f}")
    log(f"  ROPE h = {h:.0f}  (40% of |gap|)")
    log(f"  Precision s_max = {s_max:.0f}")

    # === Collect BBG demonstrations (train) ===
    log("\nCollecting BBG demonstrations (500 train episodes)...")
    t0 = time.time()
    demo_features, demo_perturbations = collect_bbg_demonstrations(
        config, bbg_ctrl, rn_dists, train_seeds,
    )
    log(f"  {len(demo_features)} state-action pairs in {time.time() - t0:.1f}s")

    # === Build controllers ===
    controllers = {"risk_neutral": rn_ctrl, "bbg_numerical": bbg_ctrl}

    # Demonstration recovery at ranks 1, 2, 3
    log("\nFitting demonstration recovery controllers...")
    for rank in [1, 2, 3]:
        W_r, U_r, S_r = fit_demonstration_recovery(
            demo_features, demo_perturbations, rank,
        )
        ctrl = make_demo_recovery_controller(config, W_r, rn_dists)
        label = f"demo_r{rank}"
        controllers[label] = ctrl
        log(f"  {label}: singular values = {S_r}")

    # SDRE recovery from BBG-regime exploration
    log("\nBuilding SDRE recovery controllers (BBG-regime exploration)...")
    for method in ["action_pca", "bilinear_2stage"]:
        for rank in [1, 2, 3]:
            t0 = time.time()
            label = f"sdre_{method}_r{rank}"
            log(f"  Training {label}...")
            ctrl, _ = build_sdre_from_bbg_exploration(
                config, rn_dists, bbg_ctrl, method, rank,
            )
            controllers[label] = ctrl
            log(f"    Done in {time.time() - t0:.1f}s")

    # Simple baselines for context (inlined to avoid cross-script imports)
    log("\nBuilding simple baselines...")

    def _vw(config):
        h_ = config.heston
        vegas_ = np.array([bs_call_vega_sqrt_nu(h_.spot0, o.strike, o.maturity, h_.rate, h_.nu0)
                           for o in config.book.options])
        prices_ = np.array([bs_call_price(h_.spot0, o.strike, o.maturity, h_.rate, h_.nu0)
                            for o in config.book.options])
        ts_ = np.array([config.liquidity.trade_size(p) for p in prices_])
        w_ = ts_ * vegas_
        return w_ / (np.linalg.norm(w_) + 1e-15)

    def _make_gw(cfg, rnd, alpha):
        sc = np.maximum(rnd * (1.0 + alpha), 1e-6)
        def ctrl(state, history=None):
            return OptionBookMMAction(bid_distances=sc.copy(), ask_distances=sc.copy(),
                                      hedge_trade=-state.net_delta)
        return ctrl

    def _make_gws(cfg, rnd, alpha, beta):
        w_ = _vw(cfg); vlim = cfg.control.vega_limit
        def ctrl(state, history=None):
            vn = state.portfolio_vega / vlim
            bm = 1.0 + alpha + beta * vn * w_
            am = 1.0 + alpha - beta * vn * w_
            return OptionBookMMAction(bid_distances=np.maximum(rnd * bm, 1e-6),
                                      ask_distances=np.maximum(rnd * am, 1e-6),
                                      hedge_trade=-state.net_delta)
        return ctrl

    bl_train = list(range(100))
    best_a, best_ce = 0.0, -np.inf
    for a_ in np.linspace(-0.3, 3.0, 25):
        w_ = eval_seeds(config, _make_gw(config, rn_dists, a_), bl_train)
        c_ = cara_ce(w_, gamma)
        if c_ > best_ce: best_ce, best_a = c_, a_
    controllers["global_width"] = _make_gw(config, rn_dists, best_a)
    log(f"  global_width: alpha={best_a:.3f}")

    best_a2, best_b2, best_ce2 = 0.0, 0.0, -np.inf
    for a_ in np.linspace(-0.3, 3.0, 15):
        for b_ in np.linspace(0.0, 5.0, 11):
            w_ = eval_seeds(config, _make_gws(config, rn_dists, a_, b_), bl_train)
            c_ = cara_ce(w_, gamma)
            if c_ > best_ce2: best_ce2, best_a2, best_b2 = c_, a_, b_
    controllers["global_width_skew"] = _make_gws(config, rn_dists, best_a2, best_b2)
    log(f"  global_width_skew: alpha={best_a2:.3f}, beta={best_b2:.3f}")

    # === Evaluate all on TEST split ===
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
        log(f"  {label:<30s} CE={ce:>10.0f}  mean={w.mean():>10.0f}  "
            f"std={w.std():>8.0f}  ({elapsed:.1f}s)")

    # === Recovery gate for each candidate vs BBG ===
    w_bbg = results["bbg_numerical"]
    ce_bbg = cara_ce(w_bbg, gamma)
    ce_rn = cara_ce(results["risk_neutral"], gamma)
    bbg_rn_gap = ce_bbg - ce_rn

    log(f"\n{'='*70}")
    log(f"  Recovery Gate (h={h:.0f}, s_max={s_max:.0f})")
    log(f"  BBG CE={ce_bbg:.0f}, RN CE={ce_rn:.0f}, gap={bbg_rn_gap:.0f}")
    log(f"{'='*70}")

    candidates = [k for k in results if k not in ("risk_neutral", "bbg_numerical")]
    candidates.sort()

    log(f"\n  {'Candidate':<30s} {'mean':>8s} {'sd':>8s} {'P(ROPE)':>8s} "
        f"{'GateA':>6s} {'GateB':>6s} {'BOTH':>6s} {'gap%':>7s} {'95% CrI':>22s}")
    log(f"  {'-'*106}")

    for label in candidates:
        gate = paired_recovery_gate(results[label], w_bbg, gamma, h, s_max)
        # Express gap as fraction of BBG-RN gap
        gap_frac = gate["mean"] / abs(bbg_rn_gap) if abs(bbg_rn_gap) > 1 else float("nan")
        ci = gate["ci_95"]
        log(f"  {label:<30s} {gate['mean']:>8.0f} {gate['sd_post']:>8.0f} "
            f"{gate['p_rope']:>8.3f} "
            f"{'PASS' if gate['passes_a'] else 'FAIL':>6s} "
            f"{'PASS' if gate['passes_b'] else 'FAIL':>6s} "
            f"{'PASS' if gate['passes_both'] else 'FAIL':>6s} "
            f"{gap_frac:>6.0%} [{ci[0]:>9.0f}, {ci[1]:>9.0f}]")

    # === Summary: best recovery candidate ===
    log(f"\n{'='*70}")
    log("  Recovery Summary")
    log(f"{'='*70}")

    any_pass = False
    for label in candidates:
        gate = paired_recovery_gate(results[label], w_bbg, gamma, h, s_max)
        if gate["passes_both"]:
            log(f"  PASS: {label} (mean={gate['mean']:.0f}, sd={gate['sd_post']:.0f}, "
                f"P(ROPE)={gate['p_rope']:.3f})")
            any_pass = True

    if not any_pass:
        # Find closest
        best_label, best_p = None, -1.0
        for label in candidates:
            gate = paired_recovery_gate(results[label], w_bbg, gamma, h, s_max)
            if gate["p_rope"] > best_p:
                best_p = gate["p_rope"]
                best_label = label
        log(f"  No candidate passes both gates.")
        log(f"  Closest: {best_label} (P(ROPE)={best_p:.3f})")

    # === Outcome classification ===
    passing_labels = []
    for label in candidates:
        gate = paired_recovery_gate(results[label], w_bbg, gamma, h, s_max)
        if gate["passes_both"]:
            passing_labels.append(label)

    if passing_labels:
        # Check if any low-rank candidate passes
        low_rank_pass = any("_r1" in l or "_r2" in l for l in passing_labels)
        if low_rank_pass:
            outcome = "R1"
            desc = "Strong recovery: low-rank candidate passes both ROPE and precision gates."
        else:
            outcome = "R1"
            desc = "Strong recovery: candidate passes both gates."
    else:
        # Check if any candidate is close
        best_p = max(
            paired_recovery_gate(results[l], w_bbg, gamma, h, s_max)["p_rope"]
            for l in candidates
        )
        if best_p > 0.80:
            outcome = "R2"
            desc = ("Weak/noisy recovery: best candidate near BBG on mean "
                    "but posterior too wide or ROPE not met.")
        else:
            outcome = "R3"
            desc = ("No recovery: candidates remain meaningfully away from BBG "
                    "under paired ROPE test.")

    log(f"\n  *** OUTCOME: {outcome} ***")
    log(f"  {desc}")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_recovery_equivalence_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
