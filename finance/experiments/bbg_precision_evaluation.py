"""Precision evaluation: can the best kernelized controller pass the
recovery gate with stronger inference and more test episodes?

Fixes the best exact kernelized controller from the tuning run
(ActionPCA r3, compact, alpha=3e-3, ls=2.0) and evaluates it at
N_test = 2000, 3000, 4000 using the metrics-layer paired CE posterior
(delta method + MC cross-check).

CRN pairing is already enforced: same seed -> identical Heston path
and Poisson fill sequence, action-independent draw order.

Usage:
    python finance/experiments/bbg_precision_evaluation.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "finance" / "experiments"))

import bbg_kernelized_recovery as bkr
from applications.option_mm.metrics import cara_utility, paired_ce_posterior
from applications.option_mm_bbg.spec import BBGBenchmarkConfig
from applications.option_mm_bbg.solver import (
    solve_bbg_value_function,
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
)
from applications.option_mm_bbg.sdre_recovery import (
    _compute_rn_distances,
    make_kernelized_recovery_controller,
)

BEST_ALPHA = 3e-3
BEST_LS = 2.0
BEST_STATE_REP = "compact"
TRAIN_SEEDS = list(range(500))
N_TEST_LEVELS = [2000, 3000, 4000]


def recovery_gate_from_posterior(post, h, s_max):
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


def main() -> int:
    out: list[str] = []

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma
    utility = cara_utility(gamma)
    rn_dists = _compute_rn_distances(config)

    log("=" * 72)
    log("  Precision Evaluation: Best Kernelized Controller")
    log("=" * 72)
    log(f"  Controller: ActionPCA r3, {BEST_STATE_REP}, "
        f"alpha={BEST_ALPHA:.0e}, ls={BEST_LS:.1f}")
    log(f"  Train: {len(TRAIN_SEEDS)} eps")
    log(f"  N_test levels: {N_TEST_LEVELS}")
    log(f"  Inference: delta method (headline) + bootstrap (cross-check)")

    # === HJB ===
    log("\nSolving HJB...")
    t0 = time.time()
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=20, n_vpi=40, n_time=120,
    )
    log(f"  Solved in {time.time() - t0:.1f}s")
    bbg_ctrl = make_bbg_numerical_controller(
        config, values, t_grid, nu_grid, vpi_grid
    )
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # === Build ActionPCA r3 basis + kernelized controller ===
    log("\nBuilding ActionPCA r3 basis...")
    t0 = time.time()
    _, model = bkr.build_sdre_from_bbg(
        config, rn_dists, bbg_ctrl, method="action_pca", rank=3,
    )
    U_r = model.U_r.copy()
    log(f"  U_r {U_r.shape} in {time.time()-t0:.1f}s")

    log(f"\nBuilding best exact kernelized controller "
        f"(alpha={BEST_ALPHA:.0e}, ls={BEST_LS}, {BEST_STATE_REP})...")
    t0 = time.time()
    kern_ctrl = make_kernelized_recovery_controller(
        config, U_r, bbg_ctrl, rn_dists,
        train_seeds=TRAIN_SEEDS,
        state_rep=BEST_STATE_REP,
        n_subsample=3000,
        krr_alpha=BEST_ALPHA,
        ls_multiplier=BEST_LS,
        approx="exact",
        device="cpu",
    )
    log(f"  Built in {time.time()-t0:.1f}s")

    # === ROPE calibration ===
    log("\nCalibrating ROPE from 500-ep pilot...")
    pilot_seeds = list(range(5000, 5500))
    t0 = time.time()
    w_rn_p = bkr.eval_seeds(config, rn_ctrl, pilot_seeds)
    w_bbg_p = bkr.eval_seeds(config, bbg_ctrl, pilot_seeds)
    gap_pilot = bkr.cara_ce(w_bbg_p, gamma) - bkr.cara_ce(w_rn_p, gamma)
    h = max(abs(gap_pilot) * 0.40, 1000.0)
    s_max = h
    log(f"  h = {h:.0f}, s_max = {s_max:.0f}  "
        f"(pilot BBG-RN gap = {gap_pilot:.0f}, {time.time()-t0:.1f}s)")

    # === Evaluate at each N_test level ===
    # Use a single large seed pool and take the first N from it.
    max_n = max(N_TEST_LEVELS)
    all_test_seeds = list(range(2000, 2000 + max_n))

    log(f"\nEvaluating BBG on full pool ({max_n} eps)...")
    t0 = time.time()
    w_bbg_all = bkr.eval_seeds(config, bbg_ctrl, all_test_seeds)
    log(f"  BBG done in {time.time()-t0:.1f}s")

    log(f"Evaluating candidate on full pool ({max_n} eps)...")
    t0 = time.time()
    w_kern_all = bkr.eval_seeds(config, kern_ctrl, all_test_seeds)
    log(f"  Candidate done in {time.time()-t0:.1f}s")

    log(f"Evaluating RN on full pool ({max_n} eps)...")
    t0 = time.time()
    w_rn_all = bkr.eval_seeds(config, rn_ctrl, all_test_seeds)
    log(f"  RN done in {time.time()-t0:.1f}s")

    # Anti-triviality baselines
    gw_ctrl = bkr.make_global_width_controller(config, rn_dists, -0.025)
    gws_ctrl = bkr.make_global_width_skew_controller(
        config, rn_dists, -0.300, 3.000,
    )
    log("Evaluating anti-triviality baselines...")
    t0 = time.time()
    w_gws_all = bkr.eval_seeds(config, gws_ctrl, all_test_seeds)
    log(f"  GWS done in {time.time()-t0:.1f}s")

    # === Results at each N ===
    log("\n" + "=" * 72)
    log("  Results by N_test")
    log("=" * 72)

    for n_test in N_TEST_LEVELS:
        w_bbg = w_bbg_all[:n_test]
        w_kern = w_kern_all[:n_test]
        w_rn = w_rn_all[:n_test]
        w_gws = w_gws_all[:n_test]

        ce_bbg = bkr.cara_ce(w_bbg, gamma)
        ce_kern = bkr.cara_ce(w_kern, gamma)
        ce_rn = bkr.cara_ce(w_rn, gamma)
        ce_gws = bkr.cara_ce(w_gws, gamma)

        log(f"\n  --- N_test = {n_test} ---")
        log(f"  BBG  CE = {ce_bbg:.0f}")
        log(f"  Kern CE = {ce_kern:.0f}")
        log(f"  RN   CE = {ce_rn:.0f}")
        log(f"  GWS  CE = {ce_gws:.0f}")

        # Delta method (headline)
        post_delta = paired_ce_posterior(
            w_kern, w_bbg, utility=utility, method="delta",
        )
        g_delta = recovery_gate_from_posterior(post_delta, h, s_max)

        # MC cross-check
        post_mc = paired_ce_posterior(
            w_kern, w_bbg, utility=utility, method="mc",
            n_draws=50_000, rng=np.random.default_rng(42),
        )
        g_mc = recovery_gate_from_posterior(post_mc, h, s_max)

        log(f"\n  Delta method (headline):")
        log(f"    mean={g_delta['mean']:+.1f}  sd={g_delta['sd_post']:.1f}  "
            f"P(ROPE)={g_delta['p_rope']:.4f}  "
            f"GA={'PASS' if g_delta['passes_a'] else 'FAIL'}  "
            f"GB={'PASS' if g_delta['passes_b'] else 'FAIL'}  "
            f"95% CrI=[{g_delta['ci_95'][0]:.0f}, {g_delta['ci_95'][1]:.0f}]")

        log(f"  MC cross-check (50K draws):")
        log(f"    mean={g_mc['mean']:+.1f}  sd={g_mc['sd_post']:.1f}  "
            f"P(ROPE)={g_mc['p_rope']:.4f}  "
            f"GA={'PASS' if g_mc['passes_a'] else 'FAIL'}  "
            f"GB={'PASS' if g_mc['passes_b'] else 'FAIL'}  "
            f"95% CrI=[{g_mc['ci_95'][0]:.0f}, {g_mc['ci_95'][1]:.0f}]")

        # Agreement check
        sd_agree = abs(g_delta["sd_post"] - g_mc["sd_post"]) / max(
            g_delta["sd_post"], 1e-10
        )
        log(f"  Delta-bootstrap agreement: sd_post relative diff = {sd_agree:.4f}")

        # Anti-triviality: kern vs GWS
        post_gws = paired_ce_posterior(
            w_kern, w_gws, utility=utility, method="delta",
        )
        g_gws = recovery_gate_from_posterior(post_gws, h, s_max)
        log(f"  vs GWS: mean={g_gws['mean']:+.0f}  "
            f"sd={g_gws['sd_post']:.0f}  P(+)={g_gws['p_positive']:.4f}")

    # === Summary ===
    log("\n" + "=" * 72)
    log("  Summary")
    log("=" * 72)

    # Use the largest-N results for the final call
    n_max = max(N_TEST_LEVELS)
    w_bbg_max = w_bbg_all[:n_max]
    w_kern_max = w_kern_all[:n_max]
    post_final = paired_ce_posterior(
        w_kern_max, w_bbg_max, utility=utility, method="delta",
    )
    g_final = recovery_gate_from_posterior(post_final, h, s_max)

    log(f"\n  Final result at N_test = {n_max}:")
    log(f"    mean gap = {g_final['mean']:+.1f}")
    log(f"    sd_post  = {g_final['sd_post']:.1f}")
    log(f"    P(ROPE)  = {g_final['p_rope']:.4f}")
    log(f"    Gate A   = {'PASS' if g_final['passes_a'] else 'FAIL'}")
    log(f"    Gate B   = {'PASS' if g_final['passes_b'] else 'FAIL'}")

    if g_final["passes_both"]:
        log("\n  *** Outcome: E1 — precision fixed, recovery PASSES. ***")
    elif g_final["sd_post"] < 0.9 * 8700:
        log("\n  Outcome: E2 — precision improved but gate still fails.")
    else:
        log("\n  Outcome: E3 — little to no precision gain.")

    # Scaling analysis
    log(f"\n  Scaling analysis (sd_post vs N):")
    for n_test in N_TEST_LEVELS:
        post_n = paired_ce_posterior(
            w_kern_all[:n_test], w_bbg_all[:n_test],
            utility=utility, method="delta",
        )
        predicted_sd = post_n.sd_post * np.sqrt(n_test / 2000)
        log(f"    N={n_test}: sd_post={post_n.sd_post:.1f}  "
            f"(extrapolated to N=2000: {predicted_sd:.1f})")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_precision_evaluation_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
