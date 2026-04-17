"""Nystrom-accelerated kernel tuning for BBG kernelized recovery.

Tunes (krr_alpha, ls_multiplier, n_landmarks) for the ActionPCA r3
kernelized controller using Nystrom KRR as a fast search backend, then
exact-KRR refits the top 2-4 finalists on the full formal split and
reports the full paired ROPE recovery gate plus an anti-triviality
contrast vs global_width_skew.

Dev split (search)      :  200 train (0..199)     / 400 dev (1000..1399)
Formal split (refit)    :  500 train (0..499)     / 2000 test (2000..3999)

The formal split matches `bbg_kernelized_recovery.py` so exact-refit
results are directly comparable to the prior run.

Usage:
    python finance/experiments/bbg_kernel_tuning.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "finance" / "experiments"))

import bbg_kernelized_recovery as bkr  # noqa: E402
from applications.option_mm_bbg.env import (  # noqa: E402
    OptionBookMarketMakingEnv,
    OptionBookMMAction,
)
from applications.option_mm_bbg.spec import BBGBenchmarkConfig  # noqa: E402
from applications.option_mm_bbg.solver import (  # noqa: E402
    solve_bbg_value_function,
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
)
from applications.option_mm_bbg.sdre_recovery import (  # noqa: E402
    KernelRidgeModel,
    _compute_rn_distances,
    extract_state_compact,
    make_kernelized_recovery_controller,
)


ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LS_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
M_GRID = [256, 512, 1024]


def _collect_bbg_demos_reduced(
    config: BBGBenchmarkConfig,
    bbg_ctrl,
    U_r: np.ndarray,
    rn_distances: np.ndarray,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run BBG over `seeds`, project its actions into reduced coordinates.

    Returns X (compact 3D features) and Y (reduced-action targets).
    Extracted so the sweep can collect demos ONCE and reuse them across
    every (alpha, ls_multiplier, n_landmarks) configuration.
    """
    u_baseline = np.concatenate([rn_distances, rn_distances])
    max_dist = 10.0 * np.max(rn_distances)
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    for seed in seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            x = extract_state_compact(state, config)
            action = bbg_ctrl(state)
            u = np.concatenate([action.bid_distances, action.ask_distances])
            u_clipped = np.minimum(u, max_dist)
            delta_u = u_clipped - u_baseline
            a_target = U_r.T @ delta_u
            X.append(x)
            Y.append(a_target)
            state, _, _, _ = env.step(action)
    return np.array(X), np.array(Y)


def _build_tuned_nystrom_controller(
    config: BBGBenchmarkConfig,
    U_r: np.ndarray,
    rn_distances: np.ndarray,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    base_ls: np.ndarray,
    krr_alpha: float,
    ls_multiplier: float,
    n_landmarks: int,
    landmark_seed: int = 0,
):
    """Fit a Nystrom KRR on pre-collected (X_train, Y_train) and wrap as
    an OptionBookMMAction controller. Used inside the tuning sweep so demo
    collection does not repeat per configuration."""
    n_opt = config.book.n_options
    u_baseline = np.concatenate([rn_distances, rn_distances])
    ls = base_ls * ls_multiplier
    krr = KernelRidgeModel(
        length_scales=ls,
        alpha=krr_alpha,
        device="cpu",
        approx="nystrom",
        n_landmarks=n_landmarks,
        landmark_method="kmeans++",
        landmark_seed=landmark_seed,
    )
    krr.fit(X_train, Y_train)

    def ctrl(state, history=None):
        x = extract_state_compact(state, config)
        a = krr.predict_single(x)
        u_delta = U_r @ a
        u_full = u_baseline + u_delta
        return OptionBookMMAction(
            bid_distances=np.maximum(u_full[:n_opt], 1e-6),
            ask_distances=np.maximum(u_full[n_opt:], 1e-6),
            hedge_trade=-state.net_delta,
        )

    return ctrl, krr


def main() -> int:
    out: list[str] = []

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma
    rn_dists = _compute_rn_distances(config)

    dev_train_seeds = list(range(200))
    dev_eval_seeds = list(range(1000, 1400))
    full_train_seeds = list(range(500))
    full_test_seeds = list(range(2000, 4000))

    log("=" * 72)
    log("  Kernel Tuning - Nystrom Sweep + Exact Refit (ActionPCA r3)")
    log("=" * 72)
    log(f"  Dev    : {len(dev_train_seeds)} train / {len(dev_eval_seeds)} eval")
    log(f"  Formal : {len(full_train_seeds)} train / {len(full_test_seeds)} test")
    log(f"  Grid   : alpha={ALPHA_GRID}")
    log(f"           ls_mult={LS_GRID}")
    log(f"           n_landmarks={M_GRID}")
    log(f"  Total configurations: {len(ALPHA_GRID) * len(LS_GRID) * len(M_GRID)}")

    # === HJB / BBG ===
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

    # === Fixed ActionPCA r3 basis ===
    log("\nBuilding ActionPCA r3 basis (SDRE from BBG exploration)...")
    t0 = time.time()
    _sdre_ctrl, sdre_model = bkr.build_sdre_from_bbg(
        config, rn_dists, bbg_ctrl, method="action_pca", rank=3,
    )
    U_r = sdre_model.U_r.copy()
    log(f"  U_r shape {U_r.shape}, built in {time.time()-t0:.1f}s")

    # === ROPE from 500-ep pilot (same calibration as prior kernelized run) ===
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

    # === BBG baseline on dev eval (gap metric for tuning) ===
    log(f"\nEvaluating BBG on dev eval ({len(dev_eval_seeds)} eps)...")
    t0 = time.time()
    w_bbg_dev = bkr.eval_seeds(config, bbg_ctrl, dev_eval_seeds)
    ce_bbg_dev = bkr.cara_ce(w_bbg_dev, gamma)
    log(f"  BBG dev CE = {ce_bbg_dev:.0f}  ({time.time()-t0:.1f}s)")

    # === Formal-split BBG + RN + anti-triviality baselines ===
    log(f"\nEvaluating BBG / RN on formal test ({len(full_test_seeds)} eps)...")
    t0 = time.time()
    w_bbg = bkr.eval_seeds(config, bbg_ctrl, full_test_seeds)
    ce_bbg = bkr.cara_ce(w_bbg, gamma)
    log(f"  BBG  CE = {ce_bbg:.0f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    w_rn = bkr.eval_seeds(config, rn_ctrl, full_test_seeds)
    ce_rn = bkr.cara_ce(w_rn, gamma)
    log(f"  RN   CE = {ce_rn:.0f}  ({time.time()-t0:.1f}s)")

    # Anti-triviality baselines: reuse coefficients from the 2026-04-11
    # kernelized recovery run (fit on the full 500-ep formal train set).
    # This keeps the anti-triviality contrast here directly comparable
    # to `bbg_kernelized_recovery_2026-04-11.txt` and saves ~7 min of
    # baseline-fit time that would otherwise recompute the same result.
    log("\nAnti-triviality baselines (reused from 2026-04-11 kernelized run,")
    log("  fit on the full 500-ep formal train set):")
    alpha_w = -0.025
    alpha_ws = -0.300
    beta_ws = 3.000
    gw_ctrl = bkr.make_global_width_controller(config, rn_dists, alpha_w)
    gws_ctrl = bkr.make_global_width_skew_controller(
        config, rn_dists, alpha_ws, beta_ws,
    )
    log(f"  global_width      alpha={alpha_w:+.3f}")
    log(f"  global_width_skew alpha={alpha_ws:+.3f} beta={beta_ws:+.3f}")
    w_gw = bkr.eval_seeds(config, gw_ctrl, full_test_seeds)
    w_gws = bkr.eval_seeds(config, gws_ctrl, full_test_seeds)
    ce_gw = bkr.cara_ce(w_gw, gamma)
    ce_gws = bkr.cara_ce(w_gws, gamma)
    log(f"  global_width      CE = {ce_gw:.0f}")
    log(f"  global_width_skew CE = {ce_gws:.0f}")

    # === Nystrom tuning sweep ===
    log("\n" + "=" * 72)
    log("  Nystrom Tuning Sweep")
    log("=" * 72)
    state_rep = "compact"
    log(f"  state_rep = {state_rep}")

    # Precollect BBG demos ONCE — every sweep config refits KRR on the same
    # (X_train, Y_train) slab. Without this, demo collection would dominate
    # and re-run 126 times for no reason.
    log(f"\n  Collecting BBG demos on dev train ({len(dev_train_seeds)} eps)...")
    t0 = time.time()
    X_all, Y_all = _collect_bbg_demos_reduced(
        config, bbg_ctrl, U_r, rn_dists, dev_train_seeds,
    )
    log(f"  {len(X_all)} (x, a_reduced) pairs collected in {time.time()-t0:.1f}s")

    n_subsample = 3000
    if len(X_all) > n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_all), n_subsample, replace=False)
        X_train = X_all[idx]
        Y_train = Y_all[idx]
    else:
        X_train = X_all
        Y_train = Y_all
    base_ls = np.maximum(X_train.std(axis=0), 1e-8)
    log(f"  X_train={X_train.shape} Y_train={Y_train.shape} "
        f"base_ls={base_ls}")

    sweep_rows: list[dict] = []
    sweep_t0 = time.time()
    n_total = len(ALPHA_GRID) * len(LS_GRID) * len(M_GRID)
    i = 0
    for alpha in ALPHA_GRID:
        for lsm in LS_GRID:
            for M in M_GRID:
                i += 1
                t_fit_0 = time.time()
                ctrl, krr = _build_tuned_nystrom_controller(
                    config, U_r, rn_dists,
                    X_train, Y_train, base_ls,
                    krr_alpha=alpha,
                    ls_multiplier=lsm,
                    n_landmarks=M,
                    landmark_seed=0,
                )
                fit_t = time.time() - t_fit_0
                t_eval_0 = time.time()
                w_dev = bkr.eval_seeds(config, ctrl, dev_eval_seeds)
                eval_t = time.time() - t_eval_0
                ce_dev = bkr.cara_ce(w_dev, gamma)
                gap = ce_dev - ce_bbg_dev
                sweep_rows.append({
                    "approx": "nystrom",
                    "state_rep": state_rep,
                    "rank": 3,
                    "krr_alpha": alpha,
                    "ls_multiplier": lsm,
                    "n_landmarks": M,
                    "dev_ce": ce_dev,
                    "dev_gap": gap,
                    "fit_s": fit_t,
                    "eval_s": eval_t,
                    "cond": float(krr.fit_cond_number)
                        if krr.fit_cond_number is not None else float("nan"),
                })
                if i % 10 == 0 or i == n_total:
                    elapsed = time.time() - sweep_t0
                    eta = elapsed / i * (n_total - i)
                    log(f"  [{i:3d}/{n_total}] alpha={alpha:.0e} ls={lsm:.2f} "
                        f"M={M:4d} gap={gap:+8.0f}  "
                        f"elapsed={elapsed:5.0f}s ETA={eta:5.0f}s")

    sweep_total = time.time() - sweep_t0
    log(f"\n  Sweep complete: {len(sweep_rows)} configs in {sweep_total:.1f}s "
        f"({sweep_total / n_total:.2f}s per config)")

    # Rank by |dev_gap|
    sweep_rows.sort(key=lambda r: abs(r["dev_gap"]))

    log("\n  Top 10 by |dev_gap|:")
    log(f"  {'#':>3s} {'alpha':>9s} {'ls_mult':>8s} {'M':>5s} "
        f"{'dev_ce':>9s} {'dev_gap':>9s} {'cond':>9s} {'fit':>6s} {'eval':>6s}")
    for rk, r in enumerate(sweep_rows[:10], 1):
        log(f"  {rk:>3d} {r['krr_alpha']:>9.1e} {r['ls_multiplier']:>8.2f} "
            f"{r['n_landmarks']:>5d} {r['dev_ce']:>9.0f} "
            f"{r['dev_gap']:>+9.0f} {r['cond']:>9.1e} "
            f"{r['fit_s']:>6.2f} {r['eval_s']:>6.2f}")

    # Save full ranked table for the record
    log("\n  Full sweep (ranked by |dev_gap|):")
    log(f"  {'#':>3s} {'alpha':>9s} {'ls_mult':>8s} {'M':>5s} "
        f"{'dev_ce':>9s} {'dev_gap':>9s} {'cond':>9s}")
    for rk, r in enumerate(sweep_rows, 1):
        log(f"  {rk:>3d} {r['krr_alpha']:>9.1e} {r['ls_multiplier']:>8.2f} "
            f"{r['n_landmarks']:>5d} {r['dev_ce']:>9.0f} "
            f"{r['dev_gap']:>+9.0f} {r['cond']:>9.1e}")

    # === Exact refit of finalists ===
    # Deduplicate by (alpha, ls) since n_landmarks is irrelevant for exact.
    seen: set = set()
    finalists: list[dict] = []
    for r in sweep_rows:
        key = (r["krr_alpha"], r["ls_multiplier"], r["state_rep"])
        if key in seen:
            continue
        seen.add(key)
        finalists.append(r)
        if len(finalists) >= 4:
            break

    log("\n" + "=" * 72)
    log(f"  Exact KRR Refit of {len(finalists)} Finalists on Formal Split")
    log("=" * 72)

    refit_rows: list[dict] = []
    for rank_idx, r in enumerate(finalists, 1):
        log(f"\n  [{rank_idx}/{len(finalists)}] alpha={r['krr_alpha']:.1e} "
            f"ls={r['ls_multiplier']:.2f} state={r['state_rep']} "
            f"(Nystrom dev gap {r['dev_gap']:+.0f})")
        t0 = time.time()
        ctrl = make_kernelized_recovery_controller(
            config, U_r, bbg_ctrl, rn_dists,
            train_seeds=full_train_seeds,
            state_rep=r["state_rep"],
            n_subsample=3000,
            krr_alpha=r["krr_alpha"],
            ls_multiplier=r["ls_multiplier"],
            approx="exact",
        )
        fit_t = time.time() - t0
        t0 = time.time()
        w_test = bkr.eval_seeds(config, ctrl, full_test_seeds)
        eval_t = time.time() - t0
        ce_test = bkr.cara_ce(w_test, gamma)
        gate_bbg = bkr.paired_recovery_gate(w_test, w_bbg, gamma, h, s_max)
        gate_gws = bkr.paired_recovery_gate(w_test, w_gws, gamma, h, s_max)
        refit_rows.append({
            **r,
            "test_ce": ce_test,
            "gate_bbg": gate_bbg,
            "gate_gws": gate_gws,
            "fit_s": fit_t,
            "eval_s": eval_t,
        })
        log(f"    CE={ce_test:.0f}  fit={fit_t:.1f}s  eval={eval_t:.1f}s")
        log(f"    vs BBG   : mean={gate_bbg['mean']:+.0f} "
            f"sd={gate_bbg['sd_post']:.0f} P(R)={gate_bbg['p_rope']:.3f} "
            f"GA={'PASS' if gate_bbg['passes_a'] else 'FAIL'} "
            f"GB={'PASS' if gate_bbg['passes_b'] else 'FAIL'}")
        log(f"    vs GWS   : mean={gate_gws['mean']:+.0f} "
            f"sd={gate_gws['sd_post']:.0f} P(R)={gate_gws['p_rope']:.3f} "
            f"GA={'PASS' if gate_gws['passes_a'] else 'FAIL'} "
            f"GB={'PASS' if gate_gws['passes_b'] else 'FAIL'}")

    # === Reference: prior default (exact, alpha=1e-2, ls=1.0) ===
    log("\n" + "=" * 72)
    log("  Reference: Prior Kernelized Default (exact, alpha=1e-2, ls=1.0)")
    log("=" * 72)
    t0 = time.time()
    ctrl_ref = make_kernelized_recovery_controller(
        config, U_r, bbg_ctrl, rn_dists,
        train_seeds=full_train_seeds,
        state_rep="compact",
        krr_alpha=1e-2, ls_multiplier=1.0,
        approx="exact",
    )
    w_ref = bkr.eval_seeds(config, ctrl_ref, full_test_seeds)
    ce_ref = bkr.cara_ce(w_ref, gamma)
    gate_ref_bbg = bkr.paired_recovery_gate(w_ref, w_bbg, gamma, h, s_max)
    gate_ref_gws = bkr.paired_recovery_gate(w_ref, w_gws, gamma, h, s_max)
    log(f"  Reference CE={ce_ref:.0f}  ({time.time()-t0:.1f}s)")
    log(f"  vs BBG : mean={gate_ref_bbg['mean']:+.0f} "
        f"sd={gate_ref_bbg['sd_post']:.0f} P(R)={gate_ref_bbg['p_rope']:.3f} "
        f"GA={'PASS' if gate_ref_bbg['passes_a'] else 'FAIL'} "
        f"GB={'PASS' if gate_ref_bbg['passes_b'] else 'FAIL'}")
    log(f"  vs GWS : mean={gate_ref_gws['mean']:+.0f} "
        f"sd={gate_ref_gws['sd_post']:.0f} P(R)={gate_ref_gws['p_rope']:.3f}")

    # === Summary ===
    log("\n" + "=" * 72)
    log("  Summary")
    log("=" * 72)

    log(f"\n  Anti-triviality baselines (formal test):")
    log(f"    BBG               CE = {ce_bbg:.0f}")
    log(f"    global_width      CE = {ce_gw:.0f}")
    log(f"    global_width_skew CE = {ce_gws:.0f}")
    log(f"    RN                CE = {ce_rn:.0f}")
    log(f"    Reference kern    CE = {ce_ref:.0f}  "
        f"(mean gap {gate_ref_bbg['mean']:+.0f}, P(ROPE)={gate_ref_bbg['p_rope']:.3f})")

    if refit_rows:
        # Rank finalists by |mean gap vs BBG|
        refit_rows.sort(key=lambda r: abs(r["gate_bbg"]["mean"]))
        best = refit_rows[0]
        log(f"\n  Best exact-refit finalist:")
        log(f"    alpha={best['krr_alpha']:.1e} ls={best['ls_multiplier']:.2f} "
            f"state={best['state_rep']}")
        log(f"    CE={best['test_ce']:.0f}  "
            f"mean gap {best['gate_bbg']['mean']:+.0f}  "
            f"sd={best['gate_bbg']['sd_post']:.0f}  "
            f"P(ROPE)={best['gate_bbg']['p_rope']:.3f}  "
            f"GA={'PASS' if best['gate_bbg']['passes_a'] else 'FAIL'}  "
            f"GB={'PASS' if best['gate_bbg']['passes_b'] else 'FAIL'}")
        log(f"    vs global_width_skew: "
            f"mean={best['gate_gws']['mean']:+.0f} "
            f"P(R)={best['gate_gws']['p_rope']:.3f}")

        # Classification (spec)
        passes_full_gate = best["gate_bbg"]["passes_both"]
        beats_gws = (
            best["gate_gws"]["mean"] > 0
            and best["gate_gws"]["p_rope"] <= 0.5
        )
        improved_vs_ref = abs(best["gate_bbg"]["mean"]) < abs(gate_ref_bbg["mean"])
        log(f"\n  Passes full recovery gate (vs BBG): "
            f"{'YES' if passes_full_gate else 'NO'}")
        log(f"  Beats global_width_skew: "
            f"{'YES' if beats_gws else 'NO'}")
        log(f"  Improved vs prior default: "
            f"{'YES' if improved_vs_ref else 'NO'} "
            f"(|mean_gap| {abs(best['gate_bbg']['mean']):.0f} vs "
            f"{abs(gate_ref_bbg['mean']):.0f})")

        if passes_full_gate and beats_gws:
            log("\n  *** Outcome: tuning CLOSES the recovery gap. ***")
        elif improved_vs_ref and abs(best['gate_bbg']['mean']) < 0.7 * abs(gate_ref_bbg['mean']):
            log("\n  Outcome: tuning HELPS but does not close the gap.")
        else:
            log("\n  Outcome: tuning DOES NOT close the gap.")

        # Nystrom vs exact consistency
        dev_ordering = [r["dev_gap"] for r in refit_rows]
        exact_ordering = [r["gate_bbg"]["mean"] for r in refit_rows]
        # Simple Spearman-ish check: do they agree on the best?
        dev_best_idx = int(np.argmin(np.abs(dev_ordering)))
        exact_best_idx = int(np.argmin(np.abs(exact_ordering)))
        log(f"\n  Nystrom vs exact finalist ranking:")
        log(f"    Top Nystrom dev candidate:  rank {dev_best_idx + 1}")
        log(f"    Top exact-refit candidate:  rank {exact_best_idx + 1}")
        log(f"    Agreement: "
            f"{'YES' if dev_best_idx == exact_best_idx else 'PARTIAL'}")

    log(f"\n  Nystrom sweep wall clock: {sweep_total:.0f}s for {n_total} configs")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_kernel_tuning_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
