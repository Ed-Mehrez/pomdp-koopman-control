r"""
STUDY: Heston filter target-clarification and horizon / warm-start sensitivity.

Labeled as a STUDY, not a benchmark victory claim and not a new method.
Two parts:

  Study A -- target-clarification study
      Against what quantity is each current lane actually estimating?
      Report filter quality against multiple targets:
        * spot V_true_t
        * one-step-ahead V_true_{t+1}
        * forward-averaged latent variance mean(V_true_{t+1 .. t+h})
        * (companion) forward realized variance mean(dr_S_{t+1..t+h}^2 / dt)

  Study B -- horizon- and warm-start-sensitivity study
      B1  cold-reset per episode at T in {60, 252, 500}.
          Is signature-lane weakness mostly a short-episode cold start?
      B2  warm-persistence within one long T=5000 trajectory per seed.
          Compute corr within [warm-up], [mid], [late] windows; each window
          is a WITHIN-PATH statistic on a filter that has been warming
          since t=0 of that path.  Fair caveat: within-path correlations
          are NOT directly comparable to cross-path RMSE in B1.

Fixed controller: no controller is active in this study (u = 0 throughout
rollouts).  Heston V is action-independent, so this does not change V
dynamics.  This is purely a filter diagnostic.

Forward-averaged latent variance
--------------------------------
The control-relevant forward target is argued in the parent write-up
as the quantity that appears in the CRRA log-wealth integral:

    bar V_{t, h}  :=  (1/h) * sum_{k=1..h} V_true_{t+k}.

This is NOT the instantaneous V_t; it is the expected V trajectory over
the next h steps.  Filters that smooth V toward its mean-reverting target
may track bar V_{t,h} WELL while tracking V_t imperfectly.  That is the
main finding we want the study to either confirm or reject.

Lanes included
--------------
Oracle, EWMA, bayesian_sig, hetero_kalman, blr_kf_leadlag.

No code changes to the controller or filters; this file only consumes
existing lane factories from finance/experiments/merton_value_gradient.py.

Post-study qualitative classification (at this daily-Heston config)
-------------------------------------------------------------------
Backed by the tables this script prints; the labels describe what
each lane appears to be estimating and how, NOT a universal ranking.
These classifications hold at the default config (rho=-0.7, gamma=3,
dt=1/252, CIR (kappa, theta, xi) = (2.0, 0.04, 0.3)).  Do not port
the labels to other regimes without re-running the study.

  - oracle         : reference / upper-bound lane.
  - ewma           : spot-vol tracker.  Best spot-V corr among the
                     tested lanes at this configuration; does not
                     trade spot accuracy for forward accuracy.
  - blr_kf_leadlag : warmup-sensitive model-free spot-vol tracker.
                     Closes to EWMA by T=500; strongest model-free
                     signature lane at this configuration.
  - bayesian_sig   : warmup-sensitive model-free spot-vol tracker,
                     second tier at this configuration.  No longer
                     pathological under the corrected observation
                     pipeline (was NEG_CORR before the fix).
  - hetero_kalman  : mean-reversion-prior tracker /
                     observation-discounting baseline.  The known
                     CIR dynamics + R_t = 2 V^2 / dt produces a
                     Kalman gain that appears to over-discount
                     observations at typical V; flagged for audit.

The spot-vs-forward-target mismatch hypothesis is NOT supported here:
lane rankings are preserved across spot V, one-step V_{t+1}, and
forward-averaged latent V targets, and across h in {5, 10, 20}.

The recommended next step after this study is a targeted audit of
the hetero_kalman steady-state Kalman gain under the heteroskedastic
R_t = 2 V^2 / dt, NOT a new filter method.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv
from merton_value_gradient import (
    VGConfig,
    HVGState,
    _paired_noise,
    _step_state,
    OracleVEstimator,
    EWMAVEstimator,
    BLFVEstimator,
    HeteroKalmanVEstimator,
    LeadLagBLRKFVEstimator,
)
from src.sskf.dual_target_sig_blf import DualTargetSigBLFConfig


# ==========================================================================
# Lane factory (kept local to this file so the study is self-contained)
# ==========================================================================


def make_lane_factories(cfg: VGConfig, env: HestonMertonEnv) -> Dict[str, Callable[[], object]]:
    blf_cfg = DualTargetSigBLFConfig(
        input_dim=2, sig_level=2, sig_forget=0.94,
        prior_var_mu=100.0, prior_var_v=100.0,
        process_noise_mu=1e-4, process_noise_v=1e-4,
        R_init_mu=10.0, R_init_v=0.5,
        R_adapt_halflife=50.0, winsor_v_q=0.995,
    )
    return {
        "oracle":         lambda: OracleVEstimator(),
        "ewma":           lambda: EWMAVEstimator(halflife_days=21.0, dt=cfg.dt),
        "bayesian_sig":   lambda: BLFVEstimator(dt=cfg.dt, blf_config=blf_cfg),
        "hetero_kalman":  lambda: HeteroKalmanVEstimator(env=env, dt=cfg.dt),
        "blr_kf_leadlag": lambda: LeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, ll_gamma=0.99, target_clip=2.0,
        ),
    }


# ==========================================================================
# Single rollout harness (action-free; V-dynamics independent of u)
# ==========================================================================


def _filter_rollout(
    cfg: VGConfig,
    env: HestonMertonEnv,
    lanes: Dict[str, object],
    T: int,
    noise: np.ndarray,
    V0: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    r"""Run one Heston path of length T with u=0.  Filters are NOT reset
    here (so warm-persistence is the caller's choice via `lanes`).

    Target-alignment note
    ---------------------
    The env step driven by pre-step V_t produces dr_S_t with
    E[(dr_S_t)^2 / dt | V_t] = V_t.  After the filter consumes dr_S_t,
    its posterior V_{t|t} is a belief about V_t, NOT about V_{t+1}.
    So the correct "spot V" target for V_hat_post_t is V_true_PRE_t = V_t
    (V just BEFORE the step), not V_true_POST_t = V_{t+1} (V just AFTER).

    Returns
    -------
    V_true_pre_history   : (T,)  V_t, the pre-step V used by the
                           env to generate dr_S_t.  Correct spot target.
    V_true_post_history  : (T,)  V_{t+1}, post-step V.  One-step-ahead
                           target when compared to V_hat_post_t.
    V_hat_post_history   : dict[lane_name -> (T,)]  post-update V_hat
    dr_S_history         : (T,)  underlying asset returns
    """
    state = HVGState(
        logW=0.0, V=V0, t=0, logW0=0.0,
        ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
    )
    V_true_pre = np.zeros(T)
    V_true_post = np.zeros(T)
    V_hat_post = {name: np.zeros(T) for name in lanes}
    dr_S_hist = np.zeros(T)
    for t in range(T):
        V_true_pre[t] = state.V  # V_t, the volatility that dr_S_t will be scaled by
        state, obs = _step_state(
            env, cfg, state, 0.0,
            float(noise[t, 0]), float(noise[t, 1]),
        )
        V_true_post[t] = state.V  # V_{t+1}, post-step
        dr_S_hist[t] = obs["dr_S"]
        for name, est in lanes.items():
            if isinstance(est, OracleVEstimator):
                # Oracle "cheats" by seeing V_t directly (the same V the env
                # used to scale dr_S_t).
                est.set_true_V(V_true_pre[t])
            else:
                est.observe(obs["dr_S"], cfg.dt)
            V_hat_post[name][t] = est.V_hat()
    return V_true_pre, V_true_post, V_hat_post, dr_S_hist


def _fresh_lanes(
    factories: Dict[str, Callable[[], object]], V0: float,
) -> Dict[str, object]:
    lanes = {name: factory() for name, factory in factories.items()}
    for name, est in lanes.items():
        est.reset(V0)
        if isinstance(est, OracleVEstimator):
            est.set_true_V(V0)
    return lanes


# ==========================================================================
# Study A -- target clarification
# ==========================================================================


def build_targets(
    V_true_pre: np.ndarray,
    V_true_post: np.ndarray,
    dr_S: np.ndarray,
    dt: float,
    h: int,
) -> Dict[str, np.ndarray]:
    r"""Build all targets aligned with V_hat_post_t.

    V_hat_post_t is the filter's posterior AFTER observing dr_S_t, which
    is a belief about V_t (== V_true_pre[t]).  Targets:

    * spot_V : V_true_pre[t]  (V_t, the instantaneous variance the filter
               is forming a belief about).
    * one_step_ahead_V : V_true_post[t]  (V_{t+1}).
    * forward_latent_V_h{h} : mean(V_true_post[t .. t+h-1])  == mean of
               V_{t+1} .. V_{t+h}.
    * forward_realized_V_h{h} : mean of (dr_S^2/dt) over t+1 .. t+h.

    All return arrays of shape (T,) with np.nan where undefined.
    """
    T = V_true_pre.size
    targets: Dict[str, np.ndarray] = {}
    # spot: V_{t} (pre-step V at time t)
    targets["spot_V"] = V_true_pre.copy()
    # one-step-ahead: V_{t+1} (post-step V at time t)
    targets["one_step_ahead_V"] = V_true_post.copy()
    # forward-averaged latent: mean of V_{t+1} .. V_{t+h}
    # which is mean of V_true_post[t .. t+h-1]
    fwd_lat = np.full(T, np.nan)
    if T >= h:
        cum = np.concatenate([[0.0], np.cumsum(V_true_post)])
        for t in range(T - h + 1):
            fwd_lat[t] = (cum[t + h] - cum[t]) / h
    targets[f"forward_latent_V_h{h}"] = fwd_lat
    # forward realized: mean of (dr_S)^2/dt over steps t+1 .. t+h
    # where dr_S_k uses V_k (pre-step), so this is noisy version of
    # mean(V_{t} .. V_{t+h-1}).  We drop the first dr_S to align with
    # the V_{t+1} .. V_{t+h} window conceptually.
    rv = (dr_S ** 2) / dt
    fwd_rv = np.full(T, np.nan)
    if T >= h + 1:
        cum_rv = np.concatenate([[0.0], np.cumsum(rv)])
        for t in range(T - h):
            fwd_rv[t] = (cum_rv[t + h + 1] - cum_rv[t + 1]) / h
    targets[f"forward_realized_V_h{h}"] = fwd_rv
    return targets


def _rmse_corr(pred: np.ndarray, target: np.ndarray, warmup: int) -> Tuple[float, float, int]:
    r"""Drop warm-up and NaN entries; return (rmse, corr, n_used)."""
    pred = pred[warmup:]
    target = target[warmup:]
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    p = pred[mask]
    t = target[mask]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    corr = float(np.corrcoef(p, t)[0, 1])
    return rmse, corr, int(mask.sum())


def study_A_target_clarification(
    cfg: VGConfig,
    env: HestonMertonEnv,
    n_seeds: int = 80,
    T: int = 120,
    h: int = 10,
    warmup: int = 20,
    base_seed: int = 8_000,
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    r"""For each lane, compute RMSE and corr against each target.

    Aggregation: stack per-seed arrays, then evaluate on pooled values.
    This matches how the prior filter audits were computed.
    """
    factories = make_lane_factories(cfg, env)
    lane_names = list(factories.keys())
    target_names = [
        "spot_V",
        "one_step_ahead_V",
        f"forward_latent_V_h{h}",
        f"forward_realized_V_h{h}",
    ]
    stacked_hat: Dict[str, List[np.ndarray]] = {name: [] for name in lane_names}
    stacked_targets: Dict[str, List[np.ndarray]] = {name: [] for name in target_names}

    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_seeds):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        noise = _paired_noise(cfg.rho, T, base_seed + 1 + k)
        lanes = _fresh_lanes(factories, V0)
        V_true_pre, V_true_post, V_hat_post, dr_S = _filter_rollout(
            cfg, env, lanes, T, noise, V0,
        )
        targets = build_targets(V_true_pre, V_true_post, dr_S, cfg.dt, h)
        for name in lane_names:
            stacked_hat[name].append(V_hat_post[name])
        for name in target_names:
            stacked_targets[name].append(targets[name])

    # Pool across seeds before computing stats.
    out: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for lane_name in lane_names:
        pred_all = np.concatenate(stacked_hat[lane_name])
        seeds_warmup = warmup
        lane_res: Dict[str, Tuple[float, float, int]] = {}
        # The per-seed arrays are of length T; we drop the first `warmup`
        # entries of each seed BEFORE pooling, not of the pooled array.
        # Build properly.
        pred_trim = np.concatenate(
            [a[warmup:] for a in stacked_hat[lane_name]]
        )
        for tgt_name in target_names:
            tgt_trim = np.concatenate(
                [a[warmup:] for a in stacked_targets[tgt_name]]
            )
            mask = np.isfinite(pred_trim) & np.isfinite(tgt_trim)
            if mask.sum() < 3:
                lane_res[tgt_name] = (float("nan"), float("nan"), 0)
                continue
            p = pred_trim[mask]
            q = tgt_trim[mask]
            rmse = float(np.sqrt(np.mean((p - q) ** 2)))
            corr = float(np.corrcoef(p, q)[0, 1])
            lane_res[tgt_name] = (rmse, corr, int(mask.sum()))
        out[lane_name] = lane_res
    return out


def _print_study_A_table(
    results: Dict[str, Dict[str, Tuple[float, float, int]]],
    h: int,
) -> None:
    target_order = [
        "spot_V",
        "one_step_ahead_V",
        f"forward_latent_V_h{h}",
        f"forward_realized_V_h{h}",
    ]
    header_labels = {
        "spot_V": "spot V",
        "one_step_ahead_V": "one-step V",
        f"forward_latent_V_h{h}": f"fwd_lat_V (h={h})",
        f"forward_realized_V_h{h}": f"fwd_real_V (h={h})",
    }
    print("Study A  -- per-lane filter quality against multiple targets")
    print("-" * 110)
    # Row 1: correlation
    print(f"{'lane':16s} | {'corr vs target':>18s}", end="")
    for tgt in target_order:
        print(f"  {header_labels[tgt]:>16s}", end="")
    print()
    print("-" * 110)
    for lane_name, lane_res in results.items():
        print(f"{lane_name:16s} | {'corr':>18s}", end="")
        for tgt in target_order:
            rmse, corr, n = lane_res[tgt]
            cell = (f"{corr:+.4f}" if np.isfinite(corr) else "    nan")
            print(f"  {cell:>16s}", end="")
        print()
    print("-" * 110)
    print(f"{'lane':16s} | {'RMSE vs target':>18s}", end="")
    for tgt in target_order:
        print(f"  {header_labels[tgt]:>16s}", end="")
    print()
    print("-" * 110)
    for lane_name, lane_res in results.items():
        print(f"{lane_name:16s} | {'rmse':>18s}", end="")
        for tgt in target_order:
            rmse, corr, n = lane_res[tgt]
            cell = (f"{rmse:.4f}" if np.isfinite(rmse) else "   nan")
            print(f"  {cell:>16s}", end="")
        print()
    print()


# ==========================================================================
# Study B1 -- cold-reset horizon sensitivity
# ==========================================================================


def study_B1_cold_reset_horizon(
    cfg_base: VGConfig,
    env: HestonMertonEnv,
    T_list: Tuple[int, ...] = (60, 252, 500),
    n_seeds: int = 40,
    warmup: int = 20,
    base_seed: int = 9_000,
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    r"""Cold-reset per episode; vary T.

    Only spot V is reported (the canonical filter-quality object).  If a
    lane's cold-reset corr vs spot V grows markedly with T, the
    short-episode cold-start is the dominant bottleneck.
    """
    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for T in T_list:
        # Build a config with this T.  VGConfig is frozen; use replace().
        from dataclasses import replace
        cfg = replace(cfg_base, T_steps=T)
        factories = make_lane_factories(cfg, env)
        lane_names = list(factories.keys())

        stacked_hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
        stacked_true: List[np.ndarray] = []

        v0_rng = np.random.RandomState(base_seed + T)
        for k in range(n_seeds):
            V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
            noise = _paired_noise(cfg.rho, T, base_seed + T + 1 + k)
            lanes = _fresh_lanes(factories, V0)
            V_true_pre, _, V_hat_post, _ = _filter_rollout(
                cfg, env, lanes, T, noise, V0,
            )
            for name in lane_names:
                stacked_hat[name].append(V_hat_post[name])
            stacked_true.append(V_true_pre)

        stats: Dict[str, Tuple[float, float]] = {}
        true_trim = np.concatenate([a[warmup:] for a in stacked_true])
        for name in lane_names:
            pred_trim = np.concatenate([a[warmup:] for a in stacked_hat[name]])
            mask = np.isfinite(pred_trim) & np.isfinite(true_trim)
            if mask.sum() < 3:
                stats[name] = (float("nan"), float("nan"))
                continue
            rmse = float(np.sqrt(np.mean((pred_trim[mask] - true_trim[mask]) ** 2)))
            corr = float(np.corrcoef(pred_trim[mask], true_trim[mask])[0, 1])
            stats[name] = (rmse, corr)
        out[T] = stats
    return out


def _print_study_B1_table(
    results: Dict[int, Dict[str, Tuple[float, float]]],
) -> None:
    print("Study B1  -- cold-reset horizon sensitivity (corr vs spot V, post-update)")
    print("  Each row is one horizon T.  For each lane, two columns: corr / RMSE.")
    print("-" * 100)
    T_list = sorted(results.keys())
    lane_names = list(next(iter(results.values())).keys())
    header = f"{'T':>6s}  |"
    for name in lane_names:
        header += f"  {name[:14]:>14s}"
    print(header)
    print("-" * 100)
    for T in T_list:
        line = f"{T:>6d}  |"
        for name in lane_names:
            rmse, corr = results[T][name]
            cell = (
                f"{corr:+.3f}/{rmse:.3f}"
                if np.isfinite(corr) and np.isfinite(rmse)
                else "    nan/nan   "
            )
            line += f"  {cell:>14s}"
        print(line)
    print()


# ==========================================================================
# Study B2 -- warm-persistence within one long path
# ==========================================================================


def study_B2_warm_persistence(
    cfg_base: VGConfig,
    env: HestonMertonEnv,
    T: int = 5000,
    n_seeds: int = 5,
    windows: Tuple[Tuple[int, int], ...] = (
        (20, 500),      # warming
        (500, 2500),    # mid
        (2500, 5000),   # late
    ),
    base_seed: int = 10_000,
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    r"""Single long trajectory per seed; filters warmed from t=0; compute
    per-window statistics.  Report corr and RMSE vs spot V in each window.

    STATISTICAL CAVEAT: within-window correlations here are WITHIN-PATH
    statistics on a filter that has been warming since t=0 of the same
    path.  They are NOT directly comparable to the cross-path B1 table,
    which is aggregated across INDEPENDENT cold-reset episodes.  We
    report both so the interpretation can be kept honest.
    """
    from dataclasses import replace
    cfg = replace(cfg_base, T_steps=T)
    factories = make_lane_factories(cfg, env)
    lane_names = list(factories.keys())

    # Collect per-seed arrays
    per_seed_hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
    per_seed_true: List[np.ndarray] = []

    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_seeds):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        noise = _paired_noise(cfg.rho, T, base_seed + 1 + k)
        lanes = _fresh_lanes(factories, V0)
        V_true_pre, _, V_hat_post, _ = _filter_rollout(
            cfg, env, lanes, T, noise, V0,
        )
        for name in lane_names:
            per_seed_hat[name].append(V_hat_post[name])
        per_seed_true.append(V_true_pre)

    # Per-window stats: pool the window slice across seeds
    out: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for (lo, hi) in windows:
        key = f"t=[{lo},{hi})"
        out[key] = {}
        true_pool = np.concatenate([a[lo:hi] for a in per_seed_true])
        for name in lane_names:
            pred_pool = np.concatenate([a[lo:hi] for a in per_seed_hat[name]])
            mask = np.isfinite(pred_pool) & np.isfinite(true_pool)
            if mask.sum() < 3:
                out[key][name] = {"spot_V": (float("nan"), float("nan"))}
                continue
            rmse = float(np.sqrt(np.mean((pred_pool[mask] - true_pool[mask]) ** 2)))
            corr = float(np.corrcoef(pred_pool[mask], true_pool[mask])[0, 1])
            out[key][name] = {"spot_V": (rmse, corr)}
    return out


def _print_study_B2_table(
    results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
) -> None:
    print("Study B2  -- warm-persistence within one long T=5000 path "
          "(corr vs spot V, post-update)")
    print("  Windows are within-path slices; filter warms from t=0 of that path.")
    print("  CAVEAT: NOT directly comparable to B1 (cross-path cold-reset).")
    print("-" * 100)
    windows = list(results.keys())
    lane_names = list(next(iter(results.values())).keys())
    header = f"{'window':>16s}  |"
    for name in lane_names:
        header += f"  {name[:14]:>14s}"
    print(header)
    print("-" * 100)
    for w in windows:
        line = f"{w:>16s}  |"
        for name in lane_names:
            rmse, corr = results[w][name]["spot_V"]
            cell = (
                f"{corr:+.3f}/{rmse:.3f}"
                if np.isfinite(corr) and np.isfinite(rmse)
                else "    nan/nan   "
            )
            line += f"  {cell:>14s}"
        print(line)
    print()


# ==========================================================================
# Figure
# ==========================================================================


def _plot_target_clarification(
    cfg: VGConfig,
    env: HestonMertonEnv,
    h: int,
    out_path: str,
    T: int = 300,
    seed: int = 42,
) -> None:
    r"""Plot one representative trajectory: V_true, V_hat_post per lane,
    and forward-averaged latent V target.  One figure, lane-colored lines.
    """
    from dataclasses import replace
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg2 = replace(cfg, T_steps=T)
    factories = make_lane_factories(cfg2, env)
    noise = _paired_noise(cfg.rho, T, seed)
    V0 = float(np.random.RandomState(seed).uniform(cfg.V0_low, cfg.V0_high))
    lanes = _fresh_lanes(factories, V0)
    V_true_pre, V_true_post, V_hat_post, dr_S = _filter_rollout(
        cfg2, env, lanes, T, noise, V0,
    )
    targets = build_targets(V_true_pre, V_true_post, dr_S, cfg.dt, h)
    t_axis = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(t_axis, V_true_pre, "k-", lw=1.6, label="V_true (spot = V_t)")
    ax.plot(t_axis, targets[f"forward_latent_V_h{h}"], "k--", lw=1.0,
            label=f"forward-averaged latent V, h={h}")
    ax.set_ylabel("V")
    ax.set_title(f"Heston V trajectory (seed={seed}, T={T})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis, V_true_pre, "k-", lw=1.2, alpha=0.5, label="V_true (spot)")
    palette = {
        "oracle": "tab:green",
        "ewma": "tab:orange",
        "bayesian_sig": "tab:blue",
        "hetero_kalman": "tab:purple",
        "blr_kf_leadlag": "tab:red",
    }
    for name, vh in V_hat_post.items():
        if name == "oracle":
            continue  # same as V_true
        ax.plot(t_axis, vh, color=palette.get(name, None),
                lw=1.0, alpha=0.85, label=name)
    ax.plot(t_axis, targets[f"forward_latent_V_h{h}"], "k--", lw=1.0, alpha=0.7,
            label=f"fwd-avg latent V (h={h})")
    ax.set_xlabel("step t")
    ax.set_ylabel("V / V_hat")
    ax.set_title("V_hat_post per lane vs spot V and forward-averaged V")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Heston filter study: spot vs forward-averaged V  "
        f"(rho={cfg.rho}, gamma={cfg.gamma}, dt={cfg.dt:.5f}, h={h})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Runner
# ==========================================================================


def main() -> None:
    cfg = VGConfig()
    env = HestonMertonEnv(rho=cfg.rho, gamma=cfg.gamma)

    print("=" * 110)
    print("STUDY: Heston filter target clarification and horizon / warm-start sensitivity")
    print(f"  config: rho={cfg.rho}, gamma={cfg.gamma}, dt={cfg.dt:.5f},"
          f" V0 uniform [{cfg.V0_low}, {cfg.V0_high}]")
    print("  lanes : oracle, ewma, bayesian_sig, hetero_kalman, blr_kf_leadlag")
    print("=" * 110)

    # Study A
    print()
    print("Study A -- target clarification  (n_seeds=80, T=120, h=10, warm-up=20)")
    resA = study_A_target_clarification(
        cfg, env, n_seeds=80, T=120, h=10, warmup=20,
    )
    _print_study_A_table(resA, h=10)

    # Study A sensitivity to h
    print()
    print("Study A sensitivity to forward window h (n_seeds=60, T=200, warm-up=20)")
    for h_test in (5, 10, 20):
        print()
        print(f"  (h = {h_test})")
        resA_h = study_A_target_clarification(
            cfg, env, n_seeds=60, T=200, h=h_test, warmup=20,
            base_seed=8_000 + h_test * 100,
        )
        _print_study_A_table(resA_h, h=h_test)

    # Study B1
    print()
    print("Study B1 -- cold-reset horizon sensitivity  (n_seeds=40, warm-up=20)")
    resB1 = study_B1_cold_reset_horizon(
        cfg, env, T_list=(60, 252, 500), n_seeds=40, warmup=20,
    )
    _print_study_B1_table(resB1)

    # Study B2
    print()
    print("Study B2 -- warm-persistence within one long T=5000 path (n_seeds=5)")
    resB2 = study_B2_warm_persistence(
        cfg, env, T=5000, n_seeds=5,
        windows=((20, 500), (500, 2500), (2500, 5000)),
    )
    _print_study_B2_table(resB2)

    # Figure
    plot_path = os.path.join(HERE, "study_heston_filter_targets.png")
    try:
        _plot_target_clarification(cfg, env, h=10, out_path=plot_path)
        print(f"Saved figure: {plot_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")


if __name__ == "__main__":
    main()
