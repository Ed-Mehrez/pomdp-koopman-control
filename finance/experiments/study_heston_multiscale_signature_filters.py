r"""
STUDY: multiresolution signature-filter comparison on Heston (filter-only).

Labeled as a STUDY.  NOT a benchmark victory claim; NOT a new control
method; NOT a universal ranking.  At a single Heston configuration
(rho=-0.7, gamma=3, dt=1/252, CIR (kappa,theta,xi)=(2,0.04,0.3)) we
compare the filter quality of several signature-feature lanes that
differ only in their SCALE STRUCTURE:

  * ewma                  -- single-scale realized-variance tracker (baseline)
  * blr_kf_leadlag        -- single-scale lead-lag + 3-feature BLR + outer KF
                             (existing strong signature lane, gamma=0.99)
  * bayesian_sig          -- dual-target signature BLF (existing baseline)
  * ms_forget_fixed       -- K parallel lead-lag sig states at FIXED
                             calendar-time forgetting ladder (1/5/20 days)
  * ms_forget_spectral    -- same architecture as ms_forget_fixed, but the
                             ladder is centered at a DATA-DRIVEN tau_hat
                             estimated from a pilot warm-up of EWMA(r^2/dt)
  * ms_cum_stride         -- ONE cumulative (gamma=1.0) lead-lag sig +
                             ring-buffer + Chen-level-2 window recovery at
                             strides (1/5/20 days) + per-stride BLR

All signature lanes use the CORRECTED observation pipeline (filters
consume the underlying-asset return dr_S, not the wealth return).
Controller is OFF (u=0); Heston V dynamics are action-independent, so
this is a clean filter-only comparison.

Methodological notes
--------------------
* Scale fusion.  For the multiscale lanes, the K Bayesian predictive
  Gaussians (y_k, R_k) are combined into one scalar observation of V by
  precision-weighted averaging:
      1/R_bar = sum_k 1/R_k,   y_bar = R_bar * sum_k y_k/R_k.
  Strict conditional independence across scales is NOT held; the K
  observations share the same dr_S stream.  Documented, not hidden.

* Pilot-tau for the spectral ladder.  The ladder for ms_forget_spectral
  is built ONCE from a pilot pass over `n_pilot` paths of length
  `T_pilot`.  On each path we feed dr_S^2 / dt through an EWMA (hl=21d)
  to form a smoothed realized-variance series, then call
  `estimate_variance_timescale(...)` to get the 1/e lag of the centered
  ACF.  tau_hat is the median across pilot paths.  This is a FROZEN
  pre-computation (same ladder for all evaluation seeds), not an online
  adaptation.

* Warm-up handling.  Multiresolution lanes that include short scales
  (e.g. tau ~= dt) can be effectively identical to a 1-step EWMA there;
  the long-scale heads (tau ~= 20d) take at least a few mean-reversion
  autocorrelation windows to stabilize.  Stats are reported after
  dropping the first `warmup` steps of each episode.

* One Heston config only.  Results are reported at the default config.
  Rankings may differ under different (kappa, theta, xi), a different
  observation dt, or a different warmup budget.  Caveat applied to ALL
  findings below.

Reporting contract
------------------
We report corr and RMSE against three targets:
  - spot V (V_t)
  - one-step-ahead V (V_{t+1})
  - forward-averaged latent V over h days (mean of V_{t+1..t+h})
and cold-reset horizon sensitivity at T in {60, 252}.  One slide-ready
figure shows a representative trajectory with all lanes overlaid.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv
from merton_value_gradient import (
    BLFVEstimator,
    CumulativeStrideLeadLagBLRKFVEstimator,
    EWMAVEstimator,
    HVGState,
    LeadLagBLRKFVEstimator,
    MultiScaleLeadLagBLRKFVEstimator,
    OracleVEstimator,
    VGConfig,
    _paired_noise,
    _step_state,
)
from study_heston_filter_targets_and_horizon import (
    _filter_rollout,
    _fresh_lanes,
    build_targets,
)
from src.sskf.dual_target_sig_blf import DualTargetSigBLFConfig
from src.sskf.multiscale_leadlag_filters import (
    estimate_variance_timescale,
    fixed_calendar_ladder,
    ladder_from_timescale,
)


# ==========================================================================
# Pilot tau estimation for the spectral multiresolution ladder
# ==========================================================================


def estimate_pilot_tau_years(
    cfg: VGConfig,
    env: HestonMertonEnv,
    n_pilot: int = 20,
    T_pilot: int = 500,
    ewma_halflife_days: float = 21.0,
    base_seed: int = 7_000,
) -> float:
    r"""Pilot estimate of the variance decorrelation timescale in years.

    For each pilot seed: run a u=0 Heston path, compute r^2/dt, pass
    through EWMA with hl=21d to smooth into a realized-variance proxy,
    then compute the 1/e autocorrelation lag via
    `estimate_variance_timescale`.  Return the median tau_hat across
    pilot seeds.
    """
    lam = float(np.log(2.0) / max(ewma_halflife_days, 1e-3))
    alpha = 1.0 - float(np.exp(-lam))
    taus: List[float] = []
    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_pilot):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        noise = _paired_noise(cfg.rho, T_pilot, base_seed + 1 + k)
        state = HVGState(
            logW=0.0, V=V0, t=0, logW0=0.0,
            ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
        )
        rv = np.zeros(T_pilot)
        smooth = V0
        for t in range(T_pilot):
            state, obs = _step_state(
                env, cfg, state, 0.0,
                float(noise[t, 0]), float(noise[t, 1]),
            )
            y = obs["dr_S"] ** 2 / cfg.dt
            smooth = (1.0 - alpha) * smooth + alpha * y
            rv[t] = smooth
        tau_k = estimate_variance_timescale(
            rv, dt=cfg.dt, max_lag_fraction=0.2,
            min_tau_days=2.0, max_tau_days=200.0,
        )
        taus.append(tau_k)
    return float(np.median(taus))


# ==========================================================================
# Lane factories (include existing + multiresolution variants)
# ==========================================================================


def make_lane_factories(
    cfg: VGConfig,
    env: HestonMertonEnv,
    tau_spectral_years: float,
    ms_fixed_days: Tuple[float, ...] = (1.0, 5.0, 20.0),
    ms_stride_days: Tuple[float, ...] = (1.0, 5.0, 20.0),
    ms_spectral_fan: float = 4.0,
    ms_spectral_nscales: int = 3,
) -> Dict[str, Callable[[], object]]:
    blf_cfg = DualTargetSigBLFConfig(
        input_dim=2, sig_level=2, sig_forget=0.94,
        prior_var_mu=100.0, prior_var_v=100.0,
        process_noise_mu=1e-4, process_noise_v=1e-4,
        R_init_mu=10.0, R_init_v=0.5,
        R_adapt_halflife=50.0, winsor_v_q=0.995,
    )
    taus_fixed = fixed_calendar_ladder(cfg.dt, days=ms_fixed_days)
    taus_stride = fixed_calendar_ladder(cfg.dt, days=ms_stride_days)
    taus_spectral = ladder_from_timescale(
        tau_spectral_years, n_scales=ms_spectral_nscales, fan=ms_spectral_fan,
    )
    return {
        "oracle":              lambda: OracleVEstimator(),
        "ewma":                lambda: EWMAVEstimator(halflife_days=21.0, dt=cfg.dt),
        "bayesian_sig":        lambda: BLFVEstimator(dt=cfg.dt, blf_config=blf_cfg),
        "blr_kf_leadlag":      lambda: LeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, ll_gamma=0.99, target_clip=2.0,
        ),
        "ms_forget_fixed":     lambda: MultiScaleLeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, taus_years=list(taus_fixed), target_clip=2.0,
        ),
        "ms_forget_spectral":  lambda: MultiScaleLeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, taus_years=list(taus_spectral), target_clip=2.0,
        ),
        "ms_cum_stride":       lambda: CumulativeStrideLeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, taus_years=list(taus_stride), target_clip=2.0,
        ),
    }


# ==========================================================================
# Core study: target comparison across lanes
# ==========================================================================


def study_target_comparison(
    cfg: VGConfig,
    env: HestonMertonEnv,
    factories: Dict[str, Callable[[], object]],
    n_seeds: int = 80,
    T: int = 200,
    h: int = 10,
    warmup: int = 20,
    base_seed: int = 8_500,
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    r"""Pool-across-seeds corr / RMSE against each target for each lane."""
    lane_names = list(factories.keys())
    target_names = [
        "spot_V",
        "one_step_ahead_V",
        f"forward_latent_V_h{h}",
    ]
    stacked_hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
    stacked_tgt: Dict[str, List[np.ndarray]] = {n: [] for n in target_names}

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
            stacked_tgt[name].append(targets[name])

    out: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for lane in lane_names:
        lane_res: Dict[str, Tuple[float, float, int]] = {}
        pred = np.concatenate([a[warmup:] for a in stacked_hat[lane]])
        for tgt in target_names:
            target_arr = np.concatenate(
                [a[warmup:] for a in stacked_tgt[tgt]]
            )
            mask = np.isfinite(pred) & np.isfinite(target_arr)
            if mask.sum() < 3:
                lane_res[tgt] = (float("nan"), float("nan"), 0)
                continue
            p = pred[mask]
            q = target_arr[mask]
            rmse = float(np.sqrt(np.mean((p - q) ** 2)))
            corr = float(np.corrcoef(p, q)[0, 1])
            lane_res[tgt] = (rmse, corr, int(mask.sum()))
        out[lane] = lane_res
    return out


def _print_target_table(
    results: Dict[str, Dict[str, Tuple[float, float, int]]],
    h: int,
    title: str,
) -> None:
    target_order = [
        "spot_V",
        "one_step_ahead_V",
        f"forward_latent_V_h{h}",
    ]
    labels = {
        "spot_V": "spot V",
        "one_step_ahead_V": "one-step V",
        f"forward_latent_V_h{h}": f"fwd_lat_V (h={h})",
    }
    print(title)
    print("-" * 110)
    print(f"{'lane':20s} | {'corr':>6s} |", end="")
    for tgt in target_order:
        print(f"  {labels[tgt]:>18s}", end="")
    print()
    print("-" * 110)
    for lane, lane_res in results.items():
        print(f"{lane:20s} | {'corr':>6s} |", end="")
        for tgt in target_order:
            _, corr, _ = lane_res[tgt]
            cell = f"{corr:+.4f}" if np.isfinite(corr) else "    nan"
            print(f"  {cell:>18s}", end="")
        print()
    print("-" * 110)
    print(f"{'lane':20s} | {'rmse':>6s} |", end="")
    for tgt in target_order:
        print(f"  {labels[tgt]:>18s}", end="")
    print()
    print("-" * 110)
    for lane, lane_res in results.items():
        print(f"{lane:20s} | {'rmse':>6s} |", end="")
        for tgt in target_order:
            rmse, _, _ = lane_res[tgt]
            cell = f"{rmse:.4f}" if np.isfinite(rmse) else "   nan"
            print(f"  {cell:>18s}", end="")
        print()
    print()


# ==========================================================================
# Horizon sensitivity (cold-reset)
# ==========================================================================


def study_cold_reset_horizon(
    cfg_base: VGConfig,
    env: HestonMertonEnv,
    factories_builder: Callable[[VGConfig, HestonMertonEnv], Dict[str, Callable[[], object]]],
    T_list: Tuple[int, ...] = (60, 252),
    n_seeds: int = 30,
    warmup: int = 20,
    base_seed: int = 9_500,
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for T in T_list:
        cfg = replace(cfg_base, T_steps=T)
        factories = factories_builder(cfg, env)
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


def _print_cold_reset_table(
    results: Dict[int, Dict[str, Tuple[float, float]]],
) -> None:
    print("Cold-reset horizon sensitivity (corr vs spot V, post-update / RMSE)")
    print("-" * 140)
    T_list = sorted(results.keys())
    lane_names = list(next(iter(results.values())).keys())
    header = f"{'T':>6s}  |"
    for name in lane_names:
        header += f"  {name[:18]:>18s}"
    print(header)
    print("-" * 140)
    for T in T_list:
        line = f"{T:>6d}  |"
        for name in lane_names:
            rmse, corr = results[T][name]
            cell = (
                f"{corr:+.3f}/{rmse:.3f}"
                if np.isfinite(corr) and np.isfinite(rmse)
                else "    nan/nan"
            )
            line += f"  {cell:>18s}"
        print(line)
    print()


# ==========================================================================
# One slide-ready figure
# ==========================================================================


def _plot_representative_trajectory(
    cfg: VGConfig,
    env: HestonMertonEnv,
    factories: Dict[str, Callable[[], object]],
    out_path: str,
    T: int = 300,
    seed: int = 42,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg2 = replace(cfg, T_steps=T)
    noise = _paired_noise(cfg.rho, T, seed)
    V0 = float(np.random.RandomState(seed).uniform(cfg.V0_low, cfg.V0_high))
    lanes = _fresh_lanes(factories, V0)
    V_true_pre, V_true_post, V_hat_post, dr_S = _filter_rollout(
        cfg2, env, lanes, T, noise, V0,
    )
    t_axis = np.arange(T)
    palette = {
        "oracle":             "tab:green",
        "ewma":               "tab:orange",
        "bayesian_sig":       "tab:blue",
        "blr_kf_leadlag":     "tab:red",
        "ms_forget_fixed":    "tab:purple",
        "ms_forget_spectral": "tab:brown",
        "ms_cum_stride":      "tab:cyan",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax = axes[0]
    ax.plot(t_axis, V_true_pre, "k-", lw=1.6, label="V_true (spot)")
    for name, vh in V_hat_post.items():
        if name in ("oracle",):
            continue
        if name in ("ewma", "bayesian_sig", "blr_kf_leadlag"):
            ax.plot(
                t_axis, vh, color=palette.get(name), lw=1.0, alpha=0.85,
                label=name,
            )
    ax.set_ylabel("V")
    ax.set_title("Single-scale lanes: V_hat vs V_true (spot)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis, V_true_pre, "k-", lw=1.6, label="V_true (spot)")
    for name, vh in V_hat_post.items():
        if name not in ("ms_forget_fixed", "ms_forget_spectral", "ms_cum_stride"):
            continue
        ax.plot(
            t_axis, vh, color=palette.get(name), lw=1.2, alpha=0.9,
            label=name,
        )
    ax.set_xlabel("step t")
    ax.set_ylabel("V")
    ax.set_title("Multiresolution lanes: V_hat vs V_true (spot)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Heston multiresolution signature-filter study  "
        f"(rho={cfg.rho}, gamma={cfg.gamma}, dt={cfg.dt:.5f}, seed={seed})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Summary: direct answers
# ==========================================================================


def _print_direct_answers(
    results_A: Dict[str, Dict[str, Tuple[float, float, int]]],
    h: int,
    tau_spectral_years: float,
    dt: float,
) -> None:
    def corr_of(lane: str, tgt: str) -> float:
        entry = results_A.get(lane, {}).get(tgt, None)
        return float(entry[1]) if entry is not None else float("nan")

    print("=" * 110)
    print("DIRECT ANSWERS (at this daily-Heston config; NOT a universal ranking)")
    print("=" * 110)

    best_single = max(
        ["ewma", "blr_kf_leadlag", "bayesian_sig"],
        key=lambda L: corr_of(L, "spot_V"),
    )
    best_ms = max(
        ["ms_forget_fixed", "ms_forget_spectral", "ms_cum_stride"],
        key=lambda L: corr_of(L, "spot_V"),
    )
    gap = corr_of(best_ms, "spot_V") - corr_of(best_single, "spot_V")
    print()
    print("Q1.  Does multiresolution help at this config?")
    print(f"     Best single-scale lane: {best_single} "
          f"(corr spot V = {corr_of(best_single, 'spot_V'):+.4f})")
    print(f"     Best multires   lane : {best_ms} "
          f"(corr spot V = {corr_of(best_ms, 'spot_V'):+.4f})")
    print(f"     Multires - single gap : {gap:+.4f}")
    verdict = (
        "YES, multiresolution improves spot-V tracking"
        if gap > 0.02
        else "NO material improvement (within ~0.02 corr noise)"
        if abs(gap) <= 0.02
        else "NO, multiresolution underperforms"
    )
    print(f"     Verdict               : {verdict}")

    print()
    print("Q2.  Does cumulative-stride beat fixed forgetting-factor at this config?")
    corr_cs = corr_of("ms_cum_stride", "spot_V")
    corr_ff = corr_of("ms_forget_fixed", "spot_V")
    delta = corr_cs - corr_ff
    print(f"     ms_cum_stride         (corr spot V) = {corr_cs:+.4f}")
    print(f"     ms_forget_fixed       (corr spot V) = {corr_ff:+.4f}")
    print(f"     Δ (cum_stride - forget_fixed)       = {delta:+.4f}")
    verdict_cs = (
        "YES, cumulative-stride wins" if delta > 0.02
        else "NO material difference" if abs(delta) <= 0.02
        else "NO, fixed forgetting wins"
    )
    print(f"     Verdict                              : {verdict_cs}")

    print()
    print("Q3.  Does data-driven spectral ladder beat fixed calendar (1/5/20 days)?")
    corr_sp = corr_of("ms_forget_spectral", "spot_V")
    delta2 = corr_sp - corr_ff
    tau_days = tau_spectral_years / dt
    print(f"     pilot tau_hat         = {tau_spectral_years:.4f} yr = {tau_days:.1f} days")
    print(f"     ms_forget_spectral    (corr spot V) = {corr_sp:+.4f}")
    print(f"     ms_forget_fixed       (corr spot V) = {corr_ff:+.4f}")
    print(f"     Δ (spectral - fixed)                = {delta2:+.4f}")
    verdict_sp = (
        "YES, spectral ladder wins" if delta2 > 0.02
        else "NO material difference" if abs(delta2) <= 0.02
        else "NO, fixed calendar ladder wins"
    )
    print(f"     Verdict                              : {verdict_sp}")

    print()
    print("Q4.  Recommendation for DEFAULT signature lane at this config:")
    all_sig_lanes = [
        "bayesian_sig", "blr_kf_leadlag",
        "ms_forget_fixed", "ms_forget_spectral", "ms_cum_stride",
    ]
    best = max(all_sig_lanes, key=lambda L: corr_of(L, "spot_V"))
    ewma_corr = corr_of("ewma", "spot_V")
    best_corr = corr_of(best, "spot_V")
    print(f"     Best signature lane on spot V: {best} (corr = {best_corr:+.4f})")
    print(f"     EWMA baseline (corr spot V)  : {ewma_corr:+.4f}")
    if best_corr > ewma_corr + 0.01:
        rec = best
        note = "signature lane edges EWMA on spot V"
    else:
        rec = "ewma (baseline) or " + best
        note = ("signature lane does not materially outperform EWMA on spot V; "
                "pick signature only if additional diagnostics are needed")
    print(f"     Recommendation               : {rec}")
    print(f"     Note                         : {note}")
    print("=" * 110)


# ==========================================================================
# Runner
# ==========================================================================


def main() -> None:
    cfg = VGConfig()
    env = HestonMertonEnv(rho=cfg.rho, gamma=cfg.gamma)

    print("=" * 110)
    print("STUDY: multiresolution signature-filter comparison on Heston (filter-only)")
    print(f"  config: rho={cfg.rho}  gamma={cfg.gamma}  dt={cfg.dt:.5f}"
          f"  V0 uniform [{cfg.V0_low}, {cfg.V0_high}]")
    print(f"  CIR   : kappa={env.kappa}  theta={env.theta}  xi={env.xi}"
          f"   =>  1/kappa = {1.0/env.kappa:.2f} yr"
          f"  =  {1.0/(env.kappa*cfg.dt):.1f} days")
    print("  lanes : oracle, ewma, bayesian_sig, blr_kf_leadlag, "
          "ms_forget_fixed, ms_forget_spectral, ms_cum_stride")
    print("=" * 110)

    # Pilot tau estimation
    print()
    print("Pilot tau estimation (EWMA(r^2/dt) ACF 1/e lag, median over pilot seeds)")
    tau_spectral = estimate_pilot_tau_years(
        cfg, env, n_pilot=20, T_pilot=500, ewma_halflife_days=21.0,
        base_seed=7_000,
    )
    tau_days = tau_spectral / cfg.dt
    print(f"  pilot tau_hat = {tau_spectral:.5f} yr = {tau_days:.2f} days")
    ladder = ladder_from_timescale(tau_spectral, n_scales=3, fan=4.0)
    print(f"  spectral ladder (fan=4, 3 scales): "
          f"{[f'{t*252:.1f}d' for t in ladder]}  (= {ladder} yr)")

    def factories_builder(
        cfg_local: VGConfig, env_local: HestonMertonEnv,
    ) -> Dict[str, Callable[[], object]]:
        return make_lane_factories(cfg_local, env_local, tau_spectral)

    factories = factories_builder(cfg, env)

    # Study 1: target comparison
    print()
    print("STUDY 1  -- target-clarification comparison  (n_seeds=80, T=200, h=10, warm-up=20)")
    resA = study_target_comparison(
        cfg, env, factories, n_seeds=80, T=200, h=10, warmup=20,
        base_seed=8_500,
    )
    _print_target_table(
        resA, h=10,
        title="Study 1 -- corr and RMSE vs multiple targets (pooled across seeds)",
    )

    # Study 2: cold-reset horizon sensitivity
    print()
    print("STUDY 2  -- cold-reset horizon sensitivity  (n_seeds=30, warm-up=20)")
    resB = study_cold_reset_horizon(
        cfg, env, factories_builder, T_list=(60, 252),
        n_seeds=30, warmup=20, base_seed=9_500,
    )
    _print_cold_reset_table(resB)

    # Figure
    plot_path = os.path.join(HERE, "study_heston_multiscale_signature_filters.png")
    try:
        _plot_representative_trajectory(
            cfg, env, factories, out_path=plot_path, T=300, seed=42,
        )
        print(f"Saved figure: {plot_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")

    # Direct answers
    _print_direct_answers(resA, h=10, tau_spectral_years=tau_spectral, dt=cfg.dt)


if __name__ == "__main__":
    main()
