r"""
BENCHMARK STUDY: Heston filter comparison at 5-minute bars.

This is Row 2 of the validation ladder in
`docs/benchmark_ladder_gated_compression.md`.  At 5-min bars, the
observation frequency is 78 bars/day, so the intraday-to-daily timescale
ratio is substantially larger than in the daily-Heston study.  The
expectation is that fixed-halflife EWMA becomes sensitive to halflife
choice and that multiresolution signature lanes either (a) match EWMA
across a broader range of baselines, or (b) beat the best single EWMA.

No controller (u=0).  Heston V is action-independent, so this is a pure
filter comparison.  Filters consume the underlying-asset return dr_S
(not wealth return).

Lanes compared
--------------
  ewma_5min_1d   : scalar baseline A.  EWMA(r^2/dt) at halflife = 1
                   trading day (78 bars).
  ewma_5min_5d   : scalar baseline B.  EWMA(r^2/dt) at halflife = 5
                   trading days (390 bars).
  blr_kf_leadlag : single-scale lead-lag + 3-feature BLR + outer KF,
                   gamma = 0.99 (about 100-step window).
  ms_cum_stride  : multires cumulative-stride lead-lag BLR+KF,
                   calendar strides (1 hr, 1 day, 5 days) in bars.
  ms_forget_spec : multires forgetting-factor lead-lag BLR+KF with a
                   pilot-estimated spectral ladder (data-driven tau_hat).

Heston config: same as the daily study (rho=-0.7, kappa=2.0, theta=0.04,
xi=0.3).  dt = 1/(252*78).

Reporting
---------
  Primary : pooled corr and RMSE of V_hat_post vs V_true_pre (spot V).
  Secondary: corr vs forward-averaged latent V over h = 78 bars (one day).
  Tertiary : one representative trajectory figure.

Pre-registered interpretation bars (set before running):
  - ms_cum_stride beats EWMA by > +0.03 corr  ->  clear gate flip.
  - ms_cum_stride within +-0.02 corr of EWMA AND clearly above other
    signature variants  ->  strong positive architectural result.
  - EWMA still wins comfortably (> +0.03 over all signature lanes) ->
    the easy-case gate extends to intraday Heston at this config.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv
from merton_value_gradient import (
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
from src.sskf.multiscale_leadlag_filters import (
    estimate_variance_timescale,
    fixed_calendar_ladder,
    ladder_from_timescale,
)


BARS_PER_DAY = 78
BARS_PER_YEAR = 252 * BARS_PER_DAY   # 19656
DT_5MIN = 1.0 / BARS_PER_YEAR


# ==========================================================================
# Pilot tau (5-min)
# ==========================================================================


def estimate_pilot_tau_years_5min(
    cfg: VGConfig,
    env: HestonMertonEnv,
    n_pilot: int = 8,
    T_pilot_days: int = 30,
    ewma_halflife_bars: float = BARS_PER_DAY,
    base_seed: int = 7_500,
) -> float:
    r"""Pilot tau estimation at 5-min cadence.

    EWMA of r^2/dt with halflife = ewma_halflife_bars steps (default: 78
    bars = 1 trading day).  Then estimate the 1/e lag of the smoothed
    series, capped to the CIR-admissible range.
    """
    lam = float(np.log(2.0) / max(ewma_halflife_bars, 1e-3))
    alpha = 1.0 - float(np.exp(-lam))
    taus: List[float] = []
    T_pilot = int(T_pilot_days * BARS_PER_DAY)
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
            min_tau_days=0.5, max_tau_days=250.0,
        )
        taus.append(tau_k)
    return float(np.median(taus))


# ==========================================================================
# Lane factories at 5-min
# ==========================================================================


def make_lane_factories_5min(
    cfg: VGConfig,
    env: HestonMertonEnv,
    tau_spectral_years: float,
    cum_stride_bars: Tuple[float, ...] = (BARS_PER_DAY / 6.0,  # ~1 hr
                                           BARS_PER_DAY * 1.0,  # 1 day
                                           BARS_PER_DAY * 5.0), # 5 days
    ms_spectral_fan: float = 4.0,
    ms_spectral_nscales: int = 3,
) -> Dict[str, Callable[[], object]]:
    r"""Lane factories for the 5-min study.  Scalar-EWMA halflives are
    specified in BARS (not days), since VGConfig's halflife_days field
    is actually a halflife-in-steps under the hood.
    """
    halflife_1d_bars = BARS_PER_DAY
    halflife_5d_bars = 5 * BARS_PER_DAY
    cum_taus = [float(b * cfg.dt) for b in cum_stride_bars]
    spectral_taus = ladder_from_timescale(
        tau_spectral_years, n_scales=ms_spectral_nscales, fan=ms_spectral_fan,
    )
    return {
        "oracle":         lambda: OracleVEstimator(),
        "ewma_5min_1d":   lambda: EWMAVEstimator(
            halflife_days=halflife_1d_bars, dt=cfg.dt,
        ),
        "ewma_5min_5d":   lambda: EWMAVEstimator(
            halflife_days=halflife_5d_bars, dt=cfg.dt,
        ),
        "blr_kf_leadlag": lambda: LeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, ll_gamma=0.99, target_clip=None,
        ),
        "ms_cum_stride":  lambda: CumulativeStrideLeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, taus_years=list(cum_taus), target_clip=None,
        ),
        "ms_forget_spec": lambda: MultiScaleLeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, taus_years=list(spectral_taus), target_clip=None,
        ),
    }


# ==========================================================================
# Core comparison
# ==========================================================================


def study_5min_comparison(
    cfg: VGConfig,
    env: HestonMertonEnv,
    factories: Dict[str, Callable[[], object]],
    n_seeds: int,
    T: int,
    h: int,
    warmup: int,
    base_seed: int,
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    lane_names = list(factories.keys())
    target_names = ["spot_V", "one_step_ahead_V", f"forward_latent_V_h{h}"]
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
        pred = np.concatenate([a[warmup:] for a in stacked_hat[lane]])
        lane_res: Dict[str, Tuple[float, float, int]] = {}
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


def _print_table(
    results: Dict[str, Dict[str, Tuple[float, float, int]]],
    h: int,
    title: str,
) -> None:
    order = ["spot_V", "one_step_ahead_V", f"forward_latent_V_h{h}"]
    labels = {
        "spot_V": "spot V",
        "one_step_ahead_V": "one-step V",
        f"forward_latent_V_h{h}": f"fwd_lat_V (h={h}b)",
    }
    print(title)
    print("-" * 118)
    print(f"{'lane':20s} | {'stat':>6s} |", end="")
    for tgt in order:
        print(f"  {labels[tgt]:>20s}", end="")
    print()
    print("-" * 118)
    for lane, lane_res in results.items():
        print(f"{lane:20s} | {'corr':>6s} |", end="")
        for tgt in order:
            _, corr, _ = lane_res[tgt]
            cell = f"{corr:+.4f}" if np.isfinite(corr) else "    nan"
            print(f"  {cell:>20s}", end="")
        print()
    print("-" * 118)
    for lane, lane_res in results.items():
        print(f"{lane:20s} | {'rmse':>6s} |", end="")
        for tgt in order:
            rmse, _, _ = lane_res[tgt]
            cell = f"{rmse:.4f}" if np.isfinite(rmse) else "   nan"
            print(f"  {cell:>20s}", end="")
        print()
    print()


# ==========================================================================
# Warm-start ablation (Benchmark 3)
# ==========================================================================


def warm_start_ablation(
    cfg: VGConfig,
    env: HestonMertonEnv,
    scalar_factory: Callable[[], object],
    signature_factory: Callable[[], object],
    scalar_name: str,
    signature_name: str,
    T: int,
    warm_windows: Tuple[Tuple[int, int], ...],
    n_seeds: int,
    base_seed: int,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    r"""Warm-persistent within one long path per seed.

    Filters are reset ONCE at t=0 of the path and allowed to warm through
    the path.  Stats are computed inside each (lo, hi) window.  Compare
    how much the signature lane closes vs the scalar baseline as the
    warm-up window moves later in the path.
    """
    factories = {scalar_name: scalar_factory, signature_name: signature_factory}
    per_seed_hat: Dict[str, List[np.ndarray]] = {n: [] for n in factories}
    per_seed_true: List[np.ndarray] = []
    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_seeds):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        noise = _paired_noise(cfg.rho, T, base_seed + 1 + k)
        lanes = _fresh_lanes(factories, V0)
        V_true_pre, _, V_hat_post, _ = _filter_rollout(
            cfg, env, lanes, T, noise, V0,
        )
        for name in factories:
            per_seed_hat[name].append(V_hat_post[name])
        per_seed_true.append(V_true_pre)
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for (lo, hi) in warm_windows:
        key = f"t=[{lo},{hi})"
        out[key] = {}
        true_pool = np.concatenate([a[lo:hi] for a in per_seed_true])
        for name in factories:
            pred_pool = np.concatenate([a[lo:hi] for a in per_seed_hat[name]])
            mask = np.isfinite(pred_pool) & np.isfinite(true_pool)
            if mask.sum() < 3:
                out[key][name] = (float("nan"), float("nan"))
                continue
            rmse = float(np.sqrt(np.mean((pred_pool[mask] - true_pool[mask]) ** 2)))
            corr = float(np.corrcoef(pred_pool[mask], true_pool[mask])[0, 1])
            out[key][name] = (rmse, corr)
    return out


def _print_warm_start_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    scalar_name: str, signature_name: str,
) -> None:
    print(f"Warm-start ablation (corr / RMSE vs spot V within each within-path window)")
    print("  CAVEAT: within-path stats; NOT directly comparable to cross-path cold-reset.")
    print("-" * 90)
    windows = list(results.keys())
    header = f"{'window':>16s}  |  {scalar_name[:18]:>18s}   {signature_name[:22]:>22s}"
    print(header)
    print("-" * 90)
    for w in windows:
        rmse_s, corr_s = results[w][scalar_name]
        rmse_g, corr_g = results[w][signature_name]
        cs = f"{corr_s:+.3f}/{rmse_s:.3f}" if np.isfinite(corr_s) else "nan"
        cg = f"{corr_g:+.3f}/{rmse_g:.3f}" if np.isfinite(corr_g) else "nan"
        print(f"{w:>16s}  |  {cs:>18s}   {cg:>22s}")
    print()


# ==========================================================================
# Figure
# ==========================================================================


def _plot_representative(
    cfg: VGConfig,
    env: HestonMertonEnv,
    factories: Dict[str, Callable[[], object]],
    out_path: str,
    T: int,
    seed: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg2 = replace(cfg, T_steps=T)
    noise = _paired_noise(cfg.rho, T, seed)
    V0 = float(np.random.RandomState(seed).uniform(cfg.V0_low, cfg.V0_high))
    lanes = _fresh_lanes(factories, V0)
    V_true_pre, _, V_hat_post, _ = _filter_rollout(
        cfg2, env, lanes, T, noise, V0,
    )
    t_axis = np.arange(T) / BARS_PER_DAY   # x-axis in trading days
    palette = {
        "oracle":         "tab:green",
        "ewma_5min_1d":   "tab:orange",
        "ewma_5min_5d":   "tab:olive",
        "blr_kf_leadlag": "tab:red",
        "ms_cum_stride":  "tab:cyan",
        "ms_forget_spec": "tab:brown",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    ax.plot(t_axis, V_true_pre, "k-", lw=1.4, label="V_true (spot)")
    for name in ("ewma_5min_1d", "ewma_5min_5d", "blr_kf_leadlag"):
        if name in V_hat_post:
            ax.plot(t_axis, V_hat_post[name], color=palette.get(name),
                    lw=1.0, alpha=0.85, label=name)
    ax.set_ylabel("V")
    ax.set_title("Scalar + single-scale signature lanes")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t_axis, V_true_pre, "k-", lw=1.4, label="V_true (spot)")
    for name in ("ms_cum_stride", "ms_forget_spec"):
        if name in V_hat_post:
            ax.plot(t_axis, V_hat_post[name], color=palette.get(name),
                    lw=1.2, alpha=0.9, label=name)
    ax.set_xlabel("trading days")
    ax.set_ylabel("V")
    ax.set_title("Multiresolution signature lanes")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.suptitle(
        f"Heston 5-min filter study  (dt=1/{BARS_PER_YEAR}, seed={seed}, "
        f"{T/BARS_PER_DAY:.1f} trading days)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    # Base daily config; we mutate dt, T_steps, and ewma_halflife in-place.
    cfg_base = VGConfig()
    cfg = replace(
        cfg_base,
        dt=DT_5MIN,
        T_steps=60 * BARS_PER_DAY,   # ~3 trading months
        ewma_halflife_days=float(BARS_PER_DAY),  # 1 day in bars (halflife-in-steps)
    )
    env = HestonMertonEnv(rho=cfg.rho, gamma=cfg.gamma)

    print("=" * 118)
    print("BENCHMARK 2 of validation ladder:  Heston at 5-minute bars  (filter-only, u=0)")
    print(f"  config: rho={cfg.rho}  gamma={cfg.gamma}  dt=1/{BARS_PER_YEAR}={cfg.dt:.6f}")
    print(f"  CIR   : kappa={env.kappa}  theta={env.theta}  xi={env.xi}"
          f"   1/kappa={1.0/env.kappa:.2f} yr = {1.0/(env.kappa*cfg.dt):.0f} bars"
          f" = {1.0/(env.kappa*cfg.dt*BARS_PER_DAY):.1f} days")
    print(f"  V0 uniform [{cfg.V0_low}, {cfg.V0_high}]")
    print("  Bars per day: 78  (5-min bars, 6.5-hour trading day)")
    print("=" * 118)

    # Pilot tau (spectral ladder)
    print()
    print("Pilot tau estimation (EWMA(r^2/dt) ACF 1/e lag, median over pilot seeds)")
    tau_spectral = estimate_pilot_tau_years_5min(
        cfg, env, n_pilot=8, T_pilot_days=30, ewma_halflife_bars=BARS_PER_DAY,
        base_seed=7_500,
    )
    tau_days = tau_spectral / cfg.dt / BARS_PER_DAY
    tau_bars = tau_spectral / cfg.dt
    print(f"  pilot tau_hat = {tau_spectral:.5f} yr = {tau_bars:.0f} bars = {tau_days:.2f} days")
    ladder = ladder_from_timescale(tau_spectral, n_scales=3, fan=4.0)
    ladder_days = [t / cfg.dt / BARS_PER_DAY for t in ladder]
    ladder_bars = [t / cfg.dt for t in ladder]
    print(f"  spectral ladder (fan=4, 3 scales): "
          f"{['%.1fd' % d for d in ladder_days]}  "
          f"(= {['%.0f bars' % b for b in ladder_bars]})")

    factories = make_lane_factories_5min(cfg, env, tau_spectral)
    cum_stride_days = [BARS_PER_DAY / 6.0 / BARS_PER_DAY,
                       1.0, 5.0]
    print(f"  cum-stride calendar scales: "
          f"[{cum_stride_days[0]*24:.1f} hr, {cum_stride_days[1]:.0f} d, {cum_stride_days[2]:.0f} d]"
          f"  = [~13, 78, 390 bars]")

    # Primary study
    print()
    n_seeds = 8
    T_days = 60
    T = T_days * BARS_PER_DAY       # ~4680 bars/seed
    h = BARS_PER_DAY                # forward-averaged V over 1 trading day
    warmup_days = 5
    warmup = warmup_days * BARS_PER_DAY
    print(f"STUDY  --  n_seeds={n_seeds}, T={T} bars ({T_days} trading days), "
          f"h={h} bars (1 day), warm-up={warmup} bars ({warmup_days} days)")
    res = study_5min_comparison(
        cfg, env, factories,
        n_seeds=n_seeds, T=T, h=h, warmup=warmup,
        base_seed=11_000,
    )
    _print_table(res, h=h, title="Study -- corr and RMSE vs spot V / one-step / forward-avg V")

    # Figure
    plot_path = os.path.join(HERE, "study_heston_5min_signature_filters.png")
    try:
        _plot_representative(
            cfg, env, factories, out_path=plot_path,
            T=10 * BARS_PER_DAY, seed=42,
        )
        print(f"Saved figure: {plot_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")

    # Direct answers
    def corr_of(lane: str) -> float:
        return float(res.get(lane, {}).get("spot_V", (float("nan"), float("nan"), 0))[1])

    ewma_best = max(["ewma_5min_1d", "ewma_5min_5d"], key=lambda L: corr_of(L))
    sig_lanes = ["blr_kf_leadlag", "ms_cum_stride", "ms_forget_spec"]
    sig_best = max(sig_lanes, key=lambda L: corr_of(L))

    print("=" * 118)
    print("DIRECT ANSWERS  (at this 5-min Heston config; NOT a universal ranking)")
    print("=" * 118)
    print(f"  Best scalar baseline : {ewma_best} (corr spot V = {corr_of(ewma_best):+.4f})")
    print(f"  Best signature lane  : {sig_best} (corr spot V = {corr_of(sig_best):+.4f})")
    gap = corr_of(sig_best) - corr_of(ewma_best)
    print(f"  Δ (signature - scalar): {gap:+.4f}")
    if gap > 0.03:
        verdict = "CLEAR GATE FLIP: signature beats best scalar EWMA by > +0.03 corr."
    elif gap > -0.02:
        verdict = "Within +-0.02 corr: signature competitive with EWMA at this config."
    else:
        verdict = "Gate holds: EWMA still wins comfortably; easy-case extends to intraday Heston."
    print(f"  Verdict               : {verdict}")

    # Benchmark 3: warm-start ablation
    print()
    print("=" * 118)
    print("BENCHMARK 3  --  warm-start ablation on the strongest pair  (within-path stats)")
    print("=" * 118)
    T_warm_days = 60
    T_warm = T_warm_days * BARS_PER_DAY
    warm_windows = (
        (1 * BARS_PER_DAY,  5 * BARS_PER_DAY),    # early
        (10 * BARS_PER_DAY, 30 * BARS_PER_DAY),   # mid
        (40 * BARS_PER_DAY, 60 * BARS_PER_DAY),   # late
    )
    warm_res = warm_start_ablation(
        cfg, env,
        scalar_factory=factories[ewma_best],
        signature_factory=factories[sig_best],
        scalar_name=ewma_best,
        signature_name=sig_best,
        T=T_warm,
        warm_windows=warm_windows,
        n_seeds=4,
        base_seed=13_000,
    )
    _print_warm_start_table(warm_res, scalar_name=ewma_best, signature_name=sig_best)
    print("  If signature lane's corr improves markedly from early to late windows while the")
    print("  scalar lane stays roughly flat, the residual gap is mostly cold-start rather than")
    print("  architectural.  If both stay flat at their respective levels, the gap is intrinsic.")


if __name__ == "__main__":
    main()
