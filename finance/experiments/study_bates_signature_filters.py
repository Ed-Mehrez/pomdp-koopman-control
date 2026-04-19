r"""
BENCHMARK STUDY: Bates / jump-volatility filter comparison.

Row 3 of the validation ladder (see
`docs/benchmark_ladder_gated_compression.md`).  Bates = Heston
dynamics + compound-Poisson jumps in log-price:

    dS_t / S_t    =  mu dt  +  sqrt(V_t) dW_1  +  (e^J - 1) dN_t
    dV_t          =  kappa (theta - V_t) dt  +  xi sqrt(V_t) dW_2
    dN_t          ~  Poisson(lambda_j dt)
    J             ~  N(mu_j, sigma_j^2)  i.i.d., independent of dW_1, dW_2, dN
    d<W_1, W_2>_t =  rho dt

Hypothesis: scalar r^2/dt estimators of V are contaminated by jump atoms;
jump-robust handcrafted summaries (bipower variation, winsorized EWMA)
and the model-free multiresolution signature lane should both recover.
The key thesis question: does the gate prefer
  (a) a robust scalar baseline, or
  (b) the signature lane?

This is a minimal Bates setup that deliberately does NOT use the Merton
wealth wrapper.  Filters are tested directly on simulated returns.
Controller is off by construction.

Pre-registered interpretation
-----------------------------
  - If rv_ewma deteriorates vs the Heston case and ms_cum_stride
    recovers the gap, that is a slide-worthy gate flip toward signatures.
  - If bv_ewma (bipower variation) or a winsorized EWMA wins, report
    honestly: the gate prefers a better HANDCRAFTED summary under jumps.
  - The thesis statement is not "signatures must win"; it is "plain
    scalar r^2/dt smoothers are not universally adequate once the
    observation law gets harder."
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

from src.sskf.leadlag_blr_kf import (
    LeadLagBLRKFConfig,
    LeadLagBLRKFilter,
)
from src.sskf.multiscale_leadlag_filters import (
    CumulativeStrideLeadLagBLRKFConfig,
    CumulativeStrideLeadLagBLRKFilter,
    MultiScaleLeadLagBLRKFConfig,
    MultiScaleLeadLagBLRKFilter,
    fixed_calendar_ladder,
    ladder_from_timescale,
    estimate_variance_timescale,
)


# ==========================================================================
# Simulator
# ==========================================================================


@dataclass
class BatesConfig:
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    mu: float = 0.05
    lambda_j: float = 30.0      # expected jumps per year
    mu_j: float = -0.03         # mean jump size in log-price
    sigma_j: float = 0.03       # jump-size stdev
    dt: float = 1.0 / 252.0


def simulate_bates(
    cfg: BatesConfig, T: int, V0: float, seed: int,
) -> Dict[str, np.ndarray]:
    r"""Euler + explicit-full-truncation CIR + compound Poisson jumps.

    Returns dict with keys:
        V       : (T,) pre-step variance trajectory
        dr_S    : (T,) underlying log-returns including jumps
        J       : (T,) jump atoms (0.0 where no jump)
        jump    : (T,) 0/1 indicator
    """
    rng = np.random.RandomState(int(seed))
    sqrt_dt = float(np.sqrt(cfg.dt))
    V = np.zeros(T)
    dr_S = np.zeros(T)
    J = np.zeros(T)
    jump_flag = np.zeros(T, dtype=int)
    V_prev = float(V0)
    # probability of at least one jump in an interval of length dt;
    # for small lambda*dt this is ~ lambda*dt.
    p_jump = 1.0 - float(np.exp(-cfg.lambda_j * cfg.dt))
    for t in range(T):
        V[t] = V_prev
        z1 = rng.standard_normal()
        z2 = cfg.rho * z1 + float(np.sqrt(max(1.0 - cfg.rho ** 2, 0.0))) * rng.standard_normal()
        v_use = max(V_prev, 1e-8)
        diffusion = (cfg.mu - 0.5 * v_use) * cfg.dt + float(np.sqrt(v_use)) * sqrt_dt * z1
        if rng.rand() < p_jump:
            J_t = float(rng.normal(cfg.mu_j, cfg.sigma_j))
            J[t] = J_t
            jump_flag[t] = 1
            dr_S[t] = diffusion + J_t
        else:
            dr_S[t] = diffusion
        # CIR variance update (independent diffusion branch, same z2 coupling)
        V_new = V_prev + cfg.kappa * (cfg.theta - V_prev) * cfg.dt + cfg.xi * float(np.sqrt(v_use)) * sqrt_dt * z2
        V_prev = max(V_new, 1e-8)
    return {"V": V, "dr_S": dr_S, "J": J, "jump": jump_flag}


# ==========================================================================
# Scalar baselines: EWMA of r^2/dt, Bipower EWMA, Winsorized EWMA
# ==========================================================================


class EWMAr2:
    name = "rv_ewma"

    def __init__(self, halflife_steps: float, dt: float):
        self.alpha = 1.0 - float(np.exp(-np.log(2.0) / max(halflife_steps, 1e-3)))
        self.dt = float(dt)
        self.V = 0.04

    def reset(self, V0: float):
        self.V = float(V0)

    def observe(self, r_t: float, dt: float):
        y = float(r_t) ** 2 / float(dt)
        self.V = (1.0 - self.alpha) * self.V + self.alpha * y
        self.V = max(self.V, 1e-8)

    def V_hat(self) -> float:
        return float(self.V)

    def V_interval(self) -> Tuple[float, float]:
        return (float("nan"), float("nan"))


class EWMABipower:
    r"""Bipower variation as a jump-robust variance proxy.

    BV_t = (pi/2) * |r_{t-1}| * |r_t| / dt  is an (approximately)
    unbiased V estimator that REJECTS finite-activity jumps
    asymptotically (Barndorff-Nielsen & Shephard, 2004).  We use a
    standard EWMA on the per-step BV to form a running V estimate.
    """
    name = "bv_ewma"

    def __init__(self, halflife_steps: float, dt: float):
        self.alpha = 1.0 - float(np.exp(-np.log(2.0) / max(halflife_steps, 1e-3)))
        self.dt = float(dt)
        self.V = 0.04
        self._prev_abs_r = None

    def reset(self, V0: float):
        self.V = float(V0)
        self._prev_abs_r = None

    def observe(self, r_t: float, dt: float):
        abs_r = float(np.abs(r_t))
        if self._prev_abs_r is not None:
            # Bipower per-step estimator of V (scaled by pi/2, divided by dt).
            bv = (np.pi / 2.0) * self._prev_abs_r * abs_r / float(dt)
            self.V = (1.0 - self.alpha) * self.V + self.alpha * bv
            self.V = max(self.V, 1e-8)
        self._prev_abs_r = abs_r

    def V_hat(self) -> float:
        return float(self.V)

    def V_interval(self) -> Tuple[float, float]:
        return (float("nan"), float("nan"))


class EWMAWinsorized:
    r"""Winsorized EWMA: clip r^2/dt at k * V_current BEFORE updating.

    k=4 truncates a return whose scaled square exceeds 4x the current
    V estimate.  This is a cheap ad hoc jump filter.  Clearly labeled
    heuristic, not a principled robust estimator.
    """
    name = "winsor_ewma"

    def __init__(self, halflife_steps: float, dt: float, k: float = 4.0):
        self.alpha = 1.0 - float(np.exp(-np.log(2.0) / max(halflife_steps, 1e-3)))
        self.dt = float(dt)
        self.k = float(k)
        self.V = 0.04

    def reset(self, V0: float):
        self.V = float(V0)

    def observe(self, r_t: float, dt: float):
        y = float(r_t) ** 2 / float(dt)
        cap = self.k * max(self.V, 1e-8)
        y_clipped = min(y, cap)
        self.V = (1.0 - self.alpha) * self.V + self.alpha * y_clipped
        self.V = max(self.V, 1e-8)

    def V_hat(self) -> float:
        return float(self.V)

    def V_interval(self) -> Tuple[float, float]:
        return (float("nan"), float("nan"))


# ==========================================================================
# Signature lanes (direct use of the filter classes; no Merton wrapper)
# ==========================================================================


class BLRKFLeadLagLane:
    name = "blr_kf_leadlag"

    def __init__(self, bates: BatesConfig):
        cfg = LeadLagBLRKFConfig(
            ll_gamma=0.99,
            prior_w_var=10.0, sigma_n2_init=0.01, sigma_n2_alpha=0.01,
            target_clip=None,                      # disable to not bias away jumps
            kf_kappa=bates.kappa, kf_theta=bates.theta, kf_xi=bates.xi,
            V_floor=1e-6, P_init_mult=10.0,
        )
        self.filter = LeadLagBLRKFilter(dt=bates.dt, config=cfg)

    def reset(self, V0: float):
        self.filter.reset(V0)

    def observe(self, r_t: float, dt: float):
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


class MSCumStrideLane:
    name = "ms_cum_stride"

    def __init__(self, bates: BatesConfig, days_ladder=(1.0, 5.0, 20.0)):
        strides = tuple(max(1, int(round(d))) for d in days_ladder)
        cfg = CumulativeStrideLeadLagBLRKFConfig(
            strides=strides,
            kf_kappa=bates.kappa, kf_theta=bates.theta, kf_xi=bates.xi,
            V_floor=1e-6, P_init_mult=10.0,
            prior_w_var=10.0, sigma_n2_init=0.01, sigma_n2_alpha=0.01,
            target_clip=None,
        )
        self.filter = CumulativeStrideLeadLagBLRKFilter(dt=bates.dt, config=cfg)

    def reset(self, V0: float):
        self.filter.reset(V0)

    def observe(self, r_t: float, dt: float):
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


class MSForgetSpectralLane:
    name = "ms_forget_spec"

    def __init__(self, bates: BatesConfig, taus_years: List[float]):
        from src.sskf.multiscale_leadlag_filters import gammas_from_taus
        gammas = tuple(gammas_from_taus(taus_years, bates.dt))
        cfg = MultiScaleLeadLagBLRKFConfig(
            gammas=gammas,
            kf_kappa=bates.kappa, kf_theta=bates.theta, kf_xi=bates.xi,
            V_floor=1e-6, P_init_mult=10.0,
            prior_w_var=10.0, sigma_n2_init=0.01, sigma_n2_alpha=0.01,
            target_clip=None,
        )
        self.filter = MultiScaleLeadLagBLRKFilter(dt=bates.dt, config=cfg)

    def reset(self, V0: float):
        self.filter.reset(V0)

    def observe(self, r_t: float, dt: float):
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


# ==========================================================================
# Rollout + pooled statistics
# ==========================================================================


def _fresh_lanes(
    factories: Dict[str, Callable[[], object]], V0: float,
) -> Dict[str, object]:
    lanes = {name: factory() for name, factory in factories.items()}
    for est in lanes.values():
        est.reset(V0)
    return lanes


def _filter_rollout_bates(
    cfg: BatesConfig,
    factories: Dict[str, Callable[[], object]],
    T: int, V0: float, seed: int,
) -> Dict[str, np.ndarray]:
    sim = simulate_bates(cfg, T, V0, seed)
    lanes = _fresh_lanes(factories, V0)
    V_hat_post = {name: np.zeros(T) for name in lanes}
    for t in range(T):
        r = float(sim["dr_S"][t])
        for name, est in lanes.items():
            est.observe(r, cfg.dt)
            V_hat_post[name][t] = est.V_hat()
    return {
        "V_true_pre": sim["V"],
        "dr_S": sim["dr_S"],
        "jump": sim["jump"],
        "V_hat_post": V_hat_post,
    }


def _pooled_stats(
    per_seed_pred: List[np.ndarray],
    per_seed_target: List[np.ndarray],
    mask_per_seed: Optional[List[np.ndarray]] = None,
    warmup: int = 20,
) -> Tuple[float, float, int]:
    pred_list: List[np.ndarray] = []
    tgt_list: List[np.ndarray] = []
    for k in range(len(per_seed_pred)):
        p = per_seed_pred[k][warmup:]
        t = per_seed_target[k][warmup:]
        if mask_per_seed is not None:
            m = mask_per_seed[k][warmup:]
            p = p[m]
            t = t[m]
        pred_list.append(p)
        tgt_list.append(t)
    pred = np.concatenate(pred_list) if pred_list else np.zeros(0)
    tgt = np.concatenate(tgt_list) if tgt_list else np.zeros(0)
    mask = np.isfinite(pred) & np.isfinite(tgt)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    rmse = float(np.sqrt(np.mean((pred[mask] - tgt[mask]) ** 2)))
    corr = float(np.corrcoef(pred[mask], tgt[mask])[0, 1])
    return rmse, corr, int(mask.sum())


def _jump_adjacent_mask(jump_flag: np.ndarray, radius: int) -> np.ndarray:
    r"""Boolean array flagging steps within +/- radius of any jump."""
    T = jump_flag.size
    mask = np.zeros(T, dtype=bool)
    jump_idx = np.where(jump_flag > 0)[0]
    for j in jump_idx:
        lo = max(0, j - radius)
        hi = min(T, j + radius + 1)
        mask[lo:hi] = True
    return mask


# ==========================================================================
# Study driver
# ==========================================================================


def make_factories(
    bates: BatesConfig, tau_spectral_years: float,
) -> Dict[str, Callable[[], object]]:
    r"""Scalar baselines at halflife = 21 steps (daily-cadence convention);
    signature lanes use the same (1, 5, 20)-day calendar ladder as the
    daily study.
    """
    halflife = 21.0
    return {
        "rv_ewma":       lambda: EWMAr2(halflife_steps=halflife, dt=bates.dt),
        "bv_ewma":       lambda: EWMABipower(halflife_steps=halflife, dt=bates.dt),
        "winsor_ewma":   lambda: EWMAWinsorized(halflife_steps=halflife, dt=bates.dt, k=4.0),
        "blr_kf_leadlag": lambda: BLRKFLeadLagLane(bates),
        "ms_cum_stride": lambda: MSCumStrideLane(bates, days_ladder=(1.0, 5.0, 20.0)),
        "ms_forget_spec": lambda: MSForgetSpectralLane(
            bates,
            taus_years=ladder_from_timescale(tau_spectral_years, n_scales=3, fan=4.0),
        ),
    }


def estimate_pilot_tau(
    bates: BatesConfig, n_pilot: int, T_pilot: int, base_seed: int,
    ewma_halflife_steps: float = 21.0,
) -> float:
    alpha = 1.0 - float(np.exp(-np.log(2.0) / max(ewma_halflife_steps, 1e-3)))
    v0_rng = np.random.RandomState(base_seed)
    taus: List[float] = []
    for k in range(n_pilot):
        V0 = float(v0_rng.uniform(0.02, 0.08))
        sim = simulate_bates(bates, T_pilot, V0, base_seed + 1 + k)
        rv = np.zeros(T_pilot)
        smooth = V0
        for t in range(T_pilot):
            y = sim["dr_S"][t] ** 2 / bates.dt
            smooth = (1.0 - alpha) * smooth + alpha * y
            rv[t] = smooth
        tau_k = estimate_variance_timescale(
            rv, dt=bates.dt, max_lag_fraction=0.2,
            min_tau_days=2.0, max_tau_days=200.0,
        )
        taus.append(tau_k)
    return float(np.median(taus))


def run_study(
    bates: BatesConfig, n_seeds: int, T: int, warmup: int, h: int,
    base_seed: int, tau_spectral_years: float,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float, int]]], Dict]:
    factories = make_factories(bates, tau_spectral_years)
    lane_names = list(factories.keys())
    per_seed_hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
    per_seed_V: List[np.ndarray] = []
    per_seed_fwd: List[np.ndarray] = []
    per_seed_jump: List[np.ndarray] = []
    per_seed_dr: List[np.ndarray] = []
    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_seeds):
        V0 = float(v0_rng.uniform(0.02, 0.08))
        out = _filter_rollout_bates(bates, factories, T, V0, base_seed + 1 + k)
        V_true_pre = out["V_true_pre"]
        jump_flag = out["jump"]
        V_hat_post = out["V_hat_post"]
        # Forward-averaged latent V over h (shift by 1 so post-update
        # V_hat aligns with post-step V_{t+1}..V_{t+h} window mean).
        # Implementation: build fwd(V_true_pre) by shifting V_true_pre by
        # 1 and computing the h-step rolling mean.
        T_ = V_true_pre.size
        fwd = np.full(T_, np.nan)
        if T_ > h:
            post = np.concatenate([V_true_pre[1:], V_true_pre[-1:]])  # V_{t+1} proxy at index t
            cum = np.concatenate([[0.0], np.cumsum(post)])
            for t in range(T_ - h):
                fwd[t] = (cum[t + h] - cum[t]) / h
        for name in lane_names:
            per_seed_hat[name].append(V_hat_post[name])
        per_seed_V.append(V_true_pre)
        per_seed_fwd.append(fwd)
        per_seed_jump.append(jump_flag)
        per_seed_dr.append(out["dr_S"])

    # Full-sample, jump-adjacent, calm stats
    radius = 5
    results: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    jump_masks = [_jump_adjacent_mask(j, radius) for j in per_seed_jump]
    calm_masks = [~m for m in jump_masks]
    for name in lane_names:
        stats: Dict[str, Tuple[float, float, int]] = {}
        stats["spot_V_all"]   = _pooled_stats(
            per_seed_hat[name], per_seed_V, warmup=warmup)
        stats["spot_V_jump"]  = _pooled_stats(
            per_seed_hat[name], per_seed_V, mask_per_seed=jump_masks, warmup=warmup)
        stats["spot_V_calm"]  = _pooled_stats(
            per_seed_hat[name], per_seed_V, mask_per_seed=calm_masks, warmup=warmup)
        stats[f"fwd_V_h{h}"]  = _pooled_stats(
            per_seed_hat[name], per_seed_fwd, warmup=warmup)
        results[name] = stats

    # Also return a random representative trajectory (first seed)
    trace = {
        "V_true_pre": per_seed_V[0],
        "dr_S":       per_seed_dr[0],
        "jump":       per_seed_jump[0],
        "V_hat_post": {name: per_seed_hat[name][0] for name in lane_names},
    }
    return results, trace


# ==========================================================================
# Reporting
# ==========================================================================


def _print_bates_table(
    results: Dict[str, Dict[str, Tuple[float, float, int]]], h: int,
) -> None:
    col_keys = ["spot_V_all", "spot_V_jump", "spot_V_calm", f"fwd_V_h{h}"]
    labels = {
        "spot_V_all":   "spot V (all)",
        "spot_V_jump":  "spot V (jump)",
        "spot_V_calm":  "spot V (calm)",
        f"fwd_V_h{h}":  f"fwd V (h={h})",
    }
    print("Bates study  --  pooled corr and RMSE of V_hat_post vs spot V (subset masks)")
    print("-" * 128)
    print(f"{'lane':20s} | {'stat':>6s} |", end="")
    for k in col_keys:
        print(f"  {labels[k]:>20s}", end="")
    print()
    print("-" * 128)
    for lane, stats in results.items():
        print(f"{lane:20s} | {'corr':>6s} |", end="")
        for k in col_keys:
            rmse, corr, n = stats.get(k, (float("nan"), float("nan"), 0))
            cell = f"{corr:+.4f}" if np.isfinite(corr) else "    nan"
            print(f"  {cell:>20s}", end="")
        print()
    print("-" * 128)
    for lane, stats in results.items():
        print(f"{lane:20s} | {'rmse':>6s} |", end="")
        for k in col_keys:
            rmse, corr, n = stats.get(k, (float("nan"), float("nan"), 0))
            cell = f"{rmse:.4f}" if np.isfinite(rmse) else "   nan"
            print(f"  {cell:>20s}", end="")
        print()
    print()


def _plot_representative_bates(
    trace: Dict, out_path: str, dt: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V_true = trace["V_true_pre"]
    V_hats = trace["V_hat_post"]
    jump = trace["jump"]
    T = V_true.size
    t_axis = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    ax.plot(t_axis, V_true, "k-", lw=1.4, label="V_true (spot)")
    jump_idx = np.where(jump > 0)[0]
    for j in jump_idx:
        ax.axvline(j, color="gray", alpha=0.3, lw=0.7)
    for name in ("rv_ewma", "bv_ewma", "winsor_ewma"):
        if name in V_hats:
            ax.plot(t_axis, V_hats[name], lw=1.0, alpha=0.85, label=name)
    ax.set_ylabel("V")
    ax.set_title("Scalar baselines (vertical lines = jump events)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t_axis, V_true, "k-", lw=1.4, label="V_true (spot)")
    for j in jump_idx:
        ax.axvline(j, color="gray", alpha=0.3, lw=0.7)
    for name in ("blr_kf_leadlag", "ms_cum_stride", "ms_forget_spec"):
        if name in V_hats:
            ax.plot(t_axis, V_hats[name], lw=1.2, alpha=0.9, label=name)
    ax.set_xlabel("step t")
    ax.set_ylabel("V")
    ax.set_title("Signature lanes")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.suptitle(
        f"Bates filter study (dt={dt:.5f}, lambda_j*1yr = expected jumps/year)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    bates = BatesConfig()
    print("=" * 128)
    print("BENCHMARK 3 of validation ladder:  Bates / jump-volatility filter study (filter-only)")
    print(f"  Heston block: kappa={bates.kappa}, theta={bates.theta}, xi={bates.xi}, rho={bates.rho}")
    print(f"  Jump block  : lambda={bates.lambda_j}/yr, mu_j={bates.mu_j}, sigma_j={bates.sigma_j}")
    print(f"  dt={bates.dt:.5f} (daily cadence)")
    print(f"  Expected jumps per 200-step episode: {bates.lambda_j * bates.dt * 200:.2f}")
    print("=" * 128)

    # Pilot tau
    print()
    print("Pilot tau estimation (EWMA r^2/dt ACF 1/e lag; median over pilot seeds)")
    tau_spectral = estimate_pilot_tau(
        bates, n_pilot=20, T_pilot=500, base_seed=6_000,
    )
    tau_days = tau_spectral / bates.dt
    print(f"  pilot tau_hat = {tau_spectral:.5f} yr = {tau_days:.2f} days")

    # Primary study
    n_seeds = 40
    T = 400
    warmup = 30
    h = 10
    print()
    print(f"STUDY  --  n_seeds={n_seeds}, T={T}, warm-up={warmup}, h={h}")
    results, trace = run_study(
        bates, n_seeds=n_seeds, T=T, warmup=warmup, h=h,
        base_seed=14_000, tau_spectral_years=tau_spectral,
    )
    _print_bates_table(results, h=h)

    # Figure
    plot_path = os.path.join(HERE, "study_bates_signature_filters.png")
    try:
        _plot_representative_bates(trace, out_path=plot_path, dt=bates.dt)
        print(f"Saved figure: {plot_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")

    # Direct answers
    def corr_of(lane: str, key: str) -> float:
        entry = results.get(lane, {}).get(key, None)
        return float(entry[1]) if entry is not None else float("nan")

    scalar_lanes = ["rv_ewma", "bv_ewma", "winsor_ewma"]
    sig_lanes = ["blr_kf_leadlag", "ms_cum_stride", "ms_forget_spec"]
    best_rv = "rv_ewma"
    best_scalar = max(scalar_lanes, key=lambda L: corr_of(L, "spot_V_all"))
    best_sig = max(sig_lanes, key=lambda L: corr_of(L, "spot_V_all"))

    print("=" * 128)
    print("DIRECT ANSWERS  (Bates jump-vol, at this config; NOT a universal ranking)")
    print("=" * 128)
    print(f"  rv_ewma (plain r^2/dt)  : corr spot V (all) = {corr_of(best_rv, 'spot_V_all'):+.4f}")
    print(f"  Best scalar             : {best_scalar}   corr spot V (all) = "
          f"{corr_of(best_scalar, 'spot_V_all'):+.4f}")
    print(f"  Best signature          : {best_sig}  corr spot V (all) = "
          f"{corr_of(best_sig, 'spot_V_all'):+.4f}")
    print()
    print("  Jump-adjacent subset (|t - t_jump| <= 5 bars):")
    print(f"    rv_ewma               : corr = {corr_of(best_rv, 'spot_V_jump'):+.4f}")
    print(f"    best robust scalar    : {best_scalar}   corr = {corr_of(best_scalar, 'spot_V_jump'):+.4f}")
    print(f"    best signature        : {best_sig}  corr = {corr_of(best_sig, 'spot_V_jump'):+.4f}")
    print()

    # Interpretation
    gap_rv_vs_robust = corr_of(best_scalar, "spot_V_all") - corr_of(best_rv, "spot_V_all")
    gap_rv_vs_sig = corr_of(best_sig, "spot_V_all") - corr_of(best_rv, "spot_V_all")
    gap_jump_rv_vs_robust = (corr_of(best_scalar, "spot_V_jump")
                             - corr_of(best_rv, "spot_V_jump"))
    gap_jump_rv_vs_sig = (corr_of(best_sig, "spot_V_jump")
                          - corr_of(best_rv, "spot_V_jump"))
    print("  Gaps (all):")
    print(f"    best robust - rv_ewma    = {gap_rv_vs_robust:+.4f}")
    print(f"    best signature - rv_ewma = {gap_rv_vs_sig:+.4f}")
    print("  Gaps (jump-adjacent only):")
    print(f"    best robust - rv_ewma    = {gap_jump_rv_vs_robust:+.4f}")
    print(f"    best signature - rv_ewma = {gap_jump_rv_vs_sig:+.4f}")

    if gap_jump_rv_vs_sig > 0.10 and gap_rv_vs_sig > 0.04:
        verdict = ("SIGNATURE GATE FLIP: signature lane beats rv_ewma on both full sample "
                   "(>+0.04) and jump-adjacent subset (>+0.10).")
    elif gap_jump_rv_vs_robust > 0.10 and gap_rv_vs_robust > 0.04:
        verdict = ("ROBUST-SCALAR GATE FLIP: handcrafted jump-robust scalar beats rv_ewma; "
                   "gate prefers a better handcrafted summary over signatures under jumps.")
    else:
        verdict = ("No clear flip at this config: the signal is small or mixed; see "
                   "table for nuance.")
    print(f"  Verdict                  : {verdict}")
    print("=" * 128)


if __name__ == "__main__":
    main()
