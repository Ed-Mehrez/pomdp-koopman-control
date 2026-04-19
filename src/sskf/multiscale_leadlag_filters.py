r"""
Multiresolution lead-lag signature V-filters.

Two main filter classes, both lane-API compatible (reset / observe /
V_hat / V_interval), and both generalizations of the single-scale
BLR+KF in `src/sskf/leadlag_blr_kf.py`:

  1. `MultiScaleLeadLagBLRKFilter`
       K parallel lead-lag recurrent log-signature states at different
       forgetting factors gamma_k.  Per scale: a 3-feature BLR on
       [Levy QV area, ret_lead, bias] targeting r^2/dt.  Each head
       emits a predictive Gaussian (mean, variance).  The K predictions
       are combined into ONE scalar observation via precision-weighted
       (inverse-variance) fusion, then assimilated into an outer scalar
       CIR Kalman on V.

  2. `CumulativeStrideLeadLagBLRKFilter`
       ONE cumulative (gamma = 1.0) lead-lag recurrent log-signature
       state.  Recent cumulative states are stored in a ring buffer; the
       log-signature of the window [t - m, t] is recovered EXACTLY at
       level 2 via Chen's identity with a single bilinear correction.
       Per stride m_k: a 3-feature BLR on the recovered window
       [Levy QV area, ret_lead, bias], normalized by the window length
       so the Levy-area feature is a V-scale estimator at that window.
       Fusion into the outer Kalman is the same precision-combine.

Both filters preserve the Bayesian semantics at the BLR level and the
Gaussian semantics at the outer Kalman level.

Precision-weighted fusion vs sequential absorption
---------------------------------------------------
For K independent Gaussian observations y_k ~ N(V, R_k) of the same
latent V, sequential Kalman absorption gives the same posterior as the
one-shot absorption of the precision-weighted mean y_bar with noise
R_bar:
    1/R_bar = sum_k 1/R_k,    y_bar = R_bar * sum_k y_k / R_k.

We use precision-weighted fusion.  Strict independence across scales
is an approximation: the K observations share the same underlying
dr_S increments and BLR weight posteriors evolve in parallel, so
there is some cross-correlation we do not model.  Documented, not
hidden.

Scale-selection helpers
-----------------------
`estimate_variance_timescale(...)` estimates the 1/e decorrelation
lag of a realized-variance proxy (model-free, NOT from raw returns).
`ladder_from_timescale(...)` spreads a geometric ladder around that
estimate; `fixed_calendar_ladder(...)` returns a calendar-time ladder
(e.g. 1/5/20 days) that bypasses estimation.  Converters
`gammas_from_taus`, `strides_from_taus` translate scales in years
into forgetting factors or integer strides.

No universal claims about which ladder is "best"; this module only
makes the multiresolution construction available.  The accompanying
study script decides empirically.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from examples.proof_of_concept.signature_features import (
    RecurrentLeadLagLogSigMap,
)


# ==========================================================================
# Indices for lead-lag log-sig with state_dim=2 (input = [dt, r])
# ==========================================================================

# Lead channels: (time_lead, ret_lead) at indices (0, 1) of l1.
# Lag channels:  (time_lag,  ret_lag)  at indices (2, 3) of l1.
# l2 indexes antisymmetric level-2 pairs (i<j):
#   (0,1)->0, (0,2)->1, (0,3)->2, (1,2)->3, (1,3)->4, (2,3)->5.
# The Levy area between ret_lead and ret_lag (which approximates QV)
# is the (1, 3) pair = index 4 of l2 (= index 8 in [l1 | l2] concat).
_QV_IDX_IN_L2 = 4
_RET_LEAD_IDX_IN_L1 = 1


def _phi_from_l1_l2(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    r"""Semantic 3-feature [Levy_QV, ret_lead, 1.0] from (l1, l2)."""
    return np.array([
        float(l2[_QV_IDX_IN_L2]),
        float(l1[_RET_LEAD_IDX_IN_L1]),
        1.0,
    ])


# ==========================================================================
# Scale-selection helpers
# ==========================================================================


def estimate_variance_timescale(
    realized_var: np.ndarray,
    dt: float,
    max_lag_fraction: float = 0.2,
    min_tau_days: float = 2.0,
    max_tau_days: float = 120.0,
) -> float:
    r"""1/e decorrelation timescale of a realized-variance proxy.

    Model-free: do NOT use raw returns (r^2/dt is a chi-squared-noise
    estimator of V and its ACF at lag 1 is NOT V's decorrelation).
    Use a SMOOTHED RV proxy (e.g. EWMA of r^2/dt) and look at its ACF.

    Returns
    -------
    tau_years : float
        First lag at which the (centered) autocorrelation drops below
        1/e, multiplied by dt.  Clipped to [min_tau_days * dt,
        max_tau_days * dt].  Defaults to 30 days * dt when the input is
        degenerate or too short.
    """
    x = np.asarray(realized_var, dtype=float).flatten()
    mask = np.isfinite(x) & (x > 0)
    x = x[mask]
    min_tau_years = float(min_tau_days) * float(dt)
    max_tau_years = float(max_tau_days) * float(dt)
    default_tau = float(30.0 * dt)
    if x.size < 50:
        return float(np.clip(default_tau, min_tau_years, max_tau_years))
    x = x - x.mean()
    N = int(x.size)
    max_lag = max(5, int(max_lag_fraction * N))
    var0 = float(np.mean(x * x))
    if var0 <= 0:
        return float(np.clip(default_tau, min_tau_years, max_tau_years))
    thresh = np.exp(-1.0)
    tau_steps = None
    prev_acf = 1.0
    for k in range(1, max_lag + 1):
        acf_k = float(np.mean(x[: N - k] * x[k:])) / var0
        if acf_k < thresh:
            # Linear interpolate between lag k-1 (prev_acf) and lag k (acf_k)
            if prev_acf > acf_k and prev_acf > thresh:
                frac = (prev_acf - thresh) / (prev_acf - acf_k)
                tau_steps = float(k - 1) + float(frac)
            else:
                tau_steps = float(k)
            break
        prev_acf = acf_k
    if tau_steps is None:
        tau_steps = float(max_lag)
    tau_years = float(tau_steps) * float(dt)
    return float(np.clip(tau_years, min_tau_years, max_tau_years))


def ladder_from_timescale(
    tau_hat_years: float,
    n_scales: int = 3,
    fan: float = 4.0,
) -> List[float]:
    r"""Geometric ladder of scales (in years) centered at tau_hat."""
    if n_scales < 1:
        return [tau_hat_years]
    if n_scales == 1:
        return [float(tau_hat_years)]
    mids = np.linspace(-1, 1, n_scales)
    return [float(tau_hat_years * (fan ** m)) for m in mids]


def fixed_calendar_ladder(
    dt: float,
    days: Sequence[float] = (1.0, 5.0, 20.0),
) -> List[float]:
    r"""Fixed calendar-time ladder in years, ignoring any data-driven
    estimate.  Matches the old successful short/medium/long pattern."""
    return [float(d * dt) for d in days]


def gammas_from_taus(taus_years: Sequence[float], dt: float) -> List[float]:
    r"""Forgetting factor per scale: gamma_k = exp(-dt / tau_k).

    tau_k = dt  -> gamma = exp(-1) ≈ 0.368  (essentially single-step memory).
    tau_k big   -> gamma -> 1            (long memory).
    """
    out: List[float] = []
    for t in taus_years:
        g = float(np.exp(-float(dt) / max(float(t), 1e-12)))
        out.append(float(np.clip(g, 1e-6, 1.0 - 1e-9)))
    return out


def strides_from_taus(
    taus_years: Sequence[float], dt: float, min_stride: int = 1,
) -> List[int]:
    r"""Integer stride per scale: m_k = max(min_stride, round(tau_k / dt))."""
    return [
        max(int(min_stride), int(round(float(t) / max(float(dt), 1e-12))))
        for t in taus_years
    ]


# ==========================================================================
# Shared components: 3-weight BLR and outer CIR Kalman
# ==========================================================================


@dataclass
class BLRHead3:
    r"""3-feature Bayesian linear regression head.

    Posterior: w ~ N(w_mean, P_cov * sigma_n^2) under the conjugate
    formulation; here we collapse sigma_n^2 into P_cov for simplicity
    and track sigma_n^2 adaptively via EWMA of squared residual.
    """
    w_mean: np.ndarray
    P_cov: np.ndarray
    sigma_n2: float
    sigma_n2_alpha: float = 0.01
    prior_w_var: float = 10.0
    sigma_n2_init: float = 0.01

    @staticmethod
    def fresh(
        prior_w_var: float = 10.0,
        sigma_n2_init: float = 0.01,
        sigma_n2_alpha: float = 0.01,
    ) -> "BLRHead3":
        return BLRHead3(
            w_mean=np.zeros(3),
            P_cov=np.eye(3) * float(prior_w_var),
            sigma_n2=float(sigma_n2_init),
            sigma_n2_alpha=float(sigma_n2_alpha),
            prior_w_var=float(prior_w_var),
            sigma_n2_init=float(sigma_n2_init),
        )

    def reset(self) -> None:
        self.w_mean = np.zeros(3)
        self.P_cov = np.eye(3) * self.prior_w_var
        self.sigma_n2 = self.sigma_n2_init

    def predict(self, phi: np.ndarray) -> Tuple[float, float]:
        y = float(phi @ self.w_mean)
        R = max(float(phi @ self.P_cov @ phi) + self.sigma_n2, 1e-10)
        return y, R

    def update(self, phi: np.ndarray, y_obs: float) -> float:
        Cp = self.P_cov @ phi
        S = float(phi @ Cp) + self.sigma_n2
        S = max(S, 1e-18)
        K = Cp / S
        resid = float(y_obs) - float(phi @ self.w_mean)
        self.w_mean = self.w_mean + K * resid
        self.P_cov = self.P_cov - np.outer(K, Cp)
        self.P_cov = 0.5 * (self.P_cov + self.P_cov.T)
        self.sigma_n2 = max(
            (1.0 - self.sigma_n2_alpha) * self.sigma_n2
            + self.sigma_n2_alpha * (resid * resid),
            1e-8,
        )
        return resid


@dataclass
class OuterCIRKalmanV:
    r"""Scalar outer Kalman on V with CIR dynamics."""
    kappa: float
    theta: float
    xi: float
    V: float
    P: float
    V_floor: float = 1e-6

    def predict(self, dt: float) -> Tuple[float, float]:
        V_pred = self.V + self.kappa * (self.theta - self.V) * dt
        V_pred = max(V_pred, self.V_floor)
        Q = self.xi ** 2 * max(self.V, self.V_floor) * dt
        P_pred = (1.0 - self.kappa * dt) ** 2 * self.P + Q
        return V_pred, P_pred

    def absorb(self, V_pred: float, P_pred: float, y: float, R: float) -> Tuple[float, float, float]:
        R_eff = max(R, 1e-18)
        S = P_pred + R_eff
        innov = float(y) - V_pred
        z = float(innov / np.sqrt(max(S, 1e-18)))
        K = P_pred / S
        V_new = max(V_pred + K * innov, self.V_floor)
        P_new = (1.0 - K) * P_pred
        return V_new, P_new, z


def _precision_combine(
    ys: Sequence[float], Rs: Sequence[float],
) -> Tuple[float, float]:
    r"""Precision-weighted fusion of K scalar Gaussian observations.

    Returns (y_bar, R_bar) such that y_bar ~ N(V, R_bar) under the
    approximation that the K observations are conditionally
    independent given V.
    """
    precisions = np.array([1.0 / max(float(r), 1e-12) for r in Rs])
    total = float(np.sum(precisions))
    if total <= 0:
        return float(np.mean(list(ys))), float("inf")
    R_bar = 1.0 / total
    y_bar = float(np.sum([p * y for p, y in zip(precisions, ys)]) / total)
    return y_bar, R_bar


# ==========================================================================
# MultiScaleLeadLagBLRKFilter (forgetting-factor ladder)
# ==========================================================================


@dataclass
class MultiScaleLeadLagBLRKFConfig:
    gammas: Sequence[float] = ()
    kf_kappa: float = 2.0
    kf_theta: float = 0.04
    kf_xi: float = 0.3
    V_floor: float = 1e-6
    P_init_mult: float = 10.0
    prior_w_var: float = 10.0
    sigma_n2_init: float = 0.01
    sigma_n2_alpha: float = 0.01
    target_clip: Optional[float] = 2.0


class MultiScaleLeadLagBLRKFilter:
    r"""K parallel lead-lag log-sig states (different gammas) -> K
    Bayesian heads -> precision-fused -> outer CIR Kalman on V.

    Lane API: reset / observe / V_hat / V_interval.
    """
    name = "ms_leadlag_blrkf"

    def __init__(self, dt: float, config: MultiScaleLeadLagBLRKFConfig):
        if len(config.gammas) < 1:
            raise ValueError("gammas must be non-empty")
        self.dt = float(dt)
        self.cfg = config
        self.gammas = tuple(float(g) for g in config.gammas)
        self.K = len(self.gammas)
        self._ll_states = [
            RecurrentLeadLagLogSigMap(
                state_dim=2, level=2, forgetting_factor=g,
            )
            for g in self.gammas
        ]
        self._heads = [
            BLRHead3.fresh(
                prior_w_var=config.prior_w_var,
                sigma_n2_init=config.sigma_n2_init,
                sigma_n2_alpha=config.sigma_n2_alpha,
            )
            for _ in self.gammas
        ]
        self._kalman = OuterCIRKalmanV(
            kappa=config.kf_kappa,
            theta=config.kf_theta,
            xi=config.kf_xi,
            V=config.kf_theta,
            P=config.kf_xi ** 2 * config.kf_theta * self.dt * config.P_init_mult,
            V_floor=config.V_floor,
        )
        self._last_z: float = 0.0
        self._last_R_fused: float = float("nan")

    def reset(self, V0: float) -> None:
        self._kalman.V = max(float(V0), self.cfg.V_floor)
        self._kalman.P = (
            self.cfg.kf_xi ** 2 * self.cfg.kf_theta * self.dt * self.cfg.P_init_mult
        )
        for state in self._ll_states:
            state.reset()
        for head in self._heads:
            head.reset()
        self._last_z = 0.0
        self._last_R_fused = float("nan")

    def observe(self, r_t: float, dt: float) -> None:
        dx = np.array([float(dt), float(r_t)])
        # Per-scale feature extraction.
        phis: List[np.ndarray] = []
        for state in self._ll_states:
            state.update(dx)
            phis.append(_phi_from_l1_l2(state.l1, state.l2))
        # Per-scale BLR predictive posterior.
        ys = []
        Rs = []
        for head, phi in zip(self._heads, phis):
            y_pred, R_pred = head.predict(phi)
            ys.append(max(y_pred, 1e-8))
            Rs.append(R_pred)
        # Precision-fused scalar observation of V.
        y_bar, R_bar = _precision_combine(ys, Rs)
        self._last_R_fused = float(R_bar)
        # Outer Kalman predict + absorb.
        V_pred, P_pred = self._kalman.predict(dt)
        V_new, P_new, z = self._kalman.absorb(V_pred, P_pred, y_bar, R_bar)
        self._kalman.V = V_new
        self._kalman.P = P_new
        self._last_z = z
        # BLR posterior updates on the realized r^2/dt target.
        target = (float(r_t) ** 2) / float(dt)
        if self.cfg.target_clip is not None:
            target = min(target, self.cfg.target_clip)
        for head, phi in zip(self._heads, phis):
            head.update(phi, target)

    def V_hat(self) -> float:
        return float(self._kalman.V)

    def V_interval(self) -> Tuple[float, float]:
        sd = float(np.sqrt(max(self._kalman.P, 1e-18)))
        return (
            float(max(self._kalman.V - 1.645 * sd, 0.0)),
            float(self._kalman.V + 1.645 * sd),
        )

    def last_z(self) -> float:
        return float(self._last_z)

    def last_R_fused(self) -> float:
        return float(self._last_R_fused)


# ==========================================================================
# CumulativeStrideLeadLagBLRKFilter (cumulative sig + Chen-level-2 windows)
# ==========================================================================


@dataclass
class CumulativeStrideLeadLagBLRKFConfig:
    strides: Sequence[int] = ()
    kf_kappa: float = 2.0
    kf_theta: float = 0.04
    kf_xi: float = 0.3
    V_floor: float = 1e-6
    P_init_mult: float = 10.0
    prior_w_var: float = 10.0
    sigma_n2_init: float = 0.01
    sigma_n2_alpha: float = 0.01
    target_clip: Optional[float] = 2.0


class CumulativeStrideLeadLagBLRKFilter:
    r"""One cumulative (gamma=1.0) lead-lag log-sig + Chen-level-2
    window extraction at multiple strides -> K Bayesian heads ->
    precision-fused -> outer CIR Kalman on V.

    Window log-sig reconstruction
    -----------------------------
    With cumulative A_t := log sig_{[0, t]},
        a_1_window  =  A_t.l1  -  A_{t-m}.l1
        a_2_window  =  A_t.l2  -  A_{t-m}.l2  -  0.5 * [A_{t-m}.l1, A_t.l1]
    where [x, y]_ij = x_i y_j - x_j y_i is the Lie bracket on antisymmetric
    pairs (i < j).  This is EXACT at log-sig level 2 (no higher-order
    corrections because level-2 truncation is a nilpotent Lie algebra).

    Semantic phi normalization
    --------------------------
    The recovered window Levy area is ~= 0.5 * sum_{k=t-m..t-1} dx^2_k.
    Normalized by (m * dt) we get the average r^2/dt over the window,
    a V-scale estimator.  The normalized features are comparable across
    strides and across scales.
    """
    name = "cum_stride_leadlag_blrkf"

    def __init__(self, dt: float, config: CumulativeStrideLeadLagBLRKFConfig):
        if len(config.strides) < 1:
            raise ValueError("strides must be non-empty")
        self.dt = float(dt)
        self.cfg = config
        self.strides = tuple(int(m) for m in config.strides)
        self.K = len(self.strides)
        self._ll_state = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=1.0,
        )
        # Ring buffer of cumulative log-sig snapshots (l1, l2).
        self._max_stride = max(self.strides)
        self._buf_size = self._max_stride + 2
        self._l1_buf = np.zeros((self._buf_size, self._ll_state.dim_l1))
        self._l2_buf = np.zeros((self._buf_size, self._ll_state.dim_l2))
        self._buf_idx = 0
        self._buf_count = 0
        self._heads = [
            BLRHead3.fresh(
                prior_w_var=config.prior_w_var,
                sigma_n2_init=config.sigma_n2_init,
                sigma_n2_alpha=config.sigma_n2_alpha,
            )
            for _ in self.strides
        ]
        self._kalman = OuterCIRKalmanV(
            kappa=config.kf_kappa,
            theta=config.kf_theta,
            xi=config.kf_xi,
            V=config.kf_theta,
            P=config.kf_xi ** 2 * config.kf_theta * self.dt * config.P_init_mult,
            V_floor=config.V_floor,
        )
        self._last_z: float = 0.0

    def reset(self, V0: float) -> None:
        self._kalman.V = max(float(V0), self.cfg.V_floor)
        self._kalman.P = (
            self.cfg.kf_xi ** 2 * self.cfg.kf_theta * self.dt * self.cfg.P_init_mult
        )
        self._ll_state.reset()
        self._l1_buf[:] = 0.0
        self._l2_buf[:] = 0.0
        self._buf_idx = 0
        self._buf_count = 0
        for head in self._heads:
            head.reset()
        self._last_z = 0.0

    def _buf_get(self, m: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r"""Return (l1, l2) snapshots from m steps ago, or (None, None)
        if history is insufficient."""
        # After step t, _buf_idx has been incremented to t+1.  The snapshot
        # stored for step t is at index (_buf_idx - 1) % buf_size.
        # The snapshot from m steps before is at (_buf_idx - 1 - m) % buf_size,
        # provided we have at least m+1 snapshots (_buf_count > m).
        if self._buf_count <= m:
            return None, None
        idx = (self._buf_idx - 1 - m) % self._buf_size
        return self._l1_buf[idx].copy(), self._l2_buf[idx].copy()

    def _chen_window(
        self, a1_now: np.ndarray, a2_now: np.ndarray, m: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a1_past, a2_past = self._buf_get(m)
        if a1_past is None:
            return None, None
        a1_win = a1_now - a1_past
        # Level-2 antisymmetric bracket on (a_1_past, a_1_now):
        # For i < j:  [x, y]_{(i,j)} = x_i y_j - x_j y_i.
        d = a1_past.size
        bracket = np.zeros(self._ll_state.dim_l2)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                bracket[idx] = a1_past[i] * a1_now[j] - a1_past[j] * a1_now[i]
                idx += 1
        a2_win = a2_now - a2_past - 0.5 * bracket
        return a1_win, a2_win

    def _phi_from_window(
        self, a1_win: np.ndarray, a2_win: np.ndarray, m: int,
    ) -> np.ndarray:
        r"""Normalized 3-feature phi from window log-sig.

        The window Levy area at QV index ~= 0.5 * sum r_k^2 over the
        window, so 2 * area / (m * dt) ~= average r^2/dt ~= average V.
        Ret_lead divided by (m * dt) ~= average log-return / dt.
        """
        scale = max(float(m) * self.dt, 1e-12)
        qv_norm = 2.0 * float(a2_win[_QV_IDX_IN_L2]) / scale
        ret_norm = float(a1_win[_RET_LEAD_IDX_IN_L1]) / scale
        return np.array([qv_norm, ret_norm, 1.0])

    def observe(self, r_t: float, dt: float) -> None:
        dx = np.array([float(dt), float(r_t)])
        self._ll_state.update(dx)
        a1_now = self._ll_state.l1.copy()
        a2_now = self._ll_state.l2.copy()
        # Store snapshot AFTER the update.
        idx = self._buf_idx % self._buf_size
        self._l1_buf[idx] = a1_now
        self._l2_buf[idx] = a2_now
        self._buf_idx += 1
        self._buf_count = min(self._buf_count + 1, self._buf_size)
        # Per-stride BLR predictive.
        active_ys: List[float] = []
        active_Rs: List[float] = []
        active_updates: List[Tuple[int, np.ndarray]] = []
        for k, (m, head) in enumerate(zip(self.strides, self._heads)):
            a1_win, a2_win = self._chen_window(a1_now, a2_now, m)
            if a1_win is None:
                continue
            phi = self._phi_from_window(a1_win, a2_win, m)
            y_pred, R_pred = head.predict(phi)
            y_pred = max(y_pred, 1e-8)
            active_ys.append(y_pred)
            active_Rs.append(R_pred)
            active_updates.append((k, phi))
        # Outer Kalman predict.
        V_pred, P_pred = self._kalman.predict(dt)
        if not active_ys:
            # Warm-up phase: no window recoverable at any stride yet.
            self._kalman.V = max(V_pred, self.cfg.V_floor)
            self._kalman.P = P_pred
            self._last_z = 0.0
        else:
            y_bar, R_bar = _precision_combine(active_ys, active_Rs)
            V_new, P_new, z = self._kalman.absorb(V_pred, P_pred, y_bar, R_bar)
            self._kalman.V = V_new
            self._kalman.P = P_new
            self._last_z = z
        # BLR posterior updates on r^2/dt target.  Because the phi feature
        # is V-scale-normalized (mean r^2/dt over the window), the realized
        # target r_t^2/dt is the correct supervised signal.
        target = (float(r_t) ** 2) / float(dt)
        if self.cfg.target_clip is not None:
            target = min(target, self.cfg.target_clip)
        for k, phi in active_updates:
            self._heads[k].update(phi, target)

    def V_hat(self) -> float:
        return float(self._kalman.V)

    def V_interval(self) -> Tuple[float, float]:
        sd = float(np.sqrt(max(self._kalman.P, 1e-18)))
        return (
            float(max(self._kalman.V - 1.645 * sd, 0.0)),
            float(self._kalman.V + 1.645 * sd),
        )

    def last_z(self) -> float:
        return float(self._last_z)
