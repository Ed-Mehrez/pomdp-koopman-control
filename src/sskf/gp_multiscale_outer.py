r"""
Multiscale Gaussian-process outer prior for V (state-space form).

Model
-----
V_t = θ + s_t^(fast) + s_t^(slow)

with each s^(k) a zero-mean OU:
    ds^(k) = − κ_k · s^(k) · dt + ξ_k · dB_k,    B_fast ⊥ B_slow.

This is the SS-form of a sum-of-two-Matérn-1/2 (Ornstein-Uhlenbeck)
Gaussian process on V_t.  Equivalent to a 2-factor mean-reverting prior
on volatility, which is the Markovian discretisation that approximates
rough/multi-scale vol families (Bayer-Friz-Gatheral; Bayer-Breneis 2023).

Why this is the right structural alternative to CIR
---------------------------------------------------
- CIR commits to one fixed κ, one fixed θ, one fixed ξ, and the specific
  ξ√V diffusion form.  Empirically vol IS mean-reverting, but at multiple
  timescales and with regime-shifting long-run levels.
- Random walk has no mean-reversion structure -- inconsistent with the
  empirical ACF of variance.
- Multi-factor OU (this module) preserves mean-reversion at K timescales
  and has only K rates + K vols + 1 long-run mean, all of which can be
  fit from the data via the empirical ACF or PSD.

Calibration is data-driven via two-exponential fit to the empirical
realized-variance ACF.  Cross-checked against the spectral knee
diagnostic from `koopman-pricing/experiments/multiscale_pricing_benchmark.py`
(estimate_knee_frequency).

Filter cost is O((2K)²) per step where K = number of factors.  For K=2,
that's 4 floating-point ops per step beyond the BLR -- negligible.

References
----------
Hartikainen & Särkkä 2010, "Kalman filtering and smoothing solutions to
temporal Gaussian process regression models" (state-space form for GPs).
Bayer-Breneis 2023, "Markovian approximations of stochastic Volterra
equations with the fractional kernel" (multi-factor OU as Markovian
discretization of rough vol).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ==========================================================================
# Two-factor OU configuration
# ==========================================================================


@dataclass
class TwoFactorOUConfig:
    r"""Parameters of V = θ + s^(fast) + s^(slow)."""
    theta: float
    kappa_fast: float
    kappa_slow: float
    xi_fast: float
    xi_slow: float
    V_floor: float = 1e-8

    def stationary_var(self) -> float:
        r"""Total stationary variance of V (sum across factors)."""
        return (
            self.xi_fast ** 2 / (2.0 * max(self.kappa_fast, 1e-12))
            + self.xi_slow ** 2 / (2.0 * max(self.kappa_slow, 1e-12))
        )

    def half_lives_days(self, dt: float) -> Tuple[float, float]:
        r"""Half-life of each factor's mean-reversion in trading days."""
        return (
            float(np.log(2.0) / max(self.kappa_fast * dt, 1e-12)),
            float(np.log(2.0) / max(self.kappa_slow * dt, 1e-12)),
        )


# ==========================================================================
# Two-factor OU Kalman (state-space form GP outer)
# ==========================================================================


class TwoFactorOUKalmanV:
    r"""Online Kalman filter for V under two-factor OU prior.

    State: x = (s_fast, s_slow).  V observation y_obs ≈ θ + 1·x + noise.
    Exact discrete-time form with closed-form transition matrices:
        F = diag(exp(−κ_fast·dt), exp(−κ_slow·dt))
        Q = diag( ξ_k² · (1 − exp(−2κ_k·dt)) / (2κ_k)  for k in {fast, slow})
    """

    def __init__(self, dt: float, config: TwoFactorOUConfig):
        self.dt = float(dt)
        self.cfg = config
        self.F = np.diag([
            float(np.exp(-config.kappa_fast * dt)),
            float(np.exp(-config.kappa_slow * dt)),
        ])
        self.Q = np.diag([
            (config.xi_fast ** 2 / (2.0 * max(config.kappa_fast, 1e-12)))
            * (1.0 - float(np.exp(-2.0 * config.kappa_fast * dt))),
            (config.xi_slow ** 2 / (2.0 * max(config.kappa_slow, 1e-12)))
            * (1.0 - float(np.exp(-2.0 * config.kappa_slow * dt))),
        ])
        self.H = np.array([1.0, 1.0])              # 1×2 observation matrix
        # State and covariance, initialized at stationarity
        self.x = np.zeros(2)
        var_fast = config.xi_fast ** 2 / (2.0 * max(config.kappa_fast, 1e-12))
        var_slow = config.xi_slow ** 2 / (2.0 * max(config.kappa_slow, 1e-12))
        self.P = np.diag([var_fast, var_slow])
        self._last_z = 0.0

    def reset(self, V0: float) -> None:
        delta = float(V0) - self.cfg.theta
        var_fast = self.cfg.xi_fast ** 2 / (2.0 * max(self.cfg.kappa_fast, 1e-12))
        var_slow = self.cfg.xi_slow ** 2 / (2.0 * max(self.cfg.kappa_slow, 1e-12))
        total = var_fast + var_slow
        if total > 0:
            self.x = np.array([
                delta * (var_fast / total),
                delta * (var_slow / total),
            ])
        else:
            self.x = np.zeros(2)
        self.P = np.diag([var_fast, var_slow])
        self._last_z = 0.0

    def predict(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        V_pred = float(self.cfg.theta + self.H @ x_pred)
        V_pred_floored = max(V_pred, self.cfg.V_floor)
        P_V = float(self.H @ P_pred @ self.H)
        return V_pred_floored, P_V, x_pred, P_pred

    def absorb(
        self,
        V_pred: float, P_V: float,
        x_pred: np.ndarray, P_pred: np.ndarray,
        y_obs: float, R: float,
    ) -> Tuple[float, float]:
        S = P_V + R
        K = (P_pred @ self.H) / max(S, 1e-18)         # 2-vector
        innov = float(y_obs) - V_pred
        z = float(innov / np.sqrt(max(S, 1e-18)))
        x_new = x_pred + K * innov
        # Joseph form for covariance update (numerical stability)
        I = np.eye(2)
        IKH = I - np.outer(K, self.H)
        P_new = IKH @ P_pred @ IKH.T + np.outer(K, K) * R
        self.x = x_new
        self.P = P_new
        V_new = max(self.cfg.theta + float(self.H @ self.x), self.cfg.V_floor)
        self._last_z = z
        return V_new, z

    def V_hat(self) -> float:
        return float(max(self.cfg.theta + self.H @ self.x, self.cfg.V_floor))

    def V_interval(self) -> Tuple[float, float]:
        var_V = float(self.H @ self.P @ self.H)
        sd = float(np.sqrt(max(var_V, 1e-18)))
        return (
            float(max(self.V_hat() - 1.645 * sd, 0.0)),
            float(self.V_hat() + 1.645 * sd),
        )

    def last_z(self) -> float:
        return float(self._last_z)


# ==========================================================================
# Calibration: empirical ACF → two-exponential fit
# ==========================================================================


def _ewma_smooth(rv: np.ndarray, halflife_steps: float) -> np.ndarray:
    alpha = 1.0 - float(np.exp(-np.log(2.0) / max(halflife_steps, 1e-3)))
    out = np.zeros_like(rv)
    s = float(rv[0])
    for t, r in enumerate(rv):
        s = (1.0 - alpha) * s + alpha * float(r)
        out[t] = s
    return out


def _empirical_acf(centered: np.ndarray, max_lag: int) -> np.ndarray:
    var0 = float(np.mean(centered * centered))
    if var0 <= 0:
        return np.zeros(max_lag + 1)
    N = centered.size
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = float(np.mean(centered[: N - k] * centered[k:])) / var0
    return out


def calibrate_two_factor_ou(
    rv_series: np.ndarray,
    dt: float,
    halflife_smooth_steps: float = 21.0,
    max_lag_frac: float = 0.25,
    kappa_init_yr: Tuple[float, float] = (50.0, 1.0),
    seed: int = 0,
) -> TwoFactorOUConfig:
    r"""Fit a 2-factor OU prior to a pilot realized-variance series.

    Steps
    -----
    1.  EWMA-smooth the RV with the given halflife (default 21 steps).
    2.  θ̂ = mean(smoothed RV).  total Var = Var(smoothed RV).
    3.  Empirical ACF up to max_lag = max_lag_frac × N.
    4.  Fit two-exponential ACF model via Nelder-Mead (no SciPy needed):
            ρ̂(τ) ≈ w · exp(−κ_fast · τ · dt) + (1−w) · exp(−κ_slow · τ · dt)
        with constraints κ_fast > κ_slow > 0, 0 < w < 1.
    5.  Recover ξ_fast, ξ_slow from total Var(V) split by weight w:
            ξ_fast² / (2 κ_fast) = w · Var(V)
            ξ_slow² / (2 κ_slow) = (1 − w) · Var(V)

    No SciPy dependency: uses an inline Nelder-Mead with multiple inits.
    """
    from numpy.random import RandomState
    rv = np.asarray(rv_series, dtype=float).flatten()
    rv = rv[np.isfinite(rv) & (rv > 0)]
    if rv.size < 50:
        # Fallback: defaults
        return TwoFactorOUConfig(
            theta=0.04,
            kappa_fast=kappa_init_yr[0], kappa_slow=kappa_init_yr[1],
            xi_fast=0.3, xi_slow=0.05,
        )

    smoothed = _ewma_smooth(rv, halflife_smooth_steps)
    theta = float(np.mean(smoothed))
    var_total = float(np.var(smoothed, ddof=1))
    centered = smoothed - theta
    L = max(5, int(max_lag_frac * rv.size))
    acf = _empirical_acf(centered, L)
    lags = np.arange(L + 1)
    tau_yr = lags * dt

    def loss(params: np.ndarray) -> float:
        log_kf, log_ks, logit_w = float(params[0]), float(params[1]), float(params[2])
        kf = float(np.exp(log_kf))
        ks = float(np.exp(log_ks))
        if kf <= ks * 1.05:
            return 1e10                              # enforce kf > ks (factor 1.05 margin)
        w = 1.0 / (1.0 + float(np.exp(-logit_w)))
        pred = w * np.exp(-kf * tau_yr) + (1.0 - w) * np.exp(-ks * tau_yr)
        # Weighted MSE: emphasise low-lag fit (where ACF is most informative)
        weights = np.exp(-lags / max(L * 0.3, 1.0))
        return float(np.sum(weights * (acf - pred) ** 2))

    # Multistart Nelder-Mead (no SciPy): initial points from a small grid
    best_x = None
    best_v = float("inf")
    rng = RandomState(seed)
    for _ in range(8):
        kf0 = float(rng.uniform(20.0, 200.0))
        ks0 = float(rng.uniform(0.2, 5.0))
        if kf0 <= ks0 * 2:
            kf0 = ks0 * 5
        x0 = np.array([np.log(kf0), np.log(ks0), 0.0])
        x = _nelder_mead(loss, x0, max_iters=400, tol=1e-7)
        v = loss(x)
        if v < best_v:
            best_v = v
            best_x = x

    log_kf, log_ks, logit_w = float(best_x[0]), float(best_x[1]), float(best_x[2])
    kappa_fast = float(np.exp(log_kf))
    kappa_slow = float(np.exp(log_ks))
    weight = 1.0 / (1.0 + float(np.exp(-logit_w)))

    # Recover xi from the variance split
    xi_fast = float(np.sqrt(max(weight * var_total * 2.0 * kappa_fast, 0.0)))
    xi_slow = float(np.sqrt(max((1.0 - weight) * var_total * 2.0 * kappa_slow, 0.0)))

    return TwoFactorOUConfig(
        theta=theta,
        kappa_fast=kappa_fast, kappa_slow=kappa_slow,
        xi_fast=xi_fast, xi_slow=xi_slow,
    )


# ==========================================================================
# Inline Nelder-Mead (no SciPy needed)
# ==========================================================================


def _nelder_mead(
    f, x0: np.ndarray,
    alpha: float = 1.0, gamma: float = 2.0, rho: float = 0.5, sigma: float = 0.5,
    max_iters: int = 500, tol: float = 1e-7,
) -> np.ndarray:
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        v = np.array(x0, dtype=float)
        v[i] += 0.5 if v[i] == 0 else 0.05 * abs(v[i])
        simplex.append(v)
    fvals = [f(v) for v in simplex]
    for it in range(max_iters):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if fvals[-1] - fvals[0] < tol:
            break
        x_centroid = np.mean(simplex[:-1], axis=0)
        # Reflection
        x_r = x_centroid + alpha * (x_centroid - simplex[-1])
        f_r = f(x_r)
        if fvals[0] <= f_r < fvals[-2]:
            simplex[-1] = x_r; fvals[-1] = f_r; continue
        if f_r < fvals[0]:
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e; fvals[-1] = f_e
            else:
                simplex[-1] = x_r; fvals[-1] = f_r
            continue
        x_c = x_centroid + rho * (simplex[-1] - x_centroid)
        f_c = f(x_c)
        if f_c < fvals[-1]:
            simplex[-1] = x_c; fvals[-1] = f_c; continue
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            fvals[i] = f(simplex[i])
    return simplex[int(np.argmin(fvals))]
