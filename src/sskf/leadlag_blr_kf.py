r"""
Revived LEAD-LAG BLR+KF Heston variance filter.

Signature's role in this module
-------------------------------
The signature here acts as a **feature map for a Bayesian observation
model**, NOT as the filter state itself.  More specifically:

  * Lead-lag log-signature (BCH updates, gamma ~= 0.99 ~ 100-day window)
    produces 10 cumulative features of the (dt, return) stream.
  * We pick exactly THREE of those features by semantic role:
      idx 8 : Levy area between ret_lead and ret_lag ~= exponentially
              fading realized QV.  Proxy for E[r^2/dt].
      idx 1 : ret_lead level-1 component  (leverage/drift channel).
      2     : constant bias term.
  * A 3-weight Bayesian linear regression learns
        E[r^2/dt | phi_t]  ~=  phi_t^T w,    w ~ N(w_mean, P_cov * sigma_n^2)
    with adaptive sigma_n^2.
  * The BLR's predictive (mean, variance) feeds into an OUTER KALMAN
    FILTER with CIR dynamics:
        y_t = phi_t^T w_mean               (BLR posterior predictive mean)
        R_t = phi_t^T P_cov phi_t  +  sigma_n^2     (full BLR predictive var)
        V_{t+1|t} = V_t + kappa (theta - V_t) dt
        P_{t+1|t} = (1 - kappa dt)^2 P_t + xi^2 V_t dt
        Kalman update with (y_t, R_t) -> (V_t, P_t).

Design credit
-------------
Architecture revived from
  `kronic_pomdp/experiments/graduated_sanity_checks.py`  L869-943
  (the `BLR+KF` branch inside `_run_sdre_control` when `use_leadlag=True`).

The project memory note reports V_corr approximately 0.76 on Heston for
this architecture -- the strongest memory-supported signature result in
the repo.  The strength likely comes from four pieces (in order of
expected importance):

  1. Lead-lag DOUBLING of the input path, so the Levy area between lead
     and lag channels provides a clean QV estimator as a level-2
     antisymmetric component.  The current DualTargetSigBLF uses a plain
     RecurrentSignatureMap and lacks this QV-ready feature.
  2. Feature selection to 3 components (QV + drift + bias).  Low-dim
     BLR has low weight-variance and far less overfitting than a
     6-or-more-dim RLS/BLR on the full signature.
  3. Long forgetting gamma = 0.99 (~100-day window) -- matches Heston
     mean-reversion timescale 1/kappa ~= 126 days.
  4. Outer Kalman with CIR dynamics -- keeps the state mean-reverting and
     bounded regardless of BLR misfits.

What is NOT in this module
--------------------------
  * Not a full posterior over the lifted state (that's StreamingSigKKF).
  * Not a drift head.  We target r^2/dt ONLY; mu_hat would require a
    second BLR and a CdC decomposition, which we deliberately keep out
    of the revived lane to preserve its tight 3-feature footprint.
  * Not multi-scale.  The MultiScaleSigFilter variant is a separate
    future experiment; this lane uses a single gamma.
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
from examples.proof_of_concept.signature_features import (
    RecurrentLeadLagLogSigMap,
)


# ==========================================================================
# Config
# ==========================================================================


@dataclass
class LeadLagBLRKFConfig:
    r"""Config for `LeadLagBLRKFilter`.

    Signature:
        ll_gamma:         forgetting factor of the lead-lag log-sig
                          (0.99 ~ 100 trading-day effective window)
    BLR (3-feature regression on [QV_area, ret_lead, 1.0] -> r^2/dt):
        prior_w_var:      prior variance scale on BLR weights (N(0, v*I))
        sigma_n2_init:    initial observation noise variance in BLR
        sigma_n2_alpha:   EWMA coefficient for adaptive sigma_n^2
        target_clip:      upper clip on r^2/dt (robustness hack the old code used);
                          set None to disable clipping
    Outer Kalman (CIR dynamics):
        kf_kappa, kf_theta, kf_xi: known CIR parameters
        V_floor:          lower floor on V during the update
        P_init_mult:      initial outer-filter uncertainty = xi^2 theta dt * P_init_mult

    Indices into a 10-dim lead-lag log-sig with input_dim=2:
        l1:    [time_lead(0), ret_lead(1), time_lag(2), ret_lag(3)]
        l2:    pairs (i<j): (0,1)=4, (0,2)=5, (0,3)=6,
                             (1,2)=7, (1,3)=8, (2,3)=9
    """
    ll_gamma: float = 0.99
    prior_w_var: float = 10.0
    sigma_n2_init: float = 0.01
    sigma_n2_alpha: float = 0.01
    target_clip: Optional[float] = 2.0
    kf_kappa: float = 2.0
    kf_theta: float = 0.04
    kf_xi: float = 0.3
    V_floor: float = 1e-6
    P_init_mult: float = 10.0

    QV_idx: int = 8         # lead-lag Levy area between ret_lead and ret_lag
    ret_lead_idx: int = 1   # level-1 ret_lead channel


# ==========================================================================
# Filter class
# ==========================================================================


class LeadLagBLRKFilter:
    r"""Revived lead-lag BLR + outer Kalman (CIR) variance filter.

    Signature plays the role of *observation constructor* (feature map
    for a Bayesian linear regression that predicts E[r^2/dt | phi]).  The
    outer Kalman is the actual V-state filter with CIR dynamics.
    """
    name = "blr_kf_leadlag"

    def __init__(self, dt: float, config: Optional[LeadLagBLRKFConfig] = None):
        self.dt = float(dt)
        self.cfg = config or LeadLagBLRKFConfig()
        self.ll_sig = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=self.cfg.ll_gamma,
        )
        # BLR state: 3-dim weight vector, covariance sigma_n^2 * P_cov.
        # We parameterize so P_cov is dimensionless and sigma_n^2 carries scale.
        self._blr_w = np.zeros(3)
        self._blr_P = np.eye(3) * self.cfg.prior_w_var
        self._sigma_n2 = float(self.cfg.sigma_n2_init)
        # Outer Kalman state.
        self._V: float = float(self.cfg.kf_theta)
        self._P: float = (
            float(self.cfg.kf_xi) ** 2
            * float(self.cfg.kf_theta)
            * float(dt)
            * float(self.cfg.P_init_mult)
        )
        self._last_z_kf: float = 0.0
        self._last_y_obs: float = float("nan")
        self._last_R_kf: float = float("nan")

    def reset(self, V0: float):
        self.ll_sig.reset()
        # Keep BLR posterior warm across episodes in a lane-agnostic
        # setup -- but the three-lane harness calls reset() at each
        # episode start, so reset BLR too to keep episodes independent.
        self._blr_w = np.zeros(3)
        self._blr_P = np.eye(3) * self.cfg.prior_w_var
        self._sigma_n2 = float(self.cfg.sigma_n2_init)
        self._V = max(float(V0), self.cfg.V_floor)
        self._P = (
            self.cfg.kf_xi ** 2
            * self.cfg.kf_theta
            * self.dt
            * self.cfg.P_init_mult
        )
        self._last_z_kf = 0.0
        self._last_y_obs = float("nan")
        self._last_R_kf = float("nan")

    def _extract_phi(self, feat_full: np.ndarray) -> np.ndarray:
        return np.array([
            float(feat_full[self.cfg.QV_idx]),
            float(feat_full[self.cfg.ret_lead_idx]),
            1.0,
        ])

    def observe(self, r_t: float, dt: float) -> None:
        # 1. Update lead-lag log-signature.
        dx = np.array([dt, float(r_t)])
        feat_full = self.ll_sig.update(dx)
        phi = self._extract_phi(feat_full)

        # 2. BLR predictive distribution  (BEFORE posterior update):
        #       y_hat = phi^T w_mean
        #       var(y) = phi^T P_cov phi + sigma_n^2
        #    Clip y_hat positive (V is positive).
        y_hat = float(phi @ self._blr_w)
        y_hat_for_kf = max(y_hat, 1e-8)
        R_blr = max(float(phi @ self._blr_P @ phi) + self._sigma_n2, 1e-8)
        self._last_y_obs = y_hat_for_kf
        self._last_R_kf = R_blr

        # 3. Outer Kalman (CIR dynamics) predict step.
        V_pred = (
            self._V
            + self.cfg.kf_kappa * (self.cfg.kf_theta - self._V) * dt
        )
        V_pred = max(V_pred, self.cfg.V_floor)
        Q_kf = self.cfg.kf_xi ** 2 * max(self._V, self.cfg.V_floor) * dt
        P_pred = (1.0 - self.cfg.kf_kappa * dt) ** 2 * self._P + Q_kf

        # 4. Outer Kalman update using BLR observation (y_hat, R_blr).
        innov_kf = y_hat_for_kf - V_pred
        S_kf = P_pred + R_blr
        self._last_z_kf = float(innov_kf / np.sqrt(max(S_kf, 1e-18)))
        K_kf = P_pred / max(S_kf, 1e-18)
        self._V = max(V_pred + K_kf * innov_kf, self.cfg.V_floor)
        self._P = (1.0 - K_kf) * P_pred

        # 5. BLR posterior update on the REALIZED r^2/dt (Kalman-on-weights).
        target = (float(r_t) ** 2) / float(dt)
        if self.cfg.target_clip is not None:
            target = min(target, self.cfg.target_clip)
        Cp = self._blr_P @ phi
        S_w = float(phi @ Cp) + self._sigma_n2
        S_w = max(S_w, 1e-18)
        K_w = Cp / S_w
        resid = target - float(phi @ self._blr_w)
        self._blr_w = self._blr_w + K_w * resid
        self._blr_P = self._blr_P - np.outer(K_w, Cp)
        self._blr_P = 0.5 * (self._blr_P + self._blr_P.T)

        # 6. Adaptive sigma_n^2 via EWMA of squared residual.
        self._sigma_n2 = max(
            (1.0 - self.cfg.sigma_n2_alpha) * self._sigma_n2
            + self.cfg.sigma_n2_alpha * (resid * resid),
            1e-8,
        )

    def V_hat(self) -> float:
        return float(self._V)

    def V_interval(self) -> Tuple[float, float]:
        r"""Gaussian 90% interval from the outer Kalman (state posterior)."""
        sd = float(np.sqrt(max(self._P, 1e-18)))
        return (
            float(max(self._V - 1.645 * sd, 0.0)),
            float(self._V + 1.645 * sd),
        )

    def last_z(self) -> float:
        r"""Standardized innovation at the OUTER Kalman level."""
        return float(self._last_z_kf)

    def last_trace_P(self) -> float:
        r"""Outer Kalman state-variance (scalar)."""
        return float(self._P)

    def blr_weights(self) -> np.ndarray:
        return self._blr_w.copy()

    def blr_P_trace(self) -> float:
        return float(np.trace(self._blr_P))

    def last_R_kf(self) -> float:
        return float(self._last_R_kf)
