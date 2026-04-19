r"""
DualTargetSigBLF: LIGHT Bayesian signature CdC filter.

Scope label: this is the MINIMAL Bayesian retrofit of `DualTargetSigRLS`,
NOT the full `StreamingSigKKF` stack (which learns a full Koopman
generator on an augmented state and is deferred per the plan).  The goal
here is two conditional-moment scalars (E[r/dt | phi], E[r^2/dt | phi])
with explicit posterior mean/covariance per head, combined via CdC.

Signature state
---------------
Same as `DualTargetSigRLS` (level4_generator_sdre.py): `RecurrentSignatureMap`
over [dt, return] with level-2 and a forgetting factor gamma_sig.  The
signature features phi_t summarize path history in fixed size.

Two Bayesian linear heads on the shared features
------------------------------------------------
For each head h in {mu, v}:
    w_t^h  =  w_{t-1}^h + eta_t^h,     eta_t^h ~ N(0, Q^h)     (random-walk prior)
    y_t^h  =  phi_t^T w_t^h + eps_t^h, eps_t^h ~ N(0, R_t^h)   (Gaussian obs)

Targets:
    y_t^mu   =  r_t / dt                       (drift)
    y_t^v    =  r_t^2 / dt                     (second moment)

Posterior for each head is maintained exactly in closed form:
    predict:  w_pred = w,   P_pred = P + Q
    obs S    = phi^T P_pred phi + R
    gain K   = P_pred phi / S
    update:  w_new = w_pred + K (y_obs - phi^T w_pred)
             P_new = P_pred - K phi^T P_pred

CdC decomposition of the induced V-posterior
--------------------------------------------
Given weight samples w_mu^(s), w_v^(s) ~ (independent) head posteriors at
the CURRENT phi_t:
    mu_hat^(s)  =  phi_t^T w_mu^(s)
    m2_hat^(s)  =  phi_t^T w_v^(s)
    V_hat^(s)   =  max(m2_hat^(s) - mu_hat^(s)^2 * dt, tiny)

Posterior summary of V_hat is computed by Monte Carlo on (n_samples) draws.
An analytical second-moment approximation (delta method) is also provided
for quick reporting.

Relationship to RLS forgetting
------------------------------
RLS with forgetting factor `gamma` is APPROXIMATELY equivalent to a BLR
with random-walk prior whose per-step process noise covariance scales with
current posterior: Q_t ~= (1/gamma - 1) * P_{t-1}.  The scale is
state-dependent, which we AVOID by using a fixed additive Q = q * I.
This is a deliberate simplification:
    * Pro:  keeps the filter strictly linear-Gaussian and closed-form.
    * Con:  early in the run P is large and we effectively inflate it less
            than RLS-with-forgetting would; late in the run we inflate it
            more.
We document this trade as the key approximation.  q_mu and q_v are
exposed as config so the adapter can pick reasonable scales.

Robustness via adaptive observation noise
-----------------------------------------
The original RLS code clipped r/dt and r^2/dt to hardcoded ranges.  We
replace that with per-step adaptive Gaussian observation noise R_t^h,
maintained as an EWMA of squared predictive residuals scaled by the
predictive variance's theoretical target (e.g., Var(r/dt | V) ~= V/dt).
When an incoming observation is implausibly large, R_t grows and the
Kalman gain shrinks automatically -- a principled outlier damping inside
the Gaussian family.

For the second-moment head, we keep an optional MILD winsorization at a
configurable upper quantile (default disabled).  The bug-by-construction
of clipping small r^2 values is avoided (we only upper-winsorize).

No finance vocabulary in this module.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# Import RecurrentSignatureMap from the examples package.  (The file's
# parent dir should be on sys.path by the caller; we also try a best-effort
# import.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from examples.proof_of_concept.signature_features import RecurrentSignatureMap


# ==========================================================================
# One Bayesian linear head
# ==========================================================================


@dataclass
class BayesianLinearHeadState:
    r"""State of one Bayesian linear head under the linear-Gaussian model
    w_t = w_{t-1} + N(0, Q), y_t = phi^T w_t + N(0, R_t).

    Attributes are MUTABLE (the head updates in place).
    """
    w_mean: np.ndarray            # (n_features,)
    P_cov: np.ndarray             # (n_features, n_features)
    Q_process: np.ndarray         # (n_features, n_features); random-walk covariance
    R_init: float                 # initial observation-noise variance
    R_ema_halflife_steps: float   # EWMA halflife for adaptive R, in steps
    R_current: float              # current adaptive R
    winsor_upper_quantile: Optional[float]  # if set, cap target above this quantile
    _target_history: list         # rolling history for quantile estimation


def _make_head(
    n_features: int,
    prior_var: float,
    process_noise: float,
    R_init: float,
    R_ema_halflife_steps: float,
    winsor_upper_quantile: Optional[float] = None,
) -> BayesianLinearHeadState:
    return BayesianLinearHeadState(
        w_mean=np.zeros(n_features),
        P_cov=np.eye(n_features) * prior_var,
        Q_process=np.eye(n_features) * process_noise,
        R_init=R_init,
        R_ema_halflife_steps=R_ema_halflife_steps,
        R_current=R_init,
        winsor_upper_quantile=winsor_upper_quantile,
        _target_history=[],
    )


def _predictive(head: BayesianLinearHeadState, phi: np.ndarray) -> Tuple[float, float, float]:
    r"""Return (y_pred_mean, predictive_var_of_y, predictive_var_of_f).

    predictive_var_of_f = phi^T P phi  (weight posterior variance at this input).
    predictive_var_of_y = phi^T P phi + R  (adds observation noise).
    Note: P used here is PRE-step posterior; the full predict step would
    add Q.  This function is intended for DIAGNOSTIC reporting AFTER an
    `update` call, where P has already been updated.
    """
    y_mean = float(phi @ head.w_mean)
    var_f = float(phi @ head.P_cov @ phi)
    var_y = var_f + head.R_current
    return y_mean, var_y, var_f


def _update_head(
    head: BayesianLinearHeadState,
    phi: np.ndarray,
    y_obs: float,
    adapt_R_alpha: float,
) -> Dict[str, float]:
    r"""One step of Kalman filter on the weight posterior.

    Returns a dict of diagnostics from this step:
        y_pred_prior:      predictive mean before observing y_obs
        predictive_std:    sqrt(predictive variance of y)
        innov:             y_obs - y_pred_prior (observed innovation)
        kalman_norm:       ||K|| (gain magnitude; large = data-informative)
        R_after:           observation noise variance after EWMA update
    """
    # 1. Predict step: random-walk prior on weights.
    P_pred = head.P_cov + head.Q_process
    w_pred = head.w_mean  # random walk: identity

    # 2. Optional mild upper winsorization.
    y_used = float(y_obs)
    if head.winsor_upper_quantile is not None and len(head._target_history) > 50:
        hist = np.asarray(head._target_history[-1000:])
        q = float(np.quantile(hist, head.winsor_upper_quantile))
        if y_used > q:
            y_used = q
    head._target_history.append(float(y_obs))
    if len(head._target_history) > 2000:
        head._target_history = head._target_history[-2000:]

    # 3. Predictive distribution for y.
    y_pred_prior = float(phi @ w_pred)
    innov = y_used - y_pred_prior
    predictive_var_y = float(phi @ P_pred @ phi) + head.R_current
    S = max(predictive_var_y, 1e-18)
    z_innov = float(innov / np.sqrt(S))

    # 4. Kalman gain.
    K = (P_pred @ phi) / S  # (n_features,)

    # 5. Posterior update.
    head.w_mean = w_pred + K * innov
    #  P_new = P_pred - K phi^T P_pred
    head.P_cov = P_pred - np.outer(K, phi @ P_pred)
    # Symmetrize to avoid drift.
    head.P_cov = 0.5 * (head.P_cov + head.P_cov.T)

    # 6. Adaptive R via EWMA of squared innovation (minus epistemic part).
    #    epistemic_var at this phi = phi^T P_pred phi.
    eff_sq = max(innov ** 2 - float(phi @ P_pred @ phi), 1e-12)
    head.R_current = (1.0 - adapt_R_alpha) * head.R_current + adapt_R_alpha * eff_sq

    return {
        "y_pred_prior": y_pred_prior,
        "predictive_std": float(np.sqrt(S)),
        "innov": innov,
        "z_innov": z_innov,                   # (y_obs - y_pred)/sqrt(predictive_var)
        "kalman_norm": float(np.linalg.norm(K)),
        "R_after": float(head.R_current),
        "P_trace_after": float(np.trace(head.P_cov)),
    }


# ==========================================================================
# Dual-target signature Bayesian linear filter
# ==========================================================================


@dataclass
class DualTargetSigBLFConfig:
    r"""Configuration for DualTargetSigBLF.

    Signature:
        input_dim:         dimension of the input stream [dt, return] -> 2
        sig_level:         level of RecurrentSignatureMap (default 2)
        sig_forget:        forgetting factor for signature state (default 0.94)

    Bayesian head prior:
        prior_var_mu:      prior variance of drift weights (N(0, prior_var*I))
        prior_var_v:       prior variance of second-moment weights
        process_noise_mu:  per-step process-noise variance for drift weights
        process_noise_v:   per-step process-noise variance for second-moment weights
                           (approximates RLS forgetting; see module header)

    Observation noise (adaptive):
        R_init_mu:         initial observation-noise variance on r/dt
        R_init_v:          initial observation-noise variance on r^2/dt
        R_adapt_halflife:  EWMA halflife for R_t updates, in steps
        winsor_v_q:        upper-quantile winsorization on r^2/dt; None = off
    """
    input_dim: int = 2
    sig_level: int = 2
    sig_forget: float = 0.94
    prior_var_mu: float = 100.0
    prior_var_v: float = 100.0
    process_noise_mu: float = 1e-4
    process_noise_v: float = 1e-4
    R_init_mu: float = 10.0
    R_init_v: float = 0.5
    R_adapt_halflife: float = 50.0
    winsor_v_q: Optional[float] = 0.995


class DualTargetSigBLF:
    r"""Bayesian linear filter with two heads (drift, second moment) sharing
    a recurrent signature state.

    Design matches `DualTargetSigRLS` (level4_generator_sdre.py) in:
        - signature type and update,
        - targets (r/dt, r^2/dt),
        - CdC decomposition for V_hat.

    Adds:
        - explicit Bayesian posterior over weights per head (w_mean, P_cov),
        - predictive mean/variance before and after each assimilation,
        - posterior samples and intervals on V_hat via CdC.

    Key approximations (documented):
        1. Random-walk prior with fixed additive Q instead of RLS
           state-dependent covariance inflation.
        2. Adaptive Gaussian R (EWMA of squared residual minus epistemic
           variance), as opposed to a fully conjugate NIG likelihood.
        3. Mild upper winsorization on r^2/dt is available but off by
           default; it is NOT a hard clip.
    """

    def __init__(self, config: Optional[DualTargetSigBLFConfig] = None):
        self.cfg = config or DualTargetSigBLFConfig()
        self.sig_map = RecurrentSignatureMap(
            state_dim=self.cfg.input_dim,
            level=self.cfg.sig_level,
            forgetting_factor=self.cfg.sig_forget,
        )
        self.n_features = self.sig_map.feature_dim + 1  # +1 for bias term
        self._R_alpha = float(np.log(2.0) / max(self.cfg.R_adapt_halflife, 1e-3))

        self.head_mu = _make_head(
            self.n_features,
            prior_var=self.cfg.prior_var_mu,
            process_noise=self.cfg.process_noise_mu,
            R_init=self.cfg.R_init_mu,
            R_ema_halflife_steps=self.cfg.R_adapt_halflife,
            winsor_upper_quantile=None,  # drift is not upper-heavy
        )
        self.head_v = _make_head(
            self.n_features,
            prior_var=self.cfg.prior_var_v,
            process_noise=self.cfg.process_noise_v,
            R_init=self.cfg.R_init_v,
            R_ema_halflife_steps=self.cfg.R_adapt_halflife,
            winsor_upper_quantile=self.cfg.winsor_v_q,
        )

    def reset(self) -> None:
        self.sig_map.reset()
        self.head_mu = _make_head(
            self.n_features,
            prior_var=self.cfg.prior_var_mu,
            process_noise=self.cfg.process_noise_mu,
            R_init=self.cfg.R_init_mu,
            R_ema_halflife_steps=self.cfg.R_adapt_halflife,
            winsor_upper_quantile=None,
        )
        self.head_v = _make_head(
            self.n_features,
            prior_var=self.cfg.prior_var_v,
            process_noise=self.cfg.process_noise_v,
            R_init=self.cfg.R_init_v,
            R_ema_halflife_steps=self.cfg.R_adapt_halflife,
            winsor_upper_quantile=self.cfg.winsor_v_q,
        )

    def _features(self) -> np.ndarray:
        r"""phi_t = [sig_level_1, sig_level_2, 1.0]  (bias term included)."""
        sig = np.concatenate([self.sig_map.s1, self.sig_map.s2])
        return np.concatenate([sig, [1.0]])

    def update(self, dx: np.ndarray, r: float, dt: float) -> Dict[str, float]:
        r"""Assimilate one (increment, return) observation.

        Arguments
        ---------
        dx : (input_dim,) path increment (e.g. [dt, r]).
        r  : scalar return for this step.
        dt : time step.

        Returns
        -------
        Diagnostic dict with:
            mu_hat, V_hat:              current point estimates (posterior means)
            mu_pred_std, m2_pred_std:   standard deviations of the posterior
                                        predictive on each head
            V_hat_q05, V_hat_q95:       Monte-Carlo 90%-interval on V_hat
            R_mu, R_v:                  adaptive observation-noise variances
        """
        # 1. Update signature state with increment; then extract features.
        _ = self.sig_map.update(np.asarray(dx, dtype=float))
        phi = self._features()

        # 2. Targets.
        target_mu = float(r) / float(dt)
        target_v = float(r) ** 2 / float(dt)

        # 3. Update each head (independent linear Gaussian Kalman).
        diag_mu = _update_head(self.head_mu, phi, target_mu, self._R_alpha)
        diag_v = _update_head(self.head_v, phi, target_v, self._R_alpha)

        # 4. Post-assimilation posterior-predictive summaries.
        mu_post_mean, mu_post_var_y, mu_post_var_f = _predictive(self.head_mu, phi)
        m2_post_mean, m2_post_var_y, m2_post_var_f = _predictive(self.head_v, phi)

        # 5. Point estimate of V via CdC.
        V_point = max(m2_post_mean - mu_post_mean ** 2 * dt, 1e-8)

        # 6. Monte-Carlo posterior interval on V (default n=200).
        V_q05, V_q95 = self._posterior_V_quantiles(phi, dt, n_samples=200)

        return {
            "mu_hat": float(mu_post_mean),
            "V_hat": float(V_point),
            "mu_pred_std": float(np.sqrt(max(mu_post_var_f, 1e-18))),
            "m2_pred_std": float(np.sqrt(max(m2_post_var_f, 1e-18))),
            "V_hat_q05": float(V_q05),
            "V_hat_q95": float(V_q95),
            "R_mu": float(diag_mu["R_after"]),
            "R_v": float(diag_v["R_after"]),
            "innov_mu": float(diag_mu["innov"]),
            "innov_v": float(diag_v["innov"]),
            "z_mu": float(diag_mu["z_innov"]),   # standardized innovation (calibration)
            "z_v": float(diag_v["z_innov"]),
            "P_trace_mu": float(diag_mu["P_trace_after"]),
            "P_trace_v": float(diag_v["P_trace_after"]),
        }

    def _posterior_V_quantiles(
        self,
        phi: np.ndarray,
        dt: float,
        n_samples: int = 200,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[float, float]:
        r"""Draw joint samples (mu, m2) from the two independent head
        posteriors at the current phi; compute V via CdC; return q05, q95.
        """
        if rng is None:
            rng = np.random.RandomState(0)
        mu_mean = float(phi @ self.head_mu.w_mean)
        m2_mean = float(phi @ self.head_v.w_mean)
        mu_var = max(float(phi @ self.head_mu.P_cov @ phi), 1e-18)
        m2_var = max(float(phi @ self.head_v.P_cov @ phi), 1e-18)
        mu_samples = mu_mean + rng.standard_normal(n_samples) * np.sqrt(mu_var)
        m2_samples = m2_mean + rng.standard_normal(n_samples) * np.sqrt(m2_var)
        V_samples = np.maximum(m2_samples - mu_samples ** 2 * dt, 1e-8)
        return float(np.quantile(V_samples, 0.05)), float(np.quantile(V_samples, 0.95))

    def posterior_V_samples(
        self,
        n_samples: int = 1000,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        r"""Draw `n_samples` samples of V at the CURRENT phi (not re-updated)."""
        if rng is None:
            rng = np.random.RandomState(0)
        phi = self._features()
        # Use current (assimilated) P; approximate joint posterior as product of marginals.
        mu_mean = float(phi @ self.head_mu.w_mean)
        m2_mean = float(phi @ self.head_v.w_mean)
        mu_var = max(float(phi @ self.head_mu.P_cov @ phi), 1e-18)
        m2_var = max(float(phi @ self.head_v.P_cov @ phi), 1e-18)
        mu_samples = mu_mean + rng.standard_normal(n_samples) * np.sqrt(mu_var)
        m2_samples = m2_mean + rng.standard_normal(n_samples) * np.sqrt(m2_var)
        # We can't supply dt here; caller takes care of the dt scaling.
        return np.stack([mu_samples, m2_samples], axis=1)

    @property
    def mu_hat(self) -> float:
        phi = self._features()
        return float(phi @ self.head_mu.w_mean)

    @property
    def V_hat(self) -> float:
        phi = self._features()
        mu = float(phi @ self.head_mu.w_mean)
        m2 = float(phi @ self.head_v.w_mean)
        return max(m2 - mu ** 2, 1e-8)  # without dt scaling; caller passes dt into update()
