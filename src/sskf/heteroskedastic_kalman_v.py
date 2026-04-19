r"""
Heteroskedastic Kalman on variance V (Heston CIR dynamics) + hybrid
signature-modulated variant.

Scope label
-----------
This is a LIGHT filter-first alternative to the Bayesian signature CdC
filter in `dual_target_sig_blf.py`.  NOT the full StreamingSigKKF stack.

Design source (explicit credit)
-------------------------------
  * `koopman-pricing/experiments/hf_panel_transfer_benchmark.py` ->
    heteroskedastic Kalman pattern with observation noise R_t that scales
    with current state; signature smoothing/blending on top.
  * `gp-bubbles/docs/theory_signature_filter.md` ->
    signature-only is insufficient for variance filtering; need a direct
    variance-proxy observation with recursive state feedback.

Lane 1: HeteroskedasticKalmanV  (pure, NO signature)
----------------------------------------------------
State:   V_t (scalar)
Dynamics (CIR linearization):
    V_{t+1|t}  =  V_t + kappa * (theta - V_t) * dt
    P_{t+1|t}  =  (1 - kappa * dt)**2 * P_t  +  xi**2 * V_t * dt
Observation:
    y_t = r_t**2 / dt
    y_t | V_t  ~  N(V_t,  R_t)
    R_t = R_scale * 2 * V_pred**2 / dt          (heteroskedastic)
Update:
    K_t = P_pred / (P_pred + R_t)
    V_t = max(V_pred + K_t * (y_t - V_pred), V_floor)
    P_t = (1 - K_t) * P_pred

Known vs learned dynamics
-------------------------
The filter takes (kappa, theta, xi) at construction.  We treat these as
KNOWN (same setup as the `Kalman(xi=true)` lane in level4_generator_sdre.py).
A model-free version is out of scope for this pass.

Lane 2: HybridKalmanSigV  (base filter + sig-conditioned R)
-----------------------------------------------------------
Extends Lane 1 with a signature-based modifier on R_t:
    sig_score_t  =  some positive functional of RecurrentSignatureMap state
    ratio_t      =  sig_score_t / EWMA(sig_score)
    R_t          =  R_base * clip(ratio_t ** gamma_R, R_min_mult, R_max_mult)

If the signature score at time t is elevated relative to its EWMA baseline
(i.e., the recent path looks unusually "active"), R_t inflates and the
Kalman downweights the single-step observation.  If the score is low,
R_t shrinks (slightly) and the observation is trusted more.  Clipping
the modifier prevents signal loss when the sig is misbehaving.

If signatures add no information, ratio_t hovers near 1 and the hybrid
degenerates to HeteroskedasticKalmanV.  That is by design: the hybrid
should fail GRACEFULLY, not dramatically.

Reporting contract
------------------
Both classes expose:
    V_hat() -> scalar posterior mean
    V_interval() -> Gaussian-approx 90% CI
    last_z() -> standardized innovation (diagnostic)
    last_trace_P() -> current variance of the V posterior
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
from examples.proof_of_concept.signature_features import RecurrentSignatureMap


# ==========================================================================
# Pure heteroskedastic Kalman on V with CIR dynamics
# ==========================================================================


@dataclass
class HeteroKalmanConfig:
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    V_floor: float = 1e-6
    P_init_mult: float = 10.0   # initial P = xi^2 * theta * dt * P_init_mult
    R_scale: float = 1.0        # multiplier on R_base = 2 V^2 / dt (overfudge)


class HeteroskedasticKalmanV:
    r"""Pure heteroskedastic 1D Kalman on variance V with CIR dynamics.

    No signature input.  Assumes Heston (kappa, theta, xi) are known.
    """
    name = "hetero_kalman"

    def __init__(self, dt: float, config: Optional[HeteroKalmanConfig] = None):
        self.cfg = config or HeteroKalmanConfig()
        self.dt = float(dt)
        self.V: float = float(self.cfg.theta)
        self.P: float = (
            float(self.cfg.xi) ** 2
            * float(self.cfg.theta)
            * float(dt)
            * float(self.cfg.P_init_mult)
        )
        self._last_z: float = 0.0

    def reset(self, V0: float):
        self.V = max(float(V0), self.cfg.V_floor)
        self.P = (
            self.cfg.xi ** 2 * self.cfg.theta * self.dt * self.cfg.P_init_mult
        )
        self._last_z = 0.0

    def _effective_R(self, V_pred: float, dt: float) -> float:
        R_base = 2.0 * max(V_pred, self.cfg.V_floor) ** 2 / dt
        return self.cfg.R_scale * R_base

    def observe(self, r_t: float, dt: float) -> None:
        # Predict
        V_pred = self.V + self.cfg.kappa * (self.cfg.theta - self.V) * dt
        V_pred = max(V_pred, self.cfg.V_floor)
        P_pred = (
            (1.0 - self.cfg.kappa * dt) ** 2 * self.P
            + self.cfg.xi ** 2 * max(self.V, self.cfg.V_floor) * dt
        )
        # Observation
        y = float(r_t) ** 2 / float(dt)
        R_t = self._effective_R(V_pred, dt)
        # Innovation and standardized z-score
        innov = y - V_pred
        S = P_pred + R_t
        self._last_z = float(innov / np.sqrt(max(S, 1e-18)))
        # Kalman update
        K = P_pred / max(S, 1e-18)
        self.V = max(V_pred + K * innov, self.cfg.V_floor)
        self.P = (1.0 - K) * P_pred

    def V_hat(self) -> float:
        return float(self.V)

    def V_interval(self) -> Tuple[float, float]:
        sd = float(np.sqrt(max(self.P, 1e-18)))
        return (
            float(max(self.V - 1.645 * sd, 0.0)),
            float(self.V + 1.645 * sd),
        )

    def last_z(self) -> float:
        return float(self._last_z)

    def last_trace_P(self) -> float:
        return float(self.P)


# ==========================================================================
# Hybrid: heteroskedastic Kalman + signature-conditioned R modulator
# ==========================================================================


@dataclass
class HybridKalmanSigConfig:
    base: HeteroKalmanConfig = None
    sig_input_dim: int = 2
    sig_level: int = 2
    sig_forget: float = 0.94
    sig_score_ewma_halflife_steps: float = 50.0
    R_modifier_exponent: float = 1.0      # higher -> more aggressive modulation
    R_min_mult: float = 0.5               # clip on modifier (when score is low)
    R_max_mult: float = 2.0               # clip on modifier (when score is high)


class HybridKalmanSigV:
    r"""Heteroskedastic Kalman on V + signature-conditioned observation noise.

    The signature scores current path volatility via the norm of level-2
    components and compares to an EWMA baseline.  When ratio_t (current /
    baseline) is elevated, R_t is inflated; when suppressed, R_t shrinks.
    Clipping keeps the modifier in [R_min_mult, R_max_mult].

    Degenerate safety: if sig_score stays near baseline, ratio_t ≈ 1 and
    the filter behaves exactly like HeteroskedasticKalmanV.
    """
    name = "hybrid_sig_kalman"

    def __init__(self, dt: float, config: Optional[HybridKalmanSigConfig] = None):
        self.cfg = config or HybridKalmanSigConfig()
        if self.cfg.base is None:
            self.cfg = HybridKalmanSigConfig(
                base=HeteroKalmanConfig(),
                sig_input_dim=self.cfg.sig_input_dim,
                sig_level=self.cfg.sig_level,
                sig_forget=self.cfg.sig_forget,
                sig_score_ewma_halflife_steps=self.cfg.sig_score_ewma_halflife_steps,
                R_modifier_exponent=self.cfg.R_modifier_exponent,
                R_min_mult=self.cfg.R_min_mult,
                R_max_mult=self.cfg.R_max_mult,
            )
        self.dt = float(dt)
        self._kalman = HeteroskedasticKalmanV(dt=dt, config=self.cfg.base)
        self.sig_map = RecurrentSignatureMap(
            state_dim=self.cfg.sig_input_dim,
            level=self.cfg.sig_level,
            forgetting_factor=self.cfg.sig_forget,
        )
        self._score_ewma = 1.0     # will stabilize after warm-up
        self._score_alpha = 1.0 - float(
            np.exp(-np.log(2.0) / max(self.cfg.sig_score_ewma_halflife_steps, 1e-3))
        )
        self._last_modifier: float = 1.0

    def reset(self, V0: float):
        self._kalman.reset(V0)
        self.sig_map.reset()
        self._score_ewma = 1.0
        self._last_modifier = 1.0

    def _sig_score(self) -> float:
        # Norm of level-2 components: captures how "active" the recent path is.
        s2 = self.sig_map.s2
        return float(np.linalg.norm(s2)) + 1e-12

    def observe(self, r_t: float, dt: float) -> None:
        # 1. Update signature first (uses pre-step path).
        dx = np.array([dt, float(r_t)])
        self.sig_map.update(dx)
        score = self._sig_score()

        # 2. Update EWMA of the score to form a baseline.
        self._score_ewma = (
            (1.0 - self._score_alpha) * self._score_ewma
            + self._score_alpha * score
        )
        ratio = score / max(self._score_ewma, 1e-12)
        modifier = float(
            np.clip(
                ratio ** self.cfg.R_modifier_exponent,
                self.cfg.R_min_mult,
                self.cfg.R_max_mult,
            )
        )
        self._last_modifier = modifier

        # 3. Temporarily scale the Kalman's R via cfg.R_scale and step it.
        #    We restore R_scale afterward so the base config is preserved.
        original_R_scale = self._kalman.cfg.R_scale
        self._kalman.cfg = HeteroKalmanConfig(
            kappa=self._kalman.cfg.kappa,
            theta=self._kalman.cfg.theta,
            xi=self._kalman.cfg.xi,
            V_floor=self._kalman.cfg.V_floor,
            P_init_mult=self._kalman.cfg.P_init_mult,
            R_scale=original_R_scale * modifier,
        )
        self._kalman.observe(r_t, dt)
        # Restore original R_scale on the underlying filter.
        self._kalman.cfg = HeteroKalmanConfig(
            kappa=self._kalman.cfg.kappa,
            theta=self._kalman.cfg.theta,
            xi=self._kalman.cfg.xi,
            V_floor=self._kalman.cfg.V_floor,
            P_init_mult=self._kalman.cfg.P_init_mult,
            R_scale=original_R_scale,
        )

    def V_hat(self) -> float:
        return self._kalman.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self._kalman.V_interval()

    def last_z(self) -> float:
        return self._kalman.last_z()

    def last_trace_P(self) -> float:
        return self._kalman.last_trace_P()

    def last_R_modifier(self) -> float:
        return float(self._last_modifier)
