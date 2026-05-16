r"""
ABLATION: SDRE+Itô misspecification × filter × DGP, with principled handling
of clips/floors.

Goal
----
Reproduce and extend the historical Level-4 graduated-sanity-checks
result on misspecification, but with two improvements over the original:

  1.  Add the missing **dollar-outcome metric** (paired Δ log W and
      certainty-equivalent CE).  The historical experiment only reported
      filter and policy correlations.

  2.  Replace the historical action clip `pi = clip(-b/(2c), 0.01, 5.0)`
      with a **principled V-floor** treatment: bound V̂ from below at a
      "minimum credible variance" V_floor derived from a Bayesian prior
      (e.g., 5% annual vol → V_floor = 0.0025).  This implicitly caps
      π at `(μ−r)/(γ·V_floor)` without a separate action clip.

The policy here is exact SDRE+Itô:
    b = U'(W)·W·(μ − r)
    c = ½·U''(W)·W²·V̂
    π* = -b / (2c)
which for CRRA collapses to plug-in Merton:
    π* = (μ − r) / (γ · max(V̂, V_floor)).

No regression, no backward DP — purely Itô one-step optimal.

Cells = 3 filters × 2 DGPs:
  filters : oracle (cheats with true V) | sig (LeadLagBLRKF) | kalman (HeteroKalman)
  DGPs    : Heston (CIR true)           | CEV β=0.5 (CIR misspec; matches historical)

Reported metrics per cell:
  - paired Δ log W vs oracle myopic (CRN)
  - certainty-equivalent wealth CE = (E[W^(1-γ)])^(1/(1-γ))
  - filter corr / RMSE on V̂ vs latent V (post warm-up)
  - π statistics: mean, max, fraction at V_floor saturation

Pre-registered reading
----------------------
1. On Heston, all filters should have V_corr roughly matching the
   historical numbers (Kalman ~0.68, BLR+KF ~0.78).
2. On CEV β=0.5 with σ₀=0.3, the historical numbers are Kalman V_corr
   ~0.63 / π_corr ~0.74 vs BLR+KF V_corr ~0.77 / π_corr ~0.91.
3. The NEW question: does CE_BLR+KF > CE_Kalman on CEV?  If yes, the
   filter advantage translates into wealth.  If no, the filter+control
   correlation gap is dollar-neutral at this configuration.
4. V_floor saturation rate per cell tells us how often the policy hits
   the implicit cap; high saturation means V̂ collapses and we are
   relying on the floor for stability.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv, CEVEnv
from merton_value_gradient import _paired_noise, OracleVEstimator
from src.sskf.leadlag_blr_kf import LeadLagBLRKFConfig, LeadLagBLRKFilter
from src.sskf.heteroskedastic_kalman_v import (
    HeteroKalmanConfig, HeteroskedasticKalmanV,
)
from examples.proof_of_concept.signature_features import (
    RecurrentLeadLagLogSigMap,
)
from src.sskf.gp_multiscale_outer import (
    TwoFactorOUConfig,
    TwoFactorOUKalmanV,
    calibrate_two_factor_ou,
)


# ==========================================================================
# FullyModelFreeSigFilter: BLR + BPV target + random-walk outer dynamics
# --------------------------------------------------------------------------
# A signature lane WITHOUT any CIR commitment.  Reuses the lead-lag log-sig
# state and 3-feature BLR design from `LeadLagBLRKFilter`, but:
#   - target = BPV per step = (pi/2)*|r_{t-1}|*|r_t|/dt
#     (jump-robust + holds for any continuous Ito process; works under
#     Heston, CEV, Bates, GBM, ...)
#   - outer dynamics = local-level random walk: V_{t+1} = V_t + eps_t,
#     eps ~ N(0, q_proc)
#     (no CIR mean reversion, no kappa, no theta, no xi)
#
# The only hyperparameter on the dynamics side is `q_proc`, a smoothness
# prior on how fast V can change.  This is a 1-param replacement for the
# 3-param CIR model.  Compare with `LeadLagBLRKFilter` which has all of
# kf_kappa, kf_theta, kf_xi -- these all encode CIR-specific structure
# that misfires under non-CIR DGPs.
#
# Numerical: V_floor at observation step prevents negative V̂ that random
# walk can produce; the analogous CIR positivity comes for free from the
# sqrt(V) diffusion term, which random walk lacks.
# ==========================================================================


class FullyModelFreeSigFilter:
    r"""Sig lane with BPV target and random-walk outer dynamics.  No CIR.

    Lane API matches `LeadLagBLRKFilter`: reset(V0), observe(r_t, dt),
    V_hat() -> float, V_interval() -> (lo, hi).
    """
    name = "sig_full"

    def __init__(
        self, dt: float,
        ll_gamma: float = 0.99,
        prior_w_var: float = 10.0,
        sigma_n2_init: float = 0.01,
        sigma_n2_alpha: float = 0.01,
        q_proc: float = 1e-5,                    # random-walk process noise
        V_init: float = 0.04,
        V_floor: float = 1e-8,
        QV_idx_l1l2: int = 8,                    # ret_lead × ret_lag pair
        ret_lead_idx: int = 1,
    ):
        self.dt = float(dt)
        self.q_proc = float(q_proc)
        self.V_floor = float(V_floor)
        self.QV_idx = QV_idx_l1l2
        self.ret_lead_idx = ret_lead_idx
        self.ll_sig = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=ll_gamma,
        )
        self._w = np.zeros(3)
        self._P = np.eye(3) * float(prior_w_var)
        self._sigma_n2 = float(sigma_n2_init)
        self._sigma_n2_alpha = float(sigma_n2_alpha)
        self._sigma_n2_init = float(sigma_n2_init)
        self._prior_w_var = float(prior_w_var)
        self._V = float(V_init)
        self._P_outer = float(q_proc) * 100.0      # initial outer Kalman uncertainty
        self._prev_abs_r = None
        self._last_z = 0.0

    def reset(self, V0: float) -> None:
        self._w = np.zeros(3)
        self._P = np.eye(3) * self._prior_w_var
        self._sigma_n2 = self._sigma_n2_init
        self.ll_sig.reset()
        self._V = float(V0)
        self._P_outer = self.q_proc * 100.0
        self._prev_abs_r = None
        self._last_z = 0.0

    def _phi(self, feat: np.ndarray) -> np.ndarray:
        return np.array([
            float(feat[self.QV_idx]),
            float(feat[self.ret_lead_idx]),
            1.0,
        ])

    def observe(self, r_t: float, dt: float) -> None:
        # 1. Update lead-lag log-sig
        feat = self.ll_sig.update(np.array([float(dt), float(r_t)]))
        phi = self._phi(feat)

        # 2. BLR predictive (mean, variance)
        y_pred = float(np.dot(self._w, phi))
        y_pred = max(y_pred, 1e-8)
        R = max(float(phi @ self._P @ phi) + self._sigma_n2, 1e-8)

        # 3. Outer Kalman with RANDOM-WALK dynamics (no CIR)
        #      V_pred = V_prev      (no drift; no mean reversion)
        #      P_pred = P_prev + q_proc
        V_pred = self._V
        P_pred = self._P_outer + self.q_proc
        S = P_pred + R
        K = P_pred / max(S, 1e-18)
        innov = y_pred - V_pred
        self._last_z = float(innov / np.sqrt(max(S, 1e-18)))
        self._V = max(V_pred + K * innov, self.V_floor)
        self._P_outer = (1.0 - K) * P_pred

        # 4. BLR posterior update on BPV target  (jump-robust, model-free)
        abs_r = float(np.abs(r_t))
        if self._prev_abs_r is None:
            target = (r_t * r_t) / dt                # warm-up step: r²/dt fallback
        else:
            target = (np.pi / 2.0) * float(self._prev_abs_r) * abs_r / dt
        Cp = self._P @ phi
        S_w = float(phi @ Cp) + self._sigma_n2
        K_w = Cp / max(S_w, 1e-18)
        resid = target - float(np.dot(self._w, phi))
        self._w = self._w + K_w * resid
        self._P = self._P - np.outer(K_w, Cp)
        self._P = 0.5 * (self._P + self._P.T)
        self._sigma_n2 = max(
            (1.0 - self._sigma_n2_alpha) * self._sigma_n2
            + self._sigma_n2_alpha * (resid * resid),
            1e-8,
        )
        self._prev_abs_r = abs_r

    def V_hat(self) -> float:
        return float(self._V)

    def V_interval(self) -> Tuple[float, float]:
        sd = float(np.sqrt(max(self._P_outer, 1e-18)))
        return (
            float(max(self._V - 1.645 * sd, 0.0)),
            float(self._V + 1.645 * sd),
        )


# ==========================================================================
# Sig + BPV target + 2-factor OU outer (data-driven multiscale GP prior)
# ==========================================================================


class GPMultiscaleSigFilter:
    r"""Sig lane with BPV target and 2-factor OU outer dynamics.

    The outer prior is a sum-of-2-Matérn-1/2 GP on V (state-space form),
    calibrated per-DGP from a pilot RV ACF.  This is the principled
    multiscale-mean-reversion alternative to CIR.
    """
    name = "sig_gp"

    def __init__(
        self, dt: float, two_factor_cfg: TwoFactorOUConfig,
        ll_gamma: float = 0.99,
        prior_w_var: float = 10.0,
        sigma_n2_init: float = 0.01,
        sigma_n2_alpha: float = 0.01,
        QV_idx_l1l2: int = 8,
        ret_lead_idx: int = 1,
    ):
        self.dt = float(dt)
        self.QV_idx = QV_idx_l1l2
        self.ret_lead_idx = ret_lead_idx
        self.ll_sig = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=ll_gamma,
        )
        self._w = np.zeros(3)
        self._P = np.eye(3) * float(prior_w_var)
        self._sigma_n2 = float(sigma_n2_init)
        self._sigma_n2_alpha = float(sigma_n2_alpha)
        self._sigma_n2_init = float(sigma_n2_init)
        self._prior_w_var = float(prior_w_var)
        self.outer = TwoFactorOUKalmanV(dt=dt, config=two_factor_cfg)
        self._prev_abs_r: Optional[float] = None

    def reset(self, V0: float) -> None:
        self._w = np.zeros(3)
        self._P = np.eye(3) * self._prior_w_var
        self._sigma_n2 = self._sigma_n2_init
        self.ll_sig.reset()
        self.outer.reset(V0)
        self._prev_abs_r = None

    def _phi(self, feat: np.ndarray) -> np.ndarray:
        return np.array([
            float(feat[self.QV_idx]),
            float(feat[self.ret_lead_idx]),
            1.0,
        ])

    def observe(self, r_t: float, dt: float) -> None:
        # 1. Update lead-lag log-sig
        feat = self.ll_sig.update(np.array([float(dt), float(r_t)]))
        phi = self._phi(feat)

        # 2. BLR predictive (mean, variance)
        y_pred = float(np.dot(self._w, phi))
        y_pred = max(y_pred, 1e-8)
        R = max(float(phi @ self._P @ phi) + self._sigma_n2, 1e-8)

        # 3. Outer 2-factor-OU Kalman: predict + absorb
        V_pred, P_V, x_pred, P_pred_outer = self.outer.predict()
        V_new, _z = self.outer.absorb(V_pred, P_V, x_pred, P_pred_outer, y_pred, R)

        # 4. BLR posterior update on BPV target
        abs_r = float(np.abs(r_t))
        if self._prev_abs_r is None:
            target = (r_t * r_t) / dt
        else:
            target = (np.pi / 2.0) * float(self._prev_abs_r) * abs_r / dt
        Cp = self._P @ phi
        S_w = float(phi @ Cp) + self._sigma_n2
        K_w = Cp / max(S_w, 1e-18)
        resid = target - float(np.dot(self._w, phi))
        self._w = self._w + K_w * resid
        self._P = self._P - np.outer(K_w, Cp)
        self._P = 0.5 * (self._P + self._P.T)
        self._sigma_n2 = max(
            (1.0 - self._sigma_n2_alpha) * self._sigma_n2
            + self._sigma_n2_alpha * (resid * resid),
            1e-8,
        )
        self._prev_abs_r = abs_r

    def V_hat(self) -> float:
        return self.outer.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.outer.V_interval()


# ==========================================================================
# PureOnlineBayesianSigFilter -- BLR + BPV target + NO outer dynamics layer
#
# Philosophy: calibration is a smell.  A properly-Bayesian filter has
# conjugate online updates; it should not require a separate "fit
# parameters on a pilot" step.  The lead-lag log-sig features already
# encode temporal smoothing through their recurrent updates -- they make
# the BLR predictive mean a smooth function of the past 100 days.  So we
# drop the outer Kalman entirely.
#
# What's left is a single Bayesian model: a BLR with conjugate Normal-
# Inverse-Gamma updates.  Every quantity in this filter is either:
#   - a vague prior (chosen once from Bayesian intuition; not DGP-specific)
#   - a posterior, updated online from data
# No calibration step, no fixed θ/κ/ξ, no outer dynamics commitment.
#
# V̂(t) = phi_t · w_t.  Posterior variance phi_t · P_t · phi_t + σ_n²(t).
# That's the entire filter.
# ==========================================================================


class PureOnlineBayesianSigFilter:
    r"""BLR + BPV target.  No outer Kalman.  No calibration."""
    name = "sig_pure"

    def __init__(
        self, dt: float,
        ll_gamma: float = 0.99,
        prior_w_var: float = 10.0,
        sigma_n2_init: float = 0.01,
        sigma_n2_alpha: float = 0.01,
        V_floor: float = 1e-8,
        QV_idx_l1l2: int = 8,
        ret_lead_idx: int = 1,
    ):
        self.dt = float(dt)
        self.QV_idx = QV_idx_l1l2
        self.ret_lead_idx = ret_lead_idx
        self.V_floor = float(V_floor)
        self.ll_sig = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=ll_gamma,
        )
        self._w = np.zeros(3)
        self._P = np.eye(3) * float(prior_w_var)
        self._sigma_n2 = float(sigma_n2_init)
        self._sigma_n2_alpha = float(sigma_n2_alpha)
        self._sigma_n2_init = float(sigma_n2_init)
        self._prior_w_var = float(prior_w_var)
        self._V_cache = float(prior_w_var * 0 + 0.04)              # initial guess
        self._R_cache = float(prior_w_var)                         # initial uncertainty
        self._prev_abs_r: Optional[float] = None

    def reset(self, V0: float) -> None:
        self._w = np.zeros(3)
        self._P = np.eye(3) * self._prior_w_var
        self._sigma_n2 = self._sigma_n2_init
        self.ll_sig.reset()
        self._V_cache = float(V0)
        self._R_cache = self._prior_w_var
        self._prev_abs_r = None

    def _phi(self, feat: np.ndarray) -> np.ndarray:
        return np.array([
            float(feat[self.QV_idx]),
            float(feat[self.ret_lead_idx]),
            1.0,
        ])

    def observe(self, r_t: float, dt: float) -> None:
        # 1. Update lead-lag log-sig features
        feat = self.ll_sig.update(np.array([float(dt), float(r_t)]))
        phi = self._phi(feat)

        # 2. BPV-per-step target (jump-robust, model-free)
        abs_r = float(np.abs(r_t))
        if self._prev_abs_r is None:
            target = (r_t * r_t) / dt                            # warm-up: r²/dt
        else:
            target = (np.pi / 2.0) * float(self._prev_abs_r) * abs_r / dt

        # 3. BLR conjugate posterior update on (target | phi)
        Cp = self._P @ phi
        S = float(phi @ Cp) + self._sigma_n2
        K = Cp / max(S, 1e-18)
        resid = target - float(np.dot(self._w, phi))
        self._w = self._w + K * resid
        self._P = self._P - np.outer(K, Cp)
        self._P = 0.5 * (self._P + self._P.T)
        self._sigma_n2 = max(
            (1.0 - self._sigma_n2_alpha) * self._sigma_n2
            + self._sigma_n2_alpha * (resid * resid),
            1e-8,
        )

        # 4. Cache predictive mean and variance for V_hat / V_interval
        y_post = float(np.dot(self._w, phi))
        self._V_cache = max(y_post, self.V_floor)
        self._R_cache = max(float(phi @ self._P @ phi) + self._sigma_n2, 1e-12)

        self._prev_abs_r = abs_r

    def V_hat(self) -> float:
        return float(self._V_cache)

    def V_interval(self) -> Tuple[float, float]:
        sd = float(np.sqrt(max(self._R_cache, 1e-18)))
        return (
            float(max(self._V_cache - 1.645 * sd, 0.0)),
            float(self._V_cache + 1.645 * sd),
        )


# --------------------------------------------------------------------------
# Principled V_floor.  Single hyperparameter, Bayesian-prior-derived.
#
# Interpretation: we believe variance is at least V_FLOOR = (min_vol)^2.
# With min_vol = 5% annualized, V_FLOOR = 0.0025.  This corresponds to a
# truncated prior on V with support [V_FLOOR, ∞).  In the SDRE+Itô plug-in
# Merton policy π* = (μ−r) / (γ·V̂), this caps π at (μ−r)/(γ·V_FLOOR)
# without a separate action clip:
#
#   For γ=3, μ-r=0.06, V_FLOOR=0.0025: max π = 0.06/(3·0.0025) = 8.0.
#
# Documented principled alternatives we did NOT build:
#   - CVaR-constrained policy (computationally expensive)
#   - Trust region |π_t − π_{t-1}| ≤ ε (transaction-cost-style)
#   - Posterior-uncertainty-discounted plug-in: π* = (μ−r) E[1/V] / γ
#     using the filter's full posterior on V (not just the mean).
# --------------------------------------------------------------------------
V_FLOOR_PRIOR = 0.0025                 # = (5% min annualized vol)^2

# Numerical-safety floors (not principled; just to avoid sqrt(0) etc.)
NUMERICAL_EPS = 1e-12
V_FLOOR_NUMERICAL = 1e-8                # only for sqrt() inside Brownian terms


# ==========================================================================
# Filter constructors -- always assume Heston-CIR(2.0, 0.04, 0.3)
# ==========================================================================


# Module-level dict, populated by calibrate_for_dgp() before lane construction.
_CALIBRATED_TWO_FACTOR_OU: Dict[str, TwoFactorOUConfig] = {}
# DGP name selected at construction time (dict key into above).
_CURRENT_DGP_KEY: List[str] = ["heston"]


def make_filter(name: str, dt: float):
    if name == "oracle":
        return OracleVEstimator()
    if name == "sig":
        cfg = LeadLagBLRKFConfig(
            ll_gamma=0.99,
            prior_w_var=10.0, sigma_n2_init=0.01, sigma_n2_alpha=0.01,
            target_clip=None,                                # principled: drop the heuristic
            kf_kappa=2.0, kf_theta=0.04, kf_xi=0.3,         # Heston-assumed CIR (FIXED)
            V_floor=NUMERICAL_EPS, P_init_mult=10.0,
        )
        return LeadLagBLRKFilter(dt=dt, config=cfg)
    if name == "kalman":
        cfg = HeteroKalmanConfig(
            kappa=2.0, theta=0.04, xi=0.3,                  # Heston-assumed CIR (FIXED)
            V_floor=NUMERICAL_EPS, P_init_mult=10.0, R_scale=1.0,
        )
        return HeteroskedasticKalmanV(dt=dt, config=cfg)
    if name == "sig_full":
        # Fully model-free: BLR + BPV target + random-walk outer dynamics.
        # No CIR commitment anywhere.  q_proc tuned to the same scale as
        # CIR's typical step variance (xi^2 * theta * dt = 0.09*0.04*1/252
        # ≈ 1.4e-5) -- a fair smoothness prior, not DGP-specific.
        return FullyModelFreeSigFilter(
            dt=dt, ll_gamma=0.99,
            prior_w_var=10.0, sigma_n2_init=0.01, sigma_n2_alpha=0.01,
            q_proc=1.4e-5, V_init=0.04, V_floor=NUMERICAL_EPS,
        )
    if name == "sig_pure":
        # Fully online Bayesian: BLR + BPV target, NO outer dynamics layer.
        # No calibration, no fixed θ/κ/ξ.  V̂ = BLR predictive mean.
        return PureOnlineBayesianSigFilter(
            dt=dt, ll_gamma=0.99,
            prior_w_var=10.0, sigma_n2_init=0.01, sigma_n2_alpha=0.01,
            V_floor=NUMERICAL_EPS,
        )
    if name == "sig_gp":
        # Multiscale GP outer prior (2-factor OU), data-driven from pilot.
        dgp_key = _CURRENT_DGP_KEY[0]
        if dgp_key not in _CALIBRATED_TWO_FACTOR_OU:
            raise RuntimeError(
                f"sig_gp requested but no calibrated 2-factor-OU config for DGP {dgp_key!r}. "
                "Call calibrate_for_dgp(...) first."
            )
        return GPMultiscaleSigFilter(
            dt=dt,
            two_factor_cfg=_CALIBRATED_TWO_FACTOR_OU[dgp_key],
            ll_gamma=0.99,
            prior_w_var=10.0, sigma_n2_init=0.01, sigma_n2_alpha=0.01,
        )
    raise ValueError(f"unknown filter {name!r}")


# ==========================================================================
# SDRE+Itô rollout (one-step plug-in Merton at filter V̂)
# ==========================================================================


@dataclass
class SDREConfig:
    T_steps: int = 252                              # 1 trading year
    dt: float = 1.0 / 252.0
    gamma: float = 3.0                              # CRRA risk aversion
    mu: float = 0.08
    r: float = 0.02
    rho: float = -0.7
    V0_low: float = 0.02
    V0_high: float = 0.08
    V_floor: float = V_FLOOR_PRIOR                  # principled prior-derived floor


def sdre_ito_rollout(
    env, cfg: SDREConfig, V0: float, noise: np.ndarray, filter_,
) -> Dict[str, np.ndarray]:
    r"""SDRE+Itô one-step plug-in Merton rollout.

    Policy at each step:
        π_t = (μ − r) / (γ · max(V̂_t, V_floor))
    No clip.  Wealth update is multiplicative on TRUE V (env evolves with
    its own dynamics; the filter sees dr_S and updates).
    """
    logW = 0.0
    V_state = float(V0)
    filter_.reset(V0)
    if isinstance(filter_, OracleVEstimator):
        filter_.set_true_V(V_state)
    T = cfg.T_steps
    pi_hist = np.zeros(T)
    V_hat_hist = np.zeros(T)
    V_true_hist = np.zeros(T)
    saturated = np.zeros(T, dtype=bool)
    for t in range(T):
        V_hat = filter_.V_hat()
        V_hat_eff = max(V_hat, cfg.V_floor)
        saturated[t] = V_hat < cfg.V_floor
        pi = (cfg.mu - cfg.r) / (cfg.gamma * V_hat_eff)
        pi_hist[t] = pi
        V_hat_hist[t] = V_hat
        V_true_hist[t] = V_state
        # Step env on TRUE V state
        new_logW, new_V = env.step_explicit(
            logW, V_state, pi, float(noise[t, 0]), float(noise[t, 1]), cfg.dt,
        )
        # Compute dr_S (asset log-return) from TRUE V_pre and z1, for filter consumption
        sqrt_V_true = float(np.sqrt(max(V_state, V_FLOOR_NUMERICAL)))
        dr_S = (cfg.mu - 0.5 * V_state) * cfg.dt + sqrt_V_true * float(np.sqrt(cfg.dt)) * float(noise[t, 0])
        # Filter update
        if isinstance(filter_, OracleVEstimator):
            filter_.set_true_V(new_V)
        else:
            filter_.observe(dr_S, cfg.dt)
        logW = new_logW
        V_state = new_V
    return {
        "logW_T":     float(logW),
        "pi_hist":    pi_hist,
        "V_hat_hist": V_hat_hist,
        "V_true_hist":V_true_hist,
        "saturated":  saturated,
    }


# ==========================================================================
# Paired CRN evaluation
# ==========================================================================


def evaluate_cell(
    env_factory: Callable[[], object], cfg: SDREConfig,
    filter_name: str, n_test: int, base_seed: int,
    dgp_key: str = "heston",
) -> Dict[str, np.ndarray]:
    r"""For each test seed, run TWO rollouts on the same Brownian noise:
      A. Oracle policy: π = (μ-r)/(γ V_true)  -- baseline
      B. Filter policy: π = (μ-r)/(γ max(V̂, V_floor))  -- candidate

    Returns paired ΔlogW = logW_filter - logW_oracle, plus diagnostics.
    """
    _CURRENT_DGP_KEY[0] = dgp_key                                # for sig_gp lane
    delta_logW = np.zeros(n_test)
    logW_oracle = np.zeros(n_test)
    logW_filter = np.zeros(n_test)
    sat_frac = np.zeros(n_test)
    pi_means = np.zeros(n_test)
    pi_maxs = np.zeros(n_test)
    V_hat_corr = np.zeros(n_test)
    V_hat_rmse = np.zeros(n_test)
    pi_corr = np.zeros(n_test)
    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_test):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        noise = _paired_noise(cfg.rho, cfg.T_steps, base_seed + 1 + k)
        # A. Oracle policy (baseline -- always uses true V)
        env_a = env_factory()
        oracle_a = OracleVEstimator()
        out_a = sdre_ito_rollout(env_a, cfg, V0, noise, oracle_a)
        # B. Filter policy
        env_b = env_factory()
        filter_b = make_filter(filter_name, cfg.dt)
        out_b = sdre_ito_rollout(env_b, cfg, V0, noise, filter_b)
        delta_logW[k] = out_b["logW_T"] - out_a["logW_T"]
        logW_oracle[k] = out_a["logW_T"]
        logW_filter[k] = out_b["logW_T"]
        sat_frac[k] = float(np.mean(out_b["saturated"]))
        pi_means[k] = float(np.mean(out_b["pi_hist"]))
        pi_maxs[k] = float(np.max(out_b["pi_hist"]))
        # Filter quality (post-warm-up)
        warmup = max(cfg.T_steps // 5, 20)
        Vh = out_b["V_hat_hist"][warmup:]
        Vt = out_b["V_true_hist"][warmup:]
        if Vh.size >= 3 and np.isfinite(Vh).all() and np.isfinite(Vt).all():
            V_hat_rmse[k] = float(np.sqrt(np.mean((Vh - Vt) ** 2)))
            if filter_name == "oracle":
                V_hat_corr[k] = 1.0
            else:
                V_hat_corr[k] = float(np.corrcoef(Vh, Vt)[0, 1])
        # Policy correlation (pi_filter vs pi_oracle, both UNCLIPPED)
        pi_a = out_a["pi_hist"][warmup:]
        pi_b = out_b["pi_hist"][warmup:]
        if pi_a.size >= 3:
            pi_corr[k] = float(np.corrcoef(pi_a, pi_b)[0, 1])
    return {
        "delta_logW":  delta_logW,
        "logW_oracle": logW_oracle,
        "logW_filter": logW_filter,
        "sat_frac":    sat_frac,
        "pi_mean":     pi_means,
        "pi_max":      pi_maxs,
        "V_hat_corr":  V_hat_corr,
        "V_hat_rmse":  V_hat_rmse,
        "pi_corr":     pi_corr,
    }


# ==========================================================================
# Metric: certainty-equivalent wealth (exact CRRA, no lognormal Taylor)
# ==========================================================================


def certainty_equivalent(logW_terminal: np.ndarray, gamma: float) -> float:
    r"""CE = (E[W^(1-γ)])^(1/(1-γ)).  Exact for CRRA utility."""
    W = np.exp(np.asarray(logW_terminal, dtype=float))
    if abs(gamma - 1.0) < 1e-9:
        # Log utility limit
        return float(np.exp(np.mean(np.log(W))))
    one_minus_g = 1.0 - gamma
    expected_util = float(np.mean(W ** one_minus_g))
    return float(expected_util ** (1.0 / one_minus_g))


def bootstrap_ce_diff(
    logW_b: np.ndarray, logW_a: np.ndarray, gamma: float,
    n_boot: int = 2000, seed: int = 17,
) -> Dict[str, float]:
    r"""Bootstrap CI for CE(logW_b) − CE(logW_a) via paired resampling."""
    rng = np.random.RandomState(seed)
    n = logW_b.size
    boot = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[i] = certainty_equivalent(logW_b[idx], gamma) - certainty_equivalent(logW_a[idx], gamma)
    point = certainty_equivalent(logW_b, gamma) - certainty_equivalent(logW_a, gamma)
    return {
        "point": float(point),
        "se":    float(np.std(boot, ddof=1)),
        "q05":   float(np.quantile(boot, 0.05)),
        "q95":   float(np.quantile(boot, 0.95)),
    }


# ==========================================================================
# Reporting
# ==========================================================================


def print_per_cell(results: Dict[Tuple[str, str], Dict], cfg: SDREConfig) -> None:
    print()
    print("Per-cell results  (paired CRN; filter policy vs oracle policy on same noise)")
    print("-" * 132)
    print(f"  {'DGP':>8s} | {'filter':>7s} | {'CE_filter − CE_oracle':>26s} | "
          f"{'mean ΔlogW':>11s} | {'corr V̂':>7s} | {'rmse V̂':>7s} | "
          f"{'corr π':>7s} | {'sat %':>6s} | {'mean π':>7s} | {'max π':>6s}")
    print("-" * 132)
    for (dgp, fname), r in results.items():
        ce_diff = bootstrap_ce_diff(r["logW_filter"], r["logW_oracle"], cfg.gamma)
        cs = f"{ce_diff['point']:+.5f} [{ce_diff['q05']:+.5f}, {ce_diff['q95']:+.5f}]"
        print(f"  {dgp:>8s} | {fname:>7s} | {cs:>26s} | "
              f"{np.mean(r['delta_logW']):+.5f}    | "
              f"{np.mean(r['V_hat_corr']):+.3f}  | "
              f"{np.mean(r['V_hat_rmse']):.4f}  | "
              f"{np.mean(r['pi_corr']):+.3f}  | "
              f"{100*np.mean(r['sat_frac']):.2f}  | "
              f"{np.mean(r['pi_mean']):.3f}   | "
              f"{np.mean(r['pi_max']):.3f}")
    print()


def print_misspec_summary(results: Dict[Tuple[str, str], Dict], cfg: SDREConfig) -> None:
    print("Misspecification damage  Δ_misspec(filter) := CE(Heston) − CE(CEV)")
    print("  (CE_filter − CE_oracle is the score per cell)")
    print("-" * 96)
    print(f"  {'filter':>8s} | {'CE-Δ on Heston':>16s} | {'CE-Δ on CEV β=0.5':>18s} | "
          f"{'Δ_misspec':>12s}")
    print("-" * 96)
    for f in ["oracle", "sig", "sig_pure", "sig_full", "sig_gp", "kalman"]:
        rh = results.get(("heston", f)); rc = results.get(("cev", f))
        if rh is None or rc is None: continue
        h = bootstrap_ce_diff(rh["logW_filter"], rh["logW_oracle"], cfg.gamma)["point"]
        c = bootstrap_ce_diff(rc["logW_filter"], rc["logW_oracle"], cfg.gamma)["point"]
        print(f"  {f:>8s} | {h:+.5f}        | {c:+.5f}            | {h-c:+.5f}")
    print()
    print("Filter-axis comparisons within each DGP (CE-Δ differences)")
    print("-" * 96)
    print(f"  {'DGP':>8s} | {'sig − oracle':>14s} | {'kalman − oracle':>16s} | "
          f"{'sig − kalman':>14s}")
    print("-" * 96)
    for dgp in ["heston", "cev"]:
        ro = results.get((dgp, "oracle")); rs = results.get((dgp, "sig"))
        rsf = results.get((dgp, "sig_full")); rk = results.get((dgp, "kalman"))
        if any(r is None for r in (ro, rs, rsf, rk)): continue
        ce = lambda r_: bootstrap_ce_diff(r_["logW_filter"], r_["logW_oracle"], cfg.gamma)["point"]
        ceo, ces, cesf, cek = ce(ro), ce(rs), ce(rsf), ce(rk)
        print(f"  {dgp:>8s} | {ces-ceo:+.5f}        | {cek-ceo:+.5f}          | {ces-cek:+.5f}")
    print()
    # New: fully model-free vs model-committed
    print("  Fully-model-free comparison (CE-Δ differences)")
    print("-" * 96)
    print(f"  {'DGP':>8s} | {'sig_full − oracle':>18s} | {'sig_full − sig':>16s} | "
          f"{'sig_full − kalman':>18s}")
    print("-" * 96)
    for dgp in ["heston", "cev"]:
        ro = results.get((dgp, "oracle")); rs = results.get((dgp, "sig"))
        rsf = results.get((dgp, "sig_full")); rk = results.get((dgp, "kalman"))
        if any(r is None for r in (ro, rs, rsf, rk)): continue
        ce = lambda r_: bootstrap_ce_diff(r_["logW_filter"], r_["logW_oracle"], cfg.gamma)["point"]
        ceo, ces, cesf, cek = ce(ro), ce(rs), ce(rsf), ce(rk)
        print(f"  {dgp:>8s} | {cesf-ceo:+.5f}            | {cesf-ces:+.5f}        | "
              f"{cesf-cek:+.5f}")
    print()


# ==========================================================================
# Pilot calibration of the multiscale GP prior
# ==========================================================================


def calibrate_for_dgp(
    dgp_name: str, env_factory: Callable[[], object], cfg: SDREConfig,
    pilot_T: int = 2000, pilot_seed: int = 99_000,
) -> TwoFactorOUConfig:
    r"""Run a single u=0 pilot path under this DGP, compute realized
    variance, fit a 2-factor OU prior, and store under module-level dict
    for sig_gp construction.
    """
    env = env_factory()
    V0 = float(np.random.RandomState(pilot_seed).uniform(cfg.V0_low, cfg.V0_high))
    noise = _paired_noise(cfg.rho, pilot_T, pilot_seed)
    rv = np.zeros(pilot_T)
    logW = 0.0
    V = V0
    for t in range(pilot_T):
        # u=0 myopic-at-true-V is the natural pilot policy
        pi = (cfg.mu - cfg.r) / (cfg.gamma * max(V, cfg.V_floor))
        # Compute dr_S using TRUE V (same way the filter sees it)
        sqrt_V = float(np.sqrt(max(V, V_FLOOR_NUMERICAL)))
        dr_S = (cfg.mu - 0.5 * V) * cfg.dt + sqrt_V * float(np.sqrt(cfg.dt)) * float(noise[t, 0])
        rv[t] = (dr_S * dr_S) / cfg.dt
        new_logW, new_V = env.step_explicit(
            logW, V, pi, float(noise[t, 0]), float(noise[t, 1]), cfg.dt,
        )
        logW = new_logW; V = new_V
    cfg_out = calibrate_two_factor_ou(rv, dt=cfg.dt, halflife_smooth_steps=21.0)
    _CALIBRATED_TWO_FACTOR_OU[dgp_name] = cfg_out
    return cfg_out


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    cfg = SDREConfig(
        T_steps=1000, dt=1.0/252.0, gamma=3.0,                  # ~4 yr (filter warm-up matters)
        mu=0.08, r=0.02, rho=-0.7,
        V0_low=0.02, V0_high=0.08,
        V_floor=V_FLOOR_PRIOR,
    )
    n_test = 200

    print("=" * 132)
    print("SDRE+Itô misspecification ablation × filter × DGP  (principled V_floor; no policy clip)")
    print(f"  T_steps = {cfg.T_steps} (= {cfg.T_steps/252:.2f} yr)   "
          f"dt = {cfg.dt:.5f}   γ = {cfg.gamma}   (μ, r) = ({cfg.mu}, {cfg.r})   ρ = {cfg.rho}")
    print(f"  filter assumed CIR(κ=2.0, θ=0.04, ξ=0.3) — FIXED across DGPs")
    print(f"  V_floor (Bayesian prior on min variance) = {cfg.V_floor}  =  ({100*np.sqrt(cfg.V_floor):.1f}% min annual vol)²")
    print(f"  ⇒ implicit max π = (μ−r)/(γ·V_floor) = {(cfg.mu-cfg.r)/(cfg.gamma*cfg.V_floor):.2f} (no separate clip)")
    print(f"  n_test = {n_test} paired CRN seeds")
    print()
    print("  Policy: π_t = (μ−r) / (γ · max(V̂_t, V_floor))   — pure SDRE+Itô plug-in Merton")
    print("  Metric: paired Δ logW vs oracle myopic + bootstrap CI on certainty-equivalent CE-Δ")
    print("=" * 132)

    # Env factories that produce a FRESH env per rollout (so internal state resets)
    def make_heston_env_fac():
        # Use cfg parameters consistently
        return lambda: HestonMertonEnv(
            mu=cfg.mu, r=cfg.r, gamma=cfg.gamma,
            kappa=2.0, theta=0.04, xi=0.3, rho=cfg.rho,
        )
    def make_cev_env_fac():
        # CEV β=0.5, σ₀=0.3 to match historical Phase-4c setup (V≈0.09 at S=1)
        return lambda: CEVEnv(
            mu=cfg.mu, r=cfg.r, gamma=cfg.gamma,
            sigma=0.3, alpha=0.5,                                # alpha == β in CEVEnv parameterization
        )
    dgps = [
        ("heston", make_heston_env_fac()),
        ("cev",    make_cev_env_fac()),
    ]

    # ----- Per-DGP calibration of the 2-factor OU prior (pilot path) -----
    print()
    print("Pilot-calibrated multiscale GP outer prior (2-factor OU)")
    print("-" * 100)
    for dgp_name, env_fac in dgps:
        ou_cfg = calibrate_for_dgp(dgp_name, env_fac, cfg, pilot_T=2000)
        hl_f, hl_s = ou_cfg.half_lives_days(cfg.dt)
        print(f"  {dgp_name:>8s}: θ̂={ou_cfg.theta:.4f}  "
              f"κ_fast={ou_cfg.kappa_fast:6.2f}/yr (half-life {hl_f:5.1f}d)  "
              f"κ_slow={ou_cfg.kappa_slow:6.2f}/yr (half-life {hl_s:5.1f}d)  "
              f"ξ_fast={ou_cfg.xi_fast:.4f}  ξ_slow={ou_cfg.xi_slow:.4f}  "
              f"stat_var={ou_cfg.stationary_var():.5f}")
    print()

    filters = ["oracle", "sig", "sig_pure", "sig_full", "sig_gp", "kalman"]
    results: Dict[Tuple[str, str], Dict] = {}
    for dgp_name, env_fac in dgps:
        print()
        print(f"--- DGP: {dgp_name} ---")
        for fname in filters:
            print(f"  cell: {dgp_name} × {fname:>6s} ... ", end="", flush=True)
            base_seed = 30_000 + (0 if dgp_name == "heston" else 100_000)
            r = evaluate_cell(env_fac, cfg, fname, n_test=n_test, base_seed=base_seed,
                              dgp_key=dgp_name)
            results[(dgp_name, fname)] = r
            ce_d = bootstrap_ce_diff(r["logW_filter"], r["logW_oracle"], cfg.gamma)
            print(f"CE-Δ = {ce_d['point']:+.5f} [{ce_d['q05']:+.5f}, {ce_d['q95']:+.5f}]   "
                  f"corr V̂ = {np.mean(r['V_hat_corr']):+.3f}   sat = {100*np.mean(r['sat_frac']):.1f}%")

    print_per_cell(results, cfg)
    print_misspec_summary(results, cfg)


if __name__ == "__main__":
    main()
