r"""
Heston myopic-reference local value-gradient controller with SCALAR
multiplicative overlay and proper action-dependent continuation.

Control policy form
-------------------
    z_t        :=  state (logW_t, V_t, t, path features)
    pi_ref(V)  :=  (mu - r) / (gamma * V_floored)       (myopic Merton)
    pi_t       =   pi_ref(V_t) * (1 + u_t),   u_t scalar
    psi(z_t)   :=  adapter-supplied lifted features (Stage A minimal;
                                                     Stage B path-aware)

Training (paired-noise rollouts with a scalar exploration u in a
trust region [-u_max, u_max]):

    Collect (psi_t, u_t, psi_{t+1}) triples at every step t in [0, T).
    Per time step, fit a control-quadratic transition
        E[psi_{t+1} | psi_t, u]  ~  (A_0^(t) + u A_1^(t) + u^2 A_2^(t)) psi_t.

    Terminal target:  log W_T - log W_0  (paired; removes path baseline).
    Backward value recursion via src/control/local_value_gradient.py.

Decision time (on held-out seeds):

    At step t the Hamiltonian is exactly quadratic in u:
        Q_t(u)  =  beta_{t+1}^T (A_0 + u A_1 + u^2 A_2) psi_t
               =  alpha_0(z_t) + alpha_1(z_t) u + alpha_2(z_t) u^2.
    u*(z_t) = -alpha_1 / (2 alpha_2), clipped to the trust region;
    abstain (u=0) if alpha_2 >= 0 (no concave optimum).

Why this is the right route given the audit
-------------------------------------------
1. The transfer-form CQ transition model is the proper USE of the
   A_0 + u A_1 + u^2 A_2 object (as a DYNAMICS model on a compact trust
   region), not the raw-action CONTROLLER that fails per Section 3 of
   docs/theory_kronic_extrapolation.md.
2. The backward value recursion captures finite-horizon / intertemporal
   structure that the old Level-4 "SDRE" (memory: sensor-plus-myopic)
   does not.
3. The Hamiltonian's u-dependence is assembled EXPLICITLY via
   beta_{t+1}^T (A_0 + u A_1 + u^2 A_2) psi_t.  This is the step that
   src/applications/option_mm/local_value_bilinear.py punts on (see
   its lines 349-381 which reduce the optimizer back to a one-step
   stage-cost rule).  We do NOT punt here.

Honesty disclaimers
-------------------
* Terminal target is log W_T - log W_0 (paired).  This is the Kelly /
  log-utility residual; NOT the exact gamma-CRRA residual.  The held-out
  evaluation uses CRRA-style mean-variance score mean - 0.5(gamma-1) Var
  under a lognormal approximation, which matches what the residual-kernel
  adapter uses -- so the comparison is like-for-like.
* Under a FIXED uniform-exploration policy, the backward recursion fits
  V_t under that exploration, not the optimal policy's value.  The
  extracted u*(z_t) is the best one-step local improvement over the
  exploration conditional on z_t.  Over a horizon, rolling out u*(z_t)
  at every step is an APPROXIMATE dynamic programming solution -- not
  an exact Bellman solve.  This is the standard "fitted value iteration
  with a single data pass" trade-off.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv

from src.control.local_value_gradient import (
    ActionRecommendation,
    EpisodeRecord,
    ValueGradientModel,
    backward_value_iteration,
    extract_action,
    training_value_r2,
)
from src.sskf.dual_target_sig_blf import (
    DualTargetSigBLF,
    DualTargetSigBLFConfig,
)
from src.sskf.heteroskedastic_kalman_v import (
    HeteroKalmanConfig,
    HeteroskedasticKalmanV,
    HybridKalmanSigConfig,
    HybridKalmanSigV,
)
from src.sskf.leadlag_blr_kf import (
    LeadLagBLRKFConfig,
    LeadLagBLRKFilter,
)


# ==========================================================================
# V-estimator lanes (state-hat producers)
# ==========================================================================
#
# Each lane exposes an identical API:
#     reset(V0_init)      -- called at the start of every episode
#     observe(r_t, dt)    -- called with each realized log-return
#     V_hat() -> float    -- current V estimate at the just-observed step
#     V_interval() -> (float, float) -- optional 90% CI, or (nan, nan)
#     name: str
#
# Oracle uses the hidden true V (cheat; used as upper-bound baseline).
# EWMA uses a plain EWMA of r^2/dt with a fixed halflife.
# BLF uses the Bayesian dual-target signature filter (this pass's subject).


class OracleVEstimator:
    name = "oracle"

    def __init__(self):
        self._V = 0.04

    def reset(self, V0: float):
        self._V = float(V0)

    def set_true_V(self, V_true: float):
        self._V = float(V_true)

    def observe(self, r_t: float, dt: float):
        pass  # oracle reads V_true externally

    def V_hat(self) -> float:
        return float(self._V)

    def V_interval(self) -> Tuple[float, float]:
        return (float("nan"), float("nan"))


class EWMAVEstimator:
    name = "ewma"

    def __init__(self, halflife_days: float = 21.0, dt: float = 1.0 / 252.0):
        self.halflife_days = float(halflife_days)
        self.dt = float(dt)
        # EWMA on r^2/dt
        self._alpha = 1.0 - float(np.exp(-np.log(2.0) / max(halflife_days, 1e-3)))
        self._V = 0.04

    def reset(self, V0: float):
        self._V = float(V0)

    def observe(self, r_t: float, dt: float):
        y = float(r_t) ** 2 / float(dt)
        self._V = (1.0 - self._alpha) * self._V + self._alpha * y
        self._V = max(self._V, 1e-8)

    def V_hat(self) -> float:
        return float(self._V)

    def V_interval(self) -> Tuple[float, float]:
        return (float("nan"), float("nan"))


class HeteroKalmanVEstimator:
    r"""Pure heteroskedastic Kalman lane (NO signature).  Known CIR params."""
    name = "hetero_kalman"

    def __init__(self, env: HestonMertonEnv, dt: float):
        self.dt = float(dt)
        self.filter = HeteroskedasticKalmanV(
            dt=dt,
            config=HeteroKalmanConfig(
                kappa=env.kappa, theta=env.theta, xi=env.xi,
                V_floor=1e-6, P_init_mult=10.0, R_scale=1.0,
            ),
        )
        self.traj_z: list = []
        self.traj_P: list = []
        self.traj_V_lo: list = []
        self.traj_V_hi: list = []

    def reset(self, V0: float):
        self.filter.reset(V0)
        self.traj_z = []
        self.traj_P = []
        self.traj_V_lo = []
        self.traj_V_hi = []

    def observe(self, r_t: float, dt: float):
        self.filter.observe(r_t, dt)
        self.traj_z.append(self.filter.last_z())
        self.traj_P.append(self.filter.last_trace_P())
        lo, hi = self.filter.V_interval()
        self.traj_V_lo.append(lo)
        self.traj_V_hi.append(hi)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


class LeadLagBLRKFVEstimator:
    r"""Revived lane: lead-lag log-sig + 3-feature BLR + outer Kalman(CIR).

    The signature here is a FEATURE MAP for a Bayesian observation model
    (`y = E[r^2/dt | phi]`), NOT the filter state.  The outer Kalman
    with CIR dynamics is the actual V filter.  Full architecture is
    documented in `src/sskf/leadlag_blr_kf.py`.
    """
    name = "blr_kf_leadlag"

    def __init__(self, env: HestonMertonEnv, dt: float,
                 ll_gamma: float = 0.99, target_clip: Optional[float] = 2.0):
        self.dt = float(dt)
        cfg = LeadLagBLRKFConfig(
            ll_gamma=ll_gamma,
            prior_w_var=10.0,
            sigma_n2_init=0.01,
            sigma_n2_alpha=0.01,
            target_clip=target_clip,
            kf_kappa=env.kappa, kf_theta=env.theta, kf_xi=env.xi,
            V_floor=1e-6, P_init_mult=10.0,
        )
        self.filter = LeadLagBLRKFilter(dt=dt, config=cfg)
        self.traj_z: list = []
        self.traj_P: list = []
        self.traj_V_lo: list = []
        self.traj_V_hi: list = []
        self.traj_R_kf: list = []

    def reset(self, V0: float):
        self.filter.reset(V0)
        self.traj_z = []
        self.traj_P = []
        self.traj_V_lo = []
        self.traj_V_hi = []
        self.traj_R_kf = []

    def observe(self, r_t: float, dt: float):
        self.filter.observe(r_t, dt)
        self.traj_z.append(self.filter.last_z())
        self.traj_P.append(self.filter.last_trace_P())
        lo, hi = self.filter.V_interval()
        self.traj_V_lo.append(lo)
        self.traj_V_hi.append(hi)
        self.traj_R_kf.append(self.filter.last_R_kf())

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


class HybridSigKalmanVEstimator:
    r"""Hybrid heteroskedastic Kalman + signature-conditioned R."""
    name = "hybrid_sig_kalman"

    def __init__(
        self, env: HestonMertonEnv, dt: float,
        R_modifier_exponent: float = 1.0,
        R_min_mult: float = 0.5, R_max_mult: float = 2.0,
        sig_forget: float = 0.94, sig_ewma_halflife: float = 50.0,
    ):
        self.dt = float(dt)
        cfg = HybridKalmanSigConfig(
            base=HeteroKalmanConfig(
                kappa=env.kappa, theta=env.theta, xi=env.xi,
                V_floor=1e-6, P_init_mult=10.0, R_scale=1.0,
            ),
            sig_input_dim=2, sig_level=2,
            sig_forget=sig_forget,
            sig_score_ewma_halflife_steps=sig_ewma_halflife,
            R_modifier_exponent=R_modifier_exponent,
            R_min_mult=R_min_mult, R_max_mult=R_max_mult,
        )
        self.filter = HybridKalmanSigV(dt=dt, config=cfg)
        self.traj_z: list = []
        self.traj_P: list = []
        self.traj_V_lo: list = []
        self.traj_V_hi: list = []
        self.traj_R_mod: list = []

    def reset(self, V0: float):
        self.filter.reset(V0)
        self.traj_z = []
        self.traj_P = []
        self.traj_V_lo = []
        self.traj_V_hi = []
        self.traj_R_mod = []

    def observe(self, r_t: float, dt: float):
        self.filter.observe(r_t, dt)
        self.traj_z.append(self.filter.last_z())
        self.traj_P.append(self.filter.last_trace_P())
        lo, hi = self.filter.V_interval()
        self.traj_V_lo.append(lo)
        self.traj_V_hi.append(hi)
        self.traj_R_mod.append(self.filter.last_R_modifier())

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


class BLFVEstimator:
    r"""LIGHT Bayesian signature CdC filter lane wrapping `DualTargetSigBLF`.

    Controller contract
    -------------------
    The value-gradient controller consumes POSTERIOR MEAN `V_hat` only.
    Posterior samples, credible intervals, and calibration z-scores are
    SURFACED as diagnostics but are NOT propagated into action selection
    in this pass (the user's design instruction was to keep the controller
    fixed and compare filter quality, not rebuild the controller around
    posterior uncertainty).
    """
    name = "bayesian_sig"

    def __init__(
        self,
        dt: float,
        blf_config: Optional[DualTargetSigBLFConfig] = None,
    ):
        self.dt = float(dt)
        self.filter = DualTargetSigBLF(blf_config)
        self._last_diag: Dict[str, float] = {}
        # Per-episode trajectories (reset at reset(); appended per observe())
        self.traj_z_mu: List[float] = []
        self.traj_z_v: List[float] = []
        self.traj_P_mu: List[float] = []
        self.traj_P_v: List[float] = []
        self.traj_V_q05: List[float] = []
        self.traj_V_q95: List[float] = []

    def reset(self, V0: float):
        self.filter.reset()
        self._last_diag = {
            "V_hat": float(V0), "V_hat_q05": float("nan"), "V_hat_q95": float("nan"),
            "z_mu": float("nan"), "z_v": float("nan"),
            "P_trace_mu": float("nan"), "P_trace_v": float("nan"),
        }
        self.traj_z_mu = []
        self.traj_z_v = []
        self.traj_P_mu = []
        self.traj_P_v = []
        self.traj_V_q05 = []
        self.traj_V_q95 = []

    def observe(self, r_t: float, dt: float):
        dx = np.array([dt, float(r_t)])
        self._last_diag = self.filter.update(dx, float(r_t), float(dt))
        # Record diagnostics for this step
        self.traj_z_mu.append(self._last_diag.get("z_mu", float("nan")))
        self.traj_z_v.append(self._last_diag.get("z_v", float("nan")))
        self.traj_P_mu.append(self._last_diag.get("P_trace_mu", float("nan")))
        self.traj_P_v.append(self._last_diag.get("P_trace_v", float("nan")))
        self.traj_V_q05.append(self._last_diag.get("V_hat_q05", float("nan")))
        self.traj_V_q95.append(self._last_diag.get("V_hat_q95", float("nan")))

    def V_hat(self) -> float:
        return float(self._last_diag.get("V_hat", 0.04))

    def V_interval(self) -> Tuple[float, float]:
        return (
            float(self._last_diag.get("V_hat_q05", float("nan"))),
            float(self._last_diag.get("V_hat_q95", float("nan"))),
        )


# ==========================================================================
# Config
# ==========================================================================


@dataclass(frozen=True)
class VGConfig:
    rho: float = -0.7
    gamma: float = 3.0
    V_floor_for_pi_ref: float = 0.005
    T_steps: int = 60              # decision horizon (days)
    dt: float = 1.0 / 252.0
    u_max: float = 0.3
    n_train: int = 250
    n_test: int = 120
    V0_low: float = 0.02
    V0_high: float = 0.08
    ewma_halflife_days: float = 21.0
    ridge_transition: float = 1e-4
    ridge_value: float = 1e-4

    @property
    def ewma_lam(self) -> float:
        return float(np.log(2.0) / max(self.ewma_halflife_days, 1e-3))


# ==========================================================================
# State object (adapter-local, opaque to generic module)
# ==========================================================================


@dataclass
class HVGState:
    logW: float
    V: float
    t: int
    logW0: float
    ewma_r: float
    ewma_r2: float

    def clone(self) -> "HVGState":
        return HVGState(
            logW=self.logW, V=self.V, t=self.t, logW0=self.logW0,
            ewma_r=self.ewma_r, ewma_r2=self.ewma_r2,
        )


# ==========================================================================
# Lifted state maps
# ==========================================================================


class PsiA:
    r"""Stage A minimal lift: [1, log(V/theta), 1/V, tau_frac].

    Accepts an optional `V_override` so lanes with different V estimators
    (Oracle / EWMA / Bayesian signature filter) can plug in THEIR V_hat
    without changing the env state itself.  This keeps the env dynamics
    driven by true V while the controller sees only the filtered V.
    """
    name = "A_minimal"

    def __init__(self, env: HestonMertonEnv, T_steps: int):
        self.env = env
        self.T_steps = int(T_steps)
        self.theta = float(env.theta)

    @property
    def dim(self) -> int:
        return 4

    def __call__(
        self, state: HVGState, V_override: Optional[float] = None,
    ) -> np.ndarray:
        V_use = state.V if V_override is None else V_override
        V = max(V_use, 1e-8)
        tau_frac = (self.T_steps - state.t) / self.T_steps
        return np.array([
            1.0,
            float(np.log(V / self.theta)),
            1.0 / V,
            tau_frac,
        ])


class PsiB:
    r"""Stage B path-aware lift: Stage A + [EWMA(log-return), EWMA(log-return^2)]."""
    name = "B_path_aware"

    def __init__(self, env: HestonMertonEnv, T_steps: int):
        self.A = PsiA(env, T_steps)

    @property
    def dim(self) -> int:
        return 6

    def __call__(
        self, state: HVGState, V_override: Optional[float] = None,
    ) -> np.ndarray:
        base = self.A(state, V_override=V_override)
        return np.concatenate([base, [state.ewma_r, state.ewma_r2]])


# ==========================================================================
# Terminal target and stage cost
# ==========================================================================


class TerminalZero:
    r"""Terminal target: V_T(psi) == 0.

    Used together with stage cost = observed log-return r_t so the total
    backward-accumulated value equals logW_T - logW_0.  This decomposition
    keeps logW out of psi (letting Stage B path features matter) while
    preserving the physical meaning of the target.
    """
    name = "zero_terminal"

    def __call__(self, state_final: HVGState, psi_final: np.ndarray) -> float:
        return 0.0


class ObservedLogReturnStageCost:
    r"""Stage cost = observed log-return r_t = log W_{t+1} - log W_t.

    This is the exploration-observed realization, passed as scalar
    transition_observation['r_t'] by the adapter.  Note: this stage cost
    is NOT parameterized in u here; the u-dependence of r_t is assembled
    analytically at DECISION TIME in `heston_stage_cost_u_quadratic`.

    In training, we use the realized r_t for each (psi_t, u_t) tuple;
    the backward fit implicitly averages over the exploration u
    distribution.  This is standard fitted value iteration.
    """
    name = "observed_log_return"

    def __call__(self, psi_t, u, state_t, transition_observation) -> float:
        return float(transition_observation["r_t"])


def heston_stage_cost_u_quadratic(
    env: HestonMertonEnv, state: HVGState, cfg: "VGConfig",
) -> Tuple[float, float, float]:
    r"""Analytical u-quadratic of E[r_t | state, u] at decision time.

    Under the multiplicative overlay pi_t = (1+u) pi_ref(V):
        E[r_t | V, u] dt
      = (r + (1+u) pi_ref (mu-r) - 0.5 (1+u)^2 pi_ref^2 V) dt
      = const  +  u * [pi_ref (mu-r) - pi_ref^2 V] dt
              +  u^2 * [-0.5 pi_ref^2 V] dt
    With pi_ref = (mu-r)/(gamma V_floored):
        e_1 = (mu-r)^2 (1 - 1/gamma) / (gamma V_floored) * dt
        e_2 = -0.5 (mu-r)^2 / (gamma^2 V_floored) * dt
    The constant is added to alpha_0 but does not affect the argmax in u.
    """
    V = max(state.V, cfg.V_floor_for_pi_ref)
    a = float(env.mu - env.r)
    gamma = float(env.gamma)
    pi_ref = a / (gamma * V)
    pi_ref_sq = pi_ref * pi_ref
    e_0 = (env.r + pi_ref * a - 0.5 * pi_ref_sq * V) * cfg.dt
    e_1 = (pi_ref * a - pi_ref_sq * V) * cfg.dt
    e_2 = -0.5 * pi_ref_sq * V * cfg.dt
    return float(e_0), float(e_1), float(e_2)


# ==========================================================================
# Rollout helpers
# ==========================================================================


def _paired_noise(rho: float, T: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    zA = rng.standard_normal(T)
    zB = rng.standard_normal(T)
    z1 = zA
    z2 = rho * zA + np.sqrt(max(1.0 - rho * rho, 0.0)) * zB
    return np.column_stack([z1, z2])


def _pi_ref(env: HestonMertonEnv, V: float, V_floor: float) -> float:
    return float(env.merton_optimal(max(V, V_floor)))


def _step_state(
    env: HestonMertonEnv,
    cfg: VGConfig,
    state: HVGState,
    u: float,
    z1: float,
    z2: float,
) -> Tuple[HVGState, Dict[str, float]]:
    r"""Advance one step under multiplicative overlay; return (new_state, obs).

    Returned observation dict
    -------------------------
    r_t_wealth : d_logW = (r + pi*(mu-r) - 0.5*pi^2*V)*dt + pi*sqrt(V)*dB
                 Controlled wealth log-return.  Brownian term scales with
                 pi*sqrt(V), so (r_t_wealth)^2/dt ~ pi^2 * V.  APPROPRIATE
                 as the controller's stage cost; INAPPROPRIATE as a V
                 estimator input when pi != 1.

    dr_S       : d_logS = (mu - 0.5*V)*dt + sqrt(V)*dB   (underlying asset
                 log-return; pi-free).  Brownian term has E[(dr_S)^2/dt|V] = V.
                 Use this for variance-filter observations.

    pi_t, pi_ref: diagnostic.

    dr_S is reconstructed using the SAME z1 and pre-step V that env.step_explicit
    used, so it is realization-consistent with the rollout's wealth return.
    """
    pi_ref = _pi_ref(env, state.V, cfg.V_floor_for_pi_ref)
    pi_t = (1.0 + float(u)) * pi_ref
    logW_new, V_new = env.step_explicit(
        state.logW, state.V, pi_t, z1, z2, cfg.dt,
    )
    r_t_wealth = logW_new - state.logW
    # Underlying asset return from the same z1 and pre-step V
    V_pre = max(state.V, 1e-8)
    sqrt_V = float(np.sqrt(V_pre))
    dr_S = (
        (env.mu - 0.5 * V_pre) * cfg.dt
        + sqrt_V * float(np.sqrt(cfg.dt)) * float(z1)
    )
    rho_e = float(np.exp(-cfg.ewma_lam * cfg.dt))
    new_state = HVGState(
        logW=logW_new,
        V=V_new,
        t=state.t + 1,
        logW0=state.logW0,
        ewma_r=rho_e * state.ewma_r + (1.0 - rho_e) * r_t_wealth,
        ewma_r2=rho_e * state.ewma_r2 + (1.0 - rho_e) * (r_t_wealth * r_t_wealth),
    )
    return new_state, {
        "r_t": r_t_wealth,
        "dr_S": dr_S,
        "pi_t": pi_t,
        "pi_ref": pi_ref,
    }


# ==========================================================================
# Training data collection: uniform exploration of scalar u
# ==========================================================================


def collect_training_episodes(
    env: HestonMertonEnv,
    cfg: VGConfig,
    psi_map: object,
    n_episodes: int,
    base_seed: int,
    u_sampler: Optional[Callable[[np.random.RandomState], float]] = None,
    v_estimator_factory: Optional[Callable[[], object]] = None,
) -> List[EpisodeRecord]:
    r"""Paired-noise rollouts under SCALAR exploration u (iid per step).

    v_estimator_factory: if provided, constructs a FRESH per-episode
    V-estimator (Oracle / EWMA / BLF).  The controller's psi then uses
    the estimator's V_hat rather than state.V, so the controller ONLY
    sees filter output.  Env dynamics use true V regardless.
    """
    if u_sampler is None:
        def u_sampler(rng):
            return float(rng.uniform(-cfg.u_max, cfg.u_max))
    episodes: List[EpisodeRecord] = []
    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_episodes):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        noise = _paired_noise(cfg.rho, cfg.T_steps, base_seed + 1 + k)
        u_rng = np.random.RandomState(base_seed + 10_000 + k)

        state = HVGState(
            logW=0.0, V=V0, t=0, logW0=0.0,
            ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
        )
        v_est = v_estimator_factory() if v_estimator_factory is not None else None
        if v_est is not None:
            v_est.reset(V0)
        psis = np.zeros((cfg.T_steps + 1, psi_map.dim))
        us = np.zeros(cfg.T_steps)
        states_list: List[HVGState] = [state.clone()]
        trans_list: List[Dict[str, float]] = []
        # Initial psi: V_override = current estimate (or state.V if none)
        if v_est is None:
            psis[0] = psi_map(state)
        else:
            if isinstance(v_est, OracleVEstimator):
                v_est.set_true_V(state.V)
            psis[0] = psi_map(state, V_override=v_est.V_hat())
        for t in range(cfg.T_steps):
            u_t = u_sampler(u_rng)
            us[t] = u_t
            state, obs = _step_state(env, cfg, state, u_t, float(noise[t, 0]), float(noise[t, 1]))
            # r_t_wealth only used for HVGState EWMAs and the generic episode
            # record's `transitions`; filter observation below uses dr_S.
            r_t = obs["r_t"]
            # Advance V-estimator with the realized UNDERLYING asset return
            # (dr_S).  Using wealth return r_t would make the filter estimate
            # pi^2 * V rather than V -- see docstring of _step_state.
            if v_est is not None:
                if isinstance(v_est, OracleVEstimator):
                    v_est.set_true_V(state.V)  # oracle cheats each step
                else:
                    v_est.observe(obs["dr_S"], cfg.dt)
            if v_est is None:
                psis[t + 1] = psi_map(state)
            else:
                psis[t + 1] = psi_map(state, V_override=v_est.V_hat())
            states_list.append(state.clone())
            trans_list.append(obs)
        episodes.append(EpisodeRecord(
            psis=psis,
            us=us,
            states=states_list[:-1],           # state_t for t=0..T-1
            transitions=trans_list,            # length T
            terminal_state=states_list[-1],
            psi_final=psis[-1].copy(),
        ))
    return episodes


# ==========================================================================
# Held-out evaluation under CRN
# ==========================================================================


@dataclass
class HeldOutResult:
    name: str
    delta_logW: np.ndarray             # per-seed log W_T(policy) - log W_T(u=0)
    u_history: np.ndarray              # (n_test, T) per-step u applied
    abstention_flags: np.ndarray       # per-seed, fraction of steps abstained
    per_seed_V_at_decision: np.ndarray
    model_name: str
    V_hat_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))      # pre-update (controller-seen)
    V_true_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))     # pre-step V_true
    V_hat_post_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0))) # post-update (filter-audit)
    V_true_post_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))# post-step V_true
    # Calibration diagnostics for the Bayesian lane (empty for others)
    V_q05_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V_q95_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    z_mu_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    z_v_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    P_trace_mu_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    P_trace_v_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def _myopic_rollout(env: HestonMertonEnv, cfg: VGConfig, V0: float, noise: np.ndarray) -> float:
    r"""Rollout with u=0 throughout (pure myopic reference).  Return log W_T."""
    state = HVGState(
        logW=0.0, V=V0, t=0, logW0=0.0,
        ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
    )
    for t in range(cfg.T_steps):
        state, _ = _step_state(env, cfg, state, 0.0, float(noise[t, 0]), float(noise[t, 1]))
    return float(state.logW)


def _value_gradient_rollout(
    env: HestonMertonEnv, cfg: VGConfig, V0: float, noise: np.ndarray,
    psi_map: object, model: ValueGradientModel,
    v_estimator: Optional[object] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Rollout under CRN with u_t*(z_t) from value-gradient policy.

    Per-step sequence with pre-/post-update V_hat bookkeeping:
      1. V_hat_pre_t  = v_estimator.V_hat()  (filter posterior given r_0..r_{t-1})
         This is what the controller SEES at decision time for step t.
      2. Build psi_t with V_override = V_hat_pre_t, extract u_t* from the
         local Hamiltonian.
      3. Step env with u_t; obtain (r_t_wealth, dr_S_t, new V_t).
      4. Record V_true_post_t = state.V (= V_{t+1} in standard indexing,
         but consistent with the "variance observed in the just-executed
         step" convention).
      5. v_estimator.observe(dr_S_t, dt)  -> updates filter to include
         step-t information; the UNDERLYING asset return is the correct
         observation for variance filtering (wealth return scales with
         pi^2*V and would bias V_hat downward).
      6. V_hat_post_t = v_estimator.V_hat().

    Returns
    -------
    (logW_T, u_history, abstention_flags,
     V_hat_pre_history,  -- filter's forecast of V_t before observing r_t (used by controller)
     V_hat_post_history, -- filter's update for V_t after observing r_t (used by audit plots)
     V_true_pre_history, -- V_t BEFORE step t (matches what V_hat_pre forecasts)
     V_true_post_history)-- V_{t+1} AFTER step t (matches what V_hat_post estimates)
    """
    state = HVGState(
        logW=0.0, V=V0, t=0, logW0=0.0,
        ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
    )
    if v_estimator is not None:
        v_estimator.reset(V0)
        if isinstance(v_estimator, OracleVEstimator):
            v_estimator.set_true_V(state.V)
    u_history = np.zeros(cfg.T_steps)
    abstained = np.zeros(cfg.T_steps, dtype=bool)
    V_hat_pre_history = np.zeros(cfg.T_steps)
    V_hat_post_history = np.zeros(cfg.T_steps)
    V_true_pre_history = np.zeros(cfg.T_steps)
    V_true_post_history = np.zeros(cfg.T_steps)
    for t in range(cfg.T_steps):
        V_hat_pre = state.V if v_estimator is None else v_estimator.V_hat()
        V_hat_pre_history[t] = V_hat_pre
        V_true_pre_history[t] = state.V
        psi_t = (
            psi_map(state, V_override=V_hat_pre)
            if v_estimator is not None else psi_map(state)
        )
        stage_state = state.clone()
        stage_state.V = V_hat_pre
        stage_uq = heston_stage_cost_u_quadratic(env, stage_state, cfg)
        rec = extract_action(
            model, psi_t, t, u_max=cfg.u_max,
            stage_cost_quadratic=stage_uq,
        )
        u = rec.u_star
        u_history[t] = u
        abstained[t] = (not rec.concave)
        state, obs = _step_state(env, cfg, state, u, float(noise[t, 0]), float(noise[t, 1]))
        V_true_post_history[t] = state.V
        if v_estimator is not None:
            if isinstance(v_estimator, OracleVEstimator):
                v_estimator.set_true_V(state.V)
            else:
                v_estimator.observe(obs["dr_S"], cfg.dt)
            V_hat_post_history[t] = v_estimator.V_hat()
        else:
            V_hat_post_history[t] = state.V
    return (
        float(state.logW), u_history, abstained,
        V_hat_pre_history, V_hat_post_history,
        V_true_pre_history, V_true_post_history,
    )


def evaluate_controller_paired(
    env: HestonMertonEnv,
    cfg: VGConfig,
    psi_map: object,
    model: Optional[ValueGradientModel],
    base_seed: int,
    name: str,
    v_estimator_factory: Optional[Callable[[], object]] = None,
) -> HeldOutResult:
    r"""Paired (policy vs u=0) rollout per test seed under CRN.

    `v_estimator_factory` constructs a FRESH per-episode V-estimator that
    the policy uses for its psi input.  Oracle uses true V; EWMA/BLF use
    their own V_hat.  Env dynamics use true V regardless.
    """
    v0_rng = np.random.RandomState(base_seed)
    delta_logW = np.zeros(cfg.n_test)
    u_history = np.zeros((cfg.n_test, cfg.T_steps))
    abst_rate = np.zeros(cfg.n_test)
    V_decision = np.zeros(cfg.n_test)
    V_hat_hist = np.zeros((cfg.n_test, cfg.T_steps))
    V_true_hist = np.zeros((cfg.n_test, cfg.T_steps))
    V_hat_post_hist = np.zeros((cfg.n_test, cfg.T_steps))
    V_true_post_hist = np.zeros((cfg.n_test, cfg.T_steps))
    # Calibration storage (populated only for BLF lane)
    V_q05_hist = np.full((cfg.n_test, cfg.T_steps), float("nan"))
    V_q95_hist = np.full((cfg.n_test, cfg.T_steps), float("nan"))
    z_mu_hist = np.full((cfg.n_test, cfg.T_steps), float("nan"))
    z_v_hist = np.full((cfg.n_test, cfg.T_steps), float("nan"))
    P_trace_mu_hist = np.full((cfg.n_test, cfg.T_steps), float("nan"))
    P_trace_v_hist = np.full((cfg.n_test, cfg.T_steps), float("nan"))
    for k in range(cfg.n_test):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        V_decision[k] = V0
        noise = _paired_noise(cfg.rho, cfg.T_steps, base_seed + 1 + k)
        logW_my = _myopic_rollout(env, cfg, V0, noise)
        if model is None:
            logW_policy = logW_my
            abst_rate[k] = 1.0
        else:
            v_est = v_estimator_factory() if v_estimator_factory is not None else None
            (logW_policy, u_h, abst,
             Vh_pre, Vh_post, Vt_pre, Vt_post) = _value_gradient_rollout(
                env, cfg, V0, noise, psi_map, model, v_estimator=v_est,
            )
            u_history[k] = u_h
            abst_rate[k] = float(np.mean(abst))
            V_hat_hist[k] = Vh_pre
            V_true_hist[k] = Vt_pre
            V_hat_post_hist[k] = Vh_post
            V_true_post_hist[k] = Vt_post
            # Pull out BLF calibration trajectories if available
            if isinstance(v_est, BLFVEstimator):
                n_avail = min(len(v_est.traj_V_q05), cfg.T_steps)
                if n_avail > 0:
                    V_q05_hist[k, :n_avail] = v_est.traj_V_q05[:n_avail]
                    V_q95_hist[k, :n_avail] = v_est.traj_V_q95[:n_avail]
                    z_mu_hist[k, :n_avail] = v_est.traj_z_mu[:n_avail]
                    z_v_hist[k, :n_avail] = v_est.traj_z_v[:n_avail]
                    P_trace_mu_hist[k, :n_avail] = v_est.traj_P_mu[:n_avail]
                    P_trace_v_hist[k, :n_avail] = v_est.traj_P_v[:n_avail]
        delta_logW[k] = logW_policy - logW_my
    return HeldOutResult(
        name=name,
        delta_logW=delta_logW,
        u_history=u_history,
        abstention_flags=abst_rate,
        per_seed_V_at_decision=V_decision,
        model_name=getattr(psi_map, "name", "?"),
        V_hat_history=V_hat_hist,
        V_true_history=V_true_hist,
        V_hat_post_history=V_hat_post_hist,
        V_true_post_history=V_true_post_hist,
        V_q05_history=V_q05_hist,
        V_q95_history=V_q95_hist,
        z_mu_history=z_mu_hist,
        z_v_history=z_v_hist,
        P_trace_mu_history=P_trace_mu_hist,
        P_trace_v_history=P_trace_v_hist,
    )


# ==========================================================================
# Reporting with CRRA-style risk-sensitive score
# ==========================================================================


def _crra_score_bootstrap(
    y: np.ndarray, gamma: float, n_boot: int = 2000,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, float]:
    if rng is None:
        rng = np.random.RandomState(0)
    y = np.asarray(y, dtype=float)
    lam = 0.5 * (gamma - 1.0)
    point = float(np.mean(y) - lam * np.var(y, ddof=1)) if y.size > 1 else float(np.mean(y))
    boot = np.zeros(n_boot)
    n = y.size
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y[idx]
        boot[b] = float(np.mean(yb) - lam * np.var(yb, ddof=1)) if n > 1 else float(np.mean(yb))
    return {
        "point": point,
        "se": float(np.std(boot, ddof=1)),
        "q05": float(np.quantile(boot, 0.05)),
        "q95": float(np.quantile(boot, 0.95)),
    }


def _format_row(name: str, res: HeldOutResult, gamma: float) -> str:
    y = res.delta_logW
    mean_y = float(np.mean(y))
    var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    crra = _crra_score_bootstrap(y, gamma, rng=np.random.RandomState(2024))
    abst = float(np.mean(res.abstention_flags))
    u_flat = res.u_history.flatten()
    # Per-seed per-step non-abstained u statistics (drop zeros as abstain proxy)
    nonzero = u_flat[np.abs(u_flat) > 1e-8]
    if nonzero.size > 0:
        u_mean = float(np.mean(nonzero))
        u_std = float(np.std(nonzero, ddof=1)) if nonzero.size > 1 else 0.0
    else:
        u_mean = u_std = 0.0
    return (
        f"{name:28s} | {crra['point']:+.5f} [{crra['q05']:+.5f}, {crra['q95']:+.5f}] "
        f"| {mean_y:+.5f}   | {var_y:.4e} | {abst:.2f}     | "
        f"{u_mean:+.3f}  {u_std:.3f}"
    )


# ==========================================================================
# Runner
# ==========================================================================


def _three_lane_comparison():
    r"""Train and evaluate Oracle / EWMA / Bayesian-signature lanes.

    For each lane we:
      1. collect training episodes with a fresh per-episode V-estimator
         of the lane's type (so psi during training matches what the
         controller will see at test time);
      2. fit the local value-gradient model on those episodes;
      3. evaluate on held-out test seeds, paired against myopic u=0.
    Primary held-out metric: CRRA-style score
        mean(Δ logW) - 0.5*(gamma-1)*Var(Δ logW).
    """
    cfg = VGConfig()
    env = HestonMertonEnv(rho=cfg.rho, gamma=cfg.gamma)
    psi = PsiA(env, cfg.T_steps)
    terminal_target = TerminalZero()
    stage_cost = ObservedLogReturnStageCost()

    blf_cfg = DualTargetSigBLFConfig(
        input_dim=2, sig_level=2, sig_forget=0.94,
        prior_var_mu=100.0, prior_var_v=100.0,
        process_noise_mu=1e-4, process_noise_v=1e-4,
        R_init_mu=10.0, R_init_v=0.5,
        R_adapt_halflife=50.0, winsor_v_q=0.995,
    )

    lanes: List[Tuple[str, Callable[[], object]]] = [
        ("oracle",            lambda: OracleVEstimator()),
        ("ewma",              lambda: EWMAVEstimator(halflife_days=21.0, dt=cfg.dt)),
        # Legacy reference: two-head Bayesian sig CdC filter
        ("bayesian_sig",      lambda: BLFVEstimator(dt=cfg.dt, blf_config=blf_cfg)),
        # Current strong partial-observation lane
        ("hetero_kalman",     lambda: HeteroKalmanVEstimator(env=env, dt=cfg.dt)),
        # REVIVED signature lane: lead-lag + 3-feature BLR + outer CIR Kalman
        ("blr_kf_leadlag",    lambda: LeadLagBLRKFVEstimator(
            env=env, dt=cfg.dt, ll_gamma=0.99, target_clip=2.0,
        )),
    ]

    print("=" * 120)
    print("HESTON VALUE-GRADIENT CONTROLLER  --  three-lane V-filter comparison")
    print(f"  rho={cfg.rho}  gamma={cfg.gamma}  horizon={cfg.T_steps} days  dt={cfg.dt:.5f}")
    print(f"  V0 uniform [{cfg.V0_low}, {cfg.V0_high}]  u_max={cfg.u_max}")
    print(f"  n_train={cfg.n_train}  n_test={cfg.n_test}")
    print(f"  psi = PsiA minimal (dim {psi.dim})  -- identical across lanes")
    print(f"  only V in psi differs: Oracle=V_true, EWMA=EWMA(r^2)/dt, "
          f"bayesian_sig=DualTargetSigBLF.V_hat")
    print(f"  Primary metric: CRRA score = mean(Δ logW) - 0.5(γ-1) Var(Δ logW)"
          f"  [γ={cfg.gamma}]")
    print("=" * 120)

    summaries: List[Dict[str, object]] = []
    all_evals: List[HeldOutResult] = []
    for name, factory in lanes:
        print()
        print(f"--- Lane: {name} ---")
        print("  Collecting training episodes...")
        eps = collect_training_episodes(
            env, cfg, psi, n_episodes=cfg.n_train, base_seed=1_000,
            v_estimator_factory=factory,
        )
        print("  Fitting value-gradient model...")
        model = backward_value_iteration(
            eps, terminal_target=terminal_target, stage_cost=stage_cost,
            ridge_transition=cfg.ridge_transition, ridge_value=cfg.ridge_value,
        )
        r2 = training_value_r2(model, eps, terminal_target, stage_cost)
        print(f"  R² per step (select):  "
              f"t=0: {r2.get(0, float('nan')):.3f}   "
              f"t={cfg.T_steps//2}: {r2.get(cfg.T_steps//2, float('nan')):.3f}   "
              f"t={cfg.T_steps-1}: {r2.get(cfg.T_steps-1, float('nan')):.3f}")

        print("  Evaluating on held-out seeds...")
        ev = evaluate_controller_paired(
            env, cfg, psi, model=model,
            base_seed=500_000, name=f"value_grad_{name}",
            v_estimator_factory=factory,
        )
        all_evals.append(ev)

        y = ev.delta_logW
        mean_y = float(np.mean(y))
        var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
        crra_stats = _crra_score_bootstrap(
            y, cfg.gamma, rng=np.random.RandomState(2026),
        )
        abst = float(np.mean(ev.abstention_flags))
        u_nonzero = ev.u_history[np.abs(ev.u_history) > 1e-8]
        u_mean = float(np.mean(u_nonzero)) if u_nonzero.size else 0.0
        u_std = float(np.std(u_nonzero, ddof=1)) if u_nonzero.size > 1 else 0.0

        # V-estimate tracking diagnostics
        if ev.V_hat_history.size > 0 and name != "oracle":
            Vh = ev.V_hat_history
            Vt = ev.V_true_history
            rmse_Vhat = float(np.sqrt(np.mean((Vh - Vt) ** 2)))
            corr_Vhat = float(np.corrcoef(Vh.flatten(), Vt.flatten())[0, 1])
        else:
            rmse_Vhat = 0.0
            corr_Vhat = 1.0

        summaries.append({
            "name": name,
            "crra_point": crra_stats["point"],
            "crra_q05": crra_stats["q05"],
            "crra_q95": crra_stats["q95"],
            "crra_se": crra_stats["se"],
            "mean": mean_y,
            "var": var_y,
            "abstention": abst,
            "u_mean": u_mean,
            "u_std": u_std,
            "rmse_Vhat": rmse_Vhat,
            "corr_Vhat": corr_Vhat,
        })
        print(f"  CRRA score = {crra_stats['point']:+.6f} "
              f"[{crra_stats['q05']:+.6f}, {crra_stats['q95']:+.6f}]   "
              f"mean Δ logW = {mean_y:+.6f}   Var Δ logW = {var_y:.4e}   "
              f"abst={abst:.2f}  |u| mean,std = {u_mean:+.3f}, {u_std:.3f}")
        if name != "oracle":
            print(f"  V_hat diagnostics (held-out):  "
                  f"RMSE(V_hat, V_true) = {rmse_Vhat:.4f}   "
                  f"corr(V_hat, V_true) = {corr_Vhat:+.3f}")
        if name == "bayesian_sig":
            # Posterior interval summary on V_hat: we have it per-seed per-step
            # via the filter's stored diagnostics, but only the LAST value is
            # in our state.  We summarize over all seeds: average of
            # held-out V_hat vs V_true across all (seed, step) pairs.
            print(f"  (BLF posterior CI reported inside filter update diagnostics "
                  f"per step; full sample in ev.V_hat_history)")

    # ---- FILTER-FIRST AUDIT (reported BEFORE controller metrics) ----
    print()
    print("=" * 120)
    print("FILTER-FIRST AUDIT  (V_hat quality on held-out; before any controller metric)")
    print("  After fix: filters observe UNDERLYING asset return dr_S (not wealth return r).")
    print("  Audit now compares POST-update V_hat vs POST-step V_true (matching epochs).")
    print("  corr_Vhat_lead = corr(V_hat_post_t, V_true_post_{t+1}).")
    print("=" * 120)
    print(
        f"{'lane':22s} | {'RMSE V_hat':>11s} | {'corr V_hat':>11s} | "
        f"{'corr V_lead':>11s} | {'cov90%':>7s} | {'z_mean':>7s} | {'z_std':>7s} | {'flags':s}"
    )
    print("-" * 120)
    filter_stats = []
    for ev in all_evals:
        # Use POST-update V_hat vs POST-step V_true for filter quality.
        Vh = ev.V_hat_post_history if ev.V_hat_post_history.size else ev.V_hat_history
        Vt = ev.V_true_post_history if ev.V_true_post_history.size else ev.V_true_history
        if Vh.size == 0 or Vt.size == 0:
            continue
        warmup = min(10, cfg.T_steps // 6)
        vh = Vh[:, warmup:].flatten()
        vt = Vt[:, warmup:].flatten()
        rmse = float(np.sqrt(np.mean((vh - vt) ** 2)))
        corr = float(np.corrcoef(vh, vt)[0, 1])
        # Leading-step correlation (V_hat_post_t vs V_true_post_{t+1}): does
        # the post-update forecast help predict the next step?
        if Vh.shape[1] > 1:
            vh_t = Vh[:, warmup:-1].flatten()
            vt_tp1 = Vt[:, warmup + 1:].flatten()
            corr_lead = float(np.corrcoef(vh_t, vt_tp1)[0, 1])
        else:
            corr_lead = float("nan")
        # Interval coverage
        if ev.V_q05_history.size and np.isfinite(ev.V_q05_history).any():
            lo = ev.V_q05_history[:, warmup:].flatten()
            hi = ev.V_q95_history[:, warmup:].flatten()
            m = np.isfinite(lo) & np.isfinite(hi)
            if m.any():
                cov = float(np.mean((vt[m] >= lo[m]) & (vt[m] <= hi[m])))
            else:
                cov = float("nan")
        else:
            cov = float("nan")
        # Innovation z-scores
        if ev.z_mu_history.size and np.isfinite(ev.z_mu_history).any():
            zarr = ev.z_mu_history[:, warmup:].flatten()
            zarr = zarr[np.isfinite(zarr)]
            if zarr.size:
                z_mean = float(np.mean(zarr))
                z_std = float(np.std(zarr, ddof=1))
            else:
                z_mean = z_std = float("nan")
        else:
            z_mean = z_std = float("nan")
        flags = []
        if np.isfinite(corr) and corr < 0:
            flags.append("NEG_CORR")
        if np.isfinite(cov) and cov > 0.98:
            flags.append("OVER_CI")
        if np.isfinite(cov) and cov < 0.70 and cov == cov:  # not nan
            flags.append("UNDER_CI")
        if np.isfinite(z_std) and z_std > 2.0:
            flags.append("Z_WIDE")
        if np.isfinite(z_std) and 0.0 < z_std < 0.3:
            flags.append("Z_NARROW")
        flag_s = ",".join(flags) if flags else "—"
        filter_stats.append({
            "name": ev.name, "rmse": rmse, "corr": corr, "corr_lead": corr_lead,
            "cov": cov, "z_mean": z_mean, "z_std": z_std, "flags": flag_s,
        })
        print(
            f"{ev.name:22s} | {rmse:>11.5f} | {corr:>+11.4f} | "
            f"{corr_lead:>+11.4f} | "
            f"{(f'{cov:.3f}' if np.isfinite(cov) else '    —'):>7s} | "
            f"{(f'{z_mean:+.2f}' if np.isfinite(z_mean) else '    —'):>7s} | "
            f"{(f'{z_std:.2f}' if np.isfinite(z_std) else '    —'):>7s} | {flag_s}"
        )

    # ---- Side-by-side controller summary ----
    print()
    print("=" * 120)
    print("CONTROLLER METRICS (held-out, paired vs myopic u=0)")
    print("=" * 120)
    print(
        f"{'lane':18s} | CRRA score [90% CI]                 "
        f"| mean Δ logW | Var Δ logW  | abst | u mean, std  "
        f"| RMSE V_hat | corr V_hat"
    )
    print("-" * 120)
    for s in summaries:
        ci_str = (
            f"{s['crra_point']:+.5f} [{s['crra_q05']:+.5f}, {s['crra_q95']:+.5f}]"
        )
        print(
            f"{s['name']:18s} | {ci_str:36s} | "
            f"{s['mean']:+.5f}   | {s['var']:.4e}  | {s['abstention']:.2f} | "
            f"{s['u_mean']:+.3f}, {s['u_std']:.3f} | "
            f"{s['rmse_Vhat']:.4f}    | {s['corr_Vhat']:+.3f}"
        )

    # ---- BLF posterior calibration diagnostics ----
    print()
    print("=" * 120)
    print("BAYESIAN SIGNATURE LANE  -- posterior calibration (held-out)")
    print("  Controller uses posterior MEAN V_hat only; intervals and z-scores")
    print("  are diagnostics, not fed into action selection (this pass).")
    print("  This is a LIGHT Bayesian signature CdC filter, NOT full Sig-KKF.")
    print("=" * 120)
    blf_ev = next((e for e in all_evals if e.name == "value_grad_bayesian_sig"), None)
    if blf_ev is not None:
        # Drop warm-up steps (posterior is diffuse initially) + any NaNs
        warmup = min(10, cfg.T_steps // 6)
        zmu = blf_ev.z_mu_history[:, warmup:].flatten()
        zv = blf_ev.z_v_history[:, warmup:].flatten()
        zmu = zmu[np.isfinite(zmu)]
        zv = zv[np.isfinite(zv)]
        Vh = blf_ev.V_hat_history[:, warmup:].flatten()
        Vt = blf_ev.V_true_history[:, warmup:].flatten()
        Vq05 = blf_ev.V_q05_history[:, warmup:].flatten()
        Vq95 = blf_ev.V_q95_history[:, warmup:].flatten()
        mask_cov = np.isfinite(Vq05) & np.isfinite(Vq95) & np.isfinite(Vt)
        coverage_90 = float(
            np.mean((Vt[mask_cov] >= Vq05[mask_cov]) & (Vt[mask_cov] <= Vq95[mask_cov]))
        ) if mask_cov.any() else float("nan")
        interval_width = float(np.mean(Vq95[mask_cov] - Vq05[mask_cov])) if mask_cov.any() else float("nan")

        print(f"  (warm-up steps 0..{warmup-1} excluded from calibration)")
        print()
        print(f"  Innovation z-scores (should be ~ N(0,1) if calibrated):")
        if zmu.size:
            print(f"    drift head (r/dt):         "
                  f"mean={np.mean(zmu):+.3f}  std={np.std(zmu, ddof=1):.3f}  "
                  f"|z|>2: {float(np.mean(np.abs(zmu) > 2)):.3f}  "
                  f"|z|>3: {float(np.mean(np.abs(zmu) > 3)):.3f}")
        if zv.size:
            print(f"    2nd-moment head (r^2/dt):  "
                  f"mean={np.mean(zv):+.3f}  std={np.std(zv, ddof=1):.3f}  "
                  f"|z|>2: {float(np.mean(np.abs(zv) > 2)):.3f}  "
                  f"|z|>3: {float(np.mean(np.abs(zv) > 3)):.3f}")
        print()
        print(f"  V_hat 90% credible interval vs true V:")
        print(f"    empirical coverage:       {coverage_90:.3f}  "
              f"(nominal 0.90; higher = overly conservative; "
              f"lower = overconfident)")
        print(f"    mean interval width:      {interval_width:.4f}")
        print(f"    mean |V_hat - V_true|:    "
              f"{float(np.mean(np.abs(Vh - Vt))):.4f}")
        print()
        print(f"  Posterior variance trajectory (drift head trace(P)):")
        P_mu_hist = blf_ev.P_trace_mu_history
        P_mu_hist = P_mu_hist[np.isfinite(P_mu_hist)].reshape(-1, cfg.T_steps) if P_mu_hist.size else P_mu_hist
        if P_mu_hist.size:
            mean_by_t = np.nanmean(blf_ev.P_trace_mu_history, axis=0)
            print(f"    mean trace(P_mu) at t=0:   {mean_by_t[0]:.4f}")
            print(f"    mean trace(P_mu) at t=T/2: {mean_by_t[cfg.T_steps//2]:.4f}")
            print(f"    mean trace(P_mu) at t=T-1: {mean_by_t[-1]:.4f}")
            collapse = mean_by_t[-1] / max(mean_by_t[0], 1e-18)
            print(f"    end/start ratio:           {collapse:.3f}  "
                  f"(too-small ratio = posterior collapse; healthy is ~0.1-0.5 "
                  f"for a steady-state filter)")
        P_v_hist = blf_ev.P_trace_v_history
        if P_v_hist.size and np.isfinite(P_v_hist).any():
            mean_by_t_v = np.nanmean(P_v_hist, axis=0)
            print(f"    mean trace(P_v)  at t=0:   {mean_by_t_v[0]:.4f}")
            print(f"    mean trace(P_v)  at t=T-1: {mean_by_t_v[-1]:.4f}")
            collapse_v = mean_by_t_v[-1] / max(mean_by_t_v[0], 1e-18)
            print(f"    end/start ratio:           {collapse_v:.3f}")

    # ---- Gap closure across all non-oracle lanes ----
    print()
    print("=" * 120)
    print("GAP CLOSURE (fraction of Oracle - EWMA CRRA-score gap closed by each lane)")
    print("=" * 120)
    by_name = {s["name"]: s for s in summaries}
    if {"oracle", "ewma"} <= set(by_name):
        d_oracle = by_name["oracle"]["crra_point"]
        d_ewma = by_name["ewma"]["crra_point"]
        gap_total = d_oracle - d_ewma
        for lane_name in ["bayesian_sig", "hetero_kalman", "blr_kf_leadlag"]:
            if lane_name in by_name:
                d_lane = by_name[lane_name]["crra_point"]
                gap_closed = d_lane - d_ewma
                frac = gap_closed / gap_total if abs(gap_total) > 1e-12 else float("nan")
                print(
                    f"  {lane_name:22s}: "
                    f"CRRA = {d_lane:+.5f}   gain over ewma = {gap_closed:+.5f}   "
                    f"fraction of gap closed = {frac:+.3f}"
                )

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        # Left: CRRA score with CI
        ax = axes[0]
        names = [s["name"] for s in summaries]
        crra = [s["crra_point"] for s in summaries]
        q05 = [s["crra_q05"] for s in summaries]
        q95 = [s["crra_q95"] for s in summaries]
        lower = [c - l for c, l in zip(crra, q05)]
        upper = [u - c for u, c in zip(q95, crra)]
        ax.bar(names, crra, yerr=[lower, upper], capsize=8,
               color=["tab:green", "tab:orange", "tab:blue"], alpha=0.7)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel("CRRA score (mean − 0.5(γ−1) Var)")
        ax.set_title("Held-out CRRA-style score per lane (90% CI)")
        ax.grid(alpha=0.3, axis="y")

        # Middle: V_hat (POST-update) tracking across lanes, representative seed
        ax = axes[1]
        if all_evals:
            # Choose a seed with a V_true trajectory that has visible movement,
            # so the filter differences are interpretable.
            ev_oracle = next((e for e in all_evals if e.name == "value_grad_oracle"), None)
            if ev_oracle is not None and ev_oracle.V_true_post_history.size:
                per_seed_range = (
                    ev_oracle.V_true_post_history.max(axis=1)
                    - ev_oracle.V_true_post_history.min(axis=1)
                )
                # pick the median-range seed (not the most extreme; not a flat one)
                k_sorted = np.argsort(per_seed_range)
                k = int(k_sorted[len(k_sorted) // 2])
            else:
                k = 0
            T = cfg.T_steps
            t_axis = np.arange(T)
            if ev_oracle is not None:
                ax.plot(
                    t_axis, ev_oracle.V_true_post_history[k], "k-", lw=1.6,
                    label="V_true (post-step)",
                )
            lane_colors = {
                "value_grad_ewma": "tab:orange",
                "value_grad_bayesian_sig": "tab:blue",
                "value_grad_hetero_kalman": "tab:purple",
                "value_grad_blr_kf_leadlag": "tab:red",
            }
            for ev in all_evals:
                if ev.name == "value_grad_oracle":
                    continue
                c = lane_colors.get(ev.name, None)
                lbl = ev.name.replace("value_grad_", "") + "  (post-update)"
                Vh_plot = (
                    ev.V_hat_post_history[k]
                    if ev.V_hat_post_history.size
                    else ev.V_hat_history[k]
                )
                ax.plot(t_axis, Vh_plot, color=c, lw=1.1, alpha=0.85, label=lbl)
            ax.set_xlabel("step t")
            ax.set_ylabel("V")
            ax.set_title(f"V tracking (POST-update) on test seed {k} (median-range seed)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Right: Δ logW distribution
        ax = axes[2]
        palette = {
            "value_grad_oracle": "tab:green",
            "value_grad_ewma": "tab:orange",
            "value_grad_bayesian_sig": "tab:blue",
            "value_grad_hetero_kalman": "tab:purple",
            "value_grad_blr_kf_leadlag": "tab:red",
        }
        for ev in all_evals:
            c = palette.get(ev.name, None)
            lbl = ev.name.replace("value_grad_", "")
            ax.hist(ev.delta_logW, bins=22, alpha=0.35, density=True, label=lbl, color=c)
        ax.axvline(0.0, color="k", lw=0.6, linestyle=":")
        ax.set_xlabel("Δ logW held-out (paired vs u=0)")
        ax.set_ylabel("density")
        ax.set_title("Per-seed paired improvement")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Three-lane V-filter comparison in Heston value-gradient controller  "
            f"(γ={cfg.gamma}, ρ={cfg.rho}, T={cfg.T_steps} days)",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = os.path.join(HERE, "heston_three_lane.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print()
        print(f"Saved plot: {out_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


def main():
    _three_lane_comparison()
    return
    # Legacy single-lane path kept below, unused by default.
    cfg = VGConfig()
    env = HestonMertonEnv(rho=cfg.rho, gamma=cfg.gamma)
    psi_A = PsiA(env, cfg.T_steps)
    psi_B = PsiB(env, cfg.T_steps)
    terminal_target = TerminalZero()
    stage_cost = ObservedLogReturnStageCost()

    print("=" * 120)
    print("HESTON LOCAL VALUE-GRADIENT CONTROLLER")
    print(f"  rho={cfg.rho}  gamma={cfg.gamma}  horizon={cfg.T_steps} days  dt={cfg.dt:.5f}")
    print(f"  V0 uniform [{cfg.V0_low}, {cfg.V0_high}]  u_max={cfg.u_max}")
    print(f"  n_train_episodes={cfg.n_train}  n_test_seeds={cfg.n_test}")
    print(f"  V_floor_for_pi_ref={cfg.V_floor_for_pi_ref}")
    print(f"  Terminal target: {terminal_target.name}  "
          f"(Kelly / log-utility residual)")
    print(f"  Stage cost:      {stage_cost.name}")
    print(f"  Primary eval:    CRRA score = mean(Δ logW) − 0.5(γ−1) Var(Δ logW)  "
          f"[γ={cfg.gamma}]")
    print("=" * 120)

    # ---- Collect training episodes (ONCE; psi computed per-stage) ----
    print()
    print("Collecting training episodes (shared across stages)...")
    # We need to re-run with stage-specific psi for the lift
    print("  Stage A...")
    eps_A = collect_training_episodes(
        env, cfg, psi_A, n_episodes=cfg.n_train, base_seed=1_000,
    )
    print("  Stage B...")
    eps_B = collect_training_episodes(
        env, cfg, psi_B, n_episodes=cfg.n_train, base_seed=1_000,
    )

    # ---- Fit value-gradient models ----
    print()
    print("Fitting Stage A value-gradient model...")
    model_A = backward_value_iteration(
        eps_A, terminal_target=terminal_target, stage_cost=stage_cost,
        ridge_transition=cfg.ridge_transition, ridge_value=cfg.ridge_value,
    )
    print(f"  terminal r2 approx = from backward fit (see training_value_r2 below)")
    print("Fitting Stage B value-gradient model...")
    model_B = backward_value_iteration(
        eps_B, terminal_target=terminal_target, stage_cost=stage_cost,
        ridge_transition=cfg.ridge_transition, ridge_value=cfg.ridge_value,
    )

    # ---- Training R^2 per step ----
    r2_A = training_value_r2(model_A, eps_A, terminal_target, stage_cost)
    r2_B = training_value_r2(model_B, eps_B, terminal_target, stage_cost)
    print()
    print("Training R^2 of V_t fit per step (higher is better):")
    for key_t in [0, cfg.T_steps // 4, cfg.T_steps // 2, 3 * cfg.T_steps // 4, cfg.T_steps]:
        print(f"  t={key_t:3d}:  Stage A R² = {r2_A.get(key_t, float('nan')):.4f}   "
              f"Stage B R² = {r2_B.get(key_t, float('nan')):.4f}")

    # ---- Held-out evaluation ----
    print()
    print("Held-out evaluation (paired CRN vs myopic reference)...")
    eval_myopic = evaluate_controller_paired(
        env, cfg, psi_A, model=None, base_seed=500_000, name="myopic_only",
    )
    eval_A = evaluate_controller_paired(
        env, cfg, psi_A, model=model_A, base_seed=500_000, name="value_gradient_A",
    )
    eval_B = evaluate_controller_paired(
        env, cfg, psi_B, model=model_B, base_seed=500_000, name="value_gradient_B",
    )

    print()
    print("=" * 120)
    print("HELD-OUT COMPARISON  (paired vs myopic u=0 under identical noise)")
    print("=" * 120)
    print(
        f"{'controller':28s} | CRRA score [90% CI]              "
        f"| mean Δ logW | Var Δ logW   | abst_rate | u_active mean, std"
    )
    print("-" * 120)
    print(_format_row("myopic_only", eval_myopic, cfg.gamma))
    print(_format_row("value_gradient_A", eval_A, cfg.gamma))
    print(_format_row("value_gradient_B", eval_B, cfg.gamma))

    # ---- State-dependence summary ----
    print()
    print("=" * 120)
    print("STATE-DEPENDENCE  (u_t variability across test seeds and time steps)")
    print("  Large std(u across seeds at fixed t)  =>  state-dependent policy")
    print("=" * 120)
    for name, ev in [("value_gradient_A", eval_A), ("value_gradient_B", eval_B)]:
        u_h = ev.u_history        # (n_test, T)
        per_step_std = u_h.std(axis=0, ddof=1)       # (T,) across seeds at each t
        per_seed_std = u_h.std(axis=1, ddof=1)       # (n_test,) across time at each seed
        print(f"  {name}:")
        print(f"     time-averaged  std(u across seeds at fixed t) = "
              f"{float(per_step_std.mean()):.4f}")
        print(f"     seed-averaged  std(u across time at fixed seed) = "
              f"{float(per_seed_std.mean()):.4f}")

    # ---- Decomposition ----
    print()
    print("=" * 120)
    print("DECOMPOSITION  value_gradient_A vs value_gradient_B  (CRRA score)")
    print("=" * 120)
    y_A, y_B = eval_A.delta_logW, eval_B.delta_logW
    m_A, v_A = float(np.mean(y_A)), float(np.var(y_A, ddof=1))
    m_B, v_B = float(np.mean(y_B)), float(np.var(y_B, ddof=1))
    crra_A = _crra_score_bootstrap(y_A, cfg.gamma, rng=np.random.RandomState(7))["point"]
    crra_B = _crra_score_bootstrap(y_B, cfg.gamma, rng=np.random.RandomState(7))["point"]
    print(f"  mean Δ logW:  A = {m_A:+.5f}   B = {m_B:+.5f}   "
          f"Δ(B−A) = {m_B - m_A:+.5f}")
    print(f"  Var  Δ logW:  A = {v_A:.4e}   B = {v_B:.4e}   "
          f"Δ(B−A) = {v_B - v_A:+.4e}")
    print(f"  CRRA score :  A = {crra_A:+.5f}   B = {crra_B:+.5f}   "
          f"Δ(B−A) = {crra_B - crra_A:+.5f}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: u trajectories for a few test seeds (Stage A)
        ax = axes[0]
        t_axis = np.arange(cfg.T_steps)
        for k in range(0, cfg.n_test, max(1, cfg.n_test // 6)):
            ax.plot(t_axis, eval_A.u_history[k], lw=0.8, alpha=0.6,
                    label=f"seed {k} V₀={eval_A.per_seed_V_at_decision[k]:.3f}")
        ax.axhline(0.0, color="gray", lw=0.6)
        ax.set_xlabel("step t")
        ax.set_ylabel("u_t*  (Stage A)")
        ax.set_title(f"value_gradient_A  state-dependent u_t over the horizon")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

        # Right: Δ logW distribution per controller
        ax = axes[1]
        bins = 25
        ax.hist(eval_A.delta_logW, bins=bins, alpha=0.5, label="value_gradient_A",
                density=True, color="tab:blue")
        ax.hist(eval_B.delta_logW, bins=bins, alpha=0.5, label="value_gradient_B",
                density=True, color="tab:red")
        ax.axvline(0.0, color="k", lw=0.6, linestyle=":")
        ax.set_xlabel("Δ logW = logW(policy) − logW(u=0)")
        ax.set_ylabel("density")
        ax.set_title("Held-out paired improvement distribution")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Heston local value-gradient controller  "
            f"(γ={cfg.gamma}, ρ={cfg.rho}, T={cfg.T_steps} days)",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = os.path.join(HERE, "heston_value_gradient.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print()
        print(f"Saved plot: {out_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
