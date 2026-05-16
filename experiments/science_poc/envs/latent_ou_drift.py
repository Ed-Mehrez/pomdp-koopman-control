r"""
Latent-OU-drift environment: a canonical POMDP with an ergodic hidden
factor, a non-stationary cumulative observation, and exact closed-form
optimal filter and controller.

Model
-----
Hidden factor X_t (scalar, mean-zero OU, ergodic):

    dX_t = -theta * X_t * dt + sigma_X * dW_X.

Observed level S_t (cumulative; non-stationary):

    dS_t = X_t * dt + sigma_Y * dW_Y,
    S_0  = 0.

The observation INCREMENT dS_t is a noisy measurement of X_t.  The
observed LEVEL S_t has Var(S_t - S_0) -> infinity as t -> infinity (the
variance is bounded below by sigma_Y^2 * t), so `S_t` has no invariant
distribution.  Compare theory_ergodic_signatures_and_horizon_selection.md
Prop 7.1 for the structurally analogous Heston log-price argument.

Control problem (linear-Gaussian, quadratic-cost):

    maximize  E[ integral_0^T ( u_t * X_t - c * u_t**2 ) dt ]
    over observation-adapted processes u_t.

The factor of c>0 penalizes action magnitude; the linear reward
`u_t * X_t` pays to align action with the hidden factor.

Optimal control (see module-level NOTES)
---------------------------------------
By linear-Gaussian Kalman-Bucy separation + myopic LQR pointwise:

    u_t*  =  Xhat_t / (2c),
    Xhat_t  = E[X_t | (S_s)_{s<=t}]  (Kalman-Bucy filter).

The corresponding optimal value is achieved by the Kalman-filtered state.
Using the RAW level S_t as the state (e.g., u_t = S_t / (2c * T_eff))
IGNORES the noise accumulation and is provably worse.

Why this env is useful for Approach 2
-------------------------------------
- Hidden factor is ergodic (spectral gap = theta).
- Observed level is non-ergodic (variance diverges).
- Optimal filter and optimal control are closed-form.
- Raw observation level is a BAD state; filtered or fading-memory
  transform of increments is a GOOD state.
- Finance vocabulary is absent from the mathematical specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np


# ==========================================================================
# Config and env
# ==========================================================================


@dataclass(frozen=True)
class LatentOUConfig:
    r"""Immutable parameters of the latent-OU-drift model.

    theta    : hidden-factor mean-reversion rate  (spectral gap = theta)
    sigma_X  : hidden-factor diffusion
    sigma_Y  : observation diffusion on dS
    c        : quadratic action cost
    T        : horizon (time units)
    dt       : time step
    """
    theta: float = 1.0
    sigma_X: float = 0.5
    sigma_Y: float = 0.5
    c: float = 1.0
    T: float = 4.0
    dt: float = 0.02

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def stationary_var_X(self) -> float:
        r"""Stationary variance of X: sigma_X^2 / (2 theta)."""
        return float(self.sigma_X ** 2 / (2.0 * self.theta))

    @property
    def stationary_kalman_P(self) -> float:
        r"""Stationary Kalman variance P_infty solves
        0 = -2 theta P + sigma_X^2 - P^2 / sigma_Y^2.
        """
        a = 1.0 / (self.sigma_Y ** 2)
        b = 2.0 * self.theta
        c_ = -self.sigma_X ** 2
        disc = b * b - 4 * a * c_
        return float((-b + np.sqrt(disc)) / (2 * a))


class LatentOUEnv:
    r"""Simulator for the latent-OU-drift POMDP.

    Separate RNG for process noise vs observation noise vs action noise is
    exposed so that paired-noise (CRN) experiments can match noise across
    runs with different controllers.
    """

    def __init__(self, config: LatentOUConfig, seed: int = 0):
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.X: float = 0.0
        self.S: float = 0.0
        self.t: int = 0

    def reset(self, X0: float = 0.0, S0: float = 0.0, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.RandomState(self.seed)
        self.X = float(X0)
        self.S = float(S0)
        self.t = 0

    def step(
        self,
        u: float,
        zX: Optional[float] = None,
        zY: Optional[float] = None,
    ) -> Tuple[float, float]:
        r"""Advance one dt.  Returns (dS, instantaneous_reward).

        If (zX, zY) are provided, they are used as the Gaussian draws
        (enables paired-noise across controllers).  Otherwise draws are
        taken from the env's RNG.
        """
        c = self.config
        dt = c.dt
        if zX is None:
            zX = float(self.rng.standard_normal())
        if zY is None:
            zY = float(self.rng.standard_normal())
        dWX = np.sqrt(dt) * zX
        dWY = np.sqrt(dt) * zY

        # Evolve hidden factor (Euler-Maruyama)
        X_new = self.X - c.theta * self.X * dt + c.sigma_X * dWX
        # Observation increment uses pre-step X (standard Ito discretization)
        dS = self.X * dt + c.sigma_Y * dWY
        # Reward increment
        reward = (u * self.X - c.c * u * u) * dt

        self.X = float(X_new)
        self.S = float(self.S + dS)
        self.t += 1
        return float(dS), float(reward)

    def rollout(
        self,
        policy: Callable[[Dict[str, float], Dict], float],
        transform_state_fn: Optional[Callable[[Dict[str, float]], Dict]] = None,
        init_state: Optional[Dict] = None,
        noise: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Run one episode from t=0 to t=T.

        policy(context, transform_state) -> u_t:
            context is {"t_step": t, "t_time": t*dt, "S": S, "dS": dS_last}
            transform_state is the output of `transform_state_fn` applied
            to the pre-step context; if `transform_state_fn` is None the
            empty dict is passed.
        noise: optional (n_steps, 2) array of (zX, zY); used for CRN.

        Returns dict of per-step arrays:
            X_true, S, u, dS, reward
        """
        self.reset(seed=self.seed if init_state is None else init_state.get("seed", self.seed))
        n = self.config.n_steps
        X_hist = np.zeros(n)
        S_hist = np.zeros(n)
        u_hist = np.zeros(n)
        dS_hist = np.zeros(n)
        r_hist = np.zeros(n)
        dS_last = 0.0
        for t in range(n):
            context = {
                "t_step": t,
                "t_time": t * self.config.dt,
                "S": self.S,
                "dS": dS_last,
            }
            trans_state = transform_state_fn(context) if transform_state_fn else {}
            u_t = float(policy(context, trans_state))
            if noise is not None:
                zX, zY = float(noise[t, 0]), float(noise[t, 1])
            else:
                zX = zY = None
            X_hist[t] = self.X
            S_hist[t] = self.S
            u_hist[t] = u_t
            dS, r = self.step(u_t, zX=zX, zY=zY)
            dS_hist[t] = dS
            r_hist[t] = r
            dS_last = dS
        return {
            "X_true": X_hist,
            "S": S_hist,
            "u": u_hist,
            "dS": dS_hist,
            "reward": r_hist,
        }


# ==========================================================================
# Oracle baselines (closed-form references)
# ==========================================================================


def closed_form_stationary_value(config: LatentOUConfig) -> float:
    r"""Stationary per-unit-time reward under the optimal Kalman+LQG policy
    in the infinite-horizon limit (diagnostic; not used for gating).

    Under stationary Kalman (filter variance P_infty) and LQG u_t* = Xhat/(2c),
    the stationary expected per-step reward is

        E[ u_t* * X_t - c * (u_t*)**2 ] * dt
      = dt * ( E[Xhat * X] / (2c)  -  c * E[Xhat^2] / (4 c^2) )
      = dt * ( E[Xhat^2] / (2c)  -  E[Xhat^2] / (4c) )                    (because X = Xhat + err, err ⟂ Xhat)
      = dt * E[Xhat^2] / (4c)

    where E[Xhat^2] = Var(X) - P_infty under linear-Gaussian filter.
    """
    var_X = config.stationary_var_X
    P = config.stationary_kalman_P
    var_Xhat = max(var_X - P, 0.0)
    return float(var_Xhat / (4.0 * config.c))


def raw_level_scale(config: LatentOUConfig) -> float:
    r"""Effective horizon scale used by a 'raw level' heuristic.

    The raw-level baseline is u_t = S_t / (2 c * T_eff); T_eff here is
    the time-constant 1/theta (the autocorrelation scale of X), which is
    the horizon over which S accumulates "about one factor's worth of
    drift" on average.
    """
    return 1.0 / config.theta


# ==========================================================================
# Policies (callables bound to a transform/state)
# ==========================================================================


def policy_oracle_lqg(X_true: float, c: float) -> float:
    """u* = X / (2c) when we see the hidden factor directly."""
    return float(X_true / (2.0 * c))


def policy_kalman_lqg(Xhat: float, c: float) -> float:
    """u* = Xhat / (2c) when Xhat comes from the Kalman-Bucy filter."""
    return float(Xhat / (2.0 * c))


def policy_efm_linear(Z: float, gain: float) -> float:
    """Linear policy in the EFM-level-1 coordinate: u = gain * Z.

    The optimal gain for EFM-as-approximate-Kalman depends on the
    relation between the EFM decay lambda and the hidden-factor theta;
    this function exposes `gain` so the demo can sweep it.
    """
    return float(gain * Z)


def policy_raw_level(S: float, T_eff: float, c: float) -> float:
    """Heuristic raw-level baseline: u = S / (2c T_eff).

    This is the baseline that SHOULD perform poorly because S is
    non-stationary and its variance accumulates with t.
    """
    return float(S / (2.0 * c * max(T_eff, 1e-12)))
