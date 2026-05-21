r"""
Two-scale Lorenz-96 simulator (Wilks 2005 / Arnold-Moroz-Palmer form).

Slow variables X_k (k=0..K-1) and fast variables Y_{j,k} (j=0..J-1, k=0..K-1)
evolve as

    dX_k/dt = X_{k-1} (X_{k+1} - X_{k-2}) - X_k + F + U_k,
    dY_{j,k}/dt = -c b Y_{j+1,k} (Y_{j+2,k} - Y_{j-1,k}) - c Y_{j,k}
                  + (h c / b) X_k,

with periodic indices and unresolved tendency

    U_k = -(h c / b) sum_j Y_{j,k}.

The fast index runs first within each slow block, so the flattened Y array has
length K*J with cyclic shifts in the fast direction. Integration uses RK4 in
the joint (X, Y) space; only X and U_k are returned for the closure task.

This is the science-side analogue of the partially observed double-well env:
the closure problem is to learn U_k(t) from the observed history of X.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TwoScaleL96Config:
    """Two-scale Lorenz-96 parameters.

    `dt_sim` is the fine RK4 step needed for stable integration of the fast
    subsystem. `obs_subsample` is how many fine steps make up one observation
    interval; the observed `dt = dt_sim * obs_subsample`. Coarse observations
    make the closure problem genuinely non-Markovian on X alone, because U
    integrates fast information over an interval that X cannot resolve.
    """

    K: int = 9          # number of slow variables
    J: int = 8          # number of fast per slow
    F: float = 10.0     # external forcing
    h: float = 1.0      # coupling
    c: float = 10.0     # fast/slow time-scale ratio
    b: float = 10.0     # fast/slow amplitude ratio
    dt_sim: float = 0.005       # fine RK4 step
    obs_subsample: int = 4      # observation step = dt_sim * obs_subsample
    T: float = 16.0             # recorded time horizon (in MTU)
    warmup_T: float = 4.0
    x0_amp: float = 1.0
    y0_amp: float = 0.1

    @property
    def dt(self) -> float:
        return float(self.dt_sim * self.obs_subsample)

    @property
    def n_obs(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def n_sim_warmup(self) -> int:
        return int(round(self.warmup_T / self.dt_sim))


def _rhs(
    X: np.ndarray,
    Y: np.ndarray,
    config: TwoScaleL96Config,
):
    """Compute joint (X, Y) RHS for the two-scale Lorenz-96 system."""

    K, J = config.K, config.J
    F, h, c, b = config.F, config.h, config.c, config.b

    # Slow advection - X has shape (n_paths, K).
    X_m1 = np.roll(X, 1, axis=1)
    X_m2 = np.roll(X, 2, axis=1)
    X_p1 = np.roll(X, -1, axis=1)
    sum_Y = Y.sum(axis=2)                           # (n_paths, K)
    U = -(h * c / b) * sum_Y
    dX = X_m1 * (X_p1 - X_m2) - X + F + U

    # Fast advection - flatten Y to a single periodic array along the fast index.
    Y_flat = Y.reshape(Y.shape[0], -1)              # (n_paths, K*J)
    Y_flat_m1 = np.roll(Y_flat,  1, axis=1)
    Y_flat_p1 = np.roll(Y_flat, -1, axis=1)
    Y_flat_p2 = np.roll(Y_flat, -2, axis=1)
    X_per_fast = np.repeat(X, J, axis=1)            # (n_paths, K*J)

    dY_flat = -c * b * Y_flat_p1 * (Y_flat_p2 - Y_flat_m1) - c * Y_flat \
              + (h * c / b) * X_per_fast
    dY = dY_flat.reshape(Y.shape)
    return dX, dY, U


def _rk4_step(
    X: np.ndarray,
    Y: np.ndarray,
    dt: float,
    config: TwoScaleL96Config,
):
    dX1, dY1, _ = _rhs(X,            Y,            config)
    dX2, dY2, _ = _rhs(X + 0.5 * dt * dX1, Y + 0.5 * dt * dY1, config)
    dX3, dY3, _ = _rhs(X + 0.5 * dt * dX2, Y + 0.5 * dt * dY2, config)
    dX4, dY4, _ = _rhs(X + dt * dX3,       Y + dt * dY3,       config)
    X_new = X + (dt / 6.0) * (dX1 + 2.0 * dX2 + 2.0 * dX3 + dX4)
    Y_new = Y + (dt / 6.0) * (dY1 + 2.0 * dY2 + 2.0 * dY3 + dY4)
    return X_new, Y_new


def simulate_two_scale_l96(
    config: TwoScaleL96Config,
    n_paths: int,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    r"""Simulate two-scale Lorenz-96 paths and return observation-step samples.

    Integration runs at the fine `dt_sim` step. Observations are recorded every
    `obs_subsample` steps. `U_avg` is the *interval-averaged* unresolved
    tendency over the observation window -- the quantity the closure must
    reproduce to match an observed dX_k.

    Returns arrays with shape `(n_paths, n_obs, K)`:
      - X         : observed slow variables (sampled every obs_subsample steps)
      - U         : instantaneous unresolved tendency at sample times
      - U_avg     : interval-averaged unresolved tendency over each obs window
      - drift_X   : interval-averaged full slow drift dX_k/dt
      - resolved  : interval-averaged resolved part on the X stencil
      - dX        : forward difference X[:, t+1] - X[:, t] (last column zero)
    """

    rng = np.random.RandomState(seed)
    K, J = config.K, config.J
    n_paths = int(n_paths)
    sub = int(config.obs_subsample)

    X = config.x0_amp * rng.standard_normal((n_paths, K))
    Y = config.y0_amp * rng.standard_normal((n_paths, K, J))

    # Warmup: integrate at the fine step without recording.
    for _ in range(config.n_sim_warmup):
        X, Y = _rk4_step(X, Y, config.dt_sim, config)

    n_obs = config.n_obs
    X_hist = np.zeros((n_paths, n_obs, K), dtype=float)
    U_hist = np.zeros((n_paths, n_obs, K), dtype=float)
    U_avg_hist = np.zeros((n_paths, n_obs, K), dtype=float)
    drift_avg_hist = np.zeros((n_paths, n_obs, K), dtype=float)
    resolved_avg_hist = np.zeros((n_paths, n_obs, K), dtype=float)

    def _resolved(X_):
        X_m1 = np.roll(X_, 1, axis=1)
        X_m2 = np.roll(X_, 2, axis=1)
        X_p1 = np.roll(X_, -1, axis=1)
        return X_m1 * (X_p1 - X_m2) - X_ + config.F

    for t in range(n_obs):
        dX_full, _, U = _rhs(X, Y, config)
        X_hist[:, t] = X
        U_hist[:, t] = U

        # Accumulate interval averages over the next `sub` fine steps.
        U_acc = U.copy()
        drift_acc = dX_full.copy()
        resolved_acc = _resolved(X).copy()
        for _ in range(sub):
            X, Y = _rk4_step(X, Y, config.dt_sim, config)
            dX_full, _, U_next = _rhs(X, Y, config)
            U_acc += U_next
            drift_acc += dX_full
            resolved_acc += _resolved(X)
        denom = float(sub + 1)
        U_avg_hist[:, t] = U_acc / denom
        drift_avg_hist[:, t] = drift_acc / denom
        resolved_avg_hist[:, t] = resolved_acc / denom

    dX_obs = np.zeros_like(X_hist)
    dX_obs[:, :-1] = X_hist[:, 1:] - X_hist[:, :-1]

    return {
        "X": X_hist,
        "U": U_hist,
        "U_avg": U_avg_hist,
        "drift_X": drift_avg_hist,
        "resolved": resolved_avg_hist,
        "dX": dX_obs,
    }
