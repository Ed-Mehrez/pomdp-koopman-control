r"""
Generic state transforms for partially observed control.

Purpose
-------
Turn a possibly-non-ergodic observation stream into a lower-dimensional
lifted state on which prediction and control are better behaved.

All transforms share a single minimal protocol:

    class StateTransform(Protocol):
        dim: int                      # dimension of lifted state
        def reset(self) -> None: ...
        def update(self, y: ndarray, dt: float) -> ndarray: ...
        def current(self) -> ndarray: ...

`y` is an observation (e.g., an increment dY or a level Y; the transform
defines which).  `dt` is the time step.  `update` returns the new lifted
state.  `current` returns the last lifted state without consuming a new
observation.

The generic layer makes no finance-specific assumptions.  Transforms
that use domain knowledge (e.g., a linear-Gaussian Kalman filter for an
OU factor) are constructed with explicit model parameters.

Currently used by:
    - experiments/science_poc/envs/latent_ou_drift.py  (Approach 2 env)
    - experiments/science_poc/latent_ou_representation_demo.py

Expect refactor when a third non-adapter caller appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np


# ==========================================================================
# Protocol
# ==========================================================================


class StateTransform(Protocol):
    r"""Online transform of an observation stream into a lifted state.

    Implementations are responsible for stating in their docstring whether
    `update(y, dt)` expects `y` to be a LEVEL (e.g., price) or an INCREMENT
    (e.g., dS).  Mixing the two silently is a bug — the transform must
    document which convention it consumes.
    """
    dim: int
    name: str

    def reset(self) -> None: ...
    def update(self, y: np.ndarray, dt: float) -> np.ndarray: ...
    def current(self) -> np.ndarray: ...


# ==========================================================================
# Raw observation (identity transform)
# ==========================================================================


class RawLevel:
    r"""Identity transform: lifted state is the raw observed level.

    For a non-ergodic level process, this is intentionally a BAD state --
    useful only as a baseline for comparison.
    """
    name = "raw_level"

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._state = np.zeros(self.dim)

    def reset(self) -> None:
        self._state = np.zeros(self.dim)

    def update(self, y: np.ndarray, dt: float) -> np.ndarray:
        self._state = np.atleast_1d(np.asarray(y, dtype=float)).reshape(self.dim)
        return self.current()

    def current(self) -> np.ndarray:
        return self._state.copy()


# ==========================================================================
# Exponentially fading memory, level 1 (continuous-time EMA on increments)
# ==========================================================================


class EFMLevel1:
    r"""Level-1 exponentially fading-memory transform of INCREMENTS.

    Mathematical definition (Prop 7.3 of theory_ergodic_signatures.md):

        Z_t = integral_{-infty}^{t} exp(-lambda (t - s)) dY_s
            ==>  dZ_t = -lambda * Z_t * dt + dY_t.

    Discrete update with step dt and forgetting rho = exp(-lambda * dt):

        Z_{t+dt} = rho * Z_t + dY_{t+dt}.

    This transform CONSUMES increments.  Pass `y = dY` (not Y) to `update`.

    Under time-augmented Brownian input, Z_t is OU, stationary, Markov,
    with spectral gap lambda (Prop 5.4 of the theory doc).  For general
    stationary-increment observables, Z_t is stationary (Prop 7.3).
    """

    def __init__(self, dim: int, lam: float, name: str = "efm_level_1"):
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.dim = int(dim)
        self.lam = float(lam)
        self.name = name
        self._z = np.zeros(self.dim)

    def reset(self) -> None:
        self._z = np.zeros(self.dim)

    def update(self, y: np.ndarray, dt: float) -> np.ndarray:
        rho = float(np.exp(-self.lam * dt))
        dy = np.atleast_1d(np.asarray(y, dtype=float)).reshape(self.dim)
        self._z = rho * self._z + dy
        return self.current()

    def current(self) -> np.ndarray:
        return self._z.copy()


# ==========================================================================
# Scalar Kalman-Bucy filter for a 1D linear-Gaussian latent factor
# ==========================================================================


@dataclass
class KalmanLinearConfig:
    r"""Model used by `KalmanLinearFilter`.

    Hidden factor:   dX_t = -theta * X_t * dt + sigma_X * dW_X
    Observation:     dY_t =  X_t * dt + sigma_Y * dW_Y
    (continuous-time Kalman-Bucy, stationary form)

    This is the canonical linear-Gaussian latent-drift model; knowing
    these parameters is a STRONG assumption (the "oracle" baseline).
    """
    theta: float
    sigma_X: float
    sigma_Y: float


class KalmanLinearFilter:
    r"""Oracle Kalman-Bucy filter for the scalar latent-drift model.

    Continuous update over a step of length dt (explicit Euler on the
    filtering SDE):

        dXhat = -theta * Xhat * dt + (P / sigma_Y**2) * (dY - Xhat * dt)
        dP    = (-2*theta*P + sigma_X**2 - P**2 / sigma_Y**2) * dt

    This transform CONSUMES increments (`y = dY`).  State is (Xhat, P).

    NOT part of the generic representation claim: it's the oracle we
    compare EFM against.  Using it requires knowing the model parameters.
    """
    name = "kalman_linear_oracle"

    def __init__(self, config: KalmanLinearConfig):
        self.dim = 2  # (Xhat, P)
        self.config = config
        self._xhat: float = 0.0
        # Stationary variance as initial P: solves 2*theta*P = sigma_X^2 - P^2/sigma_Y^2
        c = self.config
        disc = c.theta * c.theta + (c.sigma_X * c.sigma_X) / (c.sigma_Y * c.sigma_Y)
        self._P_stat: float = c.sigma_Y * c.sigma_Y * (-c.theta + float(np.sqrt(disc)))
        self._P: float = self._P_stat

    def reset(self) -> None:
        self._xhat = 0.0
        self._P = self._P_stat

    def update(self, y: np.ndarray, dt: float) -> np.ndarray:
        dY = float(np.atleast_1d(np.asarray(y, dtype=float)).flatten()[0])
        c = self.config
        innov = dY - self._xhat * dt
        gain = self._P / (c.sigma_Y * c.sigma_Y)
        self._xhat = self._xhat - c.theta * self._xhat * dt + gain * innov
        dP = (-2.0 * c.theta * self._P + c.sigma_X * c.sigma_X
              - self._P * self._P / (c.sigma_Y * c.sigma_Y))
        self._P = max(self._P + dP * dt, 1e-12)
        return self.current()

    def current(self) -> np.ndarray:
        return np.array([self._xhat, self._P], dtype=float)


# ==========================================================================
# Spectral-gap diagnostic (empirical)
# ==========================================================================


def empirical_autocorrelation(
    series: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    r"""Biased autocorrelation of a scalar stationary-ish series up to max_lag.

    Returns an (max_lag + 1,) array with entry k = corr(x_t, x_{t+k})
    estimated by sample covariance.  Useful as a Bakry-Emery / spectral-gap
    empirical sanity check: for a transform claimed to have gap rho, the
    log-autocorrelation should decay approximately linearly with slope -rho
    (or -rho*dt if working in step units).
    """
    x = np.asarray(series, dtype=float).flatten()
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var <= 0.0:
        return np.zeros(max_lag + 1)
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    N = x.size
    for k in range(1, max_lag + 1):
        if k >= N:
            break
        out[k] = float(np.mean(x[:N - k] * x[k:])) / var
    return out
