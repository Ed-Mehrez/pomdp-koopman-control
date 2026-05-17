r"""
Two-scale partially observed double-well benchmark.

The slow coordinate X_t is observed.  A hidden OU factor H_t modulates the
double-well drift, so X_t alone is not Markov:

    dX_t = (X_t - X_t^3 + beta H_t) dt + sigma_X dW_X,
    dH_t = -kappa_H H_t dt + sigma_H dW_H.

This is the science-first analogue of the latent-volatility finance examples:
the observed coordinate has memory because an unobserved ergodic factor drives
its local generator.  Path lifts such as delays or fading-memory transforms of
dX_t should recover part of the missing Markov state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class TwoScaleDoubleWellConfig:
    """Parameters for the two-scale double-well simulator."""

    dt: float = 0.01
    T: float = 8.0
    beta: float = 1.4
    sigma_X: float = 0.15
    kappa_H: float = 1.5
    sigma_H: float = 1.0
    x0_noise: float = 0.15

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def hidden_stationary_var(self) -> float:
        return float(self.sigma_H ** 2 / (2.0 * self.kappa_H))

    @property
    def hidden_timescale(self) -> float:
        return float(1.0 / self.kappa_H)


def double_well_drift(x: np.ndarray, h: np.ndarray, config: TwoScaleDoubleWellConfig) -> np.ndarray:
    """True drift of the observed slow coordinate."""

    return x - x ** 3 + config.beta * h


def simulate_two_scale_double_well(
    config: TwoScaleDoubleWellConfig,
    n_paths: int,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    h0: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    r"""Simulate independent paths with Euler-Maruyama.

    Returns arrays with shape `(n_paths, n_steps)` for pre-step quantities
    `X`, `H`, `drift_X`, and increments `dX`.
    """

    rng = np.random.RandomState(seed)
    n_paths = int(n_paths)
    n_steps = config.n_steps
    dt = float(config.dt)
    sqrt_dt = float(np.sqrt(dt))

    if x0 is None:
        wells = rng.choice(np.array([-1.0, 1.0]), size=n_paths)
        x = wells + config.x0_noise * rng.standard_normal(n_paths)
    else:
        x = np.asarray(x0, dtype=float).reshape(n_paths).copy()

    if h0 is None:
        h = np.sqrt(config.hidden_stationary_var) * rng.standard_normal(n_paths)
    else:
        h = np.asarray(h0, dtype=float).reshape(n_paths).copy()

    X = np.zeros((n_paths, n_steps), dtype=float)
    H = np.zeros((n_paths, n_steps), dtype=float)
    drift_X = np.zeros((n_paths, n_steps), dtype=float)
    dX = np.zeros((n_paths, n_steps), dtype=float)

    for t in range(n_steps):
        bx = double_well_drift(x, h, config)
        dW_x = sqrt_dt * rng.standard_normal(n_paths)
        dW_h = sqrt_dt * rng.standard_normal(n_paths)
        dx = bx * dt + config.sigma_X * dW_x
        dh = -config.kappa_H * h * dt + config.sigma_H * dW_h

        X[:, t] = x
        H[:, t] = h
        drift_X[:, t] = bx
        dX[:, t] = dx

        x = x + dx
        h = h + dh

    return {"X": X, "H": H, "drift_X": drift_X, "dX": dX}

