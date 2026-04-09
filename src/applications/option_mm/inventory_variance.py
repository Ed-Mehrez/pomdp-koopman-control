"""Inventory-variance estimator factories for the Stage 4 v2 OMM benchmark."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from math import exp, isfinite, log, pi, sqrt
from typing import Any

import numpy as np

from .env import OptionMMState, OptionMarketMakingEnv


SQRT_TWO_PI = sqrt(2.0 * pi)
InventoryVarianceEstimator = Callable[[OptionMMState, Any | None], float]


def _normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / SQRT_TWO_PI


def _compute_constant_vega_per_share(
    env: OptionMarketMakingEnv,
    initial_state: OptionMMState,
) -> float:
    """Black-Scholes call vega per share, frozen at episode start."""
    spot = initial_state.spot
    strike = initial_state.strike
    tau = initial_state.time_to_maturity
    variance = initial_state.variance
    rate = env.heston.rate

    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if tau <= 0.0:
        return 0.0

    sigma = sqrt(max(variance, 1e-16))
    vol_sqrt_tau = sigma * sqrt(tau)
    if vol_sqrt_tau <= 1e-12:
        return 0.0

    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol_sqrt_tau
    return float(spot * sqrt(tau) * _normal_pdf(d1))


def bergault_gueant_heston_estimator(
    env: OptionMarketMakingEnv,
    initial_state: OptionMMState,
) -> InventoryVarianceEstimator:
    """Return the BG Heston inventory-variance estimator, frozen per episode."""
    vega_per_share = _compute_constant_vega_per_share(env, initial_state)
    vega_contract = env.contract.contract_multiplier * vega_per_share
    xi = env.heston.xi
    sigma_sq_inv = float((vega_contract * vega_contract * xi * xi) / 4.0)

    def estimator(state: OptionMMState, history: Any | None = None) -> float:
        del state, history
        return sigma_sq_inv

    return estimator


def empirical_sliding_window_estimator(
    window_length: int,
    env: OptionMarketMakingEnv,
) -> InventoryVarianceEstimator:
    """Return an episode-local estimator of delta-hedged contract PnL variance."""
    if window_length <= 1:
        raise ValueError("window_length must be at least 2")
    if env.dt <= 0.0 or not isfinite(env.dt):
        raise ValueError("dt must be positive and finite")

    contract_multiplier = env.contract.contract_multiplier
    hedged_diffs: deque[float] = deque(maxlen=window_length)
    prev_mid: float | None = None
    prev_spot: float | None = None
    prev_delta: float | None = None

    def estimator(state: OptionMMState, history: Any | None = None) -> float:
        del history
        nonlocal prev_mid, prev_spot, prev_delta
        if prev_mid is None or prev_spot is None or prev_delta is None:
            prev_mid = float(state.option_mid)
            prev_spot = float(state.spot)
            prev_delta = float(state.option_delta)
            return 0.0

        option_change = float(state.option_mid) - prev_mid
        spot_change = float(state.spot) - prev_spot
        hedged_contract_change = contract_multiplier * (
            option_change - prev_delta * spot_change
        )
        hedged_diffs.append(hedged_contract_change)
        prev_mid = float(state.option_mid)
        prev_spot = float(state.spot)
        prev_delta = float(state.option_delta)

        if len(hedged_diffs) < 2:
            return 0.0
        return float(np.var(np.asarray(hedged_diffs, dtype=float)) / env.dt)

    return estimator
