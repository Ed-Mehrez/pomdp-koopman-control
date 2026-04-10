"""Numerical BBG benchmark for the single-option Heston OMM env.

This module implements the reduced BBG 2020 controller for the repo's current
single-option setting. Under the constant-vega approximation, no P/Q drift gap,
and a fixed inventory grid q in {-Q, ..., Q}, the BBG HJB collapses to a
backward ODE system in time and inventory only.
"""

from __future__ import annotations

from collections.abc import Callable
from math import e, isfinite
from typing import Any

import numpy as np

from .controllers import NO_QUOTE_ASK
from .env import OptionMMAction, OptionMMState, OptionMarketMakingEnv
from .inventory_variance import compute_constant_vega_per_share


EpisodeController = Callable[[OptionMMState, Any | None], OptionMMAction]


def _h_exponential(
    p: float,
    *,
    base_intensity: float,
    distance_slope: float,
) -> float:
    """Hamiltonian for exponential intensity λ(δ) = A exp(-k δ)."""
    exponent = float(np.clip(-distance_slope * p, -700.0, 700.0))
    return (base_intensity / (e * distance_slope)) * np.exp(exponent)


def solve_bbg_value_table(
    env: OptionMarketMakingEnv,
    initial_state: OptionMMState,
    *,
    gamma: float,
    max_inventory: int,
    substeps_per_step: int = 16,
) -> np.ndarray:
    """Solve the reduced BBG backward ODE on the q-grid by explicit Euler."""
    if gamma < 0.0 or not isfinite(gamma):
        raise ValueError("gamma must be nonnegative and finite")
    if max_inventory <= 0:
        raise ValueError("max_inventory must be positive")
    if substeps_per_step <= 0:
        raise ValueError("substeps_per_step must be positive")

    q_grid = np.arange(-max_inventory, max_inventory + 1, dtype=float)
    num_q = q_grid.size
    values = np.zeros((env.horizon_steps + 1, num_q), dtype=float)

    vega_per_share = compute_constant_vega_per_share(env, initial_state)
    vega_contract = env.contract.contract_multiplier * vega_per_share
    penalty_scale = gamma * (env.heston.xi ** 2) * (vega_contract ** 2) / 8.0
    base_intensity = env.fills.base_intensity
    distance_slope = env.fills.distance_slope
    dt = env.dt / substeps_per_step

    for step_index in range(env.horizon_steps - 1, -1, -1):
        current_values = values[step_index + 1].copy()
        for _ in range(substeps_per_step):
            previous_values = current_values.copy()
            for q_index, q in enumerate(q_grid):
                penalty = penalty_scale * (q ** 2)
                bid_term = 0.0
                ask_term = 0.0
                if q_index < num_q - 1:
                    bid_term = _h_exponential(
                        previous_values[q_index] - previous_values[q_index + 1],
                        base_intensity=base_intensity,
                        distance_slope=distance_slope,
                    )
                if q_index > 0:
                    ask_term = _h_exponential(
                        previous_values[q_index] - previous_values[q_index - 1],
                        base_intensity=base_intensity,
                        distance_slope=distance_slope,
                    )
                rhs = penalty - bid_term - ask_term
                current_values[q_index] = previous_values[q_index] - dt * rhs
        values[step_index] = current_values
    return values


def solve_bbg_quote_tables(
    env: OptionMarketMakingEnv,
    initial_state: OptionMMState,
    *,
    gamma: float,
    max_inventory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q-grid and per-step bid/ask distances for the BBG controller."""
    values = solve_bbg_value_table(
        env,
        initial_state,
        gamma=gamma,
        max_inventory=max_inventory,
    )
    q_grid = np.arange(-max_inventory, max_inventory + 1, dtype=int)
    num_q = q_grid.size
    bid_distances = np.full((env.horizon_steps, num_q), np.inf, dtype=float)
    ask_distances = np.full((env.horizon_steps, num_q), np.inf, dtype=float)
    base_half_spread = 1.0 / env.fills.distance_slope
    multiplier = env.contract.contract_multiplier

    for step_index in range(env.horizon_steps):
        step_values = values[step_index]
        for q_index in range(num_q):
            if q_index < num_q - 1:
                bid_distances[step_index, q_index] = max(
                    0.0,
                    ((step_values[q_index] - step_values[q_index + 1]) / multiplier)
                    + base_half_spread,
                )
            if q_index > 0:
                ask_distances[step_index, q_index] = max(
                    0.0,
                    ((step_values[q_index] - step_values[q_index - 1]) / multiplier)
                    + base_half_spread,
                )
    return q_grid, bid_distances, ask_distances


def make_bbg_numerical(
    env: OptionMarketMakingEnv,
    initial_state: OptionMMState,
    *,
    gamma: float,
    max_inventory: int,
) -> EpisodeController:
    """Factory for the finite-γ BBG reduced-HJB controller."""
    q_grid, bid_distances, ask_distances = solve_bbg_quote_tables(
        env,
        initial_state,
        gamma=gamma,
        max_inventory=max_inventory,
    )
    min_q = int(q_grid[0])
    max_q = int(q_grid[-1])

    def controller(state: OptionMMState, history: Any | None = None) -> OptionMMAction:
        del history
        step_index = min(max(state.step_index, 0), env.horizon_steps - 1)
        q = int(np.clip(state.option_inventory, min_q, max_q))
        q_index = q - min_q
        bid_distance = bid_distances[step_index, q_index]
        ask_distance = ask_distances[step_index, q_index]
        bid_price = 0.0 if not np.isfinite(bid_distance) else max(state.option_mid - bid_distance, 0.0)
        ask_price = (
            NO_QUOTE_ASK
            if not np.isfinite(ask_distance)
            else state.option_mid + ask_distance
        )
        return OptionMMAction(
            bid_price=bid_price,
            ask_price=ask_price,
            hedge_trade=-state.net_delta,
        )

    return controller
