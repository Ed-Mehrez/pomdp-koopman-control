"""BBG reduced HJB solver on the (t, nu, V^pi) grid.

Implements eq. (4) from Baldacci, Bergault & Gueant (2020):

    0 = d_t v + a_P d_nu v + (1/2) nu xi^2 d^2_nu v
        + V^pi (a_P - a_Q) / (2 sqrt(nu))
        - (gamma xi^2 / 8) V^{pi,2}
        + sum_i sum_{j=a,b} z_i 1_{|V^pi - psi(j) z_i V^i| <= V_bar}
          H^{i,j}( (v(..., V^pi) - v(..., V^pi - psi(j) z_i V^i)) / z_i )

with v(T, nu, V^pi) = 0.

Solver: explicit Euler backward in time on a 3D grid.
"""

from __future__ import annotations

from collections.abc import Callable
from math import exp, log, sqrt
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .pricing import bs_call_vega_sqrt_nu, bs_call_price
from .spec import BBGBenchmarkConfig


def _hamiltonian_logistic(
    p: float,
    lambda_i: float,
    vega_i: float,
    alpha: float,
    beta: float,
) -> float:
    """H^{i,j}(p) = sup_{delta} Lambda(delta) * (delta - p).

    For logistic intensity Lambda(delta) = lambda / (1 + exp(alpha + beta*delta/V)).
    Solved numerically via bounded optimization.
    """
    if lambda_i <= 0.0 or vega_i <= 0.0:
        return 0.0

    def neg_obj(delta: float) -> float:
        arg = alpha + beta * delta / vega_i
        arg = float(np.clip(arg, -500.0, 500.0))
        lam = lambda_i / (1.0 + np.exp(arg))
        return -lam * (delta - p)

    # Search over a reasonable range: delta in [p - 5*V/beta, p + 5*V/beta]
    scale = max(vega_i / beta, 0.001)
    lo = max(p - 10 * scale, -10.0 * vega_i)
    hi = p + 10 * scale

    result = minimize_scalar(neg_obj, bounds=(lo, hi), method="bounded")
    return -float(result.fun)


def _optimal_quote_logistic(
    p: float,
    lambda_i: float,
    vega_i: float,
    alpha: float,
    beta: float,
    delta_min: float = -10.0,
) -> float:
    """Optimal quote distance delta* for the logistic Hamiltonian."""
    if lambda_i <= 0.0 or vega_i <= 0.0:
        return 1e6  # no-quote

    def neg_obj(delta: float) -> float:
        arg = alpha + beta * delta / vega_i
        arg = float(np.clip(arg, -500.0, 500.0))
        lam = lambda_i / (1.0 + np.exp(arg))
        return -lam * (delta - p)

    scale = max(vega_i / beta, 0.001)
    lo = max(delta_min, p - 10 * scale)
    hi = p + 10 * scale

    result = minimize_scalar(neg_obj, bounds=(lo, hi), method="bounded")
    return float(result.x)


def solve_bbg_value_function(
    config: BBGBenchmarkConfig,
    n_nu: int = 30,
    n_vpi: int = 40,
    n_time: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the reduced BBG HJB on a (t, nu, V^pi) grid.

    Returns (t_grid, nu_grid, vpi_grid, values) where
    values has shape (n_time+1, n_nu, n_vpi).
    """
    h = config.heston
    liq = config.liquidity
    ctrl = config.control
    book = config.book

    # Grids
    T = ctrl.horizon
    t_grid = np.linspace(0.0, T, n_time + 1)
    dt = T / n_time

    # nu grid: around the initial variance
    nu_lo = max(h.nu0 * 0.5, 0.005)
    nu_hi = min(h.nu0 * 2.0, 0.10)
    nu_grid = np.linspace(nu_lo, nu_hi, n_nu)
    dnu = nu_grid[1] - nu_grid[0] if n_nu > 1 else 1.0

    # V^pi grid: symmetric around 0
    vpi_lo = -ctrl.vega_limit
    vpi_hi = ctrl.vega_limit
    vpi_grid = np.linspace(vpi_lo, vpi_hi, n_vpi)
    dvpi = vpi_grid[1] - vpi_grid[0] if n_vpi > 1 else 1.0

    # Pre-compute option data at initial state
    options = book.options
    N = book.n_options
    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        for opt in options
    ])
    prices = np.array([
        bs_call_price(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        for opt in options
    ])
    lambdas = np.array([
        liq.lambda_i(h.spot0, opt.strike) for opt in options
    ])
    trade_sizes = np.array([
        liq.trade_size(p) for p in prices
    ])

    # Terminal condition: v(T, nu, V^pi) = 0
    values = np.zeros((n_time + 1, n_nu, n_vpi))

    # Backward Euler
    for t_idx in range(n_time - 1, -1, -1):
        v_next = values[t_idx + 1]

        for j_nu in range(n_nu):
            nu = nu_grid[j_nu]
            nu_pos = max(nu, 1e-8)

            for k_vpi in range(n_vpi):
                vpi = vpi_grid[k_vpi]

                # Diffusion terms (finite differences on nu)
                d_nu_v = 0.0
                d2_nu_v = 0.0
                if n_nu > 2:
                    if 0 < j_nu < n_nu - 1:
                        d_nu_v = (v_next[j_nu + 1, k_vpi] - v_next[j_nu - 1, k_vpi]) / (2 * dnu)
                        d2_nu_v = (v_next[j_nu + 1, k_vpi] - 2 * v_next[j_nu, k_vpi] + v_next[j_nu - 1, k_vpi]) / (dnu * dnu)
                    elif j_nu == 0:
                        d_nu_v = (v_next[1, k_vpi] - v_next[0, k_vpi]) / dnu
                        if n_nu >= 3:
                            d2_nu_v = (v_next[2, k_vpi] - 2 * v_next[1, k_vpi] + v_next[0, k_vpi]) / (dnu * dnu)
                    else:
                        d_nu_v = (v_next[-1, k_vpi] - v_next[-2, k_vpi]) / dnu
                        if n_nu >= 3:
                            d2_nu_v = (v_next[-1, k_vpi] - 2 * v_next[-2, k_vpi] + v_next[-3, k_vpi]) / (dnu * dnu)

                # PDE terms
                drift_term = h.a_p(nu_pos) * d_nu_v
                diffusion_term = 0.5 * nu_pos * h.xi ** 2 * d2_nu_v
                drift_gap = vpi * (h.a_p(nu_pos) - h.a_q(nu_pos)) / (2.0 * sqrt(nu_pos))
                penalty = -(config.control.gamma * h.xi ** 2 / 8.0) * vpi ** 2

                # Hamiltonian sum over options and sides
                hamiltonian_sum = 0.0
                for i in range(N):
                    z_i = trade_sizes[i]
                    v_i = vegas[i]
                    if z_i <= 0.0 or v_i <= 0.0:
                        continue

                    for side, psi in [("b", 1.0), ("a", -1.0)]:
                        # New vega after fill
                        vpi_new = vpi + psi * z_i * v_i
                        if abs(vpi_new) > ctrl.vega_limit:
                            continue

                        # Interpolate v at (nu, vpi_new)
                        k_new = (vpi_new - vpi_lo) / dvpi
                        k_lo_idx = int(np.clip(np.floor(k_new), 0, n_vpi - 2))
                        k_frac = k_new - k_lo_idx
                        k_frac = float(np.clip(k_frac, 0.0, 1.0))
                        v_at_new = (
                            (1 - k_frac) * v_next[j_nu, k_lo_idx]
                            + k_frac * v_next[j_nu, min(k_lo_idx + 1, n_vpi - 1)]
                        )

                        p = (v_next[j_nu, k_vpi] - v_at_new) / z_i
                        h_val = _hamiltonian_logistic(
                            p, lambdas[i], v_i, liq.alpha, liq.beta,
                        )
                        hamiltonian_sum += z_i * h_val

                rhs = drift_term + diffusion_term + drift_gap + penalty + hamiltonian_sum
                values[t_idx, j_nu, k_vpi] = v_next[j_nu, k_vpi] - dt * rhs

    return t_grid, nu_grid, vpi_grid, values


def make_bbg_numerical_controller(
    config: BBGBenchmarkConfig,
    values: np.ndarray,
    t_grid: np.ndarray,
    nu_grid: np.ndarray,
    vpi_grid: np.ndarray,
):
    """Build a controller that looks up optimal quotes from the solved value function."""
    from .env import OptionBookMMAction, OptionBookMMState

    h = config.heston
    liq = config.liquidity
    ctrl = config.control
    book = config.book
    N = book.n_options

    options = book.options
    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in options
    ])
    prices = np.array([
        bs_call_price(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in options
    ])
    lambdas = np.array([liq.lambda_i(h.spot0, o.strike) for o in options])
    trade_sizes = np.array([liq.trade_size(p) for p in prices])

    dvpi = vpi_grid[1] - vpi_grid[0] if len(vpi_grid) > 1 else 1.0
    dnu = nu_grid[1] - nu_grid[0] if len(nu_grid) > 1 else 1.0
    vpi_lo = vpi_grid[0]
    nu_lo = nu_grid[0]
    n_vpi = len(vpi_grid)
    n_nu = len(nu_grid)

    def _interp_v(t_idx: int, nu: float, vpi: float) -> float:
        """Bilinear interpolation on (nu, V^pi) grid at time index t_idx."""
        j_f = (nu - nu_lo) / dnu
        j_lo = int(np.clip(np.floor(j_f), 0, n_nu - 2))
        j_frac = float(np.clip(j_f - j_lo, 0.0, 1.0))

        k_f = (vpi - vpi_lo) / dvpi
        k_lo = int(np.clip(np.floor(k_f), 0, n_vpi - 2))
        k_frac = float(np.clip(k_f - k_lo, 0.0, 1.0))

        v00 = values[t_idx, j_lo, k_lo]
        v01 = values[t_idx, j_lo, min(k_lo + 1, n_vpi - 1)]
        v10 = values[t_idx, min(j_lo + 1, n_nu - 1), k_lo]
        v11 = values[t_idx, min(j_lo + 1, n_nu - 1), min(k_lo + 1, n_vpi - 1)]

        return (
            (1 - j_frac) * (1 - k_frac) * v00
            + (1 - j_frac) * k_frac * v01
            + j_frac * (1 - k_frac) * v10
            + j_frac * k_frac * v11
        )

    def controller(state: OptionBookMMState, history=None) -> OptionBookMMAction:
        t_idx = min(state.step_index, len(t_grid) - 2)
        nu = state.variance
        vpi = state.portfolio_vega

        bid_dists = np.zeros(N)
        ask_dists = np.zeros(N)

        v_here = _interp_v(t_idx, nu, vpi)

        for i in range(N):
            z_i = trade_sizes[i]
            v_i = vegas[i]
            if z_i <= 0.0 or v_i <= 0.0:
                bid_dists[i] = 1e6
                ask_dists[i] = 1e6
                continue

            # Bid: buy = increase vega
            vpi_bid = vpi + z_i * v_i
            if abs(vpi_bid) <= ctrl.vega_limit:
                v_bid = _interp_v(t_idx, nu, vpi_bid)
                p_bid = (v_here - v_bid) / z_i
                bid_dists[i] = _optimal_quote_logistic(
                    p_bid, lambdas[i], v_i, liq.alpha, liq.beta,
                )
            else:
                bid_dists[i] = 1e6

            # Ask: sell = decrease vega
            vpi_ask = vpi - z_i * v_i
            if abs(vpi_ask) <= ctrl.vega_limit:
                v_ask = _interp_v(t_idx, nu, vpi_ask)
                p_ask = (v_here - v_ask) / z_i
                ask_dists[i] = _optimal_quote_logistic(
                    p_ask, lambdas[i], v_i, liq.alpha, liq.beta,
                )
            else:
                ask_dists[i] = 1e6

        return OptionBookMMAction(
            bid_distances=np.maximum(bid_dists, 0.0),
            ask_distances=np.maximum(ask_dists, 0.0),
            hedge_trade=-state.net_delta,
        )

    return controller


def make_bbg_risk_neutral_controller(
    config: BBGBenchmarkConfig,
):
    """Risk-neutral (gamma=0) controller: set delta_infty = optimal for H with v=0."""
    from .env import OptionBookMMAction, OptionBookMMState

    h = config.heston
    liq = config.liquidity
    book = config.book
    N = book.n_options

    options = book.options
    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in options
    ])
    prices = np.array([
        bs_call_price(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in options
    ])
    lambdas = np.array([liq.lambda_i(h.spot0, o.strike) for o in options])

    # At gamma=0, v=0 everywhere, so p=0 for all quotes.
    # Optimal quote: argmax Lambda(delta) * delta
    rn_distances = np.array([
        _optimal_quote_logistic(0.0, lambdas[i], vegas[i], liq.alpha, liq.beta)
        for i in range(N)
    ])

    def controller(state: OptionBookMMState, history=None) -> OptionBookMMAction:
        return OptionBookMMAction(
            bid_distances=rn_distances.copy(),
            ask_distances=rn_distances.copy(),
            hedge_trade=-state.net_delta,
        )

    return controller
