"""BBG reduced HJB solver on the (t, nu, V^pi) grid.

Implements eq. (4) from Baldacci, Bergault & Gueant (2020).

Solver variants:
  - solve_bbg_value_function:          full 3D (t, nu, V^pi) monotone explicit
  - solve_bbg_value_function_no_gap:   reduced 2D (t, V^pi) when a_P = a_Q
"""

from __future__ import annotations

from math import exp, sqrt
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .pricing import bs_call_vega_sqrt_nu, bs_call_price
from .spec import BBGBenchmarkConfig, BBGHestonSpec


# ---------------------------------------------------------------------------
# Logistic Hamiltonian primitives
# ---------------------------------------------------------------------------


def _hamiltonian_logistic(
    p: float,
    lambda_i: float,
    vega_i: float,
    alpha: float,
    beta: float,
) -> float:
    """H^{i,j}(p) = sup_{delta} Lambda(delta) * (delta - p)."""
    if lambda_i <= 0.0 or vega_i <= 0.0:
        return 0.0

    def neg_obj(delta: float) -> float:
        arg = alpha + beta * delta / vega_i
        arg = float(np.clip(arg, -500.0, 500.0))
        lam = lambda_i / (1.0 + np.exp(arg))
        return -lam * (delta - p)

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
        return 1e6

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


# ---------------------------------------------------------------------------
# Block 2: Precomputed Hamiltonian lookup tables
# ---------------------------------------------------------------------------


class HamiltonianTable:
    """Precomputed H(p) and delta*(p) for one option on a dense p-grid."""

    def __init__(
        self,
        lambda_i: float,
        vega_i: float,
        alpha: float,
        beta: float,
        p_lo: float = -5.0,
        p_hi: float = 5.0,
        n_p: int = 500,
    ) -> None:
        self.p_grid = np.linspace(p_lo, p_hi, n_p)
        self.dp = self.p_grid[1] - self.p_grid[0] if n_p > 1 else 1.0
        self.H_values = np.empty(n_p)
        self.delta_values = np.empty(n_p)

        for idx, p in enumerate(self.p_grid):
            self.H_values[idx] = _hamiltonian_logistic(p, lambda_i, vega_i, alpha, beta)
            self.delta_values[idx] = _optimal_quote_logistic(p, lambda_i, vega_i, alpha, beta)

    def interp_H(self, p: float) -> float:
        """Linear interpolation of H(p), clamped at boundaries."""
        idx_f = (p - self.p_grid[0]) / self.dp
        idx_lo = int(np.clip(np.floor(idx_f), 0, len(self.p_grid) - 2))
        frac = float(np.clip(idx_f - idx_lo, 0.0, 1.0))
        return float((1 - frac) * self.H_values[idx_lo] + frac * self.H_values[idx_lo + 1])

    def interp_delta(self, p: float) -> float:
        """Linear interpolation of delta*(p), clamped at boundaries."""
        idx_f = (p - self.p_grid[0]) / self.dp
        idx_lo = int(np.clip(np.floor(idx_f), 0, len(self.p_grid) - 2))
        frac = float(np.clip(idx_f - idx_lo, 0.0, 1.0))
        return float((1 - frac) * self.delta_values[idx_lo] + frac * self.delta_values[idx_lo + 1])


def _build_hamiltonian_tables(
    config: BBGBenchmarkConfig,
    p_lo: float = -5.0,
    p_hi: float = 5.0,
    n_p: int = 500,
) -> list[HamiltonianTable]:
    """Build one HamiltonianTable per option."""
    h = config.heston
    liq = config.liquidity
    tables = []
    for opt in config.book.options:
        vega = bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        lam = liq.lambda_i(h.spot0, opt.strike)
        tables.append(HamiltonianTable(lam, vega, liq.alpha, liq.beta, p_lo, p_hi, n_p))
    return tables


# ---------------------------------------------------------------------------
# Block 3: No-gap reduced solver (t, V^pi) only
# ---------------------------------------------------------------------------


def solve_bbg_value_function_no_gap(
    config: BBGBenchmarkConfig,
    n_vpi: int = 80,
    n_time: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the reduced BBG HJB when a_P = a_Q (no drift gap).

    Value depends only on (t, V^pi). Returns (t_grid, vpi_grid, values)
    where values has shape (n_time+1, n_vpi).
    """
    h = config.heston
    liq = config.liquidity
    ctrl = config.control
    book = config.book

    T = ctrl.horizon
    t_grid = np.linspace(0.0, T, n_time + 1)
    dt = T / n_time

    vpi_lo = -ctrl.vega_limit
    vpi_hi = ctrl.vega_limit
    vpi_grid = np.linspace(vpi_lo, vpi_hi, n_vpi)
    dvpi = vpi_grid[1] - vpi_grid[0] if n_vpi > 1 else 1.0

    # Pre-compute option data
    N = book.n_options
    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        for opt in book.options
    ])
    prices = np.array([
        bs_call_price(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        for opt in book.options
    ])
    lambdas = np.array([liq.lambda_i(h.spot0, opt.strike) for opt in book.options])
    trade_sizes = np.array([liq.trade_size(p) for p in prices])

    # Hamiltonian tables
    tables = _build_hamiltonian_tables(config)

    # Terminal condition
    values = np.zeros((n_time + 1, n_vpi))

    # Backward solve
    for t_idx in range(n_time - 1, -1, -1):
        v_next = values[t_idx + 1]

        for k in range(n_vpi):
            vpi = vpi_grid[k]

            # Penalty term (no drift gap, no diffusion in no-gap case)
            penalty = -(ctrl.gamma * h.xi ** 2 / 8.0) * vpi ** 2

            # Hamiltonian sum
            ham_sum = 0.0
            for i in range(N):
                z_i = trade_sizes[i]
                v_i = vegas[i]
                if z_i <= 0.0 or v_i <= 0.0:
                    continue

                for psi in [1.0, -1.0]:  # bid (+), ask (-)
                    vpi_new = vpi + psi * z_i * v_i
                    if abs(vpi_new) > ctrl.vega_limit:
                        continue

                    # Positive-weight interpolation (monotone)
                    k_f = (vpi_new - vpi_lo) / dvpi
                    k_lo_idx = int(np.clip(np.floor(k_f), 0, n_vpi - 2))
                    k_frac = float(np.clip(k_f - k_lo_idx, 0.0, 1.0))
                    v_at_new = (1 - k_frac) * v_next[k_lo_idx] + k_frac * v_next[min(k_lo_idx + 1, n_vpi - 1)]

                    p = (v_next[k] - v_at_new) / z_i
                    h_val = tables[i].interp_H(p)
                    ham_sum += z_i * h_val

            rhs = penalty + ham_sum
            values[t_idx, k] = v_next[k] + dt * rhs

    return t_grid, vpi_grid, values


# ---------------------------------------------------------------------------
# Block 4: Full 3D solver with monotone scheme
# ---------------------------------------------------------------------------


def solve_bbg_value_function(
    config: BBGBenchmarkConfig,
    n_nu: int = 30,
    n_vpi: int = 40,
    n_time: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the reduced BBG HJB on a (t, nu, V^pi) grid.

    Uses monotone explicit Euler with upwind differencing in nu.
    Returns (t_grid, nu_grid, vpi_grid, values).
    """
    h = config.heston
    liq = config.liquidity
    ctrl = config.control
    book = config.book

    T = ctrl.horizon
    t_grid = np.linspace(0.0, T, n_time + 1)
    dt = T / n_time

    nu_lo = max(h.nu0 * 0.5, 0.005)
    nu_hi = min(h.nu0 * 2.0, 0.10)
    nu_grid = np.linspace(nu_lo, nu_hi, n_nu)
    dnu = nu_grid[1] - nu_grid[0] if n_nu > 1 else 1.0

    vpi_lo = -ctrl.vega_limit
    vpi_hi = ctrl.vega_limit
    vpi_grid = np.linspace(vpi_lo, vpi_hi, n_vpi)
    dvpi = vpi_grid[1] - vpi_grid[0] if n_vpi > 1 else 1.0

    N = book.n_options
    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        for opt in book.options
    ])
    prices = np.array([
        bs_call_price(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
        for opt in book.options
    ])
    lambdas = np.array([liq.lambda_i(h.spot0, opt.strike) for opt in book.options])
    trade_sizes = np.array([liq.trade_size(p) for p in prices])

    tables = _build_hamiltonian_tables(config)

    values = np.zeros((n_time + 1, n_nu, n_vpi))

    for t_idx in range(n_time - 1, -1, -1):
        v_next = values[t_idx + 1]

        for j_nu in range(n_nu):
            nu = nu_grid[j_nu]
            nu_pos = max(nu, 1e-8)

            # Monotone upwind differencing for drift term
            drift_coeff = h.a_p(nu_pos)  # kappa_P(theta_P - nu)
            diff_coeff = 0.5 * nu_pos * h.xi ** 2

            for k_vpi in range(n_vpi):
                vpi = vpi_grid[k_vpi]

                # Upwind first derivative in nu
                d_nu_v = 0.0
                if n_nu > 1:
                    if drift_coeff >= 0:
                        # Forward difference
                        if j_nu < n_nu - 1:
                            d_nu_v = (v_next[j_nu + 1, k_vpi] - v_next[j_nu, k_vpi]) / dnu
                    else:
                        # Backward difference
                        if j_nu > 0:
                            d_nu_v = (v_next[j_nu, k_vpi] - v_next[j_nu - 1, k_vpi]) / dnu

                # Central second derivative in nu
                d2_nu_v = 0.0
                if n_nu > 2 and 0 < j_nu < n_nu - 1:
                    d2_nu_v = (
                        v_next[j_nu + 1, k_vpi]
                        - 2 * v_next[j_nu, k_vpi]
                        + v_next[j_nu - 1, k_vpi]
                    ) / (dnu * dnu)

                drift_term = drift_coeff * d_nu_v
                diffusion_term = diff_coeff * d2_nu_v
                drift_gap = vpi * (h.a_p(nu_pos) - h.a_q(nu_pos)) / (2.0 * sqrt(nu_pos))
                penalty = -(ctrl.gamma * h.xi ** 2 / 8.0) * vpi ** 2

                # Hamiltonian sum with table lookup
                ham_sum = 0.0
                for i in range(N):
                    z_i = trade_sizes[i]
                    v_i = vegas[i]
                    if z_i <= 0.0 or v_i <= 0.0:
                        continue

                    for psi in [1.0, -1.0]:
                        vpi_new = vpi + psi * z_i * v_i
                        if abs(vpi_new) > ctrl.vega_limit:
                            continue

                        k_f = (vpi_new - vpi_lo) / dvpi
                        k_lo_idx = int(np.clip(np.floor(k_f), 0, n_vpi - 2))
                        k_frac = float(np.clip(k_f - k_lo_idx, 0.0, 1.0))
                        v_at_new = (
                            (1 - k_frac) * v_next[j_nu, k_lo_idx]
                            + k_frac * v_next[j_nu, min(k_lo_idx + 1, n_vpi - 1)]
                        )

                        p = (v_next[j_nu, k_vpi] - v_at_new) / z_i
                        h_val = tables[i].interp_H(p)
                        ham_sum += z_i * h_val

                rhs = drift_term + diffusion_term + drift_gap + penalty + ham_sum
                values[t_idx, j_nu, k_vpi] = v_next[j_nu, k_vpi] + dt * rhs

    return t_grid, nu_grid, vpi_grid, values


# ---------------------------------------------------------------------------
# Solver diagnostics
# ---------------------------------------------------------------------------


def solver_diagnostics(values: np.ndarray) -> dict:
    """Lightweight diagnostic summary of the solved value function."""
    return {
        "max_abs": float(np.max(np.abs(values))),
        "n_nonfinite": int(np.sum(~np.isfinite(values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "terminal_max_abs": float(np.max(np.abs(values[-1]))),
        "t0_range": (float(np.min(values[0])), float(np.max(values[0]))),
    }


# ---------------------------------------------------------------------------
# Controller factories
# ---------------------------------------------------------------------------


def make_bbg_numerical_controller(
    config: BBGBenchmarkConfig,
    values: np.ndarray,
    t_grid: np.ndarray,
    nu_grid: np.ndarray | None,
    vpi_grid: np.ndarray,
):
    """Build a controller from a solved value function.

    Works with both 3D (t, nu, V^pi) and 2D (t, V^pi) value arrays.
    """
    from .env import OptionBookMMAction, OptionBookMMState

    h = config.heston
    liq = config.liquidity
    ctrl = config.control
    N = config.book.n_options

    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in config.book.options
    ])
    prices = np.array([
        bs_call_price(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in config.book.options
    ])
    lambdas = np.array([liq.lambda_i(h.spot0, o.strike) for o in config.book.options])
    trade_sizes = np.array([liq.trade_size(p) for p in prices])
    tables = _build_hamiltonian_tables(config)

    dvpi = vpi_grid[1] - vpi_grid[0] if len(vpi_grid) > 1 else 1.0
    vpi_lo = vpi_grid[0]
    n_vpi = len(vpi_grid)

    is_3d = values.ndim == 3

    if is_3d:
        assert nu_grid is not None
        dnu = nu_grid[1] - nu_grid[0] if len(nu_grid) > 1 else 1.0
        nu_lo = nu_grid[0]
        n_nu = len(nu_grid)

    def _interp_v(t_idx: int, nu: float, vpi: float) -> float:
        k_f = (vpi - vpi_lo) / dvpi
        k_lo = int(np.clip(np.floor(k_f), 0, n_vpi - 2))
        k_frac = float(np.clip(k_f - k_lo, 0.0, 1.0))

        if is_3d:
            j_f = (nu - nu_lo) / dnu
            j_lo = int(np.clip(np.floor(j_f), 0, n_nu - 2))
            j_frac = float(np.clip(j_f - j_lo, 0.0, 1.0))
            v00 = values[t_idx, j_lo, k_lo]
            v01 = values[t_idx, j_lo, min(k_lo + 1, n_vpi - 1)]
            v10 = values[t_idx, min(j_lo + 1, n_nu - 1), k_lo]
            v11 = values[t_idx, min(j_lo + 1, n_nu - 1), min(k_lo + 1, n_vpi - 1)]
            return ((1 - j_frac) * ((1 - k_frac) * v00 + k_frac * v01)
                    + j_frac * ((1 - k_frac) * v10 + k_frac * v11))
        else:
            return float((1 - k_frac) * values[t_idx, k_lo]
                         + k_frac * values[t_idx, min(k_lo + 1, n_vpi - 1)])

    def controller(state: OptionBookMMState, history=None) -> OptionBookMMAction:
        t_idx = min(state.step_index, len(t_grid) - 2)
        nu = state.variance
        vpi = state.portfolio_vega

        bid_dists = np.full(N, 1e6)
        ask_dists = np.full(N, 1e6)
        v_here = _interp_v(t_idx, nu, vpi)

        for i in range(N):
            z_i = trade_sizes[i]
            v_i = vegas[i]
            if z_i <= 0.0 or v_i <= 0.0:
                continue

            vpi_bid = vpi + z_i * v_i
            if abs(vpi_bid) <= ctrl.vega_limit:
                v_bid = _interp_v(t_idx, nu, vpi_bid)
                p_bid = (v_here - v_bid) / z_i
                bid_dists[i] = tables[i].interp_delta(p_bid)

            vpi_ask = vpi - z_i * v_i
            if abs(vpi_ask) <= ctrl.vega_limit:
                v_ask = _interp_v(t_idx, nu, vpi_ask)
                p_ask = (v_here - v_ask) / z_i
                ask_dists[i] = tables[i].interp_delta(p_ask)

        return OptionBookMMAction(
            bid_distances=np.maximum(bid_dists, 0.0),
            ask_distances=np.maximum(ask_dists, 0.0),
            hedge_trade=-state.net_delta,
        )

    return controller


def make_bbg_risk_neutral_controller(config: BBGBenchmarkConfig):
    """Risk-neutral (gamma=0) controller: p=0 for all options."""
    from .env import OptionBookMMAction, OptionBookMMState

    h = config.heston
    liq = config.liquidity
    N = config.book.n_options

    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in config.book.options
    ])
    lambdas = np.array([liq.lambda_i(h.spot0, o.strike) for o in config.book.options])

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
