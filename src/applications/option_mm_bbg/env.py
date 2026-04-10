"""Simulation env for a BBG-style multi-option market-making book.

One underlying with Heston dynamics, N options, portfolio-vega accounting,
logistic intensity fills, delta-hedged wealth.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Tuple

import numpy as np

from .pricing import bs_call_price, bs_call_delta, bs_call_vega_sqrt_nu, price_option_book
from .spec import BBGBenchmarkConfig, BBGHestonSpec, BBGOptionBookSpec, BBGLiquiditySpec, BBGControlSpec


@dataclass(frozen=True)
class OptionBookMMAction:
    """Per-option bid/ask quote distances and stock hedge."""
    bid_distances: np.ndarray   # (n_options,) distance from mid
    ask_distances: np.ndarray   # (n_options,) distance from mid
    hedge_trade: float = 0.0


@dataclass
class OptionBookMMState:
    """Full book state."""
    step_index: int
    time: float
    spot: float
    variance: float
    option_prices: np.ndarray    # (n_options,)
    option_deltas: np.ndarray    # (n_options,)
    option_vegas: np.ndarray     # (n_options,) BBG vega = d_{sqrt(nu)} O
    option_inventories: np.ndarray  # (n_options,) integers
    portfolio_vega: float
    stock_position: float
    cash: float
    wealth: float
    net_delta: float
    done: bool = False


class OptionBookMarketMakingEnv:
    """Multi-option Heston market-making simulator with logistic fills."""

    def __init__(
        self,
        config: BBGBenchmarkConfig | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or BBGBenchmarkConfig.paper_default()
        self.config.validate()

        h = self.config.heston
        b = self.config.book
        c = self.config.control

        # Time grid
        if dt is not None and n_steps is not None:
            self.dt = dt
            self.n_steps = n_steps
        else:
            # Default: divide control horizon into ~30 steps
            self.n_steps = 30
            self.dt = c.horizon / self.n_steps

        # Option specs
        self._strikes = np.array([o.strike for o in b.options])
        self._maturities = np.array([o.maturity for o in b.options])
        self.n_options = b.n_options

        # Pre-compute initial option data
        self._initial_prices, self._initial_deltas, self._initial_vegas = (
            price_option_book(h.spot0, h.nu0, h.rate, self._strikes, self._maturities)
        )

        # Pre-compute per-option liquidity params (frozen at t=0)
        liq = self.config.liquidity
        self._lambdas = np.array([
            liq.lambda_i(h.spot0, s) for s in self._strikes
            for _ in self._maturities  # same lambda for all maturities of same strike
        ])
        self._trade_sizes = np.array([
            liq.trade_size(p) for p in self._initial_prices
        ])

        self._set_rngs(seed)
        self.state: OptionBookMMState | None = None

    def _set_rngs(self, seed: int | None) -> None:
        ss = np.random.SeedSequence(seed)
        path_s, fill_s = ss.spawn(2)
        self.path_rng = np.random.default_rng(path_s)
        self.fill_rng = np.random.default_rng(fill_s)

    def reset(self, seed: int | None = None) -> OptionBookMMState:
        if seed is not None:
            self._set_rngs(seed)

        h = self.config.heston
        inventories = np.zeros(self.n_options, dtype=float)

        self.state = self._build_state(
            step_index=0,
            time=0.0,
            spot=h.spot0,
            variance=h.nu0,
            inventories=inventories,
            stock_position=0.0,
            cash=0.0,
        )
        return self.state

    def step(
        self,
        action: OptionBookMMAction,
    ) -> Tuple[OptionBookMMState, float, bool, dict]:
        if self.state is None:
            raise RuntimeError("call reset first")
        if self.state.done:
            raise RuntimeError("episode is done")

        prev = self.state
        h = self.config.heston
        liq = self.config.liquidity
        c = self.config.control

        # Sample fills per option per side
        inventories = prev.option_inventories.copy()
        cash = prev.cash
        total_spread_capture = 0.0

        for i in range(self.n_options):
            vega_i = self._initial_vegas[i]  # frozen (Assumption 1)
            z_i = self._trade_sizes[i]

            # Bid side (market maker buys)
            bid_int = liq.intensity(action.bid_distances[i], self._lambdas[i], vega_i)
            expected_bid = bid_int * self.dt
            bid_fills = int(self.fill_rng.poisson(max(expected_bid, 0.0)))

            # Ask side (market maker sells)
            ask_int = liq.intensity(action.ask_distances[i], self._lambdas[i], vega_i)
            expected_ask = ask_int * self.dt
            ask_fills = int(self.fill_rng.poisson(max(expected_ask, 0.0)))

            # Vega risk limit check
            for _ in range(bid_fills):
                new_vega = sum(inventories[j] * self._initial_vegas[j]
                               for j in range(self.n_options)) + z_i * vega_i
                if abs(new_vega) <= c.vega_limit:
                    inventories[i] += z_i
                    cash -= z_i * (prev.option_prices[i] - action.bid_distances[i])
                    total_spread_capture += z_i * action.bid_distances[i]

            for _ in range(ask_fills):
                new_vega = sum(inventories[j] * self._initial_vegas[j]
                               for j in range(self.n_options)) - z_i * vega_i
                if abs(new_vega) <= c.vega_limit:
                    inventories[i] -= z_i
                    cash += z_i * (prev.option_prices[i] + action.ask_distances[i])
                    total_spread_capture += z_i * action.ask_distances[i]

        # Stock hedge
        stock_position = prev.stock_position + action.hedge_trade
        cash -= action.hedge_trade * prev.spot

        # Advance Heston
        spot_next, var_next = self._advance_heston(prev.spot, prev.variance)

        step_index = prev.step_index + 1
        time_next = step_index * self.dt
        done = step_index >= self.n_steps

        next_state = self._build_state(
            step_index=step_index,
            time=time_next,
            spot=spot_next,
            variance=var_next,
            inventories=inventories,
            stock_position=stock_position,
            cash=cash,
        )

        reward = next_state.wealth - prev.wealth
        info = {
            "spread_capture": total_spread_capture,
            "wealth_before": prev.wealth,
            "wealth_after": next_state.wealth,
        }
        self.state = next_state
        return next_state, reward, done, info

    def _build_state(
        self,
        step_index: int,
        time: float,
        spot: float,
        variance: float,
        inventories: np.ndarray,
        stock_position: float,
        cash: float,
    ) -> OptionBookMMState:
        h = self.config.heston
        # Price options at current (spot, variance)
        # Under constant-vega assumption, use frozen vegas
        prices = np.array([
            bs_call_price(spot, self._strikes[i], self._maturities[i], h.rate, variance)
            for i in range(self.n_options)
        ])
        deltas = np.array([
            bs_call_delta(spot, self._strikes[i], self._maturities[i], h.rate, variance)
            for i in range(self.n_options)
        ])
        vegas = self._initial_vegas  # frozen (Assumption 1)

        portfolio_vega = float(np.sum(inventories * vegas))
        net_delta = stock_position + float(np.sum(inventories * deltas))
        wealth = cash + stock_position * spot + float(np.sum(inventories * prices))

        return OptionBookMMState(
            step_index=step_index,
            time=time,
            spot=spot,
            variance=variance,
            option_prices=prices,
            option_deltas=deltas,
            option_vegas=vegas,
            option_inventories=inventories,
            portfolio_vega=portfolio_vega,
            stock_position=stock_position,
            cash=cash,
            wealth=wealth,
            net_delta=net_delta,
            done=step_index >= self.n_steps,
        )

    def _advance_heston(self, spot: float, variance: float) -> Tuple[float, float]:
        h = self.config.heston
        z_s = float(self.path_rng.normal())
        z_ind = float(self.path_rng.normal())
        z_v = h.rho * z_s + sqrt(max(1.0 - h.rho ** 2, 0.0)) * z_ind

        var_pos = max(variance, 1e-12)
        sqrt_v_dt = sqrt(var_pos * self.dt)

        # Under P measure
        var_next = variance + h.a_p(var_pos) * self.dt + h.xi * sqrt_v_dt * z_v
        var_next = max(var_next, 1e-12)

        spot_next = spot * exp(-0.5 * var_pos * self.dt + sqrt_v_dt * z_s)
        return spot_next, var_next
