"""Pure-simulation option market-making environment.

This module pins down the v1 benchmark conventions before any controller code:

- one Heston underlying;
- one long-dated European call, struck ATM at episode reset and held fixed;
- horizon strictly shorter than option maturity, so v1 has no expiry/roll logic;
- independent Poisson bid/ask fills with exponential intensity in quote distance;
- a default same-step fill policy that removes the zero-risk full-spread subsidy;
- no queue dynamics and no market impact;
- mark-to-market wealth, not IV RMSE, is the primitive reward.

The true market option mid is a Black-Scholes proxy using the current Heston
variance. That is intentionally simple: v1 should test inventory/control under
partial variance observation, not option-pricing model sophistication.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt
from typing import Literal, Optional, Tuple

import numpy as np


SQRT_TWO = sqrt(2.0)
SQRT_TWO_PI = sqrt(2.0 * pi)
SameStepBothFillsPolicy = Literal["allowed", "rejected", "mid_drift"]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT_TWO))


def _normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / SQRT_TWO_PI


def black_scholes_call_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    variance: float,
) -> float:
    """Black-Scholes European call price parameterized by instantaneous variance."""
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")

    if tau <= 0.0:
        return max(spot - strike, 0.0)

    sigma = sqrt(max(variance, 1e-16))
    vol_sqrt_tau = sigma * sqrt(tau)
    if vol_sqrt_tau <= 1e-12:
        forward_intrinsic = spot - strike * exp(-rate * tau)
        return max(forward_intrinsic, 0.0)

    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    return spot * _normal_cdf(d1) - strike * exp(-rate * tau) * _normal_cdf(d2)


def black_scholes_call_delta(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    variance: float,
) -> float:
    """Black-Scholes European call delta parameterized by instantaneous variance."""
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")

    if tau <= 0.0:
        if spot > strike:
            return 1.0
        if spot < strike:
            return 0.0
        return 0.5

    sigma = sqrt(max(variance, 1e-16))
    vol_sqrt_tau = sigma * sqrt(tau)
    if vol_sqrt_tau <= 1e-12:
        return 1.0 if spot > strike * exp(-rate * tau) else 0.0

    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol_sqrt_tau
    return _normal_cdf(d1)


@dataclass(frozen=True)
class HestonParams:
    """Heston path parameters for the v1 simulator."""

    spot0: float = 100.0
    variance0: float = 0.04
    mu: float = 0.0
    rate: float = 0.0
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    variance_floor: float = 1e-8

    def validate(self) -> None:
        if self.spot0 <= 0.0:
            raise ValueError("spot0 must be positive")
        if self.variance0 < 0.0 or self.theta < 0.0:
            raise ValueError("variance0 and theta must be nonnegative")
        if self.kappa < 0.0 or self.xi < 0.0:
            raise ValueError("kappa and xi must be nonnegative")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        if self.variance_floor < 0.0:
            raise ValueError("variance_floor must be nonnegative")


@dataclass(frozen=True)
class OptionContractSpec:
    """One fixed-strike European call contract.

    If ``strike`` is ``None``, the option is struck ATM at reset using ``spot0``.
    The strike is then fixed for the whole episode. There is no re-strike, expiry,
    settlement, or roll in v1; the episode horizon must be shorter than maturity.
    The default is intentionally long-dated, so tau is nearly stationary across
    an episode but remains a controller input in ``[q, h, V_hat, tau]``.
    """

    maturity_years: float = 1.0
    strike: Optional[float] = None
    contract_multiplier: float = 100.0

    def validate(self) -> None:
        if self.maturity_years <= 0.0:
            raise ValueError("maturity_years must be positive")
        if self.strike is not None and self.strike <= 0.0:
            raise ValueError("strike must be positive when provided")
        if self.contract_multiplier <= 0.0:
            raise ValueError("contract_multiplier must be positive")


@dataclass(frozen=True)
class FillModelSpec:
    """Frozen v1 fill model.

    Intensities are annualized and side-specific:

        lambda_bid = Lambda_0 exp(-k (mid - bid))
        lambda_ask = Lambda_0 exp(-k (ask - mid))

    Distances are signed. Crossing through the mid therefore raises intensity,
    capped by ``max_intensity``. Fills are independent Poisson counts, clipped to
    ``max_contracts_per_step`` and multiplied by ``lot_size``. Queue dynamics and
    market impact are deliberately absent in v1.

    If both sides fill in the same step, the default ``mid_drift`` policy applies
    a conservative half-tick accounting drift to matched round-trips. This avoids
    giving controllers a risk-free full-spread subsidy from interval aggregation.
    """

    base_intensity: float = 50.0
    distance_slope: float = 5.0
    max_intensity: float = 1_000.0
    max_contracts_per_step: int = 5
    lot_size: int = 1
    same_step_both_fills_policy: SameStepBothFillsPolicy = "mid_drift"
    tick_size: float = 0.01
    same_step_mid_drift_ticks: float = 0.5

    def validate(self) -> None:
        if self.base_intensity < 0.0:
            raise ValueError("base_intensity must be nonnegative")
        if self.distance_slope < 0.0:
            raise ValueError("distance_slope must be nonnegative")
        if self.max_intensity < 0.0:
            raise ValueError("max_intensity must be nonnegative")
        if self.max_contracts_per_step < 0:
            raise ValueError("max_contracts_per_step must be nonnegative")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.same_step_both_fills_policy not in {"allowed", "rejected", "mid_drift"}:
            raise ValueError(
                "same_step_both_fills_policy must be one of "
                "{'allowed', 'rejected', 'mid_drift'}"
            )
        if self.tick_size < 0.0:
            raise ValueError("tick_size must be nonnegative")
        if self.same_step_mid_drift_ticks < 0.0:
            raise ValueError("same_step_mid_drift_ticks must be nonnegative")


@dataclass(frozen=True)
class ExecutionCostSpec:
    """Execution costs for option fills and stock hedges."""

    option_fee_per_contract: float = 0.0
    stock_fee_per_share: float = 0.0
    stock_slippage_bps: float = 0.0

    def validate(self) -> None:
        if self.option_fee_per_contract < 0.0:
            raise ValueError("option_fee_per_contract must be nonnegative")
        if self.stock_fee_per_share < 0.0:
            raise ValueError("stock_fee_per_share must be nonnegative")
        if self.stock_slippage_bps < 0.0:
            raise ValueError("stock_slippage_bps must be nonnegative")


@dataclass(frozen=True)
class OptionMMAction:
    """Controller action submitted at the beginning of a step."""

    bid_price: float
    ask_price: float
    hedge_trade: float = 0.0


@dataclass(frozen=True)
class OptionMMState:
    """Mark-to-market state after a simulator step."""

    step_index: int
    time: float
    spot: float
    variance: float
    strike: float
    time_to_maturity: float
    option_mid: float
    option_delta: float
    option_inventory: int
    stock_position: float
    cash: float
    wealth: float
    net_delta: float
    done: bool = False


@dataclass(frozen=True)
class OptionMMStepInfo:
    """Diagnostics for one environment transition."""

    wealth_before: float
    wealth_after: float
    pnl: float
    spot_before: float
    spot_after: float
    variance_before: float
    variance_after: float
    option_mid_before: float
    option_mid_after: float
    option_delta_before: float
    option_delta_after: float
    bid_price: float
    ask_price: float
    bid_distance: float
    ask_distance: float
    bid_intensity: float
    ask_intensity: float
    bid_fills: int
    ask_fills: int
    bid_fills_censored: bool
    ask_fills_censored: bool
    bid_notional: float
    ask_notional: float
    same_step_both_fills_policy: str
    same_step_first_side: str
    same_step_mid_drift: float
    option_cashflow: float
    hedge_trade: float
    hedge_cashflow: float
    option_fees: float
    stock_costs: float
    spread_capture: float
    adverse_selection_cost: float
    z_spot: float
    z_variance: float
    variance_floor_bound: bool
    variance_floor_binds: int


class OptionMarketMakingEnv:
    """One-option Heston market-making simulator with Poisson fills."""

    def __init__(
        self,
        heston: HestonParams | None = None,
        contract: OptionContractSpec | None = None,
        fills: FillModelSpec | None = None,
        costs: ExecutionCostSpec | None = None,
        dt: float = 1.0 / 252.0,
        horizon_steps: int = 20,
        initial_cash: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.heston = heston or HestonParams()
        self.contract = contract or OptionContractSpec()
        self.fills = fills or FillModelSpec()
        self.costs = costs or ExecutionCostSpec()
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.initial_cash = initial_cash

        self.heston.validate()
        self.contract.validate()
        self.fills.validate()
        self.costs.validate()
        self._validate_time_grid()
        self._validate_initial_cash()

        self._set_rngs(seed)
        self.state: OptionMMState | None = None
        self.variance_floor_binds = 0

    def _set_rngs(self, seed: int | None) -> None:
        seed_sequence = np.random.SeedSequence(seed)
        path_seed, fill_seed, tie_seed = seed_sequence.spawn(3)
        self.path_rng = np.random.default_rng(path_seed)
        self.fill_rng = np.random.default_rng(fill_seed)
        self.tie_rng = np.random.default_rng(tie_seed)

    def _validate_time_grid(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        horizon_years = self.horizon_steps * self.dt
        if horizon_years >= self.contract.maturity_years:
            raise ValueError(
                "v1 excludes expiry/roll; require horizon_steps * dt < maturity_years"
            )

    def _validate_initial_cash(self) -> None:
        if not isfinite(self.initial_cash):
            raise ValueError("initial_cash must be finite")

    def reset(self, seed: int | None = None) -> OptionMMState:
        """Reset to zero inventory and an ATM fixed-strike contract."""
        if seed is not None:
            self._set_rngs(seed)
        self.variance_floor_binds = 0

        strike = self.contract.strike or self.heston.spot0
        spot = self.heston.spot0
        variance = max(self.heston.variance0, self.heston.variance_floor)
        tau = self.contract.maturity_years

        self.state = self._build_state(
            step_index=0,
            time=0.0,
            spot=spot,
            variance=variance,
            strike=strike,
            tau=tau,
            option_inventory=0,
            stock_position=0.0,
            cash=self.initial_cash,
            done=False,
        )
        return self.state

    def fill_intensities(
        self,
        bid_price: float,
        ask_price: float,
        option_mid: float | None = None,
    ) -> Tuple[float, float]:
        """Return annualized bid/ask fill intensities for the current mid."""
        if option_mid is None:
            if self.state is None:
                raise RuntimeError("reset must be called before using current mid")
            option_mid = self.state.option_mid
        self._validate_quotes(bid_price, ask_price)

        bid_distance = option_mid - bid_price
        ask_distance = ask_price - option_mid
        return (
            self._intensity_from_distance(bid_distance),
            self._intensity_from_distance(ask_distance),
        )

    def step(self, action: OptionMMAction) -> Tuple[OptionMMState, float, bool, OptionMMStepInfo]:
        """Advance one step using pre-fill quotes and a pre-fill stock hedge trade."""
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self.state.done:
            raise RuntimeError("cannot step a finished episode; call reset")
        self._validate_action(action)

        prev = self.state
        wealth_before = prev.wealth

        bid_distance = prev.option_mid - action.bid_price
        ask_distance = action.ask_price - prev.option_mid
        bid_intensity = self._intensity_from_distance(bid_distance)
        ask_intensity = self._intensity_from_distance(ask_distance)
        bid_fills, bid_fills_censored = self._sample_contract_fills(bid_intensity)
        ask_fills, ask_fills_censored = self._sample_contract_fills(ask_intensity)
        (
            bid_fills,
            ask_fills,
            bid_notional,
            ask_notional,
            same_step_first_side,
            same_step_mid_drift,
        ) = self._apply_same_step_both_fills_policy(
            bid_fills=bid_fills,
            ask_fills=ask_fills,
            bid_price=action.bid_price,
            ask_price=action.ask_price,
        )

        multiplier = self.contract.contract_multiplier
        option_inventory = prev.option_inventory + bid_fills - ask_fills
        option_cashflow = multiplier * (ask_notional - bid_notional)
        option_fees = self.costs.option_fee_per_contract * (bid_fills + ask_fills)

        hedge_trade = action.hedge_trade
        stock_costs = abs(hedge_trade) * (
            self.costs.stock_fee_per_share
            + prev.spot * self.costs.stock_slippage_bps * 1e-4
        )
        hedge_cashflow = -hedge_trade * prev.spot - stock_costs

        cash = prev.cash + option_cashflow - option_fees + hedge_cashflow
        stock_position = prev.stock_position + hedge_trade

        (
            spot_next,
            variance_next,
            z_spot,
            z_variance,
            variance_floor_bound,
        ) = self._advance_heston(
            prev.spot, prev.variance
        )
        if variance_floor_bound:
            self.variance_floor_binds += 1
        step_index = prev.step_index + 1
        time_next = step_index * self.dt
        tau_next = max(self.contract.maturity_years - time_next, 0.0)
        done = step_index >= self.horizon_steps

        next_state = self._build_state(
            step_index=step_index,
            time=time_next,
            spot=spot_next,
            variance=variance_next,
            strike=prev.strike,
            tau=tau_next,
            option_inventory=option_inventory,
            stock_position=stock_position,
            cash=cash,
            done=done,
        )

        q_from_fills = bid_fills - ask_fills
        spread_capture = multiplier * (
            bid_fills * prev.option_mid
            - bid_notional
            + ask_notional
            - ask_fills * prev.option_mid
        )
        adverse_selection_cost = -multiplier * q_from_fills * (
            next_state.option_mid - prev.option_mid
        )

        reward = next_state.wealth - wealth_before
        info = OptionMMStepInfo(
            wealth_before=wealth_before,
            wealth_after=next_state.wealth,
            pnl=reward,
            spot_before=prev.spot,
            spot_after=next_state.spot,
            variance_before=prev.variance,
            variance_after=next_state.variance,
            option_mid_before=prev.option_mid,
            option_mid_after=next_state.option_mid,
            option_delta_before=prev.option_delta,
            option_delta_after=next_state.option_delta,
            bid_price=action.bid_price,
            ask_price=action.ask_price,
            bid_distance=bid_distance,
            ask_distance=ask_distance,
            bid_intensity=bid_intensity,
            ask_intensity=ask_intensity,
            bid_fills=bid_fills,
            ask_fills=ask_fills,
            bid_fills_censored=bid_fills_censored,
            ask_fills_censored=ask_fills_censored,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
            same_step_both_fills_policy=self.fills.same_step_both_fills_policy,
            same_step_first_side=same_step_first_side,
            same_step_mid_drift=same_step_mid_drift,
            option_cashflow=option_cashflow,
            hedge_trade=hedge_trade,
            hedge_cashflow=hedge_cashflow,
            option_fees=option_fees,
            stock_costs=stock_costs,
            spread_capture=spread_capture,
            adverse_selection_cost=adverse_selection_cost,
            z_spot=z_spot,
            z_variance=z_variance,
            variance_floor_bound=variance_floor_bound,
            variance_floor_binds=self.variance_floor_binds,
        )

        self.state = next_state
        return next_state, reward, done, info

    def mark_to_market_wealth(
        self,
        cash: float,
        option_inventory: int,
        stock_position: float,
        spot: float,
        option_mid: float,
    ) -> float:
        """Return cash plus marked option and stock inventory."""
        return (
            cash
            + option_inventory * self.contract.contract_multiplier * option_mid
            + stock_position * spot
        )

    def _build_state(
        self,
        step_index: int,
        time: float,
        spot: float,
        variance: float,
        strike: float,
        tau: float,
        option_inventory: int,
        stock_position: float,
        cash: float,
        done: bool,
    ) -> OptionMMState:
        option_mid, option_delta = self._price_and_delta(spot, variance, tau, strike)
        wealth = self.mark_to_market_wealth(
            cash=cash,
            option_inventory=option_inventory,
            stock_position=stock_position,
            spot=spot,
            option_mid=option_mid,
        )
        net_delta = stock_position + (
            option_inventory * self.contract.contract_multiplier * option_delta
        )
        return OptionMMState(
            step_index=step_index,
            time=time,
            spot=spot,
            variance=variance,
            strike=strike,
            time_to_maturity=tau,
            option_mid=option_mid,
            option_delta=option_delta,
            option_inventory=option_inventory,
            stock_position=stock_position,
            cash=cash,
            wealth=wealth,
            net_delta=net_delta,
            done=done,
        )

    def _price_and_delta(
        self,
        spot: float,
        variance: float,
        tau: float,
        strike: float,
    ) -> Tuple[float, float]:
        price = black_scholes_call_price(
            spot=spot,
            strike=strike,
            tau=tau,
            rate=self.heston.rate,
            variance=variance,
        )
        delta = black_scholes_call_delta(
            spot=spot,
            strike=strike,
            tau=tau,
            rate=self.heston.rate,
            variance=variance,
        )
        return price, delta

    def _advance_heston(
        self, spot: float, variance: float
    ) -> Tuple[float, float, float, float, bool]:
        z_spot = float(self.path_rng.normal())
        z_independent = float(self.path_rng.normal())
        rho = self.heston.rho
        z_variance = rho * z_spot + sqrt(max(1.0 - rho * rho, 0.0)) * z_independent

        variance_pos = max(variance, 0.0)
        sqrt_v_dt = sqrt(variance_pos * self.dt)
        raw_variance_next = (
            variance
            + self.heston.kappa * (self.heston.theta - variance_pos) * self.dt
            + self.heston.xi * sqrt_v_dt * z_variance
        )
        variance_floor_bound = raw_variance_next < self.heston.variance_floor
        variance_next = max(raw_variance_next, self.heston.variance_floor)

        spot_next = spot * exp(
            (self.heston.mu - 0.5 * variance_pos) * self.dt + sqrt_v_dt * z_spot
        )
        return spot_next, variance_next, z_spot, z_variance, variance_floor_bound

    def _intensity_from_distance(self, quote_distance: float) -> float:
        if self.fills.base_intensity == 0.0 or self.fills.max_intensity == 0.0:
            return 0.0

        exponent = -self.fills.distance_slope * quote_distance
        exponent = float(np.clip(exponent, -700.0, 700.0))
        intensity = self.fills.base_intensity * exp(exponent)
        return float(np.clip(intensity, 0.0, self.fills.max_intensity))

    def _sample_contract_fills(self, annualized_intensity: float) -> Tuple[int, bool]:
        if self.fills.max_contracts_per_step == 0:
            return 0, False
        expected_count = max(annualized_intensity, 0.0) * self.dt
        fill_events = int(self.fill_rng.poisson(expected_count))
        clipped_events = min(fill_events, self.fills.max_contracts_per_step)
        fills = clipped_events * self.fills.lot_size
        return fills, fill_events > self.fills.max_contracts_per_step

    def _apply_same_step_both_fills_policy(
        self,
        bid_fills: int,
        ask_fills: int,
        bid_price: float,
        ask_price: float,
    ) -> Tuple[int, int, float, float, str, float]:
        bid_notional = bid_fills * bid_price
        ask_notional = ask_fills * ask_price
        if bid_fills <= 0 or ask_fills <= 0:
            return bid_fills, ask_fills, bid_notional, ask_notional, "none", 0.0

        policy = self.fills.same_step_both_fills_policy
        if policy == "allowed":
            return bid_fills, ask_fills, bid_notional, ask_notional, "both", 0.0

        first_side = "bid" if self.tie_rng.random() < 0.5 else "ask"
        if policy == "rejected":
            if first_side == "bid":
                ask_fills = 0
            else:
                bid_fills = 0
            return (
                bid_fills,
                ask_fills,
                bid_fills * bid_price,
                ask_fills * ask_price,
                first_side,
                0.0,
            )

        if policy == "mid_drift":
            matched_fills = min(bid_fills, ask_fills)
            mid_drift = self.fills.tick_size * self.fills.same_step_mid_drift_ticks
            if mid_drift <= 0.0 or matched_fills == 0:
                return bid_fills, ask_fills, bid_notional, ask_notional, first_side, 0.0

            if first_side == "bid":
                matched_ask_price = max(ask_price - mid_drift, 0.0)
                ask_notional = (
                    (ask_fills - matched_fills) * ask_price
                    + matched_fills * matched_ask_price
                )
            else:
                matched_bid_price = bid_price + mid_drift
                bid_notional = (
                    (bid_fills - matched_fills) * bid_price
                    + matched_fills * matched_bid_price
                )
            return bid_fills, ask_fills, bid_notional, ask_notional, first_side, mid_drift

        raise RuntimeError(f"unhandled same-step both-fills policy: {policy}")

    def _validate_action(self, action: OptionMMAction) -> None:
        self._validate_quotes(action.bid_price, action.ask_price)
        if not isfinite(action.hedge_trade):
            raise ValueError("hedge_trade must be finite")

    def _validate_quotes(self, bid_price: float, ask_price: float) -> None:
        if not isfinite(bid_price) or not isfinite(ask_price):
            raise ValueError("bid_price and ask_price must be finite")
        if bid_price < 0.0:
            raise ValueError("bid_price must be nonnegative")
        if ask_price < bid_price:
            raise ValueError("ask_price must be at least bid_price")
