"""Controllers for the option market-making benchmark."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Any

import numpy as np

from .env import OptionMMAction, OptionMMState, OptionMarketMakingEnv
from .metrics import UtilitySpec


NO_QUOTE_ASK = 1.0e12
EpisodeController = Callable[[OptionMMState, Any | None], OptionMMAction]
InventoryVarianceEstimator = Callable[[OptionMMState, Any | None], float]


@dataclass(frozen=True)
class ConstantSpreadContext:
    """Context for symmetric constant-spread quoting."""

    half_spread: float = 0.05

    def validate(self) -> None:
        if self.half_spread < 0.0 or not isfinite(self.half_spread):
            raise ValueError("half_spread must be nonnegative and finite")


@dataclass(frozen=True)
class ASContext:
    """Context for textbook Avellaneda-Stoikov quotes."""

    v_hat: float
    gamma_inv: float
    k_intensity: float
    horizon_remaining: float

    def validate(self) -> None:
        if self.v_hat <= 0.0 or not isfinite(self.v_hat):
            raise ValueError("v_hat must be positive and finite")
        if self.gamma_inv <= 0.0 or not isfinite(self.gamma_inv):
            raise ValueError("gamma_inv must be positive and finite")
        if self.k_intensity <= 0.0 or not isfinite(self.k_intensity):
            raise ValueError("k_intensity must be positive and finite")
        if self.horizon_remaining < 0.0 or not isfinite(self.horizon_remaining):
            raise ValueError("horizon_remaining must be nonnegative and finite")


@dataclass(frozen=True)
class LinearRuleContext:
    """Context for the pinned affine inventory and delta-exposure rule."""

    v_hat: float
    gamma_inv: float
    k_intensity: float
    horizon_remaining: float
    contract_multiplier: float = 100.0
    hedge_gain: float = 1.0
    max_abs_hedge_trade: float = 1_000.0

    def validate(self) -> None:
        _validate_common_control_params(
            v_hat=self.v_hat,
            gamma_inv=self.gamma_inv,
            k_intensity=self.k_intensity,
            horizon_remaining=self.horizon_remaining,
        )
        if self.contract_multiplier <= 0.0 or not isfinite(self.contract_multiplier):
            raise ValueError("contract_multiplier must be positive and finite")
        if self.hedge_gain < 0.0 or not isfinite(self.hedge_gain):
            raise ValueError("hedge_gain must be nonnegative and finite")
        if self.max_abs_hedge_trade < 0.0 or not isfinite(self.max_abs_hedge_trade):
            raise ValueError("max_abs_hedge_trade must be nonnegative and finite")


@dataclass(frozen=True)
class SDREContext:
    """Context for the local SDRE controller on (q, h, V_hat, tau)."""

    v_hat: float
    gamma_inv: float
    k_intensity: float
    horizon_remaining: float
    base_intensity: float = 50.0
    dt: float = 1.0 / 252.0
    contract_multiplier: float = 100.0
    hedge_penalty: float = 1e-8
    skew_penalty_floor: float = 1e-8
    max_abs_skew: float = 5.0
    max_abs_hedge_trade: float = 1_000.0

    def validate(self) -> None:
        _validate_common_control_params(
            v_hat=self.v_hat,
            gamma_inv=self.gamma_inv,
            k_intensity=self.k_intensity,
            horizon_remaining=self.horizon_remaining,
        )
        if self.base_intensity < 0.0 or not isfinite(self.base_intensity):
            raise ValueError("base_intensity must be nonnegative and finite")
        if self.dt <= 0.0 or not isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if self.contract_multiplier <= 0.0 or not isfinite(self.contract_multiplier):
            raise ValueError("contract_multiplier must be positive and finite")
        if self.hedge_penalty < 0.0 or not isfinite(self.hedge_penalty):
            raise ValueError("hedge_penalty must be nonnegative and finite")
        if self.skew_penalty_floor <= 0.0 or not isfinite(self.skew_penalty_floor):
            raise ValueError("skew_penalty_floor must be positive and finite")
        if self.max_abs_skew < 0.0 or not isfinite(self.max_abs_skew):
            raise ValueError("max_abs_skew must be nonnegative and finite")
        if self.max_abs_hedge_trade < 0.0 or not isfinite(self.max_abs_hedge_trade):
            raise ValueError("max_abs_hedge_trade must be nonnegative and finite")


def no_quote(state: OptionMMState, ctx: Any = None) -> OptionMMAction:
    """Do not post usable liquidity and do not hedge."""
    del state, ctx
    return OptionMMAction(bid_price=0.0, ask_price=NO_QUOTE_ASK, hedge_trade=0.0)


def constant_spread(
    state: OptionMMState,
    ctx: ConstantSpreadContext,
) -> OptionMMAction:
    """Symmetric quote around the current mid, with no hedge or inventory skew."""
    ctx.validate()
    return OptionMMAction(
        bid_price=max(state.option_mid - ctx.half_spread, 0.0),
        ask_price=state.option_mid + ctx.half_spread,
        hedge_trade=0.0,
    )


def make_risk_neutral_optimal(
    env: OptionMarketMakingEnv,
) -> EpisodeController:
    """Factory for the risk-neutral symmetric optimum at ±1/k."""
    half_spread = 1.0 / env.fills.distance_slope

    def controller(state: OptionMMState, history: Any | None = None) -> OptionMMAction:
        del history
        return OptionMMAction(
            bid_price=max(state.option_mid - half_spread, 0.0),
            ask_price=state.option_mid + half_spread,
            hedge_trade=-state.net_delta,
        )

    return controller


def make_linear_inventory_skew(
    env: OptionMarketMakingEnv,
    inventory_variance_estimator: InventoryVarianceEstimator,
    utility: UtilitySpec,
) -> EpisodeController:
    """Factory for the legacy first-order inventory-skew diagnostic controller."""
    multiplier = env.contract.contract_multiplier
    distance_slope = env.fills.distance_slope
    dt = env.dt
    horizon_steps = env.horizon_steps

    def controller(state: OptionMMState, history: Any | None = None) -> OptionMMAction:
        sigma_sq_inv = inventory_variance_estimator(state, history)
        gamma_local = utility.arrow_pratt(state.wealth)
        horizon_remaining = max((horizon_steps - state.step_index) * dt, 0.0)
        half_spread_base = 1.0 / distance_slope
        p_contract = (
            gamma_local
            * sigma_sq_inv
            * horizon_remaining
            * state.option_inventory
        )
        p_quote = p_contract / multiplier
        bid_distance = max(0.0, half_spread_base + p_quote)
        ask_distance = max(0.0, half_spread_base - p_quote)
        return OptionMMAction(
            bid_price=max(state.option_mid - bid_distance, 0.0),
            ask_price=state.option_mid + ask_distance,
            hedge_trade=-state.net_delta,
        )

    return controller


# Legacy aliases kept for historical experiment compatibility.
make_bergault_gueant_closed_form = make_risk_neutral_optimal
make_sdre_controller_v2 = make_linear_inventory_skew


def avellaneda_stoikov(
    state: OptionMMState,
    ctx: ASContext,
) -> OptionMMAction:
    """Textbook Avellaneda-Stoikov reservation-price and spread quote."""
    ctx.validate()
    risk_term = ctx.gamma_inv * ctx.v_hat * ctx.horizon_remaining
    reservation_price = state.option_mid - state.option_inventory * risk_term
    half_spread = 0.5 * risk_term + (
        log(1.0 + ctx.gamma_inv / ctx.k_intensity) / ctx.gamma_inv
    )
    return OptionMMAction(
        bid_price=max(reservation_price - half_spread, 0.0),
        ask_price=max(reservation_price + half_spread, 0.0),
        hedge_trade=0.0,
    )


def linear_inventory_rule(
    state: OptionMMState,
    ctx: LinearRuleContext,
) -> OptionMMAction:
    """Affine inventory skew on (q, h, V_hat, tau), with pinned hedge-to-zero."""
    ctx.validate()
    risk_term = _as_risk_term(
        v_hat=ctx.v_hat,
        gamma_inv=ctx.gamma_inv,
        horizon_remaining=ctx.horizon_remaining,
    )
    half_spread = _as_half_spread(
        v_hat=ctx.v_hat,
        gamma_inv=ctx.gamma_inv,
        k_intensity=ctx.k_intensity,
        horizon_remaining=ctx.horizon_remaining,
    )
    option_equivalent_delta = _option_equivalent_delta(state, ctx.contract_multiplier)
    reservation_price = state.option_mid - risk_term * (
        state.option_inventory + option_equivalent_delta
    )
    hedge_trade = _clip(
        -ctx.hedge_gain * state.net_delta,
        ctx.max_abs_hedge_trade,
    )
    return _quotes_from_reservation(state, reservation_price, half_spread, hedge_trade)


def sdre_controller(
    state: OptionMMState,
    ctx: SDREContext,
) -> OptionMMAction:
    """SDRE-on-(q, h, V_hat, tau) using a local Itô-quadratic expansion."""
    ctx.validate()
    skew, hedge_trade = _sdre_optimal_action(state, ctx)
    half_spread = _as_half_spread(
        v_hat=ctx.v_hat,
        gamma_inv=ctx.gamma_inv,
        k_intensity=ctx.k_intensity,
        horizon_remaining=ctx.horizon_remaining,
    )
    return _quotes_from_reservation(
        state,
        state.option_mid + skew,
        half_spread,
        hedge_trade,
    )


def _sdre_optimal_action(
    state: OptionMMState,
    ctx: SDREContext,
) -> tuple[float, float]:
    """Solve the one-step local quadratic SDRE action around the current state."""
    risk_term = _as_risk_term(
        v_hat=ctx.v_hat,
        gamma_inv=ctx.gamma_inv,
        horizon_remaining=ctx.horizon_remaining,
    )
    if risk_term == 0.0:
        return 0.0, 0.0

    half_spread = _as_half_spread(
        v_hat=ctx.v_hat,
        gamma_inv=ctx.gamma_inv,
        k_intensity=ctx.k_intensity,
        horizon_remaining=ctx.horizon_remaining,
    )
    base_lambda = ctx.base_intensity * exp(-ctx.k_intensity * half_spread)
    fill_skew_gain = 2.0 * ctx.k_intensity * base_lambda * ctx.dt
    contract_delta = max(
        abs(state.option_delta) * ctx.contract_multiplier,
        1e-6,
    )

    state_vec = np.array(
        [float(state.option_inventory), float(state.net_delta)],
        dtype=float,
    )
    dynamics = np.array(
        [
            [fill_skew_gain, 0.0],
            [contract_delta * fill_skew_gain, 1.0],
        ],
        dtype=float,
    )

    inventory_weight = max(risk_term, 1e-12)
    delta_weight = inventory_weight / (contract_delta * contract_delta)
    spread_curvature = (
        base_lambda
        * ctx.dt
        * max(
            2.0 * ctx.k_intensity - ctx.k_intensity * ctx.k_intensity * half_spread,
            ctx.skew_penalty_floor,
        )
    )
    state_cost = np.diag([inventory_weight, delta_weight])
    action_cost = np.diag(
        [
            max(spread_curvature, ctx.skew_penalty_floor),
            max(ctx.hedge_penalty, 0.0),
        ]
    )
    hessian = dynamics.T @ state_cost @ dynamics + action_cost
    gradient = dynamics.T @ state_cost @ state_vec

    try:
        action = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        action = -np.linalg.pinv(hessian) @ gradient

    skew = _clip(float(action[0]), ctx.max_abs_skew)
    hedge_trade = _clip(float(action[1]), ctx.max_abs_hedge_trade)
    return skew, hedge_trade


def _validate_common_control_params(
    v_hat: float,
    gamma_inv: float,
    k_intensity: float,
    horizon_remaining: float,
) -> None:
    if v_hat < 0.0 or not isfinite(v_hat):
        raise ValueError("v_hat must be nonnegative and finite")
    if gamma_inv <= 0.0 or not isfinite(gamma_inv):
        raise ValueError("gamma_inv must be positive and finite")
    if k_intensity <= 0.0 or not isfinite(k_intensity):
        raise ValueError("k_intensity must be positive and finite")
    if horizon_remaining < 0.0 or not isfinite(horizon_remaining):
        raise ValueError("horizon_remaining must be nonnegative and finite")


def _as_risk_term(
    v_hat: float,
    gamma_inv: float,
    horizon_remaining: float,
) -> float:
    return max(gamma_inv * v_hat * horizon_remaining, 0.0)


def _as_half_spread(
    v_hat: float,
    gamma_inv: float,
    k_intensity: float,
    horizon_remaining: float,
) -> float:
    return 0.5 * _as_risk_term(
        v_hat=v_hat,
        gamma_inv=gamma_inv,
        horizon_remaining=horizon_remaining,
    ) + (log(1.0 + gamma_inv / k_intensity) / gamma_inv)


def _option_equivalent_delta(
    state: OptionMMState,
    contract_multiplier: float,
) -> float:
    contract_delta = abs(state.option_delta) * contract_multiplier
    if contract_delta <= 1e-6:
        return 0.0
    return state.net_delta / contract_delta


def _quotes_from_reservation(
    state: OptionMMState,
    reservation_price: float,
    half_spread: float,
    hedge_trade: float,
) -> OptionMMAction:
    bid = max(reservation_price - half_spread, 0.0)
    ask = max(reservation_price + half_spread, bid)
    return OptionMMAction(
        bid_price=bid,
        ask_price=ask,
        hedge_trade=hedge_trade,
    )


def _clip(value: float, max_abs_value: float) -> float:
    if max_abs_value == 0.0:
        return 0.0
    return float(np.clip(value, -max_abs_value, max_abs_value))
