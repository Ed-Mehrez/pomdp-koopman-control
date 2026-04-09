import os
import sys

import numpy as np
import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from applications.option_mm.env import (  # noqa: E402
    ExecutionCostSpec,
    FillModelSpec,
    HestonParams,
    OptionContractSpec,
    OptionMMAction,
    OptionMarketMakingEnv,
)


def assert_dataclasses_close(left, right):
    assert type(left) is type(right)
    for name in left.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if isinstance(left_value, float):
            assert left_value == pytest.approx(right_value), name
        else:
            assert left_value == right_value, name


def test_reset_uses_fixed_atm_strike_and_excludes_expiry_roll():
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(base_intensity=0.0),
        dt=1.0 / 252.0,
        horizon_steps=2,
        seed=123,
    )
    state0 = env.reset()

    assert env.contract.maturity_years == pytest.approx(1.0)
    assert state0.strike == pytest.approx(env.heston.spot0)
    assert state0.time_to_maturity == pytest.approx(env.contract.maturity_years)

    action = OptionMMAction(
        bid_price=max(state0.option_mid - 100.0, 0.0),
        ask_price=state0.option_mid + 100.0,
        hedge_trade=0.0,
    )
    state1, _, done1, _ = env.step(action)

    assert not done1
    assert state1.strike == pytest.approx(state0.strike)
    assert state1.time_to_maturity == pytest.approx(
        env.contract.maturity_years - env.dt
    )

    action = OptionMMAction(
        bid_price=max(state1.option_mid - 100.0, 0.0),
        ask_price=state1.option_mid + 100.0,
        hedge_trade=0.0,
    )
    state2, _, done2, _ = env.step(action)

    assert done2
    assert state2.done
    assert state2.time_to_maturity > 0.0
    assert state2.strike == pytest.approx(state0.strike)


def test_horizon_touching_expiry_is_rejected_in_v1():
    with pytest.raises(ValueError, match="excludes expiry"):
        OptionMarketMakingEnv(
            contract=OptionContractSpec(maturity_years=2.0 / 252.0),
            dt=1.0 / 252.0,
            horizon_steps=2,
        )


def test_initial_cash_sets_positive_starting_wealth_for_ce_metrics():
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(base_intensity=0.0),
        initial_cash=100_000.0,
        horizon_steps=2,
    )
    state0 = env.reset()

    assert state0.cash == pytest.approx(100_000.0)
    assert state0.wealth == pytest.approx(100_000.0)
    assert state0.net_delta == pytest.approx(0.0)


def test_fill_intensity_matches_frozen_exponential_formula():
    fills = FillModelSpec(
        base_intensity=7.0,
        distance_slope=0.5,
        max_intensity=1_000.0,
    )
    env = OptionMarketMakingEnv(fills=fills, horizon_steps=2)
    option_mid = 2.0

    bid_intensity, ask_intensity = env.fill_intensities(
        bid_price=1.75,
        ask_price=2.50,
        option_mid=option_mid,
    )

    assert bid_intensity == pytest.approx(7.0 * np.exp(-0.5 * 0.25))
    assert ask_intensity == pytest.approx(7.0 * np.exp(-0.5 * 0.50))

    wider_bid, wider_ask = env.fill_intensities(
        bid_price=1.00,
        ask_price=3.00,
        option_mid=option_mid,
    )
    assert wider_bid < bid_intensity
    assert wider_ask < ask_intensity


def test_default_same_step_mid_drift_removes_full_spread_roundtrip_subsidy():
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(
            base_intensity=1_000_000.0,
            distance_slope=0.0,
            max_intensity=1_000_000.0,
            max_contracts_per_step=1,
            same_step_both_fills_policy="mid_drift",
            tick_size=0.20,
            same_step_mid_drift_ticks=0.5,
        ),
        horizon_steps=2,
        seed=11,
    )
    state0 = env.reset()
    bid_price = state0.option_mid - 0.50
    ask_price = state0.option_mid + 0.50

    _, _, _, info = env.step(
        OptionMMAction(
            bid_price=bid_price,
            ask_price=ask_price,
            hedge_trade=0.0,
        )
    )

    multiplier = env.contract.contract_multiplier
    full_spread_cashflow = multiplier * (ask_price - bid_price)
    expected_drift = 0.20 * 0.5

    assert info.bid_fills == 1
    assert info.ask_fills == 1
    assert info.same_step_both_fills_policy == "mid_drift"
    assert info.same_step_first_side in {"bid", "ask"}
    assert info.same_step_mid_drift == pytest.approx(expected_drift)
    assert info.option_cashflow == pytest.approx(
        full_spread_cashflow - multiplier * expected_drift
    )
    assert info.spread_capture == pytest.approx(info.option_cashflow)


def test_forced_bid_fill_updates_inventory_cash_and_spread_capture():
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(
            base_intensity=1_000_000.0,
            distance_slope=10.0,
            max_intensity=1_000_000.0,
            max_contracts_per_step=1,
        ),
        costs=ExecutionCostSpec(option_fee_per_contract=0.0),
        horizon_steps=2,
        seed=7,
    )
    state0 = env.reset()
    bid_price = state0.option_mid - 0.25
    ask_price = state0.option_mid + 10.0

    state1, _, _, info = env.step(
        OptionMMAction(
            bid_price=bid_price,
            ask_price=ask_price,
            hedge_trade=0.0,
        )
    )

    multiplier = env.contract.contract_multiplier
    assert info.bid_fills == 1
    assert info.ask_fills == 0
    assert state1.option_inventory == 1
    assert info.option_cashflow == pytest.approx(-bid_price * multiplier)
    assert info.spread_capture == pytest.approx(0.25 * multiplier)
    assert state1.cash == pytest.approx(info.option_cashflow)
    assert state1.wealth == pytest.approx(
        env.mark_to_market_wealth(
            cash=state1.cash,
            option_inventory=state1.option_inventory,
            stock_position=state1.stock_position,
            spot=state1.spot,
            option_mid=state1.option_mid,
        )
    )


def test_zero_fill_hedge_trade_uses_pre_step_spot_for_cash_accounting():
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(base_intensity=0.0),
        costs=ExecutionCostSpec(stock_fee_per_share=0.0, stock_slippage_bps=0.0),
        horizon_steps=2,
        seed=42,
    )
    state0 = env.reset()
    hedge_trade = 10.0

    state1, _, _, info = env.step(
        OptionMMAction(
            bid_price=max(state0.option_mid - 100.0, 0.0),
            ask_price=state0.option_mid + 100.0,
            hedge_trade=hedge_trade,
        )
    )

    assert info.bid_fills == 0
    assert info.ask_fills == 0
    assert info.hedge_cashflow == pytest.approx(-hedge_trade * state0.spot)
    assert state1.cash == pytest.approx(-hedge_trade * state0.spot)
    assert state1.stock_position == pytest.approx(hedge_trade)
    assert state1.net_delta == pytest.approx(hedge_trade)


def test_variance_floor_binding_is_reported():
    env = OptionMarketMakingEnv(
        heston=HestonParams(
            variance0=1e-8,
            theta=0.0,
            kappa=1_000.0,
            xi=0.0,
            variance_floor=1e-6,
        ),
        fills=FillModelSpec(base_intensity=0.0),
        horizon_steps=1,
        seed=5,
    )
    state0 = env.reset()

    _, _, _, info = env.step(
        OptionMMAction(
            bid_price=max(state0.option_mid - 100.0, 0.0),
            ask_price=state0.option_mid + 100.0,
            hedge_trade=0.0,
        )
    )

    assert info.variance_floor_bound
    assert info.variance_floor_binds == 1
    assert env.variance_floor_binds == 1


def test_same_seed_and_actions_give_identical_transitions():
    kwargs = dict(
        heston=HestonParams(rho=-0.4, xi=0.3),
        fills=FillModelSpec(base_intensity=75.0, distance_slope=4.0),
        horizon_steps=3,
        seed=99,
    )
    env_a = OptionMarketMakingEnv(**kwargs)
    env_b = OptionMarketMakingEnv(**kwargs)
    state_a = env_a.reset()
    state_b = env_b.reset()

    for _ in range(3):
        action_a = OptionMMAction(
            bid_price=max(state_a.option_mid - 0.20, 0.0),
            ask_price=state_a.option_mid + 0.25,
            hedge_trade=-0.5 * state_a.net_delta,
        )
        action_b = OptionMMAction(
            bid_price=max(state_b.option_mid - 0.20, 0.0),
            ask_price=state_b.option_mid + 0.25,
            hedge_trade=-0.5 * state_b.net_delta,
        )

        state_a, reward_a, done_a, info_a = env_a.step(action_a)
        state_b, reward_b, done_b, info_b = env_b.step(action_b)

        assert reward_a == pytest.approx(reward_b)
        assert done_a == done_b
        assert_dataclasses_close(state_a, state_b)
        assert_dataclasses_close(info_a, info_b)


def test_heston_path_rng_is_independent_of_fill_policy_and_actions():
    env_no_fills = OptionMarketMakingEnv(
        fills=FillModelSpec(base_intensity=0.0),
        horizon_steps=3,
        seed=1234,
    )
    env_many_fills = OptionMarketMakingEnv(
        fills=FillModelSpec(
            base_intensity=1_000_000.0,
            distance_slope=0.0,
            max_intensity=1_000_000.0,
            max_contracts_per_step=1,
            same_step_both_fills_policy="mid_drift",
        ),
        horizon_steps=3,
        seed=1234,
    )
    state_no_fills = env_no_fills.reset()
    state_many_fills = env_many_fills.reset()

    for _ in range(3):
        state_no_fills, _, _, _ = env_no_fills.step(
            OptionMMAction(
                bid_price=max(state_no_fills.option_mid - 100.0, 0.0),
                ask_price=state_no_fills.option_mid + 100.0,
                hedge_trade=0.0,
            )
        )
        state_many_fills, _, _, _ = env_many_fills.step(
            OptionMMAction(
                bid_price=state_many_fills.option_mid,
                ask_price=state_many_fills.option_mid,
                hedge_trade=0.0,
            )
        )

        assert state_no_fills.spot == pytest.approx(state_many_fills.spot)
        assert state_no_fills.variance == pytest.approx(state_many_fills.variance)
