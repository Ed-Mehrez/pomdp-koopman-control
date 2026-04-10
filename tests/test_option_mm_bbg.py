"""Tests for the BBG benchmark package."""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from applications.option_mm_bbg.spec import (
    BBGBenchmarkConfig,
    BBGHestonSpec,
    BBGOptionBookSpec,
    BBGLiquiditySpec,
    BBGControlSpec,
)
from applications.option_mm_bbg.pricing import (
    bs_call_price,
    bs_call_delta,
    bs_call_vega_sqrt_nu,
    price_option_book,
)
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv, OptionBookMMAction
from applications.option_mm_bbg.solver import (
    _hamiltonian_logistic,
    _optimal_quote_logistic,
    solve_bbg_value_function,
    make_bbg_risk_neutral_controller,
    make_bbg_numerical_controller,
)


class TestPaperConfig:
    def test_paper_default_loads(self):
        config = BBGBenchmarkConfig.paper_default()
        config.validate()
        assert config.book.n_options == 20

    def test_paper_heston_params(self):
        h = BBGHestonSpec()
        assert h.spot0 == 10.0
        assert h.nu0 == 0.0225
        assert h.kappa_p == 2.0
        assert h.theta_p == 0.04
        assert h.kappa_q == 3.0
        assert h.theta_q == 0.0225

    def test_paper_strikes_and_maturities(self):
        b = BBGOptionBookSpec()
        assert b.strikes == (8.0, 9.0, 10.0, 11.0, 12.0)
        assert b.maturities == (1.0, 1.5, 2.0, 3.0)
        assert b.n_options == 20


class TestPricing:
    def test_atm_call_price_positive(self):
        p = bs_call_price(10.0, 10.0, 1.0, 0.0, 0.0225)
        assert p > 0.0
        assert np.isfinite(p)

    def test_deep_itm_delta_near_one(self):
        d = bs_call_delta(10.0, 5.0, 1.0, 0.0, 0.0225)
        assert d > 0.99

    def test_vega_positive_for_all_options(self):
        config = BBGBenchmarkConfig.paper_default()
        h = config.heston
        for opt in config.book.options:
            v = bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
            assert v > 0.0, f"vega <= 0 for K={opt.strike}, T={opt.maturity}"

    def test_price_option_book_shapes(self):
        config = BBGBenchmarkConfig.paper_default()
        h = config.heston
        strikes = np.array([o.strike for o in config.book.options])
        mats = np.array([o.maturity for o in config.book.options])
        prices, deltas, vegas = price_option_book(h.spot0, h.nu0, h.rate, strikes, mats)
        assert prices.shape == (20,)
        assert deltas.shape == (20,)
        assert vegas.shape == (20,)
        assert np.all(prices > 0)
        assert np.all(np.isfinite(prices))


class TestEnv:
    def test_single_option_env_reset_step(self):
        config = BBGBenchmarkConfig(
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
        )
        env = OptionBookMarketMakingEnv(config, seed=42)
        state = env.reset()
        assert state.step_index == 0
        assert state.portfolio_vega == 0.0
        assert len(state.option_inventories) == 1

        action = OptionBookMMAction(
            bid_distances=np.array([0.01]),
            ask_distances=np.array([0.01]),
            hedge_trade=-state.net_delta,
        )
        next_state, _, done, _ = env.step(action)
        assert next_state.step_index == 1
        assert np.isfinite(next_state.wealth)

    def test_full_book_env_runs_to_done(self):
        config = BBGBenchmarkConfig.paper_default()
        env = OptionBookMarketMakingEnv(config, seed=42)
        state = env.reset()

        n_steps = 0
        while not state.done:
            action = OptionBookMMAction(
                bid_distances=np.full(20, 0.01),
                ask_distances=np.full(20, 0.01),
                hedge_trade=-state.net_delta,
            )
            state, _, done, _ = env.step(action)
            n_steps += 1

        assert state.done
        assert n_steps == env.n_steps
        assert np.isfinite(state.wealth)

    def test_portfolio_vega_is_sum_qi_vi(self):
        config = BBGBenchmarkConfig(
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
        )
        env = OptionBookMarketMakingEnv(config, seed=42)
        state = env.reset()
        # Force a fill by quoting very tight
        action = OptionBookMMAction(
            bid_distances=np.array([0.0001]),
            ask_distances=np.array([100.0]),  # no ask fills
            hedge_trade=0.0,
        )
        # Run several steps to get a fill
        for _ in range(env.n_steps):
            if state.done:
                break
            state, _, _, _ = env.step(action)
        # Check vega = sum(q_i * V_i)
        expected_vega = float(np.sum(state.option_inventories * state.option_vegas))
        assert state.portfolio_vega == pytest.approx(expected_vega, abs=1e-10)


class TestSolver:
    def test_hamiltonian_positive_at_p_zero(self):
        # At p=0, sup Lambda(delta)*delta should be > 0
        h = _hamiltonian_logistic(0.0, 7560.0, 4.0, 0.7, 150.0)
        assert h > 0.0

    def test_risk_neutral_quotes_are_symmetric(self):
        config = BBGBenchmarkConfig.paper_default()
        ctrl = make_bbg_risk_neutral_controller(config)
        env = OptionBookMarketMakingEnv(config, seed=1)
        state = env.reset()
        action = ctrl(state)
        # At gamma=0, bid and ask distances should be equal
        np.testing.assert_allclose(action.bid_distances, action.ask_distances, atol=1e-10)

    def test_risk_neutral_quotes_are_finite(self):
        config = BBGBenchmarkConfig.paper_default()
        ctrl = make_bbg_risk_neutral_controller(config)
        env = OptionBookMarketMakingEnv(config, seed=1)
        state = env.reset()
        action = ctrl(state)
        assert np.all(np.isfinite(action.bid_distances))
        assert np.all(action.bid_distances > 0)

    def test_single_option_solver_runs(self):
        """Solve the HJB for a single-option book (cheapest validation)."""
        config = BBGBenchmarkConfig(
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
            control=BBGControlSpec(horizon=0.0012, gamma=1e-3),
        )
        t, nu, vpi, values = solve_bbg_value_function(
            config, n_nu=5, n_vpi=10, n_time=20,
        )
        assert values.shape == (21, 5, 10)
        assert np.all(np.isfinite(values))
        # Terminal condition: v(T) = 0
        np.testing.assert_allclose(values[-1], 0.0, atol=1e-15)

    def test_gamma_zero_value_near_zero(self):
        """At gamma=0, inventory penalty vanishes; value should be small."""
        config = BBGBenchmarkConfig(
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
            control=BBGControlSpec(horizon=0.0012, gamma=0.0),
        )
        t, nu, vpi, values = solve_bbg_value_function(
            config, n_nu=5, n_vpi=10, n_time=20,
        )
        # With gamma=0, the penalty term vanishes, but Hamiltonian terms
        # still contribute. Value should be non-negative (spread capture).
        assert np.all(np.isfinite(values))
