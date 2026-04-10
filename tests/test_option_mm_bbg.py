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
    HamiltonianTable,
    _hamiltonian_logistic,
    _optimal_quote_logistic,
    _build_hamiltonian_tables,
    solve_bbg_value_function,
    solve_bbg_value_function_no_gap,
    solver_diagnostics,
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


class TestScaleAudit:
    def test_all_initial_prices_positive(self):
        config = BBGBenchmarkConfig.paper_default()
        h = config.heston
        for opt in config.book.options:
            p = bs_call_price(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
            assert p > 0.0 and np.isfinite(p)

    def test_all_vegas_positive(self):
        config = BBGBenchmarkConfig.paper_default()
        h = config.heston
        for opt in config.book.options:
            v = bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
            assert v > 0.0 and np.isfinite(v)

    def test_all_lambdas_positive(self):
        config = BBGBenchmarkConfig.paper_default()
        h = config.heston
        liq = config.liquidity
        for opt in config.book.options:
            lam = liq.lambda_i(h.spot0, opt.strike)
            assert lam > 0.0 and np.isfinite(lam)

    def test_trade_sizes_positive(self):
        config = BBGBenchmarkConfig.paper_default()
        h = config.heston
        liq = config.liquidity
        for opt in config.book.options:
            p = bs_call_price(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
            z = liq.trade_size(p)
            assert z > 0.0 and np.isfinite(z)

    def test_exactly_one_option_exceeds_vega_limit(self):
        """K=12 T=1 is the only option with z_i*V_i > V_bar."""
        config = BBGBenchmarkConfig.paper_default()
        h = config.heston
        liq = config.liquidity
        n_exceed = 0
        for opt in config.book.options:
            p = bs_call_price(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
            v = bs_call_vega_sqrt_nu(h.spot0, opt.strike, opt.maturity, h.rate, h.nu0)
            z = liq.trade_size(p)
            if z * v > config.control.vega_limit:
                n_exceed += 1
                assert opt.strike == 12.0 and opt.maturity == 1.0
        assert n_exceed == 1


class TestHamiltonianTable:
    def test_table_agrees_with_direct(self):
        """Table interpolation should agree with direct optimization."""
        table = HamiltonianTable(7560.0, 4.0, 0.7, 150.0, p_lo=-3.0, p_hi=3.0, n_p=2000)
        for p in [-1.0, 0.0, 0.5, 1.5]:
            h_direct = _hamiltonian_logistic(p, 7560.0, 4.0, 0.7, 150.0)
            h_table = table.interp_H(p)
            rel_err = abs(h_direct - h_table) / max(abs(h_direct), 1e-6)
            assert rel_err < 0.05, f"H mismatch at p={p}: {h_direct} vs {h_table} (rel={rel_err:.3f})"

    def test_delta_star_finite(self):
        table = HamiltonianTable(7560.0, 4.0, 0.7, 150.0)
        for p in [-1.0, 0.0, 1.0]:
            d = table.interp_delta(p)
            assert np.isfinite(d)

    def test_h_nonnegative(self):
        table = HamiltonianTable(7560.0, 4.0, 0.7, 150.0)
        assert np.all(table.H_values >= -1e-10)


class TestSolver:
    def test_hamiltonian_positive_at_p_zero(self):
        h = _hamiltonian_logistic(0.0, 7560.0, 4.0, 0.7, 150.0)
        assert h > 0.0

    def test_risk_neutral_quotes_are_symmetric(self):
        config = BBGBenchmarkConfig.paper_default()
        ctrl = make_bbg_risk_neutral_controller(config)
        env = OptionBookMarketMakingEnv(config, seed=1)
        state = env.reset()
        action = ctrl(state)
        np.testing.assert_allclose(action.bid_distances, action.ask_distances, atol=1e-10)

    def test_risk_neutral_quotes_are_finite(self):
        config = BBGBenchmarkConfig.paper_default()
        ctrl = make_bbg_risk_neutral_controller(config)
        env = OptionBookMarketMakingEnv(config, seed=1)
        state = env.reset()
        action = ctrl(state)
        assert np.all(np.isfinite(action.bid_distances))
        assert np.all(action.bid_distances > 0)

    def test_single_option_3d_solver_runs(self):
        config = BBGBenchmarkConfig(
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
        )
        t, nu, vpi, values = solve_bbg_value_function(config, n_nu=5, n_vpi=10, n_time=20)
        assert values.shape == (21, 5, 10)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values[-1], 0.0, atol=1e-15)

    def test_gamma_zero_3d(self):
        config = BBGBenchmarkConfig(
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
            control=BBGControlSpec(gamma=0.0),
        )
        t, nu, vpi, values = solve_bbg_value_function(config, n_nu=5, n_vpi=10, n_time=20)
        assert np.all(np.isfinite(values))


class TestNoGapSolver:
    def test_single_option_no_gap_runs(self):
        config = BBGBenchmarkConfig(
            heston=BBGHestonSpec(kappa_p=3.0, theta_p=0.0225),
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
        )
        t, vpi, values = solve_bbg_value_function_no_gap(config, n_vpi=20, n_time=30)
        assert values.shape == (31, 20)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values[-1], 0.0, atol=1e-15)

    def test_multi_option_no_gap_runs(self):
        config = BBGBenchmarkConfig.no_gap_default()
        config = BBGBenchmarkConfig(
            heston=config.heston,
            book=BBGOptionBookSpec(strikes=(9.0, 10.0, 11.0), maturities=(1.0,)),
            control=config.control,
        )
        t, vpi, values = solve_bbg_value_function_no_gap(config, n_vpi=30, n_time=30)
        assert values.shape == (31, 30)
        assert np.all(np.isfinite(values))

    def test_no_gap_controller_returns_finite_quotes(self):
        config = BBGBenchmarkConfig(
            heston=BBGHestonSpec(kappa_p=3.0, theta_p=0.0225),
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
        )
        t, vpi, values = solve_bbg_value_function_no_gap(config, n_vpi=30, n_time=30)
        ctrl = make_bbg_numerical_controller(config, values, t, None, vpi)
        env = OptionBookMarketMakingEnv(config, seed=1)
        state = env.reset()
        action = ctrl(state)
        assert np.all(np.isfinite(action.bid_distances))
        assert np.all(np.isfinite(action.ask_distances))

    def test_diagnostics_report(self):
        config = BBGBenchmarkConfig(
            heston=BBGHestonSpec(kappa_p=3.0, theta_p=0.0225),
            book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
        )
        _, _, values = solve_bbg_value_function_no_gap(config, n_vpi=20, n_time=20)
        diag = solver_diagnostics(values)
        assert diag["n_nonfinite"] == 0
        assert diag["terminal_max_abs"] == pytest.approx(0.0, abs=1e-15)
