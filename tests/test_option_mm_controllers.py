import os
import sys
from dataclasses import replace
from math import log

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from applications.option_mm.controllers import (  # noqa: E402
    ASContext,
    ConstantSpreadContext,
    LinearRuleContext,
    SDREContext,
    avellaneda_stoikov,
    constant_spread,
    linear_inventory_rule,
    make_linear_inventory_skew,
    make_risk_neutral_optimal,
    no_quote,
    sdre_controller,
)
from applications.option_mm.bbg_solver import make_bbg_numerical  # noqa: E402
from applications.option_mm.env import OptionMarketMakingEnv  # noqa: E402
from applications.option_mm.inventory_variance import (  # noqa: E402
    bergault_gueant_heston_estimator,
    empirical_sliding_window_estimator,
)
from applications.option_mm.hybrid_residual_controller import (  # noqa: E402
    BBGQuoteLookup,
    HybridTrainingBuffer,
    collect_hybrid_training_data,
    make_hybrid_residual_controller,
    train_hybrid_residual_model,
)
from applications.option_mm.local_kernel_controller import (  # noqa: E402
    KernelRewardModel,
    TrainingBuffer,
    collect_training_data,
    extract_state_features,
    make_local_kernel_controller,
    median_bandwidth,
    train_local_kernel_model,
)
from applications.option_mm.metrics import (  # noqa: E402
    UtilitySpec,
    cara_utility,
    crra_utility,
    quadratic_utility,
)

import numpy as np  # noqa: E402


def test_no_quote_returns_unfillable_quotes():
    state = OptionMarketMakingEnv(seed=1).reset()
    action = no_quote(state)

    assert action.bid_price == 0.0
    assert action.ask_price > 1e9
    assert action.hedge_trade == 0.0


def test_constant_spread_quotes_symmetrically_around_mid():
    state = OptionMarketMakingEnv(seed=1).reset()
    action = constant_spread(state, ConstantSpreadContext(half_spread=0.25))

    assert action.bid_price == pytest.approx(state.option_mid - 0.25)
    assert action.ask_price == pytest.approx(state.option_mid + 0.25)
    assert action.hedge_trade == 0.0


def test_avellaneda_stoikov_matches_textbook_formula_and_inventory_sign():
    state = OptionMarketMakingEnv(seed=1).reset()
    state = replace(state, option_inventory=3)
    ctx = ASContext(
        v_hat=0.04,
        gamma_inv=0.1,
        k_intensity=5.0,
        horizon_remaining=20.0 / 252.0,
    )
    action = avellaneda_stoikov(state, ctx)

    risk_term = ctx.gamma_inv * ctx.v_hat * ctx.horizon_remaining
    reservation = state.option_mid - state.option_inventory * risk_term
    half_spread = 0.5 * risk_term + (
        log(1.0 + ctx.gamma_inv / ctx.k_intensity) / ctx.gamma_inv
    )

    assert action.bid_price == pytest.approx(reservation - half_spread)
    assert action.ask_price == pytest.approx(reservation + half_spread)
    assert (action.bid_price + action.ask_price) / 2.0 < state.option_mid


def test_controller_contexts_validate_parameters():
    state = OptionMarketMakingEnv(seed=1).reset()

    with pytest.raises(ValueError):
        constant_spread(state, ConstantSpreadContext(half_spread=-0.01))
    with pytest.raises(ValueError):
        avellaneda_stoikov(
            state,
            ASContext(
                v_hat=0.0,
                gamma_inv=0.1,
                k_intensity=5.0,
                horizon_remaining=20.0 / 252.0,
            ),
        )


def test_linear_inventory_rule_is_finite_deterministic_and_valid():
    state = OptionMarketMakingEnv(seed=1).reset()
    state = replace(state, option_inventory=3, stock_position=-25.0)
    state = replace(state, net_delta=state.stock_position + 3 * 100.0 * state.option_delta)
    ctx = LinearRuleContext(
        v_hat=0.04,
        gamma_inv=0.1,
        k_intensity=5.0,
        horizon_remaining=20.0 / 252.0,
    )

    action_a = linear_inventory_rule(state, ctx)
    action_b = linear_inventory_rule(state, ctx)

    assert action_a == action_b
    assert action_a.bid_price <= action_a.ask_price
    assert action_a.bid_price >= 0.0
    assert abs(action_a.hedge_trade + state.net_delta) < 1e-12


def test_sdre_controller_is_finite_deterministic_and_valid():
    state = OptionMarketMakingEnv(seed=1).reset()
    state = replace(state, option_inventory=3, stock_position=-25.0)
    state = replace(state, net_delta=state.stock_position + 3 * 100.0 * state.option_delta)
    ctx = SDREContext(
        v_hat=0.04,
        gamma_inv=0.1,
        k_intensity=5.0,
        horizon_remaining=20.0 / 252.0,
    )

    action_a = sdre_controller(state, ctx)
    action_b = sdre_controller(state, ctx)

    assert action_a == action_b
    assert action_a.bid_price <= action_a.ask_price
    assert action_a.bid_price >= 0.0
    assert abs(action_a.hedge_trade) < 1_000.0


def test_augmented_controllers_reduce_to_no_skew_at_zero_risk_and_state():
    state = OptionMarketMakingEnv(seed=1).reset()
    linear_ctx = LinearRuleContext(
        v_hat=0.0,
        gamma_inv=0.1,
        k_intensity=5.0,
        horizon_remaining=20.0 / 252.0,
    )
    sdre_ctx = SDREContext(
        v_hat=0.0,
        gamma_inv=0.1,
        k_intensity=5.0,
        horizon_remaining=20.0 / 252.0,
    )

    linear_action = linear_inventory_rule(state, linear_ctx)
    sdre_action = sdre_controller(state, sdre_ctx)
    half_spread = log(1.0 + linear_ctx.gamma_inv / linear_ctx.k_intensity) / (
        linear_ctx.gamma_inv
    )

    assert linear_action.bid_price == pytest.approx(state.option_mid - half_spread)
    assert linear_action.ask_price == pytest.approx(state.option_mid + half_spread)
    assert linear_action.hedge_trade == 0.0
    assert sdre_action.bid_price == pytest.approx(state.option_mid - half_spread)
    assert sdre_action.ask_price == pytest.approx(state.option_mid + half_spread)
    assert sdre_action.hedge_trade == 0.0


def test_augmented_contexts_validate_parameters():
    state = OptionMarketMakingEnv(seed=1).reset()

    with pytest.raises(ValueError):
        linear_inventory_rule(
            state,
            LinearRuleContext(
                v_hat=-1.0,
                gamma_inv=0.1,
                k_intensity=5.0,
                horizon_remaining=20.0 / 252.0,
            ),
        )


def test_risk_neutral_optimal_is_symmetric_at_1_over_k():
    env = OptionMarketMakingEnv(seed=1)
    state = env.reset()
    state = replace(state, option_inventory=3, net_delta=12.5)
    controller = make_risk_neutral_optimal(env)

    action = controller(state)

    assert action.bid_price == pytest.approx(state.option_mid - 0.2)
    assert action.ask_price == pytest.approx(state.option_mid + 0.2)
    assert action.hedge_trade == pytest.approx(-12.5)


def test_linear_inventory_skew_recovers_risk_neutral_at_zero_inventory():
    env = OptionMarketMakingEnv(seed=1, initial_cash=100_000.0)
    state = env.reset()
    state = replace(state, net_delta=7.0)
    baseline_controller = make_risk_neutral_optimal(env)
    estimator = bergault_gueant_heston_estimator(env, state)
    linear_skew = make_linear_inventory_skew(env, estimator, crra_utility(2.0))

    assert linear_skew(state) == baseline_controller(state)


def test_linear_inventory_skew_reduces_to_risk_neutral_when_arrow_pratt_is_zero():
    """Track A identity check: AP(W) -> 0 must give bit-identical risk-neutral quotes.

    The linear-inventory-skew controller is risk-neutral quoting plus an
    Arrow-Pratt skew correction. In the
    AP(W) -> 0 limit the correction must vanish exactly, not approximately.
    Uses a stub UtilitySpec returning ``arrow_pratt = 0.0`` and a constant
    inventory variance estimator so the reduction is bit-identical at every
    step on a fixed seed, with non-trivial option inventory and net delta to
    actually exercise the AP code path. If this test fails, the AP -> BG
    reduction in ``make_linear_inventory_skew`` is broken and the benchmark
    downstream gates are meaningless.
    """
    env = OptionMarketMakingEnv(seed=1, initial_cash=100_000.0)
    state = env.reset()
    state = replace(state, option_inventory=3, net_delta=12.5)

    zero_ap_utility = UtilitySpec(
        u=lambda wealth: 0.0 * wealth,
        ce=lambda mean_u: 0.0,
        ce_grad=lambda mean_u: 0.0,
        arrow_pratt=lambda wealth: 0.0,
        requires_positive_wealth=False,
    )
    constant_estimator = lambda state, history=None: 1.0  # noqa: E731

    baseline_controller = make_risk_neutral_optimal(env)
    linear_skew = make_linear_inventory_skew(env, constant_estimator, zero_ap_utility)

    n_steps_checked = 0
    while not state.done:
        action_baseline = baseline_controller(state)
        action_linear = linear_skew(state)
        assert action_linear.bid_price == pytest.approx(
            action_baseline.bid_price, abs=1e-12, rel=0.0
        )
        assert action_linear.ask_price == pytest.approx(
            action_baseline.ask_price, abs=1e-12, rel=0.0
        )
        assert action_linear.hedge_trade == pytest.approx(
            action_baseline.hedge_trade, abs=1e-12, rel=0.0
        )
        n_steps_checked += 1
        state, _, _, _ = env.step(action_linear)

    assert n_steps_checked == env.horizon_steps


def test_linear_inventory_skew_sign_long_inventory():
    env = OptionMarketMakingEnv(seed=1)
    state = env.reset()
    state = replace(state, option_inventory=1, net_delta=0.0)
    estimator = bergault_gueant_heston_estimator(env, state)
    linear_skew = make_linear_inventory_skew(env, estimator, cara_utility(1.0e-3))

    action = linear_skew(state)
    bid_distance = state.option_mid - action.bid_price
    ask_distance = action.ask_price - state.option_mid

    assert bid_distance > 0.2
    assert ask_distance < 0.2
    assert ask_distance < bid_distance


def test_linear_inventory_skew_sign_short_inventory():
    env = OptionMarketMakingEnv(seed=1)
    state = env.reset()
    state = replace(state, option_inventory=-1, net_delta=0.0)
    estimator = bergault_gueant_heston_estimator(env, state)
    linear_skew = make_linear_inventory_skew(env, estimator, cara_utility(1.0e-3))

    action = linear_skew(state)
    bid_distance = state.option_mid - action.bid_price
    ask_distance = action.ask_price - state.option_mid

    assert bid_distance < 0.2
    assert ask_distance > 0.2
    assert bid_distance < ask_distance


def test_linear_inventory_skew_magnitude_crra_2_at_w_1e5():
    env = OptionMarketMakingEnv(seed=1)
    state = env.reset()
    state = replace(state, option_inventory=1, wealth=100_000.0, net_delta=0.0)
    estimator = bergault_gueant_heston_estimator(env, state)
    linear_skew = make_linear_inventory_skew(env, estimator, crra_utility(2.0))

    action = linear_skew(state)
    bid_distance = state.option_mid - action.bid_price
    inventory_skew = bid_distance - 0.2

    assert 0.012 <= inventory_skew <= 0.020


def test_bbg_numerical_reduces_to_risk_neutral_at_gamma_zero():
    env = OptionMarketMakingEnv(seed=1)
    state = env.reset()
    state = replace(state, option_inventory=3, net_delta=12.5)
    baseline_controller = make_risk_neutral_optimal(env)
    bbg_controller = make_bbg_numerical(
        env,
        state,
        gamma=0.0,
        max_inventory=10,
    )

    n_steps_checked = 0
    while not state.done:
        action_baseline = baseline_controller(state)
        action_bbg = bbg_controller(state)
        assert action_bbg.bid_price == pytest.approx(
            action_baseline.bid_price, abs=1e-6, rel=0.0
        )
        assert action_bbg.ask_price == pytest.approx(
            action_baseline.ask_price, abs=1e-6, rel=0.0
        )
        assert action_bbg.hedge_trade == pytest.approx(
            action_baseline.hedge_trade, abs=1e-12, rel=0.0
        )
        n_steps_checked += 1
        state, _, _, _ = env.step(action_bbg)

    assert n_steps_checked == env.horizon_steps


def test_bbg_numerical_skews_quotes_with_inventory_at_positive_gamma():
    env = OptionMarketMakingEnv(seed=1)
    state = env.reset()
    state = replace(state, option_inventory=1, net_delta=0.0)
    controller = make_bbg_numerical(
        env,
        state,
        gamma=2.0e-5,
        max_inventory=10,
    )

    action = controller(state)
    bid_distance = state.option_mid - action.bid_price
    ask_distance = action.ask_price - state.option_mid

    assert bid_distance > 0.2
    assert ask_distance < 0.2
    assert ask_distance < bid_distance


def test_arrow_pratt_factories():
    wealth = 100_000.0

    assert crra_utility(2.0).arrow_pratt(wealth) == pytest.approx(2.0e-5)
    assert cara_utility(1.0e-3).arrow_pratt(wealth) == pytest.approx(1.0e-3)
    assert quadratic_utility(1.0e-6).arrow_pratt(wealth) == pytest.approx(
        1.0e-6 / (1.0 - 1.0e-6 * wealth)
    )


def test_empirical_sliding_window_estimator_is_episode_local_and_finite():
    env = OptionMarketMakingEnv(seed=1)
    state = env.reset()
    estimator_a = empirical_sliding_window_estimator(window_length=10, env=env)
    estimator_b = empirical_sliding_window_estimator(window_length=10, env=env)

    first_a = estimator_a(state)
    first_b = estimator_b(state)
    assert first_a == pytest.approx(0.0)
    assert first_b == pytest.approx(0.0)

    for _ in range(20):
        state, _, _, _ = env.step(no_quote(state))
        value_a = estimator_a(state)
        value_b = estimator_b(state)
        assert value_a == pytest.approx(value_b)
        assert value_a >= 0.0


def test_empirical_and_bg_estimators_time_average_are_same_order_on_heston():
    empirical_values = []
    bg_values = []
    for seed in range(500):
        env = OptionMarketMakingEnv(seed=seed)
        state = env.reset()
        bg_estimator = bergault_gueant_heston_estimator(env, state)
        empirical_estimator = empirical_sliding_window_estimator(window_length=10, env=env)

        for _ in range(env.horizon_steps):
            state, _, _, _ = env.step(no_quote(state))
            empirical_values.append(empirical_estimator(state))
            bg_values.append(bg_estimator(state))

    empirical_mean = float(sum(empirical_values) / len(empirical_values))
    bg_mean = float(sum(bg_values) / len(bg_values))

    assert empirical_mean == pytest.approx(bg_mean, rel=0.3)
    with pytest.raises(ValueError):
        sdre_controller(
            state,
            SDREContext(
                v_hat=0.04,
                gamma_inv=0.1,
                k_intensity=5.0,
                horizon_remaining=20.0 / 252.0,
                base_intensity=-1.0,
            ),
        )
    with pytest.raises(ValueError):
        avellaneda_stoikov(
            state,
            ASContext(
                v_hat=0.04,
                gamma_inv=0.0,
                k_intensity=5.0,
                horizon_remaining=20.0 / 252.0,
            ),
        )
    with pytest.raises(ValueError):
        avellaneda_stoikov(
            state,
            ASContext(
                v_hat=0.04,
                gamma_inv=0.1,
                k_intensity=0.0,
                horizon_remaining=20.0 / 252.0,
            ),
        )


# ---------------------------------------------------------------------------
# Local kernel controller tests (Track B)
# ---------------------------------------------------------------------------


class TestExtractStateFeatures:
    def test_returns_4d_vector(self):
        env = OptionMarketMakingEnv(seed=1)
        state = env.reset()
        feat = extract_state_features(state, env, v_hat=0.04)
        assert feat.shape == (4,)

    def test_zero_inventory_gives_zero_q_norm(self):
        env = OptionMarketMakingEnv(seed=1)
        state = env.reset()
        feat = extract_state_features(state, env, v_hat=0.04)
        assert feat[0] == 0.0  # q_norm at reset is zero

    def test_tau_frac_is_one_at_reset(self):
        env = OptionMarketMakingEnv(seed=1)
        state = env.reset()
        feat = extract_state_features(state, env, v_hat=0.04)
        assert feat[2] == pytest.approx(1.0)

    def test_v_hat_norm_is_one_at_theta(self):
        env = OptionMarketMakingEnv(seed=1)
        state = env.reset()
        feat = extract_state_features(state, env, v_hat=env.heston.theta)
        assert feat[3] == pytest.approx(1.0)


class TestKernelRewardModel:
    def test_fit_and_predict_shape(self):
        rng = np.random.default_rng(42)
        n = 50
        features = rng.standard_normal((n, 4))
        actions = rng.standard_normal((n, 2))
        rewards = rng.standard_normal(n)

        model = KernelRewardModel(bandwidth=1.0, ridge_alpha=1e-2)
        model.fit(features, actions, rewards)
        assert model.is_fitted

        pred = model.predict(features[:5], actions[:5])
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))

    def test_prediction_interpolates_training_data(self):
        rng = np.random.default_rng(42)
        n = 30
        features = rng.standard_normal((n, 4))
        actions = rng.standard_normal((n, 2))
        # Simple linear target to test interpolation
        rewards = features[:, 0] + 2.0 * actions[:, 0]

        model = KernelRewardModel(bandwidth=1.0, ridge_alpha=1e-4)
        model.fit(features, actions, rewards)
        pred = model.predict(features, actions)

        # With low ridge and training points, should interpolate well
        residuals = np.abs(pred - rewards)
        assert np.mean(residuals) < 1.0

    def test_unfitted_model_raises(self):
        model = KernelRewardModel()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((1, 4)), np.zeros((1, 2)))

    def test_invalid_bandwidth_raises(self):
        with pytest.raises(ValueError):
            KernelRewardModel(bandwidth=-1.0)
        with pytest.raises(ValueError):
            KernelRewardModel(bandwidth=0.0)


class TestTrainingBuffer:
    def test_add_and_size(self):
        buf = TrainingBuffer()
        assert buf.size == 0

        buf.add(np.array([1.0, 2.0]), np.array([0.5, 0.1]), 3.14)
        assert buf.size == 1

    def test_as_arrays(self):
        buf = TrainingBuffer()
        buf.add(np.array([1.0, 2.0]), np.array([0.5]), 1.0)
        buf.add(np.array([3.0, 4.0]), np.array([0.6]), 2.0)
        f, a, r = buf.as_arrays()
        assert f.shape == (2, 2)
        assert a.shape == (2, 1)
        assert r.shape == (2,)


class TestCollectTrainingData:
    def test_collects_data(self):
        buf = collect_training_data(seeds=(0, 1), horizon_steps=5)
        assert buf.size == 10  # 2 seeds x 5 steps each
        f, a, r = buf.as_arrays()
        assert f.shape == (10, 4)
        assert a.shape == (10, 2)
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(r))


class TestMedianBandwidth:
    def test_returns_positive(self):
        rng = np.random.default_rng(42)
        f = rng.standard_normal((100, 4))
        a = rng.standard_normal((100, 2))
        bw = median_bandwidth(f, a)
        assert bw > 0.0


class TestMakeLocalKernelController:
    def test_controller_produces_valid_actions(self):
        # Train a tiny model
        model = train_local_kernel_model(
            training_seeds=(0, 1, 2),
            horizon_steps=5,
            max_training_samples=100,
        )

        # Deploy
        env = OptionMarketMakingEnv(horizon_steps=5, seed=99)
        state = env.reset()
        controller = make_local_kernel_controller(env, model, state)

        actions_seen = []
        while not state.done:
            action = controller(state)
            assert action.bid_price >= 0.0
            assert action.ask_price >= action.bid_price
            assert np.isfinite(action.hedge_trade)
            actions_seen.append(action)
            state, _, _, _ = env.step(action)

        assert len(actions_seen) == 5

    def test_controller_runs_full_episode(self):
        model = train_local_kernel_model(
            training_seeds=tuple(range(5)),
            horizon_steps=10,
            max_training_samples=200,
        )

        env = OptionMarketMakingEnv(horizon_steps=10, seed=42)
        state = env.reset()
        controller = make_local_kernel_controller(env, model, state)

        states = [state]
        while not state.done:
            action = controller(state)
            state, _, _, _ = env.step(action)
            states.append(state)

        assert states[-1].done
        assert np.isfinite(states[-1].wealth)

    def test_unfitted_model_raises(self):
        model = KernelRewardModel()
        env = OptionMarketMakingEnv(seed=1)
        state = env.reset()
        with pytest.raises(RuntimeError):
            make_local_kernel_controller(env, model, state)


# ---------------------------------------------------------------------------
# Hybrid BBG prior + residual controller tests
# ---------------------------------------------------------------------------


class TestBBGQuoteLookup:
    def test_action_matches_make_bbg_numerical(self):
        """BBGQuoteLookup.action must produce identical quotes to make_bbg_numerical."""
        env = OptionMarketMakingEnv(horizon_steps=5, initial_cash=100_000.0, seed=42)
        state = env.reset()
        gamma = 2.0 / state.wealth

        bbg_ctrl = make_bbg_numerical(env, state, gamma=gamma, max_inventory=10)
        lookup = BBGQuoteLookup(env, state, gamma=gamma, max_inventory=10)

        # Run a few steps and compare
        for _ in range(5):
            a_ctrl = bbg_ctrl(state)
            a_lookup = lookup.action(state)
            assert a_ctrl.bid_price == pytest.approx(a_lookup.bid_price, abs=1e-12)
            assert a_ctrl.ask_price == pytest.approx(a_lookup.ask_price, abs=1e-12)
            assert a_ctrl.hedge_trade == pytest.approx(a_lookup.hedge_trade, abs=1e-12)
            state, _, _, _ = env.step(a_ctrl)
            if state.done:
                break

    def test_distances_are_finite_and_nonnegative(self):
        env = OptionMarketMakingEnv(horizon_steps=5, initial_cash=100_000.0, seed=1)
        state = env.reset()
        lookup = BBGQuoteLookup(env, state, gamma=2e-5, max_inventory=10)
        bd, ad = lookup.distances(state)
        assert bd >= 0.0
        assert ad >= 0.0
        assert np.isfinite(bd)
        assert np.isfinite(ad)


class TestHybridTrainingBuffer:
    def test_add_and_size(self):
        buf = HybridTrainingBuffer()
        assert buf.size == 0
        buf.add(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.01, -0.005]), 3.14)
        assert buf.size == 1

    def test_as_arrays(self):
        buf = HybridTrainingBuffer()
        buf.add(np.zeros(4), np.array([0.01, 0.0]), 1.0)
        buf.add(np.ones(4), np.array([-0.01, 0.005]), 2.0)
        f, p, r = buf.as_arrays()
        assert f.shape == (2, 4)
        assert p.shape == (2, 2)
        assert r.shape == (2,)


class TestCollectHybridTrainingData:
    def test_collects_paired_data(self):
        buf = collect_hybrid_training_data(
            seeds=(0, 1), horizon_steps=5, noise_width=0.02, noise_skew=0.01,
        )
        assert buf.size == 10  # 2 seeds x 5 steps
        f, p, r = buf.as_arrays()
        assert f.shape == (10, 4)
        assert p.shape == (10, 2)  # (Δwidth, Δskew)
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(r))


class TestHybridResidualController:
    def _make_tiny_model(self):
        return train_hybrid_residual_model(
            training_seeds=(0, 1, 2, 3, 4),
            horizon_steps=5,
            max_training_samples=200,
        )

    def test_zero_perturbation_recovers_bbg(self):
        """With stencil = {(0,0)}, hybrid must return exactly BBG quotes."""
        model = self._make_tiny_model()
        env = OptionMarketMakingEnv(horizon_steps=5, initial_cash=100_000.0, seed=99)
        state = env.reset()

        # Hybrid with stencil that only includes (0, 0)
        hybrid = make_hybrid_residual_controller(
            env, model, state,
            stencil_width=(0.0,),
            stencil_skew=(0.0,),
        )
        # Reference BBG
        gamma_local = 2.0 / state.wealth
        bbg = make_bbg_numerical(env, state, gamma=gamma_local, max_inventory=10)

        for _ in range(5):
            a_hybrid = hybrid(state)
            a_bbg = bbg(state)
            assert a_hybrid.bid_price == pytest.approx(a_bbg.bid_price, abs=1e-12)
            assert a_hybrid.ask_price == pytest.approx(a_bbg.ask_price, abs=1e-12)
            assert a_hybrid.hedge_trade == pytest.approx(a_bbg.hedge_trade, abs=1e-12)
            state, _, _, _ = env.step(a_hybrid)
            if state.done:
                break

    def test_controller_produces_valid_actions(self):
        model = self._make_tiny_model()
        env = OptionMarketMakingEnv(horizon_steps=5, initial_cash=100_000.0, seed=77)
        state = env.reset()
        hybrid = make_hybrid_residual_controller(env, model, state)

        while not state.done:
            action = hybrid(state)
            assert action.bid_price >= 0.0
            assert action.ask_price >= action.bid_price
            assert np.isfinite(action.hedge_trade)
            state, _, _, _ = env.step(action)

    def test_positive_delta_width_widens_quotes(self):
        """A positive Δwidth should increase both bid and ask distance from mid."""
        env = OptionMarketMakingEnv(horizon_steps=5, initial_cash=100_000.0, seed=1)
        state = env.reset()
        gamma_local = 2.0 / state.wealth
        lookup = BBGQuoteLookup(env, state, gamma=gamma_local, max_inventory=10)
        a_base = lookup.action(state)

        from applications.option_mm.hybrid_residual_controller import _apply_perturbation
        bid_dist_0, ask_dist_0 = lookup.distances(state)
        a_wide = _apply_perturbation(state, bid_dist_0, ask_dist_0, 0.05, 0.0)

        # Wider bid distance => lower bid price
        assert a_wide.bid_price <= a_base.bid_price + 1e-12
        # Wider ask distance => higher ask price
        assert a_wide.ask_price >= a_base.ask_price - 1e-12

    def test_positive_delta_skew_widens_bid_tightens_ask(self):
        """Positive Δskew adds to bid distance, subtracts from ask distance."""
        env = OptionMarketMakingEnv(horizon_steps=5, initial_cash=100_000.0, seed=1)
        state = env.reset()
        gamma_local = 2.0 / state.wealth
        lookup = BBGQuoteLookup(env, state, gamma=gamma_local, max_inventory=10)
        bid_dist_0, ask_dist_0 = lookup.distances(state)

        from applications.option_mm.hybrid_residual_controller import _apply_perturbation
        a_skew = _apply_perturbation(state, bid_dist_0, ask_dist_0, 0.0, 0.03)
        a_base = lookup.action(state)

        # Bid wider (lower price)
        assert a_skew.bid_price <= a_base.bid_price + 1e-12
        # Ask tighter (lower price, closer to mid)
        assert a_skew.ask_price <= a_base.ask_price + 1e-12

    def test_unfitted_model_raises(self):
        model = KernelRewardModel()
        env = OptionMarketMakingEnv(initial_cash=100_000.0, seed=1)
        state = env.reset()
        with pytest.raises(RuntimeError):
            make_hybrid_residual_controller(env, model, state)
