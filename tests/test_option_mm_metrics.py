import os
import sys

import numpy as np
import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from applications.option_mm.metrics import (  # noqa: E402
    cara_utility,
    certainty_equivalent,
    crra_utility,
    paired_bayesian_bootstrap_posterior,
    paired_ce_posterior,
    paired_mean_difference_posterior,
    quadratic_utility,
)


def test_certainty_equivalent_uniform_weights_match_unweighted_formula():
    wealth = np.array([100_000.0, 100_100.0, 99_900.0, 100_050.0])
    weights = np.full(wealth.shape[0], 1.0 / wealth.shape[0])

    assert certainty_equivalent(wealth, gamma=1.0, weights=weights) == pytest.approx(
        np.exp(np.mean(np.log(wealth)))
    )
    assert certainty_equivalent(wealth, gamma=2.0, weights=weights) == pytest.approx(
        1.0 / np.mean(1.0 / wealth)
    )
    assert certainty_equivalent(wealth, gamma=3.0, weights=weights) == pytest.approx(
        np.mean(wealth ** -2.0) ** -0.5
    )
    assert certainty_equivalent(wealth, gamma=2.0, weights=weights) == (
        certainty_equivalent(wealth, gamma=2.0)
    )


def test_paired_ce_posterior_degenerate_equal_case_is_tie_neutral():
    wealth = np.full(8, 100_000.0)
    for method in ("delta", "mc", "bootstrap"):
        summary = paired_ce_posterior(
            wealth,
            wealth,
            gamma=2.0,
            method=method,
            n_draws=256,
            rng=np.random.default_rng(123),
        )

        assert summary.mean == pytest.approx(0.0, abs=1e-12)
        assert summary.sd_post == pytest.approx(0.0, abs=1e-12)
        assert summary.ci_low == pytest.approx(0.0, abs=1e-12)
        assert summary.ci_high == pytest.approx(0.0, abs=1e-12)
        assert summary.p_positive == pytest.approx(0.5)


def test_paired_ce_posterior_constant_edge_is_strictly_positive():
    wealth_b = np.full(8, 100_000.0)
    wealth_a = wealth_b + 1.0
    for method in ("delta", "mc", "bootstrap"):
        summary = paired_ce_posterior(
            wealth_a,
            wealth_b,
            gamma=2.0,
            method=method,
            n_draws=256,
            rng=np.random.default_rng(123),
        )

        assert summary.mean == pytest.approx(1.0, abs=1e-10)
        assert summary.sd_post == pytest.approx(0.0, abs=1e-10)
        assert summary.ci_low == pytest.approx(1.0, abs=1e-10)
        assert summary.ci_high == pytest.approx(1.0, abs=1e-10)
        assert summary.p_positive == pytest.approx(1.0)


def test_crra_utility_spec_matches_gamma_shortcut_for_all_methods():
    wealth_a = np.array([100_000.0, 100_200.0, 99_950.0, 100_075.0])
    wealth_b = np.array([100_010.0, 100_050.0, 99_900.0, 100_025.0])

    for method in ("delta", "mc", "bootstrap"):
        gamma_summary = paired_ce_posterior(
            wealth_a,
            wealth_b,
            gamma=2.0,
            method=method,
            n_draws=512,
            rng=np.random.default_rng(77),
        )
        utility_summary = paired_ce_posterior(
            wealth_a,
            wealth_b,
            utility=crra_utility(2.0),
            method=method,
            n_draws=512,
            rng=np.random.default_rng(77),
        )

        assert utility_summary == gamma_summary


def test_paired_ce_mc_and_bootstrap_reproducible_with_seeded_rng():
    wealth_a = np.array([100_000.0, 100_200.0, 99_950.0, 100_075.0])
    wealth_b = np.array([100_010.0, 100_050.0, 99_900.0, 100_025.0])

    for method in ("mc", "bootstrap"):
        summary_a = paired_ce_posterior(
            wealth_a,
            wealth_b,
            gamma=2.0,
            method=method,
            n_draws=512,
            rng=np.random.default_rng(999),
        )
        summary_b = paired_ce_posterior(
            wealth_a,
            wealth_b,
            gamma=2.0,
            method=method,
            n_draws=512,
            rng=np.random.default_rng(999),
        )

        assert summary_a == summary_b


def test_paired_ce_delta_mc_and_bootstrap_agree_on_smooth_case():
    rng = np.random.default_rng(20260407)
    shared = rng.normal(0.0, 80.0, size=500)
    idiosyncratic = rng.normal(0.0, 5.0, size=500)
    wealth_b = 100_000.0 + shared
    wealth_a = wealth_b + 4.0 + idiosyncratic

    delta = paired_ce_posterior(wealth_a, wealth_b, gamma=2.0, method="delta")
    mc = paired_ce_posterior(
        wealth_a,
        wealth_b,
        gamma=2.0,
        method="mc",
        n_draws=20_000,
        rng=np.random.default_rng(1),
    )
    bootstrap = paired_ce_posterior(
        wealth_a,
        wealth_b,
        gamma=2.0,
        method="bootstrap",
        n_draws=20_000,
        rng=np.random.default_rng(2),
    )

    assert mc.mean == pytest.approx(delta.mean, rel=0.05)
    assert bootstrap.mean == pytest.approx(delta.mean, rel=0.05)
    assert mc.sd_post == pytest.approx(delta.sd_post, rel=0.05)
    assert bootstrap.sd_post == pytest.approx(delta.sd_post, rel=0.05)


def test_cara_delta_and_mc_agree_on_smooth_case():
    rng = np.random.default_rng(20260407)
    shared = rng.normal(0.0, 80.0, size=500)
    idiosyncratic = rng.normal(0.0, 5.0, size=500)
    wealth_b = 100_000.0 + shared
    wealth_a = wealth_b + 4.0 + idiosyncratic
    utility = cara_utility(0.001)

    delta = paired_ce_posterior(wealth_a, wealth_b, utility=utility, method="delta")
    mc = paired_ce_posterior(
        wealth_a,
        wealth_b,
        utility=utility,
        method="mc",
        n_draws=20_000,
        rng=np.random.default_rng(1),
    )

    assert np.isfinite(delta.mean)
    assert delta.mean > 0.0
    assert delta.sd_post > 0.0
    assert 0.0 <= delta.p_positive <= 1.0
    assert mc.mean == pytest.approx(delta.mean, rel=0.05)
    assert mc.sd_post == pytest.approx(delta.sd_post, rel=0.05)


def test_cara_accepts_non_positive_wealth_and_crra_rejects_it():
    wealth_b = np.array([-2.0, 0.0, 1.0, 2.0])
    wealth_a = wealth_b + 1.0

    cara_summary = paired_ce_posterior(
        wealth_a,
        wealth_b,
        utility=cara_utility(0.1),
        method="delta",
    )
    assert np.isfinite(cara_summary.mean)

    with pytest.raises(ValueError, match="strictly positive wealth"):
        paired_ce_posterior(wealth_a, wealth_b, utility=crra_utility(2.0))
    with pytest.raises(ValueError, match="strictly positive wealth"):
        paired_ce_posterior(wealth_a, wealth_b, gamma=2.0)


def test_paired_ce_posterior_requires_exactly_one_utility_specification():
    wealth = np.full(4, 100_000.0)

    with pytest.raises(ValueError, match="exactly one"):
        paired_ce_posterior(wealth, wealth)
    with pytest.raises(ValueError, match="exactly one"):
        paired_ce_posterior(wealth, wealth, gamma=2.0, utility=crra_utility(2.0))


def test_cara_degenerate_tie_is_tie_neutral():
    wealth = np.full(8, 100_000.0)
    summary = paired_ce_posterior(
        wealth,
        wealth,
        utility=cara_utility(0.001),
        method="delta",
    )

    assert summary.mean == pytest.approx(0.0, abs=1e-12)
    assert summary.sd_post == pytest.approx(0.0, abs=1e-12)
    assert summary.p_positive == pytest.approx(0.5)


def test_paired_mean_difference_posterior_reports_student_t_posterior_sd():
    values = np.array([1.0, 2.0, 3.0])
    summary = paired_mean_difference_posterior(values)

    assert summary.mean == pytest.approx(2.0)
    assert summary.sd_post > 0.0
    assert summary.ci_low < summary.mean < summary.ci_high
    assert summary.p_positive > 0.95


def test_paired_bayesian_bootstrap_posterior_handles_weighted_functional():
    samples_a = np.array([2.0, 4.0, 6.0])
    samples_b = np.array([1.0, 1.0, 1.0])

    def weighted_mean_delta(a, b, weights):
        return float(np.sum(weights * (a - b)))

    summary = paired_bayesian_bootstrap_posterior(
        samples_a,
        samples_b,
        weighted_mean_delta,
        n_draws=256,
        rng=np.random.default_rng(123),
    )

    assert summary.mean > 0.0
    assert summary.sd_post > 0.0
    assert summary.p_positive == pytest.approx(1.0)


def test_arrow_pratt_factories_match_expected_values():
    wealth = 100_000.0

    assert crra_utility(2.0).arrow_pratt(wealth) == pytest.approx(2.0e-5)
    assert cara_utility(1.0e-3).arrow_pratt(wealth) == pytest.approx(1.0e-3)
    assert quadratic_utility(1.0e-6).arrow_pratt(wealth) == pytest.approx(
        1.0e-6 / (1.0 - 1.0e-6 * wealth)
    )
