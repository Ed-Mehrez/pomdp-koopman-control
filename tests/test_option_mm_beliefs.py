import os
import sys
from math import log

import numpy as np
import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from applications.option_mm.beliefs import (  # noqa: E402
    BootstrapParticleFilter,
    EWMAVarianceFilter,
    OracleVarianceFilter,
    RecursiveSigRLSFilter,
)
from applications.option_mm.env import HestonParams  # noqa: E402


def test_ewma_reset_sets_initial_variance():
    filt = EWMAVarianceFilter(half_life_days=5.0)
    filt.reset(initial_variance=0.04, initial_spot=100.0)

    assert filt.variance == pytest.approx(0.04)


def test_ewma_update_uses_previous_observed_spot_only():
    filt = EWMAVarianceFilter(half_life_days=1.0, annualization=252.0)
    filt.reset(initial_variance=0.04, initial_spot=100.0)

    v_hat = filt.update(101.0)
    lam = 0.5
    expected_rv = 252.0 * log(101.0 / 100.0) ** 2
    expected = lam * 0.04 + (1.0 - lam) * expected_rv

    assert v_hat == pytest.approx(expected)
    assert filt.variance == pytest.approx(expected)


def test_ewma_requires_reset_before_update():
    filt = EWMAVarianceFilter(half_life_days=5.0)

    with pytest.raises(RuntimeError, match="reset"):
        _ = filt.variance
    with pytest.raises(RuntimeError, match="reset"):
        filt.update(101.0)


def test_ewma_validates_inputs():
    with pytest.raises(ValueError):
        EWMAVarianceFilter(half_life_days=0.0)
    with pytest.raises(ValueError):
        EWMAVarianceFilter(half_life_days=5.0, annualization=0.0)

    filt = EWMAVarianceFilter(half_life_days=5.0)
    with pytest.raises(ValueError):
        filt.reset(initial_variance=0.0, initial_spot=100.0)
    with pytest.raises(ValueError):
        filt.reset(initial_variance=0.04, initial_spot=0.0)

    filt.reset(initial_variance=0.04, initial_spot=100.0)
    with pytest.raises(ValueError):
        filt.update(0.0)


def test_oracle_filter_reads_true_variance_from_state_or_kwarg():
    filt = OracleVarianceFilter()
    filt.reset(initial_variance=0.04, initial_spot=100.0)

    assert filt.update(101.0, true_variance=0.05) == pytest.approx(0.05)

    class State:
        variance = 0.03

    assert filt.update(102.0, state=State()) == pytest.approx(0.03)


def test_oracle_filter_requires_true_variance():
    filt = OracleVarianceFilter()
    filt.reset(initial_variance=0.04, initial_spot=100.0)

    with pytest.raises(ValueError, match="true_variance"):
        filt.update(101.0)


def test_bpf_tracks_known_constant_variance_path():
    filt = BootstrapParticleFilter(
        heston=HestonParams(mu=0.0, kappa=0.0, theta=0.04, xi=0.0, rho=0.0),
        n_particles=200,
        seed=123,
    )
    estimates = _run_constant_realized_variance_path(filt, variance=0.04, n_steps=100)

    assert np.all(np.isfinite(estimates))
    assert np.min(estimates) > 0.0
    assert estimates[-1] == pytest.approx(0.04)


def test_recsig_filter_learns_known_realized_variance_path():
    filt = RecursiveSigRLSFilter(
        signature_forgetting=0.99,
        blr_forgetting=0.999,
        prior_precision=0.01,
        observation_variance=0.25,
    )
    estimates = _run_constant_realized_variance_path(filt, variance=0.04, n_steps=250)

    assert np.all(np.isfinite(estimates))
    assert np.min(estimates) > 0.0
    assert estimates[-1] == pytest.approx(0.04, abs=0.02)


def test_filters_remain_stable_at_variance_floor():
    filters = [
        OracleVarianceFilter(),
        BootstrapParticleFilter(n_particles=50, seed=321),
        RecursiveSigRLSFilter(),
    ]
    for filt in filters:
        filt.reset(initial_variance=1e-8, initial_spot=100.0)
        estimate = filt.update(100.0, true_variance=1e-8)
        assert np.isfinite(estimate)
        assert estimate > 0.0


def test_bpf_reproducible_with_same_seed():
    filt_a = BootstrapParticleFilter(n_particles=50, seed=55)
    filt_b = BootstrapParticleFilter(n_particles=50, seed=55)

    estimates_a = _run_constant_realized_variance_path(filt_a, variance=0.04, n_steps=20)
    estimates_b = _run_constant_realized_variance_path(filt_b, variance=0.04, n_steps=20)

    np.testing.assert_allclose(estimates_a, estimates_b)


def test_filter_input_validation():
    with pytest.raises(ValueError):
        BootstrapParticleFilter(n_particles=0)
    with pytest.raises(ValueError):
        RecursiveSigRLSFilter(signature_forgetting=0.0)
    with pytest.raises(ValueError):
        RecursiveSigRLSFilter(blr_forgetting=0.0)


def _run_constant_realized_variance_path(filt, variance: float, n_steps: int) -> np.ndarray:
    spot = 100.0
    filt.reset(initial_variance=variance, initial_spot=spot)
    estimates = []
    log_return = np.sqrt(variance / 252.0)
    for _ in range(n_steps):
        spot *= float(np.exp(log_return))
        estimates.append(filt.update(spot, true_variance=variance))
    return np.asarray(estimates)
