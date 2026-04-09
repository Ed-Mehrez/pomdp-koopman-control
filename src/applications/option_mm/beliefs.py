"""Belief-state filters for the option market-making benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np

from .env import HestonParams


@dataclass
class EWMAVarianceFilter:
    """EWMA filter for annualized instantaneous variance."""

    half_life_days: float
    annualization: float = 252.0

    def __post_init__(self) -> None:
        if self.half_life_days <= 0.0 or not isfinite(self.half_life_days):
            raise ValueError("half_life_days must be positive and finite")
        if self.annualization <= 0.0 or not isfinite(self.annualization):
            raise ValueError("annualization must be positive and finite")
        self._lam = 0.5 ** (1.0 / self.half_life_days)
        self._var_estimate: float | None = None
        self._prev_log_spot: float | None = None

    @property
    def variance(self) -> float:
        if self._var_estimate is None:
            raise RuntimeError("filter must be reset before reading variance")
        return self._var_estimate

    def reset(self, initial_variance: float, initial_spot: float) -> None:
        """Initialize with a prior variance and the first observed spot."""
        if initial_variance <= 0.0 or not isfinite(initial_variance):
            raise ValueError("initial_variance must be positive and finite")
        if initial_spot <= 0.0 or not isfinite(initial_spot):
            raise ValueError("initial_spot must be positive and finite")
        self._var_estimate = float(initial_variance)
        self._prev_log_spot = log(initial_spot)

    def update(
        self,
        new_spot: float,
        *,
        true_variance: float | None = None,
        state: Any | None = None,
    ) -> float:
        """Update on the new spot and return annualized variance estimate."""
        del true_variance, state
        if self._var_estimate is None or self._prev_log_spot is None:
            raise RuntimeError("filter must be reset before update")
        if new_spot <= 0.0 or not isfinite(new_spot):
            raise ValueError("new_spot must be positive and finite")

        log_spot = log(new_spot)
        log_return = log_spot - self._prev_log_spot
        realized_variance = self.annualization * log_return * log_return
        self._var_estimate = (
            self._lam * self._var_estimate
            + (1.0 - self._lam) * realized_variance
        )
        self._prev_log_spot = log_spot
        return self._var_estimate


@dataclass
class OracleVarianceFilter:
    """Oracle filter that reads the true simulator variance."""

    variance_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.variance_floor < 0.0 or not isfinite(self.variance_floor):
            raise ValueError("variance_floor must be nonnegative and finite")
        self._var_estimate: float | None = None
        self._prev_log_spot: float | None = None

    @property
    def variance(self) -> float:
        if self._var_estimate is None:
            raise RuntimeError("filter must be reset before reading variance")
        return self._var_estimate

    def reset(self, initial_variance: float, initial_spot: float) -> None:
        if initial_variance < 0.0 or not isfinite(initial_variance):
            raise ValueError("initial_variance must be nonnegative and finite")
        if initial_spot <= 0.0 or not isfinite(initial_spot):
            raise ValueError("initial_spot must be positive and finite")
        self._var_estimate = max(float(initial_variance), self.variance_floor)
        self._prev_log_spot = log(initial_spot)

    def update(
        self,
        new_spot: float,
        *,
        true_variance: float | None = None,
        state: Any | None = None,
    ) -> float:
        """Return the true variance supplied by the runner."""
        if self._var_estimate is None or self._prev_log_spot is None:
            raise RuntimeError("filter must be reset before update")
        if new_spot <= 0.0 or not isfinite(new_spot):
            raise ValueError("new_spot must be positive and finite")

        if true_variance is None and state is not None:
            true_variance = getattr(state, "variance", None)
        if true_variance is None:
            raise ValueError("OracleVarianceFilter requires true_variance or state")
        if true_variance < 0.0 or not isfinite(true_variance):
            raise ValueError("true_variance must be nonnegative and finite")

        self._var_estimate = max(float(true_variance), self.variance_floor)
        self._prev_log_spot = log(new_spot)
        return self._var_estimate


@dataclass
class BootstrapParticleFilter:
    """Bootstrap particle filter for Heston variance with known model parameters."""

    heston: HestonParams | None = None
    dt: float = 1.0 / 252.0
    n_particles: int = 200
    seed: int | None = None
    variance_floor: float = 1e-8

    def __post_init__(self) -> None:
        self.heston = self.heston or HestonParams()
        self.heston.validate()
        if self.dt <= 0.0 or not isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.variance_floor <= 0.0 or not isfinite(self.variance_floor):
            raise ValueError("variance_floor must be positive and finite")

        self._rng = np.random.default_rng(self.seed)
        self._particles: np.ndarray | None = None
        self._var_estimate: float | None = None
        self._prev_log_spot: float | None = None

    @property
    def variance(self) -> float:
        if self._var_estimate is None:
            raise RuntimeError("filter must be reset before reading variance")
        return self._var_estimate

    def reset(self, initial_variance: float, initial_spot: float) -> None:
        if initial_variance <= 0.0 or not isfinite(initial_variance):
            raise ValueError("initial_variance must be positive and finite")
        if initial_spot <= 0.0 or not isfinite(initial_spot):
            raise ValueError("initial_spot must be positive and finite")
        variance = max(float(initial_variance), self.variance_floor)
        self._particles = np.full(self.n_particles, variance, dtype=float)
        self._var_estimate = variance
        self._prev_log_spot = log(initial_spot)

    def update(
        self,
        new_spot: float,
        *,
        true_variance: float | None = None,
        state: Any | None = None,
    ) -> float:
        """Update from the observed spot return and return filtered variance."""
        del true_variance, state
        if (
            self._particles is None
            or self._var_estimate is None
            or self._prev_log_spot is None
        ):
            raise RuntimeError("filter must be reset before update")
        if new_spot <= 0.0 or not isfinite(new_spot):
            raise ValueError("new_spot must be positive and finite")

        log_spot = log(new_spot)
        log_return = log_spot - self._prev_log_spot
        particles_pred = self._predict_particles()
        weights = self._return_likelihood_weights(log_return, particles_pred)

        self._var_estimate = float(np.sum(weights * particles_pred))
        resample_indices = self._rng.choice(
            self.n_particles,
            size=self.n_particles,
            replace=True,
            p=weights,
        )
        self._particles = particles_pred[resample_indices]
        self._prev_log_spot = log_spot
        return self._var_estimate

    def _predict_particles(self) -> np.ndarray:
        if self._particles is None:
            raise RuntimeError("filter must be reset before update")

        heston = self.heston
        assert heston is not None
        variance_pos = np.maximum(self._particles, self.variance_floor)
        shocks = self._rng.standard_normal(self.n_particles)
        raw_next = (
            self._particles
            + heston.kappa * (heston.theta - variance_pos) * self.dt
            + heston.xi * np.sqrt(variance_pos * self.dt) * shocks
        )
        return np.maximum(raw_next, self.variance_floor)

    def _return_likelihood_weights(
        self,
        log_return: float,
        particles_pred: np.ndarray,
    ) -> np.ndarray:
        heston = self.heston
        assert heston is not None
        variance_pos = np.maximum(particles_pred, self.variance_floor)
        obs_variance = variance_pos * self.dt
        obs_mean = (heston.mu - 0.5 * variance_pos) * self.dt
        log_weights = -0.5 * (
            np.log(2.0 * np.pi * obs_variance)
            + (log_return - obs_mean) ** 2 / obs_variance
        )
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0 or not isfinite(weight_sum):
            return np.full(self.n_particles, 1.0 / self.n_particles)
        return weights / weight_sum


@dataclass
class RecursiveSigRLSFilter:
    """Recurrent lead-lag log-signature filter with Bayesian RLS updates."""

    dt: float = 1.0 / 252.0
    signature_forgetting: float = 0.99
    blr_forgetting: float = 0.999
    prior_precision: float = 0.01
    observation_variance: float = 0.25
    variance_floor: float = 1e-8
    target_clip: float = 2.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or not isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if not 0.0 < self.signature_forgetting <= 1.0:
            raise ValueError("signature_forgetting must lie in (0, 1]")
        if not 0.0 < self.blr_forgetting <= 1.0:
            raise ValueError("blr_forgetting must lie in (0, 1]")
        if self.prior_precision <= 0.0 or not isfinite(self.prior_precision):
            raise ValueError("prior_precision must be positive and finite")
        if self.observation_variance <= 0.0 or not isfinite(self.observation_variance):
            raise ValueError("observation_variance must be positive and finite")
        if self.variance_floor <= 0.0 or not isfinite(self.variance_floor):
            raise ValueError("variance_floor must be positive and finite")
        if self.target_clip <= self.variance_floor or not isfinite(self.target_clip):
            raise ValueError("target_clip must exceed variance_floor")

        self._sig_map = _RecurrentLeadLagLogSigMap(
            input_dim=2,
            forgetting_factor=self.signature_forgetting,
        )
        self._n_features = self._sig_map.feature_dim + 1
        self._weights = np.zeros(self._n_features, dtype=float)
        self._posterior_cov = np.eye(self._n_features) / self.prior_precision
        self._var_estimate: float | None = None
        self._prev_log_spot: float | None = None

    @property
    def variance(self) -> float:
        if self._var_estimate is None:
            raise RuntimeError("filter must be reset before reading variance")
        return self._var_estimate

    def reset(self, initial_variance: float, initial_spot: float) -> None:
        if initial_variance <= 0.0 or not isfinite(initial_variance):
            raise ValueError("initial_variance must be positive and finite")
        if initial_spot <= 0.0 or not isfinite(initial_spot):
            raise ValueError("initial_spot must be positive and finite")

        self._sig_map.reset()
        self._weights = np.zeros(self._n_features, dtype=float)
        self._weights[-1] = max(float(initial_variance), self.variance_floor)
        self._posterior_cov = np.eye(self._n_features) / self.prior_precision
        self._var_estimate = max(float(initial_variance), self.variance_floor)
        self._prev_log_spot = log(initial_spot)

    def update(
        self,
        new_spot: float,
        *,
        true_variance: float | None = None,
        state: Any | None = None,
    ) -> float:
        """Update on one spot observation and return the posterior mean variance."""
        del true_variance, state
        if self._var_estimate is None or self._prev_log_spot is None:
            raise RuntimeError("filter must be reset before update")
        if new_spot <= 0.0 or not isfinite(new_spot):
            raise ValueError("new_spot must be positive and finite")

        log_spot = log(new_spot)
        log_return = log_spot - self._prev_log_spot
        features = self._features_for_increment(log_return)
        target = min(log_return * log_return / self.dt, self.target_clip)
        self._bayesian_rls_update(features, target)

        estimate = float(np.dot(self._weights, features))
        self._var_estimate = float(
            np.clip(estimate, self.variance_floor, self.target_clip)
        )
        self._prev_log_spot = log_spot
        return self._var_estimate

    def _features_for_increment(self, log_return: float) -> np.ndarray:
        sig_features = self._sig_map.update(np.array([self.dt, log_return]))
        return np.concatenate([sig_features, [1.0]])

    def _bayesian_rls_update(self, features: np.ndarray, target: float) -> None:
        cov_pred = self._posterior_cov / self.blr_forgetting
        cov_features = cov_pred @ features
        innovation_var = self.observation_variance + float(features @ cov_features)
        if innovation_var <= 0.0 or not isfinite(innovation_var):
            return

        gain = cov_features / innovation_var
        prediction = float(self._weights @ features)
        self._weights = self._weights + gain * (target - prediction)
        self._posterior_cov = cov_pred - np.outer(gain, cov_features)
        self._posterior_cov = 0.5 * (self._posterior_cov + self._posterior_cov.T)


class _RecurrentLeadLagLogSigMap:
    """Level-2 recurrent lead-lag log-signature map."""

    def __init__(self, input_dim: int, forgetting_factor: float) -> None:
        self.input_dim = input_dim
        self.lead_lag_dim = 2 * input_dim
        self.gamma = forgetting_factor
        self.dim_l1 = self.lead_lag_dim
        self.dim_l2 = self.lead_lag_dim * (self.lead_lag_dim - 1) // 2
        self.feature_dim = self.dim_l1 + self.dim_l2
        self.reset()

    def reset(self) -> None:
        self.l1 = np.zeros(self.dim_l1, dtype=float)
        self.l2 = np.zeros(self.dim_l2, dtype=float)

    def update(self, dx: np.ndarray) -> np.ndarray:
        if dx.shape != (self.input_dim,):
            raise ValueError("dx must have shape (input_dim,)")

        dx_lead = np.zeros(self.lead_lag_dim, dtype=float)
        dx_lead[: self.input_dim] = dx
        self._bch_update(dx_lead)

        dx_lag = np.zeros(self.lead_lag_dim, dtype=float)
        dx_lag[self.input_dim :] = dx
        self._bch_update(dx_lag)
        return self.features()

    def features(self) -> np.ndarray:
        return np.concatenate([self.l1, self.l2])

    def _bch_update(self, dx_ll: np.ndarray) -> None:
        l1_decayed = self.gamma * self.l1
        l2_decayed = self.gamma * self.gamma * self.l2

        bracket = np.zeros(self.dim_l2, dtype=float)
        idx = 0
        for i in range(self.lead_lag_dim):
            for j in range(i + 1, self.lead_lag_dim):
                bracket[idx] = l1_decayed[i] * dx_ll[j] - l1_decayed[j] * dx_ll[i]
                idx += 1

        self.l1 = l1_decayed + dx_ll
        self.l2 = l2_decayed + 0.5 * bracket
