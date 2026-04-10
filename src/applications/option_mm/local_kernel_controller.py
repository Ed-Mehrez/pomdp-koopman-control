"""Model-free local kernel controller for option market-making.

Track B of the salvage plan: learns a local reward landscape directly from
belief/path features, bypassing the sigma_sq_inv estimation channel.

Training:
    Run episodes with an exploration policy (risk-neutral + Gaussian noise),
    collect (state_features, action_params, one_step_reward) tuples, fit RBF
    kernel ridge regression on the joint feature-action space.

Deployment:
    Extract state features, grid-search over candidate (half_spread, skew)
    pairs, predict E[reward | state, action], choose the maximizer.

State features (4D):
    q_norm        = option_inventory / q_max
    delta_norm    = net_delta / delta_scale
    tau_frac      = (horizon_steps - step_index) / horizon_steps
    v_hat_norm    = v_hat / theta

Action params (2D):
    half_spread   = quote half-spread (bid = mid - half_spread - skew)
    skew          = inventory skew (>0 widens bid, tightens ask)

Hedge: always -net_delta (perfect rehedge, same as BBG baselines).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .beliefs import EWMAVarianceFilter
from .env import FillModelSpec, OptionMMAction, OptionMMState, OptionMarketMakingEnv


EpisodeController = Callable[[OptionMMState, Any | None], OptionMMAction]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_state_features(
    state: OptionMMState,
    env: OptionMarketMakingEnv,
    v_hat: float,
) -> np.ndarray:
    """Extract normalized state features for the kernel controller.

    Returns a 4D vector: [q_norm, delta_norm, tau_frac, v_hat_norm].
    """
    q_max = max(env.fills.max_contracts_per_step * env.horizon_steps, 1)
    delta_scale = env.contract.contract_multiplier * max(abs(state.option_delta), 0.01)
    return np.array(
        [
            state.option_inventory / q_max,
            state.net_delta / delta_scale,
            (env.horizon_steps - state.step_index) / env.horizon_steps,
            v_hat / max(env.heston.theta, 1e-8),
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Kernel ridge regression model
# ---------------------------------------------------------------------------


class KernelRewardModel:
    """RBF kernel ridge regression for E[reward | state_features, action_params].

    Training fits alpha = (K + lambda I)^{-1} y on standardized joint
    (features, actions) -> rewards.  Prediction evaluates the kernel on
    new points against the training set.
    """

    def __init__(self, bandwidth: float = 1.0, ridge_alpha: float = 1e-3) -> None:
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        if ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be nonnegative")
        self.bandwidth = bandwidth
        self.ridge_alpha = ridge_alpha
        self._X: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    @property
    def is_fitted(self) -> bool:
        return self._X is not None and self._alpha is not None

    def fit(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        """Fit KRR on the joint (features, actions) -> rewards mapping."""
        if features.ndim != 2 or actions.ndim != 2:
            raise ValueError("features and actions must be 2D")
        if features.shape[0] != actions.shape[0] or features.shape[0] != rewards.shape[0]:
            raise ValueError("features, actions, and rewards must have the same length")

        X = np.hstack([features, actions])
        y = rewards.astype(float)

        # Standardize inputs
        self._X_mean = X.mean(axis=0)
        self._X_std = np.maximum(X.std(axis=0), 1e-10)
        X_norm = (X - self._X_mean) / self._X_std

        # Standardize targets
        self._y_mean = float(y.mean())
        self._y_std = max(float(y.std()), 1e-10)
        y_norm = (y - self._y_mean) / self._y_std

        # Kernel matrix + ridge solve via Cholesky
        K = self._rbf_kernel(X_norm, X_norm)
        n = K.shape[0]
        L, low = cho_factor(K + self.ridge_alpha * np.eye(n))
        self._alpha = cho_solve((L, low), y_norm)
        self._X = X_norm

    def predict(self, features: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict E[reward] for each (features[i], actions[i]) pair."""
        if not self.is_fitted:
            raise RuntimeError("model must be fit before prediction")

        X_new = np.hstack([features, actions])
        X_norm = (X_new - self._X_mean) / self._X_std
        K_new = self._rbf_kernel(X_norm, self._X)
        y_norm = K_new @ self._alpha
        return y_norm * self._y_std + self._y_mean

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dists = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1, keepdims=True).T
            - 2.0 * X1 @ X2.T
        )
        return np.exp(-sq_dists / (2.0 * self.bandwidth**2))


def median_bandwidth(features: np.ndarray, actions: np.ndarray) -> float:
    """Median heuristic for RBF bandwidth on the joint feature-action space."""
    X = np.hstack([features, actions])
    X_std = np.maximum(X.std(axis=0), 1e-10)
    X_norm = X / X_std
    n = min(X_norm.shape[0], 2000)
    idx = np.random.default_rng(0).choice(X_norm.shape[0], size=n, replace=False)
    X_sub = X_norm[idx]
    sq_dists = (
        np.sum(X_sub**2, axis=1, keepdims=True)
        + np.sum(X_sub**2, axis=1, keepdims=True).T
        - 2.0 * X_sub @ X_sub.T
    )
    dists = np.sqrt(np.maximum(sq_dists[np.triu_indices(n, k=1)], 0.0))
    return max(float(np.median(dists)), 0.1)


# ---------------------------------------------------------------------------
# Training data collection
# ---------------------------------------------------------------------------


@dataclass
class TrainingBuffer:
    """Accumulates (state_features, action_params, reward) tuples."""

    features: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, feat: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.features.append(feat.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)

    @property
    def size(self) -> int:
        return len(self.rewards)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.features),
            np.array(self.actions),
            np.array(self.rewards),
        )


def collect_training_data(
    seeds: tuple[int, ...] | list[int],
    horizon_steps: int = 20,
    initial_cash: float = 100_000.0,
    ewma_half_life_days: float = 5.0,
    noise_spread: float = 0.03,
    noise_skew: float = 0.02,
    exploration_rng_seed: int = 42,
) -> TrainingBuffer:
    """Run episodes with exploration policy and collect training tuples.

    The exploration policy is risk-neutral (half_spread = 1/k) plus
    independent Gaussian noise on half_spread and skew.

    The reward target is the **controllable PnL** per step: spread capture
    minus adverse selection minus fees.  This strips out mark-to-market
    path noise (dominated by Heston dynamics, not the action) and isolates
    the action-dependent reward signal, yielding much higher SNR for the
    kernel regression.
    """
    exploration_rng = np.random.default_rng(exploration_rng_seed)
    buffer = TrainingBuffer()

    for seed in seeds:
        env = OptionMarketMakingEnv(
            fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
            horizon_steps=horizon_steps,
            initial_cash=initial_cash,
            seed=seed,
        )
        state = env.reset()
        ewma = EWMAVarianceFilter(half_life_days=ewma_half_life_days)
        ewma.reset(initial_variance=state.variance, initial_spot=state.spot)

        base_half_spread = 1.0 / env.fills.distance_slope

        while not state.done:
            v_hat = ewma.variance
            feat = extract_state_features(state, env, v_hat)

            # Exploration: risk-neutral baseline + Gaussian noise
            half_spread = max(0.01, base_half_spread + exploration_rng.normal(0, noise_spread))
            skew = exploration_rng.normal(0, noise_skew)
            action_params = np.array([half_spread, skew])

            bid_distance = half_spread + skew
            ask_distance = half_spread - skew
            action = OptionMMAction(
                bid_price=max(state.option_mid - bid_distance, 0.0),
                ask_price=state.option_mid + ask_distance,
                hedge_trade=-state.net_delta,
            )

            state, _, _, info = env.step(action)
            ewma.update(state.spot)

            # Reward = net spread capture minus costs.  Adverse selection
            # is omitted: it averages to zero (path and fill RNG are
            # independent) but its variance dominates the per-step
            # signal, drowning the action-dependent spread-capture peak.
            spread_reward = (
                info.spread_capture
                - info.option_fees
                - info.stock_costs
            )
            buffer.add(feat, action_params, spread_reward)

    return buffer


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


def make_local_kernel_controller(
    env: OptionMarketMakingEnv,
    model: KernelRewardModel,
    initial_state: OptionMMState,
    ewma_half_life_days: float = 5.0,
    spread_range: tuple[float, float] = (0.10, 0.30),
    skew_range: tuple[float, float] = (-0.05, 0.05),
    n_spread_candidates: int = 7,
    n_skew_candidates: int = 7,
) -> EpisodeController:
    """Factory for the trained local kernel controller.

    Creates an EWMA filter and a candidate-action grid.  At each step the
    controller predicts E[reward | features, a] for every grid point and
    chooses the argmax.

    Parameters
    ----------
    env : OptionMarketMakingEnv
        The environment (needed for feature normalization constants).
    model : KernelRewardModel
        A fitted kernel ridge regression model.
    initial_state : OptionMMState
        Reset state, used to initialize the EWMA filter.
    ewma_half_life_days : float
        Half-life for the internal EWMA variance filter.
    spread_range, skew_range : tuple[float, float]
        Bounds for the candidate action grid.
    n_spread_candidates, n_skew_candidates : int
        Grid resolution per action dimension.
    """
    if not model.is_fitted:
        raise RuntimeError("model must be fitted before building controller")

    ewma = EWMAVarianceFilter(half_life_days=ewma_half_life_days)
    ewma.reset(initial_variance=initial_state.variance, initial_spot=initial_state.spot)

    # Pre-compute candidate action grid (n_candidates, 2)
    spread_grid = np.linspace(spread_range[0], spread_range[1], n_spread_candidates)
    skew_grid = np.linspace(skew_range[0], skew_range[1], n_skew_candidates)
    action_grid = np.array(
        [[s, k] for s in spread_grid for k in skew_grid],
        dtype=float,
    )
    n_candidates = action_grid.shape[0]

    def controller(state: OptionMMState, history: Any | None = None) -> OptionMMAction:
        del history

        # Update EWMA with current observation (after first step)
        if state.step_index > 0:
            ewma.update(state.spot)
        v_hat = ewma.variance

        # Extract state features
        feat = extract_state_features(state, env, v_hat)

        # Predict reward for each candidate action
        feat_tiled = np.tile(feat, (n_candidates, 1))
        predicted_rewards = model.predict(feat_tiled, action_grid)

        # Choose best action
        best_idx = int(np.argmax(predicted_rewards))
        half_spread = float(action_grid[best_idx, 0])
        skew = float(action_grid[best_idx, 1])

        # Convert to OptionMMAction
        bid_distance = half_spread + skew
        ask_distance = half_spread - skew
        return OptionMMAction(
            bid_price=max(state.option_mid - bid_distance, 0.0),
            ask_price=state.option_mid + ask_distance,
            hedge_trade=-state.net_delta,
        )

    return controller


# ---------------------------------------------------------------------------
# End-to-end training convenience
# ---------------------------------------------------------------------------


def train_local_kernel_model(
    training_seeds: tuple[int, ...] | list[int],
    horizon_steps: int = 20,
    initial_cash: float = 100_000.0,
    ewma_half_life_days: float = 5.0,
    noise_spread: float = 0.03,
    noise_skew: float = 0.02,
    exploration_rng_seed: int = 42,
    ridge_alpha: float = 1e-3,
    max_training_samples: int = 10_000,
) -> KernelRewardModel:
    """Collect training data and fit a kernel reward model.

    If the buffer exceeds ``max_training_samples``, a random subsample is
    used for fitting (KRR scales as O(N^3), so cap at ~10K).
    """
    buffer = collect_training_data(
        seeds=training_seeds,
        horizon_steps=horizon_steps,
        initial_cash=initial_cash,
        ewma_half_life_days=ewma_half_life_days,
        noise_spread=noise_spread,
        noise_skew=noise_skew,
        exploration_rng_seed=exploration_rng_seed,
    )

    features, actions, rewards = buffer.as_arrays()

    # Subsample if necessary
    if features.shape[0] > max_training_samples:
        rng = np.random.default_rng(exploration_rng_seed + 1)
        idx = rng.choice(features.shape[0], size=max_training_samples, replace=False)
        features = features[idx]
        actions = actions[idx]
        rewards = rewards[idx]

    # Bandwidth via median heuristic
    bw = median_bandwidth(features, actions)

    model = KernelRewardModel(bandwidth=bw, ridge_alpha=ridge_alpha)
    model.fit(features, actions, rewards)
    return model
