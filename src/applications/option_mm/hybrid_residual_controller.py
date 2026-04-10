"""Hybrid BBG prior + state-action residual controller for option market-making.

Architecture:
    u(s_t) = u_0(s_t) + Δu(s_t)

where u_0 is the BBG numerical controller and Δu is a learned 2D residual
correction (Δwidth, Δskew) fitted by RBF kernel ridge regression on
state-action pairs collected from exploration around the BBG prior.

The corrected quote distances are:
    δ_bid = clip(δ_bid_0 + Δwidth + Δskew, lower=0)
    δ_ask = clip(δ_ask_0 + Δwidth - Δskew, lower=0)

Hedge is fixed to -net_delta (same as BBG).

Training target: per-step spread reward (spread_capture - fees) under
the perturbed action, conditioned on BOTH state features AND the
perturbation Δu. This keeps the project connected to control (not
generic reward regression).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .bbg_solver import solve_bbg_quote_tables
from .beliefs import EWMAVarianceFilter
from .controllers import NO_QUOTE_ASK
from .env import FillModelSpec, HestonParams, OptionMMAction, OptionMMState, OptionMarketMakingEnv
from .local_kernel_controller import KernelRewardModel, extract_state_features, median_bandwidth
from .metrics import UtilitySpec


EpisodeController = Callable[[OptionMMState, Any | None], OptionMMAction]


# ---------------------------------------------------------------------------
# BBG quote-distance lookup (reusable helper)
# ---------------------------------------------------------------------------


class BBGQuoteLookup:
    """Pre-solved BBG quote-distance tables for fast per-step lookups.

    Provides the raw bid/ask distances from mid that the BBG controller
    would produce, WITHOUT converting to OptionMMAction.  This lets the
    hybrid controller apply perturbations before clipping.
    """

    def __init__(
        self,
        env: OptionMarketMakingEnv,
        initial_state: OptionMMState,
        *,
        gamma: float,
        max_inventory: int,
    ) -> None:
        q_grid, bid_distances, ask_distances = solve_bbg_quote_tables(
            env, initial_state, gamma=gamma, max_inventory=max_inventory,
        )
        self._bid_distances = bid_distances
        self._ask_distances = ask_distances
        self._min_q = int(q_grid[0])
        self._max_q = int(q_grid[-1])
        self._horizon_steps = env.horizon_steps

    def distances(self, state: OptionMMState) -> tuple[float, float]:
        """Return (bid_distance, ask_distance) for the current state."""
        step_index = min(max(state.step_index, 0), self._horizon_steps - 1)
        q = int(np.clip(state.option_inventory, self._min_q, self._max_q))
        q_index = q - self._min_q
        return (
            float(self._bid_distances[step_index, q_index]),
            float(self._ask_distances[step_index, q_index]),
        )

    def action(self, state: OptionMMState) -> OptionMMAction:
        """Return the unperturbed BBG action (for reference / zero-residual)."""
        bid_dist, ask_dist = self.distances(state)
        bid_price = (
            0.0 if not np.isfinite(bid_dist)
            else max(state.option_mid - bid_dist, 0.0)
        )
        ask_price = (
            NO_QUOTE_ASK if not np.isfinite(ask_dist)
            else state.option_mid + ask_dist
        )
        return OptionMMAction(
            bid_price=bid_price,
            ask_price=ask_price,
            hedge_trade=-state.net_delta,
        )


# ---------------------------------------------------------------------------
# Training data collection
# ---------------------------------------------------------------------------


@dataclass
class HybridTrainingBuffer:
    """Accumulates (state_features, Δu, spread_reward) tuples."""

    features: list[np.ndarray] = field(default_factory=list)
    perturbations: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, feat: np.ndarray, delta_u: np.ndarray, reward: float) -> None:
        self.features.append(feat.copy())
        self.perturbations.append(delta_u.copy())
        self.rewards.append(reward)

    @property
    def size(self) -> int:
        return len(self.rewards)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.features),
            np.array(self.perturbations),
            np.array(self.rewards),
        )


def collect_hybrid_training_data(
    seeds: tuple[int, ...] | list[int],
    heston: HestonParams | None = None,
    horizon_steps: int = 20,
    initial_cash: float = 100_000.0,
    gamma_ce: float = 2.0,
    max_inventory: int = 10,
    ewma_half_life_days: float = 5.0,
    noise_width: float = 0.02,
    noise_skew: float = 0.01,
    exploration_rng_seed: int = 42,
) -> HybridTrainingBuffer:
    """Run episodes with BBG prior + Gaussian perturbations on (Δwidth, Δskew).

    For each step:
      1. Query BBG for (δ_bid_0, δ_ask_0).
      2. Sample perturbation (Δwidth, Δskew) ~ N(0, σ).
      3. Execute perturbed action.
      4. Record (state_features, Δu, spread_reward).
    """
    exploration_rng = np.random.default_rng(exploration_rng_seed)
    buffer = HybridTrainingBuffer()
    utility = _crra_utility_arrow_pratt(gamma_ce)

    for seed in seeds:
        env = OptionMarketMakingEnv(
            heston=heston,
            fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
            horizon_steps=horizon_steps,
            initial_cash=initial_cash,
            seed=seed,
        )
        state = env.reset()
        gamma_local = utility(state.wealth)
        bbg = BBGQuoteLookup(
            env, state, gamma=gamma_local, max_inventory=max_inventory,
        )
        ewma = EWMAVarianceFilter(half_life_days=ewma_half_life_days)
        ewma.reset(initial_variance=state.variance, initial_spot=state.spot)

        while not state.done:
            v_hat = ewma.variance
            feat = extract_state_features(state, env, v_hat)

            # BBG prior distances
            bid_dist_0, ask_dist_0 = bbg.distances(state)

            # Sample perturbation
            delta_width = exploration_rng.normal(0, noise_width)
            delta_skew = exploration_rng.normal(0, noise_skew)
            delta_u = np.array([delta_width, delta_skew])

            # Apply perturbation
            action = _apply_perturbation(
                state, bid_dist_0, ask_dist_0, delta_width, delta_skew,
            )

            state, _, _, info = env.step(action)
            ewma.update(state.spot)

            spread_reward = (
                info.spread_capture - info.option_fees - info.stock_costs
            )
            buffer.add(feat, delta_u, spread_reward)

    return buffer


def collect_hybrid_training_data_multi_cell(
    cell_params: list[HestonParams],
    seeds_per_cell: tuple[int, ...] | list[int],
    **kwargs,
) -> HybridTrainingBuffer:
    """Collect training data across multiple Heston parameter cells."""
    combined = HybridTrainingBuffer()
    for heston in cell_params:
        buf = collect_hybrid_training_data(
            seeds=seeds_per_cell,
            heston=heston,
            **kwargs,
        )
        combined.features.extend(buf.features)
        combined.perturbations.extend(buf.perturbations)
        combined.rewards.extend(buf.rewards)
    return combined


# ---------------------------------------------------------------------------
# Hybrid controller factory
# ---------------------------------------------------------------------------


def make_hybrid_residual_controller(
    env: OptionMarketMakingEnv,
    model: KernelRewardModel,
    initial_state: OptionMMState,
    *,
    gamma_ce: float = 2.0,
    max_inventory: int = 10,
    ewma_half_life_days: float = 5.0,
    stencil_width: tuple[float, ...] = (-0.03, -0.015, 0.0, 0.015, 0.03),
    stencil_skew: tuple[float, ...] = (-0.015, -0.005, 0.0, 0.005, 0.015),
) -> EpisodeController:
    """Factory for the hybrid BBG prior + residual controller.

    At each step:
      1. Query BBG for (δ_bid_0, δ_ask_0).
      2. Evaluate E[reward | features, Δu] for each candidate in the stencil.
      3. Choose the Δu that maximizes predicted reward.
      4. Apply perturbation to BBG quotes.

    With model predicting constant (or zero residual trained), this
    recovers BBG exactly when stencil includes Δu = (0, 0).
    """
    if not model.is_fitted:
        raise RuntimeError("model must be fitted before building controller")

    gamma_local = _crra_utility_arrow_pratt(gamma_ce)(initial_state.wealth)
    bbg = BBGQuoteLookup(
        env, initial_state, gamma=gamma_local, max_inventory=max_inventory,
    )
    ewma = EWMAVarianceFilter(half_life_days=ewma_half_life_days)
    ewma.reset(initial_variance=initial_state.variance, initial_spot=initial_state.spot)

    # Pre-compute candidate perturbation stencil
    stencil = np.array(
        [[w, s] for w in stencil_width for s in stencil_skew],
        dtype=float,
    )
    n_candidates = stencil.shape[0]

    def controller(state: OptionMMState, history: Any | None = None) -> OptionMMAction:
        del history

        # Update EWMA
        if state.step_index > 0:
            ewma.update(state.spot)
        v_hat = ewma.variance

        feat = extract_state_features(state, env, v_hat)
        bid_dist_0, ask_dist_0 = bbg.distances(state)

        # Predict reward for each candidate perturbation
        feat_tiled = np.tile(feat, (n_candidates, 1))
        predicted_rewards = model.predict(feat_tiled, stencil)

        # Choose best perturbation
        best_idx = int(np.argmax(predicted_rewards))
        delta_width = float(stencil[best_idx, 0])
        delta_skew = float(stencil[best_idx, 1])

        return _apply_perturbation(
            state, bid_dist_0, ask_dist_0, delta_width, delta_skew,
        )

    return controller


# ---------------------------------------------------------------------------
# End-to-end training convenience
# ---------------------------------------------------------------------------


def train_hybrid_residual_model(
    training_seeds: tuple[int, ...] | list[int],
    heston: HestonParams | None = None,
    horizon_steps: int = 20,
    initial_cash: float = 100_000.0,
    gamma_ce: float = 2.0,
    max_inventory: int = 10,
    ewma_half_life_days: float = 5.0,
    noise_width: float = 0.02,
    noise_skew: float = 0.01,
    exploration_rng_seed: int = 42,
    ridge_alpha: float = 1e-3,
    max_training_samples: int = 10_000,
) -> KernelRewardModel:
    """Collect training data and fit a kernel reward model for the residual."""
    buffer = collect_hybrid_training_data(
        seeds=training_seeds,
        heston=heston,
        horizon_steps=horizon_steps,
        initial_cash=initial_cash,
        gamma_ce=gamma_ce,
        max_inventory=max_inventory,
        ewma_half_life_days=ewma_half_life_days,
        noise_width=noise_width,
        noise_skew=noise_skew,
        exploration_rng_seed=exploration_rng_seed,
    )
    return _fit_model_from_buffer(buffer, ridge_alpha, max_training_samples,
                                  exploration_rng_seed)


def train_hybrid_residual_model_multi_cell(
    cell_params: list[HestonParams],
    seeds_per_cell: tuple[int, ...] | list[int],
    horizon_steps: int = 20,
    initial_cash: float = 100_000.0,
    gamma_ce: float = 2.0,
    max_inventory: int = 10,
    ewma_half_life_days: float = 5.0,
    noise_width: float = 0.02,
    noise_skew: float = 0.01,
    exploration_rng_seed: int = 42,
    ridge_alpha: float = 1e-3,
    max_training_samples: int = 10_000,
) -> KernelRewardModel:
    """Train residual model on data pooled across Heston parameter cells."""
    buffer = collect_hybrid_training_data_multi_cell(
        cell_params=cell_params,
        seeds_per_cell=seeds_per_cell,
        horizon_steps=horizon_steps,
        initial_cash=initial_cash,
        gamma_ce=gamma_ce,
        max_inventory=max_inventory,
        ewma_half_life_days=ewma_half_life_days,
        noise_width=noise_width,
        noise_skew=noise_skew,
        exploration_rng_seed=exploration_rng_seed,
    )
    return _fit_model_from_buffer(buffer, ridge_alpha, max_training_samples,
                                  exploration_rng_seed)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_perturbation(
    state: OptionMMState,
    bid_dist_0: float,
    ask_dist_0: float,
    delta_width: float,
    delta_skew: float,
) -> OptionMMAction:
    """Apply (Δwidth, Δskew) to BBG quote distances and return an action."""
    # Δwidth widens both symmetrically; Δskew is antisymmetric.
    bid_dist = max(0.0, bid_dist_0 + delta_width + delta_skew)
    ask_dist = max(0.0, ask_dist_0 + delta_width - delta_skew)

    bid_price = (
        0.0 if not np.isfinite(bid_dist)
        else max(state.option_mid - bid_dist, 0.0)
    )
    ask_price = (
        NO_QUOTE_ASK if not np.isfinite(ask_dist)
        else state.option_mid + ask_dist
    )
    return OptionMMAction(
        bid_price=bid_price,
        ask_price=ask_price,
        hedge_trade=-state.net_delta,
    )


def _crra_utility_arrow_pratt(gamma: float) -> Callable[[float], float]:
    """Return the Arrow-Pratt function for CRRA(gamma)."""
    return lambda wealth: gamma / wealth


def _fit_model_from_buffer(
    buffer: HybridTrainingBuffer,
    ridge_alpha: float,
    max_training_samples: int,
    rng_seed: int,
) -> KernelRewardModel:
    """Fit a KernelRewardModel from a training buffer."""
    features, perturbations, rewards = buffer.as_arrays()

    if features.shape[0] > max_training_samples:
        rng = np.random.default_rng(rng_seed + 1)
        idx = rng.choice(features.shape[0], size=max_training_samples, replace=False)
        features = features[idx]
        perturbations = perturbations[idx]
        rewards = rewards[idx]

    bw = median_bandwidth(features, perturbations)
    model = KernelRewardModel(bandwidth=bw, ridge_alpha=ridge_alpha)
    model.fit(features, perturbations, rewards)
    return model
