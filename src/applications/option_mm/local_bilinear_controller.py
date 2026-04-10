"""Prior-free local bilinear / CQ-KRONIC controller for option market-making.

Learns a local control-quadratic lifted dynamics model from state-action
transition data, then extracts the optimal action via local SDRE.

NO BBG prior inside the candidate.  BBG is a benchmark only.

Architecture:
    phi(x_{t+1}) ~ A_0 phi(x_t)
                  + w * A_w phi(x_t)
                  + s * A_s phi(x_t)
                  + w^2 * A_ww phi(x_t)
                  + ws * A_ws phi(x_t)
                  + s^2 * A_ss phi(x_t)

    where (w, s) = (half_width, skew) are the action parameters.

Policy extraction:
    J_x(w,s) = mu_dW(x,w,s) - (gamma_loc/2) sigma^2_dW(x,w,s) - lambda_q E[q^2]
    u*(x) = argmax J_x(w,s) via local quadratic solve.

State: x = (q_norm, delta_norm, tau_frac, v_norm, 1)
Lift:  phi(x) = [1, q, h, tau, v, qh, qv, hv, q^2, h^2, v^2]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .beliefs import EWMAVarianceFilter
from .env import FillModelSpec, HestonParams, OptionMMAction, OptionMMState, OptionMarketMakingEnv


EpisodeController = Callable[[OptionMMState, Any | None], OptionMMAction]


# ---------------------------------------------------------------------------
# State normalization and lift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateNorm:
    """Normalization constants derived from env config."""
    q_max: float
    delta_max: float
    tau_max: float
    v_scale: float  # theta

    @classmethod
    def from_env(cls, env: OptionMarketMakingEnv) -> "StateNorm":
        return cls(
            q_max=max(float(env.fills.max_contracts_per_step * env.horizon_steps), 1.0),
            delta_max=env.contract.contract_multiplier * 0.5,  # rough scale
            tau_max=float(env.horizon_steps),
            v_scale=max(env.heston.theta, 1e-8),
        )


def normalize_state(
    state: OptionMMState,
    norm: StateNorm,
) -> np.ndarray:
    """Return 5D normalized state: (q, h, tau, v, 1)."""
    return np.array([
        state.option_inventory / norm.q_max,
        state.net_delta / norm.delta_max,
        (norm.tau_max - state.step_index) / norm.tau_max,
        state.variance / norm.v_scale,
        1.0,
    ], dtype=float)


def lift_state(x: np.ndarray) -> np.ndarray:
    """Quadratic lift: [1, q, h, tau, v, qh, qv, hv, q^2, h^2, v^2]."""
    q, h, tau, v = x[0], x[1], x[2], x[3]
    return np.array([
        1.0, q, h, tau, v,
        q * h, q * v, h * v,
        q * q, h * h, v * v,
    ], dtype=float)


LIFT_DIM = 11  # dimension of lift_state output


# ---------------------------------------------------------------------------
# Training data collection
# ---------------------------------------------------------------------------


@dataclass
class BilinearTrainingBuffer:
    """Accumulates (x_t, u_t, x_{t+1}, dW_t, q_{t+1}) tuples."""
    x_list: list[np.ndarray] = field(default_factory=list)
    u_list: list[np.ndarray] = field(default_factory=list)
    x_next_list: list[np.ndarray] = field(default_factory=list)
    dw_list: list[float] = field(default_factory=list)
    q_next_list: list[float] = field(default_factory=list)
    spread_capture_list: list[float] = field(default_factory=list)

    def add(
        self,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        dw: float,
        q_next: float,
        spread_capture: float,
    ) -> None:
        self.x_list.append(x.copy())
        self.u_list.append(u.copy())
        self.x_next_list.append(x_next.copy())
        self.dw_list.append(dw)
        self.q_next_list.append(q_next)
        self.spread_capture_list.append(spread_capture)

    @property
    def size(self) -> int:
        return len(self.dw_list)


def collect_bilinear_training_data(
    seeds: list[int] | tuple[int, ...],
    heston: HestonParams | None = None,
    horizon_steps: int = 20,
    initial_cash: float = 100_000.0,
    width_range: tuple[float, float] = (0.10, 0.30),
    skew_range: tuple[float, float] = (-0.05, 0.05),
    exploration_rng_seed: int = 42,
) -> BilinearTrainingBuffer:
    """Run episodes with uniform exploration over (width, skew) box."""
    rng = np.random.default_rng(exploration_rng_seed)
    buffer = BilinearTrainingBuffer()

    for seed in seeds:
        env = OptionMarketMakingEnv(
            heston=heston,
            fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
            horizon_steps=horizon_steps,
            initial_cash=initial_cash,
            seed=seed,
        )
        norm = StateNorm.from_env(env)
        state = env.reset()

        while not state.done:
            x = normalize_state(state, norm)

            # Generic exploration: uniform over action box
            w = rng.uniform(width_range[0], width_range[1])
            s = rng.uniform(skew_range[0], skew_range[1])
            u = np.array([w, s])

            bid_dist = w + s
            ask_dist = w - s
            action = OptionMMAction(
                bid_price=max(state.option_mid - bid_dist, 0.0),
                ask_price=state.option_mid + max(ask_dist, 0.0),
                hedge_trade=-state.net_delta,
            )

            next_state, reward, _, info = env.step(action)
            x_next = normalize_state(next_state, norm)
            dw = next_state.wealth - state.wealth
            q_next = float(next_state.option_inventory) / norm.q_max

            buffer.add(x, u, x_next, dw, q_next, info.spread_capture)
            state = next_state

    return buffer


# ---------------------------------------------------------------------------
# Local CQ-KRONIC fitter
# ---------------------------------------------------------------------------


class LocalBilinearModel:
    """Kernel-weighted local control-quadratic dynamics model.

    Fits phi(x_{t+1}) as a function of (phi(x_t), w, s) with quadratic
    action dependence, using RBF kernel weights centered at query point.

    Also fits scalar models for dW (wealth increment) with the same
    control-quadratic structure.
    """

    def __init__(
        self,
        X: np.ndarray,       # (N, 5) normalized states
        U: np.ndarray,       # (N, 2) actions
        X_next: np.ndarray,  # (N, 5) next states
        dW: np.ndarray,      # (N,) wealth increments
        Q_next: np.ndarray,  # (N,) next inventory (normalized)
        bandwidth: float = 1.0,
        ridge: float = 1e-3,
    ) -> None:
        self.X = X
        self.U = U
        self.X_next = X_next
        self.dW = dW
        self.Q_next = Q_next
        self.bandwidth = bandwidth
        self.ridge = ridge

        # Pre-compute lifted features and CQ regressors
        self._Phi = np.array([lift_state(x) for x in X])       # (N, 11)
        self._Phi_next = np.array([lift_state(x) for x in X_next])
        self._build_cq_regressors()

    def _build_cq_regressors(self) -> None:
        """Build the (N, 6*LIFT_DIM) regressor matrix for CQ structure."""
        N = self.X.shape[0]
        w = self.U[:, 0]
        s = self.U[:, 1]
        P = self._Phi  # (N, 11)

        # CQ regressors: [phi, w*phi, s*phi, w^2*phi, ws*phi, s^2*phi]
        self._Z = np.hstack([
            P,                                    # A_0 phi
            P * w[:, None],                       # A_w phi
            P * s[:, None],                       # A_s phi
            P * (w * w)[:, None],                 # A_ww phi
            P * (w * s)[:, None],                 # A_ws phi
            P * (s * s)[:, None],                 # A_ss phi
        ])  # (N, 6 * LIFT_DIM)

    def _kernel_weights(self, x_query: np.ndarray) -> np.ndarray:
        """RBF kernel weights centered at x_query."""
        diffs = self.X - x_query[None, :]
        sq_dists = np.sum(diffs * diffs, axis=1)
        return np.exp(-sq_dists / (2.0 * self.bandwidth ** 2))

    def fit_local_dw(self, x_query: np.ndarray) -> np.ndarray:
        """Fit local CQ model for wealth increment at x_query.

        Returns coefficient vector theta of length 6*LIFT_DIM such that:
            dW ~ Z @ theta  (kernel-weighted local regression)
        """
        K = self._kernel_weights(x_query)
        return self._solve_weighted_ridge(self._Z, self.dW, K)

    def fit_local_q_next(self, x_query: np.ndarray) -> np.ndarray:
        """Fit local CQ model for next inventory at x_query."""
        K = self._kernel_weights(x_query)
        return self._solve_weighted_ridge(self._Z, self.Q_next, K)

    def predict_dw_quadratic(
        self,
        x_query: np.ndarray,
        theta_dw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract quadratic coefficients of dW in (w, s) at x_query.

        Returns (H, c, d) such that dW(w,s) ~ d + c^T [w,s] + 0.5 [w,s]^T H [w,s].
        """
        phi = lift_state(x_query)
        d_lift = LIFT_DIM

        # theta is [A_0, A_w, A_s, A_ww, A_ws, A_ss] each length d_lift
        a0 = theta_dw[0:d_lift] @ phi
        aw = theta_dw[d_lift:2*d_lift] @ phi
        a_s = theta_dw[2*d_lift:3*d_lift] @ phi
        aww = theta_dw[3*d_lift:4*d_lift] @ phi
        aws = theta_dw[4*d_lift:5*d_lift] @ phi
        ass_ = theta_dw[5*d_lift:6*d_lift] @ phi

        # dW(w,s) = a0 + aw*w + as*s + aww*w^2 + aws*ws + ass*s^2
        d = float(a0)
        c = np.array([float(aw), float(a_s)])
        H = np.array([
            [2.0 * float(aww), float(aws)],
            [float(aws), 2.0 * float(ass_)],
        ])
        return H, c, d

    def predict_q_quadratic(
        self,
        x_query: np.ndarray,
        theta_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract quadratic coefficients of q_next in (w, s)."""
        phi = lift_state(x_query)
        d_lift = LIFT_DIM

        a0 = theta_q[0:d_lift] @ phi
        aw = theta_q[d_lift:2*d_lift] @ phi
        a_s = theta_q[2*d_lift:3*d_lift] @ phi
        aww = theta_q[3*d_lift:4*d_lift] @ phi
        aws = theta_q[4*d_lift:5*d_lift] @ phi
        ass_ = theta_q[5*d_lift:6*d_lift] @ phi

        d = float(a0)
        c = np.array([float(aw), float(a_s)])
        H = np.array([
            [2.0 * float(aww), float(aws)],
            [float(aws), 2.0 * float(ass_)],
        ])
        return H, c, d

    def _solve_weighted_ridge(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """Weighted ridge regression: min_theta sum K_i (Z_i theta - y_i)^2 + ridge ||theta||^2."""
        sqrt_K = np.sqrt(np.maximum(K, 0.0))
        Zw = Z * sqrt_K[:, None]
        yw = y * sqrt_K
        p = Z.shape[1]
        gram = Zw.T @ Zw + self.ridge * np.eye(p)
        try:
            theta = np.linalg.solve(gram, Zw.T @ yw)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(gram, Zw.T @ yw, rcond=None)[0]
        return theta


# ---------------------------------------------------------------------------
# Local SDRE action extraction
# ---------------------------------------------------------------------------


def local_sdre_action(
    model: LocalBilinearModel,
    x_query: np.ndarray,
    gamma_local: float,
    lambda_q: float = 0.0,
    width_range: tuple[float, float] = (0.10, 0.30),
    skew_range: tuple[float, float] = (-0.05, 0.05),
) -> tuple[float, float]:
    """Extract the optimal (width, skew) from the local CQ model.

    Maximizes:
        J(w,s) = mu_dW(w,s) - (gamma_local/2) * var_proxy - lambda_q * E[q^2]

    For phase 1 we use mu_dW only (no explicit variance model).
    The quadratic structure gives a closed-form optimum.
    """
    theta_dw = model.fit_local_dw(x_query)
    H_dw, c_dw, d_dw = model.predict_dw_quadratic(x_query, theta_dw)

    # Objective: J = d + c^T u + 0.5 u^T H u  (maximize)
    # Inventory penalty: -lambda_q * (d_q + c_q^T u + 0.5 u^T H_q u)^2
    # For phase 1, use the simple quadratic:
    #   J(u) = c_dw^T u + 0.5 u^T H_dw u   (drop constant d)
    # with optional inventory penalty on E[q_next^2]
    H_obj = H_dw.copy()
    c_obj = c_dw.copy()

    if lambda_q > 0.0:
        theta_q = model.fit_local_q_next(x_query)
        H_q, c_q, d_q = model.predict_q_quadratic(x_query, theta_q)
        # Approximate E[q^2] ~ (d_q + c_q^T u)^2 + variance_term
        # Gradient of lambda_q * (d_q + c_q^T u)^2 w.r.t. u:
        #   2 * lambda_q * (d_q + c_q^T u) * c_q
        # Hessian: 2 * lambda_q * c_q c_q^T
        H_obj -= 2.0 * lambda_q * np.outer(c_q, c_q)
        c_obj -= 2.0 * lambda_q * d_q * c_q

    # Solve: u* = -H_obj^{-1} c_obj (for maximization, if H_obj is negative definite)
    # Check if H_obj is negative semi-definite (concave)
    eigvals = np.linalg.eigvalsh(H_obj)
    if eigvals[-1] < -1e-10:
        # Concave: use closed-form
        try:
            u_star = -np.linalg.solve(H_obj, c_obj)
        except np.linalg.LinAlgError:
            u_star = np.array([np.mean(width_range), 0.0])
    else:
        # Not concave: fallback to grid search over small candidate set
        u_star = _grid_search_action(H_dw, c_dw, d_dw, width_range, skew_range)

    # Clip to admissible box
    w = float(np.clip(u_star[0], width_range[0], width_range[1]))
    s = float(np.clip(u_star[1], skew_range[0], skew_range[1]))
    return w, s


def _grid_search_action(
    H: np.ndarray,
    c: np.ndarray,
    d: float,
    width_range: tuple[float, float],
    skew_range: tuple[float, float],
    n_grid: int = 11,
) -> np.ndarray:
    """Numerical fallback: grid search over action box."""
    ws = np.linspace(width_range[0], width_range[1], n_grid)
    ss = np.linspace(skew_range[0], skew_range[1], n_grid)
    best_val = -np.inf
    best_u = np.array([np.mean(width_range), 0.0])
    for w in ws:
        for s in ss:
            u = np.array([w, s])
            val = d + c @ u + 0.5 * u @ H @ u
            if val > best_val:
                best_val = val
                best_u = u
    return best_u


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


def make_local_bilinear_controller(
    env: OptionMarketMakingEnv,
    model: LocalBilinearModel,
    initial_state: OptionMMState,
    gamma_ce: float = 2.0,
    lambda_q: float = 0.0,
    width_range: tuple[float, float] = (0.10, 0.30),
    skew_range: tuple[float, float] = (-0.05, 0.05),
) -> EpisodeController:
    """Factory for the prior-free local bilinear controller.

    At each step:
      1. Normalize state.
      2. Fit local CQ model at current state.
      3. Extract optimal (width, skew) via SDRE.
      4. Convert to OptionMMAction.

    No BBG prior is used inside this controller.
    """
    norm = StateNorm.from_env(env)
    gamma_local = gamma_ce / initial_state.wealth

    def controller(state: OptionMMState, history: Any | None = None) -> OptionMMAction:
        del history
        x = normalize_state(state, norm)
        w, s = local_sdre_action(
            model, x, gamma_local, lambda_q, width_range, skew_range,
        )

        bid_dist = w + s
        ask_dist = w - s
        return OptionMMAction(
            bid_price=max(state.option_mid - bid_dist, 0.0),
            ask_price=state.option_mid + max(ask_dist, 0.0),
            hedge_trade=-state.net_delta,
        )

    return controller


# ---------------------------------------------------------------------------
# Instrumented version (records action choices)
# ---------------------------------------------------------------------------


def make_instrumented_bilinear_controller(
    env: OptionMarketMakingEnv,
    model: LocalBilinearModel,
    initial_state: OptionMMState,
    gamma_ce: float = 2.0,
    lambda_q: float = 0.0,
    width_range: tuple[float, float] = (0.10, 0.30),
    skew_range: tuple[float, float] = (-0.05, 0.05),
) -> tuple[EpisodeController, list[tuple[float, float]]]:
    """Like make_local_bilinear_controller but records (w, s) choices."""
    norm = StateNorm.from_env(env)
    gamma_local = gamma_ce / initial_state.wealth
    choices: list[tuple[float, float]] = []

    def controller(state: OptionMMState, history: Any | None = None) -> OptionMMAction:
        del history
        x = normalize_state(state, norm)
        w, s = local_sdre_action(
            model, x, gamma_local, lambda_q, width_range, skew_range,
        )
        choices.append((round(w, 6), round(s, 6)))
        bid_dist = w + s
        ask_dist = w - s
        return OptionMMAction(
            bid_price=max(state.option_mid - bid_dist, 0.0),
            ask_price=state.option_mid + max(ask_dist, 0.0),
            hedge_trade=-state.net_delta,
        )

    return controller, choices


# ---------------------------------------------------------------------------
# End-to-end training convenience
# ---------------------------------------------------------------------------


def train_bilinear_model(
    training_seeds: list[int] | tuple[int, ...],
    heston: HestonParams | None = None,
    horizon_steps: int = 20,
    initial_cash: float = 100_000.0,
    width_range: tuple[float, float] = (0.10, 0.30),
    skew_range: tuple[float, float] = (-0.05, 0.05),
    exploration_rng_seed: int = 42,
    bandwidth: float | None = None,
    ridge: float = 1e-3,
    max_training_samples: int = 10_000,
) -> LocalBilinearModel:
    """Collect training data and build a LocalBilinearModel."""
    buffer = collect_bilinear_training_data(
        seeds=training_seeds,
        heston=heston,
        horizon_steps=horizon_steps,
        initial_cash=initial_cash,
        width_range=width_range,
        skew_range=skew_range,
        exploration_rng_seed=exploration_rng_seed,
    )

    X = np.array(buffer.x_list)
    U = np.array(buffer.u_list)
    X_next = np.array(buffer.x_next_list)
    dW = np.array(buffer.dw_list)
    Q_next = np.array(buffer.q_next_list)

    # Subsample if needed
    if X.shape[0] > max_training_samples:
        rng = np.random.default_rng(exploration_rng_seed + 1)
        idx = rng.choice(X.shape[0], size=max_training_samples, replace=False)
        X, U, X_next, dW, Q_next = X[idx], U[idx], X_next[idx], dW[idx], Q_next[idx]

    # Bandwidth: median heuristic on state space
    if bandwidth is None:
        n_sub = min(X.shape[0], 2000)
        idx_sub = np.random.default_rng(0).choice(X.shape[0], size=n_sub, replace=False)
        X_sub = X[idx_sub]
        sq_d = (
            np.sum(X_sub**2, axis=1, keepdims=True)
            + np.sum(X_sub**2, axis=1, keepdims=True).T
            - 2.0 * X_sub @ X_sub.T
        )
        dists = np.sqrt(np.maximum(sq_d[np.triu_indices(n_sub, k=1)], 0.0))
        bandwidth = max(float(np.median(dists)), 0.1)

    return LocalBilinearModel(
        X=X, U=U, X_next=X_next, dW=dW, Q_next=Q_next,
        bandwidth=bandwidth, ridge=ridge,
    )


def train_bilinear_model_multi_cell(
    cell_params: list[HestonParams],
    seeds_per_cell: list[int] | tuple[int, ...],
    **kwargs,
) -> LocalBilinearModel:
    """Train on data pooled across multiple Heston parameter cells."""
    all_x, all_u, all_xn, all_dw, all_qn = [], [], [], [], []
    for heston in cell_params:
        buf = collect_bilinear_training_data(
            seeds=seeds_per_cell, heston=heston, **kwargs,
        )
        all_x.extend(buf.x_list)
        all_u.extend(buf.u_list)
        all_xn.extend(buf.x_next_list)
        all_dw.extend(buf.dw_list)
        all_qn.extend(buf.q_next_list)

    X = np.array(all_x)
    U = np.array(all_u)
    X_next = np.array(all_xn)
    dW = np.array(all_dw)
    Q_next = np.array(all_qn)

    max_n = kwargs.get("max_training_samples", 10_000)
    rng_seed = kwargs.get("exploration_rng_seed", 42)
    if X.shape[0] > max_n:
        rng = np.random.default_rng(rng_seed + 1)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X, U, X_next, dW, Q_next = X[idx], U[idx], X_next[idx], dW[idx], Q_next[idx]

    bw = kwargs.get("bandwidth", None)
    if bw is None:
        n_sub = min(X.shape[0], 2000)
        idx_sub = np.random.default_rng(0).choice(X.shape[0], size=n_sub, replace=False)
        X_sub = X[idx_sub]
        sq_d = (
            np.sum(X_sub**2, axis=1, keepdims=True)
            + np.sum(X_sub**2, axis=1, keepdims=True).T
            - 2.0 * X_sub @ X_sub.T
        )
        dists = np.sqrt(np.maximum(sq_d[np.triu_indices(n_sub, k=1)], 0.0))
        bw = max(float(np.median(dists)), 0.1)

    return LocalBilinearModel(
        X=X, U=U, X_next=X_next, dW=dW, Q_next=Q_next,
        bandwidth=bw, ridge=kwargs.get("ridge", 1e-3),
    )
