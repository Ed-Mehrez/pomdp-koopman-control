"""Data-driven reduced-action SDRE recovery controller for the BBG benchmark.

Two methods for learning a low-dimensional action subspace:

  Option 2 (PRIMARY) — BilinearSVD:
      Learn bilinear control channels from dynamics regression. SVD of the
      value-weighted control matrix B gives a reduced action basis. SDRE
      optimization in the reduced space uses learned revenue curvature +
      analytical vega penalty.

  Option 1 (SECONDARY) — ActionPCA:
      Eigendecomposition of the action-value Hessian (constructed from
      learned revenue curvature + vega penalty). Same SDRE optimization.

Theory separation:
  - BBG benchmark (solver.py) = solved HJB, external reference only
  - This module = data-driven controller, no HJB at action time
  - Risk-neutral distances used as exploration center (competitive MM spread,
    not from the solved value function)
  - BBG env used for simulation (exploration + evaluation)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg
import torch

from .env import OptionBookMMAction, OptionBookMMState, OptionBookMarketMakingEnv
from .pricing import bs_call_price, bs_call_vega_sqrt_nu
from .spec import BBGBenchmarkConfig


Controller = Callable[[OptionBookMMState, Any], OptionBookMMAction]


# ---------------------------------------------------------------------------
# State representations
# ---------------------------------------------------------------------------


def extract_state_compact(
    state: OptionBookMMState,
    config: BBGBenchmarkConfig,
) -> np.ndarray:
    """Compact state: (tau_frac, nu_norm, vpi_norm) — 3D."""
    return extract_state_features(state, config)


def extract_state_rich(
    state: OptionBookMMState,
    config: BBGBenchmarkConfig,
    _cache: dict | None = None,
) -> np.ndarray:
    """Richer state with portfolio structure — 7D.

    (tau_frac, nu_norm, vpi_norm, vpi_short, vpi_long,
     vega_concentration, dist_to_limit)
    """
    horizon = config.control.horizon
    tau_frac = state.time / horizon if horizon > 0 else 0.0
    nu_norm = state.variance / config.heston.nu0
    vl = config.control.vega_limit
    vpi_norm = state.portfolio_vega / vl

    # Short- vs long-maturity vega decomposition
    mats = np.array([o.maturity for o in config.book.options])
    qi_vi = state.option_inventories * state.option_vegas
    short_mask = mats <= 1.5
    vpi_short = float(np.sum(qi_vi[short_mask])) / vl
    vpi_long = float(np.sum(qi_vi[~short_mask])) / vl

    # Concentration: how much vega is in the largest single position
    abs_qi_vi = np.abs(qi_vi)
    vega_conc = float(np.max(abs_qi_vi)) / (abs(state.portfolio_vega) + 1e-6)
    vega_conc = min(vega_conc, 10.0)  # clip for numerical safety

    # Distance to vega limit
    dist_to_limit = 1.0 - abs(state.portfolio_vega) / vl

    return np.array([tau_frac, nu_norm, vpi_norm, vpi_short, vpi_long,
                     vega_conc, dist_to_limit])


# ---------------------------------------------------------------------------
# Matern 3/2 kernel with ARD
# ---------------------------------------------------------------------------


def matern32_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scales: np.ndarray,
) -> np.ndarray:
    """ARD Matern 3/2 kernel matrix.

    k(x, x') = (1 + sqrt(3) r) exp(-sqrt(3) r)
    where r = sqrt(sum_d ((x_d - x'_d) / l_d)^2)
    """
    X1s = X1 / length_scales[None, :]
    X2s = X2 / length_scales[None, :]
    sq1 = np.sum(X1s ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2s ** 2, axis=1, keepdims=True).T
    dist_sq = np.maximum(sq1 + sq2 - 2.0 * X1s @ X2s.T, 0.0)
    dist = np.sqrt(dist_sq)
    s3d = np.sqrt(3.0) * dist
    return (1.0 + s3d) * np.exp(-s3d)


def matern32_kernel_torch(
    X1: torch.Tensor,
    X2: torch.Tensor,
    length_scales: torch.Tensor,
) -> torch.Tensor:
    """Torch version of the ARD Matern 3/2 kernel."""
    X1s = X1 / length_scales.unsqueeze(0)
    X2s = X2 / length_scales.unsqueeze(0)
    sq1 = torch.sum(X1s ** 2, dim=1, keepdim=True)
    sq2 = torch.sum(X2s ** 2, dim=1, keepdim=True).T
    dist_sq = torch.clamp(sq1 + sq2 - 2.0 * (X1s @ X2s.T), min=0.0)
    dist = torch.sqrt(dist_sq)
    s3d = np.sqrt(3.0) * dist
    return (1.0 + s3d) * torch.exp(-s3d)


def _kmeans_pp_landmarks(
    X: np.ndarray,
    M: int,
    length_scales: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """D^2-sampling (kmeans++) landmark selection.

    Distances are measured in length-scale-normalized Euclidean space if
    `length_scales` is provided, so landmarks are diverse under the same
    metric the kernel uses.
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    N = X.shape[0]
    if M >= N:
        return X.copy()
    if length_scales is not None:
        Xs = X / length_scales[None, :]
    else:
        Xs = X
    idx = np.empty(M, dtype=np.int64)
    idx[0] = int(rng.integers(N))
    min_sq = np.sum((Xs - Xs[idx[0]]) ** 2, axis=1)
    for m in range(1, M):
        total = float(min_sq.sum())
        if total <= 0.0:
            remaining = np.setdiff1d(np.arange(N), idx[:m])
            if remaining.size == 0:
                idx[m:] = idx[m - 1]
                break
            idx[m] = int(rng.choice(remaining))
        else:
            probs = np.maximum(min_sq / total, 0.0)
            probs = probs / probs.sum()
            idx[m] = int(rng.choice(N, p=probs))
        new_sq = np.sum((Xs - Xs[idx[m]]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, new_sq)
    return X[idx].copy()


class KernelRidgeModel:
    """KRR with ARD Matern 3/2 for multi-output regression.

    Backends
    --------
    approx = "exact":
        Solve (K + alpha I) w = Y with the full N x N kernel. GPU via
        torch if available.
    approx = "nystrom":
        Restrict the solution to the span of M landmarks chosen from X,
        solve (K_nm^T K_nm + alpha K_mm) beta = K_nm^T Y. Used as a fast
        search backend for hyperparameter tuning; final claims should
        be refit with approx = "exact".
    """

    def __init__(
        self,
        length_scales: np.ndarray,
        alpha: float = 1e-2,
        device: str | None = None,
        approx: str = "exact",
        n_landmarks: int | None = None,
        landmark_method: str = "kmeans++",
        landmark_seed: int | None = None,
    ):
        self.length_scales = length_scales
        self.alpha = alpha
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        if approx not in ("exact", "nystrom"):
            raise ValueError(f"approx must be 'exact' or 'nystrom', got {approx!r}")
        self.approx = approx
        self.n_landmarks = n_landmarks
        if landmark_method not in ("kmeans++", "random"):
            raise ValueError(
                f"landmark_method must be 'kmeans++' or 'random', got {landmark_method!r}"
            )
        self.landmark_method = landmark_method
        self.landmark_seed = landmark_seed
        # Exact-mode state
        self.X_train: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.X_train_t: torch.Tensor | None = None
        self.weights_t: torch.Tensor | None = None
        self.length_scales_t: torch.Tensor | None = None
        self.L_chol: np.ndarray | None = None  # Cholesky for posterior variance
        # Nystrom-mode state
        self.landmarks: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        # Diagnostics
        self.fit_cond_number: float | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        if self.approx == "exact":
            self._fit_exact(X, Y)
        else:
            self._fit_nystrom(X, Y)

    def _fit_exact(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit: weights = (K + alpha I)^{-1} Y.

        Stores Cholesky factor L of (K + alpha I) on CPU for posterior
        variance computation via predict_with_variance.
        """
        self.X_train = X.copy()
        K_np = matern32_kernel(X, X, self.length_scales)
        N = K_np.shape[0]
        A = K_np + self.alpha * np.eye(N)
        # Cholesky for variance (always CPU float64)
        self.L_chol = linalg.cholesky(A, lower=True)

        if self.device == "cuda":
            x_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)
            y_t = torch.as_tensor(Y, dtype=torch.float64, device=self.device)
            ls_t = torch.as_tensor(
                self.length_scales, dtype=torch.float64, device=self.device
            )
            K_t = matern32_kernel_torch(x_t, x_t, ls_t)
            eye_t = torch.eye(N, dtype=torch.float64, device=self.device)
            w_t = torch.linalg.solve(K_t + self.alpha * eye_t, y_t)
            self.X_train_t = x_t
            self.weights_t = w_t
            self.length_scales_t = ls_t
            self.weights = w_t.detach().cpu().numpy()
        else:
            self.weights = linalg.cho_solve(
                (self.L_chol, True), Y,
            )

    def _fit_nystrom(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Nystrom KRR: solve (K_nm^T K_nm + alpha K_mm) beta = K_nm^T Y.

        CPU / numpy only. Uses a Cholesky solve with jitter fallback to
        lstsq; records the condition number of the normal-equation matrix
        as a diagnostic.
        """
        N = X.shape[0]
        M = self.n_landmarks if self.n_landmarks is not None else min(512, N)
        M = min(M, N)

        if self.landmark_method == "kmeans++":
            Z = _kmeans_pp_landmarks(X, M, self.length_scales, self.landmark_seed)
        else:
            rng = np.random.default_rng(
                self.landmark_seed if self.landmark_seed is not None else 0
            )
            idx = rng.choice(N, M, replace=False)
            Z = X[idx].copy()

        K_nm = matern32_kernel(X, Z, self.length_scales)          # (N, M)
        K_mm = matern32_kernel(Z, Z, self.length_scales)          # (M, M)

        A = K_nm.T @ K_nm + self.alpha * K_mm
        A = 0.5 * (A + A.T)                                       # symmetrize
        # Diagonal jitter scaled to trace for stability
        jitter = 1e-8 * max(float(np.trace(A)) / M, 1.0)
        A = A + jitter * np.eye(M)
        b = K_nm.T @ Y

        try:
            self.beta = linalg.solve(A, b, assume_a="pos")
        except (linalg.LinAlgError, np.linalg.LinAlgError):
            self.beta = linalg.lstsq(A, b)[0]

        try:
            eigs = linalg.eigvalsh(A)
            emin = float(eigs.min())
            emax = float(eigs.max())
            self.fit_cond_number = emax / max(emin, 1e-30)
        except Exception:
            self.fit_cond_number = None

        self.landmarks = Z

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict: f(x) = K(x, X_train) @ weights (exact) or
        K(x, Z) @ beta (nystrom)."""
        if self.approx == "nystrom":
            K_new = matern32_kernel(X_new, self.landmarks, self.length_scales)
            return K_new @ self.beta
        if self.device == "cuda" and self.X_train_t is not None:
            x_new_t = torch.as_tensor(X_new, dtype=torch.float64, device=self.device)
            K_new_t = matern32_kernel_torch(
                x_new_t, self.X_train_t, self.length_scales_t
            )
            pred_t = K_new_t @ self.weights_t
            return pred_t.detach().cpu().numpy()
        K_new = matern32_kernel(X_new, self.X_train, self.length_scales)
        return K_new @ self.weights

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """Predict for a single input vector."""
        return self.predict(x[None, :])[0]

    def predict_with_variance(
        self, X_new: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and GP posterior variance.

        Returns
        -------
        mean : (M, d_out) predicted values
        var : (M,) GP posterior variance per test point.
            For a stationary kernel, k(x,x) = 1, so
            var(x*) = 1 - k_*^T (K + alpha I)^{-1} k_*
            This is the same scalar for all outputs (shared kernel).
        """
        if self.approx == "nystrom" or self.L_chol is None:
            mean = self.predict(X_new)
            # No Cholesky available — return NaN variance
            return mean, np.full(X_new.shape[0], np.nan)

        k_star = matern32_kernel(X_new, self.X_train, self.length_scales)  # (M, N)
        mean = k_star @ self.weights  # (M, d_out)

        # var(x*) = k(x*, x*) - k_*^T L^{-T} L^{-1} k_*
        #         = 1 - ||v||^2  where L v = k_*^T
        v = linalg.solve_triangular(
            self.L_chol, k_star.T, lower=True,
        )  # (N, M)
        var = 1.0 - np.sum(v ** 2, axis=0)  # (M,)
        var = np.maximum(var, 0.0)
        return mean, var

    def predict_single_with_variance(
        self, x: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict mean and variance for a single input."""
        mean, var = self.predict_with_variance(x[None, :])
        return mean[0], float(var[0])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SDRERecoveryConfig:
    """Configuration for the SDRE recovery controller."""

    n_explore_episodes: int = 500
    explore_noise_std: float = 0.3   # log-normal multiplicative noise on delta
    rank: int = 3                    # reduced action dimension
    ridge_alpha: float = 1e-3        # regularization for regressions
    method: str = "bilinear"         # "bilinear", "action_pca", or "bilinear_2stage"
    bilinear_overspace: int = 10     # overspace rank for 2-stage bilinear
    explore_seed: int = 42


# ---------------------------------------------------------------------------
# Exploration data
# ---------------------------------------------------------------------------


@dataclass
class ExplorationData:
    """Per-step transition data from exploratory episodes."""

    state_features: np.ndarray    # (N, 3)  tau_frac, nu_norm, vpi_norm
    actions: np.ndarray           # (N, 2*n_options)
    vpi_pre: np.ndarray           # (N,)
    vpi_post: np.ndarray          # (N,)
    inventory_changes: np.ndarray # (N, n_options)
    spread_captures: np.ndarray   # (N,)


def extract_state_features(
    state: OptionBookMMState,
    config: BBGBenchmarkConfig,
) -> np.ndarray:
    """Normalized state features: (tau_frac, nu_norm, vpi_norm)."""
    horizon = config.control.horizon
    tau_frac = state.time / horizon if horizon > 0 else 0.0
    nu_norm = state.variance / config.heston.nu0
    vpi_norm = state.portfolio_vega / config.control.vega_limit
    return np.array([tau_frac, nu_norm, vpi_norm])


def _compute_rn_distances(config: BBGBenchmarkConfig) -> np.ndarray:
    """Risk-neutral optimal quote distances (p=0, competitive MM spread).

    Uses the logistic intensity model parameters (simulation environment),
    NOT the solved HJB value function.
    """
    from .solver import _optimal_quote_logistic

    h = config.heston
    liq = config.liquidity
    n_opt = config.book.n_options
    vegas = np.array([
        bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
        for o in config.book.options
    ])
    lambdas = np.array([
        liq.lambda_i(h.spot0, o.strike) for o in config.book.options
    ])
    return np.array([
        _optimal_quote_logistic(0.0, lambdas[i], vegas[i], liq.alpha, liq.beta)
        for i in range(n_opt)
    ])


def collect_exploration_data(
    config: BBGBenchmarkConfig,
    rn_distances: np.ndarray,
    sdre_config: SDRERecoveryConfig,
) -> ExplorationData:
    """Run exploratory episodes with log-normal noise on risk-neutral quotes.

    At each step, independently sample:
        delta^{bid}_i = rn_dist_i * exp(N(0, sigma^2))
        delta^{ask}_i = rn_dist_i * exp(N(0, sigma^2))
    """
    n_opt = config.book.n_options
    sigma = sdre_config.explore_noise_std
    rng = np.random.default_rng(sdre_config.explore_seed)

    all_sf, all_u, all_vpi_pre, all_vpi_post = [], [], [], []
    all_dinv, all_spread = [], []

    for ep in range(sdre_config.n_explore_episodes):
        env = OptionBookMarketMakingEnv(config, seed=ep)
        state = env.reset()

        while not state.done:
            sf = extract_state_features(state, config)
            vpi_pre = state.portfolio_vega
            inv_pre = state.option_inventories.copy()

            # Log-normal exploration noise
            bid_dists = rn_distances * np.exp(rng.normal(0, sigma, n_opt))
            ask_dists = rn_distances * np.exp(rng.normal(0, sigma, n_opt))
            bid_dists = np.maximum(bid_dists, 1e-6)
            ask_dists = np.maximum(ask_dists, 1e-6)

            action = OptionBookMMAction(
                bid_distances=bid_dists,
                ask_distances=ask_dists,
                hedge_trade=-state.net_delta,
            )
            u = np.concatenate([bid_dists, ask_dists])

            next_state, _, _, info = env.step(action)

            all_sf.append(sf)
            all_u.append(u)
            all_vpi_pre.append(vpi_pre)
            all_vpi_post.append(next_state.portfolio_vega)
            all_dinv.append(next_state.option_inventories - inv_pre)
            all_spread.append(info["spread_capture"])

            state = next_state

    return ExplorationData(
        state_features=np.array(all_sf),
        actions=np.array(all_u),
        vpi_pre=np.array(all_vpi_pre),
        vpi_post=np.array(all_vpi_post),
        inventory_changes=np.array(all_dinv),
        spread_captures=np.array(all_spread),
    )


# ---------------------------------------------------------------------------
# Regression utility
# ---------------------------------------------------------------------------


def _ridge_regression(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Ridge regression: Y = X @ beta + eps. Returns beta.

    X: (N, d), Y: (N,) or (N, p). Returns (d,) or (d, p).
    """
    XtX = X.T @ X
    reg = alpha * np.eye(XtX.shape[0])
    XtY = X.T @ Y
    return linalg.solve(XtX + reg, XtY, assume_a="pos")


# ---------------------------------------------------------------------------
# Option 2: Bilinear control model
# ---------------------------------------------------------------------------


class BilinearControlModel:
    """Learn control channels from dynamics, SVD for reduced action basis.

    Fits bilinear dynamics:
        Y = X_state @ A + (U - u_center) @ B + epsilon

    Response Y is value-weighted:
        [spread_capture, z_1 V_1 Δq_1, ..., z_N V_N Δq_N]
    giving B ∈ R^{2N × (N+1)}, with rank up to N+1.

    The SVD of B reveals the principal action directions ranked by
    their importance for the value-relevant state changes.
    """

    def __init__(self, config: BBGBenchmarkConfig, ridge_alpha: float = 1e-3):
        self.config = config
        self.n_options = config.book.n_options
        self.m = 2 * self.n_options   # full action dim
        self.ridge_alpha = ridge_alpha

        h = config.heston
        self._vegas = np.array([
            bs_call_vega_sqrt_nu(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
            for o in config.book.options
        ])
        prices = np.array([
            bs_call_price(h.spot0, o.strike, o.maturity, h.rate, h.nu0)
            for o in config.book.options
        ])
        self._trade_sizes = np.array([
            config.liquidity.trade_size(p) for p in prices
        ])
        self._vega_weights = self._trade_sizes * self._vegas

        # Fitted quantities (populated by fit + reduce)
        self.B: np.ndarray | None = None       # (m, n_resp) control channel
        self.U_r: np.ndarray | None = None     # (m, rank) reduced basis
        self.S_r: np.ndarray | None = None     # (rank,) singular values
        self.u_center: np.ndarray | None = None
        self.u_scale: np.ndarray | None = None  # (m,) per-dim std for normalization

        self.vega_channel: np.ndarray | None = None  # (m,) in ORIGINAL action units
        self.rev_linear: np.ndarray | None = None     # (m,) in ORIGINAL action units
        self.rev_quad: np.ndarray | None = None       # (m,) in ORIGINAL action units

    def fit(self, data: ExplorationData) -> None:
        """Ridge regression for control channels and revenue model.

        Actions are normalized per-dimension before regression so that
        the SVD and SDRE operate in properly scaled coordinates.
        The learned channels (vega_channel, rev_linear, rev_quad) are
        stored in ORIGINAL action units for compatibility with ActionPCA.
        """
        N = len(data.actions)
        n_opt = self.n_options

        self.u_center = data.actions.mean(axis=0)
        self.u_scale = np.maximum(data.actions.std(axis=0), 1e-12)
        U_c = data.actions - self.u_center
        U_norm = U_c / self.u_scale    # normalized actions

        # State features with intercept
        X_state = np.column_stack([np.ones(N), data.state_features])
        d_s = X_state.shape[1]

        # --- Bilinear dynamics: value-weighted response ---
        Y_inv_w = data.inventory_changes * self._vega_weights[None, :]
        Y_spread = data.spread_captures[:, None]
        Y = np.column_stack([Y_spread, Y_inv_w])   # (N, n_opt+1)

        X_full_norm = np.column_stack([X_state, U_norm])
        beta_norm = _ridge_regression(X_full_norm, Y, self.ridge_alpha)
        self.B = beta_norm[d_s:]   # (m, n_opt+1) in NORMALIZED action units

        # --- Vega channel (original units for SDRE compatibility) ---
        X_full_raw = np.column_stack([X_state, U_c])
        beta_vpi = _ridge_regression(X_full_raw, data.vpi_post - data.vpi_pre,
                                     self.ridge_alpha)
        self.vega_channel = beta_vpi[d_s:]   # (m,) original units

        # --- Revenue model (original units) ---
        X_rev = np.column_stack([X_state, U_c, U_c ** 2])
        beta_rev = _ridge_regression(X_rev, data.spread_captures, self.ridge_alpha)
        self.rev_linear = beta_rev[d_s : d_s + self.m]
        self.rev_quad = beta_rev[d_s + self.m :]
        # Enforce concavity: revenue curvature must be non-positive
        self.rev_quad = np.minimum(self.rev_quad, -1e-12)

    def reduce(self, rank: int) -> None:
        """SVD of B to extract reduced action basis.

        B is in normalized action units. The SVD basis U_norm gives
        directions in normalized space. We convert to original-unit
        directions by multiplying by u_scale, then re-normalizing
        each column to unit norm.
        """
        U_norm, S, _ = np.linalg.svd(self.B, full_matrices=False)
        effective_rank = min(rank, U_norm.shape[1])

        # Convert normalized-space directions to original-unit directions
        U_orig = U_norm[:, :effective_rank] * self.u_scale[:, None]
        # Re-normalize columns
        norms = np.linalg.norm(U_orig, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        self.U_r = U_orig / norms
        self.S_r = S[:effective_rank]

    def explained_variance(self) -> np.ndarray:
        """Fraction of variance explained by each singular direction."""
        _, S, _ = np.linalg.svd(self.B, full_matrices=False)
        total = np.sum(S ** 2)
        if total < 1e-30:
            return S * 0.0
        return S ** 2 / total


# ---------------------------------------------------------------------------
# Option 1: Action PCA model
# ---------------------------------------------------------------------------


class ActionPCAModel:
    """Eigendecomposition of the action-value Hessian for reduced basis.

    Constructs:
        H = diag(rev_quad) - c_pen * outer(vega_channel, vega_channel)

    The eigenvectors of H (ordered by most negative eigenvalue) give
    the principal action directions where the risk-adjusted objective
    has the most curvature.
    """

    def __init__(self, config: BBGBenchmarkConfig, ridge_alpha: float = 1e-3):
        self.config = config
        self.n_options = config.book.n_options
        self.m = 2 * self.n_options
        self.ridge_alpha = ridge_alpha

        self.U_r: np.ndarray | None = None
        self.S_r: np.ndarray | None = None
        self.u_center: np.ndarray | None = None

        self.vega_channel: np.ndarray | None = None
        self.rev_linear: np.ndarray | None = None
        self.rev_quad: np.ndarray | None = None
        self.H_full: np.ndarray | None = None

    def fit(
        self,
        bilinear_model: BilinearControlModel,
        gamma: float,
        xi: float,
        dt: float,
    ) -> None:
        """Construct Hessian from bilinear model's learned components."""
        self.u_center = bilinear_model.u_center
        self.vega_channel = bilinear_model.vega_channel
        self.rev_linear = bilinear_model.rev_linear
        self.rev_quad = bilinear_model.rev_quad

        c_pen = gamma * xi ** 2 / 8.0 * dt
        self.H_full = (
            np.diag(self.rev_quad)
            - c_pen * np.outer(self.vega_channel, self.vega_channel)
        )

    def reduce(self, rank: int) -> None:
        """Eigendecomposition of H for principal action directions."""
        eigvals, eigvecs = np.linalg.eigh(self.H_full)
        # Most negative eigenvalue = most curvature = most important
        idx = np.argsort(eigvals)   # ascending (most negative first)
        effective_rank = min(rank, len(eigvals))
        self.U_r = eigvecs[:, idx[:effective_rank]]
        self.S_r = np.abs(eigvals[idx[:effective_rank]])


# ---------------------------------------------------------------------------
# SDRE solve
# ---------------------------------------------------------------------------


def _sdre_solve(
    vpi: float,
    U_r: np.ndarray,
    vega_channel: np.ndarray,
    rev_linear: np.ndarray,
    rev_quad: np.ndarray,
    gamma: float,
    xi: float,
    dt: float,
    reg: float = 1e-8,
) -> np.ndarray:
    """Solve the local quadratic SDRE in reduced coordinates.

    Objective in reduced space (perturbation a from baseline):
        J(a) = g_r^T a + a^T H_r a
    where:
        g_r = rev_lin_r - 2 c_pen V^pi vega_r
        H_r = Rev_quad_r - c_pen outer(vega_r, vega_r)

    Returns optimal reduced-action perturbation a*.
    """
    rank = U_r.shape[1]

    vega_r = U_r.T @ vega_channel
    rev_lin_r = U_r.T @ rev_linear
    Rev_quad_r = U_r.T @ np.diag(rev_quad) @ U_r

    c_pen = gamma * xi ** 2 / 8.0 * dt

    g_r = rev_lin_r - 2.0 * c_pen * vpi * vega_r
    H_r = Rev_quad_r - c_pen * np.outer(vega_r, vega_r)

    # Regularize to ensure strict negative definiteness
    H_r -= reg * np.eye(rank)

    try:
        a_star = -0.5 * linalg.solve(H_r, g_r)
    except linalg.LinAlgError:
        a_star = np.zeros(rank)

    return a_star


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


def make_sdre_recovery_controller(
    config: BBGBenchmarkConfig,
    sdre_config: SDRERecoveryConfig,
    rn_distances: np.ndarray | None = None,
    return_model: bool = False,
) -> Controller | tuple[Controller, BilinearControlModel | ActionPCAModel]:
    """Build a data-driven SDRE recovery controller.

    Phases:
      1. Compute risk-neutral baseline (competitive MM spread, no HJB).
      2. Collect exploration data with log-normal noise on baseline quotes.
      3. Fit bilinear dynamics model (revenue + vega channels).
      4. Reduce to a rank-r action subspace (SVD or eigendecomposition).
      5. Return a controller that solves a local quadratic SDRE per step.

    No BBG solver or value function is used at action time.
    """
    n_opt = config.book.n_options
    h = config.heston

    if rn_distances is None:
        rn_distances = _compute_rn_distances(config)

    # Phase 1-2: collect exploration data
    data = collect_exploration_data(config, rn_distances, sdre_config)

    # Phase 3: fit bilinear model (both methods need it for revenue + vega)
    bilinear = BilinearControlModel(config, sdre_config.ridge_alpha)
    bilinear.fit(data)

    env_dt = config.control.horizon / 30  # default env dt

    if sdre_config.method == "bilinear":
        bilinear.reduce(sdre_config.rank)
        model = bilinear
    elif sdre_config.method == "action_pca":
        model = ActionPCAModel(config, sdre_config.ridge_alpha)
        model.fit(bilinear, config.control.gamma, h.xi, env_dt)
        model.reduce(sdre_config.rank)
        # Share learned revenue/vega channels
        model.vega_channel = bilinear.vega_channel
        model.rev_linear = bilinear.rev_linear
        model.rev_quad = bilinear.rev_quad
    elif sdre_config.method == "bilinear_2stage":
        # Two-stage: bilinear SVD for dynamics overspace, then Hessian
        # eigendecomposition within the overspace for the final basis.
        overspace = min(sdre_config.bilinear_overspace, 2 * n_opt)
        bilinear.reduce(overspace)
        U_over = bilinear.U_r  # (m, overspace) dynamics-relevant overspace

        # Build Hessian within the overspace
        c_pen = config.control.gamma * h.xi ** 2 / 8.0 * env_dt
        vc_proj = U_over.T @ bilinear.vega_channel
        rq_proj = U_over.T @ np.diag(bilinear.rev_quad) @ U_over
        H_over = rq_proj - c_pen * np.outer(vc_proj, vc_proj)

        # Eigendecomposition within overspace
        eigvals, eigvecs = np.linalg.eigh(H_over)
        idx = np.argsort(eigvals)  # most negative first
        k = min(sdre_config.rank, len(eigvals))
        V_inner = eigvecs[:, idx[:k]]

        # Map back to full action space: U_final = U_over @ V_inner
        U_final = U_over @ V_inner
        # Re-normalize columns
        norms = np.linalg.norm(U_final, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-15)

        bilinear.U_r = U_final / norms
        bilinear.S_r = np.abs(eigvals[idx[:k]])
        model = bilinear
    else:
        raise ValueError(f"Unknown method: {sdre_config.method}")

    # Phase 4: build controller closure
    u_baseline = np.concatenate([rn_distances, rn_distances])
    U_r = model.U_r
    vega_channel = model.vega_channel
    rev_linear = model.rev_linear
    rev_quad = model.rev_quad
    gamma = config.control.gamma
    xi = h.xi

    # Maximum perturbation as fraction of baseline (trust-region safeguard)
    max_pert_frac = 0.8

    def controller(state: OptionBookMMState, history: Any = None) -> OptionBookMMAction:
        a_star = _sdre_solve(
            vpi=state.portfolio_vega,
            U_r=U_r,
            vega_channel=vega_channel,
            rev_linear=rev_linear,
            rev_quad=rev_quad,
            gamma=gamma,
            xi=xi,
            dt=env_dt,
        )
        u_delta = U_r @ a_star
        # Proportional clamp: perturbation ≤ max_pert_frac * baseline
        u_delta = np.clip(u_delta, -max_pert_frac * u_baseline, max_pert_frac * u_baseline)
        u_full = u_baseline + u_delta
        bid_dists = np.maximum(u_full[:n_opt], 1e-6)
        ask_dists = np.maximum(u_full[n_opt:], 1e-6)
        return OptionBookMMAction(
            bid_distances=bid_dists,
            ask_distances=ask_dists,
            hedge_trade=-state.net_delta,
        )

    if return_model:
        return controller, model
    return controller


# ---------------------------------------------------------------------------
# Kernelized state-conditioned reduced-action controller
# ---------------------------------------------------------------------------


def make_kernelized_recovery_controller(
    config: BBGBenchmarkConfig,
    U_r: np.ndarray,
    bbg_ctrl: Controller,
    rn_distances: np.ndarray,
    train_seeds: list[int],
    state_rep: str = "compact",
    n_subsample: int = 3000,
    krr_alpha: float = 1e-2,
    ls_multiplier: float = 1.0,
    device: str | None = None,
    approx: str = "exact",
    n_landmarks: int | None = None,
    landmark_method: str = "kmeans++",
    landmark_seed: int | None = None,
    return_model: bool = False,
) -> Controller:
    """Kernelized reduced-action controller trained to recover BBG.

    Keeps U_r fixed and learns a(x) = KRR(state -> reduced_coords)
    trained on BBG's projected actions.

    Parameters
    ----------
    U_r : (m, rank) fixed reduced action basis
    bbg_ctrl : BBG benchmark controller (for demonstrations)
    rn_distances : risk-neutral baseline distances
    train_seeds : episode seeds for BBG demonstration collection
    state_rep : "compact" (3D) or "rich" (7D)
    n_subsample : max training points for KRR (memory bound)
    krr_alpha : KRR ridge parameter
    ls_multiplier : multiply auto length-scales by this factor
    approx : "exact" (default) or "nystrom" — Nystrom is a fast search
        backend for hyperparameter tuning; final claims should refit
        with "exact".
    n_landmarks : number of Nystrom landmarks (only used if approx="nystrom")
    landmark_method : "kmeans++" (default) or "random"
    landmark_seed : RNG seed for landmark selection
    return_model : also return the fitted KernelRidgeModel (default False)
    """
    n_opt = config.book.n_options
    u_baseline = np.concatenate([rn_distances, rn_distances])
    max_dist = 10.0 * np.max(rn_distances)

    feat_fn = extract_state_compact if state_rep == "compact" else extract_state_rich

    # --- Collect BBG demonstrations ---
    all_features, all_targets = [], []
    for seed in train_seeds:
        env = OptionBookMarketMakingEnv(config, seed=seed)
        state = env.reset()
        while not state.done:
            x = feat_fn(state, config)
            action = bbg_ctrl(state)
            u = np.concatenate([action.bid_distances, action.ask_distances])
            u_clipped = np.minimum(u, max_dist)
            delta_u = u_clipped - u_baseline
            a_target = U_r.T @ delta_u  # project to reduced coordinates

            all_features.append(x)
            all_targets.append(a_target)
            state, _, _, _ = env.step(action)

    X_all = np.array(all_features)
    Y_all = np.array(all_targets)

    # Subsample for KRR scalability
    N = len(X_all)
    if N > n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, n_subsample, replace=False)
        X_train = X_all[idx]
        Y_train = Y_all[idx]
    else:
        X_train = X_all
        Y_train = Y_all

    # Auto length-scales from data spread
    ls = np.maximum(X_train.std(axis=0), 1e-8) * ls_multiplier

    # Fit KRR
    krr = KernelRidgeModel(
        length_scales=ls,
        alpha=krr_alpha,
        device=device,
        approx=approx,
        n_landmarks=n_landmarks,
        landmark_method=landmark_method,
        landmark_seed=landmark_seed,
    )
    krr.fit(X_train, Y_train)

    # --- Build controller ---
    def controller(state: OptionBookMMState, history: Any = None) -> OptionBookMMAction:
        x = feat_fn(state, config)
        a = krr.predict_single(x)
        u_delta = U_r @ a
        u_full = u_baseline + u_delta
        return OptionBookMMAction(
            bid_distances=np.maximum(u_full[:n_opt], 1e-6),
            ask_distances=np.maximum(u_full[n_opt:], 1e-6),
            hedge_trade=-state.net_delta,
        )

    if return_model:
        return controller, krr
    return controller


# ---------------------------------------------------------------------------
# Level 2: deterministic encoder + Bayesian KRR head
# ---------------------------------------------------------------------------


def make_level2_controller(
    config: BBGBenchmarkConfig,
    U_r: np.ndarray,
    bbg_ctrl: Controller,
    rn_distances: np.ndarray,
    train_seeds: list[int],
    demo_cache: dict[str, np.ndarray] | None = None,
    encoder_type: str = "compact",
    n_subsample: int = 3000,
    krr_alpha: float = 1e-2,
    ls_multiplier: float = 1.0,
    deepsets_latent_dim: int = 8,
    deepsets_hidden_dim: int = 32,
    deepsets_element_dim: int = 16,
    deepsets_epochs: int = 200,
    deepsets_lr: float = 1e-3,
    deepsets_seed: int = 0,
    return_diagnostics: bool = False,
) -> Controller | tuple[Controller, dict]:
    """Level-2 controller: deterministic encoder + Bayesian KRR head.

    The encoder maps the full option-book state to a latent z.
    The Bayesian KRR head maps z → reduced action coordinates with
    posterior variance.

    Parameters
    ----------
    U_r : (m, rank) fixed reduced action basis (ActionPCA r3)
    bbg_ctrl : BBG benchmark controller for demonstrations
    rn_distances : risk-neutral baseline distances
    train_seeds : episode seeds for demonstration collection
    demo_cache : optional precomputed BBG demonstration bundle with keys:
        - Y_all
        - compact_all
        - rich_all
        - per_opt_all
        - global_all
    encoder_type : "compact" (3D), "rich" (7D), or "deepsets" (learned)
    n_subsample : max training points for KRR
    krr_alpha : KRR ridge parameter
    ls_multiplier : auto length-scale multiplier
    deepsets_* : DeepSets hyperparameters (only used if encoder_type="deepsets")
    return_diagnostics : return (controller, diagnostics_dict)

    Returns
    -------
    controller : the Level-2 controller
    diagnostics : (if return_diagnostics) dict with model, encoder info, etc.
    """
    n_opt = config.book.n_options
    u_baseline = np.concatenate([rn_distances, rn_distances])
    max_dist = 10.0 * np.max(rn_distances)

    # --- Collect or reuse BBG demonstrations ---
    if demo_cache is None:
        all_per_opt, all_global, all_targets = [], [], []

        if encoder_type == "deepsets":
            from .state_encoder import (
                extract_per_option_features,
                extract_global_features,
            )

        for seed in train_seeds:
            env = OptionBookMarketMakingEnv(config, seed=seed)
            state = env.reset()
            while not state.done:
                action = bbg_ctrl(state)
                u = np.concatenate([action.bid_distances, action.ask_distances])
                u_clipped = np.minimum(u, max_dist)
                delta_u = u_clipped - u_baseline
                a_target = U_r.T @ delta_u

                if encoder_type == "deepsets":
                    all_per_opt.append(extract_per_option_features(state, config))
                    all_global.append(extract_global_features(state, config))
                elif encoder_type == "compact":
                    all_global.append(extract_state_compact(state, config))
                elif encoder_type == "rich":
                    all_global.append(extract_state_rich(state, config))
                else:
                    raise ValueError(f"Unknown encoder_type: {encoder_type!r}")

                all_targets.append(a_target)
                state, _, _, _ = env.step(action)

        Y_all = np.array(all_targets)
    else:
        Y_all = np.array(demo_cache["Y_all"])

    N_total = len(Y_all)

    # Subsample
    rng = np.random.default_rng(42)
    if N_total > n_subsample:
        idx = rng.choice(N_total, n_subsample, replace=False)
    else:
        idx = np.arange(N_total)

    diagnostics: dict[str, Any] = {
        "encoder_type": encoder_type,
        "n_demos": N_total,
        "n_train": len(idx),
    }

    if encoder_type == "deepsets":
        from .state_encoder import train_deepsets_encoder

        if demo_cache is None:
            per_opt_all = np.array(all_per_opt)
            global_all = np.array(all_global)
        else:
            per_opt_all = np.array(demo_cache["per_opt_all"])
            global_all = np.array(demo_cache["global_all"])

        # Train encoder on ALL data, then subsample for KRR
        encoder, normalizer, train_info = train_deepsets_encoder(
            per_opt_all,
            global_all,
            Y_all,
            latent_dim=deepsets_latent_dim,
            hidden_dim=deepsets_hidden_dim,
            element_dim=deepsets_element_dim,
            n_epochs=deepsets_epochs,
            lr=deepsets_lr,
            seed=deepsets_seed,
            verbose=False,
        )
        diagnostics["encoder_train_info"] = train_info
        diagnostics["encoder"] = encoder
        diagnostics["normalizer"] = normalizer

        # Extract latent representations
        import torch as _torch
        encoder.eval()
        with _torch.no_grad():
            po_norm = normalizer.normalize_per_opt(
                per_opt_all.reshape(-1, per_opt_all.shape[-1]),
            ).reshape(per_opt_all.shape)
            gl_norm = normalizer.normalize_global(global_all)
            po_t = _torch.as_tensor(po_norm, dtype=_torch.float32)
            gl_t = _torch.as_tensor(gl_norm, dtype=_torch.float32)
            Z_all = encoder(po_t, gl_t).numpy().astype(np.float64)

        X_train = Z_all[idx]
        Y_train = Y_all[idx]
        diagnostics["Z_all"] = Z_all
    else:
        if demo_cache is None:
            X_all = np.array(all_global)
        elif encoder_type == "compact":
            X_all = np.array(demo_cache["compact_all"])
        elif encoder_type == "rich":
            X_all = np.array(demo_cache["rich_all"])
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
        X_train = X_all[idx]
        Y_train = Y_all[idx]
        diagnostics["X_all"] = X_all

    diagnostics["X_train"] = X_train
    diagnostics["Y_train"] = Y_train
    diagnostics["Y_all"] = Y_all

    # Auto length-scales
    ls = np.maximum(X_train.std(axis=0), 1e-8) * ls_multiplier

    # Fit Bayesian KRR (exact mode, CPU — needed for posterior variance)
    krr = KernelRidgeModel(
        length_scales=ls,
        alpha=krr_alpha,
        device="cpu",
        approx="exact",
    )
    krr.fit(X_train, Y_train)
    diagnostics["krr"] = krr

    # --- Build controller closure ---
    if encoder_type == "deepsets":
        import torch as _torch
        from .state_encoder import (
            extract_per_option_features as _extract_po,
            extract_global_features as _extract_gl,
        )
        _encoder = encoder
        _normalizer = normalizer

        def controller(state: OptionBookMMState, history: Any = None) -> OptionBookMMAction:
            po = _extract_po(state, config)[None, :]  # (1, n_opt, d)
            gl = _extract_gl(state, config)[None, :]   # (1, d_g)
            po_n = _normalizer.normalize_per_opt(
                po.reshape(-1, po.shape[-1]),
            ).reshape(po.shape)
            gl_n = _normalizer.normalize_global(gl)
            with _torch.no_grad():
                z = _encoder(
                    _torch.as_tensor(po_n, dtype=_torch.float32),
                    _torch.as_tensor(gl_n, dtype=_torch.float32),
                ).numpy().astype(np.float64)
            a = krr.predict_single(z[0])
            u_delta = U_r @ a
            u_full = u_baseline + u_delta
            return OptionBookMMAction(
                bid_distances=np.maximum(u_full[:n_opt], 1e-6),
                ask_distances=np.maximum(u_full[n_opt:], 1e-6),
                hedge_trade=-state.net_delta,
            )
    else:
        feat_fn = extract_state_compact if encoder_type == "compact" else extract_state_rich

        def controller(state: OptionBookMMState, history: Any = None) -> OptionBookMMAction:
            x = feat_fn(state, config)
            a = krr.predict_single(x)
            u_delta = U_r @ a
            u_full = u_baseline + u_delta
            return OptionBookMMAction(
                bid_distances=np.maximum(u_full[:n_opt], 1e-6),
                ask_distances=np.maximum(u_full[n_opt:], 1e-6),
                hedge_trade=-state.net_delta,
            )

    if return_diagnostics:
        return controller, diagnostics
    return controller
