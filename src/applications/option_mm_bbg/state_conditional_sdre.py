"""State-conditional reduced SDRE controller, exploration-only training.

Stage A of the de-cheated Level-2 line. Differences from `sdre_recovery`:

  - The reduced-space coefficients (rev_lin_r, rev_quad_diag_r, vega_r)
    are predicted from an 11-D hand-summary state feature vector via
    Bayesian linear regression, rather than held as a single global scalar.
  - No BBG actions, BBG coordinates, or BBG solver enter training. BBG is
    evaluation-only.
  - The reduced action basis U_r is reused from BilinearControlModel
    (exploration-only), exactly as in the original SDRE recovery.

The head predicts control primitives (reduced linear revenue slope, diagonal
reduced revenue curvature, reduced vega channel); the local quadratic control
law is assembled inside the controller.

Usage:
    cfg = StateConditionalConfig(rank=3, n_explore_episodes=300)
    ctrl, head = make_state_conditional_controller(
        bbg_config, cfg, return_model=True,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import linalg

from .env import OptionBookMMAction, OptionBookMMState, OptionBookMarketMakingEnv
from .pricing import bs_call_vega_sqrt_nu
from .sdre_recovery import (
    BilinearControlModel,
    ExplorationData,
    SDRERecoveryConfig,
    _compute_rn_distances,
    collect_exploration_data,
    extract_state_rich,
)
from .spec import BBGBenchmarkConfig


Controller = Callable[[OptionBookMMState, Any], OptionBookMMAction]


# ---------------------------------------------------------------------------
# Extended state features (11-D, generic summaries only)
# ---------------------------------------------------------------------------


def extract_state_rich_extended(
    state: OptionBookMMState,
    config: BBGBenchmarkConfig,
) -> np.ndarray:
    """11-D generic state summary, no BBG coordinates.

    Extends `extract_state_rich` (7-D) with four global inventory/wealth
    summaries that are cheap to compute and do not reference the BBG solver:

        7:  log_wealth_norm     = log(W / W_0)         wealth scale
        8:  inv_l1              = (sum |q_i|) / Q_ref  total inventory mag
        9:  inv_tilt_moneyness  = sum q_i * log(K_i/S) / Q_ref
        10: inv_tilt_maturity   = sum q_i * T_i / (Q_ref * mean_T)

    Q_ref is the configured inventory scale; it keeps all features O(1).
    """
    base = extract_state_rich(state, config)

    h = config.heston
    wealth = max(state.wealth, 1e-6)
    log_wealth_norm = float(np.log(wealth / h.spot0))

    options = config.book.options
    q = state.option_inventories
    # Reference inventory unit: one notional-sized trade at spot0.
    # Keeps inventory features O(1) without coupling to vega_limit (already
    # used in vpi_norm) or to any BBG-derived quantity.
    inv_scale = max(
        config.liquidity.notional_per_trade / max(h.spot0, 1e-6),
        1.0,
    )

    abs_q_sum = float(np.sum(np.abs(q)))
    inv_l1 = abs_q_sum / inv_scale

    log_mny = np.array([np.log(opt.strike / state.spot) for opt in options])
    mats = np.array([opt.maturity for opt in options])
    mean_T = max(float(np.mean(mats)), 1e-6)

    inv_tilt_mny = float(np.sum(q * log_mny)) / inv_scale
    inv_tilt_mat = float(np.sum(q * mats)) / (inv_scale * mean_T)

    extra = np.array([log_wealth_norm, inv_l1, inv_tilt_mny, inv_tilt_mat])
    return np.concatenate([base, extra])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateConditionalConfig:
    """Configuration for the state-conditional Level-2 controller.

    Exploration and basis learning settings reuse the original SDRE recovery
    machinery (log-normal noise on RN baseline). The `rank` here is the
    reduced action dimension used by both the basis and the head.
    """

    n_explore_episodes: int = 300
    explore_noise_std: float = 0.3
    explore_seed: int = 42

    rank: int = 3
    basis_method: str = "bilinear_2stage"  # reuses sdre_recovery methods
    bilinear_overspace: int = 10
    ridge_alpha_basis: float = 1e-3

    # Head regularization (Bayesian ridge)
    head_alpha_spread: float = 1e-2
    head_alpha_vega: float = 1e-2
    head_feature_map: str = "linear"  # "linear" (identity+intercept)

    # Concavity floor: rev_quad_diag is clipped to <= -floor * ref_scale,
    # where ref_scale = |min(global bilinear rev_quad)| computed at fit time.
    # A data-scale floor prevents the head from predicting near-zero
    # curvature, which would explode the QP.
    concavity_floor_rel: float = 1e-2

    # Trust region on u_delta (fraction of baseline). Kept tight for
    # Stage A: the head's per-step targets are noisy; large perturbations
    # off the exploration distribution are risky.
    max_pert_frac: float = 0.3


# ---------------------------------------------------------------------------
# Bayesian ridge regression (closed-form)
# ---------------------------------------------------------------------------


class BayesianRidge:
    """Ridge regression with posterior variance from the standard Bayesian
    interpretation.

    Fit: MAP under likelihood N(y | X beta, sigma^2 I) with prior
    beta ~ N(0, (1/alpha) I).  The ridge penalty is applied at the
    FIXED ``alpha`` (not rescaled by sigma^2) — keeping the estimator
    identical to classical ridge with penalty alpha.

    mu_post    = (X^T X + alpha I)^{-1} X^T y
    sigma2_hat = ||y - X mu||^2 / (n - eff_df)
    Sigma_post = sigma2_hat * (X^T X + alpha I)^{-1}

    eff_df = trace(X (X^T X + alpha I)^{-1} X^T), the usual ridge DoF.
    """

    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self.mu: np.ndarray | None = None
        self.Sigma: np.ndarray | None = None
        self.sigma2: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianRidge":
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape
        XtX = X.T @ X
        Xty = X.T @ y
        A = XtX + self.alpha * np.eye(d)
        A_inv = linalg.solve(A, np.eye(d), assume_a="pos")
        mu = A_inv @ Xty
        resid = y - X @ mu
        # Effective degrees of freedom for ridge
        H_diag = np.einsum("ij,jk,ik->i", X, A_inv, X)
        eff_df = float(np.sum(H_diag))
        denom = max(n - eff_df, 1.0)
        sigma2 = float(np.sum(resid ** 2) / denom)
        sigma2 = max(sigma2, 1e-12)
        self.mu = mu
        self.Sigma = sigma2 * A_inv
        self.sigma2 = sigma2
        return self

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return X @ self.mu

    def predict_mean_var(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = X @ self.mu
        # Var per row = sigma^2 + x^T Sigma x
        quad = np.einsum("ij,jk,ik->i", X, self.Sigma, X)
        var = self.sigma2 + np.maximum(quad, 0.0)
        return mean, var

    def r2(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = X @ self.mu
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot < 1e-30:
            return 0.0
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# State-conditional head
# ---------------------------------------------------------------------------


@dataclass
class HeadPrediction:
    rev_lin_r: np.ndarray       # (r,)
    rev_quad_diag_r: np.ndarray # (r,) — clipped negative
    vega_r: np.ndarray          # (r,)


class StateConditionalHead:
    """Predicts reduced-space control primitives as a function of state.

    Parameterization (linear feature map, phi(z) = [1, z_std]):
        rev_lin_r(z)        = W_lin.T @ phi(z)        shape (r,)
        rev_quad_diag_r(z)  = W_quad.T @ phi(z)       shape (r,)
        vega_r(z)           = W_vega.T @ phi(z)       shape (r,)

    Fit strategy: one Bayesian ridge per regression problem, with feature
    expansion [phi(z); phi(z) ⊗ a; phi(z) ⊗ a^2] for spread and
    [phi(z); phi(z) ⊗ a] for Δv_pi. Weights for the different z × a blocks
    are extracted directly from the posterior mean.
    """

    def __init__(self, config: StateConditionalConfig, rank: int, d_state: int):
        self.config = config
        self.rank = rank
        self.d_state = d_state
        self.feature_dim = d_state + 1  # [1, z_std_1, ..., z_std_d]

        self.z_mean: np.ndarray | None = None
        self.z_scale: np.ndarray | None = None
        self.a_scale: np.ndarray | None = None

        self.W_lin: np.ndarray | None = None     # (F, r)
        self.W_quad: np.ndarray | None = None    # (F, r)
        self.W_vega: np.ndarray | None = None    # (F, r)

        self.spread_model: BayesianRidge | None = None
        self.vega_model: BayesianRidge | None = None

        # Fit diagnostics
        self.spread_r2: float = 0.0
        self.vega_r2: float = 0.0
        self.spread_r2_test: float | None = None
        self.vega_r2_test: float | None = None

        # Absolute concavity floor (set from global bilinear at fit time).
        # rev_quad_diag_r(z) is clipped to <= -concavity_floor_abs.
        self.concavity_floor_abs: float = 1e-10

    # -- feature map ------------------------------------------------------

    def _phi(self, z_std: np.ndarray) -> np.ndarray:
        """Linear feature map: [1, z_std]. z_std: (N, d_state) or (d_state,)."""
        if z_std.ndim == 1:
            return np.concatenate([[1.0], z_std])
        n = z_std.shape[0]
        return np.column_stack([np.ones(n), z_std])

    def _standardize_z(self, Z: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.z_mean = Z.mean(axis=0)
            self.z_scale = np.maximum(Z.std(axis=0), 1e-6)
        return (Z - self.z_mean) / self.z_scale

    # -- spread feature expansion ----------------------------------------

    def _expand_spread(self, Phi: np.ndarray, A: np.ndarray) -> np.ndarray:
        """[phi; phi ⊗ a; phi ⊗ a^2] — shape (N, F*(1+2r))."""
        n, F = Phi.shape
        r = A.shape[1]
        phi_a = (Phi[:, :, None] * A[:, None, :]).reshape(n, F * r)
        phi_a2 = (Phi[:, :, None] * (A[:, None, :] ** 2)).reshape(n, F * r)
        return np.concatenate([Phi, phi_a, phi_a2], axis=1)

    def _expand_vega(self, Phi: np.ndarray, A: np.ndarray) -> np.ndarray:
        """[phi; phi ⊗ a] — shape (N, F*(1+r))."""
        n, F = Phi.shape
        r = A.shape[1]
        phi_a = (Phi[:, :, None] * A[:, None, :]).reshape(n, F * r)
        return np.concatenate([Phi, phi_a], axis=1)

    # -- fit --------------------------------------------------------------

    def fit(
        self,
        Z: np.ndarray,
        A: np.ndarray,
        spread: np.ndarray,
        dvpi: np.ndarray,
        test_idx: np.ndarray | None = None,
    ) -> None:
        """Fit two Bayesian ridges, extract (W_lin, W_quad, W_vega).

        Z: (N, d_state) raw state features.
        A: (N, r) reduced-space action perturbations.
        spread: (N,) realized per-step spread capture.
        dvpi:  (N,) realized per-step vpi change.
        test_idx: optional held-out mask indices for R^2 reporting.
        """
        assert A.shape[1] == self.rank
        assert Z.shape[1] == self.d_state

        Z_std = self._standardize_z(Z, fit=True)
        Phi = self._phi(Z_std)   # (N, F)
        F = Phi.shape[1]
        r = self.rank

        # Action scaling to keep feature magnitudes balanced
        self.a_scale = np.maximum(A.std(axis=0), 1e-6)
        A_std = A / self.a_scale

        train_mask = np.ones(Z.shape[0], dtype=bool)
        if test_idx is not None and len(test_idx) > 0:
            train_mask[test_idx] = False

        # --- Spread model: s ~ phi + phi⊗a + phi⊗a^2 -----------------
        X_spr = self._expand_spread(Phi, A_std)
        self.spread_model = BayesianRidge(alpha=self.config.head_alpha_spread)
        self.spread_model.fit(X_spr[train_mask], spread[train_mask])
        self.spread_r2 = self.spread_model.r2(X_spr[train_mask], spread[train_mask])
        if test_idx is not None and test_idx.size > 0:
            self.spread_r2_test = self.spread_model.r2(
                X_spr[~train_mask], spread[~train_mask]
            )

        theta_spr = self.spread_model.mu  # shape (F + F*r + F*r,)
        # Slice out blocks
        # block 0: phi only (F,) -> intercept in state, unused by controller
        # block 1: phi⊗a (F*r,)
        # block 2: phi⊗a^2 (F*r,)
        W_lin_std = theta_spr[F : F + F * r].reshape(F, r)       # in a_std units
        W_quad_std = theta_spr[F + F * r : F + 2 * F * r].reshape(F, r)

        # Unscale: spread = sum_j rev_lin_r_j * a_j + 0.5 * rev_quad_diag_r_j * a_j^2
        # where a_j = A_std_j * a_scale_j. The BLR fit for phi⊗a_std gives
        # coeff_std[j] = rev_lin_r_j * a_scale_j. And phi⊗a_std^2 gives
        # coeff_std2[j] = 0.5 * rev_quad_diag_r_j * a_scale_j^2.
        # Hence:
        self.W_lin = W_lin_std / self.a_scale[None, :]
        self.W_quad = 2.0 * W_quad_std / (self.a_scale[None, :] ** 2)

        # --- Vega model: Δv ~ phi + phi⊗a ---------------------------
        X_veg = self._expand_vega(Phi, A_std)
        self.vega_model = BayesianRidge(alpha=self.config.head_alpha_vega)
        self.vega_model.fit(X_veg[train_mask], dvpi[train_mask])
        self.vega_r2 = self.vega_model.r2(X_veg[train_mask], dvpi[train_mask])
        if test_idx is not None and test_idx.size > 0:
            self.vega_r2_test = self.vega_model.r2(
                X_veg[~train_mask], dvpi[~train_mask]
            )

        theta_veg = self.vega_model.mu  # shape (F + F*r,)
        W_vega_std = theta_veg[F : F + F * r].reshape(F, r)
        # Δv = sum_j vega_r_j * a_j, a_j = A_std_j * a_scale_j
        # coeff_std[j] = vega_r_j * a_scale_j  =>  vega_r_j = coeff_std[j] / a_scale_j
        self.W_vega = W_vega_std / self.a_scale[None, :]

    # -- predict ----------------------------------------------------------

    def predict(self, z: np.ndarray) -> HeadPrediction:
        """Predict reduced coefficients at a single state z ∈ R^{d_state}."""
        z_std = (z - self.z_mean) / self.z_scale
        phi = self._phi(z_std)  # (F,)

        rev_lin_r = phi @ self.W_lin
        rev_quad_diag_r = phi @ self.W_quad
        vega_r = phi @ self.W_vega

        # Concavity floor: rev_quad_diag must be <= -floor
        rev_quad_diag_r = np.minimum(rev_quad_diag_r, -self.concavity_floor_abs)

        return HeadPrediction(
            rev_lin_r=rev_lin_r,
            rev_quad_diag_r=rev_quad_diag_r,
            vega_r=vega_r,
        )

    def predict_variance(self, z: np.ndarray) -> dict:
        """Posterior predictive variance at z (scalar per regression).

        Returns the expansion-point predictive variance for the spread model
        at perturbation a = 0, and for the vega model at a = 0. These serve
        as a coarse state-level uncertainty summary (not per-coefficient).
        """
        z_std = (z - self.z_mean) / self.z_scale
        phi = self._phi(z_std)[None, :]  # (1, F)
        # Spread at a=0: X = [phi; 0; 0]
        F = phi.shape[1]
        r = self.rank
        X_spr = np.concatenate(
            [phi, np.zeros((1, F * r)), np.zeros((1, F * r))], axis=1
        )
        _, var_spr = self.spread_model.predict_mean_var(X_spr)
        X_veg = np.concatenate([phi, np.zeros((1, F * r))], axis=1)
        _, var_veg = self.vega_model.predict_mean_var(X_veg)
        return {"spread_var": float(var_spr[0]), "vega_var": float(var_veg[0])}


# ---------------------------------------------------------------------------
# Reduced SDRE solve with state-conditional coefficients
# ---------------------------------------------------------------------------


def _sdre_solve_state_conditional(
    vpi: float,
    pred: HeadPrediction,
    gamma: float,
    xi: float,
    dt: float,
    rel_reg: float = 1e-4,
) -> np.ndarray:
    """Solve argmax g_r^T a - 0.5 a^T Q_r a in reduced coordinates.

    Q_r = -2 diag(rev_quad_diag_r) + 2 c_pen vega_r vega_r^T (positive def.)
    g_r = rev_lin_r - 2 c_pen V^pi vega_r
    a*  = Q_r^{-1} g_r.

    Regularization is RELATIVE to the Q_r scale so it works across
    problem magnitudes.
    """
    r = pred.rev_lin_r.shape[0]
    c_pen = gamma * xi ** 2 / 8.0 * dt

    g_r = pred.rev_lin_r - 2.0 * c_pen * vpi * pred.vega_r
    Q_r = -2.0 * np.diag(pred.rev_quad_diag_r) + 2.0 * c_pen * np.outer(
        pred.vega_r, pred.vega_r
    )
    scale = float(np.max(np.abs(np.diag(Q_r))))
    Q_r = Q_r + rel_reg * max(scale, 1e-12) * np.eye(r)

    try:
        a_star = linalg.solve(Q_r, g_r, assume_a="pos")
    except linalg.LinAlgError:
        a_star = np.zeros(r)
    return a_star


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


def make_state_conditional_controller(
    config: BBGBenchmarkConfig,
    sc_config: StateConditionalConfig,
    rn_distances: np.ndarray | None = None,
    held_out_frac: float = 0.2,
    return_model: bool = False,
) -> Controller | tuple[Controller, dict]:
    """Build the de-cheated Level-2 controller.

    Pipeline (all BBG-free):
      1. Compute RN baseline distances (competitive MM spread, no HJB).
      2. Run exploration episodes with log-normal noise (reuses sdre_recovery).
      3. Fit BilinearControlModel + reduce to rank r. Yields U_r, u_center.
      4. Extract extended per-step state features z_i.
      5. Project per-step actions onto U_r:  a_i = U_r^T (u_i - u_center).
      6. Fit StateConditionalHead on (z, a, spread, dvpi) with hold-out.
      7. Return a closure that reads z, predicts (g_r, Q_r), solves QP.

    BBG appears only as the simulation env.
    """
    h = config.heston
    n_opt = config.book.n_options

    if rn_distances is None:
        rn_distances = _compute_rn_distances(config)

    # Step 2: exploration (reuse existing loop)
    sdre_like = SDRERecoveryConfig(
        n_explore_episodes=sc_config.n_explore_episodes,
        explore_noise_std=sc_config.explore_noise_std,
        explore_seed=sc_config.explore_seed,
        rank=sc_config.rank,
        method=sc_config.basis_method,
        bilinear_overspace=sc_config.bilinear_overspace,
        ridge_alpha=sc_config.ridge_alpha_basis,
    )
    data: ExplorationData = collect_exploration_data(
        config, rn_distances, sdre_like
    )

    # Step 3: fit bilinear + reduce (same methods as sdre_recovery)
    bilinear = BilinearControlModel(config, sc_config.ridge_alpha_basis)
    bilinear.fit(data)

    env_dt = config.control.horizon / 30

    if sc_config.basis_method == "bilinear":
        bilinear.reduce(sc_config.rank)
        U_r = bilinear.U_r
    elif sc_config.basis_method == "bilinear_2stage":
        overspace = min(sc_config.bilinear_overspace, 2 * n_opt)
        bilinear.reduce(overspace)
        U_over = bilinear.U_r
        c_pen = config.control.gamma * h.xi ** 2 / 8.0 * env_dt
        vc_proj = U_over.T @ bilinear.vega_channel
        rq_proj = U_over.T @ np.diag(bilinear.rev_quad) @ U_over
        H_over = rq_proj - c_pen * np.outer(vc_proj, vc_proj)
        eigvals, eigvecs = np.linalg.eigh(H_over)
        idx = np.argsort(eigvals)
        k = min(sc_config.rank, len(eigvals))
        V_inner = eigvecs[:, idx[:k]]
        U_final = U_over @ V_inner
        norms = np.maximum(np.linalg.norm(U_final, axis=0, keepdims=True), 1e-15)
        U_r = U_final / norms
    else:
        raise ValueError(f"unknown basis_method: {sc_config.basis_method}")

    u_center = bilinear.u_center

    # Step 4-5: build training matrices for the head
    # Extend state features per step by re-simulating the SAME exploration
    # rollouts: the per-step extended features are not stored by
    # collect_exploration_data, so we replay with identical seed/policy.
    Z_ext, A_red, spread_vec, dvpi_vec = _collect_extended_features(
        config, rn_distances, sc_config, U_r, u_center,
    )

    # Step 6: fit head with hold-out
    d_state = Z_ext.shape[1]
    rng = np.random.default_rng(sc_config.explore_seed + 7)
    n_total = Z_ext.shape[0]
    n_test = int(held_out_frac * n_total)
    test_idx = rng.choice(n_total, size=n_test, replace=False)

    head = StateConditionalHead(sc_config, rank=sc_config.rank, d_state=d_state)
    # Data-scale concavity floor: take the GLOBAL reduced revenue curvature
    # (diag of U_r^T diag(rev_quad) U_r), and floor the head's predicted
    # rev_quad_diag_r magnitude at a small fraction of that scale.
    global_rq_r = np.diag(U_r.T @ np.diag(bilinear.rev_quad) @ U_r)
    ref_scale = float(np.max(np.abs(global_rq_r)))
    head.concavity_floor_abs = max(
        sc_config.concavity_floor_rel * ref_scale, 1e-10
    )
    head.fit(Z_ext, A_red, spread_vec, dvpi_vec, test_idx=test_idx)

    # Step 7: controller closure
    u_baseline = u_center.copy()
    gamma = config.control.gamma
    xi = h.xi
    max_pert_frac = sc_config.max_pert_frac

    def controller(
        state: OptionBookMMState, history: Any = None
    ) -> OptionBookMMAction:
        z = extract_state_rich_extended(state, config)
        pred = head.predict(z)
        a_star = _sdre_solve_state_conditional(
            vpi=state.portfolio_vega,
            pred=pred,
            gamma=gamma, xi=xi, dt=env_dt,
        )
        u_delta = U_r @ a_star

        # Trust region: u_delta clipped to +/- max_pert_frac * baseline
        cap = max_pert_frac * np.abs(u_baseline)
        u_delta = np.clip(u_delta, -cap, cap)
        u_full = u_baseline + u_delta
        u_full = np.maximum(u_full, 1e-6)

        bid_dists = u_full[:n_opt]
        ask_dists = u_full[n_opt:]
        return OptionBookMMAction(
            bid_distances=bid_dists,
            ask_distances=ask_dists,
            hedge_trade=-state.net_delta,
        )

    if return_model:
        diagnostics = {
            "head": head,
            "U_r": U_r,
            "u_center": u_center,
            "bilinear": bilinear,
            "n_transitions": n_total,
            "n_test": n_test,
        }
        return controller, diagnostics
    return controller


# ---------------------------------------------------------------------------
# Replay to collect extended per-step features (Z, A_reduced, s, dvpi)
# ---------------------------------------------------------------------------


def _collect_extended_features(
    config: BBGBenchmarkConfig,
    rn_distances: np.ndarray,
    sc_config: StateConditionalConfig,
    U_r: np.ndarray,
    u_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the same exploration rollouts but record extended state
    features and per-step reduced actions.

    The RNG sequence is identical to `collect_exploration_data` so the
    action trajectories match exactly; this gives per-step z vectors
    aligned with the bilinear fit.
    """
    n_opt = config.book.n_options
    sigma = sc_config.explore_noise_std
    rng = np.random.default_rng(sc_config.explore_seed)

    Z_list, A_list, S_list, DV_list = [], [], [], []

    for ep in range(sc_config.n_explore_episodes):
        env = OptionBookMarketMakingEnv(config, seed=ep)
        state = env.reset()

        while not state.done:
            z = extract_state_rich_extended(state, config)

            bid_dists = rn_distances * np.exp(rng.normal(0, sigma, n_opt))
            ask_dists = rn_distances * np.exp(rng.normal(0, sigma, n_opt))
            bid_dists = np.maximum(bid_dists, 1e-6)
            ask_dists = np.maximum(ask_dists, 1e-6)

            vpi_pre = state.portfolio_vega
            action = OptionBookMMAction(
                bid_distances=bid_dists,
                ask_distances=ask_dists,
                hedge_trade=-state.net_delta,
            )
            u = np.concatenate([bid_dists, ask_dists])
            a_red = U_r.T @ (u - u_center)

            next_state, _, _, info = env.step(action)

            Z_list.append(z)
            A_list.append(a_red)
            S_list.append(info["spread_capture"])
            DV_list.append(next_state.portfolio_vega - vpi_pre)

            state = next_state

    return (
        np.array(Z_list),
        np.array(A_list),
        np.array(S_list),
        np.array(DV_list),
    )
