r"""
Generic residual-kernel control: GP/KRR model for a residual objective
`y = f(z, u) + noise` around a reference action u = 0, with posterior-aware
decision rules including first-class abstention to the reference.

Design motivation
-----------------
Let `a_ref(z)` be a strong reference controller.  The executed action is
    a(z, u)  =  a_ref(z)  +  Delta a(z, u),
with u a compact low-dimensional local coordinate.  Define the residual
short-horizon improvement objective
    y(z, u)  :=  Q(z, a(z, u))  -  Q(z, a_ref(z)),
so y(z, 0) = 0 by construction.

The kernel head learns ONLY this residual on a compact overlay domain,
not the global raw-action geometry.  Proposition 5.1 of
docs/theory_kronic_extrapolation.md shows that this converts the
extrapolation breakdown into a local approximation problem on a compact
domain.

Responsibility split
--------------------
This module is response-agnostic and finance-agnostic:
    - it does not know what `y` is;
    - it does not know what the reference action is;
    - it does not know how the residual is computed.
All of that lives in the domain adapter.

Currently used by:
    - finance/experiments/merton_residual_kernel.py   (Heston)

Exact-GP posterior formulation
------------------------------
We use a single scalar ridge parameter `alpha` interpreted as the
Gaussian noise variance in the GP likelihood.  Training:
    K = k(X, X),    K_alpha = K + alpha * I,
    fit:  L  = chol(K_alpha),   dual  = K_alpha^{-1} y.
Posterior at a query X_q:
    mean     = k(X_q, X) @ dual
    var(f)   = k_diag(X_q) - sum_ij k(X_q, X_i) K_alpha^{-1}_{ij} k(X_j, X_q)
    var(y)   = var(f) + alpha      (predictive noise)

For exact reproducibility we use a Cholesky solve, not a matrix inverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np


# ===========================================================================
# Kernel
# ===========================================================================


@dataclass(frozen=True)
class RBFKernel:
    r"""Anisotropic squared-exponential kernel.

    k(x, x')  =  amplitude_sq * exp( -0.5 * sum_d ((x_d - x'_d) / length_scale_d)^2 ).

    length_scale : (d,) positive per-dimension lengths, or a single scalar.
    amplitude_sq : signal variance prior (default 1).
    """
    length_scale: np.ndarray
    amplitude_sq: float = 1.0

    def _ls_vec(self, d: int) -> np.ndarray:
        ls = np.asarray(self.length_scale, dtype=float)
        if ls.ndim == 0:
            ls = np.full(d, float(ls))
        if ls.shape[0] != d:
            raise ValueError(f"length_scale shape {ls.shape} != input dim {d}")
        return ls

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        ls = self._ls_vec(X1.shape[1])
        Z1 = X1 / ls
        Z2 = X2 / ls
        # squared distances
        a = np.sum(Z1 ** 2, axis=1, keepdims=True)
        b = np.sum(Z2 ** 2, axis=1, keepdims=True)
        sqdist = a + b.T - 2.0 * (Z1 @ Z2.T)
        sqdist = np.maximum(sqdist, 0.0)
        return self.amplitude_sq * np.exp(-0.5 * sqdist)

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self.amplitude_sq)


def median_heuristic_length_scale(X: np.ndarray) -> np.ndarray:
    r"""Per-dimension length scale = median of absolute pairwise differences.

    A standard quick-and-dirty choice.  Returns a (d,) vector.
    """
    X = np.atleast_2d(X)
    n, d = X.shape
    ls = np.ones(d)
    for j in range(d):
        diffs = np.abs(X[:, j:j+1] - X[:, j:j+1].T)
        triu = diffs[np.triu_indices(n, k=1)]
        if triu.size == 0:
            ls[j] = 1.0
        else:
            m = float(np.median(triu[triu > 0])) if np.any(triu > 0) else 1.0
            ls[j] = max(m, 1e-6)
    return ls


# ===========================================================================
# GP residual model
# ===========================================================================


@dataclass
class GPResidualModel:
    r"""Gaussian-Process residual model y = f(x) + N(0, alpha) on a compact
    joint domain x = (z, u).

    Parameters
    ----------
    kernel : RBFKernel
    alpha  : float, GP noise variance / ridge parameter.  If the residual
             target is exactly quadratic per path, alpha represents
             seed-to-seed dispersion of that quadratic.
    """
    kernel: RBFKernel
    alpha: float = 1e-3
    # Fit state (populated by .fit):
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    L_chol: Optional[np.ndarray] = None
    dual_coefs: Optional[np.ndarray] = None
    y_mean: float = 0.0
    y_scale: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, standardize: bool = True) -> "GPResidualModel":
        r"""Fit via Cholesky of K + alpha I, standardizing y for numerical
        conditioning.  `standardize=False` preserves the physical scale of
        y at the cost of some conditioning.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if standardize:
            self.y_mean = float(np.mean(y))
            self.y_scale = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
            if self.y_scale <= 0:
                self.y_scale = 1.0
        else:
            self.y_mean = 0.0
            self.y_scale = 1.0
        y_s = (y - self.y_mean) / self.y_scale

        K = self.kernel(X, X)
        K_alpha = K + self.alpha * np.eye(X.shape[0])
        # Jitter for numerical safety
        jitter = 1e-8
        while True:
            try:
                L = np.linalg.cholesky(K_alpha + jitter * np.eye(X.shape[0]))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > 1.0:
                    raise RuntimeError("GP Cholesky failed even with large jitter")
        self.L_chol = L
        self.dual_coefs = np.linalg.solve(L.T, np.linalg.solve(L, y_s))
        self.X_train = X
        self.y_train = y
        return self

    def posterior(self, X_q: np.ndarray, include_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        r"""Posterior mean and variance at query points.

        include_noise: if True, adds alpha (predictive variance of y);
                       if False, returns variance of f (epistemic).
        """
        if self.L_chol is None:
            raise RuntimeError("call fit() before posterior()")
        X_q = np.asarray(X_q, dtype=float)
        if X_q.ndim == 1:
            X_q = X_q.reshape(-1, 1)
        k_qx = self.kernel(X_q, self.X_train)
        k_qq = self.kernel.diag(X_q)
        mean_s = k_qx @ self.dual_coefs
        # var = k_qq - k_qx^T (K + alpha I)^{-1} k_qx, computed via the
        # Cholesky factor L for stability: v = L^{-1} k_qx^T ; var = k_qq - sum(v^2).
        V = np.linalg.solve(self.L_chol, k_qx.T)
        var_s = k_qq - np.sum(V ** 2, axis=0)
        var_s = np.maximum(var_s, 0.0)
        if include_noise:
            var_s = var_s + self.alpha
        # Un-standardize
        mean = self.y_mean + self.y_scale * mean_s
        var = (self.y_scale ** 2) * var_s
        return mean, var


# ===========================================================================
# Posterior decision primitives on a grid of overlay coords
# ===========================================================================


def posterior_improvement_over_baseline(
    mean: np.ndarray,
    var: np.ndarray,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""P(y > baseline | D) under Gaussian marginal at each grid point.

    Uses the approximation that the posterior is approximately Gaussian
    with the given mean and variance.  For an exact GP this is exact.
    """
    std = np.sqrt(np.maximum(var, 1e-18))
    z = (mean - baseline) / std
    return 0.5 * (1.0 + _erf_nz(z / np.sqrt(2.0)))


def expected_improvement_over_baseline(
    mean: np.ndarray,
    var: np.ndarray,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""E[max(y - baseline, 0) | D] under Gaussian marginal."""
    std = np.sqrt(np.maximum(var, 1e-18))
    z = (mean - baseline) / std
    cdf = 0.5 * (1.0 + _erf_nz(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return (mean - baseline) * cdf + std * pdf


def lower_credible_bound(
    mean: np.ndarray,
    var: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    r"""alpha-quantile of the Gaussian marginal at each grid point.

    Lower alpha -> more pessimistic (lower LCB).
    """
    std = np.sqrt(np.maximum(var, 1e-18))
    return mean + std * _ppf_nz(alpha)


def _erf_nz(z: np.ndarray) -> np.ndarray:
    r"""Hastings-style erf approximation to avoid scipy dep.  Good to ~1.5e-7."""
    # Abramowitz & Stegun 7.1.26; vectorized.
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    sign = np.sign(z)
    z = np.abs(z)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    return sign * y


def _ppf_nz(q: float) -> float:
    r"""Beasley-Springer-Moro rational approximation for the standard normal
    inverse CDF.  Avoids scipy dependency.  Good to ~1e-7 for q in [1e-6, 1-1e-6].
    """
    # Rational approximation (Wichura, Algorithm AS 241 simplified).
    p = float(q)
    if not (0.0 < p < 1.0):
        raise ValueError("q must be in (0, 1)")
    # central region
    plow = 0.02425
    phigh = 1 - plow
    if plow <= p <= phigh:
        # Peter J. Acklam's approximation
        a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        r = p - 0.5
        rsq = r * r
        num = (((((a[0] * rsq + a[1]) * rsq + a[2]) * rsq + a[3]) * rsq + a[4]) * rsq + a[5]) * r
        den = (((((b[0] * rsq + b[1]) * rsq + b[2]) * rsq + b[3]) * rsq + b[4]) * rsq + 1.0)
        return float(num / den)
    # tail
    if p < plow:
        q_t = np.sqrt(-2 * np.log(p))
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]
        return float((((((c[0] * q_t + c[1]) * q_t + c[2]) * q_t + c[3]) * q_t + c[4]) * q_t + c[5]) /
                     ((((d[0] * q_t + d[1]) * q_t + d[2]) * q_t + d[3]) * q_t + 1.0))
    # upper tail
    return -_ppf_nz(1 - p)


# ===========================================================================
# Action selection with first-class abstention
# ===========================================================================


DecisionRule = Literal[
    "reference_abstention_ei",
    "grid_expected_improvement",
    "grid_probability_of_improvement",
    "grid_lower_credible_bound",
    "always_zero",
]


@dataclass
class ActionDecision:
    action: float
    rationale: str
    recommended_idx: int
    diagnostics: Dict[str, float] = field(default_factory=dict)


def select_action_with_abstention(
    u_grid: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_var: np.ndarray,
    *,
    rule: DecisionRule = "reference_abstention_ei",
    baseline_value: float = 0.0,
    baseline_u: float = 0.0,
    poi_threshold: float = 0.8,
    ei_min_threshold: float = 0.0,
    lcb_alpha: float = 0.1,
) -> ActionDecision:
    r"""Posterior-aware action selection on a grid.

    `baseline_value` is the value of Q at u = baseline_u.  For residual
    models where y(z, 0) = 0 by construction, pass baseline_value = 0.

    Rules
    -----
    reference_abstention_ei (default):
        u_EI = argmax EI on grid.  If P(y(u_EI) > baseline) < poi_threshold,
        return u = baseline_u (abstain).  Otherwise return u_EI.

    grid_expected_improvement:
        argmax EI, no abstention logic.

    grid_probability_of_improvement:
        argmax PoI, no abstention.

    grid_lower_credible_bound:
        argmax (lower credible bound) -- risk-averse choice.

    always_zero:
        always return baseline_u.
    """
    u_grid = np.asarray(u_grid)
    mean = np.asarray(posterior_mean)
    var = np.asarray(posterior_var)

    ei = expected_improvement_over_baseline(mean, var, baseline=baseline_value)
    poi = posterior_improvement_over_baseline(mean, var, baseline=baseline_value)
    lcb = lower_credible_bound(mean, var, alpha=lcb_alpha) - baseline_value

    diagnostics = {
        "max_ei": float(np.max(ei)),
        "max_poi": float(np.max(poi)),
        "max_lcb": float(np.max(lcb)),
    }

    if rule == "always_zero":
        idx = int(np.argmin(np.abs(u_grid - baseline_u)))
        return ActionDecision(
            action=float(baseline_u),
            rationale="rule=always_zero",
            recommended_idx=idx,
            diagnostics=diagnostics,
        )

    if rule == "grid_expected_improvement":
        idx = int(np.argmax(ei))
        return ActionDecision(
            action=float(u_grid[idx]),
            rationale=f"argmax EI: u={u_grid[idx]:+.4f}, EI={ei[idx]:.4e}",
            recommended_idx=idx,
            diagnostics=diagnostics,
        )

    if rule == "grid_probability_of_improvement":
        idx = int(np.argmax(poi))
        return ActionDecision(
            action=float(u_grid[idx]),
            rationale=f"argmax PoI: u={u_grid[idx]:+.4f}, PoI={poi[idx]:.3f}",
            recommended_idx=idx,
            diagnostics=diagnostics,
        )

    if rule == "grid_lower_credible_bound":
        idx = int(np.argmax(lcb))
        return ActionDecision(
            action=float(u_grid[idx]),
            rationale=f"argmax LCB: u={u_grid[idx]:+.4f}, LCB={lcb[idx]:+.4e}",
            recommended_idx=idx,
            diagnostics=diagnostics,
        )

    if rule == "reference_abstention_ei":
        idx_ei = int(np.argmax(ei))
        ei_best = float(ei[idx_ei])
        poi_at_best = float(poi[idx_ei])
        if ei_best <= ei_min_threshold:
            idx_ref = int(np.argmin(np.abs(u_grid - baseline_u)))
            return ActionDecision(
                action=float(baseline_u),
                rationale=(
                    f"abstain: max EI = {ei_best:.3e} <= threshold = {ei_min_threshold:.3e}"
                ),
                recommended_idx=idx_ref,
                diagnostics=diagnostics,
            )
        if poi_at_best < poi_threshold:
            idx_ref = int(np.argmin(np.abs(u_grid - baseline_u)))
            return ActionDecision(
                action=float(baseline_u),
                rationale=(
                    f"abstain at u={baseline_u}: argmax-EI at u={u_grid[idx_ei]:+.4f} has "
                    f"P(y>baseline)={poi_at_best:.3f} < threshold {poi_threshold:.2f}"
                ),
                recommended_idx=idx_ref,
                diagnostics=diagnostics,
            )
        return ActionDecision(
            action=float(u_grid[idx_ei]),
            rationale=(
                f"u_EI={u_grid[idx_ei]:+.4f}: "
                f"P(y>baseline|D)={poi_at_best:.3f} >= {poi_threshold:.2f}"
            ),
            recommended_idx=idx_ei,
            diagnostics=diagnostics,
        )

    raise ValueError(f"Unknown rule: {rule}")


# ===========================================================================
# Posterior sampling at query points (independent-per-point marginal)
# ===========================================================================


def sample_marginal_posterior(
    gp: "GPResidualModel",
    X_q: np.ndarray,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
    include_noise: bool = False,
) -> np.ndarray:
    r"""Independent-per-point marginal posterior samples from a GP.

    Returns an (n_samples, n_query) array.  For per-query decision rules
    on a grid (e.g. EI/PoI/LCB per grid point) this is the right object;
    for joint-function uncertainty across query points you would need the
    joint posterior (not implemented here -- trivial to add with the
    Cholesky factor of the joint covariance).

    This helper is response-agnostic: the generic module does NOT know
    whether the GP output is a Kelly residual, a CRRA score, or anything
    else.  Domain adapters combine samples from multiple GPs (e.g.,
    mean + second-moment) into domain-specific scores.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    mean, var = gp.posterior(X_q, include_noise=include_noise)
    std = np.sqrt(np.maximum(var, 1e-18))
    eps = rng.standard_normal((n_samples, mean.size))
    return mean[None, :] + eps * std[None, :]


# ===========================================================================
# Helper: extrapolation flag on a grid
# ===========================================================================


def extrapolation_risk_per_grid_point(
    X_grid: np.ndarray,
    X_train: np.ndarray,
    length_scale: np.ndarray,
) -> np.ndarray:
    r"""Return, per grid point, the distance (in kernel-scaled units) to the
    nearest training point.  Large values flag extrapolation.
    """
    X_grid = np.atleast_2d(X_grid)
    X_train = np.atleast_2d(X_train)
    ls = np.asarray(length_scale, dtype=float)
    if ls.ndim == 0:
        ls = np.full(X_grid.shape[1], float(ls))
    D = ((X_grid[:, None, :] - X_train[None, :, :]) / ls) ** 2
    sqdist = D.sum(axis=-1)
    return np.sqrt(np.min(sqdist, axis=1))
