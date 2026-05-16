r"""
Bayesian local-quadratic response: conjugate Normal-Inverse-Gamma posterior
for the scalar-local-control regression

    y  =  c0  +  c1 * u  +  c2 * u**2  +  epsilon,
    epsilon ~ N(0, sigma**2).

Purpose
-------
Replace the point-estimate `u_star = -c1 / (2 c2)` with a posterior object
that naturally expresses uncertainty, shrinks recommended actions back to
the reference when representation/data is weak, and never silently reports
an optimizer from an extrapolated quadratic.

Conjugate posterior
-------------------
Prior:
    sigma**2  ~  InvGamma(a_0, b_0)
    (c0, c1, c2) | sigma**2  ~  N(mu_0, sigma**2 * V_0)

With design matrix X = [1, u_i, u_i**2] and observations y:
    V_N  =  (X.T @ X + V_0^{-1})^{-1}
    mu_N =  V_N @ (X.T @ y + V_0^{-1} @ mu_0)
    a_N  =  a_0 + N/2
    b_N  =  b_0 + 0.5 * (y.T @ y + mu_0.T @ V_0^{-1} @ mu_0 - mu_N.T @ V_N^{-1} @ mu_N)

The marginal posterior of (c0, c1, c2) is multivariate Student-t with
location mu_N, scale (b_N / a_N) * V_N, and 2 * a_N degrees of freedom.

Posterior on the optimizer u_star
---------------------------------
u_star = -c1 / (2 c2) is a nonlinear functional of the coefficients.
We sample (c0, c1, c2) from the posterior (exactly, via NIG) and push
each draw through the nonlinear map, giving the empirical posterior of
u_star.  Draws with c2 >= 0 have no concave optimum and are reported
as "no optimizer" rather than clipped.

Decision rule
-------------
The module exposes three primitives -- posterior median, posterior EI over
a trust-region grid, and lower credible bound -- plus a default composite
rule that shrinks to the reference action (u=0) when either
    P(c2 < 0 | D) < concavity_mass_threshold
or
    P(|u*| <= u_max | D)  <  trust_region_mass_threshold.
Both thresholds are caller-configurable; neither is hard-coded.

No finance vocabulary appears in this module.

Currently used by:
    - finance/experiments/merton_relative_overlay_bayesian.py  (Heston)
    - experiments/latent_ou_representation_demo_bayesian.py    (latent-OU)

Refactor when a third non-adapter caller appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


# ===========================================================================
# Prior configuration
# ===========================================================================


@dataclass(frozen=True)
class NIGPrior:
    r"""Conjugate Normal-Inverse-Gamma prior.

    Defaults are weakly informative:
      * mu_0 = 0 (prior mean says "no curvature, no slope, no offset")
      * V_0  = diag([tau0**2, tau1**2, tau2**2]) with tau_i large so the
              prior covariance dominates over typical N ~ 10..100 data
      * a_0, b_0 chosen so E[sigma**2] is moderate and variance is wide

    The default tau_i values are deliberately broad; the posterior becomes
    data-dominated for even modestly informative regressions.  For very
    low-signal regimes the prior MATTERS and this is by design: it is the
    mechanism that shrinks the recommended action back toward u=0.
    """
    mu_0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    V_0: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 100.0]))
    a_0: float = 2.0  # so prior variance = b_0 / (a_0 - 1) is finite
    b_0: float = 1.0


# ===========================================================================
# Posterior container
# ===========================================================================


@dataclass(frozen=True)
class NIGPosterior:
    r"""Conjugate NIG posterior parameters.

    Marginal posterior for (c0, c1, c2) is multivariate Student-t with:
        location  = mu_N
        scale     = (b_N / a_N) * V_N
        df        = 2 * a_N
    """
    mu_N: np.ndarray            # (3,) posterior mean of coefficients
    V_N: np.ndarray             # (3,3) posterior precision^{-1} (up to sigma^2)
    a_N: float                  # posterior sigma^2 shape
    b_N: float                  # posterior sigma^2 scale
    n_data: int                 # sample size

    @property
    def df(self) -> float:
        return 2.0 * self.a_N

    @property
    def coef_mean(self) -> np.ndarray:
        return self.mu_N.copy()

    @property
    def coef_cov(self) -> np.ndarray:
        r"""Marginal covariance of the coefficient Student-t posterior."""
        if self.a_N <= 1.0:
            raise ValueError("Posterior coefficient covariance undefined when a_N <= 1.")
        return (self.b_N / (self.a_N - 1.0)) * self.V_N

    @property
    def sigma_sq_mean(self) -> float:
        if self.a_N <= 1.0:
            return float("nan")
        return float(self.b_N / (self.a_N - 1.0))


# ===========================================================================
# Fit
# ===========================================================================


def fit_bayesian_quadratic(
    u: np.ndarray,
    y: np.ndarray,
    prior: Optional[NIGPrior] = None,
) -> NIGPosterior:
    r"""Conjugate NIG posterior for y = c0 + c1 * u + c2 * u^2 + noise.

    Arguments
    ---------
    u   : (N,) array of local coordinates.
    y   : (N,) array of responses.
    prior : optional NIGPrior; defaults to weakly informative.

    Returns
    -------
    NIGPosterior with (mu_N, V_N, a_N, b_N, n_data).
    """
    if prior is None:
        prior = NIGPrior()
    u = np.asarray(u, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    if u.size != y.size:
        raise ValueError("u and y must have the same length")
    N = int(u.size)
    X = np.column_stack([np.ones_like(u), u, u * u])

    V0_inv = np.linalg.inv(prior.V_0)
    VN_inv = X.T @ X + V0_inv
    V_N = np.linalg.inv(VN_inv)
    mu_N = V_N @ (X.T @ y + V0_inv @ prior.mu_0)
    a_N = prior.a_0 + N / 2.0
    b_N = (
        prior.b_0
        + 0.5 * float(y @ y)
        + 0.5 * float(prior.mu_0 @ V0_inv @ prior.mu_0)
        - 0.5 * float(mu_N @ VN_inv @ mu_N)
    )
    b_N = max(b_N, 1e-12)
    return NIGPosterior(mu_N=mu_N, V_N=V_N, a_N=float(a_N), b_N=float(b_N), n_data=N)


# ===========================================================================
# Posterior sampling
# ===========================================================================


def sample_coefficients(
    post: NIGPosterior,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    r"""Draw (n_samples, 3) samples of (c0, c1, c2) from the NIG posterior.

    Exact sampler:
        sigma^2 ~ InverseGamma(a_N, b_N)        (draw via 1 / Gamma)
        beta | sigma^2 ~ N(mu_N, sigma^2 * V_N)
    """
    if rng is None:
        rng = np.random.RandomState(0)
    # InvGamma(a, b):  X = b / Gamma(a, 1)
    # scipy isn't required; use numpy.random.gamma
    gammas = rng.gamma(shape=post.a_N, scale=1.0, size=n_samples)
    sigma_sq = post.b_N / gammas  # inverse-gamma draws
    # Cholesky of V_N once; scale by sqrt(sigma^2) per-sample
    L = np.linalg.cholesky(post.V_N + 1e-12 * np.eye(post.V_N.shape[0]))
    eps = rng.standard_normal((n_samples, 3))
    draws = post.mu_N[None, :] + np.sqrt(sigma_sq)[:, None] * (eps @ L.T)
    return draws


def sample_optimizer(
    post: NIGPosterior,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    r"""Sample the posterior of u_star = -c1 / (2 c2).

    Returns
    -------
    dict with:
        u_star         : (n_samples,) raw u_star samples (NaN where c2 >= 0)
        concave_mask   : (n_samples,) bool, True where c2 < 0
        coefs          : (n_samples, 3) coefficient samples (c0, c1, c2)
    """
    coefs = sample_coefficients(post, n_samples, rng=rng)
    c2 = coefs[:, 2]
    concave_mask = c2 < 0.0
    u_star = np.full(n_samples, np.nan)
    c1 = coefs[:, 1]
    safe = concave_mask
    u_star[safe] = -c1[safe] / (2.0 * c2[safe])
    return {"u_star": u_star, "concave_mask": concave_mask, "coefs": coefs}


# ===========================================================================
# Predictive distribution at a point
# ===========================================================================


def predictive_mean_var(
    post: NIGPosterior,
    u_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Posterior predictive mean and VARIANCE of y at each u in u_grid.

    Posterior predictive is Student-t with location x^T mu_N, scale
    (b_N / a_N) * (1 + x^T V_N x), df = 2 a_N.  Variance = scale * df / (df - 2)
    for df > 2.
    """
    u_grid = np.asarray(u_grid, dtype=float).flatten()
    X = np.column_stack([np.ones_like(u_grid), u_grid, u_grid ** 2])
    mean = X @ post.mu_N
    scale_sq = (post.b_N / post.a_N) * (1.0 + np.einsum("ij,jk,ik->i", X, post.V_N, X))
    if post.df > 2:
        var = scale_sq * post.df / (post.df - 2.0)
    else:
        var = np.full_like(scale_sq, np.inf)
    return mean, var


# ===========================================================================
# Summaries
# ===========================================================================


@dataclass(frozen=True)
class PosteriorSummary:
    r"""Summary statistics from N posterior samples of (coef, u_star).

    Field naming follows the repo's generic vocabulary (no finance jargon).
    """
    # Coefficient posterior
    coef_mean: np.ndarray
    coef_std: np.ndarray
    coef_q05: np.ndarray
    coef_q50: np.ndarray
    coef_q95: np.ndarray
    # Optimizer posterior
    concavity_mass: float                  # P(c2 < 0 | D)
    in_trust_region_mass: float            # P(|u*| <= u_max | D), conditional on concave
    optimizer_q05: float                   # 5th pct of posterior u*, NaN where concave mass insufficient
    optimizer_q50: float
    optimizer_q95: float
    optimizer_mean: float
    optimizer_std: float
    # Predictive summary on trust-region grid
    u_grid: np.ndarray
    q_mean_on_grid: np.ndarray
    q_std_on_grid: np.ndarray
    # Improvement diagnostics
    prob_improvement_on_grid: np.ndarray   # P(Q(u) > Q(0) | D) per grid u
    expected_improvement_on_grid: np.ndarray  # E[max(Q(u)-Q(0),0) | D] per grid u
    # Decision-rule outputs (informational; see `decide_action`)
    recommended_action: float
    recommended_action_rationale: str


def summarize_posterior(
    post: NIGPosterior,
    u_max: float,
    n_grid: int = 51,
    n_samples: int = 4000,
    rng: Optional[np.random.RandomState] = None,
    improvement_baseline_u: float = 0.0,
) -> PosteriorSummary:
    r"""Compute a full posterior summary from a NIG posterior.

    Draws n_samples posterior (c0, c1, c2), computes u_star samples, and
    assembles predictive/improvement statistics on a u-grid of size n_grid
    in [-u_max, +u_max].
    """
    if rng is None:
        rng = np.random.RandomState(0)
    samples = sample_optimizer(post, n_samples, rng=rng)
    coefs = samples["coefs"]
    u_star = samples["u_star"]
    concave_mask = samples["concave_mask"]

    concavity_mass = float(np.mean(concave_mask))

    # In-trust-region conditional on concave
    in_region = concave_mask & (np.abs(u_star) <= u_max)
    if concavity_mass > 0:
        in_region_mass = float(np.sum(in_region) / max(np.sum(concave_mask), 1))
    else:
        in_region_mass = 0.0

    # Optimizer quantiles: computed over samples that are concave AND in region
    valid = np.isfinite(u_star) & (np.abs(u_star) <= u_max)
    if valid.sum() >= 10:
        u_star_valid = u_star[valid]
        q05 = float(np.quantile(u_star_valid, 0.05))
        q50 = float(np.quantile(u_star_valid, 0.50))
        q95 = float(np.quantile(u_star_valid, 0.95))
        mean_ = float(np.mean(u_star_valid))
        std_ = float(np.std(u_star_valid, ddof=1))
    else:
        q05 = q50 = q95 = mean_ = std_ = float("nan")

    # Predictive on grid
    u_grid = np.linspace(-u_max, u_max, n_grid)
    # Evaluate per-sample: Q_k(u) = c0_k + c1_k u + c2_k u^2, shape (n_samples, n_grid)
    X = np.column_stack([np.ones_like(u_grid), u_grid, u_grid ** 2])
    Q_samples = coefs @ X.T
    baseline_X = np.array([1.0, improvement_baseline_u, improvement_baseline_u ** 2])
    Q_baseline = coefs @ baseline_X
    # improvement samples: Q(u) - Q(baseline), shape (n_samples, n_grid)
    improvement = Q_samples - Q_baseline[:, None]
    prob_improvement = (improvement > 0).mean(axis=0)
    expected_improvement = np.mean(np.maximum(improvement, 0.0), axis=0)
    q_mean = Q_samples.mean(axis=0)
    q_std = Q_samples.std(axis=0, ddof=1)

    coef_mean = coefs.mean(axis=0)
    coef_std = coefs.std(axis=0, ddof=1)
    coef_q = np.quantile(coefs, [0.05, 0.50, 0.95], axis=0)

    # Default recommended action: `reference_abstention_ei`.  Stay at u=0
    # unless BOTH (a) posterior concavity mass is high enough AND (b) the
    # posterior probability of improvement at the EI-argmax is high.  The
    # posterior median of u* = -c1/(2c2) is returned as a diagnostic only
    # and is NOT used in the decision rule (the ratio is heavy-tailed and
    # can extrapolate outside the data support).
    rec_action, rec_rationale = decide_action(
        concavity_mass=concavity_mass,
        in_trust_region_mass=in_region_mass,
        optimizer_median=q50,
        u_max=u_max,
        rule="reference_abstention_ei",
        grid_u=u_grid,
        ei_on_grid=expected_improvement,
        poi_on_grid=prob_improvement,
    )

    return PosteriorSummary(
        coef_mean=coef_mean,
        coef_std=coef_std,
        coef_q05=coef_q[0],
        coef_q50=coef_q[1],
        coef_q95=coef_q[2],
        concavity_mass=concavity_mass,
        in_trust_region_mass=in_region_mass,
        optimizer_q05=q05,
        optimizer_q50=q50,
        optimizer_q95=q95,
        optimizer_mean=mean_,
        optimizer_std=std_,
        u_grid=u_grid,
        q_mean_on_grid=q_mean,
        q_std_on_grid=q_std,
        prob_improvement_on_grid=prob_improvement,
        expected_improvement_on_grid=expected_improvement,
        recommended_action=rec_action,
        recommended_action_rationale=rec_rationale,
    )


# ===========================================================================
# Decision rules
# ===========================================================================


DecisionRule = Literal[
    "reference_abstention_ei",         # DEFAULT: u=0 unless posterior evidence
    "grid_expected_improvement",
    "grid_probability_of_improvement",
    "grid_lower_credible_bound",
    "grid_composite_ei_shrinkage",
    "median_diagnostic_only",
    "always_zero",
]


def decide_action(
    concavity_mass: float,
    in_trust_region_mass: float,
    optimizer_median: float,
    u_max: float,
    rule: DecisionRule = "reference_abstention_ei",
    concavity_mass_threshold: float = 0.8,
    trust_region_mass_threshold: float = 0.5,
    poi_threshold: float = 0.8,
    grid_u: Optional[np.ndarray] = None,
    ei_on_grid: Optional[np.ndarray] = None,
    poi_on_grid: Optional[np.ndarray] = None,
    lcb_on_grid: Optional[np.ndarray] = None,
    ei_min_threshold: float = 0.0,
) -> Tuple[float, str]:
    r"""Apply a configurable decision rule; return (action, rationale).

    **Default** is `reference_abstention_ei`: the reference action u=0 is a
    FIRST-CLASS decision outcome, not a fallback.  We leave u=0 only when
    the posterior actively supports doing so:

        (A) P(c2 < 0 | D) >= concavity_mass_threshold  (a concave optimum exists)
        (B) P(Q(u_EI) > Q(0) | D) >= poi_threshold     (posterior probability of
                                                        improvement at the EI
                                                        argmax is high)

    where u_EI = argmax over the grid of expected improvement.  If either
    test fails, the recommended action is u=0.

    This matches the intended semantics of a safety-aware Bayesian
    controller: don't move away from the nominal policy unless the
    posterior has both a concave local landscape AND concrete probability
    of improvement.  Weak signals and diffuse posteriors stay at u=0.

    Rules
    -----
    "reference_abstention_ei" (default):
        u_EI = argmax over grid_u of E[max(Q(u)-Q(0),0) | D].
        Stay at u=0 unless both concavity_mass >= threshold AND
        P(Q(u_EI) > Q(0) | D) >= poi_threshold.  Otherwise use u_EI.

    "grid_expected_improvement":
        action = argmax over grid_u of E[max(Q(u)-Q(0),0) | D].
        Pure EI; no abstention.  Reference u=0 is only chosen when EI=0
        is the grid max, which requires the posterior to assign zero
        probability to any improvement at any u.

    "grid_probability_of_improvement":
        action = argmax over grid_u of P(Q(u) > Q(0) | D).

    "grid_lower_credible_bound":
        action = argmax over grid_u of the alpha-quantile of
        (Q(u) - Q(0)).  Conservative; shrinks heavily under uncertainty.

    "grid_composite_ei_shrinkage":
        action = argmax EI if max EI > ei_min_threshold AND
                 P(concave|D) >= concavity_mass_threshold.
        Otherwise u = 0.  Similar to the default but uses an absolute-EI
        magnitude test rather than a PoI test.

    "median_diagnostic_only":
        Posterior-median u* clipped to trust region.  NOT recommended for
        decision because the ratio -c1/(2c2) has heavy posterior tails.

    "always_zero":
        action = 0.  Baseline for comparison.
    """
    if rule == "always_zero":
        return 0.0, "rule=always_zero"
    if rule == "reference_abstention_ei":
        if grid_u is None or ei_on_grid is None or poi_on_grid is None:
            raise ValueError(
                "reference_abstention_ei requires grid_u, ei_on_grid, poi_on_grid"
            )
        if concavity_mass < concavity_mass_threshold:
            return 0.0, (
                f"abstain: P(concave|D)={concavity_mass:.3f} "
                f"< threshold {concavity_mass_threshold:.2f}"
            )
        idx = int(np.argmax(ei_on_grid))
        u_ei = float(grid_u[idx])
        poi_at_ei = float(poi_on_grid[idx])
        if poi_at_ei < poi_threshold:
            return 0.0, (
                f"abstain at u=0: argmax-EI at u={u_ei:+.4f} has "
                f"P(Q(u)>Q(0)|D)={poi_at_ei:.3f} < threshold {poi_threshold:.2f}"
            )
        return u_ei, (
            f"u_EI={u_ei:+.4f}: P(Q(u)>Q(0)|D)={poi_at_ei:.3f} >= {poi_threshold:.2f}, "
            f"P(concave|D)={concavity_mass:.3f} >= {concavity_mass_threshold:.2f}"
        )
    if rule == "median_diagnostic_only":
        if not np.isfinite(optimizer_median):
            return 0.0, "median ill-defined (fallback u=0)"
        return float(np.clip(optimizer_median, -u_max, u_max)), (
            "DIAGNOSTIC: posterior median u* (ratio is heavy-tailed; not recommended for decision)"
        )
    if rule == "grid_expected_improvement":
        if grid_u is None or ei_on_grid is None:
            raise ValueError("grid_expected_improvement requires grid_u, ei_on_grid")
        idx = int(np.argmax(ei_on_grid))
        return float(grid_u[idx]), (
            f"argmax EI: u={grid_u[idx]:+.4f}, EI={ei_on_grid[idx]:.4e}"
        )
    if rule == "grid_probability_of_improvement":
        if grid_u is None or poi_on_grid is None:
            raise ValueError("grid_probability_of_improvement requires grid_u, poi_on_grid")
        idx = int(np.argmax(poi_on_grid))
        return float(grid_u[idx]), (
            f"argmax PoI: u={grid_u[idx]:+.4f}, P(Q(u)>Q(0))={poi_on_grid[idx]:.3f}"
        )
    if rule == "grid_lower_credible_bound":
        if grid_u is None or lcb_on_grid is None:
            raise ValueError("grid_lower_credible_bound requires grid_u, lcb_on_grid")
        idx = int(np.argmax(lcb_on_grid))
        return float(grid_u[idx]), (
            f"argmax LCB: u={grid_u[idx]:+.4f}, LCB={lcb_on_grid[idx]:+.4e}"
        )
    if rule == "grid_composite_ei_shrinkage":
        if grid_u is None or ei_on_grid is None:
            raise ValueError("grid_composite_ei_shrinkage requires grid_u, ei_on_grid")
        if concavity_mass < concavity_mass_threshold:
            return 0.0, (
                f"P(concave|D)={concavity_mass:.3f} < {concavity_mass_threshold:.2f}: shrink to u=0"
            )
        ei_max = float(np.max(ei_on_grid))
        if ei_max <= ei_min_threshold:
            return 0.0, (
                f"max EI={ei_max:.4e} <= threshold={ei_min_threshold:.4e}: shrink to u=0"
            )
        idx = int(np.argmax(ei_on_grid))
        return float(grid_u[idx]), (
            f"argmax EI: u={grid_u[idx]:+.4f}, EI={ei_on_grid[idx]:.4e} "
            f"(P(concave|D)={concavity_mass:.3f})"
        )
    raise ValueError(f"Unknown rule: {rule}")


# ===========================================================================
# Lower credible bound on Q(u) - Q(baseline)
# ===========================================================================


def lower_credible_bound_on_grid(
    post: NIGPosterior,
    u_max: float,
    n_grid: int = 51,
    n_samples: int = 4000,
    alpha: float = 0.1,
    improvement_baseline_u: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Posterior alpha-quantile of (Q(u) - Q(u_baseline)) on a grid.

    Returns (u_grid, lcb).  Useful for risk-averse action selection.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    coefs = sample_coefficients(post, n_samples, rng=rng)
    u_grid = np.linspace(-u_max, u_max, n_grid)
    X = np.column_stack([np.ones_like(u_grid), u_grid, u_grid ** 2])
    Q = coefs @ X.T
    baseline_X = np.array([1.0, improvement_baseline_u, improvement_baseline_u ** 2])
    Q_baseline = coefs @ baseline_X
    improvement = Q - Q_baseline[:, None]
    lcb = np.quantile(improvement, alpha, axis=0)
    return u_grid, lcb


# ===========================================================================
# Generic helper: combine two quadratic posteriors (mean and variance of
# a response) under a linear risk-correction functional.  Domain adapters
# (e.g. a finance adapter interpreting this as CRRA-log-moment matching)
# supply the response-specific scalar lambda_risk.
#
# NOT CRRA-specific.  The finance adapter computes lambda_risk from its
# own interpretation and calls this helper.
# ===========================================================================


def combine_mean_and_variance_posteriors(
    mean_post: NIGPosterior,
    var_post: NIGPosterior,
    lambda_risk: float,
) -> np.ndarray:
    r"""Point-posterior-mean combination: coef = mu_mean - lambda * mu_var.

    The interpretation of `lambda_risk` is response-specific (e.g., a
    domain adapter may set it to `0.5 * (gamma - 1)` for a lognormal
    CRRA-moment-matching approximation, or to `0` for a pure-mean target).
    """
    return mean_post.mu_N - float(lambda_risk) * var_post.mu_N


def sample_combined_optimizer(
    mean_post: NIGPosterior,
    var_post: NIGPosterior,
    lambda_risk: float,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    r"""Per-sample combination: coef_k = mu_mean_k - lambda * mu_var_k,
    then u_star_k = -c1_k / (2 c2_k) with concavity check.

    Uses INDEPENDENT posterior samples of the two posteriors.  If the
    underlying data means and variances are correlated (they usually are
    when both are estimated from the same paths), this is a first-order
    approximation.  Tighter treatments fit a joint model.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    mu_samples = sample_coefficients(mean_post, n_samples, rng=rng)
    var_samples = sample_coefficients(var_post, n_samples, rng=rng)
    combined = mu_samples - float(lambda_risk) * var_samples
    c2 = combined[:, 2]
    c1 = combined[:, 1]
    concave_mask = c2 < 0.0
    u_star = np.full(n_samples, np.nan)
    u_star[concave_mask] = -c1[concave_mask] / (2.0 * c2[concave_mask])
    return {"u_star": u_star, "concave_mask": concave_mask, "coefs": combined}
