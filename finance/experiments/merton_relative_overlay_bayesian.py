r"""
Approach 1 Bayesian benchmark: Heston/CRRA multiplicative overlay with a
posterior over the local optimizer u*.

What changed from merton_relative_overlay_benchmark.py
------------------------------------------------------
1. RESPONSE VARIABLE changed from U(W_T) = W_T^(1-gamma)/(1-gamma) to
   log W_T.  Under the multiplicative constant-u overlay
       pi_t = (1+u) * pi_myopic(V_t),
   per-path log-wealth is EXACTLY quadratic in u (derivation in the
   docstring of the Heston estimator below).  This replaces polynomially
   heavy CRRA tails (which overflowed at H = 1260) with lognormal tails
   on a sum of bounded quadratic paths.

2. POSTERIOR OBJECT.  We fit a conjugate Normal-Inverse-Gamma Bayesian
   posterior on [1, u, u^2] pooled across seeds, using
   src/control/bayesian_local_quadratic.py.  Report posterior mean, 90%
   credible interval, and concavity / trust-region masses.

3. CRRA INTERPRETATION.  The raw posterior target is the Kelly
   (log-utility) optimum u*_Kelly = -c1 / (2 c2) of expected log-wealth.
   To compare against the gamma-specific CRRA theory target
   u* = rho * xi * p_gamma / (mu - r), we ALSO fit a second Bayesian
   regression to the per-u sample-variance of log W_T and combine via
   the lognormal approximation

       CRRA_score(u)  =  mu(u)  -  0.5 * (gamma - 1) * var(u),

   with both moments posterior-sampled and combined per sample.

Benchmark target interpretation (explicit, not silent)
------------------------------------------------------
- Stationary Heston-CRRA theory: u*_CRRA = rho * xi * p_gamma / (mu-r).
  p_gamma depends on gamma (see merton_theory.py).  V-independent.
- Stationary Heston-LOG theory:  u*_Kelly = rho * xi * p_1 / (mu-r).
  p_1 is the same ODE's root with gamma -> 1 substituted in. Different
  exponent; different target; we report it for reference.
- The BAYESIAN output gives posterior summaries for both.  They are
  NUMERICALLY DIFFERENT; the user should expect them to differ by the
  gamma-correction term, not agree.

No finance vocabulary leaks into src/control/bayesian_local_quadratic.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv
from merton_theory import stationary_heston_crra_theory

from src.control.bayesian_local_quadratic import (
    NIGPrior,
    NIGPosterior,
    fit_bayesian_quadratic,
    sample_coefficients,
    sample_optimizer,
    sample_combined_optimizer,
    summarize_posterior,
)


# ==========================================================================
# Finance-specific: CRRA-log-moment approximation (NOT in src/control)
# ==========================================================================
#
# IMPORTANT INTERPRETATION NOTE (please read before reading u_CRRA below)
# -----------------------------------------------------------------------
# The regression target in this script is `log W_T`, not CRRA utility
# `W_T^{1-gamma} / (1-gamma)`.  This is a deliberate numerical stability
# choice:
#
#   1. Under constant-u multiplicative overlay, log W_T is EXACTLY
#      quadratic in u per path (no exponentials).  CRRA utility has
#      polynomial tails that overflowed at H=1260.
#
#   2. The posterior on log W_T gives the Kelly / log-utility optimum,
#      u*_Kelly = argmax_u E[log W_T | u].  This is NOT the exact CRRA
#      gamma optimum.
#
#   3. Under LOGNORMAL W_T, the CRRA-gamma optimum is the argmax of
#
#          mu(u) - 0.5 * (gamma - 1) * sigma^2(u)
#
#      where mu(u) = E[log W_T | u], sigma^2(u) = Var[log W_T | u].
#      Both are (approximately) quadratic in u, so the CRRA-corrected
#      optimum can be computed from two separate Bayesian quadratic fits.
#
#   4. This lognormal-CRRA mapping is an APPROXIMATION.  It would be exact
#      under lognormal wealth, which holds asymptotically for the
#      constant-u overlay but not exactly.  Report u_CRRA outputs as
#      "CRRA-approximation under lognormal", never as "exact gamma=3
#      CRRA recovery."


def crra_lambda_risk(gamma: float) -> float:
    r"""Lognormal-moment-matching risk coefficient: 0.5 * (gamma - 1).

    Combined coefficient vector for `mu(u) - 0.5*(gamma-1)*var(u)`:
        combined = mu_coefs - 0.5 * (gamma - 1) * var_coefs.

    For gamma = 1 (log utility): combined = mu_coefs (Kelly).
    For gamma > 1: a risk penalty shifts the optimum toward less risky
    actions.  Interpretation is approximate (see header comment).
    """
    return 0.5 * (float(gamma) - 1.0)


# ==========================================================================
# Rollout: log W_T under multiplicative overlay, per seed per u
# ==========================================================================


@dataclass(frozen=True)
class HestonBayesianConfig:
    rho: float
    gamma: float
    V0: float
    H: int
    dt: float
    u_max: float
    n_grid: int
    n_seeds: int
    n_posterior_samples: int = 8000


def _paired_noise(rho: float, H: int, seed: int) -> np.ndarray:
    r"""Correlated (z1, z2) with Heston asset-vs-variance correlation."""
    rng = np.random.RandomState(seed)
    zA = rng.standard_normal(H)
    zB = rng.standard_normal(H)
    z1 = zA
    z2 = rho * zA + np.sqrt(max(1.0 - rho * rho, 0.0)) * zB
    return np.column_stack([z1, z2])


def _logW_T_per_u(
    env: HestonMertonEnv,
    V0: float,
    H: int,
    dt: float,
    u_grid: np.ndarray,
    noise: np.ndarray,
    V_floor_for_pi_ref: float = 0.005,
) -> np.ndarray:
    r"""For ONE paired-noise seed, return log W_T evaluated at each u in u_grid.

    Multiplicative overlay:
        pi_t          = (1+u) * pi_myopic(V_t_floored)
        pi_myopic(V)  = (mu - r) / (gamma * V)

    `V_floor_for_pi_ref` caps the leverage implied by pi_ref at

        pi_cap  =  (mu-r) / (gamma * V_floor)

    without modifying env V dynamics.  This is a CONTROLLER-LEVEL safety
    measure: the env V path evolves normally, but the reference policy
    treats V < V_floor as if V == V_floor.  Without this floor, transient
    CIR excursions toward V = 0 in the discretization give transient
    leverage of 1e6 and corrupt the response variable across u.

    The interpretation is unchanged: the reference policy IS still a
    V-adaptive myopic controller; we only remove its unrealistic
    divergence at numerically small V.
    """
    logW_at_u = np.zeros(len(u_grid))
    for k, u in enumerate(u_grid):
        logW = 0.0
        V = float(V0)
        for t in range(H):
            V_for_ref = max(V, V_floor_for_pi_ref)
            pi_ref = env.merton_optimal(V_for_ref)
            pi_t = (1.0 + float(u)) * pi_ref
            logW, V = env.step_explicit(
                logW, V, pi_t,
                float(noise[t, 0]), float(noise[t, 1]), dt,
            )
        logW_at_u[k] = logW
    return logW_at_u


def collect_log_wealth_data(
    env: HestonMertonEnv,
    cfg: HestonBayesianConfig,
    base_seed: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Returns (u_pooled, logW_pooled, logW_matrix).

    logW_matrix is (n_seeds, n_grid): one row per seed.  Useful for
    computing per-u variance of log W_T across seeds.

    CRN invariant: within each seed `k`, the SAME correlated noise matrix
    drives every u in u_grid.  The posterior therefore reflects only the
    inter-seed response variation, not Monte Carlo noise that could have
    been paired away within a seed.
    """
    u_grid = np.linspace(-cfg.u_max, cfg.u_max, cfg.n_grid)
    logW_matrix = np.zeros((cfg.n_seeds, cfg.n_grid))
    for k in range(cfg.n_seeds):
        noise = _paired_noise(cfg.rho, cfg.H, base_seed + k)  # ONCE per seed
        # Same `noise` used for every u in u_grid -> CRN across the grid
        logW_matrix[k] = _logW_T_per_u(env, cfg.V0, cfg.H, cfg.dt, u_grid, noise)
    u_pooled = np.tile(u_grid, cfg.n_seeds)
    logW_pooled = logW_matrix.flatten()
    return u_pooled, logW_pooled, logW_matrix


# ==========================================================================
# Theoretical targets (both Kelly and gamma-CRRA)
# ==========================================================================


def theory_targets(env: HestonMertonEnv, V0: float) -> Dict[str, float]:
    theory_gamma = stationary_heston_crra_theory(env, V0)
    p_gamma = float(theory_gamma.exponent_p)
    a = float(env.mu - env.r)
    u_CRRA = env.rho * env.xi * p_gamma / a if abs(a) > 1e-12 else 0.0

    # Kelly target: same HJB root but with gamma = 1 (log utility).
    # Construct a surrogate env with gamma=1 to reuse the theory solver.
    kelly_env = HestonMertonEnv(
        mu=env.mu, r=env.r, gamma=1.0,
        kappa=env.kappa, theta=env.theta, xi=env.xi, rho=env.rho,
    )
    theory_kelly = stationary_heston_crra_theory(kelly_env, V0)
    p_kelly = float(theory_kelly.exponent_p)
    u_Kelly = env.rho * env.xi * p_kelly / a if abs(a) > 1e-12 else 0.0

    return {
        "p_gamma": p_gamma,
        "p_kelly": p_kelly,
        "u_star_CRRA": float(u_CRRA),
        "u_star_Kelly": float(u_Kelly),
    }


# ==========================================================================
# Two-moment CRRA combination: fit mean(log W) AND var(log W) separately
# ==========================================================================


def fit_mean_and_variance_posteriors(
    u_pooled: np.ndarray,
    logW_matrix: np.ndarray,
    prior_mean: NIGPrior,
    prior_var: NIGPrior,
) -> Tuple[object, object, np.ndarray]:
    r"""Returns (mean_post, var_post, per_u_variance_estimates).

    - mean_post: NIGPosterior of [1, u, u^2] on pooled log W_T.
    - var_post:  NIGPosterior of [1, u, u^2] on per-u SAMPLE variance of log W_T.
    - per_u_variance_estimates: (n_grid,) of sample variance of log W_T at each u.
    """
    n_seeds, n_grid = logW_matrix.shape
    u_grid = u_pooled[:n_grid]
    # Mean posterior: pooled regression
    mean_post = fit_bayesian_quadratic(u_pooled, logW_matrix.flatten(), prior=prior_mean)
    # Per-u sample variance of log W_T (one value per u)
    per_u_mean = logW_matrix.mean(axis=0)
    per_u_var = logW_matrix.var(axis=0, ddof=1)
    # Fit variance posterior on (u_grid, per_u_var)
    var_post = fit_bayesian_quadratic(u_grid, per_u_var, prior=prior_var)
    return mean_post, var_post, per_u_var


# ==========================================================================
# Reporting
# ==========================================================================


def _format_ci(q05, q50, q95):
    return f"{q50:+.4f}  [{q05:+.4f}, {q95:+.4f}]"


def run_single_cell(
    env: HestonMertonEnv,
    cfg: HestonBayesianConfig,
    label: str = "",
    print_output: bool = True,
) -> Dict:
    u_pooled, logW_pooled, logW_matrix = collect_log_wealth_data(env, cfg)

    prior_mean = NIGPrior(
        mu_0=np.zeros(3),
        V_0=np.diag([10.0, 10.0, 10.0]),
        a_0=2.0, b_0=1.0,
    )
    prior_var = NIGPrior(
        mu_0=np.zeros(3),
        V_0=np.diag([10.0, 10.0, 10.0]),
        a_0=2.0, b_0=1.0,
    )
    mean_post, var_post, per_u_var = fit_mean_and_variance_posteriors(
        u_pooled, logW_matrix, prior_mean, prior_var,
    )

    # Kelly (log-utility) posterior from mean_post directly.
    # summarize_posterior uses grid_expected_improvement by default for the
    # recommended action; posterior median u* is a DIAGNOSTIC only.
    kelly_summary = summarize_posterior(
        mean_post, u_max=cfg.u_max,
        n_grid=cfg.n_grid, n_samples=cfg.n_posterior_samples,
        rng=np.random.RandomState(42),
    )
    # CRRA-LOGNORMAL-APPROXIMATION posterior combining mean and variance.
    # NOT exact gamma-CRRA recovery; this is the lognormal closed-form
    # mapping through `log E[W^{1-gamma}] = (1-gamma) mu + 0.5 (1-gamma)^2 sigma^2`,
    # which is first-order correct under lognormal W_T and approximate otherwise.
    crra_samples = sample_combined_optimizer(
        mean_post, var_post,
        lambda_risk=crra_lambda_risk(env.gamma),
        n_samples=cfg.n_posterior_samples,
        rng=np.random.RandomState(43),
    )
    u_star_crra = crra_samples["u_star"]
    conc_crra = crra_samples["concave_mask"]
    valid_crra = np.isfinite(u_star_crra) & (np.abs(u_star_crra) <= cfg.u_max)
    if valid_crra.sum() >= 10:
        crra_q50 = float(np.quantile(u_star_crra[valid_crra], 0.5))
        crra_q05 = float(np.quantile(u_star_crra[valid_crra], 0.05))
        crra_q95 = float(np.quantile(u_star_crra[valid_crra], 0.95))
    else:
        crra_q50 = crra_q05 = crra_q95 = float("nan")
    crra_concavity_mass = float(np.mean(conc_crra))
    crra_in_tr = float(
        np.sum(np.abs(u_star_crra[conc_crra]) <= cfg.u_max) / max(conc_crra.sum(), 1)
    )

    targets = theory_targets(env, cfg.V0)
    # Sign correctness posterior mass: P(sign(u*) == sign(theory))
    theory_CRRA = targets["u_star_CRRA"]
    theory_Kelly = targets["u_star_Kelly"]
    if abs(theory_CRRA) > 1e-10 and valid_crra.sum() > 10:
        psign_crra = float(
            np.mean(np.sign(u_star_crra[valid_crra]) == np.sign(theory_CRRA))
        )
    else:
        psign_crra = float("nan")

    kelly_u = kelly_summary
    # Posterior improvement mass at the theoretical target u
    # P(Q(u_target) > Q(0) | D)   for both targets, via kelly posterior
    def p_improve_at(target_u: float) -> float:
        if not np.isfinite(target_u) or abs(target_u) > cfg.u_max:
            return float("nan")
        coefs = sample_coefficients(
            mean_post, n_samples=cfg.n_posterior_samples,
            rng=np.random.RandomState(7),
        )
        x = np.array([1.0, target_u, target_u ** 2])
        x0 = np.array([1.0, 0.0, 0.0])
        improvement = coefs @ (x - x0)
        return float(np.mean(improvement > 0))

    p_imp_at_theory_kelly = p_improve_at(theory_Kelly)
    p_imp_at_theory_crra = p_improve_at(theory_CRRA)

    result = {
        "label": label,
        "config": cfg,
        "env_rho": env.rho,
        "env_gamma": env.gamma,
        "theory_targets": targets,
        # Kelly
        "kelly_u_star_q50": kelly_u.optimizer_q50,
        "kelly_u_star_q05": kelly_u.optimizer_q05,
        "kelly_u_star_q95": kelly_u.optimizer_q95,
        "kelly_concavity_mass": kelly_u.concavity_mass,
        "kelly_in_tr_mass": kelly_u.in_trust_region_mass,
        "kelly_recommended": kelly_u.recommended_action,
        "kelly_rationale": kelly_u.recommended_action_rationale,
        # CRRA combined
        "crra_u_star_q50": crra_q50,
        "crra_u_star_q05": crra_q05,
        "crra_u_star_q95": crra_q95,
        "crra_concavity_mass": crra_concavity_mass,
        "crra_in_tr_mass": crra_in_tr,
        "crra_psign_match": psign_crra,
        # Coefficients
        "mean_coef_mean": mean_post.mu_N,
        "var_coef_mean": var_post.mu_N,
        # P(improvement at theory u)
        "p_improvement_at_theory_kelly": p_imp_at_theory_kelly,
        "p_improvement_at_theory_crra": p_imp_at_theory_crra,
        # Expose the posterior summary for plotting
        "kelly_summary": kelly_u,
        "mean_post": mean_post,
        "var_post": var_post,
    }

    if print_output:
        print(f"[{label}]  rho={env.rho:+.2f}  gamma={env.gamma}  "
              f"V0={cfg.V0}  H={cfg.H}  n_seeds={cfg.n_seeds}  u_max={cfg.u_max}")
        print(f"  Response variable: log W_T (NOT CRRA utility).  "
              f"Kelly target is exact; CRRA target is lognormal-approx.")
        print(f"  Theory: u*_Kelly (exact for log-utility) = {theory_Kelly:+.4f}")
        print(f"  Theory: u*_CRRA(gamma={env.gamma}, approx via lognormal) "
              f"= {theory_CRRA:+.4f}")
        print(f"  mean(log W_T) coefficient posterior mean: "
              f"c0={mean_post.mu_N[0]:+.4f}  c1={mean_post.mu_N[1]:+.4f}  "
              f"c2={mean_post.mu_N[2]:+.4f}")
        print(f"  var(log W_T)  coefficient posterior mean: "
              f"c0={var_post.mu_N[0]:+.4f}  c1={var_post.mu_N[1]:+.4f}  "
              f"c2={var_post.mu_N[2]:+.4f}")
        print(f"  KELLY (log-utility) posterior:")
        print(f"     RECOMMENDED ACTION (grid-EI rule) = {kelly_u.recommended_action:+.4f}")
        print(f"       rationale: {kelly_u.recommended_action_rationale}")
        print(f"     diagnostic posterior median u* = "
              f"{kelly_u.optimizer_q50:+.4f}  "
              f"(90% CI [{kelly_u.optimizer_q05:+.4f}, {kelly_u.optimizer_q95:+.4f}])")
        print(f"     diagnostic P(concave|D) = {kelly_u.concavity_mass:.3f}   "
              f"P(u_ratio in TR|D) = {kelly_u.in_trust_region_mass:.3f}")
        print(f"     P(Q(u_Kelly_theory) > Q(0) | D) = {p_imp_at_theory_kelly:.3f}   "
              f"(structural alignment with Kelly theory)")
        print(f"  CRRA (lognormal-approx, gamma={env.gamma}) posterior:")
        print(f"     diagnostic posterior median u* = "
              f"{_format_ci(crra_q05, crra_q50, crra_q95)}")
        print(f"     diagnostic P(concave|D) = {crra_concavity_mass:.3f}   "
              f"P(u_ratio in TR|D) = {crra_in_tr:.3f}")
        if np.isfinite(psign_crra):
            print(f"     P(sign(u*_CRRA) = sign(theory_CRRA)|D) = {psign_crra:.3f}")
        print(f"     P(Q(u_CRRA_theory) > Q(0) | D) = {p_imp_at_theory_crra:.3f}   "
              f"(structural alignment with CRRA theory under lognormal approx)")
    return result


# ==========================================================================
# Sweeps
# ==========================================================================


def sweep_rho(
    rhos=(-0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.7),
    gamma=3.0,
    V0=0.04,
    H_list=(126, 1260),
    dt=1.0 / 252.0,
    u_max=0.3,
    n_grid=21,
    n_seeds=30,
) -> List[Dict]:
    rows = []
    for H in H_list:
        print()
        print("#" * 100)
        print(f"# RHO sweep at H = {H}")
        print("#" * 100)
        for rho in rhos:
            env = HestonMertonEnv(rho=rho, gamma=gamma)
            cfg = HestonBayesianConfig(
                rho=rho, gamma=gamma, V0=V0,
                H=H, dt=dt, u_max=u_max, n_grid=n_grid, n_seeds=n_seeds,
            )
            row = run_single_cell(env, cfg, label=f"rho={rho:+.1f},H={H}")
            rows.append(row)
    return rows


def sweep_V(
    V_list=(0.02, 0.03, 0.04, 0.06, 0.08),
    rho=-0.7, gamma=3.0, H=1260,
    dt=1.0 / 252.0, u_max=0.3, n_grid=21, n_seeds=30,
) -> List[Dict]:
    rows = []
    print()
    print("#" * 100)
    print(f"# V sweep at rho={rho}, H={H}")
    print("#" * 100)
    env = HestonMertonEnv(rho=rho, gamma=gamma)
    for V0 in V_list:
        cfg = HestonBayesianConfig(
            rho=rho, gamma=gamma, V0=V0,
            H=H, dt=dt, u_max=u_max, n_grid=n_grid, n_seeds=n_seeds,
        )
        row = run_single_cell(env, cfg, label=f"V0={V0:.3f}")
        rows.append(row)
    return rows


# ==========================================================================
# Plot
# ==========================================================================


def _plot_rho_sweep(rows: List[Dict], out_path: str, V0: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_H = {}
    for r in rows:
        by_H.setdefault(r["config"].H, []).append(r)

    fig, axes = plt.subplots(1, len(by_H), figsize=(6 * len(by_H), 5), sharey=False)
    if len(by_H) == 1:
        axes = [axes]

    for ax, (H, rs) in zip(axes, sorted(by_H.items())):
        rhos = [r["env_rho"] for r in rs]
        q50_k = [r["kelly_u_star_q50"] for r in rs]
        q05_k = [r["kelly_u_star_q05"] for r in rs]
        q95_k = [r["kelly_u_star_q95"] for r in rs]
        q50_c = [r["crra_u_star_q50"] for r in rs]
        q05_c = [r["crra_u_star_q05"] for r in rs]
        q95_c = [r["crra_u_star_q95"] for r in rs]
        th_k = [r["theory_targets"]["u_star_Kelly"] for r in rs]
        th_c = [r["theory_targets"]["u_star_CRRA"] for r in rs]

        rhos = np.array(rhos)
        q05_k, q50_k, q95_k = map(np.array, (q05_k, q50_k, q95_k))
        q05_c, q50_c, q95_c = map(np.array, (q05_c, q50_c, q95_c))
        th_k, th_c = map(np.array, (th_k, th_c))

        ax.plot(rhos, th_k, "k--", lw=1.2, label="theory u*_Kelly")
        ax.plot(rhos, th_c, "k-", lw=1.2, label=f"theory u*_CRRA(gamma)")
        ax.errorbar(rhos - 0.02, q50_k, yerr=[q50_k - q05_k, q95_k - q50_k],
                    fmt="o", capsize=3, label="posterior u*_Kelly (median, 90% CI)")
        ax.errorbar(rhos + 0.02, q50_c, yerr=[q50_c - q05_c, q95_c - q50_c],
                    fmt="s", capsize=3, label="posterior u*_CRRA (median, 90% CI)")
        ax.set_title(f"H = {H} steps ({'~1 mean-reversion' if H <= 200 else '~5 mean-reversions'})")
        ax.set_xlabel("rho (correlation)")
        ax.set_ylabel("u = (pi - pi_ref) / pi_ref")
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Heston multiplicative overlay: Bayesian posterior u* vs rho  "
        f"(V0={V0}, response = log W_T)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Runner
# ==========================================================================


def main():
    print("=" * 100)
    print("HESTON MULTIPLICATIVE OVERLAY: BAYESIAN POSTERIOR (log W_T response)")
    print("  coordinate u = (pi - pi_ref(V)) / pi_ref(V),   pi_ref(V) = myopic Merton(V)")
    print("  Bayesian NIG posterior on [1, u, u^2];")
    print("  Kelly target: u* = rho * xi * p_1    / (mu - r)")
    print("  CRRA  target: u* = rho * xi * p_gamma/ (mu - r)  (gamma-specific exponent)")
    print("=" * 100)

    rho_rows = sweep_rho(gamma=3.0)
    V_rows = sweep_V()

    print()
    print("=" * 100)
    print("SUMMARY - rho sweep at H = 1260 (stationary-approximate regime)")
    print("=" * 100)
    print("rho   | u_Kelly_th | u_Kelly_post (CI)            | "
          "u_CRRA_th  | u_CRRA_post (CI)           | P(concave)| P(TR) | P(sign_CRRA)")
    print("-" * 130)
    for r in rho_rows:
        if r["config"].H != 1260:
            continue
        th_k = r["theory_targets"]["u_star_Kelly"]
        th_c = r["theory_targets"]["u_star_CRRA"]
        psign = r["crra_psign_match"] if np.isfinite(r["crra_psign_match"]) else float("nan")
        print(
            f"{r['env_rho']:+.2f} | {th_k:+.4f}    | "
            f"{_format_ci(r['kelly_u_star_q05'], r['kelly_u_star_q50'], r['kelly_u_star_q95'])} | "
            f"{th_c:+.4f}   | "
            f"{_format_ci(r['crra_u_star_q05'], r['crra_u_star_q50'], r['crra_u_star_q95'])} | "
            f"{r['crra_concavity_mass']:7.3f}   | {r['crra_in_tr_mass']:5.3f} | {psign:5.3f}"
        )

    print()
    print("=" * 100)
    print(f"SUMMARY - V sweep at rho={V_rows[0]['env_rho']:+.1f}, "
          f"H = {V_rows[0]['config'].H} (V-invariance structural test)")
    print("=" * 100)
    print("V0    | u_CRRA_theory | posterior u*_CRRA (CI)        | P(concave)| P(TR) ")
    print("-" * 100)
    for r in V_rows:
        th_c = r["theory_targets"]["u_star_CRRA"]
        print(
            f"{r['config'].V0:.3f} | {th_c:+.4f}       | "
            f"{_format_ci(r['crra_u_star_q05'], r['crra_u_star_q50'], r['crra_u_star_q95'])} | "
            f"{r['crra_concavity_mass']:7.3f}   | {r['crra_in_tr_mass']:5.3f}"
        )
    u_medians = [r["crra_u_star_q50"] for r in V_rows if np.isfinite(r["crra_u_star_q50"])]
    if len(u_medians) >= 3:
        cv = np.std(u_medians, ddof=1) / max(abs(np.mean(u_medians)), 1e-12)
        print()
        print(f"V-sweep invariance (on posterior u*_CRRA medians): "
              f"CV = {cv:.3f}    (small = V-invariance supported)")

    # Plot
    plot_dir = HERE
    plot_path = os.path.join(plot_dir, "heston_rho_posterior.png")
    _plot_rho_sweep(rho_rows, plot_path, V0=0.04)
    print()
    print(f"Saved rho-sweep plot: {plot_path}")


if __name__ == "__main__":
    main()
