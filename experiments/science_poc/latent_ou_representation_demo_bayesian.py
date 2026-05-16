r"""
Approach 2 Bayesian demo: posterior over the linear-feedback gain across
three state representations (raw level, EFM transform, Kalman oracle).

What this script asks
---------------------
For each state representation z_t = T(path), the linear-feedback policy is

    u_t  =  g * z_t,                         g in R.

The per-episode reward R(g) under CRN is exactly quadratic in g:

    R(g)  =  g * sum_t z_t * X_t * dt  -  c * g**2 * sum_t z_t**2 * dt
          =  c1_path * g  +  c2_path * g**2.

We therefore fit a Bayesian NIG posterior on [1, g, g^2] pooled across
paired-noise seeds and ask:

    1. How TIGHT is the posterior on the optimal gain g* = -c1/(2c2)?
    2. What is P(c2 < 0 | D) (concavity; "is there a concave optimum at all?")
    3. Does the posterior SHRINK to g = 0 when the representation is bad
       (raw nonstationary level) and SHARPEN when the representation is
       good (Kalman)?

Theoretical gain targets
------------------------
With config c = 1 and the LQG closed form u_t* = Xhat_t / (2c):
  - Kalman oracle (z_t = Xhat_t):     g* = 1/(2c) = 0.5
  - Hidden-factor oracle (z_t = X_t): g* = 1/(2c) = 0.5
  - EFM level-1 at lambda = theta:    g* = 2 lambda / (lambda + theta) / (2c)
                                         = 1 * 1 / (2 * 1) = 0.5  (matched case)
  - Raw level z_t = S_t:              no fixed g* makes sense; S_t is
                                      non-stationary, so we expect
                                      a wide / flat posterior.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.control.state_transform import (
    EFMLevel1,
    KalmanLinearConfig,
    KalmanLinearFilter,
)
from src.control.bayesian_local_quadratic import (
    NIGPrior,
    fit_bayesian_quadratic,
    summarize_posterior,
)
from experiments.science_poc.envs.latent_ou_drift import (
    LatentOUConfig,
    LatentOUEnv,
)


# ==========================================================================
# Reward collection: for each (representation, gain, seed), run an episode
# ==========================================================================


def _paired_noise(config: LatentOUConfig, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.standard_normal((config.n_steps, 2))


def _run_episode_with_gain(
    config: LatentOUConfig,
    transform_factory: Callable[[], object],
    gain: float,
    noise: np.ndarray,
) -> float:
    r"""Run one episode with policy u_t = gain * z_t; return total reward.

    `transform_factory()` constructs a fresh stateful transform for this
    episode (reset-free across gains to match the same noise path).
    """
    env = LatentOUEnv(config)
    env.reset(seed=0)
    transform = transform_factory()
    dS_prev = 0.0
    total_reward = 0.0
    for t in range(config.n_steps):
        z = transform.update(np.array([dS_prev]), config.dt)[0]
        # For KalmanLinearFilter, the filter returns (Xhat, P); use only Xhat
        if hasattr(transform, "_P"):
            z = transform.current()[0]
        u_t = float(gain * z)
        dS, r = env.step(u_t, zX=float(noise[t, 0]), zY=float(noise[t, 1]))
        total_reward += r
        dS_prev = dS
    return total_reward


def _raw_level_adapter(config: LatentOUConfig):
    r"""A stateful transform that returns S_t via accumulating dS."""
    class _RawS:
        def __init__(self):
            self.S = 0.0
        def update(self, y, dt):
            self.S += float(y[0])
            return np.array([self.S])
        def current(self):
            return np.array([self.S])
    return _RawS


def collect_pooled_data(
    config: LatentOUConfig,
    transform_factory: Callable[[], object],
    gain_grid: np.ndarray,
    n_seeds: int,
    base_seed: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Returns (gains_pooled, rewards_pooled) with n_seeds * len(gain_grid) rows.

    CRN invariant: within each seed, the SAME noise matrix is reused across
    every value of `gain_grid`.  This ensures that posterior uncertainty
    on (c0, c1, c2) reflects inter-seed response variation, NOT
    intra-grid Monte Carlo noise that could have been differenced out.
    """
    gains_all = []
    rewards_all = []
    for k in range(n_seeds):
        noise = _paired_noise(config, base_seed + k)  # drawn ONCE per seed
        for g in gain_grid:
            # Same `noise` passed to every gain in this seed -> CRN across grid
            r = _run_episode_with_gain(config, transform_factory, float(g), noise)
            gains_all.append(float(g))
            rewards_all.append(r)
    return np.array(gains_all), np.array(rewards_all)


# ==========================================================================
# Runner per representation
# ==========================================================================


@dataclass
class RepresentationResult:
    name: str
    g_star_median: float
    g_star_q05: float
    g_star_q95: float
    g_star_std: float
    concavity_mass: float
    in_trust_region_mass: float
    recommended_action: float
    recommended_rationale: str
    coef_mean: np.ndarray
    coef_std: np.ndarray
    posterior_mean_on_grid: np.ndarray
    posterior_std_on_grid: np.ndarray
    ei_on_grid: np.ndarray
    g_grid: np.ndarray
    reward_obs_pairs: Tuple[np.ndarray, np.ndarray]  # (gains, rewards)


def run_representation(
    name: str,
    config: LatentOUConfig,
    transform_factory: Callable[[], object],
    gain_grid: np.ndarray,
    n_seeds: int,
    gain_max_for_trust_region: float,
    n_grid_posterior: int = 51,
) -> RepresentationResult:
    gains, rewards = collect_pooled_data(
        config, transform_factory, gain_grid, n_seeds=n_seeds,
    )
    post = fit_bayesian_quadratic(gains, rewards, prior=NIGPrior())
    summary = summarize_posterior(
        post,
        u_max=gain_max_for_trust_region,
        n_grid=n_grid_posterior,
        n_samples=8000,
        rng=np.random.RandomState(42),
    )
    return RepresentationResult(
        name=name,
        g_star_median=summary.optimizer_q50,
        g_star_q05=summary.optimizer_q05,
        g_star_q95=summary.optimizer_q95,
        g_star_std=summary.optimizer_std,
        concavity_mass=summary.concavity_mass,
        in_trust_region_mass=summary.in_trust_region_mass,
        recommended_action=summary.recommended_action,
        recommended_rationale=summary.recommended_action_rationale,
        coef_mean=summary.coef_mean,
        coef_std=summary.coef_std,
        posterior_mean_on_grid=summary.q_mean_on_grid,
        posterior_std_on_grid=summary.q_std_on_grid,
        ei_on_grid=summary.expected_improvement_on_grid,
        g_grid=summary.u_grid,
        reward_obs_pairs=(gains, rewards),
    )


# ==========================================================================
# Plot
# ==========================================================================


def _plot_posteriors(results: List[RepresentationResult], out_path: str, config: LatentOUConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: posterior mean Q(g) +/- 2 sd per representation
    ax = axes[0]
    for res in results:
        g = res.g_grid
        m = res.posterior_mean_on_grid
        s = res.posterior_std_on_grid
        ax.plot(g, m, lw=1.5, label=f"{res.name}  median g*={res.g_star_median:+.3f}")
        ax.fill_between(g, m - 2 * s, m + 2 * s, alpha=0.18)
    ax.axvline(0.5, color="k", linestyle="--", lw=0.8, label="LQG optimal g*=0.5")
    ax.set_xlabel("gain g")
    ax.set_ylabel("posterior Q(g) = E[reward | g]")
    ax.set_title("Posterior predictive (mean ± 2σ) of Q(g) vs control gain")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Right: posterior histogram of g*
    ax = axes[1]
    for res in results:
        gains, rewards = res.reward_obs_pairs
        post = fit_bayesian_quadratic(gains, rewards)
        rng = np.random.RandomState(7)
        from src.control.bayesian_local_quadratic import sample_optimizer
        samples = sample_optimizer(post, n_samples=20000, rng=rng)
        valid = np.isfinite(samples["u_star"]) & (np.abs(samples["u_star"]) <= 2.0)
        if valid.sum() > 100:
            ax.hist(samples["u_star"][valid], bins=80, alpha=0.4,
                    density=True, label=f"{res.name}  concave mass = {res.concavity_mass:.2f}")
    ax.axvline(0.5, color="k", linestyle="--", lw=0.8)
    ax.set_xlim(-1.5, 2.0)
    ax.set_xlabel("g* = -c1/(2 c2)  (posterior samples in trust region)")
    ax.set_ylabel("density")
    ax.set_title("Posterior over the optimizer g* per representation")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Latent-OU LQG: Bayesian posterior over linear-feedback gain\n"
        f"theta={config.theta}, sigma_X={config.sigma_X}, sigma_Y={config.sigma_Y},"
        f" T={config.T}, dt={config.dt}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================


def main():
    config = LatentOUConfig(
        theta=1.0, sigma_X=0.5, sigma_Y=0.5, c=1.0, T=4.0, dt=0.02,
    )
    kalman_cfg = KalmanLinearConfig(
        theta=config.theta, sigma_X=config.sigma_X, sigma_Y=config.sigma_Y,
    )
    n_seeds = 60
    gain_grid = np.linspace(-0.5, 1.5, 21)
    gain_max_for_trust_region = 1.5

    # Representation factories
    raw_factory = _raw_level_adapter(config)
    efm_factory = lambda: EFMLevel1(dim=1, lam=config.theta)
    kalman_factory = lambda: KalmanLinearFilter(kalman_cfg)

    reps = [
        ("raw_level_S", raw_factory),
        ("efm_level1_lam=theta", efm_factory),
        ("kalman_oracle_Xhat", kalman_factory),
    ]

    print("=" * 100)
    print(
        f"LATENT-OU BAYESIAN POSTERIOR DEMO  (n_seeds={n_seeds}, "
        f"gains in [{gain_grid[0]}, {gain_grid[-1]}], step {gain_grid[1]-gain_grid[0]:.2f})"
    )
    print(f"Optimal gain (theory): g*_LQG = 1/(2c) = {1/(2*config.c):.3f}")
    print("=" * 100)

    results: List[RepresentationResult] = []
    for name, factory in reps:
        res = run_representation(
            name, config, factory, gain_grid, n_seeds,
            gain_max_for_trust_region=gain_max_for_trust_region,
        )
        results.append(res)
        print()
        print(f"[{name}]")
        print(f"  Coefficient posterior mean ({'[c0, c1, c2]'})")
        print(f"     mean = [{res.coef_mean[0]:+.4f}, {res.coef_mean[1]:+.4f}, {res.coef_mean[2]:+.4f}]")
        print(f"     std  = [{res.coef_std[0]:.4f}, {res.coef_std[1]:.4f}, {res.coef_std[2]:.4f}]")
        print(f"  P(concave | D)             = {res.concavity_mass:.3f}")
        print(f"  P(g* in trust region | D)  = {res.in_trust_region_mass:.3f}")
        print(f"  posterior g* median        = {res.g_star_median:+.3f}")
        print(f"  posterior g* 90% CI        = [{res.g_star_q05:+.3f}, {res.g_star_q95:+.3f}]")
        print(f"  posterior g* std           = {res.g_star_std:.3f}")
        print(f"  recommended action (g)     = {res.recommended_action:+.4f}")
        print(f"     rationale               = {res.recommended_rationale}")

    # Rank by distance-to-LQG-target
    print()
    print("=" * 100)
    print("Ranking by posterior location + uncertainty:")
    print("=" * 100)
    print(f"{'representation':26s} | g*_med  | IQR width | P(concave) | P(TR)  | |g*_med − 0.5|")
    print("-" * 100)
    for res in results:
        iqr = res.g_star_q95 - res.g_star_q05
        dist = abs(res.g_star_median - 0.5) if np.isfinite(res.g_star_median) else float("nan")
        print(
            f"{res.name:26s} | {res.g_star_median:+.3f}  | {iqr:8.3f} | "
            f"{res.concavity_mass:9.3f} | {res.in_trust_region_mass:5.3f} | {dist:8.3f}"
        )

    # Plot
    out_path = os.path.join(HERE, "latent_ou_posterior.png")
    _plot_posteriors(results, out_path, config)
    print()
    print(f"Saved posterior plot: {out_path}")


if __name__ == "__main__":
    main()
