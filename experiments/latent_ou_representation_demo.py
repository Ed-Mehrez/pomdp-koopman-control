r"""
Approach 2 representation demo: raw observed level vs EFM-level-1 vs
Kalman-filtered state in the latent-OU-drift POMDP.

What this script tests
----------------------
Core Approach 2 claim (from theory_ergodic_signatures_and_horizon_selection.md,
Prop 7.1 and Prop 7.3):

    Raw non-ergodic observation levels are the WRONG state for prediction
    and control.  A fading-memory transform of increments is a RIGHT state.

Concretely, in the canonical latent-OU-drift env (experiments/envs/
latent_ou_drift.py):

    - Hidden factor X_t is OU, ergodic, spectral gap = theta.
    - Observed level S_t is cumulative, Var(S_t) ~ O(t), non-ergodic.
    - Optimal control (LQG + Kalman-Bucy): u_t* = Xhat_t / (2c).

We compare controllers under paired noise:

    * hidden_oracle   : u = X_true / (2c)          (upper bound)
    * kalman_oracle   : u = Xhat_Kalman / (2c)     (requires model)
    * efm_level_1     : u = gain * Z_t             (model-free; Z is EFM of dS)
                        for a sweep over lambda (EFM decay rate)
    * raw_level       : u = S / (2 c T_eff)        (theoretically bad)

Success criterion
-----------------
1. raw_level total reward is noticeably worse than the rest.
2. efm_level_1 at lambda = theta approaches kalman_oracle.
3. The empirical autocorrelation of Z_t decays at approximately lambda,
   confirming the Bakry-Emery / spectral-gap prediction.

NOTE: The demo fixes the EFM gain analytically at lambda=theta so the
comparison with Kalman is apples-to-apples (both attempt to estimate the
same latent drift).  A grid over lambda shows that mis-tuned lambda
degrades performance, which is itself an Approach-II theoretical
prediction (Remark 5.3: lambda selection is a live knob).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

from src.control.state_transform import (
    EFMLevel1,
    KalmanLinearConfig,
    KalmanLinearFilter,
    empirical_autocorrelation,
)
from experiments.envs.latent_ou_drift import (
    LatentOUConfig,
    LatentOUEnv,
    closed_form_stationary_value,
)


# ==========================================================================
# Paired-noise rollout
# ==========================================================================


def _paired_noise(config: LatentOUConfig, seed: int) -> np.ndarray:
    r"""(n_steps, 2) iid standard-normal matrix; columns are (zX, zY)."""
    rng = np.random.RandomState(seed)
    return rng.standard_normal((config.n_steps, 2))


def _run_episode(
    config: LatentOUConfig,
    policy: Callable[[float, float, float, int], float],
    noise: np.ndarray,
    transform_update: Callable[[float, float], float] = None,
) -> Dict[str, np.ndarray]:
    r"""Generic episode runner that exposes the raw state variables to the
    caller's policy.

    policy(X_true, S, Z, t_step) -> u_t
        (the policy receives the hidden factor too, but is free to ignore
        it; that lets us build an oracle baseline with the same harness)

    transform_update(dS_prev, dt) -> Z_t  (EFM lifted state), called each step
    """
    env = LatentOUEnv(config)
    env.reset(seed=0)  # reset is noise-agnostic with CRN
    n = config.n_steps
    X_hist = np.zeros(n)
    S_hist = np.zeros(n)
    Z_hist = np.zeros(n)
    u_hist = np.zeros(n)
    r_hist = np.zeros(n)
    dS_prev = 0.0
    Z_t = 0.0
    for t in range(n):
        if transform_update is not None:
            Z_t = float(transform_update(dS_prev, config.dt))
        X_hist[t] = env.X
        S_hist[t] = env.S
        Z_hist[t] = Z_t
        u_t = float(policy(env.X, env.S, Z_t, t))
        u_hist[t] = u_t
        dS, r = env.step(u_t, zX=float(noise[t, 0]), zY=float(noise[t, 1]))
        r_hist[t] = r
        dS_prev = dS
    return {"X": X_hist, "S": S_hist, "Z": Z_hist, "u": u_hist, "reward": r_hist}


# ==========================================================================
# Controllers
# ==========================================================================


def policy_hidden_oracle(c: float):
    def pol(X, S, Z, t):
        return X / (2.0 * c)
    return pol


def policy_kalman(filter_obj: KalmanLinearFilter, c: float):
    def pol(X, S, Z, t):
        xhat, _ = filter_obj.current()
        return xhat / (2.0 * c)
    return pol


def policy_efm_linear(gain: float):
    def pol(X, S, Z, t):
        return gain * Z
    return pol


def policy_raw_level(T_eff: float, c: float):
    def pol(X, S, Z, t):
        return S / (2.0 * c * T_eff)
    return pol


# ==========================================================================
# Efm + Kalman wiring under paired noise
# ==========================================================================


def _run_all_policies_paired(
    config: LatentOUConfig,
    efm_lambdas: List[float],
    kalman_cfg: KalmanLinearConfig,
    n_seeds: int,
    base_seed: int = 1000,
) -> Dict[str, np.ndarray]:
    r"""Returns per-seed total reward for each policy.

    Policies run on EXACTLY the same noise path per seed (CRN).  The EFM
    transforms are re-initialized at each seed to share the warm-start.
    """
    T_eff = 1.0 / config.theta

    results: Dict[str, np.ndarray] = {
        "hidden_oracle": np.zeros(n_seeds),
        "kalman_oracle": np.zeros(n_seeds),
        "raw_level": np.zeros(n_seeds),
    }
    for lam in efm_lambdas:
        results[f"efm_lam={lam:g}"] = np.zeros(n_seeds)

    # Store autocorrelations of Z_t and S_t for the first seed only (diagnostic)
    diag_Z_autocorr: Dict[str, np.ndarray] = {}
    diag_S_autocorr: np.ndarray = np.zeros(1)

    for k in range(n_seeds):
        seed = base_seed + k
        noise = _paired_noise(config, seed)

        # 1. Hidden oracle
        def pol_h(X, S, Z, t):
            return X / (2.0 * config.c)
        out_h = _run_episode(config, pol_h, noise, transform_update=None)
        results["hidden_oracle"][k] = float(out_h["reward"].sum())

        # 2. Kalman oracle
        kf = KalmanLinearFilter(kalman_cfg)
        def kalman_update(dS_prev, dt):
            arr = kf.update(np.array([dS_prev]), dt)
            return float(arr[0])
        def pol_kf(X, S, Z, t):
            return Z / (2.0 * config.c)
        out_k = _run_episode(config, pol_kf, noise, transform_update=kalman_update)
        results["kalman_oracle"][k] = float(out_k["reward"].sum())

        # 3. Raw-level baseline
        def pol_raw(X, S, Z, t):
            return S / (2.0 * config.c * T_eff)
        out_r = _run_episode(config, pol_raw, noise, transform_update=None)
        results["raw_level"][k] = float(out_r["reward"].sum())

        # 4. EFM-level-1 at a sweep of lambdas
        for lam in efm_lambdas:
            efm = EFMLevel1(dim=1, lam=lam)
            # EFM approximates the Kalman gain around X; choose gain that
            # makes E[Z^2] match E[X^2] stationary variance under matched
            # input dynamics.  For u* = X/(2c) and Z ~ X/lambda_eff scaling,
            # the corresponding linear policy is u = Z / (2c * 1) because
            # the EMA of X*dt + noise*dW has gain 1 in the dt -> 0 limit.
            def efm_update(dS_prev, dt, efm=efm):
                arr = efm.update(np.array([dS_prev]), dt)
                return float(arr[0])
            # The linear EFM policy that matches LQG asymptotically
            # (see docstring derivation notes at end of module):
            # for Z_t = sum e^{-lam*h} * dS, the expected value is
            #   E[Z_t] = sum e^{-lam*h} * E[dS] = 0, and stationary variance
            # is sigma_Y^2/(2 lam) + sigma_X^2/(2 lam*(lam+theta)*theta)
            # dominated by sigma_X^2/(2*lam*theta*(lam+theta)) when SNR
            # favors signal.  The matched linear gain is 2*lam/(lam+theta)
            # which reduces to 1 when lam = theta.
            gain = (2.0 * lam / (lam + config.theta)) / (2.0 * config.c)
            def pol_efm(X, S, Z, t, g=gain):
                return g * Z
            out_e = _run_episode(config, pol_efm, noise, transform_update=efm_update)
            results[f"efm_lam={lam:g}"][k] = float(out_e["reward"].sum())

            # Diagnostic: on the first seed, capture Z autocorrelation
            if k == 0:
                diag_Z_autocorr[f"efm_lam={lam:g}"] = empirical_autocorrelation(
                    out_e["Z"], max_lag=int(3.0 / lam / config.dt),
                )
        if k == 0:
            # Compute S autocorrelation too (should show NO decay for a
            # non-ergodic process)
            diag_S_autocorr = empirical_autocorrelation(
                out_h["S"], max_lag=min(200, config.n_steps // 2),
            )

    return {
        "totals": results,
        "Z_autocorr": diag_Z_autocorr,
        "S_autocorr": diag_S_autocorr,
    }


# ==========================================================================
# Reporting
# ==========================================================================


def _print_table(totals: Dict[str, np.ndarray]):
    n_seeds = next(iter(totals.values())).size
    hidden = totals["hidden_oracle"]
    print("=" * 90)
    print(
        f"LATENT-OU-DRIFT REPRESENTATION DEMO  (paired noise, n_seeds={n_seeds})"
    )
    print(
        "  Reward per episode = integral(u_t * X_t - c * u_t^2) dt."
        "  Larger is better."
    )
    print("=" * 90)
    header = (
        "policy                 |   mean(R)   |   std(R)  |   SE(R)  "
        "| regret-from-hidden | t(vs hidden)"
    )
    print(header)
    print("-" * len(header))
    for name, arr in totals.items():
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        se = std / np.sqrt(arr.size)
        # paired regret: hidden - self
        if name == "hidden_oracle":
            regret = 0.0
            t_vs = 0.0
            regret_line = "baseline            "
        else:
            diffs = hidden - arr
            regret = float(np.mean(diffs))
            reg_se = float(np.std(diffs, ddof=1)) / np.sqrt(arr.size)
            t_vs = regret / reg_se if reg_se > 0 else float("inf")
            regret_line = f"{regret:+.4f} ± {reg_se:.4f}"
        print(
            f"{name:22s} | {mean:+8.3f}    | {std:7.3f}   | {se:6.3f}   "
            f"| {regret_line:20s} | {t_vs:+7.2f}"
        )


def _print_autocorr(Z_ac: Dict[str, np.ndarray], S_ac: np.ndarray, config: LatentOUConfig):
    print()
    print("=" * 70)
    print("Spectral-gap sanity check: empirical autocorrelations (seed 1000)")
    print("  For an OU with gap lam, expect log(corr) ~ -lam * (k * dt).")
    print("  For the non-ergodic S, expect autocorrelation NOT to decay.")
    print("=" * 70)
    print("policy                  k=1       k=5       k=20      k=50    implied_gap")
    print("-" * 70)
    for name, ac in Z_ac.items():
        dt = config.dt
        vals = []
        for k in [1, 5, 20, 50]:
            if k < ac.size:
                vals.append(ac[k])
            else:
                vals.append(float("nan"))
        # Fit log(|ac|) ~ -gap * (k*dt) over k in [1, 20] to estimate gap
        ks = np.arange(1, min(21, ac.size))
        safe = np.abs(ac[ks]) > 1e-6
        if safe.sum() >= 3:
            logs = np.log(np.abs(ac[ks[safe]]))
            times = ks[safe] * dt
            slope, _ = np.polyfit(times, logs, 1)
            gap_hat = -slope
        else:
            gap_hat = float("nan")
        print(
            f"{name:22s}  {vals[0]:+.3f}   {vals[1]:+.3f}   {vals[2]:+.3f}   "
            f"{vals[3]:+.3f}   {gap_hat:+.3f}"
        )
    # Raw S
    print()
    print("S_t (raw level; should NOT decay):")
    for k in [1, 5, 20, 50]:
        if k < S_ac.size:
            print(f"  corr(S_0, S_k)  at k={k:3d}  =  {S_ac[k]:+.4f}")


# ==========================================================================
# Runner
# ==========================================================================


def main():
    config = LatentOUConfig(
        theta=1.0, sigma_X=0.5, sigma_Y=0.5, c=1.0, T=4.0, dt=0.02,
    )
    kalman_cfg = KalmanLinearConfig(
        theta=config.theta, sigma_X=config.sigma_X, sigma_Y=config.sigma_Y,
    )
    efm_lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]  # theta=1.0 is in the middle
    n_seeds = 60

    print("=" * 90)
    print("CONFIG")
    print(f"  theta={config.theta}  sigma_X={config.sigma_X}  sigma_Y={config.sigma_Y}  "
          f"c={config.c}")
    print(f"  T={config.T}  dt={config.dt}  n_steps={config.n_steps}  "
          f"n_seeds={n_seeds}")
    print(f"  Stationary Var(X) = {config.stationary_var_X:.4f}")
    print(f"  Stationary Kalman P_infty = {config.stationary_kalman_P:.4f}")
    print(f"  Stationary optimal per-step reward "
          f"= {closed_form_stationary_value(config):.6f}")
    print(f"  EFM-lambda sweep = {efm_lambdas}   "
          f"(theta={config.theta} is the optimal under matched model)")
    print()

    out = _run_all_policies_paired(
        config, efm_lambdas, kalman_cfg, n_seeds=n_seeds,
    )
    _print_table(out["totals"])
    _print_autocorr(out["Z_autocorr"], out["S_autocorr"], config)


if __name__ == "__main__":
    main()
