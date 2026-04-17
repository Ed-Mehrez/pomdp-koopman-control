r"""
Non-finance smoke test for `src/control/local_around_reference`.

Purpose
-------
Prove the generic gate abstractions don't smuggle finance concepts.  A 1D
LQR-style control-affine system with NO pi, V, rho, gamma, CRRA, or
Heston vocabulary exercises the same API the Heston adapter will use.

If this test passes, the boundary between the generic layer
(`src/control/local_around_reference.py`) and finance adapters is clean.

System
------
Scalar state x in R, dynamics

    x_{t+1} = x_t + (-a * x_t + b * u) * dt + sigma * sqrt(dt) * Z_t,  Z_t ~ N(0,1)

Reward over H steps with constant action u:

    R(u; x_0, path) = - x_H^2 - lambda * u^2 * H * dt.

The (approximate, constant-u) optimum at x_0 = 1 is negative.
For x_0 = 0, there is no optimum (null regime): reward is concave in u
with peak near u = 0.

Test 1: x_0 = 1.0 -> gate detects a negative action_delta (|t| >= 2,
        sign-match >= 0.75).
Test 2: x_0 = 0.0 -> null.  action_delta mean is near zero, |t| modest.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

from src.control.local_around_reference import (
    GateObjective,
    GateSpec,
    IdentityNormalizer,
    run_signal_gate,
)


@dataclass(frozen=True)
class ScalarLqrState:
    x: float


class LqrTerminalRewardEstimator:
    r"""Paired-noise constant-action rollout of scalar LQR.

    Returns an array of rewards aligned with `u_grid`, evaluated under a
    SINGLE paired-noise draw identified by `seed`.
    """

    def __init__(self, a=1.0, b=1.0, sigma=0.2, H=80, dt=0.02, lam=0.05):
        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.H = int(H)
        self.dt = float(dt)
        self.lam = float(lam)

    def __call__(self, u_grid, state, ref_action, normalizer, seed):
        rng = np.random.RandomState(seed)
        # Paired-noise: SAME dW path for every u.
        dW = rng.standard_normal(self.H) * np.sqrt(self.dt)
        rewards = np.zeros(len(u_grid))
        for k, u in enumerate(u_grid):
            action_delta = normalizer.to_action_delta(float(u), state, ref_action)
            action = float(ref_action) + action_delta
            x = float(state.x)
            for t in range(self.H):
                x = x + (-self.a * x + self.b * action) * self.dt + self.sigma * dW[t]
            rewards[k] = -x * x - self.lam * action * action * self.H * self.dt
        return rewards


def _run_cell(state_x: float, u_max_list, target_overlay, label: str, n_seeds: int = 25):
    spec = GateSpec(
        label=label,
        state=ScalarLqrState(x=state_x),
        reference_action=0.0,
        target_overlay=target_overlay,
        normalizers=[IdentityNormalizer()],
        u_max_list=u_max_list,
        meta={"x0": state_x},
    )
    objective = GateObjective(
        name="neg_terminal_x_squared_plus_action_penalty",
        horizon=80,
        dt=0.02,
        notes="scalar LQR-like: R = -x_H^2 - lam * u^2 * H * dt",
    )
    estimator = LqrTerminalRewardEstimator(
        a=1.0, b=1.0, sigma=0.2, H=80, dt=0.02, lam=0.05,
    )
    return run_signal_gate(
        [spec], objective, estimator, n_seeds=n_seeds, n_grid=41,
    )


def test_detects_nonzero_optimum():
    # Analytical constant-action optimum with a = b = 1, H*dt = 1.6, lam = 0.05.
    # Closed form: under constant u, x_H = A*x0 + B*u + noise with A = (1-a*dt)^H,
    # B = (1 - A)/a * b. Maximizing -x_H^2 - lam*u^2*H*dt gives
    #   u* = -B * A * x0 / (B^2 + lam * H * dt).
    # For x0 = 2: u* ≈ -0.44.
    target = -0.44
    rows = _run_cell(state_x=2.0, u_max_list=[1.0, 2.0],
                     target_overlay=target, label="x0=2")
    failures = []
    for r in rows:
        if r.n_valid < 15:
            failures.append(
                f"u_max={r.u_max}: n_valid={r.n_valid} < 15 (quadratic fit failing)"
            )
        if not np.isfinite(r.t_stat) or abs(r.t_stat) < 2.0:
            failures.append(
                f"u_max={r.u_max}: |t|={abs(r.t_stat):.2f} < 2 "
                f"(mean={r.action_delta_mean:+.3f}, SE={r.action_delta_se:.3e})"
            )
        if not np.isfinite(r.sign_match) or r.sign_match < 0.75:
            failures.append(
                f"u_max={r.u_max}: sign_match={r.sign_match:.2f} < 0.75"
            )
    print("[nonzero-optimum] per cell:")
    for r in rows:
        print(
            f"  u_max={r.u_max:.2f}  mean={r.action_delta_mean:+.3f}  "
            f"SE={r.action_delta_se:.3e}  t={r.t_stat:+.2f}  "
            f"t_vs_theory={r.t_stat_vs_theory:+.2f}  "
            f"sign={r.sign_match:.2f}  ratio={r.effect_ratio:+.2f}  "
            f"conc={r.concave_frac:.2f}"
        )
    if failures:
        raise AssertionError("\n  ".join(["nonzero-optimum FAILED:"] + failures))
    print("[nonzero-optimum] PASS\n")


def test_null_state_is_null():
    rows = _run_cell(state_x=0.0, u_max_list=[2.0],
                     target_overlay=None, label="x0=0")
    failures = []
    print("[null-state] per cell:")
    for r in rows:
        print(
            f"  u_max={r.u_max:.2f}  mean={r.action_delta_mean:+.4f}  "
            f"SE={r.action_delta_se:.3e}  t={r.t_stat:+.2f}  "
            f"conc={r.concave_frac:.2f}   (target=None, sign/ratio undefined)"
        )
        if not np.isnan(r.sign_match):
            failures.append(
                f"target_overlay=None but sign_match is finite: {r.sign_match}"
            )
        if not np.isnan(r.effect_ratio):
            failures.append(
                f"target_overlay=None but effect_ratio is finite: {r.effect_ratio}"
            )
        if abs(r.action_delta_mean) > 0.5:
            failures.append(
                f"|mean|={abs(r.action_delta_mean):.3f} > 0.5 for x_0=0"
            )
    if failures:
        raise AssertionError("\n  ".join(["null-state FAILED:"] + failures))
    print("[null-state] PASS\n")


if __name__ == "__main__":
    test_detects_nonzero_optimum()
    test_null_state_is_null()
    print("All LQR smoke tests passed.  Generic gate boundary is clean.")
