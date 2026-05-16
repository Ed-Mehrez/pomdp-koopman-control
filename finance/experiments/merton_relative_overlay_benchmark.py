r"""
Approach 1 benchmark: Heston/CRRA stationary multiplicative overlay.

Theory source
-------------
- docs/theory_crra_eigenfunction.md, Prop 14.2 (Stationary Multiplicative Overlay)
- docs/signature_based_filtering_control.md, Prop 5.1 + Cor 5.2 (reference-relative
  coordinate is the natural local coordinate for multiplicative benchmarks)

Claim being tested
------------------
The stationary-HJB ansatz Phi(v) = v^p gives

    pi*(v)  =  pi_myopic(v) * (1 + rho*xi*p / (mu-r)),

so in the reference-relative coordinate

    u_t  =  (pi_t - pi_ref(V_t)) / pi_ref(V_t),

the target is a V-INDEPENDENT CONSTANT

    u*  =  rho * xi * p / (mu - r).

We verify three STRUCTURAL properties of this target:

    (A) correct SIGN of u* across rho sign flips;
    (B) V-INVARIANCE of the recovered u* (the key structural claim of the
        multiplicative form: sweeping V_0 should leave u_hat ~ constant);
    (C) approximate MAGNITUDE recovery of u* at large horizon (where the
        stationary ansatz becomes tight).

Finite-horizon honesty
----------------------
For any finite H, the true Bellman optimum is
    pi*(t,v) = pi_myopic(v) + (rho*xi/gamma) * partial_v Phi(t,v) / Phi(t,v),
and Phi(t,v) is time-dependent.  As t -> T the terminal condition Phi(T,v)=1
forces the hedging overlay to zero, so the T-averaged optimal u is SMALLER
in magnitude than the stationary u*.

We therefore run the benchmark at two horizons:
    H_stationary   =  5 / kappa   (many mean-reversion times; stationary is
                                   tight, ~exp(-5) residual)
    H_finite       =  1 / kappa   (one mean-reversion time; finite-H
                                   contraction is expected and reported).

For H = H_stationary we expect u_hat ~ u* theory.
For H = H_finite  we expect |u_hat| to be SYSTEMATICALLY smaller than |u*|;
that is a correct structural prediction, NOT a failure.

This script does NOT silently change the benchmark target if H_finite
underreports.  It simply reports the finite-H ratio and flags it.
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

from src.control.local_around_reference import (
    ControlNormalizer,
    GateObjective,
    GateSpec,
    ReferenceProportionalNormalizer,
    run_signal_gate,
)


# ==========================================================================
# State object (finance adapter-local; not in src/control)
# ==========================================================================


@dataclass(frozen=True)
class HestonState:
    logW: float
    V: float


# ==========================================================================
# Paired estimator with STATE-ADAPTIVE multiplicative overlay
# ==========================================================================


class HestonTerminalCRRAEstimatorMultiplicative:
    r"""Paired-noise rollout with state-adaptive multiplicative overlay.

    At each step,
        pi_ref_t  =  pi_myopic(V_t)  =  (mu - r) / (gamma * V_t),
        pi_t      =  pi_ref_t  +  normalizer.to_action_delta(u, (V_t), pi_ref_t),
    and the normalizer IS `ReferenceProportionalNormalizer` (re-applied at
    each step so the overlay is pi_t = (1 + u) * pi_ref_t).

    The reported `action_delta_0 = u * pi_myopic(V_0)` equals the analytical
    hedging demand at V_0 = V_ref when u = rho * xi * p / (mu - r).
    """

    def __init__(self, env: HestonMertonEnv, H: int, dt: float):
        self.env = env
        self.H = int(H)
        self.dt = float(dt)

    def _paired_noise(self, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        zA = rng.standard_normal(self.H)
        zB = rng.standard_normal(self.H)
        rho = self.env.rho
        z1 = zA
        z2 = rho * zA + np.sqrt(max(1.0 - rho * rho, 0.0)) * zB
        return np.column_stack([z1, z2])

    def _pi_ref(self, V: float) -> float:
        return float(self.env.merton_optimal(max(V, 1e-8)))

    def __call__(self, u_grid, state, ref_action, normalizer, seed):
        noise = self._paired_noise(seed)
        gamma = self.env.gamma
        out = np.zeros(len(u_grid))
        for k, u in enumerate(u_grid):
            logW, V = float(state.logW), float(state.V)
            for t in range(self.H):
                pi_nominal = self._pi_ref(V)
                state_t = HestonState(logW=logW, V=V)
                delta_t = normalizer.to_action_delta(
                    float(u), state_t, pi_nominal,
                )
                pi_t = pi_nominal + delta_t
                logW, V = self.env.step_explicit(
                    logW, V, pi_t,
                    float(noise[t, 0]), float(noise[t, 1]), self.dt,
                )
            W = max(float(np.exp(logW)), 1e-12)
            if abs(gamma - 1.0) < 1e-12:
                out[k] = np.log(W)
            else:
                out[k] = W ** (1.0 - gamma) / (1.0 - gamma)
        return out


# ==========================================================================
# Theoretical overlay target in the LOCAL coordinate u
# ==========================================================================


def relative_target(env: HestonMertonEnv, V: float) -> float:
    r"""u* = rho * xi * p / (mu - r)  (V-independent under stationary ansatz)."""
    theory = stationary_heston_crra_theory(env, V)
    a = float(env.mu - env.r)
    if abs(a) < 1e-12:
        return 0.0
    return float(env.rho * env.xi * theory.exponent_p / a)


# ==========================================================================
# Sweep driver
# ==========================================================================


def _run_cell(
    env: HestonMertonEnv,
    V0: float,
    H: int,
    dt: float,
    n_seeds: int,
    u_max: float,
    n_grid: int,
) -> Dict[str, float]:
    r"""Runs a single (rho, V0, H) cell and returns raw stats."""
    theory = stationary_heston_crra_theory(env, V0)
    pi_ref_0 = float(theory.myopic_pi)
    target_u_local = relative_target(env, V0)
    target_action_delta = target_u_local * pi_ref_0  # reported action_delta at V0

    normalizer = ReferenceProportionalNormalizer()
    spec = GateSpec(
        label=f"rho={env.rho:+.1f},V0={V0:.3f},H={H}",
        state=HestonState(logW=0.0, V=V0),
        reference_action=pi_ref_0,
        target_overlay=target_action_delta,
        normalizers=[normalizer],
        u_max_list=[u_max],
        meta={
            "rho": float(env.rho), "V0": float(V0), "H": int(H),
            "exponent_p": float(theory.exponent_p),
            "target_u_local": float(target_u_local),
            "pi_ref_0": pi_ref_0,
        },
    )
    objective = GateObjective(
        name="terminal_CRRA_utility",
        horizon=H, dt=dt,
        notes="state-adaptive multiplicative overlay: pi_t = (1 + u) * pi_myopic(V_t)",
    )
    estimator = HestonTerminalCRRAEstimatorMultiplicative(env, H=H, dt=dt)
    rows = run_signal_gate(
        specs=[spec], objective=objective, estimator=estimator,
        n_seeds=n_seeds, n_grid=n_grid,
    )
    row = rows[0]
    # Convert pooled action_delta back to LOCAL u-coordinate for a cleaner
    # comparison against u* = rho * xi * p / (mu - r).
    u_pool = row.action_delta_pooled / pi_ref_0 if pi_ref_0 != 0 else float("nan")
    u_pool_se = row.action_delta_pooled_se / abs(pi_ref_0) if pi_ref_0 != 0 else float("nan")
    t_u = u_pool / u_pool_se if u_pool_se > 0 else float("inf")
    if abs(target_u_local) > 1e-12:
        t_u_vs_theory = (u_pool - target_u_local) / u_pool_se if u_pool_se > 0 else float("inf")
        ratio_u = u_pool / target_u_local
    else:
        t_u_vs_theory = float("nan")
        ratio_u = float("nan")
    return {
        "rho": env.rho, "V0": V0, "H": H,
        "exponent_p": theory.exponent_p,
        "target_u_local": target_u_local,
        "pi_ref_0": pi_ref_0,
        "target_action_delta": target_action_delta,
        "u_pool": u_pool,
        "u_pool_se": u_pool_se,
        "t_u_H0_zero": t_u,
        "t_u_H0_theory": t_u_vs_theory,
        "ratio_u": ratio_u,
        "action_delta_pool": row.action_delta_pooled,
        "action_delta_pool_se": row.action_delta_pooled_se,
        "t_ad_zero": row.t_pooled,
        "t_ad_theory": row.t_pooled_vs_theory,
        "effect_ratio_ad": row.effect_ratio_pooled,
        "concave_frac": row.concave_frac,
    }


def sweep_rho(
    rhos: List[float] = (-0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.7),
    gamma: float = 3.0,
    V0: float = 0.04,
    horizons: List[int] = (126, 1260),     # H_finite ~ 1/kappa=126d; H_stationary 5/kappa
    dt: float = 1.0 / 252.0,
    n_seeds: int = 30,
    u_max: float = 0.3,
    n_grid: int = 41,
) -> List[Dict]:
    r"""Rho sweep at fixed V0; two horizons to expose finite-H mismatch."""
    rows = []
    for H in horizons:
        for rho in rhos:
            env = HestonMertonEnv(rho=rho, gamma=gamma)
            row = _run_cell(env, V0, H, dt, n_seeds, u_max, n_grid)
            rows.append(row)
    return rows


def sweep_V(
    V_list: List[float] = (0.02, 0.03, 0.04, 0.06, 0.08),
    rho: float = -0.7,
    gamma: float = 3.0,
    H: int = 1260,
    dt: float = 1.0 / 252.0,
    n_seeds: int = 30,
    u_max: float = 0.3,
    n_grid: int = 41,
) -> List[Dict]:
    r"""V sweep at fixed rho; theory predicts u* is V-independent."""
    rows = []
    env = HestonMertonEnv(rho=rho, gamma=gamma)
    for V0 in V_list:
        row = _run_cell(env, V0, H, dt, n_seeds, u_max, n_grid)
        rows.append(row)
    return rows


# ==========================================================================
# Reporting
# ==========================================================================


def _fmt(val):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "   nan "
    return f"{val:+7.4f}"


def _print_rho_sweep(rows: List[Dict]):
    print("=" * 128)
    print("APPROACH 1: HESTON STATIONARY MULTIPLICATIVE OVERLAY  -- rho sweep")
    print("  coordinate u = (pi - pi_ref(V))/pi_ref(V)   (reference-relative)")
    print("  theoretical target u* = rho * xi * p / (mu - r)   (V-INDEPENDENT under stationary ansatz)")
    print("=" * 128)
    print("H    | rho    | V0    |   u*     |  u_hat  |  SE(u)  | t(H0:0) | t(H0:th) | ratio_u | sign ok? | conc")
    print("-" * 128)
    by_H = {}
    for row in rows:
        by_H.setdefault(row["H"], []).append(row)
    for H in sorted(by_H.keys()):
        for row in by_H[H]:
            sign_ok = "n/a" if abs(row["target_u_local"]) < 1e-8 else (
                "Y" if np.sign(row["u_pool"]) == np.sign(row["target_u_local"]) else "n"
            )
            print(
                f"{H:4d} | {row['rho']:+.2f}  | {row['V0']:.3f} | "
                f"{_fmt(row['target_u_local'])}  | {_fmt(row['u_pool'])} | "
                f"{row['u_pool_se']:.4f} | {row['t_u_H0_zero']:+7.2f} | "
                f"{row['t_u_H0_theory']:+7.2f}  | {_fmt(row['ratio_u'])} | {sign_ok:^8s} | "
                f"{row['concave_frac']:.2f}"
            )
        print("-" * 128)


def _print_V_sweep(rows: List[Dict]):
    print()
    print("=" * 118)
    print("APPROACH 1: HESTON STATIONARY MULTIPLICATIVE OVERLAY  -- V sweep at fixed rho")
    print("  Structural test: u_hat should be V-INVARIANT (stationary-ansatz claim).")
    print(
        "  If u_hat depends strongly on V, either the stationary ansatz is "
        "off OR the finite-H bias is V-sensitive."
    )
    print("=" * 118)
    print("rho    | V0     |   u*     |  u_hat  |  SE(u)  | t(H0:0) | t(H0:th) | ratio_u | conc")
    print("-" * 118)
    for row in rows:
        print(
            f"{row['rho']:+.2f}  | {row['V0']:.3f}  | "
            f"{_fmt(row['target_u_local'])}  | {_fmt(row['u_pool'])} | "
            f"{row['u_pool_se']:.4f} | {row['t_u_H0_zero']:+7.2f} | "
            f"{row['t_u_H0_theory']:+7.2f}  | {_fmt(row['ratio_u'])} | "
            f"{row['concave_frac']:.2f}"
        )
    # V-invariance summary: coefficient of variation of u_hat across the V grid.
    u_hats = np.array([r["u_pool"] for r in rows if np.isfinite(r["u_pool"])])
    u_targets = np.array([r["target_u_local"] for r in rows if np.isfinite(r["u_pool"])])
    if u_hats.size >= 3:
        mean_u = float(np.mean(u_hats))
        std_u = float(np.std(u_hats, ddof=1))
        cv_u = abs(std_u / mean_u) if abs(mean_u) > 1e-12 else float("inf")
        mean_th = float(np.mean(u_targets))
        print()
        print(
            f"V-sweep invariance:   mean(u_hat) = {mean_u:+.4f}   std = {std_u:.4f}   "
            f"CV = {cv_u:.3f}    theory mean u* = {mean_th:+.4f}"
        )
        print(
            "  Interpretation: small CV (<~0.2) + ratio_u in [0.5, 2] "
            "= structural claim of V-invariant u* supported."
        )


def _print_finite_H_discount(rho_rows: List[Dict]):
    r"""Compare u_hat at H_finite vs H_stationary (same rho, same V0)."""
    by_rho = {}
    for row in rho_rows:
        by_rho.setdefault(row["rho"], []).append(row)
    print()
    print("=" * 88)
    print("FINITE-HORIZON DISCOUNT (empirical)")
    print(
        "  ratio_H = u_hat(H_finite) / u_hat(H_stationary);  expected < 1 in magnitude"
    )
    print("  because finite-H Phi(t,v) decays to terminal 1 at t=T.")
    print("=" * 88)
    print("rho   | u_hat(H_fin) | u_hat(H_sta) | ratio_H (signed) | theory u*")
    print("-" * 88)
    for rho in sorted(by_rho.keys()):
        rows = by_rho[rho]
        if len(rows) < 2:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["H"])
        h_fin, h_sta = rows_sorted[0], rows_sorted[-1]
        if abs(h_sta["u_pool"]) < 1e-10:
            ratio_H = float("nan")
        else:
            ratio_H = h_fin["u_pool"] / h_sta["u_pool"]
        print(
            f"{rho:+.2f} | {_fmt(h_fin['u_pool'])}    | "
            f"{_fmt(h_sta['u_pool'])}    | {_fmt(ratio_H)}         | "
            f"{_fmt(h_fin['target_u_local'])}"
        )


def main():
    print("Running rho sweep (two horizons)...")
    rho_rows = sweep_rho()
    _print_rho_sweep(rho_rows)
    _print_finite_H_discount(rho_rows)

    print()
    print("Running V sweep at rho=-0.7 (stationary horizon)...")
    V_rows = sweep_V(rho=-0.7)
    _print_V_sweep(V_rows)


if __name__ == "__main__":
    main()
