r"""
Heston/Merton adapter for the generic local-around-reference signal gate.

Phase 0 audit for `docs/note_merton_local_bilinear_spec.md`.

What this script does
---------------------
Maps the Heston/Merton benchmark onto the generic API in
`src/control/local_around_reference.py`:

    reference_action  <-  myopic Merton pi_ref = (mu - r) / (gamma * V_0)
    action_delta      <-  pi - pi_ref
    target_overlay    <-  analytical stationary-Heston hedging demand
    effect_samples    <-  terminal CRRA utility under paired-noise rollouts

Nothing finance-specific leaks back into `src/control/local_around_reference`.

Baseline labeling (honest)
--------------------------
The null `target_overlay = 0` here is MYOPIC Merton.  Under CRRA with known
drift mu and variance observed at V_0, the Level 4 local Ito-quadratic /
SDRE policy is algebraically identical to myopic Merton:

    pi_SDRE = -b / (2c)
            = -(U'(W) W (mu-r)) / (2 * 0.5 U''(W) W^2 V_0)
            = (mu - r) / (gamma * V_0)            (CRRA, gamma = -U''W/U')
            = pi_myopic.

So the `Delta = 0` baseline is BEST labeled as **"sensor-plus-myopic"**:
it validates the variance filter but carries NO intertemporal hedging
correction.  A detectable `action_delta != 0` is the signature of
intertemporal hedging that the Phase 1 local-bilinear path is meant to
capture.  Phase 1 is therefore only worth building if the gate can
distinguish the theoretical intertemporal overlay from the
sensor-plus-myopic null.
"""

from __future__ import annotations

import collections
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv
from merton_theory import stationary_heston_crra_theory

from src.control.local_around_reference import (
    ControlNormalizer,
    GateObjective,
    GateResult,
    GateSpec,
    IdentityNormalizer,
    ReferenceProportionalNormalizer,
    StateScaledNormalizer,
    run_signal_gate,
)


# ===========================================================================
# Finance state object (adapter-local; not in src/control)
# ===========================================================================


@dataclass(frozen=True)
class HestonState:
    logW: float
    V: float


# ===========================================================================
# Paired estimator: terminal CRRA utility over H Heston steps, CRN u-grid
# ===========================================================================


class HestonTerminalCRRAEstimator:
    r"""Rollout of `HestonMertonEnv.step_explicit` under common-random-number
    (CRN) paths with a V-adaptive reference policy and a
    STATE-ADAPTIVE overlay.

    Why state-adaptive?
    -------------------
    The stationary Heston-CRRA optimal policy is

        pi*(V)  =  pi_myopic(V) * (1 + rho*xi*p / (mu-r))

    i.e. a CONSTANT-RATIO correction multiplying pi_myopic(V).  An
    additive-constant overlay pi_t = pi_ref(V_t) + Delta cannot track a
    multiplicative V-dependent correction except at one specific V.
    Applying the normalizer once at the initial state freezes the overlay
    to that V and makes the estimand differ from the analytical hedging
    demand for reasons that are not signal-level but model-specification.

    Re-applying the normalizer at each step lets
    `ReferenceProportionalNormalizer` realize
        pi_t = pi_ref(V_t) + u * pi_ref(V_t) = (1 + u) * pi_ref(V_t),
    which matches the Heston-CRRA functional structure exactly.  In that
    parameterization the target u* = rho*xi*p/(mu-r) is a V-invariant
    constant, and the reported action_delta_0 = u * pi_ref(V_0) equals
    the analytical hedging demand at V_0 by construction.

    The per-step normalizer call is a general principle (not Heston-only):
    any domain whose nominal policy has state-dependent structure needs it.

    Correlation handling
    --------------------
    `HestonMertonEnv.step_explicit(logW, V, pi, z1, z2, dt)` treats z1 and
    z2 as independent Gaussians.  Adapter injects correlation rho: z1
    drives asset, z2 = rho*z1 + sqrt(1-rho^2)*z_perp drives variance.
    """

    def __init__(self, env: HestonMertonEnv, H: int, dt: float,
                 overlay_mode: str = "state_adaptive"):
        r"""
        overlay_mode:
            "state_adaptive"   - re-apply normalizer at each step with
                                 current (state_t, pi_ref(V_t));
                                 principled default for domains whose
                                 nominal policy is state-dependent.
            "init_state"       - apply normalizer once at initial
                                 (state_0, ref_action_0); overlay is a
                                 time-constant additive deviation.
                                 Kept for ablation only.
        """
        self.env = env
        self.H = int(H)
        self.dt = float(dt)
        if overlay_mode not in ("state_adaptive", "init_state"):
            raise ValueError(f"Unknown overlay_mode: {overlay_mode}")
        self.overlay_mode = overlay_mode

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
        init_action_delta = normalizer.to_action_delta(
            0.0, state, ref_action,
        )  # zero-check, not actually used
        for k, u in enumerate(u_grid):
            logW, V = float(state.logW), float(state.V)
            for t in range(self.H):
                pi_nominal = self._pi_ref(V)
                if self.overlay_mode == "state_adaptive":
                    # Re-evaluate normalizer at CURRENT state and CURRENT reference.
                    state_t = HestonState(logW=logW, V=V)
                    delta_t = normalizer.to_action_delta(
                        float(u), state_t, pi_nominal,
                    )
                else:  # "init_state"
                    delta_t = normalizer.to_action_delta(
                        float(u), state, ref_action,
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


# ===========================================================================
# Spec construction from (rho, gamma, V0) regime scan
# ===========================================================================


def build_specs(
    rhos: List[float],
    gamma: float,
    V0: float,
    u_max_list: List[float],
) -> Tuple[List[GateSpec], List[HestonMertonEnv]]:
    normalizers: List[ControlNormalizer] = [
        IdentityNormalizer(),
        StateScaledNormalizer(
            scale_fn=lambda s: float(np.sqrt(max(s.V, 1e-8))),
            name="sqrt_V",
        ),
        ReferenceProportionalNormalizer(),
    ]
    specs: List[GateSpec] = []
    envs: List[HestonMertonEnv] = []
    for rho in rhos:
        env = HestonMertonEnv(rho=rho, gamma=gamma)
        theory = stationary_heston_crra_theory(env, V0)
        specs.append(GateSpec(
            label=f"rho={rho:+.1f}",
            state=HestonState(logW=0.0, V=V0),
            reference_action=float(theory.myopic_pi),
            target_overlay=float(theory.hedging_demand),
            normalizers=normalizers,
            u_max_list=list(u_max_list),
            meta={
                "rho": float(rho), "gamma": float(gamma), "V0": float(V0),
                "exponent_p": float(theory.exponent_p),
                "myopic_pi": float(theory.myopic_pi),
            },
        ))
        envs.append(env)
    return specs, envs


# ===========================================================================
# Reporting (adapter-layer; applies pass/fail thresholds on top of raw stats)
# ===========================================================================


def _pass_pooled(row: GateResult, t_threshold: float,
                 effect_ratio_min: float = 0.5,
                 effect_ratio_max: float = 2.0) -> bool:
    r"""Primary gate: pooled coefficients must satisfy ALL of:
      (1) detectable against null:   |t_pooled|            >= t_threshold
      (2) consistent with theory:    |t_pooled_vs_theory|  <= t_threshold
      (3) effect magnitude in range: effect_ratio in [effect_ratio_min, max]

    Check (3) rejects false positives where the SE is so wide that any
    value in a huge range passes check (2).  Without it, an estimate that
    is 10x off in magnitude but has a 20x-wide confidence interval would
    spuriously "pass."
    """
    if not (np.isfinite(row.t_pooled)
            and np.isfinite(row.t_pooled_vs_theory)
            and np.isfinite(row.effect_ratio_pooled)):
        return False
    return (
        abs(row.t_pooled) >= t_threshold
        and abs(row.t_pooled_vs_theory) <= t_threshold
        and effect_ratio_min <= row.effect_ratio_pooled <= effect_ratio_max
    )


def _pass_per_seed(row: GateResult, t_threshold: float, sign_threshold: float) -> bool:
    r"""Secondary gate: per-seed mean passes detectability + sign match."""
    return (
        np.isfinite(row.t_stat)
        and abs(row.t_stat) >= t_threshold
        and np.isfinite(row.sign_match)
        and row.sign_match >= sign_threshold
    )


def _print_header(objective: GateObjective, gamma: float, V0: float,
                  n_seeds: int, n_grid: int):
    print("=" * 158)
    print("HESTON SIGNAL GATE  (generic local-around-reference core + finance adapter)")
    print(f"  objective:       {objective.name}")
    print(f"  horizon:         {objective.horizon} steps  dt={objective.dt:.5f}")
    print(f"  baseline label:  target=0  ==  myopic Merton  ==  SENSOR-PLUS-MYOPIC")
    print(f"                   (local Ito-quadratic / SDRE under CRRA with known mu)")
    print(f"  regime:          gamma={gamma}  V0={V0}  n_seeds={n_seeds}  "
          f"n_grid={n_grid}")
    print(f"  notes:           {objective.notes}")
    print(f"  primary gate:    POOLED estimator (pool c1,c2 across seeds, then")
    print(f"                   delta = -c1_pool/(2 c2_pool), delta-method SE)")
    print(f"  pass criterion:  |t_p(H0:0)|>=2  AND  |t_p(H0:th)|<=2  AND  "
          f"ratio_pool in [0.5, 2.0]")
    print("=" * 158)
    header = (
        "rho    | normalizer      | u_max | target_d |"
        "  Δ̂_pool    |  SE_pool   |  t_p(H0:0) | t_p(H0:th) | ratio_p |"
        "  Δ̂_seed    | t_s(H0:0) | sign | conc | P_pool"
    )
    print(header)
    print("-" * len(header))


def _print_row(row: GateResult, t_threshold: float, sign_threshold: float):
    rho = row.meta.get("rho", float("nan"))
    passes_pool = "Y" if _pass_pooled(row, t_threshold) else "n"
    td = row.target_overlay if row.target_overlay is not None else float("nan")
    rp = row.effect_ratio_pooled
    rp_str = f"{rp:+6.2f}" if np.isfinite(rp) else "   nan"
    print(
        f"{rho:+.1f}   | {row.normalizer_name:15s} | {row.u_max:.2f}  | "
        f"{td:+.4f}  | "
        f"{row.action_delta_pooled:+.6f}  | {row.action_delta_pooled_se:.6f}  | "
        f"{row.t_pooled:+7.2f}   | {row.t_pooled_vs_theory:+7.2f}   | "
        f"{rp_str} | "
        f"{row.action_delta_mean:+.6f}  | "
        f"{row.t_stat:+6.2f}  | {row.sign_match:4.2f} | "
        f"{row.concave_frac:.2f} | {passes_pool}"
    )


def _summarize(rows: List[GateResult], t_threshold: float, sign_threshold: float,
               abs_rho_min: float):
    print()
    print("=" * 110)
    print(
        f"PASS SUMMARY (POOLED estimator):  requiring |t_p(H0:0)|>={t_threshold} "
        f"AND |t_p(H0:th)|<={t_threshold}"
    )
    print(f"restricted to |rho|>={abs_rho_min} rows (nontrivial hedging demand):")
    print("=" * 110)
    bucket = collections.defaultdict(list)
    for r in rows:
        rho = float(r.meta.get("rho", 0.0))
        if abs(rho) < abs_rho_min - 1e-9:
            continue
        bucket[(r.normalizer_name, r.u_max)].append(
            (rho, _pass_pooled(r, t_threshold), r)
        )
    any_full = False
    any_partial = False
    for key, items in sorted(bucket.items()):
        n_pass = sum(int(p) for _, p, _ in items)
        n_tot = len(items)
        if n_pass == n_tot and n_tot > 0:
            marker = "PASS   "
            any_full = True
        elif n_pass > 0:
            marker = "partial"
            any_partial = True
        else:
            marker = "FAIL   "
        per_row = ", ".join(f"rho={r:+.1f}:{'Y' if p else 'n'}" for r, p, _ in items)
        print(f"  {key[0]:18s}  u_max={key[1]:.2f}  {n_pass}/{n_tot}  [{marker}]   {per_row}")
    print()
    if any_full:
        print("RECOMMENDATION: at least one (normalizer, u_max) CLEARS the gate.")
        print("  -> Phase 1 local-bilinear experiment is justified in that cell.")
        print("     Use the cell with highest mean |t| across rho as the first")
        print("     baseline; record its (normalizer, u_max) as the nominal Phase 1")
        print("     trust region.")
    elif any_partial:
        print("RECOMMENDATION: PARTIAL pass only.  DO NOT proceed to the")
        print("  signature-lifted Phase 1 fit yet.  Options before escalating:")
        print("    (a) increase n_seeds 4x in the partial-pass cells and re-test;")
        print("    (b) widen the regime scan (gamma in {3,5,7}, V0 in {0.01,0.04})")
        print("        and check whether signal strengthens predictably;")
        print("    (c) if widening does not help, the daily regime is the wrong")
        print("        operating point; consider weekly dt or longer horizon.")
    else:
        print("RECOMMENDATION: FULL FAIL across tested configurations.")
        print("  The Phase 1 local-bilinear path as specified will not recover")
        print("  intertemporal hedging demand in this regime.  Consider:")
        print("    (a) a regime with larger gamma and/or smaller V0;")
        print("    (b) a thicker-dt regime so CRN variance reduction has")
        print("        more room to shrink the paired-noise floor;")
        print("    (c) abandoning 'recover analytical Delta' as the Phase 1")
        print("        target; use the infrastructure for a control problem")
        print("        where the intertemporal signal is larger.")


# ===========================================================================
# Runner
# ===========================================================================


def main(
    rhos=(0.0, -0.3, -0.5, -0.7, -0.9),
    u_max_list=(0.05, 0.15, 0.30, 0.50),
    n_seeds: int = 20,
    n_grid: int = 41,
    H: int = 126,
    dt: float = 1.0 / 252.0,
    gamma: float = 3.0,
    V0: float = 0.04,
    t_threshold: float = 2.0,
    sign_threshold: float = 0.75,
    abs_rho_min: float = 0.3,
):
    specs, envs = build_specs(list(rhos), gamma, V0, list(u_max_list))
    objective = GateObjective(
        name="terminal_CRRA_utility",
        horizon=H,
        dt=dt,
        notes=(
            "pi_t = pi_ref(V_t) + normalizer.to_action_delta(u, state_t, pi_ref(V_t));"
            " state-adaptive overlay so that ref_proportional realizes the Heston-CRRA"
            " multiplicative correction structure exactly.  CRN across u-grid."
        ),
    )

    _print_header(objective, gamma, V0, n_seeds, n_grid)
    all_rows: List[GateResult] = []
    for spec, env in zip(specs, envs):
        estimator = HestonTerminalCRRAEstimator(env, H=H, dt=dt)
        rows = run_signal_gate(
            specs=[spec], objective=objective, estimator=estimator,
            n_seeds=n_seeds, n_grid=n_grid,
        )
        for r in rows:
            _print_row(r, t_threshold, sign_threshold)
            all_rows.append(r)

    _summarize(all_rows, t_threshold, sign_threshold, abs_rho_min)


if __name__ == "__main__":
    main()
