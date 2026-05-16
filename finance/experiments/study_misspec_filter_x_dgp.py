r"""
ABLATION: misspecification × filter × DGP, with value-gradient controller.

Goal
----
Pin where misspecification damage shows up in the filter+controller stack.
The historical Level-4 graduated-sanity-checks result (`MEMORY.md`) showed
that on CEV (with CIR-assuming filters), the model-free signature filter
HOLDS while the known-CIR Kalman DEGRADES at the **filter level**.  This
study asks the matching CONTROL question: paired-CRN CRRA score on
held-out seeds.

Cells = filters × DGPs (vg controller throughout):
  filters : oracle (cheats with true V) | sig (LeadLagBLRKF) | kalman (HeteroKalman)
  DGPs    : Heston (CIR true)           | CEV (CIR misspec)
  ⇒ 6 cells.

All filters that need CIR parameters use Heston defaults (kappa=2, theta=0.04,
xi=0.3) regardless of the underlying DGP.  This is the misspecification:
under CEV, both the kalman filter and the sig lane's outer KF assume CIR
that is wrong.  The sig lane's BLR target (r^2/dt → V) is structurally
model-free and continues to hold under CEV.

Why no separate "controller-off" axis
-------------------------------------
The existing `_step_state` uses `pi_ref(state.V_true)` (oracle myopic
reference at the env level), so setting u=0 collapses across filters --
all (filter, u=0) cells are numerically identical.  Any honest
controller-axis ablation would require a forked rollout where pi_ref
uses V_hat from the filter; that's a separate experiment.  Here we keep
the controller fixed (value-gradient) and isolate the filter axis.

Reporting
---------
For each cell:
  - CRRA paired score = mean(ΔlogW) − ½(γ−1)·Var(ΔlogW), bootstrap CI
  - Filter corr / RMSE vs latent V on held-out seeds
  - Per-step abstention rate

Pre-registered reading
----------------------
1. Heston DGP: oracle ≥ sig ≥ kalman is the expected ordering (matches
   memory's V-corr pattern).
2. CEV DGP: oracle ≥ sig » kalman is the prediction (kalman degrades
   harder under CIR misspecification).
3. The QUANTITY of interest is `Δ_misspec(filter) :=
   CRRA(filter, Heston) − CRRA(filter, CEV)`.  If
   Δ_misspec(kalman) >> Δ_misspec(sig), the model-free signature lane
   genuinely buys robustness *at the control level*, not just at the
   filter-quality level.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv, CEVEnv
from merton_value_gradient import (
    VGConfig,
    HVGState,
    PsiA,
    TerminalZero,
    ObservedLogReturnStageCost,
    OracleVEstimator,
    EWMAVEstimator,
    LeadLagBLRKFVEstimator,
    HeteroKalmanVEstimator,
    collect_training_episodes,
    evaluate_controller_paired,
    HeldOutResult,
    _crra_score_bootstrap,
)
from src.control.local_value_gradient import (
    backward_value_iteration,
    training_value_r2,
)


# ==========================================================================
# Heston-assumed proxy env: lets filter factories run on any DGP
# ==========================================================================


class _HestonAssumedProxy:
    r"""Minimal duck-typed env that exposes the Heston-CIR attributes the
    filter factories read at construction time (kappa, theta, xi).  Used
    so we can build filter lanes that ASSUME CIR(2.0, 0.04, 0.3) even
    when the underlying DGP is CEV.

    NOT a step-able env -- only used for filter factory construction.
    """

    def __init__(self, kappa: float = 2.0, theta: float = 0.04, xi: float = 0.3):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        # PsiA also reads env.theta for the log(V/theta) feature.
        # The proxy already exposes it; PsiA can use this proxy.


# ==========================================================================
# DGPs: matched-scale defaults so V scales are comparable
# ==========================================================================


def make_heston_env(rho: float = -0.7, gamma: float = 3.0) -> HestonMertonEnv:
    return HestonMertonEnv(
        mu=0.08, r=0.02, gamma=gamma,
        kappa=2.0, theta=0.04, xi=0.3, rho=rho,
    )


def make_cev_env(gamma: float = 3.0, alpha: float = 0.5) -> CEVEnv:
    r"""sigma=0.2, var(S=1) = sigma^2 = 0.04 matches Heston theta.

    alpha=0.5  → var(S) = sigma^2 / S  (mild inverse leverage; closer to Heston in spirit).
    alpha=2.0  → var(S) = sigma^2 * S^2 (positive leverage; direction OPPOSITE to Heston rho<0).
                 This is a CLEARLY misspecified DGP for a CIR-assuming filter.
    """
    return CEVEnv(
        mu=0.08, r=0.02, gamma=gamma,
        sigma=0.2, alpha=alpha,
    )


# ==========================================================================
# Filter factory builders -- ALWAYS Heston-assumed, regardless of DGP
# ==========================================================================


def filter_factory(name: str, dt: float) -> Callable[[], object]:
    proxy = _HestonAssumedProxy()
    if name == "oracle":
        return lambda: OracleVEstimator()
    if name == "sig":
        return lambda: LeadLagBLRKFVEstimator(
            env=proxy, dt=dt, ll_gamma=0.99, target_clip=2.0,
        )
    if name == "kalman":
        return lambda: HeteroKalmanVEstimator(env=proxy, dt=dt)
    if name == "ewma":
        return lambda: EWMAVEstimator(halflife_days=21.0, dt=dt)
    raise ValueError(f"unknown filter name {name!r}")


# ==========================================================================
# One cell: train + evaluate
# ==========================================================================


def run_cell(
    dgp_name: str, env, cfg: VGConfig, filter_name: str,
    n_train: int, n_test: int, base_seed_train: int, base_seed_test: int,
) -> Dict[str, float]:
    # Use the Heston-assumed proxy for psi so the controller's lifted state
    # representation (specifically its theta-based log(V/theta) feature) is
    # FIXED across DGPs.  This is part of the misspecification: under CEV,
    # the controller still assumes theta=0.04 in its state representation.
    psi = PsiA(_HestonAssumedProxy(), cfg.T_steps)
    terminal = TerminalZero()
    stage_cost = ObservedLogReturnStageCost()
    factory = filter_factory(filter_name, cfg.dt)

    # Train
    cfg_train = replace(cfg, n_train=n_train)
    eps = collect_training_episodes(
        env, cfg_train, psi,
        n_episodes=n_train, base_seed=base_seed_train,
        v_estimator_factory=factory,
    )
    model = backward_value_iteration(
        eps, terminal_target=terminal, stage_cost=stage_cost,
        ridge_transition=cfg.ridge_transition, ridge_value=cfg.ridge_value,
    )
    r2_t0 = training_value_r2(model, eps, terminal, stage_cost).get(0, float("nan"))

    # Evaluate (paired vs myopic at u=0; both share Brownian noise)
    cfg_eval = replace(cfg, n_test=n_test)
    ev = evaluate_controller_paired(
        env, cfg_eval, psi, model=model,
        base_seed=base_seed_test,
        name=f"{dgp_name}_{filter_name}",
        v_estimator_factory=factory,
    )

    # Aggregate
    y = ev.delta_logW
    crra = _crra_score_bootstrap(y, cfg.gamma, rng=np.random.RandomState(2026))
    abst = float(np.mean(ev.abstention_flags))
    Vh = ev.V_hat_post_history if ev.V_hat_post_history.size else ev.V_hat_history
    Vt = ev.V_true_post_history if ev.V_true_post_history.size else ev.V_true_history
    if Vh.size > 0 and Vt.size > 0:
        rmse_V = float(np.sqrt(np.mean((Vh - Vt) ** 2)))
        flat_h = Vh.flatten(); flat_t = Vt.flatten()
        mask = np.isfinite(flat_h) & np.isfinite(flat_t)
        corr_V = float(np.corrcoef(flat_h[mask], flat_t[mask])[0, 1]) if mask.sum() >= 3 else float("nan")
    else:
        rmse_V = float("nan"); corr_V = float("nan")

    return {
        "dgp":         dgp_name,
        "filter":      filter_name,
        "crra_point":  crra["point"],
        "crra_q05":    crra["q05"],
        "crra_q95":    crra["q95"],
        "crra_se":     crra["se"],
        "mean_dlw":    float(np.mean(y)),
        "var_dlw":     float(np.var(y, ddof=1)) if y.size > 1 else 0.0,
        "abstention":  abst,
        "filter_corr": corr_V,
        "filter_rmse": rmse_V,
        "r2_t0":       float(r2_t0),
    }


# ==========================================================================
# Reporting
# ==========================================================================


def _print_cell_table(rows: List[Dict[str, float]]) -> None:
    dgps = sorted({r["dgp"] for r in rows})
    filters = ["oracle", "sig", "kalman"]
    print()
    print("Per-cell results")
    print("-" * 124)
    print(f"  {'DGP':>8s} | {'filter':>10s} | {'CRRA score':>26s} | "
          f"{'mean ΔlogW':>11s} | {'Var ΔlogW':>10s} | {'corr V̂':>7s} | "
          f"{'rmse V̂':>7s} | {'abst':>5s}")
    print("-" * 124)
    for dgp in dgps:
        for f in filters:
            r = next((r for r in rows if r["dgp"] == dgp and r["filter"] == f), None)
            if r is None:
                continue
            cs = (f"{r['crra_point']:+.5f} "
                  f"[{r['crra_q05']:+.5f}, {r['crra_q95']:+.5f}]")
            print(f"  {dgp:>8s} | {f:>10s} | {cs:>26s} | "
                  f"{r['mean_dlw']:+.5f}    | {r['var_dlw']:.4e} | "
                  f"{r['filter_corr']:+.3f}  | {r['filter_rmse']:.4f}  | "
                  f"{r['abstention']:.2f}")
        print()


def _print_misspec_summary(rows: List[Dict[str, float]]) -> None:
    by = {(r["dgp"], r["filter"]): r for r in rows}
    filters = ["oracle", "sig", "kalman"]
    print("Misspecification damage  Δ_misspec(filter) := CRRA(Heston) − CRRA(CEV)")
    print("-" * 96)
    print(f"  {'filter':>10s} | {'CRRA on Heston':>16s} | {'CRRA on CEV':>16s} | "
          f"{'Δ_misspec':>14s}")
    print("-" * 96)
    for f in filters:
        rh = by.get(("heston", f))
        rc = by.get(("cev", f))
        if rh is None or rc is None:
            continue
        d = rh["crra_point"] - rc["crra_point"]
        print(f"  {f:>10s} | {rh['crra_point']:+.5f}        | {rc['crra_point']:+.5f}        | "
              f"{d:+.5f}")
    print()
    # Filter-axis comparisons within each DGP
    print("Filter-axis comparisons within each DGP (CRRA differences)")
    print("-" * 96)
    print(f"  {'DGP':>8s} | {'sig − oracle':>14s} | {'kalman − oracle':>16s} | "
          f"{'sig − kalman':>14s}")
    print("-" * 96)
    for dgp in ["heston", "cev"]:
        ro = by.get((dgp, "oracle")); rs = by.get((dgp, "sig")); rk = by.get((dgp, "kalman"))
        if any(r is None for r in (ro, rs, rk)):
            continue
        print(f"  {dgp:>8s} | {rs['crra_point']-ro['crra_point']:+.5f}        | "
              f"{rk['crra_point']-ro['crra_point']:+.5f}          | "
              f"{rs['crra_point']-rk['crra_point']:+.5f}")
    print()


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    cfg = VGConfig(
        rho=-0.7, gamma=3.0,
        T_steps=60, dt=1.0/252.0, u_max=0.3,
        n_train=200, n_test=120,
        V0_low=0.02, V0_high=0.08,
        ewma_halflife_days=21.0,
    )

    print("=" * 124)
    print("ABLATION: misspecification × filter × DGP  (value-gradient controller throughout)")
    print(f"  config: rho={cfg.rho}, gamma={cfg.gamma}, dt={cfg.dt:.5f}, "
          f"T_steps={cfg.T_steps}, u_max={cfg.u_max}")
    print(f"  filter assumed CIR: kappa=2.0, theta=0.04, xi=0.3 (FIXED across DGPs)")
    print(f"  DGPs: Heston (κ=2, θ=0.04, ξ=0.3, ρ=-0.7) | CEV (σ=0.2, α=0.5)")
    print(f"  n_train={cfg.n_train}, n_test={cfg.n_test}")
    print("=" * 124)

    rows: List[Dict[str, float]] = []
    # CEV alpha controls how strongly misspecified the DGP is relative to a
    # CIR-assuming filter.  Default to alpha=2.0 (positive S→var leverage) which
    # is structurally opposite to Heston's rho<0; a much harder regime for the
    # Kalman+CIR filter than the gentler alpha=0.5 case.
    cev_alpha = 2.0
    print(f"  CEV alpha = {cev_alpha} (var(S) = sigma^2 * S^{2*cev_alpha-2:.1f})")
    for dgp_name, env in [
        ("heston", make_heston_env(rho=cfg.rho, gamma=cfg.gamma)),
        ("cev",    make_cev_env(gamma=cfg.gamma, alpha=cev_alpha)),
    ]:
        print()
        print(f"--- DGP: {dgp_name} ---")
        for filter_name in ["oracle", "sig", "kalman"]:
            print(f"  cell: {dgp_name} × {filter_name} ... ", end="", flush=True)
            r = run_cell(
                dgp_name=dgp_name, env=env, cfg=cfg,
                filter_name=filter_name,
                n_train=cfg.n_train, n_test=cfg.n_test,
                base_seed_train=20_000 + (0 if dgp_name == "heston" else 1) * 100_000,
                base_seed_test =80_000 + (0 if dgp_name == "heston" else 1) * 100_000,
            )
            rows.append(r)
            print(f"CRRA = {r['crra_point']:+.5f}  "
                  f"corr V̂ = {r['filter_corr']:+.3f}  "
                  f"abst = {r['abstention']:.2f}")

    _print_cell_table(rows)
    _print_misspec_summary(rows)


if __name__ == "__main__":
    main()
