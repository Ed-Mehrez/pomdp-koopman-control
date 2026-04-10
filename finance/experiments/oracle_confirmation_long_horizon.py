"""Targeted oracle confirmation: N=20000 on the strongest long-horizon cells.

Follow-up to oracle_screening_grid.py. The screening grid showed all H=20 cells
negative and H=60 cells flipping positive with 3 LEAN (P>0.80) cells. This script
runs N=20000 paired seeds on the 3 strongest cells to confirm or reject.

Decision rule:
  - P(>0) >= 0.95 in at least one cell → channel alive in that regime
  - P(>0) < 0.95 everywhere at N=20000 → kill the estimation channel for OMM
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.controllers import (  # noqa: E402
    make_linear_inventory_skew,
    make_risk_neutral_optimal,
)
from applications.option_mm.env import (  # noqa: E402
    FillModelSpec,
    HestonParams,
    OptionMarketMakingEnv,
)
from applications.option_mm.inventory_variance import (  # noqa: E402
    oracle_heston_estimator,
)
from applications.option_mm.metrics import (  # noqa: E402
    UtilitySpec,
    cara_utility,
    crra_utility,
    paired_ce_posterior,
)


@dataclass(frozen=True)
class Cell:
    label: str
    xi: float
    kappa: float
    horizon_steps: int
    utility_name: str
    utility: UtilitySpec


CELLS = [
    # Strongest from screening: long+slow+high-xi, CARA (P=0.8642)
    Cell("long+slow+high-xi", 1.0, 1.0, 60, "CARA", cara_utility(2.0e-5)),
    # Second: long+slow+high-xi, CRRA (P=0.8443)
    Cell("long+slow+high-xi", 1.0, 1.0, 60, "CRRA", crra_utility(2.0)),
    # Third: long-horizon+high-xi, CARA (P=0.8031)
    Cell("long-horizon+high-xi", 1.0, 2.0, 60, "CARA", cara_utility(2.0e-5)),
]

N_SEEDS = 20_000


def make_env(cell: Cell, seed: int) -> OptionMarketMakingEnv:
    return OptionMarketMakingEnv(
        heston=HestonParams(xi=cell.xi, kappa=cell.kappa),
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=cell.horizon_steps,
        initial_cash=100_000.0,
        seed=seed,
    )


def run_episode(
    env: OptionMarketMakingEnv,
    controller,
) -> float:
    state = env.reset()
    while not state.done:
        state, _, _, _ = env.step(controller(state))
    return state.wealth


def run_cell(cell: Cell) -> dict:
    bg_wealth = np.empty(N_SEEDS)
    oracle_wealth = np.empty(N_SEEDS)

    for seed in range(N_SEEDS):
        env_bg = make_env(cell, seed)
        bg_controller = make_risk_neutral_optimal(env_bg)
        bg_wealth[seed] = run_episode(env_bg, bg_controller)

        env_oracle = make_env(cell, seed)
        oracle_estimator = oracle_heston_estimator(env_oracle)
        oracle_controller = make_linear_inventory_skew(
            env_oracle, oracle_estimator, cell.utility,
        )
        oracle_wealth[seed] = run_episode(env_oracle, oracle_controller)

    posterior = paired_ce_posterior(
        oracle_wealth,
        bg_wealth,
        utility=cell.utility,
        method="delta",
    )
    mc_posterior = paired_ce_posterior(
        oracle_wealth,
        bg_wealth,
        utility=cell.utility,
        method="mc",
        n_draws=10_000,
        rng=np.random.default_rng(42),
    )

    bg_ce = cell.utility.ce(float(np.mean(cell.utility.u(bg_wealth))))
    oracle_ce = cell.utility.ce(float(np.mean(cell.utility.u(oracle_wealth))))

    return {
        "label": cell.label,
        "utility": cell.utility_name,
        "xi": cell.xi,
        "kappa": cell.kappa,
        "horizon": cell.horizon_steps,
        "bg_ce": bg_ce,
        "oracle_ce": oracle_ce,
        "delta_mean": posterior.mean,
        "delta_sd_post": posterior.sd_post,
        "delta_p_positive": posterior.p_positive,
        "delta_ci_low": posterior.ci_low,
        "delta_ci_high": posterior.ci_high,
        "mc_mean": mc_posterior.mean,
        "mc_sd_post": mc_posterior.sd_post,
        "mc_p_positive": mc_posterior.p_positive,
    }


def main() -> int:
    results: list[dict] = []
    for idx, cell in enumerate(CELLS):
        print(
            f"[{idx + 1}/{len(CELLS)}] {cell.label} / {cell.utility_name} "
            f"(xi={cell.xi}, kappa={cell.kappa}, H={cell.horizon_steps}, N={N_SEEDS})"
        )
        result = run_cell(cell)
        results.append(result)
        print(
            f"  ΔCE={result['delta_mean']:.4f}  "
            f"sd_post={result['delta_sd_post']:.4f}  "
            f"P(>0)={result['delta_p_positive']:.6f}  "
            f"95% CrI=[{result['delta_ci_low']:.2f}, {result['delta_ci_high']:.2f}]"
        )
        print(
            f"  MC check: mean={result['mc_mean']:.4f}  "
            f"sd_post={result['mc_sd_post']:.4f}  "
            f"P(>0)={result['mc_p_positive']:.6f}"
        )

    # --- summary ---
    print("\n" + "=" * 90)
    print(f"ORACLE CONFIRMATION — N={N_SEEDS}, oracle linear-skew vs risk-neutral ±1/k")
    print("Decision rule: P(>0) >= 0.95 in at least one cell → channel alive")
    print("=" * 90)

    any_confirmed = False
    for r in results:
        confirmed = r["delta_p_positive"] >= 0.95
        if confirmed:
            any_confirmed = True
        tag = "CONFIRMED" if confirmed else "NOT CONFIRMED"
        print(
            f"\n  [{tag}] {r['label']} / {r['utility']}"
            f"  (xi={r['xi']}, kappa={r['kappa']}, H={r['horizon']})"
        )
        print(
            f"    Risk-neutral CE: {r['bg_ce']:.4f}"
        )
        print(
            f"    Oracle CE: {r['oracle_ce']:.4f}"
        )
        print(
            f"    ΔCE (oracle - BG): {r['delta_mean']:.4f} ± {r['delta_sd_post']:.4f}"
            f"  P(>0) = {r['delta_p_positive']:.6f}"
        )
        print(
            f"    95% CrI: [{r['delta_ci_low']:.2f}, {r['delta_ci_high']:.2f}]"
        )

    print("\n" + "-" * 90)
    if any_confirmed:
        confirmed_cells = [
            r for r in results if r["delta_p_positive"] >= 0.95
        ]
        print(
            f"VERDICT: Channel ALIVE in {len(confirmed_cells)} cell(s). "
            "Proceed to model-free estimator work in confirmed regime(s) only."
        )
    else:
        best = max(results, key=lambda r: r["delta_p_positive"])
        print(
            f"VERDICT: Channel NOT CONFIRMED at P>=0.95. "
            f"Best cell: {best['label']}/{best['utility']} at P={best['delta_p_positive']:.4f}. "
            "Kill the sigma_sq_inv estimation channel for OMM."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
