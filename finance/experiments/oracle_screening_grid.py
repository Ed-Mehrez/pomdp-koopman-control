"""Oracle screening gate: does inventory skew EVER help across a Heston stress grid?

Cheap diagnostic: for each (xi, kappa, horizon) regime, run the risk-neutral
±1/k controller against the oracle linear-inventory-skew controller (perfect-V
skew) under CRRA(gamma=2) and CARA(alpha=2e-5). If oracle never beats the
risk-neutral limit, the entire sigma_sq_inv estimation channel is dead for OMM
and no amount of signature/kernel work can revive it.

This is a standalone diagnostic — it does not modify the locked Stage 4 v2 runner.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, replace
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
class RegimeSpec:
    xi: float
    kappa: float
    horizon_steps: int
    label: str


REGIMES = [
    # --- default baseline (replicate known result) ---
    RegimeSpec(xi=0.5, kappa=2.0, horizon_steps=20, label="default"),
    # --- higher vol-of-vol: V wanders more, vega changes more ---
    RegimeSpec(xi=1.0, kappa=2.0, horizon_steps=20, label="high-xi"),
    RegimeSpec(xi=1.5, kappa=2.0, horizon_steps=20, label="very-high-xi"),
    # --- slower mean reversion: V persists away from theta ---
    RegimeSpec(xi=0.5, kappa=1.0, horizon_steps=20, label="slow-kappa"),
    RegimeSpec(xi=1.0, kappa=1.0, horizon_steps=20, label="slow-kappa+high-xi"),
    # --- faster mean reversion: V snaps back, frozen estimate is OK ---
    RegimeSpec(xi=0.5, kappa=5.0, horizon_steps=20, label="fast-kappa"),
    RegimeSpec(xi=1.0, kappa=5.0, horizon_steps=20, label="fast-kappa+high-xi"),
    # --- longer horizons: more time for V to wander ---
    RegimeSpec(xi=0.5, kappa=2.0, horizon_steps=60, label="long-horizon"),
    RegimeSpec(xi=1.0, kappa=2.0, horizon_steps=60, label="long-horizon+high-xi"),
    RegimeSpec(xi=1.0, kappa=1.0, horizon_steps=60, label="long+slow+high-xi"),
]

UTILITIES: list[tuple[str, UtilitySpec]] = [
    ("CRRA", crra_utility(2.0)),
    ("CARA", cara_utility(2.0e-5)),
]

N_SEEDS = 5_000
SEED_ENTROPY = 20260410


def make_env(regime: RegimeSpec, seed: int) -> OptionMarketMakingEnv:
    return OptionMarketMakingEnv(
        heston=HestonParams(xi=regime.xi, kappa=regime.kappa),
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=regime.horizon_steps,
        initial_cash=100_000.0,
        seed=seed,
    )


def run_episode_bg(env: OptionMarketMakingEnv) -> float:
    state = env.reset()
    controller = make_risk_neutral_optimal(env)
    while not state.done:
        state, _, _, _ = env.step(controller(state))
    return state.wealth


def run_episode_oracle(
    env: OptionMarketMakingEnv,
    utility: UtilitySpec,
) -> float:
    state = env.reset()
    estimator = oracle_heston_estimator(env)
    controller = make_linear_inventory_skew(env, estimator, utility)
    while not state.done:
        state, _, _, _ = env.step(controller(state))
    return state.wealth


def run_regime(
    regime: RegimeSpec,
    utility_name: str,
    utility: UtilitySpec,
    seeds: list[int],
) -> dict:
    bg_wealth = np.empty(len(seeds))
    oracle_wealth = np.empty(len(seeds))

    for i, seed in enumerate(seeds):
        env_bg = make_env(regime, seed)
        bg_wealth[i] = run_episode_bg(env_bg)

        env_oracle = make_env(regime, seed)
        oracle_wealth[i] = run_episode_oracle(env_oracle, utility)

    posterior = paired_ce_posterior(
        oracle_wealth,
        bg_wealth,
        utility=utility,
        method="delta",
    )

    return {
        "regime": regime.label,
        "utility": utility_name,
        "xi": regime.xi,
        "kappa": regime.kappa,
        "horizon": regime.horizon_steps,
        "bg_ce": utility.ce(float(np.mean(utility.u(bg_wealth)))),
        "oracle_ce": utility.ce(float(np.mean(utility.u(oracle_wealth)))),
        "delta_ce": posterior.mean,
        "sd_post": posterior.sd_post,
        "p_positive": posterior.p_positive,
    }


def main() -> int:
    seeds = list(range(N_SEEDS))

    results: list[dict] = []
    n_total = len(REGIMES) * len(UTILITIES)
    for idx, (regime, (utility_name, utility)) in enumerate(
        itertools.product(REGIMES, UTILITIES)
    ):
        print(
            f"[{idx + 1}/{n_total}] {regime.label} / {utility_name} "
            f"(xi={regime.xi}, kappa={regime.kappa}, H={regime.horizon_steps})"
        )
        result = run_regime(regime, utility_name, utility, seeds)
        results.append(result)

        tag = "WIN" if result["p_positive"] >= 0.95 else (
            "LEAN" if result["p_positive"] >= 0.80 else "LOSE"
        )
        print(
            f"  {tag}  oracle-risk-neutral: "
            f"ΔCE={result['delta_ce']:.4f}  "
            f"sd_post={result['sd_post']:.4f}  "
            f"P(>0)={result['p_positive']:.4f}"
        )

    # --- summary table ---
    print("\n" + "=" * 100)
    print("ORACLE SCREENING GRID — oracle linear-skew vs risk-neutral ±1/k")
    print("Question: does perfect-V inventory skew EVER beat the risk-neutral limit?")
    print("=" * 100)
    print(
        f"{'regime':<25s} {'util':<6s} {'xi':>4s} {'κ':>4s} "
        f"{'H':>3s} {'ΔCE':>10s} {'sd_post':>10s} {'P(>0)':>8s} {'verdict':>7s}"
    )
    print("-" * 100)

    any_win = False
    for r in results:
        if r["p_positive"] >= 0.95:
            verdict = "WIN"
            any_win = True
        elif r["p_positive"] >= 0.80:
            verdict = "LEAN"
        elif r["p_positive"] <= 0.05:
            verdict = "LOSE*"
        elif r["p_positive"] <= 0.20:
            verdict = "LOSE"
        else:
            verdict = "NULL"
        print(
            f"{r['regime']:<25s} {r['utility']:<6s} {r['xi']:>4.1f} {r['kappa']:>4.1f} "
            f"{r['horizon']:>3d} {r['delta_ce']:>10.4f} {r['sd_post']:>10.4f} "
            f"{r['p_positive']:>8.4f} {verdict:>7s}"
        )

    print("-" * 100)
    if any_win:
        print(
            "RESULT: Oracle beats BG in at least one regime. "
            "The sigma_sq_inv channel is alive — proceed to model-free estimator work."
        )
    else:
        lean_count = sum(1 for r in results if r["p_positive"] >= 0.80)
        if lean_count > 0:
            print(
                f"RESULT: No decisive oracle win, but {lean_count} regimes lean positive. "
                "Consider targeted follow-up at higher N or higher risk aversion."
            )
        else:
            print(
                "RESULT: Oracle never beats BG. "
                "The inventory-skew correction is net value-destroying across all regimes tested. "
                "Kill the sigma_sq_inv estimation channel for OMM."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
