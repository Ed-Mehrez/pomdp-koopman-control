"""Phase 2: Single-option BBG reduction validation.

Runs the BBG benchmark package on a single-option book to validate
that the solver, env, and controllers are numerically coherent.

Usage:
    python finance/experiments/bbg_single_option_validation.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm_bbg.spec import (
    BBGBenchmarkConfig, BBGHestonSpec, BBGOptionBookSpec, BBGControlSpec,
)
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv, OptionBookMMAction
from applications.option_mm_bbg.solver import (
    solve_bbg_value_function,
    solve_bbg_value_function_no_gap,
    solver_diagnostics,
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
)


def main() -> int:
    out: list[str] = []
    def log(msg=""):
        print(msg)
        out.append(msg)

    log("=" * 60)
    log("  BBG Single-Option Validation")
    log("=" * 60)

    # Use no-gap config for stable validation (a_P = a_Q eliminates nu dependence)
    # Vega limit sized to single-option: 2 fills max = 2 * z * V ~ 6.6M
    config = BBGBenchmarkConfig(
        heston=BBGHestonSpec(kappa_p=3.0, theta_p=0.0225),  # match Q
        book=BBGOptionBookSpec(strikes=(10.0,), maturities=(1.0,)),
        control=BBGControlSpec(vega_limit=5.0e6),  # tighter for 1-option case
    )
    config.validate()
    log(f"\nConfig: K=10, T=1yr, gamma={config.control.gamma}")
    log(f"Horizon: {config.control.horizon} yr, S0={config.heston.spot0}, nu0={config.heston.nu0}")
    log(f"Using no-gap solver (a_P = a_Q)")

    # Solve HJB (no-gap: 2D on (t, V^pi))
    t0 = time.time()
    log("\nSolving HJB (no-gap)...")
    t_grid, vpi_grid, values = solve_bbg_value_function_no_gap(
        config, n_vpi=80, n_time=120,
    )
    diag = solver_diagnostics(values)
    log(f"  Grid: {values.shape}, time: {time.time()-t0:.1f}s")
    log(f"  Value range: [{diag['min']:.2f}, {diag['max']:.2f}]")
    log(f"  Non-finite: {diag['n_nonfinite']}")

    # Build controllers
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, None, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # Run episodes
    n_episodes = 100
    log(f"\nRunning {n_episodes} episodes...")

    def run_episodes(ctrl, label):
        wealths = []
        spreads = []
        for seed in range(n_episodes):
            env = OptionBookMarketMakingEnv(config, seed=seed)
            state = env.reset()
            total_spread = 0.0
            while not state.done:
                action = ctrl(state)
                state, _, _, info = env.step(action)
                total_spread += info["spread_capture"]
            wealths.append(state.wealth)
            spreads.append(total_spread)
        wealths = np.array(wealths)
        spreads = np.array(spreads)
        log(f"\n  {label}:")
        log(f"    mean wealth: {wealths.mean():.4f}")
        log(f"    std wealth:  {wealths.std():.4f}")
        log(f"    mean spread capture: {spreads.mean():.4f}")
        return wealths

    w_bbg = run_episodes(bbg_ctrl, "BBG numerical")
    w_rn = run_episodes(rn_ctrl, "Risk-neutral")

    # Paired comparison
    diff = w_bbg - w_rn
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = std_diff / np.sqrt(n_episodes)
    from scipy.stats import norm
    if se_diff > 0:
        p_positive = float(norm.cdf(mean_diff / se_diff))
    else:
        p_positive = 0.5 if mean_diff == 0 else (1.0 if mean_diff > 0 else 0.0)

    log(f"\n  BBG - RiskNeutral:")
    log(f"    mean diff: {mean_diff:.6f}")
    log(f"    sd_post:   {se_diff:.6f}")
    log(f"    P(>0):     {p_positive:.5f}")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_single_option_validation_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nResults saved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
