"""Phase 3A: Full multi-option BBG benchmark validation.

Paper-default 20-option book, full 3D HJB solver, BBG numerical vs risk-neutral.

Usage:
    python finance/experiments/bbg_multi_option_benchmark.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from applications.option_mm_bbg.spec import BBGBenchmarkConfig
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv
from applications.option_mm_bbg.solver import (
    solve_bbg_value_function,
    solver_diagnostics,
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
)


def main() -> int:
    out: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()

    log("=" * 70)
    log("  Full Multi-Option BBG Benchmark (Paper Default)")
    log("=" * 70)
    log(f"  Options: {config.book.n_options}")
    log(f"  Gamma: {config.control.gamma}")
    log(f"  Horizon: {config.control.horizon} yr")
    log(f"  Vega limit: {config.control.vega_limit:.0e}")

    # Solve HJB
    n_nu, n_vpi, n_time = 20, 40, 120
    t0 = time.time()
    log(f"\nSolving 3D HJB (grid {n_time}x{n_nu}x{n_vpi})...")
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=n_nu, n_vpi=n_vpi, n_time=n_time,
    )
    solve_time = time.time() - t0
    diag = solver_diagnostics(values)
    log(f"  Time: {solve_time:.1f}s")
    log(f"  Value range: [{diag['min']:.0f}, {diag['max']:.0f}]")
    log(f"  Non-finite: {diag['n_nonfinite']}")
    log(f"  t=0 range: [{diag['t0_range'][0]:.0f}, {diag['t0_range'][1]:.0f}]")

    # Build controllers
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    rn_ctrl = make_bbg_risk_neutral_controller(config)

    # Run episodes
    n_episodes = 200
    log(f"\nRunning {n_episodes} episodes...")

    def run_episodes(ctrl, label):
        wealths, spreads, avg_vega, avg_delta = [], [], [], []
        t1 = time.time()
        for seed in range(n_episodes):
            env = OptionBookMarketMakingEnv(config, seed=seed)
            state = env.reset()
            total_spread = 0.0
            vega_abs_sum, delta_abs_sum, n_steps = 0.0, 0.0, 0
            while not state.done:
                action = ctrl(state)
                state, _, _, info = env.step(action)
                total_spread += info["spread_capture"]
                vega_abs_sum += abs(state.portfolio_vega)
                delta_abs_sum += abs(state.net_delta)
                n_steps += 1
            wealths.append(state.wealth)
            spreads.append(total_spread)
            avg_vega.append(vega_abs_sum / max(n_steps, 1))
            avg_delta.append(delta_abs_sum / max(n_steps, 1))

        elapsed = time.time() - t1
        w = np.array(wealths)
        s = np.array(spreads)
        v = np.array(avg_vega)
        d = np.array(avg_delta)
        log(f"\n  {label} ({elapsed:.1f}s):")
        log(f"    wealth:  mean={w.mean():.2f}, std={w.std():.2f}")
        log(f"    spread:  mean={s.mean():.2f}")
        log(f"    |vega|:  mean={v.mean():.0f}")
        log(f"    |delta|: mean={d.mean():.2f}")
        return w

    w_bbg = run_episodes(bbg_ctrl, "BBG numerical")
    w_rn = run_episodes(rn_ctrl, "Risk-neutral")

    # Paired comparison
    diff = w_bbg - w_rn
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    se_d = std_d / np.sqrt(n_episodes)
    from scipy.stats import norm
    p_pos = float(norm.cdf(mean_d / se_d)) if se_d > 0 else 0.5

    log(f"\n  BBG numerical - Risk-neutral:")
    log(f"    mean = {mean_d:.2f}")
    log(f"    sd_post = {se_d:.2f}")
    log(f"    P(>0) = {p_pos:.5f}")
    log(f"    95% CrI = [{mean_d - 1.96*se_d:.2f}, {mean_d + 1.96*se_d:.2f}]")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_multi_option_benchmark_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
