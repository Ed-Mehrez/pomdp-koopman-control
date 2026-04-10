"""BBG 3D solver grid-sensitivity diagnostics.

Runs the full 3D solver at coarse/medium/fine grids and reports value ranges,
stability, and representative controller behavior.
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


GRIDS = {
    "coarse": (30, 10, 20),
    "medium": (60, 15, 30),
    "fine":   (120, 20, 40),
}


def main() -> int:
    out: list[str] = []
    def log(msg=""):
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    log("=" * 70)
    log("  BBG 3D Solver Grid-Sensitivity Diagnostics")
    log("=" * 70)

    results = {}
    for label, (n_t, n_nu, n_vpi) in GRIDS.items():
        log(f"\n--- {label} grid: (n_time={n_t}, n_nu={n_nu}, n_vpi={n_vpi}) ---")
        t0 = time.time()
        t_grid, nu_grid, vpi_grid, vals = solve_bbg_value_function(
            config, n_nu=n_nu, n_vpi=n_vpi, n_time=n_t,
        )
        elapsed = time.time() - t0
        diag = solver_diagnostics(vals)
        log(f"  Runtime: {elapsed:.1f}s")
        log(f"  Value range: [{diag['min']:.0f}, {diag['max']:.0f}]")
        log(f"  Non-finite: {diag['n_nonfinite']}")
        log(f"  t=0 range: [{diag['t0_range'][0]:.0f}, {diag['t0_range'][1]:.0f}]")
        log(f"  Terminal max|v|: {diag['terminal_max_abs']:.6f}")

        # Representative quotes from controller
        ctrl = make_bbg_numerical_controller(config, vals, t_grid, nu_grid, vpi_grid)
        env = OptionBookMarketMakingEnv(config, seed=0)
        state = env.reset()
        action = ctrl(state)

        bid_finite = action.bid_distances[action.bid_distances < 1e5]
        ask_finite = action.ask_distances[action.ask_distances < 1e5]
        n_censored_bid = int(np.sum(action.bid_distances >= 1e5))
        n_censored_ask = int(np.sum(action.ask_distances >= 1e5))

        log(f"  Quotes at t=0, V^pi=0:")
        log(f"    Finite bid dists: min={bid_finite.min():.4f}, max={bid_finite.max():.4f}, n={len(bid_finite)}")
        log(f"    Finite ask dists: min={ask_finite.min():.4f}, max={ask_finite.max():.4f}, n={len(ask_finite)}")
        log(f"    Censored: bid={n_censored_bid}, ask={n_censored_ask} of {config.book.n_options}")

        # Run 20 episodes to check controller wealth
        wealths = []
        for seed in range(20):
            env = OptionBookMarketMakingEnv(config, seed=seed)
            state = env.reset()
            while not state.done:
                state, _, _, _ = env.step(ctrl(state))
            wealths.append(state.wealth)
        w = np.array(wealths)
        log(f"  20-episode wealth: mean={w.mean():.0f}, std={w.std():.0f}")

        results[label] = {
            "diag": diag,
            "mean_wealth": float(w.mean()),
            "bid_range": (float(bid_finite.min()), float(bid_finite.max())),
            "n_censored": n_censored_bid + n_censored_ask,
        }

    # Cross-grid comparison
    log(f"\n--- Cross-Grid Summary ---")
    log(f"{'Grid':<10s} {'v_min':>10s} {'v_max':>10s} {'wealth':>10s} {'bid_lo':>8s} {'bid_hi':>8s} {'cens':>5s}")
    for label in GRIDS:
        r = results[label]
        d = r["diag"]
        log(f"{label:<10s} {d['min']:>10.0f} {d['max']:>10.0f} {r['mean_wealth']:>10.0f} "
            f"{r['bid_range'][0]:>8.4f} {r['bid_range'][1]:>8.4f} {r['n_censored']:>5d}")

    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    rf = results_dir / f"bbg_solver_diagnostics_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
