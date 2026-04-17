"""Full multi-option BBG benchmark with risk-adjusted metrics.

Paper-default 20-option book, full 3D HJB solver, BBG numerical vs risk-neutral.
Reports CARA certainty equivalent and mean-variance surrogate in addition to
raw wealth, providing a benchmark-consistent comparison.

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


# ---------------------------------------------------------------------------
# Risk-adjusted metrics
# ---------------------------------------------------------------------------


def cara_ce(wealths: np.ndarray, gamma: float) -> float:
    """CARA certainty equivalent: CE = -1/gamma * ln(E[exp(-gamma * W)]).

    Uses log-sum-exp trick for numerical stability.
    """
    if gamma == 0.0:
        return float(np.mean(wealths))
    neg_gamma_w = -gamma * wealths
    max_val = float(np.max(neg_gamma_w))
    log_mean_exp = max_val + np.log(np.mean(np.exp(neg_gamma_w - max_val)))
    return -log_mean_exp / gamma


def mean_var_surrogate(wealths: np.ndarray, gamma: float) -> float:
    """Mean-variance surrogate: E[W] - (gamma/2) Var(W)."""
    return float(np.mean(wealths) - 0.5 * gamma * np.var(wealths))


def bootstrap_ce_diff(
    w_a: np.ndarray,
    w_b: np.ndarray,
    gamma: float,
    n_boot: int = 10_000,
    seed: int = 999,
) -> dict:
    """Bootstrap posterior for CE_a - CE_b using paired resampling."""
    rng = np.random.default_rng(seed)
    n = len(w_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = cara_ce(w_a[idx], gamma) - cara_ce(w_b[idx], gamma)
    return {
        "mean": float(np.mean(diffs)),
        "sd_post": float(np.std(diffs)),
        "p_pos": float(np.mean(diffs > 0)),
        "ci_95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
    }


def main() -> int:
    out: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma

    log("=" * 70)
    log("  Full Multi-Option BBG Benchmark (Paper Default)")
    log("=" * 70)
    log(f"  Options: {config.book.n_options}")
    log(f"  Gamma: {gamma}")
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
        ce = cara_ce(w, gamma)
        mv = mean_var_surrogate(w, gamma)
        log(f"    CARA CE (gamma={gamma}): {ce:.2f}")
        log(f"    mean-var surrogate:      {mv:.2f}")
        return w

    w_bbg = run_episodes(bbg_ctrl, "BBG numerical")
    w_rn = run_episodes(rn_ctrl, "Risk-neutral")

    # --- Raw wealth paired comparison ---
    diff = w_bbg - w_rn
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    se_d = std_d / np.sqrt(n_episodes)
    from scipy.stats import norm
    p_pos = float(norm.cdf(mean_d / se_d)) if se_d > 0 else 0.5

    log(f"\n  === Raw wealth: BBG numerical - Risk-neutral ===")
    log(f"    mean = {mean_d:.2f}")
    log(f"    sd_post = {se_d:.2f}")
    log(f"    P(>0) = {p_pos:.5f}")
    log(f"    95% CrI = [{mean_d - 1.96*se_d:.2f}, {mean_d + 1.96*se_d:.2f}]")

    # --- CARA CE paired comparison (bootstrap) ---
    log(f"\n  === CARA CE: BBG numerical - Risk-neutral (gamma={gamma}) ===")
    boot = bootstrap_ce_diff(w_bbg, w_rn, gamma)
    log(f"    mean = {boot['mean']:.2f}")
    log(f"    sd_post = {boot['sd_post']:.2f}")
    log(f"    P(>0) = {boot['p_pos']:.5f}")
    log(f"    95% CrI = [{boot['ci_95'][0]:.2f}, {boot['ci_95'][1]:.2f}]")

    # --- Mean-variance surrogate ---
    mv_bbg = mean_var_surrogate(w_bbg, gamma)
    mv_rn = mean_var_surrogate(w_rn, gamma)
    log(f"\n  === Mean-variance surrogate (gamma={gamma}) ===")
    log(f"    BBG numerical: {mv_bbg:.2f}")
    log(f"    Risk-neutral:  {mv_rn:.2f}")
    log(f"    Difference:    {mv_bbg - mv_rn:.2f}")

    # --- Interpretation ---
    ce_bbg = cara_ce(w_bbg, gamma)
    ce_rn = cara_ce(w_rn, gamma)
    log(f"\n  === Interpretation ===")
    if boot["p_pos"] > 0.95:
        log(f"    BBG WINS under CARA CE: BBG earns {boot['mean']:.0f} more CE-EUR")
        log(f"    The variance reduction from risk management dominates the lower mean wealth.")
    elif boot["p_pos"] < 0.05:
        log(f"    BBG LOSES under CARA CE: RN earns {-boot['mean']:.0f} more CE-EUR")
        log(f"    Even with risk adjustment, variance reduction does not compensate.")
    else:
        log(f"    INCONCLUSIVE under CARA CE (P(>0)={boot['p_pos']:.3f})")
        log(f"    The risk-adjusted difference is not clearly signed.")

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
