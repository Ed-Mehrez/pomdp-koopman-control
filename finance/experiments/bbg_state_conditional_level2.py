"""De-cheated Level-2 controller experiment.

Tests whether STATE CONDITIONING of the reduced-space control coefficients
(rev_lin_r, rev_quad_diag_r, vega_r) adds value over the existing global
SDRE recovery, using exploration-only training. NO BBG ACTIONS are used as
supervision.

Baselines (ranked by role):
  - risk_neutral        (floor; simple MM spread)
  - sdre_global         (PRIMARY A/B; current make_sdre_recovery_controller)
  - sdre_state_cond     (candidate; new state-conditional head)
  - bbg_numerical       (ORACLE REFERENCE ONLY; solved HJB — not a training
                          target, shown for top-end comparison)

Modes:
  smoke : 200 explore, 50 pilot + 50 eval seeds  (code smoke + sanity)
  dev   : 300 explore, 100 pilot + 100 eval seeds  (pattern diagnosis)
  formal: 500 explore, pilot derives N via power calc  (claims; gated)

Usage:
    python finance/experiments/bbg_state_conditional_level2.py smoke
    python finance/experiments/bbg_state_conditional_level2.py dev
    python finance/experiments/bbg_state_conditional_level2.py formal
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from applications.option_mm.metrics import cara_utility, paired_ce_posterior
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv
from applications.option_mm_bbg.sdre_recovery import (
    SDRERecoveryConfig,
    _compute_rn_distances,
    make_sdre_recovery_controller,
)
from applications.option_mm_bbg.solver import (
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
    solve_bbg_value_function,
)
from applications.option_mm_bbg.spec import BBGBenchmarkConfig
from applications.option_mm_bbg.state_conditional_sdre import (
    StateConditionalConfig,
    make_state_conditional_controller,
    extract_state_rich_extended,
)


# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeConfig:
    name: str
    n_explore: int
    n_eval: int
    n_pilot: int
    rank: int


MODES = {
    "smoke":  ModeConfig("smoke",  n_explore=200, n_eval=50,  n_pilot=50,  rank=3),
    "dev":    ModeConfig("dev",    n_explore=300, n_eval=100, n_pilot=100, rank=3),
    "formal": ModeConfig("formal", n_explore=500, n_eval=400, n_pilot=100, rank=3),
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episodes(config, ctrl, label, seeds, log):
    """Run one episode per seed and return (wealths, aux)."""
    wealths, spreads, vega_mag = [], [], []
    t0 = time.time()
    for seed in seeds:
        env = OptionBookMarketMakingEnv(config, seed=int(seed))
        state = env.reset()
        spread_sum, vega_abs, n_steps = 0.0, 0.0, 0
        while not state.done:
            action = ctrl(state)
            state, _, _, info = env.step(action)
            spread_sum += info["spread_capture"]
            vega_abs += abs(state.portfolio_vega)
            n_steps += 1
        wealths.append(state.wealth)
        spreads.append(spread_sum)
        vega_mag.append(vega_abs / max(n_steps, 1))
    w = np.asarray(wealths)
    elapsed = time.time() - t0
    log(f"  {label:<28s} n={len(seeds):3d}  "
        f"mean_W={w.mean():9.0f}  sd_W={w.std():8.0f}  "
        f"spread={np.mean(spreads):8.0f}  |vega|={np.mean(vega_mag):6.0f}  "
        f"({elapsed:.1f}s)")
    return w


def cara_ce_point(w, gamma):
    util = cara_utility(gamma)
    return float(util.ce(float(np.mean(util.u(w)))))


def paired_report(label, w_a, w_b, gamma, log):
    post = paired_ce_posterior(w_a, w_b, gamma=gamma, method="delta")
    log(f"  {label:<34s} mean={post.mean:9.0f}  "
        f"sd={post.sd_post:8.0f}  "
        f"P(>0)={post.p_positive:.4f}  "
        f"95% CrI=[{post.ci_low:9.0f}, {post.ci_high:9.0f}]")
    return post


# ---------------------------------------------------------------------------
# Head diagnostics: uncertainty distribution on eval states
# ---------------------------------------------------------------------------


def head_uncertainty_audit(config, head, eval_seeds, log):
    """Compute per-state predictive variance over the eval distribution."""
    spreads_var, vega_var = [], []
    for seed in eval_seeds[: min(20, len(eval_seeds))]:
        env = OptionBookMarketMakingEnv(config, seed=int(seed))
        state = env.reset()
        while not state.done:
            z = extract_state_rich_extended(state, config)
            v = head.predict_variance(z)
            spreads_var.append(v["spread_var"])
            vega_var.append(v["vega_var"])
            # Use RN action to advance (we just need trajectory through
            # state space, exact action is not critical for diagnostics)
            state, _, _, _ = env.step(env.null_action() if hasattr(env, "null_action")
                                       else _step_dummy(state, config))
    s = np.asarray(spreads_var)
    v = np.asarray(vega_var)
    log(f"  head variance (spread): "
        f"median={np.median(s):.2e}  p90={np.percentile(s, 90):.2e}")
    log(f"  head variance (vega):   "
        f"median={np.median(v):.2e}  p90={np.percentile(v, 90):.2e}")


def _step_dummy(state, config):
    """Fallback advance action for audit walks; uses RN-like quotes."""
    from applications.option_mm_bbg.env import OptionBookMMAction
    return OptionBookMMAction(
        bid_distances=np.full(config.book.n_options, 0.2),
        ask_distances=np.full(config.book.n_options, 0.2),
        hedge_trade=-state.net_delta,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=list(MODES), default="smoke", nargs="?")
    parser.add_argument("--include-bbg-oracle", action="store_true",
                        help="solve BBG HJB and include as oracle reference")
    parser.add_argument("--rank", type=int, default=None)
    args = parser.parse_args()

    mode = MODES[args.mode]
    rank = args.rank if args.rank is not None else mode.rank

    out: list[str] = []
    def log(msg=""):
        print(msg)
        out.append(msg)

    config = BBGBenchmarkConfig.paper_default()
    gamma = config.control.gamma

    log("=" * 78)
    log(f"  BBG State-Conditional Level-2  ({mode.name})")
    log("=" * 78)
    log(f"  Options: {config.book.n_options}  gamma: {gamma}")
    log(f"  Exploration: {mode.n_explore}  Eval: {mode.n_eval}  rank: {rank}")
    log(f"  BBG is evaluation-only (no action supervision).")

    rn_dists = _compute_rn_distances(config)

    # -- Train controllers -------------------------------------------------
    log("\nTraining global SDRE (primary baseline)...")
    t0 = time.time()
    sdre_ctrl, sdre_model = make_sdre_recovery_controller(
        config,
        SDRERecoveryConfig(
            method="bilinear_2stage", rank=rank,
            n_explore_episodes=mode.n_explore,
            bilinear_overspace=10, explore_seed=42,
        ),
        rn_distances=rn_dists,
        return_model=True,
    )
    log(f"  trained in {time.time() - t0:.1f}s")
    ev = sdre_model.explained_variance()
    log(f"  singular values (top 5): "
        f"{np.array2string(sdre_model.S_r, precision=3)}")
    log(f"  cumulative EV at rank {rank}: {ev[:rank].sum():.4f}")

    log("\nTraining state-conditional SDRE (candidate)...")
    t0 = time.time()
    sc_ctrl, sc_diag = make_state_conditional_controller(
        config,
        StateConditionalConfig(
            rank=rank,
            n_explore_episodes=mode.n_explore,
            explore_seed=42,
            basis_method="bilinear_2stage",
            bilinear_overspace=10,
        ),
        rn_distances=rn_dists,
        return_model=True,
    )
    log(f"  trained in {time.time() - t0:.1f}s")
    h = sc_diag["head"]
    log(f"  head fit (N={sc_diag['n_transitions']}, test={sc_diag['n_test']}):")
    log(f"    spread R^2  train={h.spread_r2:.4f}  test={h.spread_r2_test:.4f}")
    log(f"    vega   R^2  train={h.vega_r2:.4f}   test={h.vega_r2_test:.4f}")

    rn_ctrl = make_bbg_risk_neutral_controller(config)

    bbg_ctrl = None
    if args.include_bbg_oracle:
        log("\nSolving BBG HJB (ORACLE REFERENCE ONLY — not used in training)...")
        t0 = time.time()
        t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
            config, n_nu=20, n_vpi=40, n_time=120,
        )
        bbg_ctrl = make_bbg_numerical_controller(
            config, values, t_grid, nu_grid, vpi_grid,
        )
        log(f"  HJB solved in {time.time() - t0:.1f}s")

    # -- Evaluate ----------------------------------------------------------
    eval_seed_rng = np.random.default_rng(10_000)
    eval_seeds = eval_seed_rng.integers(100_000, 1_000_000, size=mode.n_eval)

    log(f"\nEvaluating on {mode.n_eval} held-out seeds...")
    w_rn = run_episodes(config, rn_ctrl, "risk_neutral (floor)", eval_seeds, log)
    w_sdre = run_episodes(config, sdre_ctrl, "sdre_global (primary A/B)",
                          eval_seeds, log)
    w_sc = run_episodes(config, sc_ctrl, "sdre_state_cond (candidate)",
                        eval_seeds, log)
    w_bbg = None
    if bbg_ctrl is not None:
        w_bbg = run_episodes(config, bbg_ctrl, "bbg_numerical (ORACLE REF)",
                             eval_seeds, log)

    # -- Paired posteriors -------------------------------------------------
    log("\n" + "-" * 78)
    log("Paired CE posteriors (CARA, delta method)")
    log("-" * 78)

    # Primary contrast: candidate vs global SDRE
    paired_report("state_cond - sdre_global [PRIMARY]", w_sc, w_sdre, gamma, log)
    # Floor checks
    paired_report("state_cond - risk_neutral",          w_sc,   w_rn,   gamma, log)
    paired_report("sdre_global - risk_neutral",         w_sdre, w_rn,   gamma, log)
    if w_bbg is not None:
        paired_report("sdre_global - bbg_numerical [ORACLE]",
                      w_sdre, w_bbg, gamma, log)
        paired_report("state_cond  - bbg_numerical [ORACLE]",
                      w_sc,   w_bbg, gamma, log)

    # -- Summary table -----------------------------------------------------
    log("\n" + "-" * 78)
    log("CE summary")
    log("-" * 78)
    log(f"  {'controller':<30s} {'CE':>12s}  {'mean W':>12s}  {'sd W':>12s}")
    for label, w in (
        ("risk_neutral",         w_rn),
        ("sdre_global",          w_sdre),
        ("sdre_state_cond",      w_sc),
    ) + (
        (("bbg_numerical (REF)", w_bbg),) if w_bbg is not None else ()
    ):
        log(f"  {label:<30s} {cara_ce_point(w, gamma):>12.0f}  "
            f"{w.mean():>12.0f}  {w.std():>12.0f}")

    # -- Head uncertainty audit on dev+ ------------------------------------
    if mode.name in {"dev", "formal"}:
        log("\n" + "-" * 78)
        log("Head predictive-variance distribution over eval states")
        log("-" * 78)
        try:
            head_uncertainty_audit(config, h, eval_seeds, log)
        except Exception as e:  # noqa: BLE001
            log(f"  [audit skipped: {e}]")

    # -- Notes / discipline reminders --------------------------------------
    log("\n" + "-" * 78)
    log("Notes")
    log("-" * 78)
    log("  - Primary claim target: state_cond - sdre_global CE contrast.")
    log("  - BBG is oracle reference (training uses NO BBG actions).")
    log("  - Per-step head R^2 is small by construction: Poisson arrival noise")
    log("    dominates per-step spread. Controller quality is the metric, not R^2.")
    if mode.name != "formal":
        log(f"  - {mode.name!r} mode: dev/smoke. Do NOT cite as a formal claim.")

    # -- Save --------------------------------------------------------------
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"bbg_state_conditional_level2_{mode.name}_{date.today()}.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(out))
    log(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
