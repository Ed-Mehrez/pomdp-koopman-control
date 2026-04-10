"""Prior-free local bilinear value-gradient recovery benchmark.

The candidate learns local value gradients via backward recursion on a
CQ transfer model, then extracts actions from the local quadratic
Hamiltonian.  No BBG prior inside the candidate.

Usage:
    python finance/experiments/option_mm_local_value_gradient.py [--pilot]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.bbg_solver import make_bbg_numerical  # noqa: E402
from applications.option_mm.controllers import (  # noqa: E402
    make_linear_inventory_skew,
    make_risk_neutral_optimal,
)
from applications.option_mm.env import (  # noqa: E402
    FillModelSpec,
    OptionMMAction,
    OptionMMState,
    OptionMarketMakingEnv,
)
from applications.option_mm.inventory_variance import oracle_heston_estimator  # noqa: E402
from applications.option_mm.local_value_bilinear import (  # noqa: E402
    collect_episode_data,
    compute_value_gradients,
    make_instrumented_value_gradient_controller,
    make_local_value_gradient_controller,
    train_value_gradient_controller_data,
)
from applications.option_mm.local_bilinear_controller import (  # noqa: E402
    collect_bilinear_training_data,
)
from applications.option_mm.metrics import (  # noqa: E402
    EpisodeSummary,
    UtilitySpec,
    aggregate_episode_summaries,
    crra_utility,
    paired_ce_posterior,
    summarize_episode,
)


@dataclass(frozen=True)
class VGConfig:
    seed_sequence_entropy: int = 20260410
    n_train_episodes: int = 200
    n_test_episodes: int = 50
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    gamma_ce: float = 2.0
    max_inventory: int = 10
    width_range: tuple[float, float] = (0.10, 0.30)
    skew_range: tuple[float, float] = (-0.05, 0.05)
    exploration_rng_seed: int = 42
    ridge: float = 1e-3
    lambda_q: float = 0.0


def run_episode(
    strategy: str, seed: int, config: VGConfig, utility: UtilitySpec,
    *, episodes=None, betas=None, bandwidth=None,
    episodes_perm=None, betas_perm=None,
    fixed_w=0.20, fixed_s=0.0,
    choices_acc=None,
):
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()
    local_choices = None

    if strategy == "risk_neutral_optimal":
        ctrl = make_risk_neutral_optimal(env)
    elif strategy == "bbg_numerical":
        gamma = utility.arrow_pratt(state.wealth)
        ctrl = make_bbg_numerical(env, state, gamma=gamma, max_inventory=config.max_inventory)
    elif strategy == "linear_inventory_skew":
        est = oracle_heston_estimator(env)
        ctrl = make_linear_inventory_skew(env, est, utility)
    elif strategy == "global_fixed_action":
        def ctrl(st, h=None):
            bd = fixed_w + fixed_s
            ad = fixed_w - fixed_s
            return OptionMMAction(
                bid_price=max(st.option_mid - bd, 0.0),
                ask_price=st.option_mid + max(ad, 0.0),
                hedge_trade=-st.net_delta,
            )
    elif strategy == "local_value_gradient":
        if choices_acc is not None:
            ctrl, local_choices = make_instrumented_value_gradient_controller(
                env, episodes, betas, state,
                gamma_ce=config.gamma_ce, lambda_q=config.lambda_q,
                bandwidth=bandwidth, ridge=config.ridge,
                width_range=config.width_range, skew_range=config.skew_range,
            )
        else:
            ctrl = make_local_value_gradient_controller(
                env, episodes, betas, state,
                gamma_ce=config.gamma_ce, lambda_q=config.lambda_q,
                bandwidth=bandwidth, ridge=config.ridge,
                width_range=config.width_range, skew_range=config.skew_range,
            )
    elif strategy == "permuted_null":
        ctrl = make_local_value_gradient_controller(
            env, episodes_perm, betas_perm, state,
            gamma_ce=config.gamma_ce, lambda_q=config.lambda_q,
            bandwidth=bandwidth, ridge=config.ridge,
            width_range=config.width_range, skew_range=config.skew_range,
        )
    else:
        raise ValueError(strategy)

    states = [state]
    infos = []
    while not state.done:
        action = ctrl(state)
        state, _, _, info = env.step(action)
        states.append(state)
        infos.append(info)

    if local_choices is not None and choices_acc is not None:
        choices_acc.extend(local_choices)
    return states, infos


def run_strategy(name, seeds, config, utility, **kwargs):
    summaries = []
    for seed in seeds:
        states, infos = run_episode(name, seed, config, utility, **kwargs)
        summaries.append(summarize_episode(
            states=states, infos=infos, inventory_limit=config.max_inventory,
        ))
    return summaries


def paired_post(a, b, utility):
    wa = [s.terminal_wealth for s in a]
    wb = [s.terminal_wealth for s in b]
    return paired_ce_posterior(wa, wb, utility=utility, method="delta")


def ce_from(summaries, utility):
    w = np.array([s.terminal_wealth for s in summaries])
    return utility.ce(float(np.mean(utility.u(w))))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)

    config = VGConfig()
    if args.pilot:
        config = VGConfig(n_train_episodes=50, n_test_episodes=30)

    utility = crra_utility(config.gamma_ce)
    ss = np.random.SeedSequence(config.seed_sequence_entropy)
    child = ss.spawn(config.n_train_episodes + config.n_test_episodes)
    all_ints = [int(cs.generate_state(1)[0]) for cs in child]
    train_seeds = all_ints[:config.n_train_episodes]
    test_seeds = all_ints[config.n_train_episodes:]

    out = []
    def log(msg=""):
        print(msg)
        out.append(msg)

    log("=" * 70)
    log("  VALUE-GRADIENT RECOVERY: Prior-Free CQ-KRONIC + Backward Recursion")
    log("=" * 70)
    log(f"Train: {config.n_train_episodes} eps  |  Test: {config.n_test_episodes} eps")

    # Train
    t0 = time.time()
    log("\n--- Training ---")
    episodes, betas, bw = train_value_gradient_controller_data(
        training_seeds=train_seeds,
        utility_u=utility.u,
        gamma_ce=config.gamma_ce,
        initial_cash=config.initial_cash,
        lambda_q=config.lambda_q,
        horizon_steps=config.horizon_steps,
        width_range=config.width_range,
        skew_range=config.skew_range,
        exploration_rng_seed=config.exploration_rng_seed,
        ridge=config.ridge,
    )
    log(f"  Episodes: {len(episodes)}, bandwidth: {bw:.3f}")
    log(f"  Beta norms: " + ", ".join(f"t{t}={np.linalg.norm(b):.2f}" for t, b in enumerate(betas[:5])))
    log(f"  Train time: {time.time()-t0:.1f}s")

    # Global fixed-action baseline
    buf = collect_bilinear_training_data(
        seeds=train_seeds, horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        width_range=config.width_range, skew_range=config.skew_range,
        exploration_rng_seed=config.exploration_rng_seed,
    )
    U = np.array(buf.u_list)
    dW = np.array(buf.dw_list)
    n_bins = 10
    w_edges = np.linspace(config.width_range[0], config.width_range[1], n_bins + 1)
    s_edges = np.linspace(config.skew_range[0], config.skew_range[1], n_bins + 1)
    best_w, best_s, best_mean = 0.20, 0.0, -np.inf
    for i in range(n_bins):
        for j in range(n_bins):
            mask = ((U[:,0]>=w_edges[i])&(U[:,0]<w_edges[i+1])
                    &(U[:,1]>=s_edges[j])&(U[:,1]<s_edges[j+1]))
            if mask.sum() >= 3:
                m = float(np.mean(dW[mask]))
                if m > best_mean:
                    best_mean, best_w, best_s = m, (w_edges[i]+w_edges[i+1])/2, (s_edges[j]+s_edges[j+1])/2
    log(f"  Global best fixed: w={best_w:.3f}, s={best_s:.3f}, mean_dW={best_mean:.2f}")

    # Permuted null
    log("  Building permuted-null...")
    eps_perm, betas_perm, _ = train_value_gradient_controller_data(
        training_seeds=train_seeds,
        utility_u=utility.u,
        gamma_ce=config.gamma_ce,
        initial_cash=config.initial_cash,
        lambda_q=config.lambda_q,
        horizon_steps=config.horizon_steps,
        width_range=config.width_range,
        skew_range=config.skew_range,
        exploration_rng_seed=config.exploration_rng_seed + 999,  # different exploration
        ridge=config.ridge,
        bandwidth=bw,
    )
    # Shuffle the terminal wealth for permuted value gradients
    for ep in eps_perm:
        ep_rng = np.random.default_rng(hash(ep.terminal_wealth) & 0xFFFFFFFF)
        ep.terminal_wealth = config.initial_cash + ep_rng.normal(0, 100)
    betas_perm = compute_value_gradients(
        eps_perm, utility.u, config.gamma_ce / config.initial_cash,
        config.lambda_q, bw, config.ridge, config.horizon_steps,
    )

    # Evaluate
    log("\n--- Evaluation ---")
    strategies = [
        "risk_neutral_optimal", "bbg_numerical", "linear_inventory_skew",
        "global_fixed_action", "local_value_gradient", "permuted_null",
    ]
    results = {}
    vg_choices = []
    for strat in strategies:
        t1 = time.time()
        kw = {}
        if strat == "global_fixed_action":
            kw["fixed_w"] = best_w
            kw["fixed_s"] = best_s
        elif strat == "local_value_gradient":
            kw["episodes"] = episodes
            kw["betas"] = betas
            kw["bandwidth"] = bw
            kw["choices_acc"] = vg_choices
        elif strat == "permuted_null":
            kw["episodes_perm"] = eps_perm
            kw["betas_perm"] = betas_perm
            kw["bandwidth"] = bw
        results[strat] = run_strategy(strat, test_seeds, config, utility, **kw)
        log(f"  {strat}: {time.time()-t1:.1f}s")

    # Results
    log("\n" + "=" * 70)
    log("  RESULTS")
    log("=" * 70)
    log(f"\n  {'Strategy':<35s} {'CE':>12s} {'Spread Cap':>12s}")
    log("  " + "-" * 61)
    ce = {}
    for name, sums in results.items():
        agg = aggregate_episode_summaries(sums)
        ce[name] = ce_from(sums, utility)
        log(f"  {name:<35s} {ce[name]:>12.3f} {agg.gross_spread_capture_mean:>12.3f}")

    contrasts = [
        ("local_value_gradient", "linear_inventory_skew"),
        ("local_value_gradient", "bbg_numerical"),
        ("local_value_gradient", "risk_neutral_optimal"),
        ("local_value_gradient", "global_fixed_action"),
        ("local_value_gradient", "permuted_null"),
    ]
    log(f"\n  {'Contrast':<55s} {'Mean':>8s} {'sd':>8s} {'P(>0)':>8s}")
    log("  " + "-" * 81)
    for na, nb in contrasts:
        p = paired_post(results[na], results[nb], utility)
        log(f"  {na+' - '+nb:<55s} {p.mean:>8.3f} {p.sd_post:>8.3f} {p.p_positive:>8.5f}")

    # Gap closure
    gap = ce["bbg_numerical"] - ce["linear_inventory_skew"]
    gc = (ce["local_value_gradient"] - ce["linear_inventory_skew"]) / gap if abs(gap) > 1e-6 else float("nan")
    log(f"\n  Gap closure: {gc:.3f}")
    log(f"    vg CE={ce['local_value_gradient']:.3f}, linear={ce['linear_inventory_skew']:.3f}, bbg={ce['bbg_numerical']:.3f}")

    # Action usage
    if vg_choices:
        ws = [c[0] for c in vg_choices]
        ss = [c[1] for c in vg_choices]
        log(f"\n  Action usage (N={len(vg_choices)}):")
        log(f"    mean w={np.mean(ws):.4f}, std={np.std(ws):.4f}, range=[{np.min(ws):.4f},{np.max(ws):.4f}]")
        log(f"    mean s={np.mean(ss):.5f}, std={np.std(ss):.5f}")
        log(f"    unique w (4dp): {len(set(round(w,4) for w in ws))}")

    # Gradient diagnostics
    log(f"\n  Gradient diagnostics:")
    for t in range(min(5, len(betas))):
        if betas[t] is not None:
            log(f"    beta[{t}] norm={np.linalg.norm(betas[t]):.4f}, "
                f"max={np.max(np.abs(betas[t])):.4f}")

    # Save
    results_dir = PROJECT_ROOT / "finance" / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    rf = results_dir / f"option_mm_value_gradient_{suffix}_{date.today()}.txt"
    with open(rf, "w") as f:
        f.write("\n".join(out))
    log(f"\nResults saved to {rf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
