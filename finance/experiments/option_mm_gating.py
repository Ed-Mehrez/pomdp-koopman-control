"""Stage-2 gating benchmark for option market making.

Runs constant-spread and textbook Avellaneda-Stoikov with the same EWMA variance
belief on paired Heston seeds. This runner is a gate, not a tuning script.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.beliefs import EWMAVarianceFilter  # noqa: E402
from applications.option_mm.controllers import (  # noqa: E402
    ASContext,
    ConstantSpreadContext,
    avellaneda_stoikov,
    constant_spread,
)
from applications.option_mm.env import FillModelSpec, OptionMMAction  # noqa: E402
from applications.option_mm.env import OptionMMState, OptionMarketMakingEnv
from applications.option_mm.metrics import (  # noqa: E402
    EpisodeSummary,
    PosteriorSummary,
    aggregate_episode_summaries,
    cara_utility,
    crra_utility,
    paired_ce_posterior,
    paired_mean_difference_posterior,
    summarize_episode,
)


@dataclass(frozen=True)
class GatingConfig:
    # N=5000 from power calc: per-seed SNR~0.058, needs N>=810 for P>=0.95, N=5000 gives ~4sigma margin.
    seeds: tuple[int, ...] = tuple(range(5_000))
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    half_spread: float = 0.05
    ewma_half_life_days: float = 5.0
    gamma_inv: float = 0.1
    gamma_ce: float = 2.0
    cara_alpha: float = 0.001
    inventory_limit: int = 10
    posterior_draws: int = 5_000


@dataclass(frozen=True)
class StrategyRun:
    name: str
    summaries: list[EpisodeSummary]

    @property
    def terminal_wealth(self) -> np.ndarray:
        return np.asarray([summary.terminal_wealth for summary in self.summaries])

    @property
    def total_pnl(self) -> np.ndarray:
        return np.asarray([summary.total_pnl for summary in self.summaries])

    @property
    def gross_spread_capture(self) -> np.ndarray:
        return np.asarray([summary.gross_spread_capture for summary in self.summaries])

    @property
    def total_pnl_minus_gross_spread(self) -> np.ndarray:
        return self.total_pnl - self.gross_spread_capture

    @property
    def net_delta_rms(self) -> np.ndarray:
        return np.asarray([summary.net_delta_rms for summary in self.summaries])


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


ActionFn = Callable[
    [OptionMMState, EWMAVarianceFilter, OptionMarketMakingEnv, GatingConfig],
    OptionMMAction,
]


def constant_spread_action(
    state: OptionMMState,
    filt: EWMAVarianceFilter,
    env: OptionMarketMakingEnv,
    config: GatingConfig,
) -> OptionMMAction:
    del filt, env
    return constant_spread(state, ConstantSpreadContext(config.half_spread))


def avellaneda_stoikov_action(
    state: OptionMMState,
    filt: EWMAVarianceFilter,
    env: OptionMarketMakingEnv,
    config: GatingConfig,
) -> OptionMMAction:
    horizon_remaining = max((config.horizon_steps - state.step_index) * env.dt, 0.0)
    return avellaneda_stoikov(
        state,
        ASContext(
            v_hat=filt.variance,
            gamma_inv=config.gamma_inv,
            k_intensity=env.fills.distance_slope,
            horizon_remaining=horizon_remaining,
        ),
    )


def run_strategy(
    name: str,
    config: GatingConfig,
    action_fn: ActionFn,
) -> StrategyRun:
    summaries: list[EpisodeSummary] = []
    for seed in config.seeds:
        states, infos = run_episode(seed=seed, config=config, action_fn=action_fn)
        summaries.append(
            summarize_episode(
                states=states,
                infos=infos,
                inventory_limit=config.inventory_limit,
            )
        )
    return StrategyRun(name=name, summaries=summaries)


def run_episode(
    seed: int,
    config: GatingConfig,
    action_fn: ActionFn,
) -> tuple[list[OptionMMState], list]:
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()
    filt = EWMAVarianceFilter(half_life_days=config.ewma_half_life_days)
    filt.reset(initial_variance=state.variance, initial_spot=state.spot)
    states = [state]
    infos = []

    while not state.done:
        action = action_fn(state, filt, env, config)
        state, _, _, info = env.step(action)
        filt.update(state.spot)
        states.append(state)
        infos.append(info)

    return states, infos


def posterior_summaries(
    config: GatingConfig,
    wealth_as: np.ndarray,
    wealth_constant: np.ndarray,
) -> tuple[PosteriorSummary, PosteriorSummary, PosteriorSummary, PosteriorSummary]:
    crra = crra_utility(config.gamma_ce)
    cara = cara_utility(config.cara_alpha)
    seed_sequence = np.random.SeedSequence(20260407)
    crra_mc_seed, cara_mc_seed = seed_sequence.spawn(2)
    crra_delta = paired_ce_posterior(
        wealth_as,
        wealth_constant,
        utility=crra,
        method="delta",
    )
    crra_mc = paired_ce_posterior(
        wealth_as,
        wealth_constant,
        utility=crra,
        method="mc",
        n_draws=config.posterior_draws,
        rng=np.random.default_rng(crra_mc_seed),
    )
    cara_delta = paired_ce_posterior(
        wealth_as,
        wealth_constant,
        utility=cara,
        method="delta",
    )
    cara_mc = paired_ce_posterior(
        wealth_as,
        wealth_constant,
        utility=cara,
        method="mc",
        n_draws=config.posterior_draws,
        rng=np.random.default_rng(cara_mc_seed),
    )
    return crra_delta, crra_mc, cara_delta, cara_mc


def build_gates(
    as_run: StrategyRun,
    constant_run: StrategyRun,
    crra_delta: PosteriorSummary,
    crra_mc: PosteriorSummary,
    cara_delta: PosteriorSummary,
    cara_mc: PosteriorSummary,
) -> list[GateResult]:
    spread_ratio = (
        np.mean(as_run.gross_spread_capture)
        / max(np.mean(constant_run.gross_spread_capture), 1e-12)
    )
    return [
        GateResult(
            "CRRA posterior P(delta CE > 0) >= 0.95",
            crra_delta.p_positive >= 0.95,
            f"p_positive={crra_delta.p_positive:.6f}",
        ),
        GateResult(
            "CRRA delta and MC agree on mean and sd_post",
            _posterior_agrees(crra_delta, crra_mc),
            _agreement_detail(crra_delta, crra_mc),
        ),
        GateResult(
            "A-S mean spread capture exceeds constant spread",
            np.mean(as_run.gross_spread_capture) > np.mean(
                constant_run.gross_spread_capture
            ),
            f"spread_ratio={spread_ratio:.6f}",
        ),
        GateResult(
            "CARA direction is nonnegative",
            cara_delta.p_positive >= 0.5,
            f"p_positive={cara_delta.p_positive:.6f}",
        ),
        GateResult(
            "CARA delta and MC agree on mean and sd_post",
            _posterior_agrees(cara_delta, cara_mc),
            _agreement_detail(cara_delta, cara_mc),
        ),
    ]


def print_strategy_summary(run: StrategyRun) -> None:
    agg = aggregate_episode_summaries(run.summaries)
    pnl_minus_gross_summary = paired_mean_difference_posterior(
        run.total_pnl_minus_gross_spread
    )
    print(f"\n[{run.name}]")
    print(f"  terminal_wealth_mean       {agg.terminal_wealth_mean:12.6f}")
    print(f"  terminal_wealth_std        {agg.terminal_wealth_std:12.6f}")
    print(f"  total_pnl_mean             {agg.total_pnl_mean:12.6f}")
    print(f"  total_pnl_std              {agg.total_pnl_std:12.6f}")
    print(f"  max_drawdown_mean          {agg.max_drawdown_mean:12.6f}")
    print(f"  final_inventory_mean       {agg.final_inventory_mean:12.6f}")
    print(f"  final_inventory_std        {agg.final_inventory_std:12.6f}")
    print(f"  gross_spread_mean          {agg.gross_spread_capture_mean:12.6f}")
    print(
        f"  mean_total_pnl_minus_gross_spread "
        f"{pnl_minus_gross_summary.mean:12.6f}"
    )
    print(
        f"  total_pnl_minus_gross_sd_post     "
        f"{pnl_minus_gross_summary.sd_post:12.6f}"
    )
    print(f"  net_spread_mean            {agg.net_spread_capture_mean:12.6f}")
    print(f"  adverse_selection_mean     {agg.adverse_selection_cost_mean:12.6f}")
    print(f"  option_turnover_mean       {agg.option_turnover_mean:12.6f}")
    print(f"  net_delta_rms_mean         {agg.net_delta_rms_mean:12.6f}")
    print(f"  censoring_rate_mean        {agg.censoring_rate_mean:12.8f}")
    print(f"  variance_floor_rate_mean   {agg.variance_floor_binding_rate_mean:12.8f}")


def print_posterior(name: str, summary: PosteriorSummary) -> None:
    print(f"  {name:12s} mean={summary.mean: .6f} sd_post={summary.sd_post: .6f} "
          f"CrI=[{summary.ci_low: .6f}, {summary.ci_high: .6f}] "
          f"P(>0)={summary.p_positive: .6f}")


def _posterior_agrees(a: PosteriorSummary, b: PosteriorSummary) -> bool:
    return _relative_close(a.mean, b.mean) and _relative_close(a.sd_post, b.sd_post)


def _agreement_detail(a: PosteriorSummary, b: PosteriorSummary) -> str:
    mean_rel = _relative_error(a.mean, b.mean)
    sd_rel = _relative_error(a.sd_post, b.sd_post)
    return f"mean_rel={mean_rel:.6f}, sd_post_rel={sd_rel:.6f}"


def _relative_close(a: float, b: float, tolerance: float = 0.05) -> bool:
    return _relative_error(a, b) <= tolerance


def _relative_error(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main() -> int:
    config = GatingConfig()
    constant_run = run_strategy(
        "constant_spread",
        config=config,
        action_fn=constant_spread_action,
    )
    as_run = run_strategy(
        "avellaneda_stoikov_ewma",
        config=config,
        action_fn=avellaneda_stoikov_action,
    )
    crra_delta, crra_mc, cara_delta, cara_mc = posterior_summaries(
        config,
        wealth_as=as_run.terminal_wealth,
        wealth_constant=constant_run.terminal_wealth,
    )
    pnl_summary = paired_mean_difference_posterior(
        as_run.total_pnl - constant_run.total_pnl
    )
    gates = build_gates(
        as_run=as_run,
        constant_run=constant_run,
        crra_delta=crra_delta,
        crra_mc=crra_mc,
        cara_delta=cara_delta,
        cara_mc=cara_mc,
    )

    print("Option MM Stage-2 Gating")
    print(f"seeds={len(config.seeds)}, horizon_steps={config.horizon_steps}")
    print_strategy_summary(constant_run)
    print_strategy_summary(as_run)
    print("\n[posterior summaries]")
    print_posterior("CRRA delta", crra_delta)
    print_posterior("CRRA MC", crra_mc)
    print_posterior("CARA delta", cara_delta)
    print_posterior("CARA MC", cara_mc)
    print_posterior("mean PnL", pnl_summary)

    spread_ratio = (
        np.mean(as_run.gross_spread_capture)
        / np.mean(constant_run.gross_spread_capture)
    )
    net_delta_ratio = np.mean(as_run.net_delta_rms) / np.mean(constant_run.net_delta_rms)
    print("\n[stage-2 sentence]")
    print(
        f"On {len(config.seeds)} paired seeds "
        "(same Heston path per seed via path_rng), "
        f"A-S-with-EWMA beats constant-spread by analytic Bayesian posterior "
        f"delta CE = {crra_delta.mean:.6f} with sd_post = {crra_delta.sd_post:.6f} "
        f"under CRRA(gamma=2), 95% CrI = [{crra_delta.ci_low:.6f}, "
        f"{crra_delta.ci_high:.6f}], P(delta CE > 0 | data) = "
        f"{crra_delta.p_positive:.6f}. Under CARA(alpha=0.001), delta CE = "
        f"{cara_delta.mean:.6f} with sd_post = {cara_delta.sd_post:.6f} and "
        f"P(>0) = {cara_delta.p_positive:.6f}. Delta-method and MC posterior "
        "agreement is shown above. Spread capture is "
        f"{spread_ratio:.6f}x constant-spread; net delta exposure RMS is "
        f"{net_delta_ratio:.6f}x lower."
    )

    print("\n[stage-2 gates]")
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"  {status:4s}  {gate.name}: {gate.detail}")

    return 0 if all(gate.passed for gate in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
