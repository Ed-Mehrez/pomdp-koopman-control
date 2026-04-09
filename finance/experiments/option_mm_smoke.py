"""Stage-1 smoke test for the option market-making simulator.

This is intentionally boring: no beliefs, no A-S baseline, no SDRE. The goal is
to lock the measurement layer and check that the simulator produces sensible
wealth, inventory, fill, and health diagnostics before controller work starts.
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

from applications.option_mm.env import (  # noqa: E402
    FillModelSpec,
    OptionMMAction,
    OptionMMState,
    OptionMarketMakingEnv,
)
from applications.option_mm.metrics import (  # noqa: E402
    AggregateSummary,
    EpisodeSummary,
    aggregate_episode_summaries,
    certainty_equivalent,
    paired_mean_difference_posterior,
    summarize_episode,
)


NO_QUOTE_ASK = 1.0e12


@dataclass(frozen=True)
class SmokeConfig:
    seeds: tuple[int, ...] = tuple(range(500))
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    half_spread: float = 0.05
    gamma_ce: float = 2.0
    inventory_limit: int = 10


@dataclass(frozen=True)
class StrategyRun:
    name: str
    summaries: list[EpisodeSummary]
    wealth_paths: list[np.ndarray]
    inventory_paths: list[np.ndarray]

    @property
    def aggregate(self) -> AggregateSummary:
        return aggregate_episode_summaries(self.summaries)

    @property
    def terminal_wealth(self) -> np.ndarray:
        return np.asarray([summary.terminal_wealth for summary in self.summaries])

    @property
    def total_pnl(self) -> np.ndarray:
        return np.asarray([summary.total_pnl for summary in self.summaries])

    @property
    def final_inventory(self) -> np.ndarray:
        return np.asarray([summary.final_inventory for summary in self.summaries])

    @property
    def gross_spread_capture(self) -> np.ndarray:
        return np.asarray([summary.gross_spread_capture for summary in self.summaries])


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def no_quote_baseline(state: OptionMMState, _: SmokeConfig) -> OptionMMAction:
    """No posted liquidity and no hedge.

    The env requires finite quotes, so the smoke test pairs this action with a
    zero-intensity fill model to make the no-op path exact.
    """
    return OptionMMAction(bid_price=0.0, ask_price=NO_QUOTE_ASK, hedge_trade=0.0)


def constant_spread_baseline(state: OptionMMState, config: SmokeConfig) -> OptionMMAction:
    """Symmetric quote, no hedge, no inventory skew."""
    return OptionMMAction(
        bid_price=max(state.option_mid - config.half_spread, 0.0),
        ask_price=state.option_mid + config.half_spread,
        hedge_trade=0.0,
    )


def run_strategy(
    name: str,
    config: SmokeConfig,
    action_fn: Callable[[OptionMMState, SmokeConfig], OptionMMAction],
    fill_base_intensity: float,
    same_step_policy: str = "mid_drift",
    horizon_steps: int | None = None,
) -> StrategyRun:
    summaries: list[EpisodeSummary] = []
    wealth_paths: list[np.ndarray] = []
    inventory_paths: list[np.ndarray] = []

    for seed in config.seeds:
        states, infos = run_episode(
            seed=seed,
            config=config,
            action_fn=action_fn,
            fill_base_intensity=fill_base_intensity,
            same_step_policy=same_step_policy,
            horizon_steps=horizon_steps or config.horizon_steps,
        )
        summaries.append(
            summarize_episode(
                states=states,
                infos=infos,
                inventory_limit=config.inventory_limit,
            )
        )
        wealth_paths.append(np.asarray([state.wealth for state in states], dtype=float))
        inventory_paths.append(
            np.asarray([state.option_inventory for state in states], dtype=float)
        )

    return StrategyRun(
        name=name,
        summaries=summaries,
        wealth_paths=wealth_paths,
        inventory_paths=inventory_paths,
    )


def run_episode(
    seed: int,
    config: SmokeConfig,
    action_fn: Callable[[OptionMMState, SmokeConfig], OptionMMAction],
    fill_base_intensity: float,
    same_step_policy: str,
    horizon_steps: int,
) -> tuple[list[OptionMMState], list]:
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(
            base_intensity=fill_base_intensity,
            same_step_both_fills_policy=same_step_policy,
        ),
        horizon_steps=horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()
    states = [state]
    infos = []

    while not state.done:
        action = action_fn(state, config)
        state, _, _, info = env.step(action)
        states.append(state)
        infos.append(info)

    return states, infos


def run_inventory_growth_check(config: SmokeConfig) -> dict[int, float]:
    growth: dict[int, float] = {}
    for horizon in (5, 10, 20):
        run = run_strategy(
            name=f"constant_spread_T{horizon}",
            config=config,
            action_fn=constant_spread_baseline,
            fill_base_intensity=50.0,
            horizon_steps=horizon,
        )
        growth[horizon] = _sample_std(run.final_inventory)
    return growth


def check_paired_seed_reproducibility(config: SmokeConfig) -> bool:
    states_a, _ = run_episode(
        seed=777,
        config=config,
        action_fn=constant_spread_baseline,
        fill_base_intensity=50.0,
        same_step_policy="mid_drift",
        horizon_steps=config.horizon_steps,
    )
    states_b, _ = run_episode(
        seed=777,
        config=config,
        action_fn=constant_spread_baseline,
        fill_base_intensity=50.0,
        same_step_policy="mid_drift",
        horizon_steps=config.horizon_steps,
    )
    wealth_a = np.asarray([state.wealth for state in states_a])
    wealth_b = np.asarray([state.wealth for state in states_b])
    inventory_a = np.asarray([state.option_inventory for state in states_a])
    inventory_b = np.asarray([state.option_inventory for state in states_b])
    return bool(np.array_equal(wealth_a, wealth_b) and np.array_equal(inventory_a, inventory_b))


def build_checks(
    config: SmokeConfig,
    no_quote: StrategyRun,
    constant_spread: StrategyRun,
    inventory_growth: dict[int, float],
    paired_reproducible: bool,
    subsidy_delta: float,
) -> list[CheckResult]:
    no_quote_pnl_mean = float(np.mean(no_quote.total_pnl))
    no_quote_pnl_std = _sample_std(no_quote.total_pnl)
    if no_quote_pnl_std == 0.0:
        no_quote_pnl_ok = no_quote_pnl_mean == 0.0
        no_quote_pnl_detail = f"mean={no_quote_pnl_mean:.6g}, std=0"
    else:
        tolerance = 0.5 * no_quote_pnl_std
        no_quote_pnl_ok = abs(no_quote_pnl_mean) <= tolerance
        no_quote_pnl_detail = (
            f"mean={no_quote_pnl_mean:.6g}, std={no_quote_pnl_std:.6g}, "
            f"tol={tolerance:.6g}"
        )

    no_quote_wealth_std = _sample_std(no_quote.terminal_wealth)
    constant_agg = constant_spread.aggregate
    final_inventory_std = _sample_std(constant_spread.final_inventory)
    final_inventory_mean = float(np.mean(constant_spread.final_inventory))

    inv_std_5 = inventory_growth[5]
    inv_std_10 = inventory_growth[10]
    inv_std_20 = inventory_growth[20]
    growth_ratio_20_5 = inv_std_20 / inv_std_5 if inv_std_5 > 0.0 else float("inf")
    growth_ok = (
        abs(inv_std_5 - 1.243) <= 0.08
        and abs(inv_std_10 - 1.758) <= 0.11
        and abs(inv_std_20 - 2.486) <= 0.16
        and 1.85 <= growth_ratio_20_5 <= 2.15
    )

    return [
        CheckResult(
            "no_quote mean PnL approximately zero",
            no_quote_pnl_ok,
            no_quote_pnl_detail,
        ),
        CheckResult(
            "no_quote inventory always zero",
            bool(np.all(no_quote.final_inventory == 0.0)),
            f"max_abs_final_inventory={np.max(np.abs(no_quote.final_inventory)):.6g}",
        ),
        CheckResult(
            "no_quote terminal wealth std exactly zero",
            no_quote_wealth_std == 0.0,
            f"terminal_wealth_std={no_quote_wealth_std:.6g}",
        ),
        CheckResult(
            "constant_spread spread capture positive for every seed",
            bool(np.all(constant_spread.gross_spread_capture > 0.0)),
            (
                f"min={np.min(constant_spread.gross_spread_capture):.6g}, "
                f"mean={constant_agg.gross_spread_capture_mean:.6g}"
            ),
        ),
        CheckResult(
            "constant_spread inventory mean small vs std",
            abs(final_inventory_mean) <= 0.5 * final_inventory_std,
            f"mean={final_inventory_mean:.6g}, std={final_inventory_std:.6g}",
        ),
        CheckResult(
            "constant_spread inventory std grows roughly with sqrt(T)",
            growth_ok,
            (
                f"std_5={inv_std_5:.6g}, std_10={inv_std_10:.6g}, "
                f"std_20={inv_std_20:.6g}, ratio_20_5={growth_ratio_20_5:.6g}"
            ),
        ),
        CheckResult(
            "default censoring rate below 0.01%",
            constant_agg.censoring_rate_mean < 1.0e-4,
            f"rate={constant_agg.censoring_rate_mean:.6g}",
        ),
        CheckResult(
            "default variance-floor binding rate below 0.1%",
            constant_agg.variance_floor_binding_rate_mean < 1.0e-3,
            f"rate={constant_agg.variance_floor_binding_rate_mean:.6g}",
        ),
        CheckResult(
            "paired-seed reproducibility",
            paired_reproducible,
            "identical wealth/inventory paths for duplicate envs",
        ),
        CheckResult(
            "same-step subsidy delta quantified",
            subsidy_delta >= -1.0e-12,
            f"allowed_minus_mid_drift_gross_spread_mean={subsidy_delta:.6g}",
        ),
    ]


def print_summary(config: SmokeConfig, run: StrategyRun) -> None:
    agg = run.aggregate
    ce = certainty_equivalent(run.terminal_wealth, gamma=config.gamma_ce)
    total_pnl_post = paired_mean_difference_posterior(run.total_pnl)
    pnl_minus_gross_spread = run.total_pnl - run.gross_spread_capture
    pnl_minus_gross_post = paired_mean_difference_posterior(pnl_minus_gross_spread)
    print(f"\n[{run.name}]")
    print(f"  terminal_wealth_mean     {agg.terminal_wealth_mean:12.6f}")
    print(f"  terminal_wealth_std      {agg.terminal_wealth_std:12.6f}")
    print(f"  CRRA_CE_gamma_{config.gamma_ce:g}      {ce:12.6f}")
    print(f"  total_pnl_mean           {agg.total_pnl_mean:12.6f}")
    print(f"  total_pnl_sd_post        {total_pnl_post.sd_post:12.6f}")
    print(f"  total_pnl_std            {agg.total_pnl_std:12.6f}")
    print(f"  mean_total_pnl_minus_gross_spread {pnl_minus_gross_post.mean:6.6f}")
    print(f"  total_pnl_minus_gross_sd_post {pnl_minus_gross_post.sd_post:5.6f}")
    print(f"  pnl_vol_mean             {agg.pnl_vol_mean:12.6f}")
    print(f"  max_drawdown_mean        {agg.max_drawdown_mean:12.6f}")
    print(f"  max_drawdown_max         {agg.max_drawdown_max:12.6f}")
    print(f"  final_inventory_mean     {agg.final_inventory_mean:12.6f}")
    print(f"  final_inventory_std      {agg.final_inventory_std:12.6f}")
    print(f"  inventory_abs_mean       {agg.inventory_abs_mean:12.6f}")
    print(f"  inventory_abs_p95_mean   {agg.inventory_abs_p95_mean:12.6f}")
    print(f"  inventory_abs_max_mean   {agg.inventory_abs_max_mean:12.6f}")
    print(f"  time_at_inv_limit_mean   {agg.time_at_inventory_limit_mean:12.6f}")
    print(f"  gross_spread_mean        {agg.gross_spread_capture_mean:12.6f}")
    print(f"  net_spread_mean          {agg.net_spread_capture_mean:12.6f}")
    print(f"  adverse_selection_mean   {agg.adverse_selection_cost_mean:12.6f}")
    print(f"  mean_adverse_selection_cost {agg.adverse_selection_cost_mean:9.6f}")
    print(f"  option_turnover_mean     {agg.option_turnover_mean:12.6f}")
    print(f"  stock_turnover_mean      {agg.stock_turnover_mean:12.6f}")
    print(f"  fees_and_costs_mean      {agg.total_fees_and_costs_mean:12.6f}")
    print(f"  net_delta_rms_mean       {agg.net_delta_rms_mean:12.6f}")
    print(f"  censoring_rate_mean      {agg.censoring_rate_mean:12.8f}")
    print(f"  variance_floor_rate_mean {agg.variance_floor_binding_rate_mean:12.8f}")
    print(f"  both_fill_rate_mean      {agg.same_step_both_fill_rate_mean:12.8f}")


def print_checks(checks: list[CheckResult]) -> None:
    print("\n[stage-1 checks]")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  {status:4s}  {check.name}: {check.detail}")


def _sample_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def main() -> int:
    config = SmokeConfig()

    no_quote = run_strategy(
        name="no_quote",
        config=config,
        action_fn=no_quote_baseline,
        fill_base_intensity=0.0,
    )
    constant_spread = run_strategy(
        name="constant_spread_mid_drift",
        config=config,
        action_fn=constant_spread_baseline,
        fill_base_intensity=50.0,
        same_step_policy="mid_drift",
    )
    constant_spread_allowed = run_strategy(
        name="constant_spread_allowed",
        config=config,
        action_fn=constant_spread_baseline,
        fill_base_intensity=50.0,
        same_step_policy="allowed",
    )

    inventory_growth = run_inventory_growth_check(config)
    paired_reproducible = check_paired_seed_reproducibility(config)
    subsidy_delta = (
        constant_spread_allowed.aggregate.gross_spread_capture_mean
        - constant_spread.aggregate.gross_spread_capture_mean
    )
    checks = build_checks(
        config=config,
        no_quote=no_quote,
        constant_spread=constant_spread,
        inventory_growth=inventory_growth,
        paired_reproducible=paired_reproducible,
        subsidy_delta=subsidy_delta,
    )

    print("Option MM Stage-1 Smoke")
    print(f"seeds={len(config.seeds)}, horizon_steps={config.horizon_steps}")
    print(f"initial_cash={config.initial_cash:.2f}, half_spread={config.half_spread:.4f}")
    print_summary(config, no_quote)
    print_summary(config, constant_spread)
    print_summary(config, constant_spread_allowed)
    print("\n[inventory growth]")
    for horizon, std in sorted(inventory_growth.items()):
        print(f"  horizon={horizon:2d}  final_inventory_std={std:.6f}")
    print(
        "\n[same-step subsidy] "
        f"allowed_minus_mid_drift_gross_spread_mean={subsidy_delta:.6f}"
    )
    print_checks(checks)

    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
