"""Stage-4 control-structure ablation for option market making.

Uses the Stage-3 decision to keep EWMA fixed and compares controller structure:
textbook A-S, a pinned affine inventory rule, and a local SDRE rule.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.beliefs import EWMAVarianceFilter  # noqa: E402
from applications.option_mm.controllers import (  # noqa: E402
    ASContext,
    ConstantSpreadContext,
    LinearRuleContext,
    SDREContext,
    avellaneda_stoikov,
    constant_spread,
    linear_inventory_rule,
    no_quote,
    sdre_controller,
)
from applications.option_mm.env import FillModelSpec, OptionMMAction  # noqa: E402
from applications.option_mm.env import OptionMMState, OptionMarketMakingEnv
from applications.option_mm.metrics import (  # noqa: E402
    EpisodeSummary,
    PosteriorSummary,
    UtilitySpec,
    aggregate_episode_summaries,
    cara_utility,
    crra_utility,
    paired_ce_posterior,
    summarize_episode,
)


POSTERIOR_PROB_95_Z = 1.6448536269514722
STAGE2_EWMA_MINUS_CONSTANT_CRRA = 26.967789651505882
STAGE2_CROSS_STAGE_TOL = 1e-9
CONTROLLER_ORDER = (
    "sdre",
    "linear_rule",
    "as_ewma",
    "constant_spread",
    "no_quote",
)


@dataclass(frozen=True)
class ControlAblationConfig:
    seed_sequence_entropy: int = 20260407
    pilot_seeds: tuple[int, ...] = tuple(range(100))
    # N pinned from power calc (2026-04-08): N=100 pilot gave SDRE-linear SNR=0.091
    # (N>=326 for P>=0.95) and SDRE-AS SNR=0.048 (N>=1161); N=5000 preserves
    # Stage-2/3 paired seeds and gives the EWMA-constant check exactly.
    formal_seeds: tuple[int, ...] = tuple(range(5_000))
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    half_spread: float = 0.05
    ewma_half_life_days: float = 5.0
    gamma_inv: float = 0.1
    gamma_ce: float = 2.0
    cara_alpha: float = 2.0e-5
    inventory_limit: int = 10
    posterior_draws: int = 5_000


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    kind: Literal["no_quote", "constant_spread", "as", "linear", "sdre"]
    uses_filter: bool


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
    def net_delta_rms(self) -> np.ndarray:
        return np.asarray([summary.net_delta_rms for summary in self.summaries])


@dataclass(frozen=True)
class PilotPowerRow:
    contrast: str
    mean_diff: float
    per_seed_sd: float
    per_seed_snr: float
    required_n_for_95: int | None


@dataclass(frozen=True)
class PairedPosteriorCheck:
    contrast: str
    controller_a: str
    controller_b: str
    utility_name: str
    delta: PosteriorSummary
    mc: PosteriorSummary

    @property
    def mean_rel_error(self) -> float:
        return _relative_error(self.delta.mean, self.mc.mean)

    @property
    def sd_post_rel_error(self) -> float:
        return _relative_error(self.delta.sd_post, self.mc.sd_post)

    @property
    def passes_agreement(self) -> bool:
        return self.mean_rel_error <= 0.05 and self.sd_post_rel_error <= 0.05


def controller_specs() -> tuple[ControllerSpec, ...]:
    return (
        ControllerSpec("no_quote", "no_quote", uses_filter=False),
        ControllerSpec("constant_spread", "constant_spread", uses_filter=False),
        ControllerSpec("as_ewma", "as", uses_filter=True),
        ControllerSpec("linear_rule", "linear", uses_filter=True),
        ControllerSpec("sdre", "sdre", uses_filter=True),
    )


def run_all_strategies(
    seeds: tuple[int, ...],
    config: ControlAblationConfig,
) -> dict[str, StrategyRun]:
    return {
        spec.name: run_strategy(spec=spec, seeds=seeds, config=config)
        for spec in controller_specs()
    }


def run_strategy(
    spec: ControllerSpec,
    seeds: tuple[int, ...],
    config: ControlAblationConfig,
) -> StrategyRun:
    summaries: list[EpisodeSummary] = []
    for seed in seeds:
        states, infos = run_episode(seed=seed, spec=spec, config=config)
        summaries.append(
            summarize_episode(
                states=states,
                infos=infos,
                inventory_limit=config.inventory_limit,
            )
        )
    return StrategyRun(name=spec.name, summaries=summaries)


def run_episode(
    seed: int,
    spec: ControllerSpec,
    config: ControlAblationConfig,
) -> tuple[list[OptionMMState], list]:
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()
    variance_filter = None
    if spec.uses_filter:
        variance_filter = EWMAVarianceFilter(half_life_days=config.ewma_half_life_days)
        variance_filter.reset(initial_variance=state.variance, initial_spot=state.spot)

    states = [state]
    infos = []
    while not state.done:
        action = action_for_spec(spec, state, variance_filter, env, config)
        state, _, _, info = env.step(action)
        if variance_filter is not None:
            variance_filter.update(state.spot)
        states.append(state)
        infos.append(info)

    return states, infos


def action_for_spec(
    spec: ControllerSpec,
    state: OptionMMState,
    variance_filter: EWMAVarianceFilter | None,
    env: OptionMarketMakingEnv,
    config: ControlAblationConfig,
) -> OptionMMAction:
    if spec.kind == "no_quote":
        return no_quote(state)
    if spec.kind == "constant_spread":
        return constant_spread(state, ConstantSpreadContext(config.half_spread))
    if variance_filter is None:
        raise RuntimeError("filtered controller requires EWMA variance filter")

    horizon_remaining = max((config.horizon_steps - state.step_index) * env.dt, 0.0)
    if spec.kind == "as":
        return avellaneda_stoikov(
            state,
            ASContext(
                v_hat=variance_filter.variance,
                gamma_inv=config.gamma_inv,
                k_intensity=env.fills.distance_slope,
                horizon_remaining=horizon_remaining,
            ),
        )
    if spec.kind == "linear":
        return linear_inventory_rule(
            state,
            LinearRuleContext(
                v_hat=variance_filter.variance,
                gamma_inv=config.gamma_inv,
                k_intensity=env.fills.distance_slope,
                horizon_remaining=horizon_remaining,
                contract_multiplier=env.contract.contract_multiplier,
            ),
        )
    if spec.kind == "sdre":
        return sdre_controller(
            state,
            SDREContext(
                v_hat=variance_filter.variance,
                gamma_inv=config.gamma_inv,
                k_intensity=env.fills.distance_slope,
                horizon_remaining=horizon_remaining,
                base_intensity=env.fills.base_intensity,
                dt=env.dt,
                contract_multiplier=env.contract.contract_multiplier,
            ),
        )
    raise ValueError(f"unknown controller kind: {spec.kind}")


def pilot_power_rows(runs: dict[str, StrategyRun]) -> list[PilotPowerRow]:
    rows = []
    for contrast, name_a, name_b in (
        ("sdre_minus_linear", "sdre", "linear_rule"),
        ("sdre_minus_as", "sdre", "as_ewma"),
    ):
        diffs = runs[name_a].terminal_wealth - runs[name_b].terminal_wealth
        mean_diff = float(np.mean(diffs))
        per_seed_sd = _sample_sd(diffs)
        snr = 0.0 if per_seed_sd == 0.0 else abs(mean_diff) / per_seed_sd
        required_n = None
        if snr > 0.0 and isfinite(snr):
            required_n = int(ceil((POSTERIOR_PROB_95_Z / snr) ** 2))
        rows.append(
            PilotPowerRow(
                contrast=contrast,
                mean_diff=mean_diff,
                per_seed_sd=per_seed_sd,
                per_seed_snr=float(snr),
                required_n_for_95=required_n,
            )
        )
    return rows


def absolute_ce_table(
    runs: dict[str, StrategyRun],
    utility: UtilitySpec,
) -> dict[str, float]:
    return {
        name: utility.ce(float(np.mean(utility.u(runs[name].terminal_wealth))))
        for name in CONTROLLER_ORDER
    }


def paired_posterior_checks(
    runs: dict[str, StrategyRun],
    config: ControlAblationConfig,
) -> list[PairedPosteriorCheck]:
    utilities = (
        ("CRRA", crra_utility(config.gamma_ce)),
        ("CARA", cara_utility(config.cara_alpha)),
    )
    pairs = list(itertools.combinations(CONTROLLER_ORDER, 2))
    seed_sequence = np.random.SeedSequence(config.seed_sequence_entropy)
    mc_seeds = seed_sequence.spawn(len(utilities) * len(pairs))

    checks: list[PairedPosteriorCheck] = []
    seed_index = 0
    for utility_name, utility in utilities:
        for name_a, name_b in pairs:
            contrast = f"{name_a}_minus_{name_b}"
            delta = paired_ce_posterior(
                runs[name_a].terminal_wealth,
                runs[name_b].terminal_wealth,
                utility=utility,
                method="delta",
            )
            mc = paired_ce_posterior(
                runs[name_a].terminal_wealth,
                runs[name_b].terminal_wealth,
                utility=utility,
                method="mc",
                n_draws=config.posterior_draws,
                rng=np.random.default_rng(mc_seeds[seed_index]),
            )
            seed_index += 1
            checks.append(
                PairedPosteriorCheck(
                    contrast=contrast,
                    controller_a=name_a,
                    controller_b=name_b,
                    utility_name=utility_name,
                    delta=delta,
                    mc=mc,
                )
            )
    return checks


def contribution_table(
    ce_crra: dict[str, float],
) -> dict[str, float]:
    best_ce = max(ce_crra[name] for name in CONTROLLER_ORDER)
    anchor_ce = ce_crra["constant_spread"]
    gap = best_ce - anchor_ce
    if abs(gap) <= 1e-12:
        return {name: float("nan") for name in CONTROLLER_ORDER}
    return {
        name: (ce_crra[name] - anchor_ce) / gap
        for name in CONTROLLER_ORDER
    }


def print_pilot_power(rows: list[PilotPowerRow]) -> None:
    print("\n[pilot power calc, N=100]")
    for row in rows:
        required = (
            "inf" if row.required_n_for_95 is None else str(row.required_n_for_95)
        )
        print(
            f"  {row.contrast:18s} mean_diff={row.mean_diff: .6f} "
            f"per_seed_sd={row.per_seed_sd: .6f} "
            f"per_seed_snr={row.per_seed_snr: .6f} "
            f"N_for_P95~{required}"
        )


def print_strategy_summaries(runs: dict[str, StrategyRun]) -> None:
    print("\n[strategy summaries]")
    for name in CONTROLLER_ORDER:
        agg = aggregate_episode_summaries(runs[name].summaries)
        print(
            f"  {name:16s} terminal_wealth={agg.terminal_wealth_mean: .6f} "
            f"total_pnl={agg.total_pnl_mean: .6f} "
            f"spread={agg.gross_spread_capture_mean: .6f} "
            f"net_delta_rms={agg.net_delta_rms_mean: .6f}"
        )


def print_absolute_ce(ce_crra: dict[str, float], ce_cara: dict[str, float]) -> None:
    print("\n[absolute CE]")
    print("  controller          CRRA_gamma_2       CARA_alpha_2e-5")
    for name in CONTROLLER_ORDER:
        print(f"  {name:16s} {ce_crra[name]:16.6f} {ce_cara[name]:20.6f}")


def print_contribution_table(
    ce_crra: dict[str, float],
    contributions: dict[str, float],
) -> None:
    print("\n[controller contribution, CRRA]")
    print("  controller          CE          delta_vs_constant  best_gap_fraction")
    for name in CONTROLLER_ORDER:
        delta_vs_constant = ce_crra[name] - ce_crra["constant_spread"]
        print(
            f"  {name:16s} {ce_crra[name]:12.6f} "
            f"{delta_vs_constant:18.6f} {100.0 * contributions[name]:18.3f}%"
        )


def print_posterior_checks(checks: list[PairedPosteriorCheck]) -> None:
    print("\n[pairwise posterior checks]")
    for check in checks:
        print(
            f"  {check.utility_name:4s} {check.contrast:38s} "
            f"delta_mean={check.delta.mean: .6f} "
            f"delta_sd_post={check.delta.sd_post: .6f} "
            f"delta_P(>0)={check.delta.p_positive: .6f} "
            f"mc_mean={check.mc.mean: .6f} "
            f"mc_sd_post={check.mc.sd_post: .6f} "
            f"mean_rel={check.mean_rel_error: .6f} "
            f"sd_post_rel={check.sd_post_rel_error: .6f}"
        )


def print_stage_sentence(
    config: ControlAblationConfig,
    checks: list[PairedPosteriorCheck],
    outcome: str,
) -> None:
    crra = _check_lookup(checks, "CRRA")
    cara = _check_lookup(checks, "CARA")
    sdre_linear_crra = crra["sdre_minus_linear_rule"]
    sdre_linear_cara = cara["sdre_minus_linear_rule"]
    sdre_as_crra = crra["sdre_minus_as_ewma"]
    linear_as_crra = crra["linear_rule_minus_as_ewma"]
    ewma_constant_crra = crra["as_ewma_minus_constant_spread"]

    print("\n[stage-4 sentence]")
    print(
        f"On {len(config.formal_seeds)} paired seeds "
        f"(same SeedSequence({config.seed_sequence_entropy}) as Stages 2/3, "
        "all filtered controllers consuming EWMAVarianceFilter), under "
        "CRRA(gamma=2) the SDRE controller on (q, h, V_hat, tau) achieves "
        f"delta CE vs linear-rule = {sdre_linear_crra.delta.mean:.6f} with "
        f"sd_post = {sdre_linear_crra.delta.sd_post:.6f}, "
        f"P(delta CE > 0) = {sdre_linear_crra.delta.p_positive:.6f}. "
        f"Under CARA(alpha=2e-5), delta CE = {sdre_linear_cara.delta.mean:.6f} "
        f"with sd_post = {sdre_linear_cara.delta.sd_post:.6f} and "
        f"P(>0) = {sdre_linear_cara.delta.p_positive:.6f}. "
        f"SDRE vs A-S: delta CE = {sdre_as_crra.delta.mean:.6f} "
        f"(P = {sdre_as_crra.delta.p_positive:.6f}). "
        f"Linear-rule vs A-S: delta CE = {linear_as_crra.delta.mean:.6f} "
        f"(P = {linear_as_crra.delta.p_positive:.6f}). "
        f"EWMA-constant reproduces the Stage 2 number "
        f"{ewma_constant_crra.delta.mean:.6f}. "
        f"The result supports {outcome}."
    )


def build_gate_results(
    checks: list[PairedPosteriorCheck],
) -> tuple[list[tuple[str, bool, str]], str]:
    results: list[tuple[str, bool, str]] = []
    for check in checks:
        results.append(
            (
                f"{check.utility_name} {check.contrast} delta-vs-MC agreement",
                check.passes_agreement,
                (
                    f"mean_rel={check.mean_rel_error:.6f}, "
                    f"sd_post_rel={check.sd_post_rel_error:.6f}"
                ),
            )
        )

    crra = _check_lookup(checks, "CRRA")
    cara = _check_lookup(checks, "CARA")
    cross_stage = crra["as_ewma_minus_constant_spread"].delta.mean
    cross_stage_error = abs(cross_stage - STAGE2_EWMA_MINUS_CONSTANT_CRRA)
    results.append(
        (
            "EWMA-constant reproduces Stage-2 CRRA delta CE",
            cross_stage_error <= STAGE2_CROSS_STAGE_TOL,
            f"error={cross_stage_error:.12g}",
        )
    )

    sdre_linear_ship = (
        crra["sdre_minus_linear_rule"].delta.p_positive >= 0.95
        and cara["sdre_minus_linear_rule"].delta.p_positive >= 0.95
    )
    sdre_as_direction = (
        crra["sdre_minus_as_ewma"].delta.p_positive >= 0.5
        and cara["sdre_minus_as_ewma"].delta.p_positive >= 0.5
    )
    results.append(
        (
            "ship rule SDRE-linear P(delta CE > 0) >= 0.95 under CRRA and CARA",
            sdre_linear_ship,
            (
                f"CRRA={crra['sdre_minus_linear_rule'].delta.p_positive:.6f}, "
                f"CARA={cara['sdre_minus_linear_rule'].delta.p_positive:.6f}"
            ),
        )
    )
    results.append(
        (
            "sanity rule SDRE-A-S direction nonnegative under CRRA and CARA",
            sdre_as_direction,
            (
                f"CRRA={crra['sdre_minus_as_ewma'].delta.p_positive:.6f}, "
                f"CARA={cara['sdre_minus_as_ewma'].delta.p_positive:.6f}"
            ),
        )
    )

    outcome = _classify_outcome(crra, cara)
    return results, outcome


def print_gates(results: list[tuple[str, bool, str]]) -> None:
    print("\n[stage-4 gates]")
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4s}  {name}: {detail}")


def _classify_outcome(
    crra: dict[str, PairedPosteriorCheck],
    cara: dict[str, PairedPosteriorCheck],
) -> str:
    sdre_linear_ship = (
        crra["sdre_minus_linear_rule"].delta.p_positive >= 0.95
        and cara["sdre_minus_linear_rule"].delta.p_positive >= 0.95
    )
    sdre_as_ship = crra["sdre_minus_as_ewma"].delta.p_positive >= 0.95
    linear_as_ship = crra["linear_rule_minus_as_ewma"].delta.p_positive >= 0.95
    if sdre_linear_ship and sdre_as_ship:
        return "outcome 1: SDRE clears gate and beats A-S"
    if not sdre_linear_ship and linear_as_ship:
        return "outcome 2: state-space win; ship the linear rule"
    return "outcome 3: augmented controllers tie A-S"


def _check_lookup(
    checks: list[PairedPosteriorCheck],
    utility_name: str,
) -> dict[str, PairedPosteriorCheck]:
    return {
        check.contrast: check
        for check in checks
        if check.utility_name == utility_name
    }


def _relative_error(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _sample_sd(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def main() -> int:
    config = ControlAblationConfig()

    print("Option MM Stage-4 Control Ablation")
    print(
        f"pilot_seeds={len(config.pilot_seeds)}, "
        f"formal_seeds={len(config.formal_seeds)}, "
        f"horizon_steps={config.horizon_steps}"
    )

    pilot_runs = run_all_strategies(seeds=config.pilot_seeds, config=config)
    print_pilot_power(pilot_power_rows(pilot_runs))

    formal_runs = run_all_strategies(seeds=config.formal_seeds, config=config)
    ce_crra = absolute_ce_table(formal_runs, crra_utility(config.gamma_ce))
    ce_cara = absolute_ce_table(formal_runs, cara_utility(config.cara_alpha))
    contributions = contribution_table(ce_crra)
    checks = paired_posterior_checks(formal_runs, config)
    gate_results, outcome = build_gate_results(checks)

    print_strategy_summaries(formal_runs)
    print_absolute_ce(ce_crra, ce_cara)
    print_contribution_table(ce_crra, contributions)
    print_posterior_checks(checks)
    print_stage_sentence(config, checks, outcome)
    print_gates(gate_results)

    numerical_results = [
        passed
        for name, passed, _ in gate_results
        if not name.startswith("ship rule")
    ]
    return 0 if all(numerical_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
