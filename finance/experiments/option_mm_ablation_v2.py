"""Stage-4 v2 validation benchmark for option market making.

Validates the multiplier-corrected BG+Davis-Lleo closed-form controller against
the Bergault-Guéant analytic baseline, with a cross-stage wiring check against
the locked Stage-2 EWMA-vs-constant result.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path

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
    make_bergault_gueant_closed_form,
    make_sdre_controller_v2,
    no_quote,
)
from applications.option_mm.env import FillModelSpec, OptionMMAction  # noqa: E402
from applications.option_mm.env import OptionMMState, OptionMarketMakingEnv
from applications.option_mm.inventory_variance import (  # noqa: E402
    bergault_gueant_heston_estimator,
    empirical_sliding_window_estimator,
)
from applications.option_mm.metrics import (  # noqa: E402
    EpisodeSummary,
    PosteriorSummary,
    UtilitySpec,
    aggregate_episode_summaries,
    cara_utility,
    crra_utility,
    paired_ce_posterior,
    quadratic_utility,
    summarize_episode,
)


TARGET_SD_POST = 0.5
STAGE2_EWMA_MINUS_CONSTANT_CRRA = 26.967789651505882
STAGE2_CROSS_STAGE_TOL = 1e-9
SHARED_ORDER = (
    "no_quote",
    "constant_spread",
    "as_ewma",
    "bergault_gueant_closed_form",
)
SDRE_ORDER = (
    "sdre_v2_heston",
    "sdre_v2_empirical",
)
SCENARIO_ORDER = SHARED_ORDER + SDRE_ORDER


@dataclass(frozen=True)
class V2AblationConfig:
    seed_sequence_entropy: int = 20260407
    pilot_seeds: tuple[int, ...] = tuple(range(200))
    # Pilot result (2026-04-09): CRRA sdre_v2_heston minus BG at N=200 gave
    # mean_diff=-7.134105, per_seed_sd=105.573530, per_seed_snr=0.067575.
    # To hit the pre-registered target sd_post<=0.5 for the validation contrast
    # would require N~44,584, so the formal N=5000 run is intentionally stopped
    # rather than silently bumped. The Stage-2 cross-stage wiring check still uses
    # tuple(range(5000)) with the same seed entropy when the formal run is enabled.
    formal_seeds: tuple[int, ...] = tuple(range(5_000))
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    half_spread: float = 0.05
    ewma_half_life_days: float = 5.0
    gamma_inv: float = 0.1
    gamma_ce: float = 2.0
    cara_alpha: float = 2.0e-5
    quadratic_k: float = 2.0e-5
    empirical_window_length: int = 10
    inventory_limit: int = 10
    posterior_draws: int = 5_000


@dataclass(frozen=True)
class StrategyRun:
    name: str
    summaries: list[EpisodeSummary]

    @property
    def terminal_wealth(self) -> np.ndarray:
        return np.asarray([summary.terminal_wealth for summary in self.summaries], dtype=float)

    @property
    def total_pnl(self) -> np.ndarray:
        return np.asarray([summary.total_pnl for summary in self.summaries], dtype=float)

    @property
    def gross_spread_capture(self) -> np.ndarray:
        return np.asarray([summary.gross_spread_capture for summary in self.summaries], dtype=float)

    @property
    def net_delta_rms(self) -> np.ndarray:
        return np.asarray([summary.net_delta_rms for summary in self.summaries], dtype=float)


@dataclass(frozen=True)
class PilotPowerResult:
    mean_diff: float
    per_seed_sd: float
    per_seed_snr: float
    target_sd_post: float
    required_n_for_target_sd: int | None


@dataclass(frozen=True)
class PairedPosteriorCheck:
    contrast: str
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


def utility_specs(config: V2AblationConfig) -> tuple[tuple[str, UtilitySpec], ...]:
    return (
        ("CRRA", crra_utility(config.gamma_ce)),
        ("CARA", cara_utility(config.cara_alpha)),
        ("QUADRATIC", quadratic_utility(config.quadratic_k)),
    )


def run_shared_strategies(
    seeds: tuple[int, ...],
    config: V2AblationConfig,
) -> dict[str, StrategyRun]:
    return {
        name: run_shared_strategy(name=name, seeds=seeds, config=config)
        for name in SHARED_ORDER
    }


def run_shared_strategy(
    name: str,
    seeds: tuple[int, ...],
    config: V2AblationConfig,
) -> StrategyRun:
    summaries: list[EpisodeSummary] = []
    for seed in seeds:
        states, infos = run_shared_episode(name=name, seed=seed, config=config)
        summaries.append(
            summarize_episode(
                states=states,
                infos=infos,
                inventory_limit=config.inventory_limit,
            )
        )
    return StrategyRun(name=name, summaries=summaries)


def run_shared_episode(
    name: str,
    seed: int,
    config: V2AblationConfig,
) -> tuple[list[OptionMMState], list]:
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()
    controller = None
    variance_filter = None
    if name == "bergault_gueant_closed_form":
        controller = make_bergault_gueant_closed_form(env)
    elif name == "as_ewma":
        variance_filter = EWMAVarianceFilter(half_life_days=config.ewma_half_life_days)
        variance_filter.reset(initial_variance=state.variance, initial_spot=state.spot)

    states = [state]
    infos = []
    while not state.done:
        if name == "no_quote":
            action = no_quote(state)
        elif name == "constant_spread":
            action = constant_spread(state, ConstantSpreadContext(config.half_spread))
        elif name == "as_ewma":
            assert variance_filter is not None
            horizon_remaining = max((config.horizon_steps - state.step_index) * env.dt, 0.0)
            action = avellaneda_stoikov(
                state,
                ASContext(
                    v_hat=variance_filter.variance,
                    gamma_inv=config.gamma_inv,
                    k_intensity=env.fills.distance_slope,
                    horizon_remaining=horizon_remaining,
                ),
            )
        elif name == "bergault_gueant_closed_form":
            assert controller is not None
            action = controller(state)
        else:
            raise ValueError(f"unknown shared strategy: {name}")

        state, _, _, info = env.step(action)
        if variance_filter is not None:
            variance_filter.update(state.spot)
        states.append(state)
        infos.append(info)

    return states, infos


def run_sdre_strategy(
    estimator_name: str,
    seeds: tuple[int, ...],
    config: V2AblationConfig,
    utility: UtilitySpec,
) -> StrategyRun:
    summaries: list[EpisodeSummary] = []
    for seed in seeds:
        states, infos = run_sdre_episode(
            estimator_name=estimator_name,
            seed=seed,
            config=config,
            utility=utility,
        )
        summaries.append(
            summarize_episode(
                states=states,
                infos=infos,
                inventory_limit=config.inventory_limit,
            )
        )
    return StrategyRun(name=estimator_name, summaries=summaries)


def run_sdre_episode(
    estimator_name: str,
    seed: int,
    config: V2AblationConfig,
    utility: UtilitySpec,
) -> tuple[list[OptionMMState], list]:
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()
    if estimator_name == "sdre_v2_heston":
        estimator = bergault_gueant_heston_estimator(env, state)
    elif estimator_name == "sdre_v2_empirical":
        estimator = empirical_sliding_window_estimator(
            window_length=config.empirical_window_length,
            env=env,
        )
    else:
        raise ValueError(f"unknown estimator strategy: {estimator_name}")
    controller = make_sdre_controller_v2(env, estimator, utility)

    states = [state]
    infos = []
    while not state.done:
        action = controller(state)
        state, _, _, info = env.step(action)
        states.append(state)
        infos.append(info)
    return states, infos


def run_sdre_strategies_by_utility(
    seeds: tuple[int, ...],
    config: V2AblationConfig,
) -> dict[str, dict[str, StrategyRun]]:
    return {
        utility_name: {
            name: run_sdre_strategy(name, seeds, config, utility)
            for name in SDRE_ORDER
        }
        for utility_name, utility in utility_specs(config)
    }


def combined_runs_for_utility(
    shared_runs: dict[str, StrategyRun],
    sdre_runs: dict[str, StrategyRun],
) -> dict[str, StrategyRun]:
    runs = dict(shared_runs)
    runs.update(sdre_runs)
    return runs


def pilot_power_result(
    bg_run: StrategyRun,
    sdre_run: StrategyRun,
    target_sd_post: float = TARGET_SD_POST,
) -> PilotPowerResult:
    diffs = sdre_run.terminal_wealth - bg_run.terminal_wealth
    mean_diff = float(np.mean(diffs))
    per_seed_sd = _sample_sd(diffs)
    per_seed_snr = 0.0 if per_seed_sd == 0.0 else abs(mean_diff) / per_seed_sd
    required_n = None
    if per_seed_sd > 0.0 and isfinite(per_seed_sd):
        required_n = int(ceil((per_seed_sd / target_sd_post) ** 2))
    return PilotPowerResult(
        mean_diff=mean_diff,
        per_seed_sd=per_seed_sd,
        per_seed_snr=float(per_seed_snr),
        target_sd_post=target_sd_post,
        required_n_for_target_sd=required_n,
    )


def absolute_ce_table(
    runs: dict[str, StrategyRun],
    utility: UtilitySpec,
) -> dict[str, float]:
    return {
        name: utility.ce(float(np.mean(utility.u(runs[name].terminal_wealth))))
        for name in SCENARIO_ORDER
    }


def paired_posterior_checks(
    runs: dict[str, StrategyRun],
    utility_name: str,
    utility: UtilitySpec,
    config: V2AblationConfig,
    *,
    mc_seed_entropy: int,
) -> list[PairedPosteriorCheck]:
    pairs = list(itertools.combinations(SCENARIO_ORDER, 2))
    seed_sequence = np.random.SeedSequence([config.seed_sequence_entropy, mc_seed_entropy])
    mc_seeds = seed_sequence.spawn(len(pairs))
    checks: list[PairedPosteriorCheck] = []
    for idx, (name_a, name_b) in enumerate(pairs):
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
            rng=np.random.default_rng(mc_seeds[idx]),
        )
        checks.append(
            PairedPosteriorCheck(
                contrast=contrast,
                utility_name=utility_name,
                delta=delta,
                mc=mc,
            )
        )
    return checks


def print_pilot(result: PilotPowerResult, pilot_n: int) -> None:
    required = "inf" if result.required_n_for_target_sd is None else str(result.required_n_for_target_sd)
    print("\n[pilot power calc]")
    print(
        f"  N={pilot_n} "
        f"mean_diff={result.mean_diff:.6f} "
        f"per_seed_sd={result.per_seed_sd:.6f} "
        f"per_seed_snr={result.per_seed_snr:.6f} "
        f"target_sd_post={result.target_sd_post:.6f} "
        f"required_N_for_target~{required}"
    )


def print_strategy_summaries(runs: dict[str, StrategyRun], utility_name: str) -> None:
    print(f"\n[strategy summaries, {utility_name}]")
    for name in SCENARIO_ORDER:
        agg = aggregate_episode_summaries(runs[name].summaries)
        print(
            f"  {name:24s} terminal_wealth={agg.terminal_wealth_mean: .6f} "
            f"total_pnl={agg.total_pnl_mean: .6f} "
            f"spread={agg.gross_spread_capture_mean: .6f} "
            f"net_delta_rms={agg.net_delta_rms_mean: .6f}"
        )


def print_absolute_ce(ce: dict[str, float], utility_name: str) -> None:
    print(f"\n[absolute CE, {utility_name}]")
    for name in SCENARIO_ORDER:
        print(f"  {name:24s} {ce[name]:16.6f}")


def print_posterior_checks(checks: list[PairedPosteriorCheck]) -> None:
    utility_name = checks[0].utility_name if checks else "UNKNOWN"
    print(f"\n[pairwise posterior checks, {utility_name}]")
    for check in checks:
        print(
            f"  {check.contrast:52s} "
            f"delta_mean={check.delta.mean: .6f} "
            f"delta_sd_post={check.delta.sd_post: .6f} "
            f"delta_P(>0)={check.delta.p_positive: .6f} "
            f"mc_mean={check.mc.mean: .6f} "
            f"mc_sd_post={check.mc.sd_post: .6f} "
            f"mean_rel={check.mean_rel_error: .6f} "
            f"sd_post_rel={check.sd_post_rel_error: .6f}"
        )


def print_gates(
    headline_checks: dict[str, PairedPosteriorCheck],
    bg_constant_crra: PairedPosteriorCheck,
    cross_stage_error: float,
    all_checks: list[PairedPosteriorCheck],
) -> None:
    print("\n[stage-4 v2 gates]")
    for utility_name in ("CRRA", "CARA", "QUADRATIC"):
        check = headline_checks[utility_name]
        z_like = float("inf") if check.delta.sd_post == 0.0 else abs(check.delta.mean) / check.delta.sd_post
        passed = z_like < 2.0
        print(
            f"  {'PASS' if passed else 'FAIL':4s}  "
            f"{utility_name} |ΔCE_sdre_v2_heston_minus_BG| / sd_post < 2: "
            f"value={z_like:.6f}"
        )
    bg_constant_pass = bg_constant_crra.delta.p_positive >= 0.95
    print(
        f"  {'PASS' if bg_constant_pass else 'FAIL':4s}  "
        "CRRA BG minus constant_spread P(>0) >= 0.95: "
        f"P={bg_constant_crra.delta.p_positive:.6f}"
    )
    cross_stage_pass = cross_stage_error <= STAGE2_CROSS_STAGE_TOL
    print(
        f"  {'PASS' if cross_stage_pass else 'FAIL':4s}  "
        "EWMA-constant reproduces Stage-2 CRRA delta CE: "
        f"error={cross_stage_error:.12g}"
    )
    agreement_pass = all(check.passes_agreement for check in all_checks)
    worst_mean = max(check.mean_rel_error for check in all_checks)
    worst_sd = max(check.sd_post_rel_error for check in all_checks)
    print(
        f"  {'PASS' if agreement_pass else 'FAIL':4s}  "
        "delta-vs-MC agreement <= 5% for all reported contrasts: "
        f"max_mean_rel={worst_mean:.6f}, max_sd_post_rel={worst_sd:.6f}"
    )


def print_stage_sentence(
    config: V2AblationConfig,
    crra_headline: PairedPosteriorCheck,
    cara_headline: PairedPosteriorCheck,
    quadratic_headline: PairedPosteriorCheck,
    cross_stage_value: float,
    outcome: str,
) -> None:
    print("\n[stage-4 v2 sentence]")
    print(
        f"On {len(config.formal_seeds)} paired seeds "
        f"(same SeedSequence({config.seed_sequence_entropy}) as Stages 2/3), "
        "under CRRA(gamma=2) sdre_v2_heston achieves "
        f"delta CE vs bergault_gueant_closed_form = {crra_headline.delta.mean:.6f} "
        f"with sd_post = {crra_headline.delta.sd_post:.6f} and "
        f"P(>0) = {crra_headline.delta.p_positive:.6f}. "
        f"Under CARA(alpha={config.cara_alpha:.1e}), delta CE = {cara_headline.delta.mean:.6f} "
        f"with sd_post = {cara_headline.delta.sd_post:.6f} and "
        f"P(>0) = {cara_headline.delta.p_positive:.6f}. "
        f"Under quadratic(k={config.quadratic_k:.1e}), delta CE = {quadratic_headline.delta.mean:.6f} "
        f"with sd_post = {quadratic_headline.delta.sd_post:.6f} and "
        f"P(>0) = {quadratic_headline.delta.p_positive:.6f}. "
        f"EWMA-constant reproduces the Stage-2 CRRA delta CE {cross_stage_value:.6f}. "
        f"The result supports {outcome}."
    )


def classify_outcome(
    headline_checks: dict[str, PairedPosteriorCheck],
    bg_constant_crra: PairedPosteriorCheck,
) -> str:
    all_validate = all(
        (float("inf") if check.delta.sd_post == 0.0 else abs(check.delta.mean) / check.delta.sd_post) < 2.0
        for check in headline_checks.values()
    )
    bg_beats_constant = bg_constant_crra.delta.p_positive >= 0.95
    if all_validate and bg_beats_constant:
        return "v2-validation"
    if all(
        check.delta.mean > 0.0
        and (float("inf") if check.delta.sd_post == 0.0 else abs(check.delta.mean) / check.delta.sd_post) >= 2.0
        for check in headline_checks.values()
    ):
        return "v2-improvement"
    if all(
        check.delta.mean < 0.0
        and (float("inf") if check.delta.sd_post == 0.0 else abs(check.delta.mean) / check.delta.sd_post) >= 2.0
        for check in headline_checks.values()
    ):
        return "v2-disagreement"
    return "v2-still-null"


def _sample_sd(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _relative_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def main() -> int:
    config = V2AblationConfig()

    pilot_shared = run_shared_strategies(config.pilot_seeds, config)
    crra_utility_spec = crra_utility(config.gamma_ce)
    pilot_sdre = run_sdre_strategy(
        "sdre_v2_heston",
        config.pilot_seeds,
        config,
        crra_utility_spec,
    )
    pilot = pilot_power_result(
        bg_run=pilot_shared["bergault_gueant_closed_form"],
        sdre_run=pilot_sdre,
    )
    print_pilot(pilot, pilot_n=len(config.pilot_seeds))
    if pilot.required_n_for_target_sd is None or pilot.required_n_for_target_sd > len(config.formal_seeds):
        print(
            "\n[stage-4 v2 stop] "
            "Pilot indicates N=5000 is not enough for the target sd_post precision."
        )
        return 1

    shared_formal = run_shared_strategies(config.formal_seeds, config)
    sdre_by_utility = run_sdre_strategies_by_utility(config.formal_seeds, config)

    scenario_results: dict[str, dict[str, StrategyRun]] = {}
    ce_tables: dict[str, dict[str, float]] = {}
    posterior_checks: dict[str, list[PairedPosteriorCheck]] = {}
    for utility_index, (utility_name, utility) in enumerate(utility_specs(config)):
        scenario_runs = combined_runs_for_utility(shared_formal, sdre_by_utility[utility_name])
        scenario_results[utility_name] = scenario_runs
        ce_tables[utility_name] = absolute_ce_table(scenario_runs, utility)
        posterior_checks[utility_name] = paired_posterior_checks(
            scenario_runs,
            utility_name,
            utility,
            config,
            mc_seed_entropy=utility_index + 1,
        )
        print_strategy_summaries(scenario_runs, utility_name)
        print_absolute_ce(ce_tables[utility_name], utility_name)
        print_posterior_checks(posterior_checks[utility_name])

    crra_lookup = {check.contrast: check for check in posterior_checks["CRRA"]}
    cara_lookup = {check.contrast: check for check in posterior_checks["CARA"]}
    quadratic_lookup = {check.contrast: check for check in posterior_checks["QUADRATIC"]}
    cross_stage_value = crra_lookup["as_ewma_minus_constant_spread"].delta.mean
    cross_stage_error = abs(cross_stage_value - STAGE2_EWMA_MINUS_CONSTANT_CRRA)
    headline_checks = {
        "CRRA": crra_lookup["sdre_v2_heston_minus_bergault_gueant_closed_form"],
        "CARA": cara_lookup["sdre_v2_heston_minus_bergault_gueant_closed_form"],
        "QUADRATIC": quadratic_lookup["sdre_v2_heston_minus_bergault_gueant_closed_form"],
    }
    bg_constant_crra = crra_lookup["bergault_gueant_closed_form_minus_constant_spread"]
    all_checks = (
        posterior_checks["CRRA"]
        + posterior_checks["CARA"]
        + posterior_checks["QUADRATIC"]
    )
    outcome = classify_outcome(headline_checks, bg_constant_crra)

    print_gates(headline_checks, bg_constant_crra, cross_stage_error, all_checks)
    print_stage_sentence(
        config,
        headline_checks["CRRA"],
        headline_checks["CARA"],
        headline_checks["QUADRATIC"],
        cross_stage_value,
        outcome,
    )

    if cross_stage_error > STAGE2_CROSS_STAGE_TOL:
        return 1
    if not all(check.passes_agreement for check in all_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
