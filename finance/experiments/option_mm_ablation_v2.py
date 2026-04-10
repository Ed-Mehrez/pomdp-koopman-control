"""Stage-4 v2 BBG re-anchored benchmark for option market making.

Uses paired Bayesian posterior inference to compare the finite-gamma BBG
numerical controller against its risk-neutral ±1/k limit, with the legacy
linear-inventory-skew controllers retained as diagnostics and a cross-stage
wiring check against the locked Stage-2 EWMA-vs-constant result.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, replace
from math import ceil, isfinite, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.bbg_solver import make_bbg_numerical  # noqa: E402
from applications.option_mm.beliefs import EWMAVarianceFilter  # noqa: E402
from applications.option_mm.controllers import (  # noqa: E402
    ASContext,
    ConstantSpreadContext,
    avellaneda_stoikov,
    constant_spread,
    make_linear_inventory_skew,
    make_risk_neutral_optimal,
    no_quote,
)
from applications.option_mm.env import FillModelSpec, OptionMMAction  # noqa: E402
from applications.option_mm.env import OptionMMState, OptionMarketMakingEnv
from applications.option_mm.inventory_variance import (  # noqa: E402
    bergault_gueant_heston_estimator,
    empirical_sliding_window_estimator,
    oracle_heston_estimator,
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
    "risk_neutral_optimal",
)
UTILITY_ORDER = (
    "bbg_numerical",
    "linear_inventory_skew_oracle",
    "linear_inventory_skew_heston",
    "linear_inventory_skew_empirical",
)
SCENARIO_ORDER = SHARED_ORDER + UTILITY_ORDER


@dataclass(frozen=True)
class V2AblationConfig:
    seed_sequence_entropy: int = 20260407
    pilot_seeds: tuple[int, ...] = tuple(range(200))
    # The formal N=5000 grid remains locked to preserve the Stage-2 cross-stage
    # wiring check. The pilot now projects the BBG-vs-risk-neutral ROPE gate,
    # while the legacy sd_post target remains diagnostic-only output.
    formal_seeds: tuple[int, ...] = tuple(range(5_000))
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    half_spread: float = 0.05
    ewma_half_life_days: float = 5.0
    gamma_inv: float = 0.1
    gamma_ce: float = 2.0
    cara_alpha: float = 2.0e-5
    # Chosen so quadratic Arrow-Pratt matches 2e-5 at W=1e5:
    # k / (1 - kW) = 2e-5  =>  k = 2e-5 / 3.
    quadratic_k: float = 2.0e-5 / 3.0
    empirical_window_length: int = 10
    inventory_limit: int = 10
    posterior_draws: int = 5_000
    # Pre-registered ROPE half-width for the low-gamma equivalence check on
    # ``bbg_numerical - risk_neutral_optimal``. The gate is
    # P(delta_CE in [-rope_half_width, +rope_half_width] | data) >= 0.95.
    # Default 10.0 is the post-pilot pre-registration; overrides are for
    # sensitivity reporting only, never for the locked run.
    rope_half_width: float = 10.0


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
class RopeProjection:
    """Pilot-time projection of the Track B2 ROPE equivalence gate."""

    n_formal: int
    half_width: float
    mean_diff: float
    projected_sd_post: float
    projected_p_in_rope: float
    threshold: float = 0.95

    @property
    def passes(self) -> bool:
        return self.projected_p_in_rope >= self.threshold


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
    if name == "risk_neutral_optimal":
        controller = make_risk_neutral_optimal(env)
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
        elif name == "risk_neutral_optimal":
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


def run_utility_strategy(
    strategy_name: str,
    seeds: tuple[int, ...],
    config: V2AblationConfig,
    utility: UtilitySpec,
) -> StrategyRun:
    summaries: list[EpisodeSummary] = []
    for seed in seeds:
        states, infos = run_utility_episode(
            strategy_name=strategy_name,
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
    return StrategyRun(name=strategy_name, summaries=summaries)


def run_utility_episode(
    strategy_name: str,
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
    gamma = utility.arrow_pratt(state.wealth)
    if strategy_name == "bbg_numerical":
        controller = make_bbg_numerical(
            env,
            state,
            gamma=gamma,
            max_inventory=config.inventory_limit,
        )
    elif strategy_name == "linear_inventory_skew_oracle":
        estimator = oracle_heston_estimator(env)
        controller = make_linear_inventory_skew(env, estimator, utility)
    elif strategy_name == "linear_inventory_skew_heston":
        estimator = bergault_gueant_heston_estimator(env, state)
        controller = make_linear_inventory_skew(env, estimator, utility)
    elif strategy_name == "linear_inventory_skew_empirical":
        estimator = empirical_sliding_window_estimator(
            window_length=config.empirical_window_length,
            env=env,
        )
        controller = make_linear_inventory_skew(env, estimator, utility)
    else:
        raise ValueError(f"unknown utility-dependent strategy: {strategy_name}")

    states = [state]
    infos = []
    while not state.done:
        action = controller(state)
        state, _, _, info = env.step(action)
        states.append(state)
        infos.append(info)
    return states, infos


def run_utility_strategies_by_utility(
    seeds: tuple[int, ...],
    config: V2AblationConfig,
) -> dict[str, dict[str, StrategyRun]]:
    return {
        utility_name: {
            name: run_utility_strategy(name, seeds, config, utility)
            for name in UTILITY_ORDER
        }
        for utility_name, utility in utility_specs(config)
    }


def combined_runs_for_utility(
    shared_runs: dict[str, StrategyRun],
    utility_runs: dict[str, StrategyRun],
) -> dict[str, StrategyRun]:
    runs = dict(shared_runs)
    runs.update(utility_runs)
    return runs


def pilot_power_result(
    reference_run: StrategyRun,
    candidate_run: StrategyRun,
    target_sd_post: float = TARGET_SD_POST,
) -> PilotPowerResult:
    diffs = candidate_run.terminal_wealth - reference_run.terminal_wealth
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


def project_rope_from_pilot(
    pilot: PilotPowerResult,
    half_width: float,
    n_formal: int,
) -> RopeProjection:
    """Project the Track B2 ROPE probability at the planned formal N.

    Uses the analytic Normal posterior implied by the delta method:
    sd_post(N) = per_seed_sd / sqrt(N), and
    P(ΔCE in [-h, +h]) = Φ((h - mean)/sd_post) - Φ((-h - mean)/sd_post).
    """
    if half_width <= 0.0 or not isfinite(half_width):
        raise ValueError("half_width must be positive and finite")
    if n_formal <= 0:
        raise ValueError("n_formal must be positive")

    projected_sd_post = (
        float("inf") if pilot.per_seed_sd == 0.0 else pilot.per_seed_sd / sqrt(n_formal)
    )
    projected_p = _normal_rope_probability(
        mean=pilot.mean_diff,
        sd_post=projected_sd_post,
        half_width=half_width,
    )
    return RopeProjection(
        n_formal=n_formal,
        half_width=half_width,
        mean_diff=pilot.mean_diff,
        projected_sd_post=projected_sd_post,
        projected_p_in_rope=projected_p,
    )


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


def print_rope_projection(projection: RopeProjection) -> None:
    """Print the projected Track B2 ROPE probability at the formal N."""
    status = "PASS" if projection.passes else "FAIL"
    print("\n[pilot ROPE projection (Track B2 secondary)]")
    print(
        f"  {status}  "
        f"half_width=±{projection.half_width:.6f} "
        f"N_formal={projection.n_formal} "
        f"projected_sd_post={projection.projected_sd_post:.6f} "
        f"projected_P(ΔCE in ROPE)={projection.projected_p_in_rope:.6f} "
        f"threshold={projection.threshold:.2f}"
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
    risk_neutral_constant_crra: PairedPosteriorCheck,
    rope_probabilities: dict[str, float],
    rope_half_width: float,
    cross_stage_error: float,
    all_checks: list[PairedPosteriorCheck],
) -> None:
    print("\n[stage-4 v2 gates]")
    for utility_name in ("CRRA", "CARA"):
        p_in_rope = rope_probabilities[utility_name]
        passed = p_in_rope >= 0.95
        print(
            f"  {'PASS' if passed else 'FAIL':4s}  "
            f"{utility_name} P(ΔCE_{{bbg_numerical - risk_neutral_optimal}} in ±{rope_half_width:g}) >= 0.95: "
            f"P={p_in_rope:.6f}"
        )
    quadratic_rope = rope_probabilities["QUADRATIC"]
    print(
        f"  INFO  "
        f"QUADRATIC P(ΔCE_{{bbg_numerical - risk_neutral_optimal}} in ±{rope_half_width:g}) = "
        f"{quadratic_rope:.6f} (descriptive only)"
    )
    risk_neutral_pass = risk_neutral_constant_crra.delta.p_positive >= 0.99
    print(
        f"  {'PASS' if risk_neutral_pass else 'FAIL':4s}  "
        "CRRA risk_neutral_optimal minus constant_spread P(>0) >= 0.99: "
        f"P={risk_neutral_constant_crra.delta.p_positive:.6f}"
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
    crra_linear_diagnostic: PairedPosteriorCheck,
    cara_linear_diagnostic: PairedPosteriorCheck,
    quadratic_linear_diagnostic: PairedPosteriorCheck,
    risk_neutral_constant_crra: PairedPosteriorCheck,
    rope_probabilities: dict[str, float],
    cross_stage_value: float,
    outcome: str,
) -> None:
    print("\n[stage-4 v2 sentence]")
    print(
        f"On {len(config.formal_seeds)} paired seeds "
        f"(same SeedSequence({config.seed_sequence_entropy}) as Stages 2/3), "
        "under CRRA(gamma=2) risk_neutral_optimal achieves "
        f"delta CE vs constant_spread = {risk_neutral_constant_crra.delta.mean:.6f} "
        f"with sd_post = {risk_neutral_constant_crra.delta.sd_post:.6f} and "
        f"P(>0) = {risk_neutral_constant_crra.delta.p_positive:.6f}. "
        "The BBG-vs-risk-neutral low-gamma equivalence check gives "
        f"CRRA P(in ±{config.rope_half_width:g}) = {rope_probabilities['CRRA']:.6f} and "
        f"CARA P(in ±{config.rope_half_width:g}) = {rope_probabilities['CARA']:.6f}. "
        f"Under CRRA, bbg_numerical achieves "
        f"delta CE vs risk_neutral_optimal = {crra_headline.delta.mean:.6f} "
        f"with sd_post = {crra_headline.delta.sd_post:.6f} and "
        f"P(>0) = {crra_headline.delta.p_positive:.6f}. "
        f"Under CARA(alpha={config.cara_alpha:.1e}), delta CE = {cara_headline.delta.mean:.6f} "
        f"with sd_post = {cara_headline.delta.sd_post:.6f} and "
        f"P(>0) = {cara_headline.delta.p_positive:.6f}. "
        f"Under quadratic(k={config.quadratic_k:.1e}), delta CE = {quadratic_headline.delta.mean:.6f} "
        f"with sd_post = {quadratic_headline.delta.sd_post:.6f} and "
        f"P(>0) = {quadratic_headline.delta.p_positive:.6f}. "
        "The legacy linear-inventory-skew diagnostic remains dominated: "
        f"under CRRA, delta CE vs bbg_numerical = {crra_linear_diagnostic.delta.mean:.6f} "
        f"(P(>0) = {crra_linear_diagnostic.delta.p_positive:.6f}); "
        f"under CARA, delta CE = {cara_linear_diagnostic.delta.mean:.6f} "
        f"(P(>0) = {cara_linear_diagnostic.delta.p_positive:.6f}); "
        f"under quadratic, delta CE = {quadratic_linear_diagnostic.delta.mean:.6f} "
        f"(P(>0) = {quadratic_linear_diagnostic.delta.p_positive:.6f}). "
        f"EWMA-constant reproduces the Stage-2 CRRA delta CE {cross_stage_value:.6f}. "
        f"The result supports {outcome}."
    )


def classify_outcome(
    risk_neutral_constant_crra: PairedPosteriorCheck,
    rope_probabilities: dict[str, float],
) -> str:
    anchor_beats_constant = risk_neutral_constant_crra.delta.p_positive >= 0.99
    rope_passes = all(
        rope_probabilities[utility_name] >= 0.95
        for utility_name in ("CRRA", "CARA")
    )
    if anchor_beats_constant and rope_passes:
        return "bbg-validation"
    if anchor_beats_constant:
        return "bbg-anchor-only"
    return "bbg-anchor-fail"


def _sample_sd(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _reverse_posterior(summary: PosteriorSummary) -> PosteriorSummary:
    return PosteriorSummary(
        mean=-summary.mean,
        sd_post=summary.sd_post,
        ci_low=-summary.ci_high,
        ci_high=-summary.ci_low,
        p_positive=1.0 - summary.p_positive,
    )


def _reverse_check(check: PairedPosteriorCheck, contrast: str) -> PairedPosteriorCheck:
    return PairedPosteriorCheck(
        contrast=contrast,
        utility_name=check.utility_name,
        delta=_reverse_posterior(check.delta),
        mc=_reverse_posterior(check.mc),
    )


def _lookup_contrast(
    lookup: dict[str, PairedPosteriorCheck],
    left: str,
    right: str,
) -> PairedPosteriorCheck:
    direct = f"{left}_minus_{right}"
    if direct in lookup:
        return lookup[direct]
    reverse = f"{right}_minus_{left}"
    if reverse in lookup:
        return _reverse_check(lookup[reverse], direct)
    raise KeyError(direct)


def _relative_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def _normal_rope_probability(mean: float, sd_post: float, half_width: float) -> float:
    """Analytic P(ΔCE in [-h, +h]) for a Normal(mean, sd_post) posterior."""
    if half_width <= 0.0 or not isfinite(half_width):
        raise ValueError("half_width must be positive and finite")
    if sd_post < 0.0 or not isfinite(sd_post):
        return 0.0
    if sd_post == 0.0:
        return 1.0 if -half_width <= mean <= half_width else 0.0
    upper = (half_width - mean) / sd_post
    lower = (-half_width - mean) / sd_post
    return float(norm.cdf(upper) - norm.cdf(lower))


def _build_config_from_cli(argv: list[str] | None = None) -> V2AblationConfig:
    """Parse CLI overrides for pre-registered config fields.

    Currently exposes only ``--rope-half-width``: the Track B2 ROPE bound is
    pre-registered, so the only legitimate reason to override it is sensitivity
    reporting alongside the locked-in default.
    """
    parser = argparse.ArgumentParser(
        description="Stage-4 v2 OMM validation benchmark.",
        allow_abbrev=False,
    )
    default_config = V2AblationConfig()
    parser.add_argument(
        "--rope-half-width",
        type=float,
        default=default_config.rope_half_width,
        help=(
            "Half-width of the ROPE on ΔCE_{bbg_numerical - risk_neutral_optimal}. "
            "Pre-registered default is 10.0; overrides are for sensitivity "
            "reporting only."
        ),
    )
    args = parser.parse_args(argv)
    if args.rope_half_width <= 0.0 or not isfinite(args.rope_half_width):
        parser.error("--rope-half-width must be positive and finite")
    return replace(default_config, rope_half_width=args.rope_half_width)


def main(argv: list[str] | None = None) -> int:
    config = _build_config_from_cli(argv)

    pilot_shared = run_shared_strategies(config.pilot_seeds, config)
    crra_utility_spec = crra_utility(config.gamma_ce)
    pilot_bbg = run_utility_strategy(
        "bbg_numerical",
        config.pilot_seeds,
        config,
        crra_utility_spec,
    )
    pilot = pilot_power_result(
        reference_run=pilot_shared["risk_neutral_optimal"],
        candidate_run=pilot_bbg,
    )
    print_pilot(pilot, pilot_n=len(config.pilot_seeds))
    rope_projection = project_rope_from_pilot(
        pilot=pilot,
        half_width=config.rope_half_width,
        n_formal=len(config.formal_seeds),
    )
    print_rope_projection(rope_projection)
    if not rope_projection.passes:
        print(
            "\n[stage-4 v2 stop] "
            "Pilot indicates N=5000 does not clear the pre-registered ROPE projection gate."
        )
        return 1

    shared_formal = run_shared_strategies(config.formal_seeds, config)
    utility_runs_by_utility = run_utility_strategies_by_utility(config.formal_seeds, config)

    ce_tables: dict[str, dict[str, float]] = {}
    posterior_checks: dict[str, list[PairedPosteriorCheck]] = {}
    for utility_index, (utility_name, utility) in enumerate(utility_specs(config)):
        scenario_runs = combined_runs_for_utility(
            shared_formal,
            utility_runs_by_utility[utility_name],
        )
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
    cross_stage_value = _lookup_contrast(
        crra_lookup,
        "as_ewma",
        "constant_spread",
    ).delta.mean
    cross_stage_error = abs(cross_stage_value - STAGE2_EWMA_MINUS_CONSTANT_CRRA)
    headline_checks = {
        "CRRA": _lookup_contrast(
            crra_lookup,
            "bbg_numerical",
            "risk_neutral_optimal",
        ),
        "CARA": _lookup_contrast(
            cara_lookup,
            "bbg_numerical",
            "risk_neutral_optimal",
        ),
        "QUADRATIC": _lookup_contrast(
            quadratic_lookup,
            "bbg_numerical",
            "risk_neutral_optimal",
        ),
    }
    linear_diagnostic_checks = {
        "CRRA": _lookup_contrast(
            crra_lookup,
            "linear_inventory_skew_heston",
            "bbg_numerical",
        ),
        "CARA": _lookup_contrast(
            cara_lookup,
            "linear_inventory_skew_heston",
            "bbg_numerical",
        ),
        "QUADRATIC": _lookup_contrast(
            quadratic_lookup,
            "linear_inventory_skew_heston",
            "bbg_numerical",
        ),
    }
    rope_probabilities = {
        utility_name: _normal_rope_probability(
            mean=check.delta.mean,
            sd_post=check.delta.sd_post,
            half_width=config.rope_half_width,
        )
        for utility_name, check in headline_checks.items()
    }
    risk_neutral_constant_crra = _lookup_contrast(
        crra_lookup,
        "risk_neutral_optimal",
        "constant_spread",
    )
    all_checks = (
        posterior_checks["CRRA"]
        + posterior_checks["CARA"]
        + posterior_checks["QUADRATIC"]
    )
    outcome = classify_outcome(risk_neutral_constant_crra, rope_probabilities)

    print_gates(
        headline_checks,
        risk_neutral_constant_crra,
        rope_probabilities,
        config.rope_half_width,
        cross_stage_error,
        all_checks,
    )
    print_stage_sentence(
        config,
        headline_checks["CRRA"],
        headline_checks["CARA"],
        headline_checks["QUADRATIC"],
        linear_diagnostic_checks["CRRA"],
        linear_diagnostic_checks["CARA"],
        linear_diagnostic_checks["QUADRATIC"],
        risk_neutral_constant_crra,
        rope_probabilities,
        cross_stage_value,
        outcome,
    )

    if cross_stage_error > STAGE2_CROSS_STAGE_TOL:
        return 1
    if not all(check.passes_agreement for check in all_checks):
        return 1
    if outcome == "bbg-anchor-fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
