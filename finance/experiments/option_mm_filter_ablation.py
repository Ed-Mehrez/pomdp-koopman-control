"""Stage-3 filter ablation for option market making.

Keeps the Stage-2 Avellaneda-Stoikov controller fixed and varies only the
variance filter. The ordering of filters is the result, not a tuning target.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from applications.option_mm.beliefs import (  # noqa: E402
    BootstrapParticleFilter,
    EWMAVarianceFilter,
    OracleVarianceFilter,
    RecursiveSigRLSFilter,
)
from applications.option_mm.controllers import (  # noqa: E402
    ASContext,
    ConstantSpreadContext,
    avellaneda_stoikov,
    constant_spread,
    no_quote,
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
CONTRASTS = (
    ("oracle_minus_bpf", "oracle", "bpf"),
    ("bpf_minus_recsig", "bpf", "recsig_rls"),
    ("recsig_minus_ewma", "recsig_rls", "ewma"),
    ("ewma_minus_constant", "ewma", "constant_spread"),
)
DISPLAY_ORDER = (
    "oracle",
    "bpf",
    "recsig_rls",
    "ewma",
    "constant_spread",
    "no_quote",
)


@dataclass(frozen=True)
class FilterAblationConfig:
    seed_sequence_entropy: int = 20260407
    pilot_seeds: tuple[int, ...] = tuple(range(100))
    # Formal N=5000 reuses the Stage-2 paired path grid; pilot power is printed first.
    formal_seeds: tuple[int, ...] = tuple(range(5_000))
    horizon_steps: int = 20
    initial_cash: float = 100_000.0
    half_spread: float = 0.05
    ewma_half_life_days: float = 5.0
    gamma_inv: float = 0.1
    gamma_ce: float = 2.0
    cara_alpha: float = 2.0e-5
    inventory_limit: int = 10
    bpf_particles: int = 200
    recsig_signature_forgetting: float = 0.99
    recsig_blr_forgetting: float = 0.999
    posterior_draws: int = 5_000


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: Literal["no_quote", "constant_spread", "as"]
    filter_factory: Callable[[int, OptionMarketMakingEnv, FilterAblationConfig], Any] | None


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


def make_oracle_filter(
    seed: int,
    env: OptionMarketMakingEnv,
    config: FilterAblationConfig,
) -> OracleVarianceFilter:
    del seed, config
    return OracleVarianceFilter(variance_floor=env.heston.variance_floor)


def make_bpf_filter(
    seed: int,
    env: OptionMarketMakingEnv,
    config: FilterAblationConfig,
) -> BootstrapParticleFilter:
    return BootstrapParticleFilter(
        heston=env.heston,
        dt=env.dt,
        n_particles=config.bpf_particles,
        seed=_child_seed(config.seed_sequence_entropy, seed, stream=31),
        variance_floor=max(env.heston.variance_floor, 1e-8),
    )


def make_recsig_filter(
    seed: int,
    env: OptionMarketMakingEnv,
    config: FilterAblationConfig,
) -> RecursiveSigRLSFilter:
    del seed
    return RecursiveSigRLSFilter(
        dt=env.dt,
        signature_forgetting=config.recsig_signature_forgetting,
        blr_forgetting=config.recsig_blr_forgetting,
        variance_floor=max(env.heston.variance_floor, 1e-8),
    )


def make_ewma_filter(
    seed: int,
    env: OptionMarketMakingEnv,
    config: FilterAblationConfig,
) -> EWMAVarianceFilter:
    del seed, env
    return EWMAVarianceFilter(half_life_days=config.ewma_half_life_days)


def strategy_specs() -> tuple[StrategySpec, ...]:
    return (
        StrategySpec("no_quote", "no_quote", None),
        StrategySpec("constant_spread", "constant_spread", None),
        StrategySpec("ewma", "as", make_ewma_filter),
        StrategySpec("recsig_rls", "as", make_recsig_filter),
        StrategySpec("bpf", "as", make_bpf_filter),
        StrategySpec("oracle", "as", make_oracle_filter),
    )


def run_all_strategies(
    seeds: tuple[int, ...],
    config: FilterAblationConfig,
) -> dict[str, StrategyRun]:
    return {
        spec.name: run_strategy(spec=spec, seeds=seeds, config=config)
        for spec in strategy_specs()
    }


def run_strategy(
    spec: StrategySpec,
    seeds: tuple[int, ...],
    config: FilterAblationConfig,
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
    spec: StrategySpec,
    config: FilterAblationConfig,
) -> tuple[list[OptionMMState], list]:
    env = OptionMarketMakingEnv(
        fills=FillModelSpec(same_step_both_fills_policy="mid_drift"),
        horizon_steps=config.horizon_steps,
        initial_cash=config.initial_cash,
        seed=seed,
    )
    state = env.reset()
    variance_filter = None
    if spec.filter_factory is not None:
        variance_filter = spec.filter_factory(seed, env, config)
        variance_filter.reset(initial_variance=state.variance, initial_spot=state.spot)

    states = [state]
    infos = []
    while not state.done:
        action = action_for_spec(spec, state, variance_filter, env, config)
        state, _, _, info = env.step(action)
        if variance_filter is not None:
            variance_filter.update(
                state.spot,
                true_variance=state.variance,
                state=state,
            )
        states.append(state)
        infos.append(info)

    return states, infos


def action_for_spec(
    spec: StrategySpec,
    state: OptionMMState,
    variance_filter: Any,
    env: OptionMarketMakingEnv,
    config: FilterAblationConfig,
) -> OptionMMAction:
    if spec.kind == "no_quote":
        return no_quote(state)
    if spec.kind == "constant_spread":
        return constant_spread(state, ConstantSpreadContext(config.half_spread))
    if variance_filter is None:
        raise RuntimeError("A-S strategy requires a variance filter")

    horizon_remaining = max((config.horizon_steps - state.step_index) * env.dt, 0.0)
    return avellaneda_stoikov(
        state,
        ASContext(
            v_hat=variance_filter.variance,
            gamma_inv=config.gamma_inv,
            k_intensity=env.fills.distance_slope,
            horizon_remaining=horizon_remaining,
        ),
    )


def pilot_power_rows(
    runs: dict[str, StrategyRun],
) -> list[PilotPowerRow]:
    rows = []
    for contrast, name_a, name_b in CONTRASTS:
        diffs = runs[name_a].terminal_wealth - runs[name_b].terminal_wealth
        mean_diff = float(np.mean(diffs))
        per_seed_sd = _sample_sd(diffs)
        if per_seed_sd == 0.0:
            snr = float("inf") if mean_diff != 0.0 else 0.0
        else:
            snr = abs(mean_diff) / per_seed_sd
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
        for name in DISPLAY_ORDER
    }


def paired_posterior_checks(
    runs: dict[str, StrategyRun],
    config: FilterAblationConfig,
) -> list[PairedPosteriorCheck]:
    crra = crra_utility(config.gamma_ce)
    cara = cara_utility(config.cara_alpha)
    utilities = (("CRRA", crra), ("CARA", cara))
    seed_sequence = np.random.SeedSequence(config.seed_sequence_entropy)
    mc_seeds = seed_sequence.spawn(len(CONTRASTS) * len(utilities))
    checks: list[PairedPosteriorCheck] = []

    seed_index = 0
    for utility_name, utility in utilities:
        for contrast, name_a, name_b in CONTRASTS:
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
                    utility_name=utility_name,
                    delta=delta,
                    mc=mc,
                )
            )
    return checks


def contribution_table(
    ce_crra: dict[str, float],
) -> dict[str, float]:
    oracle_gap = ce_crra["oracle"] - ce_crra["constant_spread"]
    rows = {}
    for name in ("oracle", "bpf", "recsig_rls", "ewma"):
        if abs(oracle_gap) <= 1e-12:
            rows[name] = float("nan")
        else:
            rows[name] = (
                (ce_crra[name] - ce_crra["constant_spread"]) / oracle_gap
            )
    return rows


def print_pilot_power(rows: list[PilotPowerRow]) -> None:
    print("\n[pilot power calc, N=100]")
    for row in rows:
        required = (
            "inf" if row.required_n_for_95 is None else str(row.required_n_for_95)
        )
        print(
            f"  {row.contrast:20s} mean_diff={row.mean_diff: .6f} "
            f"per_seed_sd={row.per_seed_sd: .6f} "
            f"per_seed_snr={row.per_seed_snr: .6f} "
            f"N_for_P95~{required}"
        )


def print_strategy_summaries(runs: dict[str, StrategyRun]) -> None:
    print("\n[strategy summaries]")
    for name in DISPLAY_ORDER:
        agg = aggregate_episode_summaries(runs[name].summaries)
        print(
            f"  {name:16s} terminal_wealth={agg.terminal_wealth_mean: .6f} "
            f"total_pnl={agg.total_pnl_mean: .6f} "
            f"spread={agg.gross_spread_capture_mean: .6f} "
            f"net_delta_rms={agg.net_delta_rms_mean: .6f}"
        )


def print_absolute_ce(ce_crra: dict[str, float], ce_cara: dict[str, float]) -> None:
    print("\n[absolute CE]")
    print("  strategy              CRRA_gamma_2       CARA_alpha_2e-5")
    for name in DISPLAY_ORDER:
        print(f"  {name:16s} {ce_crra[name]:16.6f} {ce_cara[name]:20.6f}")


def print_contribution_table(
    ce_crra: dict[str, float],
    contributions: dict[str, float],
) -> None:
    print("\n[filter contribution, CRRA]")
    print("  filter              CE          delta_vs_constant  oracle_gap_fraction")
    for name in ("oracle", "bpf", "recsig_rls", "ewma"):
        delta_vs_constant = ce_crra[name] - ce_crra["constant_spread"]
        print(
            f"  {name:12s} {ce_crra[name]:12.6f} "
            f"{delta_vs_constant:18.6f} {100.0 * contributions[name]:18.3f}%"
        )


def print_posterior_checks(checks: list[PairedPosteriorCheck]) -> None:
    print("\n[pairwise posterior checks]")
    for check in checks:
        print(
            f"  {check.utility_name:4s} {check.contrast:20s} "
            f"delta_mean={check.delta.mean: .6f} "
            f"delta_sd_post={check.delta.sd_post: .6f} "
            f"delta_CrI=[{check.delta.ci_low: .6f}, {check.delta.ci_high: .6f}] "
            f"delta_P(>0)={check.delta.p_positive: .6f} "
            f"mc_mean={check.mc.mean: .6f} mc_sd_post={check.mc.sd_post: .6f} "
            f"mean_rel={check.mean_rel_error: .6f} "
            f"sd_post_rel={check.sd_post_rel_error: .6f}"
        )


def print_success_sentence(
    config: FilterAblationConfig,
    ce_crra: dict[str, float],
    ce_cara: dict[str, float],
    contributions: dict[str, float],
    checks: list[PairedPosteriorCheck],
) -> None:
    crra_checks = [check for check in checks if check.utility_name == "CRRA"]
    max_rel = max(
        max(check.mean_rel_error, check.sd_post_rel_error) for check in checks
    )
    cara_order = _ordering(ce_cara)
    crra_order = _ordering(ce_crra)
    ordering_text = (
        "the same orderings hold"
        if cara_order == crra_order
        else f"the CARA ordering is {cara_order}"
    )
    crra_by_contrast = {check.contrast: check.delta for check in crra_checks}

    print("\n[stage-3 sentence]")
    print(
        f"On {len(config.formal_seeds)} paired seeds "
        f"(same Heston path per seed via path_rng, SeedSequence({config.seed_sequence_entropy})), "
        f"A-S with each filter achieves CE under CRRA(gamma=2): "
        f"oracle = {ce_crra['oracle']:.6f}, BPF = {ce_crra['bpf']:.6f}, "
        f"RecSig = {ce_crra['recsig_rls']:.6f}, EWMA = {ce_crra['ewma']:.6f}, "
        f"constant_spread = {ce_crra['constant_spread']:.6f}. "
        "Pairwise paired delta CE under CRRA: "
        f"oracle-BPF = {crra_by_contrast['oracle_minus_bpf'].mean:.6f} +/- "
        f"{crra_by_contrast['oracle_minus_bpf'].sd_post:.6f} "
        f"(P={crra_by_contrast['oracle_minus_bpf'].p_positive:.6f}), "
        f"BPF-RecSig = {crra_by_contrast['bpf_minus_recsig'].mean:.6f} +/- "
        f"{crra_by_contrast['bpf_minus_recsig'].sd_post:.6f} "
        f"(P={crra_by_contrast['bpf_minus_recsig'].p_positive:.6f}), "
        f"RecSig-EWMA = {crra_by_contrast['recsig_minus_ewma'].mean:.6f} +/- "
        f"{crra_by_contrast['recsig_minus_ewma'].sd_post:.6f} "
        f"(P={crra_by_contrast['recsig_minus_ewma'].p_positive:.6f}), "
        f"EWMA-constant = {crra_by_contrast['ewma_minus_constant'].mean:.6f} +/- "
        f"{crra_by_contrast['ewma_minus_constant'].sd_post:.6f} "
        f"(P={crra_by_contrast['ewma_minus_constant'].p_positive:.6f}). "
        "Filter contribution: oracle captures 100% by definition; "
        f"BPF captures {100.0 * contributions['bpf']:.3f}%, "
        f"RecSig captures {100.0 * contributions['recsig_rls']:.3f}%, "
        f"EWMA captures {100.0 * contributions['ewma']:.3f}%. "
        f"Under CARA(alpha=2e-5) {ordering_text}. "
        f"Delta-method and MC posteriors agree to within {100.0 * max_rel:.3f}% "
        "relative across all 8 contrasts."
    )


def build_gate_results(checks: list[PairedPosteriorCheck]) -> list[tuple[str, bool, str]]:
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

    direction_ok, direction_detail = direction_consistency(checks)
    results.append(("CRRA/CARA confident direction consistency", direction_ok, direction_detail))
    return results


def direction_consistency(checks: list[PairedPosteriorCheck]) -> tuple[bool, str]:
    by_contrast: dict[str, dict[str, PosteriorSummary]] = {}
    for check in checks:
        by_contrast.setdefault(check.contrast, {})[check.utility_name] = check.delta

    mismatches = []
    for contrast, summaries in by_contrast.items():
        crra = summaries["CRRA"]
        cara = summaries["CARA"]
        confident = (
            crra.p_positive >= 0.95
            or crra.p_positive <= 0.05
            or cara.p_positive >= 0.95
            or cara.p_positive <= 0.05
        )
        if confident and np.sign(crra.mean) != np.sign(cara.mean):
            mismatches.append(contrast)

    if mismatches:
        return False, "mismatches=" + ",".join(mismatches)
    return True, "no confident direction mismatches"


def print_gates(results: list[tuple[str, bool, str]]) -> None:
    print("\n[stage-3 numerical gates]")
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4s}  {name}: {detail}")


def _child_seed(entropy: int, seed: int, stream: int) -> int:
    seed_sequence = np.random.SeedSequence([entropy, seed, stream])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _relative_error(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _sample_sd(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _ordering(ce_values: dict[str, float]) -> str:
    names = ("oracle", "bpf", "recsig_rls", "ewma", "constant_spread")
    ordered = sorted(names, key=lambda name: ce_values[name], reverse=True)
    return " > ".join(ordered)


def main() -> int:
    config = FilterAblationConfig()

    print("Option MM Stage-3 Filter Ablation")
    print(
        f"pilot_seeds={len(config.pilot_seeds)}, "
        f"formal_seeds={len(config.formal_seeds)}, "
        f"horizon_steps={config.horizon_steps}"
    )

    pilot_runs = run_all_strategies(seeds=config.pilot_seeds, config=config)
    print_pilot_power(pilot_power_rows(pilot_runs))

    formal_runs = run_all_strategies(seeds=config.formal_seeds, config=config)
    crra = crra_utility(config.gamma_ce)
    cara = cara_utility(config.cara_alpha)
    ce_crra = absolute_ce_table(formal_runs, crra)
    ce_cara = absolute_ce_table(formal_runs, cara)
    contributions = contribution_table(ce_crra)
    checks = paired_posterior_checks(formal_runs, config)
    gate_results = build_gate_results(checks)

    print_strategy_summaries(formal_runs)
    print_absolute_ce(ce_crra, ce_cara)
    print_contribution_table(ce_crra, contributions)
    print_posterior_checks(checks)
    print_success_sentence(config, ce_crra, ce_cara, contributions, checks)
    print_gates(gate_results)

    return 0 if all(passed for _, passed, _ in gate_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
