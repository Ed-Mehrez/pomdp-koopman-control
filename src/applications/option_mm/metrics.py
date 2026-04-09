"""Metrics for the option market-making benchmark.

The measurement layer is deliberately independent of controller code. It consumes
environment states and step diagnostics, then reports wealth, inventory, spread,
and simulator-health quantities that should be stable before any policy tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import norm, t

from .env import OptionMMStepInfo, OptionMMState


@dataclass(frozen=True)
class PosteriorSummary:
    """Posterior summary for a Bayesian contrast."""

    mean: float
    sd_post: float
    ci_low: float
    ci_high: float
    p_positive: float


@dataclass(frozen=True)
class UtilitySpec:
    """Closed-form utility specification for delta-method CE inference."""

    u: Callable[[np.ndarray], np.ndarray]
    ce: Callable[[float], float]
    ce_grad: Callable[[float], float]
    arrow_pratt: Callable[[float], float]
    requires_positive_wealth: bool


@dataclass(frozen=True)
class EpisodeSummary:
    """Per-episode summary statistics."""

    terminal_wealth: float
    total_pnl: float
    mean_pnl: float
    pnl_vol: float
    max_drawdown: float
    final_inventory: int
    inventory_mean: float
    inventory_abs_mean: float
    inventory_abs_p95: float
    inventory_abs_max: float
    time_at_inventory_limit: float
    gross_spread_capture: float
    net_spread_capture: float
    adverse_selection_cost: float
    option_turnover: float
    stock_turnover: float
    total_fees_and_costs: float
    net_delta_rms: float
    censoring_rate: float
    variance_floor_binding_rate: float
    same_step_both_fill_rate: float
    n_steps: int


@dataclass(frozen=True)
class AggregateSummary:
    """Cross-episode mean/std summary."""

    n_episodes: int
    terminal_wealth_mean: float
    terminal_wealth_std: float
    total_pnl_mean: float
    total_pnl_std: float
    mean_pnl_mean: float
    pnl_vol_mean: float
    max_drawdown_mean: float
    max_drawdown_max: float
    final_inventory_mean: float
    final_inventory_std: float
    inventory_abs_mean: float
    inventory_abs_p95_mean: float
    inventory_abs_max_mean: float
    inventory_abs_max_max: float
    time_at_inventory_limit_mean: float
    gross_spread_capture_mean: float
    net_spread_capture_mean: float
    adverse_selection_cost_mean: float
    option_turnover_mean: float
    stock_turnover_mean: float
    total_fees_and_costs_mean: float
    net_delta_rms_mean: float
    censoring_rate_mean: float
    variance_floor_binding_rate_mean: float
    same_step_both_fill_rate_mean: float


def crra_utility(gamma: float) -> UtilitySpec:
    """CRRA utility, with log utility at gamma=1."""
    if gamma < 0.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and nonnegative")

    if abs(gamma - 1.0) < 1e-12:
        return UtilitySpec(
            u=lambda wealth: np.log(wealth),
            ce=lambda mean_u: float(np.exp(mean_u)),
            ce_grad=lambda mean_u: float(np.exp(mean_u)),
            arrow_pratt=lambda wealth: float(gamma / wealth),
            requires_positive_wealth=True,
        )
    if abs(gamma - 2.0) < 1e-12:
        return UtilitySpec(
            u=lambda wealth: -1.0 / wealth,
            ce=lambda mean_u: _crra_power_ce(mean_u, gamma),
            ce_grad=lambda mean_u: float(1.0 / (mean_u * mean_u)),
            arrow_pratt=lambda wealth: float(gamma / wealth),
            requires_positive_wealth=True,
        )

    exponent = 1.0 - gamma

    def _u(wealth: np.ndarray) -> np.ndarray:
        return wealth ** exponent / exponent

    def _ce(mean_u: float) -> float:
        return _crra_power_ce(mean_u, gamma)

    def _ce_grad(mean_u: float) -> float:
        ce = _ce(mean_u)
        return float(ce ** gamma)

    return UtilitySpec(
        u=_u,
        ce=_ce,
        ce_grad=_ce_grad,
        arrow_pratt=lambda wealth: float(gamma / wealth),
        requires_positive_wealth=True,
    )


def cara_utility(alpha: float) -> UtilitySpec:
    """CARA utility U(W) = -exp(-alpha W), valid for any real wealth."""
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be positive and finite")

    def _ce(mean_u: float) -> float:
        if mean_u >= 0.0:
            raise ValueError("expected utility is outside the CARA CE domain")
        return float(-np.log(-mean_u) / alpha)

    def _ce_grad(mean_u: float) -> float:
        if mean_u >= 0.0:
            raise ValueError("expected utility is outside the CARA CE domain")
        return float(-1.0 / (alpha * mean_u))

    return UtilitySpec(
        u=lambda wealth: -np.exp(-alpha * wealth),
        ce=_ce,
        ce_grad=_ce_grad,
        arrow_pratt=lambda wealth: float(alpha),
        requires_positive_wealth=False,
    )


def quadratic_utility(k: float) -> UtilitySpec:
    """Quadratic utility U(W) = W - 0.5 k W^2 on the domain 1 - kW > 0."""
    if k <= 0.0 or not np.isfinite(k):
        raise ValueError("k must be positive and finite")

    def _u(wealth: np.ndarray) -> np.ndarray:
        return wealth - 0.5 * k * wealth * wealth

    def _ce(mean_u: float) -> float:
        discriminant = 1.0 - 2.0 * k * mean_u
        if discriminant <= 0.0:
            raise ValueError("expected utility is outside the quadratic CE domain")
        return float((1.0 - np.sqrt(discriminant)) / k)

    def _ce_grad(mean_u: float) -> float:
        discriminant = 1.0 - 2.0 * k * mean_u
        if discriminant <= 0.0:
            raise ValueError("expected utility is outside the quadratic CE domain")
        return float(1.0 / np.sqrt(discriminant))

    def _arrow_pratt(wealth: float) -> float:
        denominator = 1.0 - k * wealth
        if denominator <= 0.0:
            raise ValueError("quadratic utility Arrow-Pratt is undefined for 1 - kW <= 0")
        return float(k / denominator)

    return UtilitySpec(
        u=_u,
        ce=_ce,
        ce_grad=_ce_grad,
        arrow_pratt=_arrow_pratt,
        requires_positive_wealth=False,
    )


def certainty_equivalent(
    terminal_wealth: Sequence[float],
    gamma: float,
    weights: Sequence[float] | None = None,
) -> float:
    """Return weighted CRRA certainty equivalent for positive terminal wealth."""
    wealth = np.asarray(terminal_wealth, dtype=float)
    if wealth.ndim != 1 or wealth.size == 0:
        raise ValueError("terminal_wealth must be a nonempty 1D array")
    if not np.all(np.isfinite(wealth)):
        raise ValueError("terminal_wealth must be finite")
    if np.any(wealth <= 0.0):
        raise ValueError("CRRA certainty equivalent requires terminal wealth > 0")
    if gamma < 0.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and nonnegative")

    utility = crra_utility(gamma)
    if weights is None:
        normalized_weights = np.full(wealth.shape[0], 1.0 / wealth.shape[0])
    else:
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.shape != wealth.shape:
            raise ValueError("weights must have the same shape as terminal_wealth")
        if not np.all(np.isfinite(normalized_weights)):
            raise ValueError("weights must be finite")
        if np.any(normalized_weights < 0.0):
            raise ValueError("weights must be nonnegative")
        weight_sum = float(np.sum(normalized_weights))
        if weight_sum <= 0.0:
            raise ValueError("weights must have positive total mass")
        normalized_weights = normalized_weights / weight_sum

    return utility.ce(float(np.sum(normalized_weights * utility.u(wealth))))


def paired_ce_posterior(
    wealth_a: Sequence[float],
    wealth_b: Sequence[float],
    *,
    gamma: float | None = None,
    utility: UtilitySpec | None = None,
    method: Literal["delta", "mc", "bootstrap"] = "delta",
    n_draws: int = 5_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> PosteriorSummary:
    """Posterior over the paired CE contrast CE(W_a) - CE(W_b).

    Specify exactly one of ``gamma`` (CRRA shortcut) or ``utility``. The default
    method is the analytic Bayesian delta method on paired utility means.
    """
    if (gamma is None) == (utility is None):
        raise ValueError("specify exactly one of `gamma` or `utility`")
    if utility is None:
        utility = crra_utility(float(gamma))

    a, b = _validate_paired_arrays(wealth_a, wealth_b, "wealth_a", "wealth_b")
    if utility.requires_positive_wealth and (np.any(a <= 0.0) or np.any(b <= 0.0)):
        raise ValueError("utility requires strictly positive wealth")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if method not in {"delta", "mc", "bootstrap"}:
        raise ValueError("method must be one of {'delta', 'mc', 'bootstrap'}")

    if method == "bootstrap":
        return _paired_ce_bayesian_bootstrap(
            a,
            b,
            utility=utility,
            n_draws=n_draws,
            alpha=alpha,
            rng=rng,
        )

    utilities_a = utility.u(a)
    utilities_b = utility.u(b)
    utility_means = np.array([np.mean(utilities_a), np.mean(utilities_b)])
    utility_cov = _paired_mean_covariance(utilities_a, utilities_b)

    if method == "mc":
        generator = rng or np.random.default_rng()
        utility_mean_draws = generator.multivariate_normal(
            mean=utility_means,
            cov=utility_cov,
            size=n_draws,
            check_valid="ignore",
        )
        delta = np.asarray(
            [
                utility.ce(draw[0]) - utility.ce(draw[1])
                for draw in utility_mean_draws
            ],
            dtype=float,
        )
        return posterior_summary(delta, alpha=alpha)

    ce_a = utility.ce(utility_means[0])
    ce_b = utility.ce(utility_means[1])
    gradient = np.array(
        [
            utility.ce_grad(utility_means[0]),
            -utility.ce_grad(utility_means[1]),
        ]
    )
    var_delta = float(gradient @ utility_cov @ gradient)
    sd_post = float(np.sqrt(max(var_delta, 0.0)))
    mean = ce_a - ce_b
    return _normal_posterior_summary(mean=mean, sd_post=sd_post, alpha=alpha)


def paired_bayesian_bootstrap_posterior(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    functional: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_draws: int = 5_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> PosteriorSummary:
    """Bayesian bootstrap for arbitrary paired weighted functionals."""
    a, b = _validate_paired_arrays(samples_a, samples_b, "samples_a", "samples_b")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    generator = rng or np.random.default_rng()
    weights = generator.dirichlet(np.ones(a.shape[0]), size=n_draws)
    delta = np.asarray(
        [functional(a, b, weights[draw_idx]) for draw_idx in range(n_draws)],
        dtype=float,
    )
    return posterior_summary(delta, alpha=alpha)


def paired_mean_difference_posterior(
    diffs: Sequence[float],
    alpha: float = 0.05,
) -> PosteriorSummary:
    """Student-t posterior for the mean of paired scalar differences."""
    values = np.asarray(diffs, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("diffs must be a nonempty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("diffs must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    n = values.size
    mean = float(np.mean(values))
    if n < 2:
        return _point_mass_posterior_summary(mean, alpha=alpha)

    sample_sd = _sample_std(values)
    if sample_sd == 0.0:
        return _point_mass_posterior_summary(mean, alpha=alpha)

    df = n - 1
    scale = sample_sd / np.sqrt(n)
    ci_low, ci_high = t.ppf([alpha / 2.0, 1.0 - alpha / 2.0], df=df, loc=mean, scale=scale)
    if df > 2:
        sd_post = float(scale * np.sqrt(df / (df - 2.0)))
    else:
        sd_post = float("inf")
    return PosteriorSummary(
        mean=mean,
        sd_post=sd_post,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_positive=float(t.cdf(mean / scale, df=df)),
    )


def posterior_summary(draws: Sequence[float], alpha: float = 0.05) -> PosteriorSummary:
    """Summarize posterior draws with a central credible interval."""
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("draws must be a nonempty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("draws must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    zero_ties = np.isclose(values, 0.0, rtol=0.0, atol=1e-12)
    p_positive = np.mean(values > 0.0) + 0.5 * np.mean(zero_ties)
    return PosteriorSummary(
        mean=float(np.mean(values)),
        sd_post=_sample_std(values),
        ci_low=float(np.quantile(values, alpha / 2.0)),
        ci_high=float(np.quantile(values, 1.0 - alpha / 2.0)),
        p_positive=float(min(p_positive, 1.0)),
    )


def _paired_ce_bayesian_bootstrap(
    wealth_a: np.ndarray,
    wealth_b: np.ndarray,
    utility: UtilitySpec,
    n_draws: int,
    alpha: float,
    rng: np.random.Generator | None,
) -> PosteriorSummary:
    generator = rng or np.random.default_rng()
    weights = generator.dirichlet(np.ones(wealth_a.shape[0]), size=n_draws)
    delta = np.empty(n_draws)
    for draw_idx in range(n_draws):
        ce_a = utility.ce(float(np.sum(weights[draw_idx] * utility.u(wealth_a))))
        ce_b = utility.ce(float(np.sum(weights[draw_idx] * utility.u(wealth_b))))
        delta[draw_idx] = ce_a - ce_b
    return posterior_summary(delta, alpha=alpha)


def _validate_paired_arrays(
    values_a: Sequence[float],
    values_b: Sequence[float],
    name_a: str,
    name_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must be paired with the same shape")
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name_a} and {name_b} must be nonempty 1D arrays")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{name_a} and {name_b} must be finite")
    return a, b


def _crra_power_ce(mean_u: float, gamma: float) -> float:
    exponent = 1.0 - gamma
    scaled = exponent * mean_u
    if scaled <= 0.0:
        raise ValueError("expected utility is outside the CRRA CE domain")
    return float(scaled ** (1.0 / exponent))


def _paired_mean_covariance(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    if values_a.size < 2:
        return np.zeros((2, 2), dtype=float)
    paired_values = np.column_stack([values_a, values_b])
    return np.cov(paired_values, rowvar=False, ddof=1) / values_a.size


def _normal_posterior_summary(
    mean: float,
    sd_post: float,
    alpha: float,
) -> PosteriorSummary:
    if sd_post == 0.0:
        return _point_mass_posterior_summary(mean, alpha=alpha)
    z_value = norm.ppf(1.0 - alpha / 2.0)
    return PosteriorSummary(
        mean=float(mean),
        sd_post=float(sd_post),
        ci_low=float(mean - z_value * sd_post),
        ci_high=float(mean + z_value * sd_post),
        p_positive=float(norm.cdf(mean / sd_post)),
    )


def _point_mass_posterior_summary(mean: float, alpha: float) -> PosteriorSummary:
    if mean > 0.0:
        p_positive = 1.0
    elif mean < 0.0:
        p_positive = 0.0
    else:
        p_positive = 0.5
    return PosteriorSummary(
        mean=float(mean),
        sd_post=0.0,
        ci_low=float(mean),
        ci_high=float(mean),
        p_positive=p_positive,
    )


def summarize_episode(
    states: Sequence[OptionMMState],
    infos: Sequence[OptionMMStepInfo],
    inventory_limit: int | None = None,
) -> EpisodeSummary:
    """Summarize one episode from the reset state plus step diagnostics."""
    if len(states) != len(infos) + 1:
        raise ValueError("states must contain reset state plus one state per step")
    if not infos:
        raise ValueError("infos must contain at least one step")

    wealth = np.asarray([state.wealth for state in states], dtype=float)
    pnl = np.asarray([info.pnl for info in infos], dtype=float)
    inventory = np.asarray([state.option_inventory for state in states[1:]], dtype=float)
    abs_inventory = np.abs(inventory)
    net_delta = np.asarray([state.net_delta for state in states[1:]], dtype=float)

    gross_spread = float(sum(info.spread_capture for info in infos))
    adverse_selection = float(sum(info.adverse_selection_cost for info in infos))
    option_fees = float(sum(info.option_fees for info in infos))
    stock_costs = float(sum(info.stock_costs for info in infos))
    total_costs = option_fees + stock_costs

    bid_fills = np.asarray([info.bid_fills for info in infos], dtype=float)
    ask_fills = np.asarray([info.ask_fills for info in infos], dtype=float)
    hedge_trades = np.asarray([info.hedge_trade for info in infos], dtype=float)
    censored_sides = sum(
        int(info.bid_fills_censored) + int(info.ask_fills_censored) for info in infos
    )
    same_step_both_fills = sum(
        int(info.bid_fills > 0 and info.ask_fills > 0) for info in infos
    )
    floor_binds = sum(int(info.variance_floor_bound) for info in infos)

    if inventory_limit is None:
        time_at_limit = float("nan")
    else:
        if inventory_limit <= 0:
            raise ValueError("inventory_limit must be positive when provided")
        time_at_limit = float(np.mean(abs_inventory >= inventory_limit))

    running_peak = np.maximum.accumulate(wealth)
    drawdowns = running_peak - wealth

    return EpisodeSummary(
        terminal_wealth=float(wealth[-1]),
        total_pnl=float(wealth[-1] - wealth[0]),
        mean_pnl=float(np.mean(pnl)),
        pnl_vol=_sample_std(pnl),
        max_drawdown=float(np.max(drawdowns)),
        final_inventory=int(states[-1].option_inventory),
        inventory_mean=float(np.mean(inventory)),
        inventory_abs_mean=float(np.mean(abs_inventory)),
        inventory_abs_p95=float(np.quantile(abs_inventory, 0.95)),
        inventory_abs_max=float(np.max(abs_inventory)),
        time_at_inventory_limit=time_at_limit,
        gross_spread_capture=gross_spread,
        net_spread_capture=gross_spread - adverse_selection - option_fees,
        adverse_selection_cost=adverse_selection,
        option_turnover=float(np.sum(bid_fills + ask_fills)),
        stock_turnover=float(np.sum(np.abs(hedge_trades))),
        total_fees_and_costs=total_costs,
        net_delta_rms=float(np.sqrt(np.mean(net_delta * net_delta))),
        censoring_rate=float(censored_sides / (2 * len(infos))),
        variance_floor_binding_rate=float(floor_binds / len(infos)),
        same_step_both_fill_rate=float(same_step_both_fills / len(infos)),
        n_steps=len(infos),
    )


def aggregate_episode_summaries(
    summaries: Sequence[EpisodeSummary],
) -> AggregateSummary:
    """Aggregate per-episode summaries by simple cross-seed mean/std."""
    if not summaries:
        raise ValueError("summaries must be nonempty")

    return AggregateSummary(
        n_episodes=len(summaries),
        terminal_wealth_mean=_mean(summaries, "terminal_wealth"),
        terminal_wealth_std=_std(summaries, "terminal_wealth"),
        total_pnl_mean=_mean(summaries, "total_pnl"),
        total_pnl_std=_std(summaries, "total_pnl"),
        mean_pnl_mean=_mean(summaries, "mean_pnl"),
        pnl_vol_mean=_mean(summaries, "pnl_vol"),
        max_drawdown_mean=_mean(summaries, "max_drawdown"),
        max_drawdown_max=_max(summaries, "max_drawdown"),
        final_inventory_mean=_mean(summaries, "final_inventory"),
        final_inventory_std=_std(summaries, "final_inventory"),
        inventory_abs_mean=_mean(summaries, "inventory_abs_mean"),
        inventory_abs_p95_mean=_mean(summaries, "inventory_abs_p95"),
        inventory_abs_max_mean=_mean(summaries, "inventory_abs_max"),
        inventory_abs_max_max=_max(summaries, "inventory_abs_max"),
        time_at_inventory_limit_mean=_nanmean(summaries, "time_at_inventory_limit"),
        gross_spread_capture_mean=_mean(summaries, "gross_spread_capture"),
        net_spread_capture_mean=_mean(summaries, "net_spread_capture"),
        adverse_selection_cost_mean=_mean(summaries, "adverse_selection_cost"),
        option_turnover_mean=_mean(summaries, "option_turnover"),
        stock_turnover_mean=_mean(summaries, "stock_turnover"),
        total_fees_and_costs_mean=_mean(summaries, "total_fees_and_costs"),
        net_delta_rms_mean=_mean(summaries, "net_delta_rms"),
        censoring_rate_mean=_mean(summaries, "censoring_rate"),
        variance_floor_binding_rate_mean=_mean(
            summaries, "variance_floor_binding_rate"
        ),
        same_step_both_fill_rate_mean=_mean(summaries, "same_step_both_fill_rate"),
    )


def _values(summaries: Sequence[EpisodeSummary], field_name: str) -> np.ndarray:
    return np.asarray([getattr(summary, field_name) for summary in summaries], dtype=float)


def _mean(summaries: Sequence[EpisodeSummary], field_name: str) -> float:
    return float(np.mean(_values(summaries, field_name)))


def _nanmean(summaries: Sequence[EpisodeSummary], field_name: str) -> float:
    values = _values(summaries, field_name)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _std(summaries: Sequence[EpisodeSummary], field_name: str) -> float:
    return _sample_std(_values(summaries, field_name))


def _max(summaries: Sequence[EpisodeSummary], field_name: str) -> float:
    return float(np.max(_values(summaries, field_name)))


def _sample_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))
