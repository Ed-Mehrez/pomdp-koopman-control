r"""
Local-around-reference control: generic tooling.

Core abstraction
----------------
Given a nominal controller `reference_action(state)` and a local coordinate
`u` in a trust region around zero, evaluate / detect / fit the best local
correction under paired-noise variance reduction.

Generic terminology (domain-agnostic)
-------------------------------------
- reference_action:    the nominal action at a state.
- action_delta:        action - reference_action.
- control_coordinate:  local coordinate u (possibly a normalized
                       re-parameterization of action_delta).
- target_overlay:      the analytical action_delta we are trying to detect,
                       if any.  None => null gate (directionless).
- effect_samples:      the domain-defined scalar that the quadratic fit
                       is regressed on (e.g., terminal utility, rollout
                       return, Lyapunov decrement).

Finance-specific vocabulary (pi, V, rho, gamma, hedging demand, CRRA)
belongs in a finance adapter that maps onto these names, not here.

Currently used by:
    - finance/experiments/merton_local_signal_gate.py   (Heston benchmark)
    - tests/test_local_around_reference_lqr.py          (non-finance sanity)

Expect refactor when a third non-adapter caller appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np


# ===========================================================================
# Types
# ===========================================================================


Action = float
State = Any


# ===========================================================================
# Control normalizer (symmetric: to_action_delta and to_local are inverses)
# ===========================================================================


class ControlNormalizer(Protocol):
    r"""Map between a local coordinate `u` and an action_delta.

    The two methods MUST be mutual inverses for every (state, ref_action):
        to_local(to_action_delta(u, s, r), s, r) == u   (up to numerical error).

    Attributes
    ----------
    name : short identifier for reporting.
    """

    name: str

    def to_action_delta(self, u: float, state: State, ref_action: Action) -> float: ...
    def to_local(self, action_delta: float, state: State, ref_action: Action) -> float: ...


class IdentityNormalizer:
    name = "identity"

    def to_action_delta(self, u, state, ref_action):
        return float(u)

    def to_local(self, action_delta, state, ref_action):
        return float(action_delta)


class StateScaledNormalizer:
    r"""Normalizer with a state-dependent scalar:

        action_delta = u / scale_fn(state)
        u            = action_delta * scale_fn(state)

    E.g. `scale_fn = lambda s: sqrt(s.V)` gives a sqrt-variance normalization.
    The concept is domain-agnostic: any state-dependent amplitude scale works.
    """

    def __init__(self, scale_fn: Callable[[State], float], name: str = "state_scaled"):
        self.scale_fn = scale_fn
        self.name = name

    def to_action_delta(self, u, state, ref_action):
        s = max(float(self.scale_fn(state)), 1e-12)
        return float(u) / s

    def to_local(self, action_delta, state, ref_action):
        s = max(float(self.scale_fn(state)), 1e-12)
        return float(action_delta) * s


class ReferenceProportionalNormalizer:
    r"""Normalizer proportional to the reference action:

        action_delta = u * ref_action
        u            = action_delta / ref_action
    """

    name = "ref_proportional"

    def to_action_delta(self, u, state, ref_action):
        return float(u) * float(ref_action)

    def to_local(self, action_delta, state, ref_action):
        r = float(ref_action) if abs(ref_action) > 1e-12 else 1e-12
        return float(action_delta) / r


# ===========================================================================
# Trust region + local quadratic fit
# ===========================================================================


def trust_region_grid(u_max: float, n: int = 41) -> np.ndarray:
    return np.linspace(-u_max, u_max, n)


@dataclass(frozen=True)
class LocalQuadraticFit:
    c0: float
    c1: float
    c2: float
    concave: bool
    u_star: float
    clipped: bool

    @classmethod
    def from_samples(
        cls,
        u_grid: np.ndarray,
        effect_samples: np.ndarray,
        u_max: float,
    ) -> "LocalQuadraticFit":
        X = np.column_stack([np.ones_like(u_grid), u_grid, u_grid ** 2])
        coef, *_ = np.linalg.lstsq(X, effect_samples, rcond=None)
        c0, c1, c2 = (float(v) for v in coef)
        concave = c2 < -1e-18
        clipped = False
        if concave:
            u_star = -c1 / (2 * c2)
            if abs(u_star) > u_max:
                u_star = float(np.clip(u_star, -u_max, u_max))
                clipped = True
        else:
            u_star = float("nan")
        return cls(c0, c1, c2, concave, u_star, clipped)


# ===========================================================================
# Paired estimator protocol (the single domain hook)
# ===========================================================================


class PairedEffectEstimator(Protocol):
    r"""Return a `len(u_grid)`-array of effect samples under shared paired noise.

    The domain adapter decides internally:
      - what an "effect" is (terminal utility, rollout return, Lyapunov
        decrement, ...);
      - how the seed constructs paired noise;
      - how a local coordinate `u` maps to the action sequence via
        (normalizer, state, ref_action).

    Contract
    --------
    Every element of the returned array corresponds to u_grid[k], evaluated
    under the SAME paired-noise draw identified by `seed`.
    """

    def __call__(
        self,
        u_grid: np.ndarray,
        state: State,
        ref_action: Action,
        normalizer: ControlNormalizer,
        seed: int,
    ) -> np.ndarray: ...


# ===========================================================================
# Per-seed action_delta estimate
# ===========================================================================


def action_delta_for_seed(
    u_grid: np.ndarray,
    u_max: float,
    effect_samples: np.ndarray,
    state: State,
    ref_action: Action,
    normalizer: ControlNormalizer,
) -> Dict[str, float]:
    fit = LocalQuadraticFit.from_samples(u_grid, effect_samples, u_max)
    if fit.concave:
        action_delta = normalizer.to_action_delta(fit.u_star, state, ref_action)
    else:
        action_delta = float("nan")
    return {
        "action_delta": float(action_delta),
        "u_star": fit.u_star,
        "concave": fit.concave,
        "clipped": fit.clipped,
        "c1": fit.c1,
        "c2": fit.c2,
    }


# ===========================================================================
# Specs, objective, results (raw statistics only)
# ===========================================================================


@dataclass(frozen=True)
class GateObjective:
    r"""Domain-agnostic description of what the gate is probing.

    Stamped onto every `GateResult` so horizon / dt / semantics travel with
    the output.  Keeps the machinery from being implicitly "daily one-step"
    or "terminal log-wealth" specific.
    """
    name: str
    horizon: int
    dt: float
    notes: str = ""


@dataclass(frozen=True)
class GateSpec:
    r"""One (state, regime) cell of the sweep.

    Parameters
    ----------
    label            : short identifier (e.g., "rho=-0.7" or "x0=1.0").
    state            : domain-defined state object.
    reference_action : nominal action at `state`.
    target_overlay   : expected action_delta if known (e.g., analytical
                       theory); None for a directionless null gate.
    normalizers      : list of ControlNormalizer instances to sweep.
    u_max_list       : trust-region sizes to sweep.
    meta             : free-form adapter labels for reporting.
    """
    label: str
    state: State
    reference_action: Action
    target_overlay: Optional[float]
    normalizers: Sequence[ControlNormalizer]
    u_max_list: Sequence[float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    r"""Raw statistics for one (spec, normalizer, u_max) cell.

    Two estimators are surfaced:

    * Per-seed estimator
        Each seed independently produces `action_delta = -c1/(2*c2)` from its
        own local quadratic fit, then aggregates means/SEs across seeds.
        This honestly exposes seed-level variability of the nonlinear
        collapse, but when `c1` has high paired-noise variance and `c2` is
        small, the per-seed estimator is inefficient.

    * Pooled-coefficient estimator
        Pool the linear-fit coefficients across seeds first, then collapse:
            c1_pool  = mean_seeds(c1_seed),     c2_pool = mean_seeds(c2_seed)
            action_delta_pooled = -c1_pool / (2 * c2_pool).
        Uncertainty from a delta-method on (c1_pool, c2_pool).  This is
        much more efficient under shared-signal / paired-noise regimes
        because linear aggregation precedes the nonlinear ratio.

    The adapter / config layer is responsible for turning these raw
    statistics into pass/fail decisions.
    """
    spec_label: str
    normalizer_name: str
    u_max: float
    objective: GateObjective
    target_overlay: Optional[float]
    n_seeds: int
    n_valid: int

    # ---- per-seed estimator stats ----
    action_delta_mean: float
    action_delta_std: float
    action_delta_se: float
    snr: float                    # |mean| / std
    t_stat: float                 # mean / se              (H0: mean = 0)
    t_stat_vs_theory: float       # (mean - target) / se   (H0: mean = target)
    sign_match: float             # frac(sign(Δ̂) == sign(target))
    effect_ratio: float           # mean / target          (1.0 = perfect recovery)
    concave_frac: float
    clip_frac: float
    samples: np.ndarray           # raw per-seed action_delta samples

    # ---- pooled-coefficient estimator stats ----
    action_delta_pooled: float    # -mean(c1)/(2*mean(c2))   if mean(c2) < 0
    action_delta_pooled_se: float # delta-method SE from (c1_seeds, c2_seeds)
    t_pooled: float               # pooled / SE         (H0: pooled = 0)
    t_pooled_vs_theory: float     # (pooled - target) / SE  (H0: pooled = target)
    effect_ratio_pooled: float    # pooled / target
    c1_seeds: np.ndarray          # per-seed linear coefficients
    c2_seeds: np.ndarray          # per-seed quadratic coefficients

    meta: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Aggregation
# ===========================================================================


def _pooled_delta_stats(
    c1_seeds: np.ndarray,
    c2_seeds: np.ndarray,
    state: State,
    ref_action: Action,
    normalizer: ControlNormalizer,
    target_overlay: Optional[float],
) -> Dict[str, float]:
    r"""Pooled-coefficient delta* estimate with delta-method SE.

    Let c1, c2 be the per-seed linear/quadratic regression coefficients in
    the LOCAL coordinate u.  We form
        c1_pool = mean(c1),   c2_pool = mean(c2),
        u_pool  = -c1_pool / (2 * c2_pool)         (if c2_pool < 0)
    and then map u_pool back through the normalizer to action space:
        action_delta_pooled = normalizer.to_action_delta(u_pool, state, ref_action).

    Uncertainty on u_pool via first-order delta method:
        Var(u_pool) ≈ (d u / d c1)^2 * Var(c1_pool)
                    + (d u / d c2)^2 * Var(c2_pool)
                    + 2 * (d u / d c1)(d u / d c2) * Cov(c1_pool, c2_pool)
    with d u / d c1 = -1/(2 c2_pool), d u / d c2 = c1_pool / (2 c2_pool^2).

    SE on action_delta_pooled obtained by scaling u_pool SE through the
    normalizer's local slope d(action_delta)/d u (finite-difference).
    """
    nan = float("nan")
    out = {
        "pooled": nan, "pooled_se": nan,
        "t_pooled": nan, "t_pooled_vs_theory": nan, "effect_ratio_pooled": nan,
    }
    c1_valid = c1_seeds[np.isfinite(c1_seeds)]
    c2_valid = c2_seeds[np.isfinite(c2_seeds)]
    n = min(c1_valid.size, c2_valid.size)
    if n < 3:
        return out
    # Pair them back up (they should be same length by construction).
    pair_mask = np.isfinite(c1_seeds) & np.isfinite(c2_seeds)
    c1p = c1_seeds[pair_mask]
    c2p = c2_seeds[pair_mask]
    n_pair = c1p.size
    if n_pair < 3:
        return out
    c1_mean = float(np.mean(c1p))
    c2_mean = float(np.mean(c2p))
    if c2_mean >= -1e-18:
        # Non-concave pooled quadratic: pooled estimator undefined.
        return out
    u_pool = -c1_mean / (2 * c2_mean)
    # Delta-method SE on u_pool.
    var_c1 = float(np.var(c1p, ddof=1)) / n_pair
    var_c2 = float(np.var(c2p, ddof=1)) / n_pair
    cov_12 = float(np.cov(c1p, c2p, ddof=1)[0, 1]) / n_pair
    du_dc1 = -1.0 / (2 * c2_mean)
    du_dc2 = c1_mean / (2 * c2_mean ** 2)
    var_u = (du_dc1 ** 2) * var_c1 + (du_dc2 ** 2) * var_c2 + 2 * du_dc1 * du_dc2 * cov_12
    se_u = float(np.sqrt(max(var_u, 0.0)))
    # Map u_pool and its SE through the normalizer to action space.
    action_pool = float(normalizer.to_action_delta(u_pool, state, ref_action))
    # Normalizer is locally approximately linear around u_pool; use a small
    # FD step to get its slope in action-coordinate.
    eps = max(1e-6, 0.001 * (abs(u_pool) + 1.0))
    a_plus = float(normalizer.to_action_delta(u_pool + eps, state, ref_action))
    a_minus = float(normalizer.to_action_delta(u_pool - eps, state, ref_action))
    d_action_du = (a_plus - a_minus) / (2 * eps)
    se_action = abs(d_action_du) * se_u
    t_pooled = action_pool / se_action if se_action > 0 else float("inf")
    if target_overlay is not None and abs(target_overlay) > 1e-12:
        t_pooled_vs_theory = (action_pool - target_overlay) / se_action if se_action > 0 else float("inf")
        effect_ratio_pooled = action_pool / target_overlay
    else:
        t_pooled_vs_theory = nan
        effect_ratio_pooled = nan
    return {
        "pooled": action_pool,
        "pooled_se": se_action,
        "t_pooled": t_pooled,
        "t_pooled_vs_theory": t_pooled_vs_theory,
        "effect_ratio_pooled": effect_ratio_pooled,
    }


def _aggregate(
    samples: np.ndarray,
    target_overlay: Optional[float],
    concave_flags: np.ndarray,
    clipped_flags: np.ndarray,
) -> Dict[str, float]:
    valid = samples[np.isfinite(samples)]
    n_valid = int(valid.size)
    if n_valid < 3:
        nan = float("nan")
        return {
            "n_valid": n_valid,
            "mean": nan, "std": nan, "se": nan,
            "snr": nan, "t_stat": nan, "t_stat_vs_theory": nan,
            "sign_match": nan, "effect_ratio": nan,
            "concave_frac": float(np.mean(concave_flags)) if concave_flags.size else nan,
            "clip_frac": float(np.mean(clipped_flags)) if clipped_flags.size else nan,
        }
    mean_d = float(np.mean(valid))
    std_d = float(np.std(valid, ddof=1))
    se_d = std_d / np.sqrt(n_valid) if std_d > 0 else 0.0
    snr = abs(mean_d) / std_d if std_d > 0 else float("inf")
    t_stat = mean_d / se_d if se_d > 0 else float("inf")
    if target_overlay is not None and abs(target_overlay) > 1e-12:
        sign_match = float(np.mean(np.sign(valid) == np.sign(target_overlay)))
        effect_ratio = mean_d / target_overlay
        t_stat_vs_theory = (mean_d - target_overlay) / se_d if se_d > 0 else float("inf")
    else:
        sign_match = float("nan")
        effect_ratio = float("nan")
        t_stat_vs_theory = float("nan")
    return {
        "n_valid": n_valid,
        "mean": mean_d, "std": std_d, "se": se_d,
        "snr": snr, "t_stat": t_stat, "t_stat_vs_theory": t_stat_vs_theory,
        "sign_match": sign_match, "effect_ratio": effect_ratio,
        "concave_frac": float(np.mean(concave_flags)),
        "clip_frac": float(np.mean(clipped_flags)),
    }


# ===========================================================================
# Main gate loop
# ===========================================================================


def run_signal_gate(
    specs: Sequence[GateSpec],
    objective: GateObjective,
    estimator: PairedEffectEstimator,
    n_seeds: int = 20,
    n_grid: int = 41,
    base_seed: int = 1000,
) -> List[GateResult]:
    r"""Run the signal gate over (spec × normalizer × u_max).

    For each cell:
      - build u_grid = linspace(-u_max, +u_max, n_grid);
      - for seed in [base_seed, base_seed + n_seeds):
          effects = estimator(u_grid, state, ref_action, normalizer, seed)
          action_delta_sample = argmax of local quadratic fit (if concave);
      - aggregate raw statistics across seeds.

    Returns raw `GateResult` rows.  No pass/fail thresholds are applied
    here; that belongs in the adapter.
    """
    out: List[GateResult] = []
    for spec in specs:
        for normalizer in spec.normalizers:
            for u_max in spec.u_max_list:
                u_grid = trust_region_grid(u_max, n_grid)
                samples = np.full(n_seeds, np.nan, dtype=float)
                c1_seeds = np.full(n_seeds, np.nan, dtype=float)
                c2_seeds = np.full(n_seeds, np.nan, dtype=float)
                concave_flags = np.zeros(n_seeds, dtype=bool)
                clipped_flags = np.zeros(n_seeds, dtype=bool)
                for k in range(n_seeds):
                    effects = estimator(
                        u_grid, spec.state, spec.reference_action,
                        normalizer, base_seed + k,
                    )
                    per_seed = action_delta_for_seed(
                        u_grid, float(u_max), effects,
                        spec.state, spec.reference_action, normalizer,
                    )
                    samples[k] = per_seed["action_delta"]
                    c1_seeds[k] = per_seed["c1"]
                    c2_seeds[k] = per_seed["c2"]
                    concave_flags[k] = per_seed["concave"]
                    clipped_flags[k] = per_seed["clipped"]
                agg = _aggregate(
                    samples, spec.target_overlay, concave_flags, clipped_flags,
                )
                pooled = _pooled_delta_stats(
                    c1_seeds, c2_seeds,
                    spec.state, spec.reference_action, normalizer,
                    spec.target_overlay,
                )
                out.append(GateResult(
                    spec_label=spec.label,
                    normalizer_name=normalizer.name,
                    u_max=float(u_max),
                    objective=objective,
                    target_overlay=spec.target_overlay,
                    n_seeds=int(n_seeds),
                    n_valid=int(agg["n_valid"]),
                    action_delta_mean=float(agg["mean"]),
                    action_delta_std=float(agg["std"]),
                    action_delta_se=float(agg["se"]),
                    snr=float(agg["snr"]),
                    t_stat=float(agg["t_stat"]),
                    t_stat_vs_theory=float(agg["t_stat_vs_theory"]),
                    sign_match=float(agg["sign_match"]),
                    effect_ratio=float(agg["effect_ratio"]),
                    concave_frac=float(agg["concave_frac"]),
                    clip_frac=float(agg["clip_frac"]),
                    samples=samples.copy(),
                    action_delta_pooled=float(pooled["pooled"]),
                    action_delta_pooled_se=float(pooled["pooled_se"]),
                    t_pooled=float(pooled["t_pooled"]),
                    t_pooled_vs_theory=float(pooled["t_pooled_vs_theory"]),
                    effect_ratio_pooled=float(pooled["effect_ratio_pooled"]),
                    c1_seeds=c1_seeds.copy(),
                    c2_seeds=c2_seeds.copy(),
                    meta=dict(spec.meta),
                ))
    return out
