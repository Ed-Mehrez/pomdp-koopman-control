r"""
BENCHMARK STUDY: Marcus-inspired follow-up on the Bates lambda sweep (Part B).

Context
-------
Part A (`study_bates_lambda_sweep.py`) showed that as `lambda_j` rises
from 10 -> 30 -> 60 /yr, the residual gap
    winsor_ewma - ms_cum_stride_bv_target
GROWS on both the all-samples and jump-adjacent subsets.  Target
recovery (bv_target - raw) on jump-adjacent windows is large and flat;
target fix alone does not close the widening residual gap.  This is
suggestive of a real jump-aware REPRESENTATION bottleneck.

Terminology (important)
-----------------------
We do NOT claim this is a Marcus-signature implementation.  Marcus is
the correct jump-aware rough-path theory; a full implementation is
deliberately out of scope.  What we implement here is a **Marcus-
inspired jump-aware input/target correction**:

  - oracle: corrects the signature lane's continuous target using the
    TRUE simulated jump atom J_t.  Target = max(r_t^2/dt - J_t^2/dt, 0).
    This is the representation ceiling assuming perfect jump
    identification.  NOT deployable.
  - proxy: replaces the oracle with a running jump-energy proxy computed
    from a smoothed bipower variation.  Target =
        max(r_t^2/dt - bv_smooth_t, 0)
    which is positive only when the instantaneous r^2/dt clearly exceeds
    the smooth continuous variance estimate.  This is a practical
    jump-aware correction with NO oracle information.

Both lanes preserve the EXISTING cumulative-stride signature state and
feature map.  Only the supervised target is jump-aware.  This is a
TARGET-level correction, not a new rough-path construction.

Design principle (honest framing):
  - Part A isolated the effect of raw-vs-bv_target swap.
  - Part B isolates the further effect of jump-aware target correction
    (either with or without oracle information).
  - The fraction of the oracle gain captured by the proxy tells us how
    practically accessible the representation ceiling is.

Pre-registered bars
-------------------
  1. Representation bottleneck confirmed:
       marcus_oracle - bv_target > +0.05 corr on jump-adjacent at lambda=30
       AND non-decreasing from 10 -> 30 -> 60.
  2. Practical proxy success:
       marcus_proxy captures at least half of (oracle - bv_target) on
       jump-adjacent corr at lambda=30.
  3. Negative Marcus result:
       marcus_oracle adds < +0.02 corr.  Target fix already captured it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from study_bates_signature_filters import (
    BatesConfig,
    EWMABipower,
    EWMAWinsorized,
    _jump_adjacent_mask,
    simulate_bates,
)
from study_bates_signature_proxy_channels import (
    CumStrideSigConfig,
    CumStrideSigFilter,
    target_bv_per_step,
)


# ==========================================================================
# Jump-aware signature lanes (Marcus-inspired input/target correction)
# ==========================================================================


def _sig_cfg(bates: BatesConfig) -> CumStrideSigConfig:
    return CumStrideSigConfig(
        strides=(1, 5, 20),
        kappa=bates.kappa, theta=bates.theta, xi=bates.xi, dt=bates.dt,
    )


class MarcusOracleLane:
    r"""Signature lane with oracle jump-aware target.

    Target per step: max(r_t^2 / dt - J_t^2 / dt, 0)
    where J_t is the TRUE simulated jump atom.  The filter's signature
    state and feature map are unchanged; only the supervised target is
    corrected using oracle jump information.

    Not deployable; purely an ablation to measure the representation
    ceiling under perfect jump identification.
    """
    name = "ms_cum_stride_marcus_oracle"

    def __init__(self, bates: BatesConfig):
        self._J_sq_over_dt: float = 0.0
        def _target(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
            return max((r_t * r_t) / dt - self._J_sq_over_dt, 0.0)
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=_target, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)
        self._J_sq_over_dt = 0.0

    def observe(self, r_t: float, dt: float, J_t: float = 0.0):
        self._J_sq_over_dt = (float(J_t) ** 2) / float(dt)
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


class MarcusProxyLane:
    r"""Signature lane with proxy (no-oracle) jump-aware target.

    Target per step: max(r_t^2 / dt - bv_smooth_t, 0)
    where bv_smooth_t is a running EWMA of per-step bipower variation:
        bv_t = (pi/2) * |r_{t-1}| * |r_t| / dt,
    smoothed with halflife 21 (steps).  The target is strictly
    non-negative; it measures positive excess of r_t^2/dt over a
    smooth continuous-variance estimate (a pragmatic jump-energy
    indicator).  No oracle information.

    One threshold / hyperparameter (halflife = 21 steps).  Kept
    FIXED across all lambdas; no tuning sweep.
    """
    name = "ms_cum_stride_marcus_proxy"
    HALFLIFE_STEPS = 21.0

    def __init__(self, bates: BatesConfig):
        self._alpha = 1.0 - float(np.exp(-np.log(2.0) / self.HALFLIFE_STEPS))
        self._bv_smooth: float = 0.04
        self._prev_abs_r: Optional[float] = None
        self._current_jump_proxy: float = 0.0
        self._dt = bates.dt
        def _target(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
            rv = (r_t * r_t) / dt
            return max(rv - self._current_jump_proxy, 0.0)
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=_target, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)
        self._bv_smooth = float(V0)
        self._prev_abs_r = None
        self._current_jump_proxy = 0.0

    def observe(self, r_t: float, dt: float, J_t: float = 0.0):
        # 1) compute per-step BV
        abs_r = float(np.abs(r_t))
        if self._prev_abs_r is None:
            bv_t = (r_t * r_t) / dt                   # warm-up: fall back to r^2/dt
        else:
            bv_t = (np.pi / 2.0) * float(self._prev_abs_r) * abs_r / dt
        # 2) update smoothed BV estimator
        self._bv_smooth = (1.0 - self._alpha) * self._bv_smooth + self._alpha * bv_t
        # 3) compute jump proxy as positive excess of r^2/dt over bv_smooth
        rv = (r_t * r_t) / dt
        self._current_jump_proxy = max(rv - self._bv_smooth, 0.0)
        # 4) step the underlying sig filter (target uses _current_jump_proxy)
        self.filter.observe(r_t, dt)
        self._prev_abs_r = abs_r

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


class BVTargetLane:
    r"""Reference signature lane: BV-per-step target (no jump-aware correction).

    Wrapper so its observe() signature matches the Marcus lanes (which
    take a J_t argument).  Target = (pi/2)|r_{t-1}||r_t|/dt, as in the
    Part A and proxy-channel studies.
    """
    name = "ms_cum_stride_bv_target"

    def __init__(self, bates: BatesConfig):
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=target_bv_per_step, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)

    def observe(self, r_t: float, dt: float, J_t: float = 0.0):
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()


# ==========================================================================
# Scalar baseline wrappers (uniform observe(r_t, dt, J_t) signature)
# ==========================================================================


class _ScalarWrap:
    def __init__(self, core, name: str):
        self.core = core
        self.name = name

    def reset(self, V0: float):
        self.core.reset(V0)

    def observe(self, r_t: float, dt: float, J_t: float = 0.0):
        self.core.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.core.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.core.V_interval()


# ==========================================================================
# Rollout / pooled stats  (lanes observed with (r_t, dt, J_t))
# ==========================================================================


def make_factories(bates: BatesConfig) -> Dict[str, Callable[[], object]]:
    halflife = 21.0
    return {
        "bv_ewma":                       lambda: _ScalarWrap(
            EWMABipower(halflife_steps=halflife, dt=bates.dt), "bv_ewma",
        ),
        "winsor_ewma":                   lambda: _ScalarWrap(
            EWMAWinsorized(halflife_steps=halflife, dt=bates.dt, k=4.0), "winsor_ewma",
        ),
        "ms_cum_stride_bv_target":       lambda: BVTargetLane(bates),
        "ms_cum_stride_marcus_oracle":   lambda: MarcusOracleLane(bates),
        "ms_cum_stride_marcus_proxy":    lambda: MarcusProxyLane(bates),
    }


def _fresh_lanes(factories, V0: float) -> Dict[str, object]:
    lanes = {name: f() for name, f in factories.items()}
    for est in lanes.values():
        est.reset(V0)
    return lanes


def _rollout(
    bates: BatesConfig, factories: Dict[str, Callable[[], object]],
    T: int, V0: float, seed: int,
) -> Dict[str, np.ndarray]:
    sim = simulate_bates(bates, T, V0, seed)
    lanes = _fresh_lanes(factories, V0)
    V_hat = {name: np.zeros(T) for name in lanes}
    for t in range(T):
        r = float(sim["dr_S"][t])
        J = float(sim["J"][t])
        for name, est in lanes.items():
            est.observe(r, bates.dt, J)
            V_hat[name][t] = est.V_hat()
    return {
        "V_true_pre": sim["V"],
        "jump":       sim["jump"],
        "V_hat":      V_hat,
        "dr_S":       sim["dr_S"],
    }


def _pooled(pred_list, tgt_list, mask_list=None, warmup=30):
    p_all, t_all = [], []
    for i in range(len(pred_list)):
        p = pred_list[i][warmup:]
        t = tgt_list[i][warmup:]
        if mask_list is not None:
            m = mask_list[i][warmup:]
            p = p[m]; t = t[m]
        p_all.append(p); t_all.append(t)
    p = np.concatenate(p_all) if p_all else np.zeros(0)
    q = np.concatenate(t_all) if t_all else np.zeros(0)
    mask = np.isfinite(p) & np.isfinite(q)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    rmse = float(np.sqrt(np.mean((p[mask] - q[mask]) ** 2)))
    corr = float(np.corrcoef(p[mask], q[mask])[0, 1])
    return rmse, corr, int(mask.sum())


def run_sweep(
    bates_base: BatesConfig, lambdas: Tuple[float, ...],
    n_seeds: int, T: int, warmup: int, base_seed: int,
):
    out: Dict[float, Dict[str, Dict[str, Tuple[float, float, int]]]] = {}
    trace_by_lambda: Dict[float, Dict] = {}
    for lam in lambdas:
        bates = replace(bates_base, lambda_j=float(lam))
        factories = make_factories(bates)
        lane_names = list(factories.keys())
        hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
        V_list: List[np.ndarray] = []
        jump_list: List[np.ndarray] = []
        dr_list: List[np.ndarray] = []
        v0_rng = np.random.RandomState(base_seed + int(lam))
        for k in range(n_seeds):
            V0 = float(v0_rng.uniform(0.02, 0.08))
            r_out = _rollout(bates, factories, T, V0, base_seed + int(lam) * 1000 + k)
            for name in lane_names:
                hat[name].append(r_out["V_hat"][name])
            V_list.append(r_out["V_true_pre"])
            jump_list.append(r_out["jump"])
            dr_list.append(r_out["dr_S"])
        jump_masks = [_jump_adjacent_mask(j, radius=5) for j in jump_list]
        calm_masks = [~m for m in jump_masks]
        lane_res: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
        for name in lane_names:
            lane_res[name] = {
                "all":  _pooled(hat[name], V_list, warmup=warmup),
                "jump": _pooled(hat[name], V_list, mask_list=jump_masks, warmup=warmup),
                "calm": _pooled(hat[name], V_list, mask_list=calm_masks, warmup=warmup),
            }
        out[float(lam)] = lane_res
        # Store a representative trace (first seed) for possible inset plot
        trace_by_lambda[float(lam)] = {
            "V_true": V_list[0],
            "jump":   jump_list[0],
            "dr_S":   dr_list[0],
            "V_hat":  {name: hat[name][0] for name in lane_names},
        }
    return out, trace_by_lambda


# ==========================================================================
# Reporting
# ==========================================================================


def _print_per_lambda(results) -> None:
    lane_order = [
        "bv_ewma", "winsor_ewma",
        "ms_cum_stride_bv_target",
        "ms_cum_stride_marcus_oracle",
        "ms_cum_stride_marcus_proxy",
    ]
    for lam, lane_res in sorted(results.items()):
        print(f"lambda_j = {lam:.0f} / yr")
        print("-" * 108)
        print(f"  {'lane':32s} | {'corr(all)':>9s} | {'corr(jump)':>10s} | "
              f"{'corr(calm)':>10s} | {'RMSE(all)':>9s}")
        print("-" * 108)
        for name in lane_order:
            if name not in lane_res: continue
            r_all, c_all, _ = lane_res[name]["all"]
            r_j,   c_j,   _ = lane_res[name]["jump"]
            r_c,   c_c,   _ = lane_res[name]["calm"]
            print(f"  {name:32s} | {c_all:+.4f}   | {c_j:+.4f}    | "
                  f"{c_c:+.4f}    | {r_all:.4f}")
        print()


def _print_incremental_gains(results) -> None:
    print("INCREMENTAL GAINS per lambda")
    print("  oracle_gain = corr(marcus_oracle) - corr(bv_target)")
    print("  proxy_gain  = corr(marcus_proxy)  - corr(bv_target)")
    print("  proxy_fraction_of_oracle = proxy_gain / oracle_gain (when oracle_gain > 0)")
    print("-" * 108)
    print(f"  {'lambda':>7s} | {'subset':>6s} | {'oracle_gain':>12s} | {'proxy_gain':>11s} | {'proxy/oracle':>13s}")
    print("-" * 108)
    for lam, lane_res in sorted(results.items()):
        for subset in ("all", "jump"):
            c_bv  = lane_res["ms_cum_stride_bv_target"][subset][1]
            c_or  = lane_res["ms_cum_stride_marcus_oracle"][subset][1]
            c_px  = lane_res["ms_cum_stride_marcus_proxy"][subset][1]
            og = c_or - c_bv
            pg = c_px - c_bv
            if og > 1e-6:
                frac = pg / og
                frac_str = f"{frac:+.2f}"
            else:
                frac_str = "n/a"
            print(f"  {lam:7.0f} | {subset:>6s} | {og:+.4f}      | {pg:+.4f}     | {frac_str:>13s}")
    print()


def _plot_sweep(results, trace_by_lambda, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = sorted(results.keys())
    lane_order = [
        "bv_ewma", "winsor_ewma",
        "ms_cum_stride_bv_target",
        "ms_cum_stride_marcus_oracle",
        "ms_cum_stride_marcus_proxy",
    ]
    palette = {
        "bv_ewma":                     "tab:purple",
        "winsor_ewma":                 "tab:green",
        "ms_cum_stride_bv_target":     "tab:cyan",
        "ms_cum_stride_marcus_oracle": "tab:brown",
        "ms_cum_stride_marcus_proxy":  "tab:orange",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: corr(all)
    ax = axes[0, 0]
    for name in lane_order:
        y = [results[l][name]["all"][1] for l in lams]
        ax.plot(lams, y, "o-", color=palette.get(name), lw=1.5, label=name)
    ax.set_xlabel("lambda_j (jumps / year)")
    ax.set_ylabel("corr(V_hat, V_true)  (all samples)")
    ax.set_title("Corr, all samples")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")

    # Panel 2: corr(jump-adjacent)
    ax = axes[0, 1]
    for name in lane_order:
        y = [results[l][name]["jump"][1] for l in lams]
        ax.plot(lams, y, "o-", color=palette.get(name), lw=1.5, label=name)
    ax.set_xlabel("lambda_j")
    ax.set_ylabel("corr (jump-adjacent)")
    ax.set_title("Corr, jump-adjacent subset")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")

    # Panel 3: corr(calm)
    ax = axes[1, 0]
    for name in lane_order:
        y = [results[l][name]["calm"][1] for l in lams]
        ax.plot(lams, y, "o-", color=palette.get(name), lw=1.5, label=name)
    ax.set_xlabel("lambda_j")
    ax.set_ylabel("corr (calm subset)")
    ax.set_title("Corr, calm subset")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")

    # Panel 4: oracle_gain and proxy_gain over bv_target, jump-adjacent
    ax = axes[1, 1]
    y_oracle_jump = [
        results[l]["ms_cum_stride_marcus_oracle"]["jump"][1]
        - results[l]["ms_cum_stride_bv_target"]["jump"][1]
        for l in lams
    ]
    y_proxy_jump = [
        results[l]["ms_cum_stride_marcus_proxy"]["jump"][1]
        - results[l]["ms_cum_stride_bv_target"]["jump"][1]
        for l in lams
    ]
    y_oracle_all = [
        results[l]["ms_cum_stride_marcus_oracle"]["all"][1]
        - results[l]["ms_cum_stride_bv_target"]["all"][1]
        for l in lams
    ]
    y_proxy_all = [
        results[l]["ms_cum_stride_marcus_proxy"]["all"][1]
        - results[l]["ms_cum_stride_bv_target"]["all"][1]
        for l in lams
    ]
    ax.plot(lams, y_oracle_jump, "o-",  color=palette["ms_cum_stride_marcus_oracle"],
            lw=1.8, label="oracle - bv_target (jump)")
    ax.plot(lams, y_proxy_jump, "s--", color=palette["ms_cum_stride_marcus_proxy"],
            lw=1.8, label="proxy - bv_target (jump)")
    ax.plot(lams, y_oracle_all, "o-",  color=palette["ms_cum_stride_marcus_oracle"],
            lw=1.0, alpha=0.6, label="oracle - bv_target (all)")
    ax.plot(lams, y_proxy_all, "s--", color=palette["ms_cum_stride_marcus_proxy"],
            lw=1.0, alpha=0.6, label="proxy - bv_target (all)")
    ax.axhline(0.0, color="gray", lw=0.7)
    ax.axhline(0.05, color="gray", lw=0.5, linestyle=":")
    ax.set_xlabel("lambda_j")
    ax.set_ylabel("incremental corr over bv_target")
    ax.set_title("Incremental gains over bv_target (pre-reg bar: jump oracle > 0.05)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Bates lambda sweep: Part B (Marcus-inspired target corrections)", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    bates_base = BatesConfig()
    lambdas = (10.0, 30.0, 60.0)
    n_seeds = 40
    T = 400
    warmup = 30
    print("=" * 108)
    print("BENCHMARK STUDY  --  Bates Marcus-inspired sweep (Part B)")
    print(f"  lambda sweep: {lambdas}  /yr")
    print(f"  n_seeds={n_seeds}, T={T}, warm-up={warmup}, dt={bates_base.dt:.5f}")
    print(f"  oracle: target = max(r^2/dt - J_t^2/dt, 0)  (uses true J_t)")
    print(f"  proxy : target = max(r^2/dt - bv_smooth_t, 0)  (halflife=21 steps, fixed)")
    print("=" * 108)

    results, traces = run_sweep(
        bates_base=bates_base, lambdas=lambdas,
        n_seeds=n_seeds, T=T, warmup=warmup, base_seed=17_000,
    )
    _print_per_lambda(results)
    _print_incremental_gains(results)

    out_path = os.path.join(HERE, "study_bates_marcus_lambda_sweep.png")
    try:
        _plot_sweep(results, traces, out_path)
        print(f"Saved figure: {out_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")

    # Pre-registered bar checks
    lams = sorted(results.keys())
    og_jump = [
        results[l]["ms_cum_stride_marcus_oracle"]["jump"][1]
        - results[l]["ms_cum_stride_bv_target"]["jump"][1]
        for l in lams
    ]
    pg_jump = [
        results[l]["ms_cum_stride_marcus_proxy"]["jump"][1]
        - results[l]["ms_cum_stride_bv_target"]["jump"][1]
        for l in lams
    ]
    og30_jump = og_jump[lams.index(30.0)] if 30.0 in lams else float("nan")
    pg30_jump = pg_jump[lams.index(30.0)] if 30.0 in lams else float("nan")
    og_non_decreasing = all(og_jump[i] >= og_jump[i - 1] - 0.01 for i in range(1, len(og_jump)))
    proxy_fraction = (pg30_jump / og30_jump) if og30_jump > 1e-6 else float("nan")

    print()
    print("=" * 108)
    print("PART B SUMMARY  (pre-registered bar checks)")
    print("=" * 108)
    print(f"  oracle_gain (jump) per lambda   : {[f'{v:+.4f}' for v in og_jump]}")
    print(f"  proxy_gain  (jump) per lambda   : {[f'{v:+.4f}' for v in pg_jump]}")
    print(f"  oracle non-decreasing 10->60    : {og_non_decreasing}")
    print(f"  oracle gain at lambda=30 (jump) : {og30_jump:+.4f}   (bar: > +0.05)")
    print(f"  proxy  gain at lambda=30 (jump) : {pg30_jump:+.4f}")
    print(f"  proxy / oracle at lambda=30     : "
          f"{proxy_fraction:+.2f}" if np.isfinite(proxy_fraction) else "  proxy / oracle at lambda=30     : n/a")

    # Verdict
    if og30_jump > 0.05 and og_non_decreasing:
        repbar = "CONFIRMED (> +0.05 at lambda=30 and non-decreasing 10->60)"
    else:
        repbar = "NOT confirmed"
    if np.isfinite(proxy_fraction) and proxy_fraction >= 0.5 and pg30_jump > 0.02:
        proxybar = "SUCCESS (proxy captures at least half of oracle gain)"
    elif og30_jump < 0.02:
        proxybar = "n/a (oracle gain too small to evaluate)"
    else:
        proxybar = "PROXY DOES NOT CAPTURE HALF of oracle gain"
    print(f"  Representation bottleneck       : {repbar}")
    print(f"  Practical proxy result          : {proxybar}")
    print("=" * 108)


if __name__ == "__main__":
    main()
