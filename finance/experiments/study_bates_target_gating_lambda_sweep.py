r"""
BENCHMARK STUDY: target-level gating on Bates, lambda sweep (Part B v2).

Why this script exists
----------------------
The prior Part B (`study_bates_marcus_lambda_sweep.py`) had two honest
defects flagged in review:

  1. The "oracle" target `max(r_t^2/dt - J_t^2/dt, 0)` is NOT a true
     continuous-variance oracle.  With `r_t = c_t + J_t` (continuous
     diffusion plus jump atom), `r_t^2 = c_t^2 + 2 c_t J_t + J_t^2`, so
     subtracting only `J_t^2/dt` leaves a `2 c_t J_t / dt` cross-term.
     It is a partial correction, NOT a representation ceiling.

  2. The "proxy" target `max(r_t^2/dt - bv_smooth_t, 0)` subtracted from
     `r_t^2/dt` reduces algebraically to `min(r_t^2/dt, bv_smooth_t)` --
     a HARD CLIP / WINSOR-style target, not a distinct posterior gate.

Further, the "Marcus" naming was misleading: nothing here lifts the
path to include jump information at the representation level.  We stay
at the standard continuous level-2 lead-lag signature.  Nothing here is
Marcus.

This study replaces all of that with three honest target-level lanes:

  - `ms_cum_stride_hard_proxy`    the old "proxy" renamed honestly.
                                  Target = min(rv_t, bv_smooth_t).
  - `ms_cum_stride_soft_gate`     NEW.  Target-level soft probabilistic
                                  gate (see below).  Online, no
                                  label-based tuning.
  - `ms_cum_stride_oracle_dejump` Target = (r_t - J_t)^2 / dt, the
                                  actual "perfect jump identification"
                                  target-level oracle.  Not deployable.
  - `ms_cum_stride_oracle_latent` UNATTAINABLE CEILING.  Target =
                                  V_true_pre[t].  Noise-free latent
                                  supervision.  Upper bound on what the
                                  current signature representation can
                                  reach with a perfect target.

Plus the existing reference:
  - `ms_cum_stride_bv_target`    target = (pi/2)|r_{t-1}||r_t|/dt.

Scalars: `rv_ewma`, `bv_ewma`, `winsor_ewma`.

Soft gate design (no label tuning)
----------------------------------
Let `rv_t = r_t^2/dt`, `c_t = bv_smooth_t` (online EWMA of per-step
bipower variation, halflife = 21 steps; same smoother used throughout
this study for any online scale reference).  Under NO JUMP, `rv_t / V_t`
is chi-squared-1 noise around 1 (`E[rv_t] = V_t`), with typical V-scale
dispersion.  Define a scale-free excess:

    z_t = max(rv_t - c_t, 0) / max(c_t, 1e-8).

Map it to a jump posterior via the exponential survival function:

    q_t = 1 - exp(-z_t)  in [0, 1].

  - q_t = 0.63 when rv_t exceeds c_t by one V-scale (typical moderate
    fluctuation under no-jump is `~V`; borderline case).
  - q_t = 0.86 at 2 V-scales; 0.95 at 3 V-scales.

Soft continuous target:

    target_soft = rv_t - q_t * max(rv_t - c_t, 0)
                = (1 - q_t) * rv_t + q_t * min(rv_t, c_t).

Interpolation between raw rv_t (when q_t = 0) and hard clip min(rv_t,
c_t) (when q_t = 1).  NO threshold / label tuning.  The only
"parameter" is the halflife of bv_smooth, which is held fixed at the
same 21-step halflife used across the scalar baselines.

Same seed bank as Part A for DIRECT COMPARABILITY
-------------------------------------------------
base_seed = 16_000 and the same seed construction formula.  So the
per-lambda tables here are numerically comparable to Part A's tables.

Pre-registered interpretation
-----------------------------
1. If `soft_gate - hard_proxy` is materially positive on CALM subset
   (with little or no loss on jump subset), soft gating improves
   target design and should become the default target-level Bates
   signature proxy.
2. If `soft_gate` and `hard_proxy` are equivalent, the target-level
   gating is not a meaningful bottleneck -- hard clip is enough.
3. If `oracle_dejump - soft_gate` is small, jump identification is no
   longer the bottleneck; residual is representation / posterior
   noise.  If it grows with lambda, the case for a jump-aware
   representation (true Marcus lift) is stronger.
4. `oracle_latent` is labeled as unattainable ceiling, NOT a
   comparator.  The gap `oracle_latent - oracle_dejump` isolates the
   per-step target noise floor independent of jumps.
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
    EWMAr2,
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
# Signature lanes with pluggable targets (target-level only; no rep change)
# ==========================================================================


def _sig_cfg(bates: BatesConfig) -> CumStrideSigConfig:
    return CumStrideSigConfig(
        strides=(1, 5, 20),
        kappa=bates.kappa, theta=bates.theta, xi=bates.xi, dt=bates.dt,
    )


class BVTargetLane:
    name = "ms_cum_stride_bv_target"

    def __init__(self, bates: BatesConfig):
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=target_bv_per_step, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)

    def observe(self, r_t: float, dt: float, J_t: float = 0.0, V_true: float = 0.0):
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()

    def q_t(self) -> float:
        return float("nan")


class _BVSmoothMixin:
    r"""Shared: online per-step bipower variation smoother (halflife = 21 steps)."""
    HALFLIFE_STEPS = 21.0

    def _init_bv(self, V0: float) -> None:
        self._alpha_bv = 1.0 - float(np.exp(-np.log(2.0) / self.HALFLIFE_STEPS))
        self._bv_smooth = float(V0)
        self._prev_abs_r: Optional[float] = None

    def _update_bv_smooth(self, r_t: float, dt: float) -> float:
        abs_r = float(np.abs(r_t))
        if self._prev_abs_r is None:
            bv_t = (r_t * r_t) / dt
        else:
            bv_t = (np.pi / 2.0) * float(self._prev_abs_r) * abs_r / dt
        self._bv_smooth = (1.0 - self._alpha_bv) * self._bv_smooth + self._alpha_bv * bv_t
        self._prev_abs_r = abs_r
        return self._bv_smooth


class HardProxyLane(_BVSmoothMixin):
    r"""HONEST RELABEL of the previous 'marcus_proxy' lane.

    Target algebraically reduces to hard clip:
        target = rv_t - max(rv_t - bv_smooth_t, 0) = min(rv_t, bv_smooth_t).
    Kept as a REFERENCE POINT to compare soft gating against.
    """
    name = "ms_cum_stride_hard_proxy"

    def __init__(self, bates: BatesConfig):
        self._current_clip: float = 0.0
        def _target(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
            rv = (r_t * r_t) / dt
            return min(rv, self._current_clip)
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=_target, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)
        self._init_bv(V0)
        self._current_clip = float(V0)

    def observe(self, r_t: float, dt: float, J_t: float = 0.0, V_true: float = 0.0):
        self._current_clip = self._update_bv_smooth(r_t, dt)
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()

    def q_t(self) -> float:
        return float("nan")


class SoftGateLane(_BVSmoothMixin):
    r"""Target-level SOFT probabilistic jump gate.

    q_t in [0, 1] from online path info only.  No label tuning.
        c_t      = bv_smooth_t
        z_t      = max(rv_t - c_t, 0) / max(c_t, 1e-8)
        q_t      = 1 - exp(-z_t)
        target   = rv_t - q_t * max(rv_t - c_t, 0)
    """
    name = "ms_cum_stride_soft_gate"

    def __init__(self, bates: BatesConfig):
        self._current_c: float = 0.04
        self._current_q: float = 0.0
        self._last_q: float = 0.0
        def _target(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
            rv = (r_t * r_t) / dt
            c = max(self._current_c, 1e-8)
            excess = max(rv - c, 0.0)
            z = excess / c
            q = 1.0 - float(np.exp(-z))
            self._last_q = q
            self._current_q = q
            return rv - q * excess
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=_target, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)
        self._init_bv(V0)
        self._current_c = float(V0)
        self._current_q = 0.0
        self._last_q = 0.0

    def observe(self, r_t: float, dt: float, J_t: float = 0.0, V_true: float = 0.0):
        self._current_c = self._update_bv_smooth(r_t, dt)
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()

    def q_t(self) -> float:
        return float(self._last_q)


class OracleDejumpLane:
    r"""CORRECTED oracle target: supervises on the ACTUAL continuous r^2/dt.

    target = (r_t - J_t)^2 / dt.  This is the supervised-target ceiling
    under perfect jump identification.  Not deployable.
    """
    name = "ms_cum_stride_oracle_dejump"

    def __init__(self, bates: BatesConfig):
        self._current_J: float = 0.0
        def _target(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
            c = float(r_t) - self._current_J
            return (c * c) / dt
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=_target, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)
        self._current_J = 0.0

    def observe(self, r_t: float, dt: float, J_t: float = 0.0, V_true: float = 0.0):
        self._current_J = float(J_t)
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()

    def q_t(self) -> float:
        return float("nan")


class OracleLatentLane:
    r"""UNATTAINABLE CEILING.  Target = latent V_true_pre[t] directly.

    Noise-free supervision.  Use this to measure how much of the
    residual is per-step target noise (even a perfect de-jumped r^2/dt
    is chi-squared-noisy around V).  Purely an upper bound; NOT a
    practical comparator.
    """
    name = "ms_cum_stride_oracle_latent"

    def __init__(self, bates: BatesConfig):
        self._current_V: float = 0.04
        def _target(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
            return float(self._current_V)
        self.filter = CumStrideSigFilter(
            _sig_cfg(bates), target_fn_single=_target, name=self.name,
        )

    def reset(self, V0: float):
        self.filter.reset(V0)
        self._current_V = float(V0)

    def observe(self, r_t: float, dt: float, J_t: float = 0.0, V_true: float = 0.04):
        self._current_V = float(V_true)
        self.filter.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.filter.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.filter.V_interval()

    def q_t(self) -> float:
        return float("nan")


# ==========================================================================
# Scalar baseline wrappers with uniform observe(r_t, dt, J_t, V_true)
# ==========================================================================


class _ScalarWrap:
    def __init__(self, core, name: str):
        self.core = core
        self.name = name

    def reset(self, V0: float):
        self.core.reset(V0)

    def observe(self, r_t: float, dt: float, J_t: float = 0.0, V_true: float = 0.0):
        self.core.observe(r_t, dt)

    def V_hat(self) -> float:
        return self.core.V_hat()

    def V_interval(self) -> Tuple[float, float]:
        return self.core.V_interval()

    def q_t(self) -> float:
        return float("nan")


# ==========================================================================
# Rollout + pooled stats
# ==========================================================================


def make_factories(bates: BatesConfig) -> Dict[str, Callable[[], object]]:
    halflife = 21.0
    return {
        "rv_ewma":                       lambda: _ScalarWrap(
            EWMAr2(halflife_steps=halflife, dt=bates.dt), "rv_ewma"),
        "bv_ewma":                       lambda: _ScalarWrap(
            EWMABipower(halflife_steps=halflife, dt=bates.dt), "bv_ewma"),
        "winsor_ewma":                   lambda: _ScalarWrap(
            EWMAWinsorized(halflife_steps=halflife, dt=bates.dt, k=4.0), "winsor_ewma"),
        "ms_cum_stride_bv_target":       lambda: BVTargetLane(bates),
        "ms_cum_stride_hard_proxy":      lambda: HardProxyLane(bates),
        "ms_cum_stride_soft_gate":       lambda: SoftGateLane(bates),
        "ms_cum_stride_oracle_dejump":   lambda: OracleDejumpLane(bates),
        "ms_cum_stride_oracle_latent":   lambda: OracleLatentLane(bates),
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
    q_trace = np.zeros(T)          # soft-gate q_t per step
    soft_name = "ms_cum_stride_soft_gate"
    for t in range(T):
        r = float(sim["dr_S"][t])
        J = float(sim["J"][t])
        V_true_t = float(sim["V"][t])
        for name, est in lanes.items():
            est.observe(r, bates.dt, J, V_true_t)
            V_hat[name][t] = est.V_hat()
        if soft_name in lanes:
            q_trace[t] = lanes[soft_name].q_t()
    return {
        "V_true_pre": sim["V"],
        "jump":       sim["jump"],
        "V_hat":      V_hat,
        "q_trace":    q_trace,
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
    out: Dict[float, Dict] = {}
    for lam in lambdas:
        bates = replace(bates_base, lambda_j=float(lam))
        factories = make_factories(bates)
        lane_names = list(factories.keys())
        hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
        V_list: List[np.ndarray] = []
        jump_list: List[np.ndarray] = []
        q_list: List[np.ndarray] = []
        v0_rng = np.random.RandomState(base_seed + int(lam))
        for k in range(n_seeds):
            V0 = float(v0_rng.uniform(0.02, 0.08))
            r_out = _rollout(bates, factories, T, V0, base_seed + int(lam) * 1000 + k)
            for name in lane_names:
                hat[name].append(r_out["V_hat"][name])
            V_list.append(r_out["V_true_pre"])
            jump_list.append(r_out["jump"])
            q_list.append(r_out["q_trace"])
        jump_masks = [_jump_adjacent_mask(j, radius=5) for j in jump_list]
        calm_masks = [~m for m in jump_masks]
        lane_res: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
        for name in lane_names:
            lane_res[name] = {
                "all":  _pooled(hat[name], V_list, warmup=warmup),
                "jump": _pooled(hat[name], V_list, mask_list=jump_masks, warmup=warmup),
                "calm": _pooled(hat[name], V_list, mask_list=calm_masks, warmup=warmup),
            }
        # q_t distribution stats from soft_gate lane
        q_trim = np.concatenate([a[warmup:] for a in q_list])
        jump_trim = np.concatenate([jm[warmup:] for jm in jump_masks])
        calm_trim = np.concatenate([cm[warmup:] for cm in calm_masks])
        q_stats = {
            "mean_all":  float(np.mean(q_trim)),
            "mean_jump": float(np.mean(q_trim[jump_trim])) if jump_trim.any() else float("nan"),
            "mean_calm": float(np.mean(q_trim[calm_trim])) if calm_trim.any() else float("nan"),
            "q90_all":   float(np.quantile(q_trim, 0.90)),
            "q99_all":   float(np.quantile(q_trim, 0.99)),
            "q50_jump":  float(np.quantile(q_trim[jump_trim], 0.5)) if jump_trim.any() else float("nan"),
            "q50_calm":  float(np.quantile(q_trim[calm_trim], 0.5)) if calm_trim.any() else float("nan"),
        }
        # Per-lambda sample of q_t for histograms
        out[float(lam)] = {
            "lane_res": lane_res,
            "q_stats":  q_stats,
            "q_trim":   q_trim,
            "jump_trim": jump_trim,
            "calm_trim": calm_trim,
        }
    return out


# ==========================================================================
# Reporting
# ==========================================================================


LANE_ORDER = [
    "rv_ewma", "bv_ewma", "winsor_ewma",
    "ms_cum_stride_bv_target",
    "ms_cum_stride_hard_proxy",
    "ms_cum_stride_soft_gate",
    "ms_cum_stride_oracle_dejump",
    "ms_cum_stride_oracle_latent",
]


def _print_per_lambda(results) -> None:
    for lam, d in sorted(results.items()):
        lane_res = d["lane_res"]
        print(f"lambda_j = {lam:.0f} / yr")
        print("-" * 112)
        print(f"  {'lane':32s} | {'corr(all)':>9s} | {'corr(jump)':>10s} | "
              f"{'corr(calm)':>10s} | {'RMSE(all)':>9s}")
        print("-" * 112)
        for name in LANE_ORDER:
            if name not in lane_res: continue
            r_all, c_all, _ = lane_res[name]["all"]
            r_j,   c_j,   _ = lane_res[name]["jump"]
            r_c,   c_c,   _ = lane_res[name]["calm"]
            label = name
            if name == "ms_cum_stride_oracle_latent":
                label = name + " [ceiling]"
            print(f"  {label:32s} | {c_all:+.4f}   | {c_j:+.4f}    | "
                  f"{c_c:+.4f}    | {r_all:.4f}")
        print()


def _print_q_stats(results) -> None:
    print("Soft-gate q_t distribution (per lambda, post-warm-up)")
    print("-" * 112)
    print(f"  {'lambda':>7s} | {'mean(all)':>9s} | {'mean(jump)':>10s} | "
          f"{'mean(calm)':>10s} | {'median(jump)':>12s} | {'median(calm)':>12s} | "
          f"{'q90(all)':>8s} | {'q99(all)':>8s}")
    print("-" * 112)
    for lam, d in sorted(results.items()):
        q = d["q_stats"]
        print(f"  {lam:7.0f} | {q['mean_all']:.4f}    | {q['mean_jump']:.4f}     | "
              f"{q['mean_calm']:.4f}     | {q['q50_jump']:.4f}       | {q['q50_calm']:.4f}       | "
              f"{q['q90_all']:.4f}   | {q['q99_all']:.4f}")
    print()


def _print_deltas(results) -> None:
    print("Derived deltas per lambda (corr differences)")
    print("  bv_target - rv_ewma                 (target fix over plain scalar)")
    print("  hard_proxy - bv_target              (does hard clip help over BV per-step?)")
    print("  soft_gate  - hard_proxy             (does soft gating beat hard clip?)")
    print("  oracle_dejump - soft_gate           (residual jump identification gap)")
    print("  oracle_latent - oracle_dejump       (per-step target noise floor)")
    print("  winsor_ewma - oracle_latent          (scalar vs sig-ceiling comparison)")
    print("-" * 118)
    print(f"  {'lambda':>7s} | {'subset':>6s} | {'bvt-rv':>7s} | "
          f"{'hp-bvt':>7s} | {'sg-hp':>7s} | {'ord-sg':>7s} | "
          f"{'olat-ord':>8s} | {'win-olat':>8s}")
    print("-" * 118)
    for lam, d in sorted(results.items()):
        lane_res = d["lane_res"]
        def c(name, sub):
            return lane_res[name][sub][1]
        for sub in ("all", "jump", "calm"):
            bvt_rv  = c("ms_cum_stride_bv_target", sub)     - c("rv_ewma", sub)
            hp_bvt  = c("ms_cum_stride_hard_proxy", sub)    - c("ms_cum_stride_bv_target", sub)
            sg_hp   = c("ms_cum_stride_soft_gate", sub)     - c("ms_cum_stride_hard_proxy", sub)
            ord_sg  = c("ms_cum_stride_oracle_dejump", sub) - c("ms_cum_stride_soft_gate", sub)
            olat_ord = c("ms_cum_stride_oracle_latent", sub) - c("ms_cum_stride_oracle_dejump", sub)
            win_olat = c("winsor_ewma", sub)                - c("ms_cum_stride_oracle_latent", sub)
            print(f"  {lam:7.0f} | {sub:>6s} | {bvt_rv:+.4f} | {hp_bvt:+.4f} | "
                  f"{sg_hp:+.4f} | {ord_sg:+.4f} | {olat_ord:+.4f}  | {win_olat:+.4f}")
    print()


def _plot(results, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = sorted(results.keys())
    palette = {
        "rv_ewma":                     "tab:olive",
        "bv_ewma":                     "tab:purple",
        "winsor_ewma":                 "tab:green",
        "ms_cum_stride_bv_target":     "tab:cyan",
        "ms_cum_stride_hard_proxy":    "tab:red",
        "ms_cum_stride_soft_gate":     "tab:orange",
        "ms_cum_stride_oracle_dejump": "tab:brown",
        "ms_cum_stride_oracle_latent": "black",
    }
    linestyle = {"ms_cum_stride_oracle_latent": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for (row, col, subset, title) in [
        (0, 0, "all",  "Corr, all samples"),
        (0, 1, "jump", "Corr, jump-adjacent (+/-5 bars)"),
        (1, 0, "calm", "Corr, calm subset"),
    ]:
        ax = axes[row, col]
        for name in LANE_ORDER:
            y = [results[l]["lane_res"][name][subset][1] for l in lams]
            ls = linestyle.get(name, "-")
            lw = 1.8 if name == "ms_cum_stride_oracle_latent" else 1.4
            ax.plot(lams, y, ls + "o", color=palette.get(name), lw=lw, label=name, markersize=5)
        ax.set_xlabel("lambda_j")
        ax.set_ylabel(f"corr({subset})")
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")

    # Panel (1,1): q_t separation histograms (one per lambda)
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 41)
    colors = ["tab:blue", "tab:orange", "tab:red"]
    for i, lam in enumerate(lams):
        d = results[lam]
        q = d["q_trim"]
        jmask = d["jump_trim"]
        cmask = d["calm_trim"]
        ax.hist(q[jmask], bins=bins, alpha=0.4, color=colors[i], density=True,
                label=f"lam={lam:.0f} jump")
        ax.hist(q[cmask], bins=bins, alpha=0.25, color=colors[i],
                density=True, histtype="step",
                label=f"lam={lam:.0f} calm")
    ax.set_xlabel("q_t  (soft-gate jump confidence)")
    ax.set_ylabel("density")
    ax.set_title("Soft-gate q_t: separation of jump vs calm (both subsets per lambda)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right", ncol=1)

    fig.suptitle(
        "Bates target-level gating sweep (Part B v2): soft gate + corrected oracle + latent ceiling",
        fontsize=12,
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
    base_seed = 16_000                                  # SAME as Part A

    print("=" * 112)
    print("BENCHMARK STUDY  --  Bates target-level gating (Part B v2, corrected)")
    print(f"  lambda sweep: {lambdas}  /yr")
    print(f"  n_seeds={n_seeds}, T={T}, warm-up={warmup}, dt={bates_base.dt:.5f}")
    print(f"  SEED BANK MATCHED TO PART A  (base_seed={base_seed})")
    print()
    print("  Honest lane naming and framing:")
    print("    hard_proxy     : target = min(rv_t, bv_smooth_t)  (hard clip)")
    print("    soft_gate      : q_t = 1 - exp(-max(rv_t - c_t, 0)/c_t);"
          "  target = rv_t - q_t * max(rv_t - c_t, 0)")
    print("    oracle_dejump  : target = (r_t - J_t)^2 / dt  (NOT deployable)")
    print("    oracle_latent  : target = V_true_pre[t]       (UNATTAINABLE ceiling)")
    print("=" * 112)

    results = run_sweep(
        bates_base=bates_base, lambdas=lambdas,
        n_seeds=n_seeds, T=T, warmup=warmup, base_seed=base_seed,
    )
    _print_per_lambda(results)
    _print_q_stats(results)
    _print_deltas(results)

    out_path = os.path.join(HERE, "study_bates_target_gating_lambda_sweep.png")
    try:
        _plot(results, out_path)
        print(f"Saved figure: {out_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")

    # Pre-registered interpretation summary
    lams = sorted(results.keys())
    def corr_of(name: str, lam: float, sub: str) -> float:
        return float(results[lam]["lane_res"][name][sub][1])

    print()
    print("=" * 112)
    print("PART B v2 INTERPRETATION (pre-registered)")
    print("=" * 112)

    # 1. soft_gate vs hard_proxy on CALM subset; bar: materially positive, little/no loss on jump.
    sg_minus_hp_calm = [corr_of("ms_cum_stride_soft_gate", l, "calm")
                        - corr_of("ms_cum_stride_hard_proxy", l, "calm") for l in lams]
    sg_minus_hp_jump = [corr_of("ms_cum_stride_soft_gate", l, "jump")
                        - corr_of("ms_cum_stride_hard_proxy", l, "jump") for l in lams]
    print(f"  (1) soft_gate - hard_proxy (calm) per lambda : {[f'{v:+.4f}' for v in sg_minus_hp_calm]}")
    print(f"       soft_gate - hard_proxy (jump) per lambda: {[f'{v:+.4f}' for v in sg_minus_hp_jump]}")
    calm_gain_avg = float(np.mean(sg_minus_hp_calm))
    jump_loss = float(np.min(sg_minus_hp_jump))
    if calm_gain_avg > 0.02 and jump_loss > -0.01:
        v1 = "SOFT GATE WINS on calm with no loss on jump.  Make it the default target-level proxy."
    elif calm_gain_avg > 0.0 and jump_loss > -0.02:
        v1 = "Soft gate slightly helps calm, minimal jump loss.  Mild positive."
    elif calm_gain_avg > -0.01 and jump_loss > -0.01:
        v1 = "Soft gate ~= hard proxy.  Target-level gating is not the bottleneck."
    else:
        v1 = "Soft gate does NOT beat hard proxy.  Hard clip is fine."
    print(f"       Verdict (1): {v1}")

    # 2. oracle_dejump - soft_gate: jump identification residual
    ord_minus_sg_jump = [corr_of("ms_cum_stride_oracle_dejump", l, "jump")
                         - corr_of("ms_cum_stride_soft_gate", l, "jump") for l in lams]
    ord_minus_sg_calm = [corr_of("ms_cum_stride_oracle_dejump", l, "calm")
                         - corr_of("ms_cum_stride_soft_gate", l, "calm") for l in lams]
    print(f"  (2) oracle_dejump - soft_gate (jump) per lam : {[f'{v:+.4f}' for v in ord_minus_sg_jump]}")
    print(f"       oracle_dejump - soft_gate (calm) per lam: {[f'{v:+.4f}' for v in ord_minus_sg_calm]}")
    growing_j = ord_minus_sg_jump[-1] > ord_minus_sg_jump[0] + 0.02
    if abs(float(np.max(ord_minus_sg_jump))) < 0.02:
        v2 = "Jump identification is NOT the main bottleneck.  soft_gate ~= oracle_dejump."
    elif growing_j:
        v2 = "Jump identification gap grows with lambda.  Representation fix (not just target) is more warranted."
    else:
        v2 = "Jump identification gap is small / flat.  Residual is mostly elsewhere."
    print(f"       Verdict (2): {v2}")

    # 3. oracle_latent - oracle_dejump  (per-step target noise floor)
    olat_minus_ord = [corr_of("ms_cum_stride_oracle_latent", l, "all")
                      - corr_of("ms_cum_stride_oracle_dejump", l, "all") for l in lams]
    print(f"  (3) oracle_latent - oracle_dejump (all) per lam: {[f'{v:+.4f}' for v in olat_minus_ord]}")
    v3 = ("Latent supervision closes substantial extra corr over de-jumped r^2/dt;"
          " per-step target noise is a real floor."
          if np.max(olat_minus_ord) > 0.03 else
          "Latent supervision adds little over de-jumped r^2/dt.  The per-step noise floor is small.")
    print(f"       Verdict (3): {v3}")

    # 4. winsor vs oracle_latent  (scalar vs sig-ceiling comparison)
    win_minus_olat = [corr_of("winsor_ewma", l, "all")
                      - corr_of("ms_cum_stride_oracle_latent", l, "all") for l in lams]
    print(f"  (4) winsor_ewma - oracle_latent (all) per lam: {[f'{v:+.4f}' for v in win_minus_olat]}")
    if np.max(win_minus_olat) > 0.02:
        v4 = ("Even with PERFECT supervision, the signature representation does not beat winsor_ewma."
              "  Sig lane has a structural ceiling below scalar at this config.")
    elif np.min(win_minus_olat) < -0.02:
        v4 = ("With perfect supervision, signature representation surpasses winsor_ewma."
              "  Residual is target design, not representation.")
    else:
        v4 = "Signature ceiling matches scalar approximately under perfect supervision."
    print(f"       Verdict (4): {v4}")

    print("=" * 112)


if __name__ == "__main__":
    main()
