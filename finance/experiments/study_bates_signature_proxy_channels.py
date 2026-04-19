r"""
BENCHMARK STUDY: signature-derived proxy channels on Bates jumps.

Context
-------
The Bates filter study (`study_bates_signature_filters.py`) showed a
clear failure mode for the current signature lane:

  - online BLR targets are raw `r_t^2 / dt`
  - a few jump-contaminated steps corrupt BLR posterior weights
  - calm-window corr is reasonable (~0.58), but jump-window corr collapses

The winsorized EWMA scalar wins that benchmark NOT because signatures
"cannot see" the latent V, but because they are trained on the wrong
local characteristic.

This follow-up study tests a focused, honest hypothesis:
  Is the Bates signature failure primarily a TARGET problem (what we
  supervise the BLR on), or a REPRESENTATION problem (the multires
  cumulative-stride sig state cannot distinguish continuous variation
  from jumps)?

Conceptual framing (kept explicit in comments):
  - EXTERNAL handcrafted proxy  -- e.g. BV or winsorized EWMA.
                                   Computed from r_t alone, no sig state.
  - SIGNATURE-DERIVED PROXY     -- a local characteristic that uses the
                                   existing sig features/state as input
                                   to a small learned readout (BLR).
  - TARGET                      -- the supervised signal the BLR is
                                   updated toward.  Swapping this is a
                                   TARGET change, not an architecture
                                   change.

Theory reference
----------------
From `docs/hida_malliavin_signature_unification.md`:
  - Level-2 lead-lag geometry captures QV-type information.
  - Level-3 is the natural home for jump / skew information.
  - The standard continuous signature is insufficient for jumps; a
    Marcus extension is the principled cure.  Marcus is OUT of scope
    for this pass.  We stay at the existing level-2 lead-lag state and
    test whether target/channel engineering alone closes the gap.

Signature lanes compared
------------------------
  A. ms_cum_stride_raw         -- existing lane, target = r_t^2/dt.
  B. ms_cum_stride_bv_target   -- same sig state, target = BV per step
                                   = (pi/2) * |r_{t-1}| * |r_t| / dt.
  C. ms_cum_stride_two_channel -- same sig state; two BLR heads per
                                   stride -> two outer CIR Kalmans:
                                   continuous channel (BV target) +
                                   jump channel (max(r^2/dt - BV, 0)
                                   target).  V_hat = continuous only
                                   (the latent V we compare against is
                                   continuous-Heston by construction).

Scalar baselines
----------------
  rv_ewma      -- EWMA of r^2/dt.                 (plain, non-robust)
  bv_ewma      -- EWMA of (pi/2)|r_{t-1}||r_t|/dt.(robust handcrafted)
  winsor_ewma  -- EWMA of min(r^2/dt, k*V_running).(robust handcrafted)

All signature lanes share the SAME signature state and stride ladder
as `ms_cum_stride_raw`: strides (1, 5, 20) days.  No hyperparameter
sweep.  Only the SUPERVISED TARGET differs.

Pre-registered strong-positive bar
----------------------------------
A "meaningful recovery" is declared ONLY if BOTH hold:
  - best signature-proxy lane (B or C) improves over
    `ms_cum_stride_raw` by > +0.10 corr on the jump-adjacent subset,
  AND
  - best signature-proxy lane closes at least HALF the gap from
    `ms_cum_stride_raw` to `winsor_ewma` on overall corr.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

# Reuse the Bates simulator + scalar baselines + jump-adjacent mask
# from the prior study.  NO new simulator.
from study_bates_signature_filters import (
    BatesConfig,
    EWMAr2,
    EWMABipower,
    EWMAWinsorized,
    _jump_adjacent_mask,
    simulate_bates,
)
from examples.proof_of_concept.signature_features import (
    RecurrentLeadLagLogSigMap,
)
from src.sskf.multiscale_leadlag_filters import (
    BLRHead3,
    OuterCIRKalmanV,
    _precision_combine,
    _QV_IDX_IN_L2,
    _RET_LEAD_IDX_IN_L1,
)


# ==========================================================================
# Minimal cumulative-stride sig filter with PLUGGABLE target function(s).
# --------------------------------------------------------------------------
# Reuses the exact architecture of CumulativeStrideLeadLagBLRKFilter
# (one cumulative lead-lag log-sig + Chen-level-2 window recovery at
# calendar strides + per-stride BLR -> precision-fused -> outer CIR
# Kalman), but exposes the TARGET function as a constructor argument.
# When a single target is specified, we get one outer Kalman.  When a
# TARGET PAIR is specified, we get a two-channel filter with two
# independent outer Kalmans (for diagnostics).
# ==========================================================================


def _chen_level2_window(
    a1_past: np.ndarray, a1_now: np.ndarray,
    a2_past: np.ndarray, a2_now: np.ndarray,
    dim_l2: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Exact Chen identity at level 2 on lead-lag log-sig:
      a_2_window = a_2_now - a_2_past - 0.5 * [a_1_past, a_1_now]
    """
    a1_win = a1_now - a1_past
    d = a1_past.size
    bracket = np.zeros(dim_l2)
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            bracket[idx] = a1_past[i] * a1_now[j] - a1_past[j] * a1_now[i]
            idx += 1
    a2_win = a2_now - a2_past - 0.5 * bracket
    return a1_win, a2_win


def _phi_from_window(a1_win: np.ndarray, a2_win: np.ndarray, m: int, dt: float) -> np.ndarray:
    r"""Same normalization as `CumulativeStrideLeadLagBLRKFilter._phi_from_window`."""
    scale = max(float(m) * dt, 1e-12)
    return np.array([
        2.0 * float(a2_win[_QV_IDX_IN_L2]) / scale,
        float(a1_win[_RET_LEAD_IDX_IN_L1]) / scale,
        1.0,
    ])


@dataclass
class CumStrideSigConfig:
    strides: Tuple[int, ...]
    kappa: float
    theta: float
    xi: float
    dt: float
    V_floor: float = 1e-6
    P_init_mult: float = 10.0
    prior_w_var: float = 10.0
    sigma_n2_init: float = 0.01
    sigma_n2_alpha: float = 0.01


class CumStrideSigFilter:
    r"""Signature state + pluggable target.  Supports one-channel or two-channel.

    Target function signature:
      target_fn(r_t: float, dt: float, prev_abs_r: Optional[float],
                V_running: float) -> float

    For the two-channel variant, pass a PAIR of target functions
    (cont_target_fn, jump_target_fn).  Each channel maintains its own
    BLR heads, its own outer Kalman, and its own posterior V estimate.
    """

    def __init__(
        self, cfg: CumStrideSigConfig,
        target_fn_single: Optional[Callable] = None,
        target_fn_pair: Optional[Tuple[Callable, Callable]] = None,
        name: str = "cum_stride_sig",
    ):
        assert (target_fn_single is not None) != (target_fn_pair is not None), (
            "specify exactly one of target_fn_single or target_fn_pair"
        )
        self.cfg = cfg
        self.name = name
        self.target_fn_single = target_fn_single
        self.target_fn_pair = target_fn_pair
        self.two_channel = target_fn_pair is not None
        self.strides = tuple(int(m) for m in cfg.strides)
        self.K = len(self.strides)

        # Single cumulative lead-lag log-sig state.
        self._ll = RecurrentLeadLagLogSigMap(
            state_dim=2, level=2, forgetting_factor=1.0,
        )
        self._max_stride = max(self.strides)
        self._buf_size = self._max_stride + 2
        self._l1_buf = np.zeros((self._buf_size, self._ll.dim_l1))
        self._l2_buf = np.zeros((self._buf_size, self._ll.dim_l2))
        self._buf_idx = 0
        self._buf_count = 0

        # Per-channel per-stride BLR heads + one outer Kalman per channel.
        self._n_channels = 2 if self.two_channel else 1
        self._heads: List[List[BLRHead3]] = [
            [
                BLRHead3.fresh(
                    prior_w_var=cfg.prior_w_var,
                    sigma_n2_init=cfg.sigma_n2_init,
                    sigma_n2_alpha=cfg.sigma_n2_alpha,
                )
                for _ in self.strides
            ]
            for _ in range(self._n_channels)
        ]
        P0 = cfg.xi ** 2 * cfg.theta * cfg.dt * cfg.P_init_mult
        self._kf = [
            OuterCIRKalmanV(
                kappa=cfg.kappa, theta=cfg.theta, xi=cfg.xi,
                V=cfg.theta, P=P0, V_floor=cfg.V_floor,
            )
            for _ in range(self._n_channels)
        ]

        # Running state for target computations (prev |r|, V running).
        self._prev_abs_r: Optional[float] = None
        self._last_z: float = 0.0

    def reset(self, V0: float) -> None:
        V0 = max(float(V0), self.cfg.V_floor)
        P0 = self.cfg.xi ** 2 * self.cfg.theta * self.cfg.dt * self.cfg.P_init_mult
        for kf in self._kf:
            kf.V = V0
            kf.P = P0
        self._ll.reset()
        self._l1_buf[:] = 0.0
        self._l2_buf[:] = 0.0
        self._buf_idx = 0
        self._buf_count = 0
        for ch in self._heads:
            for h in ch:
                h.reset()
        self._prev_abs_r = None
        self._last_z = 0.0

    def _buf_get(self, m: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._buf_count <= m:
            return None, None
        idx = (self._buf_idx - 1 - m) % self._buf_size
        return self._l1_buf[idx].copy(), self._l2_buf[idx].copy()

    def observe(self, r_t: float, dt: float) -> None:
        # 1) advance sig state
        dx = np.array([float(dt), float(r_t)])
        self._ll.update(dx)
        a1_now = self._ll.l1.copy()
        a2_now = self._ll.l2.copy()
        idx_buf = self._buf_idx % self._buf_size
        self._l1_buf[idx_buf] = a1_now
        self._l2_buf[idx_buf] = a2_now
        self._buf_idx += 1
        self._buf_count = min(self._buf_count + 1, self._buf_size)

        # 2) per-channel predictive Gaussians from per-stride BLRs
        per_channel_active: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(self._n_channels)]
        per_channel_ys: List[List[float]] = [[] for _ in range(self._n_channels)]
        per_channel_Rs: List[List[float]] = [[] for _ in range(self._n_channels)]
        for k, m in enumerate(self.strides):
            a1_past, a2_past = self._buf_get(m)
            if a1_past is None:
                continue
            a1_win, a2_win = _chen_level2_window(
                a1_past, a1_now, a2_past, a2_now, self._ll.dim_l2,
            )
            phi = _phi_from_window(a1_win, a2_win, m, self.cfg.dt)
            for ch in range(self._n_channels):
                y_pred, R_pred = self._heads[ch][k].predict(phi)
                per_channel_ys[ch].append(max(y_pred, 1e-8))
                per_channel_Rs[ch].append(R_pred)
                per_channel_active[ch].append((k, phi))

        # 3) per-channel outer Kalman predict+absorb
        for ch in range(self._n_channels):
            V_pred, P_pred = self._kf[ch].predict(dt)
            if not per_channel_ys[ch]:
                self._kf[ch].V = max(V_pred, self.cfg.V_floor)
                self._kf[ch].P = P_pred
                continue
            y_bar, R_bar = _precision_combine(per_channel_ys[ch], per_channel_Rs[ch])
            V_new, P_new, z = self._kf[ch].absorb(V_pred, P_pred, y_bar, R_bar)
            self._kf[ch].V = V_new
            self._kf[ch].P = P_new
            if ch == 0:
                self._last_z = z

        # 4) per-channel BLR posterior update on the SUPERVISED target(s)
        if self.two_channel:
            cont_fn, jump_fn = self.target_fn_pair
            # Use the V running of the CONTINUOUS channel as a reference (useful for winsor-style).
            V_run = self._kf[0].V
            t_cont = float(cont_fn(r_t, dt, self._prev_abs_r, V_run))
            t_jump = float(jump_fn(r_t, dt, self._prev_abs_r, V_run))
            for k, phi in per_channel_active[0]:
                self._heads[0][k].update(phi, t_cont)
            for k, phi in per_channel_active[1]:
                self._heads[1][k].update(phi, t_jump)
        else:
            V_run = self._kf[0].V
            t_val = float(self.target_fn_single(r_t, dt, self._prev_abs_r, V_run))
            for k, phi in per_channel_active[0]:
                self._heads[0][k].update(phi, t_val)

        self._prev_abs_r = float(np.abs(r_t))

    def V_hat(self) -> float:
        return float(self._kf[0].V)

    def V_hat_cont(self) -> float:
        return float(self._kf[0].V)

    def V_hat_jump(self) -> float:
        if not self.two_channel:
            return float("nan")
        return float(self._kf[1].V)

    def last_z(self) -> float:
        return float(self._last_z)

    def V_interval(self) -> Tuple[float, float]:
        sd = float(np.sqrt(max(self._kf[0].P, 1e-18)))
        return (
            float(max(self._kf[0].V - 1.645 * sd, 0.0)),
            float(self._kf[0].V + 1.645 * sd),
        )


# ==========================================================================
# Target functions
# ==========================================================================


def target_raw_r2(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
    return (r_t * r_t) / dt


def target_bv_per_step(r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float) -> float:
    r"""Bipower-variation per-step: (pi/2) * |r_{t-1}| * |r_t| / dt.

    Warm-up: when prev_abs_r is None (very first step), fall back to
    min(r^2/dt, 3*V_run) to avoid injecting a large atom into the prior.
    Documented warm-up: one step.
    """
    abs_r = float(np.abs(r_t))
    if prev_abs_r is None:
        return min((r_t * r_t) / dt, 3.0 * max(V_run, 1e-8))
    return (np.pi / 2.0) * float(prev_abs_r) * abs_r / dt


def target_jump_residual(
    r_t: float, dt: float, prev_abs_r: Optional[float], V_run: float,
) -> float:
    r"""Jump channel target: positive residual of r^2/dt over BV per step.

    rv = r_t^2 / dt.  bv = (pi/2) |r_{t-1}| |r_t| / dt.  target = max(rv - bv, 0).
    """
    rv = (r_t * r_t) / dt
    if prev_abs_r is None:
        return 0.0
    abs_r = float(np.abs(r_t))
    bv = (np.pi / 2.0) * float(prev_abs_r) * abs_r / dt
    return max(rv - bv, 0.0)


# ==========================================================================
# Study driver
# ==========================================================================


def _make_bates_factories(
    bates: BatesConfig,
) -> Dict[str, Callable[[], object]]:
    sig_cfg = CumStrideSigConfig(
        strides=(1, 5, 20),
        kappa=bates.kappa, theta=bates.theta, xi=bates.xi, dt=bates.dt,
    )
    halflife = 21.0
    return {
        "rv_ewma":                   lambda: EWMAr2(halflife_steps=halflife, dt=bates.dt),
        "bv_ewma":                   lambda: EWMABipower(halflife_steps=halflife, dt=bates.dt),
        "winsor_ewma":               lambda: EWMAWinsorized(halflife_steps=halflife, dt=bates.dt, k=4.0),
        "ms_cum_stride_raw":         lambda: CumStrideSigFilter(
            sig_cfg, target_fn_single=target_raw_r2, name="ms_cum_stride_raw",
        ),
        "ms_cum_stride_bv_target":   lambda: CumStrideSigFilter(
            sig_cfg, target_fn_single=target_bv_per_step,
            name="ms_cum_stride_bv_target",
        ),
        "ms_cum_stride_two_channel": lambda: CumStrideSigFilter(
            sig_cfg,
            target_fn_pair=(target_bv_per_step, target_jump_residual),
            name="ms_cum_stride_two_channel",
        ),
    }


def _fresh_lanes(factories: Dict[str, Callable[[], object]], V0: float) -> Dict[str, object]:
    lanes = {name: factory() for name, factory in factories.items()}
    for est in lanes.values():
        est.reset(V0)
    return lanes


def _rollout(
    bates: BatesConfig, factories: Dict[str, Callable[[], object]],
    T: int, V0: float, seed: int,
) -> Dict[str, np.ndarray]:
    sim = simulate_bates(bates, T, V0, seed)
    lanes = _fresh_lanes(factories, V0)
    V_hat_post = {name: np.zeros(T) for name in lanes}
    # Two-channel diagnostics (only filled for the two-channel lane).
    V_hat_jump_post = np.full(T, np.nan)
    # Per-step BV reference target (for channel diagnostics).
    bv_series = np.zeros(T)
    prev_abs_r: Optional[float] = None
    for t in range(T):
        r = float(sim["dr_S"][t])
        for name, est in lanes.items():
            est.observe(r, bates.dt)
            V_hat_post[name][t] = est.V_hat()
            if name == "ms_cum_stride_two_channel" and hasattr(est, "V_hat_jump"):
                V_hat_jump_post[t] = est.V_hat_jump()
        # Per-step BV reference (for corr(cont channel, BV)):
        abs_r = float(np.abs(r))
        if prev_abs_r is None:
            bv_series[t] = (r * r) / bates.dt
        else:
            bv_series[t] = (np.pi / 2.0) * prev_abs_r * abs_r / bates.dt
        prev_abs_r = abs_r
    return {
        "V_true_pre": sim["V"],
        "dr_S":       sim["dr_S"],
        "jump":       sim["jump"],
        "V_hat_post": V_hat_post,
        "V_hat_jump": V_hat_jump_post,
        "bv_series":  bv_series,
    }


def _pooled(pred_list: List[np.ndarray], tgt_list: List[np.ndarray],
            mask_list: Optional[List[np.ndarray]] = None, warmup: int = 30,
            ) -> Tuple[float, float, int]:
    p_all: List[np.ndarray] = []
    t_all: List[np.ndarray] = []
    for i in range(len(pred_list)):
        p = pred_list[i][warmup:]
        t = tgt_list[i][warmup:]
        if mask_list is not None:
            m = mask_list[i][warmup:]
            p = p[m]
            t = t[m]
        p_all.append(p)
        t_all.append(t)
    p_cat = np.concatenate(p_all) if p_all else np.zeros(0)
    t_cat = np.concatenate(t_all) if t_all else np.zeros(0)
    mask = np.isfinite(p_cat) & np.isfinite(t_cat)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    rmse = float(np.sqrt(np.mean((p_cat[mask] - t_cat[mask]) ** 2)))
    corr = float(np.corrcoef(p_cat[mask], t_cat[mask])[0, 1])
    return rmse, corr, int(mask.sum())


def run_proxy_study(
    bates: BatesConfig, n_seeds: int, T: int, warmup: int,
    base_seed: int,
):
    factories = _make_bates_factories(bates)
    lane_names = list(factories.keys())
    hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
    V_jump_list: List[np.ndarray] = []
    V_list: List[np.ndarray] = []
    bv_list: List[np.ndarray] = []
    jump_list: List[np.ndarray] = []
    dr_list: List[np.ndarray] = []
    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_seeds):
        V0 = float(v0_rng.uniform(0.02, 0.08))
        out = _rollout(bates, factories, T, V0, base_seed + 1 + k)
        for name in lane_names:
            hat[name].append(out["V_hat_post"][name])
        V_list.append(out["V_true_pre"])
        V_jump_list.append(out["V_hat_jump"])
        bv_list.append(out["bv_series"])
        jump_list.append(out["jump"])
        dr_list.append(out["dr_S"])
    jump_masks = [_jump_adjacent_mask(j, radius=5) for j in jump_list]
    calm_masks = [~m for m in jump_masks]

    results: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for name in lane_names:
        stats = {}
        stats["spot_V_all"]  = _pooled(hat[name], V_list, warmup=warmup)
        stats["spot_V_jump"] = _pooled(hat[name], V_list, mask_list=jump_masks, warmup=warmup)
        stats["spot_V_calm"] = _pooled(hat[name], V_list, mask_list=calm_masks, warmup=warmup)
        results[name] = stats

    # Channel diagnostics for two-channel lane:
    # - continuous channel V_hat_cont vs per-step BV: should be high
    #   (BV is the target the continuous channel is trained on)
    # - jump channel V_hat_jump vs jump indicator: should be positive
    channel_diag: Dict[str, Tuple[float, float, int]] = {}
    cont_hat = hat["ms_cum_stride_two_channel"]
    # corr(cont channel, BV series smoothed over same 21-halflife for fair scale)
    bv_smooth_list: List[np.ndarray] = []
    alpha = 1.0 - float(np.exp(-np.log(2.0) / 21.0))
    for bv in bv_list:
        smooth = np.zeros_like(bv)
        s = bv[0]
        for i, b in enumerate(bv):
            s = (1.0 - alpha) * s + alpha * b
            smooth[i] = s
        bv_smooth_list.append(smooth)
    channel_diag["cont_vs_bv_smoothed"] = _pooled(cont_hat, bv_smooth_list, warmup=warmup)
    # corr(jump channel, jump-energy) where jump-energy = (rv - bv_smoothed)_+
    jump_energy_list = []
    for dr, bvs in zip(dr_list, bv_smooth_list):
        rv = (dr ** 2) / bates.dt
        jump_energy_list.append(np.maximum(rv - bvs, 0.0))
    channel_diag["jump_vs_jump_energy"] = _pooled(
        V_jump_list, jump_energy_list, warmup=warmup,
    )
    # corr(jump channel, binary jump indicator)
    jump_ind_list = [j.astype(float) for j in jump_list]
    channel_diag["jump_vs_jump_indicator"] = _pooled(
        V_jump_list, jump_ind_list, warmup=warmup,
    )

    # Hold a single representative trajectory for the plot (first seed).
    trace = {
        "V_true_pre": V_list[0],
        "dr_S":       dr_list[0],
        "jump":       jump_list[0],
        "V_hat_post": {name: hat[name][0] for name in lane_names},
        "V_hat_jump": V_jump_list[0],
        "bv_smooth":  bv_smooth_list[0],
    }
    return results, channel_diag, trace


# ==========================================================================
# Reporting
# ==========================================================================


def _print_table(results) -> None:
    cols = ["spot_V_all", "spot_V_jump", "spot_V_calm"]
    labels = {"spot_V_all": "spot V (all)",
              "spot_V_jump": "spot V (jump)",
              "spot_V_calm": "spot V (calm)"}
    print("Bates signature-proxy study  --  pooled corr / RMSE vs latent spot V")
    print("-" * 108)
    print(f"{'lane':30s} | {'stat':>5s} |", end="")
    for k in cols:
        print(f"  {labels[k]:>16s}", end="")
    print()
    print("-" * 108)
    for lane, stats in results.items():
        print(f"{lane:30s} | {'corr':>5s} |", end="")
        for k in cols:
            _, c, _ = stats.get(k, (float("nan"), float("nan"), 0))
            cell = f"{c:+.4f}" if np.isfinite(c) else "    nan"
            print(f"  {cell:>16s}", end="")
        print()
    print("-" * 108)
    for lane, stats in results.items():
        print(f"{lane:30s} | {'rmse':>5s} |", end="")
        for k in cols:
            r, _, _ = stats.get(k, (float("nan"), float("nan"), 0))
            cell = f"{r:.4f}" if np.isfinite(r) else "   nan"
            print(f"  {cell:>16s}", end="")
        print()
    print()


def _print_channel_diag(diag) -> None:
    print("Two-channel diagnostics  (corr / RMSE; predictions from the two-channel lane)")
    print("-" * 72)
    for key, (rmse, corr, n) in diag.items():
        cell_c = f"{corr:+.4f}" if np.isfinite(corr) else "nan"
        cell_r = f"{rmse:.4f}" if np.isfinite(rmse) else "nan"
        print(f"  {key:26s}  corr={cell_c:>8s}   rmse={cell_r:>8s}   (n={n})")
    print()


def _plot(trace, out_path: str, dt: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V_true = trace["V_true_pre"]
    V_hats = trace["V_hat_post"]
    V_jump = trace["V_hat_jump"]
    bv_sm = trace["bv_smooth"]
    jump = trace["jump"]
    T = V_true.size
    t_axis = np.arange(T)
    jump_idx = np.where(jump > 0)[0]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # Panel 1: scalar baselines
    ax = axes[0]
    ax.plot(t_axis, V_true, "k-", lw=1.4, label="V_true (spot)")
    for j in jump_idx:
        ax.axvline(j, color="gray", alpha=0.3, lw=0.7)
    for name in ("rv_ewma", "bv_ewma", "winsor_ewma"):
        if name in V_hats:
            ax.plot(t_axis, V_hats[name], lw=1.0, alpha=0.85, label=name)
    ax.set_ylabel("V")
    ax.set_title("Panel 1 -- scalar baselines (vertical lines = jump events)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # Panel 2: signature lanes
    ax = axes[1]
    ax.plot(t_axis, V_true, "k-", lw=1.4, label="V_true (spot)")
    for j in jump_idx:
        ax.axvline(j, color="gray", alpha=0.3, lw=0.7)
    for name in ("ms_cum_stride_raw", "ms_cum_stride_bv_target", "ms_cum_stride_two_channel"):
        if name in V_hats:
            ax.plot(t_axis, V_hats[name], lw=1.2, alpha=0.9, label=name)
    ax.set_ylabel("V")
    ax.set_title("Panel 2 -- signature lanes (same sig state; different supervised target)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # Panel 3: proxy-channel diagnostics for the two-channel lane
    ax = axes[2]
    ax.plot(t_axis, V_hats["ms_cum_stride_two_channel"], lw=1.2,
            label="cont. channel V_hat (two-channel)")
    ax.plot(t_axis, bv_sm, lw=1.0, alpha=0.75,
            label="BV smoothed (halflife=21d)")
    # jump channel on secondary axis
    ax2 = ax.twinx()
    ax2.plot(t_axis, V_jump, "r-", lw=1.0, alpha=0.6,
             label="jump channel V_hat_jump (right)")
    for j in jump_idx:
        ax.axvline(j, color="gray", alpha=0.3, lw=0.7)
    ax.set_xlabel("step t")
    ax.set_ylabel("cont-channel V / BV")
    ax2.set_ylabel("jump channel V")
    ax.set_title("Panel 3 -- two-channel diagnostics: continuous vs BV, jump channel near events")
    # combined legend
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Bates signature-derived proxy channels (dt={dt:.5f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    bates = BatesConfig()
    print("=" * 108)
    print("BENCHMARK FOLLOW-UP:  signature-derived proxy channels on Bates jumps")
    print("  Same Heston block, same jump block as `study_bates_signature_filters.py`.")
    print(f"  lambda_j = {bates.lambda_j}/yr, mu_j = {bates.mu_j}, sigma_j = {bates.sigma_j}")
    print(f"  dt = {bates.dt:.5f} (daily cadence).  Signature lanes share strides (1, 5, 20)d.")
    print("  ONLY the supervised target changes between signature lanes.")
    print("=" * 108)

    n_seeds = 40
    T = 400
    warmup = 30
    print(f"STUDY  --  n_seeds={n_seeds}, T={T}, warm-up={warmup}")
    results, channel_diag, trace = run_proxy_study(
        bates, n_seeds=n_seeds, T=T, warmup=warmup, base_seed=15_000,
    )
    print()
    _print_table(results)
    _print_channel_diag(channel_diag)

    # Figure
    plot_path = os.path.join(HERE, "study_bates_signature_proxy_channels.png")
    try:
        _plot(trace, out_path=plot_path, dt=bates.dt)
        print(f"Saved figure: {plot_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")

    # Direct answers
    def corr(lane: str, key: str = "spot_V_all") -> float:
        entry = results.get(lane, {}).get(key, None)
        return float(entry[1]) if entry is not None else float("nan")

    raw = corr("ms_cum_stride_raw")
    bv_t = corr("ms_cum_stride_bv_target")
    tc   = corr("ms_cum_stride_two_channel")
    best_sig_proxy = max(
        [("bv_target", bv_t), ("two_channel", tc)],
        key=lambda x: x[1] if np.isfinite(x[1]) else -np.inf,
    )
    winsor = corr("winsor_ewma")
    raw_jump = corr("ms_cum_stride_raw", "spot_V_jump")
    bv_t_jump = corr("ms_cum_stride_bv_target", "spot_V_jump")
    tc_jump = corr("ms_cum_stride_two_channel", "spot_V_jump")
    best_sig_proxy_jump = max(bv_t_jump, tc_jump) if np.isfinite(bv_t_jump) or np.isfinite(tc_jump) else float("nan")

    jump_gain = best_sig_proxy_jump - raw_jump
    gap_raw_to_winsor = winsor - raw
    gap_closed = best_sig_proxy[1] - raw if gap_raw_to_winsor > 0 else 0.0
    half_gap = 0.5 * gap_raw_to_winsor

    print("=" * 108)
    print("DIRECT ANSWERS")
    print("=" * 108)
    print(f"  ms_cum_stride_raw          corr spot V (all)  = {raw:+.4f}   (jump) = {raw_jump:+.4f}")
    print(f"  ms_cum_stride_bv_target    corr spot V (all)  = {bv_t:+.4f}   (jump) = {bv_t_jump:+.4f}")
    print(f"  ms_cum_stride_two_channel  corr spot V (all)  = {tc:+.4f}   (jump) = {tc_jump:+.4f}")
    print(f"  winsor_ewma                corr spot V (all)  = {winsor:+.4f}")
    print()
    print(f"  best signature-proxy lane  : {best_sig_proxy[0]} (all corr = {best_sig_proxy[1]:+.4f})")
    print(f"  jump-adjacent gain vs raw  : {jump_gain:+.4f}   (bar for recovery: > +0.10)")
    print(f"  raw -> winsor gap (all)    : {gap_raw_to_winsor:+.4f}")
    print(f"  proxy lane gap closure     : {gap_closed:+.4f}   (bar for recovery: > {half_gap:+.4f})")

    if jump_gain > 0.10 and gap_closed > half_gap:
        verdict = ("MEANINGFUL RECOVERY: signature-proxy lane beats ms_cum_stride_raw by "
                   ">+0.10 corr on jump-adjacent AND closes >= half the raw->winsor gap.  "
                   "Bates failure was primarily a TARGET problem.")
    elif jump_gain > 0.10:
        verdict = ("PARTIAL RECOVERY (jump-adjacent only): target fix closes a real part of "
                   "the jump-window gap but does NOT close half the raw->winsor gap overall. "
                   "Target matters; representation may also matter.")
    elif gap_closed > half_gap:
        verdict = ("PARTIAL RECOVERY (overall only): target fix closes overall gap but "
                   "jump-adjacent gain < 0.10.  Probably a mix of target + scale effects.")
    else:
        verdict = ("NO RECOVERY: target swap does not materially close the Bates gap.  "
                   "At this config, signature REPRESENTATION is also a bottleneck.")
    print(f"  Verdict                    : {verdict}")
    print("=" * 108)


if __name__ == "__main__":
    main()
