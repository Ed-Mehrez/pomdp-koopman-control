r"""
BENCHMARK STUDY: Bates lambda-sweep (Part A of the Bates follow-up).

Why this study
--------------
The prior Bates study at `lambda_j = 30/yr` showed:
  - raw `r^2/dt` target for the signature lane fails near jumps
  - a BV-style target (per-step `(pi/2)*|r_{t-1}|*|r_t|/dt`) recovers
    ~58% of the raw->winsor gap and clears the pre-registered bar
  - a residual gap to `winsor_ewma` remains

Before deciding whether that residual is a REPRESENTATION problem (the
motivation for Marcus-style extensions), we first characterize how the
two derived gaps behave under a jump-intensity sweep.  If
  target_recovery_gap = bv_target - raw
grows sharply with lambda and
  residual_gap = winsor - bv_target
also grows sharply, the representation argument becomes compelling.
If the residual stays small/flat, Marcus is lower priority than
expected.

Scope
-----
- Sweep: lambda_j in {10, 30, 60}/yr.
- All other Bates params fixed (mu_j, sigma_j, Heston block, dt).
- Lanes:
    scalars     : rv_ewma, bv_ewma, winsor_ewma
    signatures  : ms_cum_stride_raw, ms_cum_stride_bv_target
- Metrics per lambda:
    corr(all), corr(jump-adjacent), corr(calm), RMSE(all)
- Derived gaps per lambda:
    target recovery  = bv_target - raw           (all & jump-adjacent)
    residual to winsor = winsor - bv_target       (all & jump-adjacent)

Pre-registered interpretation
-----------------------------
1. target_recovery grows with lambda  -> target mismatch dominates more
    as jumps intensify.
2. residual_gap grows with lambda     -> genuine representation bottleneck.
3. residual_gap stays flat/small      -> target fix does most of the work;
    Marcus is lower priority than expected.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Tuple

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
    target_raw_r2,
    target_bv_per_step,
)


# ==========================================================================
# Lane factories
# ==========================================================================


def make_factories(bates: BatesConfig) -> Dict[str, Callable[[], object]]:
    sig_cfg = CumStrideSigConfig(
        strides=(1, 5, 20),
        kappa=bates.kappa, theta=bates.theta, xi=bates.xi, dt=bates.dt,
    )
    halflife = 21.0
    return {
        "rv_ewma":                 lambda: EWMAr2(halflife_steps=halflife, dt=bates.dt),
        "bv_ewma":                 lambda: EWMABipower(halflife_steps=halflife, dt=bates.dt),
        "winsor_ewma":             lambda: EWMAWinsorized(halflife_steps=halflife, dt=bates.dt, k=4.0),
        "ms_cum_stride_raw":       lambda: CumStrideSigFilter(
            sig_cfg, target_fn_single=target_raw_r2, name="ms_cum_stride_raw",
        ),
        "ms_cum_stride_bv_target": lambda: CumStrideSigFilter(
            sig_cfg, target_fn_single=target_bv_per_step,
            name="ms_cum_stride_bv_target",
        ),
    }


# ==========================================================================
# Rollout / pooled stats
# ==========================================================================


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
        for name, est in lanes.items():
            est.observe(r, bates.dt)
            V_hat[name][t] = est.V_hat()
    return {
        "V_true_pre": sim["V"],
        "jump":       sim["jump"],
        "V_hat":      V_hat,
    }


def _pooled(pred_list, tgt_list, mask_list=None, warmup=30) -> Tuple[float, float, int]:
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
) -> Dict[float, Dict[str, Dict[str, Tuple[float, float, int]]]]:
    out: Dict[float, Dict[str, Dict[str, Tuple[float, float, int]]]] = {}
    for lam in lambdas:
        bates = replace(bates_base, lambda_j=float(lam))
        factories = make_factories(bates)
        lane_names = list(factories.keys())
        hat: Dict[str, List[np.ndarray]] = {n: [] for n in lane_names}
        V_list: List[np.ndarray] = []
        jump_list: List[np.ndarray] = []
        v0_rng = np.random.RandomState(base_seed + int(lam))
        for k in range(n_seeds):
            V0 = float(v0_rng.uniform(0.02, 0.08))
            r_out = _rollout(bates, factories, T, V0, base_seed + int(lam) * 1000 + k)
            for name in lane_names:
                hat[name].append(r_out["V_hat"][name])
            V_list.append(r_out["V_true_pre"])
            jump_list.append(r_out["jump"])
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
    return out


# ==========================================================================
# Reporting
# ==========================================================================


def _print_per_lambda(results) -> None:
    lane_names = list(next(iter(results.values())).keys())
    for lam, lane_res in sorted(results.items()):
        print(f"lambda_j = {lam:.0f} / yr")
        print("-" * 96)
        print(f"  {'lane':26s} | {'stat':>5s} | {'corr(all)':>10s} | "
              f"{'corr(jump)':>10s} | {'corr(calm)':>10s} | {'RMSE(all)':>10s}")
        print("-" * 96)
        for name in lane_names:
            r_all, c_all, _ = lane_res[name]["all"]
            r_j,   c_j,   _ = lane_res[name]["jump"]
            r_c,   c_c,   _ = lane_res[name]["calm"]
            print(f"  {name:26s} | {'corr':>5s} | {c_all:+.4f}    | "
                  f"{c_j:+.4f}    | {c_c:+.4f}    | {r_all:.4f}")
        print()


def _print_derived_gaps(results) -> None:
    print("DERIVED GAPS per lambda")
    print("  target_recovery = corr(ms_cum_stride_bv_target) - corr(ms_cum_stride_raw)")
    print("  residual_gap    = corr(winsor_ewma) - corr(ms_cum_stride_bv_target)")
    print("-" * 96)
    print(f"  {'lambda':>7s} | {'subset':>6s} | {'target_recovery':>17s} | {'residual_gap':>14s}")
    print("-" * 96)
    for lam, lane_res in sorted(results.items()):
        for subset in ("all", "jump"):
            c_raw  = lane_res["ms_cum_stride_raw"][subset][1]
            c_bv   = lane_res["ms_cum_stride_bv_target"][subset][1]
            c_wins = lane_res["winsor_ewma"][subset][1]
            tr = c_bv - c_raw
            rg = c_wins - c_bv
            print(f"  {lam:7.0f} | {subset:>6s} | {tr:+.4f}            | {rg:+.4f}")
    print()


def _plot_sweep(results, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = sorted(results.keys())
    lane_names = list(next(iter(results.values())).keys())
    palette = {
        "rv_ewma":                 "tab:olive",
        "bv_ewma":                 "tab:purple",
        "winsor_ewma":             "tab:green",
        "ms_cum_stride_raw":       "tab:red",
        "ms_cum_stride_bv_target": "tab:cyan",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # Panel 1: corr(all)
    ax = axes[0]
    for name in lane_names:
        y = [results[l][name]["all"][1] for l in lams]
        ax.plot(lams, y, "o-", color=palette.get(name), label=name, lw=1.5)
    ax.set_xlabel("lambda_j (jumps / year)")
    ax.set_ylabel("corr(V_hat, V_true)  (all samples)")
    ax.set_title("Corr, all samples")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")

    # Panel 2: corr(jump-adjacent)
    ax = axes[1]
    for name in lane_names:
        y = [results[l][name]["jump"][1] for l in lams]
        ax.plot(lams, y, "o-", color=palette.get(name), label=name, lw=1.5)
    ax.set_xlabel("lambda_j")
    ax.set_ylabel("corr (jump-adjacent, +/- 5 bars)")
    ax.set_title("Corr, jump-adjacent subset")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")

    # Panel 3: residual gap winsor - bv_target, all and jump
    ax = axes[2]
    y_all  = [results[l]["winsor_ewma"]["all"][1]  - results[l]["ms_cum_stride_bv_target"]["all"][1]
              for l in lams]
    y_jump = [results[l]["winsor_ewma"]["jump"][1] - results[l]["ms_cum_stride_bv_target"]["jump"][1]
              for l in lams]
    ax.plot(lams, y_all, "o-",  color="tab:orange", label="residual (all)", lw=1.8)
    ax.plot(lams, y_jump, "s--", color="tab:red",    label="residual (jump)", lw=1.8)
    ax.axhline(0.0, color="gray", lw=0.7)
    ax.set_xlabel("lambda_j")
    ax.set_ylabel("winsor_ewma - ms_cum_stride_bv_target  (corr)")
    ax.set_title("Residual gap to winsor")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Bates lambda sweep: Part A (no Marcus-inspired lanes)", fontsize=12,
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
    print("=" * 96)
    print("BENCHMARK STUDY  --  Bates lambda sweep (Part A)")
    print(f"  lambda sweep: {lambdas}  /yr")
    print(f"  n_seeds={n_seeds}, T={T}, warm-up={warmup}, dt={bates_base.dt:.5f}")
    print(f"  mu_j={bates_base.mu_j}, sigma_j={bates_base.sigma_j}")
    print(f"  Heston: kappa={bates_base.kappa}, theta={bates_base.theta}, "
          f"xi={bates_base.xi}, rho={bates_base.rho}")
    print("=" * 96)

    results = run_sweep(
        bates_base=bates_base, lambdas=lambdas,
        n_seeds=n_seeds, T=T, warmup=warmup, base_seed=16_000,
    )
    _print_per_lambda(results)
    _print_derived_gaps(results)

    out_path = os.path.join(HERE, "study_bates_lambda_sweep.png")
    try:
        _plot_sweep(results, out_path)
        print(f"Saved figure: {out_path}")
    except Exception as e:
        print(f"Figure skipped: {e}")

    # Summary interpretation (arithmetic, no narrative spin)
    lams = sorted(results.keys())
    def lookup(lam, lane, subset):
        return results[lam][lane][subset][1]
    print()
    print("=" * 96)
    print("PART A SUMMARY (monotonic tests)")
    print("=" * 96)
    tr_all  = [lookup(l, "ms_cum_stride_bv_target", "all")  - lookup(l, "ms_cum_stride_raw", "all")  for l in lams]
    tr_jump = [lookup(l, "ms_cum_stride_bv_target", "jump") - lookup(l, "ms_cum_stride_raw", "jump") for l in lams]
    rg_all  = [lookup(l, "winsor_ewma", "all")  - lookup(l, "ms_cum_stride_bv_target", "all")  for l in lams]
    rg_jump = [lookup(l, "winsor_ewma", "jump") - lookup(l, "ms_cum_stride_bv_target", "jump") for l in lams]
    print(f"  target_recovery (all)  across lambdas : {[f'{v:+.4f}' for v in tr_all]}")
    print(f"  target_recovery (jump) across lambdas : {[f'{v:+.4f}' for v in tr_jump]}")
    print(f"  residual_gap     (all)  across lambdas : {[f'{v:+.4f}' for v in rg_all]}")
    print(f"  residual_gap     (jump) across lambdas : {[f'{v:+.4f}' for v in rg_jump]}")
    def grow(v):
        return "growing" if v[-1] > v[0] + 0.02 else ("shrinking" if v[-1] < v[0] - 0.02 else "flat")
    print(f"  target_recovery (all)  trend: {grow(tr_all)}")
    print(f"  target_recovery (jump) trend: {grow(tr_jump)}")
    print(f"  residual_gap    (all)  trend: {grow(rg_all)}")
    print(f"  residual_gap    (jump) trend: {grow(rg_jump)}")


if __name__ == "__main__":
    main()
