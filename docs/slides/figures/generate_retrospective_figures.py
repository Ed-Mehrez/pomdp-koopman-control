r"""Generate three pedagogical figures for the retrospective Marp deck.

Outputs (saved alongside this script):
  fig_pomdp_loop.png            -- POMDP filter+control loop schematic
  fig_signature_qv.png          -- Heston path + cumulative QV vs cumulative Levy area
  fig_benchmark_summary.png     -- two-panel: (a) cross-benchmark EWMA-vs-signature
                                              (b) Bates lane landscape at lambda=30

No experiment is rerun.  The Bates panel uses numbers already produced
by `study_bates_target_gating_lambda_sweep.py` (base_seed=16_000, lambda=30).
The signature panel runs ONE short Heston path (T=500 steps) inline to
demonstrate that the lead-lag Levy area tracks cumulative QV.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.proof_of_concept.signature_features import RecurrentLeadLagLogSigMap


# ==========================================================================
# Figure 1: POMDP loop schematic
# ==========================================================================


def make_pomdp_loop():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    def box(x, y, w, h, label, color="#dfe9f7", edge="#2c3e50"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=11)

    def arrow(x1, y1, x2, y2, label=None, lab_y_off=0.18, color="#2c3e50"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.0, color=color)
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + lab_y_off, label,
                    ha="center", va="center", fontsize=10, color=color, style="italic")

    # World box: hidden state V_t and dynamics
    box(0.5, 3.5, 2.4, 1.4, r"hidden state $V_t$" + "\n(stoch. variance)", color="#fde9d9")
    # Observation
    box(3.6, 3.5, 2.2, 1.4, r"observation $r_t$" + "\n(realised return)", color="#e8f4dd")
    # Filter posterior
    box(6.4, 3.5, 3.0, 1.4, r"filter posterior $\hat{V}_t$" + "\n" + r"$=\mathbb{E}[V_t\mid r_{1:t}]$", color="#dfe9f7")
    # Decision / action
    box(6.4, 0.4, 3.0, 1.4, r"action $u_t$" + "\n(hedge / quote / etc.)", color="#f5d6e0")
    # Stage cost / reward
    box(0.5, 0.4, 2.4, 1.4, r"stage cost $c(V_t, u_t)$", color="#e6e0f0")

    # Arrows
    arrow(2.9, 4.2, 3.6, 4.2, "emit (noisy)")
    arrow(5.8, 4.2, 6.4, 4.2, "Bayes update")
    arrow(7.9, 3.5, 7.9, 1.8, "policy")
    arrow(6.4, 1.1, 2.9, 1.1, "decision affects payoff")
    arrow(0.7, 1.8, 0.7, 3.5, "advances dynamics")
    # Loop arrow back from V_t along the top
    a = FancyArrowPatch((1.7, 4.9), (1.7, 5.3),
                        arrowstyle="-", linewidth=1.0, color="#888")
    ax.add_patch(a)
    a = FancyArrowPatch((1.7, 5.3), (8.0, 5.3),
                        arrowstyle="-", linewidth=1.0, color="#888")
    ax.add_patch(a)
    a = FancyArrowPatch((8.0, 5.3), (8.0, 4.9),
                        arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#888")
    ax.add_patch(a)
    ax.text(4.85, 5.45, "next time step", ha="center", va="bottom",
            fontsize=10, style="italic", color="#666")

    ax.set_title(
        "Partially-observed control loop: latent variance is never seen directly.\n"
        "Filter forms a posterior; the policy acts on that posterior; the world "
        "evolves; repeat.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig_pomdp_loop.png", dpi=130)
    plt.close(fig)


# ==========================================================================
# Figure 2: signature / Levy area = QV intuition
# ==========================================================================


def make_signature_qv():
    rng = np.random.RandomState(7)
    T = 500
    dt = 1.0 / 252.0
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    V = np.zeros(T); r = np.zeros(T)
    V_prev = float(theta)
    sqrt_dt = np.sqrt(dt)
    for t in range(T):
        V[t] = V_prev
        z1 = rng.standard_normal()
        z2 = rho * z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.standard_normal()
        sV = np.sqrt(max(V_prev, 1e-8))
        r[t] = (-0.5 * V_prev) * dt + sV * sqrt_dt * z1
        V_new = V_prev + kappa * (theta - V_prev) * dt + xi * sV * sqrt_dt * z2
        V_prev = max(V_new, 1e-8)

    # Cumulative quadratic variation
    cumQV = np.cumsum(r ** 2)

    # Cumulative log-signature lead-lag with gamma=1.0; pull QV-Levy area
    sig_map = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
    QV_IDX_IN_L2 = 4
    cum_levy = np.zeros(T)
    for t in range(T):
        sig_map.update(np.array([dt, float(r[t])]))
        # 2 * Levy area between (ret_lead, ret_lag) ~= cumulative sum r^2
        cum_levy[t] = 2.0 * float(sig_map.l2[QV_IDX_IN_L2])

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(np.cumsum(r), color="#2c3e50", lw=1.0)
    ax.set_ylabel("log-price")
    ax.set_title("(1) Heston path: log-price (cumulative log-return)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(cumQV, color="#c0392b", lw=1.4, label=r"cumulative QV $= \sum_{s \leq t} r_s^2$")
    ax.set_ylabel("cumulative QV")
    ax.set_title("(2) What we want: cumulative quadratic variation = the integrated variance signal")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(cumQV, color="#c0392b", lw=1.4, label=r"cumulative QV (target)")
    ax.plot(cum_levy, color="#2980b9", lw=1.2, linestyle="--",
            label=r"$2\times$ lead-lag Levy area (signature, level 2)")
    ax.set_xlabel("step t  (daily bars, dt = 1/252)")
    ax.set_ylabel("variance proxy")
    ax.set_title(
        "(3) The signature picks this up automatically: the level-2 Levy area "
        "between ret$_{\\mathrm{lead}}$ and ret$_{\\mathrm{lag}}$ tracks QV without us telling it to."
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "fig_signature_qv.png", dpi=130)
    plt.close(fig)


# ==========================================================================
# Figure 3: benchmark summary  (two panels)
# ==========================================================================


def make_benchmark_summary():
    # Cross-benchmark headline numbers (corr spot V).  Sources:
    #   Daily Heston   : study_heston_multiscale_signature_filters.py
    #   5-min Heston   : study_heston_5min_signature_filters.py
    #   Bates @ lam=30 : study_bates_lambda_sweep.py + study_bates_target_gating_lambda_sweep.py
    benchmarks = ["Daily Heston", "5-min Heston", "Bates raw\n(λ=30)", "Bates BV target\n(λ=30)", "Bates soft gate\n(λ=30)"]
    best_scalar = [0.8229, 0.9668, 0.7479, 0.7479, 0.7479]      # winsor_ewma at Bates, EWMA at Heston
    best_sig    = [0.8149, 0.9705, 0.4352, 0.5768, 0.6430]      # ms_cum_stride family
    sig_label   = ["ms_cum_stride", "ms_cum_stride", "ms_cum_stride_raw",
                   "ms_cum_stride_bv_target", "ms_cum_stride_soft_gate"]

    # Bates lane landscape at lam=30 (same seed bank).
    bates_lanes = [
        ("rv_ewma",                      0.4923, "scalar"),
        ("bv_ewma",                      0.6678, "scalar (robust)"),
        ("winsor_ewma",                  0.7479, "scalar (robust)"),
        ("ms_cum_stride_raw",            0.4352, "signature"),
        ("ms_cum_stride_bv_target",      0.5768, "signature"),
        ("ms_cum_stride_hard_proxy",     0.6441, "signature"),
        ("ms_cum_stride_soft_gate",      0.6430, "signature"),
        ("ms_cum_stride_oracle_dejump",  0.6583, "oracle"),
        ("ms_cum_stride_oracle_latent",  0.7227, "ceiling"),
    ]
    # Family colors
    family_color = {"scalar": "#7f8c8d",
                    "scalar (robust)": "#27ae60",
                    "signature": "#2980b9",
                    "oracle": "#e67e22",
                    "ceiling": "#000000"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel 1: cross-benchmark grouped bars ---
    ax = axes[0]
    x = np.arange(len(benchmarks))
    width = 0.36
    bars1 = ax.bar(x - width / 2, best_scalar, width, label="best scalar / EWMA", color="#27ae60")
    bars2 = ax.bar(x + width / 2, best_sig, width, label="best signature lane", color="#2980b9")
    for xi_, h, lab in zip(x + width / 2, best_sig, sig_label):
        ax.text(xi_, h + 0.015, lab, rotation=45, ha="left", va="bottom", fontsize=7, color="#2980b9")
    for xi_, h in zip(x - width / 2, best_scalar):
        ax.text(xi_, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#27ae60")
    for xi_, h in zip(x + width / 2, best_sig):
        ax.text(xi_, h - 0.04, f"{h:.3f}", ha="center", va="top", fontsize=8, color="white", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("corr(V_hat, latent V)  pooled across seeds")
    ax.set_title("(A) Cross-benchmark: best handcrafted scalar vs best signature lane")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # --- Panel 2: Bates lane landscape at lam=30 ---
    ax = axes[1]
    names = [n for (n, _, _) in bates_lanes]
    vals  = [v for (_, v, _) in bates_lanes]
    fams  = [f for (_, _, f) in bates_lanes]
    colors = [family_color[f] for f in fams]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    # vertical line for winsor reference
    ax.axvline(0.7479, color="#27ae60", lw=1.2, linestyle="--", alpha=0.7)
    ax.text(0.7479, len(names) - 0.3, "  winsor_ewma", color="#27ae60", fontsize=8,
            ha="left", va="center")
    # vertical line for oracle_latent ceiling
    ax.axvline(0.7227, color="black", lw=1.2, linestyle=":", alpha=0.7)
    ax.text(0.7227, -0.4, "oracle_latent (ceiling)", color="black", fontsize=8,
            ha="center", va="bottom")
    ax.set_xlim(0.0, 0.85)
    ax.set_xlabel("corr(V_hat, latent V) at λ_j = 30 / yr")
    ax.set_title("(B) Bates lane landscape (same seed bank, λ=30)")
    # legend handles
    handles = [plt.Rectangle((0, 0), 1, 1, color=family_color[f]) for f in
               ["scalar", "scalar (robust)", "signature", "oracle", "ceiling"]]
    labels = ["scalar (raw)", "scalar (robust)", "signature", "oracle (target uses J_t)", "ceiling (target = V_true)"]
    ax.legend(handles, labels, loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Cross-benchmark summary  (filter-only; corr of V̂ against latent spot V)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig_benchmark_summary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    make_pomdp_loop()
    make_signature_qv()
    make_benchmark_summary()
    for name in ("fig_pomdp_loop.png", "fig_signature_qv.png", "fig_benchmark_summary.png"):
        path = HERE / name
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")
