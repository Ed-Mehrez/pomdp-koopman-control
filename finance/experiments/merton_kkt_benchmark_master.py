r"""
Master Benchmark: KKT vs. Bilinear vs. Analytical Theory
========================================================
Compares the intertemporal hedging demand across:
1. Analytical Heston (Exact steady-state)
2. Bilinear CQ-KRONIC (Parametric / Signatures)
3. Kernel Koopman Tensor (Non-parametric / Nystrom)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

# paths
sys.path.insert(0, os.path.dirname(__file__))
from merton_kronic_kernel_tensor import KernelMertonController, HestonMertonEnv
from merton_kronic_bilinear import CQKRONICMerton
from merton_theory import canonical_state_history, stationary_heston_crra_theory


THEORY_DEVIATION_FLAG = 0.10

def _validate_rows(rows):
    theory = np.array([row["theory_delta"] for row in rows], dtype=float)
    if not np.all(np.isfinite(theory)):
        raise RuntimeError("Analytical theory produced non-finite values.")
    if np.ptp(theory) <= 1e-6:
        raise RuntimeError(
            "Analytical theory column is flat/degenerate across rho; refusing to emit a benchmark plot."
        )

    bad_rows = []
    for row in rows:
        numeric_fields = [
            row["myopic_pi"],
            row["kkt_pi"],
            row["kkt_delta"],
            row["bilinear_pi"],
            row["bilinear_delta"],
            row["kkt_operator_r2"],
            row["kkt_utility_r2"],
            row["bilinear_operator_r2"],
            row["bilinear_utility_r2"],
        ]
        if not np.all(np.isfinite(numeric_fields)):
            bad_rows.append(row["rho"])
    if bad_rows:
        rho_fmt = ", ".join(f"{rho:+.1f}" for rho in bad_rows)
        raise RuntimeError(f"Non-finite benchmark row(s) for rho={rho_fmt}.")


def run_master_comparison(n_paths=1000, n_mc=100, horizon=10, v_test=0.04):
    rho_list = [0.0, -0.3, -0.5, -0.7, -0.9]
    rows = []
    
    print("=" * 85)
    print(f"MASTER BENCHMARK AUDIT: KKT vs BILINEAR vs THEORY (N={n_paths}, MC={n_mc}, H={horizon})")
    print("=" * 85)
    print(
        f"{'rho':>5s} | {'Theory':>8s} | {'Myopic':>8s} | {'KKT':>8s} | "
        f"{'dKKT':>8s} | {'Bilin':>8s} | {'dBil':>8s} | {'OpR2':>6s} | "
        f"{'UR2':>6s} | {'BQ':>6s} | {'BU':>6s} | {'Flags':>10s}"
    )
    print("-" * 85)

    for rho in rho_list:
        env = HestonMertonEnv(rho=rho, gamma=3.0)
        theory = stationary_heston_crra_theory(env, v_test)
        myopic = theory.myopic_pi
        eval_history = canonical_state_history(v_test, myopic)
        
        np.random.seed(42); torch.manual_seed(42)
        kkt_ctrl = KernelMertonController(env, horizon=horizon, mx=60)
        X0, XT, pi_a, U_T = kkt_ctrl.generate_training_data(n_paths=n_paths, n_mc=n_mc)
        kkt_ctrl.train(X0, XT, pi_a, U_T)
        kkt_eval = kkt_ctrl.evaluate_state(eval_history)
        
        np.random.seed(42); torch.manual_seed(42)
        bilinear_ctrl = CQKRONICMerton(env, depth=3, mode='transfer')
        psi_X, psi_Y, pi_b, U_Y = bilinear_ctrl.generate_training_data(n_paths=200, n_steps=20)
        bilinear_ctrl.train(psi_X, psi_Y, pi_b, U_Y)
        bilinear_eval = bilinear_ctrl.evaluate_state(eval_history)

        flags = []
        if not bilinear_eval["concave"]:
            flags.append("BILIN_GRID")
        if abs(kkt_eval["pi_opt"] - theory.optimal_pi) > THEORY_DEVIATION_FLAG:
            flags.append("KKT_FAR")
        if abs(bilinear_eval["pi_opt"] - theory.optimal_pi) > THEORY_DEVIATION_FLAG:
            flags.append("BILIN_FAR")
        flag_text = ",".join(flags) if flags else "OK"

        row = {
            "rho": rho,
            "theory_delta": theory.hedging_demand,
            "myopic_pi": myopic,
            "theory_pi": theory.optimal_pi,
            "kkt_pi": kkt_eval["pi_opt"],
            "kkt_delta": kkt_eval["pi_opt"] - myopic,
            "bilinear_pi": bilinear_eval["pi_opt"],
            "bilinear_delta": bilinear_eval["pi_opt"] - myopic,
            "kkt_operator_r2": kkt_ctrl.operator_r2,
            "kkt_utility_r2": kkt_ctrl.utility_r2,
            "bilinear_operator_r2": bilinear_ctrl.koopman_fit_r2,
            "bilinear_utility_r2": bilinear_ctrl.utility_fit_r2,
            "flags": flag_text,
        }
        rows.append(row)

        print(
            f"{rho:+.1f} | {row['theory_delta']:+.4f} | {row['myopic_pi']:+.4f} | "
            f"{row['kkt_pi']:+.4f} | {row['kkt_delta']:+.4f} | {row['bilinear_pi']:+.4f} | "
            f"{row['bilinear_delta']:+.4f} | {row['kkt_operator_r2']:.4f} | "
            f"{row['kkt_utility_r2']:.4f} | {row['bilinear_operator_r2']:.4f} | "
            f"{row['bilinear_utility_r2']:.4f} | {row['flags']}"
        )

    _validate_rows(rows)

    rhos = np.array([row["rho"] for row in rows], dtype=float)
    plt.figure(figsize=(10, 6))
    plt.plot(rhos, [row["theory_delta"] for row in rows], 'k-', label='Theory (Analytical)', lw=2)
    plt.scatter(rhos, [row["kkt_delta"] for row in rows], color='blue', s=80, label='KKT (Nystrom Kernel)', alpha=0.7)
    plt.scatter(rhos, [row["bilinear_delta"] for row in rows], color='red', marker='x', s=80, label='Bilinear (Signatures)', alpha=0.7)
    
    plt.title(f"Intertemporal Hedging Demand Comparison (V={v_test})")
    plt.xlabel("Correlation (rho)")
    plt.ylabel("Hedging Demand Delta (pi* - pi_myopic)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("finance/experiments/master_benchmark_plot.png")
    print("\n[Benchmark] Plot saved to finance/experiments/master_benchmark_plot.png")

if __name__ == "__main__":
    run_master_comparison()
