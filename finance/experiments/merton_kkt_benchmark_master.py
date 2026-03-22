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
from merton_kronic_kernel_tensor import KernelMertonController, analytical_hedging_demand, HestonMertonEnv
from merton_kronic_bilinear import CQKRONICMerton

def run_master_comparison(n_paths=1000, n_mc=100, horizon=10):
    rho_list = [0.0, -0.3, -0.5, -0.7, -0.9]
    V_test = 0.04
    
    # Results storage: rho -> {method: pi*}
    results = {r: {} for r in rho_list}
    
    print("=" * 85)
    print(f"MASTER BENCHMARK: KKT vs BILINEAR vs THEORY (N={n_paths}, MC={n_mc}, H={horizon})")
    print("=" * 85)
    print(f"{'rho':>5s} | {'Theory':>8s} | {'KKT':>8s} | {'Bilinear':>8s} | {'Myopic':>8s}")
    print("-" * 85)

    for rho in rho_list:
        # 1. Env and Analytical
        env = HestonMertonEnv(rho=rho)
        theory_hedge = analytical_hedging_demand(env)
        myopic      = env.merton_optimal(V_test)
        
        # 2. RUN KKT (Kernel Nystrom)
        # We use identical seeds for fair comparison if possible
        np.random.seed(42); torch.manual_seed(42)
        kkt_ctrl = KernelMertonController(env, horizon=horizon, n_landmarks=60)
        # Use our best-found parameters
        X0, XT, pi_a, U_T = kkt_ctrl.generate_training_data(n_paths=n_paths, n_mc=n_mc)
        kkt_ctrl.train(X0, XT, pi_a, U_T)
        pi_kkt, _, _, _ = kkt_ctrl.find_optimal_pi([[0.0, V_test]] * 3)
        
        # 3. RUN BILINEAR (Signatures)
        np.random.seed(42); torch.manual_seed(42)
        bilinear_ctrl = CQKRONICMerton(env, depth=3, mode='transfer')
        psi_X, psi_Y, pi_b, U_Y = bilinear_ctrl.generate_training_data(n_paths=200, n_steps=20)
        bilinear_ctrl.train(psi_X, psi_Y, pi_b, U_Y)
        # Bilinear optimal pi search
        pi_bilinear, _, _, _ = bilinear_ctrl.find_optimal_pi([[0.0, V_test]] * 3)
        
        # Store (pi - myopic) to get the hedging demand delta
        results[rho]['Theory']   = theory_hedge
        results[rho]['KKT']      = pi_kkt - myopic
        results[rho]['Bilinear'] = pi_bilinear - myopic
        results[rho]['Myopic']   = 0.0
        
        print(f"{rho:+.1f} | {theory_hedge:+.4f} | {results[rho]['KKT']:+.4f} | {results[rho]['Bilinear']:+.4f} | {0.0:+.4f}")

        # Save partial plot
        rhos_partial = np.array([r for r in results if 'KKT' in results[r]])
        if len(rhos_partial) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(rhos_partial, [results[r]['Theory'] for r in rhos_partial], 'k-', label='Theory (Analytical)', lw=2)
            plt.scatter(rhos_partial, [results[r]['KKT'] for r in rhos_partial], color='blue', s=80, label='KKT (Nystrom Kernel)', alpha=0.7)
            plt.scatter(rhos_partial, [results[r]['Bilinear'] for r in rhos_partial], color='red', marker='x', s=80, label='Bilinear (Signatures)', alpha=0.7)
            plt.title(f"Intertemporal Hedging Demand (V={V_test}) - Partial")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig("finance/experiments/master_benchmark_plot_partial.png")
            plt.close()

    # Plotting
    rhos = np.array(rho_list)
    plt.figure(figsize=(10, 6))
    plt.plot(rhos, [results[r]['Theory'] for r in rho_list], 'k-', label='Theory (Analytical)', lw=2)
    plt.scatter(rhos, [results[r]['KKT'] for r in rho_list], color='blue', s=80, label='KKT (Nystrom Kernel)', alpha=0.7)
    plt.scatter(rhos, [results[r]['Bilinear'] for r in rho_list], color='red', marker='x', s=80, label='Bilinear (Signatures)', alpha=0.7)
    
    plt.title(f"Intertemporal Hedging Demand Comparison (V={V_test})")
    plt.xlabel("Correlation (rho)")
    plt.ylabel("Hedging Demand Delta (pi* - pi_myopic)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("finance/experiments/master_benchmark_plot.png")
    print("\n[Benchmark] Plot saved to finance/experiments/master_benchmark_plot.png")

if __name__ == "__main__":
    run_master_comparison()
