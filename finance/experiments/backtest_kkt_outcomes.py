r"""
Backtesting Wealth Outcomes & Allocations (FIXED)
=================================================
Performs long-horizon simulations to test the KKT framework in practice.
Compares:
1. KKT (Full-Kernel Momentum)
2. Myopic Baseline

Environments:
- HestonMerton (Standard)
- CEV Dynamics (Non-linear Vol)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from merton_kronic_bilinear import HestonMertonEnv, CEVEnv
from merton_kronic_kernel_tensor import CrossValidatedKKT

def run_backtest(env_name="HestonMerton", rho=-0.7, n_paths=100, horizon=60):
    """
    horizon: Number of steps to simulate out-of-sample.
    """
    dt = 1/252
    if env_name == "HestonMerton":
        env = HestonMertonEnv(rho=rho, gamma=3.0)
        v0 = 0.04
    elif env_name == "CEV":
        env = CEVEnv(alpha=0.5, sigma=0.2, gamma=3.0)
        v0 = 0.04 # S=1 -> vol=0.2 -> var=0.04
    else:
        raise ValueError("Invalid Environment")

    print(f"\n[Backtest] Environment: {env_name} | Rho: {rho if env_name=='HestonMerton' else 'N/A'} | Horizon: {horizon} days")
    
    # 1. Train the KKT using Empirical Risk Minimization (Robust Shrinkage)
    ctrl = CrossValidatedKKT(env, mx=60, npi=15, switching_penalty=0.01)
    ctrl.train_with_sanity_check(n_train=1500, n_val=200)
    
    # 2. Out-of-sample Simulation
    n_sim_paths = 50
    wealth_kkt = np.ones((n_sim_paths, horizon + 1))
    wealth_myopic = np.ones((n_sim_paths, horizon + 1))
    pi_kkt = np.zeros((n_sim_paths, horizon))
    pi_myopic = np.zeros((n_sim_paths, horizon))
    
    for i in range(n_sim_paths):
        # Initial states
        lw_kkt, v_kkt = 0.0, v0
        lw_myo, v_myo = 0.0, v0
        
        # Histories for signature extraction [State: logW, V, pi_prev]
        hist_kkt = [(0, v0, 0.5)] * 3
        pi_prev_kkt = 0.5
        
        for t in range(horizon):
            # Decide actions
            p_kkt = ctrl.find_optimal_pi(hist_kkt)
            p_myo = env.merton_optimal(v_myo)
            
            pi_kkt[i, t] = p_kkt
            pi_myopic[i, t] = p_myo
            
            # Step environments with paired noise
            z1 = np.random.randn()
            z2 = getattr(env, 'rho', 0.0) * z1 + np.sqrt(1 - getattr(env, 'rho', 0.0)**2) * np.random.randn()
            
            lw_kkt, v_kkt = env.step_explicit(lw_kkt, v_kkt, p_kkt, z1, z2, dt=dt)
            lw_myo, v_myo = env.step_explicit(lw_myo, v_myo, p_myo, z1, z2, dt=dt)
            
            wealth_kkt[i, t+1] = np.exp(lw_kkt)
            wealth_myopic[i, t+1] = np.exp(lw_myo)
            
            # Update history
            hist_kkt.append((lw_kkt, v_kkt, pi_prev_kkt))
            hist_kkt = hist_kkt[1:] 
            pi_prev_kkt = p_kkt

    # 3. Calculate Performance Metrics
    def get_metrics(w_paths):
        rets = np.diff(w_paths, axis=1) / w_paths[:, :-1]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)
        end_wealth = w_paths[:, -1]
        return np.mean(sharpe), np.mean(end_wealth), np.std(end_wealth)

    s_kkt, w_kkt_m, w_kkt_s = get_metrics(wealth_kkt)
    s_myo, w_myo_m, w_myo_s = get_metrics(wealth_myopic)
    
    print("-" * 40)
    print(f"KKT: Sharpe={s_kkt:.4f}, FinalWealth={w_kkt_m:.4f} (std={w_kkt_s:.4f})")
    print(f"Myopic: Sharpe={s_myo:.4f}, FinalWealth={w_myo_m:.4f} (std={w_myo_s:.4f})")
    print("-" * 40)

    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    t_axis = np.arange(horizon + 1)
    # Plot first path
    ax1.plot(t_axis, wealth_kkt[0], 'b-', label='KKT Wealth', alpha=0.8)
    ax1.plot(t_axis, wealth_myopic[0], 'r--', label='Myopic Wealth', alpha=0.8)
    ax1.set_title(f"Wealth Trajectories ({env_name})")
    ax1.set_ylabel("Wealth")
    ax1.legend(); ax1.grid(True, alpha=0.2)

    ax2.plot(np.arange(horizon), pi_kkt[0], 'b-', label='KKT Allocation (pi)')
    ax2.plot(np.arange(horizon), pi_myopic[0], 'r--', label='Myopic Allocation')
    ax2.set_title("Allocation Weights over Time")
    ax2.set_xlabel("Steps (Days)"); ax2.set_ylabel("pi")
    ax2.legend(); ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plot_name = f"finance/experiments/backtest_fixed_{env_name}_{rho}.png"
    plt.savefig(plot_name)
    print(f"[Done] Result plot saved as {plot_name}")

if __name__ == "__main__":
    os.makedirs("finance/experiments", exist_ok=True)
    run_backtest(env_name="HestonMerton", rho=-0.7, horizon=60)
    run_backtest(env_name="CEV", horizon=60)
