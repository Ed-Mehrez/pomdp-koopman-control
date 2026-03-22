r"""
Master Benchmark Simulation for KRONIC Controllers
==================================================

This script runs multi-year temporal simulations benchmarking three strategies
on paired stochastic volatility (Heston) paths over MULTIPLE seeds:

1. Analytical Merton (Theoretical Baseline)
2. Online Signature KRONIC (Lead-Lag Signatory + Exact RLS Correction)
3. Offline Signature KRONIC (Lead-Lag Signatory + Fixed Global Mapping)

We observe average PnL convergence across parallel universes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import torch
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from merton_kronic_online import OnlineKRONICMerton, HestonMertonEnv
from merton_kronic_signatures import SignatureKRONICMerton

def explicit_step(env, state, pi_action, z1, z2, dt):
    """Manually advances the state using provided random normal shocks."""
    log_W, pi, v = state
    dB = np.sqrt(dt) * z1
    dB_v = np.sqrt(dt) * z2
    
    v_safe = max(v, 1e-8)
    drift_W = env.r + pi_action * (env.mu - env.r) - 0.5 * pi_action**2 * v_safe
    d_log_W = drift_W * dt + pi_action * np.sqrt(v_safe) * dB
    
    new_v = v + env.kappa * (env.theta - v_safe) * dt + env.xi * np.sqrt(v_safe) * dB_v
    new_v = max(new_v, 1e-8)
    
    return np.array([log_W + d_log_W, pi_action, new_v])


def run_temporal_simulation(T_years=5, seeds=[42, 100, 404]):
    print("=" * 60)
    print("Phase 3 & 4: Multi-Seed Temporal Benchmarking")
    print("=" * 60)
    
    env = HestonMertonEnv()
    dt = 1/252
    n_steps = int(T_years * 252)

    print("\n1. Initializing and Training Global Offline Models...")
    
    # --- Online KRONIC ---
    print("   -> Online Signature KRONIC...")
    online_kronic = OnlineKRONICMerton(env, n_eigs=30)
    psi_X_o, psi_Y_o, U_o = online_kronic.generate_safe_training_data()
    online_kronic.train_offline(psi_X_o, psi_Y_o, U_o)

    # --- Signature KRONIC ---
    print("   -> Offline Signature KRONIC...")
    sig_kronic = SignatureKRONICMerton(env)
    psi_X_s, psi_Y_s, U_s = sig_kronic.generate_training_data()
    sig_kronic.train(psi_X_s, psi_Y_s, U_s)

    # Result Trackers
    all_W_merton = []
    all_W_online = []
    all_W_sig = []
    
    time_axis = np.arange(n_steps) / 252.0

    print(f"\n2. Running Simulation across {len(seeds)} Random Seeds...")
    
    pi_candidates = np.linspace(0.1, 8.0, 80)
    
    time_axis = np.linspace(0, T_years, int(T_years * 252))
    all_W_merton = []
    all_W_online = []
    all_W_sig = []
    
    # Global Initial Portfolio conditions across all seeds
    V0 = 0.04
    W0 = 1.0
    pi0_sig = 0.1
    pi0_opt = 0.1
    for seed_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\n   === Seed {seed} ({seed_idx + 1}/{len(seeds)}) ===")
        
        # We clone the models so online RLS updates don't bleed across seeds
        import copy
        kronic_online_run = copy.deepcopy(online_kronic)
        
        z1_traj = np.random.randn(n_steps)
        z2_traj = env.rho * z1_traj + np.sqrt(1 - env.rho**2) * np.random.randn(n_steps)

        # Initialize states for each strategy
        # state = [log_W, pi, v]
        state_merton = np.array([np.log(W0), env.merton_optimal(V0), V0])
        state_online = np.array([np.log(W0), 0.1, V0]) # Initial pi for online/sig is arbitrary, will be updated
        state_sig = np.array([np.log(W0), 0.1, V0])
        
        # We need sliding histories for Lead-Lag Signatures
        # The initial pi for online/sig is arbitrary, as it will be immediately overwritten by the first optimal_pi call
        hist_online = deque([[np.log(W0), pi0_opt, V0]] * 3, maxlen=3)
        hist_sig = deque([[np.log(W0), pi0_sig, V0]] * 3, maxlen=3)
        # Merton sliding window
        hist_merton = deque([[np.log(W0), env.merton_optimal(V0), V0]] * 3, maxlen=3)
        
        W_merton_run = np.zeros(n_steps)
        W_online_run = np.zeros(n_steps)
        W_sig_run = np.zeros(n_steps)
        
        pi_merton_run = np.zeros(n_steps)
        pi_online_run = np.zeros(n_steps)
        pi_sig_run = np.zeros(n_steps)
        
        for t in range(n_steps):
            if t > 0 and t % 500 == 0:
                print(f"       Step {t} / {n_steps}...")
                
            V_curr = state_merton[2]
            
            # A. Optimal Actions
            pi_merton = env.merton_optimal(V_curr)
            pi_online, _ = kronic_online_run.find_optimal_pi(list(hist_online), pi_candidates)
            pi_sig, _ = sig_kronic.find_optimal_pi(list(hist_sig), pi_candidates)
            
            W_merton_run[t] = np.exp(state_merton[0])
            W_online_run[t] = np.exp(state_online[0])
            W_sig_run[t] = np.exp(state_sig[0])
            
            pi_merton_run[t] = pi_merton
            pi_online_run[t] = pi_online
            pi_sig_run[t] = pi_sig
            
            # B. Simulation Steps
            z1, z2 = z1_traj[t], z2_traj[t]
            
            # 1. Merton
            state_merton_next = explicit_step(env, state_merton, pi_merton, z1, z2, dt)
            hist_merton.append(list(state_merton_next))
            state_merton = state_merton_next
            
            # 2. Online Signature Tracker
            state_online_next = explicit_step(env, state_online, pi_online, z1, z2, dt)
            
            # Online tracking requires length-3 sliding windows!
            # List structure: [t-2, t-1, t] -> [t-1, t, t+1]
            p_t = np.array(list(hist_online))
            p_next = np.array(list(hist_online)[1:] + [list(state_online_next)])
            
            # Wealth Translation Invariance: center logW on current time
            shift_W = p_t[-1, 0]
            p_t[:, 0] -= shift_W
            p_next[:, 0] -= shift_W
            
            online_window_t = torch.tensor([p_t], dtype=torch.float64, device=kronic_online_run.device)
            online_window_next = torch.tensor([p_next], dtype=torch.float64, device=kronic_online_run.device)
            
            with torch.no_grad():
                psi_t = kronic_online_run.kgedmd.extractor.extract(online_window_t)[0].cpu().numpy()
                psi_next = kronic_online_run.kgedmd.extractor.extract(online_window_next)[0].cpu().numpy()
                
            kronic_online_run.kgedmd.update_online(psi_t, psi_next)
            
            hist_online.append(list(state_online_next))
            state_online = state_online_next
            
            # 3. Offline Global Signature
            state_sig_next = explicit_step(env, state_sig, pi_sig, z1, z2, dt)
            hist_sig.append(list(state_sig_next))
            state_sig = state_sig_next

        all_W_merton.append(W_merton_run)
        all_W_online.append(W_online_run)
        all_W_sig.append(W_sig_run)
        
        # Track the allocations for the first seed (for visualization of the path)
        if seed == seeds[0]:
            all_pi_merton = pi_merton_run.copy()
            all_pi_online = pi_online_run.copy()
            all_pi_sig = pi_sig_run.copy()
        
        print(f"       Terminal W (Merton) : {W_merton_run[-1]:.3f}")
        print(f"       Terminal W (Online) : {W_online_run[-1]:.3f}")
        print(f"       Terminal W (Offline): {W_sig_run[-1]:.3f}")

    print("\n4. Saving Plots...")
    
    # Compute Medians and Std Devs
    mean_merton = np.mean(all_W_merton, axis=0)
    std_merton = np.std(all_W_merton, axis=0)
    
    mean_online = np.mean(all_W_online, axis=0)
    std_online = np.std(all_W_online, axis=0)
    
    mean_sig = np.mean(all_W_sig, axis=0)
    std_sig = np.std(all_W_sig, axis=0)
    
    # --- Plot 1: Terminal Wealth ---
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, mean_merton, 'k--', label='Analytical Merton')
    plt.fill_between(time_axis, mean_merton - std_merton, mean_merton + std_merton, color='k', alpha=0.1)
    
    plt.plot(time_axis, mean_online, 'r-', label='Online Signature KRONIC (Adaptive)')
    plt.fill_between(time_axis, mean_online - std_online, mean_online + std_online, color='r', alpha=0.1)
    
    plt.plot(time_axis, mean_sig, 'b-', label='Offline Signature KRONIC (Global Base)')
    plt.fill_between(time_axis, mean_sig - std_sig, mean_sig + std_sig, color='b', alpha=0.1)
    
    plt.ylabel('Average Portfolio Wealth ($W_t$)')
    plt.xlabel('Time (Years)')
    plt.title(f'KRONIC PnL Multi-Seed Benchmark ({len(seeds)} Parallel Universes)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('finance/experiments/merton_kronic_simulation_master.png', dpi=150)
    print("Saved master benchmark plot to finance/experiments/merton_kronic_simulation_master.png")
    
    # --- Plot 2: Allocation Trajectories (Seed 0) ---
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, all_pi_merton, 'k--', alpha=0.7, label='Analytical Merton')
    plt.plot(time_axis, all_pi_online, 'r-', alpha=0.8, label='Online Signature KRONIC (Adaptive)')
    plt.plot(time_axis, all_pi_sig, 'b-', alpha=0.8, linewidth=2, label='Offline Signature KRONIC (Global Base)')
    
    plt.ylabel(r'Allocation to Risky Asset ($\pi_t$)')
    plt.xlabel('Time (Years)')
    plt.title(f'Dynamic Allocation Trajectories (Seed {seeds[0]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('finance/experiments/merton_kronic_allocations.png', dpi=150)
    print("Saved allocations plot to finance/experiments/merton_kronic_allocations.png")

if __name__ == "__main__":
    run_temporal_simulation(T_years=5, seeds=[42, 100, 404, 777, 999])
