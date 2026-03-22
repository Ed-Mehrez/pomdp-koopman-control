import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.finance.continuous_kyle import SigFilterMM, KoopmanSDREInsider

def run_experiment():
    np.random.seed(42)
    os.makedirs('examples/finance/results', exist_ok=True)
    
    # 1. Market Parameters
    T = 1.0
    dt = 0.005
    steps = int(T / dt)
    t_space = np.linspace(0, T, steps)
    
    P_0 = 100.0
    Sigma_0 = 100.0  # Prior variance of v. std_dev = 10.
    sigma_z = 1.0    # Noise trader volatility
    
    # In continuous Kyle, Kyle's lambda is structurally known
    lam = np.sqrt(Sigma_0) / (sigma_z * np.sqrt(T)) # = 10.0
    
    # 2. Market Maker Koopman Filter
    # In Kyle, Price is a martingale. It has NO drift.
    # z = [1, P_t - P_0]
    L_mm = np.array([[0.0, 0.0],
                     [0.0, 0.0]])
    # dP_t = lam * dY_t
    g_mm = np.array([0.0, lam])
    # P_t = [P_0, 1] @ z
    w_mm = np.array([P_0, 1.0])
    
    mm = SigFilterMM(L_mm, g_mm, w_mm)
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: CONDITIONAL ON A FIXED v = 110 
    # (Shows the Insider's Brownian Bridge behavior and Pricing Convergence)
    # -----------------------------------------------------------------
    print("Running Experiment 1: Conditional Paths (Fixed v=110)")
    v_fixed = 110.0
    num_paths = 50
    
    # Initialize insider 
    insider_fixed = KoopmanSDREInsider(L_mm, g_mm, w_mm, T=T)
    
    Y_paths_cond = np.zeros((num_paths, steps))
    P_paths_cond = np.zeros((num_paths, steps))
    
    for path_idx in tqdm(range(num_paths), desc="Conditional Paths"):
        z_t = np.array([1.0, 0.0])
        X_t, Y_t = 0.0, 0.0
        
        for i in range(steps):
            
            theta_t = insider_fixed.compute_exact_singular_rate(z_t, v_fixed, t_space[i])
            
            dX_t = theta_t * dt
            dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
            dY_t = dX_t + dZ_t
            
            z_next, P_t = mm.filter_step(z_t, dY_t, dt)
            
            Y_t += dY_t
            X_t += dX_t
            z_t = np.array(z_next)
            
            Y_paths_cond[path_idx, i] = Y_t
            P_paths_cond[path_idx, i] = P_t


    # -----------------------------------------------------------------
    # EXPERIMENT 2: UNCONDITIONAL (Random v)
    # (Shows the Market Maker's view: Y_t is standard Brownian Motion)
    # -----------------------------------------------------------------
    print("Running Experiment 2: Unconditional Paths (Random v)")
    num_paths_uncond = 10000 # With the new decoupled DRE, we can run 10k paths in seconds
    Y_paths_uncond = np.zeros((num_paths_uncond, steps))
    
    # We can reuse the same insider instance
    for path_idx in tqdm(range(num_paths_uncond), desc="Unconditional Paths"):
        v_rand = np.random.normal(P_0, np.sqrt(Sigma_0))
        
        z_t = np.array([1.0, 0.0])
        Y_t = 0.0
        
        for i in range(steps):
            theta_t = insider_fixed.compute_exact_singular_rate(z_t, v_rand, t_space[i])
            
            dX_t = theta_t * dt
            dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
            dY_t = dX_t + dZ_t
            
            z_next, _ = mm.filter_step(z_t, dY_t, dt)
            Y_t += dY_t
            z_t = np.array(z_next)
            
            Y_paths_uncond[path_idx, i] = Y_t


    # -----------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------
    
    # Plot 1: Pricing Convergence (Conditional)
    plt.figure(figsize=(10, 6))
    for i in range(10): # Plot first 10 paths to avoid clutter
        plt.plot(t_space, P_paths_cond[i], color='purple', alpha=0.3)
    plt.axhline(v_fixed, color='red', linestyle='--', label=f'True Value $v={v_fixed}$')
    plt.axhline(P_0, color='blue', linestyle='--', label=f'Initial $P_0={P_0}$')
    plt.title("Pricing Convergence to Fair Value (Positive Prices)")
    plt.xlabel("Time $t$")
    plt.ylabel("Price $P_t$")
    plt.legend()
    plt.grid(True)
    plt.savefig('examples/finance/results/pricing_convergence.png')
    plt.close()

    # Plot 2: Brownian Bridge Variance (Conditional Order Flow)
    plt.figure(figsize=(10, 6))
    emp_var_cond = np.var(Y_paths_cond, axis=0)
    # Exact Brownian Bridge Variance from 0 to Y_T: Var = sigma_Z^2 * t * (1 - t/T)
    # Because Y_T = (v - P_0) / lambda is deterministically known to the insider!
    theo_var_bridge = (sigma_z**2) * t_space * (1 - t_space/T)
    
    plt.plot(t_space, emp_var_cond, label='Empirical Variance of $Y_t$ (Conditional)', color='red', linewidth=2)
    plt.plot(t_space, theo_var_bridge, label=r'Theoretical Brownian Bridge Var: $t(1-t/T)$', color='black', linestyle='--')
    plt.title("Insider's View: Total Order Flow forms a Brownian Bridge")
    plt.xlabel("Time $t$")
    plt.ylabel("Variance of $Y_t$")
    plt.legend()
    plt.grid(True)
    plt.savefig('examples/finance/results/koopman_bridge_variance.png')
    plt.close()

    # Plot 3: Brownian Motion Variance (Unconditional Order Flow)
    plt.figure(figsize=(10, 6))
    emp_var_uncond = np.var(Y_paths_uncond, axis=0)
    theo_var_bm = (sigma_z**2) * t_space
    
    plt.plot(t_space, emp_var_uncond, label='Empirical Variance of $Y_t$ (Unconditional)', color='blue', linewidth=2)
    plt.plot(t_space, theo_var_bm, label=r'Theoretical Brownian Motion Var: $t$', color='black', linestyle='--')
    plt.title("Market Maker's View: Total Order Flow is perfectly absorbed as a Brownian Motion")
    plt.xlabel("Time $t$")
    plt.ylabel("Variance of $Y_t$")
    plt.legend()
    plt.grid(True)
    plt.savefig('examples/finance/results/koopman_bm_variance.png')
    plt.close()

    print("Experiment successful! See examples/finance/results/ for the updated plots.")

if __name__ == "__main__":
    run_experiment()
