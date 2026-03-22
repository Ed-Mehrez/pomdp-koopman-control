import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.finance.adaptive_kyle import AdaptiveMM, AdaptiveInsider

def run_adaptive_experiment():
    np.random.seed(42)
    
    T = 1.0
    dt = 0.005
    steps = int(T / dt)
    t_space = np.linspace(0, T, steps)
    
    P_0 = 100.0
    Sigma_0 = 100.0
    sigma_z = 1.0
    lam = np.sqrt(Sigma_0) / (sigma_z * np.sqrt(T))
    
    num_episodes = 50
    
    # Initialize Adaptive Agents
    # forgetting factor 1.0 means infinite memory (perfect for stationary convergence over many episodes)
    # Using 0.999 keeps it slightly adaptive.
    mm = AdaptiveMM(dt=dt, lam=lam, P_0=P_0, level=1, forgetting_factor=1.0)
    
    # To kickstart RLS, we can inject a tiny prior on A, but zero is fine.
    
    insider = AdaptiveInsider(T=T, dt=dt)
    
    # Trackers
    A_history = np.zeros((num_episodes, steps, 3)) # Tracking [a, b, c] from MM filter
    Y_final_paths = np.zeros((num_episodes, steps))
    
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        # Reset episode state (Leaves learned RLS A and P matrices intact!)
        mm.reset()
        
        # New true value
        v = np.random.normal(P_0, np.sqrt(Sigma_0))
        
        z_t = np.array([1.0, 0.0, 0.0, 0.0]) # [1, t, Y_t, P_t - P_0]
        
        for i in range(steps):
            t = t_space[i]
            
            # 1. MM provides current Koopman operator
            L_mm, g_mm = mm.get_koopman_matrices()
            
            A_history[ep, i, :] = mm.kkf.A[2, :3]
            
            # 2. Insider instantly localizes SDRE and trades
            theta_t = insider.compute_optimal_rate(z_t, v, P_0, t, L_mm, g_mm)
            
            # Clamp extreme theta during early exploration phases
            theta_t = np.clip(theta_t, -500, 500)
            
            # 3. Market executes
            dX_t = theta_t * dt
            dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
            dY_t = dX_t + dZ_t
            
            # 4. MM updates filter and sets new price
            P_next = mm.filter_step(dY_t)
            
            # Update state for insider next step
            z_t[1] = t + dt
            z_t[2] += dY_t
            z_t[3] = P_next - P_0
            
            if ep == num_episodes - 1:
                Y_final_paths[ep, i] = z_t[2]
                
    # ----------- Plotting -----------
    os.makedirs("examples/finance/results", exist_ok=True)
    
    # 1. Plot RLS Coefficients over episodes
    plt.figure(figsize=(10, 6))
    epochs = np.arange(num_episodes)
    # Take the end-of-episode coefficient matrix
    a_vals = A_history[:, -1, 0]
    b_vals = A_history[:, -1, 1]
    c_vals = A_history[:, -1, 2]
    
    plt.plot(epochs, a_vals, label='Intercept (a)')
    plt.plot(epochs, b_vals, label='Time weight (b)')
    plt.plot(epochs, c_vals, label='Order flow weight (c)')
    plt.title("Convergence of Market Maker's Koopman Filter via RLS")
    plt.xlabel("Episode")
    plt.ylabel("Coefficient Value")
    plt.grid()
    plt.legend()
    plt.savefig("examples/finance/results/adaptive_convergence.png")
    plt.close()
    
    # 2. Plot final episode Y path
    plt.figure(figsize=(10, 6))
    plt.plot(t_space, Y_final_paths[-1, :], label='Y_t (Final Episode)')
    plt.title("Total Order Flow (Y_t) after Learning Stabilization")
    plt.xlabel("Time t")
    plt.ylabel("Cumulative Order Flow")
    plt.grid()
    plt.legend()
    plt.savefig("examples/finance/results/adaptive_final_path.png")
    plt.close()
    
    print("Adaptive experiment completed. Plots saved to examples/finance/results/")

if __name__ == "__main__":
    run_adaptive_experiment()
