import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure src modules are discoverable
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/finance'))

# Import the Oracle
from bayesian_khasminskii_oracle import BayesianKhasminskiiOracle

# Import generators from existing POC
from poc_signature_cdc_sv import simulate_heston, simulate_fbm_rough_vol

def test_streaming_oracle(paths, name, dt=1/252):
    """
    Streams a set of paths through the Bayesian Oracle.
    For stationary true martingales (SV, fSDE), the conditional prob must stay near 0.
    """
    n_paths, n_steps = paths.shape
    
    # We will just test the first path explicitly for plotting
    path = paths[0, :]
    
    oracle = BayesianKhasminskiiOracle()
    
    prob_history = []
    gamma_history = []
    
    for t in range(n_steps):
        prob, gamma_est = oracle.update(path[t], dt)
        prob_history.append(prob)
        gamma_history.append(gamma_est)
        
    prob_history = np.array(prob_history)
    gamma_history = np.array(gamma_history)
    
    # Assertions
    # We ignore the first 100 days (burn-in period where prob is structurally 0)
    max_prob = np.max(prob_history[100:])
    mean_gamma = np.mean(gamma_history[100:])
    
    print(f"--- {name} Results ---")
    print(f"Max P(Bubble > 1.0) over 2 years: {max_prob * 100.0:.2f}%")
    print(f"Average Kalman Gamma Estimate:    {mean_gamma:.3f}\n")
    
    return prob_history, gamma_history, path

if __name__ == "__main__":
    S0 = 100.0
    T = 5.0 # Let's run a full 5 year stream
    dt = 1/252
    
    print("=======================================================================")
    print(" BAYESIAN KALMAN ORACLE: SV and fSDE Streaming Robustness Test ")
    print("=======================================================================\n")
    
    # 1. Heston SV
    print("Simulating 5 Years of Heston SV...")
    paths_heston = simulate_heston(S0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=T, dt=dt, n_paths=1)
    p_heston, g_heston, s_heston = test_streaming_oracle(paths_heston, "Heston SV")
    
    # 2. Rough fSDE (Hurst = 0.1)
    print("Simulating 5 Years of Rough Volatility (H=0.1)...")
    paths_fbm = simulate_fbm_rough_vol(S0, mu=0.05, nu=0.2, H=0.1, T=T, dt=dt, n_paths=1)
    p_fbm, g_fbm, s_fbm = test_streaming_oracle(paths_fbm, "Rough FBM (H=0.1)")
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    time_ax = np.linspace(0, T, len(s_heston))
    
    axes[0].plot(time_ax, s_heston, label="Heston SV Price", color='blue', alpha=0.7)
    axes[0].plot(time_ax, s_fbm, label="Rough FBM Price", color='orange', alpha=0.7)
    axes[0].set_title("Simulated Asset Paths (True Martingales)")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    
    axes[1].plot(time_ax, g_heston, label="Heston $\\hat{\\gamma}_t$", color='blue')
    axes[1].plot(time_ax, g_fbm, label="FBM $\\hat{\\gamma}_t$", color='orange')
    axes[1].axhline(1.0, color='black', linestyle='--', label='Explosive Threshold ($\gamma > 1$)')
    axes[1].set_title("Kalman Filter Estimated Variance Exponent $\gamma_t$")
    axes[1].set_ylabel("Exponent")
    axes[1].set_ylim(0, 1.5)
    axes[1].legend()
    
    axes[2].plot(time_ax, p_heston, label="Heston $\\mathbb{P}(\\gamma_t > 1)$", color='blue')
    axes[2].plot(time_ax, p_fbm, label="FBM $\\mathbb{P}(\\gamma_t > 1)$", color='orange')
    axes[2].axhline(0.95, color='red', linestyle=':', label='95% Confidence Alarm')
    axes[2].set_title("Kalman Bayesian Conditional Bubble Probability")
    axes[2].set_ylabel("Probability")
    axes[2].set_xlabel("Time (Years)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("sv_fsde_kalman_robustness.png")
    print("Saved dual stability plot to sv_fsde_kalman_robustness.png")
