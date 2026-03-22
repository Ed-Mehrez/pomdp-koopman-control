import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure src modules are discoverable
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

from sskf.streaming_sig_kkf import StreamingSigKKF
from poc_signature_cdc_sv import simulate_heston, simulate_fbm_rough_vol

def test_sig_kkf_streaming(paths, name, dt=1/252):
    """
    Streams a stationary stochastic volatility or fractional path through the
    multidimensional Signature-Koopman Kalman Filter.
    Returns the real-time history of the Koopman maximum eigenvalue to prove stability.
    """
    n_paths, n_steps = paths.shape
    path = paths[0, :]
    
    # Initialize the Sig-KKF with Level 2 signatures
    # Forgetting factor 0.99 for stable structural learning
    kkf = StreamingSigKKF(dt=dt, level=2, forgetting_factor=0.99, process_noise=1e-5)
    kkf.reset(path[0])
    
    max_eigenvalues = []
    
    for t in range(1, n_steps):
        # Update filter with new observation
        kkf.update(path[t])
        
        # Extract the instantaneous Signature-Koopman Generator A
        # The eigenvalues of A dictate the structural stability of the system
        A_curr = kkf.get_generator()
        
        # Compute maximum real part of the eigenvalues
        eigvals = np.linalg.eigvals(A_curr)
        max_real_eig = np.max(np.real(eigvals))
        
        max_eigenvalues.append(max_real_eig)
        
    # Ignore initial burn-in period where the matrix is ill-conditioned
    burn_in = 200
    safe_max_eig = np.max(max_eigenvalues[burn_in:])
    
    print(f"--- {name} Sig-KKF Results ---")
    print(f"Max Koopman Re(\\lambda) (post burn-in): {safe_max_eig:.4f}\n")
    
    return path, max_eigenvalues

if __name__ == "__main__":
    S0 = 100.0
    T = 5.0 # 5 years
    dt = 1/252
    
    print("=======================================================================")
    print(" PHASE III: Sig-KKF Streaming on SV & Fractional Rough Volatility")
    print("=======================================================================\n")
    
    # 1. Heston SV
    print("Streaming 5 Years of Heston SV through Sig-KKF...")
    paths_heston = simulate_heston(S0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=T, dt=dt, n_paths=1)
    s_heston, eigs_heston = test_sig_kkf_streaming(paths_heston, "Heston SV")
    
    # 2. Rough fSDE (Hurst = 0.1)
    print("Streaming 5 Years of Rough FBM through Sig-KKF...")
    paths_fbm = simulate_fbm_rough_vol(S0, mu=0.05, nu=0.2, H=0.1, T=T, dt=dt, n_paths=1)
    s_fbm, eigs_fbm = test_sig_kkf_streaming(paths_fbm, "Rough FBM (H=0.1)")
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    time_ax = np.linspace(0, T, len(s_heston))
    
    axes[0].plot(time_ax, s_heston, label="Heston SV Price", color='blue', alpha=0.7)
    axes[0].plot(time_ax, s_fbm, label="Rough FBM Price", color='orange', alpha=0.7)
    axes[0].set_title("Simulated Asset Paths (True Martingales)")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    
    # Pad eigenvalue arrays to match time_ax length (first element missing)
    eigs_h_padded = [0] + eigs_heston
    eigs_f_padded = [0] + eigs_fbm
    
    axes[1].plot(time_ax, eigs_h_padded, label="Heston Sig-KKF Max Re(\\lambda)", color='blue')
    axes[1].plot(time_ax, eigs_f_padded, label="FBM Sig-KKF Max Re(\\lambda)", color='orange')
    axes[1].axhline(0.0, color='black', linestyle='--', label='Stability Threshold (Re(\\lambda) = 0)')
    axes[1].set_title("Multi-Dimensional Koopman Signature Stability")
    axes[1].set_ylabel("Max Real Eigenvalue")
    axes[1].set_xlabel("Time (Years)")
    axes[1].set_ylim(-5, 5) # Focus on the stability band, ignoring initial transient spikes
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("sig_kkf_sv_fsde_robustness.png")
    print("Saved dual stability plot to sig_kkf_sv_fsde_robustness.png")
