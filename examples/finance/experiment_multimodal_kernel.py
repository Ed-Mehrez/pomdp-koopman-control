import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.finance.adaptive_kyle_kernel import AdaptiveMM_Kernel, AdaptiveInsider_Kernel
from src.sskf.streaming_sig_kkf import SignatureState

def get_bimodal_v():
    """Samples v from a bimodal distribution: N(80, 10^2) or N(120, 10^2)"""
    if np.random.rand() < 0.5:
        return np.random.normal(80, 10)
    else:
        return np.random.normal(120, 10)

def run_multimodal_experiment():
    np.random.seed(42)
    
    T = 1.0
    dt = 0.005
    steps = int(T / dt)
    t_space = np.linspace(0, T, steps)
    
    P_0 = 100.0
    lam_init = 1.0 # Initial linear guess before kernel dictionary builds up
    sigma_z = 1.0
    
    # Kernel methods need some history to build a good support dictionary
    num_episodes = 250
    
    # Initialize Kernel Adaptive Agents
    mm = AdaptiveMM_Kernel(dt=dt, lam_linear=lam_init, P_0=P_0)
    insider = AdaptiveInsider_Kernel(T=T, dt=dt)
    
    # Trackers
    Y_final_paths = []
    Pricing_Curves = []
    
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        # Reset episode state (Leaves learned RKHS Dictionary Alpha matrices intact!)
        mm.reset()
        
        # Draw from bimodal true value distribution
        v = get_bimodal_v()
        
        for i in range(steps):
            t = t_space[i]
            
            # 1. MM provides current Price P_t and evaluates RKHS derivative \lambda_t
            # We take a finite-difference probe of the kernel
            dY_test = 0.01
            lam_t = mm.evaluate_price_derivative(dY_test)
            
            # 2. Insider instantly localizes SDRE and trades
            theta_t = insider.compute_optimal_rate(v, t, mm.P_t, lam_t)
            
            # Clamp extreme theta
            theta_t = np.clip(theta_t, -500, 500)
            
            # 3. Market executes
            dX_t = theta_t * dt
            dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
            dY_t = dX_t + dZ_t
            
            # 4. MM updates filter and moves price forward
            _ = mm.filter_step(dY_t)
            
            if ep == num_episodes - 1:
                Y_final_paths.append(mm.Y_t)
                
        # --- End of Episode Update for MM Pricing Rule ---
        # The MM now observes the true v. They add the terminal Z_T to their dictionary
        # and recompute the Kernel Ridge Regression.
        mm.end_of_episode_update(v)
        
        # Every 25 episodes, we sample the learned pricing curve over a grid to track convergence
        if ep % 25 == 0 or ep == num_episodes - 1:
            Y_eval = np.linspace(-30, 30, 100)
            P_eval = np.zeros_like(Y_eval)
            
            if len(mm.kkf.support_points) > 5:
                for idx, y in enumerate(Y_eval):
                    # We create a dummy signature representing a straight path from (0,0) to (1, y)
                    # For a straight path, Levi area is 0. 
                    sig_test = SignatureState(level=mm.kkf.level, store_path=mm.kkf.store_path)
                    if mm.kkf.store_path:
                        sig_test.t_history = [0.0]
                        sig_test.x_history = [0.0]
                        
                    # [1, dt, dx]  We just extend from 0 to 1 in one go for the terminal signature
                    sig_test.extend(T, y)
                    
                    k_vec = np.array([sig_test.kernel_with(sp[0], kernel_type=mm.kkf.kernel_type) for sp in mm.kkf.support_points])
                    P_eval[idx] = k_vec @ mm.kkf.alpha
            else:
                P_eval = P_0 + lam_init * Y_eval
                
            Pricing_Curves.append(P_eval)

    # ----------- Plotting -----------
    os.makedirs("examples/finance/results", exist_ok=True)
    
    # Plot the learned RKHS Non-Linear Pricing Curve over time
    plt.figure(figsize=(10, 6))
    
    Y_eval = np.linspace(-30, 30, 100)
    colors = plt.cm.viridis(np.linspace(0, 1, len(Pricing_Curves)))
    
    for j, P_curve in enumerate(Pricing_Curves):
        label = f"Ep {j*25}" if j < len(Pricing_Curves)-1 else f"Final (Ep {num_episodes})"
        alpha = 0.3 if j < len(Pricing_Curves)-1 else 1.0
        linewidth = 1 if j < len(Pricing_Curves)-1 else 3
        
        plt.plot(Y_eval, P_curve, color=colors[j], alpha=alpha, linewidth=linewidth, label=label)
        
    plt.axhline(120, color='gray', linestyle='--', alpha=0.5, label='Upper Mode (120)')
    plt.axhline(100, color='black', linestyle=':', alpha=0.5, label='Prior Mean (100)')
    plt.axhline(80, color='gray', linestyle='--', alpha=0.5, label='Lower Mode (80)')
    
    plt.title("Evolution of the RKHS Signature Kernel S-Curve Pricing Rule")
    plt.xlabel("Terminal Order Flow Y_T")
    plt.ylabel("Price Prediction P_T")
    plt.grid()
    
    # Put legend outside if it's too big
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("examples/finance/results/multimodal_pricing_curve.png")
    plt.close()
    
    print("Multimodal RKHS Kernel experiment completed. Plots saved to examples/finance/results/")

if __name__ == "__main__":
    run_multimodal_experiment()
