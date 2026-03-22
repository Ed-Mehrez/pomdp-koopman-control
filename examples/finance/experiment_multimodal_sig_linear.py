import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.finance.adaptive_kyle_sig_linear import AdaptiveMM_SigLinear, AdaptiveInsider_SigLinear

def get_bimodal_v():
    """Samples v from a bimodal distribution: N(80, 10^2) or N(120, 10^2)"""
    if np.random.rand() < 0.5:
        return np.random.normal(80, 10)
    else:
        return np.random.normal(120, 10)

def run_multimodal_experiment():
    np.random.seed(42)
    torch.manual_seed(42)
    
    T = 1.0
    dt = 0.005
    steps = int(T / dt)
    t_space = np.linspace(0, T, steps)
    
    P_0 = 100.0
    lam_init = 1.0 
    sigma_z = 1.0
    depth = 4 # Full polynomial projection up to degree 4! (340 features)
    
    num_episodes = 250 
    
    # Exact GPU Signature RLS
    mm = AdaptiveMM_SigLinear(dt=dt, lam_linear=lam_init, P_0=P_0, depth=depth)
    insider = AdaptiveInsider_SigLinear(T=T, dt=dt)
    
    Pricing_Curves = []
    
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        # Reset episode state 
        mm.reset()
        v = get_bimodal_v()
        
        for i in range(steps):
            t = t_space[i]
            
            # 1. Evaluate Price Derivative
            dY_test = 0.01
            lam_t = mm.evaluate_price_derivative(dY_test)
            
            # 2. Insider calculates exact SDRE instantaneous control
            theta_t = insider.compute_optimal_rate(v, t, mm.P_t, lam_t)
            theta_t = np.clip(theta_t, -500, 500)
            
            # 3. Step Market
            dX_t = theta_t * dt
            dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
            dY_t = dX_t + dZ_t
            
            # 4. MM updates internal PyTorch tracker
            _ = mm.filter_step(dY_t)
            
        # Train Weights
        mm.end_of_episode_update(v)
        
        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            Y_eval = np.linspace(-30, 30, 100)
            P_eval = np.zeros_like(Y_eval)
            
            # To securely sample the pure geometric mapping of terminal order flow
            for idx, y in enumerate(Y_eval):
                # We write the terminal line into the path tensor manually
                dummy_path = torch.zeros(1, 2, 2, device=mm.device, dtype=torch.float64)
                dummy_path[0, 1, 0] = T
                dummy_path[0, 1, 1] = y
                
                # Manual extraction 
                # (Notice: dummy_path is manually generated here, we MUST apply the same scale!)
                dummy_scaled = dummy_path.clone()
                dummy_scaled[:, :, 0] /= 1.0
                dummy_scaled[:, :, 1] /= 5.0
                
                Z_sig = mm._get_signature(dummy_scaled)
                Z_feat = torch.cat([Z_sig, torch.tensor([1.0], device=mm.device, dtype=torch.float64)])
                
                P_eval[idx] = torch.dot(mm.w, Z_feat).item()
                
            Pricing_Curves.append(P_eval)

    # ----------- Plotting -----------
    os.makedirs("examples/finance/results", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    Y_eval = np.linspace(-30, 30, 100)
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(Pricing_Curves))))
    
    for j, P_curve in enumerate(Pricing_Curves):
        label = f"Sample {j+1}" if j < len(Pricing_Curves)-1 else f"Final (Ep {num_episodes})"
        alpha = 0.3 if j < len(Pricing_Curves)-1 else 1.0
        linewidth = 1 if j < len(Pricing_Curves)-1 else 3
        
        plt.plot(Y_eval, P_curve, color=colors[j], alpha=alpha, linewidth=linewidth, label=label)
        
    plt.axhline(120, color='gray', linestyle='--', alpha=0.5, label='Upper Mode (120)')
    plt.axhline(100, color='black', linestyle=':', alpha=0.5, label='Prior Mean (100)')
    plt.axhline(80, color='gray', linestyle='--', alpha=0.5, label='Lower Mode (80)')
    
    plt.title(f"Evolution of PyTorch Signature Linear Pricing Rule (Depth {depth})")
    plt.xlabel("Terminal Order Flow Y_T")
    plt.ylabel("Price Prediction P_T")
    plt.grid()
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("examples/finance/results/multimodal_pricing_sig_linear.png")
    plt.close()
    
    print("Multimodal Signature Linear RLS experiment completed. Saved to examples/finance/results/")

if __name__ == "__main__":
    run_multimodal_experiment()
