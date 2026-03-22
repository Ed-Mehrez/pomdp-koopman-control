import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.finance.adaptive_kyle_poly import AdaptiveMM_Poly, AdaptiveInsider_Poly

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
    lam_init = 1.0 # Initial linear guess
    sigma_z = 1.0
    
    # We use 500 episodes because fitting 5 polynomials takes slightly longer to stabilize
    num_episodes = 500 
    
    # Initialize Polynomial Adaptive Agents
    mm = AdaptiveMM_Poly(dt=dt, lam_linear=lam_init, P_0=P_0, forgetting_factor=1.0)
    insider = AdaptiveInsider_Poly(T=T, dt=dt)
    
    # Trackers
    Y_final_paths = []
    V_final = []
    W_history = np.zeros((num_episodes, 5)) # Tracking MM's pricing weights [w0, w1, w2, w3, w4]
    
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        # Reset episode state
        mm.reset()
        
        # Draw from bimodal true value distribution
        v = get_bimodal_v()
        V_final.append(v)
        
        # Polynomial State: [1, t, Y, Y^2, Y^3]
        Z_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) 
        
        for i in range(steps):
            t = t_space[i]
            
            # 1. MM provides current Koopman operator
            L_mm, g_mm, w_mm = mm.get_koopman_matrices()
            
            # 2. Insider instantly localizes SDRE and trades
            theta_t = insider.compute_optimal_rate(Z_t, v, t, L_mm, g_mm, w_mm)
            
            # Clamp extreme theta
            theta_t = np.clip(theta_t, -500, 500)
            
            # 3. Market executes
            dX_t = theta_t * dt
            dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
            dY_t = dX_t + dZ_t
            
            # 4. MM updates filter
            _ = mm.filter_step(dY_t)
            
            # Update state for next step
            Y_next = Z_t[2] + dY_t
            Z_t = np.array([1.0, t + dt, Y_next, Y_next**2, Y_next**3])
            
            if ep == num_episodes - 1:
                Y_final_paths.append(Z_t[2])
                
        # --- End of Episode Update for MM Pricing Rule ---
        # The MM now observes the true v. They want to find w_mm such that w_mm^T Z_T \approx v
        Z_T = Z_t 
        
        # Stochastic gradient descent on the 5 pricing weights
        # We use a very small learning rate since Y^3 can be quite large
        learning_rate = 1e-4
        prediction = np.dot(mm.w_mm, Z_T)
        error = prediction - v
        
        # Gradient clip to prevent exploding weights due to Y^3
        grad = error * Z_T
        grad = np.clip(grad, -10.0, 10.0)
        
        mm.w_mm = mm.w_mm - learning_rate * grad
        
        W_history[ep, :] = mm.w_mm
                
    # ----------- Plotting -----------
    os.makedirs("examples/finance/results", exist_ok=True)
    
    # 1. Plot Pricing Weights over episodes
    plt.figure(figsize=(10, 6))
    epochs = np.arange(num_episodes)
    plt.plot(epochs, W_history[:, 0], label='Bias (w0)')
    plt.plot(epochs, W_history[:, 1], label='Time Weight (w1)')
    plt.plot(epochs, W_history[:, 2], label='Linear Flow Weight (w2)')
    plt.title("Convergence of Market Maker's Linear Weights")
    plt.xlabel("Episode")
    plt.ylabel("Weight Value")
    plt.grid()
    plt.legend()
    plt.savefig("examples/finance/results/multimodal_weights_linear.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, W_history[:, 3], label='Quadratic Weight (w3)', color='purple')
    plt.plot(epochs, W_history[:, 4], label='Cubic Weight (w4)', color='brown')
    plt.title("Convergence of Market Maker's Non-Linear (Cubic) Weights")
    plt.xlabel("Episode")
    plt.ylabel("Weight Value")
    plt.grid()
    plt.legend()
    plt.savefig("examples/finance/results/multimodal_weights_nonlinear.png")
    plt.close()
    
    # 2. Plot the learned Non-Linear Pricing Curve P(Y)
    Y_eval = np.linspace(-30, 30, 200)
    P_eval = np.zeros_like(Y_eval)
    
    w_final = W_history[-1, :]
    for i, y in enumerate(Y_eval):
        # [1, T, Y_T, Y_T^2, Y_T^3]
        Z_test = np.array([1.0, T, y, y**2, y**3])
        P_eval[i] = np.dot(w_final, Z_test)
        
    plt.figure(figsize=(10, 6))
    plt.plot(Y_eval, P_eval, 'r-', linewidth=2, label='Learned Price P(Y)')
    plt.axhline(120, color='gray', linestyle='--', alpha=0.5, label='Upper Mode (120)')
    plt.axhline(100, color='black', linestyle=':', alpha=0.5, label='Prior Mean (100)')
    plt.axhline(80, color='gray', linestyle='--', alpha=0.5, label='Lower Mode (80)')
    
    # Optional: Plot the true theoretical Bayesian posterior S-curve for comparison
    sigma_y = sigma_z * np.sqrt(T) # Unconditional variance of noise roughly
    # P(v=120) = 0.5, P(v=80) = 0.5
    # The signal is y. But insider trading makes y correlate with v.
    # We don't have the exact closed form instantly, but we can visualize the learned one.
    
    plt.title("Emergent Non-Linear S-Curve Pricing Rule (Koopman Polynomial)")
    plt.xlabel("Terminal Order Flow Y_T")
    plt.ylabel("Price P_T")
    plt.grid()
    plt.legend()
    plt.savefig("examples/finance/results/multimodal_pricing_curve.png")
    plt.close()
    
    print("Multimodal Polynomial experiment completed. Plots saved to examples/finance/results/")

if __name__ == "__main__":
    run_multimodal_experiment()
