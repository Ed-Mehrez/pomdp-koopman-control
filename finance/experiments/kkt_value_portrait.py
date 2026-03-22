
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

# paths
sys.path.insert(0, os.path.dirname(__file__))
from merton_kronic_kernel_tensor import KernelMertonController, HestonMertonEnv

def portrait():
    env = HestonMertonEnv(rho=-0.5)
    ctrl = KernelMertonController(env, horizon=10, n_landmarks=60, n_pca=15)
    
    # 1. Train
    print("Generating training data...")
    X0, XT, pi_a, U_T = ctrl.generate_training_data(n_paths=1000, n_mc=100)
    ctrl.train(X0, XT, pi_a, U_T)
    
    # 2. Pick a test state
    v_test = 0.04
    hist = np.array([[0.0, v_test], [0.0, v_test], [0.0, v_test]])
    
    with torch.no_grad():
        t_hist = torch.tensor(np.array([hist]), dtype=torch.float64)
        x_feat = ctrl.extractor.extract(t_hist)[0].cpu().numpy()
    
    pi_grid = np.linspace(0.01, 5.0, 100)
    
    plt.figure(figsize=(10, 6))
    
    # 3. Use Controller to find peak (handles Log-Growth)
    pi_star, c0, c1, c2 = ctrl.find_optimal_pi(x_feat)
    
    # Reconstruct for plot
    u_sign = np.sign(ctrl.kkt._U_mu)
    total_T = ctrl.kkt.horizon * ctrl.kkt.dt
    lambda_grid = c0 + c1 * pi_grid + c2 * pi_grid**2
    V_grid = u_sign * np.exp(lambda_grid * total_T)
    
    # Normalize for Plot
    v_norm = (V_grid - V_grid.min()) / (V_grid.max() - V_grid.min() + 1e-9)
    plt.plot(pi_grid, v_norm, 'b-', label='KKT Value Landscape $V(\pi)$', lw=2)
    plt.axvline(pi_star, color='r', linestyle='--', label=f'KKT Peak = {pi_star:.2f}')
    
    # Merton Myopic Baseline
    m_opt = (env.mu - env.r) / (env.gamma * v_test)
    plt.axvline(m_opt, color='k', linestyle=':', label=f'Merton Myopic = {m_opt:.2f}')
    
    plt.title(f"KKT Value Portrait (Log-Growth, rho=-0.5, V=0.04)")
    plt.xlabel("Allocation $\pi$")
    plt.ylabel("Normalized Physical Utility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("finance/experiments/kkt_final_landscape.png")
    print(f"KKT Peak: {pi_star:.4f}, Merton: {m_opt:.4f}")
    print("Portrait saved to finance/experiments/kkt_final_landscape.png")

if __name__ == "__main__":
    portrait()
