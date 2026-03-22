r"""
Full-Kernel Stress-Test: CEV Dynamics & CARA Utility (with Momentum)
===================================================================
Evaluates the robustness of the Dual-RBF Action-Momentum Koopman Tensor
across non-linear environments and non-standard utilities.

Tests:
1. CEV Model: dS/S = mu*dt + sigma*S**(alpha-1)*dW
2. CARA Utility: U(W) = -exp(-eta * W)
3. Action Momentum: Training with [logW, V, pi_prev]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from merton_kronic_bilinear import HestonMertonEnv, CEVEnv, CARAUtility
from merton_kronic_kernel_tensor import KernelMertonController

def run_stress_test():
    V_TEST = 0.04
    N_PATHS = 1000
    N_MC = 100

    print("=" * 72)
    print("FULL-KERNEL STRESS-TEST: CEV & CARA (MOMENTUM)")
    print("=" * 72)

    # ---------------------------------------------------------
    # Test 1: CEV Dynamics (alpha = 0.5, Square-Root Vol)
    # ---------------------------------------------------------
    print("\n[Phase 1] CEV Dynamics (alpha=0.5, sigma=0.2)")
    env_cev = CEVEnv(alpha=0.5, sigma=0.2)
    myopic_cev = env_cev.merton_optimal(V_TEST)
    
    # KKT Dual-RBF
    ctrl_cev = KernelMertonController(env_cev)
    X0, XT, pi, UT = ctrl_cev.generate_training_data(N_PATHS, N_MC, momentum=True)
    ctrl_cev.train(X0, XT, pi, UT)
    # Test point: V=0.04, pi_prev=0.5 (Neutral)
    pi_cev = ctrl_cev.find_optimal_pi([(0, V_TEST, 0.5)] * 3)
    
    print(f"   CEV Myopic (pi_m) = {myopic_cev:.4f}")
    print(f"   KKT Action-RBF (pi*) = {pi_cev:.4f} | Delta = {pi_cev - myopic_cev:+.4f}")

    # ---------------------------------------------------------
    # Test 2: CARA Utility (Exponential)
    # ---------------------------------------------------------
    print("\n[Phase 2] CARA Utility (eta=1.5) on Heston")
    env_heston = HestonMertonEnv(rho=-0.7)
    cara = CARAUtility(eta=1.5)
    
    def generate_cara_momentum_data(ctrl, n_paths, n_mc):
        X0_l, XT_l, pi_l, U_l = [], [], [], []
        win, dt = 3, 1/252
        for _ in range(n_paths):
            lw0, v0 = 0.0, np.random.uniform(0.01, 0.09)
            pic = np.random.uniform(0.1, 3.0)
            pi_prev = np.random.uniform(0.1, 3.0)
            
            p0 = []
            lw, v = lw0, v0
            for _ in range(win):
                lw, v, _ = ctrl.env.step_momentum(lw, v, pic, pi_prev, dt=dt)
                p0.append((lw, v, pi_prev))
            p0 = np.array(p0); p0[:, 0] -= p0[-1, 0]
            with torch.no_grad():
                psi0 = ctrl.extractor.extract(torch.tensor(np.array([p0]), dtype=torch.float64))[0].cpu().numpy()
            
            lwp, vp, _ = p0[-1]
            Ur = []
            for _ in range(n_mc):
                lr, vr = lwp, vp
                for _ in range(ctrl.horizon): lr, vr = ctrl.env.step(lr, vr, pic, dt=dt)
                Ur.append(cara(np.exp(lr - lw0)))
            X0_l.append(psi0); XT_l.append(psi0)
            pi_l.append(pic); U_l.append(np.mean(Ur))
        ctrl.scaler.fit(X0_l)
        return np.array(X0_l), np.array(XT_l), np.array(pi_l), np.array(U_l)

    ctrl_cara = KernelMertonController(env_heston)
    X0, XT, pi, UT = generate_cara_momentum_data(ctrl_cara, N_PATHS, N_MC)
    print("   Fitting Adaptive CARA RBF...")
    ctrl_cara.train(X0, XT, pi, UT)
    pi_cara = ctrl_cara.find_optimal_pi([(0, 0.04, 0.75)] * 3)
    
    myopic_heston = env_heston.merton_optimal(0.04)
    print(f"   Heston Myopic (pi_m) = {myopic_heston:.4f}")
    print(f"   CARA KKT RBF (pi*) = {pi_cara:.4f} | Delta = {pi_cara - myopic_heston:+.4f}")

    # ---------------------------------------------------------
    # Visualization: Non-Parametric Landscape
    # ---------------------------------------------------------
    pi_grid = np.linspace(0.01, 5.0, 150)[:, None]
    Xp = ctrl_cara.kkt._transform(X0[0][None])
    # rbf_gram expects (N, D)
    psi = np.exp(-np.sum((Xp - ctrl_cara.kkt._lm_x)**2, axis=1) / (2 * ctrl_cara.kkt.sigma_x**2))
    
    # Reconstruction logic: V = w_base * psi + (psi \otimes Psi_P) * w_skill
    Psi_P = np.exp(-np.sum((pi_grid[:, None] - ctrl_cara.kkt._lm_pi[None, :])**2, axis=-1) / (2 * ctrl_cara.kkt.sigma_pi**2))
    Psi_J = np.array([np.kron(psi, Psi_P[i]) for i in range(len(pi_grid))])
    
    V = (ctrl_cara.kkt.w_util_base @ psi) + (Psi_J @ ctrl_cara.kkt.w_util)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pi_grid.flatten(), V, 'b-', lw=2, label='Joint RBF Value')
    plt.axvline(pi_cara, color='r', ls='--', label='KKT Optimal')
    plt.title("CARA Utility Landscape - Full-Kernel Dual-RBF")
    plt.xlabel("Allocation (pi)"); plt.ylabel("Predicted Utility")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("finance/experiments/cara_rbf_momentum.png")
    print("\n[Result] Landscape saved to finance/experiments/cara_rbf_momentum.png")

if __name__ == "__main__":
    run_stress_test()
