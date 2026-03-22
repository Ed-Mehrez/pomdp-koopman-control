r"""
Verification: Heston-Merton Baseline (Full-Kernel Momentum)
===========================================================
Calculates the hedging demand sweep to verify the KKT Delta calibration.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from merton_kronic_bilinear import HestonMertonEnv
from merton_kronic_kernel_tensor import KernelMertonController

def analytical_hedging_demand(env):
    mu, r, gamma = env.mu, env.r, env.gamma
    kappa, theta, xi, rho = env.kappa, env.theta, env.xi, env.rho
    a = mu - r
    # Effective kappa for the volatility process under the 'optimal' Merton drift
    # This is a simplification of the Campbell-Viceira / Chacko-Viceira result
    kappa_eff = kappa - rho * xi * (1-gamma) * a / (gamma) # Simplified
    gamma_coeff = -(1-gamma) * a**2 / (2 * gamma**2)
    disc = kappa_eff**2 - 4 * 0.5 * xi**2 * gamma_coeff
    if disc < 0: return float('nan')
    B = (kappa_eff - np.sqrt(disc)) / xi**2 # Vol hedge sensitivity
    return rho * xi * (1-gamma) * B / gamma

def run_verification_sweep():
    V_TEST = 0.04
    N_PATHS = 1200 # Higher N for better SNR
    N_MC = 100
    RHO_LIST = [0.0, -0.3, -0.5, -0.7, -0.9]

    print("=" * 72)
    print(f"HESTON-MERTON VERIFICATION SWEEP (Full-Kernel Momentum, N={N_PATHS})")
    print("=" * 72)
    print(f"{'rho':>5s} | {'Theory':>8s} | {'Myopic':>8s} | {'pi*':>8s} | {'Delta':>8s} | {'R2':>6s}")
    print("-" * 72)

    for rho in RHO_LIST:
        np.random.seed(42); torch.manual_seed(42)
        env = HestonMertonEnv(rho=rho, gamma=3.0)
        theory = analytical_hedging_demand(env)
        myopic = env.merton_optimal(V_TEST)
        
        ctrl = KernelMertonController(env, mx=60, npi=15)
        X0, XT, pi, UT = ctrl.generate_training_data(N_PATHS, N_MC, momentum=True)
        ctrl.train(X0, XT, pi, UT)
        
        # Test point: V=0.04, pi_prev = myopic (Stationary assumption for test)
        pi_opt = ctrl.find_optimal_pi([(0, V_TEST, myopic)] * 3)
        
        # Calculate R2 manually for reporting
        Xp = ctrl.kkt._transform(X0)
        Psi_X = ctrl.kkt.rbf_gram(Xp, ctrl.kkt._lm_x, ctrl.kkt.sigma_x)
        Psi_P = ctrl.kkt.rbf_gram(pi[:, None], ctrl.kkt._lm_pi, ctrl.kkt.sigma_pi)
        Psi_J = np.array([np.kron(Psi_X[i], Psi_P[i]) for i in range(len(X0))])
        Un = (UT - ctrl.kkt._U_mu) / ctrl.kkt._U_std
        Up = (Psi_X @ (ctrl.kkt.w_util_base / ctrl.kkt._U_std) + Psi_J @ (ctrl.kkt.w_util / ctrl.kkt._U_std))
        r2 = 1 - np.mean((Un - Up)**2) / np.var(Un)

        print(f"{rho:+.1f} | {theory:+.4f} | {myopic:+.4f} | {pi_opt:+.4f} | {pi_opt-myopic:+.4f} | {r2:.4f}")

if __name__ == "__main__":
    # Add rbf_gram to KKT scope if missing (it's global in current tensor file)
    import merton_kronic_kernel_tensor as mkt
    if not hasattr(mkt.KernelKoopmanTensor, 'rbf_gram'):
        mkt.KernelKoopmanTensor.rbf_gram = staticmethod(mkt.rbf_gram)
    
    run_verification_sweep()
