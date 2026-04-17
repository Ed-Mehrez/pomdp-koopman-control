r"""
Verification: Heston-Merton Baseline (Full-Kernel Momentum)
===========================================================
Calculates the hedging demand sweep to verify the KKT Delta calibration.
"""

import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from merton_kronic_bilinear import HestonMertonEnv
from merton_kronic_kernel_tensor import KernelMertonController
from merton_theory import canonical_state_history, stationary_heston_crra_theory

def run_verification_sweep():
    V_TEST = 0.04
    N_PATHS = 1200 # Higher N for better SNR
    N_MC = 100
    RHO_LIST = [0.0, -0.3, -0.5, -0.7, -0.9]

    print("=" * 72)
    print(f"HESTON-MERTON VERIFICATION SWEEP (Full-Kernel Momentum, N={N_PATHS})")
    print("=" * 72)
    print(f"{'rho':>5s} | {'Theory':>8s} | {'Myopic':>8s} | {'pi*':>8s} | {'Delta':>8s} | {'p':>6s} | {'OpR2':>6s} | {'UR2':>6s}")
    print("-" * 72)

    for rho in RHO_LIST:
        np.random.seed(42); torch.manual_seed(42)
        env = HestonMertonEnv(rho=rho, gamma=3.0)
        theory = stationary_heston_crra_theory(env, V_TEST)
        myopic = theory.myopic_pi
        
        ctrl = KernelMertonController(env, mx=60, npi=15)
        X0, XT, pi, UT = ctrl.generate_training_data(N_PATHS, N_MC, momentum=True)
        ctrl.train(X0, XT, pi, UT)
        
        eval_history = canonical_state_history(V_TEST, myopic)
        eval_diag = ctrl.evaluate_state(eval_history)
        pi_opt = eval_diag["pi_opt"]

        print(
            f"{rho:+.1f} | {theory.hedging_demand:+.4f} | {myopic:+.4f} | "
            f"{pi_opt:+.4f} | {pi_opt-myopic:+.4f} | {theory.exponent_p:+.3f} | "
            f"{ctrl.operator_r2:.4f} | {ctrl.utility_r2:.4f}"
        )

if __name__ == "__main__":
    run_verification_sweep()
