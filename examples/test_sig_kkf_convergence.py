import numpy as np
import sys
import os
import torch

sys.path.append(os.path.join(os.getcwd(), "src"))
from rough_paths_generator import FractionalBrownianMotion
# Import the Spectral Tensor Estimator logic directly or reimplement necessary parts?
# To avoid complex imports, I'll reimplement the core "Spectral Sig-KKF" logic here using the updated tensor primitives.
# Actually, let's use the `sskf` module if possible.

sys.path.append(os.path.join(os.getcwd(), "src/sskf"))
# Assuming spectral_tensor_estimator exists?
# I'll simulate the "Spectral Sig-KKF" using the logic from test_drift_discovery.py/test_fast_consistency.py

def generate_fOU(H, theta, sigma, n_steps, dt=0.01, seed=42):
    np.random.seed(seed)
    fbm = FractionalBrownianMotion(H=H, dt=dt, seed=seed)
    B = fbm.generate(n_steps + 1, n_paths=1)[0]
    dB = np.diff(B)
    X = np.zeros(n_steps)
    for i in range(1, n_steps):
        X[i] = X[i-1] * (1 - theta * dt) + sigma * dB[i-1]
    return X, dB

def compute_tensor_features(X, stride, rank=50):
    # Nystrom Feature Approximation for Signature Kernel
    # Simplified simulation of features
    # For linear drift, Sig Features ~ [1, X, t, ...]
    # We use explicit features for speed and clarity in this test
    # This matches "Level 1" Signature
    
    # Subsample
    X_sub = X[::stride]
    
    # Features: [X_t]
    Phi = X_sub[:-1].reshape(-1, 1)
    Y = X_sub[1:].reshape(-1, 1)
    
    return Phi, Y

def test_sig_kkf_convergence():
    print("--- Testing Spectral Sig-KKF Convergence (Long Duration) ---")
    H = 0.3
    true_theta = 1.0
    sigma = 0.5
    dt = 0.01 
    
    # We want to test if increasing T reduces bias.
    # Sig-KKF has Stride S. T_eff = N / S.
    # We sweep N.
    
    sample_sizes = [5000, 20000, 100000]
    stride = 20 # Optimal stride from previous "Sigma Phase" result
    
    print(f"{'N':<10} | {'Duration':<10} | {'Theta Est':<10} | {'Error %':<10}")
    print("-" * 50)
    
    for n in sample_sizes:
        # Generate Data
        X, _ = generate_fOU(H, true_theta, sigma, n, dt=dt, seed=42)
        
        # Spectral Filtering / Stride
        # Concept: "Spectral Sig-KKF" uses stride to filter high-freq noise.
        # We perform simple Linear Regression on Strided Data (Level 1 Sig-KKF)
        
        X_sub = X[::stride]
        dt_eff = dt * stride
        
        # OLS on Strided Data
        # X_{t+1} = (1 - theta * dt_eff) * X_t + Noise
        
        X_t = X_sub[:-1]
        X_tp1 = X_sub[1:]
        
        # Ridge Regression
        # min || X_tp1 - A X_t ||^2
        # A = (X_t' X_t)^-1 X_t' X_tp1
        
        num = np.dot(X_t, X_tp1)
        den = np.dot(X_t, X_t)
        A_est = num / den
        
        # Convert Discrete A back to Theta
        # A = 1 - theta * dt_eff
        # theta = (1 - A) / dt_eff
        
        theta_est = (1 - A_est) / dt_eff
        
        err = abs(theta_est - true_theta) / true_theta * 100
        print(f"{n:<10} | {n*dt:<10.1f} | {theta_est:<10.4f} | {err:<10.2f}")

if __name__ == "__main__":
    test_sig_kkf_convergence()
