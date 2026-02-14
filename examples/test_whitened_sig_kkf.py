import numpy as np
import scipy.linalg
import scipy.special
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src"))
from rough_paths_generator import FractionalBrownianMotion

def generate_fOU(H, theta, sigma, n_steps, dt=0.01, seed=42):
    np.random.seed(seed)
    fbm = FractionalBrownianMotion(H=H, dt=dt, seed=seed)
    B = fbm.generate(n_steps + 1, n_paths=1)[0]
    dB = np.diff(B)
    X = np.zeros(n_steps)
    for i in range(1, n_steps):
        X[i] = X[i-1] * (1 - theta * dt) + sigma * dB[i-1]
    return X, dB

def compute_fgn_covariance(H, n, dt):
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k + 1)**(2*H) + np.abs(k - 1)**(2*H) - 2 * np.abs(k)**(2*H)) * (dt**(2*H))
    C = scipy.linalg.toeplitz(gamma)
    return C

def test_whitened_sig_kkf():
    print("--- Testing Whitened Sig-KKF (Cholesky + Koopman) ---")
    H = 0.3
    true_theta = 1.0
    sigma = 0.5
    dt = 0.1 # Long Span T=400
    N = 4000 
    
    # Generate Data
    X, dB_true = generate_fOU(H, true_theta, sigma, N, dt=dt, seed=42)
    X_state = X[:-1]
    dX = np.diff(X)
    n_obs = len(dX)
    
    # 1. Whitening (The "Bridge")
    print("Computing Covariance and Cholesky...")
    Sigma_Noise = compute_fgn_covariance(H, n_obs, dt)
    try:
        L = scipy.linalg.cholesky(Sigma_Noise, lower=True)
    except scipy.linalg.LinAlgError:
        Sigma_Noise += 1e-6 * np.eye(n_obs)
        L = scipy.linalg.cholesky(Sigma_Noise, lower=True)
        
    print("Whitening Data...")
    dX_w = scipy.linalg.solve_triangular(L, dX, lower=True)
    X_w = scipy.linalg.solve_triangular(L, X_state, lower=True)
    
    # 2. Sig-KKF KRR on Whitened Data
    # Feature Space: Phi(x_t) = [1, x_t] (for Linear Propagator)
    # Dynamics: x_{t+1} = (1 - theta*dt) x_t + noise
    # In Whitened space: dX_w ~ -theta * X_w * dt + WhiteNoise
    # This implies standard linear regression is valid.
    
    # Let's perform Ridge Regression (Koopman Style)
    # min || dX_w - (A X_w) ||^2
    # This is equivalent to Exact MLE for linear features.
    
    num = np.dot(X_w, dX_w)
    den = np.dot(X_w, X_w)
    A_est = num / den
    
    # Convert back to Theta
    # dX_w = -theta * X_w * dt
    # A_est = -theta * dt
    theta_est = -A_est / dt
    
    print(f"Estimated Theta (Whitened Sig-KKF): {theta_est:.4f}")
    
    err = abs(theta_est - true_theta) / true_theta * 100
    print(f"Error: {err:.2f}%")
    
    if err < 5.0:
        print("SUCCESS: Whitened Sig-KKF bridges the gap!")
    else:
        print("RESULT: Bias persists.")

if __name__ == "__main__":
    test_whitened_sig_kkf()
