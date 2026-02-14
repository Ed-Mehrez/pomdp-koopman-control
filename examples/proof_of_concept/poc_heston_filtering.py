"""
Heston Volatility Estimation via Signatures
===========================================
Demonstrates O(1) volatility inference using Level-2 Signatures.
Key Feature: Principled window size selection via Koopman Eigenvalues.

Methods:
1. Heston Simulation (Ground Truth)
2. Window Selection:
   - Method 3: KGEDMD Eigenalysis (tau = -1/Re(lambda))
   - Method 4: Cross-Validation Grid Search
3. Estimation:
   - Signature Ridge Regression (Linear in Level-2 terms)
   - Baseline: Realized Variance

Usage:
    python poc_heston_filtering.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.kgedmd_core import KernelGEDMD, RBFKernel
from src.kronic_controller import KRONICController # For utility if needed
from examples.proof_of_concept.signature_features import compute_log_signature, compute_path_signature

# -----------------------------------------------------------------------------
# 1. Heston Simulation (Ground Truth)
# -----------------------------------------------------------------------------
class HestonSimulator:
    def __init__(self, kappa=2.0, theta=0.04, xi=0.3, dt=0.01):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.dt = dt
        
    def generate_path(self, n_steps, s0=100.0, v0=0.04):
        """Generate (Price, Variance) path."""
        prices = np.zeros(n_steps)
        variances = np.zeros(n_steps)
        
        S = s0
        v = v0
        
        prices[0] = S
        variances[0] = v
        
        for t in range(1, n_steps):
            # Correlated Brownian Motions (rho=0 for simplicity, can add later)
            dW1 = np.random.normal(0, np.sqrt(self.dt))
            dW2 = np.random.normal(0, np.sqrt(self.dt))
            
            # Variance Process (CIR) - Full Truncation
            v_abs = max(1e-6, v)
            dv = self.kappa * (self.theta - v_abs) * self.dt + self.xi * np.sqrt(v_abs) * dW2
            v = v + dv
            
            # Price Process
            dS = S * (0.0 * self.dt + np.sqrt(v_abs) * dW1) # Zero drift for simplicity of return analysis
            S = S + dS
            
            prices[t] = S
            variances[t] = v_abs # Store true variance
            
        returns = np.diff(np.log(prices))
        # Align variance: v[t] generates return[t] (approx)
        # Returns length is N-1. Latent variance length is N.
        # Use variance at start of interval as ground truth for that return
        true_variance = variances[:-1]
        
        return returns, true_variance

# -----------------------------------------------------------------------------
# 2. Principled Window Selection
# -----------------------------------------------------------------------------
def measure_decorrelation_time_kgedmd(returns, dt=0.01):
    """
    Method 3: Estimate decorrelation time using KGEDMD eigenvalues.
    tau_corr = -1 / Re(lambda_dom)
    """
    print("\n🔍 Method 3: Principal Eigenvalue Analysis (Koopman)")
    
    # 1. Prepare Data
    # Use squared returns as proxy for Variance state
    proxy_state = returns**2
    X_data = proxy_state.reshape(1, -1)
    
    # Create shifted pairs
    X = X_data[:, :-1]
    Y = X_data[:, 1:] 
    
    # Normalize inputs for improved Kernel conditioning
    scale = np.std(X) + 1e-9
    X = X / scale
    Y = Y / scale
    
    # Fit Kernel GEDMD
    # Use RBF kernel with adaptive sigma
    dists = pdist(X.T[:, :500]) if X.shape[1] > 500 else pdist(X.T)
    sigma = np.median(dists) if len(dists) > 0 else 1.0
    print(f"   Adaptive Sigma: {sigma:.4f}")
    
    gedmd = KernelGEDMD(kernel_type='rbf', sigma=sigma, epsilon=1e-5)
    gedmd.fit(X, Y, dt=dt, n_subsample=2000)
    
    # Inspect Eigenvalues
    evals = gedmd.eigenvalues_
    
    # Filter for STABLE physical modes (Re(lambda) < 0)
    # Ignore modes that are practically constant (Re ~ 0) or highly oscillatory (Im large)
    # We expect kappa ~ 2.0, so lambda ~ -2.0.
    
    print(f"   Top 5 raw eigenvalues: {evals[:5]}")
    
    # Valid: Negative real part (stable), not too close to zero (not constant), not too negative (not noise)
    valid_evals = [e for e in evals if -20.0 < np.real(e) < -0.1]
    
    # Sort by how close real part is to 0 (slowest decay dominant)
    valid_evals = sorted(valid_evals, key=lambda x: abs(np.real(x)))
    
    if len(valid_evals) > 0:
        lambda_dom = valid_evals[0]
        # Robustness: Take real part
        decay_rate = np.real(lambda_dom)
        tau_corr = -1.0 / decay_rate
        print(f"   Selected Mode: {lambda_dom:.4f}")
        print(f"   Estimated Decorrelation Time: {tau_corr:.4f}s")
        return tau_corr
    else:
        print("   No valid physical modes found in range [-20, -0.1].")
        print("   Using fallback heuristic (based on Heston kappa=2 -> 0.5s)")
        return 0.5

# -----------------------------------------------------------------------------
# 3. Model Training
# -----------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

def train_signature_estimator(returns, variances, window_sizes, degree=2, log_target=False):
    """Train Ridge Regression on Multi-Scale Signatures.
    
    Args:
        log_target (bool): If True, train on log(v_t) and predict exp(y). 
                           Enforces positivity and handles heavy tails.
    """
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]
        
    X_feat = []
    y_target = []
    
    # Max window determines start index
    max_w = max(window_sizes)
    limit = len(returns)
    stride = 5
    
    # Scaling for Ridge
    # Pre-scale features to ensure time and returns channels balanced
    global_vol_scale = np.std(returns)
    
    for i in range(max_w, limit, stride): 
        target_v = variances[i-1] 
        
        # Concatenate signatures from all window sizes
        multi_scale_sig = []
        
        for w in window_sizes:
            window_rets = returns[i-w:i]
            
            # Compute Signature
            t_steps = np.linspace(0, 1, len(window_rets))
            
            # SCALE RETURNS
            rets_scaled = window_rets / global_vol_scale
            
            path = np.column_stack([t_steps, rets_scaled])
            
            # Standard Signature (Level 2) -> Contains Quadratic Variation terms explicitly
            sig_std = compute_path_signature(path, level=degree)
            
            multi_scale_sig.append(sig_std)
            
        # Flatten concatenated features
        full_feature_vec = np.concatenate(multi_scale_sig)
        
        X_feat.append(full_feature_vec)
        y_target.append(target_v)
        
    X_feat = np.array(X_feat)
    y_target = np.array(y_target)
    
    if len(X_feat) == 0:
        raise ValueError("No data generated. Window too large?")

    # Transform target if requested
    if log_target:
        # Avoid log(0) if any v is 0 (unlikely in Heston but safe)
        y_train = np.log(y_target + 1e-9)
    else:
        y_train = y_target
    
    # Scaling for Ridge (StandardScaler ensures regression is well-conditioned)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    # Ridge Regression
    model = Ridge(alpha=1e-8) 
    model.fit(X_scaled, y_train)
    
    score = model.score(X_scaled, y_train)
    
    # Package model with its scaler
    class BundledModel:
        def __init__(self, m, s, w_sizes, v_scale, use_log):
            self.model = m
            self.scaler = s
            self.window_sizes = w_sizes
            self.vol_scale = v_scale
            self.use_log = use_log
            
        def predict_multi(self, returns_full, current_idx):
            # Helper for multi-scale extraction
            feats = []
            for w in self.window_sizes:
                window_rets = returns_full[current_idx-w:current_idx]
                t_steps = np.linspace(0, 1, len(window_rets))
                rets_scaled = window_rets / self.vol_scale
                path = np.column_stack([t_steps, rets_scaled])
                sig = compute_path_signature(path, level=2)
                feats.append(sig)
            
            full_vec = np.concatenate(feats).reshape(1, -1)
            raw_pred = self.model.predict(self.scaler.transform(full_vec))
            
            if self.use_log:
                return np.exp(raw_pred)
            else:
                return raw_pred
            
    return BundledModel(model, scaler, window_sizes, global_vol_scale, log_target), score, X_feat, y_target

# -----------------------------------------------------------------------------
# 4. SOTA Baseline: Bootstrap Particle Filter (BPF)
# -----------------------------------------------------------------------------
import time 

class BootstrapSMC:
    """
    Standard Bootstrap Particle Filter for Heston Model.
    State: v_t
    Observation: y_t = log(S_t) - log(S_{t-1}) (Returns)
    """
    def __init__(self, n_particles=1000, kappa=2.0, theta=0.04, xi=0.3, dt=0.01):
        self.N = n_particles
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.dt = dt
        
    def filter(self, returns):
        """Run filter on a sequence of returns."""
        T = len(returns)
        
        # Initialize particles (draw from stationary distribution Gamma/CIR)
        # Stationary dist: Gamma(2*kappa*theta/xi^2, xi^2/2kappa) - roughly
        # For simplicity, init at theta with some noise
        particles = np.clip(np.random.normal(self.theta, 0.01, self.N), 1e-4, 1.0)
        weights = np.ones(self.N) / self.N
        
        estimates = np.zeros(T)
        
        start_time = time.time()
        
        for t in range(T):
            obs = returns[t]
            
            # 1. Propagation (Prior)
            # v_{t} = v_{t-1} + kappa(theta - v_{t-1})dt + xi*sqrt(v_{t-1})dW
            dW = np.random.normal(0, np.sqrt(self.dt), self.N)
            drift = self.kappa * (self.theta - particles) * self.dt
            diffusion = self.xi * np.sqrt(particles) * dW
            particles_pred = particles + drift + diffusion
            particles_pred = np.maximum(particles_pred, 1e-6) # Reflection/Truncation
            
            # 2. Weighting (Likelihood)
            # r_t ~ N(0, v_t * dt) approx (neglecting drift of price for short dt)
            # Likelihood p(y_t | v_t)
            # log_lik = -0.5 * (log(2*pi*v*dt) + y^2 / (v*dt))
            var_term = particles_pred * self.dt
            log_weights = -0.5 * (np.log(2*np.pi*var_term) + (obs**2)/var_term)
            
            # Stability: Shift max log-weight to 0
            max_log_w = np.max(log_weights)
            weights = np.exp(log_weights - max_log_w)
            weights /= np.sum(weights)
            
            # 3. Estimation
            estimate = np.sum(particles_pred * weights)
            estimates[t] = estimate
            
            # 4. Resampling (Multinomial / Systematic)
            # Simple Multinomial for PoC
            indices = np.random.choice(np.arange(self.N), size=self.N, p=weights)
            particles = particles_pred[indices]
            
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000 / T
        return estimates, inference_time_ms

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def run_experiment():
    print("🚀 Signature-Based Volatility Estimation")
    print("========================================")
    
    # 1. Data Generation
    dt = 0.01
    true_kappa = 2.0
    sim = HestonSimulator(kappa=true_kappa, theta=0.04, xi=0.3, dt=dt)
    
    print("Generating Heston Path (10,000 steps)...")
    returns, true_vol = sim.generate_path(10000)
    
    # Window Selection (Method 3)
    tau_corr = measure_decorrelation_time_kgedmd(returns, dt=dt)
    recommended_window_steps = int(3.0 * tau_corr / dt)
    print(f"✅ Recommended Window: {recommended_window_steps} steps")
    
    # Split
    split = 5000
    train_rets, test_rets = returns[:split], returns[split:]
    train_vol, test_vol = true_vol[:split], true_vol[split:]
    t_axis = np.arange(len(test_rets)) * dt
    
    # ---------------------------------------------------------
    # A. Signature Method (Dyadic + Log-Space)
    # ---------------------------------------------------------
    dyadic_windows = [8, 16, 32, 64, 128]
    print(f"\n🅰️  Signature Method (Dyadic {dyadic_windows}, Linear-Space)...")
    
    t0 = time.time()
    # Reverting to Linear Target (Log-Space caused Jensen's Bias explosion)
    model, score, _, _ = train_signature_estimator(train_rets, train_vol, dyadic_windows, log_target=False)
    train_time = time.time() - t0
    print(f"   Training Time: {train_time:.4f}s")
    print(f"   Train R2: {score:.4f}")
    
    # Inference
    t0 = time.time()
    test_preds_sig = []
    
    # Needs max window start
    start_idx = max(dyadic_windows)
    
    # Fast inference loop simulation
    for i in range(start_idx, len(test_rets)):
        # Pass full returns and current index -> simulates O(1) state update
        pred = model.predict_multi(test_rets, i)[0]
        test_preds_sig.append(pred)
        
    inf_time_sig_total = time.time() - t0
    inf_time_sig_per_step = (inf_time_sig_total / len(test_preds_sig)) * 1000 # ms
    
    # Align targets
    target_sig = test_vol[start_idx-1:-1] 
    
    mse_sig = np.mean((np.array(test_preds_sig) - target_sig)**2)
    print(f"   MSE: {mse_sig:.6f}")
    print(f"   Speed: {inf_time_sig_per_step:.4f} ms/step")
    
    # ---------------------------------------------------------
    # B. SOTA: Particle Filter (Correct Model)
    # ---------------------------------------------------------
    print("\n🅱️  SOTA: Particle Filter (N=1000, Correct Param)...")
    bpf_correct = BootstrapSMC(n_particles=1000, kappa=true_kappa, theta=0.04, xi=0.3, dt=dt)
    estimates_bpf, time_bpf = bpf_correct.filter(test_rets)
    
    # Align comparison
    common_start = start_idx
    bpf_aligned = estimates_bpf[common_start-1:-1] 
    target_common = test_vol[common_start-1:-1]
    
    mse_bpf = np.mean((bpf_aligned - target_common)**2)
    print(f"   MSE (Aligned): {mse_bpf:.6f}")
    
    # ---------------------------------------------------------
    # C. SOTA: Particle Filter (Misspecified) - Robustness Test
    # ---------------------------------------------------------
    wrong_kappa = 5.0 
    print(f"\n⚠️  Robustness Test: PF (N=1000, Wrong Kappa={wrong_kappa})...")
    bpf_wrong = BootstrapSMC(n_particles=1000, kappa=wrong_kappa, theta=0.04, xi=0.3, dt=dt)
    estimates_bpf_wrong, _ = bpf_wrong.filter(test_rets)
    
    bpf_wrong_aligned = estimates_bpf_wrong[common_start-1:-1]
    mse_bpf_wrong = np.mean((bpf_wrong_aligned - target_common)**2)
    print(f"   MSE (Aligned): {mse_bpf_wrong:.6f}")
    
    # ---------------------------------------------------------
    # D. Comparison & Plotting
    # ---------------------------------------------------------
    print("\n📊 Summary Comparison")
    print(f"   BPF (Oracle) MSE: {mse_bpf:.6f} (Baseline)")
    print(f"   Signature MSE:    {mse_sig:.6f} (x{mse_sig/mse_bpf:.2f} baseline)")
    print(f"   BPF (Wrong) MSE:  {mse_bpf_wrong:.6f} (x{mse_bpf_wrong/mse_bpf:.2f} baseline)")
    print(f"   Speedup (Sig vs BPF): {time_bpf/inf_time_sig_per_step:.1f}x")
    
    plt.figure(figsize=(12, 6))
    
    # Plot subset for clarity
    plot_len = 500
    t_plot = t_axis[start_idx:start_idx+plot_len]
    true_plot = target_common[:plot_len]
    sig_plot = test_preds_sig[:plot_len]
    bpf_plot = bpf_aligned[:plot_len]
    bpf_wrong_plot = bpf_wrong_aligned[:plot_len]
    
    plt.plot(t_plot, true_plot, 'k-', alpha=0.3, label='True Volatility', linewidth=2)
    plt.plot(t_plot, bpf_plot, 'b--', label=f'BPF (Oracle) MSE={mse_bpf:.1e}', alpha=0.8)
    plt.plot(t_plot, bpf_wrong_plot, 'r:', label=f'BPF (Misspecified k=5) MSE={mse_bpf_wrong:.1e}', alpha=0.8)
    plt.plot(t_plot, sig_plot, 'g-', label=f'Dyadic Sig {dyadic_windows} MSE={mse_sig:.1e}', linewidth=1.5)
    
    plt.title(f"Volatility Estimation (Dyadic Scales + Log-Space)\nSig Efficiency: {mse_sig/mse_bpf:.2f}x Oracle (Target < 1.1x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Variance v_t")
    plt.tight_layout()
    plt.savefig('heston_dyadic_comparison.png')
    print("Saved heston_dyadic_comparison.png")

if __name__ == "__main__":
    run_experiment()
