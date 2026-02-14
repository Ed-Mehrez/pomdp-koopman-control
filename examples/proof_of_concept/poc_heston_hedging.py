"""
Heston Option Hedging via KRONIC
================================
Learning the Variance-Optimal Hedge Ratio (Delta) using Signatures.

Physics:
    dS = r S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2
    corr(dW1, dW2) = rho

Target:
    Minimize Var(PnL_T)
    PnL_T = V_PF_T - max(S_T - K, 0)
    V_PF_t matches Option Price C(t) along the path.

Reference:
    Heston (1993) Semi-Analytical Solution for Call Price.
"""

import os
import sys
with open("debug.log", "w") as f: f.write("Starting imports...\n")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

with open("debug.log", "a") as f: f.write("Imports done. Starting experiment...\n")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from examples.proof_of_concept.signature_features import compute_log_signature
from src.online_koopman import OnlineKoopman

# -----------------------------------------------------------------------------
# 1. Heston Analytics (Ground Truth)
# -----------------------------------------------------------------------------
class HestonAnalytics:
    """Semi-Analytical Solution for Heston Call Option."""
    
    def __init__(self, kappa, theta, xi, rho, r=0.0):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r

    def bs_delta(self, S, K, T, v):
        if T <= 1e-4: return 1.0 if S > K else 0.0
        vol_eff = np.sqrt(max(1e-9, v))
        d1 = (np.log(S/K) + (self.r + 0.5 * v) * T) / (vol_eff * np.sqrt(T))
        return norm.cdf(d1)

    def bs_vega(self, S, K, T, v):
        if T <= 1e-4: return 0.0
        vol_eff = np.sqrt(max(1e-9, v))
        d1 = (np.log(S/K) + (self.r + 0.5 * v) * T) / (vol_eff * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def heston_mv_delta(self, S, K, T, v):
        # Minimum Variance Delta
        delta_bs = self.bs_delta(S, K, T, v)
        vega_bs = self.bs_vega(S, K, T, v)
        vol = np.sqrt(max(1e-9, v))
        if vol < 1e-4 or S < 1e-4: return delta_bs
        
        # dC/dv = Vega_BS * (1 / 2*vol)
        # Cov(dv, dS) = rho * xi * vol * S * vol * dt = rho * xi * v * S dt
        # Var(dS) = S^2 * v dt
        # Adj = (dC/dv * Cov) / Var
        #     = (Vega / 2v^0.5) * (rho xi v S) / (S^2 v)
        #     = (Vega * rho * xi) / (2 * S * v^0.5)
        
        adj = (self.rho * self.xi * vega_bs) / (2 * S * vol)
        return delta_bs + adj

    def price_call(self, S, K, T, v):
        """
        Approximate Heston Price using Black-Scholes with Instantaneous Variance.
        For a PoC training target, this is sufficient to test the Controller's
        ability to learn a hedge ratio that accounts for vol-of-vol noise if present,
        or at least recover the BS Delta.
        
        Ideally, we'd use the full Fourier Transform integration here.
        """
        if T <= 1e-4: return max(S - K, 0.0)
        
        vol_eff = np.sqrt(max(1e-9, v))
        d1 = (np.log(S/K) + (self.r + 0.5 * v) * T) / (vol_eff * np.sqrt(T))
        d2 = d1 - vol_eff * np.sqrt(T)
        price = S * norm.cdf(d1) - K * np.exp(-self.r*T) * norm.cdf(d2)
        return price

# -----------------------------------------------------------------------------
# 2. Simulation Environment
# -----------------------------------------------------------------------------
class HestonHedgingEnv:
    def __init__(self, kappa=2.0, theta=0.04, xi=0.3, rho=-0.9, dt=0.01):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho # Strong negative correlation leverage effect
        self.dt = dt
        self.pricer = HestonAnalytics(kappa, theta, xi, rho, r=0.0)
        
    def generate_episode(self, n_steps, S0=100.0, K=100.0, T_maturity=1.0):
        # Generate Path
        S = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        C = np.zeros(n_steps + 1)
        Delta_BS = np.zeros(n_steps + 1)
        Delta_MV = np.zeros(n_steps + 1)
        
        S[0] = S0
        v[0] = self.theta # Start at mean
        
        # Time array
        times = np.linspace(0, T_maturity, n_steps + 1)
        dt = times[1] - times[0]
        
        # Correlated Brownians
        Z1 = np.random.normal(0, 1, n_steps)
        Z2 = np.random.normal(0, 1, n_steps)
        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        for i in range(n_steps):
            v_curr = max(1e-6, v[i])
            S_curr = S[i]
            t_curr = times[i]
            tau = T_maturity - t_curr
            
            # Ground Truths
            C[i] = self.pricer.price_call(S_curr, K, tau, v_curr)
            Delta_BS[i] = self.pricer.bs_delta(S_curr, K, tau, v_curr)
            Delta_MV[i] = self.pricer.heston_mv_delta(S_curr, K, tau, v_curr)
            
            # Dynamics
            dS = np.sqrt(v_curr) * S_curr * W1[i] * np.sqrt(dt)
            # Full truncation scheme for variance
            dv = self.kappa * (self.theta - v_curr) * dt + self.xi * np.sqrt(v_curr) * W2[i] * np.sqrt(dt)
            
            S[i+1] = S_curr + dS
            v[i+1] = max(1e-6, v_curr + dv)
            
        C[-1] = max(S[-1] - K, 0) # Payoff
        Delta_BS[-1] = 0.0 # Expired
        Delta_MV[-1] = 0.0
        
        return S, v, C, Delta_BS, Delta_MV, times

# -----------------------------------------------------------------------------
# 3. Particle Filter Hedger (Benchmark)
# -----------------------------------------------------------------------------
class BootstrapSMC:
    """Standard Bootstrap Particle Filter for Heston Volatility."""
    def __init__(self, n_particles=1000, kappa=2.0, theta=0.04, xi=0.3, dt=0.01):
        self.N = n_particles
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.dt = dt
        self.particles = np.clip(np.random.normal(theta, 0.01, self.N), 1e-4, 1.0)
        self.weights = np.ones(self.N) / self.N
        
    def update(self, return_obs):
        # 1. Propagation
        dW = np.random.normal(0, np.sqrt(self.dt), self.N)
        drift = self.kappa * (self.theta - self.particles) * self.dt
        diffusion = self.xi * np.sqrt(self.particles) * dW
        particles_pred = self.particles + drift + diffusion
        particles_pred = np.maximum(particles_pred, 1e-6)
        
        # 2. Weighting (Likelihood: r ~ N(0, v*dt))
        var_term = particles_pred * self.dt
        log_weights = -0.5 * (np.log(2*np.pi*var_term) + (return_obs**2)/var_term)
        max_log = np.max(log_weights)
        weights = np.exp(log_weights - max_log)
        weights /= np.sum(weights)
        
        # 3. Estimation
        v_hat = np.sum(particles_pred * weights)
        
        # 4. Resampling
        indices = np.random.choice(np.arange(self.N), size=self.N, p=weights)
        self.particles = particles_pred[indices]
        
        return v_hat

class BPFHedger:
    def __init__(self, pricer, kappa=2.0, theta=0.04, xi=0.3, dt=0.01):
        self.pf = BootstrapSMC(n_particles=1000, kappa=kappa, theta=theta, xi=xi, dt=dt)
        self.pricer = pricer
        
    def predict_delta(self, S, K, T, return_obs):
        # Update filter
        v_hat = self.pf.update(return_obs)
        # Compute MV Delta using estimated vol
        return self.pricer.heston_mv_delta(S, K, T, v_hat)

class LMSHedger:
    """Least Mean Squares (SGD) Online Delta Learner."""
    def __init__(self, n_features, learning_rate=0.001, scaler_alpha=0.001, weight_decay=0.0):
        self.n = n_features
        self.lr = learning_rate
        self.alpha = scaler_alpha # Tunable scaler rate
        self.wd = weight_decay    # L2 Regularization
        self.w = np.zeros(n_features)
        
        self.mean = np.zeros(n_features) 
        self.std = np.ones(n_features)
        self.n_samples = 0
        
    def predict(self, z):
        # Normalize
        z_norm = (z - self.mean) / (self.std + 1e-6)
        raw_pred = np.dot(self.w, z_norm)
        # CRITICAL: Clip prediction to reasonable Delta range prevents explosion
        return np.clip(raw_pred, -1.0, 2.0)
        
    def update(self, z, target):
        self.n_samples += 1
        if self.n_samples == 1:
            self.mean = z
            self.std = np.ones_like(z)
        else:
            # Slow adaptation of stats
            self.mean = (1-self.alpha)*self.mean + self.alpha*z
            self.std = (1-self.alpha)*self.std + self.alpha*np.abs(z - self.mean)
            
        z_norm = (z - self.mean) / (self.std + 1e-6)
        
        pred = np.dot(self.w, z_norm)
        err = pred - target
        
        # SGD Step: w_new = w - lr * (err * x + wd * w)
        grad = err * z_norm + self.wd * self.w
        self.w -= self.lr * grad 
        return
        
        return error**2

def execute_online_run(lr, alpha, n_episodes, env, offline_model, X_off, quiet=False):
    """Executes a single online learning run."""
    print(f"👉 Run: LR={lr}, Alpha={alpha}, Eps={n_episodes}", flush=True)
    
    # Init Hedger
    z_sample = X_off[0]
    # Augment features: z_raw + K@z_raw + Bias
    # Original: len(z_sample) + 1
    # New: len(z_sample) * 2 + 1
    dim_raw = len(z_sample)
    n_features = dim_raw * 2 + 1 
    
    lms_hedger = LMSHedger(n_features, learning_rate=lr, scaler_alpha=alpha)
    
    # Online Koopman (Recursive EDMD)
    # Forgetting factor 0.995 (slow adaptation for stability)
    rk = OnlineKoopman(dim_raw, forgetting_factor=0.995, lambda_reg=1.0)
    prev_z_raw = None
    
    # Config
    steps_per_ep = 100
    window_len = 50
    S0 = 100.0
    K = 100.0
    
    # --- Burn-In Phase ---
    n_burn_in = 50
    if not quiet: print(f"🔥 Burn-In Phase ({n_burn_in} eps)...", flush=True)
    
    for ep in range(n_burn_in):
        prev_z_raw = None # Reset state for new episode
        S, v, C, Delta_BS, Delta_MV, times = env.generate_episode(steps_per_ep, S0=100.0, K=100.0, T_maturity=1.0)
        history_buffer = [np.log(S[0])] * window_len
        for t in range(steps_per_ep - 1):
            dS = S[t+1] - S[t]
            dC = C[t+1] - C[t]
            
            t_grid = np.linspace(0, 1, len(history_buffer))
            path = np.column_stack([t_grid, np.array(history_buffer)])
            sig = compute_log_signature(path, level=2)
            z_raw = np.concatenate([sig, [np.log(S[t]/K), 1.0 - times[t]]])
            
            # --- Koopman Update & Augmentation ---
            if prev_z_raw is not None:
                rk.update(prev_z_raw, z_raw)
                
            z_pred = rk.predict(z_raw)
            # Augmented State: [Features, Predicted_Next_Features, Bias]
            z = np.concatenate([z_raw, z_pred, [1.0]])
            
            prev_z_raw = z_raw.copy()
            
            u_star = Delta_BS[t] if abs(dS) < 1e-3 else dC / dS
            u_star = np.clip(u_star, -0.5, 1.5)
            lms_hedger.update(z, u_star)
            
            history_buffer.pop(0)
            history_buffer.append(np.log(S[t+1]))
            
    # --- Eval Phase ---
    if not quiet: print(f"📉 Evaluation Phase ({n_episodes} eps)...", flush=True)
    
    cum_var_on = 0
    hist_on = []
    # Track baselines only if not quiet (for final plot)
    hist_bs, hist_off = [], []
    cum_var_bs, cum_var_off = 0, 0
    
    steps_total = 0
    
    for ep in range(n_episodes):
        prev_z_raw = None # Reset state for new episode
        S, v, C, Delta_BS, Delta_MV, times = env.generate_episode(steps_per_ep, S0=100.0, K=100.0, T_maturity=1.0)
        history_buffer = [np.log(S[0])] * window_len
        
        for t in range(steps_per_ep - 1):
            steps_total += 1
            dS = S[t+1] - S[t]
            dC = C[t+1] - C[t]
            
            # --- Features ---
            t_grid = np.linspace(0, 1, len(history_buffer))
            path = np.column_stack([t_grid, np.array(history_buffer)])
            sig = compute_log_signature(path, level=2)
            z_raw = np.concatenate([sig, [np.log(S[t]/K), 1.0 - times[t]]])
            
            # --- Koopman Update & Augmentation ---
            if prev_z_raw is not None:
                rk.update(prev_z_raw, z_raw)
            
            z_pred = rk.predict(z_raw)
            z_online = np.concatenate([z_raw, z_pred, [1.0]])
            
            prev_z_raw = z_raw.copy()
            
            # --- Predictions ---
            u_on = lms_hedger.predict(z_online)
            
            # --- PnL ---
            err_on = (u_on * dS - dC)**2
            cum_var_on += err_on
            hist_on.append(cum_var_on / steps_total)
            
            if not quiet:
                u_bs = Delta_BS[t]
                pred_off = offline_model.predict([z_raw])
                u_off = pred_off[0][0] if pred_off.ndim > 1 else pred_off[0]
                
                err_bs = (u_bs * dS - dC)**2
                err_off = (u_off * dS - dC)**2
                
                cum_var_bs += err_bs
                cum_var_off += err_off
                
                hist_bs.append(cum_var_bs / steps_total)
                hist_off.append(cum_var_off / steps_total)
            
            # --- Online Update ---
            u_star = Delta_BS[t] if abs(dS) < 1e-3 else dC / dS
            u_star = np.clip(u_star, -0.5, 1.5)
            # Update LMS
            lms_hedger.update(z_online, u_star)
            
            history_buffer.pop(0)
            history_buffer.append(np.log(S[t+1]))
            
    final_var = hist_on[-1]
    if quiet:
        return final_var
    else:
        return hist_on, hist_bs, hist_off

def run_tuning_and_final():
    print("🚀 Heston Hedging: Online Learning Tuning", flush=True)
    
    # Common Env Setup
    dt = 0.01
    env = HestonHedgingEnv(dt=dt, kappa=2.0, theta=0.04, xi=0.5, rho=-0.9)
    S0, K = 100.0, 100.0
    steps_per_ep = 100
    window_len = 50
    
    # 1. Train Offline Baseline (Once)
    print("🧠 Training Offline Baseline (50 eps)...", flush=True)
    X_off, U_off = [], []
    for _ in range(50):
        S, v, C, Delta_BS, Delta_MV, times = env.generate_episode(steps_per_ep, S0=np.random.uniform(80, 120), K=K, T_maturity=1.0)
        history_buffer = [np.log(S[0])] * window_len
        for t in range(steps_per_ep - 1):
            log_S_window = np.array(history_buffer)
            t_grid = np.linspace(0, 1, len(log_S_window))
            path = np.column_stack([t_grid, log_S_window])
            sig = compute_log_signature(path, level=2)
            z = np.concatenate([sig, [np.log(S[t]/K), 1.0 - times[t]]])
            dS = S[t+1] - S[t]
            dC = C[t+1] - C[t]
            u_star = Delta_BS[t] if abs(dS) < 1e-3 else dC / dS
            u_star = np.clip(u_star, -0.5, 1.5)
            X_off.append(z)
            U_off.append([u_star])
            history_buffer.pop(0)
            history_buffer.append(np.log(S[t+1]))
            
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    offline_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    offline_model.fit(X_off, U_off)
    
    # 2. Grid Search
    print("\n🔎 Starting Grid Search (500 eps per config)...", flush=True)
    lrs = [0.001, 0.005, 0.01, 0.05]
    alphas = [0.0001, 0.001, 0.01]
    
    best_var = float('inf')
    best_config = (0.005, 0.001)
    
    for lr in lrs:
        for alpha in alphas:
            var = execute_online_run(lr, alpha, 500, env, offline_model, X_off, quiet=True)
            print(f"   [Result] LR={lr}, Alpha={alpha} -> Var={var:.4f}")
            if var < best_var:
                best_var = var
                best_config = (lr, alpha)
                
    print(f"\n🏆 Best Config: LR={best_config[0]}, Alpha={best_config[1]} (Var={best_var:.4f})")
    
    # 3. Final Run
    print("\n🎬 Running Final Validation (2000 eps)...", flush=True)
    hist_on, hist_bs, hist_off = execute_online_run(best_config[0], best_config[1], 2000, env, offline_model, X_off, quiet=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(hist_bs, 'k--', label='Black-Scholes')
    plt.plot(hist_off, 'b-', label='Offline Sig')
    plt.plot(hist_on, 'r-', label=f'Online Sig (LR={best_config[0]})')
    plt.xlabel('Steps')
    plt.ylabel('Average PnL Variance (Cumulative)')
    plt.title('Online Hedging Adaptation (Tuned)')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('heston_online_learning.png')
    
    with open("debug.log", "a") as f: 
        f.write(f"Tuning Complete. Best: {best_config}. Final Var: {hist_on[-1]:.4f}\n")
    print("📸 Saved heston_online_learning.png")

if __name__ == "__main__":
    run_tuning_and_final()
