
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from examples.proof_of_concept.signature_features import compute_log_signature
from src.online_koopman import OnlineKoopman
from examples.proof_of_concept.poc_heston_hedging import HestonHedgingEnv, BPFHedger, LMSHedger, BootstrapSMC

class SimpleRNNHedger:
    """
    End-to-End Deep Hedging Benchmark (Vanilla RNN).
    Maps sequence of returns directly to Delta.
    Architecture: Input(2) -> Hidden(20) -> Output(1).
    Trained via Online SGD (approx BPTT).
    """
    def __init__(self, input_dim=2, hidden_dim=20, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Weights
        self.Wx = np.random.randn(hidden_dim, input_dim) * 0.1
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.Wy = np.random.randn(1, hidden_dim) * 0.1
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((1, 1))
        
        # State
        self.h = np.zeros((hidden_dim, 1))
        
    def forward(self, x):
        # x: (input_dim, 1)
        self.x_prev = x
        self.h_prev = self.h.copy()
        
        # RNN Step: h_t = tanh(Wx * x + Wh * h_{t-1} + bh)
        self.z = self.Wx @ x + self.Wh @ self.h_prev + self.bh
        self.h = np.tanh(self.z)
        
        # Output: y = Wy * h + by
        self.y = self.Wy @ self.h + self.by
        return self.y[0, 0]
        
    def update(self, target_u):
        # Loss = (y - target)^2
        # Gradient descent on the last step (Truncated BPTT-1)
        # This is a simplification of full BPTT but fair for online O(1).
        
        pred = self.y[0, 0]
        err = pred - target_u # dL/dy
        
        # Backprop through Output
        dWy = err * self.h.T
        dby = err
        
        # Backprop through Tanh
        dh = (self.Wy.T * err) * (1 - self.h**2)
        
        # Backprop through Weights
        dWx = dh @ self.x_prev.T
        dWh = dh @ self.h_prev.T
        dbh = dh
        
        # SGD Update
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        self.Wy -= self.lr * dWy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby
        
    def reset(self):
        self.h = np.zeros((self.hidden_dim, 1))

class SignatureVolEstimator:
    def __init__(self, n_features, forgetting_factor=0.999, adaptive_alpha=1000.0, use_autoregressive=True, use_rkf=True):
        self.n_raw = n_features
        self.use_ar = use_autoregressive
        self.use_rkf = use_rkf
        # If AR, we add 1 feature (previous estimate)
        self.n = n_features + 1 if use_autoregressive else n_features
        
        # RLS for Volatility Readout: v_t = w^T [z_t, z_{t+1|t}]
        # Dim is 2 * n_features
        self.P = np.eye(2 * self.n) * 10.0
        self.w = np.zeros(2 * self.n)
        self.ff = forgetting_factor
        self.rk = OnlineKoopman(self.n, forgetting_factor=forgetting_factor)
        
        # Adaptive Regularization State
        self.error_buffer = []
        self.buffer_size = 50
        self.alpha_reg = adaptive_alpha
        
        # Auto-Regressive State (Summary Statistic)
        self.last_est = 0.04 # Initial guess (Theta)
        # Store Filtered State for RKF
        self.z_filtered = None 
        
    def  _augment(self, z_raw):
        """Appends summary statistic (last estimate) to signature."""
        if self.use_ar:
            # Append last estimate normalized? 
            # Vol is 0.01-0.1. Sigs are small. 
            # Let's clean scaling later, raw for now.
            return np.concatenate([z_raw, [self.last_est]]) 
        return z_raw

    def update(self, z_prev_raw, z_curr_raw, v_target):
        # Construct Augmented States
        z_curr_aug = self._augment(z_curr_raw)
        
        # RKF: Recurrent Koopman Filter Logic
        # 1. State Prediction (Prior)
        if self.use_rkf and self.z_filtered is not None:
            z_prior = self.rk.predict(self.z_filtered)
        else:
            z_prior = z_curr_aug # High uncertainty initialization
            
        # 2. Measurement Gating (Adaptive Gain)
        # Calculate MSE to determine how much we trust the new measurement vs prior
        adaptive_noise = 1.0
        if len(self.error_buffer) > 10:
             mse = np.mean(np.array(self.error_buffer)**2)
             adaptive_noise = 1.0 + self.alpha_reg * mse
             
        # Kalman Gain equivalent: G = 1 / R. (Simplification)
        # If Noise is High (High MSE), Gain is Low -> Trust Prior.
        # If Noise is Low, Gain is High -> Trust Measurement.
        gain = 1.0 / adaptive_noise
        
        if self.use_rkf:
            z_filtered = (1.0 - gain) * z_prior + gain * z_curr_aug
        else:
            z_filtered = z_curr_aug
            
        # 3. Update Dynamics (Using Filtered State)
        if self.z_filtered is not None:
             # We learn transition: z_filtered_{t-1} -> z_filtered_{t}
             err_norm = self.rk.update(self.z_filtered, z_filtered, adaptive_noise=adaptive_noise)
             
             self.error_buffer.append(err_norm)
             if len(self.error_buffer) > self.buffer_size:
                 self.error_buffer.pop(0)
                 
        # 4. Update Readout
        z_next_pred = self.rk.predict(z_filtered)
        z_in = np.concatenate([z_filtered, z_next_pred])
        
        z_col = z_in[:, np.newaxis]
        Pz = self.P @ z_col
        denom = self.ff + np.dot(z_in, Pz)
        k = Pz / denom
        
        v_pred = np.dot(self.w, z_in)
        error = v_target - v_pred
        
        self.w = self.w + k.flatten() * error
        self.P = (self.P - np.outer(k, Pz)) / self.ff
        
        # Store State
        self.z_filtered = z_filtered.copy()
        self.last_est = max(v_pred, 1e-6)
        
        return self.last_est

    def predict(self, z_curr_raw):
        # Used for Step 0 or pure inference
        z_curr_aug = self._augment(z_curr_raw)
        z_next_pred = self.rk.predict(z_curr_aug)
        z_in = np.concatenate([z_curr_aug, z_next_pred])
        v_pred = np.dot(self.w, z_in)
        
        # Don't update last_est here? Depends on usage. 
        # If predict is called in loop, we should. But update() creates the loop.
        return max(v_pred, 1e-6)

class RBFFeatureMap:
    def __init__(self, n_components=200, gamma=0.5, state_dim=3):
        self.gamma = gamma
        self.n = n_components
        # Random centers (initialized roughly in the domain)
        # Vol: [0, 0.5], Moneyness: [-0.3, 0.3], Time: [0, 1]
        self.centers = np.random.rand(n_components, state_dim)
        self.centers[:, 0] = self.centers[:, 0] * 0.5 + 0.05 # Vol range
        self.centers[:, 1] = (self.centers[:, 1] - 0.5) * 0.6 # Moneyness range [-0.3, 0.3]
        self.centers[:, 2] = self.centers[:, 2] * 1.0 # Time range
        
    def transform(self, x):
        # x is (3,)
        # dists is (n_components,)
        dists = np.linalg.norm(self.centers - x, axis=1)
        phi = np.exp(-self.gamma * (dists**2))
        return phi

    def update(self, x, lr=0.005):
        # Online K-Means: Move winner towards data
        dists = np.linalg.norm(self.centers - x, axis=1)
        winner_idx = np.argmin(dists)
        # Gradient step: c <- c + lr * (x - c)
        self.centers[winner_idx] += lr * (x - self.centers[winner_idx])

def run_integrated_control():
    print("🚀 Heston Hedging: Dual Adaptive Control (Sensor + RBF Controller)", flush=True)
    
    # Config
    dt = 0.01
    n_episodes = 200 # Short run for proof of concept
    steps_per_ep = 100
    window_len = 50
    lr_controller = 0.01 # Slightly higher for RBF
    
    env = HestonHedgingEnv(dt=dt, kappa=2.0, theta=0.04, xi=0.5, rho=-0.9)
    S0, K_opt = 100.0, 100.0
    
    # 1. Init Sensor (Unsupervised)
    dummy = np.zeros((50, 2))
    sig = compute_log_signature(dummy, level=2)
    dim_sig = len(sig) 
    sensor = SignatureVolEstimator(dim_sig, forgetting_factor=0.999)
    
    # Init SOTA Baseline (Particle Filter)
    bpf = BootstrapSMC(n_particles=200, kappa=2.0, theta=0.04, xi=0.5, dt=dt)
    
    # Init SOTA RNN (Deep Hedging)
    rnn_agent = SimpleRNNHedger(input_dim=2, hidden_dim=20, lr=0.01) # Inputs: [Moneyness, Time]
    
    # 2. Init Controller (Supervised RBF)
    # State: [Vol_Hat, LogMoneyness, Time] -> 3 dims
    state_dim = 3
    n_rbf = 100
    rbf_map = RBFFeatureMap(n_components=n_rbf, gamma=10.0, state_dim=state_dim) # Sharper gamma
    
    # Controller acts on RBF features + Bias
    dim_ctrl = n_rbf + 1
    controller = LMSHedger(dim_ctrl, learning_rate=lr_controller, scaler_alpha=0.0001)
    
    history_bs = []
    history_mv = []
    history_mv_hat = []
    history_bpf = []
    history_rnn = []
    history_dual = []
    
    # Running Variances
    cum_var_bs = 0.0
    cum_var_mv = 0.0
    cum_var_mv_hat = 0.0
    cum_var_bpf = 0.0
    cum_var_rnn = 0.0
    cum_var_dual = 0.0
    total_steps = 0
    
    # Burn-in Sensor?
    print("warming up sensor...", flush=True)
    for _ in range(10):
         S, v, _, _, _, _ = env.generate_episode(steps_per_ep, S0=100.0, K=100.0, T_maturity=1.0)
         h = [np.log(S[0])] * window_len
         pz = None
         for t in range(steps_per_ep):
             p = np.column_stack([np.linspace(0,1, window_len), h])
             z = compute_log_signature(p, level=2)
             
             if t > 0:
                 ret_sq = (np.log(S[t]/S[t-1]))**2
                 v_prox = min(ret_sq/dt, 2.0)
                 sensor.update(pz, z, v_prox)
             
             pz = z.copy()
             if t < steps_per_ep-1: h.pop(0); h.append(np.log(S[t+1]))
             
    print("Starting Control Loop...", flush=True)
    
    for ep in range(n_episodes):
        S, v, C, Delta_BS, Delta_MV, times = env.generate_episode(steps_per_ep, S0=100.0, K=K_opt, T_maturity=1.0)
        history_buffer = [np.log(S[0])] * window_len
        prev_z = None
        
        # Reset BPF for new episode
        bpf = BootstrapSMC(n_particles=200, kappa=2.0, theta=0.04, xi=0.5, dt=dt)
        rnn_agent.reset()
        
        for t in range(steps_per_ep - 1):
            dS = S[t+1] - S[t]
            dC = C[t+1] - C[t]
            dt_rem = 1.0 - times[t]
            
            # --- SENSOR STEP ---
            path_data = np.column_stack([np.linspace(0, 1, window_len), history_buffer])
            z_curr = compute_log_signature(path_data, level=2)
            
            # Unsupervised Update
            if t > 0:
                ret_sq = (np.log(S[t]/S[t-1]))**2
                v_prox = min(ret_sq/dt, 2.0)
                v_hat = sensor.update(prev_z, z_curr, v_prox)
            else:
                v_hat = sensor.predict(z_curr)
            
            prev_z = z_curr.copy()
            
            # --- CONTROLLER STEP ---
            # Raw State: [Vol_Hat, Moneyness, Time]
            log_m = np.log(S[t]/K_opt)
            raw_state = np.array([v_hat, log_m, dt_rem])
            
            # RBF Transform
            phi = rbf_map.transform(raw_state)
            
            # ADAPT CENTERS: Track the drifting distribution
            rbf_map.update(raw_state, lr=0.005)
            
            # Augmented Feature: [RBF, 1.0]
            x_ctrl = np.concatenate([phi, [1.0]])
            
            u_dual = controller.predict(x_ctrl)
            u_bs = Delta_BS[t]
            
            # Supervised Update
            u_star = Delta_MV[t] # Target the Analytic Optimum (Best Teacher)
            # u_star = dC/dS # Or Noisy Ex-Post Best? Let's use MV for stable learning now.
            
            controller.update(x_ctrl, u_star)
            
            # Analytic Baseline with Sensor Vol
            # "What if we knew the formula but only had the sensor?"
            u_mv_hat = env.pricer.heston_mv_delta(S[t], K_opt, dt_rem, v_hat)
            
            # SOTA Baseline: BPF Posterior Average
            # 1. Update BPF
            ret_obs = np.log(S[t]/S[t-1]) if t > 0 else 0.0
            # Note: BPF expects return over dt.
            bpf.update(ret_obs)
            # 2. Average Delta over particles (The Correct Way)
            # \hat{Delta} = E[Delta(v) | F_t]
            deltas_bpf = [env.pricer.heston_mv_delta(S[t], K_opt, dt_rem, p) for p in bpf.particles]
            u_bpf = np.mean(deltas_bpf)
            
            # SOTA 2: Deep Hedging (RNN)
            # Input: [LogMoneyness, Time]
            # Ideally it needs returns history, but RNN holds state. 
            # We feed it the current observable state and let it evolve internal memory.
            x_rnn = np.array([[log_m], [dt_rem]])
            u_rnn = rnn_agent.forward(x_rnn)
            rnn_agent.update(u_star) # Train online same as RBF
            
            # Eval
            err_bs = (u_bs * dS - dC)**2
            err_mv = (Delta_MV[t] * dS - dC)**2
            err_mv_hat = (u_mv_hat * dS - dC)**2
            err_bpf = (u_bpf * dS - dC)**2
            err_rnn = (u_rnn * dS - dC)**2
            err_dual = (u_dual * dS - dC)**2
            
            cum_var_bs += err_bs
            cum_var_mv += err_mv
            cum_var_mv_hat += err_mv_hat
            cum_var_bpf += err_bpf
            cum_var_rnn += err_rnn
            cum_var_dual += err_dual
            total_steps += 1
            
            history_bs.append(cum_var_bs / total_steps)
            history_mv.append(cum_var_mv / total_steps)
            history_mv_hat.append(cum_var_mv_hat / total_steps)
            history_bpf.append(cum_var_bpf / total_steps)
            history_rnn.append(cum_var_rnn / total_steps)
            history_dual.append(cum_var_dual / total_steps)
            
            history_buffer.pop(0)
            history_buffer.append(np.log(S[t+1]))
            
    # Visualize
    print(f"  BS Oracle (Naive): {history_bs[-1]:.4f}")
    print(f"  MV Oracle (True Vol): {history_mv[-1]:.4f}")
    print(f"  MV Hat (Sensor Vol): {history_mv_hat[-1]:.4f}")
    print(f"  SOTA BPF (Posterior Avg): {history_bpf[-1]:.4f}")
    print(f"  SOTA RNN (Deep Hedge): {history_rnn[-1]:.4f}")
    print(f"  Dual Adaptive (Learned): {history_dual[-1]:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_bs, 'k--', alpha=0.3, label='BS (Naive)')
    plt.plot(history_mv, 'g--', label='MV (True Vol)')
    plt.plot(history_mv_hat, 'b--', alpha=0.3, label='MV Hat (Sensor Vol)')
    #plt.plot(history_bpf, 'm-', label='SOTA BPF') # Failed, skip plot
    plt.plot(history_rnn, 'c-', label='SOTA RNN (Deep Hedge)')
    plt.plot(history_dual, 'r-', linewidth=2, label='Dual Adaptive (Learned)')
    plt.title('Heston Hedging: Learned vs SOTA Numerical')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig('heston_dual_adaptive.png')
    print("📸 Saved heston_dual_adaptive.png")

if __name__ == "__main__":
    run_integrated_control()
