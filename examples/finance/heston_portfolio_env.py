import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import torch
import torch.optim as optim

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from sskf.online_path_features import RandomProjectionNystrom
from sskf.streaming_sig_kkf import LogSignatureState

class HestonPortfolioEnv:
    """
    Simulates a continuous-time market with a risk-free asset and a risky asset
    driven by Heston stochastic volatility.
    
    Agents must dynamically choose a portfolio fraction pi_t to maximize Expected
    CRRA Utility of terminal wealth U(W_T) = (W_T^(1-gamma)) / (1-gamma).
    """
    def __init__(self, T=1.0, N=100000, r=0.02, mu=0.08, kappa=5.0, theta=0.04, sigma=0.5, rho=-0.7, V0=0.04, W0=1.0, gamma_risk=2.0, sigma_noise=0.005):
        self.T = T
        self.N = N
        self.dt = T / N
        self.r = r
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
        self.V0 = V0
        self.W0 = W0
        self.gamma_risk = gamma_risk # Coefficient of Relative Risk Aversion
        
        # Microstructure noise standard deviation
        self.sigma_noise = sigma_noise
        
        self.times = np.linspace(0, T, N+1)
        self._simulate_market()

    def _simulate_market(self):
        """Pre-simulates the exact market paths."""
        np.random.seed(42)  # For reproducible benchmarking
        
        self.S_true = np.zeros(self.N + 1)
        self.V_true = np.zeros(self.N + 1)
        self.S_true[0] = 1.0
        self.V_true[0] = self.V0
        
        Z1 = np.random.randn(self.N)
        Z2 = np.random.randn(self.N)
        W_S = np.sqrt(self.dt) * Z1
        W_V = np.sqrt(self.dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
        
        for i in range(self.N):
            v_plus = max(self.V_true[i], 0)
            self.S_true[i+1] = self.S_true[i] + self.mu * self.S_true[i] * self.dt + np.sqrt(v_plus) * self.S_true[i] * W_S[i]
            self.V_true[i+1] = self.V_true[i] + self.kappa * (self.theta - v_plus) * self.dt + self.sigma * np.sqrt(v_plus) * W_V[i]
            
        self.log_prices_true = np.log(self.S_true)
        # Add bid-ask microstructure noise
        noise = np.random.normal(0, self.sigma_noise, self.N + 1)
        self.log_prices_observed = self.log_prices_true + noise
        
        self.returns_observed = np.diff(self.log_prices_observed)
        self.returns_true = np.diff(self.log_prices_true)

    def evaluate_policy(self, policy_fn, name="Agent"):
        """
        Evaluates a portfolio function pi_t = policy_fn(t, obs_history).
        Simulates the continuous-time self-financing wealth equation:
        dW_t = W_t [r + pi_t(mu - r)] dt + W_t pi_t sqrt(V_t) dW_t^S
        """
        W = np.zeros(self.N + 1)
        W[0] = self.W0
        allocations = np.zeros(self.N)
        
        # We need the true Brownian increments of the asset to correctly integrate wealth
        # dS/S = mu dt + sqrt(V) dW_S  =>  sqrt(V) dW_S = dS/S - mu dt
        # For numeric stability and exactness, we use the simulated returns
        asset_returns = self.S_true[1:] / self.S_true[:-1] - 1.0
        
        t0 = time.time()
        for i in range(self.N):
            t = self.times[i]
            
            # The policy relies on the OBSERVABLE history up to time i.
            # (To keep it fast, the policy_fn should maintain its own online state internally if needed)
            pi_t = policy_fn(i, self)
            
            # Clip leverage to reasonable bounds [0, 4] to prevent total bankruptcy on noise
            # (Softened from 2.0 to show the structural allocation curve rather than a flat "bang-bang" roof)
            pi_t = np.clip(pi_t, 0.0, 4.0)
            allocations[i] = pi_t
            
            # Wealth evolution
            # dW_t/W_t = r dt + pi_t * (dS_t/S_t - r dt)
            dw_over_w = self.r * self.dt + pi_t * (asset_returns[i] - self.r * self.dt)
            W[i+1] = W[i] * (1.0 + dw_over_w)
            
            # Bankruptcy condition
            if W[i+1] <= 0:
                W[i+1:] = 1e-10
                break
                
        duration = time.time() - t0
        
        # Calculate Terminal CRRA Utility
        W_T = W[-1]
        if self.gamma_risk == 1.0:
            utility = np.log(W_T)
        else:
            utility = (W_T**(1.0 - self.gamma_risk)) / (1.0 - self.gamma_risk)
            
        print(f"[{name}] Evaluated in {duration:.2f}s | Terminal Wealth: {W_T:.4f} | CRRA Utility: {utility:.4f}")
        
        return W, allocations, utility


# --- Baselines and Oracle ---

def static_policy(i, env):
    """
    Naive Mean-Variance Optimal allocation assuming constant long-term volatility.
    pi_t = (mu - r) / (gamma * theta)
    """
    return (env.mu - env.r) / (env.gamma_risk * env.theta)

def oracle_policy(i, env):
    """
    Oracle Merton: Perfectly observes the true hidden Volatility V_t.
    No hedging demand, just myopic Merton for baseline.
    pi_t = (mu - r) / (gamma * V_t)
    """
    # Use max to prevent division by zero near 0 variance
    v_t = max(env.V_true[i], 1e-5)
    return (env.mu - env.r) / (env.gamma_risk * v_t)


class KoopmanSignaturePolicy:
    """
    Nystrom Signature Koopman Controller.
    Filters the unobservable Volatility dynamically using only noisy ticks,
    then executes the Merton continuous portfolio allocation.
    """
    def __init__(self, window=1000, n_landmarks=100, dt=1e-5):
        self.window = window
        self.dt = dt
        
        self.extractor = RandomProjectionNystrom(
            dim=2, depth=2, projection_dim=200, n_landmarks=n_landmarks,
            use_leadlag=True, kernel_bandwidth=1.0, feature_mode='joint'
        )
        
        # RLS (Kalman) State for Observation mapping
        self.A = np.zeros(n_landmarks) 
        self.P = np.eye(n_landmarks) * 1.0 
        self.ff = 1.0 - (1.0 / window)  # principled forget factor
        
        self.path_buffer = np.zeros((window, 2))
        self.path_buffer[:, 0] = np.arange(window) * dt
        
        self.k_n = max(1, int(np.sqrt(window))) # A&J Pre-average optimal block size
        self.last_pred_V = 0.04 # fallback prior
        
        # We need to compute Bipower Variation online for the RLS target
        self.recent_returns = np.zeros(window)
        self.recent_cross_abs = np.zeros(window)
        self.bv_sum = 0.0
        
    def reset_learning_state(self):
        """Resets the online state buffers for a fresh evaluation pass."""
        self.A = np.zeros(self.extractor.n_landmarks) 
        self.P = np.eye(self.extractor.n_landmarks) * 1.0 
        
        self.path_buffer = np.zeros((self.window, 2))
        self.path_buffer[:, 0] = np.arange(self.window) * self.dt
        
        self.recent_returns = np.zeros(self.window)
        self.recent_cross_abs = np.zeros(self.window)
        self.bv_sum = 0.0
        self.last_pred_V = 0.04
        
        # We do NOT reset the extractor landmarks because we want to use the same basis!

    def __call__(self, i, env):
        # We start trading only after we have enough history to fill the window
        if i < self.window:
            return (env.mu - env.r) / (env.gamma_risk * env.theta) # Static prior fallback
            
        ret_t = env.returns_observed[i]
        
        # 1. Update Bipower Variation rolling sum (O(1) update)
        old_ret = self.recent_returns[0]
        self.recent_returns[:-1] = self.recent_returns[1:]
        self.recent_returns[-1] = ret_t
        
        if i > 1:
            new_cross = np.abs(self.recent_returns[-1]) * np.abs(self.recent_returns[-2])
            old_cross = self.recent_cross_abs[0]
            self.recent_cross_abs[:-1] = self.recent_cross_abs[1:]
            self.recent_cross_abs[-1] = new_cross
            
            self.bv_sum = self.bv_sum + new_cross - old_cross
            
        # 2. Update Path Buffer
        self.path_buffer[:-1, 1] = self.path_buffer[1:, 1]
        self.path_buffer[-1, 1] = self.path_buffer[-2, 1] + ret_t
        
        # 3. Kalman Nystrom Filter Step (Once every k_n ticks as theoretically optimal)
        if i % self.k_n == 0:
            bv_target_val = (np.pi / 2.0) * self.bv_sum
            # A&J Pre-averaging trick structural fallback inside block loop
            target_std = np.sqrt(max(bv_target_val / (self.window * self.dt), 1e-10))
            
            if len(self.extractor.landmarks) < self.extractor.n_landmarks:
                self.extractor.update(self.path_buffer)
                self.last_pred_V = target_std**2
            else:
                phi = self.extractor.nystrom_embedding(self.path_buffer)
                
                pred_std = self.A @ phi
                error = target_std - pred_std
                
                Pz = self.P @ phi
                gain = Pz / (self.ff + phi @ Pz)
                self.A = self.A + error * gain
                self.P = (self.P - np.outer(gain, Pz)) / self.ff
                
                self.last_pred_V = (pred_std)**2
                
            if np.random.rand() < 0.05:
                self.extractor.update(self.path_buffer)
                
        # 4. Final Koopman Allocation 
        V_koopman = max(self.last_pred_V, 1e-5)
        pi_t = (env.mu - env.r) / (env.gamma_risk * V_koopman)
        return pi_t

class TrueHedgingKoopmanPolicy(KoopmanSignaturePolicy):
    """
    Extends the Koopman policy to extract True Intertemporal Hedging Demand.
    Projects the CRRA Value Function onto the Koopman Eigenfunctions via EDMD,
    using the analytical spatial gradient of the continuous operator to explicitly
    solve the HJB Cross-Derivative without Riccati equations.
    """
    def __init__(self, window=1000, n_landmarks=100, dt=1e-5):
        super().__init__(window, n_landmarks, dt)
        
        self.phi_history = []
        self.dphi_history = []
        self.cov_history = []
        self.v_history = []
        self.w_value = np.zeros(n_landmarks) # Value function projection weights
        self.is_trained = False
        
        self.cov_ewma = np.zeros(n_landmarks) # Covariance between dR and dPhi
        self.cov_alpha = 0.01 # EWMA decay
        self.last_phi = np.zeros(n_landmarks)
        
    def reset_learning_state(self, preserve_weights=True):
        super().reset_learning_state()
        self.cov_ewma = np.zeros(self.extractor.n_landmarks)
        self.last_phi = np.zeros(self.extractor.n_landmarks)
        
        if not preserve_weights:
            self.phi_history = []
            self.dphi_history = []
            self.cov_history = []
            self.v_history = []
            self.w_value = np.zeros(self.extractor.n_landmarks)
            self.is_trained = False
            
    def fit_value_function_hjb(self, env_params, window):
        """
        Calculates the Optimal Koopman Value Function Weights by solving the 
        Continuous Intertemporal HJB Residual globally over the Nystrom basis.
        """
        # Convert tracked state histories to tensors
        phi_mat = torch.tensor(np.array(self.phi_history), dtype=torch.float32)
        dphi_mat = torch.tensor(np.array(self.dphi_history), dtype=torch.float32) / self.dt
        cov_mat = torch.tensor(np.array(self.cov_history), dtype=torch.float32)
        v_mat = torch.tensor(np.array(self.v_history), dtype=torch.float32)
        
        # 1. Extract Continuous-Time Generator L via Regularized GEDMD
        # As per Klus (2020), diffusion variance dominates drift in pure finite differences (1/dt vs 1/sqrt(dt)).
        # We MUST apply weak regression (lam=1e-6) to cleanly extract the undamped expected generator over the tiny Nyström features.
        lam_gedmd = 1e-6
        R_dim = self.extractor.n_landmarks
        
        Gram = phi_mat.T @ phi_mat + lam_gedmd * torch.eye(R_dim, dtype=torch.float32)
        Cross = phi_mat.T @ dphi_mat
        
        # L = Gram^-1 Cross
        L_generator = torch.linalg.solve(Gram, Cross)
        
        N_samples = len(phi_mat)
        
        # Environmental constants
        mu = env_params['mu']
        r = env_params['r']
        gamma = env_params['gamma_risk']
        
        # --- LINEAR KOOPMAN POLICY ITERATION ---
        import scipy.linalg as la
        
        # 1. Policy Evaluation: Fix allocation to robust myopic pi_M
        v_safe = np.clip(v_mat.numpy(), 1e-5, None)
        pi_m = (mu - r) / (gamma * v_safe)
        
        # 2. Construct the Linear Continuous Bellman Operator M_total
        # Equation: [S_t + (1-gamma) pi_m Cov_t / dt + L_drift] f(x) = lambda f(x)
        
        # Scalar multiplier S_t
        term1_factor = (1.0 - gamma) * (r + pi_m * (mu - r))
        term2_factor = -0.5 * gamma * (1.0 - gamma) * (pi_m**2) * v_safe
        S_t = term1_factor + term2_factor
        
        Phi_np = phi_mat.numpy()
        M_0 = S_t[:, None] * Phi_np  # (N, K)
        
        Cov_np = cov_mat.numpy() / self.dt
        M_cov = ((1.0 - gamma) * pi_m)[:, None] * Cov_np  # (N, K)
        
        L_np = L_generator.numpy()
        M_drift = Phi_np @ L_np.T  # (N, K)
        
        # Total trace operator M_total * w = lambda * Phi * w
        M_total = M_0 + M_cov + M_drift
        
        # 3. Solve Generalized Eigenvalue Problem via Normal Equations
        # (Phi^T M_total) w = lambda (Phi^T Phi) w
        A_mat = Phi_np.T @ M_total
        B_mat = Phi_np.T @ Phi_np
        
        eigvals, eigvecs = la.eig(A_mat, B_mat)
        
        # 4. Filter for the Principal Value Eigenfunction (Largest Growth Rate)
        # We only consider analytically stable (real) eigenfunctions mapping the geometric space
        real_indices = np.where(np.abs(np.imag(eigvals)) < 1e-5)[0]
        if len(real_indices) > 0:
            best_idx = real_indices[np.argmax(np.real(eigvals)[real_indices])]
        else:
            best_idx = np.argmax(np.real(eigvals))
            
        lambda_val = np.real(eigvals[best_idx])
        w_raw = np.real(eigvecs[:, best_idx])
        
        # The Value Function MUST evaluate to a positive mapping over the domain
        # Check the empirical average value against the Nystrom trace
        if np.mean(Phi_np @ w_raw) < 0:
            w_raw = -w_raw
            
        w_norm = w_raw / np.linalg.norm(w_raw)
        
        print(f"Koopman Policy Iteration GEP completed.")
        print(f"Discovered Principal Eigenvalue (Growth Rate): {lambda_val:.4f}")
        
        self.w_value = w_norm
        self.L_generator_np = L_np
        self.is_trained = True
        
    def __call__(self, i, env):
        # Fallback prior
        if i < self.window:
            return (env.mu - env.r) / (env.gamma_risk * env.theta) 
            
        ret_t = env.returns_observed[i]
        
        # 1. Update Bipower Variation rolling sum
        old_ret = self.recent_returns[0]
        self.recent_returns[:-1] = self.recent_returns[1:]
        self.recent_returns[-1] = ret_t
        
        if i > 1:
            new_cross = np.abs(self.recent_returns[-1]) * np.abs(self.recent_returns[-2])
            old_cross = self.recent_cross_abs[0]
            self.recent_cross_abs[:-1] = self.recent_cross_abs[1:]
            self.recent_cross_abs[-1] = new_cross
            self.bv_sum = self.bv_sum + new_cross - old_cross
            
        # 2. Update Path Buffer
        self.path_buffer[:-1, 1] = self.path_buffer[1:, 1]
        self.path_buffer[-1, 1] = self.path_buffer[-2, 1] + ret_t
        
        # Extract features every step for true EDMD fidelity and Covariance tracking
        if len(self.extractor.landmarks) >= self.extractor.n_landmarks:
            phi = self.extractor.nystrom_embedding(self.path_buffer)
            
            # Online Cross-Covariance tracking between Asset Returns and Koopman State
            dPhi = phi - self.last_phi
            dR = ret_t
            self.cov_ewma = (1 - self.cov_alpha) * self.cov_ewma + self.cov_alpha * (dR * dPhi)
            
            if not self.is_trained:
                # IMPORTANT: Use the OLD state (Itô pre-point geometry) to avoid anticipating the Brownian noise
                # E[Phi_t * (Phi_{t+1} - Phi_t)] vs E[Phi_{t+1} * (Phi_{t+1} - Phi_t)] -> latter explodes as 1/dt
                self.phi_history.append(self.last_phi.copy())
                self.dphi_history.append(dPhi)
                self.cov_history.append(self.cov_ewma.copy())
                self.v_history.append(self.last_pred_V)
                
            # Update current state filters
            pred_std = self.A @ phi
            self.last_pred_V = pred_std**2
            self.last_phi = phi
                
            
        # 3. Kalman Nystrom Filter Step
        if i % self.k_n == 0:
            bv_target_val = (np.pi / 2.0) * self.bv_sum
            target_std = np.sqrt(max(bv_target_val / (self.window * self.dt), 1e-10))
            
            if len(self.extractor.landmarks) < self.extractor.n_landmarks:
                self.extractor.update(self.path_buffer)
                self.last_pred_V = target_std**2
            else:
                error = target_std - pred_std
                Pz = self.P @ phi
                gain = Pz / (self.ff + phi @ Pz)
                self.A = self.A + error * gain
                self.P = (self.P - np.outer(gain, Pz)) / self.ff
                
            if np.random.rand() < 0.05:
                self.extractor.update(self.path_buffer)
                
        # 4. Final Koopman Allocation 
        V_koopman = max(self.last_pred_V, 1e-5)
        pi_myopic = (env.mu - env.r) / (env.gamma_risk * V_koopman)
        
        if self.is_trained:
            # TRUE INTERTEMPORAL HEDGING DEMAND via Koopman spatial gradients
            # Reconstruct positive-definite value function projection
            f_phi = np.dot(self.w_value, self.last_phi)
            if f_phi > 1e-5:
                # The exact analytical gradient of the Koopman Value mapping is identically `w_value`
                cov_term = np.dot(self.w_value, self.cov_ewma) / self.dt
                # pi_H = (w^T Cov(dR, dPhi) / dt) / (gamma * V * w^T Phi)
                pi_hedging = cov_term / (env.gamma_risk * V_koopman * f_phi)
                # Soft clip the hedging correction locally
                pi_hedging = np.clip(pi_hedging, -4.0, 4.0)
            else:
                pi_hedging = 0.0
                
            pi_t = pi_myopic + pi_hedging
            # Ensure final bounds against leverage blowouts
            pi_t = np.clip(pi_t, 0.0, 5.0)
        else:
            pi_t = pi_myopic
            
        return pi_t


if __name__ == "__main__":
    print("Initializing Heston Portfolio Environment...")
    env = HestonPortfolioEnv(T=1.0, N=100000, gamma_risk=3.0)
    
    print("\nEvaluating Static Policy...")
    W_stat, alloc_stat, u_stat = env.evaluate_policy(static_policy, name="Static (Mean-Var)")
    
    print("\nEvaluating Oracle Myopic Policy (Perfect V_t)...")
    W_orac, alloc_orac, u_orac = env.evaluate_policy(oracle_policy, name="Oracle Myopic")
    
    print("\nEvaluating Koopman Signature Controller (Robust Myopic)...")
    sig_controller = KoopmanSignaturePolicy(window=1000, n_landmarks=100, dt=env.dt)
    W_sig, alloc_sig, u_sig = env.evaluate_policy(sig_controller, name="Koopman Robust Myopic")
    
    # Koopman True Hedging Demand Control
    print("\nEvaluating Koopman True Hedging Controller (Joint HJB Opt)...")
    
    import os
    import pickle
    checkpoint_file = "hedge_controller_trained.pkl"
    
    # Environmental constants for HJB optimization
    env_params = {
        'mu': env.mu,
        'r': env.r,
        'gamma_risk': env.gamma_risk
    }

    if os.path.exists(checkpoint_file):
        print(f"Loading trained Koopman weights from {checkpoint_file}...")
        with open(checkpoint_file, "rb") as f:
            state_dict = pickle.load(f)
        
        hedge_controller = TrueHedgingKoopmanPolicy(window=1000, n_landmarks=100, dt=env.dt)
        hedge_controller.w_value = state_dict['w_value']
        hedge_controller.L_generator_np = state_dict['L_generator_np']
        hedge_controller.is_trained = state_dict['is_trained']
        hedge_controller.extractor.landmarks = state_dict['landmarks']
        hedge_controller.A = state_dict['A']
        hedge_controller.P = state_dict['P']
    else:
        hedge_controller = TrueHedgingKoopmanPolicy(window=1000, n_landmarks=100, dt=env.dt)
        # Needs to see the whole state space geometry first
        W_train, _, _ = env.evaluate_policy(hedge_controller, name="HJB State Generation")
        
        print("Solving continuous Joint HJB Optimization for Value Eigenfunctions...")
        hedge_controller.fit_value_function_hjb(env_params, hedge_controller.window)
        
        state_dict = {
            'w_value': hedge_controller.w_value,
            'L_generator_np': hedge_controller.L_generator_np,
            'is_trained': hedge_controller.is_trained,
            'landmarks': hedge_controller.extractor.landmarks,
            'A': hedge_controller.A,
            'P': hedge_controller.P
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(state_dict, f)
        print(f"Saved trained controller weights to {checkpoint_file}")
        
    # Clear histories from RAM before final eval
    hedge_controller.phi_history = []
    hedge_controller.dphi_history = []
    hedge_controller.cov_history = []
    hedge_controller.v_history = []
    
    W_hedge, alloc_hedge, u_hedge = env.evaluate_policy(hedge_controller, name="Koopman True Hedging")
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(env.times, W_stat, label=f"Static (U={u_stat:.2f})", color='gray')
    plt.plot(env.times, W_sig, label=f"Robust Myopic (U={u_sig:.2f})", color='blue', linewidth=2)
    plt.plot(env.times, W_hedge, label=f"True Hedging (U={u_hedge:.2f})", color='purple', linewidth=2)
    plt.plot(env.times, W_orac, label=f"Oracle (U={u_orac:.2f})", color='green', linestyle='--')
    plt.title("Portfolio Wealth Trajectories (CRRA Utility $\gamma=3.0$)")
    plt.ylabel("Wealth $W_t$")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    # Subsample for faster rendering
    step = 100
    plt.plot(env.times[:-1][::step], alloc_stat[::step], label="Static Alloc", color='gray', linestyle='--')
    plt.plot(env.times[:-1][::step], alloc_sig[::step], label="Robust Myopic Alloc", color='blue', alpha=0.8)
    plt.plot(env.times[:-1][::step], alloc_hedge[::step], label="True Hedging Alloc", color='purple', alpha=0.8)
    plt.title("Dynamic Optimal Allocation Fraction $\pi_t$")
    plt.ylabel("Fraction in Risky Asset")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(env.times[:-1][::step], env.V_true[:-1][::step], label="True Hidden $V_t$", color='black', linestyle='--')
    # Because alloc = (mu-r)/(gamma*V) approximately, we can reverse engineer the robust controller's internal V_t
    v_sig = (env.mu - env.r) / (env.gamma_risk * np.maximum(alloc_sig, 1e-10))
    plt.plot(env.times[:-1][::step], v_sig[::step], label="Koopman Filtered $\hat{V}_t$", color='blue', alpha=0.7)
    plt.title("Koopman Filter Hidden State Tracking")
    plt.ylabel("Stochastic Variance")
    plt.xlabel("Time")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heston_portfolio_benchmark.png')
    print("\nSaved benchmark plot to heston_portfolio_benchmark.png")
