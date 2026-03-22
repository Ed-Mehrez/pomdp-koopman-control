r"""
Online KRONIC Controller for the Merton Portfolio Problem
=============================================================

This script implements Solution B from the theory document: the Online
Koopman Controller with Exploration-Correction.

It demonstrates how the offline Definitive KRONIC controller fails when 
extrapolating (due to RBF decay on the variance penalty), but rapidly 
self-corrects using a rank-1 Recursive Least Squares (RLS) update to its
empirical Koopman generator matrices when it observes the massive variance
of an unsafe allocation.

Key additions:
1. Online rank-1 Sherman-Morrison updates to the inverse Gram matrix $P^{(t)}$.
2. Real-time updates to the Koopman generator $A^{(t)}$.
3. Dynamic eigenvalue evaluation and action rejection in real-time.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq
from scipy.linalg import eig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import signatory
from merton_kronic_signatures import SignatoryFeatureExtractor
import time
import os

np.random.seed(42)

# =============================================================================
# 1. Environment and State Normalization (Unchanged)
# =============================================================================

class HestonMertonEnv:
    def __init__(self, mu=0.08, r=0.02, gamma=2.0,
                 kappa=2.0, theta=0.04, xi=0.3, rho=-0.7):
        self.mu = mu
        self.r = r
        self.gamma = gamma
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def compute_drift(self, state):
        log_W, pi, V = state
        V_safe = max(V, 1e-8)
        drift_log_W = self.r + pi * (self.mu - self.r) - 0.5 * pi**2 * V_safe
        drift_pi = 0.0
        drift_V = self.kappa * (self.theta - V_safe)
        return np.array([drift_log_W, drift_pi, drift_V])

    def compute_diffusion_matrix(self, state):
        log_W, pi, V = state
        V_safe = max(V, 1e-8)
        Sigma = np.zeros((3, 3))
        Sigma[0, 0] = pi**2 * V_safe
        Sigma[2, 2] = self.xi**2 * V_safe
        Sigma[0, 2] = pi * self.xi * V_safe * self.rho
        Sigma[2, 0] = Sigma[0, 2]
        return Sigma

    def merton_optimal(self, V):
        return (self.mu - self.r) / (self.gamma * max(V, 1e-8))

    def step(self, state, action=0.0, dt=1/252):
        log_W, pi, v = state
        z1 = np.random.randn()
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.randn()
        dB = np.sqrt(dt) * z1
        dB_v = np.sqrt(dt) * z2
        
        drift_W = self.r + pi * (self.mu - self.r) - 0.5 * pi**2 * v
        d_log_W = drift_W * dt + pi * np.sqrt(max(v, 1e-8)) * dB
        
        new_v = v + self.kappa * (self.theta - v) * dt
        new_v += self.xi * np.sqrt(max(v, 1e-8)) * dB_v
        new_v = max(new_v, 1e-8)
        
        return np.array([log_W + d_log_W, pi + action, new_v]), 0.0

class StateNormalizer:
    def __init__(self):
        self.mean = None
        self.scale = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        
    def transform(self, X):
        return (X - self.mean) / self.scale
        
    def inv_scale_matrix(self):
        return np.diag(1.0 / self.scale)

# =============================================================================
# 2. Online Signature EDMD Tracker
# =============================================================================

class OnlineSignatureEDMD:
    def __init__(self, dt, n_landmarks=30, epsilon=1e-3, n_eigs=30):
        self.dt = dt
        self.epsilon = epsilon
        self.n_eigs = n_eigs
        self.n_landmarks = n_landmarks
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = SignatoryFeatureExtractor(depth=3, device=self.device)
        self.d_feat = self.extractor.sig_dim + 1
        
        self.eigenvectors = None
        self.eigenvalues = None
        
        self.landmarks_sig = None  # (m, D)
        self.C_inv = None          # (m, m) Inner Product Matrix Inverse
        
        self.P = None      
        self.A_T = None    
        self.updates = 0
        
    def _project_nystrom(self, psi_matrix):
        """Projects full sigs onto the Nystrom span: K(X, L) @ C_inv"""
        # Linear kernel between sample signatures and landmark signatures
        K_XL = np.dot(psi_matrix, self.landmarks_sig.T) # (N, m)
        return K_XL @ self.C_inv # (N, m)
        
    def fit_offline(self, psi_X, psi_Y):
        n = len(psi_X)
        m = min(self.n_landmarks, n)
        
        # 1. Select Nystrom Landmarks using K-Means Clustering (Noise Sub-sampling)
        print(f"    [Offline] Executing K-Means Clustering for {m} Nystrom Landmarks...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=m, random_state=42, n_init='auto')
        kmeans.fit(psi_X)
        self.landmarks_sig = kmeans.cluster_centers_ # (m, D)
        
        # 2. Compute inner product matrix of landmarks
        C = np.dot(self.landmarks_sig, self.landmarks_sig.T) # (m, m)
        self.C_inv = np.linalg.inv(C + self.epsilon * np.eye(m))
        
        # 3. Project all training data into the m-dimensional Nystrom span
        print(f"    [Offline] Projecting {self.d_feat}D Signatures to {m}D Nystrom Span...")
        psi_X_m = self._project_nystrom(psi_X)
        psi_Y_m = self._project_nystrom(psi_Y)
        
        print("    [Offline] Computing Gram matrix (G00) in Nystrom Span...")
        G00 = (psi_X_m.T @ psi_X_m) / n + self.epsilon * np.eye(m)
        G10 = (psi_X_m.T @ psi_Y_m) / n
        
        print("    [Offline] Initializing Bayesian Kalman Filter Matrices...")
        
        self.A_T = np.linalg.inv(G00) @ G10     # Discrete Koopman Operator (m x m)
        
        # 4. Initialize Structural Measurement Noise (R)
        # R controls the expected instantaneous variance of the target signatures.
        # It is strictly the empirical MSE of the offline training functional map per-dimension.
        pred_Y_m = psi_X_m @ self.A_T
        self.R = np.mean((psi_Y_m - pred_Y_m)**2)
        
        # 5. Initialize the Bayesian Prior Covariance (P_0) from the Nystrom span precision 
        # Scaled by the actual measurement noise
        self.P = np.linalg.inv(G00) * (self.R / n)
        
        # 6. Extract structural Process Noise Covariance (Q)
        # We set Q = 0 to enforce strict Bayesian convergence (1/t gain decay).
        # This prevents the variance from exploding during Heston vol jumps.
        self.Q = np.zeros_like(self.P)
        
        self._update_eigensystem()
        
    def _update_eigensystem(self):
        eigenvals, eigenvecs = eig(self.A_T)
        
        valid_idx = np.abs(eigenvals) > 1e-8
        cont_eigenvals = np.zeros_like(eigenvals, dtype=complex)
        cont_eigenvals[valid_idx] = np.log(eigenvals[valid_idx]) / self.dt
        
        idx = np.argsort(np.real(cont_eigenvals))[::-1]
        self.eigenvalues = cont_eigenvals[idx][:self.n_eigs]
        self.eigenvectors = eigenvecs[:, idx][:, :self.n_eigs]

    def update_online(self, psi_t, psi_next):
        """
        Bayesian Koopman Kalman Filter (Sig-KKF) update in the compressed Nystrom span.
        """
        # Project full signatures into the constrained m-dimensional Nystrom base
        psi_t_m = self._project_nystrom(psi_t.reshape(1, -1))[0]
        psi_next_m = self._project_nystrom(psi_next.reshape(1, -1))[0]
        
        # -------------------------------------------------------------
        # Kalman Filter Equations
        # -------------------------------------------------------------
        
        # 1. Prior Prediction (Extrapolate uncertainty via feature-scaled drift)
        P_prior = self.P + self.Q
        
        # 2. Kalman Gain (Innovation scale)
        S_t = np.dot(psi_t_m, P_prior @ psi_t_m) + self.R
        K_t = (P_prior @ psi_t_m) / S_t  # Shape: (m,)
        
        # 3. Measurement Residual (Target is next state signature)
        prediction = self.A_T.T @ psi_t_m
        error = psi_next_m - prediction
        
        # 4. Posterior State Update (Drift the Koopman Matrix)
        self.A_T = self.A_T + np.outer(K_t, error)
        
        # 5. Posterior Covariance Update
        m = self.P.shape[0]
        self.P = (np.eye(m) - np.outer(K_t, psi_t_m)) @ P_prior
        
        self.updates += 1
        
        if self.updates % 20 == 0:
            self._update_eigensystem()


# =============================================================================
# 3. KRONIC Controller
# =============================================================================

class OnlineKRONICMerton:
    def __init__(self, env, n_eigs=30, horizon=None):
        self.env = env
        self.n_eigs = n_eigs
        self.dt = 1/252
        self.horizon = horizon if horizon is not None else self.dt  # Default to 1-step DP evaluation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.kgedmd = None
        self.w_utility = None
        
    def generate_safe_training_data(self, n_paths=40, n_steps=252):
        print(f"2. Generating Safe Offline Training Data (Lead-Lag)...")
        psi_X = []
        psi_Y = []
        U_all = []
        
        with torch.no_grad():
            extractor = SignatoryFeatureExtractor(depth=3, device=self.device)
            
            for path in range(n_paths):
                log_W = np.random.normal(0, 0.1)
                pi_val = np.random.uniform(0.1, 1.2)  # SAFE bounds
                V_val = np.random.uniform(0.02, 0.08)
                
                state_traj = [[log_W, pi_val, V_val]]
                
                for t in range(n_steps):
                    state = state_traj[-1]
                    action = 0.02 * np.random.randn()
                    next_state, _ = self.env.step(state, action, dt=self.dt)
                    state_traj.append(list(next_state))
                    
                    if np.random.rand() < 0.02:
                        state_traj[-1][1] = np.random.uniform(0.1, 1.2)
                        
                    if len(state_traj) >= 4: # Fixed sliding windows of length 3
                        # Path for psi_X is (t-2, t-1, t)
                        p_X = np.array(state_traj[-4:-1])
                        # Path for psi_Y is (t-1, t, t+1)
                        p_Y = np.array(state_traj[-3:])
                        
                        # Wealth Translation Invariance: center logW on current time
                        shift_W = p_X[-1, 0]
                        p_X[:, 0] -= shift_W
                        p_Y[:, 0] -= shift_W
                        
                        path_X_tensor = torch.tensor([p_X], dtype=torch.float64, device=self.device)
                        path_Y_tensor = torch.tensor([p_Y], dtype=torch.float64, device=self.device)
                        
                        f_X = extractor.extract(path_X_tensor)[0].cpu().numpy()
                        f_Y = extractor.extract(path_Y_tensor)[0].cpu().numpy()
                        
                        psi_X.append(f_X)
                        psi_Y.append(f_Y)
                        
                        # Utility evaluated on SHIFTED wealth (centered at current time)
                        W_s = np.exp(p_Y[-1, 0])  # = exp(Δ logW from current)
                        U = (W_s**(1 - self.env.gamma)) / (1 - self.env.gamma)
                        U_all.append(U)
                        
        idx = np.random.choice(len(psi_X), min(2000, len(psi_X)), replace=False)
        return np.array(psi_X)[idx], np.array(psi_Y)[idx], np.array(U_all)[idx]
        
    def train_offline(self, psi_X, psi_Y, U_array):
        print("3. Fitting Offline Base KGEDMD Model...")
        self.kgedmd = OnlineSignatureEDMD(dt=self.dt, n_eigs=self.n_eigs)
        self.kgedmd.fit_offline(psi_X, psi_Y)
        
        print("4. Projecting Utility onto Nystrom psi_Y (growth-rate utility)...")
        # Train w on psi_Y so that w·ψ_Y ≈ U(exp(ΔlogW))
        # At inference: w · A · ψ_X predicts this via Koopman evolution
        psi_Y_m = self.kgedmd._project_nystrom(psi_Y)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(psi_Y_m, U_array)
        self.w_utility = ridge.coef_
        print("   Done offline training.")

    def compute_online_eigenvalues(self):
        return np.real(self.kgedmd.eigenvalues)

    def find_optimal_pi(self, history_traj, pi_candidates):
        steps = max(1, int(self.horizon / self.dt))
        discrete_A_horizon = np.linalg.matrix_power(self.kgedmd.A_T, steps)
        
        B = len(pi_candidates)
        curr_state = history_traj[-1]
        
        # Wealth Translation Invariance: center logW on current time
        shift_W = history_traj[-1][0]
        shifted_history = []
        for state in history_traj:
            shifted_history.append([state[0] - shift_W, state[1], state[2]])
        
        path_X = torch.tensor(shifted_history, dtype=torch.float64, device=self.device)
        path_X = path_X.unsqueeze(0).repeat(B, 1, 1) # (B, 3, 3)
        
        # Overwrite pi in the current state x_t so the Signature captures the action
        path_X[:, -1, 1] = torch.tensor(pi_candidates, dtype=torch.float64, device=self.device)
        
        with torch.no_grad():
            # Extract signature features for exactly path length 3
            psi_0s = self.kgedmd.extractor.extract(path_X).cpu().numpy()
            
        # VERY IMPORTANT: Compress the 259D tensor down into the 30D Nyström space FIRST.
        psi_0s_m = self.kgedmd._project_nystrom(psi_0s)
            
        # Predict the future signature features using the COMPRESSED Koopman operator
        Psi_T_preds = psi_0s_m @ discrete_A_horizon
        
        EU_objs = np.sum(self.w_utility * Psi_T_preds, axis=1)
        
        best_idx = np.argmax(EU_objs)
        return pi_candidates[best_idx], EU_objs

# =============================================================================
# 4. Online Simulation Hook
# =============================================================================

def run_online_experiment():
    print("=" * 60)
    print("Online KRONIC controller (RLS Corrected) for Merton-Heston")
    print("=" * 60)
    
    env = HestonMertonEnv()
    kronic = OnlineKRONICMerton(env, n_eigs=30, horizon=0.1)
    
    # Offline safe phase
    X_train, X_train_scaled = kronic.generate_safe_training_data()
    kronic.train_offline(X_train, X_train_scaled)
    
    # ---------------------------------------------------------
    # Online Phase: Observe Extrapolation Collapse and Recovery
    # ---------------------------------------------------------
    print("\n[Online Phase] Provoking Extrapolation Error...")
    
    # We query the controller with a global search grid. Because it was trained
    # ONLY on pi in [0.1, 1.2], it will hallucinate that pi=4.0 is optimal due
    # to RBF decay on the variance penalty.
    pi_candidates = np.linspace(0.1, 4.0, 100)
    V_test = 0.05
    W0 = 1.0
    
    # Step 0: The Baseline Hallucination
    pi_opt_0, objs_0 = kronic.find_optimal_pi(W0, V_test, pi_candidates)
    merton_opt = env.merton_optimal(V_test)
    
    print(f"  Step 0: Analytical Optimum     = {merton_opt:.3f}")
    print(f"  Step 0: Offline KRONIC Optimum = {pi_opt_0:.3f} (Hallucination predicted!)")
    
    # We now step the environment to that exact hallucinated allocation for ONE DAY (dt=1/252)
    dt = 1/252
    state_t = np.array([np.log(W0), pi_opt_0, V_test])
    state_next, _ = env.step(state_t, action=0.0, dt=dt)
    
    print("\n  Executing ONE step at hallucinated optimum...")
    # Update the Koopman operator online!
    z_t_scaled = kronic.normalizer.transform(state_t.reshape(1, -1))[0]
    z_next_scaled = kronic.normalizer.transform(state_next.reshape(1, -1))[0]
    
    kronic.kgedmd.update_online(z_t_scaled, z_next_scaled, dt)
    print("  RLS Rank-1 Koopman Update complete.")
    
    # Step 1: Evaluating the Post-RLS Objective Landscape
    pi_opt_1, objs_1 = kronic.find_optimal_pi(np.exp(state_next[0]), state_next[2], pi_candidates)
    
    print(f"  Step 1: Corrected KRONIC Optimum = {pi_opt_1:.3f}")
    
    # Plotting the real-time correction
    plt.figure(figsize=(8, 6))
    plt.plot(pi_candidates, objs_0, 'r--', label='Step 0 (Prior Hallucination)')
    plt.plot(pi_candidates, objs_1, 'g-', linewidth=2, label='Step 1 (Post-RLS Correction)')
    plt.axvline(merton_opt, color='k', linestyle=':', label='Merton Analytical')
    plt.xlabel(r'Candidate Allocation ($\pi$)')
    plt.ylabel('Expected Utility Objective')
    plt.title('Real-Time KRONIC Suboptimality Correction via RLS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('finance/experiments/merton_kronic_online_correction.png', dpi=150)
    print("\nSaved plot to finance/experiments/merton_kronic_online_correction.png")

if __name__ == "__main__":
    run_online_experiment()
