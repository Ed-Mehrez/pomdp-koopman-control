r"""
Definitive KRONIC Controller for the Merton Portfolio Problem
=============================================================

This script implements the mathematically correct KRONIC (Koopman operator
based control) methodology for the Merton problem with Heston stochastic
volatility.

Key features (addressing previous failed attempts):
1. State space: (log W, π, V) with proper normalization for RBF kernel.
2. Generator: Includes the full drift AND the second-order Itô diffusion
   terms (including cross-variance) for both wealth and volatility.
3. Eigenvalues: Uses the unbiased $L\psi_i / \psi_i$ estimator pointwise
   rather than the biased $L(U)/U$ estimator.
4. Utility: Projections allow general utility forms $U(W)$.
5. Optimization: Discovers the optimal allocation $\pi^*$ by evaluating
   the expectation $\sum w_i e^{\lambda_i(\pi) t} \psi_i(x_0)$.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq, eig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import time
import os

np.random.seed(42)

# =============================================================================
# 1. Environment and State Normalization
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
        """b(X) for state = [log W, pi, V]"""
        log_W, pi, V = state
        V_safe = max(V, 1e-8)
        
        drift_log_W = self.r + pi * (self.mu - self.r) - 0.5 * pi**2 * V_safe
        drift_pi = 0.0
        drift_V = self.kappa * (self.theta - V_safe)
        
        return np.array([drift_log_W, drift_pi, drift_V])

    def compute_diffusion_matrix(self, state):
        """a(X)a(X)^T for state = [log W, pi, V]"""
        log_W, pi, V = state
        V_safe = max(V, 1e-8)
        
        Sigma = np.zeros((3, 3))
        # var(d log W) = pi^2 * V * dt
        Sigma[0, 0] = pi**2 * V_safe
        # var(dV) = xi^2 * V * dt
        Sigma[2, 2] = self.xi**2 * V_safe
        # cov(d log W, dV) = pi * xi * V * rho * dt
        Sigma[0, 2] = pi * self.xi * V_safe * self.rho
        Sigma[2, 0] = Sigma[0, 2]
        
        return Sigma

    def merton_optimal(self, V):
        """Myopic Merton rule π* = (μ-r)/(γV)"""
        return (self.mu - self.r) / (self.gamma * max(V, 1e-8))

    def step(self, state, action=0.0, dt=1/252):
        """Evolve state [log_W, pi, V] via Euler-Maruyama."""
        log_W, pi, v = state
        
        # Correlated Brownians
        z1 = np.random.randn()
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.randn()
        dB = np.sqrt(dt) * z1
        dB_v = np.sqrt(dt) * z2
        
        # Wealth dynamics (log scale)
        drift_W = self.r + pi * (self.mu - self.r) - 0.5 * pi**2 * v
        d_log_W = drift_W * dt + pi * np.sqrt(max(v, 1e-8)) * dB
        
        # Variance dynamics
        new_v = v + self.kappa * (self.theta - v) * dt
        new_v += self.xi * np.sqrt(max(v, 1e-8)) * dB_v
        new_v = max(new_v, 1e-8)
        
        new_state = np.array([log_W + d_log_W, pi + action, new_v])
        return new_state, 0.0


class StateNormalizer:
    """Scales states so RBF kernel works equally well across dimensions."""
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
        """Returns the diagonal matrix S such that unscaled cov Sigma_unscaled 
           transforms to Sigma_scaled = S * Sigma_unscaled * S"""
        return np.diag(1.0 / self.scale)


# =============================================================================
# 2. Fully Correct KGEDMD 
# =============================================================================

class DefinitiveKGEDMD:
    def __init__(self, kernel_bandwidth=1.0, epsilon=1e-6, n_eigs=50):
        self.sigma_sq = kernel_bandwidth ** 2
        self.epsilon = epsilon
        self.n_eigs = n_eigs
        
        self.eigenvectors = None
        self.eigenvalues = None
        self.X_train_scaled = None
        
    def rbf_gram(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        sq_dists = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-sq_dists / (2 * self.sigma_sq))
        
    def fit(self, X_train_scaled, drifts_scaled, diffusions_scaled):
        """
        Fits Koopman generator using EDMD with full Itô correction.
        """
        self.X_train_scaled = X_train_scaled
        n = len(X_train_scaled)
        
        # 1. G00 (Gram matrix)
        print("    Computing Gram matrix...")
        G00 = self.rbf_gram(X_train_scaled)
        
        # 2. G10 (Generator inner products)
        print("    Computing Generator matrix with Itô correction...")
        G10 = np.zeros((n, n))
        
        for i in range(n):
            xi = X_train_scaled[i]
            drift_i = drifts_scaled[i]
            Sigma_i = diffusions_scaled[i]
            
            trace_Sigma = np.trace(Sigma_i)
            
            for j in range(n):
                xj = X_train_scaled[j]
                diff = xi - xj
                
                k_ij = G00[i, j]
                
                # First-order term (Drift)
                # L_{x_i} k(x_i, x_j) = -1/sigma^2 * (x_i - x_j)^T b(x_i) * k(x_i, x_j)
                term1 = -np.dot(drift_i, diff) / self.sigma_sq * k_ij
                
                # Second-order term (Diffusion)
                # 1/2 Tr(Sigma(x_i) Hessian_xi k)
                quads = np.einsum('ni,ij,nj->n', diff[np.newaxis, :], Sigma_i, diff[np.newaxis, :])[0]
                term2 = 0.5 * (k_ij / self.sigma_sq) * (quads / self.sigma_sq - trace_Sigma)
                
                G10[i, j] = term1 + term2
                
        # 3. Solve Generalized Eigenvalue Problem
        print("    Solving Eigenvalue Problem...")
        # L K = K A^T  =>  A_T = K^+ L K
        A_T, _, _, _ = lstsq(G00 + self.epsilon * np.eye(n), G10, rcond=None)
        
        eigenvals, eigenvecs = eig(A_T)
        
        # Sort by real part (largest first, typically near 0 for conservative systems)
        idx = np.argsort(np.real(eigenvals))[::-1]
        
        self.eigenvalues = eigenvals[idx][:self.n_eigs]
        self.eigenvectors = eigenvecs[:, idx][:, :self.n_eigs]
        
    def transform(self, X_test_scaled):
        """Evaluate eigenfunctions at test queries."""
        K_test = self.rbf_gram(X_test_scaled, self.X_train_scaled)
        # psi_j(x) = sum_i K(x, x_i) v_{i,j}
        # In reality, keeping it complex or real depends on context. Since we
        # want real utilities, we take the real part (eigenvalues usually appear 
        # in complex conjugate pairs anyway or are strictly real).
        return np.real(K_test @ self.eigenvectors)
        
    def compute_generator(self, X_test_scaled, drifts_scaled, diffusions_scaled):
        """
        Evaluate L(psi_j) at novel states analytically.
        This provides the un-biased estimator.
        """
        m = len(X_test_scaled)
        n = len(self.X_train_scaled)
        
        L_psi = np.zeros((m, self.n_eigs), dtype=complex)
        
        for i in range(m):
            xi = X_test_scaled[i]
            drift_i = drifts_scaled[i]
            Sigma_i = diffusions_scaled[i]
            trace_Sigma = np.trace(Sigma_i)
            
            # Compute L_{x_i} k(x_i, X_train)
            diffs = xi - self.X_train_scaled  # (n_train, dim) Note xi is 1D, X_train is 2D
            sq_dists = np.sum(diffs**2, axis=1)
            K_vec = np.exp(-sq_dists / (2 * self.sigma_sq))
            
            # First-order term (Drift)
            # L_{x_i} k(x_i, x_j) = -1/sigma^2 * (x_i - x_j)^T b(x_i) * k(x_i, x_j)
            term1 = -(diffs @ drift_i) / self.sigma_sq * K_vec
            
            # Second-order term (Diffusion)
            # 1/2 Tr(Sigma(x_i) Hessian_xi k)
            quads = np.einsum('ni,ij,nj->n', diffs, Sigma_i, diffs)
            term2 = 0.5 * (K_vec / self.sigma_sq) * (quads / self.sigma_sq - trace_Sigma)
            
            L_K_vec = term1 + term2  # (n_train,)
            
            # psi_j(x) = sum_k K(x, x_k) v_{k,j}
            # L psi_j(x) = sum_k L_x K(x, x_k) v_{k,j}
            L_psi[i, :] = L_K_vec @ self.eigenvectors
            
        return np.real(L_psi)


# =============================================================================
# 3. KRONIC Controller
# =============================================================================

class TrueKRONICMerton:
    def __init__(self, env, n_eigs=30, horizon=1.0):
        self.env = env
        self.n_eigs = n_eigs
        self.horizon = horizon
        self.normalizer = StateNormalizer()
        self.kgedmd = None
        self.w_utility = None
        self.psi_scaler = None
        
    def generate_training_data(self, n_paths=100, n_steps=504):
        """Generates a rich dataset of (logW, pi, V) states using random trajectories."""
        print(f"2. Generating Training Data ({n_paths} paths, {n_steps} steps)...")
        X_all = []
        
        for path in range(n_paths):
            # Random initial conditions for diversity
            log_W = np.random.normal(0, 0.1)
            pi = np.random.uniform(0.3, 1.5)  # Diverse allocations
            V = np.random.uniform(0.02, 0.08)  # Diverse volatilities
            
            state = np.array([log_W, pi, V])
            
            for t in range(n_steps):
                X_all.append(state.copy())
                
                # Evolve with some exploration noise
                action = 0.05 * np.random.randn()  # Small random rebalancing
                state, _ = self.env.step(state, action)
                
                # Occasionally reset pi for exploration
                if np.random.rand() < 0.01:
                    state[1] = np.random.uniform(0.3, 1.5)
                    
        X_train = np.array(X_all)
        print(f"   Generated {len(X_train)} samples.")
        
        # Subsample for efficiency
        n_kgedmd = min(3000, len(X_train))
        idx = np.random.choice(len(X_train), n_kgedmd, replace=False)
        X_kgedmd = X_train[idx]
        
        self.normalizer.fit(X_kgedmd)
        return X_kgedmd, self.normalizer.transform(X_kgedmd)
        
    def train(self, X_train, X_train_scaled):
        t0 = time.time()
        print("1. Normalizing state space...")
        # Normalizer is now fitted and X_train_scaled is provided by generate_training_data
        
        print("2. Computing scaled drifts and diffusions...")
        n = len(X_train)
        drifts_scaled = np.zeros_like(X_train_scaled)
        diffusions_scaled = np.zeros((n, 3, 3))
        
        S_mat = self.normalizer.inv_scale_matrix()
        
        for i in range(n):
            state = X_train[i]
            b = self.env.compute_drift(state)
            a_sq = self.env.compute_diffusion_matrix(state)
            
            # Scale drift: dx_scaled/dt = S * dx/dt
            drifts_scaled[i] = S_mat @ b
            # Scale diffusion: S * a_sq * S^T
            diffusions_scaled[i] = S_mat @ a_sq @ S_mat.T
            
        print("3. Fitting Definitive KGEDMD...")
        self.kgedmd = DefinitiveKGEDMD(kernel_bandwidth=1.5, n_eigs=self.n_eigs)
        self.kgedmd.fit(X_train_scaled, drifts_scaled, diffusions_scaled)
        
        print("4. Projecting Utility U(W) onto Eigenfunctions...")
        # Compute U(W) for training data
        W = np.exp(X_train[:, 0])
        U = (W**(1 - self.env.gamma)) / (1 - self.env.gamma)
        
        psi_train = self.kgedmd.transform(X_train_scaled)
        
        self.psi_scaler = StandardScaler()
        psi_scaled = self.psi_scaler.fit_transform(psi_train)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(psi_scaled, U)
        self.w_utility = ridge.coef_
        
        U_pred = ridge.predict(psi_scaled)
        r2 = 1 - np.var(U - U_pred) / np.var(U)
        print(f"    Utility projection R²: {r2:.4f}")
        print(f"    Training completed in {time.time()-t0:.2f}s")
        
    def compute_robust_eigenvalues(self, pi, V):
        """Estimate robust lambda_i for a specific pi and V using multiple samples."""
        n_mc = 30
        X_test = np.column_stack([
            np.zeros(n_mc),  # log W doesn't matter for eigenvalues
            np.full(n_mc, pi),
            np.full(n_mc, V)
        ])
        
        X_test_scaled = self.normalizer.transform(X_test)
        drifts_scaled = np.zeros_like(X_test_scaled)
        diffusions_scaled = np.zeros((n_mc, 3, 3))
        S_mat = self.normalizer.inv_scale_matrix()
        
        for i in range(n_mc):
            state = X_test[i]
            b = self.env.compute_drift(state)
            a_sq = self.env.compute_diffusion_matrix(state)
            drifts_scaled[i] = S_mat @ b
            diffusions_scaled[i] = S_mat @ a_sq @ S_mat.T
            
        psi_val = self.kgedmd.transform(X_test_scaled)
        L_psi_val = self.kgedmd.compute_generator(X_test_scaled, drifts_scaled, diffusions_scaled)
        
        lambda_i = np.zeros(self.n_eigs)
        for i in range(self.n_eigs):
            valid = np.abs(psi_val[:, i]) > 1e-8
            if np.sum(valid) > 5:
                ratios = np.real(L_psi_val[valid, i] / psi_val[valid, i])
                lambda_i[i] = np.median(ratios)
                
        return lambda_i

    def find_optimal_pi(self, W0, V0, pi_candidates):
        """
        Finds the expected utility for each candidate pi and selects the best.
        """
        n_cand = len(pi_candidates)
        EU_objs = np.zeros(n_cand)
        
        # Target allocation loop
        for i, pi in enumerate(pi_candidates):
            lambda_i = self.compute_robust_eigenvalues(pi, V0)
            
            # Crucial fix: initial state assumes we rebalance to target pi
            z0 = np.array([[np.log(W0), pi, V0]]) 
            z0_scaled = self.normalizer.transform(z0)
            psi_0 = self.kgedmd.transform(z0_scaled)[0]
            
            psi_0_scaled = (psi_0 - self.psi_scaler.mean_) / self.psi_scaler.scale_
            
            # E[U] = sum_j w_j e^(lambda_j * t) psi_j(W0, pi, V0)
            EU_objs[i] = np.sum(self.w_utility * np.exp(lambda_i * self.horizon) * psi_0_scaled)
            
        best_idx = np.argmax(EU_objs)
        return pi_candidates[best_idx], EU_objs

# =============================================================================
# 4. Evaluation and Visualization
# =============================================================================

def run_experiment():
    print("=" * 60)
    print("Definitive KRONIC controller for Merton-Heston")
    print("=" * 60)
    
    env = HestonMertonEnv()
    kronic = TrueKRONICMerton(env, n_eigs=40, horizon=0.1)
    
    print("\n[Phase 1] Training")
    # Generate data and train
    X_train, X_train_scaled = kronic.generate_training_data(n_paths=100, n_steps=504)
    kronic.train(X_train, X_train_scaled)
    
    print("\n[Phase 2] Evaluating KRONIC Policy vs Merton")
    V_test = np.linspace(0.015, 0.12, 12)
    pi_candidates = np.linspace(0.1, 4.0, 100)
    
    W0 = 1.0  # Normalized initial wealth
    
    pi_kronic_list = []
    pi_merton_list = []
    errors = []
    
    print(f"\n{'V':>8} | {'π*_Merton':>10} | {'π*_KRONIC':>10} | {'Error %':>8}")
    print("-" * 45)
    
    for V in V_test:
        pi_merton = (env.mu - env.r) / (env.gamma * V)
        # Revert candidate ranges to global search, no more cheating bounds!
        pi_candidates = np.linspace(0.1, 4.0, 100)
        
        pi_kronic, _ = kronic.find_optimal_pi(W0=1.0, V0=V, pi_candidates=pi_candidates)
        
        err = abs(pi_kronic - pi_merton) / pi_merton * 100
        errors.append(err)
        
        pi_kronic_list.append(pi_kronic)
        pi_merton_list.append(pi_merton)
        
        print(f"{V:8.4f} | {pi_merton:10.3f} | {pi_kronic:10.3f} | {err:7.1f}%")
        
    print(f"\nMean Absolute Error vs Myopic Merton: {np.mean(errors):.2f}%")
    
    # Validation Plots
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(V_test, pi_merton_list, 'k-', linewidth=2, label='Merton (Analytical)')
    plt.plot(V_test, pi_kronic_list, 'g--o', label='KRONIC (Learned)')
    plt.xlabel('Variance (V)')
    plt.ylabel(r'Optimal Allocation ($\pi^*$)')
    plt.title('Learned Policy vs Ground Truth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Plot the objective landscape for a specific V to prove it works
    V_mid = V_test[len(V_test)//2]
    _, EU_objs = kronic.find_optimal_pi(W0, V_mid, pi_candidates)
    
    plt.plot(pi_candidates, EU_objs, 'b-')
    plt.axvline(env.merton_optimal(V_mid), color='k', linestyle='--', label='Merton Obj Max')
    plt.xlabel(r'Candidate Allocation ($\pi$)')
    plt.ylabel('Expected Utility')
    plt.title(f'Objective Landscape (V={V_mid:.4f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('finance/experiments/merton_kronic_definitive.png', dpi=150)
    print("\nSaved plot to finance/experiments/merton_kronic_definitive.png")

if __name__ == "__main__":
    run_experiment()
