r"""
Control-Quadratic Koopman (CQ-KRONIC) for the Merton Portfolio Problem
=======================================================================

Two modes:

1. Transfer Operator mode:
   ψ_{t+1} = A₀·ψ_t + π·A₁·ψ_t + π²·A₂·ψ_t

2. Generator mode (preferred):
   (ψ_{t+1} - ψ_t)/dt = L₀·ψ_t + π·L₁·ψ_t + π²·L₂·ψ_t

   L₁ captures: (μ-r)·∂/∂logW + ρξV·∂²/∂logW∂V  (hedging term!)
   L₂ captures: -½V·∂/∂logW + ½V·∂²/∂logW²

Optimal allocation: π* = -c₁ / (2·c₂) where cᵢ = w·Lᵢ·ψ
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import torch
import signatory
import time

np.random.seed(42)

# =============================================================================
# 1. Environment (state = logW, V only -- pi is the control)
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

    def merton_optimal(self, V):
        return (self.mu - self.r) / (self.gamma * max(V, 1e-8))

    def step(self, logW, V, pi, dt=1/252):
        """Step the state forward by dt given the control action pi."""
        z1 = np.random.randn()
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.randn()
        dB = np.sqrt(dt) * z1
        dB_v = np.sqrt(dt) * z2
        
        v_safe = max(V, 1e-8)
        drift_W = self.r + pi * (self.mu - self.r) - 0.5 * pi**2 * v_safe
        d_logW = drift_W * dt + pi * np.sqrt(v_safe) * dB
        
        new_V = V + self.kappa * (self.theta - v_safe) * dt + self.xi * np.sqrt(v_safe) * dB_v
        new_V = max(new_V, 1e-8)
        
        return logW + d_logW, new_V

    def step_momentum(self, logW, V, pi, pi_prev, dt=1/252):
        """Step with previous action as part of state."""
        lw, v = self.step(logW, V, pi, dt)
        return lw, v, pi  # Return (logW, V, pi_prev)

    def step_explicit(self, logW, V, pi, z1, z2, dt=1/252):
        """Step with explicit noise for paired comparisons."""
        dB = np.sqrt(dt) * z1
        dB_v = np.sqrt(dt) * z2
        
        v_safe = max(V, 1e-8)
        drift_W = self.r + pi * (self.mu - self.r) - 0.5 * pi**2 * v_safe
        d_logW = drift_W * dt + pi * np.sqrt(v_safe) * dB
        
        new_V = V + self.kappa * (self.theta - v_safe) * dt + self.xi * np.sqrt(v_safe) * dB_v
        new_V = max(new_V, 1e-8)
        
        return logW + d_logW, new_V


class CEVEnv:
    """
    Constant Elasticity of Variance (CEV) model.
    dS/S = mu*dt + sigma*S**(alpha-1)*dW
    Volatility is state-dependent: vol(S) = sigma * S**(alpha-1)
    """
    def __init__(self, mu=0.08, r=0.02, gamma=3.0, sigma=0.2, alpha=0.5):
        self.mu, self.r, self.gamma = mu, r, gamma
        self.sigma, self.alpha = sigma, alpha
        self.S = 1.0 # Initial price level

    def step(self, logW, var, pi, dt=1/252):
        # We assume var is the 'effective' variance sigma^2 * S^(2alpha-2)
        # S can be recovered from var: S = (var / sigma^2)^(1 / (2alpha-2))
        S = (max(var, 1e-8) / self.sigma**2)**(1 / (2*self.alpha - 2))
        vol = self.sigma * (S**(self.alpha - 1))
        
        dW = np.random.normal(0, np.sqrt(dt))
        
        # Asset Dynamics: dS/S = mu*dt + vol*dW
        new_S = S * (1 + self.mu * dt + vol * dW)
        new_S = max(1e-3, new_S)
        new_var = self.sigma**2 * (new_S**(2*self.alpha - 2))
        
        # Wealth Dynamics
        mu_p = self.r + pi * (self.mu - self.r)
        d_logW = (mu_p - 0.5 * (pi**2) * vol**2) * dt + pi * vol * dW
        
        return logW + d_logW, new_var

    def step_momentum(self, logW, var, pi, pi_prev, dt=1/252):
        lw, v = self.step(logW, var, pi, dt)
        return lw, v, pi

    def step_explicit(self, logW, var, pi, z1, z2_unused, dt=1/252):
        # CEV has only one source of noise (asset-driven vol)
        S = (max(var, 1e-8) / self.sigma**2)**(1 / (2*self.alpha - 2))
        vol = self.sigma * (S**(self.alpha - 1))
        dW = np.sqrt(dt) * z1
        
        new_S = S * (1 + self.mu * dt + vol * dW)
        new_S = max(1e-3, new_S)
        new_var = self.sigma**2 * (new_S**(2*self.alpha - 2))
        
        mu_p = self.r + pi * (self.mu - self.r)
        d_logW = (mu_p - 0.5 * (pi**2) * vol**2) * dt + pi * vol * dW
        
        return logW + d_logW, new_var

    def merton_optimal(self, var):
        return (self.mu - self.r) / (self.gamma * max(var, 1e-8))

    def step_momentum(self, logW, V_unused, pi, pi_prev, dt=1/252):
        lw, v = self.step(logW, V_unused, pi, dt)
        return lw, v, pi


class CARAUtility:
    """Exponential (CARA) Utility: U(W) = -exp(-eta * W)"""
    def __init__(self, eta=1.0):
        self.eta = eta
    def __call__(self, W):
        return -np.exp(-self.eta * W)

# =============================================================================
# 2. Feature Extractor (state-only: logW, V → 2D lead-lag signatures)
# =============================================================================

class StateSignatureExtractor:
    """Extracts lead-lag signatures from state paths.
    Supports 2D (logW, V) or 3D (logW, V, pi_prev) state paths.
    """
    def __init__(self, depth=3, device='cpu'):
        self.depth = depth
        self.device = device
        
    def extract(self, paths):
        """
        paths: torch.Tensor of shape (B, L, D) - D=2 or 3
        Returns: (B, sig_dim)
        """
        B, L, D = paths.shape
        # Manual Lead-Lag: (B, 2L-1, D*2)
        # repeated = paths.repeat_interleave(2, dim=1)
        # lead = repeated[:, :-1, :]
        # lag = repeated[:, 1:, :]
        # path_ll = torch.cat([lead, lag], dim=2)
        
        # Actually, let's use the simpler path augmentation for robustness
        # signatory doesn't have a direct 'lead_lag' helper in all versions
        # we'll do it explicitly:
        path = paths.to(self.device).to(torch.float64)
        path_repeated = torch.repeat_interleave(path, 2, dim=1)
        lead = path_repeated[:, 1:, :]
        lag = path_repeated[:, :-1, :]
        path_ll = torch.cat([lead, lag], dim=2)
        
        sig = signatory.signature(path_ll, self.depth)
        return sig


# =============================================================================
# 3. Control-Quadratic Koopman Model
# =============================================================================

class ControlQuadraticKoopman:
    """
    Fits the control-quadratic Koopman model.
    
    Modes:
      'transfer':  ψ_{t+1} = A₀·ψ_t + π·A₁·ψ_t + π²·A₂·ψ_t  (Ridge)
      'generator': (ψ_{t+1} - ψ_t)/dt = L₀·ψ_t + ...           (Ridge)
      'kkf':       Bayesian Kalman Filter with noise-calibrated prior
    """
    def __init__(self, n_features, alpha=1e-3, mode='transfer', dt=1/252):
        self.n_features = n_features
        self.alpha = alpha
        self.mode = mode
        self.dt = dt
        self.A0 = None  # L₀ in generator/kkf mode
        self.A1 = None  # L₁ in generator/kkf mode
        self.A2 = None  # L₂ in generator/kkf mode
    
    def fit(self, psi_X, psi_Y, pi_actions):
        N, d = psi_X.shape
        
        # Build features: [ψ | π·ψ | π²·ψ]
        pi = pi_actions.reshape(-1, 1)
        features = np.hstack([psi_X, pi * psi_X, pi**2 * psi_X])
        D = 3 * d
        
        # Choose targets
        if self.mode == 'generator':
            targets = (psi_Y - psi_X) / self.dt
        else:  # 'transfer' and 'kkf' both use transfer targets
            targets = psi_Y
        
        if self.mode == 'kkf':
            coeffs = self._fit_kkf(features, targets, N, D, d)
        elif self.mode == 'kkf_split':
            coeffs = self._fit_kkf_split(psi_X, psi_Y, pi_actions, N, d)
        else:
            G = features.T @ features / N + self.alpha * np.eye(D)
            H = features.T @ targets / N
            coeffs = np.linalg.solve(G, H)
        
        self.A0 = coeffs[:d, :]
        self.A1 = coeffs[d:2*d, :]
        self.A2 = coeffs[2*d:, :]
        
        # R²
        pred = features @ coeffs
        ss_res = np.sum((targets - pred)**2)
        ss_tot = np.sum((targets - targets.mean(axis=0))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"    [CQ-{self.mode.upper()}] Fit R² = {r2:.6f}")
        print(f"    [CQ-{self.mode.upper()}] |L₀| = {np.linalg.norm(self.A0):.4f}, "
              f"|L₁| = {np.linalg.norm(self.A1):.4f}, "
              f"|L₂| = {np.linalg.norm(self.A2):.4f}")
        
        return r2
    
    def _fit_kkf(self, features, targets, N, D, d):
        """
        Bayesian KKF with PCA + generator targets.
        
        Uses (ψ_Y - ψ_X)/dt targets (subtracts identity to make noise meaningful)
        with PCA dimensionality reduction for well-determined regression.
        Then converts generator L back to transfer A = I + L·dt.
        """
        from sklearn.decomposition import PCA
        
        # --- Recover raw ψ_X, ψ_Y, π ---
        psi_X_block = features[:, :d]
        # Recover π from features: features[:, d:2d] = π·ψ, so use bias term
        bias_idx = 0
        pi_recovered = features[:, d + bias_idx] / (features[:, bias_idx] + 1e-30)
        
        # Generator targets: (ψ_Y - ψ_X)/dt
        # targets was already set to ψ_Y in the calling code, so compute δ
        gen_targets = (targets - psi_X_block) / self.dt  # (N, d)
        
        # --- PCA reduction ---
        n_components = min(30, d - 1, N // 10)
        self._pca = PCA(n_components=n_components)
        psi_r = self._pca.fit_transform(psi_X_block)  # (N, n_comp)
        gen_targets_r = gen_targets @ self._pca.components_.T  # (N, n_comp)
        
        # Build reduced CQ features
        pi = pi_recovered.reshape(-1, 1)
        features_r = np.hstack([psi_r, pi * psi_r, pi**2 * psi_r])
        D_r = 3 * n_components
        
        print(f"    [KKF] PCA: {d}D → {n_components}D, features: {D_r}, N/D = {N/D_r:.1f}")
        
        # --- OLS on generator targets ---
        G_ols = features_r.T @ features_r + 1e-8 * np.eye(D_r)
        theta_ols = np.linalg.solve(G_ols, features_r.T @ gen_targets_r)
        
        residuals = gen_targets_r - features_r @ theta_ols
        dof = max(N - D_r, 1)
        sigma2_obs = np.sum(residuals**2) / (dof * n_components)
        sigma2_obs = max(sigma2_obs, 1e-10)
        
        # --- Bayesian posterior ---
        # Prior scale: how big should operator entries be?
        # Use the ratio of target variance to feature variance
        # This gives a scale-appropriate regularization
        target_var = np.var(gen_targets_r)
        feature_var = np.var(features_r) + 1e-10
        sigma2_prior = target_var / feature_var  # Expected coefficient scale²
        alpha_kkf = sigma2_obs / sigma2_prior
        
        print(f"    [KKF] σ²_obs = {sigma2_obs:.4f}, σ²_prior = {sigma2_prior:.4f}, α = {alpha_kkf:.6f}")
        
        G_bayes = features_r.T @ features_r / N + alpha_kkf * np.eye(D_r)
        theta_bayes = np.linalg.solve(G_bayes, features_r.T @ gen_targets_r / N)
        
        # Posterior covariance
        self._P_post = sigma2_obs * np.linalg.inv(features_r.T @ features_r + N * alpha_kkf * np.eye(D_r))
        
        # --- Map back to full-dim transfer operators ---
        V = self._pca.components_  # (n_comp, d)
        L0_r = theta_bayes[:n_components, :]
        L1_r = theta_bayes[n_components:2*n_components, :]
        L2_r = theta_bayes[2*n_components:, :]
        
        # Generator in original space
        L0_full = V.T @ L0_r @ V
        L1_full = V.T @ L1_r @ V
        L2_full = V.T @ L2_r @ V
        
        print(f"    [KKF] |L₀| = {np.linalg.norm(L0_full):.4f}, "
              f"|L₁| = {np.linalg.norm(L1_full):.4f}, "
              f"|L₂| = {np.linalg.norm(L2_full):.4f}")
        
        # Convert generator → transfer: A = I + L·dt
        coeffs = np.zeros((3 * d, d))
        coeffs[:d, :] = np.eye(d) + L0_full * self.dt   # A₀ = I + L₀·dt
        coeffs[d:2*d, :] = L1_full * self.dt             # A₁ = L₁·dt
        coeffs[2*d:, :] = L2_full * self.dt               # A₂ = L₂·dt
        
        return coeffs
    
    def _fit_kkf_split(self, psi_X, psi_Y, pi_actions, N, d):
        """
        Two-stage differentiated regularization.
        
        Stage 1: Fit A₀ from ψ_X → ψ_Y (identity-dominated, light regularization)
        Stage 2: Fit A₁, A₂ from the RESIDUAL ψ_Y - A₀·ψ_X with KKF-calibrated α
        
        This separates the identity signal from the tiny control-dependent signal,
        allowing proper noise estimation for the hedging operators.
        """
        pi = pi_actions.reshape(-1, 1)
        
        # --- Stage 1: Fit A₀ (identity-dominated, light regularization) ---
        G0 = psi_X.T @ psi_X / N + 1e-6 * np.eye(d)
        H0 = psi_X.T @ psi_Y / N
        A0 = np.linalg.solve(G0, H0)
        
        # Residual after removing A₀ contribution
        residual = psi_Y - psi_X @ A0  # (N, d) — the control-dependent part
        
        print(f"    [KKF-SPLIT] Stage 1: |A₀| = {np.linalg.norm(A0):.4f}, "
              f"residual std = {np.std(residual):.6f}")
        
        # --- Stage 2: Fit A₁, A₂ from residual ---
        # residual ≈ π·A₁·ψ + π²·A₂·ψ + noise
        features_ctrl = np.hstack([pi * psi_X, pi**2 * psi_X])  # (N, 2d)
        D_ctrl = 2 * d
        
        # OLS pass for noise estimation
        G_ols = features_ctrl.T @ features_ctrl + 1e-8 * np.eye(D_ctrl)
        theta_ols = np.linalg.solve(G_ols, features_ctrl.T @ residual)
        
        ols_residual = residual - features_ctrl @ theta_ols
        dof = max(N - D_ctrl, 1)
        sigma2_obs = np.sum(ols_residual**2) / (dof * d)
        sigma2_obs = max(sigma2_obs, 1e-10)
        
        # Signal variance: how big is the control-dependent residual?
        sigma2_signal = np.var(residual)
        # Feature variance
        sigma2_feat = np.var(features_ctrl) + 1e-10
        # Prior: expected coefficient scale
        sigma2_prior = sigma2_signal / sigma2_feat
        alpha_ctrl = sigma2_obs / sigma2_prior
        
        print(f"    [KKF-SPLIT] Stage 2: σ²_obs = {sigma2_obs:.6f}, "
              f"σ²_prior = {sigma2_prior:.6f}, α_ctrl = {alpha_ctrl:.6f}")
        
        # Bayesian fit on control-dependent operators
        G_ctrl = features_ctrl.T @ features_ctrl / N + alpha_ctrl * np.eye(D_ctrl)
        H_ctrl = features_ctrl.T @ residual / N
        theta_ctrl = np.linalg.solve(G_ctrl, H_ctrl)
        
        A1 = theta_ctrl[:d, :]
        A2 = theta_ctrl[d:, :]
        
        print(f"    [KKF-SPLIT] |A₁| = {np.linalg.norm(A1):.6f}, "
              f"|A₂| = {np.linalg.norm(A2):.6f}")
        
        coeffs = np.zeros((3 * d, d))
        coeffs[:d, :] = A0
        coeffs[d:2*d, :] = A1
        coeffs[2*d:, :] = A2
        
        return coeffs
# 4. CQ-KRONIC Controller
# =============================================================================

class CQKRONICMerton:
    def __init__(self, env, depth=3, mode='transfer'):
        self.env = env
        self.dt = 1/252
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = StateSignatureExtractor(depth=depth, device=self.device)
        self.koopman = None
        self.w_utility = None
        self.koopman_fit_r2 = float("nan")
        self.utility_fit_r2 = float("nan")
        
    def generate_training_data(self, n_paths=200, n_steps=252):
        """Generate (state, action, next_state) tuples with diverse pi coverage."""
        print("1. Generating Training Data with Diverse π Coverage...")
        
        psi_X_list = []
        psi_Y_list = []
        pi_list = []
        U_Y_list = []
        
        with torch.no_grad():
            for path_idx in range(n_paths):
                # Random initial conditions
                logW = np.random.normal(0, 0.1)
                V = np.random.uniform(0.02, 0.08)
                
                # CRITICAL: Vary pi broadly across the training set
                # Use a different fixed pi for each path (covers the full range)
                pi_base = np.random.uniform(0.1, 5.0)
                
                # Build state trajectory (logW, V only)
                state_traj = [[logW, V]]
                pi_traj = []
                
                for t in range(n_steps):
                    # Pi varies with some noise around the base for this path
                    pi_t = max(0.01, pi_base + 0.1 * np.random.randn())
                    pi_traj.append(pi_t)
                    
                    logW_next, V_next = self.env.step(logW, V, pi_t, dt=self.dt)
                    state_traj.append([logW_next, V_next])
                    logW, V = logW_next, V_next
                    
                    if len(state_traj) >= 4:
                        # Sliding windows of length 3
                        raw_X = np.array(state_traj[-4:-1])  # (3, 2)
                        raw_Y = np.array(state_traj[-3:])     # (3, 2)
                        
                        # Wealth Translation Invariance
                        shift = raw_X[-1, 0]
                        raw_X[:, 0] -= shift
                        raw_Y[:, 0] -= shift
                        
                        path_X = torch.from_numpy(raw_X[None]).to(device=self.device, dtype=torch.float64)
                        path_Y = torch.from_numpy(raw_Y[None]).to(device=self.device, dtype=torch.float64)
                        
                        f_X = self.extractor.extract(path_X)[0].cpu().numpy()
                        f_Y = self.extractor.extract(path_Y)[0].cpu().numpy()
                        
                        psi_X_list.append(f_X)
                        psi_Y_list.append(f_Y)
                        pi_list.append(pi_t)
                        
                        # Growth-rate utility for the next step
                        W_Y_s = np.exp(raw_Y[-1, 0])
                        U = (W_Y_s**(1 - self.env.gamma)) / (1 - self.env.gamma)
                        U_Y_list.append(U)
        
        psi_X = np.array(psi_X_list)
        psi_Y = np.array(psi_Y_list)
        pi_arr = np.array(pi_list)
        U_Y = np.array(U_Y_list)
        
        # Subsample if too many
        if len(psi_X) > 10000:
            idx = np.random.choice(len(psi_X), 10000, replace=False)
            psi_X, psi_Y, pi_arr, U_Y = psi_X[idx], psi_Y[idx], pi_arr[idx], U_Y[idx]
        
        print(f"   Generated {len(psi_X)} training samples, "
              f"π range: [{pi_arr.min():.2f}, {pi_arr.max():.2f}], "
              f"sig dim: {psi_X.shape[1]}")
        
        return psi_X, psi_Y, pi_arr, U_Y
    
    def train(self, psi_X, psi_Y, pi_arr, U_Y):
        """Fit the control-quadratic Koopman model and utility weights."""
        print(f"2. Fitting Control-Quadratic Koopman ({self.mode.upper()} mode)...")
        d = psi_X.shape[1]
        self.koopman = ControlQuadraticKoopman(
            n_features=d, mode=self.mode, dt=self.dt
        )
        self.koopman_fit_r2 = self.koopman.fit(psi_X, psi_Y, pi_arr)
        
        print("3. Fitting Growth-Rate Utility on psi_Y...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(psi_Y, U_Y)
        self.w_utility = ridge.coef_
        self.utility_fit_r2 = float(ridge.score(psi_Y, U_Y))
        print(f"   Utility R² (on psi_Y): {self.utility_fit_r2:.4f}")
        
        # Initialize: myopic (no future value)
        self.w_V = np.zeros_like(self.w_utility)
        self.w_combined = self.w_utility.copy()
        
        # Store training data for SKVI
        self._psi_X = psi_X
    
    def run_skvi(self, n_epochs=50, gamma=0.99, batch_size=2048, 
                 pi_min=0.01, pi_max=10.0):
        """
        Soft Koopman Value Iteration (inspired by KoopmanRL).
        
        Iterates the Bellman equation:
            V(ψ) = max_π [(w_r + γ·w_V) · K(π) · ψ]
        
        where K(π) = A₀ + π·A₁ + π²·A₂ is our CQ operator.
        
        Starting from w_V = 0 (myopic), each iteration propagates 
        hedging demand from future periods into the value function.
        """
        A0 = self.koopman.A0
        A1 = self.koopman.A1
        A2 = self.koopman.A2
        N = self._psi_X.shape[0]
        d = self._psi_X.shape[1]
        
        print(f"\n4. SKVI: {n_epochs} epochs, γ={gamma:.3f}")
        
        for epoch in range(n_epochs):
            # Sample a batch
            idx = np.random.choice(N, min(batch_size, N), replace=False)
            psi_batch = self._psi_X[idx]  # (B, d)
            B = len(idx)
            
            # Effective weight: reward + discounted future value
            w_eff = self.w_utility + gamma * self.w_V  # (d,)
            
            # For each state, find optimal π and compute Bellman target
            # Q(π) = w_eff · K(π) · ψ = c₀ + π·c₁ + π²·c₂
            c0_batch = psi_batch @ A0.T @ w_eff  # (B,)
            c1_batch = psi_batch @ A1.T @ w_eff  # (B,)
            c2_batch = psi_batch @ A2.T @ w_eff  # (B,)
            
            # Optimal π per state (analytical from quadratic)
            concave = c2_batch < 0
            pi_star = np.where(
                concave,
                np.clip(-c1_batch / (2 * c2_batch + 1e-30), pi_min, pi_max),
                pi_min  # fallback if not concave
            )
            
            # Bellman target: V*(ψ) = c₀ + π*·c₁ + π*²·c₂
            targets = c0_batch + pi_star * c1_batch + pi_star**2 * c2_batch
            
            # Update w_V via OLS: min ||w_V · ψ - targets||²
            ridge = Ridge(alpha=1.0, fit_intercept=False)
            ridge.fit(psi_batch, targets)
            w_V_new = ridge.coef_
            
            # Check convergence
            delta = np.linalg.norm(w_V_new - self.w_V) / (np.linalg.norm(self.w_V) + 1e-10)
            self.w_V = w_V_new
            self.w_combined = self.w_utility + gamma * self.w_V
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Compute Bellman error on batch
                V_pred = psi_batch @ self.w_V
                be = np.mean((V_pred - targets)**2)
                frac_concave = np.mean(concave)
                print(f"   Epoch {epoch+1:3d}: BE={be:.6f}, "
                      f"Δw={delta:.6f}, "
                      f"concave={frac_concave:.1%}, "
                      f"|w_V|={np.linalg.norm(self.w_V):.4f}")
                
                if delta < 1e-6:
                    print(f"   Converged at epoch {epoch+1}")
                    break
        
        print(f"   SKVI complete. |w_utility|={np.linalg.norm(self.w_utility):.4f}, "
              f"|w_V|={np.linalg.norm(self.w_V):.4f}")
    
    def evaluate_state(self, state_history, pi_min=0.01, pi_max=10.0):
        """
        Compute the optimal allocation using the quadratic structure.
        
        Uses w_combined = w_utility + γ·w_V from SKVI (if run),
        which includes hedging demand from future periods.
        """
        if self.koopman is None or self.w_combined is None:
            raise RuntimeError("CQKRONICMerton must be trained before evaluation")

        # Wealth Translation Invariance
        hist = np.asarray(state_history, dtype=float)
        if hist.ndim != 2 or hist.shape[1] < 2:
            raise ValueError("state_history must have shape (T, 2) or (T, 3)")
        if hist.shape[1] > 2:
            hist = hist[:, :2]
        if not np.all(np.isfinite(hist)):
            raise ValueError("state_history contains non-finite values")
        shift = hist[-1, 0]
        hist[:, 0] -= shift
        
        path = torch.from_numpy(hist[None]).to(device=self.device, dtype=torch.float64)
        with torch.no_grad():
            psi = self.extractor.extract(path)[0].cpu().numpy()
        if not np.all(np.isfinite(psi)):
            raise ValueError("non-finite bilinear signature features during evaluation")
        
        # Use combined weights (includes SKVI if trained)
        w = self.w_combined
        
        c0 = w @ (self.koopman.A0 @ psi)
        c1 = w @ (self.koopman.A1 @ psi)
        c2 = w @ (self.koopman.A2 @ psi)
        if not np.all(np.isfinite([c0, c1, c2])):
            raise ValueError("non-finite bilinear quadratic coefficients during allocation search")
        
        if c2 >= 0:
            pi_grid = np.linspace(pi_min, pi_max, 200)
            objs = c0 + c1 * pi_grid + c2 * pi_grid**2
            if not np.all(np.isfinite(objs)):
                raise ValueError("non-finite bilinear objective grid during allocation search")
            idx = int(np.argmax(objs))
            pi_opt = float(pi_grid[idx])
            concave = False
        else:
            pi_star = -c1 / (2 * c2)
            pi_opt = float(np.clip(pi_star, pi_min, pi_max))
            concave = True
        
        return {
            "pi_opt": pi_opt,
            "c0": float(c0),
            "c1": float(c1),
            "c2": float(c2),
            "concave": concave,
        }

    def find_optimal_pi(self, state_history, pi_min=0.01, pi_max=10.0):
        diag = self.evaluate_state(state_history, pi_min=pi_min, pi_max=pi_max)
        return diag["pi_opt"], diag["c0"], diag["c1"], diag["c2"]


# =============================================================================
# 5. Experiment: Verify the System
# =============================================================================

def run_cq_experiment(mode='transfer'):
    print("=" * 60)
    print(f"Control-Quadratic KRONIC ({mode.upper()} Mode)")
    print("=" * 60)
    
    env = HestonMertonEnv()
    cq = CQKRONICMerton(env, depth=3, mode=mode)
    
    psi_X, psi_Y, pi_arr, U_Y = cq.generate_training_data(n_paths=200, n_steps=252)
    cq.train(psi_X, psi_Y, pi_arr, U_Y)
    
    print(f"\n{'V':>8s}  {'Merton':>8s}  {'CQ':>10s}  {'c1':>10s}  {'c2':>10s}  {'Concave?':>8s}")
    print("-" * 60)
    
    for V_test in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        history = [[0.0, V_test]] * 3
        pi_opt, c0, c1, c2 = cq.find_optimal_pi(history)
        merton_opt = env.merton_optimal(V_test)
        concave = "YES" if c2 < 0 else "NO"
        print(f"  {V_test:.2f}    {merton_opt:.3f}     {pi_opt:.3f}      {c1:.6f}  {c2:.6f}  {concave}")
    
    return cq


def run_hedging_comparison():
    """Compare Transfer vs Generator vs KKF hedging demand against theory."""
    from merton_theory import canonical_state_history, stationary_heston_crra_theory

    V_test = 0.04
    
    print("=" * 75)
    print("HEDGING DEMAND: Transfer vs Generator vs KKF vs Analytical")
    print("=" * 75)
    print(f"\n  {'rho':>5s}  {'Theory':>8s}  {'Transfer':>10s}  {'Generator':>10s}  {'KKF':>10s}")
    print("  " + "-" * 50)
    
    for rho in [0.0, -0.3, -0.5, -0.7, -0.9]:
        env = HestonMertonEnv(rho=rho)
        theory_hedge = stationary_heston_crra_theory(env, V_test).hedging_demand
        eval_history = canonical_state_history(V_test, env.merton_optimal(V_test))

        results = {}
        for mode in ['transfer', 'generator', 'kkf']:
            np.random.seed(42); torch.manual_seed(42)
            cq = CQKRONICMerton(env, depth=3, mode=mode)
            pX, pY, pi, UY = cq.generate_training_data(n_paths=60, n_steps=100)
            cq.train(pX, pY, pi, UY)
            pi_opt, _, _, _ = cq.find_optimal_pi(eval_history)
            results[mode] = pi_opt - env.merton_optimal(V_test)
        
        print(f"  {rho:+.1f}   {theory_hedge:+.4f}    {results['transfer']:+.4f}     {results['generator']:+.4f}     {results['kkf']:+.4f}")


if __name__ == "__main__":
    import sys
    if '--compare' in sys.argv:
        run_hedging_comparison()
    elif '--generator' in sys.argv:
        run_cq_experiment(mode='generator')
    elif '--kkf' in sys.argv:
        run_cq_experiment(mode='kkf')
    else:
        run_cq_experiment(mode='transfer')
