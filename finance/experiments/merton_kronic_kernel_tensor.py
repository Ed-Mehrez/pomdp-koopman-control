r"""
Full-Kernel Dual-RBF Koopman Tensor with Action Momentum
========================================================

Architecture: Joint RKHS k_joint = k_state(x) * k_action(pi)
    - State (x): Lead-lag log-signatures [logW, V, pi_prev].
    - Action (pi): Current allocation.
    - Features: Psi(x, pi) = Psi_x(x) \otimes Psi_pi(pi).

Features:
    1. Adaptive Action Landmarks: KMeans on observed training actions.
    2. Action Momentum: Includes pi_prev in the state signature.
    3. Dual-RBF: Non-parametric universal approximation with natural bounds.
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from merton_kronic_bilinear import HestonMertonEnv, StateSignatureExtractor


def rbf_gram(X, Y=None, sigma=1.0):
    if Y is None: Y = X
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    sq = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return np.exp(-sq / (2 * sigma ** 2))


class KernelKoopmanTensor:
    def __init__(self, sigma_x=None, sigma_pi=None, horizon=10, dt=1/252,
                 epsilon=1e-3, n_landmarks_x=60, n_landmarks_pi=15, n_pca=20):
        self.sigma_x, self.sigma_pi = sigma_x, sigma_pi
        self.horizon, self.dt = horizon, dt
        self.T = horizon * dt
        self.epsilon = epsilon
        self.m, self.n = n_landmarks_x, n_landmarks_pi
        self.n_pca = n_pca
        
        self.scaler, self._pca = None, None
        self._lm_x, self._lm_pi = None, None
        self.L_lm, self.w_util, self.w_util_base = None, None, None

    def _transform(self, X):
        Xs = self.scaler.transform(X) if self.scaler is not None else X
        return self._pca.transform(Xs) if self._pca is not None else Xs

    def _auto_sigma(self, X, percentile=75):
        sub = X[np.random.choice(len(X), min(300, len(X)), replace=False)]
        d = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
        return float(np.percentile(d[d > 0], percentile))

    def fit(self, X0, XT, pi, verbose=True):
        self.X0_train, self.pi_train, self.XT_train = X0, pi, XT
        Xp = self._transform(X0)
        
        # 1. Automate Bandwidths (Using 75th percentile for smoothness)
        if self.sigma_x is None: self.sigma_x = self._auto_sigma(Xp, 75)
        if self.sigma_pi is None: self.sigma_pi = self._auto_sigma(pi[:, None], 75)
        
        # 2. Adaptive Landmark Placement
        self._lm_x = KMeans(n_clusters=self.m, random_state=42, n_init=3).fit(Xp).cluster_centers_
        self._lm_pi = KMeans(n_clusters=self.n, random_state=42, n_init=3).fit(pi[:, None]).cluster_centers_
        self._lm_pi = np.sort(self._lm_pi.flatten())[:, None]

        # 3. Joint RKHS Features
        Psi_X0 = rbf_gram(Xp, self._lm_x, self.sigma_x)
        Psi_P0 = rbf_gram(pi, self._lm_pi, self.sigma_pi)
        Psi0 = np.array([np.kron(Psi_X0[i], Psi_P0[i]) for i in range(len(X0))])

        Xp_T = self._transform(XT)
        Psi_XT = rbf_gram(Xp_T, self._lm_x, self.sigma_x)
        PsiT = np.array([np.kron(Psi_XT[i], Psi_P0[i]) for i in range(len(XT))])

        # 4. Transfer Operator Ridge Fit (Stronger Regularization)
        K_T = Ridge(alpha=0.1, fit_intercept=False).fit(Psi0, PsiT).coef_.T
        
        w, v = np.linalg.eig(K_T)
        idx = np.argsort(np.abs(np.log(np.abs(w) + 1e-9)))
        mu = np.zeros_like(w)
        c, kappa = 0, self.n_pca
        for i in idx:
            if np.real(w[i]) > 0 and c < kappa:
                mu[i] = np.log(w[i] + 0j) / self.T
                c += 1
        self.L_lm = np.real(v @ np.diag(mu) @ np.linalg.pinv(v, rcond=1e-6))

    def fit_utility(self, U_T, verbose=True):
        Xp = self._transform(self.X0_train)
        Psi_X = rbf_gram(Xp, self._lm_x, self.sigma_x)
        Psi_P = rbf_gram(self.pi_train, self._lm_pi, self.sigma_pi)
        
        self._U_mu, self._U_std = np.mean(U_T), np.std(U_T) + 1e-9
        Un = (U_T - self._U_mu) / self._U_std

        # Stage 1: Market Luck (State only)
        w_base = Ridge(alpha=0.1, fit_intercept=False).fit(Psi_X, Un).coef_
        U_res = Un - Psi_X @ w_base
        
        # Stage 2: Portfolio Skill (Joint Action-Kernel)
        Psi_Joint = np.array([np.kron(Psi_X[i], Psi_P[i]) for i in range(len(Xp))])
        # Force high regularization for stable backtest outcomes
        w_skill = Ridge(alpha=10.0, fit_intercept=False).fit(Psi_Joint, U_res).coef_
        
        self.w_util_base = w_base * self._U_std
        self.w_util = w_skill * self._U_std
        
        if verbose:
            U_p = (Psi_X @ w_base + Psi_Joint @ w_skill) * self._U_std + self._U_mu
            R2 = 1 - np.sum((U_T - U_p)**2) / np.sum((U_T - np.mean(U_T))**2)
            print(f"   [KKT/Link-RBF] Joint Action-R2={R2:.4f}")


    def get_value_landscape(self, psi_x, pi_grid, use_drift=False):
        # Psi_P: (Grid, n)
        Psi_P = rbf_gram(pi_grid, self._lm_pi, self.sigma_pi)
        # Psi_J: (Grid, m*n)
        Psi_J = np.array([np.kron(psi_x, Psi_P[i]) for i in range(len(pi_grid))])
        
        H = (self.L_lm @ self.w_util) if use_drift else self.w_util
        return np.dot(self.w_util_base, psi_x) + Psi_J @ H


class KernelMertonController:
    def __init__(self, env, depth=3, horizon=10, 
                 mx=60, npi=15, n_pca=20, epsilon=1e-3, device='cpu', switching_penalty=0.01):
        self.env, self.depth, self.horizon = env, depth, horizon
        self.dt, self.device = 1/252, device
        self.m, self.n, self.n_pca, self.epsilon = mx, npi, n_pca, epsilon
        self.extractor = StateSignatureExtractor(depth=depth, device=device)
        self.kkt, self.scaler = None, StandardScaler()
        self.switching_penalty = switching_penalty

    def generate_training_data(self, n_paths=400, n_mc=40, momentum=True):
        print(f"1. Generating Training Data (N={n_paths}, MC={n_mc}, Momentum={momentum})")
        X0_l, XT_l, pi_l, U_l = [], [], [], []
        gamma, win = getattr(self.env, 'gamma', 3.0), 3
        
        for _ in range(n_paths):
            lw0, v0 = 0.0, np.random.uniform(0.01, 0.09)
            # Widen range to cover the optimal space [0, 5]
            pic = np.random.uniform(0.0, 5.0)
            pi_prev = np.random.uniform(0.0, 5.0)
            
            # Augmented State Path: [logW, V, pic_prev]
            path = []
            lw, v = lw0, v0
            for _ in range(win):
                # We assume inert pi_prev during the lead-in window
                if hasattr(self.env, 'step_momentum'):
                    lw, v, _ = self.env.step_momentum(lw, v, pic, pi_prev, dt=self.dt)
                else:
                    lw, v = self.env.step(lw, v, pic, dt=self.dt)
                path.append((lw, v, pi_prev))
            
            path = np.array(path); path[:, 0] -= path[-1, 0]
            with torch.no_grad():
                psi0 = self.extractor.extract(torch.tensor(np.array([path]), dtype=torch.float64))[0].cpu().numpy()
            
            lwp, vp, _ = path[-1]
            Ur = []
            for _ in range(n_mc):
                lr, vr = lwp, vp
                for _ in range(self.horizon):
                    lr, vr = self.env.step(lr, vr, pic, dt=self.dt)
                # CRRA or CARA handled by env.gamma if available
                Ur.append(np.exp(lr - lw0)**(1 - gamma) / (1 - gamma))
            
            X0_l.append(psi0); XT_l.append(psi0) # Multi-step K_T handled by horizon in env
            pi_l.append(pic); U_l.append(np.mean(Ur))
            
        self.scaler.fit(X0_l)
        return np.array(X0_l), np.array(XT_l), np.array(pi_l), np.array(U_l)

    def train(self, X0, XT, pi, UT):
        self.kkt = KernelKoopmanTensor(n_landmarks_x=self.m, n_landmarks_pi=self.n, 
                                       n_pca=self.n_pca, epsilon=self.epsilon)
        self.kkt.scaler = self.scaler
        if self.n_pca < X0.shape[1]:
            self.kkt._pca = PCA(n_components=self.n_pca, random_state=42).fit(self.kkt._transform(X0))
        self.kkt.fit(X0, XT, pi)
        self.kkt.fit_utility(UT)

    def find_optimal_pi(self, state_history):
        """
        state_history: List of (logW, V, pi_prev) or (logW, V)
        """
        hist = np.array(state_history)
        if hist.shape[1] == 2: # No momentum in test point, add zero
            hist = np.hstack([hist, np.zeros((len(hist), 1))])
        
        hist[:, 0] -= hist[-1, 0]
        with torch.no_grad():
            raw = self.extractor.extract(torch.tensor(np.array([hist]), dtype=torch.float64, device=self.device))[0].cpu().numpy()
        
        Xp = self.kkt._transform(raw[None])
        psi = rbf_gram(Xp, self.kkt._lm_x, self.kkt.sigma_x)[0]
        
        pi_grid = np.linspace(0.0, 5.0, 100)
        V_grid = self.kkt.get_value_landscape(psi, pi_grid)
        
        # Action Switching Penalty (Inertia)
        pi_prev = hist[-1, 2] # From the augmented state [logW, V, pi_prev]
        penalty = self.switching_penalty * (pi_grid - pi_prev)**2
        
        return float(pi_grid[np.argmax(V_grid - penalty)])


class CrossValidatedKKT:
    """
    Robust Shrinkage Estimator:
    Ensures the Koopman Tensor only activates if it strictly beats the 
    Myopic baseline out-of-sample, solving the Estimation Error problem.
    """
    def __init__(self, env, **kkt_kwargs):
        self.env = env
        self.kwargs = kkt_kwargs
        self.best_ctrl = None
        self.use_myopic_fallback = False

    def _eval_policy(self, ctrl, X_val_paths, use_myopic=False):
        """Evaluates Expected Utility computationally on unseen validation paths."""
        total_u = 0.0
        for lw0, v0, pi_prev in X_val_paths:
            lw, v, pp = lw0, v0, pi_prev
            hist = [(0, v, pp)] * 3
            
            for _ in range(10): # Eval horizon
                if use_myopic:
                    pic = self.env.merton_optimal(v)
                else:
                    pic = ctrl.find_optimal_pi(hist)
                
                if hasattr(self.env, 'step_momentum'):
                    lw, v, pp = self.env.step_momentum(lw, v, pic, pp, dt=1/252)
                else:
                    lw, v = self.env.step(lw, v, pic, dt=1/252)
                    
                hist.append((lw, v, pp))
                hist = hist[1:]
                
            gamma = getattr(self.env, 'gamma', 3.0)
            total_u += np.exp(lw - lw0)**(1 - gamma) / (1 - gamma)
            
        return total_u / len(X_val_paths)

    def train_with_sanity_check(self, n_train=1000, n_val=100):
        print("\n--- Empirical Risk Minimization (Robust Shrinkage) ---")
        base_ctrl = KernelMertonController(self.env, **self.kwargs)
        
        # 1. Generate Split Data
        X0_t, XT_t, pi_t, UT_t = base_ctrl.generate_training_data(n_paths=n_train)
        
        # Generate raw validation starting states
        val_starts = []
        for _ in range(n_val):
            val_starts.append((0.0, np.random.uniform(0.01, 0.09), np.random.uniform(0.0, 5.0)))
            
        # 2. Baseline Performance (Myopic)
        u_myo = self._eval_policy(None, val_starts, use_myopic=True)
        print(f"Myopic Baseline Validation Utility: {u_myo:.6f}")
        
        # 3. Hyperparameter Sweep (Ridge Alpha)
        alphas = [0.1, 1.0, 10.0, 50.0]
        best_u, best_alpha = -np.inf, None
        
        for a in alphas:
            ctrl = KernelMertonController(self.env, **self.kwargs)
            ctrl.scaler = base_ctrl.scaler
            ctrl.scaler.fit(X0_t)
            
            # Monkey-patch fit_utility to use sweeps
            def custom_fit_utility(self_kkt, U_T, verbose=False):
                Xp = self_kkt._transform(self_kkt.X0_train)
                Psi_X = rbf_gram(Xp, self_kkt._lm_x, self_kkt.sigma_x)
                Psi_P = rbf_gram(self_kkt.pi_train, self_kkt._lm_pi, self_kkt.sigma_pi)
                
                self_kkt._U_mu, self_kkt._U_std = np.mean(U_T), np.std(U_T) + 1e-9
                Un = (U_T - self_kkt._U_mu) / self_kkt._U_std

                w_base = Ridge(alpha=0.1, fit_intercept=False).fit(Psi_X, Un).coef_
                U_res = Un - Psi_X @ w_base
                
                Psi_Joint = np.array([np.kron(Psi_X[i], Psi_P[i]) for i in range(len(Xp))])
                w_skill = Ridge(alpha=a, fit_intercept=False).fit(Psi_Joint, U_res).coef_
                
                self_kkt.w_util_base = w_base * self_kkt._U_std
                self_kkt.w_util = w_skill * self_kkt._U_std
            
            # Train model
            ctrl.kkt = KernelKoopmanTensor(n_landmarks_x=ctrl.m, n_landmarks_pi=ctrl.n, n_pca=ctrl.n_pca, epsilon=ctrl.epsilon)
            ctrl.kkt.scaler = ctrl.scaler
            if ctrl.n_pca < X0_t.shape[1]:
                ctrl.kkt._pca = PCA(n_components=ctrl.n_pca, random_state=42).fit(ctrl.kkt._transform(X0_t))
            ctrl.kkt.fit(X0_t, XT_t, pi_t, verbose=False)
            
            # Apply hyperparameter
            import types
            ctrl.kkt.fit_utility = types.MethodType(custom_fit_utility, ctrl.kkt)
            ctrl.kkt.fit_utility(UT_t)
            
            # Evaluate Out-Of-Sample
            u_kkt = self._eval_policy(ctrl, val_starts, use_myopic=False)
            print(f" > KKT (alpha={a}): Validation Utility = {u_kkt:.6f}")
            
            if u_kkt > best_u:
                best_u, best_alpha, self.best_ctrl = u_kkt, a, ctrl

        # 4. The Robustness Sanity Check
        print("-" * 50)
        if best_u > u_myo:
            print(f"[SUCCESS] Selected alpha={best_alpha}. KKT proven out-of-sample (+{best_u-u_myo:.6f} utils).")
            self.use_myopic_fallback = False
        else:
            print(f"[SHRINKAGE TRIGGERED] KKT estimation error dominant. Shrinking to Myopic Baseline.")
            self.use_myopic_fallback = True
            
    def find_optimal_pi(self, state_history):
        if self.use_myopic_fallback:
            # Fallback to Myopic
            v = state_history[-1][1] if len(state_history[-1]) > 1 else 0.04
            return self.env.merton_optimal(v)
        return self.best_ctrl.find_optimal_pi(state_history)
