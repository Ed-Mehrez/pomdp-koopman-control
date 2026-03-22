"""
Stage 2: LP Control in Lifted Eigenfunction Space
==================================================

The key insight from Koopman theory:

In eigenfunction coordinates, the value function becomes LINEAR:
  E[U(W_T)] = Σ wᵢ e^{λᵢ(π)T} ψᵢ(z₀)

where:
  - ψᵢ(z) are Koopman eigenfunctions
  - λᵢ(π) are eigenvalues (depend on control π)
  - wᵢ are weights (from initial condition)

This makes optimal control a LINEAR PROGRAM in eigenfunction space!

Architecture:
1. Stage 1: RecSig-RLS filter gives σ̂²_t (from stage1_documented_approach.py)
2. Stage 2: KGEDMD learns eigenfunctions φ(W, π, σ̂²)
3. Stage 3: LP finds optimal π* by maximizing expected utility

Two targeting approaches:
A. Observable targeting: Train on r²/dt (noisy but observable)
B. Latent targeting: Train on V (smoother, like value function targeting)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eig
from scipy.optimize import linprog, minimize_scalar
from examples.proof_of_concept.signature_features import RecurrentSignatureMap

np.random.seed(42)


# =============================================================================
# Heston-Merton Environment
# =============================================================================
class HestonMertonEnv:
    """
    Merton portfolio problem with Heston stochastic volatility.

    State: (W, π, V) - Wealth, allocation, variance
    Control: π (portfolio allocation to risky asset)
    Dynamics:
      dW/W = (r + π(μ-r))dt + π√V dB₁
      dV = κ(θ-V)dt + ξ√V dB₂
      dB₁·dB₂ = ρdt
    """
    def __init__(self, mu=0.08, r=0.02, kappa=2.0, theta=0.04, xi=0.3,
                 rho=-0.7, gamma=2.0, dt=1/252):
        self.mu = mu
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.gamma = gamma
        self.dt = dt

    def step(self, W, V, pi):
        """Single step of dynamics."""
        sqrt_dt = np.sqrt(self.dt)
        v = max(V, 1e-8)

        z1 = np.random.randn()
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.randn()

        # Wealth dynamics
        dW = W * ((self.r + pi * (self.mu - self.r)) * self.dt
                  + pi * np.sqrt(v) * sqrt_dt * z1)
        W_new = max(W + dW, 1e-6)

        # Variance dynamics
        dV = self.kappa * (self.theta - v) * self.dt + self.xi * np.sqrt(v) * sqrt_dt * z2
        V_new = max(V + dV, 1e-8)

        # Return for signature filter
        ret = np.log(W_new / W)

        return W_new, V_new, ret

    def crra_utility(self, W):
        """CRRA utility function."""
        if self.gamma == 1:
            return np.log(W)
        return W**(1 - self.gamma) / (1 - self.gamma)

    def merton_optimal_pi(self, V):
        """Myopic Merton allocation (ignores vol dynamics)."""
        return (self.mu - self.r) / (self.gamma * V)

    def chacko_viceira_pi(self, V):
        """
        Chacko-Viceira (2005) approximate optimal allocation under stochastic vol.

        π* = π_myopic + π_hedging

        The hedging demand accounts for:
        - Leverage effect (ρ < 0 → vol spikes when prices drop)
        - Mean reversion (κ pulls V back to θ)
        - Vol of vol (ξ)

        Approximate formula (Campbell & Viceira 2002, eq 6.27):
        π_hedging ≈ (1 - 1/γ) × (ρ × ξ / V) × (1 / (2κ))

        This reduces allocation when ρ < 0 to hedge against vol risk.
        """
        pi_myopic = (self.mu - self.r) / (self.gamma * V)

        # Hedging demand (approximate)
        # Factor: (1 - 1/γ) captures intertemporal hedging motive
        # For γ > 1: want to hedge against bad states
        # For γ < 1: want to speculate on vol
        if self.gamma > 1:
            hedging_factor = (1 - 1/self.gamma)
        else:
            hedging_factor = 0  # Log utility → no hedging demand

        # Hedging term: negative when ρ < 0
        pi_hedging = hedging_factor * (self.rho * self.xi / np.sqrt(V)) * (1 / (2 * self.kappa))

        return pi_myopic + pi_hedging


# =============================================================================
# Stage 1: Volatility Filter (from documented approach)
# =============================================================================
class VolatilityFilter:
    """
    RecSig-RLS filter from Stage 1.

    Supports two targeting modes:
    A. 'observable': Train on r²/dt (noisy but generalizable)
    B. 'latent': Train on V (smoother, requires oracle for training)
    """
    def __init__(self, forgetting_factor=0.94, rls_ff=0.999, dt=1/252,
                 target_mode='observable'):
        self.ff = forgetting_factor
        self.rls_ff = rls_ff
        self.dt = dt
        self.target_mode = target_mode
        self.name = f"RecSig-RLS({target_mode})"

        # Signature state (2D: time, log-price)
        self.sig_map = RecurrentSignatureMap(state_dim=2, level=2,
                                              forgetting_factor=forgetting_factor)

        # RLS state
        self.n_features = self.sig_map.feature_dim + 1
        self.w = np.zeros(self.n_features)
        self.P = np.eye(self.n_features) * 100.0

        self.last_log_W = None

    def reset(self, W0=1.0):
        self.sig_map.reset()
        self.last_log_W = np.log(W0)

    def update(self, W, V_true=None):
        """
        Update filter with new wealth observation.

        Args:
            W: Current wealth
            V_true: True variance (only used if target_mode='latent')
        """
        log_W = np.log(W)

        if self.last_log_W is None:
            self.last_log_W = log_W
            return 0.04  # Return long-run mean

        # Compute increment
        d_log_W = log_W - self.last_log_W
        dx = np.array([self.dt, d_log_W])

        # Update signature
        sig_features = self.sig_map.update(dx)
        features = np.concatenate([sig_features, [1.0]])

        # Target based on mode
        if self.target_mode == 'latent' and V_true is not None:
            target = V_true
        else:
            # Observable: r²/dt as variance proxy
            target = d_log_W**2 / self.dt
            target = min(target, 1.0)

        # RLS update
        z = features[:, np.newaxis]
        Pz = self.P @ z
        denom = self.rls_ff + (z.T @ Pz)[0, 0]
        k = Pz / denom

        pred = np.dot(self.w, features)
        error = target - pred

        self.w = self.w + k.flatten() * error
        self.P = (self.P - k @ Pz.T) / self.rls_ff

        self.last_log_W = log_W

        return max(pred, 1e-6)


# =============================================================================
# Stage 1b: Estimate Volatility Dynamics from Observables
# =============================================================================
class VolatilityDynamicsEstimator:
    """
    Estimates Heston-like vol dynamics from observable r²/dt.

    Model: dV = κ(θ - V)dt + ξ√V dB

    From r²/dt observations, we estimate:
    - θ̂ = E[r²/dt] (long-run variance)
    - κ̂ from autocorrelation decay
    - ξ̂ from variance of variance
    """
    def __init__(self):
        self.kappa_hat = None
        self.theta_hat = None
        self.xi_hat = None

    def fit(self, r2_dt_series, dt=1/252):
        """
        Estimate parameters from r²/dt time series.

        Uses moment matching and autocorrelation:
        1. θ̂ = mean(r²/dt)
        2. κ̂ from lag-1 autocorrelation: ρ(1) ≈ exp(-κ·dt)
        3. ξ̂ from variance of variance
        """
        r2 = np.array(r2_dt_series)
        n = len(r2)

        # 1. Long-run mean
        self.theta_hat = np.mean(r2)

        # 2. Mean reversion from autocorrelation
        # ρ(lag) ≈ exp(-κ·lag·dt) for OU process
        if n > 10:
            r2_centered = r2 - self.theta_hat
            # Lag-1 autocorrelation
            rho1 = np.corrcoef(r2_centered[:-1], r2_centered[1:])[0, 1]
            rho1 = max(0.01, min(0.99, rho1))  # Clip to valid range
            self.kappa_hat = -np.log(rho1) / dt
        else:
            self.kappa_hat = 2.0  # Default

        # 3. Vol-of-vol from variance of squared returns
        # For Heston: Var[V] = θξ² / (2κ)
        # So: ξ² = 2κ·Var[V] / θ
        var_v = np.var(r2)
        if self.theta_hat > 0:
            xi_sq = 2 * self.kappa_hat * var_v / self.theta_hat
            self.xi_hat = np.sqrt(max(xi_sq, 0.01))
        else:
            self.xi_hat = 0.3  # Default

        # Bound estimates to reasonable ranges
        self.kappa_hat = np.clip(self.kappa_hat, 0.5, 10.0)
        self.xi_hat = np.clip(self.xi_hat, 0.1, 2.0)

        return self

    def predict_drift(self, sigma2):
        """μ_V(σ²) = κ(θ - σ²)"""
        return self.kappa_hat * (self.theta_hat - sigma2)

    def predict_diffusion(self, sigma2):
        """σ_V(σ²) = ξ√σ²"""
        return self.xi_hat * np.sqrt(max(sigma2, 1e-8))


# =============================================================================
# Stage 2: KGEDMD for Eigenfunction Learning
# =============================================================================
class KGEDMDController:
    """
    Kernel Generator EDMD with Nyström approximation.

    Learns eigenfunctions of the Koopman generator from trajectory data.
    Uses these to find optimal allocation via eigenvalue maximization.

    NEW: Can incorporate estimated vol dynamics (κ̂, θ̂, ξ̂) for proper
    generator computation in the certainty-equivalence sense.
    """
    def __init__(self, n_landmarks=100, gamma_rbf=1.0, vol_dynamics=None):
        self.n_landmarks = n_landmarks
        self.gamma_rbf = gamma_rbf
        self.vol_dynamics = vol_dynamics  # VolatilityDynamicsEstimator

        self.nystroem = None
        self.scaler = StandardScaler()
        self.eigenfunctions = None
        self.eigenvalues = None
        self.A_matrix = None

    def collect_data(self, env, filter_obs, filter_lat, n_episodes=50,
                     n_steps=252, pi_range=(0.2, 1.5)):
        """
        Collect trajectory data for KGEDMD.

        Returns:
            states: (N, 5) array [W, π, σ̂²_obs, σ̂²_lat, V_true]
            utilities: (N,) array of CRRA utilities
            r2_dt_series: list of r²/dt values for vol dynamics estimation
            episode_ends: list of indices where episodes end (for masking)
        """
        print("  Collecting trajectory data...")

        states = []
        utilities = []
        r2_dt_series = []  # For estimating vol dynamics
        episode_ends = []  # Track episode boundaries

        for ep in range(n_episodes):
            W = 1.0
            V = env.theta
            filter_obs.reset(W)
            filter_lat.reset(W)

            # Random exploration policy
            pi_base = np.random.uniform(*pi_range)

            for t in range(n_steps):
                # Random perturbation for exploration
                pi = pi_base + 0.1 * np.random.randn()
                pi = np.clip(pi, 0.1, 2.0)

                # Get volatility estimates
                sigma2_obs = filter_obs.update(W, V)
                sigma2_lat = filter_lat.update(W, V)

                # Record state
                states.append([W, pi, sigma2_obs, sigma2_lat, V])
                utilities.append(env.crra_utility(W))

                # Step environment
                W_old = W
                W, V, ret = env.step(W, V, pi)

                # Record r²/dt for vol dynamics estimation
                r2_dt = ret**2 / env.dt
                r2_dt_series.append(r2_dt)

            # Mark episode end
            episode_ends.append(len(states) - 1)

        states = np.array(states)
        utilities = np.array(utilities)

        print(f"  Collected {len(states)} state-action pairs across {n_episodes} episodes")

        return states, utilities, r2_dt_series, episode_ends

    def fit_growth_rate(self, states, returns, use_latent=False, episode_ends=None):
        """
        Fit a model of the growth rate g(π, σ²) directly from log-returns.

        This is simpler and more robust than eigenfunction-based prediction.

        The model: g(π, σ²) = r + π(μ-r) - (γ/2)π²σ²
        We fit this as a polynomial in (π, σ², π², πσ², σ²²)
        """
        print("  Fitting growth rate model...")

        # Select σ² based on targeting mode
        if use_latent:
            sigma2_col = 3  # σ̂²_lat
        else:
            sigma2_col = 2  # σ̂²_obs

        # Features: [1, π, σ², π², πσ²]
        pi = states[:-1, 1]
        sigma2 = states[:-1, sigma2_col]
        r = returns[:-1]  # One-step log returns

        # Mask episode boundaries
        if episode_ends is not None:
            valid_mask = np.ones(len(pi), dtype=bool)
            for end_idx in episode_ends:
                if end_idx < len(valid_mask):
                    valid_mask[end_idx] = False
            pi = pi[valid_mask]
            sigma2 = sigma2[valid_mask]
            r = r[valid_mask]

        # Design matrix (polynomial features)
        X = np.column_stack([
            np.ones(len(pi)),  # constant (r)
            pi,                 # π term (μ-r)
            pi**2 * sigma2,     # π²σ² term (-γ/2)
        ])

        # Target: annualized log return
        dt = 1/252
        y = r / dt  # Annualized

        # Ridge regression
        ridge_alpha = 0.01
        XtX = X.T @ X
        Xty = X.T @ y
        self.growth_coeffs = np.linalg.solve(XtX + ridge_alpha * np.eye(3), Xty)

        print(f"    Fitted growth rate: g(π,σ²) ≈ {self.growth_coeffs[0]:.4f} + {self.growth_coeffs[1]:.4f}π - {-self.growth_coeffs[2]:.4f}π²σ²")
        print(f"    Implied γ ≈ {-2*self.growth_coeffs[2]:.2f}")

        return self

    def find_optimal_pi_growth(self, sigma2):
        """
        Find optimal π by maximizing the fitted growth rate.

        g(π, σ²) = c₀ + c₁π + c₂π²σ²
        dg/dπ = c₁ + 2c₂πσ² = 0
        π* = -c₁ / (2c₂σ²)
        """
        if not hasattr(self, 'growth_coeffs'):
            return 0.75  # Default

        c1, c2 = self.growth_coeffs[1], self.growth_coeffs[2]
        if c2 != 0 and sigma2 > 0:
            pi_opt = -c1 / (2 * c2 * sigma2)
            return np.clip(pi_opt, 0.1, 2.0)
        return 0.75

    def fit(self, states, utilities, use_latent=False, episode_ends=None):
        """
        Fit KGEDMD model.

        Args:
            states: (N, 5) array [W, π, σ̂²_obs, σ̂²_lat, V_true]
            utilities: (N,) array of CRRA utilities
            use_latent: If True, use latent σ² estimate
            episode_ends: list of indices where episodes end (to mask boundaries)
        """
        print("  Fitting KGEDMD...")

        # Select features based on targeting mode
        if use_latent:
            # Use latent-trained filter estimate
            Z = states[:, [0, 1, 3]]  # W, π, σ̂²_lat
            print("    Using LATENT targeting (σ̂²_lat)")
        else:
            # Use observable-trained filter estimate
            Z = states[:, [0, 1, 2]]  # W, π, σ̂²_obs
            print("    Using OBSERVABLE targeting (σ̂²_obs)")

        # Normalize features
        Z_scaled = self.scaler.fit_transform(Z)

        # Nyström approximation for kernel features
        self.nystroem = Nystroem(kernel='rbf', gamma=self.gamma_rbf,
                                  n_components=self.n_landmarks, random_state=42)
        Phi = self.nystroem.fit_transform(Z_scaled)

        print(f"    Kernel features: {Phi.shape[1]}")

        # Compute generator approximation via finite differences
        # G ≈ (Φ_{t+1} - Φ_t) / dt
        dt = 1/252
        Phi_now = Phi[:-1]
        Phi_next = Phi[1:]
        G_approx = (Phi_next - Phi_now) / dt
        U_now = utilities[:-1]

        # Mask out episode boundaries (don't use transitions across episodes)
        if episode_ends is not None:
            valid_mask = np.ones(len(Phi_now), dtype=bool)
            for end_idx in episode_ends:
                if end_idx < len(valid_mask):
                    valid_mask[end_idx] = False
            Phi_now = Phi_now[valid_mask]
            G_approx = G_approx[valid_mask]
            U_now = U_now[valid_mask]
            print(f"    Masked {(~valid_mask).sum()} episode boundaries, using {valid_mask.sum()} transitions")

        # Solve for A matrix: G ≈ Φ_now @ A.T
        # Using RIDGE regression for stability (prevents spurious positive eigenvalues)
        ridge_alpha = 0.1  # Regularization strength
        Phi_T_Phi = Phi_now.T @ Phi_now
        reg_matrix = ridge_alpha * np.eye(Phi_T_Phi.shape[0])
        A = np.linalg.solve(Phi_T_Phi + reg_matrix, Phi_now.T @ G_approx)
        self.A_matrix = A.T

        # Eigendecomposition
        eigenvalues, eigenvectors = eig(self.A_matrix)

        # CRITICAL: Clip positive eigenvalues to zero (enforce stability)
        # Positive eigenvalues cause exp(λT) to explode
        eigenvalues_clipped = np.where(
            np.real(eigenvalues) > 0,
            -0.01 + 1j * np.imag(eigenvalues),  # Small negative instead of positive
            eigenvalues
        )
        n_clipped = np.sum(np.real(eigenvalues) > 0)
        if n_clipped > 0:
            print(f"    WARNING: Clipped {n_clipped} positive eigenvalues to -0.01")

        # Sort by real part (most stable/dominant modes)
        idx = np.argsort(-np.real(eigenvalues_clipped))
        self.eigenvalues = eigenvalues_clipped[idx]
        self.eigenvectors = eigenvectors[:, idx]

        print(f"    Top eigenvalues: {np.real(self.eigenvalues[:5])}")

        # Store utility projection onto eigenfunctions
        self.utility_weights = np.linalg.lstsq(Phi_now, U_now, rcond=None)[0]

        # Store training data statistics for diagnostics
        self.train_Z = Z
        self.train_U = utilities

        return self

    def diagnose(self, env, n_test_points=5):
        """
        Diagnose what KGEDMD is learning.
        """
        print("\n  === KGEDMD Diagnostics ===")

        # 1. Check eigenvalue spectrum
        real_eigs = np.real(self.eigenvalues[:10])
        print(f"  Top 10 eigenvalues (real part): {real_eigs}")
        print(f"  All negative? {all(real_eigs < 0)}")

        # 2. Check utility predictions vs grid search
        print("\n  Utility landscape at (W=1.0, σ²=0.04):")
        pi_grid = np.linspace(0.2, 1.5, 7)
        for pi in pi_grid:
            u_pred = self.predict_utility(1.0, pi, 0.04, T=1.0)
            print(f"    π={pi:.2f}: U_pred={u_pred:.4f}")

        # 3. Check optimal π at different σ²
        print("\n  Optimal π at different σ² (should decrease with σ²):")
        for sigma2 in [0.02, 0.04, 0.06, 0.08]:
            pi_opt, u_opt = self.find_optimal_pi(1.0, sigma2, method='scalar')
            merton_pi = (env.mu - env.r) / (env.gamma * sigma2)
            print(f"    σ²={sigma2:.2f}: π*_kgedmd={pi_opt:.3f}, π*_merton={merton_pi:.3f}")

        # 4. Check training data range
        print("\n  Training data statistics:")
        print(f"    W range: [{self.train_Z[:, 0].min():.3f}, {self.train_Z[:, 0].max():.3f}]")
        print(f"    π range: [{self.train_Z[:, 1].min():.3f}, {self.train_Z[:, 1].max():.3f}]")
        print(f"    σ² range: [{self.train_Z[:, 2].min():.4f}, {self.train_Z[:, 2].max():.4f}]")
        print(f"    U range: [{self.train_U.min():.4f}, {self.train_U.max():.4f}]")

    def predict_utility(self, W, pi, sigma2, T=1.0):
        """
        Predict expected utility at time T using Koopman propagation.

        E[U(W_T)] ≈ Σ wᵢ e^{λᵢT} ψᵢ(z₀)
        """
        z = np.array([[W, pi, sigma2]])
        z_scaled = self.scaler.transform(z)
        phi = self.nystroem.transform(z_scaled)[0]

        # Project to eigenspace
        c = self.eigenvectors.T @ phi

        # Propagate
        prop = np.exp(np.real(self.eigenvalues) * T) * c

        # Back to utility
        phi_T = self.eigenvectors @ prop
        U_T = np.real(np.dot(phi_T, self.utility_weights))

        return U_T

    def predict_growth_rate(self, pi, sigma2):
        """
        Predict effective growth rate g(π, σ²) using the analytical formula.

        For CRRA with risk aversion γ:
          g(π, σ²) = r + π(μ-r) - (γ/2)π²σ²

        The optimal π* = argmax g(π, σ²) = (μ-r)/(γσ²) (Merton's formula)

        This is more direct than trying to learn eigenfunctions.
        """
        # Use stored env parameters (must be set during fit)
        mu = 0.08  # Will be overwritten
        r = 0.02
        gamma = 2.0

        if self.vol_dynamics is not None:
            # Use estimated long-run variance as reference
            pass

        # Theoretical growth rate
        g = r + pi * (mu - r) - (gamma / 2) * pi**2 * sigma2
        return g

    def find_optimal_pi(self, W, sigma2, T=1.0, pi_grid=None, method='scalar'):
        """
        Find optimal allocation.

        Methods:
        - 'grid': Grid search (baseline)
        - 'scalar': Scipy minimize_scalar (faster)
        - 'lp': Linear program over discretized actions (principled)
        """
        if method == 'scalar':
            # Scalar optimization (fastest for 1D control)
            def neg_utility(pi):
                return -self.predict_utility(W, pi, sigma2, T)

            result = minimize_scalar(neg_utility, bounds=(0.1, 2.0), method='bounded')
            return result.x, -result.fun

        elif method == 'lp':
            # LP formulation: maximize over probability distribution on discrete π
            # c @ p = expected utility (we negate for minimization)
            # A_eq @ p = 1 (probabilities sum to 1)
            # p >= 0

            if pi_grid is None:
                pi_grid = np.linspace(0.1, 2.0, 20)

            # Compute expected utility for each discrete action
            utilities = np.array([self.predict_utility(W, pi, sigma2, T) for pi in pi_grid])

            # LP: minimize -c @ p s.t. A_eq @ p = b_eq, p >= 0
            c = -utilities  # Negative because linprog minimizes
            A_eq = np.ones((1, len(pi_grid)))
            b_eq = np.array([1.0])

            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * len(pi_grid))

            if result.success:
                # Optimal action is the one with highest probability (pure strategy)
                best_idx = np.argmax(result.x)
                return pi_grid[best_idx], -result.fun
            else:
                # Fallback to grid search
                best_idx = np.argmax(utilities)
                return pi_grid[best_idx], utilities[best_idx]

        else:  # grid
            if pi_grid is None:
                pi_grid = np.linspace(0.1, 2.0, 20)

            utilities = [self.predict_utility(W, pi, sigma2, T) for pi in pi_grid]
            best_idx = np.argmax(utilities)

            return pi_grid[best_idx], utilities[best_idx]


# =============================================================================
# Main Experiment
# =============================================================================
def run_stage2_experiment():
    print("=" * 70)
    print("STAGE 2: LP Control in Lifted Eigenfunction Space")
    print("=" * 70)
    print("""
Architecture:
1. Stage 1 Filter: RecSig-RLS estimates σ̂²
2. Stage 2 KGEDMD: Learns eigenfunctions φ(W, π, σ̂²)
3. Control: Find optimal π* via eigenvalue maximization

Two targeting modes compared:
A. OBSERVABLE: Train on r²/dt (generalizable)
B. LATENT: Train on V (smoother, oracle for training)
    """)

    env = HestonMertonEnv()

    def certainty_equivalent(wealths, gamma):
        """Compute certainty equivalent."""
        utilities = [env.crra_utility(w) for w in wealths if w > 0]
        if len(utilities) == 0:
            return 0
        mean_U = np.mean(utilities)
        if gamma == 1:
            return np.exp(mean_U)
        return ((1 - gamma) * mean_U) ** (1 / (1 - gamma))

    # Initialize filters with both targeting modes
    filter_obs = VolatilityFilter(dt=env.dt, target_mode='observable')
    filter_lat = VolatilityFilter(dt=env.dt, target_mode='latent')

    # ==========================================================================
    # Collect training data (reduced for speed)
    # ==========================================================================
    print("\n[1] Collecting training data...")
    states, utilities, r2_dt_series, episode_ends = KGEDMDController().collect_data(
        env, filter_obs, filter_lat,
        n_episodes=30, n_steps=126  # Reduced for faster iteration
    )

    # ==========================================================================
    # Stage 1b: Estimate volatility dynamics from observables
    # ==========================================================================
    print("\n[1b] Estimating volatility dynamics from r²/dt...")
    vol_dynamics = VolatilityDynamicsEstimator()
    vol_dynamics.fit(r2_dt_series, dt=env.dt)

    print(f"""
    Estimated vs True Heston Parameters:
    ------------------------------------
    Parameter   Estimated    True
    κ (reversion) {vol_dynamics.kappa_hat:8.3f}    {env.kappa:.3f}
    θ (long-run)  {vol_dynamics.theta_hat:8.4f}   {env.theta:.4f}
    ξ (vol-of-vol){vol_dynamics.xi_hat:8.3f}    {env.xi:.3f}
    """)

    # ==========================================================================
    # Train KGEDMD with estimated vol dynamics
    # ==========================================================================
    print("\n[2] Training KGEDMD models...")

    kgedmd_obs = KGEDMDController(n_landmarks=50, vol_dynamics=vol_dynamics)
    kgedmd_obs.fit(states, utilities, use_latent=False, episode_ends=episode_ends)

    kgedmd_lat = KGEDMDController(n_landmarks=50, vol_dynamics=vol_dynamics)
    kgedmd_lat.fit(states, utilities, use_latent=True, episode_ends=episode_ends)

    # Diagnose what KGEDMD learned
    print("\n[2b] KGEDMD Diagnostics (latent targeting):")
    kgedmd_lat.diagnose(env)

    # ==========================================================================
    # Evaluate control policies over MULTIPLE SEEDS
    # ==========================================================================
    print("\n[3] Evaluating control policies (multiple seeds)...")

    n_seeds = 5  # Multiple training seeds
    n_eval_episodes = 20
    n_eval_steps = 504  # 2 years - longer horizon to see state-dependent benefits

    # Chacko-Viceira with estimated parameters (fully autonomous)
    def cv_estimated(sigma2):
        """C-V using filtered σ̂² and estimated (κ̂, θ̂, ξ̂)."""
        sigma2 = max(sigma2, 0.01)
        pi_myopic = (env.mu - env.r) / (env.gamma * sigma2)
        if env.gamma > 1:
            hedging_factor = (1 - 1/env.gamma)
            # Use ESTIMATED parameters
            pi_hedging = hedging_factor * (env.rho * vol_dynamics.xi_hat / np.sqrt(sigma2)) * (1 / (2 * vol_dynamics.kappa_hat))
        else:
            pi_hedging = 0
        return pi_myopic + pi_hedging

    policies_base = {
        'Merton (true V)': lambda W, V, sigma2, kgedmd: env.merton_optimal_pi(V),
        'Merton (filtered)': lambda W, V, sigma2, kgedmd: env.merton_optimal_pi(max(sigma2, 0.01)),
        'C-V (true)': lambda W, V, sigma2, kgedmd: env.chacko_viceira_pi(V),
        'C-V (estimated)': lambda W, V, sigma2, kgedmd: cv_estimated(sigma2),
        'KGEDMD-LP': lambda W, V, sigma2, kgedmd: kgedmd.find_optimal_pi(W, sigma2, method='lp')[0],
        'Constant': lambda W, V, sigma2, kgedmd: (env.mu - env.r) / (env.gamma * env.theta),
    }

    # Results across seeds
    all_seed_results = {name: [] for name in policies_base}
    allocation_traces = {}

    # Use single training, multiple eval seeds (faster)
    for seed in range(n_seeds):
        print(f"\n  === Eval Seed {seed+1}/{n_seeds} ===")

        # Evaluate each policy with different eval seed
        seed_results = {name: [] for name in policies_base}

        for ep in range(n_eval_episodes):
            for name, policy_fn in policies_base.items():
                np.random.seed(5000 + seed * 1000 + ep)

                W = 1.0
                V = env.theta
                filter_lat.reset(W)

                pis_chosen = []
                for t in range(n_eval_steps):
                    sigma2_lat = filter_lat.update(W, V)
                    pi = policy_fn(W, V, sigma2_lat, kgedmd_lat)
                    pi = np.clip(pi, 0.1, 3.0)
                    pis_chosen.append(pi)
                    W, V, _ = env.step(W, V, pi)

                seed_results[name].append(W)

                if seed == 0 and ep == 0 and name not in allocation_traces:
                    allocation_traces[name] = pis_chosen

        # Store CE for this seed
        for name in policies_base:
            wealths = np.array(seed_results[name])
            ce = certainty_equivalent(wealths[wealths > 0], env.gamma)
            all_seed_results[name].append(ce)
            print(f"    {name}: CE = {ce:.4f}")

    # Aggregate results across seeds
    results = {}
    for name in policies_base:
        ces = np.array(all_seed_results[name])
        results[name] = {
            'ce_mean': np.mean(ces),
            'ce_std': np.std(ces),
            'ces': ces
        }

    # ==========================================================================
    # Compute metrics (aggregated across seeds)
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"RESULTS (averaged over {n_seeds} seeds)")
    print("=" * 70)

    cv_true_ce = results['C-V (true)']['ce_mean']
    cv_est_ce = results['C-V (estimated)']['ce_mean']
    ce_const = results['Constant']['ce_mean']

    print(f"\n{'Policy':<20} {'CE Mean':>12} {'CE Std':>10} {'vs Const':>12}")
    print("-" * 60)

    policy_order = ['Merton (true V)', 'Merton (filtered)', 'C-V (true)', 'C-V (estimated)', 'KGEDMD-LP', 'Constant']
    for name in policy_order:
        r = results[name]
        diff_pct = (r['ce_mean'] / ce_const - 1) * 100  # % difference vs Constant
        print(f"{name:<20} {r['ce_mean']:>12.4f} {r['ce_std']:>10.4f} {diff_pct:>+10.2f}%")

    # ==========================================================================
    # Analysis: Oracle vs Autonomous (key comparison)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: Oracle vs Fully Autonomous Methods")
    print("=" * 70)

    ce_lp = results['KGEDMD-LP']['ce_mean']
    ce_lp_std = results['KGEDMD-LP']['ce_std']
    ce_merton_true = results['Merton (true V)']['ce_mean']
    ce_merton_filt = results['Merton (filtered)']['ce_mean']
    cv_true_std = results['C-V (true)']['ce_std']
    cv_est_std = results['C-V (estimated)']['ce_std']

    print(f"""
ORACLE methods (use true V):
  - Merton (true V): CE = {ce_merton_true:.4f} ± {results['Merton (true V)']['ce_std']:.4f}
  - C-V (true):      CE = {cv_true_ce:.4f} ± {cv_true_std:.4f}

AUTONOMOUS methods (use only observables):
  - Merton (filtered): CE = {ce_merton_filt:.4f} ± {results['Merton (filtered)']['ce_std']:.4f}
  - C-V (estimated):   CE = {cv_est_ce:.4f} ± {cv_est_std:.4f}  <-- Uses estimated (κ̂,θ̂,ξ̂)
  - KGEDMD-LP:         CE = {ce_lp:.4f} ± {ce_lp_std:.4f}
  - Constant:          CE = {ce_const:.4f} ± {results['Constant']['ce_std']:.4f}

Key question: Does C-V (estimated) match C-V (true)?
  - Gap: {(cv_est_ce - cv_true_ce):.4f} ({(cv_est_ce/cv_true_ce - 1)*100:+.2f}%)
  - If gap is small → parameter estimation works!
    """)

    # Analyze allocation differences
    print("\n" + "=" * 70)
    print("ALLOCATION ANALYSIS")
    print("=" * 70)

    for name in policy_order:
        if name in allocation_traces:
            pis = np.array(allocation_traces[name])
            print(f"{name:<20}: mean π = {np.mean(pis):.3f}, std = {np.std(pis):.3f}")

    print(f"""
Key insight:
- Merton (myopic): π = (μ-r)/(γV) - ignores vol dynamics
- Chacko-Viceira: adds hedging term (ρξ/√V)/(2κ) - SOTA for stochastic vol
- With ρ = {env.rho} < 0: hedging demand is NEGATIVE (reduce allocation)

The hedging demand compensates for:
1. Leverage effect: vol spikes → prices drop
2. Mean reversion: V returns to θ = {env.theta}
3. Vol-of-vol risk: ξ = {env.xi}
    """)

    # ==========================================================================
    # Save plot
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: CE across seeds (boxplot)
    ax = axes[0]
    plot_policies = ['Merton (true V)', 'Merton (filtered)', 'C-V (true)', 'C-V (estimated)', 'KGEDMD-LP', 'Constant']
    data = [results[name]['ces'] for name in plot_policies]
    bp = ax.boxplot(data, tick_labels=['Mert\n(V)', 'Mert\n(σ̂²)', 'C-V\n(V)', 'C-V\n(σ̂²)', 'KGEDMD', 'Const'])
    ax.set_ylabel('Certainty Equivalent')
    ax.set_title(f'CE Distribution ({n_seeds} seeds)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: CE comparison with error bars
    ax = axes[1]
    labels = ['Mert\n(V)', 'Mert\n(σ̂²)', 'C-V\n(V)', 'C-V\n(σ̂²)', 'KGEDMD', 'Const']
    ces = [results[n]['ce_mean'] for n in plot_policies]
    stds = [results[n]['ce_std'] for n in plot_policies]
    colors = ['blue', 'cyan', 'green', 'lightgreen', 'purple', 'gray']
    ax.bar(range(len(labels)), ces, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Certainty Equivalent')
    ax.set_title('Mean CE ± Std')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/stage2_lp_control.png', dpi=150)
    print("\nSaved: kronic_pomdp/experiments/stage2_lp_control.png")

    return results


if __name__ == "__main__":
    results = run_stage2_experiment()
