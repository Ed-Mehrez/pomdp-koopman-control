"""
CdC Kernel Estimators: Connection to Aït-Sahalia & Jacod

This file establishes the theoretical and empirical connection between:
1. A&J spot volatility estimators (kernel regression on squared increments)
2. CdC-based kernel methods (same regression, different priors/kernels)
3. KGEDMD eigenfunction learning (generator → CdC → σ²)

Key insight: A&J is a SPECIAL CASE of CdC kernel regression with:
- Box/uniform kernel (instead of Gaussian)
- No regularization
- No prior mean function

Our approach generalizes by:
- Allowing Gaussian/RBF kernels (better bias-variance)
- Incorporating parametric priors (CIR: σ²∝V, CEV: σ²∝V^β)
- Adding eigenfunction pricing (L → eigenfunctions → E[f(V_t)|V_0])

For multidimensional extension, see Section 4.

FUTURE EXTENSION: Log-Signature σ² Estimation
----------------------------------------------
For Markov diffusions, lead-lag log-signature Lévy area = QV = ∫σ²dt,
which is equivalent to A&J. BUT the signature approach generalizes to:
- Rough volatility (H < 0.5): σ depends on path history
- Multi-scale: different forgetting factors γ capture different horizons
- Filtering: when V is latent, sig features are model-free (Level 4 BLR+KF)
- Multi-asset: signature kernels handle cross-asset dependencies
Keep this method "in the back pocket" for harder problems.
See: examples/proof_of_concept/signature_features.py (RecurrentLeadLagLogSigMap)
"""

import numpy as np
from typing import Callable, Optional, Tuple
from scipy.spatial.distance import pdist, squareform


def simulate_cir(kappa: float, theta: float, xi: float,
                 T: int, dt: float, seed: int = 42) -> np.ndarray:
    """Simulate CIR: dV = κ(θ - V)dt + ξ√V dW"""
    np.random.seed(seed)
    V = np.zeros(T)
    V[0] = theta
    sqrt_dt = np.sqrt(dt)

    for t in range(1, T):
        v = max(V[t-1], 1e-8)
        V[t] = max(v + kappa * (theta - v) * dt + xi * np.sqrt(v) * sqrt_dt * np.random.randn(), 1e-8)

    return V


# =============================================================================
# SECTION 1: A&J Estimator (Nadaraya-Watson on squared increments)
# =============================================================================

class AitSahaliaJacodEstimator:
    """
    Aït-Sahalia & Jacod spot volatility estimator.

    This is Nadaraya-Watson kernel regression:
        σ̂²(V) = Σᵢ (ΔVᵢ)²/Δt · K_h(Vᵢ - V) / Σᵢ K_h(Vᵢ - V)

    With box kernel, this is local averaging of squared increments.
    """

    def __init__(self, bandwidth: float = None, kernel: str = 'box'):
        """
        Args:
            bandwidth: Kernel bandwidth (auto if None)
            kernel: 'box' (original A&J) or 'gaussian' (smoothed)
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.V_data = None
        self.squared_inc = None

    def fit(self, V: np.ndarray, dt: float):
        """Fit on time series data."""
        self.dt = dt
        self.V_data = V[:-1].copy()
        self.squared_inc = (V[1:] - V[:-1])**2 / dt

        if self.bandwidth is None:
            self.bandwidth = 1.06 * np.std(self.V_data) * len(self.V_data)**(-1/5)

        return self

    def _kernel_weights(self, V_query: float) -> np.ndarray:
        """Compute kernel weights for query point."""
        u = (self.V_data - V_query) / self.bandwidth

        if self.kernel == 'box':
            weights = (np.abs(u) <= 1).astype(float)
        elif self.kernel == 'gaussian':
            weights = np.exp(-0.5 * u**2)
        elif self.kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return weights

    def predict(self, V: np.ndarray) -> np.ndarray:
        """Predict σ²(V) at query points."""
        V = np.atleast_1d(V)
        result = np.zeros_like(V, dtype=float)

        for i, v in enumerate(V):
            w = self._kernel_weights(v)
            w_sum = np.sum(w)
            if w_sum > 1e-10:
                result[i] = np.sum(w * self.squared_inc) / w_sum
            else:
                result[i] = np.mean(self.squared_inc)  # Fallback

        return result


# =============================================================================
# SECTION 2: CdC Kernel Ridge Regression (Generalized A&J)
# =============================================================================

class CdCKernelEstimator:
    """
    CdC-based σ² estimation via kernel regression.

    Two modes available:
    1. 'nw' (Nadaraya-Watson): Σ K(x,x_i)·y_i / Σ K(x,x_i)
       - EQUIVALENT to A&J when using same kernel/bandwidth
       - No regularization needed
       - Normalized weights (sums to 1)

    2. 'krr' (Kernel Ridge Regression): K·(K+λI)^{-1}·y
       - Global smoother with regularization
       - Better for extrapolation
       - Different from A&J

    For equivalence to A&J:
    - Use mode='nw', same bandwidth, gaussian kernel
    """

    def __init__(self, bandwidth: float = None,
                 prior_fn: Callable = None,
                 regularization: float = 1e-4,
                 mode: str = 'nw'):
        """
        Args:
            bandwidth: RBF kernel bandwidth (auto if None)
            prior_fn: Prior mean σ²(V), e.g., lambda V: ξ²*V for CIR
            regularization: Ridge penalty λ (only used in 'krr' mode)
            mode: 'nw' (Nadaraya-Watson, matches A&J) or 'krr' (kernel ridge)
        """
        self.bandwidth = bandwidth
        self.prior_fn = prior_fn
        self.regularization = regularization
        self.mode = mode

        self.V_train = None
        self.targets = None  # For NW mode
        self.alpha = None    # For KRR mode

    def fit(self, V: np.ndarray, dt: float, max_points: int = None):
        """
        Fit kernel regression on squared increments.

        For EXACT A&J equivalence:
        - Use mode='nw', no prior
        - Don't set max_points (use all data like A&J)
        """
        self.dt = dt

        V_t = V[:-1].copy()
        squared_inc = (V[1:] - V[:-1])**2 / dt

        # Only subsample if explicitly requested (NOT by default for A&J equivalence)
        if max_points is not None and len(V_t) > max_points:
            idx = np.random.choice(len(V_t), max_points, replace=False)
            V_t = V_t[idx]
            squared_inc = squared_inc[idx]

        self.V_train = V_t

        # Auto bandwidth: use SAME rule as A&J for fair comparison
        if self.bandwidth is None:
            # Silverman's rule (same as A&J)
            self.bandwidth = 1.06 * np.std(V_t) * len(V_t)**(-1/5)

        # Target: residuals from prior (or raw if no prior)
        if self.prior_fn is not None:
            self.targets = squared_inc - self.prior_fn(V_t)
        else:
            self.targets = squared_inc

        if self.mode == 'krr':
            # KRR: α = (K + λI)⁻¹ y
            K = self._kernel_matrix(V_t, V_t)
            n = len(V_t)
            self.alpha = np.linalg.solve(K + self.regularization * np.eye(n), self.targets)

        # For NW mode, we just store targets and compute weights at prediction time

        return self

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel matrix."""
        X1 = np.atleast_1d(X1).reshape(-1, 1)
        X2 = np.atleast_1d(X2).reshape(-1, 1)
        sq_dists = (X1 - X2.T)**2
        return np.exp(-sq_dists / (2 * self.bandwidth**2))

    def predict(self, V: np.ndarray) -> np.ndarray:
        """Predict σ²(V)."""
        V = np.atleast_1d(V)

        if self.mode == 'nw':
            # Nadaraya-Watson: weighted average with normalized weights
            # This is EXACTLY what A&J does with Gaussian kernel
            K = self._kernel_matrix(V, self.V_train)
            # Normalize rows to sum to 1
            K_sum = K.sum(axis=1, keepdims=True)
            K_sum = np.maximum(K_sum, 1e-10)  # Avoid division by zero
            weights = K / K_sum
            correction = weights @ self.targets
        else:
            # KRR: use pre-computed alpha
            K = self._kernel_matrix(V, self.V_train)
            correction = K @ self.alpha

        if self.prior_fn is not None:
            return self.prior_fn(V) + correction
        else:
            return correction


# =============================================================================
# SECTION 3: KGEDMD → CdC → σ² (Eigenfunction bonus)
# =============================================================================

class KGEDMDCdCEstimator:
    """
    Use KGEDMD to learn generator L, then extract σ² via CdC identity.

    CdC identity: σ²(V) = L(V²) - 2V·L(V)

    CRITICAL FIX: L_matrix acts on Nyström COEFFICIENTS, not function values!

    In Nyström basis φᵢ(x) = k(x, zᵢ):
    - Function f(x) ≈ Σᵢ αᵢ φᵢ(x) where α = K_MM⁻¹ @ f(landmarks)
    - Generator acts: (Lf)(x) ≈ Σᵢ (L_matrix @ α)ᵢ φᵢ(x)

    So to apply L to f:
    1. Get coefficients: α = K_MM⁻¹ @ f_landmarks
    2. Apply L in coeff space: β = L_matrix @ α
    3. Evaluate at query: (Lf)(x) = k(x, landmarks) @ β

    BONUS: Once we have L's eigenfunctions, we can price derivatives!

    NEW: Direct σ² estimation via squared-increment regression (bypasses CdC).
    """

    def __init__(self, n_landmarks: int = 100, regularization: float = 1e-4,
                 sigma_method: str = 'direct'):
        """
        Args:
            n_landmarks: Number of Nyström landmarks
            regularization: Ridge penalty
            sigma_method: 'cdc' (via generator) or 'direct' (via squared increments)
        """
        self.n_landmarks = n_landmarks
        self.regularization = regularization
        self.sigma_method = sigma_method

        self.landmarks = None
        self.L_matrix = None
        self.K_MM_inv = None
        self.bandwidth = None
        self.sigma_sq_coeffs = None  # For direct method

    def fit(self, V: np.ndarray, dt: float):
        """Learn generator from consecutive transitions in time series V."""
        V_t = V[:-1].flatten()
        V_next = V[1:].flatten()
        return self.fit_pairs(V_t, V_next, dt, V_data=V.copy())

    def fit_pairs(self, V_t: np.ndarray, V_next: np.ndarray, dt: float,
                  V_data: np.ndarray = None):
        """Learn generator from transition pairs (V_t, V_next) at timestep dt.

        Unlike fit(), this accepts pre-selected transition pairs that need not
        come from a consecutive time series. Use this when subsampling transitions.
        """
        self.dt = dt
        self._V_data = V_data  # Store for CV methods (may be None)

        V_t = np.atleast_1d(V_t).flatten()
        V_next = np.atleast_1d(V_next).flatten()
        V_t_2d = V_t.reshape(-1, 1)
        V_next_2d = V_next.reshape(-1, 1)

        # Squared increments for direct σ² estimation
        squared_inc = (V_next - V_t)**2 / dt

        # Select landmarks via farthest point sampling for better coverage
        n = min(self.n_landmarks, len(V_t) // 5)
        idx = self._select_landmarks_fps(V_t, n)
        self.landmarks = V_t[idx]

        # Auto bandwidth: median heuristic
        dists = pdist(self.landmarks.reshape(-1, 1))
        self.bandwidth = np.median(dists) if len(dists) > 0 else np.std(self.landmarks)

        # Kernel matrices
        K_MM = self._kernel_matrix(self.landmarks, self.landmarks)
        K_t = self._kernel_matrix(V_t, self.landmarks)
        K_next = self._kernel_matrix(V_next, self.landmarks)

        # Store K_MM inverse for coefficient conversion
        self.K_MM_inv = np.linalg.inv(K_MM + self.regularization * np.eye(n))

        # Koopman in Nyström basis: K_next ≈ K_t @ Koopman
        # Koopman maps coefficients: α' = Koopman @ α
        K_gram = K_t.T @ K_t + self.regularization * np.eye(n)
        self.Koopman = np.linalg.solve(K_gram, K_t.T @ K_next)

        # Generator: L = (K - I) / dt
        # This acts on COEFFICIENTS in Nyström basis
        self.L_matrix = (self.Koopman - np.eye(n)) / dt

        # Direct σ² estimation: kernel regression on (ΔV)²/dt
        # This is equivalent to A&J but in Nyström basis
        # σ²(V) ≈ k(V, landmarks) @ sigma_sq_coeffs
        # where sigma_sq_coeffs = (K_t'K_t + λI)^{-1} K_t' @ squared_inc
        self.sigma_sq_coeffs = np.linalg.solve(K_gram, K_t.T @ squared_inc)

        # Store for predictive variance computation
        self._K_gram_inv = np.linalg.inv(K_gram)
        self._residual_var = float(np.mean(
            (squared_inc - K_t @ self.sigma_sq_coeffs)**2))

        return self

    def _select_landmarks_fps(self, X: np.ndarray, m: int) -> np.ndarray:
        """Farthest Point Sampling for better coverage."""
        n = len(X)
        if m >= n:
            return np.arange(n)

        indices = [np.random.randint(n)]
        min_dists = np.full(n, np.inf)

        for _ in range(m - 1):
            last_idx = indices[-1]
            dists_to_last = (X - X[last_idx])**2
            min_dists = np.minimum(min_dists, dists_to_last)
            next_idx = np.argmax(min_dists)
            indices.append(next_idx)

        return np.array(indices)

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel."""
        X1 = np.atleast_1d(X1).reshape(-1, 1)
        X2 = np.atleast_1d(X2).reshape(-1, 1)
        sq_dists = (X1 - X2.T)**2
        return np.exp(-sq_dists / (2 * self.bandwidth**2))

    def _function_to_coeffs(self, f_at_landmarks: np.ndarray) -> np.ndarray:
        """
        Convert function values at landmarks to Nyström coefficients.

        If f(x) ≈ Σᵢ αᵢ k(x, zᵢ), then at landmarks:
        f(z_j) = Σᵢ αᵢ k(z_j, zᵢ) = (K_MM @ α)_j

        So: α = K_MM⁻¹ @ f_landmarks
        """
        return self.K_MM_inv @ f_at_landmarks

    def _coeffs_to_function(self, coeffs: np.ndarray, X_query: np.ndarray) -> np.ndarray:
        """
        Evaluate function from Nyström coefficients.

        f(x) = Σᵢ αᵢ k(x, zᵢ) = k(x, landmarks) @ α
        """
        K = self._kernel_matrix(X_query, self.landmarks)
        return K @ coeffs

    def generator_on_function(self, f_at_landmarks: np.ndarray,
                               X_query: np.ndarray = None) -> np.ndarray:
        """
        Apply generator L to function f, evaluated at query points.

        Steps:
        1. Convert f values → Nyström coefficients
        2. Apply L in coefficient space
        3. Convert back to function values

        If X_query is None, evaluates at landmarks.
        """
        # Step 1: Function values → coefficients
        alpha = self._function_to_coeffs(f_at_landmarks)

        # Step 2: Apply L in coefficient space
        beta = self.L_matrix @ alpha

        # Step 3: Coefficients → function values at query points
        if X_query is None:
            X_query = self.landmarks

        return self._coeffs_to_function(beta, X_query)

    def sigma_squared_direct(self, V_query: np.ndarray) -> np.ndarray:
        """
        Extract σ²(V) via DIRECT kernel regression on squared increments.

        This bypasses the CdC identity entirely and regresses (ΔV)²/dt
        directly using the same Nyström basis as the Koopman operator.

        Equivalent to A&J/CdC-NW but in Nyström feature space.
        """
        V_query = np.atleast_1d(V_query)
        K = self._kernel_matrix(V_query, self.landmarks)
        sigma_sq = K @ self.sigma_sq_coeffs
        return np.maximum(sigma_sq, 1e-10)

    def sigma_squared_cdc(self, V_query: np.ndarray, ito_correction: bool = True,
                          method: str = None) -> np.ndarray:
        """
        Extract σ²(V).

        Args:
            V_query: Query points
            ito_correction: Apply Itô correction for discrete time (CdC method only)
            method: 'direct' (regression on squared increments) or
                    'cdc' (via generator). Default uses self.sigma_method.

        The 'direct' method is recommended as it matches A&J accuracy while
        still providing the Koopman operator for pricing.
        """
        if method is None:
            method = self.sigma_method

        if method == 'direct':
            return self.sigma_squared_direct(V_query)

        # CdC method via generator
        V_query = np.atleast_1d(V_query)

        # Function values at landmarks
        f_V = self.landmarks  # V itself (identity function)
        f_V2 = self.landmarks**2  # V²

        # Apply generator L (correctly handling Nyström coefficients)
        L_V = self.generator_on_function(f_V, V_query)  # = μ(V)
        L_V2 = self.generator_on_function(f_V2, V_query)  # = 2V·μ + σ² + μ²·dt

        # Raw CdC identity: σ²_raw = L(V²) - 2V·L(V) = σ² + μ²·dt
        sigma_sq_raw = L_V2 - 2 * V_query * L_V

        if ito_correction:
            # Correct for discrete-time bias: subtract μ²·dt
            mu_squared_dt = L_V**2 * self.dt
            sigma_sq = sigma_sq_raw - mu_squared_dt
        else:
            sigma_sq = sigma_sq_raw

        return np.maximum(sigma_sq, 1e-10)  # Ensure positive

    def drift_cdc(self, V_query: np.ndarray) -> np.ndarray:
        """
        Extract drift μ(V) directly: μ = L(V)

        This is a byproduct of the generator learning.
        """
        V_query = np.atleast_1d(V_query)
        f_V = self.landmarks

        return self.generator_on_function(f_V, V_query)

    def sigma_squared_grid(self, grid: np.ndarray) -> np.ndarray:
        """σ²(x) on a grid, using the configured method (direct or cdc)."""
        return self.sigma_squared_cdc(grid, method=self.sigma_method)

    def predictive_variance(self, V_query: np.ndarray) -> np.ndarray:
        """
        Predictive variance of σ̂²(x) from the kernel ridge regression.

        For KRR with model σ̂²(x) = k(x, Z)ᵀ α where α = (K'K + λI)⁻¹ K'y,
        the predictive variance at a query point x is:

            Var[σ̂²(x)] = σ²_noise · k(x,Z)ᵀ (K'K + λI)⁻¹ k(x,Z)

        where σ²_noise is the residual variance of the kernel regression.

        This gives the uncertainty of the MEAN prediction, not the total
        observation variance. The total observation uncertainty is:
            τ²(x) = Var[σ̂²(x)] + σ²_noise
        but for the α regression we want Var[σ̂²(x)] alone — it tells us
        how much the kernel smoother's output can vary at each point.

        Returns array of predictive variances at each query point.
        """
        V_query = np.atleast_1d(V_query)
        K_q = self._kernel_matrix(V_query, self.landmarks)

        # Var[f̂(x)] = σ²_noise · k_x' (K'K + λI)^{-1} k_x
        # _K_gram_inv = (K'K + λI)^{-1} is already stored from fit()
        # For each query point: v_i = k_i' @ _K_gram_inv @ k_i
        # Vectorized: V = diag(K_q @ _K_gram_inv @ K_q')
        pred_var = self._residual_var * np.sum(
            (K_q @ self._K_gram_inv) * K_q, axis=1)

        return np.maximum(pred_var, 1e-20)

    def fit_alpha_bayesian(self, n_posterior_samples=200, method='weighted'):
        """Fit log(σ²) ~ α·log(S) with proper uncertainty propagation.

        Three methods available:
        - 'naive': Evaluate σ² at landmarks, fit BayesianRidge ignoring
          estimation uncertainty. Fast but overconfident at α≈2.
        - 'weighted': Evaluate σ² on a grid, use analytic predictive
          variance from the kernel regression as observation weights in
          BayesianRidge. Properly calibrated uncertainty.
        - 'cv': Temporal block cross-validation. Fit Koopman on K-1 blocks,
          evaluate σ² on held-out block. Most robust but slower.

        Returns:
            dict with alpha_mean, alpha_sd, p_bubble, method
        """
        if method == 'naive':
            return self._fit_alpha_naive(n_posterior_samples)
        elif method == 'weighted':
            return self._fit_alpha_weighted(n_posterior_samples)
        elif method == 'cv':
            return self._fit_alpha_cv(n_posterior_samples)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'naive', 'weighted', or 'cv'.")

    def _fit_alpha_naive(self, n_posterior_samples=200):
        """Original: evaluate σ² at landmarks, plain BayesianRidge."""
        from sklearn.linear_model import BayesianRidge
        from scipy import stats

        sigma2_at_landmarks = self.sigma_squared_grid(self.landmarks)

        valid = (self.landmarks > 1e-4) & (sigma2_at_landmarks > 1e-8)
        if np.sum(valid) < 10:
            return {'alpha_mean': np.nan, 'alpha_sd': np.nan,
                    'p_bubble': 0.0, 'method': 'naive'}

        log_S = np.log(self.landmarks[valid]).reshape(-1, 1)
        log_sig2 = np.log(sigma2_at_landmarks[valid])

        brr = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                            lambda_1=1e-6, lambda_2=1e-6,
                            fit_intercept=True, compute_score=True)
        brr.fit(log_S, log_sig2)

        alpha_mean = float(brr.coef_[0])
        alpha_sd = float(np.sqrt(brr.sigma_[0, 0])) if hasattr(brr, 'sigma_') else 0.5

        if alpha_sd > 0:
            p_bubble = float(stats.norm.cdf((alpha_mean - 2.0) / alpha_sd))
        else:
            p_bubble = 1.0 if alpha_mean > 2.0 else 0.0

        return {
            'alpha_mean': alpha_mean, 'alpha_sd': alpha_sd,
            'p_bubble': p_bubble, 'method': 'naive',
            'C_fit': float(np.exp(brr.intercept_)),
        }

    def _fit_alpha_weighted(self, n_posterior_samples=200, n_blocks=10):
        """
        Jackknife-calibrated α estimation via temporal block resampling.

        The naive BayesianRidge on smoothed σ²(S) gives correct point estimates
        but underestimates uncertainty because:
        1. The KRR prediction variance is small (many data points per kernel width)
        2. Variance-based weighting doesn't capture boundary bias

        Instead, we use the delete-block jackknife (Kunsch 1989):
        1. Get α from the full-data fit (best point estimate)
        2. Split data into n_blocks temporal blocks
        3. For each block k, refit KGEDMDCdCEstimator on data minus block k
        4. Compute α_k from each leave-one-out fit
        5. Jackknife SD: sqrt((K-1)/K · Σ(α_k - ᾱ)²)

        The jackknife captures BOTH variance (from finite-sample noise in σ²
        estimation) AND bias (from boundary effects, path-dependent kernel
        coverage). This naturally accounts for the heteroskedastic noise
        in (ΔS)²/dt without needing explicit variance formulas.
        """
        from sklearn.linear_model import BayesianRidge
        from scipy import stats

        # Full-data α: evaluate on geomspace grid (more robust than at landmarks,
        # which can cluster near data density maxima and miss tail behavior)
        S_lo = np.percentile(self.landmarks, 5)
        S_hi = np.percentile(self.landmarks, 95)
        S_eval = np.geomspace(max(S_lo, 1e-4), S_hi, 200)
        sigma2_eval = self.sigma_squared_grid(S_eval)

        valid = (S_eval > 1e-4) & (sigma2_eval > 1e-8)
        if np.sum(valid) < 10:
            return {'alpha_mean': np.nan, 'alpha_sd': np.nan,
                    'p_bubble': 0.0, 'method': 'weighted'}

        log_S_full = np.log(S_eval[valid]).reshape(-1, 1)
        log_sig2_full = np.log(sigma2_eval[valid])
        brr_full = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                  lambda_1=1e-6, lambda_2=1e-6,
                                  fit_intercept=True)
        brr_full.fit(log_S_full, log_sig2_full)
        alpha_full = float(brr_full.coef_[0])
        C_fit_full = float(np.exp(brr_full.intercept_))

        if np.isnan(alpha_full):
            return {'alpha_mean': np.nan, 'alpha_sd': np.nan,
                    'p_bubble': 0.0, 'method': 'weighted'}

        # Delete-block jackknife for SD calibration
        V = self._V_data
        N = len(V) - 1
        block_size = N // n_blocks

        if block_size < 50:
            # Not enough data for jackknife — use BayesianRidge SD from grid fit
            alpha_sd_brr = float(np.sqrt(brr_full.sigma_[0, 0])) if hasattr(brr_full, 'sigma_') else 0.5
            p_bubble = float(stats.norm.cdf((alpha_full - 2.0) / max(alpha_sd_brr, 0.02)))
            return {'alpha_mean': alpha_full, 'alpha_sd': alpha_sd_brr,
                    'p_bubble': p_bubble, 'method': 'weighted',
                    'C_fit': C_fit_full}

        alpha_jk = []
        for k in range(n_blocks):
            block_start = k * block_size
            block_end = min((k + 1) * block_size, N)

            # Training indices: everything except block k
            train_idx = np.concatenate([
                np.arange(0, block_start),
                np.arange(block_end, N)
            ])

            if len(train_idx) < 100:
                continue

            V_train_t = V[train_idx]
            V_train_next = V[train_idx + 1]
            sq_inc = ((V_train_next - V_train_t) ** 2 / self.dt)

            # Fit a fresh estimator on training data
            est_k = KGEDMDCdCEstimator(
                n_landmarks=self.n_landmarks,
                regularization=self.regularization,
                sigma_method=self.sigma_method,
            )

            n_lm = min(est_k.n_landmarks, len(train_idx) // 5)
            if n_lm < 10:
                continue

            idx_lm = est_k._select_landmarks_fps(V_train_t, n_lm)
            est_k.landmarks = V_train_t[idx_lm]

            dists = pdist(est_k.landmarks.reshape(-1, 1))
            est_k.bandwidth = np.median(dists) if len(dists) > 0 else np.std(est_k.landmarks)

            K_MM = est_k._kernel_matrix(est_k.landmarks, est_k.landmarks)
            K_t = est_k._kernel_matrix(V_train_t, est_k.landmarks)

            K_gram = K_t.T @ K_t + est_k.regularization * np.eye(n_lm)
            est_k.sigma_sq_coeffs = np.linalg.solve(K_gram, K_t.T @ sq_inc)

            # Compute α from this fold's fit
            sig2_at_lm = est_k.sigma_squared_direct(est_k.landmarks)
            valid = (est_k.landmarks > 1e-4) & (sig2_at_lm > 1e-8)

            if np.sum(valid) < 10:
                continue

            log_S = np.log(est_k.landmarks[valid]).reshape(-1, 1)
            log_sig2 = np.log(sig2_at_lm[valid])

            brr_k = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                   lambda_1=1e-6, lambda_2=1e-6,
                                   fit_intercept=True)
            brr_k.fit(log_S, log_sig2)
            alpha_jk.append(float(brr_k.coef_[0]))

        if len(alpha_jk) < 3:
            alpha_sd_brr = float(np.sqrt(brr_full.sigma_[0, 0])) if hasattr(brr_full, 'sigma_') else 0.5
            p_bubble = float(stats.norm.cdf((alpha_full - 2.0) / max(alpha_sd_brr, 0.02)))
            return {'alpha_mean': alpha_full, 'alpha_sd': alpha_sd_brr,
                    'p_bubble': p_bubble, 'method': 'weighted',
                    'C_fit': C_fit_full}

        alpha_jk = np.array(alpha_jk)
        K_eff = len(alpha_jk)
        alpha_jk_mean = np.mean(alpha_jk)

        # Jackknife variance: Var(θ̂) = (K-1)/K · Σ(θ̂_k - θ̄)²
        alpha_var_jk = (K_eff - 1) / K_eff * np.sum((alpha_jk - alpha_jk_mean) ** 2)
        alpha_sd_jk = max(np.sqrt(alpha_var_jk), 0.02)

        # Use full-data α as point estimate, jackknife SD for uncertainty
        alpha_mean = alpha_full

        if alpha_sd_jk > 0:
            p_bubble = float(stats.norm.cdf((alpha_mean - 2.0) / alpha_sd_jk))
        else:
            p_bubble = 1.0 if alpha_mean > 2.0 else 0.0

        return {
            'alpha_mean': alpha_mean, 'alpha_sd': alpha_sd_jk,
            'p_bubble': p_bubble, 'method': 'weighted',
            'C_fit': C_fit_full,
            'diagnostics': {
                'n_blocks': K_eff,
                'alpha_jk_range': (float(np.min(alpha_jk)), float(np.max(alpha_jk))),
                'alpha_jk_mean': float(alpha_jk_mean),
            }
        }

    def _fit_alpha_cv(self, n_posterior_samples=200, n_folds=5):
        """
        Temporal block cross-validation for held-out σ² estimates.

        Splits the time series into n_folds contiguous blocks.
        For each fold: fit a fresh KGEDMDCdCEstimator on the other folds,
        evaluate σ² on the held-out block's data points.
        Collect all held-out (S, σ̂²) pairs and fit BayesianRidge.

        The held-out σ² estimates have proper noise level — the BayesianRidge
        posterior automatically reflects the true estimation uncertainty
        without any ad-hoc SD inflation.
        """
        from sklearn.linear_model import BayesianRidge
        from scipy import stats

        V = self._V_data
        N = len(V) - 1
        fold_size = N // n_folds

        if fold_size < 50:
            # Not enough data for CV, fall back to weighted
            return self._fit_alpha_weighted(n_posterior_samples)

        all_S = []
        all_sig2 = []

        for fold in range(n_folds):
            # Held-out block: [fold*fold_size, (fold+1)*fold_size)
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, N)

            # Training data: everything except the held-out block
            train_idx = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, N)
            ])

            if len(train_idx) < 100:
                continue

            # Build training series: need consecutive pairs for transitions
            # Use the training indices to select transition pairs
            V_train_t = V[train_idx]
            V_train_next = V[train_idx + 1]

            # Fit a fresh estimator on training data
            est_fold = KGEDMDCdCEstimator(
                n_landmarks=self.n_landmarks,
                regularization=self.regularization,
                sigma_method=self.sigma_method,
            )
            # Manual fit using training pairs
            V_t_2d = V_train_t.reshape(-1, 1)
            V_next_2d = V_train_next.reshape(-1, 1)
            sq_inc_train = ((V_next_2d - V_t_2d)**2 / self.dt).flatten()

            n_lm = min(est_fold.n_landmarks, len(V_t_2d) // 5)
            if n_lm < 5:
                continue
            idx_lm = est_fold._select_landmarks_fps(V_t_2d.flatten(), n_lm)
            est_fold.landmarks = V_t_2d[idx_lm].flatten()

            dists = pdist(est_fold.landmarks.reshape(-1, 1))
            est_fold.bandwidth = np.median(dists) if len(dists) > 0 else np.std(est_fold.landmarks)

            K_MM = est_fold._kernel_matrix(est_fold.landmarks, est_fold.landmarks)
            K_t = est_fold._kernel_matrix(V_t_2d.flatten(), est_fold.landmarks)
            est_fold.K_MM_inv = np.linalg.inv(
                K_MM + est_fold.regularization * np.eye(n_lm))
            K_gram = K_t.T @ K_t + est_fold.regularization * np.eye(n_lm)
            est_fold.sigma_sq_coeffs = np.linalg.solve(K_gram, K_t.T @ sq_inc_train)

            # Evaluate σ² on held-out data points
            test_S = V[test_start:test_end]
            test_sig2 = est_fold.sigma_squared_direct(test_S)

            all_S.append(test_S)
            all_sig2.append(test_sig2)

        if len(all_S) == 0:
            return self._fit_alpha_weighted(n_posterior_samples)

        S_held = np.concatenate(all_S)
        sig2_held = np.concatenate(all_sig2)

        # Filter valid
        valid = (S_held > 1e-4) & (sig2_held > 1e-8)
        if np.sum(valid) < 20:
            return self._fit_alpha_weighted(n_posterior_samples)

        log_S = np.log(S_held[valid]).reshape(-1, 1)
        log_sig2 = np.log(sig2_held[valid])

        brr = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                            lambda_1=1e-6, lambda_2=1e-6,
                            fit_intercept=True, compute_score=True)
        brr.fit(log_S, log_sig2)

        alpha_mean = float(brr.coef_[0])
        alpha_sd = float(np.sqrt(brr.sigma_[0, 0])) if hasattr(brr, 'sigma_') else 0.5

        if alpha_sd > 0:
            p_bubble = float(stats.norm.cdf((alpha_mean - 2.0) / alpha_sd))
        else:
            p_bubble = 1.0 if alpha_mean > 2.0 else 0.0

        return {
            'alpha_mean': alpha_mean, 'alpha_sd': alpha_sd,
            'p_bubble': p_bubble, 'method': 'cv',
            'C_fit': float(np.exp(brr.intercept_)),
            'diagnostics': {
                'n_folds': n_folds,
                'n_held_out': int(np.sum(valid)),
            }
        }

    def price_payoff_koopman(self, payoff_fn: Callable, V_0: float,
                              T: float) -> float:
        """
        Price E[f(V_T)|V_0] using Koopman semigroup directly (NO eigenfunctions).

        Method: E[f(V_T)|V_0] = K^n f  where n = T/dt

        In Nyström basis:
        1. α = K_MM⁻¹ @ f(landmarks)  (payoff → coefficients)
        2. β = Koopman^n @ α          (apply n-step Koopman)
        3. price = k(V_0, landmarks) @ β  (evaluate at V_0)

        This avoids eigendecomposition! For moderate horizons, matrix power is fast.
        """
        n_steps = int(T / self.dt)

        # Step 1: Payoff to coefficients
        f_landmarks = payoff_fn(self.landmarks)
        alpha = self._function_to_coeffs(f_landmarks)

        # Step 2: Apply Koopman n times
        # For numerical stability with large n, use eigendecomposition
        if n_steps <= 100:
            # Direct matrix power
            beta = np.linalg.matrix_power(self.Koopman, n_steps) @ alpha
        else:
            # Use eigendecomposition for large n
            eigvals, eigvecs = np.linalg.eig(self.Koopman)
            eigvecs_inv = np.linalg.inv(eigvecs)
            # Koopman^n = V @ diag(λ^n) @ V^{-1}
            lambda_n = eigvals ** n_steps
            beta = eigvecs @ (lambda_n * (eigvecs_inv @ alpha))
            beta = np.real(beta)

        # Step 3: Evaluate at V_0
        K_star = self._kernel_matrix(np.array([V_0]), self.landmarks)
        price = (K_star @ beta)[0]

        return price

    def price_payoff_krr(self, V_data: np.ndarray, payoff_fn: Callable,
                         V_0: float, T: float, bandwidth: float = None) -> float:
        """
        Price E[f(V_T)|V_0] using direct KRR on (V_t, f(V_{t+T})) pairs.

        This is the simplest approach: kernel regression of payoff on initial state.

        Pros:
        - No eigenfunction extraction needed
        - Works for any horizon T
        - Simpler than Koopman powers

        Cons:
        - Needs raw data (not just trained model)
        - Horizon-specific (need different regression per T)
        """
        n_steps = int(T / self.dt)

        if n_steps >= len(V_data) - 1:
            raise ValueError(f"T={T} too long for data length {len(V_data)}")

        # Create regression pairs
        V_t = V_data[:-n_steps]  # Initial states
        V_T = V_data[n_steps:]   # States at horizon T
        y = payoff_fn(V_T)       # Payoff values

        # Bandwidth
        if bandwidth is None:
            bandwidth = self.bandwidth

        # Kernel regression (Nadaraya-Watson)
        sq_dists = (V_t - V_0)**2
        weights = np.exp(-sq_dists / (2 * bandwidth**2))
        weights_sum = np.sum(weights)

        if weights_sum > 1e-10:
            price = np.sum(weights * y) / weights_sum
        else:
            price = np.mean(y)  # Fallback

        return price


# =============================================================================
# SECTION 4: Nyström + KNN Landmarking for Scalability
# =============================================================================

class NystromCdCEstimator:
    """
    CdC estimator with Nyström approximation for scalability.

    Key ideas:
    1. Select m << n landmarks via KNN-based sampling
    2. Approximate full kernel: K ≈ K_{nM} K_{MM}⁻¹ K_{Mn}
    3. Complexity: O(nm²) instead of O(n³)

    For multidimensional case, this handles curse of dimensionality:
    - Work in d-dimensional state space X ∈ ℝᵈ
    - Use product kernel: k(X, X') = exp(-||X - X'||² / 2σ²)
    - Landmarks chosen to cover state space efficiently
    """

    def __init__(self, n_landmarks: int = 100,
                 landmark_method: str = 'knn_fps',
                 prior_fn: Callable = None,
                 regularization: float = 1e-4):
        """
        Args:
            n_landmarks: Number of Nyström landmarks m
            landmark_method: 'random', 'knn_fps' (farthest point), 'kmeans'
            prior_fn: Prior mean for σ²
            regularization: Ridge penalty
        """
        self.n_landmarks = n_landmarks
        self.landmark_method = landmark_method
        self.prior_fn = prior_fn
        self.regularization = regularization

    def _select_landmarks_knn_fps(self, X: np.ndarray, m: int) -> np.ndarray:
        """
        Farthest Point Sampling for landmark selection.

        Greedily selects points that maximize minimum distance to existing landmarks.
        This gives better coverage than random sampling.
        """
        n = len(X)
        if m >= n:
            return np.arange(n)

        # Start with random point
        indices = [np.random.randint(n)]
        min_dists = np.full(n, np.inf)

        for _ in range(m - 1):
            # Update min distances to current landmarks
            last_idx = indices[-1]
            dists_to_last = np.sum((X - X[last_idx])**2, axis=1)
            min_dists = np.minimum(min_dists, dists_to_last)

            # Select farthest point
            next_idx = np.argmax(min_dists)
            indices.append(next_idx)

        return np.array(indices)

    def _select_landmarks_kmeans(self, X: np.ndarray, m: int) -> np.ndarray:
        """K-means clustering for landmark selection."""
        from scipy.cluster.vq import kmeans2
        centroids, _ = kmeans2(X, m, minit='++')

        # Find nearest data point to each centroid
        indices = []
        for c in centroids:
            dists = np.sum((X - c)**2, axis=1)
            indices.append(np.argmin(dists))

        return np.array(indices)

    def fit(self, X: np.ndarray, Y: np.ndarray, dt: float = 1.0):
        """
        Fit Nyström CdC estimator.

        Args:
            X: (n, d) state observations (multivariate)
            Y: (n,) squared increments / dt (target for σ²)
            dt: time step
        """
        self.dt = dt
        n, d = X.shape if X.ndim > 1 else (len(X), 1)
        X = X.reshape(n, -1)  # Ensure 2D

        m = min(self.n_landmarks, n // 5)

        # Select landmarks
        if self.landmark_method == 'knn_fps':
            landmark_idx = self._select_landmarks_knn_fps(X, m)
        elif self.landmark_method == 'kmeans':
            landmark_idx = self._select_landmarks_kmeans(X, m)
        else:  # random
            landmark_idx = np.random.choice(n, m, replace=False)

        self.landmarks = X[landmark_idx]
        self.d = d

        # Auto bandwidth: median heuristic on landmarks
        if not hasattr(self, 'bandwidth') or self.bandwidth is None:
            dists = pdist(self.landmarks)
            self.bandwidth = np.median(dists) if len(dists) > 0 else np.std(X)

        # Kernel matrices for Nyström
        K_MM = self._kernel_matrix(self.landmarks, self.landmarks)
        K_nM = self._kernel_matrix(X, self.landmarks)

        # Regularized inverse of K_MM
        K_MM_inv = np.linalg.inv(K_MM + self.regularization * np.eye(m))

        # Target: residuals from prior
        if self.prior_fn is not None:
            targets = Y - self.prior_fn(X)
        else:
            targets = Y

        # Nyström KRR: α = K_MM⁻¹ K_Mn (K_nM K_MM⁻¹ K_Mn + λI)⁻¹ y
        # Simplified: work in landmark space
        # α_M = (K_MM + λI)⁻¹ K_Mn y / n (approx)
        self.alpha = K_MM_inv @ (K_nM.T @ targets) / n

        return self

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel for d-dimensional data."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        # Squared Euclidean distances
        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1) - 2 * X1 @ X2.T

        return np.exp(-sq_dists / (2 * self.bandwidth**2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict σ²(X) at query points."""
        X = np.atleast_2d(X)
        K = self._kernel_matrix(X, self.landmarks)
        correction = K @ self.alpha

        if self.prior_fn is not None:
            return self.prior_fn(X) + correction
        else:
            return correction


# =============================================================================
# SECTION 5: Multivariate CdC for Covariance Matrices
# =============================================================================

class MultivariateCdCEstimator:
    """
    Multivariate CdC for covariance matrix estimation.

    For d assets with state X ∈ ℝᵈ, we estimate the d×d covariance matrix Σ(X).

    Method: Kernel regression on outer product of increments.

    Target: (ΔXᵢ)(ΔXⱼ) / dt → Σᵢⱼ(X)

    This GENERALIZES A&J multivariate by:
    1. Working in kernel feature space (curse of dimensionality)
    2. Structured priors: factor models, sparse Σ
    3. Nyström for scalability: O(nm²) instead of O(n³)

    The CdC identity for generator L:
        Σᵢⱼ(X) = Γ(Xᵢ, Xⱼ)(X) = L(XᵢXⱼ) - Xᵢ·L(Xⱼ) - Xⱼ·L(Xᵢ)
    """

    def __init__(self, n_landmarks: int = 100,
                 factor_dim: int = None,
                 regularization: float = 1e-4):
        """
        Args:
            n_landmarks: Nyström landmarks
            factor_dim: If set, use low-rank prior Σ ≈ FF' (factor model)
            regularization: Ridge penalty
        """
        self.n_landmarks = n_landmarks
        self.factor_dim = factor_dim
        self.regularization = regularization

    def fit(self, X: np.ndarray, dt: float):
        """
        Fit multivariate covariance estimator.

        Args:
            X: (T, d) multivariate time series
            dt: time step
        """
        self.dt = dt
        T, d = X.shape
        self.d = d

        # Increments
        dX = X[1:] - X[:-1]  # (T-1, d)
        X_t = X[:-1]  # States at which to evaluate

        # Target: outer products / dt
        # For each pair (i,j), target is dXᵢ·dXⱼ/dt
        # Flatten to (T-1, d²) for regression
        outer_prods = np.einsum('ti,tj->tij', dX, dX) / dt  # (T-1, d, d)
        self.cov_targets = outer_prods.reshape(T-1, d*d)

        # Select landmarks
        n = T - 1
        m = min(self.n_landmarks, n // 5)
        landmark_idx = self._select_landmarks_fps(X_t, m)
        self.landmarks = X_t[landmark_idx]

        # Auto bandwidth
        dists = pdist(self.landmarks)
        self.bandwidth = np.median(dists) if len(dists) > 0 else np.std(X_t)

        # Kernel matrices
        K_MM = self._kernel_matrix(self.landmarks, self.landmarks)
        K_nM = self._kernel_matrix(X_t, self.landmarks)

        # Solve for each (i,j) entry of covariance via proper KRR
        # Must use K_gram = K_nM^T K_nM (not K_MM) for correct regression
        K_gram = K_nM.T @ K_nM + self.regularization * np.eye(m)
        self.alpha = np.linalg.solve(K_gram, K_nM.T @ self.cov_targets)  # (m, d²)

        # Store for uncertainty estimation (hat matrix, GP posterior)
        self._K_nM = K_nM
        self._K_gram = K_gram
        self._K_MM = K_MM

        # Effective degrees of freedom: df_eff = trace(H) where H = K_nM @ K_gram^{-1} @ K_nM^T
        # Computing full H is O(n²m), but trace(H) = trace(K_gram^{-1} @ K_nM^T @ K_nM)
        #                                          = trace(K_gram^{-1} @ (K_gram - λI))
        #                                          = m - λ * trace(K_gram^{-1})
        K_gram_inv = np.linalg.inv(K_gram)
        self._df_eff = m - self.regularization * np.trace(K_gram_inv)

        # Store raw data for directional test
        self._X_t = X_t
        self._dX = dX

        return self

    def _select_landmarks_fps(self, X: np.ndarray, m: int) -> np.ndarray:
        """Farthest Point Sampling."""
        n = len(X)
        if m >= n:
            return np.arange(n)

        indices = [np.random.randint(n)]
        min_dists = np.full(n, np.inf)

        for _ in range(m - 1):
            last_idx = indices[-1]
            dists_to_last = np.sum((X - X[last_idx])**2, axis=1)
            min_dists = np.minimum(min_dists, dists_to_last)
            next_idx = np.argmax(min_dists)
            indices.append(next_idx)

        return np.array(indices)

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel."""
        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-sq_dists / (2 * self.bandwidth**2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict covariance matrix Σ(X) at query points.

        Returns: (n_query, d, d) array of covariance matrices
        """
        X = np.atleast_2d(X)
        n_query = len(X)

        K = self._kernel_matrix(X, self.landmarks)
        cov_flat = K @ self.alpha  # (n_query, d²)

        # Reshape and symmetrize
        cov = cov_flat.reshape(n_query, self.d, self.d)

        # Enforce symmetry: (Σ + Σᵀ) / 2
        cov = (cov + np.transpose(cov, (0, 2, 1))) / 2

        return cov

    def _inflate_alpha_sd(self, sd_raw, n_eval_valid):
        """Inflate BayesianRidge SD to account for kernel-induced correlation.

        The kernel predictions at n_eval points are correlated, so the effective
        sample size is df_eff (from the hat matrix), not n_eval.
        Inflate SD by sqrt(n_eval / df_eff).
        """
        if not hasattr(self, '_df_eff') or self._df_eff <= 0:
            return sd_raw
        inflation = np.sqrt(max(1.0, n_eval_valid / self._df_eff))
        return sd_raw * inflation

    def _get_eval_points(self, n_eval=30):
        """Get well-spaced evaluation points from training data (NOT landmarks).

        Using held-out points avoids interpolation bias at Nyström centers.
        Subsample to ~n_eval points spread across the range of ||X||.
        """
        X_t = self._X_t
        n = len(X_t)

        # Stratified subsample: sort by norm, take evenly spaced points
        norms = np.linalg.norm(X_t, axis=1)
        sorted_idx = np.argsort(norms)
        step = max(1, n // n_eval)
        eval_idx = sorted_idx[::step][:n_eval]

        return X_t[eval_idx]

    def _estimate_alpha_1d(self, z, dz, dt, n_landmarks=80):
        """Estimate tail exponent α via GP with blocked time-series CV.

        Model (Rasmussen & Williams §2.7, eq. 2.42):
            log σ²(z) = α·log|z| + c + f(z),   f ~ GP(0, k_SE)

        GP hyperparameter σ_f is selected via blocked time-series CV
        (R&W §5.4 + Arlot & Celisse 2010), NOT marginal likelihood.

        This is critical for non-ergodic processes (GBM): marginal likelihood
        assumes exchangeable observations and selects σ_f ≈ 0 (power law fits),
        while blocked CV detects that different temporal blocks have different
        α estimates → selects σ_f > 0 → wider posterior on α.

        Returns:
            (alpha_mean, alpha_sd) or (nan, nan) if insufficient data
        """
        n = len(z)
        sq_inc = dz ** 2 / dt

        # Split into temporal blocks for CV and noise estimation
        n_blocks = min(10, max(5, n // 500))
        block_len = n // n_blocks

        # Per-block NW estimates: the foundation for both noise variance
        # and blocked CV. Each block gives independent NW estimates.
        m = min(n_landmarks, n // 5)
        quantiles = np.linspace(0.01, 0.99, m)
        landmarks = np.quantile(z, quantiles)

        ldists = np.abs(np.diff(landmarks))
        bw = np.median(ldists) if len(ldists) > 0 else np.std(z)
        bw = max(bw, 1e-8)

        # Full NW estimates (using all data)
        diff = (landmarks[:, None] - z[None, :]) / bw
        K_nw = np.exp(-0.5 * diff ** 2)
        K_sum = K_nw.sum(axis=1)
        K_sq_sum = (K_nw ** 2).sum(axis=1)
        valid_lm = K_sum > 1e-10
        sigma2_nw = np.zeros(m)
        n_eff = np.zeros(m)
        sigma2_nw[valid_lm] = (K_nw[valid_lm] @ sq_inc) / K_sum[valid_lm]
        n_eff[valid_lm] = K_sum[valid_lm] ** 2 / K_sq_sum[valid_lm]

        valid = valid_lm & (np.abs(landmarks) > 1e-4) & (sigma2_nw > 1e-8) & (n_eff > 2)
        if np.sum(valid) < 10:
            return np.nan, np.nan

        x = np.log(np.abs(landmarks[valid]))
        y = np.log(sigma2_nw[valid])
        nv = len(x)

        # Block-based noise variance (Priestley's effective sample size)
        valid_idx = np.where(valid)[0]
        noise_var = np.full(nv, 0.0)
        for jj, j in enumerate(valid_idx):
            block_ests = []
            for b in range(n_blocks):
                sl = slice(b * block_len, min((b + 1) * block_len, n))
                K_b = K_nw[j, sl]
                K_b_sum = K_b.sum()
                if K_b_sum > 1e-10:
                    est_b = (K_b @ sq_inc[sl]) / K_b_sum
                    if est_b > 1e-10:
                        block_ests.append(np.log(est_b))
            if len(block_ests) >= 3:
                noise_var[jj] = np.var(block_ests, ddof=1) / len(block_ests)
            else:
                noise_var[jj] = 2.0 / n_eff[j]
        # Floor at independence assumption
        noise_var = np.maximum(noise_var, 2.0 / n_eff[valid])

        # Per-block α estimates for blocked bootstrap SE
        block_alphas = []
        for b in range(n_blocks):
            sl = slice(b * block_len, min((b + 1) * block_len, n))
            z_b = z[sl]
            sq_b = sq_inc[sl]
            diff_b = (landmarks[:, None] - z_b[None, :]) / bw
            K_b = np.exp(-0.5 * diff_b ** 2)
            K_b_sum = K_b.sum(axis=1)
            valid_b = valid & (K_b_sum > 1e-10)
            if np.sum(valid_b) < 5:
                continue
            s2_b = np.zeros(m)
            s2_b[K_b_sum > 1e-10] = (K_b[K_b_sum > 1e-10] @ sq_b) / K_b_sum[K_b_sum > 1e-10]
            mask_b = valid_b & (s2_b > 1e-10)
            if np.sum(mask_b) < 5:
                continue
            x_b = np.log(np.abs(landmarks[mask_b]))
            y_b = np.log(s2_b[mask_b])
            # WLS with n_eff weights
            n_eff_b = np.zeros(m)
            K_b_sq = (K_b ** 2).sum(axis=1)
            n_eff_b[K_b_sum > 1e-10] = K_b_sum[K_b_sum > 1e-10] ** 2 / K_b_sq[K_b_sum > 1e-10]
            w_b = n_eff_b[mask_b]
            H_b = np.column_stack([x_b, np.ones(len(x_b))])
            WH = H_b * w_b[:, None]
            try:
                beta_b = np.linalg.solve(WH.T @ H_b, WH.T @ y_b)
                block_alphas.append(beta_b[0])
            except np.linalg.LinAlgError:
                continue

        # Blocked bootstrap SE: SD of per-block α estimates / √n_blocks
        # This captures temporal correlation + model misspecification
        if len(block_alphas) >= 3:
            block_alpha_sd = np.std(block_alphas, ddof=1) / np.sqrt(len(block_alphas))
        else:
            block_alpha_sd = None

        # GP setup
        H = np.column_stack([x, np.ones(nv)])
        Sigma_n = np.diag(noise_var)
        x_range = x.max() - x.min()
        ell = max(x_range / 4.0, 0.1)
        sq_dists = (x[:, None] - x[None, :]) ** 2
        K_base = np.exp(-sq_dists / (2 * ell ** 2))

        # Blocked time-series CV for σ_f selection (R&W §5.4).
        # Split GP observations into temporal folds based on which time
        # block dominates each landmark's NW estimate.
        lm_block_id = np.zeros(nv, dtype=int)
        for jj, j in enumerate(valid_idx):
            block_weights = np.zeros(n_blocks)
            for b in range(n_blocks):
                sl = slice(b * block_len, min((b + 1) * block_len, n))
                block_weights[b] = K_nw[j, sl].sum()
            lm_block_id[jj] = np.argmax(block_weights)

        def _blocked_cv_mse(log_sf):
            """Leave-one-block-out CV for GP with parametric mean."""
            sf2 = np.exp(2 * log_sf)
            total_mse = 0.0
            n_test = 0
            unique_blocks = np.unique(lm_block_id)
            for fold in unique_blocks:
                test_mask = lm_block_id == fold
                train_mask = ~test_mask
                if np.sum(train_mask) < 3 or np.sum(test_mask) < 1:
                    continue
                # Train GP
                x_tr, y_tr = x[train_mask], y[train_mask]
                x_te, y_te = x[test_mask], y[test_mask]
                nv_tr = len(x_tr)
                H_tr = np.column_stack([x_tr, np.ones(nv_tr)])
                H_te = np.column_stack([x_te, np.ones(len(x_te))])
                Sigma_tr = np.diag(noise_var[train_mask])
                C_tr = sf2 * K_base[np.ix_(train_mask, train_mask)] + Sigma_tr
                try:
                    L_tr = np.linalg.cholesky(C_tr)
                except np.linalg.LinAlgError:
                    return 1e10
                Cinv_y = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, y_tr))
                Cinv_H = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, H_tr))
                A = H_tr.T @ Cinv_H
                try:
                    beta_hat = np.linalg.solve(A, H_tr.T @ Cinv_y)
                except np.linalg.LinAlgError:
                    return 1e10
                # Predict at test points: mean function + GP posterior mean
                r_tr = y_tr - H_tr @ beta_hat
                Cinv_r = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, r_tr))
                K_te_tr = sf2 * K_base[np.ix_(test_mask, train_mask)]
                y_pred = H_te @ beta_hat + K_te_tr @ Cinv_r
                total_mse += np.sum((y_te - y_pred) ** 2)
                n_test += len(y_te)
            return total_mse / max(1, n_test)

        # Grid search over σ_f (including σ_f = 0)
        log_sf_grid = np.concatenate([[-20], np.linspace(-4, 2, 20)])
        cv_scores = np.array([_blocked_cv_mse(lsf) for lsf in log_sf_grid])
        best_log_sf = log_sf_grid[np.argmin(cv_scores)]
        sf2_opt = np.exp(2 * best_log_sf)
        if best_log_sf <= -19:
            sf2_opt = 0.0

        # Build optimal C and compute posterior on β = (α, c)
        C = sf2_opt * K_base + Sigma_n
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return np.nan, np.nan

        # R&W eq. 2.42
        A = H.T @ C_inv @ H
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.nan, np.nan

        beta_hat = A_inv @ H.T @ C_inv @ y
        alpha_mean = float(beta_hat[0])
        gp_alpha_sd = float(np.sqrt(max(0, A_inv[0, 0])))

        # Final SD: max of GP posterior SD and blocked bootstrap SD.
        # GP captures model misspecification (non-power-law via σ_f > 0).
        # Bootstrap captures temporal correlation + finite-sample effects.
        if block_alpha_sd is not None:
            alpha_sd = max(gp_alpha_sd, block_alpha_sd)
        else:
            alpha_sd = gp_alpha_sd

        return alpha_mean, alpha_sd

    def alpha_per_asset(self, n_landmarks=80):
        """Fit growth exponent α per asset via 1D NW projection.

        For each asset i, projects to z = X_i (i-th coordinate) and
        estimates σ̂²(z) via Nadaraya-Watson on (ΔX_i)²/dt, then fits
        log(σ̂²) ~ α_i · log(X_i) via BayesianRidge.

        Same 1D pipeline as directional_alpha_test (consistent methodology).

        Returns:
            dict with alpha_means, alpha_sds, p_bubbles (arrays of length d)
        """
        from scipy import stats

        alpha_means = np.zeros(self.d)
        alpha_sds = np.zeros(self.d)
        p_bubbles = np.zeros(self.d)

        for i in range(self.d):
            z = self._X_t[:, i]
            dz = self._dX[:, i]
            alpha_means[i], alpha_sds[i] = self._estimate_alpha_1d(
                z, dz, self.dt, n_landmarks)

            if np.isnan(alpha_means[i]):
                p_bubbles[i] = 0.0
                continue

            if alpha_sds[i] > 0:
                z_score = (alpha_means[i] - 2.0) / alpha_sds[i]
                p_bubbles[i] = float(stats.norm.cdf(z_score))
            else:
                p_bubbles[i] = 1.0 if alpha_means[i] > 2.0 else 0.0

        return {
            'alpha_means': alpha_means,
            'alpha_sds': alpha_sds,
            'p_bubbles': p_bubbles,
        }

    def directional_alpha_test(self, n_directions=36, n_landmarks_1d=80,
                                candidate_directions=None):
        """
        Test for bubbles along portfolio directions via 1D projection.

        For each direction w on the unit sphere:
          1. Project: z_t = w^T X_t (scalar process)
          2. Squared increments: (Δz)²/dt → target
          3. 1D Nadaraya-Watson kernel regression: σ̂²_w(z)
          4. BayesianRidge: log(σ̂²_w) ~ α_w · log(|z|) + C

        This avoids the 2D covariance estimation entirely, using the
        well-tested 1D pipeline (α̂=2.00 for GBM, α̂=2.52 for CEV β=2.5).

        Bubble ⟺ ∃ direction w with α_w > 2 (Feller criterion on projection).

        Args:
            n_directions: Number of directions to sample (2D: half-circle)
            n_landmarks_1d: Landmarks for 1D NW regression per direction
            candidate_directions: Optional (n_dir, d) array of specific directions

        Returns dict with:
            - alpha_max: Maximum α across all directions
            - alpha_max_sd: SD of the maximizing direction's α
            - w_star: The direction achieving alpha_max
            - p_bubble: P(bubble | data) via max-z statistic
            - all_alphas, all_sds, all_directions: per-direction results
            - per_asset: Results from alpha_per_asset() for comparison
        """
        from sklearn.linear_model import BayesianRidge
        from scipy import stats

        d = self.d
        X_t = self._X_t
        dX = self._dX
        dt = self.dt
        n = len(X_t)

        # Generate candidate directions
        if candidate_directions is not None:
            directions = np.array(candidate_directions)
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        elif d == 2:
            angles = np.linspace(0, np.pi, n_directions, endpoint=False)
            directions = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            rng = np.random.RandomState(42)
            random_dirs = rng.randn(n_directions - d, d)
            random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
            directions = np.vstack([np.eye(d), random_dirs])

        all_alphas = np.zeros(len(directions))
        all_sds = np.zeros(len(directions))

        for i, w in enumerate(directions):
            z = X_t @ w
            dz = dX @ w
            all_alphas[i], all_sds[i] = self._estimate_alpha_1d(
                z, dz, dt, n_landmarks_1d)

        # Find maximum α direction
        valid_mask = ~np.isnan(all_alphas)
        if not np.any(valid_mask):
            return {
                'alpha_max': np.nan, 'alpha_max_sd': np.nan,
                'w_star': np.zeros(d), 'p_bubble': 0.0,
                'all_alphas': all_alphas, 'all_directions': directions,
                'per_asset': self.alpha_per_asset(),
            }

        best_idx = np.nanargmax(all_alphas)
        alpha_max = all_alphas[best_idx]
        alpha_max_sd = all_sds[best_idx]
        w_star = directions[best_idx]

        # P(bubble) via max-z statistic with proper multiple testing.
        # Under H0 (all α_w = 2): z_w = (α̂_w - 2)/σ̂_w ~ N(0,1).
        # T = max_w z_w. P(bubble) = Φ(T)^n_eff (CDF of max-normal).
        n_valid = np.sum(valid_mask)
        n_eff = max(d, n_valid // 4)

        z_scores = np.full(len(directions), -np.inf)
        for i in range(len(directions)):
            if valid_mask[i] and all_sds[i] > 0:
                z_scores[i] = (all_alphas[i] - 2.0) / all_sds[i]

        z_max = np.max(z_scores)
        if np.isfinite(z_max):
            p_bubble = float(stats.norm.cdf(z_max) ** n_eff)
        else:
            p_bubble = 0.0

        return {
            'alpha_max': float(alpha_max),
            'alpha_max_sd': float(alpha_max_sd),
            'w_star': w_star,
            'p_bubble': float(p_bubble),
            'all_alphas': all_alphas,
            'all_sds': all_sds,
            'all_directions': directions,
            'per_asset': self.alpha_per_asset(),
        }

    def conditional_feller_test(self, price_idx=0, vol_proxy_idx=None,
                                n_vol_bins=5, n_landmarks_1d=80):
        """Test for bubbles in asset `price_idx` conditional on a volatility proxy.

        Handles non-separable σ²(X,Y) where the marginal test might miss a
        bubble that only appears in certain vol regimes (JPS 2022 Remark 6).

        For each vol-proxy quantile bin, estimates α on the price sub-series
        falling in that bin.  Bubble iff ANY bin has P(α > 2) > 0.5.

        Args:
            price_idx: Index of the asset to test.
            vol_proxy_idx: Index of the vol proxy variable in X. If None,
                use realized QV of the price process (window=100).
            n_vol_bins: Number of quantile bins for the vol proxy.
            n_landmarks_1d: Landmarks for the 1D GP α estimator per bin.

        Returns:
            dict with per-bin and aggregate results.
        """
        from scipy import stats

        X_t = self._X_t
        dX = self._dX
        dt = self.dt
        n = len(X_t)

        z_all = X_t[:, price_idx]
        dz_all = dX[:, price_idx]

        # --- Vol proxy ---
        if vol_proxy_idx is not None:
            vol_proxy = X_t[:, vol_proxy_idx]
        else:
            # Realized QV from price increments
            w = 100
            sq_inc = dz_all ** 2
            vol_proxy = np.zeros(n)
            # Cumulative sum for efficient windowed average
            cs = np.concatenate([[0.0], np.cumsum(sq_inc)])
            for t in range(n):
                t0 = max(0, t - w)
                length = t - t0
                if length > 0:
                    vol_proxy[t] = (cs[t] - cs[t0]) / (length * dt)
                else:
                    vol_proxy[t] = sq_inc[t] / dt if t < len(sq_inc) else 0.0

        # --- Quantile bins ---
        bin_edges = np.quantile(vol_proxy, np.linspace(0, 1, n_vol_bins + 1))
        # Ensure unique edges (can happen with discrete data)
        bin_edges = np.unique(bin_edges)
        actual_n_bins = len(bin_edges) - 1
        if actual_n_bins < 1:
            return {
                'alpha_per_bin': np.array([np.nan]),
                'sd_per_bin': np.array([np.nan]),
                'p_bubble_per_bin': np.array([0.0]),
                'vol_bin_edges': bin_edges,
                'vol_bin_counts': np.array([n]),
                'alpha_max': np.nan,
                'alpha_max_sd': np.nan,
                'p_bubble': 0.0,
                'alpha_weighted': np.nan,
                'sd_weighted': np.nan,
            }

        alpha_per_bin = np.full(actual_n_bins, np.nan)
        sd_per_bin = np.full(actual_n_bins, np.nan)
        p_bubble_per_bin = np.zeros(actual_n_bins)
        vol_bin_counts = np.zeros(actual_n_bins, dtype=int)

        for b in range(actual_n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < actual_n_bins - 1:
                mask = (vol_proxy >= lo) & (vol_proxy < hi)
            else:
                mask = (vol_proxy >= lo) & (vol_proxy <= hi)

            vol_bin_counts[b] = int(np.sum(mask))
            if vol_bin_counts[b] < 200:
                # Too few points for reliable estimation
                continue

            z_bin = z_all[mask]
            dz_bin = dz_all[mask]

            a_mean, a_sd = self._estimate_alpha_1d(
                z_bin, dz_bin, dt, n_landmarks_1d)
            alpha_per_bin[b] = a_mean
            sd_per_bin[b] = a_sd

            if np.isnan(a_mean) or np.isnan(a_sd) or a_sd <= 0:
                p_bubble_per_bin[b] = 0.0
            else:
                z_score = (a_mean - 2.0) / a_sd
                p_bubble_per_bin[b] = float(stats.norm.cdf(z_score))

        # --- Aggregate with Šidák correction for multiple bins ---
        valid = ~np.isnan(alpha_per_bin) & ~np.isnan(sd_per_bin) & (sd_per_bin > 0)
        n_valid = int(np.sum(valid))

        # Max alpha across bins
        if np.any(valid):
            best_bin = np.nanargmax(alpha_per_bin)
            alpha_max = float(alpha_per_bin[best_bin])
            alpha_max_sd = float(sd_per_bin[best_bin])
        else:
            alpha_max = np.nan
            alpha_max_sd = np.nan

        # Šidák correction: P(bubble) = Φ(z_max)^n_bins under H0
        # Same logic as directional_alpha_test
        z_scores = np.full(actual_n_bins, -np.inf)
        for b in range(actual_n_bins):
            if valid[b] and sd_per_bin[b] > 0:
                z_scores[b] = (alpha_per_bin[b] - 2.0) / sd_per_bin[b]

        z_max = np.max(z_scores)
        n_eff_bins = max(1, n_valid)
        if np.isfinite(z_max):
            p_bubble = float(stats.norm.cdf(z_max) ** n_eff_bins)
        else:
            p_bubble = 0.0

        # Precision-weighted average
        if np.any(valid):
            prec = 1.0 / sd_per_bin[valid] ** 2
            alpha_weighted = float(np.sum(prec * alpha_per_bin[valid]) / np.sum(prec))
            sd_weighted = float(1.0 / np.sqrt(np.sum(prec)))
        else:
            alpha_weighted = np.nan
            sd_weighted = np.nan

        return {
            'alpha_per_bin': alpha_per_bin,
            'sd_per_bin': sd_per_bin,
            'p_bubble_per_bin': p_bubble_per_bin,
            'vol_bin_edges': bin_edges,
            'vol_bin_counts': vol_bin_counts,
            'alpha_max': alpha_max,
            'alpha_max_sd': alpha_max_sd,
            'p_bubble': p_bubble,
            'alpha_weighted': alpha_weighted,
            'sd_weighted': sd_weighted,
        }


def multivariate_cdc_example():
    """
    Example: Estimate 2D covariance from correlated process.
    """
    print("\n" + "=" * 70)
    print("MULTIVARIATE CdC: 2D COVARIANCE ESTIMATION")
    print("=" * 70)

    # Simulate correlated 2D process
    np.random.seed(42)
    T = 10000
    dt = 0.01
    d = 2

    # Correlation structure: ρ = 0.6
    rho = 0.6
    sigma1, sigma2 = 0.2, 0.3

    # True covariance matrix
    Sigma_true = np.array([
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2]
    ])

    print(f"True Σ:\n{Sigma_true}")
    print(f"ρ = {rho}, σ₁ = {sigma1}, σ₂ = {sigma2}")

    # Simulate
    L = np.linalg.cholesky(Sigma_true)
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t] = X[t-1] + L @ np.random.randn(d) * np.sqrt(dt)

    # Fit estimator
    estimator = MultivariateCdCEstimator(n_landmarks=100)
    estimator.fit(X, dt)

    # Evaluate at origin
    Sigma_est = estimator.predict(np.array([[0.0, 0.0]]))[0]

    print(f"\nEstimated Σ:\n{Sigma_est}")
    print(f"\nFrobenius error: {np.linalg.norm(Sigma_est - Sigma_true):.4f}")
    print(f"Relative error: {np.linalg.norm(Sigma_est - Sigma_true) / np.linalg.norm(Sigma_true):.1%}")

    # Compare estimated vs true correlation
    rho_est = Sigma_est[0, 1] / np.sqrt(Sigma_est[0, 0] * Sigma_est[1, 1])
    print(f"\nEstimated ρ: {rho_est:.3f} (true: {rho})")

    return estimator


# =============================================================================
# COMPARISON EXPERIMENT
# =============================================================================

def compare_estimators():
    """Compare A&J, CdC-KRR, and KGEDMD-CdC estimators."""
    print("=" * 70)
    print("COMPARING σ² ESTIMATORS")
    print("=" * 70)
    print("""
Connection: A&J is a special case of CdC kernel regression!

A&J: Nadaraya-Watson on squared increments (box or Gaussian kernel)
CdC-KRR: Same regression with RBF kernel + ridge + optional prior
KGEDMD-CdC: Learn generator, extract σ² via Carré du Champ identity
    """)

    # Simulate CIR
    kappa, theta, xi = 2.0, 0.04, 0.3
    dt = 0.01
    T = 20000

    print(f"CIR: dV = {kappa}({theta} - V)dt + {xi}√V dW")
    print(f"True σ²(V) = ξ²V = {xi**2:.4f}·V")
    print(f"T = {T} steps, dt = {dt}\n")

    V = simulate_cir(kappa, theta, xi, T, dt, seed=42)

    # Fit estimators
    print("Fitting estimators...")

    aj_box = AitSahaliaJacodEstimator(kernel='box')
    aj_box.fit(V, dt)

    aj_gauss = AitSahaliaJacodEstimator(kernel='gaussian')
    aj_gauss.fit(V, dt)

    # CdC-NW (Nadaraya-Watson) should MATCH A&J Gaussian
    cdc_nw = CdCKernelEstimator(prior_fn=None, mode='nw')
    cdc_nw.fit(V, dt)

    # CdC-KRR is different (global smoother with regularization)
    cdc_krr = CdCKernelEstimator(prior_fn=None, mode='krr')
    cdc_krr.fit(V, dt)

    cdc_cir_prior = CdCKernelEstimator(prior_fn=lambda v: xi**2 * v, mode='nw')
    cdc_cir_prior.fit(V, dt)

    kgedmd_cdc = KGEDMDCdCEstimator(n_landmarks=200)
    kgedmd_cdc.fit(V, dt)

    # Evaluate
    print("\n" + "-" * 70)
    print("σ² ESTIMATION ACCURACY")
    print("-" * 70)

    V_test = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    sigma_sq_true = xi**2 * V_test

    print(f"{'V':<8} {'True':>10} {'A&J Box':>10} {'A&J Gauss':>10} {'CdC-NW':>10} {'CdC-KRR':>10} {'CdC+Prior':>10} {'KGEDMD':>10}")
    print("-" * 80)

    for i, v in enumerate(V_test):
        true_val = sigma_sq_true[i]
        aj_b = aj_box.predict(np.array([v]))[0]
        aj_g = aj_gauss.predict(np.array([v]))[0]
        cdc_nw_v = cdc_nw.predict(np.array([v]))[0]
        cdc_krr_v = cdc_krr.predict(np.array([v]))[0]
        cdc_p = cdc_cir_prior.predict(np.array([v]))[0]
        kg = kgedmd_cdc.sigma_squared_cdc(np.array([v]))[0]

        print(f"{v:<8.3f} {true_val:>10.5f} {aj_b:>10.5f} {aj_g:>10.5f} {cdc_nw_v:>10.5f} {cdc_krr_v:>10.5f} {cdc_p:>10.5f} {kg:>10.5f}")

    # Compute RMSE
    print("\n" + "-" * 70)
    print("RMSE COMPARISON")
    print("-" * 70)

    V_eval = np.linspace(0.02, 0.08, 50)
    sigma_sq_true_eval = xi**2 * V_eval

    methods = {
        'A&J Box': aj_box.predict(V_eval),
        'A&J Gaussian': aj_gauss.predict(V_eval),
        'CdC-NW (≈A&J)': cdc_nw.predict(V_eval),
        'CdC-KRR (global)': cdc_krr.predict(V_eval),
        'CdC-NW (CIR prior)': cdc_cir_prior.predict(V_eval),
        'KGEDMD-CdC': kgedmd_cdc.sigma_squared_cdc(V_eval),
    }

    for name, pred in methods.items():
        rmse = np.sqrt(np.mean((pred - sigma_sq_true_eval)**2))
        rel_rmse = rmse / np.mean(sigma_sq_true_eval)
        print(f"{name:<25} RMSE: {rmse:.6f}  ({rel_rmse:.1%} relative)")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. A&J (Gaussian) = CdC-NW: EXACT equivalence when using same kernel/bandwidth
   - Both are Nadaraya-Watson: σ̂²(x) = Σᵢ K(x,xᵢ)yᵢ / Σᵢ K(x,xᵢ)
   - CdC-KRR (ridge) is DIFFERENT: uses (K+λI)⁻¹ instead of normalized weights

2. CdC-NW with prior σ²∝V: Most accurate (incorporates known structure)
3. KGEDMD-CdC: Comparable accuracy PLUS eigenfunctions for pricing

The CdC identity Γ(V,V) = L(V²) - 2V·L(V) extracts σ² from generator L.
This is algebraically equivalent to regression on squared increments.

NOVELTY for paper:
- A&J = CdC-NW ⊂ KGEDMD-CdC (increasing generality)
- KGEDMD adds: eigenfunctions → spectral pricing (Linetsky connection)
- Multidimensional: KGEDMD handles d×d covariance via Γ(Xᵢ, Xⱼ)
    """)

    return methods


# =============================================================================
# SECTION 6: Softplus Smoothing for Non-Smooth Payoffs
# =============================================================================

def softplus(x: np.ndarray, beta: float = 50.0) -> np.ndarray:
    """
    Smooth approximation to max(x, 0): softplus(x) = log(1 + exp(βx)) / β

    As β → ∞, softplus → max(x, 0)
    β ≈ 50 is a good default for variance-scale payoffs
    """
    # Numerically stable implementation
    return np.where(x > 20/beta, x, np.log1p(np.exp(beta * x)) / beta)


def softplus_debiased(x: np.ndarray, beta: float = 50.0) -> np.ndarray:
    """
    Debiased softplus: softplus(x) - softplus(0)

    This removes the bias at x=0 so that the smoothed payoff matches
    the raw payoff for deep ITM/OTM regions.
    """
    bias = np.log(2.0) / beta  # softplus(0) = log(2)/β
    return softplus(x, beta) - bias


def softplus_call(V: np.ndarray, K: float, beta: float = 50.0, debiased: bool = True) -> np.ndarray:
    """Smoothed call payoff: softplus(V - K)"""
    if debiased:
        return softplus_debiased(V - K, beta)
    return softplus(V - K, beta)


def softplus_put(V: np.ndarray, K: float, beta: float = 50.0, debiased: bool = True) -> np.ndarray:
    """Smoothed put payoff: softplus(K - V)"""
    if debiased:
        return softplus_debiased(K - V, beta)
    return softplus(K - V, beta)


def softplus_straddle(V: np.ndarray, K: float, beta: float = 50.0, debiased: bool = True) -> np.ndarray:
    """Smoothed straddle payoff: |V - K| ≈ softplus(V-K) + softplus(K-V)"""
    return softplus_call(V, K, beta, debiased) + softplus_put(V, K, beta, debiased)


def compare_pricing_methods():
    """
    Compare pricing methods:
    1. Direct KRR on (V_0, payoff(V_T)) pairs
    2. Koopman semigroup: K^n @ f
    3. Monte Carlo benchmark

    WITH AND WITHOUT softplus smoothing for option payoffs.
    """
    print("\n" + "=" * 70)
    print("KERNEL PRICING METHODS (NO EIGENFUNCTIONS)")
    print("=" * 70)
    print("""
Two kernel methods for E[f(V_T)|V_0]:

1. Direct KRR: Kernel regression on (V_t, f(V_{t+T})) pairs
   - Simplest, but horizon-specific

2. Koopman Power: K^{T/dt} @ f_coeffs
   - Uses learned Koopman, works for any horizon
   - Avoids eigendecomposition

+ Softplus smoothing: smooth approximation to max(x,0)
  softplus(x) = log(1 + exp(βx)) / β → max(x,0) as β → ∞
    """)

    # CIR simulation
    kappa, theta, xi = 2.0, 0.04, 0.3
    dt = 0.01
    T_data = 30000

    print(f"CIR: dV = {kappa}({theta} - V)dt + {xi}√V dW")
    print(f"Data: {T_data} steps, dt={dt}\n")

    V = simulate_cir(kappa, theta, xi, T_data, dt, seed=42)

    # Fit KGEDMD
    estimator = KGEDMDCdCEstimator(n_landmarks=150)
    estimator.fit(V, dt)

    # Test pricing
    V_0 = 0.04  # Start at theta
    T_horizon = 0.5  # 6 months

    # Payoff: variance swap = V_T (identity)
    identity = lambda x: x

    # Payoff: call option on variance (with and without smoothing)
    K_strike = 0.04
    beta_smooth = 200.0  # Higher beta = sharper approximation to max()

    call_raw = lambda x: np.maximum(x - K_strike, 0)
    call_smooth = lambda x: softplus_call(x, K_strike, beta=beta_smooth, debiased=False)

    # Put option
    put_raw = lambda x: np.maximum(K_strike - x, 0)
    put_smooth = lambda x: softplus_put(x, K_strike, beta=beta_smooth, debiased=False)

    # Straddle
    straddle_raw = lambda x: np.abs(x - K_strike)
    straddle_smooth = lambda x: softplus_straddle(x, K_strike, beta=beta_smooth, debiased=False)

    print("-" * 70)
    print(f"PRICING: V_0 = {V_0}, T = {T_horizon}, K = {K_strike}")
    print("-" * 70)

    # Monte Carlo benchmark
    n_steps = int(T_horizon / dt)
    n_mc = 50000
    np.random.seed(123)

    V_paths = np.zeros(n_mc)
    V_paths[:] = V_0

    for _ in range(n_steps):
        v = np.maximum(V_paths, 1e-8)
        V_paths = np.maximum(
            v + kappa * (theta - v) * dt + xi * np.sqrt(v) * np.sqrt(dt) * np.random.randn(n_mc),
            1e-8
        )

    E_V_mc = np.mean(V_paths)
    E_call_mc = np.mean(call_raw(V_paths))
    E_put_mc = np.mean(put_raw(V_paths))
    E_straddle_mc = np.mean(straddle_raw(V_paths))

    # Analytic E[V_T|V_0] for CIR
    E_V_analytic = theta + (V_0 - theta) * np.exp(-kappa * T_horizon)

    print("\n=== VARIANCE SWAP (Linear Payoff) ===")
    print(f"{'Method':<25} {'E[V_T|V_0]':>12} {'Error':>12}")
    print("-" * 50)
    print(f"{'Analytic':.<25} {E_V_analytic:>12.6f} {'-':>12}")
    print(f"{'Monte Carlo':.<25} {E_V_mc:>12.6f} {'-':>12}")

    E_V_krr = estimator.price_payoff_krr(V, identity, V_0, T_horizon)
    E_V_koop = estimator.price_payoff_koopman(identity, V_0, T_horizon)

    print(f"{'Direct KRR':.<25} {E_V_krr:>12.6f} {abs(E_V_krr - E_V_mc)/E_V_mc:>12.2%}")
    print(f"{'Koopman Power':.<25} {E_V_koop:>12.6f} {abs(E_V_koop - E_V_mc)/E_V_mc:>12.2%}")

    # === FAIR COMPARISON: Same payoff for MC and Kernel ===
    # Softplus smoothing helps kernel methods approximate the conditional expectation,
    # but we must compare both methods on the SAME payoff

    # MC with softplus payoff (for fair comparison)
    E_call_mc_smooth = np.mean(call_smooth(V_paths))
    E_put_mc_smooth = np.mean(put_smooth(V_paths))
    E_straddle_mc_smooth = np.mean(straddle_smooth(V_paths))

    print("\n=== CALL OPTION: max(V - K, 0) ===")
    print(f"{'Method':<30} {'Price':>12} {'Error':>12}")
    print("-" * 55)
    print(f"{'MC (raw payoff)':.<30} {E_call_mc:>12.6f} {'-':>12}")

    # Raw payoff - compare to raw MC
    E_call_krr_raw = estimator.price_payoff_krr(V, call_raw, V_0, T_horizon)
    E_call_koop_raw = estimator.price_payoff_koopman(call_raw, V_0, T_horizon)

    print(f"{'KRR (raw)':.<30} {E_call_krr_raw:>12.6f} {abs(E_call_krr_raw - E_call_mc)/E_call_mc:>12.2%}")
    print(f"{'Koopman (raw)':.<30} {E_call_koop_raw:>12.6f} {abs(E_call_koop_raw - E_call_mc)/E_call_mc:>12.2%}")

    # Softplus payoff - compare to softplus MC (FAIR comparison)
    print(f"\n{'MC (softplus payoff)':.<30} {E_call_mc_smooth:>12.6f} {'-':>12}")
    E_call_krr_smooth = estimator.price_payoff_krr(V, call_smooth, V_0, T_horizon)
    E_call_koop_smooth = estimator.price_payoff_koopman(call_smooth, V_0, T_horizon)

    print(f"{'KRR (softplus)':.<30} {E_call_krr_smooth:>12.6f} {abs(E_call_krr_smooth - E_call_mc_smooth)/E_call_mc_smooth:>12.2%}")
    print(f"{'Koopman (softplus)':.<30} {E_call_koop_smooth:>12.6f} {abs(E_call_koop_smooth - E_call_mc_smooth)/E_call_mc_smooth:>12.2%}")

    print("\n=== PUT OPTION: max(K - V, 0) ===")
    print(f"{'Method':<30} {'Price':>12} {'Error':>12}")
    print("-" * 55)
    print(f"{'MC (raw payoff)':.<30} {E_put_mc:>12.6f} {'-':>12}")

    E_put_krr_raw = estimator.price_payoff_krr(V, put_raw, V_0, T_horizon)
    E_put_koop_raw = estimator.price_payoff_koopman(put_raw, V_0, T_horizon)

    print(f"{'KRR (raw)':.<30} {E_put_krr_raw:>12.6f} {abs(E_put_krr_raw - E_put_mc)/E_put_mc:>12.2%}")
    print(f"{'Koopman (raw)':.<30} {E_put_koop_raw:>12.6f} {abs(E_put_koop_raw - E_put_mc)/E_put_mc:>12.2%}")

    print(f"\n{'MC (softplus payoff)':.<30} {E_put_mc_smooth:>12.6f} {'-':>12}")
    E_put_krr_smooth = estimator.price_payoff_krr(V, put_smooth, V_0, T_horizon)
    E_put_koop_smooth = estimator.price_payoff_koopman(put_smooth, V_0, T_horizon)

    print(f"{'KRR (softplus)':.<30} {E_put_krr_smooth:>12.6f} {abs(E_put_krr_smooth - E_put_mc_smooth)/E_put_mc_smooth:>12.2%}")
    print(f"{'Koopman (softplus)':.<30} {E_put_koop_smooth:>12.6f} {abs(E_put_koop_smooth - E_put_mc_smooth)/E_put_mc_smooth:>12.2%}")

    print("\n=== STRADDLE: |V - K| ===")
    print(f"{'Method':<30} {'Price':>12} {'Error':>12}")
    print("-" * 55)
    print(f"{'MC (raw payoff)':.<30} {E_straddle_mc:>12.6f} {'-':>12}")

    E_str_krr_raw = estimator.price_payoff_krr(V, straddle_raw, V_0, T_horizon)
    E_str_koop_raw = estimator.price_payoff_koopman(straddle_raw, V_0, T_horizon)

    print(f"{'KRR (raw)':.<30} {E_str_krr_raw:>12.6f} {abs(E_str_krr_raw - E_straddle_mc)/E_straddle_mc:>12.2%}")
    print(f"{'Koopman (raw)':.<30} {E_str_koop_raw:>12.6f} {abs(E_str_koop_raw - E_straddle_mc)/E_straddle_mc:>12.2%}")

    print(f"\n{'MC (softplus payoff)':.<30} {E_straddle_mc_smooth:>12.6f} {'-':>12}")
    E_str_krr_smooth = estimator.price_payoff_krr(V, straddle_smooth, V_0, T_horizon)
    E_str_koop_smooth = estimator.price_payoff_koopman(straddle_smooth, V_0, T_horizon)

    print(f"{'KRR (softplus)':.<30} {E_str_krr_smooth:>12.6f} {abs(E_str_krr_smooth - E_straddle_mc_smooth)/E_straddle_mc_smooth:>12.2%}")
    print(f"{'Koopman (softplus)':.<30} {E_str_koop_smooth:>12.6f} {abs(E_str_koop_smooth - E_straddle_mc_smooth)/E_straddle_mc_smooth:>12.2%}")

    print("\n" + "=" * 70)
    print("ADVANTAGE OF KERNEL PRICING")
    print("=" * 70)
    print("""
Both methods work WITHOUT eigenfunction extraction:

1. Direct KRR: Simple kernel regression, horizon-specific
2. Koopman Power: K^n in Nyström basis, any horizon, same trained model

For variance swaps (integrated payoffs), eigenfunction approach is better
because ∫e^{λt}dt has closed form. But for point-in-time payoffs, kernel
methods are simpler and equally accurate.
    """)


def simulate_cev(beta: float, sigma0: float, T: int, dt: float, seed: int = 42,
                  kappa: float = 2.0, theta: float = 0.04) -> np.ndarray:
    """
    Simulate CEV with mean-reversion: dV = κ(θ - V)dt + σ₀ V^{β/2} dW

    The diffusion coefficient is σ(V) = σ₀ V^{β/2}, so σ²(V) = σ₀² V^β.

    For β = 1: This is CIR (σ² ∝ V)
    For β = 0: This is OU (σ² = const)
    For β = 1.5: Intermediate power law
    """
    np.random.seed(seed)
    V = np.zeros(T)
    V[0] = theta  # Start at mean
    sqrt_dt = np.sqrt(dt)

    for t in range(1, T):
        v = max(V[t-1], 1e-8)
        drift = kappa * (theta - v) * dt
        diffusion = sigma0 * (v ** (beta / 2)) * sqrt_dt * np.random.randn()
        V[t] = max(v + drift + diffusion, 1e-8)

    return V


def simulate_ou(kappa: float, theta: float, sigma: float,
                T: int, dt: float, seed: int = 42) -> np.ndarray:
    """Simulate OU: dV = κ(θ - V)dt + σ dW"""
    np.random.seed(seed)
    V = np.zeros(T)
    V[0] = theta
    sqrt_dt = np.sqrt(dt)

    for t in range(1, T):
        V[t] = V[t-1] + kappa * (theta - V[t-1]) * dt + sigma * sqrt_dt * np.random.randn()

    return V


def comprehensive_sigma_estimation():
    """
    Comprehensive σ² estimation quality test across multiple dynamics.

    Tests:
    1. CIR: σ²(V) = ξ²V (linear in V)
    2. CEV: σ²(V) = σ₀²V^β (power law)
    3. OU: σ²(V) = σ² (constant)

    For each, compare:
    - A&J (box kernel) - baseline
    - CdC-KRR (no prior) - should match A&J
    - CdC-KRR (correct prior) - best accuracy
    - KGEDMD-CdC - learns generator → σ²
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE σ² ESTIMATION ACROSS DYNAMICS")
    print("=" * 70)

    dt = 0.01
    T = 20000

    results = {}

    # === CIR: σ²(V) = ξ²V ===
    print("\n--- CIR: dV = κ(θ - V)dt + ξ√V dW ---")
    kappa, theta, xi = 2.0, 0.04, 0.3
    V_cir = simulate_cir(kappa, theta, xi, T, dt, seed=42)

    # True σ²
    sigma_sq_cir = lambda v: xi**2 * v

    # Test points
    V_test = np.linspace(0.02, 0.08, 30)

    # Fit estimators
    aj = AitSahaliaJacodEstimator(kernel='gaussian')
    aj.fit(V_cir, dt)

    # CdC-NW should match A&J
    cdc_nw = CdCKernelEstimator(prior_fn=None, mode='nw')
    cdc_nw.fit(V_cir, dt)

    cdc_p = CdCKernelEstimator(prior_fn=lambda v: xi**2 * v, mode='nw')
    cdc_p.fit(V_cir, dt)

    kgedmd = KGEDMDCdCEstimator(n_landmarks=150)
    kgedmd.fit(V_cir, dt)

    # Compute errors
    true_vals = sigma_sq_cir(V_test)
    results['CIR'] = {
        'A&J': np.sqrt(np.mean((aj.predict(V_test) - true_vals)**2)) / np.mean(true_vals),
        'CdC-NW': np.sqrt(np.mean((cdc_nw.predict(V_test) - true_vals)**2)) / np.mean(true_vals),
        'CdC+Prior': np.sqrt(np.mean((cdc_p.predict(V_test) - true_vals)**2)) / np.mean(true_vals),
        'KGEDMD': np.sqrt(np.mean((kgedmd.sigma_squared_cdc(V_test) - true_vals)**2)) / np.mean(true_vals),
    }

    print(f"True: σ²(V) = {xi**2:.4f}·V")
    for method, rmse in results['CIR'].items():
        print(f"  {method:<12}: {rmse:.1%} relative RMSE")

    # === CEV: σ²(V) = σ₀²V^β ===
    print("\n--- CEV: dV = κ(θ-V)dt + σ₀ V^{β/2} dW ---")
    beta_cev, sigma0_cev = 1.5, 0.3  # More reasonable vol-of-vol
    V_cev = simulate_cev(beta_cev, sigma0_cev, T, dt, seed=42, kappa=2.0, theta=0.04)

    sigma_sq_cev = lambda v: sigma0_cev**2 * np.power(np.maximum(v, 1e-8), beta_cev)

    # Test at data range
    V_test_cev = np.percentile(V_cev, np.linspace(10, 90, 30))

    aj_cev = AitSahaliaJacodEstimator(kernel='gaussian')
    aj_cev.fit(V_cev, dt)

    cdc_cev_nw = CdCKernelEstimator(prior_fn=None, mode='nw')
    cdc_cev_nw.fit(V_cev, dt)

    cdc_cev_p = CdCKernelEstimator(prior_fn=lambda v: sigma0_cev**2 * np.power(np.maximum(v, 1e-8), beta_cev), mode='nw')
    cdc_cev_p.fit(V_cev, dt)

    kgedmd_cev = KGEDMDCdCEstimator(n_landmarks=150)
    kgedmd_cev.fit(V_cev, dt)

    true_vals_cev = sigma_sq_cev(V_test_cev)
    results['CEV'] = {
        'A&J': np.sqrt(np.mean((aj_cev.predict(V_test_cev) - true_vals_cev)**2)) / np.mean(true_vals_cev),
        'CdC-NW': np.sqrt(np.mean((cdc_cev_nw.predict(V_test_cev) - true_vals_cev)**2)) / np.mean(true_vals_cev),
        'CdC+Prior': np.sqrt(np.mean((cdc_cev_p.predict(V_test_cev) - true_vals_cev)**2)) / np.mean(true_vals_cev),
        'KGEDMD': np.sqrt(np.mean((kgedmd_cev.sigma_squared_cdc(V_test_cev) - true_vals_cev)**2)) / np.mean(true_vals_cev),
    }

    print(f"True: σ²(V) = {sigma0_cev**2:.4f}·V^{beta_cev}")
    for method, rmse in results['CEV'].items():
        print(f"  {method:<12}: {rmse:.1%} relative RMSE")

    # === OU: σ²(V) = σ² (constant) ===
    print("\n--- OU: dV = κ(θ - V)dt + σ dW ---")
    kappa_ou, theta_ou, sigma_ou = 1.0, 0.04, 0.01
    V_ou = simulate_ou(kappa_ou, theta_ou, sigma_ou, T, dt, seed=42)

    sigma_sq_ou = lambda v: sigma_ou**2 * np.ones_like(v)

    V_test_ou = np.linspace(0.02, 0.06, 30)

    aj_ou = AitSahaliaJacodEstimator(kernel='gaussian')
    aj_ou.fit(V_ou, dt)

    cdc_ou_nw = CdCKernelEstimator(prior_fn=None, mode='nw')
    cdc_ou_nw.fit(V_ou, dt)

    cdc_ou_p = CdCKernelEstimator(prior_fn=lambda v: sigma_ou**2 * np.ones_like(v), mode='nw')
    cdc_ou_p.fit(V_ou, dt)

    kgedmd_ou = KGEDMDCdCEstimator(n_landmarks=150)
    kgedmd_ou.fit(V_ou, dt)

    true_vals_ou = sigma_sq_ou(V_test_ou)
    results['OU'] = {
        'A&J': np.sqrt(np.mean((aj_ou.predict(V_test_ou) - true_vals_ou)**2)) / np.mean(true_vals_ou),
        'CdC-NW': np.sqrt(np.mean((cdc_ou_nw.predict(V_test_ou) - true_vals_ou)**2)) / np.mean(true_vals_ou),
        'CdC+Prior': np.sqrt(np.mean((cdc_ou_p.predict(V_test_ou) - true_vals_ou)**2)) / np.mean(true_vals_ou),
        'KGEDMD': np.sqrt(np.mean((kgedmd_ou.sigma_squared_cdc(V_test_ou) - true_vals_ou)**2)) / np.mean(true_vals_ou),
    }

    print(f"True: σ²(V) = {sigma_ou**2:.6f} (constant)")
    for method, rmse in results['OU'].items():
        print(f"  {method:<12}: {rmse:.1%} relative RMSE")

    # === Summary Table ===
    print("\n" + "=" * 70)
    print("SUMMARY: Relative RMSE by Dynamics and Method")
    print("=" * 70)
    print(f"{'Dynamics':<10} {'A&J':>12} {'CdC-NW':>12} {'CdC+Prior':>12} {'KGEDMD':>12}")
    print("-" * 60)
    for dyn, errs in results.items():
        print(f"{dyn:<10} {errs['A&J']:>12.1%} {errs['CdC-NW']:>12.1%} {errs['CdC+Prior']:>12.1%} {errs['KGEDMD']:>12.1%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. A&J = CdC-NW: Both are Nadaraya-Watson kernel regression on squared increments
   With same kernel (Gaussian) and bandwidth, they are IDENTICAL

2. CdC+Prior beats all when prior is correct: Incorporates parametric knowledge
   - CIR: prior σ² ∝ V
   - CEV: prior σ² ∝ V^β
   - OU: prior σ² = const

3. KGEDMD-CdC: Comparable to A&J, but ALSO gives:
   - Generator eigenfunctions → spectral pricing
   - Drift μ(V) via L(V)
   - Koopman semigroup for conditional expectations

4. When to use each:
   - A&J: Quick spot vol estimation, existing implementations
   - CdC+Prior: Best accuracy when parametric form known
   - KGEDMD: Full dynamics learning + derivative pricing
    """)

    return results


def pricing_accuracy_by_moneyness():
    """
    Test pricing accuracy as a function of option moneyness.

    Hypothesis: Softplus smoothing helps most for ATM options
    where the payoff kink is most problematic.
    """
    print("\n" + "=" * 70)
    print("PRICING ACCURACY BY MONEYNESS")
    print("=" * 70)

    # CIR simulation
    kappa, theta, xi = 2.0, 0.04, 0.3
    dt = 0.01
    T_data = 30000

    V = simulate_cir(kappa, theta, xi, T_data, dt, seed=42)

    # Fit KGEDMD
    estimator = KGEDMDCdCEstimator(n_landmarks=150)
    estimator.fit(V, dt)

    V_0 = 0.04
    T_horizon = 0.5
    n_steps = int(T_horizon / dt)

    # Monte Carlo paths
    n_mc = 50000
    np.random.seed(123)
    V_paths = np.zeros(n_mc)
    V_paths[:] = V_0
    for _ in range(n_steps):
        v = np.maximum(V_paths, 1e-8)
        V_paths = np.maximum(
            v + kappa * (theta - v) * dt + xi * np.sqrt(v) * np.sqrt(dt) * np.random.randn(n_mc),
            1e-8
        )

    # Test different strikes (moneyness levels)
    strikes = np.array([0.02, 0.03, 0.04, 0.05, 0.06])  # From deep ITM to OTM
    moneyness = V_0 / strikes  # M > 1 is ITM for calls
    beta_smooth = 200.0

    print(f"\nV_0 = {V_0}, T = {T_horizon}")
    print(f"\nKey: Err-R = error vs raw MC, Err-S = error vs softplus MC (FAIR comparison)")
    print(f"\n{'Strike':>8} {'M':>6} {'MC-Raw':>10} {'Koop-R':>10} {'Err-R':>7} {'MC-SP':>10} {'Koop-SP':>10} {'Err-S':>7}")
    print("-" * 80)

    for K, m in zip(strikes, moneyness):
        # MC benchmarks (both raw and softplus)
        call_mc_raw = np.mean(np.maximum(V_paths - K, 0))
        call_sp_fn = lambda x, k=K: softplus_call(x, k, beta=beta_smooth, debiased=False)
        call_mc_sp = np.mean(call_sp_fn(V_paths))

        # Koopman pricing
        call_raw_fn = lambda x, k=K: np.maximum(x - k, 0)
        price_raw = estimator.price_payoff_koopman(call_raw_fn, V_0, T_horizon)
        price_sp = estimator.price_payoff_koopman(call_sp_fn, V_0, T_horizon)

        # Errors (FAIR comparison: raw vs raw, softplus vs softplus)
        err_raw = abs(price_raw - call_mc_raw) / call_mc_raw if call_mc_raw > 1e-8 else 0
        err_sp = abs(price_sp - call_mc_sp) / call_mc_sp if call_mc_sp > 1e-8 else 0

        itm_otm = "ITM" if m > 1.1 else ("ATM" if abs(m - 1) < 0.15 else "OTM")
        print(f"{K:>8.3f} {m:>6.2f} {call_mc_raw:>10.6f} {price_raw:>10.6f} {err_raw:>7.1%} {call_mc_sp:>10.6f} {price_sp:>10.6f} {err_sp:>7.1%}  [{itm_otm}]")

    print("\n" + "=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    print("""
Softplus smoothing typically helps most for:
1. ATM options: The kink at V=K causes kernel approximation issues
2. OTM options: Small prices are sensitive to approximation error

For deep ITM options, the payoff is nearly linear → raw payoff works fine.
    """)


# =============================================================================
# SECTION 7: Signature-Augmented KGEDMD for Non-Markov Processes
# =============================================================================

class SigAugmentedKGEDMDEstimator:
    """
    KGEDMD with cumulative signature features as augmented state.

    For Markov processes, σ²(S_t) depends only on S_t → standard 1D KRR
    suffices. For non-Markov processes (rough vol, fSDE), σ²_t depends on
    path history. This estimator augments the state with running
    lead-lag log-signature features, so the KRR conditions on history:

        state_t = (S_t, QV_t, leverage_t)  from RecurrentLeadLagLogSigMap

    The σ² regression then conditions on BOTH price level and recent vol:
        σ̂²(state_t) = RBF_KRR(S_t, QV_t, leverage_t)

    For the α test, we still regress log(σ̂²) ~ α·log(S):
    - For Markov: QV is determined by S → regression residual is small
    - For non-Markov: QV varies independently → goes into residual
    - Either way, α reflects the true price-level scaling exponent

    Why this works when the PDE signature kernel doesn't:
    The PDE kernel computes global path similarity. After normalization,
    paths at different price levels look the same → KRR is constant → α≈0.
    Here we use EXPLICIT sig features (QV, leverage) as extra dimensions
    for the RBF kernel, preserving both price-level AND history information.

    Same math as KGEDMDCdCEstimator, just on augmented state space.
    """

    def __init__(self, n_landmarks: int = 80, regularization: float = 1e-3,
                 sig_gamma: float = 0.99, sig_level: int = 2):
        """
        Args:
            n_landmarks: Number of Nyström landmarks
            regularization: Ridge penalty for KRR
            sig_gamma: Forgetting factor for RecurrentLeadLagLogSigMap
                       (0.99 ≈ 100-step window, 1.0 = cumulative)
            sig_level: Signature truncation level (2 gives QV + leverage)
        """
        self.n_landmarks = n_landmarks
        self.regularization = regularization
        self.sig_gamma = sig_gamma
        self.sig_level = sig_level

    def fit(self, S: np.ndarray, dt: float):
        """
        Learn σ²(state) from price time series with sig-augmented state.

        1. Run RecurrentLeadLagLogSigMap to get sig features at each step
        2. Build augmented state (S_t, sig_features_t) at each step
        3. Fit KRR on augmented state → (ΔS)²/dt
        4. Store for α regression

        Args:
            S: 1D price time series
            dt: Time step
        """
        import sys, os
        # Import from proof_of_concept
        poc_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
            'examples', 'proof_of_concept')
        if poc_dir not in sys.path:
            sys.path.insert(0, poc_dir)
        from signature_features import RecurrentLeadLagLogSigMap

        S = np.asarray(S).flatten()
        self.dt = dt
        self._S_data = S.copy()
        N = len(S) - 1  # number of transitions

        # Step 1: Compute running signature features
        sig_map = RecurrentLeadLagLogSigMap(
            state_dim=1, level=self.sig_level,
            forgetting_factor=self.sig_gamma)

        # For d=1, level=2: 3 features [lead_disp, lag_disp, levy_area]
        # QV = 2 * |levy_area|
        sig_features = np.zeros((N, sig_map.feature_dim))
        for t in range(N):
            dx = np.array([S[t + 1] - S[t]])
            feats = sig_map.update(dx)
            sig_features[t] = feats

        # Skip warmup period for signature features
        warmup = max(50, int(1.0 / (1.0 - self.sig_gamma + 1e-8) * 0.5))
        warmup = min(warmup, N // 5)

        S_t = S[warmup:-1] if warmup > 0 else S[:-1]
        S_next = S[warmup + 1:] if warmup > 0 else S[1:]
        sig_feats = sig_features[warmup:]
        n_valid = len(S_t)

        # Step 2: Build augmented state
        # Use (log(S), QV, leverage) — log(S) works better than raw S for RBF
        log_S = np.log(np.maximum(S_t, 1e-8))
        qv = 2.0 * np.abs(sig_feats[:, 2]) if sig_feats.shape[1] >= 3 else np.zeros(n_valid)

        # Normalize features to comparable scales for RBF kernel
        log_S_std = max(np.std(log_S), 1e-8)
        qv_std = max(np.std(qv), 1e-8)

        # Augmented state: (log_S / std, qv / std)
        # Additional sig features (displacement) are correlated with S → skip
        X_aug = np.column_stack([
            log_S / log_S_std,
            qv / qv_std,
        ])

        # Store normalization for prediction
        self._log_S_std = log_S_std
        self._qv_std = qv_std
        self._S_t = S_t  # raw S values for α regression

        # Regression target: (ΔS)²/dt
        squared_inc = (S_next - S_t) ** 2 / dt

        # Step 3: Fit KRR on augmented state
        m = min(self.n_landmarks, n_valid // 5)
        idx_lm = self._select_landmarks_fps(X_aug, m)
        self.landmarks = X_aug[idx_lm]
        self.landmarks_S = S_t[idx_lm]  # raw S at landmarks

        # ARD bandwidth: separate length scale per dimension.
        # Optimize QV weight via LOO residual on a small grid.
        # If QV is redundant (Markov), its weight → 0 (bandwidth → ∞).
        #
        # Base bandwidth: median heuristic on log_S dimension alone
        log_S_lm = self.landmarks[:, 0]
        dists_1d = pdist(log_S_lm.reshape(-1, 1))
        bw_base = np.median(dists_1d) if len(dists_1d) > 0 else 1.0
        bw_base = max(bw_base, 0.1)

        # Grid search over QV weight: w ∈ {0, 0.25, 0.5, 1.0}
        # w=0 means ignore QV (pure 1D), w=1 means equal weight
        # Use GCV (generalized cross-validation) to select.
        # IMPORTANT: Only use QV if it improves GCV by > 5% over pure 1D.
        # This prevents QV from adding noise on Markov processes where
        # QV is redundant with S (improves in-sample but hurts α estimation).
        gcv_scores = {}

        for w_qv in [0.0, 0.25, 0.5, 1.0]:
            bw = np.array([bw_base, bw_base / max(w_qv, 0.01)])
            K_nM_w = self._kernel_matrix_ard(X_aug, self.landmarks, bw)
            K_gram_w = K_nM_w.T @ K_nM_w + self.regularization * np.eye(m)
            try:
                coeffs_w = np.linalg.solve(K_gram_w, K_nM_w.T @ squared_inc)
                resid = squared_inc - K_nM_w @ coeffs_w
                # Approximate LOO via GCV: ||r||² / (1 - trace(H)/n)²
                # where H = K_nM (K_nM'K_nM + λI)^{-1} K_nM' is the hat matrix
                H_trace = np.sum(K_nM_w * np.linalg.solve(K_gram_w, K_nM_w.T).T)
                gcv = np.mean(resid ** 2) / max(1 - H_trace / n_valid, 0.1) ** 2
            except np.linalg.LinAlgError:
                gcv = np.inf

            gcv_scores[w_qv] = gcv

        # Select: use QV only if it improves GCV by > 5% over w=0 (pure 1D)
        gcv_1d = gcv_scores[0.0]
        best_w = 0.0
        best_gcv = gcv_1d

        for w_qv in [0.25, 0.5, 1.0]:
            if gcv_scores[w_qv] < best_gcv and \
               (gcv_1d - gcv_scores[w_qv]) / max(gcv_1d, 1e-10) > 0.05:
                best_gcv = gcv_scores[w_qv]
                best_w = w_qv

        best_bw = np.array([bw_base, bw_base / max(best_w, 0.01)])
        self.bandwidth_ard = best_bw
        self._qv_weight = best_w

        K_nM = self._kernel_matrix_ard(X_aug, self.landmarks, self.bandwidth_ard)
        K_gram = K_nM.T @ K_nM + self.regularization * np.eye(m)
        self.sigma_sq_coeffs = np.linalg.solve(K_gram, K_nM.T @ squared_inc)

        # Store for jackknife
        self._X_aug = X_aug
        self._squared_inc = squared_inc
        self._K_gram_inv = np.linalg.inv(K_gram)
        self._residual_var = float(np.mean(
            (squared_inc - K_nM @ self.sigma_sq_coeffs) ** 2))

        return self

    def _select_landmarks_fps(self, X: np.ndarray, m: int) -> np.ndarray:
        """Farthest Point Sampling on augmented state."""
        n = X.shape[0]
        if m >= n:
            return np.arange(n)

        indices = [np.random.randint(n)]
        min_dists = np.full(n, np.inf)

        for _ in range(m - 1):
            last = indices[-1]
            d = np.sum((X - X[last]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d)
            indices.append(np.argmax(min_dists))

        return np.array(indices)

    def _kernel_matrix_ard(self, X1: np.ndarray, X2: np.ndarray,
                            bw: np.ndarray = None) -> np.ndarray:
        """ARD RBF kernel: separate bandwidth per dimension."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if bw is None:
            bw = self.bandwidth_ard
        bw = np.atleast_1d(bw)
        # Scale each dimension by its bandwidth
        X1_s = X1 / bw
        X2_s = X2 / bw
        sq_dists = np.sum(X1_s ** 2, axis=1, keepdims=True) + \
                   np.sum(X2_s ** 2, axis=1) - 2 * X1_s @ X2_s.T
        return np.exp(-0.5 * sq_dists)

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel using ARD bandwidths."""
        return self._kernel_matrix_ard(X1, X2)

    def sigma_squared_direct(self, X_query: np.ndarray) -> np.ndarray:
        """σ² at query points in augmented state space."""
        K = self._kernel_matrix(X_query, self.landmarks)
        return np.maximum(K @ self.sigma_sq_coeffs, 1e-10)

    def fit_alpha_bayesian(self, n_posterior_samples=200, n_blocks=10):
        """
        Fit log(σ̂²) ~ α·log(S) with jackknife uncertainty.

        The augmented-state KRR gives σ̂²(S_t, QV_t) conditioned on history.
        Regressing log(σ̂²) on log(S) gives α: the path-dependent part
        (captured by QV_t) goes into the residual.

        Returns: dict with alpha_mean, alpha_sd, p_bubble
        """
        from sklearn.linear_model import BayesianRidge
        from scipy import stats

        # Evaluate σ̂² at a SUBSAMPLE of training points (not just landmarks).
        #
        # Why not landmarks? Landmarks live in augmented (log_S, QV) space.
        # For Markov CEV, QV ∝ S^β, so the QV dimension absorbs S-scaling.
        # Evaluating σ̂² at landmarks and regressing on log(S) gives α ≈ β/2.
        #
        # Evaluating at training points samples the TRUE joint distribution.
        # σ̂²(S_t, QV_t) is the smoothed local vol at each time step.
        # Regressing log(σ̂²_t) on log(S_t) gives the MARGINAL relationship:
        # - For Markov: QV provides no extra info → α = β
        # - For non-Markov: QV-dependent part → residual → α still = β
        max_eval = 2000
        n_pts = len(self._X_aug)
        if n_pts > max_eval:
            eval_idx = np.random.choice(n_pts, max_eval, replace=False)
        else:
            eval_idx = np.arange(n_pts)

        X_eval = self._X_aug[eval_idx]
        S_eval = self._S_t[eval_idx]
        sigma2_eval = self.sigma_squared_direct(X_eval)

        valid = (S_eval > 1e-4) & (sigma2_eval > 1e-8)
        if np.sum(valid) < 20:
            return {'alpha_mean': np.nan, 'alpha_sd': np.nan,
                    'p_bubble': 0.0, 'method': 'sig_augmented'}

        log_S = np.log(S_eval[valid]).reshape(-1, 1)
        log_sig2 = np.log(sigma2_eval[valid])

        brr_full = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                  lambda_1=1e-6, lambda_2=1e-6,
                                  fit_intercept=True)
        brr_full.fit(log_S, log_sig2)
        alpha_full = float(brr_full.coef_[0])

        if np.isnan(alpha_full):
            return {'alpha_mean': np.nan, 'alpha_sd': np.nan,
                    'p_bubble': 0.0, 'method': 'sig_augmented'}

        # Delete-block jackknife for SD
        n_pts = len(self._X_aug)
        block_size = n_pts // n_blocks

        if block_size < 50:
            alpha_sd = float(np.sqrt(brr_full.sigma_[0, 0])) if hasattr(brr_full, 'sigma_') else 0.5
            alpha_sd = max(alpha_sd, 0.05)
            p_bubble = float(stats.norm.cdf((alpha_full - 2.0) / alpha_sd))
            return {'alpha_mean': alpha_full, 'alpha_sd': alpha_sd,
                    'p_bubble': p_bubble, 'method': 'sig_augmented'}

        alpha_jk = []
        for k in range(n_blocks):
            b_start = k * block_size
            b_end = min((k + 1) * block_size, n_pts)

            train_idx = np.concatenate([
                np.arange(0, b_start),
                np.arange(b_end, n_pts)
            ])

            if len(train_idx) < 100:
                continue

            X_train = self._X_aug[train_idx]
            y_train = self._squared_inc[train_idx]
            S_train = self._S_t[train_idx]

            m_k = min(self.n_landmarks, len(train_idx) // 5)
            if m_k < 10:
                continue

            idx_lm = self._select_landmarks_fps(X_train, m_k)
            lm_k = X_train[idx_lm]
            lm_S_k = S_train[idx_lm]

            K_nM_k = self._kernel_matrix(X_train, lm_k)
            K_gram_k = K_nM_k.T @ K_nM_k + self.regularization * np.eye(m_k)

            try:
                coeffs_k = np.linalg.solve(K_gram_k, K_nM_k.T @ y_train)
            except np.linalg.LinAlgError:
                continue

            # Evaluate σ̂² at training points (not landmarks)
            K_train_lm = self._kernel_matrix(X_train, lm_k)
            sig2_k = np.maximum(K_train_lm @ coeffs_k, 1e-10)

            valid_k = (S_train > 1e-4) & (sig2_k > 1e-8)
            if np.sum(valid_k) < 20:
                continue

            # Subsample for speed
            n_valid_k = int(np.sum(valid_k))
            if n_valid_k > 500:
                sub_idx = np.random.choice(n_valid_k, 500, replace=False)
                S_sub = S_train[valid_k][sub_idx]
                sig2_sub = sig2_k[valid_k][sub_idx]
            else:
                S_sub = S_train[valid_k]
                sig2_sub = sig2_k[valid_k]

            brr_k = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                   lambda_1=1e-6, lambda_2=1e-6,
                                   fit_intercept=True)
            brr_k.fit(np.log(S_sub).reshape(-1, 1),
                       np.log(sig2_sub))
            alpha_jk.append(float(brr_k.coef_[0]))

        if len(alpha_jk) < 3:
            alpha_sd = float(np.sqrt(brr_full.sigma_[0, 0])) if hasattr(brr_full, 'sigma_') else 0.5
            alpha_sd = max(alpha_sd, 0.05)
        else:
            alpha_jk = np.array(alpha_jk)
            K_eff = len(alpha_jk)
            alpha_var_jk = (K_eff - 1) / K_eff * np.sum(
                (alpha_jk - np.mean(alpha_jk)) ** 2)
            alpha_sd = max(np.sqrt(alpha_var_jk), 0.02)

        p_bubble = float(stats.norm.cdf((alpha_full - 2.0) / alpha_sd))

        return {
            'alpha_mean': alpha_full,
            'alpha_sd': alpha_sd,
            'p_bubble': p_bubble,
            'method': 'sig_augmented',
            'C_fit': float(np.exp(brr_full.intercept_)),
            'diagnostics': {
                'n_landmarks': len(self.landmarks),
                'bandwidth_ard': list(self.bandwidth_ard),
                'qv_weight': self._qv_weight,
                'sig_gamma': self.sig_gamma,
                'residual_var': self._residual_var,
                'n_jk_blocks': len(alpha_jk) if isinstance(alpha_jk, np.ndarray) else 0,
            }
        }


###############################################################################
# Section 8: Theory-Aligned Signature Generator + CdC Estimator
###############################################################################


class SigKGEDMDCdCEstimator:
    """
    Theory-aligned: Signature features → Koopman generator → CdC → σ²(S).

    Combines:
    - RecurrentLeadLagLogSigMap for path-dependent state augmentation
    - KGEDMD to learn the full Koopman generator L on augmented state
    - CdC identity to extract σ²_SS measure-invariantly: σ² = L(S²) - 2S·L(S)

    Why CdC matters:
    - Measure-invariant: annihilates drift, works under any ELMM (P or Q)
    - For non-Markov: generator on sig-augmented state captures path dependence
    - For Markov: GCV sets w_qv=0 → collapses to 1D generator (same as KGEDMDCdCEstimator)

    Also provides direct σ² regression as fallback (same as SigAugmentedKGEDMDEstimator).
    """

    def __init__(self, n_landmarks: int = 80, regularization: float = 1e-3,
                 sig_gamma: float = 0.99, sig_level: int = 2):
        self.n_landmarks = n_landmarks
        self.regularization = regularization
        self.sig_gamma = sig_gamma
        self.sig_level = sig_level

    def fit(self, S: np.ndarray, dt: float):
        """
        Learn Koopman generator on sig-augmented state.

        Steps:
        1. Compute running log-sig features (QV via Lévy area)
        2. Build augmented state (log_S/std, QV/std)
        3. ARD bandwidth + GCV model selection
        4. Learn Koopman → Generator L on augmented state pairs
        5. Direct σ² regression as fallback
        """
        import sys, os
        poc_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
            'examples', 'proof_of_concept')
        if poc_dir not in sys.path:
            sys.path.insert(0, poc_dir)
        from signature_features import RecurrentLeadLagLogSigMap

        S = np.asarray(S).flatten()
        self.dt = dt
        self._S_data = S.copy()
        N = len(S) - 1

        # Step 1: Running signature features
        sig_map = RecurrentLeadLagLogSigMap(
            state_dim=1, level=self.sig_level,
            forgetting_factor=self.sig_gamma)

        sig_features = np.zeros((N, sig_map.feature_dim))
        for t in range(N):
            dx = np.array([S[t + 1] - S[t]])
            sig_features[t] = sig_map.update(dx)

        # Skip warmup
        warmup = max(50, int(1.0 / (1.0 - self.sig_gamma + 1e-8) * 0.5))
        warmup = min(warmup, N // 5)

        S_t = S[warmup:-1] if warmup > 0 else S[:-1]
        S_next = S[warmup + 1:] if warmup > 0 else S[1:]
        sig_t = sig_features[warmup:]
        # Sig features at t+1 (for Koopman next-state)
        sig_next = sig_features[min(warmup + 1, N - 1):]
        # Align lengths
        n_valid = min(len(S_t), len(sig_t), len(sig_next))
        S_t = S_t[:n_valid]
        S_next = S_next[:n_valid]
        sig_t = sig_t[:n_valid]
        sig_next = sig_next[:n_valid]

        # Step 2: Augmented state
        log_S_t = np.log(np.maximum(S_t, 1e-8))
        log_S_next = np.log(np.maximum(S_next, 1e-8))
        qv_t = 2.0 * np.abs(sig_t[:, 2]) if sig_t.shape[1] >= 3 else np.zeros(n_valid)
        qv_next = 2.0 * np.abs(sig_next[:, 2]) if sig_next.shape[1] >= 3 else np.zeros(n_valid)

        log_S_std = max(np.std(log_S_t), 1e-8)
        qv_std = max(np.std(qv_t), 1e-8)
        self._log_S_std = log_S_std
        self._qv_std = qv_std

        X_aug_t = np.column_stack([log_S_t / log_S_std, qv_t / qv_std])
        X_aug_next = np.column_stack([log_S_next / log_S_std, qv_next / qv_std])

        self._S_t = S_t
        self._S_next = S_next
        squared_inc = (S_next - S_t) ** 2 / dt

        # Step 3: Landmarks + ARD bandwidth + GCV
        m = min(self.n_landmarks, n_valid // 5)
        idx_lm = self._select_landmarks_fps(X_aug_t, m)
        self.landmarks = X_aug_t[idx_lm]
        self.landmarks_S_raw = S_t[idx_lm]  # Raw S at landmarks (for CdC)

        # Base bandwidth: median heuristic on log_S dimension
        log_S_lm = self.landmarks[:, 0]
        dists_1d = pdist(log_S_lm.reshape(-1, 1))
        bw_base = max(np.median(dists_1d) if len(dists_1d) > 0 else 1.0, 0.1)

        # GCV grid search over QV weight
        gcv_scores = {}
        for w_qv in [0.0, 0.25, 0.5, 1.0]:
            bw = np.array([bw_base, bw_base / max(w_qv, 0.01)])
            K_nM_w = self._kernel_matrix_ard(X_aug_t, self.landmarks, bw)
            K_gram_w = K_nM_w.T @ K_nM_w + self.regularization * np.eye(m)
            try:
                coeffs_w = np.linalg.solve(K_gram_w, K_nM_w.T @ squared_inc)
                resid = squared_inc - K_nM_w @ coeffs_w
                H_trace = np.sum(K_nM_w * np.linalg.solve(K_gram_w, K_nM_w.T).T)
                gcv = np.mean(resid ** 2) / max(1 - H_trace / n_valid, 0.1) ** 2
            except np.linalg.LinAlgError:
                gcv = np.inf
            gcv_scores[w_qv] = gcv

        gcv_1d = gcv_scores[0.0]
        best_w = 0.0
        best_gcv = gcv_1d
        for w_qv in [0.25, 0.5, 1.0]:
            if gcv_scores[w_qv] < best_gcv and \
               (gcv_1d - gcv_scores[w_qv]) / max(gcv_1d, 1e-10) > 0.05:
                best_gcv = gcv_scores[w_qv]
                best_w = w_qv

        self.bandwidth_ard = np.array([bw_base, bw_base / max(best_w, 0.01)])
        self._qv_weight = best_w

        # Step 4: Learn Koopman generator on augmented state
        K_t = self._kernel_matrix_ard(X_aug_t, self.landmarks)
        K_next = self._kernel_matrix_ard(X_aug_next, self.landmarks)
        K_MM = self._kernel_matrix_ard(self.landmarks, self.landmarks)

        K_gram = K_t.T @ K_t + self.regularization * np.eye(m)
        self.Koopman = np.linalg.solve(K_gram, K_t.T @ K_next)
        self.L_matrix = (self.Koopman - np.eye(m)) / dt
        self.K_MM_inv = np.linalg.inv(K_MM + self.regularization * np.eye(m))

        # Step 5: Direct σ² regression (fallback)
        self.sigma_sq_coeffs = np.linalg.solve(K_gram, K_t.T @ squared_inc)

        # Store for jackknife
        self._X_aug_t = X_aug_t
        self._X_aug_next = X_aug_next
        self._squared_inc = squared_inc
        self._K_gram_inv = np.linalg.inv(K_gram)
        self._residual_var = float(np.mean(
            (squared_inc - K_t @ self.sigma_sq_coeffs) ** 2))

        return self

    def fit_koopman_at_horizon(self, k: int) -> dict:
        """Fit Koopman operator at horizon k*dt using stored augmented states.

        Must call fit() first. Reuses same landmarks and bandwidth.
        At horizon Δt = k*dt, the Koopman eigenvalue μ = e^{λ·Δt} is larger,
        making explosive modes detectable above regularization noise.

        Returns dict with koopman_eigvals, gen_eigvals, K_horizon, n_pairs,
        horizon_dt, and diagnostics (kernel coverage, semigroup ratio).
        """
        if not hasattr(self, '_X_aug_t'):
            raise RuntimeError("Call fit() before fit_koopman_at_horizon()")
        if k < 1:
            raise ValueError("k must be >= 1")

        X_aug = self._X_aug_t
        n = len(X_aug)
        if k >= n:
            raise ValueError(f"k={k} >= n_samples={n}")

        X_source = X_aug[:-k]
        X_target = self._X_aug_t  # use _X_aug_t not _X_aug_next for horizon k
        # For horizon k: pair (X_aug[i], X_aug[i+k])
        # But _X_aug_t was built from post-warmup data aligned with _X_aug_next
        # We need X_aug at t and X_aug at t+k
        # _X_aug_t[i] corresponds to time i, _X_aug_t[i+k] to time i+k
        X_target = X_aug[k:]
        n_pairs = len(X_source)

        m = len(self.landmarks)
        K_t = self._kernel_matrix_ard(X_source, self.landmarks)
        K_next = self._kernel_matrix_ard(X_target, self.landmarks)

        K_gram = K_t.T @ K_t + self.regularization * np.eye(m)
        K_horizon = np.linalg.solve(K_gram, K_t.T @ K_next)

        koopman_eigvals = np.linalg.eig(K_horizon)[0]
        delta_t = k * self.dt

        # Generator eigenvalues via log (NOT (μ-1)/Δt)
        # For complex μ: log(μ) = log|μ| + i·arg(μ)
        gen_eigvals = np.log(koopman_eigvals.astype(complex)) / delta_t

        # --- Diagnostics ---

        # Diagnostic 1: Kernel coverage at target points
        K_next_lm = self._kernel_matrix_ard(X_target, self.landmarks)
        coverage = K_next_lm.sum(axis=1)
        min_coverage = float(coverage.min())
        frac_low_coverage = float(np.mean(coverage < 0.1))

        # Diagnostic 2: Semigroup consistency
        # Compare with single-step leading eigenvalue
        single_step_eigvals = np.linalg.eig(self.Koopman)[0]
        lead_single = single_step_eigvals[np.argmax(np.abs(single_step_eigvals))]
        lead_horizon = koopman_eigvals[np.argmax(np.abs(koopman_eigvals))]
        log_lead_single = np.log(complex(lead_single)) if abs(lead_single) > 1e-10 else 0.0
        log_lead_horizon = np.log(complex(lead_horizon)) if abs(lead_horizon) > 1e-10 else 0.0
        semigroup_ratio = (np.real(log_lead_horizon) / np.real(log_lead_single)
                          if abs(np.real(log_lead_single)) > 1e-10 else np.nan)

        # Diagnostic 3: Effective explosion probability
        if hasattr(self, '_S_data'):
            S = self._S_data
            S_99 = np.percentile(S, 99.9)
            frac_extreme = float(np.mean(S[k:] > S_99 * 2)) if len(S) > k else 0.0
        else:
            frac_extreme = 0.0

        return {
            'koopman_eigvals': koopman_eigvals,
            'gen_eigvals': gen_eigvals,
            'K_horizon': K_horizon,
            'n_pairs': n_pairs,
            'horizon_dt': delta_t,
            'k': k,
            # Diagnostics
            'min_coverage': min_coverage,
            'frac_low_coverage': frac_low_coverage,
            'semigroup_ratio': float(np.real(semigroup_ratio)),
            'frac_extreme': frac_extreme,
        }

    # --- Kernel and landmark utilities ---

    def _select_landmarks_fps(self, X: np.ndarray, m: int) -> np.ndarray:
        """Farthest Point Sampling."""
        n = X.shape[0]
        if m >= n:
            return np.arange(n)
        indices = [np.random.randint(n)]
        min_dists = np.full(n, np.inf)
        for _ in range(m - 1):
            last = indices[-1]
            d = np.sum((X - X[last]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d)
            indices.append(np.argmax(min_dists))
        return np.array(indices)

    def _kernel_matrix_ard(self, X1: np.ndarray, X2: np.ndarray,
                            bw: np.ndarray = None) -> np.ndarray:
        """ARD RBF kernel."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if bw is None:
            bw = self.bandwidth_ard
        bw = np.atleast_1d(bw)
        X1_s = X1 / bw
        X2_s = X2 / bw
        sq_dists = np.sum(X1_s ** 2, axis=1, keepdims=True) + \
                   np.sum(X2_s ** 2, axis=1) - 2 * X1_s @ X2_s.T
        return np.exp(-0.5 * sq_dists)

    # --- Generator and CdC machinery ---

    def _function_to_coeffs(self, f_at_landmarks: np.ndarray) -> np.ndarray:
        """Function values at landmarks → Nyström coefficients."""
        return self.K_MM_inv @ f_at_landmarks

    def _coeffs_to_function(self, coeffs: np.ndarray,
                             X_query: np.ndarray) -> np.ndarray:
        """Nyström coefficients → function values at query points."""
        K = self._kernel_matrix_ard(X_query, self.landmarks)
        return K @ coeffs

    def generator_on_function(self, f_at_landmarks: np.ndarray,
                               X_query: np.ndarray = None) -> np.ndarray:
        """
        Apply generator L to function f, evaluated at query points.

        1. Convert f values → Nyström coefficients: α = K_MM⁻¹ @ f
        2. Apply L in coefficient space: β = L @ α
        3. Evaluate: (Lf)(x) = k(x, landmarks) @ β
        """
        alpha = self._function_to_coeffs(f_at_landmarks)
        beta = self.L_matrix @ alpha
        if X_query is None:
            X_query = self.landmarks
        return self._coeffs_to_function(beta, X_query)

    def sigma_squared_cdc(self, X_query_aug: np.ndarray,
                           S_query_raw: np.ndarray,
                           ito_correction: bool = True) -> np.ndarray:
        """
        Extract σ²_S via CdC on log(S) (measure-invariant).

        Key insight: Apply CdC to g=log(S) instead of S directly.
        Since landmarks live in log-space, g is nearly LINEAR in the
        first coordinate — much easier to approximate in the RKHS.

        By Itô: σ²_g(z) = L(g²)(z) - 2g·L(g)(z) = σ² of log-returns
        Then: σ²_S(S) = S² · σ²_g

        For GBM: σ²_g = σ₀² (constant!) — trivially learned.
        For CEV with σ(S)=c·S^(β/2): σ²_g = c²·S^(β-2), smoother than c²·S^β.

        The α regression uses σ²_S directly:
            log(σ²_S) = log(S²) + log(σ²_g) = 2·log(S) + (β-2)·log(S) + C = β·log(S) + C

        Args:
            X_query_aug: Query points in augmented (normalized) space
            S_query_raw: Corresponding raw (unnormalized) S values
            ito_correction: Subtract μ_g²·dt discrete-time bias
        """
        X_query_aug = np.atleast_2d(X_query_aug)
        S_query_raw = np.atleast_1d(S_query_raw)

        # Function values at landmarks: log(S) and log(S)²
        # log(S) ≈ landmarks[:, 0] * log_S_std (undo normalization)
        log_S_lm = np.log(np.maximum(self.landmarks_S_raw, 1e-8))
        f_g = log_S_lm
        f_g2 = log_S_lm ** 2

        # Apply generator
        L_g = self.generator_on_function(f_g, X_query_aug)    # ≈ μ_g(z) = μ - σ²/2
        L_g2 = self.generator_on_function(f_g2, X_query_aug)  # ≈ 2g·μ_g + σ²_g + μ_g²·dt

        # CdC on log(S): σ²_g = L(g²) - 2g·L(g)
        log_S_query = np.log(np.maximum(S_query_raw, 1e-8))
        sigma_sq_g_raw = L_g2 - 2 * log_S_query * L_g

        if ito_correction:
            sigma_sq_g = sigma_sq_g_raw - L_g ** 2 * self.dt
        else:
            sigma_sq_g = sigma_sq_g_raw

        # Convert: σ²_S = S² · σ²_g
        sigma_sq_S = S_query_raw ** 2 * np.maximum(sigma_sq_g, 1e-14)

        return np.maximum(sigma_sq_S, 1e-10)

    def sigma_squared_direct(self, X_query_aug: np.ndarray) -> np.ndarray:
        """σ² via direct KRR on (ΔS)²/dt (fallback, not measure-invariant)."""
        K = self._kernel_matrix_ard(X_query_aug, self.landmarks)
        return np.maximum(K @ self.sigma_sq_coeffs, 1e-10)

    def drift(self, X_query_aug: np.ndarray) -> np.ndarray:
        """Extract drift μ(z) = L(S)(z) from the generator."""
        f_S = self.landmarks_S_raw
        return self.generator_on_function(f_S, X_query_aug)

    # --- Sturm-Liouville eigenvalue test ---

    def sturm_liouville_eigenvalues(self, n_grid: int = 200,
                                      S_range: tuple = None,
                                      n_eigenvalues: int = 10,
                                      method: str = 'direct') -> dict:
        """
        Solve the generator eigenvalue problem via Sturm-Liouville discretization.

        Instead of extracting eigenvalues from the Koopman matrix (which reflects
        conservative Euler dynamics), discretize the generator L = ½σ²(S)∂²/∂S²
        + μ(S)∂/∂S on a 1D grid with Dirichlet BCs (killed semigroup).

        This uses the LOCAL estimates σ²(S) and μ(S) — which are accurate —
        to solve for GLOBAL eigenvalues without requiring explosive trajectories.

        Bubble ⟺ leading eigenvalue λ_1 > 0 (Khasminskii criterion).

        Args:
            n_grid: Number of interior grid points
            S_range: (S_min, S_max) for the grid. Default: data range.
            n_eigenvalues: Number of leading eigenvalues to return
            method: 'direct' or 'cdc' for σ² estimation

        Returns dict with eigenvalues, eigenvectors, grid, and diagnostics.
        """
        from scipy import linalg as sp_linalg

        # Grid setup
        S_data = self._S_t
        if S_range is None:
            S_min = max(np.percentile(S_data, 0.5), 1e-4)
            S_max = np.percentile(S_data, 99.5)
        else:
            S_min, S_max = S_range

        # Interior grid points (Dirichlet: u=0 at boundaries)
        S_grid = np.linspace(S_min, S_max, n_grid + 2)[1:-1]  # interior only
        h = (S_max - S_min) / (n_grid + 1)  # grid spacing

        # Evaluate σ²(S) and μ(S) at grid points
        log_S_grid = np.log(np.maximum(S_grid, 1e-8))
        # Build augmented query points (normalized)
        # For QV, use median value (we're evaluating the marginal generator)
        qv_median = np.median(np.abs(self._X_aug_t[:, 1])) * self._qv_std
        X_aug_grid = np.column_stack([
            log_S_grid / self._log_S_std,
            np.full(n_grid, qv_median / self._qv_std)
        ])

        if method == 'direct':
            sigma2 = self.sigma_squared_direct(X_aug_grid)
        else:
            sigma2 = self.sigma_squared_cdc(X_aug_grid, S_grid)

        mu = self.drift(X_aug_grid)

        # Build finite-difference matrix for L = ½σ²∂² + μ∂
        # Second derivative: (u_{i+1} - 2u_i + u_{i-1}) / h²
        # First derivative: (u_{i+1} - u_{i-1}) / (2h)
        L = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            a = 0.5 * sigma2[i] / h**2  # diffusion coefficient
            b = mu[i] / (2 * h)          # advection coefficient

            if i > 0:
                L[i, i - 1] = a - b      # u_{i-1}
            L[i, i] = -2 * a             # u_i
            if i < n_grid - 1:
                L[i, i + 1] = a + b      # u_{i+1}

        # Solve eigenvalue problem
        eigvals, eigvecs = sp_linalg.eig(L)

        # Sort by real part (descending — we want the leading eigenvalue)
        order = np.argsort(-np.real(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Take only requested number
        eigvals = eigvals[:n_eigenvalues]
        eigvecs = eigvecs[:, :n_eigenvalues]

        # Diagnostics
        max_re = float(np.real(eigvals[0]))
        n_positive = int(np.sum(np.real(eigvals) > 0))

        # Check eigenfunction positivity for leading mode
        lead_eigfn = np.real(eigvecs[:, 0])
        if np.sum(lead_eigfn) < 0:
            lead_eigfn = -lead_eigfn  # sign convention
        frac_positive = float(np.mean(lead_eigfn > 0))

        return {
            'eigenvalues': eigvals,
            'eigenvectors': eigvecs,
            'S_grid': S_grid,
            'sigma2': sigma2,
            'mu': mu,
            'max_re_lambda': max_re,
            'n_positive': n_positive,
            'lead_eigfn_frac_positive': frac_positive,
            'h': h,
            'S_range': (S_min, S_max),
            'method': method,
        }

    # --- Alpha estimation with both methods ---

    def fit_alpha_bayesian(self, n_posterior_samples=200, n_blocks=10):
        """
        Fit log(σ̂²) ~ α·log(S) using BOTH CdC and direct methods.

        Returns dict with alpha from CdC (primary, theory-aligned)
        and direct (fallback), plus auto-selection.
        """
        from sklearn.linear_model import BayesianRidge
        from scipy import stats

        # Subsample training points for evaluation
        max_eval = 2000
        n_pts = len(self._X_aug_t)
        if n_pts > max_eval:
            eval_idx = np.random.choice(n_pts, max_eval, replace=False)
        else:
            eval_idx = np.arange(n_pts)

        X_eval = self._X_aug_t[eval_idx]
        S_eval = self._S_t[eval_idx]

        # CdC σ²
        sigma2_cdc = self.sigma_squared_cdc(X_eval, S_eval)
        # Direct σ²
        sigma2_direct = self.sigma_squared_direct(X_eval)

        # Fit α for both methods
        def _fit_alpha(sigma2_vals, S_vals):
            valid = (S_vals > 1e-4) & (sigma2_vals > 1e-8)
            if np.sum(valid) < 20:
                return np.nan, np.nan, 0.0
            log_S = np.log(S_vals[valid]).reshape(-1, 1)
            log_sig2 = np.log(sigma2_vals[valid])
            brr = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                lambda_1=1e-6, lambda_2=1e-6,
                                fit_intercept=True)
            brr.fit(log_S, log_sig2)
            alpha = float(brr.coef_[0])
            alpha_brr_sd = float(np.sqrt(brr.sigma_[0, 0])) if hasattr(brr, 'sigma_') else 0.5
            return alpha, alpha_brr_sd, brr

        alpha_cdc, sd_cdc_brr, brr_cdc = _fit_alpha(sigma2_cdc, S_eval)
        alpha_direct, sd_direct_brr, brr_direct = _fit_alpha(sigma2_direct, S_eval)

        # Delete-block jackknife for SD calibration (on both methods)
        block_size = n_pts // n_blocks

        alpha_jk_cdc = []
        alpha_jk_direct = []

        if block_size >= 50:
            for k in range(n_blocks):
                b_start = k * block_size
                b_end = min((k + 1) * block_size, n_pts)
                train_idx = np.concatenate([
                    np.arange(0, b_start), np.arange(b_end, n_pts)])

                if len(train_idx) < 100:
                    continue

                X_tr = self._X_aug_t[train_idx]
                X_next_tr = self._X_aug_next[train_idx]
                y_tr = self._squared_inc[train_idx]
                S_tr = self._S_t[train_idx]

                m_k = min(self.n_landmarks, len(train_idx) // 5)
                if m_k < 10:
                    continue

                idx_lm = self._select_landmarks_fps(X_tr, m_k)
                lm_k = X_tr[idx_lm]
                lm_S_k = S_tr[idx_lm]

                K_nM_k = self._kernel_matrix_ard(X_tr, lm_k)
                K_gram_k = K_nM_k.T @ K_nM_k + self.regularization * np.eye(m_k)

                try:
                    # Direct coeffs for this fold
                    coeffs_k = np.linalg.solve(K_gram_k, K_nM_k.T @ y_tr)

                    # Generator for this fold
                    K_next_k = self._kernel_matrix_ard(X_next_tr, lm_k)
                    Koop_k = np.linalg.solve(K_gram_k, K_nM_k.T @ K_next_k)
                    L_k = (Koop_k - np.eye(m_k)) / self.dt
                    K_MM_k = self._kernel_matrix_ard(lm_k, lm_k)
                    K_MM_inv_k = np.linalg.inv(
                        K_MM_k + self.regularization * np.eye(m_k))
                except np.linalg.LinAlgError:
                    continue

                # Evaluate at training points for this fold
                K_eval_k = self._kernel_matrix_ard(X_tr, lm_k)

                # Direct σ²
                sig2_d_k = np.maximum(K_eval_k @ coeffs_k, 1e-10)

                # CdC σ² via log(S) — same as sigma_squared_cdc
                log_S_lm_k = np.log(np.maximum(lm_S_k, 1e-8))
                f_g_k = log_S_lm_k
                f_g2_k = log_S_lm_k ** 2
                alpha_g = K_MM_inv_k @ f_g_k
                beta_g = L_k @ alpha_g
                L_g_k = K_eval_k @ beta_g
                alpha_g2 = K_MM_inv_k @ f_g2_k
                beta_g2 = L_k @ alpha_g2
                L_g2_k = K_eval_k @ beta_g2
                log_S_tr = np.log(np.maximum(S_tr, 1e-8))
                sig2_g_k = L_g2_k - 2 * log_S_tr * L_g_k - L_g_k ** 2 * self.dt
                sig2_c_k = np.maximum(S_tr ** 2 * np.maximum(sig2_g_k, 1e-14), 1e-10)

                # Alpha from each
                for sig2, alpha_list in [(sig2_d_k, alpha_jk_direct),
                                          (sig2_c_k, alpha_jk_cdc)]:
                    valid = (S_tr > 1e-4) & (sig2 > 1e-8)
                    if np.sum(valid) < 20:
                        continue
                    log_S_k = np.log(S_tr[valid]).reshape(-1, 1)
                    log_s2_k = np.log(sig2[valid])
                    brr_k = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                          lambda_1=1e-6, lambda_2=1e-6,
                                          fit_intercept=True)
                    brr_k.fit(log_S_k, log_s2_k)
                    alpha_list.append(float(brr_k.coef_[0]))

        # Compute jackknife SDs
        def _jk_sd(alpha_full, alpha_jk_list):
            if len(alpha_jk_list) >= 3:
                arr = np.array(alpha_jk_list)
                K_eff = len(arr)
                var = (K_eff - 1) / K_eff * np.sum((arr - np.mean(arr)) ** 2)
                return max(np.sqrt(var), 0.02)
            else:
                return 0.5  # fallback

        sd_cdc = _jk_sd(alpha_cdc, alpha_jk_cdc)
        sd_direct = _jk_sd(alpha_direct, alpha_jk_direct)

        # Auto-select: Direct is primary for the α test because:
        # 1. Squared increments (ΔS)²/dt are ALREADY measure-invariant
        #    (drift b²·dt² is negligible), so direct ≈ CdC in accuracy.
        # 2. CdC via generator suffers catastrophic cancellation:
        #    σ²_g = L(g²) - 2g·L(g) subtracts two approximations.
        # 3. The generator is learned for PRICING/EIGENFUNCTION value,
        #    not for σ² which the direct method handles better.
        #
        # CdC α is reported as a diagnostic (validates generator quality:
        # if CdC ≈ direct, the generator is well-learned).
        direct_valid = (not np.isnan(alpha_direct)) and (-1 < alpha_direct < 10)
        cdc_valid = (not np.isnan(alpha_cdc)) and (-1 < alpha_cdc < 10) and (sd_cdc < 5)

        if direct_valid:
            alpha_selected = alpha_direct
            sd_selected = sd_direct
            method_selected = 'direct'
        elif cdc_valid:
            alpha_selected = alpha_cdc
            sd_selected = sd_cdc
            method_selected = 'cdc_fallback'
        else:
            alpha_selected = alpha_direct if not np.isnan(alpha_direct) else 0.0
            sd_selected = max(sd_direct, 0.5)
            method_selected = 'direct'

        p_bubble = float(stats.norm.cdf(
            (alpha_selected - 2.0) / max(sd_selected, 0.01)))

        return {
            'alpha_mean': alpha_selected,
            'alpha_sd': sd_selected,
            'p_bubble': p_bubble,
            'method': f'sig_generator_{method_selected}',
            'alpha_cdc': alpha_cdc,
            'alpha_cdc_sd': sd_cdc,
            'alpha_direct': alpha_direct,
            'alpha_direct_sd': sd_direct,
            'diagnostics': {
                'n_landmarks': len(self.landmarks),
                'bandwidth_ard': list(self.bandwidth_ard),
                'qv_weight': self._qv_weight,
                'sig_gamma': self.sig_gamma,
                'method_selected': method_selected,
                'cdc_valid': cdc_valid,
                'residual_var': self._residual_var,
                'n_jk_cdc': len(alpha_jk_cdc),
                'n_jk_direct': len(alpha_jk_direct),
            }
        }

    # --- Operator-based bubble diagnostics (Qin-Linetsky / Khasminskii) ---
    #
    # Theoretical connection (Q&L 2015, Thm 3.1):
    #   Bubble ⟺ strict local martingale under Q
    #          ⟺ pricing operator has NO recurrent positive eigenfunction
    #          ⟺ process is transient (not recurrent)
    #
    # For 1D diffusions with σ²(S) ~ c·S^α:
    #   α < 2 → bounded eigenfunctions exist → recurrent → no bubble
    #   α > 2 → no bounded eigenfunction → transient → bubble
    #
    # σ²(S) is measure-invariant (Girsanov changes only drift), so the
    # P-learned operator gives correct Q-answers for boundary classification.
    #
    # IMPORTANT: These operator tests are DIAGNOSTICS, not primary tests.
    # The direct α regression (fit_alpha_bayesian) is far more reliable
    # because it directly estimates σ²(S) — the quantity that determines
    # eigenfunction existence — without going through the operator.
    #
    # Why operator tests are noisy:
    # (1) RBF basis is bounded → eigenfunction reconstruction decays at
    #     boundary regardless of true eigenfunction behavior
    # (2) K^n amplifies Koopman approximation error exponentially
    # (3) Martingale defect at dt is O(dt²), smaller than operator error
    #
    # The operator's VALUE is for pricing (eigenfunction decomposition of
    # payoffs), not for σ² extraction or bubble classification.

    def eigenfunction_spectrum(self) -> dict:
        """
        Eigenvalue spectrum of the learned Koopman operator.

        Returns eigenvalues sorted by magnitude with continuous-time rates.
        For a well-learned operator:
        - Largest |λ| ≈ 1.0 (stationary/dominant mode)
        - Real eigenvalues with |λ| < 1 are stable decaying modes
        - Complex pairs indicate oscillatory dynamics
        """
        eigvals, eigvecs = np.linalg.eig(self.Koopman)
        mag = np.abs(eigvals)
        order = np.argsort(-mag)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        mag = mag[order]

        # Continuous-time rates: μ_k = log(λ_k) / dt
        # For real positive λ: μ = log(λ)/dt
        # For complex: μ = log|λ|/dt + i·angle(λ)/dt
        ct_rates = np.log(np.maximum(mag, 1e-15)) / self.dt

        spectral_gap = float(mag[0] - mag[1]) if len(mag) > 1 else 0.0

        return {
            'eigenvalues': eigvals,
            'eigenvectors': eigvecs,
            'magnitudes': mag,
            'continuous_time_rates': ct_rates,
            'spectral_gap': spectral_gap,
            'leading_eigenvalue': complex(eigvals[0]),
            'leading_magnitude': float(mag[0]),
        }

    def eigenfunction_bubble_test(self, tail_percentile: float = 75.0,
                                   n_top_eigenfunctions: int = 3) -> dict:
        """
        Test bubble via eigenfunction growth rate (Khasminskii / Q&L criterion).

        Theory (Qin & Linetsky 2015, Thm 3.1 + Appendix F):
        - No bubble: pricing operator has a bounded positive eigenfunction π(x)
        - Bubble: no such eigenfunction exists (π grows unboundedly)

        For 1D diffusions with σ²(S) ~ c·S^α:
        - α < 2: bounded eigenfunctions exist (recurrent, no bubble)
        - α > 2: eigenfunctions unbounded (transient, bubble)

        Method: Evaluate leading Koopman eigenfunctions at training points.
        Check growth rate d(log|φ|)/d(log S) in the upper tail vs lower tail.
        If upper-tail growth exceeds lower-tail growth → accelerating → bubble.

        NOTE: σ²(S) is measure-invariant (Girsanov changes only drift), so
        the boundary spectral classification from the P-Koopman gives the
        correct Q-answer for whether bounded eigenfunctions exist.

        CAVEAT: RBF basis functions are bounded, so ALL reconstructed
        eigenfunctions decay outside landmark support. We only test WITHIN
        the well-supported data range, using percentile bands.

        Args:
            tail_percentile: percentile threshold for upper/lower tail bands
            n_top_eigenfunctions: number of leading eigenfunctions to analyze

        Returns:
            dict with growth exponents, bubble probability, diagnostics
        """
        from scipy import stats as sp_stats

        spec = self.eigenfunction_spectrum()
        eigvals = spec['eigenvalues']
        eigvecs = spec['eigenvectors']

        # Evaluate eigenfunctions at training points
        K_train = self._kernel_matrix_ard(self._X_aug_t, self.landmarks)
        S_train = self._S_t.copy()
        log_S = np.log(np.maximum(S_train, 1e-8))

        # Percentile bands
        p_lo = 100 - tail_percentile  # e.g., 25th percentile
        p_hi = tail_percentile        # e.g., 75th percentile
        S_lo = np.percentile(S_train, p_lo)
        S_hi = np.percentile(S_train, p_hi)
        S_mid_lo = np.percentile(S_train, 40)
        S_mid_hi = np.percentile(S_train, 60)

        mask_lower = (S_train >= S_lo) & (S_train < S_mid_lo)
        mask_middle = (S_train >= S_mid_lo) & (S_train <= S_mid_hi)
        mask_upper = (S_train > S_mid_hi) & (S_train <= S_hi)

        results_per_ef = []

        for k in range(min(n_top_eigenfunctions, len(eigvals))):
            lam_k = eigvals[k]
            v_k = eigvecs[:, k]

            # Skip complex eigenfunctions (take only real part)
            if np.abs(lam_k.imag) > 0.01 * np.abs(lam_k.real):
                continue

            v_k = np.real(v_k)
            lam_k = np.real(lam_k)

            # Eigenfunction values at training points
            phi_k = K_train @ v_k

            # Ensure positive (eigenfunctions can be ± ; pick sign so mean > 0)
            if np.mean(phi_k) < 0:
                phi_k = -phi_k

            # Growth rate in each band: regress log|φ| ~ p·log(S)
            def _growth_in_band(mask):
                if np.sum(mask) < 10:
                    return np.nan, np.nan
                ls = log_S[mask]
                lp = np.log(np.maximum(np.abs(phi_k[mask]), 1e-15))
                valid = np.isfinite(ls) & np.isfinite(lp)
                if np.sum(valid) < 10:
                    return np.nan, np.nan
                slope, intercept, r_val, p_val, se = sp_stats.linregress(
                    ls[valid], lp[valid])
                return slope, se

            slope_lower, se_lower = _growth_in_band(mask_lower)
            slope_middle, se_middle = _growth_in_band(mask_middle)
            slope_upper, se_upper = _growth_in_band(mask_upper)

            # Full-range growth
            valid_all = (S_train >= S_lo) & (S_train <= S_hi)
            slope_full, se_full = _growth_in_band(valid_all)

            # Acceleration: does growth increase from lower to upper tail?
            # For bubble (CEV β>2): σ²~S^β → eigenfunction growth accelerates
            # For no-bubble (GBM): σ²~S² → eigenfunction growth is constant
            if not (np.isnan(slope_upper) or np.isnan(slope_lower)):
                acceleration = slope_upper - slope_lower
                # Combined SE for the difference
                se_accel = np.sqrt(se_upper**2 + se_lower**2) if \
                    not (np.isnan(se_upper) or np.isnan(se_lower)) else 1.0
            else:
                acceleration = 0.0
                se_accel = 1.0

            results_per_ef.append({
                'eigenvalue': float(lam_k),
                'slope_lower': float(slope_lower) if not np.isnan(slope_lower) else None,
                'slope_middle': float(slope_middle) if not np.isnan(slope_middle) else None,
                'slope_upper': float(slope_upper) if not np.isnan(slope_upper) else None,
                'slope_full': float(slope_full) if not np.isnan(slope_full) else None,
                'se_full': float(se_full) if not np.isnan(se_full) else None,
                'acceleration': float(acceleration),
                'se_acceleration': float(se_accel),
            })

        # Primary diagnostic: use the leading real eigenfunction
        if results_per_ef:
            lead = results_per_ef[0]
            accel = lead['acceleration']
            se_accel = lead['se_acceleration']

            # Positive acceleration → eigenfunction grows faster at large S
            # → transient boundary → bubble
            # Map to probability: P(bubble) = P(acceleration > 0)
            if se_accel > 0:
                z_accel = accel / se_accel
                p_bubble_accel = float(sp_stats.norm.cdf(z_accel))
            else:
                p_bubble_accel = 0.5
        else:
            accel = 0.0
            se_accel = 1.0
            p_bubble_accel = 0.5

        return {
            'p_bubble_eigen': p_bubble_accel,
            'acceleration': accel,
            'se_acceleration': se_accel,
            'leading_eigenvalue': float(np.real(eigvals[0])),
            'spectral_gap': spec['spectral_gap'],
            'eigenfunctions': results_per_ef,
            'n_lower': int(np.sum(mask_lower)),
            'n_middle': int(np.sum(mask_middle)),
            'n_upper': int(np.sum(mask_upper)),
        }

    def martingale_defect(self, n_grid: int = 50) -> dict:
        """
        Diagnostic: compute E[S_{t+dt} | S_t = s] for various s.

        For a true Q-martingale: E^Q[S_{t+dt}|S_t] = S_t (no defect).
        For a strict local martingale: E^Q[S_{t+dt}|S_t] < S_t for large S.

        Under P (observed measure), S has drift: E^P[S_{t+dt}|S_t] = S_t(1+μdt).
        We estimate E^P and check if the RATIO E^P[S_{t+dt}|S_t]/S_t decreases
        at large S. Decreasing ratio = evidence of local mass loss at ∞ = bubble.

        NOTE: Confounded by state-dependent drift under P. This is a
        diagnostic complement to the α test, not a standalone decision.

        Returns:
            dict with S_grid, conditional expectations, ratio, and slope.
        """
        from scipy import stats as sp_stats

        # Use binned training data for robustness
        S_t = self._S_t.copy()
        S_next = self._S_next.copy()

        # Percentile-based grid (avoid extremes)
        S_pcts = np.percentile(S_t, np.linspace(5, 95, n_grid))
        S_grid = np.unique(S_pcts)

        # For each grid point, compute E[S_next | S_t ≈ s] using Koopman
        # Method: Apply Koopman to f(x) = S(x) = exp(x[0] * log_S_std)
        # In the Nyström basis: f at landmarks = self.landmarks_S_raw
        f_S = self.landmarks_S_raw
        alpha_S = self._function_to_coeffs(f_S)
        beta_S = self.Koopman @ alpha_S  # E[f(x_{t+1})|x_t] in coeff space

        cond_exp = np.zeros(len(S_grid))
        for i, s in enumerate(S_grid):
            # Construct augmented state for query point s
            log_s_norm = np.log(max(s, 1e-8)) / self._log_S_std
            # QV: use median QV from nearby training points
            nearby = np.abs(S_t - s) < 0.1 * s
            if np.sum(nearby) > 5:
                qv_nearby = self._X_aug_t[nearby, 1]
                qv_val = np.median(qv_nearby)
            else:
                qv_val = np.median(self._X_aug_t[:, 1])

            x_query = np.array([[log_s_norm, qv_val]])
            K_q = self._kernel_matrix_ard(x_query, self.landmarks)
            cond_exp[i] = float(K_q @ beta_S)

        # Ratio: E[S_{t+dt}|S_t=s] / s
        ratio = cond_exp / np.maximum(S_grid, 1e-8)

        # Expected ratio under constant drift: (1 + μ·dt)
        # Estimate μ from data
        mean_ratio = np.mean(S_next / S_t)

        # Normalized ratio: observed / expected
        norm_ratio = ratio / mean_ratio

        # Slope of normalized ratio vs log(S) in upper half
        upper_half = S_grid > np.median(S_grid)
        if np.sum(upper_half) >= 5:
            slope, _, r_val, _, se = sp_stats.linregress(
                np.log(S_grid[upper_half]), norm_ratio[upper_half])
        else:
            slope, se, r_val = 0.0, 1.0, 0.0

        return {
            'S_grid': S_grid,
            'conditional_expectation': cond_exp,
            'ratio': ratio,
            'normalized_ratio': norm_ratio,
            'defect_slope': float(slope),
            'defect_slope_se': float(se),
            'mean_drift_ratio': float(mean_ratio),
            'note': ('Learned under P, not Q. Negative defect_slope at large S '
                     'suggests local mass loss (bubble). Confounded by '
                     'state-dependent drift.'),
        }

    def koopman_propagation_test(self, horizons=None, n_grid: int = 30) -> dict:
        """
        Multi-step Koopman propagation test for bubble detection.

        Key idea: Apply K^n to f(S)=S and f(S)=S² at various horizons.
        For a true martingale under Q:
          E[S_T | S_0=s] = s·e^{rT}  (constant ratio across S levels)
        For a strict local martingale (bubble):
          E[S_T | S_0=s] < s·e^{rT}  for large s and large T

        Under P, drift μ replaces r, but the KEY SIGNAL is:
        Does the ratio E[S_T|S_0=s] / (s·e^{μT}) DECREASE at large s?

        At one-step dt, the defect is O(dt²) — undetectable.
        At horizon T = n·dt, the defect accumulates to detectable levels
        because the strict local martingale property implies mass loss
        to infinity that grows with T.

        Uses eigendecomposition for efficient K^n computation.

        Also propagates f(S)=S² to extract E[S²_T|S_0]:
          Variance(S_T|S_0=s) = E[S²_T|S_0=s] - (E[S_T|S_0=s])²
        For bubble processes, variance grows superlinearly with s.

        Returns:
            dict with propagation ratios at each horizon, bubble scores.
        """
        from scipy import stats as sp_stats

        if horizons is None:
            horizons = [10, 50, 100, 200]  # in steps

        # Eigendecompose for efficient K^n
        eigvals, eigvecs = np.linalg.eig(self.Koopman)
        eigvecs_inv = np.linalg.inv(eigvecs)

        # Functions at landmarks: f(S) = S and f(S) = S²
        f_S = self.landmarks_S_raw
        f_S2 = self.landmarks_S_raw ** 2
        alpha_S = self._function_to_coeffs(f_S)
        alpha_S2 = self._function_to_coeffs(f_S2)

        # Grid of S values (percentile-based, avoid extremes)
        S_t = self._S_t
        S_pcts = np.percentile(S_t, np.linspace(10, 90, n_grid))
        S_grid = np.unique(S_pcts)

        # Build query augmented states
        X_query = np.zeros((len(S_grid), 2))
        for i, s in enumerate(S_grid):
            X_query[i, 0] = np.log(max(s, 1e-8)) / self._log_S_std
            nearby = np.abs(S_t - s) < 0.15 * s
            if np.sum(nearby) > 3:
                X_query[i, 1] = np.median(self._X_aug_t[nearby, 1])
            else:
                X_query[i, 1] = np.median(self._X_aug_t[:, 1])
        K_query = self._kernel_matrix_ard(X_query, self.landmarks)

        # Estimate drift from data (for normalization)
        mean_log_return = np.mean(np.log(self._S_next / self._S_t))
        mu_est = mean_log_return / self.dt

        results_by_horizon = {}
        bubble_scores = []

        for n_steps in horizons:
            T = n_steps * self.dt

            # K^n via eigendecomposition
            lambda_n = eigvals ** n_steps
            K_power_alpha_S = eigvecs @ (lambda_n * (eigvecs_inv @ alpha_S))
            K_power_alpha_S2 = eigvecs @ (lambda_n * (eigvecs_inv @ alpha_S2))
            K_power_alpha_S = np.real(K_power_alpha_S)
            K_power_alpha_S2 = np.real(K_power_alpha_S2)

            # E[S_T | S_0 = s] and E[S²_T | S_0 = s]
            E_S = K_query @ K_power_alpha_S
            E_S2 = K_query @ K_power_alpha_S2

            # Normalized ratio: E[S_T|S_0=s] / (s · e^{μT})
            drift_factor = np.exp(mu_est * T)
            ratio = E_S / (S_grid * drift_factor)

            # Does ratio decrease at large S? (bubble signature)
            log_S = np.log(S_grid)
            upper_mask = S_grid > np.percentile(S_grid, 50)
            if np.sum(upper_mask) >= 5:
                slope, _, _, _, se = sp_stats.linregress(
                    log_S[upper_mask], ratio[upper_mask])
            else:
                slope, se = 0.0, 1.0

            # Variance ratio: Var[S_T|S_0=s] / s²
            var_S = np.maximum(E_S2 - E_S**2, 0)
            var_ratio = var_S / (S_grid**2)

            # Does variance ratio increase at large S? (bubble = superlinear var)
            if np.sum(upper_mask) >= 5:
                var_slope, _, _, _, var_se = sp_stats.linregress(
                    log_S[upper_mask], np.log(np.maximum(var_ratio[upper_mask], 1e-15)))
            else:
                var_slope, var_se = 0.0, 1.0

            # Bubble score for this horizon:
            # Negative ratio_slope + positive var_slope = bubble evidence
            # Combine: score = -ratio_slope/se + var_slope/se
            score_ratio = -slope / max(se, 1e-8)
            score_var = var_slope / max(var_se, 1e-8)

            results_by_horizon[n_steps] = {
                'T': float(T),
                'S_grid': S_grid.copy(),
                'E_S': E_S.copy(),
                'ratio': ratio.copy(),
                'ratio_slope': float(slope),
                'ratio_slope_se': float(se),
                'var_ratio': var_ratio.copy(),
                'var_slope': float(var_slope),
                'var_slope_se': float(var_se),
                'score_ratio': float(score_ratio),
                'score_var': float(score_var),
            }
            bubble_scores.append(score_ratio)

        # Aggregate: bubble evidence should INCREASE with horizon
        # (the defect accumulates). Use the longest reliable horizon.
        # Reliable = ratio still positive (Koopman hasn't blown up)
        best_score = 0.0
        best_horizon = horizons[0]
        for n_steps in horizons:
            r = results_by_horizon[n_steps]
            # Check Koopman hasn't become unstable
            if np.all(r['ratio'] > 0) and np.all(r['ratio'] < 10):
                if r['score_ratio'] > best_score:
                    best_score = r['score_ratio']
                    best_horizon = n_steps

        # P(bubble) from the best horizon's ratio slope
        best_r = results_by_horizon[best_horizon]
        if best_r['ratio_slope_se'] > 0:
            # Negative slope = bubble evidence
            p_bubble = float(sp_stats.norm.cdf(
                -best_r['ratio_slope'] / best_r['ratio_slope_se']))
        else:
            p_bubble = 0.5

        return {
            'p_bubble_propagation': p_bubble,
            'best_horizon_steps': best_horizon,
            'best_horizon_T': float(best_horizon * self.dt),
            'best_ratio_slope': float(best_r['ratio_slope']),
            'best_ratio_slope_se': float(best_r['ratio_slope_se']),
            'horizons': results_by_horizon,
            'bubble_scores': bubble_scores,
            'mu_estimated': float(mu_est),
        }


if __name__ == "__main__":
    compare_estimators()
    comprehensive_sigma_estimation()
    multivariate_cdc_example()
    compare_pricing_methods()
    pricing_accuracy_by_moneyness()
