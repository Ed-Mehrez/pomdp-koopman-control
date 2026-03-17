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

        # Solve for each (i,j) entry of covariance
        K_MM_inv = np.linalg.inv(K_MM + self.regularization * np.eye(m))
        self.alpha = K_MM_inv @ (K_nM.T @ self.cov_targets) / n  # (m, d²)

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

    def alpha_per_asset(self, n_eval=200):
        """Fit growth exponent α per asset from diagonal covariance entries.

        For each asset i, evaluates Σ_ii(X) at the landmarks and fits
        log(Σ_ii) ~ α_i · log(X_i) via BayesianRidge.

        Returns:
            dict with alpha_means, alpha_sds, p_bubbles (arrays of length d)
        """
        from sklearn.linear_model import BayesianRidge
        from scipy import stats

        # Evaluate covariance at landmarks
        cov_at_landmarks = self.predict(self.landmarks)  # (m, d, d)

        alpha_means = np.zeros(self.d)
        alpha_sds = np.zeros(self.d)
        p_bubbles = np.zeros(self.d)

        for i in range(self.d):
            sigma2_ii = cov_at_landmarks[:, i, i]
            X_i = self.landmarks[:, i]

            # Filter valid
            valid = (X_i > 1e-4) & (sigma2_ii > 1e-8)
            if np.sum(valid) < 10:
                alpha_means[i] = np.nan
                alpha_sds[i] = np.nan
                p_bubbles[i] = 0.0
                continue

            log_X = np.log(X_i[valid]).reshape(-1, 1)
            log_sig2 = np.log(sigma2_ii[valid])

            brr = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                lambda_1=1e-6, lambda_2=1e-6,
                                fit_intercept=True)
            brr.fit(log_X, log_sig2)

            alpha_means[i] = float(brr.coef_[0])
            alpha_sds[i] = float(np.sqrt(brr.sigma_[0, 0])) if hasattr(brr, 'sigma_') else 0.5

            if alpha_sds[i] > 0:
                z = (alpha_means[i] - 2.0) / alpha_sds[i]
                p_bubbles[i] = float(stats.norm.cdf(z))
            else:
                p_bubbles[i] = 1.0 if alpha_means[i] > 2.0 else 0.0

        return {
            'alpha_means': alpha_means,
            'alpha_sds': alpha_sds,
            'p_bubbles': p_bubbles,
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


if __name__ == "__main__":
    compare_estimators()
    comprehensive_sigma_estimation()
    multivariate_cdc_example()
    compare_pricing_methods()
    pricing_accuracy_by_moneyness()
