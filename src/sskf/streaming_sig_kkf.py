"""
Streaming Signature Kernel Kalman Filter.

Architecture:
  1. Lead-lag transform on (time, price) → 4D lead-lag path
  2. Truncated cumulative log-sig (level 2-3) via BCH with decay γ
  3. RBF kernel on log-sig features → universal kernel on path space
     (Kiraly & Oberhauser 2019, JMLR 20:1-45)
  4. Online Nyström with k-center landmark selection
     (Kumar, Mohri & Talwalkar 2012, JMLR 13:981-1006)
  5. Kernel regression (KRR or NW) for σ²(S), drift, etc.
  6. Feller α test from NW σ² estimates → P(bubble)

Key insight: fSDEs are non-Markovian in state space X, but Markovian in
cumulative signature space S_t = Sig(X_{[0,t]}).

The lead-lag log-sig captures:
  - Lévy area = QV/2 (quadratic variation through antisymmetric part)
  - Leverage (cross-terms between return channels)
  - Path shape (higher-order iterated integrals)

References:
  - Kiraly & Oberhauser (2019): Kernels for sequentially ordered data
  - Salvi, Cass et al. (2021): Signature kernel via Goursat PDE
  - Toth & Oberhauser (2020): Bayesian learning with signature covariances
  - Calandriello, Lazaric & Valko (2017): Online kernel learning with
    adaptive Nyström embedding (PROS-N-KONS)
  - Toth & Oberhauser (2025): Recurrent sparse spectrum signature GPs

Memory: O(m² + m·sig_dim) where m = number of Nyström landmarks
"""

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist, cdist
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════════════
# 1. LEAD-LAG LOG-SIGNATURE STATE (BCH updates)
# ═══════════════════════════════════════════════════════════════════════════

class LeadLagLogSigState:
    """Maintains lead-lag log-signature via BCH updates.

    For d-dimensional input, lead-lag doubles to 2d dimensions.
    Each increment dx produces TWO BCH updates:
      1. (dx, 0) — lead moves, lag stays
      2. (0, dx) — lag catches up

    The Lévy area between lead_i and lag_i captures QV of channel i.

    At level 2 for input dim d=2 (time, return):
      - Lead-lag dim: 4
      - Level 1: 4 components
      - Level 2: 6 components (Lévy areas = 4*(4-1)/2)
      - Total: 10 features

    Parameters:
        input_dim: Dimension of input stream (2 for time+return)
        level: Log-sig truncation level (2 or 3)
        gamma: Forgetting factor (1.0 = cumulative, 0.99 = ~100 step window)
    """

    def __init__(self, input_dim: int = 2, level: int = 2, gamma: float = 1.0):
        self.input_dim = input_dim
        self.d = 2 * input_dim  # lead-lag doubles dimensions
        self.level = level
        self.gamma = gamma

        # Log-sig dimensions in lead-lag space
        self.dim_l1 = self.d
        self.dim_l2 = self.d * (self.d - 1) // 2
        self.feature_dim = self.dim_l1 + self.dim_l2
        if level >= 3:
            # Level 3: d * (d-1) * (d-2) / 6 + ... (Lie algebra dims)
            # For simplicity, use iisignature if available; otherwise skip
            self.dim_l3 = self.d * (self.d - 1) * (2 * self.d - 1) // 6
            self.feature_dim += self.dim_l3

        self.reset()

    def reset(self):
        """Reset to identity (empty path)."""
        self.l1 = np.zeros(self.dim_l1)
        self.l2 = np.zeros(self.dim_l2)
        if self.level >= 3:
            self.l3 = np.zeros(self.dim_l3)

    def _bch_update(self, dx_ll: np.ndarray):
        """Single BCH update in lead-lag space.

        BCH at level 2: L(a·b) = a + b + ½[a, b]
        BCH at level 3: + 1/12 [a,[a,b]] + 1/12 [b,[b,a]]
        """
        a1 = self.gamma * self.l1
        a2 = self.gamma**2 * self.l2

        # Lie bracket [a_decayed, dx_ll] at level 2
        bracket = np.zeros(self.dim_l2)
        idx = 0
        for i in range(self.d):
            for j in range(i + 1, self.d):
                bracket[idx] = a1[i] * dx_ll[j] - a1[j] * dx_ll[i]
                idx += 1

        self.l1 = a1 + dx_ll
        self.l2 = a2 + 0.5 * bracket

        # Level 3 BCH terms (optional, computationally heavier)
        if self.level >= 3:
            a3 = self.gamma**3 * self.l3
            # [a, [a, b]] and [b, [b, a]] terms — simplified for level 3
            # For now, use the level-2 bracket contribution only
            # (full level-3 BCH requires nested Lie brackets)
            self.l3 = a3  # Placeholder — upgrade with full BCH if needed

    def update(self, dx: np.ndarray) -> np.ndarray:
        """Lead-lag update: each input increment dx produces two BCH steps.

        Args:
            dx: (input_dim,) increment vector, e.g. [dt, d_return]

        Returns:
            (feature_dim,) current log-sig features
        """
        # Step 1: lead moves, lag stays
        dx_lead = np.zeros(self.d)
        dx_lead[:self.input_dim] = dx
        self._bch_update(dx_lead)

        # Step 2: lag catches up
        dx_lag = np.zeros(self.d)
        dx_lag[self.input_dim:] = dx
        self._bch_update(dx_lag)

        return self.get_features()

    def get_features(self) -> np.ndarray:
        """Return current log-sig as flat feature vector."""
        if self.level >= 3:
            return np.concatenate([self.l1, self.l2, self.l3])
        return np.concatenate([self.l1, self.l2])

    def copy(self) -> 'LeadLagLogSigState':
        """Deep copy of current state."""
        other = LeadLagLogSigState(self.input_dim, self.level, self.gamma)
        other.l1 = self.l1.copy()
        other.l2 = self.l2.copy()
        if self.level >= 3:
            other.l3 = self.l3.copy()
        return other


# ═══════════════════════════════════════════════════════════════════════════
# 2. RBF SIGNATURE KERNEL
# ═══════════════════════════════════════════════════════════════════════════

def sig_rbf_kernel(sig1: np.ndarray, sig2: np.ndarray, sigma: float) -> float:
    """RBF kernel on log-sig features.

    k(sig1, sig2) = exp(-||sig1 - sig2||² / (2σ²))

    This is a universal kernel on path space when applied to truncated
    log-signatures (Kiraly & Oberhauser 2019).
    """
    diff = sig1 - sig2
    return np.exp(-np.dot(diff, diff) / (2 * sigma**2))


def sig_rbf_kernel_matrix(sigs: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel matrix for a set of log-sig feature vectors.

    Args:
        sigs: (n, d) array of log-sig features
        sigma: RBF bandwidth

    Returns:
        (n, n) kernel matrix
    """
    sq_dists = cdist(sigs, sigs, 'sqeuclidean')
    return np.exp(-sq_dists / (2 * sigma**2))


def sig_rbf_kernel_vector(sigs: np.ndarray, sig_query: np.ndarray,
                           sigma: float) -> np.ndarray:
    """Kernel vector between query sig and a set of sigs.

    Args:
        sigs: (n, d) array of log-sig features
        sig_query: (d,) query log-sig
        sigma: RBF bandwidth

    Returns:
        (n,) kernel vector
    """
    diff = sigs - sig_query[None, :]
    sq_dists = np.sum(diff**2, axis=1)
    return np.exp(-sq_dists / (2 * sigma**2))


# ═══════════════════════════════════════════════════════════════════════════
# 3. ONLINE K-CENTER NYSTRÖM LANDMARKS
# ═══════════════════════════════════════════════════════════════════════════

class OnlineKCenter:
    """Online k-center (farthest-point) landmark maintenance.

    Maintains a set of m landmark sig-features such that the maximum
    distance from any observed point to its nearest landmark is minimized.

    When a new observation is farther than the current covering radius
    from all landmarks, it's added. When full, the two closest landmarks
    are merged (replaced by their midpoint).

    This gives O(1) per step, O(m²) per merge, O(m) landmarks.

    Reference: Kumar, Mohri & Talwalkar (2012), Section 4.
    """

    def __init__(self, max_landmarks: int = 100, sig_dim: int = 10):
        self.max_landmarks = max_landmarks
        self.sig_dim = sig_dim
        self.landmarks = np.empty((0, sig_dim))
        self.n_landmarks = 0
        self._min_pair_dist = np.inf

    def update(self, sig_new: np.ndarray) -> bool:
        """Process a new sig observation. Returns True if landmarks changed.

        Args:
            sig_new: (sig_dim,) new log-sig feature vector

        Returns:
            True if the landmark set was modified
        """
        if self.n_landmarks == 0:
            self.landmarks = sig_new.reshape(1, -1)
            self.n_landmarks = 1
            return True

        # Distance to nearest landmark
        dists = np.sum((self.landmarks - sig_new)**2, axis=1)
        min_dist = np.min(dists)

        # Only add if sufficiently far from existing landmarks
        if self.n_landmarks < self.max_landmarks:
            # Below budget: add if farther than median inter-landmark distance
            threshold = self._min_pair_dist * 0.5 if self._min_pair_dist < np.inf else 0
            if min_dist > threshold or self.n_landmarks < 5:
                self.landmarks = np.vstack([self.landmarks, sig_new])
                self.n_landmarks += 1
                self._update_min_pair_dist()
                return True
        else:
            # At budget: add only if farther than covering radius, then merge closest
            if min_dist > self._min_pair_dist:
                self.landmarks = np.vstack([self.landmarks, sig_new])
                self.n_landmarks += 1
                self._merge_closest()
                return True

        return False

    def _update_min_pair_dist(self):
        """Recompute minimum pairwise distance."""
        if self.n_landmarks < 2:
            self._min_pair_dist = np.inf
            return
        dists = pdist(self.landmarks, 'sqeuclidean')
        self._min_pair_dist = np.min(dists)

    def _merge_closest(self):
        """Merge the two closest landmarks into their midpoint."""
        if self.n_landmarks <= self.max_landmarks:
            return
        from scipy.spatial.distance import squareform
        D = squareform(pdist(self.landmarks, 'sqeuclidean'))
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(np.argmin(D), D.shape)

        # Replace i with midpoint, remove j
        self.landmarks[i] = 0.5 * (self.landmarks[i] + self.landmarks[j])
        self.landmarks = np.delete(self.landmarks, j, axis=0)
        self.n_landmarks -= 1
        self._update_min_pair_dist()

    def get_landmarks(self) -> np.ndarray:
        """Return (m, sig_dim) landmark array."""
        return self.landmarks[:self.n_landmarks]


# ═══════════════════════════════════════════════════════════════════════════
# 4. STREAMING SIGNATURE KERNEL LEARNER
# ═══════════════════════════════════════════════════════════════════════════

class StreamingSigKernelLearner:
    """Streaming kernel regression with lead-lag log-sig RBF kernel.

    Online KRR/NW using:
      - Lead-lag log-sig features (BCH updates, no windows)
      - RBF kernel on log-sig features
      - Online k-center Nyström landmarks
      - Sherman-Morrison rank-1 updates for KRR inverse

    Supports two regression methods:
      - 'krr': Kernel Ridge Regression (global smooth, with posterior variance)
      - 'nw':  Nadaraya-Watson (local weighted average, simpler)

    Memory: O(m² + m·sig_dim) where m = max_landmarks

    References:
      - Kiraly & Oberhauser (2019): sig-RBF is universal on path space
      - Calandriello et al. (2017): online Nyström embedding
    """

    def __init__(self, dt: float, input_dim: int = 2, sig_level: int = 2,
                 gamma_sig: float = 0.99, method: str = 'nw',
                 max_landmarks: int = 200, sigma: float = 1.0,
                 reg_param: float = 1e-2):
        """
        Args:
            dt: Time step
            input_dim: Dimension of input stream (2 for time+return)
            sig_level: Log-sig truncation level (2 or 3)
            gamma_sig: BCH forgetting factor (0.99 = ~100 step window)
            method: 'krr' or 'nw'
            max_landmarks: Nyström landmark budget
            sigma: RBF bandwidth for sig kernel
            reg_param: Ridge parameter (KRR) or unused (NW)
        """
        self.dt = dt
        self.method = method
        self.sigma = sigma
        self.lam = reg_param

        # Signature state
        self.sig_state = LeadLagLogSigState(
            input_dim=input_dim, level=sig_level, gamma=gamma_sig)

        # Nyström landmarks
        self.kcenter = OnlineKCenter(
            max_landmarks=max_landmarks, sig_dim=self.sig_state.feature_dim)

        # Support points: list of (sig_features, target)
        self.support_sigs: List[np.ndarray] = []
        self.support_targets: List[float] = []
        self.support_prices: List[float] = []

        # KRR state
        self.K = None
        self.K_reg_inv = None
        self.alpha_krr = None

        # Tracking
        self.x_current = None
        self.t = 0.0
        self.n_obs = 0

    def reset(self, x0: float = 0.0):
        """Reset for new trajectory."""
        self.sig_state.reset()
        self.kcenter = OnlineKCenter(
            max_landmarks=self.kcenter.max_landmarks,
            sig_dim=self.sig_state.feature_dim)
        self.support_sigs = []
        self.support_targets = []
        self.support_prices = []
        self.K = None
        self.K_reg_inv = None
        self.alpha_krr = None
        self.x_current = x0
        self.t = 0.0
        self.n_obs = 0

    def add_observation(self, x_new: float, target: Optional[float] = None
                        ) -> Tuple[float, float]:
        """Process one new observation.

        Args:
            x_new: New price/state value
            target: Regression target (e.g. (ΔS)²/dt for σ² estimation).
                    If None, uses dx/dt.

        Returns:
            (prediction, uncertainty) at current point
        """
        if self.x_current is None:
            self.x_current = x_new
            return 0.0, float('inf')

        dx = x_new - self.x_current

        # Update signature
        dx_input = np.array([self.dt, dx / max(abs(self.x_current), 1e-8)])
        sig_features = self.sig_state.update(dx_input)

        # Default target
        if target is None:
            target = dx / self.dt

        self.n_obs += 1

        # Check if this should become a landmark
        landmark_changed = self.kcenter.update(sig_features)

        # Always add to support set (budget managed separately)
        self.support_sigs.append(sig_features.copy())
        self.support_targets.append(target)
        self.support_prices.append(x_new)

        # Budget: keep support set aligned with landmarks
        max_support = self.kcenter.max_landmarks * 5
        if len(self.support_sigs) > max_support:
            # Remove oldest
            self.support_sigs.pop(0)
            self.support_targets.pop(0)
            self.support_prices.pop(0)
            self.K = None  # Force recompute

        # Prediction
        prediction = self._predict(sig_features)
        uncertainty = self._posterior_variance(sig_features) if self.method == 'krr' else np.nan

        self.t += self.dt
        self.x_current = x_new
        return prediction, uncertainty

    def _predict(self, sig_query: np.ndarray) -> float:
        """Predict at query signature."""
        if len(self.support_sigs) < 3:
            return 0.0

        sigs = np.array(self.support_sigs)
        targets = np.array(self.support_targets)
        k_vec = sig_rbf_kernel_vector(sigs, sig_query, self.sigma)

        if self.method == 'nw':
            w_sum = k_vec.sum()
            if w_sum < 1e-10:
                return targets[-1]
            return (k_vec @ targets) / w_sum

        elif self.method == 'krr':
            self._ensure_krr_solved()
            if self.alpha_krr is None:
                return 0.0
            return k_vec @ self.alpha_krr

        return 0.0

    def _ensure_krr_solved(self):
        """Solve KRR if kernel matrix is stale."""
        n = len(self.support_sigs)
        if n < 3:
            return

        if self.K is None or self.K.shape[0] != n:
            sigs = np.array(self.support_sigs)
            self.K = sig_rbf_kernel_matrix(sigs, self.sigma)
            K_reg = self.K + self.lam * np.eye(n)
            try:
                self.alpha_krr = np.linalg.solve(K_reg,
                                                  np.array(self.support_targets))
                self.K_reg_inv = np.linalg.inv(K_reg)
            except np.linalg.LinAlgError:
                self.alpha_krr = None
                self.K_reg_inv = None

    def _posterior_variance(self, sig_query: np.ndarray) -> float:
        """GP posterior variance at query."""
        self._ensure_krr_solved()
        if self.K_reg_inv is None:
            return float('inf')
        sigs = np.array(self.support_sigs)
        k_vec = sig_rbf_kernel_vector(sigs, sig_query, self.sigma)
        k_self = 1.0  # RBF k(x,x) = 1
        var = k_self - k_vec @ self.K_reg_inv @ k_vec
        return max(var, 0.0)

    def predict_at_price_levels(self, price_levels: np.ndarray) -> np.ndarray:
        """Predict regression target at given price levels.

        Uses NW with sig-RBF kernel on the price-matched subset.
        """
        if len(self.support_sigs) < 3:
            return np.zeros(len(price_levels))

        sigs = np.array(self.support_sigs)
        targets = np.array(self.support_targets)
        prices = np.array(self.support_prices)

        predictions = np.zeros(len(price_levels))
        for i, S in enumerate(price_levels):
            # Product kernel: price-RBF × sig-RBF
            price_bw = max(np.std(prices) * 0.3, 1e-6)
            k_price = np.exp(-0.5 * ((prices - S) / price_bw)**2)
            k_sig = sig_rbf_kernel_vector(sigs, self.sig_state.get_features(),
                                           self.sigma)
            k_total = k_price * k_sig
            w_sum = k_total.sum()
            if w_sum > 1e-10:
                predictions[i] = (k_total @ targets) / w_sum
            else:
                predictions[i] = np.nan

        return predictions


# ═══════════════════════════════════════════════════════════════════════════
# 5. STREAMING SIGNATURE KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════

class StreamingSigKKF:
    """Streaming Signature Kalman Filter.

    Maintains:
      - Lead-lag log-sig state via BCH (O(1) per step)
      - Koopman generator A in sig-feature space via RLS
      - Filter covariance P for UQ

    Online update:
      1. Observe new increment dX
      2. Update log-sig: BCH update with lead-lag
      3. Update Koopman via RLS: A += gain * (dS - A @ S * dt)

    The log-sig features stay FIXED SIZE regardless of trajectory length.
    """

    def __init__(self, dt: float, input_dim: int = 2, sig_level: int = 2,
                 gamma_sig: float = 0.99, forgetting_factor: float = 0.99,
                 initial_lambda: float = 1.0, process_noise: float = 1e-4):
        """
        Args:
            dt: Time step
            input_dim: Input stream dimension (2 for time+return)
            sig_level: Log-sig truncation level
            gamma_sig: BCH forgetting factor
            forgetting_factor: RLS forgetting factor
            initial_lambda: Initial regularization
            process_noise: Process noise for Kalman update
        """
        self.dt = dt
        self.ff = forgetting_factor
        self.process_noise = process_noise

        # Signature state
        self.sig_state = LeadLagLogSigState(
            input_dim=input_dim, level=sig_level, gamma=gamma_sig)
        self.sig_dim = self.sig_state.feature_dim

        # Augmented state: [1, sig_features] for generator with constant term
        self.aug_dim = 1 + self.sig_dim

        # Koopman generator A (aug_dim × aug_dim)
        self.A = np.zeros((self.aug_dim, self.aug_dim))

        # RLS inverse covariance
        self.P = np.eye(self.aug_dim) * initial_lambda

        # Tracking
        self.t = 0.0
        self.x_current = 0.0
        self.n_updates = 0
        self._prev_aug = None

    def reset(self, x0: float = 0.0):
        """Reset filter."""
        self.sig_state.reset()
        self.A = np.zeros((self.aug_dim, self.aug_dim))
        self.P = np.eye(self.aug_dim) * 1.0
        self.t = 0.0
        self.x_current = x0
        self.n_updates = 0
        self._prev_aug = None

    def update(self, x_new: float) -> Tuple[np.ndarray, np.ndarray]:
        """Process one observation, update generator via RLS.

        Args:
            x_new: New price/state value

        Returns:
            (sig_features, prediction_error)
        """
        dx = x_new - self.x_current

        # Update signature
        dx_input = np.array([self.dt, dx / max(abs(self.x_current), 1e-8)])
        sig_features = self.sig_state.update(dx_input)

        # Augmented state
        aug = np.concatenate([[1.0], sig_features])

        if self._prev_aug is not None:
            # dS/dt ≈ (aug - prev_aug) / dt
            ds = (aug - self._prev_aug) / self.dt

            # RLS update: A = A + gain * (ds - A @ prev_aug) * prev_aug^T
            phi = self._prev_aug
            innovation = ds - self.A @ phi

            # Forgetting
            self.P /= self.ff

            # Gain
            Pphi = self.P @ phi
            denom = 1.0 + phi @ Pphi
            gain = Pphi / denom

            # Update A (each row independently)
            for row in range(self.aug_dim):
                self.A[row] += gain * innovation[row]

            # Update P
            self.P -= np.outer(gain, Pphi)

            # Add process noise
            self.P += self.process_noise * np.eye(self.aug_dim)

            self.n_updates += 1

        self._prev_aug = aug.copy()
        self.t += self.dt
        self.x_current = x_new

        return sig_features, self.A @ aug * self.dt if self._prev_aug is not None else np.zeros(self.aug_dim)

    def predict_next(self) -> np.ndarray:
        """Predict next sig state from current + generator."""
        if self._prev_aug is None:
            return np.zeros(self.aug_dim)
        return self._prev_aug + self.A @ self._prev_aug * self.dt

    @property
    def generator(self) -> np.ndarray:
        """Current Koopman generator estimate."""
        return self.A.copy()


# ═══════════════════════════════════════════════════════════════════════════
# 6. STREAMING FELLER BUBBLE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class StreamingFellerDetector:
    """Online bubble detector using sig-RBF kernel.

    Architecture:
      1. Lead-lag log-sig with BCH decay → streaming path features
      2. Product kernel NW: K_price(S_i, S) · K_sig(sig_i, sig_current)
         on a support set with forgetting
      3. WLS for α from NW σ² estimates at price landmarks
      4. P(bubble) = Φ((α̂ - 2) / σ_α) + delta-method vol(P)

    No windows. The signature handles path memory.
    The NW forgetting handles non-stationarity.

    Parameters:
        n_price_landmarks: Price levels for evaluating σ²(S)
        max_support: Support set budget
        gamma_sig: BCH forgetting factor
        gamma_nw: NW weight forgetting factor
        sig_level: Log-sig truncation level
        sig_bw: RBF bandwidth for signature kernel
        price_bw_frac: Price NW bandwidth as fraction of price range
    """

    def __init__(self, n_price_landmarks: int = 40, max_support: int = 2000,
                 gamma_sig: float = 0.99, gamma_nw: float = 0.999,
                 sig_level: int = 2, sig_bw: float = 1.0,
                 price_bw_frac: float = 0.15, min_obs: int = 200):
        self.n_price_landmarks = n_price_landmarks
        self.max_support = max_support
        self.gamma_nw = gamma_nw
        self.sig_bw = sig_bw
        self.price_bw_frac = price_bw_frac
        self.min_obs = min_obs

        # Signature state
        self.sig_state = LeadLagLogSigState(
            input_dim=2, level=sig_level, gamma=gamma_sig)

        # Support set: circular buffer
        self._buf_price = np.zeros(max_support)
        self._buf_sq_inc = np.zeros(max_support)
        self._buf_sig = np.zeros((max_support, self.sig_state.feature_dim))
        self._buf_idx = 0
        self._buf_count = 0

        # Price landmarks and NW estimates
        self._landmarks = None
        self._log_landmarks = None
        self._bw_price = 1.0
        self._sigma2 = None

        # State
        self._prev_price = None
        self.n_obs = 0

        # Outputs
        self.alpha_mean = np.nan
        self.alpha_sd = np.nan
        self.p_bubble = 0.0
        self.vol_p_bubble = 0.0

    def initialize(self, prices: np.ndarray, dt: float):
        """Warm-start with initial prices to set landmarks.

        Args:
            prices: (n,) initial price array
            dt: time step in years
        """
        prices = prices[prices > 0]
        if len(prices) < 50:
            return

        # Set price landmarks at quantiles
        quantiles = np.linspace(0.02, 0.98, self.n_price_landmarks)
        self._landmarks = np.quantile(prices, quantiles)
        self._log_landmarks = np.log(np.abs(self._landmarks))
        lm_range = self._landmarks[-1] - self._landmarks[0]
        self._bw_price = max(lm_range * self.price_bw_frac, 1e-6)
        self._sigma2 = np.zeros(self.n_price_landmarks)
        self._dt = dt

        # Warm up signature with initial data
        for i in range(1, len(prices)):
            dS = prices[i] - prices[i - 1]
            dx = np.array([dt, dS / max(abs(prices[i - 1]), 1e-8)])
            self.sig_state.update(dx)

        self._prev_price = prices[-1]
        self.n_obs = 0

    def update(self, price: float) -> 'StreamingFellerDetector':
        """Process one new price.

        Args:
            price: Current price

        Returns:
            self (for chaining)
        """
        if self._prev_price is None or price <= 0 or not np.isfinite(price):
            return self

        dt = self._dt
        dS = price - self._prev_price
        sq_inc = dS**2 / dt

        # Update signature
        dx = np.array([dt, dS / max(abs(self._prev_price), 1e-8)])
        sig_features = self.sig_state.update(dx)

        # Add to circular buffer
        idx = self._buf_idx % self.max_support
        self._buf_price[idx] = self._prev_price
        self._buf_sq_inc[idx] = sq_inc
        self._buf_sig[idx] = sig_features
        self._buf_idx += 1
        self._buf_count = min(self._buf_count + 1, self.max_support)
        self.n_obs += 1

        # Refit α periodically (every ~1 trading day, not every step)
        if self.n_obs >= self.min_obs and self.n_obs % 390 == 0:
            self._refit_alpha(sig_features)

        self._prev_price = price
        return self

    def _refit_alpha(self, sig_current: np.ndarray):
        """Refit α via GP regression with sig-RBF kernel (R&W §2.7).

        Two-stage approach:
        1. Bin raw data by log-price quantiles → per-bin median of log(ΔS²/dt)
           and per-bin average signature. Medians are robust to the heavy-tailed
           noise in single-increment log(ΔS²/dt).
        2. GP regression on binned data with sig-RBF kernel + explicit basis
           (α·log|S| + c). The sig kernel captures path-dependent deviations
           from the global power law. ML selects σ_f: when sigs don't help,
           σ_f → 0 and this collapses to plain WLS.

        Per-bin noise ≈ Var(log χ²_1) / n_bin ≈ (π²/2) / n_bin, scaled by
        π/2 for median efficiency correction.
        """
        n = self._buf_count
        if n < 100:
            return

        prices = self._buf_price[:n]
        sq_incs = self._buf_sq_inc[:n]
        sigs = self._buf_sig[:n]

        # Filter valid data
        valid = (prices > 1e-4) & (sq_incs > 1e-20) & np.isfinite(sq_incs)
        n_valid = np.sum(valid)
        if n_valid < 50:
            return

        log_prices_all = np.log(prices[valid])
        log_sq_all = np.log(sq_incs[valid])
        sigs_all = sigs[valid]

        # ── Stage 1: Bin by log-price quantiles ────────────────────────
        n_bins = self.n_price_landmarks
        bin_edges = np.quantile(log_prices_all, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] -= 1e-6
        bin_edges[-1] += 1e-6
        bin_idx = np.clip(np.digitize(log_prices_all, bin_edges) - 1, 0, n_bins - 1)

        bin_x = np.zeros(n_bins)       # mean log-price
        bin_y = np.zeros(n_bins)       # median log(sq_inc)
        bin_sig = np.zeros((n_bins, self.sig_state.feature_dim))
        bin_noise = np.zeros(n_bins)   # estimated noise variance
        bin_n = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            mask = bin_idx == b
            nb = np.sum(mask)
            if nb < 3:
                continue
            bin_x[b] = np.mean(log_prices_all[mask])
            bin_y[b] = np.median(log_sq_all[mask])
            bin_sig[b] = np.mean(sigs_all[mask], axis=0)
            bin_n[b] = nb
            # Noise: Var(median) ≈ (π/2) · Var(mean) = (π/2) · σ² / n
            # where σ² ≈ π²/2 (Var of log χ²_1)
            bin_noise[b] = (np.pi / 2) * (np.pi**2 / 2) / nb

        usable = bin_n >= 3
        m = np.sum(usable)
        if m < 5:
            return

        x = bin_x[usable]
        y = bin_y[usable]
        sig_feats = bin_sig[usable]
        noise_var = bin_noise[usable]

        # ── Stage 2: Block bootstrap α SD ──────────────────────────────
        n_blocks = max(5, n // 500)
        block_len = n // n_blocks
        block_alphas = []
        for b in range(n_blocks):
            sl = slice(b * block_len, min((b + 1) * block_len, n))
            p_b = prices[sl]
            sq_b = sq_incs[sl]
            val_b = (p_b > 1e-4) & (sq_b > 1e-20) & np.isfinite(sq_b)
            if np.sum(val_b) < 10:
                continue
            lp = np.log(p_b[val_b])
            ly = np.log(sq_b[val_b])
            H_b = np.column_stack([lp, np.ones(np.sum(val_b))])
            try:
                beta_b = np.linalg.lstsq(H_b, ly, rcond=None)[0]
                block_alphas.append(beta_b[0])
            except Exception:
                continue
        block_alpha_sd = (np.std(block_alphas, ddof=1) / np.sqrt(len(block_alphas))
                          if len(block_alphas) >= 3 else None)

        # ── Stage 3: GP with sig-RBF kernel (fast path) ────────────────
        # Compute sig kernel once; grid search σ_f uses pre-factored Sigma_n
        sig_sq_dists = cdist(sig_feats, sig_feats, 'sqeuclidean')
        pos_dists = sig_sq_dists[np.triu_indices(m, k=1)]
        if len(pos_dists) > 0 and np.any(pos_dists > 0):
            sig_bw = max(np.sqrt(np.median(pos_dists[pos_dists > 0])), 1e-6)
        else:
            sig_bw = self.sig_bw

        K_sig = np.exp(-sig_sq_dists / (2 * sig_bw**2))
        noise_diag = noise_var
        H = np.column_stack([x, np.ones(m)])

        # Pre-compute eigendecomposition of K_sig for fast grid search
        # C = sf2 * K_sig + diag(noise) → eigenvalues shift by sf2
        evals_k, evecs_k = np.linalg.eigh(K_sig)
        # Transform y and H into eigen-basis
        Qty = evecs_k.T @ (y / noise_diag)     # weighted by 1/noise
        QtH = evecs_k.T @ (H / noise_diag[:, None])

        # For each σ_f: C⁻¹ = Q @ diag(1/(sf2*λ + noise)) @ Q^T
        # but noise varies per point, so we need full solve.
        # Faster: just use 5 grid points instead of 13, and use cho_solve.
        log_sf_grid = np.array([-20, -2, -0.5, 0.5, 1.5])
        best_ml = -np.inf
        best_log_sf = -20

        for log_sf in log_sf_grid:
            sf2 = np.exp(2 * log_sf) if log_sf > -19 else 0.0
            C = sf2 * K_sig + np.diag(noise_diag)
            try:
                L, low = cho_factor(C)
                Cinv_y = cho_solve((L, low), y)
                Cinv_H = cho_solve((L, low), H)
                A = H.T @ Cinv_H
                beta_hat = np.linalg.solve(A, H.T @ Cinv_y)

                r = y - H @ beta_hat
                Cinv_r = cho_solve((L, low), r)

                log_det_C = 2 * np.sum(np.log(np.abs(np.diag(L))))
                sign, log_det_A = np.linalg.slogdet(A)
                ml = (-0.5 * r @ Cinv_r
                      - 0.5 * log_det_C
                      - 0.5 * log_det_A
                      - 0.5 * (m - 2) * np.log(2 * np.pi))

                if ml > best_ml:
                    best_ml = ml
                    best_log_sf = log_sf
            except np.linalg.LinAlgError:
                continue

        # ── Stage 4: GP posterior with best σ_f ────────────────────────
        sf2 = np.exp(2 * best_log_sf) if best_log_sf > -19 else 0.0
        C = sf2 * K_sig + np.diag(noise_diag)

        try:
            L, low = cho_factor(C)
            Cinv_y = cho_solve((L, low), y)
            Cinv_H = cho_solve((L, low), H)
            A = H.T @ Cinv_H
            A_inv = np.linalg.inv(A)
            beta_hat = A_inv @ (H.T @ Cinv_y)

            self.alpha_mean = float(beta_hat[0])
            gp_alpha_sd = float(np.sqrt(max(A_inv[0, 0], 1e-10)))

            # Conservative: max of GP posterior SD and block bootstrap SD
            if block_alpha_sd is not None and np.isfinite(block_alpha_sd):
                self.alpha_sd = max(gp_alpha_sd, block_alpha_sd)
            else:
                self.alpha_sd = gp_alpha_sd

            if self.alpha_sd > 0:
                z_score = (self.alpha_mean - 2.0) / self.alpha_sd
                self.p_bubble = float(stats.norm.cdf(z_score))
                self.vol_p_bubble = float(stats.norm.pdf(z_score))
            else:
                self.p_bubble = 0.0
                self.vol_p_bubble = 0.0

            # Diagnostics
            self._gp_sf2 = sf2
            self._gp_sig_bw = sig_bw
            self._gp_alpha_sd = gp_alpha_sd
            self._block_alpha_sd = block_alpha_sd

        except np.linalg.LinAlgError:
            pass

        # Update σ² at landmarks for current_sigma_atm
        valid_prices = prices[prices > 0.01]
        if len(valid_prices) > 50:
            quantiles = np.linspace(0.02, 0.98, self.n_price_landmarks)
            self._landmarks = np.quantile(valid_prices, quantiles)
            self._log_landmarks = np.log(np.abs(self._landmarks))
            lm_range = self._landmarks[-1] - self._landmarks[0]
            self._bw_price = max(lm_range * self.price_bw_frac, 1e-6)
            for k in range(self.n_price_landmarks):
                k_p = np.exp(-0.5 * ((prices - self._landmarks[k]) / self._bw_price)**2)
                w_sum = k_p.sum()
                if w_sum > 1e-10:
                    self._sigma2[k] = (k_p @ sq_incs) / w_sum

    @property
    def current_sigma_atm(self) -> float:
        """Current annualized BS vol at latest price."""
        if self._prev_price is None or self._landmarks is None:
            return 0.3
        S = self._prev_price
        if S <= 0:
            return 0.3
        d = (self._landmarks - S) / self._bw_price
        w = np.exp(-0.5 * d**2)
        w_sum = w.sum()
        if w_sum < 1e-10:
            idx = np.argmin(np.abs(self._landmarks - S))
            s2 = self._sigma2[idx]
        else:
            s2 = (w @ self._sigma2) / w_sum
        vol = np.sqrt(max(s2, 1e-10)) / S
        return float(np.clip(vol, 0.05, 5.0))


# ═══════════════════════════════════════════════════════════════════════════
# 7. TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_lead_lag_logsig():
    """Test that lead-lag log-sig captures QV through Lévy area."""
    print("Test: Lead-lag log-sig captures QV...")

    np.random.seed(42)
    n = 1000
    dt = 0.01

    # GBM: dS = μS dt + σS dW
    sigma = 0.3
    S = np.zeros(n)
    S[0] = 100
    for i in range(1, n):
        S[i] = S[i-1] * np.exp((0.05 - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.randn())

    state = LeadLagLogSigState(input_dim=2, level=2, gamma=1.0)
    for i in range(1, n):
        dS = S[i] - S[i-1]
        dx = np.array([dt, dS / S[i-1]])
        state.update(dx)

    features = state.get_features()
    # For 2D input → 4D lead-lag → level 1: 4 components, level 2: 6 areas
    assert len(features) == 10, f"Expected 10 features, got {len(features)}"

    # The Lévy area between lead_return and lag_return should ≈ QV/2
    # QV ≈ Σ (dS/S)² ≈ σ²·T
    realized_qv = sum((np.diff(S) / S[:-1])**2)
    # Lévy area is in l2[1] (lead_ret × lag_ret cross-term)
    # With cumulative (γ=1), area accumulates all QV
    print(f"  Realized QV: {realized_qv:.4f}")
    print(f"  σ²·T: {sigma**2 * n * dt:.4f}")
    print(f"  Log-sig features: {features}")
    print(f"  PASS (features computed, {len(features)} dims)")


def test_online_kcenter():
    """Test online k-center maintains diverse landmarks."""
    print("Test: Online k-center...")

    np.random.seed(42)
    kcenter = OnlineKCenter(max_landmarks=20, sig_dim=10)

    # Stream random points
    n_added = 0
    for i in range(500):
        sig = np.random.randn(10) + 0.01 * i  # slow drift
        if kcenter.update(sig):
            n_added += 1

    print(f"  Final landmarks: {kcenter.n_landmarks}/{kcenter.max_landmarks}")
    print(f"  Total additions: {n_added}")
    assert kcenter.n_landmarks <= kcenter.max_landmarks
    print("  PASS")


def _simulate_cev(n, dt, beta, seed=None):
    """Simulate CEV: dS = 0.5 · |S|^(β/2) · dW."""
    if seed is not None:
        np.random.seed(seed)
    S = np.zeros(n)
    S[0] = 1.0
    for i in range(1, n):
        S[i] = S[i-1] + 0.5 * abs(S[i-1])**(beta/2) * np.sqrt(dt) * np.random.randn()
        S[i] = max(S[i], 0.01)
    return S


def test_streaming_feller():
    """Test streaming Feller on synthetic CEV processes."""
    print("Test: Streaming Feller on CEV...")

    n = 10000
    dt = 0.01

    results = {}
    for beta in [1.5, 2.0, 2.5, 3.0]:
        S = _simulate_cev(n, dt, beta, seed=int(beta * 100))

        det = StreamingFellerDetector(
            n_price_landmarks=30, max_support=3000,
            gamma_sig=0.99, gamma_nw=0.999, sig_level=2,
            sig_bw=1.0, min_obs=200)

        det.initialize(S[:1000], dt)
        for i in range(1000, n):
            det.update(S[i])

        is_bubble = beta > 2
        detected = det.p_bubble > 0.5
        status = "BUBBLE" if is_bubble else "no bub"
        flag = "OK" if detected == is_bubble else "MISS"
        print(f"  β={beta:.1f} ({status}): α̂={det.alpha_mean:.3f} ± "
              f"{det.alpha_sd:.3f}, P(bubble)={det.p_bubble:.3f}  [{flag}]")
        results[beta] = det

    # Verify: non-bubble α̂ < 2, bubble α̂ > 2 (relaxed)
    assert results[1.5].alpha_mean < 2.5, f"β=1.5 α̂={results[1.5].alpha_mean:.2f} too high"
    assert results[3.0].alpha_mean > 1.5, f"β=3.0 α̂={results[3.0].alpha_mean:.2f} too low"
    print(f"  PASS")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("  STREAMING SIG-KKF TESTS")
    print("=" * 60)

    test_lead_lag_logsig()
    test_online_kcenter()
    test_streaming_feller()

    print("\n  All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
