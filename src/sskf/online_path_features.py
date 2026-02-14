"""
Online Path Feature Extraction for High-Dimensional Systems

Three approaches for scalable, online-capable path distribution characterization:

1. Tensor Train Signatures (TT-Sig)
   - Compress d^L signature to L × d × r² via TT decomposition
   - Online updates via TT-cross

2. Random Projection + Nyström
   - Project log-signatures to fixed dimension
   - Nyström kernel approximation with landmarks
   - Trivially online

3. Signature Kernel (PDE)
   - Compute kernel directly without explicit features
   - Uses sigkernel library
   - Online via kernel evaluations with landmarks
"""

import numpy as np
from scipy.linalg import svd
import iisignature
from typing import List, Optional, Tuple, Callable


# =============================================================================
# 1. TENSOR TRAIN SIGNATURES
# =============================================================================

class TensorTrainSignature:
    """
    Tensor Train representation of path signatures.

    Instead of storing full signature tensor (d^L entries),
    store in TT format: G_1 × G_2 × ... × G_L
    Each G_k is (r_{k-1}, d, r_k) tensor
    Total storage: O(L × d × r²) instead of O(d^L)

    Supports online updates via Chen's identity:
    S(path_1 ⊕ path_2) = S(path_1) ⊗ S(path_2)

    Usage:
        # Option 1: Fit on initial batch, then stream
        tt = TensorTrainSignature(dim=2, initial_batch_size=50)
        for point in data[:50]:
            tt.add_point(point)  # Accumulates until batch full
        # After 50 points, signature is computed. Now stream:
        for point in data[50:]:
            tt.update_point(point)  # Uses Chen's identity

        # Option 2: Fit on full path, then update
        tt = TensorTrainSignature(dim=2)
        tt.fit(path)
        tt.update(new_segment)
    """

    def __init__(self, dim: int, depth: int = 3, max_rank: int = 10,
                 initial_batch_size: int = 1):
        """
        Args:
            dim: Path dimension (including time)
            depth: Signature truncation level
            max_rank: Maximum TT rank for compression
            initial_batch_size: Number of points to accumulate before first fit.
                               Set >1 for better initial signature estimate.
        """
        self.dim = dim
        self.depth = depth
        self.max_rank = max_rank
        self.initial_batch_size = initial_batch_size
        self.sig_levels = None  # List of signature levels (not full tensor)
        self.last_point = None  # Last point of path for continuity
        self._batch_buffer = []  # Buffer for initial batch accumulation

    def _sig_to_levels(self, sig: np.ndarray) -> List[np.ndarray]:
        """Split full signature into levels."""
        levels = []
        idx = 0
        for k in range(1, self.depth + 1):
            level_size = self.dim ** k
            levels.append(sig[idx:idx + level_size])
            idx += level_size
        return levels

    def _levels_to_features(self) -> np.ndarray:
        """Convert levels to fixed-size feature vector."""
        fixed_size = self.max_rank * self.dim * self.depth
        features = np.zeros(fixed_size)

        if self.sig_levels is None:
            return features

        idx = 0
        for level in self.sig_levels:
            n_keep = min(len(level), self.max_rank * self.dim)
            if idx + n_keep <= fixed_size:
                features[idx:idx + n_keep] = level[:n_keep]
                idx += self.max_rank * self.dim

        return features

    def _chen_multiply(self, sig_a_levels: List[np.ndarray],
                       sig_b_levels: List[np.ndarray]) -> List[np.ndarray]:
        """
        Chen's identity: S(a ⊕ b) = S(a) ⊗ S(b)

        For level k: S^k(a⊕b) = Σ_{i+j=k} S^i(a) ⊗ S^j(b)
        where S^0 = 1 (scalar).

        This is the shuffle product in the tensor algebra.
        """
        result_levels = []

        for k in range(1, self.depth + 1):
            level_k = np.zeros(self.dim ** k)

            for i in range(k + 1):
                j = k - i

                if i == 0:
                    # S^0(a) ⊗ S^k(b) = S^k(b)
                    if j <= len(sig_b_levels):
                        level_k += sig_b_levels[j - 1]
                elif j == 0:
                    # S^k(a) ⊗ S^0(b) = S^k(a)
                    if i <= len(sig_a_levels):
                        level_k += sig_a_levels[i - 1]
                else:
                    # Tensor product S^i(a) ⊗ S^j(b)
                    if i <= len(sig_a_levels) and j <= len(sig_b_levels):
                        s_i = sig_a_levels[i - 1]
                        s_j = sig_b_levels[j - 1]
                        # Outer product flattened
                        outer = np.outer(s_i, s_j).flatten()
                        # Truncate to level k size
                        level_k += outer[:self.dim ** k]

            result_levels.append(level_k)

        return result_levels

    def fit(self, path: np.ndarray) -> 'TensorTrainSignature':
        """Compute signature from path (initial fit)."""
        sig = iisignature.sig(path, self.depth)
        self.sig_levels = self._sig_to_levels(sig)
        self.last_point = path[-1].copy()
        return self

    def update(self, new_segment: np.ndarray) -> 'TensorTrainSignature':
        """
        Online update using Chen's identity: S(old ⊕ new) = S(old) ⊗ S(new)

        Args:
            new_segment: New path segment to append. Should start from
                        the last point of the previous path for continuity.

        Note: The new_segment should include overlap point for proper Chen application.
        """
        if self.sig_levels is None:
            return self.fit(new_segment)

        # Compute signature of new segment
        sig_new = iisignature.sig(new_segment, self.depth)
        new_levels = self._sig_to_levels(sig_new)

        # Apply Chen's identity
        self.sig_levels = self._chen_multiply(self.sig_levels, new_levels)
        self.last_point = new_segment[-1].copy()

        return self

    def add_point(self, point: np.ndarray) -> 'TensorTrainSignature':
        """
        Add a single point, accumulating until initial_batch_size is reached.

        After the batch is full, automatically fits the signature.
        Subsequent calls will use Chen's identity for updates.

        Args:
            point: Single point (dim,) array

        Returns:
            self for chaining
        """
        point = np.atleast_1d(point)

        if self.sig_levels is None:
            # Still in batch accumulation phase
            self._batch_buffer.append(point.copy())

            if len(self._batch_buffer) >= self.initial_batch_size:
                # Batch complete - fit signature
                path = np.array(self._batch_buffer)
                self.fit(path)
                self._batch_buffer = []  # Clear buffer
        else:
            # Already fitted - use Chen's identity
            self.update_point(point)

        return self

    def update_point(self, new_point: np.ndarray) -> 'TensorTrainSignature':
        """
        Update signature with a single new point using Chen's identity.

        This is O(d^depth) instead of O(N * d^depth) for full recompute.

        Note: If initial_batch_size > 1 and batch not yet full, this will
        add to the batch buffer instead. Use add_point() for automatic handling.
        """
        new_point = np.atleast_1d(new_point)

        if self.sig_levels is None:
            # Not yet fitted - add to batch if configured
            if self.initial_batch_size > 1:
                return self.add_point(new_point)
            # Initialize with single-point path
            self.last_point = new_point.copy()
            # Initialize with zero signature (identity element)
            self.sig_levels = [np.zeros(self.dim ** k) for k in range(1, self.depth + 1)]
            return self

        # Create minimal 2-point segment
        segment = np.vstack([self.last_point, new_point])

        # Update via Chen
        return self.update(segment)

    @property
    def is_fitted(self) -> bool:
        """Check if signature has been computed (batch complete)."""
        return self.sig_levels is not None

    @property
    def n_points_buffered(self) -> int:
        """Number of points in batch buffer (before first fit)."""
        return len(self._batch_buffer)

    def to_features(self) -> np.ndarray:
        """Extract fixed-size feature vector from signature levels."""
        return self._levels_to_features()

    @property
    def compression_ratio(self) -> float:
        """Compute compression ratio vs full signature."""
        full_size = sum(self.dim ** k for k in range(1, self.depth + 1))
        kept_size = self.max_rank * self.dim * self.depth
        return full_size / max(1, kept_size)


# =============================================================================
# 2. RANDOM PROJECTION + NYSTRÖM
# =============================================================================

class RandomProjectionNystrom:
    """
    Online path features via:
    1. Log-signature (compact representation)
    2. Lead-lag transform (captures QV)
    3. Random projection to fixed dimension
    4. Nyström kernel approximation with landmarks

    Fully online: new paths just need projection + kernel eval.

    Automatic independence detection:
    - If coordinates are independent, uses marginal signatures (more efficient)
    - If coordinates are correlated, uses joint log-signature (captures interactions)
    """

    def __init__(self,
                 dim: int,
                 depth: int = 3,
                 projection_dim: int = 200,
                 n_landmarks: int = 100,
                 use_leadlag: bool = True,
                 kernel_bandwidth: float = None,  # Auto-tune if None
                 normalize_features: bool = True,
                 feature_mode: str = 'auto',  # 'auto', 'joint', or 'marginal'
                 independence_threshold: float = 0.3,
                 seed: int = 42):
        """
        Args:
            dim: Path dimension (including time)
            depth: Log-signature depth
            projection_dim: Output dimension after random projection
            n_landmarks: Number of Nyström landmarks
            use_leadlag: Include lead-lag features
            kernel_bandwidth: RBF kernel bandwidth (auto-tune if None)
            normalize_features: Whether to normalize features (recommended)
            feature_mode: How to compute features:
                - 'auto': Detect independence and choose automatically
                - 'joint': Always use joint log-signature (d-dim path)
                - 'marginal': Always use marginal log-sigs (d separate 2D paths)
            independence_threshold: Threshold for independence detection (0-1).
                Lower = stricter (requires very independent coordinates).
                Default 0.3 works well for typical SDEs.
            seed: Random seed for projection matrix
        """
        self.dim = dim
        self.depth = depth
        self.projection_dim = projection_dim
        self.n_landmarks = n_landmarks
        self.use_leadlag = use_leadlag
        self.kernel_bandwidth = kernel_bandwidth
        self.normalize_features = normalize_features
        self.feature_mode = feature_mode
        self.independence_threshold = independence_threshold
        self.seed = seed

        # Independence detection state
        self._detected_mode = None  # 'joint' or 'marginal' after detection
        self._paths_for_detection = []  # Buffer paths for initial detection
        self._n_paths_for_detection = 10  # How many paths to use for detection

        # Will be initialized on first call
        self.W_logsig = None  # Random projection for log-sig
        self.W_leadlag = None  # Random projection for lead-lag
        self.landmarks = []  # Landmark features
        self.landmark_raw = []  # Raw paths for landmarks
        self.K_mm_inv = None  # Inverse of landmark kernel matrix

        # Feature normalization stats (running estimates)
        self._feature_mean = None
        self._feature_std = None
        self._n_features_seen = 0

        # Prepare log-sig computation
        self.logsig_prep = None
        self.leadlag_prep = None

    def _compute_logsig(self, path: np.ndarray) -> np.ndarray:
        """Compute log-signature of path."""
        if self.logsig_prep is None:
            self.logsig_prep = iisignature.prepare(path.shape[1], self.depth)
        return iisignature.logsig(path, self.logsig_prep)

    def _check_independence_cross_correlation(self, paths: List[np.ndarray]) -> bool:
        """
        Method 1: Check if coordinates are independent via cross-correlation of increments.

        If max |corr(dx_i, dx_j)| < threshold for all i != j, coordinates are independent.

        Returns True if independent (should use marginal sigs).
        """
        if len(paths) < 2:
            return False

        # Stack increments from all paths
        all_increments = []
        for path in paths:
            # Exclude time column (column 0)
            state_cols = path[:, 1:] if path.shape[1] > 1 else path
            dx = np.diff(state_cols, axis=0)
            all_increments.append(dx)

        increments = np.vstack(all_increments)  # (total_steps, state_dim)
        state_dim = increments.shape[1]

        if state_dim < 2:
            return False  # 1D, no cross-correlation to check

        # Compute cross-correlations
        max_cross_corr = 0.0
        for i in range(state_dim):
            for j in range(i + 1, state_dim):
                # Handle constant columns
                std_i = np.std(increments[:, i])
                std_j = np.std(increments[:, j])
                if std_i < 1e-10 or std_j < 1e-10:
                    continue
                corr = np.abs(np.corrcoef(increments[:, i], increments[:, j])[0, 1])
                max_cross_corr = max(max_cross_corr, corr)

        return max_cross_corr < self.independence_threshold

    def _check_independence_levy_area(self, paths: List[np.ndarray]) -> bool:
        """
        Method 2: Check if coordinates are independent via Lévy area magnitude.

        The Lévy area A_{ij} = integral (x_i dx_j - x_j dx_i) is ~0 for independent
        coordinates and non-zero for correlated ones.

        We check if |A_{ij}| / sqrt(QV_i * QV_j) < threshold.

        Returns True if independent (should use marginal sigs).
        """
        if len(paths) < 2:
            return False

        all_levy_ratios = []

        for path in paths:
            # Exclude time column
            state_cols = path[:, 1:] if path.shape[1] > 1 else path
            state_dim = state_cols.shape[1]

            if state_dim < 2:
                return False  # 1D

            # Compute increments
            dx = np.diff(state_cols, axis=0)
            x_mid = (state_cols[:-1] + state_cols[1:]) / 2

            for i in range(state_dim):
                for j in range(i + 1, state_dim):
                    # Lévy area: sum of (x_i * dx_j - x_j * dx_i)
                    levy_area = np.sum(x_mid[:, i] * dx[:, j] - x_mid[:, j] * dx[:, i])

                    # Normalize by sqrt of quadratic variations
                    qv_i = np.sum(dx[:, i]**2)
                    qv_j = np.sum(dx[:, j]**2)

                    if qv_i > 1e-10 and qv_j > 1e-10:
                        levy_ratio = np.abs(levy_area) / np.sqrt(qv_i * qv_j)
                        all_levy_ratios.append(levy_ratio)

        if not all_levy_ratios:
            return False

        # Average ratio across all paths and coordinate pairs
        avg_ratio = np.mean(all_levy_ratios)
        return avg_ratio < self.independence_threshold

    def _detect_independence(self, paths: List[np.ndarray]) -> str:
        """
        Detect whether coordinates are independent using both methods.

        Uses majority vote: if both methods agree on independence, use marginal.
        Otherwise use joint (safer default).

        Returns 'joint' or 'marginal'.
        """
        cross_corr_indep = self._check_independence_cross_correlation(paths)
        levy_area_indep = self._check_independence_levy_area(paths)

        # Both methods must agree for marginal
        if cross_corr_indep and levy_area_indep:
            return 'marginal'
        return 'joint'

    def _compute_marginal_logsigs(self, path: np.ndarray) -> np.ndarray:
        """
        Compute marginal log-signatures for each coordinate separately.

        For d-dimensional state, computes d separate 2D log-sigs: (t, x_i) for each i.
        Then concatenates them.

        This is more efficient than joint log-sig for independent coordinates and
        avoids the curse of dimensionality.
        """
        state_cols = path[:, 1:] if path.shape[1] > 1 else path
        time_col = path[:, 0] if path.shape[1] > 1 else np.arange(len(path))
        state_dim = state_cols.shape[1]

        marginal_logsigs = []
        prep_2d = iisignature.prepare(2, self.depth)

        for i in range(state_dim):
            # Create 2D path (t, x_i)
            path_2d = np.column_stack([time_col, state_cols[:, i]])
            logsig_i = iisignature.logsig(path_2d, prep_2d)
            marginal_logsigs.append(logsig_i)

        return np.concatenate(marginal_logsigs)

    def _lead_lag_transform(self, path: np.ndarray) -> np.ndarray:
        """Lead-lag embedding that captures quadratic variation."""
        n, d = path.shape
        ll_path = np.zeros((2*n - 1, 2*d))

        for i in range(n):
            if i > 0:
                ll_path[2*i - 1, :d] = path[i-1]
                ll_path[2*i - 1, d:] = path[i]
            ll_path[2*i, :d] = path[i]
            ll_path[2*i, d:] = path[i]

        return ll_path

    def _get_effective_mode(self) -> str:
        """Get the effective feature mode (after auto-detection if needed)."""
        if self.feature_mode != 'auto':
            return self.feature_mode
        if self._detected_mode is not None:
            return self._detected_mode
        # Default to joint until we have enough paths to detect
        return 'joint'

    def _compute_features(self, path: np.ndarray) -> np.ndarray:
        """Compute full feature vector for a path."""
        features = []

        # Determine which mode to use
        mode = self._get_effective_mode()

        # 1. Log-signature (compact) - joint or marginal
        if mode == 'marginal' and path.shape[1] > 2:  # More than (t, x)
            logsig = self._compute_marginal_logsigs(path)
        else:
            logsig = self._compute_logsig(path)
        features.append(logsig)

        # 2. Lead-lag log-signature (QV info) - only for joint mode
        if self.use_leadlag and mode == 'joint':
            ll_path = self._lead_lag_transform(path)
            if self.leadlag_prep is None:
                self.leadlag_prep = iisignature.prepare(ll_path.shape[1], min(2, self.depth))
            ll_logsig = iisignature.logsig(ll_path, self.leadlag_prep)
            features.append(ll_logsig)

        # 3. Basic statistics for scale
        dx = np.diff(path[:, 1:], axis=0)  # Exclude time column
        stats = np.array([
            np.mean(path[:, 1:]),
            np.std(path[:, 1:]),
            np.mean(dx**2),  # QV
        ]).flatten()
        features.append(stats)

        return np.concatenate(features)

    def _update_normalization_stats(self, features: np.ndarray):
        """Update running mean/std for feature normalization."""
        self._n_features_seen += 1
        if self._feature_mean is None:
            self._feature_mean = features.copy()
            self._feature_std = np.ones_like(features)
        else:
            # Welford's online algorithm
            delta = features - self._feature_mean
            self._feature_mean += delta / self._n_features_seen
            if self._n_features_seen > 1:
                # Update variance estimate
                delta2 = features - self._feature_mean
                self._feature_std = np.sqrt(
                    ((self._n_features_seen - 2) * self._feature_std**2 +
                     delta * delta2) / (self._n_features_seen - 1)
                )
        # Prevent division by zero
        self._feature_std = np.maximum(self._feature_std, 1e-6)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        # Don't normalize if disabled, no stats yet, or only one sample seen
        # (normalizing against a single sample makes all features 0)
        if not self.normalize_features or self._feature_mean is None or self._n_features_seen < 2:
            return features
        return (features - self._feature_mean) / self._feature_std

    def _project(self, raw_features: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """Apply normalization and random projection."""
        np.random.seed(self.seed)

        # Update normalization stats if requested
        if update_stats:
            self._update_normalization_stats(raw_features)

        # Normalize
        normalized = self._normalize(raw_features)

        if self.W_logsig is None:
            # Initialize projection matrix (Gaussian random)
            d = len(normalized)
            self.W_logsig = np.random.randn(self.projection_dim, d) / np.sqrt(self.projection_dim)

        return self.W_logsig @ normalized

    def transform(self, path: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """Transform path to projected features."""
        raw = self._compute_features(path)
        return self._project(raw, update_stats=update_stats)

    def add_landmark(self, path: np.ndarray):
        """Add a new landmark path."""
        # Buffer paths for auto-detection (don't compute features yet!)
        if self.feature_mode == 'auto' and self._detected_mode is None:
            self._paths_for_detection.append(path.copy())

            # Trigger detection once we have enough paths
            if len(self._paths_for_detection) >= self._n_paths_for_detection:
                self._detected_mode = self._detect_independence(self._paths_for_detection)

                # Reset projection matrices (may have been initialized with wrong dim
                # during nystrom_embedding calls before detection completed)
                self.W_logsig = None
                self.W_leadlag = None
                self._feature_mean = None
                self._feature_std = None
                self._n_features_seen = 0

                # Now compute features for all buffered paths with detected mode
                for p in self._paths_for_detection:
                    features = self.transform(p, update_stats=True)
                    self.landmarks.append(features)
                    self.landmark_raw.append(p)

                # Clear buffer
                self._paths_for_detection = []
                self.K_mm_inv = None

            return  # Don't add to landmarks yet during detection phase

        # Normal case: detection complete or not using auto mode
        features = self.transform(path, update_stats=True)
        self.landmarks.append(features)
        self.landmark_raw.append(path.copy())

        # Invalidate cached inverse
        self.K_mm_inv = None

        # Auto-tune bandwidth after seeing some landmarks
        if self.kernel_bandwidth is None and len(self.landmarks) >= 5:
            self._auto_tune_bandwidth()

        # Keep only n_landmarks most recent
        if len(self.landmarks) > self.n_landmarks:
            self.landmarks = self.landmarks[-self.n_landmarks:]
            self.landmark_raw = self.landmark_raw[-self.n_landmarks:]

    def _auto_tune_bandwidth(self):
        """Auto-tune kernel bandwidth using median heuristic."""
        if len(self.landmarks) < 2:
            self.kernel_bandwidth = 1.0
            return

        landmarks_arr = np.array(self.landmarks)
        # Compute pairwise distances
        n = len(landmarks_arr)
        dists = []
        for i in range(min(n, 20)):  # Sample for efficiency
            for j in range(i+1, min(n, 20)):
                dists.append(np.linalg.norm(landmarks_arr[i] - landmarks_arr[j]))

        if dists:
            # Median heuristic
            self.kernel_bandwidth = np.median(dists) + 1e-6
        else:
            self.kernel_bandwidth = 1.0

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel between feature vectors."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        # Default bandwidth if not set
        bandwidth = self.kernel_bandwidth if self.kernel_bandwidth is not None else 1.0
        return np.exp(-sq_dists / (2 * bandwidth**2))

    def kernel_with_landmarks(self, path: np.ndarray) -> np.ndarray:
        """Compute kernel between path and all landmarks."""
        if len(self.landmarks) == 0:
            return np.array([])

        features = self.transform(path)
        landmarks_arr = np.array(self.landmarks)
        return self._rbf_kernel(features.reshape(1, -1), landmarks_arr).flatten()

    def nystrom_embedding(self, path: np.ndarray) -> np.ndarray:
        """
        Nyström feature embedding with FIXED output size.
        phi(x) = K_mm^{-1/2} @ K(landmarks, x)

        Always returns n_landmarks-dimensional vector.
        """
        # Fixed size output
        embedding = np.zeros(self.n_landmarks)

        if len(self.landmarks) < 2:
            # Not enough landmarks - use projected features padded
            proj = self.transform(path)
            n = min(len(proj), self.n_landmarks)
            embedding[:n] = proj[:n]
            return embedding

        # Compute K_mm inverse if needed
        if self.K_mm_inv is None:
            landmarks_arr = np.array(self.landmarks)
            K_mm = self._rbf_kernel(landmarks_arr, landmarks_arr)
            # Regularized inverse
            K_mm += 1e-6 * np.eye(len(self.landmarks))
            self.K_mm_inv = np.linalg.inv(K_mm)

        # Kernel with landmarks
        k_xm = self.kernel_with_landmarks(path)

        # Nyström embedding
        nystrom_emb = self.K_mm_inv @ k_xm
        n = len(nystrom_emb)
        embedding[:n] = nystrom_emb

        return embedding

    def update(self, path: np.ndarray, add_as_landmark: bool = True):
        """
        Online update with new path.
        Optionally add as landmark (for diverse coverage).
        """
        if add_as_landmark and len(self.landmarks) < self.n_landmarks:
            self.add_landmark(path)
        elif add_as_landmark:
            # Replace oldest landmark
            self.add_landmark(path)

    @property
    def detected_mode(self) -> Optional[str]:
        """
        The detected feature mode (after auto-detection).

        Returns None if still in detection phase, otherwise 'joint' or 'marginal'.
        """
        if self.feature_mode != 'auto':
            return self.feature_mode
        return self._detected_mode

    @property
    def effective_mode(self) -> str:
        """
        The current effective feature mode being used.

        Returns 'joint' or 'marginal' based on feature_mode setting and detection.
        """
        return self._get_effective_mode()


# =============================================================================
# 3. SIGNATURE KERNEL (PDE)
# =============================================================================

class SignatureKernelPDE:
    """
    Signature kernel via PDE solver (sigkernel library).

    Computes K(path1, path2) = <S(path1), S(path2)>_RKHS directly
    without explicit signature computation.

    O(1) in signature dimension!
    Online via kernel evaluations with landmarks.
    """

    def __init__(self,
                 n_landmarks: int = 100,
                 static_kernel: str = 'rbf',
                 static_bandwidth: float = 1.0,
                 dyadic_order: int = 2):
        """
        Args:
            n_landmarks: Number of landmark paths
            static_kernel: Base kernel ('rbf' or 'linear')
            static_bandwidth: Bandwidth for RBF static kernel
            dyadic_order: PDE discretization order (higher = more accurate)
        """
        self.n_landmarks = n_landmarks
        self.static_kernel_type = static_kernel
        self.static_bandwidth = static_bandwidth
        self.dyadic_order = dyadic_order

        self.landmarks = []  # Landmark paths (as numpy arrays)
        self.K_mm = None  # Landmark kernel matrix
        self.K_mm_inv = None

        # Try to import sigkernel
        self.sigkernel = None
        self.torch = None
        self.sig_kernel = None
        self._init_sigkernel()

    def _init_sigkernel(self):
        """Initialize sigkernel if available."""
        try:
            import torch
            import sigkernel

            self.torch = torch
            self.sigkernel = sigkernel

            if self.static_kernel_type == 'rbf':
                static_kernel = sigkernel.RBFKernel(sigma=self.static_bandwidth)
            else:
                static_kernel = sigkernel.LinearKernel()

            self.sig_kernel = sigkernel.SigKernel(
                static_kernel,
                dyadic_order=self.dyadic_order
            )

        except ImportError:
            print("Warning: sigkernel not available. Using truncated signature fallback.")
            self.sigkernel = None

    def _path_to_tensor(self, path: np.ndarray):
        """Convert numpy path to torch tensor."""
        if self.torch is None:
            raise ImportError("PyTorch required for signature kernel")
        # sigkernel expects float64 (double precision)
        return self.torch.tensor(path, dtype=self.torch.float64).unsqueeze(0)

    def kernel(self, path1: np.ndarray, path2: np.ndarray) -> float:
        """Compute signature kernel between two paths."""
        if self.sig_kernel is None:
            # Fallback: truncated signature inner product
            s1 = iisignature.sig(path1, 4)
            s2 = iisignature.sig(path2, 4)
            return np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-10)

        p1 = self._path_to_tensor(path1)
        p2 = self._path_to_tensor(path2)

        K = self.sig_kernel.compute_kernel(p1, p2)
        return K.item()

    def add_landmark(self, path: np.ndarray):
        """Add landmark path."""
        self.landmarks.append(path.copy())

        # Invalidate cached matrices
        self.K_mm = None
        self.K_mm_inv = None

        # Keep only n_landmarks
        if len(self.landmarks) > self.n_landmarks:
            self.landmarks = self.landmarks[-self.n_landmarks:]

    def _compute_landmark_kernel(self):
        """Compute kernel matrix between landmarks."""
        m = len(self.landmarks)
        self.K_mm = np.zeros((m, m))

        for i in range(m):
            for j in range(i, m):
                k_ij = self.kernel(self.landmarks[i], self.landmarks[j])
                self.K_mm[i, j] = k_ij
                self.K_mm[j, i] = k_ij

        # Regularized inverse
        self.K_mm_inv = np.linalg.inv(self.K_mm + 1e-6 * np.eye(m))

    def kernel_with_landmarks(self, path: np.ndarray) -> np.ndarray:
        """Compute kernel between path and all landmarks."""
        return np.array([self.kernel(path, lm) for lm in self.landmarks])

    def nystrom_embedding(self, path: np.ndarray) -> np.ndarray:
        """Nyström embedding using landmark kernel with fixed output size."""
        embedding = np.zeros(self.n_landmarks)

        if len(self.landmarks) < 2:
            return embedding

        if self.K_mm_inv is None:
            self._compute_landmark_kernel()

        k_xm = self.kernel_with_landmarks(path)
        nystrom_emb = self.K_mm_inv @ k_xm
        n = len(nystrom_emb)
        embedding[:n] = nystrom_emb

        return embedding

    def update(self, path: np.ndarray, add_as_landmark: bool = True):
        """Online update with new path."""
        if add_as_landmark and len(self.landmarks) < self.n_landmarks:
            self.add_landmark(path)


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

class OnlinePathMMD:
    """
    Unified interface for online path distribution comparison.

    Supports all three backends:
    - 'tt': Tensor Train signatures
    - 'rp': Random Projection + Nyström
    - 'pde': Signature Kernel via PDE
    """

    def __init__(self,
                 dim: int,
                 method: str = 'rp',
                 n_landmarks: int = 100,
                 **kwargs):
        """
        Args:
            dim: Path dimension (including time)
            method: 'tt', 'rp', or 'pde'
            n_landmarks: Number of landmarks for MMD
            **kwargs: Method-specific arguments
        """
        self.dim = dim
        self.method = method
        self.n_landmarks = n_landmarks

        if method == 'tt':
            self.extractor = TensorTrainSignature(
                dim=dim,
                depth=kwargs.get('depth', 3),
                max_rank=kwargs.get('max_rank', 10)
            )
        elif method == 'rp':
            self.extractor = RandomProjectionNystrom(
                dim=dim,
                depth=kwargs.get('depth', 3),
                projection_dim=kwargs.get('projection_dim', 200),
                n_landmarks=n_landmarks,
                use_leadlag=kwargs.get('use_leadlag', True)
            )
        elif method == 'pde':
            self.extractor = SignatureKernelPDE(
                n_landmarks=n_landmarks,
                static_bandwidth=kwargs.get('static_bandwidth', 1.0),
                dyadic_order=kwargs.get('dyadic_order', 2)
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.reference_embeddings = []
        self.test_embeddings = []

    def add_reference_path(self, path: np.ndarray):
        """Add path to reference distribution."""
        if self.method == 'tt':
            tt = TensorTrainSignature(self.dim)
            tt.fit(path)
            self.reference_embeddings.append(tt.to_features())
        elif self.method == 'rp':
            self.extractor.update(path, add_as_landmark=True)
            emb = self.extractor.nystrom_embedding(path)
            self.reference_embeddings.append(emb)
        elif self.method == 'pde':
            self.extractor.update(path, add_as_landmark=True)
            emb = self.extractor.nystrom_embedding(path)
            self.reference_embeddings.append(emb)

    def add_test_path(self, path: np.ndarray):
        """Add path to test distribution."""
        if self.method == 'tt':
            tt = TensorTrainSignature(self.dim)
            tt.fit(path)
            self.test_embeddings.append(tt.to_features())
        elif self.method == 'rp':
            emb = self.extractor.nystrom_embedding(path)
            self.test_embeddings.append(emb)
        elif self.method == 'pde':
            emb = self.extractor.nystrom_embedding(path)
            self.test_embeddings.append(emb)

    def compute_mmd(self) -> float:
        """Compute MMD between reference and test distributions."""
        if len(self.reference_embeddings) < 2 or len(self.test_embeddings) < 2:
            return float('inf')

        X = np.array(self.reference_embeddings)
        Y = np.array(self.test_embeddings)

        # Linear kernel MMD (embeddings already in RKHS)
        n, m = len(X), len(Y)

        K_xx = X @ X.T
        K_yy = Y @ Y.T
        K_xy = X @ Y.T

        mmd2 = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n-1)) + \
               (np.sum(K_yy) - np.trace(K_yy)) / (m * (m-1)) - \
               2 * np.mean(K_xy)

        return np.sqrt(max(0, mmd2))

    def reset_test(self):
        """Reset test distribution (keep reference)."""
        self.test_embeddings = []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def subsample_path(path: np.ndarray, n_points: int = 50) -> np.ndarray:
    """Subsample path to fixed number of points."""
    n = len(path)
    if n <= n_points:
        return path
    indices = np.linspace(0, n-1, n_points, dtype=int)
    return path[indices]


def add_time_column(traj: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Add time column to trajectory."""
    if traj.ndim == 1:
        traj = traj.reshape(-1, 1)
    n = len(traj)
    t = np.arange(n) * dt
    return np.column_stack([t, traj])
