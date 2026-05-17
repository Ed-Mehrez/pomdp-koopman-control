import numpy as np

def compute_path_signature(path_history: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Computes signature terms of a path.
    Level 2 is vectorized with NumPy; higher levels delegate to iisignature.
    
    Args:
        path_history: Shape (T, D) array where T is time and D is dimension.
        level: Signature truncation level.
    
    Returns:
        concatenated signature features (flat array).
    """
    if level > 2:
        import iisignature

        return iisignature.sig(path_history, level)

    # 1. Increments: dX_t = X_{t+1} - X_t
    dX = np.diff(path_history, axis=0) # Shape (T-1, D)
    
    # 2. Level 1: Total Increment (Path displacement)
    # S^1 = \int dX = X_T - X_0
    sig1 = np.sum(dX, axis=0)
    
    if level == 1:
        return sig1
        
    # 3. Level 2: Iterated Integrals
    # Efficient: S^{ij} = \sum_l dX_l^j * (\sum_{k<l} dX_k^i)
    # The term (\sum_{k<l} dX_k^i) is just the cumulative sum of dX (the path position relative to start)
    
    # Check D
    T_steps, D = dX.shape
    
    path_centered = np.cumsum(dX, axis=0) # X_l - X_l_initial
    # Shift cumsum to be exclusive for the inner integral X_k (Left point)
    # cumsum[k] is sum(0..k). We want sum(0..k-1).
    # Prepend zeros.
    path_integral = np.vstack([np.zeros(D), path_centered[:-1]])
    
    # Exact piecewise-linear level 2:
    # cross-segment terms plus same-segment 0.5 * dx_i * dx_j terms.
    sig2_matrix = path_integral.T @ dX + 0.5 * (dX.T @ dX)
    
    sig2_flat = sig2_matrix.flatten()
    
    return np.concatenate([sig1, sig2_flat])

def compute_log_signature(path: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Computes the Log-Signature (Intrinsic Geometry).
    For Level 2, this consists of:
    1. Level 1 Terms (Displacement)
    2. Level 2 Skew-Symmetric Terms (Levi Area): A_ij = (S_ij - S_ji) / 2
    
    Args:
        path: (T, d) array
        level: 2
        
    Returns:
        log_sig: flattened vector
    """
    if level > 2:
        import iisignature

        prep = iisignature.prepare(path.shape[1], level)
        return iisignature.logsig(path, prep)

    # 1. Compute Standard Signature Matrix S_ij
    dX = np.diff(path, axis=0)
    sig1 = np.sum(dX, axis=0)
    
    if level == 1:
        return sig1
        
    T_steps, D = dX.shape
    
    path_centered = np.cumsum(dX, axis=0)
    path_integral = np.vstack([np.zeros(D), path_centered[:-1]])
    
    sig2_matrix = path_integral.T @ dX
    
    # 2. Compute Levi Area (Skew Symmetric Part)
    # A_ij = 0.5 * (S_ij - S_ji) for i < j
    levi_area = []
    
    for i in range(D):
        for j in range(i + 1, D): # Strictly upper triangle
            area = 0.5 * (sig2_matrix[i, j] - sig2_matrix[j, i])
            levi_area.append(area)
            
    return np.concatenate([sig1, np.array(levi_area)])

def compute_signature_level_3(path: np.ndarray) -> np.ndarray:
    """
    Computes Full Signature up to Level 3.
    Use this for high-fidelity tasks where Level 2 Log-Sig is insufficient.
    Dim = d + d^2 + d^3.
    """
    import iisignature

    return iisignature.sig(path, 3)

def get_augmented_state_with_signatures(env_obs, history_buffer, use_log_signatures=False):
    """
    Constructs the augmented state vector [state, time, signatures].
    
    Args:
        env_obs: Current environment observation [x, x_dot, theta, theta_dot]
        history_buffer: List of past observations (raw env obs)
        use_log_signatures: If True, computes Log-Signatures (Levi Area)
    
    Returns:
        z: Augmented state vector [x, x_dot, cos, sin, theta_dot, sig...]
    """
    # 1. Embed current state to manifold [x, x_dot, cos, sin, theta_dot]
    # We do this because the "State" part of augmented vector should be geometric
    obs_flat = env_obs.flatten()
    state_embedded = np.array([
        obs_flat[0], 
        obs_flat[1], 
        np.cos(obs_flat[2]), 
        np.sin(obs_flat[2]), 
        obs_flat[3]
    ])
    
    # 2. Compute Signatures from History
    # We want signatures of the EMBEDDED path, not the raw path (to avoid periodic jumps in theta)
    if len(history_buffer) < 2:
        d = len(state_embedded) + 1 # +1 for Time Augmentation
        if use_log_signatures:
             dim_sig = d + (d * (d-1)) // 2
        else:
             dim_sig = d + d*d
        sigs = np.zeros(dim_sig)
    else:
        # Embed entire history
        path_raw = np.array(history_buffer) # (T, 4)
        
        # TIME AUGMENTATION (Critical for distinguishing equilibria)
        # Without time, S(X) is translation invariant and S(const) = 0.
        # With time, S(t, X) contains \int X dt, which captures the state value.
        t_seq = np.linspace(0, 1, len(history_buffer))
        
        path = np.column_stack([
            t_seq, # Time-augmented channel
            path_raw[:, 0],
            path_raw[:, 1],
            np.cos(path_raw[:, 2]),
            np.sin(path_raw[:, 2]),
            path_raw[:, 3]
        ])
        
        if use_log_signatures:
            sigs = compute_log_signature(path, level=2)
        else:
            sigs = compute_path_signature(path, level=2)
            
    # 3. Augment
    z = np.hstack([state_embedded, sigs])
    return z

def compute_lead_lag_path(path_1d: np.ndarray) -> np.ndarray:
    """
    Transforms 1D path into 2D Lead-Lag path.
    X_lead(t) = X(t)
    X_lag(t) = X(t-dt)
    
    This embedding is CRITICAL for capturing Quadratic Variation (Volatility)
    using Signatures, as standard 1D signatures only capture increments (Chen's Identity).
    The Lévy area between Lead and Lag is one half of quadratic variation:
    2 * Area(lead, lag) = sum of squared increments.
    """
    # Simply interleave points
    # Generic construction for path X = (x1, x2, x3...)
    # Lead-Lag Path is piecewise linear in 2D:
    # (x1, x1) -> (x2, x1) -> (x2, x2) -> (x3, x2) -> (x3, x3)...
    
    x = path_1d.flatten()
    n = len(x)
    
    # We construct 2 * n - 1 points
    lead = np.repeat(x, 2)[1:]  # x1, x2, x2, x3, x3...
    lag  = np.repeat(x, 2)[:-1] # x1, x1, x2, x2, x3...
    
    return np.column_stack([lead, lag])

def compute_lead_lag_signature(path_1d: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Computes signature of the Lead-Lag embedded path.
    Captures Quadratic Variation explicitly in the Level 2 terms.
    """
    ll_path = compute_lead_lag_path(path_1d)
    return compute_log_signature(ll_path, level=level)

class RecurrentLeadLagLogSigMap:
    """
    Recurrent Lead-Lag Log-Signature with BCH updates.

    For a d-dimensional input stream, applies lead-lag embedding to get a
    2d-dimensional path, then maintains the log-signature via BCH.

    Each input increment dx produces TWO BCH updates:
      1. (dx, 0) — lead moves, lag stays
      2. (0, dx) — lag catches up to lead

    The Levy area between lead_i and lag_i captures quadratic variation
    of channel i: 2 * Area(lead_i, lag_i) = sum of dx_i^2.

    At level 2 for input dim d:
      - Lead-lag dim: 2d
      - Log-sig level 1: 2d components
      - Log-sig level 2: 2d*(2d-1)/2 components (Levy areas)
      - Total features: 2d + d*(2d-1)

    For d=1 (returns only): 2 + 1 = 3 features (displacement + QV)
    For d=2 (time+return): 4 + 6 = 10 features
    """
    def __init__(self, state_dim, level=2, forgetting_factor=1.0):
        self.d_input = state_dim
        self.d = 2 * state_dim  # lead-lag doubles dimensions
        self.level = level
        self.gamma = forgetting_factor

        # Log-sig dimensions in lead-lag space
        self.dim_l1 = self.d
        self.dim_l2 = self.d * (self.d - 1) // 2
        self.feature_dim = self.dim_l1 + self.dim_l2

        self.reset()

    def reset(self):
        self.l1 = np.zeros(self.dim_l1)
        self.l2 = np.zeros(self.dim_l2)

    def _bch_update(self, dx_ll):
        """Single BCH update in lead-lag space."""
        a1 = self.gamma * self.l1
        a2 = self.gamma ** 2 * self.l2

        # Lie bracket [a_decayed, dx_ll] at level 2
        bracket = np.zeros(self.dim_l2)
        idx = 0
        for i in range(self.d):
            for j in range(i + 1, self.d):
                bracket[idx] = a1[i] * dx_ll[j] - a1[j] * dx_ll[i]
                idx += 1

        self.l1 = a1 + dx_ll
        self.l2 = a2 + 0.5 * bracket

    def update(self, dx):
        """
        Lead-lag update: each input increment dx produces two BCH steps.

        Step 1: lead moves by dx, lag stays: dx_ll = (dx[0], dx[1], ..., 0, 0, ...)
        Step 2: lag catches up:              dx_ll = (0, 0, ..., dx[0], dx[1], ...)

        The Levy area between lead_i and lag_i accumulates 0.5 * dx_i^2.
        """
        # Step 1: lead moves
        dx_lead = np.zeros(self.d)
        dx_lead[:self.d_input] = dx
        self._bch_update(dx_lead)

        # Step 2: lag catches up
        dx_lag = np.zeros(self.d)
        dx_lag[self.d_input:] = dx
        self._bch_update(dx_lag)

        return self.get_features()

    def get_features(self):
        return np.concatenate([self.l1, self.l2])


class RecurrentLogSignatureMap:
    """
    Recurrent Log-Signature with BCH updates and optional exponential decay.

    Works in the log-signature (Lie algebra) space. At level 2 for d dimensions:
      - Level 1: d components (displacement)
      - Level 2: d*(d-1)/2 components (Levy area / antisymmetric part)
      - Total: d + d*(d-1)/2 features (3 for d=2, vs 6 for full sig)

    BCH formula (exact at level 2):
      log(exp(a) * exp(b)) = a + b + [a,b]/2

    With decay gamma:
      a_decayed = gamma * a_old
      logsig_new = a_decayed + dx + [a_decayed, dx]/2

    The log-signature is more compact (no symmetric part) and the Levy area
    captures the essential path geometry (signed area = path ordering).
    """
    def __init__(self, state_dim, level=2, forgetting_factor=1.0):
        self.d = state_dim
        self.level = level
        self.gamma = forgetting_factor

        # Dimensions
        self.dim_l1 = self.d
        self.dim_l2 = self.d * (self.d - 1) // 2  # antisymmetric only
        self.feature_dim = self.dim_l1 + self.dim_l2

        self.reset()

    def reset(self):
        self.l1 = np.zeros(self.dim_l1)
        self.l2 = np.zeros(self.dim_l2)

    def update(self, dx):
        """
        BCH update: logsig_new = gamma*logsig_old + dx + [gamma*logsig_old, dx]/2

        For d=2: [a, b]_area = a[0]*b[1] - a[1]*b[0]  (single component)
        For general d: [a, b]_{ij} = a[i]*b[j] - a[j]*b[i] for i<j
        """
        # Decay previous state
        a1 = self.gamma * self.l1
        a2 = self.gamma ** 2 * self.l2

        # Lie bracket [a_decayed, dx] at level 2
        # For each pair (i,j) with i<j: bracket_{ij} = a1[i]*dx[j] - a1[j]*dx[i]
        bracket = np.zeros(self.dim_l2)
        idx = 0
        for i in range(self.d):
            for j in range(i + 1, self.d):
                bracket[idx] = a1[i] * dx[j] - a1[j] * dx[i]
                idx += 1

        # BCH: level 1 just adds, level 2 adds + bracket/2
        self.l1 = a1 + dx
        self.l2 = a2 + 0.5 * bracket

        return self.get_features()

    def get_features(self):
        return np.concatenate([self.l1, self.l2])


class RecurrentSignatureMap:
    """
    Implements Recurrent Signatures (Chen's Identity) for infinite memory.
    S_t = S_{t-1} (otimes) exp(dX_t)
    
    This allows the controller to track 'Global Context' (e.g. Energy, Regime)
    without a finite sliding window.
    """
    def __init__(self, state_dim, level=2, forgetting_factor=1.0):
        self.d = state_dim
        self.level = level
        self.gamma = forgetting_factor
        
        # Dimensions
        # Level 0: 1 (Scalar) - usually omitted in feature vector but needed for algebra
        # Level 1: d
        # Level 2: d^2
        self.dim_l1 = self.d
        self.dim_l2 = self.d * self.d
        self.feature_dim = self.dim_l1 + self.dim_l2
        
        self.reset()
        
    def reset(self):
        # S_0 = (1, 0, 0...)
        self.s1 = np.zeros(self.dim_l1)
        self.s2 = np.zeros(self.dim_l2) # Flattened d x d
        
    def update(self, dx):
        """
        Update signature state with increment dx.
        Chen's Identity (Truncated to Level 2):
        
        S_new = S_old (otimes) exp(dx)
        
        Let S = (1, S1, S2)
        exp(dx) = (1, dx, dx^2/2)
        
        S_new_1 = S1 + dx
        S_new_2 = S2 + S1 (otimes) dx + dx^2/2
        
        With Forgetting Factor gamma:
        S1_t = gamma * S1_{t-1} + dx
        S2_t = gamma^2 * S2_{t-1} + gamma * (S1_{t-1} (otimes) dx) + dx^2/2
        """
        # Level 1 Update
        s1_prev = self.s1.copy()
        
        self.s1 = self.gamma * self.s1 + dx
        
        # Level 2 Update
        # Term 1: Cross product S1_prev (otimes) dx
        # Outer product flattened
        cross_term = np.outer(s1_prev, dx).flatten()
        
        # Term 2: dx (otimes) dx / 2
        quad_term = 0.5 * np.outer(dx, dx).flatten()
        
        self.s2 = (self.gamma**2 * self.s2) + (self.gamma * cross_term) + quad_term
        
        return self.get_features()
        
    def get_features(self):
        return np.concatenate([self.s1, self.s2])
