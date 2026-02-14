import numpy as np

def compute_path_signature(path_history: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Computes the Level-1 and Level-2 Signature terms of a path.
    Implementation is fully vectorized using NumPy (no esig dependency).
    
    Args:
        path_history: Shape (T, D) array where T is time and D is dimension.
        level: Only degree=2 is fully implemented.
    
    Returns:
        concatenated signature features (flat array).
    """
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
    
    # S2[i, j] = sum_t path_integral[t, i] * dX[t, j]
    sig2_matrix = path_integral.T @ dX
    
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
    dX = np.diff(path, axis=0) # (T-1, D)
    T_steps, D = dX.shape
    
    # 1. Level 1 (increments)
    sig1 = np.sum(dX, axis=0) # (D,)
    
    # 2. Level 2 (Iterated Integrals)
    # S(2)_ij = integral dXi dXj
    # Efficient: S(2) = (cumsum dX) . dX
    path_centered = np.cumsum(dX, axis=0)
    # Shift: integral up to t-1 * dX_t
    aug_path = np.vstack([np.zeros(D), path_centered[:-1]])
    sig2_mat = aug_path.T @ dX # (D, D)
    sig2 = sig2_mat.flatten()
    
    # 3. Level 3
    # S(3)_ijk = integral S(2)_ij dXk
    # We need the running Level 2 signature at every step
    # Running Sig2(t)_ij = sum_{s < t} X_s_i * dX_s_j ? No.
    # Recursive: S(3)_ijk = sum_t RunningSig2(t)_ij * dX_t_k
    
    # Calculate Running Sig2
    # running_sig2[t, i, j]
    running_sig2 = np.zeros((T_steps, D, D))
    current_sig2 = np.zeros((D, D))
    # We need cumulative sum of (path_integral(t) outer dX(t))
    
    # Vectorized:
    # term_t_ij = aug_path[t]_i * dX[t]_j
    # running_sig2[t] = sum_{0 to t-1} term_s_ij ...
    # Wait, the definition is integral of path_level_{k-1} against dX.
    
    # Let's do loop for clarity on Level 3
    sig3 = np.zeros((D, D, D)) 
    
    # Construct running level 2 path
    # P2[t] is the Level 2 signature of the path up to time t
    # P2[t] = P2[t-1] + (P1[t-1] \otimes dX[t]) + 0.5 dX[t] \otimes dX[t] (Stratonovich)
    # Using Itô/Riemann sum here for simplicity:
    # S(t) = S(t-1) \otimes (1 + dX_t)
    # S(t)^2 = S(t-1)^2 + S(t-1)^1 \otimes dX_t + dX \otimes dX / 2
    
    # Simplified Iterated Sum (Chen):
    # S_ijk = sum_{t1 < t2 < t3} dX_t1_i dX_t2_j dX_t3_k
    
    # Efficient:
    # CumSum1 = cumsum(dX)
    # CumSum2 = cumsum(CumSum1 * dX) ... No, outer product.
    
    # Algo:
    # 1. R1 = CumSum(dX) 
    # 2. R2 = CumSum(R1[t-1] \otimes dX[t])
    # 3. R3 = CumSum(R2[t-1] \otimes dX[t])
    # Sig = R_terminal
    
    # Let's do this:
    R1 = np.vstack([np.zeros(D), np.cumsum(dX, axis=0)]) # (T, D)
    
    # Term for L2: R1[t] \otimes dX[t]
    # We need R1[0...T-1]
    R1_lag = R1[:-1] # (T-1, D)
    
    # Compute increments of Level 2
    # dSig2[t, i, j] = R1_lag[t, i] * dX[t, j]
    dSig2 = np.einsum('ti,tj->tij', R1_lag, dX)
    
    # Integrate to get running Level 2
    R2 = np.cumsum(dSig2, axis=0) # (T-1, D, D)
    R2 = np.vstack([np.zeros((1, D, D)), R2]) # (T, D, D)
    R2_lag = R2[:-1] # (T-1, D, D)
    
    # Term for L3: R2_lag[t] \otimes dX[t]
    # dSig3[t, i, j, k] = R2_lag[t, i, j] * dX[t, k]
    dSig3 = np.einsum('tij,tk->tijk', R2_lag, dX)
    
    sig3 = np.sum(dSig3, axis=0).flatten()
    
    return np.concatenate([sig1, sig2, sig3])

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
    The 'Area' between Lead and Lag corresponds to sum of squares.
    """
    # Simply interleave points
    # Generic construction for path X = (x1, x2, x3...)
    # Lead-Lag Path is piecewise linear in 2D:
    # (x1, x1) -> (x2, x1) -> (x2, x2) -> (x3, x2) -> (x3, x3)...
    
    x = path_1d.flatten()
    n = len(x)
    
    # We construct 2 * n - 1 points
    lead = np.repeat(x, 2)[:-1] # x1, x1, x2, x2, x3...
    lag  = np.repeat(x, 2)[1:]  # x1, x2, x2, x3, x3...
    
    return np.column_stack([lead, lag])

def compute_lead_lag_signature(path_1d: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Computes signature of the Lead-Lag embedded path.
    Captures Quadratic Variation explicitly in the Level 2 terms.
    """
    ll_path = compute_lead_lag_path(path_1d)
    return compute_log_signature(ll_path, level=level)

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
