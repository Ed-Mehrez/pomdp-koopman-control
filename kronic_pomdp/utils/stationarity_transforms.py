"""
Signature-based stationarity transforms.

Detects appropriate transforms (log, sqrt, detrend) to make non-ergodic
processes approximately stationary, enabling CdC generator extraction.

See docs/theory_signature_stationarity_transforms.md for theory.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
import sys
import os

# Add path for signature features
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../examples/proof_of_concept'))
from signature_features import compute_log_signature


def detect_transform(X: np.ndarray, window_size: int = 252,
                    min_windows: int = 5) -> Tuple[str, Dict]:
    """
    Analyze signature growth patterns to detect appropriate stationarity transform.

    Args:
        X: 1D array of process values
        window_size: Size of rolling windows for analysis
        min_windows: Minimum number of windows required

    Returns:
        (transform_name, params) where transform_name in {'none', 'log', 'sqrt', 'detrend'}
    """
    X = np.asarray(X).flatten()
    T = len(X)

    if T < window_size * min_windows:
        window_size = max(T // min_windows, 20)

    n_windows = T // window_size
    if n_windows < min_windows:
        return 'none', {}

    windows = [X[i*window_size:(i+1)*window_size] for i in range(n_windows)]

    # Test transforms and score each
    candidates = []

    # 1. No transform
    score_none = _stationarity_score(X, windows, lambda x: x)
    candidates.append(('none', {}, score_none))

    # 2. Log transform (if all positive)
    if np.all(X > 0):
        score_log = _stationarity_score(X, windows, np.log)
        candidates.append(('log', {}, score_log))
    elif np.all(X > -1e6):  # Shifted log
        shift = -np.min(X) + 1e-6
        score_log = _stationarity_score(X, windows, lambda x: np.log(x + shift))
        candidates.append(('log', {'shift': shift}, score_log))

    # 3. Sqrt transform (for positive processes)
    if np.all(X >= 0):
        score_sqrt = _stationarity_score(X, windows, np.sqrt)
        candidates.append(('sqrt', {}, score_sqrt))

    # 4. Detrend (linear)
    t = np.arange(len(X))
    slope, intercept = np.polyfit(t, X, 1)
    detrend_fn = lambda x, t=t, s=slope: x - s * np.arange(len(x))
    score_detrend = _stationarity_score(X, windows, detrend_fn)
    candidates.append(('detrend', {'slope': slope, 'intercept': intercept}, score_detrend))

    # Select best (lowest score)
    best = min(candidates, key=lambda c: c[2])
    return best[0], best[1]


def _stationarity_score(X: np.ndarray, windows: list, transform: Callable) -> float:
    """
    Score stationarity of transformed process.
    Lower = more stationary.

    Metrics:
    1. CV of window variances (should be low for stationary)
    2. Trend in signature norms (should be zero for stationary)
    3. Variance ratio: end windows vs start windows
    """
    try:
        Y = transform(X)
    except:
        return float('inf')

    if not np.all(np.isfinite(Y)):
        return float('inf')

    # Transform windows
    n_windows = len(windows)
    window_size = len(windows[0])
    Y_windows = [Y[i*window_size:(i+1)*window_size] for i in range(n_windows)]

    # Metric 1: CV of increment variances
    vars_dY = [np.var(np.diff(w)) for w in Y_windows]
    if np.mean(vars_dY) < 1e-15:
        return float('inf')  # Degenerate
    cv_var = np.std(vars_dY) / np.mean(vars_dY)

    # Metric 2: Signature norm trend
    sig_norms = []
    for w in Y_windows:
        path = np.column_stack([np.linspace(0, 1, len(w)), w])
        try:
            sig = compute_log_signature(path, level=2)
            sig_norms.append(np.linalg.norm(sig))
        except:
            sig_norms.append(np.nan)

    sig_norms = np.array(sig_norms)
    valid = np.isfinite(sig_norms)
    if valid.sum() < 3:
        trend = 1.0
    else:
        # Normalized trend
        t_idx = np.arange(len(sig_norms))[valid]
        slope = np.polyfit(t_idx, sig_norms[valid], 1)[0]
        trend = np.abs(slope) / (np.mean(sig_norms[valid]) + 1e-10)

    # Metric 3: Variance ratio (end vs start)
    n_half = n_windows // 2
    var_start = np.mean(vars_dY[:n_half])
    var_end = np.mean(vars_dY[n_half:])
    var_ratio = max(var_end / (var_start + 1e-15), var_start / (var_end + 1e-15))

    # Combined score
    score = cv_var + 5 * trend + 0.5 * np.log(var_ratio + 1)
    return score


def apply_transform(X: np.ndarray, transform_name: str,
                   params: Dict) -> np.ndarray:
    """Apply a detected transform to data."""
    X = np.asarray(X)

    if transform_name == 'none':
        return X.copy()
    elif transform_name == 'log':
        shift = params.get('shift', 0)
        return np.log(X + shift)
    elif transform_name == 'sqrt':
        return np.sqrt(np.maximum(X, 0))
    elif transform_name == 'detrend':
        slope = params.get('slope', 0)
        t = np.arange(len(X))
        return X - slope * t
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


def inverse_transform(Y: np.ndarray, transform_name: str,
                     params: Dict) -> np.ndarray:
    """Inverse of apply_transform."""
    Y = np.asarray(Y)

    if transform_name == 'none':
        return Y.copy()
    elif transform_name == 'log':
        shift = params.get('shift', 0)
        return np.exp(Y) - shift
    elif transform_name == 'sqrt':
        return Y ** 2
    elif transform_name == 'detrend':
        slope = params.get('slope', 0)
        t = np.arange(len(Y))
        return Y + slope * t
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


def signature_mmd_diagnostic(X: np.ndarray, window_size: int = 252,
                            n_test_windows: int = 5) -> Dict:
    """
    Compute signature-based diagnostics for non-stationarity.

    Returns:
        Dictionary with:
        - 'mmd_trend': Trend in rolling signature means (should be ~0 for stationary)
        - 'mmd_scores': Pairwise MMD between windows
        - 'levy_area_trend': Trend in Lévy area (detects leverage changes)
        - 'is_stationary': Boolean assessment
    """
    X = np.asarray(X).flatten()
    T = len(X)
    n_windows = T // window_size

    if n_windows < n_test_windows:
        window_size = T // n_test_windows
        n_windows = n_test_windows

    # Compute signatures for each window
    signatures = []
    levy_areas = []

    for i in range(n_windows):
        w = X[i*window_size:(i+1)*window_size]
        path = np.column_stack([np.linspace(0, 1, len(w)), w])
        sig = compute_log_signature(path, level=2)
        signatures.append(sig)
        # Lévy area is the last term for 2D
        levy_areas.append(sig[-1] if len(sig) > 2 else 0)

    signatures = np.array(signatures)
    levy_areas = np.array(levy_areas)

    # Trend in signature means
    sig_means = np.mean(signatures, axis=1)
    t_idx = np.arange(len(sig_means))
    sig_trend = np.polyfit(t_idx, sig_means, 1)[0]

    # Trend in Lévy areas
    levy_trend = np.polyfit(t_idx, levy_areas, 1)[0]

    # Simple MMD: pairwise distances between window signatures
    mmd_scores = []
    for i in range(n_windows - 1):
        dist = np.linalg.norm(signatures[i] - signatures[i+1])
        mmd_scores.append(dist)

    mmd_cv = np.std(mmd_scores) / (np.mean(mmd_scores) + 1e-10)

    # Stationarity assessment
    # Stationary if: low trend, consistent MMD, bounded Lévy area
    is_stationary = (
        np.abs(sig_trend) < 0.1 * np.std(sig_means) and
        mmd_cv < 1.0 and
        np.abs(levy_trend) < 0.1 * np.std(levy_areas)
    )

    return {
        'mmd_trend': float(sig_trend),
        'mmd_scores': mmd_scores,
        'mmd_cv': float(mmd_cv),
        'levy_area_trend': float(levy_trend),
        'is_stationary': is_stationary,
        'signatures': signatures,
        'levy_areas': levy_areas
    }


def find_best_transform(X: np.ndarray, candidates: list = None,
                        verbose: bool = False) -> Tuple[str, Dict, float]:
    """
    Find the best transform from a list of candidates.

    Args:
        X: Process data
        candidates: List of (name, params) to try. If None, uses default set.
        verbose: Print scores for each candidate

    Returns:
        (best_name, best_params, best_score)
    """
    X = np.asarray(X).flatten()

    if candidates is None:
        candidates = [
            ('none', {}),
            ('log', {'shift': 0} if np.all(X > 0) else {'shift': -np.min(X) + 1e-6}),
            ('sqrt', {}),
            ('detrend', {}),
        ]
        # Filter invalid candidates
        if not np.all(X > 0) and not np.all(X > -1e6):
            candidates = [c for c in candidates if c[0] != 'log']
        if not np.all(X >= 0):
            candidates = [c for c in candidates if c[0] != 'sqrt']

    results = []
    for name, params in candidates:
        # Apply transform
        try:
            Y = apply_transform(X, name, params)
            diag = signature_mmd_diagnostic(Y)

            # Score: combine multiple metrics
            score = (
                np.abs(diag['mmd_trend']) +
                diag['mmd_cv'] +
                np.abs(diag['levy_area_trend'])
            )

            if verbose:
                print(f"  {name}: score={score:.4f} (trend={diag['mmd_trend']:.4f}, "
                      f"cv={diag['mmd_cv']:.4f}, levy={diag['levy_area_trend']:.4f})")

            results.append((name, params, score))
        except Exception as e:
            if verbose:
                print(f"  {name}: FAILED ({e})")
            results.append((name, params, float('inf')))

    best = min(results, key=lambda x: x[2])
    return best


# ======================================================================
# Joint Optimization: Transform + Segmentation
# ======================================================================

def box_cox_transform(X: np.ndarray, lam: float, shift: float = 0) -> np.ndarray:
    """
    Box-Cox transform: (x^λ - 1)/λ for λ≠0, log(x) for λ=0.
    Handles negative values via shift.
    """
    X = np.asarray(X) + shift
    if np.any(X <= 0):
        X = X - np.min(X) + 1e-6

    if abs(lam) < 1e-10:
        return np.log(X)
    else:
        return (np.power(X, lam) - 1) / lam


def signature_growth_rate(Y: np.ndarray, window: int = 63) -> float:
    """
    Cost function: normalized signature norm.
    For stationary process, ||Sig||² / T should be O(1).
    """
    if len(Y) < window:
        return float('inf')

    path = np.column_stack([np.linspace(0, 1, len(Y)), Y])
    try:
        sig = compute_log_signature(path, level=2)
        # Normalize by length (stationary processes have ||Sig|| ~ sqrt(T))
        return np.linalg.norm(sig)**2 / len(Y)
    except:
        return float('inf')


def optimal_segmentation_dp(Y: np.ndarray, cost_fn: Callable,
                            penalty: float = 1.0,
                            min_seg_len: int = 50,
                            max_segments: int = 20) -> Tuple[list, float]:
    """
    Dynamic programming for optimal segmentation.

    DP[t] = min cost to segment Y[0:t]
    DP[t] = min_{s < t} { DP[s] + cost(Y[s:t]) + penalty }

    Returns:
        (segments, total_cost) where segments is list of (start, end) tuples
    """
    T = len(Y)
    if T < min_seg_len:
        return [(0, T)], cost_fn(Y)

    # DP arrays
    DP = np.full(T + 1, np.inf)
    DP[0] = 0
    parent = np.zeros(T + 1, dtype=int)
    n_segs = np.zeros(T + 1, dtype=int)

    # Compute costs for candidate segments (with caching)
    cost_cache = {}

    def get_cost(s, t):
        if (s, t) not in cost_cache:
            cost_cache[(s, t)] = cost_fn(Y[s:t])
        return cost_cache[(s, t)]

    # DP iteration
    for t in range(min_seg_len, T + 1):
        for s in range(max(0, t - T // 2), t - min_seg_len + 1):
            if n_segs[s] >= max_segments:
                continue

            seg_cost = get_cost(s, t)
            if not np.isfinite(seg_cost):
                continue

            total = DP[s] + seg_cost + penalty
            if total < DP[t]:
                DP[t] = total
                parent[t] = s
                n_segs[t] = n_segs[s] + 1

    # Backtrack to get segments
    if not np.isfinite(DP[T]):
        return [(0, T)], cost_fn(Y)

    segments = []
    t = T
    while t > 0:
        s = parent[t]
        segments.append((s, t))
        t = s

    return segments[::-1], DP[T]


def mdl_lambda(T: int, sig_variance: float = 1.0) -> Tuple[float, float]:
    """
    Compute MDL-principled penalties from data length.

    From BIC: penalty = ½ log(T) per parameter.
    - λ₁: penalty per segment (change point needs log(T) bits to encode)
    - λ₂: penalty per transform parameter (negligible for Box-Cox)

    Args:
        T: Length of time series
        sig_variance: Typical signature variance per unit time (for scaling)

    Returns:
        (lambda_seg, lambda_complexity)
    """
    lambda_seg = 0.5 * np.log(T) * sig_variance
    lambda_complexity = 0.5 * np.log(T) / T  # Negligible for large T
    return lambda_seg, lambda_complexity


def cv_select_lambda(X: np.ndarray,
                     lambda_grid: list = None,
                     n_folds: int = 5,
                     verbose: bool = False) -> float:
    """
    Select segment penalty λ₁ via time-series cross-validation.

    Splits data temporally, fits on early folds, evaluates growth rate on later fold.

    Args:
        X: Input time series
        lambda_grid: List of λ values to try (default: [0.5, 1, 2, 4, 8])
        n_folds: Number of temporal folds
        verbose: Print progress

    Returns:
        Best λ₁ value
    """
    X = np.asarray(X).flatten()
    T = len(X)
    fold_size = T // n_folds

    if lambda_grid is None:
        # Default grid centered around MDL value
        mdl_lam, _ = mdl_lambda(T)
        lambda_grid = [mdl_lam * f for f in [0.25, 0.5, 1.0, 2.0, 4.0]]

    best_lambda = lambda_grid[0]
    best_score = float('inf')

    for lam in lambda_grid:
        scores = []
        for fold in range(n_folds - 1):
            train_end = (fold + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, T)

            if val_end - val_start < 50:
                continue

            # Fit on training data
            result = joint_optimize(X[:train_end], lambda_seg=lam,
                                    min_seg_len=min(126, train_end // 4))
            g_lambda = result['box_cox_lambda']

            # Evaluate on validation data
            Y_val = box_cox_transform(X[val_start:val_end], g_lambda)
            val_growth = signature_growth_rate(Y_val)
            if np.isfinite(val_growth):
                scores.append(val_growth)

        if scores:
            mean_score = np.mean(scores)
            if verbose:
                print(f"  λ={lam:.2f}: CV growth={mean_score:.4f}")
            if mean_score < best_score:
                best_score = mean_score
                best_lambda = lam

    return best_lambda


def joint_optimize(X: np.ndarray,
                   lambda_seg: float = None,
                   lambda_complexity: float = None,
                   box_cox_grid: list = None,
                   min_seg_len: int = 126,
                   verbose: bool = False) -> Dict:
    """
    Joint optimization over transform and segmentation.

    Solves the variational problem:
        min_{θ,τ,K} Σ ||Sig(g_θ(X)_{segment_i})||²/len + λ₁·K + λ₂·complexity(θ)

    Args:
        X: Input time series
        lambda_seg: Penalty per segment. If None, uses MDL default ½log(T).
        lambda_complexity: Penalty for transform complexity. If None, uses MDL default.
        box_cox_grid: List of λ values for Box-Cox transform
        min_seg_len: Minimum segment length
        verbose: Print progress

    Returns:
        Dictionary with optimal transform, segments, and diagnostics
    """
    X = np.asarray(X).flatten()
    T = len(X)

    # Use MDL-principled defaults if not specified
    if lambda_seg is None or lambda_complexity is None:
        mdl_seg, mdl_comp = mdl_lambda(T)
        if lambda_seg is None:
            lambda_seg = mdl_seg
        if lambda_complexity is None:
            lambda_complexity = mdl_comp

    if box_cox_grid is None:
        box_cox_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    best_obj = float('inf')
    best_result = None

    for lam in box_cox_grid:
        # Apply transform
        try:
            Y = box_cox_transform(X, lam)
            if not np.all(np.isfinite(Y)):
                continue
        except:
            continue

        # Find optimal segmentation
        segments, seg_obj = optimal_segmentation_dp(
            Y,
            cost_fn=signature_growth_rate,
            penalty=lambda_seg,
            min_seg_len=min_seg_len
        )

        # Complexity penalty (distance from identity transform)
        complexity = abs(lam - 1.0)

        # Total objective
        obj = seg_obj + lambda_complexity * complexity

        if verbose:
            print(f"  λ={lam:.2f}: {len(segments)} segments, "
                  f"seg_cost={seg_obj:.4f}, total={obj:.4f}")

        if obj < best_obj:
            best_obj = obj
            best_result = {
                'box_cox_lambda': lam,
                'segments': segments,
                'n_segments': len(segments),
                'segment_cost': seg_obj,
                'complexity_cost': lambda_complexity * complexity,
                'total_objective': obj,
                'transformed_data': Y
            }

    if best_result is None:
        # Fallback: no transform, single segment
        best_result = {
            'box_cox_lambda': 1.0,
            'segments': [(0, len(X))],
            'n_segments': 1,
            'segment_cost': signature_growth_rate(X),
            'complexity_cost': 0,
            'total_objective': signature_growth_rate(X),
            'transformed_data': X.copy()
        }

    return best_result


def find_ergodic_segments(X: np.ndarray, transform: str = 'auto',
                          min_seg_len: int = 126,
                          verbose: bool = False) -> Dict:
    """
    High-level API: Find segments where process is approximately ergodic.

    Args:
        X: Input time series
        transform: 'auto', 'none', 'log', 'sqrt', or Box-Cox λ value
        min_seg_len: Minimum segment length
        verbose: Print diagnostics

    Returns:
        Dictionary with segments and per-segment stationarity diagnostics
    """
    X = np.asarray(X).flatten()

    # Determine transform
    if transform == 'auto':
        result = joint_optimize(X, min_seg_len=min_seg_len, verbose=verbose)
        lam = result['box_cox_lambda']
        segments = result['segments']
        Y = result['transformed_data']
    elif transform == 'none':
        lam = 1.0
        Y = X.copy()
        segments, _ = optimal_segmentation_dp(Y, signature_growth_rate,
                                               min_seg_len=min_seg_len)
    elif transform == 'log':
        lam = 0.0
        Y = box_cox_transform(X, 0.0)
        segments, _ = optimal_segmentation_dp(Y, signature_growth_rate,
                                               min_seg_len=min_seg_len)
    elif transform == 'sqrt':
        lam = 0.5
        Y = box_cox_transform(X, 0.5)
        segments, _ = optimal_segmentation_dp(Y, signature_growth_rate,
                                               min_seg_len=min_seg_len)
    else:
        # Assume numeric Box-Cox lambda
        lam = float(transform)
        Y = box_cox_transform(X, lam)
        segments, _ = optimal_segmentation_dp(Y, signature_growth_rate,
                                               min_seg_len=min_seg_len)

    # Compute per-segment diagnostics
    seg_diagnostics = []
    for start, end in segments:
        seg_Y = Y[start:end]
        diag = signature_mmd_diagnostic(seg_Y, window_size=min(63, len(seg_Y)//3))
        seg_diagnostics.append({
            'start': start,
            'end': end,
            'length': end - start,
            'is_stationary': diag['is_stationary'],
            'mmd_trend': diag['mmd_trend'],
            'growth_rate': signature_growth_rate(seg_Y)
        })

    return {
        'transform': f'box_cox(λ={lam:.2f})',
        'box_cox_lambda': lam,
        'n_segments': len(segments),
        'segments': segments,
        'segment_diagnostics': seg_diagnostics,
        'transformed_data': Y
    }


# Demo
if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("Signature-Based Stationarity Transform Detection")
    print("=" * 60)

    # Test 1: GBM (needs log)
    print("\n1. GBM (exponential growth)")
    S0, mu, sigma = 100, 0.05, 0.2
    dt = 1/252
    T = 2520  # 10 years
    dW = np.random.randn(T) * np.sqrt(dt)
    S = S0 * np.exp(np.cumsum((mu - 0.5*sigma**2)*dt + sigma*dW))

    transform, params = detect_transform(S)
    print(f"   Detected transform: {transform}, params: {params}")

    best_name, best_params, best_score = find_best_transform(S, verbose=True)
    print(f"   Best transform: {best_name} (score={best_score:.4f})")

    # Test 2: OU (already stationary)
    print("\n2. OU process (mean-reverting)")
    kappa, theta, xi = 5.0, 0.04, 0.3
    V = np.zeros(T)
    V[0] = theta
    for t in range(1, T):
        V[t] = V[t-1] + kappa*(theta - V[t-1])*dt + xi*np.sqrt(max(V[t-1], 1e-8))*dW[t]
        V[t] = max(V[t], 1e-8)

    transform, params = detect_transform(V)
    print(f"   Detected transform: {transform}, params: {params}")

    best_name, best_params, best_score = find_best_transform(V, verbose=True)
    print(f"   Best transform: {best_name} (score={best_score:.4f})")

    # Test 3: Random walk with drift (needs detrend)
    print("\n3. Random walk with drift")
    drift = 0.001
    X = np.cumsum(drift + 0.02 * np.random.randn(T))

    transform, params = detect_transform(X)
    print(f"   Detected transform: {transform}, params: {params}")

    best_name, best_params, best_score = find_best_transform(X, verbose=True)
    print(f"   Best transform: {best_name} (score={best_score:.4f})")

    # Test 4: MMD diagnostic
    print("\n4. MMD Diagnostic on GBM vs log(GBM)")
    diag_raw = signature_mmd_diagnostic(S)
    diag_log = signature_mmd_diagnostic(np.log(S))

    print(f"   Raw GBM:  stationary={diag_raw['is_stationary']}, "
          f"mmd_trend={diag_raw['mmd_trend']:.4f}")
    print(f"   log(GBM): stationary={diag_log['is_stationary']}, "
          f"mmd_trend={diag_log['mmd_trend']:.4f}")

    # Test 5: Joint optimization on regime-switching GBM
    print("\n5. Joint Optimization: Regime-switching GBM")
    # Simulate GBM with drift change at t=1260
    T = 2520
    dW = np.random.randn(T) * np.sqrt(dt)
    S_regime = np.zeros(T)
    S_regime[0] = 100
    for t in range(1, T):
        mu_t = 0.10 if t < 1260 else -0.05  # Drift switch
        S_regime[t] = S_regime[t-1] * np.exp((mu_t - 0.5*0.2**2)*dt + 0.2*dW[t])

    print("   Running joint optimization...")
    result = joint_optimize(S_regime, lambda_seg=5.0, verbose=True)
    print(f"\n   Optimal: Box-Cox λ={result['box_cox_lambda']:.2f}, "
          f"{result['n_segments']} segments")
    for i, (s, e) in enumerate(result['segments']):
        print(f"     Segment {i+1}: [{s}, {e}) (len={e-s})")

    # Test 6: High-level API
    print("\n6. High-level API: find_ergodic_segments")
    segs = find_ergodic_segments(S_regime, transform='auto', verbose=False)
    print(f"   Transform: {segs['transform']}")
    print(f"   Segments: {segs['n_segments']}")
    for sd in segs['segment_diagnostics']:
        print(f"     [{sd['start']}, {sd['end']}): "
              f"stationary={sd['is_stationary']}, "
              f"growth_rate={sd['growth_rate']:.4f}")
