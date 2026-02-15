"""
Cumulative vs Windowed Signatures Comparison
=============================================
Compares the principled cumulative signature approach (Chen's identity)
with the windowed approximation used in bates_volatility.py.

Key insight: fSDEs are non-Markovian in X, but MARKOVIAN in cumulative
signature space S_t = Sig(X_{[0,t]}).

This script benchmarks both approaches on Bates model volatility estimation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
import os
import time

# Add paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)

from examples.proof_of_concept.signature_features import compute_path_signature

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 1. Bates Model (same as bates_volatility.py)
# ============================================================================
class BatesSimulator:
    """Heston + Jumps."""
    def __init__(self, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5,
                 lambda_j=5.0, mu_j=-0.05, sigma_j=0.02, dt=0.01):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.dt = dt

    def generate_path(self, n_steps, s0=100.0, v0=0.04, seed=None):
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros(n_steps)
        V = np.zeros(n_steps)
        S[0] = s0
        V[0] = v0

        for t in range(1, n_steps):
            v_abs = max(V[t-1], 1e-6)

            z1 = np.random.normal()
            z2 = np.random.normal()
            dW1 = np.sqrt(self.dt) * z1
            dW2 = np.sqrt(self.dt) * (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2)

            dv = self.kappa * (self.theta - v_abs) * self.dt + self.xi * np.sqrt(v_abs) * dW2
            V[t] = max(V[t-1] + dv, 1e-6)

            dS_cont = S[t-1] * (0.0 * self.dt + np.sqrt(v_abs) * dW1)
            S[t] = S[t-1] + dS_cont

            if np.random.rand() < self.lambda_j * self.dt:
                J = np.random.normal(self.mu_j, self.sigma_j)
                S[t] *= np.exp(J)

        returns = np.diff(np.log(S))
        true_variance = V[:-1]
        return S, V, returns, true_variance


# ============================================================================
# 2. Windowed Signature Estimator (existing approach)
# ============================================================================
def windowed_signature_features(returns, window_sizes, degree=2):
    """Compute multi-scale windowed signatures (existing approach)."""
    max_w = max(window_sizes)
    global_vol_scale = np.std(returns) + 1e-9

    features = []
    valid_indices = []

    for i in range(max_w, len(returns)):
        multi_scale_sig = []
        for w in window_sizes:
            window_rets = returns[i-w:i]
            t_steps = np.linspace(0, 1, len(window_rets))
            rets_scaled = window_rets / global_vol_scale
            path = np.column_stack([t_steps, rets_scaled])
            sig = compute_path_signature(path, level=degree)
            multi_scale_sig.append(sig)

        features.append(np.concatenate(multi_scale_sig))
        valid_indices.append(i)

    return np.array(features), np.array(valid_indices), global_vol_scale


# ============================================================================
# 3. Cumulative Signature Estimator (Chen's identity)
# ============================================================================
def cumulative_signature_features(returns, degree=2, subsample=1):
    """
    Compute cumulative signatures using Chen's identity, with INCREMENTAL features.

    KEY INSIGHT: For volatility estimation, we need LOCAL features.
    The Koopman approach uses dS/dt (signature derivatives), not raw S.

    So we compute: [dS/dt at time t] as features for predicting v_t
    """
    n = len(returns)
    global_vol_scale = np.std(returns) + 1e-9
    dt_normalized = 1.0 / n

    # Initialize cumulative signature
    sig1 = np.zeros(2)
    sig2 = np.zeros((2, 2))

    # Previous values for computing derivatives
    sig1_prev = np.zeros(2)
    sig2_prev = np.zeros((2, 2))

    features = []
    valid_indices = []

    # Rolling window for local statistics (small window for smoothing)
    window = 20
    recent_sq_returns = []

    for i in range(n):
        dx = returns[i] / global_vol_scale
        dt = dt_normalized

        # Save previous
        sig1_prev = sig1.copy()
        sig2_prev = sig2.copy()

        # Chen's identity update
        inc_sig1 = np.array([dt, dx])
        inc_sig2 = np.outer(inc_sig1, inc_sig1) / 2.0
        sig1 = sig1 + inc_sig1
        sig2 = sig2 + inc_sig2 + np.outer(sig1_prev, inc_sig1)

        # Track recent squared returns
        recent_sq_returns.append(dx**2)
        if len(recent_sq_returns) > window:
            recent_sq_returns.pop(0)

        if i >= 50 and i % subsample == 0:
            # INCREMENTAL features (signature derivatives)
            dsig1 = sig1 - sig1_prev
            dsig2 = sig2 - sig2_prev

            # Local Lévy area increment
            d_levy = (dsig2[0, 1] - dsig2[1, 0]) / 2.0

            # Local quadratic variation (what we're estimating)
            local_qv = np.mean(recent_sq_returns) if recent_sq_returns else 0

            # Instantaneous squared return (noisy proxy)
            inst_sq_ret = dx**2

            # Feature vector using INCREMENTS (local information)
            feat = np.array([
                1.0,
                dsig1[1],  # dx (increment, normalized)
                inst_sq_ret,  # dx² (instantaneous QV proxy)
                local_qv,  # rolling QV (smoothed)
                d_levy,  # local Lévy area increment
                dsig2[1, 1],  # d(x⊗x) term
            ])
            features.append(feat)
            valid_indices.append(i)

    return np.array(features), np.array(valid_indices), global_vol_scale


def cumulative_with_iisignature(returns, degree=2, subsample=1):
    """
    Alternative: Use iisignature for proper cumulative signature computation.
    This recomputes from scratch each time (slower but exact).
    """
    try:
        import iisignature
    except ImportError:
        print("iisignature not available, using manual computation")
        return cumulative_signature_features(returns, degree, subsample)

    n = len(returns)
    global_vol_scale = np.std(returns) + 1e-9

    # Build cumulative return path
    cumsum = np.cumsum(returns / global_vol_scale)
    t = np.linspace(0, 1, n)

    features = []
    valid_indices = []

    for i in range(50, n, subsample):
        # Full path up to i
        path = np.column_stack([t[:i+1], np.concatenate([[0], cumsum[:i]])])
        sig = iisignature.sig(path, degree)

        features.append(sig)
        valid_indices.append(i)

    return np.array(features), np.array(valid_indices), global_vol_scale


# ============================================================================
# 4. Comparison Experiment
# ============================================================================
def run_comparison():
    print("=" * 70)
    print("CUMULATIVE vs WINDOWED SIGNATURES COMPARISON")
    print("=" * 70)

    # Parameters
    dt = 0.01
    n_steps = 8000
    train_split = 4000
    dyadic_windows = [16, 32, 64]  # For windowed approach

    # Generate Bates path
    print("\n1. Generating Bates model path...")
    sim = BatesSimulator(kappa=2.0, theta=0.04, xi=0.5, rho=-0.5,
                         lambda_j=5.0, mu_j=-0.05, sigma_j=0.02, dt=dt)
    S, V, returns, true_var = sim.generate_path(n_steps, seed=42)

    # Split data
    train_rets = returns[:train_split]
    test_rets = returns[train_split:]
    train_var = true_var[:train_split]
    test_var = true_var[train_split:]

    # =========================================================================
    # METHOD 1: Windowed Signatures
    # =========================================================================
    print("\n2. Computing WINDOWED signatures...")
    t0 = time.time()

    X_win_train, idx_win_train, scale_win = windowed_signature_features(
        train_rets, dyadic_windows, degree=2)
    X_win_test, idx_win_test, _ = windowed_signature_features(
        test_rets, dyadic_windows, degree=2)

    win_time = time.time() - t0
    print(f"   Time: {win_time:.2f}s")
    print(f"   Feature dim: {X_win_train.shape[1]}")

    # Train windowed model
    y_train_win = train_var[idx_win_train]
    scaler_win = StandardScaler()
    X_train_scaled_win = scaler_win.fit_transform(X_win_train)

    model_win = Ridge(alpha=1.0)
    model_win.fit(X_train_scaled_win, y_train_win)
    r2_win_train = model_win.score(X_train_scaled_win, y_train_win)

    # Test windowed model
    X_test_scaled_win = scaler_win.transform(X_win_test)
    preds_win = model_win.predict(X_test_scaled_win)
    y_test_win = test_var[idx_win_test]
    mse_win = np.mean((preds_win - y_test_win)**2)
    corr_win = np.corrcoef(preds_win, y_test_win)[0, 1]

    print(f"   Train R²: {r2_win_train:.4f}")
    print(f"   Test MSE: {mse_win:.2e}")
    print(f"   Test Corr: {corr_win:.4f}")

    # =========================================================================
    # METHOD 2: Cumulative Signatures (Chen's Identity)
    # =========================================================================
    print("\n3. Computing CUMULATIVE signatures (Chen's identity)...")
    t0 = time.time()

    X_cum_train, idx_cum_train, scale_cum = cumulative_signature_features(
        train_rets, degree=2, subsample=1)
    X_cum_test, idx_cum_test, _ = cumulative_signature_features(
        test_rets, degree=2, subsample=1)

    cum_time = time.time() - t0
    print(f"   Time: {cum_time:.2f}s")
    print(f"   Feature dim: {X_cum_train.shape[1]}")

    # Train cumulative model
    y_train_cum = train_var[idx_cum_train]
    scaler_cum = StandardScaler()
    X_train_scaled_cum = scaler_cum.fit_transform(X_cum_train)

    model_cum = Ridge(alpha=1.0)
    model_cum.fit(X_train_scaled_cum, y_train_cum)
    r2_cum_train = model_cum.score(X_train_scaled_cum, y_train_cum)

    # Test cumulative model
    X_test_scaled_cum = scaler_cum.transform(X_cum_test)
    preds_cum = model_cum.predict(X_test_scaled_cum)
    y_test_cum = test_var[idx_cum_test]
    mse_cum = np.mean((preds_cum - y_test_cum)**2)
    corr_cum = np.corrcoef(preds_cum, y_test_cum)[0, 1]

    print(f"   Train R²: {r2_cum_train:.4f}")
    print(f"   Test MSE: {mse_cum:.2e}")
    print(f"   Test Corr: {corr_cum:.4f}")

    # =========================================================================
    # METHOD 3: Cumulative via iisignature (exact, for comparison)
    # =========================================================================
    print("\n4. Computing CUMULATIVE signatures (iisignature batch)...")
    t0 = time.time()

    X_iis_train, idx_iis_train, _ = cumulative_with_iisignature(
        train_rets, degree=2, subsample=10)  # Subsample for speed
    X_iis_test, idx_iis_test, _ = cumulative_with_iisignature(
        test_rets, degree=2, subsample=10)

    iis_time = time.time() - t0
    print(f"   Time: {iis_time:.2f}s")
    print(f"   Feature dim: {X_iis_train.shape[1]}")

    # Train iisignature model
    y_train_iis = train_var[idx_iis_train]
    scaler_iis = StandardScaler()
    X_train_scaled_iis = scaler_iis.fit_transform(X_iis_train)

    model_iis = Ridge(alpha=1.0)
    model_iis.fit(X_train_scaled_iis, y_train_iis)
    r2_iis_train = model_iis.score(X_train_scaled_iis, y_train_iis)

    # Test iisignature model
    X_test_scaled_iis = scaler_iis.transform(X_iis_test)
    preds_iis = model_iis.predict(X_test_scaled_iis)
    y_test_iis = test_var[idx_iis_test]
    mse_iis = np.mean((preds_iis - y_test_iis)**2)
    corr_iis = np.corrcoef(preds_iis, y_test_iis)[0, 1]

    print(f"   Train R²: {r2_iis_train:.4f}")
    print(f"   Test MSE: {mse_iis:.2e}")
    print(f"   Test Corr: {corr_iis:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'MSE':>12} {'Corr':>8} {'Time':>8} {'Features':>10}")
    print("-" * 70)
    print(f"{'Windowed (multi-scale)':<25} {mse_win:>12.2e} {corr_win:>8.4f} {win_time:>7.2f}s {X_win_train.shape[1]:>10}")
    print(f"{'Cumulative (Chen)':<25} {mse_cum:>12.2e} {corr_cum:>8.4f} {cum_time:>7.2f}s {X_cum_train.shape[1]:>10}")
    print(f"{'Cumulative (iisig)':<25} {mse_iis:>12.2e} {corr_iis:>8.4f} {iis_time:>7.2f}s {X_iis_train.shape[1]:>10}")

    # Determine winner
    best_mse = min(mse_win, mse_cum, mse_iis)
    if best_mse == mse_cum:
        winner = "Cumulative (Chen)"
    elif best_mse == mse_iis:
        winner = "Cumulative (iisig)"
    else:
        winner = "Windowed"

    print(f"\n*** Best MSE: {winner} ***")

    # =========================================================================
    # Visualization
    # =========================================================================
    print("\n5. Generating comparison figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Windowed predictions
    ax1 = axes[0, 0]
    plot_len = min(400, len(idx_win_test))
    t_plot = np.arange(plot_len) * dt
    ax1.plot(t_plot, y_test_win[:plot_len], 'k-', alpha=0.5, linewidth=1, label='True')
    ax1.plot(t_plot, preds_win[:plot_len], 'b-', linewidth=1.5, label=f'Windowed (MSE={mse_win:.2e})')
    ax1.set_title(f'Windowed Signatures (windows={dyadic_windows})', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative predictions
    ax2 = axes[0, 1]
    plot_len_cum = min(400, len(idx_cum_test))
    t_plot_cum = np.arange(plot_len_cum) * dt
    ax2.plot(t_plot_cum, y_test_cum[:plot_len_cum], 'k-', alpha=0.5, linewidth=1, label='True')
    ax2.plot(t_plot_cum, preds_cum[:plot_len_cum], 'g-', linewidth=1.5, label=f'Cumulative (MSE={mse_cum:.2e})')
    ax2.set_title('Cumulative Signatures (Chen\'s Identity)', fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Both overlaid
    ax3 = axes[1, 0]
    common_len = min(plot_len, plot_len_cum)
    t_common = np.arange(common_len) * dt
    ax3.plot(t_common, y_test_win[:common_len], 'k-', alpha=0.5, linewidth=1, label='True')
    ax3.plot(t_common, preds_win[:common_len], 'b--', linewidth=1.5, alpha=0.7, label='Windowed')
    ax3.plot(t_common, preds_cum[:common_len], 'g-', linewidth=1.5, alpha=0.7, label='Cumulative')
    ax3.set_title('Direct Comparison', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Variance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Bar comparison
    ax4 = axes[1, 1]
    methods = ['Windowed', 'Cumulative\n(Chen)', 'Cumulative\n(iisig)']
    mses = [mse_win, mse_cum, mse_iis]
    colors = ['#3498db', '#27ae60', '#9b59b6']

    bars = ax4.bar(methods, mses, color=colors, edgecolor='black')
    ax4.set_ylabel('MSE (log scale)')
    ax4.set_title('MSE Comparison', fontweight='bold')
    ax4.set_yscale('log')

    for bar, mse in zip(bars, mses):
        ax4.annotate(f'{mse:.2e}', xy=(bar.get_x() + bar.get_width()/2, mse),
                     ha='center', va='bottom', fontsize=10)

    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cumulative vs Windowed Signatures: Bates Volatility Estimation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'cumulative_vs_windowed.png'), dpi=150)
    print("   Saved cumulative_vs_windowed.png")

    plt.close('all')
    print("\nDone!")


if __name__ == "__main__":
    run_comparison()
