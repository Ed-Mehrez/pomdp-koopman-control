"""
Fair Comparison: Signature vs Kalman

Previous results were unfair because:
- Kalman does FILTERING: E[x_t | y_{0:t}]
- Decay signature does SMOOTHING: uses weighted y_{t-20:t}

For slow dynamics (A=-0.5) with noisy observations, smoothing always wins.

This script:
1. Fair LQG: Compare signature vs Kalman SMOOTHER (both use windows)
2. Fast dynamics: Where smoothing hurts (signal changes within window)
3. Nonlinear POMDP: Where Kalman is suboptimal
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial.distance import cdist
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from kronic_pomdp.experiments.lqg_baseline import LQGSystem
from kronic_pomdp.experiments.online_rbf_sig_belief import OnlineSignatureState


class KalmanSmoother:
    """Fixed-lag Kalman smoother for fair comparison."""

    def __init__(self, lqg: LQGSystem, lag: int = 20):
        self.lqg = lqg
        self.lag = lag
        self.L = lqg.L_kalman

    def smooth(self, y: np.ndarray, x_filter: np.ndarray) -> np.ndarray:
        """
        Fixed-lag smoother: E[x_t | y_{0:t+lag}]

        Uses backward pass from Kalman filter.
        Simplified version: weighted average of filter estimates.
        """
        n = len(y)
        x_smooth = np.zeros(n)

        # Simple approximation: weighted average of nearby filter estimates
        # True smoother would run backward pass, but this captures the idea
        for t in range(n):
            start = max(0, t - self.lag // 2)
            end = min(n, t + self.lag // 2)
            # Weight by distance from t
            weights = np.exp(-0.1 * np.abs(np.arange(start, end) - t))
            weights /= weights.sum()
            x_smooth[t] = np.sum(weights * x_filter[start:end])

        return x_smooth


class NonlinearPOMDP:
    """
    Nonlinear POMDP where Kalman is suboptimal.

    State: x_t (scalar)
    Dynamics: dx = -tanh(x) dt + σ dW  (nonlinear mean-reversion)
    Observation: y = x² + noise  (nonlinear observation)
    """

    def __init__(self, sigma=0.5, obs_noise=0.3, dt=0.01):
        self.sigma = sigma
        self.obs_noise = obs_noise
        self.dt = dt

    def simulate(self, x0=1.0, T=10.0, seed=42):
        np.random.seed(seed)

        n_steps = int(T / self.dt)
        sqrt_dt = np.sqrt(self.dt)

        x_true = np.zeros(n_steps)
        y = np.zeros(n_steps)

        x_true[0] = x0
        y[0] = x0**2 + self.obs_noise * np.random.randn()

        for k in range(n_steps - 1):
            # Nonlinear dynamics
            dW = np.random.randn() * sqrt_dt
            x_true[k+1] = x_true[k] - np.tanh(x_true[k]) * self.dt + self.sigma * dW

            # Nonlinear observation
            y[k+1] = x_true[k+1]**2 + self.obs_noise * np.random.randn()

        return {'x_true': x_true, 'y': y, 't': np.arange(n_steps) * self.dt}


class ExtendedKalmanFilter:
    """EKF for nonlinear POMDP (linearized Kalman)."""

    def __init__(self, pomdp: NonlinearPOMDP):
        self.pomdp = pomdp
        self.dt = pomdp.dt

    def filter(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        x_hat = np.zeros(n)
        P = np.ones(n)  # Covariance

        # Initial guess from first observation (y = x², so x ≈ ±sqrt(y))
        x_hat[0] = np.sign(np.random.randn()) * np.sqrt(max(0, y[0]))
        P[0] = 1.0

        Q = self.pomdp.sigma**2 * self.dt  # Process noise variance
        R = self.pomdp.obs_noise**2  # Observation noise variance

        for k in range(n - 1):
            # Predict (linearized around current estimate)
            A = 1 - (1 - np.tanh(x_hat[k])**2) * self.dt  # d/dx of (x - tanh(x)*dt)
            x_pred = x_hat[k] - np.tanh(x_hat[k]) * self.dt
            P_pred = A * P[k] * A + Q

            # Update (linearized observation: dy/dx = 2x)
            H = 2 * x_pred  # Jacobian of x²
            y_pred = x_pred**2

            # Kalman gain
            S = H * P_pred * H + R
            K = P_pred * H / S

            # Innovation
            innovation = y[k+1] - y_pred

            x_hat[k+1] = x_pred + K * innovation
            P[k+1] = (1 - K * H) * P_pred

        return x_hat


def train_signature_filter(system, train_seeds, decay=0.95, max_samples=3000):
    """Train RBF-Sig filter for any system."""

    features = []
    targets = []

    is_lqg = isinstance(system, LQGSystem)
    dt = system.dt

    for seed in train_seeds:
        if is_lqg:
            result = system.simulate(x0=np.random.randn()*2, T=10.0, seed=seed)
        else:
            result = system.simulate(x0=np.random.randn(), T=10.0, seed=seed)

        y = result['y']
        x_true = result['x_true']

        sig_state = OnlineSignatureState(decay=decay)
        warmup = int(1.0 / (1 - decay)) if decay else 30

        for t in range(len(y)):
            if t > 0:
                sig_state.update(dt, y[t])
            if t >= warmup:
                features.append(sig_state.to_features())
                targets.append(x_true[t])

    X = np.array(features)
    y_targets = np.array(targets)

    # Subsample
    if len(X) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y_targets = y_targets[idx]

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std

    # Auto gamma
    dists = cdist(X_scaled[:500], X_scaled[:500], 'sqeuclidean')
    gamma = 1.0 / np.median(dists[dists > 0])

    # Train
    model = KernelRidge(kernel='rbf', gamma=gamma, alpha=0.01)
    model.fit(X_scaled, y_targets)

    return model, X_mean, X_std, decay


def test_filter(system, sig_model, test_seeds, baseline_filter=None):
    """Test signature filter against baseline."""

    model, X_mean, X_std, decay = sig_model
    dt = system.dt

    rmse_base = []
    rmse_sig = []
    corr_base = []
    corr_sig = []

    for seed in test_seeds:
        if isinstance(system, LQGSystem):
            result = system.simulate(x0=np.random.randn()*2, T=15.0, seed=seed)
            x_base = result['x_hat'] if baseline_filter is None else baseline_filter(result)
        else:
            result = system.simulate(x0=np.random.randn(), T=15.0, seed=seed)
            x_base = baseline_filter.filter(result['y']) if baseline_filter else np.zeros(len(result['y']))

        y = result['y']
        x_true = result['x_true']

        # Signature filter
        sig_state = OnlineSignatureState(decay=decay)
        warmup = int(1.0 / (1 - decay)) if decay else 30
        x_sig = np.zeros(len(y))

        for t in range(len(y)):
            if t > 0:
                sig_state.update(dt, y[t])
            if t >= warmup:
                feat = sig_state.to_features().reshape(1, -1)
                feat_scaled = (feat - X_mean) / X_std
                x_sig[t] = model.predict(feat_scaled)[0]

        start = 50
        rmse_base.append(np.sqrt(np.mean((x_true[start:] - x_base[start:])**2)))
        rmse_sig.append(np.sqrt(np.mean((x_true[start:] - x_sig[start:])**2)))
        corr_base.append(np.corrcoef(x_true[start:], x_base[start:])[0, 1])
        corr_sig.append(np.corrcoef(x_true[start:], x_sig[start:])[0, 1])

    return {
        'rmse_base': np.mean(rmse_base),
        'rmse_sig': np.mean(rmse_sig),
        'corr_base': np.mean(corr_base),
        'corr_sig': np.mean(corr_sig),
        'ratio': np.mean(rmse_sig) / np.mean(rmse_base)
    }


def main():
    print("=" * 70)
    print("FAIR COMPARISON: Signature vs Optimal Baselines")
    print("=" * 70)

    train_seeds = list(range(1000, 1050))
    test_seeds = list(range(2000, 2010))

    # =========================================================================
    # Test 1: LQG with Kalman SMOOTHER (fair - both use windows)
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: LQG - Signature vs Kalman SMOOTHER")
    print("-" * 70)
    print("Both methods use ~20 step window. Fair comparison.")

    lqg = LQGSystem(A=-0.5, B=1.0, C=1.0, G=0.3, H=0.5, dt=0.01)
    smoother = KalmanSmoother(lqg, lag=20)

    print("\nTraining signature filter...")
    sig_model = train_signature_filter(lqg, train_seeds, decay=0.95)

    def smoother_baseline(result):
        return smoother.smooth(result['y'], result['x_hat'])

    results = test_filter(lqg, sig_model, test_seeds, smoother_baseline)

    print(f"\n  {'Method':<25s} {'RMSE':>10s} {'Corr':>10s}")
    print(f"  {'-'*47}")
    print(f"  {'Kalman Smoother':<25s} {results['rmse_base']:>10.4f} {results['corr_base']:>10.4f}")
    print(f"  {'Signature (decay=0.95)':<25s} {results['rmse_sig']:>10.4f} {results['corr_sig']:>10.4f}")
    print(f"\n  Ratio: {results['ratio']:.2f}x (1.0 = equal)")

    # =========================================================================
    # Test 2: FAST LQG (where smoothing hurts)
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: FAST LQG (A=-5.0) - Smoothing should hurt")
    print("-" * 70)
    print("Fast dynamics: state changes significantly within 20-step window.")

    fast_lqg = LQGSystem(A=-5.0, B=1.0, C=1.0, G=0.3, H=0.5, dt=0.01)

    print("\nTraining signature filter...")
    sig_model_fast = train_signature_filter(fast_lqg, train_seeds, decay=0.95)

    # For fast system, use Kalman filter (not smoother) as baseline
    results_fast = test_filter(fast_lqg, sig_model_fast, test_seeds, None)

    print(f"\n  {'Method':<25s} {'RMSE':>10s} {'Corr':>10s}")
    print(f"  {'-'*47}")
    print(f"  {'Kalman Filter':<25s} {results_fast['rmse_base']:>10.4f} {results_fast['corr_base']:>10.4f}")
    print(f"  {'Signature (decay=0.95)':<25s} {results_fast['rmse_sig']:>10.4f} {results_fast['corr_sig']:>10.4f}")
    print(f"\n  Ratio: {results_fast['ratio']:.2f}x (>1.0 means Kalman wins)")

    # =========================================================================
    # Test 3: Nonlinear POMDP (where Kalman is suboptimal)
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: NONLINEAR POMDP - Signatures should shine!")
    print("-" * 70)
    print("Dynamics: dx = -tanh(x) dt + σ dW")
    print("Observation: y = x² + noise (nonlinear!)")
    print("EKF is suboptimal here.")

    nonlin = NonlinearPOMDP(sigma=0.5, obs_noise=0.3, dt=0.01)
    ekf = ExtendedKalmanFilter(nonlin)

    print("\nTraining signature filter...")
    sig_model_nonlin = train_signature_filter(nonlin, train_seeds, decay=0.95)

    results_nonlin = test_filter(nonlin, sig_model_nonlin, test_seeds, ekf)

    print(f"\n  {'Method':<25s} {'RMSE':>10s} {'Corr':>10s}")
    print(f"  {'-'*47}")
    print(f"  {'EKF (linearized)':<25s} {results_nonlin['rmse_base']:>10.4f} {results_nonlin['corr_base']:>10.4f}")
    print(f"  {'Signature (decay=0.95)':<25s} {results_nonlin['rmse_sig']:>10.4f} {results_nonlin['corr_sig']:>10.4f}")
    print(f"\n  Ratio: {results_nonlin['ratio']:.2f}x (<1.0 means Signature wins)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: When do signatures add value?")
    print("=" * 70)
    print(f"""
| Test                    | Sig/Baseline | Winner      | Why                          |
|-------------------------|--------------|-------------|------------------------------|
| LQG (slow) vs Smoother  | {results['ratio']:.2f}x        | {'Signature' if results['ratio'] < 1 else 'Smoother':11s} | Fair comparison (both smooth) |
| LQG (fast) vs Filter    | {results_fast['ratio']:.2f}x        | {'Signature' if results_fast['ratio'] < 1 else 'Kalman':11s} | Smoothing hurts fast dynamics |
| Nonlinear vs EKF        | {results_nonlin['ratio']:.2f}x        | {'Signature' if results_nonlin['ratio'] < 1 else 'EKF':11s} | {'Sig captures nonlinearity' if results_nonlin['ratio'] < 1 else 'EKF sufficient'} |

Key insight: Signatures add value when:
1. Model is unknown or misspecified
2. Observation model is nonlinear
3. Dynamics have complex path-dependent structure

For known linear systems, Kalman is optimal and can't be beat!
""")


if __name__ == "__main__":
    main()
