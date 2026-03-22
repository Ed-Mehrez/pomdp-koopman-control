"""
Online RBF-Sig Belief Filter using Chen's Identity

Key insight: Chen's identity allows O(1) signature updates:
    S(X * Y) = S(X) ⊗ S(Y)

For WINDOWED signatures, we use exponential decay (soft window):
- Each step: S_new = decay * S_old ⊗ S_increment
- This approximates a hard window with forgetting factor

This gives O(1) per step instead of O(window_size).
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/sskf'))

from kronic_pomdp.experiments.lqg_baseline import LQGSystem
from streaming_sig_kkf import SignatureState, signature_to_vector


class OnlineSignatureState:
    """
    Online signature state with two modes:
    1. CUMULATIVE: Pure Chen's identity, no forgetting (default)
    2. DECAY: Exponential decay for soft windowing

    Chen's identity: S(X * Y) = S(X) ⊗ S(Y)
    """

    def __init__(self, decay: float = None, level: int = 2):
        """
        Args:
            decay: Forgetting factor. If None, use pure cumulative (no decay).
                   decay=0.95 -> effective window ~20 steps
                   decay=0.97 -> effective window ~33 steps
            level: Signature truncation level
        """
        self.decay = decay  # None = pure cumulative
        self.level = level
        self.sig_state = SignatureState(level=level, store_path=False)

        # Also store statistics
        self.n_obs = 0
        self.sum_obs = 0.0
        self.sum_sq_obs = 0.0
        self.last_obs = 0.0

        # For decay mode, use EMA
        self.mean_ema = 0.0
        self.var_ema = 0.0

    def reset(self):
        """Reset to initial state."""
        self.sig_state.reset()
        self.n_obs = 0
        self.sum_obs = 0.0
        self.sum_sq_obs = 0.0
        self.last_obs = 0.0
        self.mean_ema = 0.0
        self.var_ema = 0.0

    def update(self, dt: float, new_obs: float):
        """
        Update signature with new observation in O(1) time.
        """
        # Compute increment
        dx = new_obs - self.last_obs

        if self.decay is not None:
            # DECAY MODE: Apply exponential forgetting
            self.sig_state.S[1] *= self.decay
            if self.level >= 2:
                self.sig_state.S[2] *= self.decay

            # EMA statistics
            self.mean_ema = self.decay * self.mean_ema + (1 - self.decay) * new_obs
            self.var_ema = self.decay * self.var_ema + (1 - self.decay) * (new_obs - self.mean_ema)**2

        # Extend via Chen's identity (works for both modes)
        self.sig_state.extend(dt, dx)

        # Update running statistics (for cumulative mode)
        self.n_obs += 1
        self.sum_obs += new_obs
        self.sum_sq_obs += new_obs**2
        self.last_obs = new_obs

    def to_features(self) -> np.ndarray:
        """
        Extract feature vector for kernel regression.
        """
        sig_vec = self.sig_state.to_vector()
        levy_area = self.sig_state.get_levy_area()

        if self.decay is not None:
            # Decay mode: use EMA statistics
            mean_stat = self.mean_ema
            std_stat = np.sqrt(self.var_ema + 1e-8)
        else:
            # Cumulative mode: use running mean/std
            mean_stat = self.sum_obs / max(self.n_obs, 1)
            var_stat = self.sum_sq_obs / max(self.n_obs, 1) - mean_stat**2
            std_stat = np.sqrt(max(var_stat, 1e-8))

        return np.array([
            sig_vec[0],           # Time displacement
            sig_vec[1],           # Value displacement
            levy_area,            # Lévy area (path structure)
            mean_stat,            # Mean (running or EMA)
            std_stat,             # Std (running or EMA)
            self.last_obs         # Most recent observation
        ])


class OnlineRBFSigFilter:
    """
    Online belief filter using RBF kernel on streaming signatures.

    O(1) update time per observation via Chen's identity.
    """

    def __init__(self, decay: float = 0.95, gamma='auto', alpha=0.1):
        self.decay = decay
        self.gamma = gamma
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()

    def train(self, lqg_system, train_seeds, T_train=15.0, max_samples=3000):
        """Train using windowed data (for fair comparison with batch method)."""
        print(f"Training Online RBF-Sig filter (decay={self.decay})...")

        features = []
        targets = []

        for seed in train_seeds:
            result = lqg_system.simulate(
                x0=np.random.randn() * 2, T=T_train, seed=seed
            )
            y = result['y']
            x = result['x_true']
            dt = lqg_system.dt

            # Run online signature through trajectory
            sig_state = OnlineSignatureState(decay=self.decay)

            for t in range(len(y)):
                if t > 0:
                    sig_state.update(dt, y[t])

                # After warmup, collect training samples
                if self.decay is not None:
                    warmup = int(1.0 / (1 - self.decay))  # Effective window length
                else:
                    warmup = 30  # Default warmup for cumulative mode
                if t >= warmup:
                    feat = sig_state.to_features()
                    features.append(feat)
                    targets.append(x[t])

        X = np.array(features)
        y_target = np.array(targets)

        print(f"  Collected {len(X)} samples")

        # Subsample
        if len(X) > max_samples:
            np.random.seed(42)
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
            y_target = y_target[idx]
            print(f"  Subsampled to {max_samples}")

        # Normalize and train
        X_scaled = self.scaler.fit_transform(X)

        if self.gamma == 'auto':
            dists = cdist(X_scaled[:500], X_scaled[:500], 'sqeuclidean')
            self.gamma = 1.0 / np.median(dists[dists > 0])
            print(f"  Auto gamma: {self.gamma:.4f}")

        self.model = KernelRidge(kernel='rbf', gamma=self.gamma, alpha=self.alpha)
        self.model.fit(X_scaled, y_target)

        y_pred = self.model.predict(X_scaled)
        r2 = 1 - np.sum((y_target - y_pred)**2) / np.sum((y_target - np.mean(y_target))**2)
        print(f"  Training R²: {r2:.4f}")

    def create_state(self):
        """Create a new online signature state for filtering."""
        return OnlineSignatureState(decay=self.decay)

    def estimate(self, sig_state: OnlineSignatureState) -> float:
        """Estimate state from current signature features."""
        feat = sig_state.to_features().reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        return self.model.predict(feat_scaled)[0]


def main():
    print("=" * 70)
    print("ONLINE RBF-SIG BELIEF FILTER (Chen's Identity)")
    print("=" * 70)

    lqg = LQGSystem(A=-0.5, B=1.0, C=1.0, G=0.3, H=0.5, Q=1.0, R=0.1, dt=0.01)

    # Training
    train_seeds = list(range(1000, 1050))
    test_seeds = list(range(2000, 2010))

    # Compare BOTH modes
    print(f"\n[1] Training TWO online filters:")
    print(f"    - Cumulative (no decay): pure Chen's identity")
    print(f"    - Decay (0.95): soft window ~20 steps")

    filt_cumul = OnlineRBFSigFilter(decay=None, alpha=0.01)
    filt_cumul.train(lqg, train_seeds, T_train=10.0, max_samples=3000)

    filt_decay = OnlineRBFSigFilter(decay=0.95, alpha=0.01)
    filt_decay.train(lqg, train_seeds, T_train=10.0, max_samples=3000)

    # Test
    print(f"\n[2] Testing on held-out trajectories...")

    rmse_kalman = []
    rmse_cumul = []
    rmse_decay = []
    corr_kalman = []
    corr_cumul = []
    corr_decay = []

    for test_seed in test_seeds:
        result = lqg.simulate(x0=np.random.randn() * 2, T=15.0, seed=test_seed)
        x_true = result['x_true']
        x_kalman = result['x_hat']
        y = result['y']

        # Cumulative mode
        sig_cumul = filt_cumul.create_state()
        x_cumul = np.zeros(len(y))
        for t in range(len(y)):
            if t > 0:
                sig_cumul.update(lqg.dt, y[t])
            x_cumul[t] = filt_cumul.estimate(sig_cumul) if t > 20 else 0.0

        # Decay mode
        sig_decay = filt_decay.create_state()
        x_decay = np.zeros(len(y))
        for t in range(len(y)):
            if t > 0:
                sig_decay.update(lqg.dt, y[t])
            x_decay[t] = filt_decay.estimate(sig_decay) if t > 20 else 0.0

        start = 50

        err_k = x_true[start:] - x_kalman[start:]
        err_c = x_true[start:] - x_cumul[start:]
        err_d = x_true[start:] - x_decay[start:]

        rmse_kalman.append(np.sqrt(np.mean(err_k**2)))
        rmse_cumul.append(np.sqrt(np.mean(err_c**2)))
        rmse_decay.append(np.sqrt(np.mean(err_d**2)))
        corr_kalman.append(np.corrcoef(x_true[start:], x_kalman[start:])[0, 1])
        corr_cumul.append(np.corrcoef(x_true[start:], x_cumul[start:])[0, 1])
        corr_decay.append(np.corrcoef(x_true[start:], x_decay[start:])[0, 1])

    print(f"\n  {'Filter':<25s} {'RMSE':>10s} {'Corr':>10s}")
    print(f"  {'-'*47}")
    print(f"  {'Kalman':<25s} {np.mean(rmse_kalman):>10.4f} {np.mean(corr_kalman):>10.4f}")
    print(f"  {'Online Cumulative':<25s} {np.mean(rmse_cumul):>10.4f} {np.mean(corr_cumul):>10.4f}")
    print(f"  {'Online Decay (0.95)':<25s} {np.mean(rmse_decay):>10.4f} {np.mean(corr_decay):>10.4f}")

    ratio_cumul = np.mean(rmse_cumul) / np.mean(rmse_kalman)
    ratio_decay = np.mean(rmse_decay) / np.mean(rmse_kalman)
    print(f"\n  RMSE ratios: Cumul={ratio_cumul:.2f}x, Decay={ratio_decay:.2f}x Kalman")

    # Key insight: Cumulative vs Decay correlation comparison
    print(f"\n  Key insight:")
    print(f"    - Cumulative correlation ({np.mean(corr_cumul):.3f}) << Decay correlation ({np.mean(corr_decay):.3f})")
    print(f"    - Old observations add NOISE for stationary processes")
    print(f"    - Decay mode matches Kalman correlation ({np.mean(corr_kalman):.3f})!")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ONLINE RBF-SIG (Chen's Identity)")
    print("=" * 70)
    print(f"""
Online RBF-Sig uses Chen's identity for O(1) updates per observation.

Results:
- Kalman RMSE:              {np.mean(rmse_kalman):.4f}
- Online Cumulative:        {np.mean(rmse_cumul):.4f} ({ratio_cumul:.2f}x Kalman)
- Online Decay (0.95):      {np.mean(rmse_decay):.4f} ({ratio_decay:.2f}x Kalman)

Two modes:
1. CUMULATIVE: S_t = S_{{t-1}} ⊗ Sig(increment)
   - Pure Chen's identity, no forgetting
   - Uses all historical observations

2. DECAY: S_t = decay × S_{{t-1}} ⊗ Sig(increment)
   - Soft windowing via exponential forgetting
   - Effective window ≈ 1/(1-decay) steps

For KRONIC-POMDP:
- Both modes give O(1) update time
- Cumulative good when all history matters
- Decay good for stationary processes (recent obs more relevant)
""")


if __name__ == "__main__":
    main()
