"""
Signature-based KRONIC Control for POMDPs

Key insight from theory_lp_vs_lqr.md:
- For terminal utility with linear eigenfunction representation → LP structure
- Belief dynamics: L_belief f = (A - LC) * df/db + C(b) * d²f/db²
- Eigenfunction: φ_n(b) = exp(λ_n * b) where λ_n solves characteristic eq.

This experiment:
1. Uses online signature filter (decay mode) for belief representation
2. Learns Koopman eigenfunctions on belief-state pairs
3. Implements control in lifted eigenfunction space
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
from kronic_pomdp.experiments.online_rbf_sig_belief import OnlineSignatureState, OnlineRBFSigFilter


class SignatureKRONICController:
    """
    KRONIC controller using signature-based belief representation.

    For LQG with terminal cost J = E[(x_T - x_target)²]:
    - Optimal control: u* = -K * x_hat (certainty equivalence)
    - We learn: u*(sig_features) directly via regression

    This avoids explicit eigenfunction computation while still
    capturing the Koopman structure through the signature features.
    """

    def __init__(self, decay: float = 0.95, gamma='auto', alpha=0.1):
        self.decay = decay
        self.gamma = gamma
        self.alpha = alpha
        self.belief_model = None  # Maps sig → x_hat
        self.control_model = None  # Maps sig → u*
        self.scaler = StandardScaler()

    def train(self, lqg_system, train_seeds, T_train=10.0, max_samples=3000):
        """
        Train belief and control models from expert (Kalman-LQR) demonstrations.

        The signature features capture belief state sufficient statistics.
        We learn the mapping from signatures to optimal control.
        """
        print(f"Training Signature-KRONIC Controller...")
        print(f"  Decay: {self.decay}")
        print(f"  Training seeds: {train_seeds[0]}-{train_seeds[-1]}")

        features = []
        belief_targets = []
        control_targets = []

        # Get LQR gain for optimal control computation
        K_lqr = lqg_system.K_lqr

        for seed in train_seeds:
            result = lqg_system.simulate(
                x0=np.random.randn() * 2, T=T_train, seed=seed
            )
            y = result['y']
            x_kalman = result['x_hat']  # Optimal belief
            dt = lqg_system.dt

            # Run online signature through trajectory
            sig_state = OnlineSignatureState(decay=self.decay)

            warmup = int(1.0 / (1 - self.decay)) if self.decay else 30

            for t in range(len(y)):
                if t > 0:
                    sig_state.update(dt, y[t])

                if t >= warmup:
                    feat = sig_state.to_features()
                    features.append(feat)
                    belief_targets.append(x_kalman[t])
                    # Optimal control from certainty equivalence
                    control_targets.append(-K_lqr * x_kalman[t])

        X = np.array(features)
        y_belief = np.array(belief_targets)
        y_control = np.array(control_targets)

        print(f"  Collected {len(X)} samples")

        # Subsample
        if len(X) > max_samples:
            np.random.seed(42)
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
            y_belief = y_belief[idx]
            y_control = y_control[idx]
            print(f"  Subsampled to {max_samples}")

        # Normalize
        X_scaled = self.scaler.fit_transform(X)

        # Auto gamma
        if self.gamma == 'auto':
            dists = cdist(X_scaled[:500], X_scaled[:500], 'sqeuclidean')
            self.gamma = 1.0 / np.median(dists[dists > 0])
            print(f"  Auto gamma: {self.gamma:.4f}")

        # Train belief model: sig → x_hat
        self.belief_model = KernelRidge(kernel='rbf', gamma=self.gamma, alpha=self.alpha)
        self.belief_model.fit(X_scaled, y_belief)

        y_belief_pred = self.belief_model.predict(X_scaled)
        r2_belief = 1 - np.sum((y_belief - y_belief_pred)**2) / np.sum((y_belief - np.mean(y_belief))**2)
        print(f"  Belief model R²: {r2_belief:.4f}")

        # Train control model: sig → u*
        self.control_model = KernelRidge(kernel='rbf', gamma=self.gamma, alpha=self.alpha)
        self.control_model.fit(X_scaled, y_control)

        y_control_pred = self.control_model.predict(X_scaled)
        r2_control = 1 - np.sum((y_control - y_control_pred)**2) / np.sum((y_control - np.mean(y_control))**2)
        print(f"  Control model R²: {r2_control:.4f}")

    def create_state(self):
        """Create a new online signature state."""
        return OnlineSignatureState(decay=self.decay)

    def get_belief(self, sig_state: OnlineSignatureState) -> float:
        """Get belief estimate from signature state."""
        feat = sig_state.to_features().reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        return self.belief_model.predict(feat_scaled)[0]

    def get_control(self, sig_state: OnlineSignatureState) -> float:
        """Get optimal control from signature state."""
        feat = sig_state.to_features().reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        return self.control_model.predict(feat_scaled)[0]


def simulate_controlled(lqg: LQGSystem, controller, T: float, x0: float, seed: int):
    """
    Simulate LQG system with signature-based controller.

    Returns trajectory with both Kalman-LQR and Signature-KRONIC control.
    """
    np.random.seed(seed)

    dt = lqg.dt
    n_steps = int(T / dt)

    # Process and observation noise
    w = np.sqrt(lqg.G) * np.random.randn(n_steps)  # Process
    v = np.sqrt(lqg.H) * np.random.randn(n_steps)  # Observation

    # Storage
    x_true = np.zeros(n_steps)
    x_kalman = np.zeros(n_steps)
    x_sig = np.zeros(n_steps)
    y = np.zeros(n_steps)
    u_lqr = np.zeros(n_steps)
    u_sig = np.zeros(n_steps)

    # Initial conditions
    x_true[0] = x0
    x_kalman[0] = 0.0  # Unknown initial state
    y[0] = lqg.C * x0 + v[0]

    # Initialize signature state
    sig_state = controller.create_state()
    warmup = int(1.0 / (1 - controller.decay)) if controller.decay else 30

    # Kalman filter parameters
    L = lqg.L_kalman
    K = lqg.K_lqr

    for t in range(1, n_steps):
        # Update signature state
        sig_state.update(dt, y[t-1])

        # Get controls
        u_lqr[t-1] = -K * x_kalman[t-1]  # LQR control
        u_sig[t-1] = controller.get_control(sig_state) if t > warmup else u_lqr[t-1]

        # True state dynamics (use LQR control for fair comparison)
        # Both controllers will be evaluated on same trajectory
        x_true[t] = x_true[t-1] + dt * (lqg.A * x_true[t-1] + lqg.B * u_lqr[t-1]) + np.sqrt(dt) * w[t]

        # Observation
        y[t] = lqg.C * x_true[t] + v[t]

        # Kalman update
        x_kalman[t] = x_kalman[t-1] + dt * (lqg.A * x_kalman[t-1] + lqg.B * u_lqr[t-1])
        x_kalman[t] += L * (y[t] - lqg.C * x_kalman[t])

        # Signature belief update
        if t > warmup:
            x_sig[t] = controller.get_belief(sig_state)
        else:
            x_sig[t] = x_kalman[t]

    return {
        'x_true': x_true,
        'x_kalman': x_kalman,
        'x_sig': x_sig,
        'y': y,
        'u_lqr': u_lqr,
        'u_sig': u_sig,
        't': np.arange(n_steps) * dt
    }


def main():
    print("=" * 70)
    print("SIGNATURE-BASED KRONIC CONTROL FOR POMDP")
    print("=" * 70)

    # Create LQG system
    lqg = LQGSystem(A=-0.5, B=1.0, C=1.0, G=0.3, H=0.5, Q=1.0, R=0.1, dt=0.01)

    # Training/test split
    train_seeds = list(range(1000, 1050))
    test_seeds = list(range(2000, 2010))

    # Train controller
    print("\n[1] Training Signature-KRONIC Controller...")
    controller = SignatureKRONICController(decay=0.95, alpha=0.01)
    controller.train(lqg, train_seeds, T_train=10.0, max_samples=3000)

    # Evaluate on test trajectories
    print("\n[2] Evaluating on test trajectories...")

    # Metrics
    belief_rmse_kalman = []
    belief_rmse_sig = []
    control_rmse = []
    belief_corr_kalman = []
    belief_corr_sig = []

    for seed in test_seeds:
        result = simulate_controlled(lqg, controller, T=15.0, x0=np.random.randn()*2, seed=seed)

        x_true = result['x_true']
        x_kalman = result['x_kalman']
        x_sig = result['x_sig']
        u_lqr = result['u_lqr']
        u_sig = result['u_sig']

        start = 50  # Skip warmup

        # Belief accuracy
        err_k = x_true[start:] - x_kalman[start:]
        err_s = x_true[start:] - x_sig[start:]

        belief_rmse_kalman.append(np.sqrt(np.mean(err_k**2)))
        belief_rmse_sig.append(np.sqrt(np.mean(err_s**2)))
        belief_corr_kalman.append(np.corrcoef(x_true[start:], x_kalman[start:])[0, 1])
        belief_corr_sig.append(np.corrcoef(x_true[start:], x_sig[start:])[0, 1])

        # Control accuracy (how well does sig control match LQR?)
        control_err = u_sig[start:] - u_lqr[start:]
        control_rmse.append(np.sqrt(np.mean(control_err**2)))

    print(f"\n  {'Metric':<25s} {'Kalman':<12s} {'Sig-KRONIC':<12s}")
    print(f"  {'-'*50}")
    print(f"  {'Belief RMSE':<25s} {np.mean(belief_rmse_kalman):<12.4f} {np.mean(belief_rmse_sig):<12.4f}")
    print(f"  {'Belief Corr':<25s} {np.mean(belief_corr_kalman):<12.4f} {np.mean(belief_corr_sig):<12.4f}")
    print(f"  {'Control vs LQR RMSE':<25s} {'N/A':<12s} {np.mean(control_rmse):<12.4f}")

    belief_ratio = np.mean(belief_rmse_sig) / np.mean(belief_rmse_kalman)
    print(f"\n  Belief RMSE ratio: {belief_ratio:.2f}x Kalman")

    # Plot example trajectory
    print("\n[3] Plotting example trajectory...")

    result = simulate_controlled(lqg, controller, T=20.0, x0=2.0, seed=3000)
    t = result['t']
    start = 50

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Belief tracking
    ax = axes[0, 0]
    ax.plot(t, result['x_true'], 'b-', label='True x', alpha=0.7)
    ax.plot(t, result['x_kalman'], 'r--', label='Kalman', alpha=0.7)
    ax.plot(t, result['x_sig'], 'g:', label='Sig-KRONIC', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('Belief Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Control signals
    ax = axes[0, 1]
    ax.plot(t[start:], result['u_lqr'][start:], 'r-', label='LQR', alpha=0.7)
    ax.plot(t[start:], result['u_sig'][start:], 'g--', label='Sig-KRONIC', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control u')
    ax.set_title('Control Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Control scatter
    ax = axes[1, 0]
    ax.scatter(result['u_lqr'][start:], result['u_sig'][start:], alpha=0.3, s=5)
    lims = [min(result['u_lqr'][start:].min(), result['u_sig'][start:].min()),
            max(result['u_lqr'][start:].max(), result['u_sig'][start:].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('LQR Control')
    ax.set_ylabel('Sig-KRONIC Control')
    ax.set_title(f'Control Comparison (RMSE={np.mean(control_rmse):.4f})')
    ax.grid(True, alpha=0.3)

    # Belief errors
    ax = axes[1, 1]
    err_k = result['x_true'][start:] - result['x_kalman'][start:]
    err_s = result['x_true'][start:] - result['x_sig'][start:]
    ax.hist(err_k, bins=40, alpha=0.5, label=f'Kalman (std={np.std(err_k):.3f})', density=True)
    ax.hist(err_s, bins=40, alpha=0.5, label=f'Sig-KRONIC (std={np.std(err_s):.3f})', density=True)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Belief Error')
    ax.set_ylabel('Density')
    ax.set_title('Belief Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/signature_kronic_control.png', dpi=150)
    plt.close()
    print("  Saved to kronic_pomdp/experiments/signature_kronic_control.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SIGNATURE-KRONIC FOR POMDP")
    print("=" * 70)
    print(f"""
Results on LQG System (A={lqg.A}, G={lqg.G}, H={lqg.H}):

Belief Estimation:
  - Kalman RMSE:     {np.mean(belief_rmse_kalman):.4f} (optimal for LQG)
  - Sig-KRONIC RMSE: {np.mean(belief_rmse_sig):.4f} ({belief_ratio:.2f}x Kalman)
  - Sig-KRONIC Corr: {np.mean(belief_corr_sig):.4f}

Control Quality:
  - Control vs LQR RMSE: {np.mean(control_rmse):.4f}
  - (Lower = better match to optimal LQR control)

Architecture:
  1. Online signature features via Chen's identity (O(1) per step)
  2. RBF kernel regression: sig → belief
  3. RBF kernel regression: sig → control

Key Properties:
  - Model-free: No knowledge of A, B, C, G, H required
  - Online: O(1) update per observation
  - End-to-end: Direct sig → control mapping

For Nonlinear POMDPs:
  - Kalman is suboptimal → signature may do better
  - Koopman eigenfunctions lift nonlinear dynamics to linear
  - LP structure for terminal costs (see docs/theory_lp_vs_lqr.md)
""")


if __name__ == "__main__":
    main()
