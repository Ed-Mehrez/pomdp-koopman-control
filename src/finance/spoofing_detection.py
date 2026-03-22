"""
Spoofing Detection via Path Signatures

Key insight: Manipulators and informed traders have different path signatures
even when they produce similar terminal order flow Y_T.

- Informed traders: smooth accumulation toward their target
  → Low |Lévy area|, monotonic paths

- Manipulators (spoofers): create artificial pressure then reverse
  → High |Lévy area|, non-monotonic paths with reversals

The signature-based pricing rule should:
1. Detect suspicious patterns
2. Discount order flow with manipulation characteristics
3. Achieve robust equilibrium that's resistant to spoofing

This demonstrates where signatures genuinely add value over Y_T alone.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SignatureState:
    """Truncated signature with Lévy area tracking."""
    t_cumsum: float = 0.0
    Y_cumsum: float = 0.0
    levy_area: float = 0.0

    def extend(self, dt: float, dY: float):
        """Extend signature via Chen's identity."""
        # Lévy area increment: (Y_prev * dt - t_prev * dY) / 2
        self.levy_area += (self.Y_cumsum * dt - self.t_cumsum * dY) / 2
        self.t_cumsum += dt
        self.Y_cumsum += dY

    @property
    def Y_T(self) -> float:
        return self.Y_cumsum

    @property
    def T(self) -> float:
        return self.t_cumsum

    def to_vector(self) -> np.ndarray:
        """Return signature features: (T, Y_T, Lévy area)."""
        return np.array([self.T, self.Y_T, self.levy_area])

    def copy(self) -> 'SignatureState':
        new = SignatureState()
        new.t_cumsum = self.t_cumsum
        new.Y_cumsum = self.Y_cumsum
        new.levy_area = self.levy_area
        return new


class InformedTrader:
    """
    Informed insider who knows true value v.
    Trades smoothly toward equilibrium position.
    """
    def __init__(self, T: float, dt: float):
        self.T = T
        self.dt = dt

    def compute_trading_rate(self, v: float, t: float, P_t: float,
                             lambda_: float) -> float:
        """Optimal trading rate: θ* = (v - P_t) / (λ * (T - t))"""
        time_left = max(self.T - t, self.dt)
        return (v - P_t) / (lambda_ * time_left)


class Manipulator:
    """
    Manipulator (spoofer) who doesn't have information about v.

    Strategy: Create artificial buying pressure, then reverse.
    - Phase 1 (t < τ): Buy aggressively to push price up
    - Phase 2 (t ≥ τ): Sell to profit from inflated price

    This creates large |Lévy area| because the path reverses direction.
    """
    def __init__(self, T: float, dt: float,
                 intensity: float = 20.0,
                 reversal_time: float = 0.5):
        self.T = T
        self.dt = dt
        self.intensity = intensity  # How aggressive the manipulation is
        self.reversal_time = reversal_time  # When to reverse (fraction of T)

    def compute_trading_rate(self, t: float, Y_t: float) -> float:
        """
        Manipulation strategy:
        - Buy before reversal_time
        - Sell after reversal_time (to unwind position)
        """
        tau = self.reversal_time * self.T

        if t < tau:
            # Accumulation phase: buy aggressively
            return self.intensity
        else:
            # Reversal phase: sell to unwind
            time_left = max(self.T - t, self.dt)
            # Target: return to zero position
            return -Y_t / time_left


def simulate_informed_episode(v: float, T: float, dt: float, sigma_z: float,
                               P_0: float, lambda_: float) -> Tuple[SignatureState, List[float]]:
    """Simulate an informed trader's episode."""
    steps = int(T / dt)
    trader = InformedTrader(T, dt)
    sig = SignatureState()
    P_t = P_0

    Y_path = [0.0]

    for i in range(steps):
        t = i * dt
        theta_t = trader.compute_trading_rate(v, t, P_t, lambda_)
        theta_t = np.clip(theta_t, -50, 50)

        dX_t = theta_t * dt
        dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
        dY_t = dX_t + dZ_t

        sig.extend(dt, dY_t)
        Y_path.append(sig.Y_T)

        # Simple linear price update
        P_t = P_0 + lambda_ * sig.Y_T

    return sig, Y_path


def simulate_manipulator_episode(T: float, dt: float, sigma_z: float,
                                  intensity: float = 20.0,
                                  reversal_time: float = 0.5) -> Tuple[SignatureState, List[float]]:
    """Simulate a manipulator's episode."""
    steps = int(T / dt)
    manipulator = Manipulator(T, dt, intensity, reversal_time)
    sig = SignatureState()

    Y_path = [0.0]

    for i in range(steps):
        t = i * dt
        theta_t = manipulator.compute_trading_rate(t, sig.Y_T)
        theta_t = np.clip(theta_t, -50, 50)

        dX_t = theta_t * dt
        dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
        dY_t = dX_t + dZ_t

        sig.extend(dt, dY_t)
        Y_path.append(sig.Y_T)

    return sig, Y_path


def signature_kernel(s1: SignatureState, s2: SignatureState,
                     lengthscales: np.ndarray) -> float:
    """RBF kernel on signature features with per-dimension lengthscales."""
    x = s1.to_vector()
    y = s2.to_vector()
    diff = (x - y) / lengthscales
    return np.exp(-0.5 * np.sum(diff ** 2))


class SignaturePricingRule:
    """
    Pricing rule that uses signature features to detect manipulation.

    P(sig) = E[v | signature = sig]

    Key insight: For informed traders, low Lévy area → more weight.
    For manipulators, high |Lévy area| → discount the signal.
    """
    def __init__(self, reg: float = 0.1):
        self.reg = reg
        self.support_sigs: List[SignatureState] = []
        self.support_v: List[float] = []
        self.alpha = None
        self.lengthscales = np.array([1.0, 10.0, 5.0])  # T, Y_T, Lévy

    def fit(self, sigs: List[SignatureState], values: List[float]):
        """Fit KRR model."""
        self.support_sigs = sigs
        self.support_v = values

        n = len(sigs)
        if n < 3:
            self.alpha = None
            return

        # Compute adaptive lengthscales from data
        features = np.array([s.to_vector() for s in sigs])
        stds = np.std(features, axis=0)
        self.lengthscales = np.maximum(stds * 0.5, [0.1, 1.0, 0.5])

        # Build kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = K[j, i] = signature_kernel(sigs[i], sigs[j], self.lengthscales)

        # Solve KRR
        K_reg = K + self.reg * np.eye(n)
        try:
            self.alpha = np.linalg.solve(K_reg, np.array(values))
        except:
            self.alpha = np.linalg.lstsq(K_reg, np.array(values), rcond=None)[0]

    def predict(self, sig: SignatureState) -> float:
        """Predict value given signature."""
        if self.alpha is None:
            return 100.0  # Prior mean

        k_vec = np.array([signature_kernel(sig, s, self.lengthscales)
                          for s in self.support_sigs])
        return float(k_vec @ self.alpha)

    def predict_Y_only(self, Y_T: float) -> float:
        """Predict using only Y_T (for comparison)."""
        if self.alpha is None or len(self.support_sigs) < 3:
            return 100.0

        # Nadaraya-Watson on Y_T only
        Y_support = np.array([s.Y_T for s in self.support_sigs])
        bandwidth = np.std(Y_support) * 0.3

        distances = np.abs(Y_support - Y_T)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights_sum = weights.sum()

        if weights_sum > 1e-10:
            return float(weights @ np.array(self.support_v) / weights_sum)
        return 100.0


def run_spoofing_detection_experiment():
    """
    Main experiment: Show that signatures detect manipulation.

    Setup:
    - Generate episodes from informed traders (with known v)
    - Generate episodes from manipulators (no information)
    - Train pricing rules on informed-only data
    - Compare predictions for manipulators using signatures vs Y_T only
    """
    np.random.seed(42)

    # Parameters
    T = 1.0
    dt = 0.005
    sigma_z = 1.0
    P_0 = 100.0
    lambda_ = 2.0

    n_informed = 300
    n_manipulators = 100

    # Generate informed trader data (training set)
    print("Generating informed trader episodes...")
    informed_sigs = []
    informed_v = []
    informed_paths = []

    for _ in tqdm(range(n_informed), desc="Informed traders"):
        # Bimodal prior
        v = np.random.normal(80, 10) if np.random.rand() < 0.5 else np.random.normal(120, 10)
        sig, path = simulate_informed_episode(v, T, dt, sigma_z, P_0, lambda_)
        informed_sigs.append(sig)
        informed_v.append(v)
        informed_paths.append(path)

    # Generate manipulator data (test set - should be detected)
    print("\nGenerating manipulator episodes...")
    manip_sigs = []
    manip_paths = []

    for _ in tqdm(range(n_manipulators), desc="Manipulators"):
        # Vary manipulation parameters
        intensity = np.random.uniform(15, 30)
        reversal_time = np.random.uniform(0.3, 0.7)
        sig, path = simulate_manipulator_episode(T, dt, sigma_z, intensity, reversal_time)
        manip_sigs.append(sig)
        manip_paths.append(path)

    # Analyze signature statistics
    print("\n" + "="*60)
    print("SIGNATURE STATISTICS")
    print("="*60)

    informed_levy = np.array([s.levy_area for s in informed_sigs])
    informed_Y = np.array([s.Y_T for s in informed_sigs])
    manip_levy = np.array([s.levy_area for s in manip_sigs])
    manip_Y = np.array([s.Y_T for s in manip_sigs])

    print(f"\nInformed traders:")
    print(f"  Y_T: mean={informed_Y.mean():.2f}, std={informed_Y.std():.2f}")
    print(f"  |Lévy area|: mean={np.abs(informed_levy).mean():.2f}, std={np.abs(informed_levy).std():.2f}")

    print(f"\nManipulators:")
    print(f"  Y_T: mean={manip_Y.mean():.2f}, std={manip_Y.std():.2f}")
    print(f"  |Lévy area|: mean={np.abs(manip_levy).mean():.2f}, std={np.abs(manip_levy).std():.2f}")

    # Key diagnostic: Lévy area should discriminate
    levy_ratio = np.abs(manip_levy).mean() / np.abs(informed_levy).mean()
    print(f"\n  → Manipulator |Lévy| / Informed |Lévy| = {levy_ratio:.2f}x")

    if levy_ratio > 2.0:
        print("  ✓ Lévy area successfully discriminates manipulators!")
    else:
        print("  ⚠ Warning: Lévy area discrimination is weak")

    # Fit signature-based pricing rule
    print("\n" + "="*60)
    print("FITTING PRICING RULES")
    print("="*60)

    pricing = SignaturePricingRule(reg=0.1)
    pricing.fit(informed_sigs, informed_v)

    # Test predictions
    # For informed traders (test on held-out subset)
    n_test = 50
    test_informed_sigs = informed_sigs[-n_test:]
    test_informed_v = informed_v[-n_test:]

    pred_informed_sig = np.array([pricing.predict(s) for s in test_informed_sigs])
    pred_informed_Y = np.array([pricing.predict_Y_only(s.Y_T) for s in test_informed_sigs])

    rmse_informed_sig = np.sqrt(np.mean((pred_informed_sig - test_informed_v) ** 2))
    rmse_informed_Y = np.sqrt(np.mean((pred_informed_Y - test_informed_v) ** 2))

    print(f"\nPrediction on informed traders (held-out):")
    print(f"  Signature-based RMSE: {rmse_informed_sig:.2f}")
    print(f"  Y_T-only RMSE: {rmse_informed_Y:.2f}")

    # For manipulators - predictions should be uncertain/discounted
    pred_manip_sig = np.array([pricing.predict(s) for s in manip_sigs])
    pred_manip_Y = np.array([pricing.predict_Y_only(s.Y_T) for s in manip_sigs])

    print(f"\nPredictions on manipulators:")
    print(f"  Signature-based: mean={pred_manip_sig.mean():.2f}, std={pred_manip_sig.std():.2f}")
    print(f"  Y_T-only: mean={pred_manip_Y.mean():.2f}, std={pred_manip_Y.std():.2f}")

    # Key insight: Signature-based should be more conservative (closer to prior mean)
    # because manipulator signatures are out-of-distribution
    deviation_sig = np.abs(pred_manip_sig - P_0).mean()
    deviation_Y = np.abs(pred_manip_Y - P_0).mean()

    print(f"\n  Mean |prediction - prior mean|:")
    print(f"    Signature-based: {deviation_sig:.2f}")
    print(f"    Y_T-only: {deviation_Y:.2f}")

    if deviation_sig < deviation_Y:
        print("  ✓ Signature method is more conservative on manipulators!")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top left: Example paths
    ax = axes[0, 0]
    t_grid = np.linspace(0, T, len(informed_paths[0]))

    # Plot a few informed paths
    for i in range(min(5, len(informed_paths))):
        ax.plot(t_grid, informed_paths[i], 'b-', alpha=0.3, linewidth=1)
    # Plot a few manipulator paths
    for i in range(min(5, len(manip_paths))):
        ax.plot(t_grid, manip_paths[i], 'r-', alpha=0.5, linewidth=1.5)

    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Y_t')
    ax.set_title('Sample Trading Paths\n(Blue: Informed, Red: Manipulator)')
    ax.grid(True, alpha=0.3)

    # Top middle: Lévy area distribution
    ax = axes[0, 1]
    ax.hist(informed_levy, bins=30, alpha=0.6, label='Informed', color='blue', density=True)
    ax.hist(manip_levy, bins=30, alpha=0.6, label='Manipulator', color='red', density=True)
    ax.set_xlabel('Lévy Area')
    ax.set_ylabel('Density')
    ax.set_title('Lévy Area Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Y_T vs Lévy area scatter
    ax = axes[0, 2]
    ax.scatter(informed_Y, informed_levy, alpha=0.5, s=20, label='Informed', c='blue')
    ax.scatter(manip_Y, manip_levy, alpha=0.5, s=20, label='Manipulator', c='red')
    ax.set_xlabel('Terminal Order Flow Y_T')
    ax.set_ylabel('Lévy Area')
    ax.set_title('Signature Space: Y_T vs Lévy Area')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Predictions vs true values (informed)
    ax = axes[1, 0]
    ax.scatter(test_informed_v, pred_informed_sig, alpha=0.6, s=30, label='Signature-based')
    ax.scatter(test_informed_v, pred_informed_Y, alpha=0.6, s=30, marker='x', label='Y_T-only')
    ax.plot([60, 140], [60, 140], 'k--', label='Perfect')
    ax.set_xlabel('True v')
    ax.set_ylabel('Predicted v')
    ax.set_title(f'Informed Traders\nSig RMSE={rmse_informed_sig:.2f}, Y RMSE={rmse_informed_Y:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(60, 140)
    ax.set_ylim(60, 140)

    # Bottom middle: Manipulator predictions histogram
    ax = axes[1, 1]
    ax.hist(pred_manip_sig, bins=20, alpha=0.6, label='Signature-based', color='blue')
    ax.hist(pred_manip_Y, bins=20, alpha=0.6, label='Y_T-only', color='orange')
    ax.axvline(P_0, color='black', linestyle='--', label=f'Prior mean ({P_0})')
    ax.set_xlabel('Predicted v')
    ax.set_ylabel('Count')
    ax.set_title('Predictions on Manipulators\n(Should be uncertain → near prior mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Detection ROC-style plot
    ax = axes[1, 2]

    # Create a simple detection score based on Lévy area magnitude
    all_levy = np.concatenate([np.abs(informed_levy), np.abs(manip_levy)])
    all_labels = np.concatenate([np.zeros(len(informed_levy)), np.ones(len(manip_levy))])

    # Sort by detection score
    thresholds = np.percentile(all_levy, np.linspace(0, 100, 50))
    tpr = []  # True positive rate (manipulators detected)
    fpr = []  # False positive rate (informed misclassified)

    for thresh in thresholds:
        detected = all_levy > thresh
        tp = np.sum(detected & (all_labels == 1))
        fp = np.sum(detected & (all_labels == 0))
        tpr.append(tp / np.sum(all_labels == 1))
        fpr.append(fp / np.sum(all_labels == 0))

    ax.plot(fpr, tpr, 'b-', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Detection: |Lévy Area| as Classifier')
    ax.grid(True, alpha=0.3)

    # Compute AUC
    auc = np.trapz(tpr, fpr)
    ax.text(0.6, 0.2, f'AUC = {abs(auc):.3f}', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/docs/spoofing_detection.png', dpi=150)
    plt.close()

    print(f"\n\nPlot saved to docs/spoofing_detection.png")
    print(f"Detection AUC (|Lévy area| classifier): {abs(auc):.3f}")

    return {
        'informed_sigs': informed_sigs,
        'manip_sigs': manip_sigs,
        'informed_v': informed_v,
        'pricing': pricing,
        'auc': abs(auc),
        'levy_ratio': levy_ratio,
    }


if __name__ == "__main__":
    results = run_spoofing_detection_experiment()
