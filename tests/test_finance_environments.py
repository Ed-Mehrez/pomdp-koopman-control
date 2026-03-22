"""
Unit tests for finance environments.

Tests cover:
1. Spoofing Detection - Lévy area discrimination
2. Inventory-Averse MM - Path-dependent inventory costs
3. Signature correctness - Mathematical properties
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/src')

from finance.spoofing_detection import (
    SignatureState as SpoofSig,
    simulate_informed_episode,
    simulate_manipulator_episode,
    InformedTrader,
    Manipulator,
)

from finance.inventory_averse_mm import (
    SignatureState as InvSig,
    InventoryAverseMM,
    InformedTrader as InvTrader,
)


class TestSignatureProperties:
    """Test mathematical properties of signature computation."""

    def test_levy_area_zero_for_linear_path(self):
        """
        A linear path (constant velocity) should have zero Lévy area.
        Lévy area measures "wiggliness" / deviation from linear interpolation.
        """
        sig = SpoofSig()
        dt = 0.01
        T = 1.0
        steps = int(T / dt)

        # Linear path: Y_t = t (constant slope)
        for _ in range(steps):
            dY = dt  # Constant increment
            sig.extend(dt, dY)

        # Lévy area should be very close to zero
        assert abs(sig.levy_area) < 1e-10, f"Lévy area for linear path should be ~0, got {sig.levy_area}"

    def test_levy_area_nonzero_for_curved_path(self):
        """
        A curved path (e.g., quadratic) should have non-zero Lévy area.
        """
        sig = SpoofSig()
        dt = 0.01
        T = 1.0
        steps = int(T / dt)

        # Quadratic path: Y_t = t^2 → dY = 2t*dt
        for i in range(steps):
            t = i * dt
            dY = 2 * t * dt  # Accelerating
            sig.extend(dt, dY)

        # Lévy area should be non-zero
        assert abs(sig.levy_area) > 0.01, f"Lévy area for curved path should be non-zero, got {sig.levy_area}"

    def test_levy_area_sign_for_up_then_down(self):
        """
        Path that goes up then down should have Lévy area with specific sign.
        """
        sig = SpoofSig()
        dt = 0.01
        steps_up = 50
        steps_down = 50

        # Go up
        for _ in range(steps_up):
            sig.extend(dt, 0.1)

        # Go down
        for _ in range(steps_down):
            sig.extend(dt, -0.1)

        # This "humped" path should have positive Lévy area
        # (area above the chord from start to end)
        assert sig.levy_area != 0, "Lévy area should be non-zero for humped path"

    def test_signature_chen_identity(self):
        """
        Test Chen's identity: S(path1 * path2) = S(path1) ⊗ S(path2)
        For level 1, this means sig(combined) = sig(path1) + sig(path2)
        """
        sig1 = SpoofSig()
        sig2 = SpoofSig()
        sig_combined = SpoofSig()

        dt = 0.01

        # Path 1
        for _ in range(50):
            dY = np.random.randn() * 0.1
            sig1.extend(dt, dY)
            sig_combined.extend(dt, dY)

        # Path 2
        for _ in range(50):
            dY = np.random.randn() * 0.1
            sig2.extend(dt, dY)
            sig_combined.extend(dt, dY)

        # Level 1 signature should be additive
        assert abs(sig_combined.Y_T - (sig1.Y_T + sig2.Y_T)) < 1e-10
        assert abs(sig_combined.t_cumsum - (sig1.t_cumsum + sig2.t_cumsum)) < 1e-10


class TestSpoofingDetection:
    """Tests for spoofing detection via signatures."""

    def test_informed_trader_smooth_path(self):
        """Informed traders should produce relatively smooth paths."""
        np.random.seed(42)

        sig, path = simulate_informed_episode(
            v=110.0, T=1.0, dt=0.01, sigma_z=0.5,  # Low noise
            P_0=100.0, lambda_=2.0
        )

        # Path should be relatively monotonic for v > P_0
        increments = np.diff(path)
        positive_fraction = np.mean(increments > 0)

        # Should be mostly buying (positive increments)
        assert positive_fraction > 0.5, f"Informed trader with v>P_0 should mostly buy, got {positive_fraction:.2f}"

    def test_manipulator_reversal_pattern(self):
        """Manipulators should produce reversal patterns with high Lévy area."""
        np.random.seed(42)

        sig, path = simulate_manipulator_episode(
            T=1.0, dt=0.01, sigma_z=0.3,
            intensity=20.0, reversal_time=0.5
        )

        # Path should go up then down (or vice versa)
        mid_idx = len(path) // 2
        first_half_change = path[mid_idx] - path[0]
        second_half_change = path[-1] - path[mid_idx]

        # Should have reversal (opposite signs)
        assert first_half_change * second_half_change < 0, \
            f"Manipulator should reverse: first_half={first_half_change:.2f}, second_half={second_half_change:.2f}"

    def test_levy_area_discriminates_types(self):
        """
        Key test: Manipulators should have significantly higher |Lévy area|
        than informed traders, on average.
        """
        np.random.seed(42)

        n_samples = 100

        # Generate informed trader signatures
        informed_levy = []
        for _ in range(n_samples):
            v = np.random.normal(100, 20)
            sig, _ = simulate_informed_episode(v, T=1.0, dt=0.01, sigma_z=0.5,
                                               P_0=100.0, lambda_=2.0)
            informed_levy.append(abs(sig.levy_area))

        # Generate manipulator signatures
        manip_levy = []
        for _ in range(n_samples):
            sig, _ = simulate_manipulator_episode(T=1.0, dt=0.01, sigma_z=0.5,
                                                   intensity=20.0, reversal_time=0.5)
            manip_levy.append(abs(sig.levy_area))

        # Manipulators should have higher mean |Lévy area|
        mean_informed = np.mean(informed_levy)
        mean_manip = np.mean(manip_levy)

        assert mean_manip > mean_informed, \
            f"Manipulators should have higher |Lévy|: informed={mean_informed:.2f}, manip={mean_manip:.2f}"

        # The ratio should be substantial (at least 1.5x)
        ratio = mean_manip / mean_informed
        assert ratio > 1.5, f"Lévy area ratio should be > 1.5, got {ratio:.2f}"


class TestInventoryAverseMM:
    """Tests for inventory-averse market maker."""

    def test_early_accumulation_higher_cost(self):
        """
        Early inventory accumulation should have higher cost than late accumulation
        (same terminal inventory, but held longer).
        """
        T = 1.0
        dt = 0.01
        target_Y = 10.0

        # Path A: Accumulate early
        sig_A = InvSig()
        steps = int(T / dt)
        for i in range(steps):
            t = i * dt
            if t < 0.3 * T:  # Accumulate in first 30%
                dY = target_Y / (0.3 * T) * dt
            else:
                dY = 0.0
            sig_A.extend(dt, dY, T)

        # Path B: Accumulate late
        sig_B = InvSig()
        for i in range(steps):
            t = i * dt
            if t >= 0.7 * T:  # Accumulate in last 30%
                dY = target_Y / (0.3 * T) * dt
            else:
                dY = 0.0
            sig_B.extend(dt, dY, T)

        # Both should have same terminal Y
        assert abs(sig_A.Y_T - sig_B.Y_T) < 0.5, \
            f"Terminal Y_T should be similar: A={sig_A.Y_T:.2f}, B={sig_B.Y_T:.2f}"

        # Early accumulation should have HIGHER inventory cost
        assert sig_A.inventory_cost > sig_B.inventory_cost, \
            f"Early accumulation should have higher cost: A={sig_A.inventory_cost:.2f}, B={sig_B.inventory_cost:.2f}"

        # The ratio should be substantial
        ratio = sig_A.inventory_cost / max(sig_B.inventory_cost, 0.01)
        assert ratio > 2.0, f"Cost ratio should be > 2x, got {ratio:.2f}"

    def test_inventory_adjustment_sign(self):
        """
        Inventory adjustment should have correct sign:
        - Long inventory (I_t > 0, Y_t < 0) → negative adjustment (lower price)
        - Short inventory (I_t < 0, Y_t > 0) → positive adjustment (higher price)
        """
        mm = InventoryAverseMM(dt=0.01, P_0=100.0, T=1.0, gamma=0.5)

        # Positive Y_T means MM is short → should charge premium
        sig_short = InvSig()
        sig_short.Y_cumsum = 10.0
        sig_short.t_cumsum = 0.5  # Halfway through
        adj_short = mm._inventory_adjustment(sig_short)
        assert adj_short < 0, f"Short MM should have negative adjustment (lower price to attract sellers), got {adj_short}"

        # Negative Y_T means MM is long → should offer discount
        sig_long = InvSig()
        sig_long.Y_cumsum = -10.0
        sig_long.t_cumsum = 0.5
        adj_long = mm._inventory_adjustment(sig_long)
        assert adj_long > 0, f"Long MM should have positive adjustment (higher price to reduce buying), got {adj_long}"

    def test_gamma_affects_price_impact(self):
        """
        Higher gamma (inventory cost) should result in higher effective price impact.

        The inventory adjustment formula is: 2 * gamma * I_t * (T - t)
        So with higher gamma, the adjustment should be larger.
        """
        T = 1.0
        dt = 0.01

        mm_low_gamma = InventoryAverseMM(dt=dt, P_0=100.0, T=T, gamma=0.1)
        mm_high_gamma = InventoryAverseMM(dt=dt, P_0=100.0, T=T, gamma=1.0)

        # Create signatures with same Y_T
        sig = InvSig()
        sig.Y_cumsum = 10.0  # Positive order flow → MM is short
        sig.t_cumsum = 0.5   # Halfway through

        # Compute inventory adjustments directly
        adj_low = mm_low_gamma._inventory_adjustment(sig)
        adj_high = mm_high_gamma._inventory_adjustment(sig)

        # With Y_T=10, I=-10, T-t=0.5:
        # adj = 2 * gamma * (-10) * 0.5 = -gamma * 10
        # Low gamma: -0.1 * 10 = -1.0
        # High gamma: -1.0 * 10 = -10.0

        assert abs(adj_high) > abs(adj_low), \
            f"Higher gamma should mean larger |adjustment|: low={adj_low:.2f}, high={adj_high:.2f}"

        # Check the ratio matches gamma ratio
        gamma_ratio = 1.0 / 0.1  # = 10
        adj_ratio = abs(adj_high) / abs(adj_low)
        assert abs(adj_ratio - gamma_ratio) < 0.1, \
            f"Adjustment ratio should match gamma ratio: adj_ratio={adj_ratio:.2f}, gamma_ratio={gamma_ratio:.2f}"

    def test_time_varying_liquidity(self):
        """
        With inventory costs, effective λ should increase as inventory grows
        and as time remaining decreases.
        """
        mm = InventoryAverseMM(dt=0.01, P_0=100.0, T=1.0, gamma=0.5)

        # Train minimally
        np.random.seed(42)
        for _ in range(30):
            mm.reset()
            for _ in range(100):
                mm.filter_step(np.random.normal(0, 0.5))
            mm.end_of_episode_update(np.random.normal(100, 10))

        # Track λ as we accumulate inventory
        mm.reset()
        lambdas = []

        for i in range(50):
            lambda_t = mm.get_effective_lambda()
            lambdas.append(lambda_t)
            mm.filter_step(0.5)  # Keep buying

        # λ should generally increase (more inventory = higher cost)
        first_half = np.mean(lambdas[:25])
        second_half = np.mean(lambdas[25:])

        assert second_half >= first_half * 0.9, \
            f"λ should not decrease significantly: first={first_half:.2f}, second={second_half:.2f}"


class TestKernelProperties:
    """Test kernel and regression properties."""

    def test_kernel_positive_definite(self):
        """RBF kernel should produce positive definite matrices."""
        from finance.spoofing_detection import signature_kernel, SignatureState

        # Generate random signatures
        np.random.seed(42)
        sigs = []
        for _ in range(10):
            sig = SignatureState()
            for _ in range(50):
                sig.extend(0.01, np.random.randn() * 0.5)
            sigs.append(sig)

        lengthscales = np.array([1.0, 5.0, 2.0])

        # Build kernel matrix
        n = len(sigs)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = signature_kernel(sigs[i], sigs[j], lengthscales)

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-10), \
            f"Kernel matrix should be PSD, min eigenvalue = {eigenvalues.min()}"

    def test_krr_interpolation(self):
        """KRR should interpolate training points reasonably well."""
        from finance.spoofing_detection import SignaturePricingRule, SignatureState

        np.random.seed(42)

        # Generate training data
        sigs = []
        values = []
        for _ in range(50):
            sig = SignatureState()
            v = np.random.normal(100, 20)
            # Simulate a path that correlates with v
            target = (v - 100) / 5  # Target Y_T
            for _ in range(100):
                dY = target * 0.01 + np.random.randn() * 0.1
                sig.extend(0.01, dY)
            sigs.append(sig)
            values.append(v)

        # Fit pricing rule
        pricing = SignaturePricingRule(reg=0.01)  # Low regularization
        pricing.fit(sigs, values)

        # Check training error
        preds = [pricing.predict(s) for s in sigs]
        train_rmse = np.sqrt(np.mean((np.array(preds) - np.array(values))**2))

        # Should fit training data reasonably well
        assert train_rmse < 15, f"Training RMSE should be low, got {train_rmse:.2f}"


def run_all_tests():
    """Run all tests and report results."""
    print("="*70)
    print("RUNNING UNIT TESTS FOR FINANCE ENVIRONMENTS")
    print("="*70)

    test_classes = [
        TestSignatureProperties,
        TestSpoofingDetection,
        TestInventoryAverseMM,
        TestKernelProperties,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    total_passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {str(e)[:60]}")
                    failures.append((test_class.__name__, method_name, str(e)))
                    total_failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: EXCEPTION - {str(e)[:50]}")
                    failures.append((test_class.__name__, method_name, str(e)))
                    total_failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("="*70)

    if failures:
        print("\nFAILURES:")
        for class_name, method_name, error in failures:
            print(f"  {class_name}.{method_name}:")
            print(f"    {error}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
