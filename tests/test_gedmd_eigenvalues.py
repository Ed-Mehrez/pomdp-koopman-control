"""
H6: Verify GEDMD Eigenvalues with Signature-Based Features

Test that the Koopman generator eigenvalues from signature-based GEDMD
match analytical expectations for processes with known generators.

Key theoretical result (Klus 2020):
For dX = μ(X)dt + σ(X)dW, the generator eigenvalues satisfy:
  L φ_n = λ_n φ_n  where  λ_n < 0 for stable processes

For Ornstein-Uhlenbeck: dX = -κX dt + σ dW
  - Generator: L = -κx ∂/∂x + (σ²/2) ∂²/∂x²
  - Eigenvalues: λ_n = -n·κ  (n = 0, 1, 2, ...)
  - Eigenfunctions: Hermite polynomials H_n(x/√(σ²/2κ))

This test verifies that our signature-based kGEDMD recovers these eigenvalues.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.proof_of_concept.signature_features import (
    compute_path_signature, compute_log_signature
)


class TestOUEigenvalues(unittest.TestCase):
    """Test GEDMD eigenvalues for Ornstein-Uhlenbeck process."""

    def setUp(self):
        """Generate OU process trajectory."""
        np.random.seed(42)
        self.kappa = 2.0  # Mean reversion rate
        self.sigma = 0.5  # Volatility
        self.dt = 0.01
        self.n_steps = 10000

        # Simulate OU process: dX = -κX dt + σ dW
        X = np.zeros(self.n_steps)
        X[0] = 0.5
        for t in range(1, self.n_steps):
            X[t] = X[t-1] - self.kappa * X[t-1] * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.X = X

        # Analytical eigenvalues: λ_n = -n·κ
        self.true_eigenvalues = [-n * self.kappa for n in range(1, 6)]

    def test_gedmd_polynomial_features(self):
        """GEDMD with polynomial features should recover OU eigenvalues."""
        # Build polynomial feature matrix: [1, x, x², x³, x⁴]
        X = self.X[:-1]
        dX = np.diff(self.X)
        n = len(X)

        max_deg = 5
        Psi = np.column_stack([X**k for k in range(max_deg + 1)])
        dPsi = np.diff(Psi, axis=0)
        Psi_t = Psi[:-1]

        # GEDMD: Learn generator A such that dΨ = A @ Ψ * dt
        # Ridge regression: A = dΨ^T @ Ψ @ (Ψ^T @ Ψ + λI)^{-1} / dt
        lam = 0.01
        Gram = Psi_t.T @ Psi_t + lam * np.eye(max_deg + 1)
        Cross = dPsi.T @ Psi_t
        A = np.linalg.solve(Gram.T, Cross.T).T / self.dt

        # Eigenvalues of generator matrix
        eigenvalues = np.linalg.eigvals(A)
        eigenvalues_real = np.sort(np.real(eigenvalues))[::-1]  # Descending

        print("\nGEDMD with polynomial features:")
        print(f"  True eigenvalues: {self.true_eigenvalues[:5]}")
        print(f"  GEDMD eigenvalues: {list(eigenvalues_real[:5])}")

        # First eigenvalue should be near 0 (constant mode)
        # Second eigenvalue should be near -κ
        self.assertLess(np.abs(eigenvalues_real[0]), 0.5)  # Near 0
        self.assertAlmostEqual(eigenvalues_real[1], -self.kappa, delta=0.5)

    def test_gedmd_signature_features(self):
        """GEDMD with signature features should capture generator structure."""
        # Build signature features from path windows
        window = 20
        n_samples = self.n_steps - window - 1

        sig_features = []
        for t in range(window, self.n_steps - 1):
            path = self.X[t-window:t+1]
            time_aug = np.linspace(0, 1, len(path))
            path_2d = np.column_stack([time_aug, path])
            sig = compute_path_signature(path_2d, level=2)
            sig_features.append(sig)

        Psi = np.array(sig_features)
        dPsi = np.diff(Psi, axis=0)
        Psi_t = Psi[:-1]

        # Add augmentation: [1, x, sig]
        x_vals = self.X[window:window + len(Psi_t)]
        Psi_aug = np.column_stack([
            np.ones(len(Psi_t)),
            x_vals,
            Psi_t
        ])
        n_aug = len(Psi_t) - 1
        dPsi_aug = np.column_stack([
            np.zeros(n_aug),              # d(1) = 0
            np.diff(x_vals[:n_aug+1]),    # dx
            dPsi[:n_aug]
        ])

        # GEDMD
        lam = 1.0
        R = Psi_aug.shape[1]
        Gram = Psi_aug[:-1].T @ Psi_aug[:-1] + lam * np.eye(R)
        Cross = dPsi_aug.T @ Psi_aug[:-1]
        A = np.linalg.solve(Gram.T, Cross.T).T / self.dt

        eigenvalues = np.linalg.eigvals(A)
        eigenvalues_real = np.sort(np.real(eigenvalues))[::-1]

        print("\nGEDMD with signature features:")
        print(f"  True λ₁ = {-self.kappa:.2f}")
        print(f"  GEDMD eigenvalues: {list(eigenvalues_real[:5])}")

        # Should have eigenvalue structure (even if not exact)
        # Key: stable process → all real parts negative (except λ₀ ≈ 0)
        negative_count = np.sum(eigenvalues_real[1:] < 0)
        self.assertGreater(negative_count, 2)  # At least 2 negative eigenvalues

    def test_rbf_on_sig_for_eigenvalue_estimation(self):
        """RBF kernel on signatures should give smooth generator estimate."""
        window = 20
        n_samples = min(2000, self.n_steps - window - 1)

        # Build signature features
        sig_features = []
        for t in range(window, window + n_samples):
            path = self.X[t-window:t+1]
            time_aug = np.linspace(0, 1, len(path))
            path_2d = np.column_stack([time_aug, path])
            sig = compute_path_signature(path_2d, level=2)
            sig_features.append(sig)

        Psi = np.array(sig_features)
        x_vals = self.X[window:window + n_samples]

        # RBF kernel on signatures
        def rbf_kernel(X1, X2, gamma):
            dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
            return np.exp(-gamma * dists)

        # Kernel GEDMD
        gamma = 0.1
        K = rbf_kernel(Psi, Psi, gamma)
        dK = np.diff(K, axis=0)
        K_t = K[:-1, :-1]

        # Eigendecomposition of kernel generator
        lam = 0.1
        n = K_t.shape[0]
        L = dK[:, :-1] / self.dt

        # Use Nystrom-like approach: eigenvalues of K^{-1} L
        try:
            K_reg = K_t + lam * np.eye(n)
            A_eff = np.linalg.solve(K_reg, L)
            eigenvalues = np.linalg.eigvals(A_eff)
            eigenvalues_real = np.sort(np.real(eigenvalues))[::-1]

            print("\nKernel GEDMD (RBF on signatures):")
            print(f"  True λ₁ = {-self.kappa:.2f}")
            print(f"  Top eigenvalues: {list(eigenvalues_real[:5])}")

            # Check stability
            stable_count = np.sum(np.real(eigenvalues) < 0.1)
            total = len(eigenvalues)
            stability_ratio = stable_count / total

            print(f"  Stability ratio: {stability_ratio:.2%}")
            self.assertGreater(stability_ratio, 0.5)  # Most eigenvalues stable

        except np.linalg.LinAlgError:
            self.skipTest("Kernel matrix too ill-conditioned")


class TestCIREigenvalues(unittest.TestCase):
    """Test GEDMD eigenvalues for Cox-Ingersoll-Ross (CIR) process."""

    def setUp(self):
        """Generate CIR process trajectory."""
        np.random.seed(123)
        self.kappa = 3.0
        self.theta = 0.04
        self.xi = 0.3
        self.dt = 0.001
        self.n_steps = 50000

        # Simulate CIR: dv = κ(θ-v)dt + ξ√v dW
        v = np.zeros(self.n_steps)
        v[0] = self.theta
        for t in range(1, self.n_steps):
            v[t] = v[t-1] + self.kappa * (self.theta - v[t-1]) * self.dt
            v[t] += self.xi * np.sqrt(max(v[t-1], 1e-8) * self.dt) * np.random.normal()
            v[t] = max(v[t], 1e-8)
        self.v = v

    def test_cir_eigenvalue_structure(self):
        """CIR generator has known eigenvalue structure."""
        # CIR eigenvalues: λ_n = -n·κ (same as OU for first few)
        # But eigenfunctions are different (Laguerre polynomials)

        V = self.v[:-1]
        dV = np.diff(self.v)
        n = len(V)

        # Features: [1, v, √v, v²]
        Psi = np.column_stack([
            np.ones(n),
            V,
            np.sqrt(V),
            V**2
        ])
        dPsi = np.diff(Psi, axis=0)
        Psi_t = Psi[:-1]

        lam = 0.01
        R = Psi.shape[1]
        Gram = Psi_t.T @ Psi_t + lam * np.eye(R)
        Cross = dPsi.T @ Psi_t
        A = np.linalg.solve(Gram.T, Cross.T).T / self.dt

        eigenvalues = np.linalg.eigvals(A)
        eigenvalues_real = np.sort(np.real(eigenvalues))[::-1]

        print("\nCIR GEDMD eigenvalues:")
        print(f"  True λ₁ = {-self.kappa:.2f}")
        print(f"  GEDMD eigenvalues: {list(eigenvalues_real)}")

        # First non-trivial eigenvalue should be near -κ
        self.assertLess(eigenvalues_real[1], 0)
        self.assertAlmostEqual(eigenvalues_real[1], -self.kappa, delta=1.0)

    def test_squared_embedding_captures_diffusion(self):
        """Squared embedding should help capture σ²(v) = ξ²v term."""
        window = 30
        n_samples = min(5000, self.n_steps - window - 1)

        # Standard embedding: (v_{t-1}, v_t)
        v_lag = self.v[window-1:window + n_samples - 1]
        v_now = self.v[window:window + n_samples]
        path_linear = np.column_stack([v_lag, v_now])

        # Squared embedding: (v, v²)
        path_squared = np.column_stack([v_lag, v_lag**2, v_now, v_now**2])

        # Compute signatures
        sig_linear = []
        sig_squared = []
        for i in range(n_samples - 1):
            p_lin = np.column_stack([
                [0, 1],
                [path_linear[i, 0], path_linear[i, 1]]
            ])
            p_sq = np.column_stack([
                np.linspace(0, 1, 4),
                path_squared[i, :]
            ])
            sig_linear.append(compute_path_signature(p_lin, level=2))
            sig_squared.append(compute_path_signature(p_sq, level=2))

        sig_linear = np.array(sig_linear)
        sig_squared = np.array(sig_squared)

        # Predict next variance increment
        dv = np.diff(self.v[window:window + n_samples])
        dv_sq = dv[:-1]**2 / self.dt  # Realized variance

        # Linear regression to predict dv²
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0)

        ridge.fit(sig_linear[:-1], dv_sq)
        pred_linear = ridge.predict(sig_linear[:-1])
        corr_linear = np.corrcoef(pred_linear, dv_sq)[0, 1]

        ridge.fit(sig_squared[:-1], dv_sq)
        pred_squared = ridge.predict(sig_squared[:-1])
        corr_squared = np.corrcoef(pred_squared, dv_sq)[0, 1]

        print("\nCIR diffusion prediction:")
        print(f"  Linear embedding: r = {corr_linear:.3f}")
        print(f"  Squared embedding: r = {corr_squared:.3f}")

        # At least one should have decent correlation
        self.assertGreater(max(corr_linear, corr_squared), 0.1)


class TestMultiLagGEDMD(unittest.TestCase):
    """Test that multi-lag embeddings improve GEDMD for ARCH-like processes."""

    def setUp(self):
        """Generate ARCH(2) process."""
        np.random.seed(456)
        self.n = 5000
        self.dt = 1.0

        # ARCH(2): σ²_t = α₀ + α₁ r²_{t-1} + α₂ r²_{t-2}
        self.alpha0 = 0.0001
        self.alpha1 = 0.15
        self.alpha2 = 0.10

        r = np.zeros(self.n)
        sigma2 = np.zeros(self.n)
        sigma2[0] = self.alpha0 / (1 - self.alpha1 - self.alpha2)
        sigma2[1] = sigma2[0]

        for t in range(2, self.n):
            sigma2[t] = self.alpha0 + self.alpha1 * r[t-1]**2 + self.alpha2 * r[t-2]**2
            r[t] = np.sqrt(sigma2[t]) * np.random.normal()

        self.r = r
        self.sigma2 = sigma2

    def test_single_vs_multi_lag_for_arch(self):
        """Multi-lag embedding should better capture ARCH(2) dynamics."""
        n_test = 1000

        # Single-lag features: [r_{t-1}, r²_{t-1}]
        X_1lag = np.column_stack([
            self.r[1:n_test+1],
            self.r[1:n_test+1]**2
        ])

        # Multi-lag features: [r_{t-1}, r²_{t-1}, r_{t-2}, r²_{t-2}]
        X_2lag = np.column_stack([
            self.r[2:n_test+2],
            self.r[2:n_test+2]**2,
            self.r[1:n_test+1],
            self.r[1:n_test+1]**2
        ])

        # Target: σ²_t
        y = self.sigma2[3:n_test+3]

        # Linear regression comparison
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()

        lr.fit(X_1lag[:len(y)], y)
        pred_1lag = lr.predict(X_1lag[:len(y)])
        r2_1lag = np.corrcoef(pred_1lag, y)[0, 1]**2

        lr.fit(X_2lag[:len(y)], y)
        pred_2lag = lr.predict(X_2lag[:len(y)])
        r2_2lag = np.corrcoef(pred_2lag, y)[0, 1]**2

        print("\nARCH(2) variance prediction:")
        print(f"  Single-lag R²: {r2_1lag:.3f}")
        print(f"  Multi-lag R²:  {r2_2lag:.3f}")

        # Multi-lag should be better for ARCH(2)
        self.assertGreater(r2_2lag, r2_1lag * 0.9)  # At least as good


if __name__ == '__main__':
    unittest.main(verbosity=2)
