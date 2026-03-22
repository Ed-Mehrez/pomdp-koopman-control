import unittest
import jax.numpy as jnp
import numpy as np
from src.finance.continuous_kyle import SigFilterMM, KoopmanSDREInsider

class TestKoopmanKyle(unittest.TestCase):

    def setUp(self):
        # Create toy 2D dimensional Sig-KKF system 
        self.dim_z = 2
        
        # Example Koopman Drift Matrix L_mm
        # Say feature 0 is constant 1. feature 1 is E[Y_t].
        self.L_mm = jnp.array([[0.0, 0.0],
                               [0.0, -0.1]])
                               
        # Diffusion matrix (assume linear map)
        self.g_mm = jnp.array([0.0, 1.0])
        
        # Linear projection weights for price: P_t = 0.5 * feature_1
        self.w_mm = jnp.array([0.0, 0.5])
        
        self.v = 10.0

    def test_sig_filter_mm_step(self):
        """Test continuous time Euler update of the filter."""
        mm = SigFilterMM(self.L_mm, self.g_mm, self.w_mm)
        
        # Initial state z_0 = [1.0, 0.0]
        z_t = jnp.array([1.0, 0.0])
        
        dt = 0.01
        dY_t = 0.05
        
        z_next, P_next = mm.filter_step(z_t, dY_t, dt)
        
        # dz = L*z*dt + g*dY
        # dz[0] = 0*1*dt + 0*dY = 0 -> z_next[0] = 1.0
        # dz[1] = -0.1*0*dt + 1.0*0.05 = 0.05 -> z_next[1] = 0.05
        np.testing.assert_allclose(z_next[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(z_next[1], 0.05, atol=1e-6)
        
        # P_next = 0.5 * z_next[1] = 0.025
        np.testing.assert_allclose(P_next, 0.025, atol=1e-6)

    def test_koopman_sdre_lqr_matrices(self):
        """Validate the Riccati setup constructs correct dimensions and values."""
        insider = KoopmanSDREInsider(self.v, self.L_mm, self.g_mm, self.w_mm, gamma_risk=0.0)
        
        A, B, Q, R, S = insider._construct_lqr_matrices()
        
        # Expanded state is dim_z + 1 = 3
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(B.shape, (3, 1))
        self.assertEqual(Q.shape, (3, 3))
        self.assertEqual(R.shape, (1, 1))
        self.assertEqual(S.shape, (3, 1))
        
        # Check A embeds L_mm
        np.testing.assert_array_equal(A[:2, :2], np.array(self.L_mm))
        
        # Check B embeds g_mm
        np.testing.assert_array_equal(B[:2, 0], np.array(self.g_mm))
        
        # Check S (cross terms for min E[(w^T z - v)*theta])
        # S = w/2 for z components, -v/2 for the constant 1 component
        np.testing.assert_allclose(S[0, 0], self.w_mm[0]/2)
        np.testing.assert_allclose(S[1, 0], self.w_mm[1]/2)
        np.testing.assert_allclose(S[2, 0], -self.v/2)

    def test_compute_optimal_rate(self):
        """Test the CARE solver yields a valid trading rate."""
        # Using a slightly stabilized system for Riccati convergence
        L_mm = jnp.array([[-0.1, 0.0],
                          [0.0, -0.5]])
        
        insider = KoopmanSDREInsider(self.v, L_mm, self.g_mm, self.w_mm)
        z_t = jnp.array([1.0, 0.0])
        
        # Time t doesn't matter for infinite horizon CARE
        theta_t = insider.compute_optimal_rate(z_t, t=0.0)
        
        # A valid float rate should be computed without singular errors
        self.assertIsInstance(theta_t, float)
        self.assertFalse(np.isnan(theta_t))

if __name__ == '__main__':
    unittest.main()
