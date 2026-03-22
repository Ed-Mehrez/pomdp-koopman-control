import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple, Any

class KyleAssetEnvironment:
    """
    Simulates the true (hidden) value 'v' and the continuous-time noise trader flow.
    """
    def __init__(self, key: jax.Array, v_prior_sampler: Callable):
        self.key = key
        self.v_prior_sampler = v_prior_sampler
        
    def reset(self) -> float:
        """Sample a new true value v."""
        self.key, subkey = jax.random.split(self.key)
        return self.v_prior_sampler(subkey)

class SigFilterMM:
    """
    The Market Maker agent employing Kolgomorov/Koopman linear filtering
    in the signature feature space.
    """
    def __init__(self, L_mm: jnp.ndarray, g_mm: jnp.ndarray, w_mm: jnp.ndarray):
        """
        L_mm: The continuous-time drift operator (Koopman Matrix) for the signature.
        g_mm: The continuous-time diffusion operator for the signature (Stratonovich mapped).
        w_mm: The linear projection weights mapping the signature state back to expected value.
        """
        self.L_mm = L_mm
        self.g_mm = g_mm
        self.w_mm = w_mm
        
        self.dim_z = L_mm.shape[0]
        
    def filter_step(self, z_t: jnp.ndarray, dY_t: float, dt: float) -> Tuple[jnp.ndarray, float]:
        """
        Step the filter forward in continuous time using Euler-Maruyama.
        NOTE: Since signatures naturally evolve via Stratonovich integration, 
        the discrete EM step must account for this (or we simulate with small enough dt).
        
        dz_t = L_mm @ z_t * dt + g_mm(z_t) * dY_t
        """
        # Linear drift in feature space
        drift = self.L_mm @ z_t
        
        # Diffusion handling. If g_mm is 1D it's constant. If 2D it's linear in state.
        if self.g_mm.ndim == 1:
            diffusion = self.g_mm
        else:
            diffusion = self.g_mm @ z_t 
        
        z_next = z_t + drift * dt + diffusion * dY_t
        
        # Calculate price estimate P_t
        P_t = jnp.dot(self.w_mm, z_next)
        
        return z_next, P_t


class KoopmanSDREInsider:
    """
    The Insider computing their optimal control strategy by treating the 
    MM's Koopman filter dynamics as their state environment, leading to a 
    finite-dimensional LQR/SDRE problem.
    """
    def __init__(self, L_mm: jnp.ndarray, g_mm: jnp.ndarray, w_mm: jnp.ndarray, gamma_risk: float = 0.0, T: float = 1.0):
        """
        The Insider knows the MM's parameters exactly (Koopman Rationality).
        L_mm: Koopman drift matrix for the MM's filter.
        g_mm: Koopman diffusion matrix for the MM's filter.
        w_mm: Linear projection weights for P_t.
        gamma_risk: Risk aversion parameter.
        T: Total trading time (horizon).
        """
        self.L_mm = L_mm
        self.g_mm = g_mm
        self.w_mm = w_mm
        self.gamma_risk = gamma_risk
        self.T = T
        self.dim_z = L_mm.shape[0]
        
    def _construct_lqr_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Constructs the A, B, Q, R matrices for the continuous-time LQR problem.
        The expanded state is x_t = [z_t, 1]. The constant 1 handles the affine true value v.
        
        System dynamics:
        d(z_t) = (L_mm @ z_t + g_mm @ z_t * theta_t) dt + (noise terms)
        
        Let's assume a simplified constant diffusion vector g_mm for now to make control linear.
        If g_mm is a matrix, the control is bilinear and requires true SDRE.
        For Kerry Back continuous Kyle, the diffusion impact is usually constant relative to order flow:
        dz_t = L_mm @ z_t dt + G dY_t  where G is a vector.
        dY_t = theta_t dt + dZ_t
        So dz_t = (L_mm @ z_t + G * theta_t) dt + G dZ_t
        """
        import numpy as np
        
        # We assume g_mm is a vector here for standard LQR. If it's a matrix acting on z, 
        # we need State-Dependent Riccati Equation (SDRE) where B(x) = g_mm @ z_t.
        # For our first Paradigm A test, we will enforce g_mm is a constant vector (first level signature).
        
        # State: x = [z_t^T, v]^T  (dimension: dim_z + 1)
        dim_x = self.dim_z + 1
        A = np.zeros((dim_x, dim_x))
        A[:self.dim_z, :self.dim_z] = self.L_mm
        
        # Add a tiny decay to the constant 'v' state to satisfy CARE stabilizability.
        # An uncontrollable eigenvalue exactly at 0 will cause scipy.linalg.solve_continuous_are to fail.
        A[-1, -1] = -1e-8
        
        # B matrix maps theta_t to state derivative
        B = np.zeros((dim_x, 1))
        # If g_mm is 1D (a vector), it's exactly B. If it's 2D, we evaluate at current state for SDRE.
        g_np = np.array(self.g_mm)
        if g_np.ndim == 1:
            B[:self.dim_z, 0] = g_np
        else:
            raise ValueError("For standard LQR, g_mm must be a constant vector. If matrix, use SDRE.")
            
        # Cost function: max E[int (v - P_t) theta_t dt] 
        # Rewritten as min 1/2 x^T Q x + x^T S u + 1/2 u^T R u
        # P_t = w_mm^T z_t
        # Cost = (w_mm^T z_t - v) theta_t
        
        Q = np.zeros((dim_x, dim_x)) # No pure state cost unless risk aversion is added
        
        # Cross cost S: x^T S u
        # We want to minimize (w_mm^T z_t - v) theta_t
        # x = [z^T, v]^T
        # x^T S u = z^T S_z theta + v * S_v theta
        # So S_z = w_mm / 2. S_v = -1 / 2
        S = np.zeros((dim_x, 1))
        S[:self.dim_z, 0] = np.array(self.w_mm) / 2.0  
        S[-1, 0] = -0.5
        
        # Add tiny regularization to R to approximate the singular control
        R = np.zeros((1, 1))
        R[0, 0] = 1e-4
        
        return A, B, Q, R, S

    def compute_exact_singular_rate(self, z_t: jnp.ndarray, v: float, t: float) -> float:
        """
        Computes the exact optimal trading rate $\theta_t$ for the singular Koopman HJB.
        Since $R=0$ (risk-neutral, no execution penalty), the problem is singular.
        The exact analytical solution over the Koopman state space is:
        \theta_t = (v - w^T z_t) / ((w^T g) * (T - t))
        """
        P_t = jnp.dot(self.w_mm, z_t)
        lam = jnp.dot(self.w_mm, self.g_mm)
        
        # Prevent explosive discretization step at the exact boundary
        # Clamping to dt (0.005) stabilizes the numerical SDE Euler-Maruyama step
        time_left = max(self.T - t, 0.005)
        
        theta_t = (v - P_t) / (lam * time_left)
        return float(theta_t)

