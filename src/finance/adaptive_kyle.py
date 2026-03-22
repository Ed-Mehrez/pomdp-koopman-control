import numpy as np
from typing import Tuple
import sys
import os

# Ensure we can import from sskf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sskf.streaming_sig_kkf import StreamingSigKKF

class AdaptiveMM:
    """
    Market Maker that adaptively learns the Koopman drift of the order flow 
    using the Streaming Signature Kalman Filter (RLS).
    """
    def __init__(self, dt: float, lam: float, P_0: float, level: int = 1, forgetting_factor: float = 0.999):
        self.dt = dt
        self.lam = lam
        self.P_0 = P_0
        self.P_t = P_0
        self.Y_t = 0.0
        self.t = 0.0
        self.level = level
        
        # Initialize streaming kernel filter to learn order flow drift
        self.kkf = StreamingSigKKF(dt=dt, level=level, process_noise=1e-4, forgetting_factor=forgetting_factor)
        
    def reset(self):
        self.kkf.reset(0.0)
        self.P_t = self.P_0
        self.Y_t = 0.0
        self.t = 0.0
        
    def filter_step(self, dY_t: float) -> float:
        """
        Observes order flow increment, updates the learned drift, 
        and updates the price as a martingale relative to expectations.
        """
        self.Y_t += dY_t
        
        # Update KKF with new cumulative observation
        drift_pred, _ = self.kkf.update(self.Y_t)
        
        # P_{t+dt} = P_t + lambda * (Surprise in Order Flow)
        # Surprise = dY_t - expected_drift * dt
        surprise = dY_t - drift_pred * self.dt
        self.P_t = self.P_t + self.lam * surprise
        
        self.t += self.dt
        return self.P_t
        
    def get_koopman_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the instant matrix form L_mm and g_mm from the learned RLS parameters.
        The state space is z_t = [1, t, Y_t, P_t - P_0].
        """
        # A_drift corresponds to the learned coefficients for [1, t, Y_t]
        A_drift = self.kkf.A[2, :] 
        a, b, c = A_drift[0], A_drift[1], A_drift[2]
        
        # Construct continuous-time L_mm (dimension 4x4)
        L_mm = np.zeros((4, 4))
        # d(1)/dt = 0
        L_mm[0, :] = [0, 0, 0, 0]
        # d(t)/dt = 1
        L_mm[1, :] = [1, 0, 0, 0]
        # d(Y)/dt (Intrinsic state expectation, usually 0 or drift)
        L_mm[2, :] = [0, 0, 0, 0] 
        # d(P)/dt = -lam * expected_drift
        L_mm[3, :] = [-self.lam * a, -self.lam * b, -self.lam * c, 0]
        
        # Diffusion matrix impacts: dt terms = 0. Y_t takes 1 dY_t. P_t takes lam dY_t.
        g_mm = np.array([0, 0, 1.0, self.lam])
        
        return L_mm, g_mm


class AdaptiveInsider:
    """
    Insider bounding their reality: they observe the MM's current pricing rule 
    L_mm(t) and solve a local discrete-time Riccati Equation (DARE) backward to t.
    """
    def __init__(self, T: float, dt: float):
        self.T = T
        self.dt = dt
        
    def compute_optimal_rate(self, z_t: np.ndarray, v: float, P_0: float, t: float, L_mm: np.ndarray, g_mm: np.ndarray) -> float:
        """
        Solves the discrete-time backwards induction local SDRE given the MM's 
        current (frozen) belief matrix L_mm.
        """
        n_steps = int(round((self.T - t) / self.dt))
        if n_steps <= 0:
            return 0.0
            
        dim_z = 4
        dim_x = dim_z + 1 # x = [z_t, v_eff]
        
        # Continuous matrices wrapper
        A_c = np.zeros((dim_x, dim_x))
        A_c[:dim_z, :dim_z] = L_mm
        A_c[-1, -1] = -1e-8 # Stabilizer for true value state
        
        B_c = np.zeros((dim_x, 1))
        B_c[:dim_z, 0] = g_mm
        
        # Discrete-time equivalents
        A_d = np.eye(dim_x) + A_c * self.dt
        B_d = B_c
        
        Q_d = np.zeros((dim_x, dim_x))
        
        # Cross Cost S: Inside cost is E[u_n(v - P_n)dt]. 
        # But we formulate min cost: -u_n(v - P_n)dt = u_n(P_n - v)dt
        # P_n - v = z_t[3] - (v - P_0) 
        S_d = np.zeros((dim_x, 1))
        S_d[3, 0] = 0.5 * self.dt
        S_d[4, 0] = -0.5 * self.dt
        
        # Execution Cost
        # To strictly enforce the order flow limit natively, we penalize magnitude
        # Discrete LQR organically cures singular control through B_d, but tracking
        # error limits benefit from small Tikanov reg R.
        lam = g_mm[3]
        R_d = np.array([[lam * self.dt]]) 
        
        # DARE Backward sweep
        P = np.zeros((dim_x, dim_x))
        K = np.zeros((1, dim_x))
        
        for _ in range(n_steps):
            R_eff = R_d + B_d.T @ P @ B_d
            try:
                # K = (R + B^T P B)^-1 (B^T P A + S^T)
                K = np.linalg.inv(R_eff) @ (B_d.T @ P @ A_d + S_d.T)
            except np.linalg.LinAlgError:
                break
                
            # P = A^T P A + Q - (A^T P B + S) K
            P = A_d.T @ P @ A_d + Q_d - (A_d.T @ P @ B_d + S_d) @ K
            
        # Execute current step optimal policy
        v_eff = v - P_0
        x_t = np.concatenate([z_t, [v_eff]])
        
        theta_t = -(K @ x_t)[0]
        return float(theta_t)
