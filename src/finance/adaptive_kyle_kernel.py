import numpy as np
from typing import Tuple
import sys
import os

# Ensure we can import from sskf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sskf.streaming_sig_kkf import StreamingSigKernelLearner, SignatureState

class AdaptiveMM_Kernel:
    """
    Market Maker that adaptively learns the Koopman mapping of the order flow 
    using the infinite-dimensional Signature Kernel RKHS.
    """
    def __init__(self, dt: float, lam_linear: float, P_0: float):
        self.dt = dt
        self.P_0 = P_0
        self.P_t = P_0
        self.Y_t = 0.0
        self.t = 0.0
        
        # We use the Kernel Learner to map order flow features Z_t -> P_t directly.
        # Since the true value v is only revealed at T, this is a regression problem:
        # P_T = v  ==>  f_kernel(Z_T) \approx v
        # We maintain a budget of historic terminal signatures and use Kernel Ridge Regression (KRR).
        self.kkf = StreamingSigKernelLearner(
            dt=dt, 
            level=2, 
            method='krr',
            kernel_type='untruncated', # Upgrade to PDE solver for infinite-dimensional features!
            reg_param=1.0,  
            max_budget=200, 
            mode='budgeted'
        )
        
        # We need a linear backup for the very beginning before the kernel has support points
        self.lam_linear = lam_linear
        
    def reset(self):
        # We DO NOT reset the kkf dictionary. The KRR learns across episodes!
        # But we must reset the current signature path tracking for the new episode.
        self.kkf.x_origin = 0.0
        self.kkf.t = 0.0
        self.kkf.x_current = 0.0
        self.kkf.sig_current = SignatureState(level=self.kkf.level, store_path=self.kkf.store_path)
        if self.kkf.store_path:
            self.kkf.sig_current.t_history = [0.0]
            self.kkf.sig_current.x_history = [0.0]
        
        self.P_t = self.P_0
        self.Y_t = 0.0
        self.t = 0.0
        
    def filter_step(self, dY_t: float) -> float:
        """
        Observes order flow increment, tracks signature state.
        In the exact kernel framework, the price process P_t is the conditional expectation E[v | Y_t].
        By martingale property of Koopman, P_t = f_kernel(Z_t).
        """
        self.Y_t += dY_t
        self.t += self.dt
        
        # Step the signature state forward manually (since we aren't adding it to the kernel dictionary yet)
        self.kkf.sig_current.extend(self.dt, dY_t)
        self.kkf.x_current += dY_t
        self.kkf.t += self.dt
        
        # Evaluate current price from Kernel dictionary
        if len(self.kkf.support_points) < 5:
            # Fallback to linear before sufficient support points exist
            self.P_t = self.P_0 + self.lam_linear * self.Y_t
        else:
            # P_t = f_kernel(Z_t) = k(Z_t, Z_vocab) @ alpha
            k_vec = np.array([self.kkf.sig_current.kernel_with(sp[0], kernel_type=self.kkf.kernel_type) for sp in self.kkf.support_points])
            price_pred = k_vec @ self.kkf.alpha
            self.P_t = price_pred
            
        return self.P_t
        
    def end_of_episode_update(self, final_v: float):
        """
        At $t=T$, the true value $v$ is revealed. The MM adds the final signature Z_T 
        to their kernel dictionary with the target $v$, and recomputes the KRR weights alpha.
        """
        # We abuse the `add_observation` which normally tracks continuous state.
        # But here, we only add the TERMINAL signature Z_T to the dictionary to map Z_T -> v.
        
        # Copy the final signature state
        sig_T = SignatureState(level=self.kkf.level, store_path=self.kkf.store_path)
        sig_T.S = {k: v.copy() for k, v in self.kkf.sig_current.S.items()}
        if self.kkf.store_path:
            sig_T.t_history = self.kkf.sig_current.t_history.copy()
            sig_T.x_history = self.kkf.sig_current.x_history.copy()
        
        self.kkf.support_points.append((sig_T, final_v, self.Y_t))
        
        if len(self.kkf.support_points) > self.kkf.max_budget:
            self.kkf.support_points.pop(0)
            
        self.kkf._update_kernel_matrix()
        
    def evaluate_price_derivative(self, dY_test: float = 1.0) -> float:
        """
        Computes the instantaneous linear price impact $\lambda(t)$ by evaluating 
        the directional derivative of the RKHS function along the order flow $dY$.
        This gives the Insider their instantaneous execution cost `lam_instant`.
        """
        if len(self.kkf.support_points) < 5:
            return self.lam_linear
            
        # P_t(Y_t)
        P_current = self.P_t
        
        # P_{t+dt}(Y_t + dY)
        # We branch a temporary signature state to peek ahead
        sig_peek = SignatureState(level=self.kkf.level, store_path=self.kkf.store_path)
        sig_peek.S = {k: v.copy() for k, v in self.kkf.sig_current.S.items()}
        if self.kkf.store_path:
            sig_peek.t_history = self.kkf.sig_current.t_history.copy()
            sig_peek.x_history = self.kkf.sig_current.x_history.copy()
            
        sig_peek.extend(self.dt, dY_test) # Tiny step
        
        k_vec_peek = np.array([sig_peek.kernel_with(sp[0], kernel_type=self.kkf.kernel_type) for sp in self.kkf.support_points])
        P_peek = k_vec_peek @ self.kkf.alpha
        
        # finite difference lambda = dP / dY
        lam_instant = (P_peek - P_current) / dY_test
        return max(lam_instant, 1e-4) # strict positivity floor


class AdaptiveInsider_Kernel:
    """
    Insider bounding their reality against the non-parametric Signature Kernel Market Maker.
    Because the MM's mapping Z_t -> P_t is nonlinear, the Insider solves the 
    continuous-time optimal control by freezing the instantaneous $\lambda_t = dP/dY$ 
    and solving the 1D locally linearized continuous Kyle SDRE to the horizon.
    """
    def __init__(self, T: float, dt: float):
        self.T = T
        self.dt = dt
        
    def compute_optimal_rate(self, v: float, t: float, P_t: float, lam_instant: float) -> float:
        """
        Since the MM's global RKHS mapping is complex, the Insider localizes it.
        At any instant $t$, if the MM evaluates price impact as $\lambda_t$ and current price $P_t$,
        the mathematically exact continuous-time singular HJB solution to hit $v$ is:
        \theta_t = (v - P_t) / (\lambda_t * (T - t))
        """
        time_left = max(self.T - t, self.dt) # Clamp to prevent div by zero at boundary
        
        theta_t = (v - P_t) / (lam_instant * time_left)
        return float(theta_t)
