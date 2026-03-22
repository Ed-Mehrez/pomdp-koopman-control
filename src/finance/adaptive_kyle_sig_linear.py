import numpy as np
import torch
import signatory
from typing import Tuple

class AdaptiveMM_SigLinear:
    """
    Market Maker that adaptively learns the Koopman mapping of the order flow 
    using the *Full* Signature via `signatory` PyTorch backend, 
    with Exact Linear RLS.
    
    This operates completely on the GPU to avoid PCIe transfer overhead.
    """
    def __init__(self, dt: float, lam_linear: float, P_0: float, depth: int=4):
        self.dt = dt
        self.P_0 = P_0
        self.depth = depth
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.t = 0.0
        self.Y_t = 0.0
        self.P_t = P_0
        self.step_idx = 0
        
        # We pre-allocate a maximum trajectory tensor on GPU to avoid lists
        self.max_steps = 1000
        self.path_tensor = torch.zeros(1, self.max_steps, 2, device=self.device, dtype=torch.float64)
        self.path_tensor[0, 0, :] = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float64)
        
        # Dimensions
        self.d = 4 # After Lead-Lag
        self.sig_dim = signatory.signature_channels(self.d, self.depth)
        
        # The weight vector maps the Signature directly to Price P_T
        self.w = torch.zeros(self.sig_dim + 1, device=self.device, dtype=torch.float64) # +1 for bias
        self.w[-1] = P_0
        
        # RLS Tracking for the Weights
        self.P_rls = torch.eye(self.sig_dim + 1, device=self.device, dtype=torch.float64) * 1.0
        
        self.lam_linear = lam_linear
        
        # Minimum steps before predicting full signatures (to avoid early singular paths)
        self.warmup_steps = 5
        
    def reset(self):
        self.t = 0.0
        self.Y_t = 0.0
        self.P_t = self.P_0
        self.step_idx = 0
        self.path_tensor.zero_()
        self.path_tensor[0, 0, :] = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float64)
        
    def _extract_path(self, steps_to_include) -> torch.Tensor:
        """Slices the active valid pre-allocated path block and scales it."""
        p = self.path_tensor[:, :steps_to_include, :].clone()
        # Moderate scaling to prevent Float64 explosion, but still allow polynomials to breathe
        p[:, :, 0] /= 1.0  
        p[:, :, 1] /= 5.0 
        return p
        
    def _get_signature(self, p_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the Lead-Lag *Full* Signature exactly on the GPU tensor."""
        # 1. Vectorized Lead-Lag Repeat Interleave
        # p_tensor is [1, L, 2]
        repeated = p_tensor.repeat_interleave(2, dim=1)
        lead = repeated[:, :-1, :]
        lag = repeated[:, 1:, :]
        ll_path = torch.cat([lead, lag], dim=2)
        
        # 2. Extract Full Signature
        # Adds dimensional basis [d, d^2, d^3...]
        s = signatory.signature(ll_path, self.depth)
        return s[0] # [sig_dim]
        
    def filter_step(self, dY_t: float) -> float:
        self.t += self.dt
        self.Y_t += dY_t
        self.step_idx += 1
        
        # Write to pre-allocated tensor
        self.path_tensor[0, self.step_idx, 0] = self.t
        self.path_tensor[0, self.step_idx, 1] = self.Y_t
        
        if self.step_idx < self.warmup_steps:
            self.P_t = self.P_0 + self.lam_linear * self.Y_t
            return self.P_t
            
        # 1. Get full GPU signature up to current valid step
        valid_path = self._extract_path(self.step_idx + 1)
        Z_sig = self._get_signature(valid_path)
        
        # 2. Add bias
        Z_feat = torch.cat([Z_sig, torch.tensor([1.0], device=self.device, dtype=torch.float64)])
        
        # 3. Predict
        P_tensor = torch.dot(self.w, Z_feat)
        self.P_t = P_tensor.item()
        return self.P_t
        
    def end_of_episode_update(self, final_v: float):
        """
        At $t=T$, the true value $v$ is revealed.
        """
        if self.step_idx < self.warmup_steps:
            return
            
        valid_path = self._extract_path(self.step_idx + 1)
        Z_sig = self._get_signature(valid_path)
        Z_feat = torch.cat([Z_sig, torch.tensor([1.0], device=self.device, dtype=torch.float64)])
        
        # Exact PyTorch RLS over Signature Features
        # Using a slight forgetting factor to allow the MM to unlearn the initial naive linear guess
        decay = 0.995
        
        Pz = torch.mv(self.P_rls, Z_feat)
        denom = decay + torch.dot(Z_feat, Pz)
        gain = Pz / denom
        
        prediction = torch.dot(self.w, Z_feat)
        error = final_v - prediction.item()
        
        self.w = self.w + gain * error
        self.P_rls = (self.P_rls - torch.outer(gain, Pz)) / decay
        
    def evaluate_price_derivative(self, dY_test: float = 0.01) -> float:
        """
        Computes $\lambda_t = dP_t / dY_t$ by cleanly stepping the GPU Signature.
        """
        if self.step_idx < self.warmup_steps:
            return self.lam_linear
            
        P_curr = self.P_t
        
        # Peek ahead 
        self.path_tensor[0, self.step_idx + 1, 0] = self.t + self.dt
        self.path_tensor[0, self.step_idx + 1, 1] = self.Y_t + dY_test
        
        peek_path = self._extract_path(self.step_idx + 2)
        Z_peek_sig = self._get_signature(peek_path)
        Z_peek_feat = torch.cat([Z_peek_sig, torch.tensor([1.0], device=self.device, dtype=torch.float64)])
        
        P_peek_tensor = torch.dot(self.w, Z_peek_feat)
        P_peek = P_peek_tensor.item()
        
        lam = (P_peek - P_curr) / dY_test
        return max(lam, 1e-4) # Floor to prevent singular control


class AdaptiveInsider_SigLinear:
    """
    Insider bounding their reality against the Truncated Signature Market Maker.
    """
    def __init__(self, T: float, dt: float):
        self.T = T
        self.dt = dt
        
    def compute_optimal_rate(self, v: float, t: float, P_t: float, lam_instant: float) -> float:
        """
        Calculates exact continuous singular control hitting the true value exactly at T.
        """
        time_left = max(self.T - t, self.dt) 
        
        theta_t = (v - P_t) / (lam_instant * time_left)
        return float(theta_t)
