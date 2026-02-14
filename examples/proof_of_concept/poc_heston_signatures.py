"""
Heston Volatility Control via KRONIC + Signatures
=================================================
Validating the architecture on a Stochastic Volatility Stabilization problem.

Physics:
    dX_t = (r - 0.5 v_t) dt + sqrt(v_t) dW1  (Observed Log-Price)
    dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW2  (Hidden Variance)
    
Control:
    We control 'kappa' (mean reversion speed) to force variance to target.
    u \in [-1, 1] => kappa_eff = kappa_base * (1 + u)
    
Features:
    Input: Path of X_t (Log-Price).
    Transform: Log-Signatures (Level 2).
    Theory: Sig(X) contains info about Quadratic Variation <X,X> ~ Integrated Variance.
    
Objective:
    Stabilize v_t around theta.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.kronic_controller import KRONICController
from src.kgedmd_core import RBFKernel
from examples.proof_of_concept.signature_features import get_augmented_state_with_signatures

class HestonEnv:
    def __init__(self, kappa=2.0, theta=0.04, xi=0.3, dt=0.01):
        self.kappa_base = kappa
        self.theta = theta
        self.xi = xi
        self.dt = dt
        self.state = np.array([0.0, theta]) # [LogPrice X, Variance v]
        
    def reset(self):
        # Start at target variance
        self.state = np.array([0.0, self.theta])
        return self.state.copy()
        
    def step(self, u):
        # u in [-1, 1] controls mean reversion speed
        # kappa_eff = kappa * (1 + u)
        u_clipped = np.clip(u, -0.9, 5.0) # Allow increasing speed significantly
        kappa_eff = self.kappa_base * (1.0 + u_clipped)
        
        X, v = self.state
        
        # Euler-Maruyama
        dW1 = np.random.normal(0, np.sqrt(self.dt))
        dW2 = np.random.normal(0, np.sqrt(self.dt))
        # Correlation rho? Assuming 0 for simplicity or add later.
        
        # Heston Dynamics
        # Ensure non-negative variance (Reflection or Full Truncation)
        v_abs = max(1e-6, v)
        
        dX = -0.5 * v_abs * self.dt + np.sqrt(v_abs) * dW1
        dv = kappa_eff * (self.theta - v_abs) * self.dt + self.xi * np.sqrt(v_abs) * dW2
        
        self.state = self.state + np.array([dX, dv])
        
        # Make reduced observation (Log Price Only? Or we assume we see V for training?)
        # For KRONIC training, we usually assume full state observability OR delay embedding.
        # "Signature Control" implies we only see X (Log Price).
        # But we need GROUND TRUTH V to calculate Cost for LQR training.
        # So: Train on (X, v), Deploy on Sig(X).
        # Actually, if we train on (X, v), the Koopman operator learns dynamics of (X, v).
        # If we deploy, we need to reconstruct v.
        # Wait, the documentation says "Augment state with Signatures".
        # So the state is z = [X, v, Sig(X)].
        # If v is hidden at test time, we can't use it in 'z'.
        # LQR u = -K z. If z contains v, we need v.
        # HYPOTHESIS: Signatures ARE the proxy for v.
        # So we should define state z = [Sig(X)].
        # BUT we need a cost function J = (v - theta)^2.
        # We can learn Cost Mapping J(z) approx J(v).
        
        # Let's train assuming we have access to v (Oracle training), 
        # but check if the "Learned Dynamics" can predict v evolution from Signatures.
        
        return self.state.copy()

def run_heston_experiment():
    print("🚀 Starting Heston Volatility Control Verification")
    
    # Configuration
    # Using fewer episodes but NO subsampling to preserve temporal structure
    n_episodes = 20
    steps_per_episode = 200
    dt = 0.01 # Higher frequency sampling
    
    env = HestonEnv(dt=dt)
    
    # Controller Setup
    # Target state: We want Sig(X) to look like "low volatility path"
    # For now, use zeros as placeholder target (cost function will be overridden)
    target_z = np.zeros(3) # Will be resized after data collection
    
    controller = KRONICController(
        kernel=RBFKernel(sigma=2.0),
        target_state=target_z,
        cost_weights={'Q': 1.0, 'R': 0.1},
        verbose=True
    )
    
    # Data Collection
    X_train_data = []
    U_train_data = []
    X_next_train_data = []
    
    history_window = 10
    
    # Data Collection Arrays
    X_train_data = []
    U_train_data = []
    X_next_train_data = []
    Cost_data = [] # True cost associated with state
    
    history_window = 20 # Longer window for Volatility extraction
    
    print("📊 Collecting Training Data...")
    for ep in range(n_episodes):
        obs = env.reset()
        # Randomize start variance
        env.state[1] = np.random.uniform(0.01, 0.09)
        obs = env.state
        
        history_buffer = [] # Store X (Log Price)
        
        for t in range(steps_per_episode):
            # Control Input: Random excitation
            u = np.random.uniform(-0.5, 0.5, size=(1,))
            
            # Record History (Log Price)
            # Obs is [X, v]. We only see X.
            history_buffer.append([obs[0]]) 
            if len(history_buffer) > history_window:
                history_buffer.pop(0)
                
            # Construct Feature Vector z = [LogSig(X)]
            # We treat 'X' itself as less important than its variation (Sig)
            z = get_augmented_state_with_signatures(
                np.array([obs[0], 0.0, 0.0, 0.0]), # Padding to match dim=4 expectation of helper?
                # Wait, helper expects [x, x_dot, theta, theta_dot].
                # We need a custom/simplified helper for 1D Heston Price.
                # Let's write a simple inline signature calculator here or update helper.
                # Using the raw compute_log_signature form is better.
                [], # Dummy history
                use_log_signatures=True 
            )
            # FIX: The helper is tied to Cartpole. We should use compute_log_signature directly.
            pass
            
            # Step
            next_obs = env.step(u[0])
            
            # Cost: (v - theta)^2
            # Goal: Drive variance to theta
            current_v = obs[1]
            cost = (current_v - env.theta)**2 * 1000.0 # Scale up
            Cost_data.append(cost)
            
            # OBSOLETE: See below for fixed loop
            obs = next_obs
            
    # RE-IMPLEMENTING: Single long trajectory (realistic for Heston)
    # In practice, we only have ONE realized price path
    X_train_data = []
    U_train_data = []
    X_next_train_data = []
    Cost_data = []
    
    from examples.proof_of_concept.signature_features import compute_log_signature
    
    # Single long trajectory
    total_steps = 5000
    
    obs = env.reset()
    env.state[1] = 0.06 # Start above target vol
    
    history_buffer = []
    
    # Warmup history
    for _ in range(history_window):
        obs = env.step(0.0)
        history_buffer.append([obs[0]])
         
    print(f"    Collecting {total_steps} steps from single trajectory...")
    
    # Also collect DIFFUSION COVARIANCE for stochastic Generator EDMD
    # For Heston on LogSig space: The diffusion matrix σσᵀ depends on volatility v
    # But in our feature space z = LogSig(returns), we need the induced diffusion.
    # Approximation: Use variance of returns as proxy for local diffusion.
    # Z shape needs to be (d, d, m) where d = feature dimension
    Z_data = [] # Will store local diffusion estimates
         
    for t in range(total_steps):
        # Random Control (exploration)
        u = np.random.uniform(-0.5, 0.5, size=(1,))
        
        # 1. Construct State z = LogSig(RETURNS)
        # Convert price history to returns (differences)
        prices = np.array(history_buffer).flatten()
        returns = np.diff(prices) # (Window-1,)
        
        # Add time channel to returns [t, dX]
        t_steps = np.linspace(0, 1, len(returns))
        path_aug = np.column_stack([t_steps, returns])
        
        z = compute_log_signature(path_aug, level=2)
        
        # 2. Step
        next_obs = env.step(u[0])
        
        # 3. Next State (on returns)
        next_history = history_buffer[1:] + [[next_obs[0]]]
        next_prices = np.array(next_history).flatten()
        next_returns = np.diff(next_prices)
        path_next_aug = np.column_stack([t_steps, next_returns])
        z_next = compute_log_signature(path_next_aug, level=2)
        
        # 4. Store
        X_train_data.append(z)
        U_train_data.append(u)
        X_next_train_data.append(z_next)
        
        # 5. Compute local diffusion estimate
        # For returns dX_t = sqrt(v) dW, QV = v * dt
        # Local variance of returns window ~ integrated variance
        current_v = obs[1] # True (hidden) variance - used for training
        # Diffusion matrix in observation space: σ = [[0], [sqrt(v)]] for [t, X]
        # In LogSig space, we approximate as scaled identity (simplified)
        d_z = len(z)
        # Diffusion scales with sqrt(v), variance scales with v
        local_diffusion = current_v * np.eye(d_z) # Simplified: σσᵀ ≈ v * I
        Z_data.append(local_diffusion)
        
        # Cost (Hidden V driven guidance)
        v = obs[1]
        cost = (v - env.theta)**2 * 10000.0
        Cost_data.append(cost)
        
        obs = next_obs
        history_buffer = next_history

    X_train = np.array(X_train_data)
    U_train = np.array(U_train_data)
    X_next_train = np.array(X_next_train_data)
    Costs = np.array(Cost_data)
    Z_diffusion_raw = np.array(Z_data).transpose(1, 2, 0) # (m, d, d) -> (d, d, m)
    
    # Pad Z_diffusion to match augmented state dimension [x, u]
    # Current: (d_z, d_z, m) = (3, 3, 5000)
    # Required: (d_z + n_controls, d_z + n_controls, m) = (4, 4, 5000)
    n_controls = U_train.shape[1]
    d_z = Z_diffusion_raw.shape[0]
    d_augmented = d_z + n_controls
    Z_diffusion = np.zeros((d_augmented, d_augmented, len(X_train)))
    Z_diffusion[:d_z, :d_z, :] = Z_diffusion_raw # Place original in top-left
    # Control channels have zero diffusion (deterministic control input)
    
    print(f"    Data Shape: {X_train.shape}")
    print(f"    Cost Range: {Costs.min():.4f} - {Costs.max():.4f}")
    
    # Train KRONIC
    # Note: KRONIC fit() usually takes Q_diag.
    # We need to manually inject the "Learned Cost" step or pass costs.
    # kronic_controller implementation of fit() doesn't officially take 'costs' argument?
    # Let's look at kronic_controller.py...
    # It takes *target_z* and *Q_diag*.
    # BUT we want to learn Cost(z) from data pairs (z, true_cost).
    # The current KRONIC implementation assumes Cost(z) = (z-Ref)^T Q (z-Ref).
    # It doesn't support arbitrary cost regression from data in the public `fit` API?
    # Actually, `_learn_cost_mapping` uses `self.state_costs`.
    
    # Hack: Inject costs into controller before fitting
    controller.state_costs = Costs # Used by _learn_cost_mapping
    
    # Dummy Target/Q (ignored because we override cost mapping?)
    # Wait, `_learn_cost_mapping` is called inside `fit`.
    # AND `fit` computes `state_costs` using `target_z`.
    # We need to OVERRIDE this computation.
    
    # Plan: Pass dummy target, then overwrite `controller.state_costs` inside a derived class or patch?
    # Or rely on `fit` to compute costs?
    # If we pass target_z = [Sig(TargetVolPath)], Q = [High penalty on Level 2 terms],
    # KRONIC will compute costs = (z - z_target)^T Q (z - z_target).
    # This assumes we know the "Signature of the Target Variance".
    # Target Variance is constant theta.
    # Path with constant variance theta -> X is Brownian Motion with scale theta.
    # Expected Signature terms: Level 2 = theta * T.
    
    # EASIER APPROACH: PATCH KRONIC `fit` logic to accept 'external_costs'.
    # Or just subclass it quickly here.
    
    class HestonKRONIC(KRONICController):
        def fit(self, X_train, X_dot, U_train, dt, external_costs=None, **kwargs):
            self.external_costs = external_costs
            super().fit(X_train, X_dot, U_train, dt, **kwargs)
            
        def _learn_cost_mapping(self):
            # Override to use external costs
            if self.external_costs is not None:
                # Use the implemented Enhanced Cost Mapping with provided costs
                from src.regularized_cost_reconstruction import enhanced_cost_mapping
                # ...
                # Simple version: copy logic but use self.external_costs
                # Need to run eigenfunction eval on training data
                Z_train_norm = self.Z_scaler.transform(self.Z_train)
                Z_centers = self.gedmd_model.X_train_.T
                K = self.kernel(Z_train_norm, Z_centers)
                phi = K @ self.intrinsic_basis_complex
                phi_real = np.hstack([np.real(phi), np.imag(phi)])
                
                Q_phi, _ = enhanced_cost_mapping(phi_real, self.external_costs)
                return Q_phi
            else:
                return super()._learn_cost_mapping()

    # Create target from data mean (placeholder)
    target_z = np.zeros(X_train.shape[1])
    
    # Adaptive Bandwidth: Median Distance Heuristic
    from scipy.spatial.distance import pdist
    subsample_idx = np.random.choice(len(X_train), min(500, len(X_train)), replace=False)
    distances = pdist(X_train[subsample_idx])
    sigma = np.median(distances)
    print(f"    Adaptive Sigma (Median Distance): {sigma:.4f}")
    
    controller = HestonKRONIC(
        kernel=RBFKernel(sigma=sigma),
        target_state=target_z,
        cost_weights={'Q': 1.0, 'R': 0.1},
        verbose=True
    )
    
    # Compute X_dot
    X_dot_train = (X_next_train - X_train) / dt
    
    # No subsampling - keep full trajectories intact for time-series integrity
    # Pass Z_diffusion for stochastic Generator EDMD
    print(f"    Z_diffusion Shape: {Z_diffusion.shape}")
    controller.fit(X_train, X_dot_train, U_train, dt, 
                   external_costs=Costs,
                   Z_diffusion=Z_diffusion)
    
    # Evaluation Loop
    print("\n🎬 Evaluating Heston Control...")
    obs = env.reset()
    env.state[1] = 0.08 # Start High
    history = []
    # Warmup
    for _ in range(history_window):
        obs = env.step(0.0)
        history.append([obs[0]])
        
    v_history = []
    
    for t in range(300):
        # Feature: Signature of RETURNS
        prices = np.array(history[-history_window:]).flatten()
        returns = np.diff(prices)
        t_arr = np.linspace(0, 1, len(returns))
        path_aug = np.column_stack([t_arr, returns])
        z = compute_log_signature(path_aug, level=2)
        
        u = controller.control(z)
        
        obs = env.step(u[0])
        history.append([obs[0]])
        v_history.append(obs[1])
        
    plt.plot(v_history)
    plt.axhline(env.theta, color='r', linestyle='--')
    plt.title("Heston Volatility Stabilization")
    plt.savefig("heston_result.png")
    print("Saved heston_result.png")

if __name__ == "__main__":
    run_heston_experiment()
