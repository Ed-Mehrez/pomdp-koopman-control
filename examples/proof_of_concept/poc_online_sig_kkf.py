import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.cartpole_env import CartPoleEnv
from examples.proof_of_concept.signature_features import RecurrentSignatureMap
from examples.proof_of_concept.particle_mpc import ParticleMPC
from src.kronic_tensor_utils import koopman_tensor_reduction
from src.kronic_online_tensor import RecursiveTensorRLS
import matplotlib.animation as animation

def render_cartpole(state, ax):
    x, x_dot, theta, theta_dot = state
    cart_width = 1.0
    cart_height = 0.5
    pole_length = 1.0 # Visual length
    
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=1)
    
    # Cart
    cart = plt.Rectangle((x - cart_width/2, -cart_height/2), cart_width, cart_height, color='blue')
    ax.add_patch(cart)
    
    # Pole
    # Theta 0 is Up. pi is Down.
    # Endpoint
    pole_x = x + pole_length * np.sin(theta)
    pole_y = pole_length * np.cos(theta)
    
    ax.plot([x, pole_x], [0, pole_y], color='red', lw=3)
    ax.plot([x], [0], 'ko') # Hinge


def run_online_adaptive_experiment():
    print("🚀 Starting Online Adaptive Sig-KKF Experiment")
    
    # ---------------------------
    # EXPERIMENT CONFIGURATION
    # ---------------------------
    ENABLE_BURNIN = True    # Experiment A
    ENABLE_TUNING = True    # Experiment B
    
    print(f"   Configs: BurnIn={ENABLE_BURNIN}, Tuning={ENABLE_TUNING}")

    
    # 1. Setup
    env = CartPoleEnv(dt=0.02)
    obs = env.reset()
    env.state[2] = np.pi # Start Down
    
    # Features
    sig_map = RecurrentSignatureMap(state_dim=5, level=2, forgetting_factor=0.98)
    sig_map.reset()
    obs_embedded = np.array([obs[0], obs[1], np.cos(obs[2]), np.sin(obs[2]), obs[3]])
    z_curr = sig_map.update(obs_embedded * 0.1)
    
    feature_dim = len(z_curr)
    control_dim = 1
    
    # Learner
    # High forgetting factor (slow forgetting) to retain global knowledge?
    # Or low to adapt fast?
    # Start high (0.995) to build stable model, then maybe adapt.
    rls = RecursiveTensorRLS(feature_dim, control_dim, lambda_forget=0.995, delta_init=10.0)
    
    # Projection matrix (Random initially? Or just identity?)
    # We need projection for Tensor Reduction.
    # But RLS learns Full Rank model. Reduction happens on the fly.
    
    # Decoder C? 
    # We need to learn C online too.
    # y = C z
    # RLS for C.
    rls_decoder = RecursiveTensorRLS(feature_dim, 0, lambda_forget=0.995, delta_init=10.0) 
    # Hack: use RLS with u=0 to learn linear map Z -> Y.
    # Output dim is 4. My RLS class assumes output dim matches target z_next (n).
    # We need a separate simple RLS or just use Ridge on buffer.
    
    # Simple buffer for decoder (Observation map is usually static / easier)
    # Z_buffer = []
    # Y_buffer = []
    # Wait, we can assume we know C approx?
    # Sig[0:5] corresponds to time-integrated state.
    # Actually, let's just use the First 5 Features as State Proxy?
    # Feature 0 is x-x0 approx.
    # It requires drift Correction.
    
    # Plan: Use Tensor-MPPI with "Feature Cost".
    # We define target feature z_target (Upright).
    # We optimize distance to z_target.
    
    # Target Z:
    # Run a virtual upright episode to get Z_target.
    sig_map_ref = RecurrentSignatureMap(state_dim=5, level=2, forgetting_factor=0.98)
    sig_map_ref.reset()
    dummy_emb = np.array([0, 0, 1.0, 0.0, 0])
    z_target = sig_map_ref.update(dummy_emb * 0.1)
    for _ in range(50): z_target = sig_map_ref.update(np.zeros(5))
    
    print(f"   Target Z Norm: {np.linalg.norm(z_target):.4f}")
    
    # Controller Helper
    class AdaptiveTensorModel:
        def __init__(self, A, B, N, proj, dt):
            self.A = A
            self.B = B
            self.N = N
            self.proj = proj
            self.dt = dt
            
        def step_single(self, z_red, u):
            # Reduced Dynamics
            # But A, B, N provided are FULL rank from RLS.
            # We assume we project them here or they are already reduced.
            pass
            
            # If using HOSVD online, we reduce A, B, N every step.
            # That's expensive? 30x30 SVD is cheap.
            
            # Step Full Dynamics (30 dim is fine for MPPI rollout)
            # z_new = A z + B u + N (z u)
            # Note RLS learns Discrete map directly.
            
            N_z = np.dot(self.N[0], z_red) # (n, n) * (n,) -> (n,)
            z_next = np.dot(self.A, z_red) + (self.B.flatten() * u) + (N_z * u)
            
            # No decoder ideally. Just cost on Z distance.
            return z_next
            
    traj_theta = []
    traj_u = []
    obs_history = []
    
    # Exploration Noise schedule
    noise_std = 5.0
    
    if ENABLE_BURNIN:
        print("🔥 Starting Burn-In Phase (50 Steps)...")
        for b_step in range(50):
             # Random Kick (Bang-Bang for excitement)
             u_burn = np.random.choice([-10.0, 10.0])
             
             # Step
             next_obs, _, done, _ = env.step(u_burn)
             if abs(next_obs[0]) > 2.4:
                 obs = env.reset()
                 env.state[2] = np.pi # Force Down
                 sig_map.reset()
                 obs_embedded = np.array([obs[0], obs[1], np.cos(obs[2]), np.sin(obs[2]), obs[3]])
                 z_curr = sig_map.update(obs_embedded * 0.1)
                 continue
                 
             # Feature Update
             obs_embedded_next = np.array([next_obs[0], next_obs[1], np.cos(next_obs[2]), np.sin(next_obs[2]), next_obs[3]])
             dx = obs_embedded_next - obs_embedded
             z_next = sig_map.update(dx)
             
             # RLS Update
             rls.update(z_curr, u_burn, z_next)

             
             # Iterate
             obs = next_obs
             obs_embedded = obs_embedded_next
             z_curr = z_next
             
        print("🔥 Burn-In Complete. RLS Trace:", np.trace(rls.P))
    
    print("🎬 Starting Adaptive Loop (400 steps)...")
    
    for t in range(400):
        # 1. Update Model (Offline Phase of Step)
        # We need data from PREVIOUS step.
        # But at t=0 we have none.
        
        # 2. Plan (Control Phase)
        # Get latest matrices
        A_est, B_est, N_est = rls.get_model()
        
        # Controller: MPPI on Z-space
        # Cost: ||z - z_target||_Q
        # Q: Prioritize first 5 dims
        
        # Simple specific MPPI implementation here
        H = 40 # Increased Horizon to allow discovery of Swing-Up
        N_samples = 50
        
        # Increase duration to allow for stabilization after swing-up
        Max_Steps = 600
        
        U_noise = np.random.uniform(-10, 10, (N_samples, H))
        costs = np.zeros(N_samples)
        
        # Rollout
        for i in range(N_samples):
            z_sim = z_curr.copy()
            c_cum = 0
            for t_rollout in range(H):
                u_samp = U_noise[i, t_rollout]
                
                # Dynamics with Thompson Sampling (Uncertainty Injection)
                N_z = np.dot(N_est[0], z_sim)
                lin_term = np.dot(A_est, z_sim) + np.dot(B_est, u_samp).flatten()
                bilin_term = np.dot(N_z, u_samp)
                
                z_pred = lin_term + bilin_term
                
                # Thompson Sampling: Perturb prediction by epistemic uncertainty
                # Var = phi^T P phi
                var = rls.get_variance(z_sim.flatten(), u_samp)
                sigma = np.sqrt(var)
                
                # Sample from posterior (approximate)
                # We add noise to the latent state transition
                scale = 1.0
                if ENABLE_TUNING:
                    scale = 5.0 # Boost exploration in Tuning Mode
                    
                z_sim = z_pred + np.random.normal(0, sigma * scale, size=z_pred.shape) 

                
                # Cost (Pure Task Cost)
                diff = z_sim - z_target
                
                # State cost weights
                w = np.ones_like(z_sim) * 0.1
                # Upright is roughly z[2] (cos) ~ 1 ? No, z is embedding.
                # Actually we can compute cost on OBSERVABLE (decoder not available).
                # But we know z approx [x, xd, cos, sin ...]
                w[2] = 10.0 # Cos theta weighting
                w[3] = 1.0  # Sin theta - keep it 0
                
                if ENABLE_TUNING:
                    # Improved Cost: Penalize distance to Top more aggressively
                    # And maybe relax x-penalty?
                    # Recurrent Sig target is tricky.
                    w[2] = 20.0

                
                step_cost = np.sum(w * diff**2)
                c_cum += step_cost
            costs[i] = c_cum
            
        # Optimize
        min_c = np.min(costs)
        weights = np.exp(-(costs - min_c) / 1.0) # Temp 1.0
        weights /= (np.sum(weights) + 1e-10)
        
        u_opt = np.sum(weights * U_noise[:, 0])
        
        # Exploration: Principled Only (Variance Bonus)
        # We assume the Variance Bonus in the Cost Function drives the exploration.
        # But we still need base noise for the optimizer to "see" gradients.
        
        # Adaptive Control + MPPI Noise
        u_applied = u_opt
             
        u_applied = np.clip(u_applied, -10, 10)
        
        # 3. Step Environment
        next_obs, _, done, _ = env.step(u_applied)
        # Check fail
        if abs(next_obs[0]) > 2.4:
            # We don't stop, we just reset state but keep learning? No, episode ends.
            # For adaptive control, we want to learn from failure.
            # But env resets.
            print("   💥 Crash! Resetting env but keeping memory.")
            obs = env.reset()
            # Force Down on reset to continue swing-up attempt
            env.state[2] = np.pi 
            obs[2] = np.pi
            # Reset Sig Map? Yes, path broken.
            sig_map.reset()
            obs_embedded = np.array([obs[0], obs[1], np.cos(obs[2]), np.sin(obs[2]), obs[3]])
            z_curr = sig_map.update(obs_embedded * 0.1)
            continue
            
        # 4. Update Features
        obs_embedded_next = np.array([next_obs[0], next_obs[1], np.cos(next_obs[2]), np.sin(next_obs[2]), next_obs[3]])
        dx = obs_embedded_next - obs_embedded
        z_next = sig_map.update(dx)
        
        # 5. RLS Update (The Core)
        # Learn mapping z_curr, u_applied -> z_next
        rls.update(z_curr, u_applied, z_next)

        
        # Log
        traj_theta.append(next_obs[2])
        traj_u.append(u_applied)
        obs_history.append(next_obs)
        
        # Iterate
        obs = next_obs
        obs_embedded = obs_embedded_next
        z_curr = z_next
        
        if t % 20 == 0:
            print(f"Step {t}: Theta={next_obs[2]:.2f}, u={u_applied:.2f}, RLS Trace={np.trace(rls.P):.2f}")

    # Plot
    plt.figure()
    plt.plot(np.cos(traj_theta))
    plt.title("Online Adaptive Sig-KKF Swingup")
    plt.savefig('online_adaptive_result.png')
    
    final_cos = np.mean(np.cos(traj_theta)[-50:])
    print(f"🏁 Online Adaptive Result: Final Cos={final_cos:.3f}")

    # Generate GIF
    print("🎥 Generating GIF (Frames 100-300)...")
    fig_anim, ax_anim = plt.subplots(figsize=(6, 4))
    
    # Slice interesting part (Stabilization usually after 150)
    frames = obs_history[100:350] if len(obs_history) > 350 else obs_history
    
    def update(frame_idx):
        render_cartpole(frames[frame_idx], ax_anim)
        ax_anim.set_title(f"Online Sig-KKF (Step {100+frame_idx})")
        
    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=50)
    ani.save('online_sig_kkf_swingup.gif', writer='pillow', fps=20)
    print("✅ GIF Saved to 'online_sig_kkf_swingup.gif'")

if __name__ == "__main__":
    run_online_adaptive_experiment()
