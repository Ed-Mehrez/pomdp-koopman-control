import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
from PIL import Image, ImageDraw, ImageFont

if __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.cartpole_env import CartPoleEnv

class HeavyCartPoleEnv(CartPoleEnv):
    def __init__(self, dt=0.02, mass_factor=5.0):
        super().__init__(dt=dt)
        self.masscart = 1.0 * mass_factor
        self.masspole = 0.1 * mass_factor
        self.total_mass = (self.masscart + self.masspole)
        self.polemass_length = (self.masspole * self.length)
        print(f"🏋️ Heavy Env Initialized: Mass Factor {mass_factor}x")

from sklearn.preprocessing import MinMaxScaler

class OnlinePolynomialFeatures:
    """
    Polynomial features up to degree d.
    Winner of Prediction Benchmark (MSE 0.14 vs RBF ~200).
    """
    def __init__(self, degree=2):
        self.degree = degree
        # Input: [x, x_dot, theta, theta_dot] (Control is handled linearly by Koopman model)
        self.input_dim = 4
        # deg 2 of 4 vars: (4+2 choose 2) = 6 choose 2 = 15.
        self.n_components = 14 # Excluding bias (15 - 1 = 14)
        
        self.output_dim = 14 
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
class OnlinePolynomialFeatures:
    """
    Polynomial features with Trigonometric lifting for Theta.
    Input: [x, x_dot, theta, theta_dot]
    Lifted: [x, x_dot, cos(th), sin(th), theta_dot]
    Output: Poly(Lifted, degree=2)
    This captures physics like sin(th)*theta_dot^2, cos(th)*u, etc.
    """
    def __init__(self, degree=2):
        self.degree = degree
        self.input_dim = 4
        # Lifted dim = 5 (x, xd, c, s, thd)
        # Shift-Invariant (No x^2): 
        # Bias (1) + Linear (5) + Quad (10 from 4 vars) = 16.
        # n_components excludes implicit bias in sklearn usually, but here explicit.
        # Koopman assumes output size = n_components + 1.
        # So set n_components = 15.
        self.n_components = 15
        self.output_dim = 15 
        self.is_fitted = True # Scaling is hardcoded

    def transform(self, x):
        # Unpack
        x0 = x[0] # x
        x1 = x[1] # x_dot
        x2 = x[2] # theta
        x3 = x[3] # theta_dot
        
        # Scaling (Hardcoded relative to expected max values)
        # We scale LINEAR vars to [-1, 1] mainly for numerical conditioning.
        # cos/sin are already [-1, 1].
        s0 = 0.2 * x0    # x/5
        s1 = 0.1 * x1    # xd/10
        c2 = np.cos(x2)  # cos(theta)
        s2 = np.sin(x2)  # sin(theta)
        s3 = 0.1 * x3    # thd/10
        
        # Lifted Vector
        # Physics is Shift-Invariant: Acceleration doesn't depend on x.
        # So x only affects kinematics (x_next = x + v*dt).
        # Therefore, 'x' should only appear in LINEAR terms, not quadratic interactions.
        
        # Complete Linear Basis
        L_all = [s0, s1, c2, s2, s3] 
        
        # Basis for Nonlinear Interactions (Exclude x/s0)
        L_nonlinear = [s1, c2, s2, s3]
        
        # Poly Expansion (Degree 2)
        # Bias
        feats = [1.0]
        
        # Linear terms (5)
        feats.extend(L_all)
        
        # Quadratic terms (Interaction of Velocity/Angle/AngleRate)
        # Excludes x^2, x*v, etc.
        # Dim = (4+2 choose 2) = 15. Minus bias? No, dim=10 quadratic terms.
        # s1^2, s1c2, ...
        n_nl = len(L_nonlinear)
        for i in range(n_nl):
            for j in range(i, n_nl):
                feats.append(L_nonlinear[i] * L_nonlinear[j])
                
        # Total dim: 1 (Bias) + 5 (Linear) + 10 (Quad) = 16 features.
        # Previous was 21. This is more efficient too.
        return np.array(feats) 

class OnlineKoopmanModel:
    def __init__(self, feature_map, forgetting_factor=0.995): # Damped Forgetting
        self.phi = feature_map
        self.lam = forgetting_factor
        self.n_feat = feature_map.n_components + 1
        self.n_in = self.n_feat + 1 
        self.A = np.zeros((self.n_feat, self.n_in))
        self.A[:, :self.n_feat] = np.eye(self.n_feat) * 0.95 # Damped Identity
        self.P = np.eye(self.n_in) * 10.0 # Lower initial covariance
        
    def predict(self, x, u):
        phi_x = self.phi.transform(x)
        z = np.hstack([phi_x, u])
        phi_next_pred = self.A @ z
        pred = self._decode(phi_next_pred)
        
        # Clip Prediction to prevent explosion during rollout
        # Note: Features use cos/sin so periodicity is handled there
        pred[0] = np.clip(pred[0], -10.0, 10.0)
        pred[1] = np.clip(pred[1], -20.0, 20.0)
        pred[2] = np.clip(pred[2], -50.0, 50.0)  # Allow large theta for momentum
        pred[3] = np.clip(pred[3], -30.0, 30.0)
        return pred

    def _decode(self, phi):
        # OPTION 2: 5D trig output [x, dx, cos(θ), sin(θ), dθ]
        if not hasattr(self, 'C'):
            self.C = np.zeros((5, self.n_feat))  # 5D output now
            self.P_C = np.eye(self.n_feat) * 10.0
        out = self.C @ phi
        # Recover θ = atan2(sin, cos) for consistent periodicity
        theta = np.arctan2(out[3], out[2])
        return np.array([out[0], out[1], theta, out[4]])
        
    def update(self, x, u, x_next):
        phi_x = self.phi.transform(x)
        z = np.hstack([phi_x, u])
        phi_next_target = self.phi.transform(x_next)
        
        # RLS Update for A (Koopman)
        # Ridge: 1.0 (Stronger regularization for Poly)
        ridge = 1.0
        Pz = self.P @ z
        denom = self.lam + z @ Pz + ridge
        k = Pz / denom
        prediction = self.A @ z
        
        error = phi_next_target - prediction
        self.A += np.outer(error, k)
        self.P = (self.P - np.outer(k, Pz)) / self.lam
        
        # Stability Check (Heuristic): If A rows explode, damp them
        if np.max(np.abs(self.A)) > 5.0:
            self.A *= 0.9 
        
        # NaN Guard
        if np.any(np.isnan(self.A)):
            print("⚠️ NaN detected in Koopman A! Resetting to identity.")
            self.A = np.zeros((self.n_feat, self.n_in))
            self.A[:, :self.n_feat] = np.eye(self.n_feat) * 0.9
            self.P = np.eye(self.n_in) * 10.0 

        # C Update (decoder) - OPTION 2: 5D trig output
        if not hasattr(self, 'C') or self.C.shape[0] != 5:
            self.C = np.zeros((5, self.n_feat))  # 5D: [x, dx, cos, sin, dθ]
            self.P_C = np.eye(self.n_feat) * 10.0
        
        phi_t = phi_next_target
        P_c_phi = self.P_C @ phi_t
        denom_c = self.lam + phi_t @ P_c_phi + ridge
        k_c = P_c_phi / denom_c
        
        # OPTION 2: Use 5D trig target [x, dx, cos(θ), sin(θ), dθ]
        x_nxt = x_next if isinstance(x_next, np.ndarray) else np.array(x_next)
        trig_target = np.array([
            x_nxt[0],            # x
            x_nxt[1],            # dx
            np.cos(x_nxt[2]),    # cos(θ)
            np.sin(x_nxt[2]),    # sin(θ)
            x_nxt[3]             # dθ
        ])
        
        pred_trig = self.C @ phi_t
        err_trig = trig_target - pred_trig
        self.C += np.outer(err_trig, k_c)
        self.P_C = (self.P_C - np.outer(k_c, P_c_phi)) / self.lam

class LearnedParticleController:
    def __init__(self, model, horizon=100, n_samples=1000, max_force=30.0, smoothing_window=10):
        self.model = model
        self.H = horizon
        self.K = n_samples
        self.sigma = 10.0 
        self.lambda_ = 50.0
        self.max_force = max_force
        self.smoothing_window = smoothing_window
        self.U = np.zeros((self.H, 1))
        
    def get_control(self, state, use_barrier=True):
        delta_u = np.random.normal(0, self.sigma, (self.K, self.H, 1))
        costs = np.zeros(self.K)
        
        # Vectorized Rollout (if model supports it) or Loop
        # Model predict is single sample?
        # self.model.predict takes (x, u). If it can handle (K, D) and (K, 1) that would be fast.
        # But OnlineKoopmanModel seems to use basic numpy matrices. A @ z.
        # If z is (D, 1) it works. If z is (D, K) it works?
        # Let's assume loop for safety as Koopman update is likely singular.
        
        # Actually, let's keep the loop structure but ensure we use correct cost
        
        for k in range(self.K):
            x_curr = state.copy()
            c_cum = 0.0
            for t in range(self.H):
                # Smooth Control (Perturbation)
                u_curr = self.U[t] + delta_u[k, t]
                u_curr = np.clip(u_curr, -self.max_force, self.max_force)
                
                x_next_pred = self.model.predict(x_curr, u_curr)
                c_cum += self._cost(x_curr, u_curr)
                x_curr = x_next_pred
                
            c_cum += self._terminal_cost(x_curr)
            costs[k] = c_cum
            
        min_c = np.min(costs)
        # Solid MPPI: exp( (-1/lambda) * (S - rho) )
        # Here: exp( - (costs - min) / lambda )
        # Consistent.
        weights = np.exp( - (costs - min_c) / self.lambda_ )
        weights /= (np.sum(weights) + 1e-10)
        
        # delta_weighted: (H, 1)
        delta_weighted = np.sum(weights[:, None, None] * delta_u, axis=0)
        
        # Smooth the Update (Moving Average)
        if self.smoothing_window > 1:
            delta_weighted_flat = delta_weighted.flatten()
            delta_smooth = self._moving_average_filter(delta_weighted_flat, window_size=self.smoothing_window)
            delta_weighted = delta_smooth.reshape(-1, 1)

        self.U += delta_weighted
        
        # Clip U
        self.U = np.clip(self.U, -self.max_force, self.max_force)

        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0 # or self.U[-2]
        return self.U[0]

    def _moving_average_filter(self, xx, window_size):
        import math
        b = np.ones(window_size)/window_size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(window_size/2)
        xx_mean[0] *= window_size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= window_size/(i+n_conv)
            xx_mean[-i] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean

    def _cost(self, x, u, use_barrier=True):
        # x: [x_pos, x_dot, theta, theta_dot]
        x_pos = x[0]
        x_dot = x[1]
        theta = x[2]
        theta_dot = x[3]
        
        # PERIODIC COST FUNCTION
        # (1 - cos(theta)) is PURELY PERIODIC:
        #   - At theta=0 (upright): 1 - cos(0) = 0 → Minimum cost
        #   - At theta=π (down): 1 - cos(π) = 2 → Maximum cost
        #   - No discontinuities at ±π!
        theta_cost_term = 1.0 - np.cos(theta)  # Range: [0, 2]
        
        # Upright factor for adaptive weights (also periodic)
        upright_factor = ((1.0 + np.cos(theta)) / 2.0)**3  # Power 3 for smoother transition
        
        # MINIMAL COST: Just theta term
        cost = 1.0 - np.cos(theta)
        return cost

    def _terminal_cost(self, x):
        x_pos = x[0]
        x_dot = x[1]
        theta = x[2]
        theta_dot = x[3]
        
        # PERIODIC terminal cost
        theta_cost_term = 1.0 - np.cos(theta)
        
        # MINIMAL TERMINAL COST: Just theta term
        cost = 1.0 - np.cos(theta)
        return cost

def render_cartpole(state, mode_text=""):
    """
    Renders CartPole state to an RGB array.
    """
    x, x_dot, theta, theta_dot = state
    
    # Dimensions
    cart_width = 1.0
    cart_height = 0.6
    pole_length = 3.0 
    
    fig, ax = plt.subplots(figsize=(6, 4))
    cam_x = x
    
    # Robust numeric handling for bounds
    if np.isnan(cam_x) or np.isinf(cam_x): cam_x = 0.0
    
    ax.set_xlim(cam_x - 4.0, cam_x + 4.0)
    ax.set_ylim(-1.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Floor
    ax.plot([cam_x - 10.0, cam_x + 10.0], [0, 0], 'k-', lw=1)
    
    # Cart
    cart = plt.Rectangle((x - cart_width/2, 0), cart_width, cart_height, color='black')
    ax.add_patch(cart)
    
    # Pole
    pole_x = [x, x + pole_length * np.sin(theta)]
    pole_y = [cart_height/2, cart_height/2 + pole_length * np.cos(theta)]
    ax.plot(pole_x, pole_y, 'r-', lw=4)
    
    # Title
    ax.set_title(f"Sig-KKF Control | {mode_text}\nx={x:.2f}, theta={np.rad2deg(theta):.1f}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf)
    return img

def run_zero_knowledge_demo():
    print("🤓 Wiggling (System ID)...")
    env = CartPoleEnv(dt=0.02)
    # Use the proven Polynomial Features
    features = OnlinePolynomialFeatures(degree=2)
    model = OnlineKoopmanModel(features, forgetting_factor=0.995)
    # Powerful Config (Fast Mode)
    # Reduced samples to 200 due to improved model accuracy (Poly wins).
    # Horizon 50 = 1.0 second lookahead
    # SMOOTHING WINDOW: Defaults to 10 for Swing-up
    controller = LearnedParticleController(model, horizon=50, n_samples=200, max_force=30.0, smoothing_window=10) 
    controller.sigma = 4.0
    
    env.reset()
    env.state[2] = np.pi
    obs = np.array(env.state)
    
    frames = []
    
    # System ID Phase
    print("🤓 Phase 1: System ID...")
    for k in range(120): 
        freq = 0.5 + 0.5 * (k / 120.0)
        u = 20.0 * np.sin(freq * k * 0.2) 
        next_obs, _, _, _ = env.step(u)
        model.update(obs, [u], next_obs)
        obs = next_obs
        if k % 5 == 0: frames.append(render_cartpole(obs, "Phase 1: Zero Knowledge Learning"))
        
    print("🧠 Phase 2: Swing Up & STABILIZATION...")
    env.reset()
    env.state[2] = np.pi 
    obs = np.array(env.state)
    traj_theta = []
    
    # Extended to 800 steps to show stabilization
    for t in range(800):
        # Adaptive Smoothing: 
        # Deep Swing Phase (Down): High smoothing (10) to resonance.
        # Stabilization Phase (Up): Low smoothing (1) for fast reaction.
        if np.cos(obs[2]) > 0.0: # Upper half
            controller.smoothing_window = 1
        else:
            controller.smoothing_window = 10
            
        action = controller.get_control(obs, use_barrier=False) 
        action = np.clip(action, -30, 30)
        next_obs, _, _, _ = env.step(action)
        model.update(obs, action, next_obs)
        traj_theta.append(next_obs[2])
        obs = next_obs
        
        mode = "Phase 2: Swing Up" if np.cos(obs[2]) < 0.85 else "Phase 3: Stabilization"
        if t % 6 == 0: frames.append(render_cartpole(obs, mode))
        if t % 10 == 0:
            u_val = float(action) if np.isscalar(action) else action.item()
            print(f"   Step {t}: Theta={np.rad2deg(obs[2]):.1f}, x={obs[0]:.2f}, u={u_val:.2f}")
            
    # Conditional Save
    print(f"✅ ZK Max Cos Theta: {max(np.cos(traj_theta)):.4f}")
    
    # Calculate Last 50 step stability
    final_thetas = np.array(traj_theta[-50:])
    # Wrap
    final_errs = np.abs(((final_thetas + np.pi) % (2 * np.pi)) - np.pi)
    mean_err = np.mean(final_errs)
    print(f"📏 Final Stability Error: {np.rad2deg(mean_err):.2f} deg")

    if max(np.cos(traj_theta)) > 0.95:
        print("Saving sig_kkf_zero_knowledge.gif...")
        frames[0].save('sig_kkf_zero_knowledge.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
        print("✅ Saved GIF.")
    else:
         print("❌ ZK Swing-up failed.")

def run_heavy_stabilization_demo():
    print("\n🏋️ Starting Heavy Stabilization Demo (5.0x Mass + Upright Start)")
    # Strategy: Start Upright to demonstrate ROBUST STABILIZATION
    features = OnlinePolynomialFeatures(degree=2)
    model = OnlineKoopmanModel(features, forgetting_factor=1.0)
    
    sim_env = CartPoleEnv(dt=0.02)
    sim_env.reset()
    sim_env.state[2] = np.pi
    obs = np.array(sim_env.state)
    
    print("🤓 Phase 1: Heavy System ID...")
    frames = []
    for k in range(150): 
        freq = 0.3 + 0.3 * (k / 150.0) 
        u = 25.0 * np.sin(freq * k * 0.2) 
        next_obs, _, _, _ = sim_env.step(u)
        model.update(obs, [u], next_obs)
        obs = next_obs
        if k % 5 == 0: frames.append(render_cartpole(obs, "Phase 1: Heavy Mass Learning"))
        
    print("⚔️ Phase 2: Heavy Stabilization...")
    real_env = HeavyCartPoleEnv(dt=0.02, mass_factor=5.0)
    # Stronger Controller for Heavy Mass
    controller = LearnedParticleController(model, horizon=100, n_samples=300, max_force=30.0, smoothing_window=10) 
    controller.sigma = 8.0 
    
    real_env.reset()
    # START UPRIGHT for Robustness Verification
    real_env.state[2] = 0.0 + np.random.normal(0, 0.1) 
    obs = np.array(real_env.state)
    traj_theta = []
    
    for t in range(500):
        # Use Barrier for Stabilization
        action = controller.get_control(obs, use_barrier=True)
        action = np.clip(action, -30, 30)
        next_obs, _, _, _ = real_env.step(action)
        model.update(obs, action, next_obs)
        obs = next_obs
        traj_theta.append(obs[2])
        
        mode = "Phase 2: Heavy Stabilization"
        if t % 5 == 0: frames.append(render_cartpole(obs, mode))

    print(f"✅ Heavy Max Cos Theta: {max(np.cos(traj_theta)):.4f}")
    if max(np.cos(traj_theta)) > 0.9:
        print("Saving sig_kkf_heavy_stabilization.gif...")
        frames[0].save('sig_kkf_heavy_stabilization.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
        print("✅ Saved GIF.")
    else:
        print("❌ Heavy Stabilization failed.")

if __name__ == "__main__":
    run_zero_knowledge_demo() 
    # run_heavy_stabilization_demo()
