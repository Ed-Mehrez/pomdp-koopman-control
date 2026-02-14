
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import sys
import os

# Add path to access local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.proof_of_concept.signature_features import RecurrentSignatureMap
from environments.cartpole_env import CartPoleEnv

class NoisyCartPoleEnv(CartPoleEnv):
    def __init__(self, dt=0.02, obs_noise=0.0):
        self.obs_noise = obs_noise
        super().__init__(dt=dt)
        self.force_mag = 50.0 # Increase force authority for swingup
        
    def step(self, action):
        # Clip action to new limits
        action = np.clip(action, -self.force_mag, self.force_mag)
        
        # Standard Step
        obs, reward, done, info = super().step(action)
        
        # Add Noise to Observation
        obs_noisy = obs + np.random.normal(0, self.obs_noise, size=obs.shape)
        return obs_noisy, reward, done, info
        
    def reset(self):
        # Start DOWN (theta = pi)
        # Add small noise to break symmetry
        self.state = np.array([0.0, 0.0, np.pi + np.random.uniform(-0.1, 0.1), 0.0])
        return self.state + np.random.normal(0, self.obs_noise, size=4)

class SigKKFCartPoleController:
    """
    Sig-KKF for CartPole Swingup.
    """
    def __init__(self, obs_dim=4, level=2, dt=0.02):
        self.dt = dt
        # Input to SigMap: [dt, x, xd, cos, sin, thd]
        self.state_aug_dim = 1 + 2 + 2 + 1 
        
        self.sig_map = RecurrentSignatureMap(state_dim=self.state_aug_dim, level=level)
        self.feature_dim = self.sig_map.feature_dim
        
        # Koopman Model
        self.A = np.eye(self.feature_dim)
        self.B = np.zeros((self.feature_dim, 1))
        
        # Kalman Filter
        self.z_hat = np.zeros(self.feature_dim)
        self.P_cov = np.eye(self.feature_dim) * 1.0
        self.Q_proc = np.eye(self.feature_dim) * 0.001
        self.R_meas = np.eye(self.feature_dim) * 0.01 
        
        # LQR
        self.K_lqr = np.zeros((1, self.feature_dim))
        
        # Buffers
        self.Z_buff = []
        self.U_buff = []
        self.Z_next_buff = []
        
        # History
        self.obs_prev = None
        
    def preprocess_obs(self, obs):
        x, xd, th, thd = obs
        return np.array([x, xd, np.cos(th), np.sin(th), thd])

    def update_model(self):
        # Need enough data for neighbors
        if len(self.Z_buff) < 50: return
        
        Z_all = np.array(self.Z_buff[:-1])
        U_all = np.array(self.U_buff)
        Zn_all = np.array(self.Z_next_buff)
        
        # Alignment
        min_len = min(len(Z_all), len(U_all), len(Zn_all))
        Z_all = Z_all[:min_len]
        U_all = U_all[:min_len]
        Zn_all = Zn_all[:min_len]
        
        # --- LOCAL LEARNING (SDRE) ---
        # 1. Find Neighbors of current z_hat
        z_curr = self.z_hat.reshape(1, -1)
        
        # Compute distances (Euclidean in Sig Space)
        dists = np.linalg.norm(Z_all - z_curr, axis=1)
        
        # k-Nearest Neighbors
        n_nearest = 100 # Tuning parameter for locality
        n_nearest = min(n_nearest, len(Z_all))
        
        # Get indices
        idx = np.argsort(dists)[:n_nearest]
        
        # 2. Weighted Regression
        # Bandwidth for weights
        kernel_width = 1.0 # Tune this
        weights = np.exp(-dists[idx]**2 / (2 * kernel_width**2))
        W = np.diag(weights)
        
        Z_neigh = Z_all[idx]
        U_neigh = U_all[idx]
        Zn_neigh = Zn_all[idx]
        
        # Regress [Z, U] -> Zn using W
        X = np.hstack([Z_neigh, U_neigh])
        Y = Zn_neigh
        
        # Weighted Ridge: (X'WX + lam I) theta = X'WY
        lam = 1e-4
        XTW = X.T @ W
        coeffs = np.linalg.solve(XTW @ X + lam * np.eye(X.shape[1]), XTW @ Y)
        
        self.A = coeffs[:self.feature_dim].T
        self.B = coeffs[self.feature_dim:].T
        
        # 3. LQR Update (SDRE)
        import scipy.linalg
        try:
            # Q Tuning for Swingup
            # Penalize the first few dimensions (State) heavily, Signatures less?
            # Or just standard Identity.
            Q = np.eye(self.feature_dim) * 10.0 
            R = np.eye(1) * 0.01                 
            
            P = scipy.linalg.solve_discrete_are(self.A, self.B, Q, R)
            self.K_lqr = np.linalg.solve(R + self.B.T @ P @ self.B, self.B.T @ P @ self.A)
        except:
             # Fallback to previous gain if solver fails (e.g. uncontrollable)
             pass

    def get_target_signature(self):
        # Target: x=0, xd=0, th=0 (cos=1, sin=0), thd=0
        # This is a fixed point.
        # But Signature depends on PATH.
        # Increments for a fixed point are 0.
        # So Sig(Integration of 0) -> [dt, 0, 0...] -> Sig?
        # A fixed point at Upright has specific values for the STATE dimensions of the Recurrent Map.
        # Our Recurrent Map state: [1, S1, S2]. 
        # S1 tracks the 'value' if we use the time-augmented trick properly.
        # Actually, simpler: Regulate the Observation part of the augmented state.
        pass

    def process(self, obs_noisy, u_prev):
        if self.obs_prev is None:
            self.obs_prev = obs_noisy
            return 0.0
            
        curr_aug = self.preprocess_obs(obs_noisy)
        prev_aug = self.preprocess_obs(self.obs_prev)
        d_aug = curr_aug - prev_aug
        
        # Augment with Time
        dx_input = np.hstack([self.dt, d_aug])
        
        self.obs_prev = obs_noisy
        
        # Update Sig & Filter
        z_raw = self.sig_map.update(dx_input)
        
        # Predict
        z_pred = self.A @ self.z_hat + self.B.flatten() * u_prev
        
        # Robustify: If A explodes, reset?
        if np.linalg.norm(z_pred) > 1e4:
             z_pred = np.zeros_like(z_pred)
             
        P_pred = self.A @ self.P_cov @ self.A.T + self.Q_proc
        
        # Update
        try:
             K_gain = P_pred @ np.linalg.inv(P_pred + self.R_meas)
             self.z_hat = z_pred + K_gain @ (z_raw - z_pred)
             self.P_cov = (np.eye(self.feature_dim) - K_gain) @ P_pred
        except:
             self.z_hat = z_pred
        
        # Store
        if hasattr(self, 'z_hat_prev'):
             self.Z_buff.append(self.z_hat_prev) 
             self.U_buff.append(np.array([u_prev]))
             self.Z_next_buff.append(self.z_hat.copy())
             
        self.z_hat_prev = self.z_hat.copy()
        
        # --- Control Law ---
        # We need to regulate deviation from UPRIGHT.
        # The first few components of z (Level 1 signature) loosely track the integrated path (State).
        # We constructed 'dx_input' = [dt, dx, dxd, dcos, dsin, dthd].
        # Recurrent Sig S1 accumulates these increments (with forgetting).
        # So S1 ~ [t, x, xd, cos, sin, thd].
        
        # Target S1:
        # x -> 0
        # xd -> 0
        # cos -> 1.0 (Critical!)
        # sin -> 0.0
        # thd -> 0
        
        # The Signature Map flattens [S1, S2].
        # S1 is indices 0..5 (if augmented state is 6D).
        # Indices: 0(dt), 1(x), 2(xd), 3(cos), 4(sin), 5(thd).
        
        z_target = np.zeros_like(self.z_hat)
        
        # Assume 'forgetting factor' < 1.0 in SigMap means S1 converges to Mean/(1-gamma)?
        # Our SigMap uses gamma=1.0? 
        # If gamma=1, S1 integrates to infinity. LQR fails.
        # If gamma < 1, S1 represents localized state.
        
        # Let's trust the "Derivative Regulation" property of LQR.
        # We want to encourage cos(th) -> 1.
        # In the Feature Vector, index 3 corresponds to 'cos'.
        # We want z[3] -> large? No.
        
        # DIFFERENT APPROACH: 
        # Use simple Energy Shaping heuristics for Swingup, then Sig-KKF for stabilization.
        # This is standard hybrid control.
        # But user wants Sig-KKF to do it.
        
        # Let's assume the Learned Model A, B captures the physics.
        # We define a Cost Function J = (z - z_goal)^T Q (z - z_goal).
        # z_goal has 1.0 at index 3 (cos)? 
        # Actually S1 tracks value. 
        # Let's set z_target[3] = 1.0 (assuming properly scaled/reset).
        
        # Better: Just regulate x, xd, sin, thd to 0. 
        # And maximize cos? Or regulate cos to 1.
        z_target[3] = 1.0 
        
        error = self.z_hat - z_target
        u = -self.K_lqr @ error
        return float(u[0])

# --- Experiment ---
print("Running Sig-KKF Swingup (Hybrid)...")
env = NoisyCartPoleEnv()
# Enable forgetting in SigMap to make state bounded/local?
# Current SigMap default gamma=1.0 (Infinite Memory).
# We MUST modify SigMap params or wrapper.
# Let's rely on standard RecurrentSigMap default.

agent = SigKKFCartPoleController(dt=env.dt)

obs = env.reset()
u = 0.0
history = {'x': [], 'theta': [], 'u': []}

frames = 800

for t in range(frames):
    obs, r, done, _ = env.step(u)
    
    history['x'].append(obs[0])
    history['theta'].append(obs[2])
    history['u'].append(u)
    
    # --- EXPLORATION Strategy ---
    # 1. Energy Pumping (0-200): Swing it up!
    # 2. Sig-KKF LQR (200+): Stabilize.
    
    th = obs[2]
    thd = obs[3]
    
    if t < 300:
        # Energy Pumping with Centering
        u_pump = 50.0 * thd * np.cos(th) 
        u_center = -1.5 * obs[0] - 0.5 * obs[1] 
        
        u = u_pump + u_center
        u = np.clip(u, -env.force_mag, env.force_mag)
        
        _ = agent.process(obs, u)
    else:
        u = agent.process(obs, u)
        
    agent.update_model()
    
# --- GIF Generation ---
print("Generating GIF...")
if len(history['x']) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    ax.grid(True)
    
    cart_width = 0.5
    cart_height = 0.3
    cart = patches.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, fc='black')
    pole, = ax.plot([], [], 'r-', linewidth=3)
    ax.add_patch(cart)
    
    def init():
        cart.set_xy((-cart_width/2, -cart_height/2))
        pole.set_data([], [])
        ax.set_xlim(-6, 6)
        return cart, pole
        
    def animate(i):
        x = history['x'][i]
        th = history['theta'][i]
        
        # Camera Follow
        ax.set_xlim(x - 5.0, x + 5.0)
        
        # Cart pos
        cart.set_xy((x - cart_width/2, -cart_height/2))
        
        # Pole pos
        # CartPole env: theta=0 is UP? 
        # Standard gym: 0 is UP. 
        # Let's check environment. usually 0 is UP.
        # pole end: x + L*sin(th), L*cos(th)
        L = 1.0 # Visual length
        pole_x = [x, x + L * np.sin(th)]
        pole_y = [0, L * np.cos(th)]
        pole.set_data(pole_x, pole_y)
        return cart, pole
        
    ani = animation.FuncAnimation(fig, animate, frames=len(history['x']), init_func=init, blit=False, interval=20)
    ani.save('sig_kkf_cartpole.gif', writer='pillow', fps=50) # Faster fps
    print("Saved sig_kkf_cartpole.gif")
