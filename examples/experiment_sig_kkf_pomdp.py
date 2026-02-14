
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path to access local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.proof_of_concept.signature_features import RecurrentSignatureMap

class NoisyDoubleWellEnv:
    """
    POMDP Version of Double Well.
    True State: x (position), v (velocity)
    Observation: y = x + noise
    Potential: V(x) = (x^2 - 1)^2
    """
    def __init__(self, dt=0.05, obs_noise=0.3):
        self.dt = dt
        self.obs_noise = obs_noise
        self.state = np.array([-1.0]) # Start in left well
        
    def reset(self):
        self.state = np.array([-1.0]) + np.random.normal(0, 0.1, 1)
        return self.observe()
        
    def step(self, u):
        # Langevin Dynamics: dx = -V'(x)dt + u*dt + sigma*dW
        x = self.state[0]
        # V'(x) = 4*x^3 - 4*x
        force = -(4 * x**3 - 4 * x)
        
        # Clip control for realism
        u = np.clip(u, -5.0, 5.0)
        
        # Update
        dx = force * self.dt + u * self.dt # + np.random.normal(0, 0.1) * np.sqrt(self.dt)
        self.state += dx
        
        # Partial Observation
        y = self.observe()
        
        # Reward (Target: Right Well +1.0)
        dist = np.abs(self.state[0] - 1.0)
        reward = -dist
        
        return y, reward, False, {}
        
    def observe(self):
        # We only see noisy position
        return self.state + np.random.normal(0, self.obs_noise, 1)

class SigKKFController:
    """
    Signature-based Koopman Kalman Filter Controller.
    Pipeline:
    1. y_t (Noisy Obs) -> RecurrentSignatureMap -> z_t (Signature State)
    2. Kalman Filter on z_t (denoising in feature space) -> z_hat_t
    3. LQR Control on z_hat_t: u = -K z_hat_t
    """
    def __init__(self, obs_dim=1, level=2, dt=0.05):
        self.dt = dt
        self.sig_map = RecurrentSignatureMap(state_dim=obs_dim+1, level=level) # +1 for Control aug? No, just obs path.
        self.feature_dim = self.sig_map.feature_dim
        
        # 1. Koopman Model (Linear in Signature Space)
        # z_{t+1} = A z_t + B u_t
        self.A = np.eye(self.feature_dim)
        self.B = np.zeros((self.feature_dim, 1))
        
        # 2. Kalman Filter State
        self.z_hat = np.zeros(self.feature_dim)
        self.P_cov = np.eye(self.feature_dim) * 1.0
        self.Q_proc = np.eye(self.feature_dim) * 0.01
        self.R_meas = np.eye(self.feature_dim) * 0.1 # This is tricky, we don't observe z directly? 
        # Actually in Sig-KKF, we treat the 'Raw Signature' (computed from noisy y) as the 'Measurement'.
        
        # 3. LQR Gain
        self.K_lqr = np.zeros((1, self.feature_dim))
        
        # Buffers for Online Learning
        self.Z_buff = []
        self.U_buff = []
        self.Z_next_buff = []
        
    def update_model(self):
        # Solve Least Squares for A, B using collected buffer
        # z_next = A z + B u
        if len(self.Z_buff) < 50: return
        
        Z = np.array(self.Z_buff[:-1])
        U = np.array(self.U_buff)
        Zn = np.array(self.Z_next_buff)
        
        # Regress [Z, U] -> Zn
        X = np.hstack([Z, U])
        target = Zn
        
        # Ridge Regression
        lam = 0.1
        coeffs = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ target)
        
        self.A = coeffs[:self.feature_dim].T
        self.B = coeffs[self.feature_dim:].T
        
        # Update LQR (Algebraic Riccati)
        # Check controllability first?
        # For simplicity, solve DARE or just use B.T P (approx)
        import scipy.linalg
        try:
            Q_lqr = np.eye(self.feature_dim)
            R_lqr = np.eye(1) * 0.1
            P = scipy.linalg.solve_discrete_are(self.A, self.B, Q_lqr, R_lqr)
            self.K_lqr = np.linalg.solve(R_lqr + self.B.T @ P @ self.B, self.B.T @ P @ self.A)
            # print("LQR Updated")
        except:
            pass
            
    def process(self, y_obs, u_prev):
        # 1. Compute Raw Signature z_raw from noisy observation
        # We feed (y_obs - y_prev) as increment?
        # Or we rely on Lead-Lag?
        # Simple for now: just use raw increments of y
        if not hasattr(self, 'y_prev'):
            self.y_prev = y_obs
            return 0.0
            
        dy = y_obs - self.y_prev
        self.y_prev = y_obs
        
        # Augment with Time Increment dt
        # Input to SigMap must be size d=2 (Time, Obs)
        dx_aug = np.array([self.dt, dy[0]])
        
        # Update Recurrent Signature (Noisy)
        z_raw = self.sig_map.update(dx_aug)
        
        # 2. Kalman Filter Step
        # Predict: z_pred = A z_hat + B u
        z_pred = self.A @ self.z_hat + self.B.flatten() * u_prev
        P_pred = self.A @ self.P_cov @ self.A.T + self.Q_proc
        
        # Update: Innovation = z_raw - z_pred (Treating z_raw as 'sensor')
        # This is a bit non-standard (Sig-KKF usually filters y then computes sig), 
        # but let's try 'Filtering in Sig Space' directly.
        K_gain = P_pred @ np.linalg.inv(P_pred + self.R_meas)
        self.z_hat = z_pred + K_gain @ (z_raw - z_pred)
        self.P_cov = (np.eye(self.feature_dim) - K_gain) @ P_pred
        
        # 3. Store for Learning
        self.Z_buff.append(self.z_hat.copy()) # Use filtered z for training? Or raw? 
        # Usually train on clean, but we only have estimates.
        # Let's train on z_hat_prev -> z_hat_curr
        if hasattr(self, 'z_hat_prev'):
            self.Z_buff[-1] = self.z_hat_prev # Fix buffer alignment
            self.U_buff.append(np.array([u_prev]))
            self.Z_next_buff.append(self.z_hat.copy())
            
        self.z_hat_prev = self.z_hat.copy()
        
        # 4. Control
        # u = -K z
        u = -self.K_lqr @ self.z_hat
        return u[0]

# --- Main Loop ---
print("Running Sig-KKF Experiment on Noisy Double Well...")

env = NoisyDoubleWellEnv()
agent = SigKKFController(dt=env.dt)

Y_hist = []
X_true_hist = []
U_hist = []

obs = env.reset()
u = 0.0

for t in range(500):
    # Step
    obs, r, _, _ = env.step(u)
    
    Y_hist.append(obs)
    X_true_hist.append(env.state[0])
    
    # Agent
    if t < 50:
        # Exploration / Data Collection
        u = np.random.normal(0, 2.0)
    else:
        # Closed Loop
        u = agent.process(obs, u)
        
    # Online Learning (simulate every step)
    agent.update_model()
    U_hist.append(u)

# Plot
plt.figure(figsize=(10,6))
plt.plot(X_true_hist, 'k-', linewidth=2, label='True State (Hidden)')
plt.plot(Y_hist, 'g.', alpha=0.3, label='Noisy Obs')
plt.plot(U_hist, 'r-', alpha=0.5, label='Control')
plt.axhline(1.0, color='b', linestyle='--', label='Target')
plt.legend()
plt.title('Sig-KKF Control of Noisy Double Well (POMDP)')
plt.savefig('sig_kkf_pomdp_result.png')
print("Done. Saved plot.")
