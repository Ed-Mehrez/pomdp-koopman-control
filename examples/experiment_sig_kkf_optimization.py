
import sys
import os
sys.path.append(os.getcwd()) # Add project root

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from environments.cartpole_env import CartPoleEnv
from examples.proof_of_concept.experiment_sig_kkf_learning import OnlinePolynomialFeatures, OnlineKoopmanModel, LearnedParticleController

def run_simulation(params):
    """
    Runs a simulation with given controller parameters and returns stability score.
    Params: [w_x, w_th_top, w_xd, w_thd_top, smoothing_window]
    """
    w_x, w_th_top, w_xd, w_thd_top, smoothing = params
    
    # Clip smoothing to int
    smoothing = int(max(1, min(10, smoothing)))
    
    env = CartPoleEnv(dt=0.02)
    features = OnlinePolynomialFeatures(degree=2)
    model = OnlineKoopmanModel(features, forgetting_factor=0.995)
    
    # Use baseline robust settings
    controller = LearnedParticleController(
        model, 
        horizon=50, 
        n_samples=200, 
        max_force=30.0, 
        smoothing_window=smoothing
    )
    controller.sigma = 4.0
    
    def adaptive_cost(x_curr, u_curr):
        if x_curr.ndim == 1:
            x_curr = x_curr.reshape(1, -1)
        if u_curr is not None and u_curr.ndim == 1:
            u_curr = u_curr.reshape(1, -1)
            
        x_pos = x_curr[:, 0]
        x_dot = x_curr[:, 1]
        theta = x_curr[:, 2]
        theta_dot = x_curr[:, 3]
        
        # Periodic theta
        theta_norm = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Nonlinear Factor
        upright_factor = ((1.0 + np.cos(theta)) / 2.0)**5
        
        # Weights
        w_th = 2.0 + w_th_top * upright_factor
        w_thd = 0.1 + w_thd_top * upright_factor
        
        cost = (w_x * x_pos**2 + 
                w_th * theta_norm**2 + 
                w_xd * x_dot**2 + 
                w_thd * theta_dot**2)
        
        # Action cost
        if u_curr is not None:
             cost += 0.01 * u_curr[0]**2
             
        return cost

    # Mock the cost function
    controller._cost = adaptive_cost
    
    # Phase 1: System ID (Fast)
    obs = np.array(env.reset())
    env.state[2] = np.pi # Down
    obs = np.array(env.state)
    
    for k in range(100): # Fast ID
        u = 20.0 * np.sin(k * 0.1)
        next_obs, _, _, _ = env.step(u)
        model.update(obs, [u], next_obs)
        obs = next_obs
        
    # Phase 2: Swing Up
    env.reset()
    env.state[2] = np.pi
    obs = np.array(env.state)
    
    total_upright_time = 0
    stability_penalty = 0
    
    for t in range(500): # 10 seconds
        action = controller.get_control(obs, use_barrier=False)
        next_obs, _, _, _ = env.step(action)
        model.update(obs, action, next_obs)
        obs = next_obs
        
        theta = obs[2]
        cos_theta = np.cos(theta)
        
        # Stability Score
        if cos_theta > 0.9: # Upright cone
            total_upright_time += 1
            
        # Drift Penalty
        if abs(obs[0]) > 4.0:
            stability_penalty += 100 # Reset simulation effectively
            break
            
    # Objective: Minimize Negative Upright Time + Drift Penalty
    score = -total_upright_time + stability_penalty
    print(f"Params: {params} -> Score: {score}")
    return score

def optimize():
    print("🚀 Starting Differential Evolution Optimization...")
    bounds = [
        (1.0, 20.0),   # w_x
        (20.0, 200.0), # w_th_top
        (0.01, 5.0),   # w_xd
        (5.0, 50.0),   # w_thd_top
        (1.0, 10.0)    # smoothing
    ]
    
    result = differential_evolution(
        run_simulation, 
        bounds, 
        maxiter=5, # Fast run
        popsize=3,  # Small pop
        disp=True,
        workers=1 # Single threaded for now
    )
    
    print("\n✅ Best Parameters Found:")
    print(result.x)
    return result.x

if __name__ == "__main__":
    optimize()
