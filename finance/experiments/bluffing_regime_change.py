
import numpy as np
import matplotlib.pyplot as plt
import iisignature
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

def simulate_gbm(n_steps, dt=0.005, mu=0.05, sigma=0.2, S0=100.0, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, n_steps*dt, n_steps+1)
    W = np.random.standard_normal(size=n_steps)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t[:-1] + sigma * W
    S = S0 * np.exp(X)
    return np.diff(np.log(S)) # Log returns

def simulate_heston_matched(n_steps, dt=0.005, theta=0.04, kappa=2.0, xi=0.3, rho=-0.7, seed=43):
    # theta = long run variance. If sigma_gbm = 0.2, then sigma^2 = 0.04.
    # So we set theta = 0.04 to match the GBM volatility on average.
    np.random.seed(seed)
    v = np.zeros(n_steps + 1)
    v[0] = theta
    
    Z1 = np.random.normal(0, 1, n_steps)
    Z2 = np.random.normal(0, 1, n_steps)
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    S = np.zeros(n_steps+1)
    S[0] = 100.0
    
    # Heston
    for t in range(n_steps):
        v_t = max(0, v[t])
        dv = kappa * (theta - v_t) * dt + xi * np.sqrt(v_t * dt) * W2[t]
        v[t+1] = v_t + dv
        
        dS = S[t] * np.sqrt(v_t * dt) * W1[t] # Zero drift
        S[t+1] = S[t] + dS
        
    return np.diff(np.log(S)) # Log returns

def run_bluffing_detection():
    print("Simulating Regime Change (GBM -> Heston)...")
    n_steps_1 = 1000
    n_steps_2 = 1000
    dt = 0.005
    
    # Regime 1: GBM (Standard, "Safe")
    # Vol = 20%
    rets_1 = simulate_gbm(n_steps_1, dt=dt, sigma=0.20)
    
    # Regime 2: Heston (Stochastic Vol, "Danger", but matched avg vol)
    # Theta = 0.04 <=> Vol 20%
    rets_2 = simulate_heston_matched(n_steps_2, dt=dt, theta=0.04)
    
    # Stitch
    rets = np.concatenate([rets_1, rets_2])
    price_path = np.exp(np.cumsum(np.concatenate([[0], rets]))) * 100.0
    
    # --- Detection: Rolling Signatures ---
    print("Computing Rolling Signatures...")
    window = 50
    degree = 2 
    
    # We use (Time, Cumulative Return) path for signature
    # Signature of independent increments (GBM) vs dependent (Heston)
    # Level 2 sig has 1s, 11, 12, 21, 22 terms.
    # Area term 12 - 21 captures dependency? 
    # Actually for 1D path (Time, X), Area computes roughly integral X dt - integral t dX.
    # We need something that captures volatility clustering. 
    # Maybe (Time, Cumsum_Ret, Cumsum_SqRet) path?
    # Adding "Lead-Lag" transformation or "Time-joined" path is standard.
    # Let's use (Time, Price) path.
    
    # Path: (Time, Cumulative Return, Quadratic Return)
    # This exposes the correlation between Price and Volatility (Leverage) to the signature
    def get_rolling_sigs(series, w):
        path_sigs = []
        t_seq = np.linspace(0, w*dt, w+1)
        for i in range(w, len(series)):
            seg = series[i-w:i]
            y_path = np.cumsum(np.concatenate([[0], seg]))
            q_path = np.cumsum(np.concatenate([[0], seg**2]))
            
            # 3D Path: Time, Returns, RealizedVariance
            path = np.column_stack([t_seq, y_path, q_path])
            
            sig = iisignature.sig(path, degree)
            path_sigs.append(sig)
        return np.array(path_sigs)
    
    sigs = get_rolling_sigs(rets, window)
    
    # Baseline: First 500 steps (Assumed known "Normal" state)
    train_end = 500
    baseline_sigs = sigs[:train_end]
    baseline_mean = np.mean(baseline_sigs, axis=0)
    baseline_cov = np.cov(baseline_sigs, rowvar=False)
    # Regularize cov
    baseline_cov_inv = np.linalg.pinv(baseline_cov + 1e-6 * np.eye(baseline_cov.shape[0]))
    
    # Anomaly Score: Mahalanobis Distance
    distances = []
    for s in sigs:
        diff = s - baseline_mean
        d = np.sqrt(diff.T @ baseline_cov_inv @ diff)
        distances.append(d)
        
    distances = np.array(distances)
    
    # Padding for alignment
    distances_padded = np.concatenate([np.zeros(window), distances])
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(price_path, 'k-', linewidth=1.5)
    ax1.axvline(n_steps_1, color='r', linestyle='--', label="Regime Change (GBM -> Heston)")
    ax1.set_title("Price Path (Vol matched, Structure changes)")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(distances_padded, 'b-', linewidth=1)
    ax2.axvline(n_steps_1, color='r', linestyle='--')
    ax2.set_title("Signature Anomaly Score (Mahalanobis Dist)")
    ax2.set_ylabel("Deviation from Normal")
    ax2.set_xlabel("Time Step")
    ax2.grid(True, alpha=0.3)
    
    # Highlight the regime 2
    ax2.axvspan(n_steps_1, len(price_path), color='r', alpha=0.1, label="Heston Regime")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bluffing_detection.png'), dpi=150)
    print("Bluffing detection figure saved to bluffing_detection.png")
    
    # Check if detection successful (Mean score in Regime 2 > Mean Score in Regime 1)
    score_1 = np.mean(distances[train_end:n_steps_1-window])
    score_2 = np.mean(distances[n_steps_1-window:])
    print(f"Mean Score Regime 1: {score_1:.2f}")
    print(f"Mean Score Regime 2: {score_2:.2f}")
    print(f"Ratio: {score_2/score_1:.2f}")

if __name__ == "__main__":
    run_bluffing_detection()
