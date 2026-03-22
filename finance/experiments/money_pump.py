
import numpy as np
import matplotlib.pyplot as plt
import iisignature
from sklearn.linear_model import LinearRegression, Ridge
import sys
import os

# Ensure clean output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

def simulate_heston(n_steps=5000, dt=0.005, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, seed=42):
    """Simulate Heston Model (Stochastic Volatility)"""
    np.random.seed(seed)
    S = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    S[0] = 100.0
    v[0] = theta
    
    Z1 = np.random.normal(0, 1, n_steps)
    Z2 = np.random.normal(0, 1, n_steps)
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    for t in range(n_steps):
        v_t = max(0, v[t])
        dv = kappa * (theta - v_t) * dt + xi * np.sqrt(v_t * dt) * W2[t]
        v[t+1] = v_t + dv
        
        dS = S[t] * np.sqrt(v_t * dt) * W1[t] # Assume zero drift for simplicity in vol trading
        S[t+1] = S[t] + dS
        
    log_ret = np.diff(np.log(S))
    # True "Realized Variance" over the step is roughly v_t * dt 
    # But for trading, we bet on the NEXT squared return r_{t+1}^2
    true_next_sq_ret = log_ret**2 # This is the noisy proxy for realized variance
    
    return log_ret, v[:-1]

def run_money_pump():
    print("Simulating Heston 'Money Pump' Scenario...")
    log_ret, true_vol_state = simulate_heston(n_steps=4000)
    
    # Game: Trading "Next Step Variance"
    # Target: r_{t+1}^2
    target = log_ret[1:]**2
    
    # We need a training period (burn-in)
    train_size = 2000
    test_size = 1900
    
    # --- Player A: The "Linear" Market Maker ---
    # Uses Rolling Moving Average of past squared returns is the standard "Linear" forecast
    # Model: E[r_{t+1}^2] ~ Mean(r_{t-k}^2 ... r_t^2)
    window = 20
    
    def get_rolling_preds(series, w):
        preds = np.zeros(len(series))
        for i in range(w, len(series)):
            preds[i] = np.mean(series[i-w:i])
        return preds
        
    preds_A = get_rolling_preds(log_ret[:-1]**2, window)
    
    # --- Player B: The "Signature" Arbitrager ---
    # Uses Signatures of the path (Lead-Lag, Vol-of-Vol info) to predict variance
    print("Training Signature Model...")
    
    def make_sig_features(rets, w, degree=2):
        feats = []
        t_seq = np.linspace(0, w*0.005, w+1)
        # Precompute signature object? iterating is slow but ok for 4000
        for i in range(w, len(rets)):
            # Segment: Price path? Or Cumulative Return path?
            # Use Time-Augmented Cumulative Return Loop
            segment_r = rets[i-w:i]
            path_val = np.cumsum(segment_r)
            path_val = path_val - path_val[0] # Zero start
            path = np.column_stack([t_seq[:-1], path_val]) # Simple T, X path
            
            sig = iisignature.sig(path, degree)
            feats.append(sig)
        return np.array(feats)
    
    # Prepare data for B
    # X needs to align with 'preds_A' indexing
    # preds_A[i] is forecast for target[i] made at time i
    # features made at time i use rets[:i]
    
    X_sigs = make_sig_features(log_ret[:-1], window)
    # X_sigs[0] corresponds to index 'window' in log_ret
    
    # Align everything
    # Valid indices: window to end
    start_idx = window
    
    # Targets for regression
    y_all = log_ret[start_idx+1:]**2 # Next step sq ret
    
    # X for regression
    X_all = X_sigs[:-1] # Drop last because we have no target for it
    
    # Check lengths
    common_len = min(len(y_all), len(X_all))
    X_all = X_all[:common_len]
    y_all = y_all[:common_len]
    
    # Split
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test = X_all[train_size:train_size+test_size]
    y_test = y_all[train_size:train_size+test_size]
    
    # Train B
    model_B = Ridge(alpha=1e-3)
    model_B.fit(X_train, y_train)
    preds_B_test = model_B.predict(X_test)
    preds_A_test = preds_A[window:][:-1][:common_len][train_size:train_size+test_size]
    
    # --- The Trading Game ---
    # Market Price = Price A (Market Maker)
    # Trader B buys if B_pred > A_pred, Sells if B_pred < A_pred
    # Actually, let's say they trade against each other.
    # PnL for B = Position * (Realized - MarketPrice)
    
    realized_test = y_test
    
    # Strategy: B trades deviation from A
    # Position ~ (Pred_B - Pred_A)
    # Scale position for realism (e.g. max leverage)
    position = np.sign(preds_B_test - preds_A_test) 
    
    # Profit Calculation
    # If B buys (Pos=1) at Price A, and Realized is Higher, B makes money.
    pnl_per_trade = position * (realized_test - preds_A_test)
    cumulative_pnl = np.cumsum(pnl_per_trade)
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(cumulative_pnl, 'g-', linewidth=2, label="Sig-Trader (Player B) Wealth")
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.title("The 'Money Pump': Signature Trader vs Linear Market Maker")
    plt.ylabel("Cumulative PnL (Variance Swap Units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Show prediction quality snippet
    zoom = 100
    plt.plot(realized_test[:zoom], 'k-', alpha=0.2, label="Realized SqRet")
    plt.plot(preds_A_test[:zoom], 'b--', alpha=0.6, label="Linear (MA) Prediction")
    plt.plot(preds_B_test[:zoom], 'r-', alpha=0.8, label="Sig Prediction")
    plt.title("Prediction Quality (First 100 steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'money_pump.png'), dpi=150)
    print("Money Pump figure saved to money_pump.png")
    print(f"Final PnL: {cumulative_pnl[-1]:.4f}")

if __name__ == "__main__":
    run_money_pump()
