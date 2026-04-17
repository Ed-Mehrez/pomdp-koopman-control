import numpy as np
import matplotlib.pyplot as plt
import iisignature
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
try:
    from finance.signature_volatility import SignatureVolatilityEstimator
except ImportError:
    SignatureVolatilityEstimator = None
from sskf.streaming_sig_kkf import LogSignatureState
from sskf.online_path_features import RandomProjectionNystrom

def simulate_heston(T=1.0, N=10000, S0=1.0, V0=0.04, kappa=5.0, theta=0.04, sigma=0.5, rho=-0.7, mu=0.05, sigma_noise=0.0):
    """
    Simulate Heston model using Euler-Maruyama scheme.
    Optionally injects i.i.d. Gaussian microstructure noise into the observed price.
    dS_t = mu * S_t * dt + sqrt(V_t) * S_t * dW_t^S
    dV_t = kappa * (theta - V_t) * dt + sigma * sqrt(V_t) * dW_t^V
    dW_t^S * dW_t^V = rho * dt
    """
    dt = T / N
    S = np.zeros(N+1)
    V = np.zeros(N+1)
    S[0] = S0
    V[0] = V0
    
    # Generate correlated Brownian motions
    Z1 = np.random.randn(N)
    Z2 = np.random.randn(N)
    W_S = np.sqrt(dt) * Z1
    W_V = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
    
    for i in range(N):
        # Truncation scheme for V to prevent negative variance
        v_plus = max(V[i], 0)
        
        S[i+1] = S[i] + mu * S[i] * dt + np.sqrt(v_plus) * S[i] * W_S[i]
        V[i+1] = V[i] + kappa * (theta - v_plus) * dt + sigma * np.sqrt(v_plus) * W_V[i]
        
    return S, V, dt

def realized_variance(returns, window):
    """Computes rolling Realized Variance (RV) over a window."""
    squared_ret = returns**2
    rv = np.convolve(squared_ret, np.ones(window), mode='valid')
    # Pad beginning
    rv = np.concatenate([np.ones(window-1) * rv[0], rv])
    return rv

def bipower_variation(returns, window):
    """Computes rolling Bipower Variation (BV) over a window."""
    abs_ret = np.abs(returns)
    cross_terms = abs_ret[1:] * abs_ret[:-1]
    bv = (np.pi / 2.0) * np.convolve(cross_terms, np.ones(window), mode='valid')
    # Pad beginning
    bv = np.concatenate([np.ones(window) * bv[0], bv])
    return bv

def rolling_mean_causal(values, window):
    """Causal rolling average aligned with the local-window estimators."""
    kernel = np.ones(window) / window
    out = np.convolve(values, kernel, mode='valid')
    return np.concatenate([np.full(window - 1, out[0]), out])

def lead_lag_transform(path):
    """Computes the lead-lag transform of a 1D path."""
    n = len(path)
    ll_path = np.zeros((2*n-1, 2))
    ll_path[0::2, 0] = path
    ll_path[1::2, 0] = path[1:]
    ll_path[0::2, 1] = path
    ll_path[1::2, 1] = path[:-1]
    return ll_path

def signature_volatility_estimator(returns, window, level=2):
    """
    Computes a linear projection from Lead-Lag Signatures.
    Because computing signatures over rolling windows of high-frequency data can be slow
    if done naively, we use a stride or simple block-based approach for demonstration.
    """
    n = len(returns)
    path = np.cumsum(returns)
    path = np.concatenate([[0], path])
    
    sig_vols = np.zeros(n)
    
    # Pre-compute to save time - we will extract the signature for blocks
    # Actually, the Lead-Lag signature's specific cross term is exactly the quadratic variation.
    # Sig(X_LL)_2 contains \int X_lead dX_lag.
    
    # We will just do a block based approach for speed
    step = max(1, window // 10)
    for i in range(window, n, step):
        block_path = path[i-window:i]
        ll_block = lead_lag_transform(block_path)
        sig = iisignature.sig(ll_block, level)
        # For a 2D path, level 2 has 2^1 + 2^2 = 6 terms
        # Term index 0: X_lead, 1: X_lag
        # Term index 2: X_lead \otimes X_lead
        # Term index 3: X_lead \otimes X_lag
        # Term index 4: X_lag \otimes X_lead
        # Term index 5: X_lag \otimes X_lag
        # Quadratic variation is proportional to term 3 - term 4 (Area)
        # 1/2 * (path[-1]-path[0])^2 - \int X dX
        qv_sig = sig[3] - sig[4] 
        sig_vols[i:min(i+step, n)] = qv_sig
        
    return sig_vols

def logsignature_volatility_estimator(returns, window, level=2):
    """
    Computes volatility using the native BCH Log-Signature stream.
    We maintain a log-signature over the rolling window. For speed in Python,
    we compute the windowed log-signature directly taking increments.
    """
    n = len(returns)
    sig_vols = np.zeros(n)
    
    # Block-based for Python speed
    step = max(1, window // 10)
    for i in range(window, n, step):
        # Fresh log-signature state for the window
        lsig_state = LogSignatureState(level=level)
        block_returns = returns[i-window:i]
        
        # To get the Area (QV) we need the Lead-Lag embedded path increments
        # Lead-lag increments are (dx_lead, dx_lag).
        # We process them sequentially: first (dx, 0), then (0, dx)
        dt_fake = 1.0/window
        for dx in block_returns:
            # Step 1: Lead moves
            lsig_state.extend(dt_fake, dx) # using dt element as the lead component and x as lag
            # Step 2: Lag moves 
            # (Wait, standard 2D log signature of (X_lead, X_lag) allows Area to just be computed natively)
            pass
            
        # Actually, since Lead-Lag area is exactly 1/2 sum(dx^2), we can just pull it natively.
        # But to genuinely test the log-signature BCH:
        lsig_state_ll = LogSignatureState(level=level)
        for dx in block_returns:
            # Lead move (dx, 0)
            lsig_state_ll.extend(dx, 0.0) 
            # Lag move (0, dx)
            lsig_state_ll.extend(0.0, dx)
            
        qv_sig = lsig_state_ll.get_levy_area() * 2.0  # Area = QV/2
        sig_vols[i:min(i+step, n)] = qv_sig
        
    return sig_vols

def cumulative_logsig_kalman_filter(returns, bv_targets, window, dt):
    """
    Uses the Nystrom Koopman Kalman Filter.
    We feed the cumulative exact path history into a RandomProjectionNystrom extractor,
    which maps the Log-Signature to an RBF Landmark space. 
    A simple RLS filter learns the drift in this non-linear feature space.
    """
    # 1. Feature Extractor
    n_landmarks = 100
    extractor = RandomProjectionNystrom(
        dim=2, 
        depth=2, 
        projection_dim=200, 
        n_landmarks=n_landmarks,
        use_leadlag=True,  # Essential for capturing Quadratic Variation perfectly in the features
        kernel_bandwidth=1.0,
        feature_mode='joint'
    )
    
    # RLS State
    nystrom_dim = n_landmarks
    A = np.zeros(nystrom_dim) # We just need to learn the output projection to Volatility
    P = np.eye(nystrom_dim) * 1.0  # Lower initial covariance so we don't overfit to noise
    
    # Forget factor scaled exactly to the theoretical Spot Volatility rolling block
    ff = 1.0 - (1.0/window)  
    
    n = len(returns)
    pred_vols = np.full(n, np.nan)
    
    # We step through, maintaining the cumulative path. Since we use A&J block comparisons,
    # we'll use a sliding bounded path of size 'window' rather than full history to be strictly fair to A&J
    # and to keep extraction O(1) in memory.
    
    # Aït-Sahalia & Jacod Chapter 7: Optimal pre-averaging block size scales with sqrt(N)
    k_n = max(1, int(np.sqrt(window)))
    
    # We update the Kalman observation every k_n steps, which optimally separates continuous path from tick noise
    step = k_n
    
    # For fair benchmarking against A&J RV, the baseline signature filter needs an identical sliding window boundary
    path_buffer = np.zeros((window, 2))
    times = np.arange(window) * dt
    path_buffer[:, 0] = times
    
    for i in range(1, n):
        # Update sliding window
        # We integrate the returns to get the local price path
        if i < window:
            path_buffer[i, 1] = path_buffer[i-1, 1] + returns[i]
        else:
            # Shift and append
            path_buffer[:-1, 1] = path_buffer[1:, 1]
            path_buffer[-1, 1] = path_buffer[-2, 1] + returns[i]
        
        # Every 'step', observe Volatility and update Filter
        if i >= window and i % step == 0:
            # A&J Pre-averaging trick: We smooth the local Bipower Variation proxy over the optimal block
            # length k_n before passing to the Kalman filter.
            smoothed_bv = np.mean(bv_targets[i-k_n:i])
            target_std = np.sqrt(max(smoothed_bv / (window * dt), 1e-10))
            
            # Extract Nystrom features of the local geometric path
            # We seed it with a few landmarks initially
            if len(extractor.landmarks) < n_landmarks:
                extractor.update(path_buffer)
                pred_vols[i:min(i+step, n)] = target_std**2
                continue
                
            phi = extractor.nystrom_embedding(path_buffer)
            
            # RLS Update
            pred = A @ phi
            error = target_std - pred
            
            Pz = P @ phi
            gain = Pz / (ff + phi @ Pz)
            A = A + error * gain
            P = (P - np.outer(gain, Pz)) / ff
            
            # Record Prediction (squaring back to variance)
            pred_vols[i:min(i+step, n)] = (pred)**2
            
            # Periodically update landmarks with new geometry
            if np.random.rand() < 0.05:
                extractor.update(path_buffer)
            
    return pred_vols

def run_benchmark():
    np.random.seed(42)
    # Simulate High Frequency Data
    N = 100000
    
    # Enable Microstructure Noise to demonstrate the blind-spot in A&J Box filters
    # sigma_noise = 0.001 represents a 10 basis point bid-ask bounce/flicker
    sigma_noise = 0.005 
    
    S_true, V, dt = simulate_heston(T=1.0, N=N)
    
    # The actual true efficient price the market maker wants to track
    log_prices_true = np.log(S_true)
    
    # The corrupted Realized price with Bid-Ask bounce/ticks 
    noise = np.random.normal(0, sigma_noise, len(log_prices_true))
    log_prices_observed = log_prices_true + noise
    
    # We operate natively on the returns
    returns = np.diff(log_prices_observed)
    returns_true = np.diff(log_prices_true)
    
    # We want to estimate spot volatility.
    # In Ait-Sahalia & Jacod, spot volatility at t is estimated via RV over a local block k_n \Delta_n
    # Let block size be 1000 steps
    window = 1000
    
    print("Computing Econometric Estimators...")
    t0 = time.time()
    rv = realized_variance(returns, window)
    bv = bipower_variation(returns, window)
    # Spot volatility is RV / window duration in time
    spot_vol_rv = rv / (window * dt)
    spot_vol_bv = bv / (window * dt)
    print(f"Done in {time.time() - t0:.2f} s")
    
    print("Computing Signature Estimators...")
    t0 = time.time()
    sig_qv = signature_volatility_estimator(returns, window)
    spot_vol_sig = sig_qv / (window * dt)
    print(f"Done in {time.time() - t0:.2f} s")
    
    spot_vol_sig_rbf = np.full(N, np.nan)
    eval_size = 10000
    if SignatureVolatilityEstimator is not None:
        print("Computing RBF Signature Estimator (from src.finance)...")
        t0 = time.time()
        # Limit data size for RBF Kernel fitting to prevent memory scaling issues
        train_size = min(N // 2, 5000)
        train_returns = returns[:train_size]
        # Use BV as the target for training (Robust to Microstructure Noise)
        train_qv = bv[:train_size]

        estimator = SignatureVolatilityEstimator(window=window, gamma=0.5, alpha=0.1)

        # Train to predict spot variance (BV) - NOTE: TARGET IS BV, NOT TRUE VOL (V_true)
        # This ensures we aren't leaking the unobservable true state into the model
        sigs_train = estimator._extract_signatures(train_returns)
        targets_train = train_qv[window:window + len(sigs_train)]
        min_len = min(len(sigs_train), len(targets_train))

        from sklearn.kernel_ridge import KernelRidge
        estimator.train_sigs = sigs_train[:min_len]
        estimator.model = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.5)
        estimator.model.fit(sigs_train[:min_len], targets_train[:min_len])

        test_returns = returns[:eval_size]
        sigs_test = estimator._extract_signatures(test_returns)
        pred_vol = estimator.model.predict(sigs_test)
        spot_vol_sig_rbf[window:window+len(pred_vol)] = pred_vol
        print(f"Done in {time.time() - t0:.2f} s")
    else:
        print("Skipping RBF Signature Estimator: src/finance/signature_volatility.py is not present.")
    
    print("Computing Log-Signature (BCH) Estimators...")
    t0 = time.time()
    lsig_qv = logsignature_volatility_estimator(returns, window)
    spot_vol_lsig = lsig_qv / (window * dt)
    print(f"Done in {time.time() - t0:.2f} s")
    
    print("Computing Cumulative LogSig Kalman Filter...")
    t0 = time.time()
    spot_vol_c_lsig = cumulative_logsig_kalman_filter(returns, bv, window, dt)
    print(f"Done in {time.time() - t0:.2f} s")
    
    # Align latent target with the estimator horizon.
    # RV/BV/signature windowed estimators recover average variance over the local
    # block, so comparing against the causal rolling mean of V is the fair metric.
    V_true = V[1:]
    V_target = rolling_mean_causal(V_true, window)
    
    # Discard first 'window' steps for metrics
    valid_idx = slice(window, N)
    
    mse_rv = np.mean((spot_vol_rv[valid_idx] - V_target[valid_idx])**2)
    mse_bv = np.mean((spot_vol_bv[valid_idx] - V_target[valid_idx])**2)
    mse_sig = np.mean((spot_vol_sig[valid_idx] - V_target[valid_idx])**2)
    mse_lsig = np.mean((spot_vol_lsig[valid_idx] - V_target[valid_idx])**2)
    
    # For Cumulative Kalman Filter, compute MSE on valid evaluated segment
    valid_c_idx = ~np.isnan(spot_vol_c_lsig) & (np.arange(N) >= window) & (np.arange(N) < eval_size)
    mse_c_lsig = np.mean((spot_vol_c_lsig[valid_c_idx] - V_target[valid_c_idx])**2) if np.sum(valid_c_idx) > 0 else np.nan
    
    # For RBF, only compute MSE where valid
    valid_rbf_idx = slice(window, eval_size)
    mse_sig_rbf = (
        np.mean((spot_vol_sig_rbf[valid_rbf_idx] - V_target[valid_rbf_idx])**2)
        if SignatureVolatilityEstimator is not None
        else np.nan
    )
    
    print("-" * 30)
    print("MSE of Window-Matched Variance Estimation:")
    print(f"Realized Variance (RV): {mse_rv:.6e}")
    print(f"Bipower Variation (BV): {mse_bv:.6e}")
    # Note: area of Lead-Lag perfectly matches QV, so MSE_sig should equal MSE_rv theoretically
    print(f"Lead-Lag Signature  : {mse_sig:.6e}")
    print(f"Log-Sig (BCH)       : {mse_lsig:.6e}")
    print(f"Cumulative Log-Sig Kalman: {mse_c_lsig:.6e}")
    if SignatureVolatilityEstimator is not None:
        print(f"RBF SigKKF Estimator: {mse_sig_rbf:.6e}")
    print("-" * 30)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    time_axes = np.linspace(0, 1.0, N)
    eval_slice = slice(0, eval_size)
    plt.plot(time_axes[eval_slice], V_target[eval_slice], label='True Window-Averaged Variance', alpha=0.5, color='gray')
    plt.plot(time_axes[eval_slice], spot_vol_rv[eval_slice], label=f'A&J Local RV', color='blue', linestyle='--')
    plt.plot(time_axes[eval_slice], spot_vol_bv[eval_slice], label=f'A&J Local BV', color='green', linestyle=':')
    plt.plot(time_axes[eval_slice], spot_vol_lsig[eval_slice], label=f'Log-Signature (BCH)', color='orange', linestyle='-.')
    t_eval = time_axes[window:eval_size]
    plt.plot(t_eval, spot_vol_c_lsig[window:eval_size], label='Cumulative Nystrom Kalman Filter', alpha=0.9, linestyle='--', color='magenta', linewidth=2)
    
    plt.title('Heston Spot Volatility: Causal Estimation & Cumulative Log-Signatures')
    plt.xlabel('Time')
    plt.ylabel('Variance $V_t$')
    plt.legend()
    plt.savefig('heston_volatility_benchmark.png')
    plt.close()

if __name__ == "__main__":
    run_benchmark()
