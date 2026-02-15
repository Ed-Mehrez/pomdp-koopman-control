"""
LQR-Style Derivatives Hedging with Hidden Stochastic Volatility

When volatility is stochastic and unobserved, delta hedging becomes a
partial observation control problem:
- State: (S_t, V_t) where V is hidden
- Control: hedge ratio δ_t
- Objective: minimize hedging error variance

Standard delta-gamma hedging assumes known volatility.
With hidden vol, we need:
1. State estimation (Sig-KKF)
2. Optimal feedback control (LQR-like)

This experiment shows:
- Option hedging under Heston model
- Standard delta-hedging with wrong vol
- Sig-KKF enhanced hedging
- Vega-adjusted hedging with vol estimates

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# 1. Black-Scholes Greeks
# ============================================================================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """Black-Scholes delta."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (dC/d_sigma)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


# ============================================================================
# 2. Heston Model Simulation
# ============================================================================

def simulate_heston(n_steps=252, dt=1/252, S0=100, v0=0.04,
                    mu=0.05, r=0.02, kappa=0.3, theta=0.04, xi=0.6, rho=-0.7,
                    seed=42):
    """Simulate Heston model for option hedging."""
    np.random.seed(seed)

    S = np.zeros(n_steps)
    v = np.zeros(n_steps)
    S[0] = S0
    v[0] = v0

    for t in range(1, n_steps):
        z1 = np.random.randn()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()

        v[t] = v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(max(v[t-1], 0) * dt) * z2
        v[t] = max(v[t], 1e-8)

        S[t] = S[t-1] * np.exp((r - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1)

    returns = np.diff(np.log(S))
    return S, v, returns


# ============================================================================
# 3. Volatility Estimation
# ============================================================================

def estimate_volatility_sigkkf(returns, n_prices, train_frac=0.3):
    """Estimate volatility using signature-style features."""
    n = len(returns)
    train_end = int(n * train_frac)

    window = 10
    features = []

    for t in range(n):
        if t < window:
            feat = [0, 0, 0, 0, 0]
        else:
            recent = returns[t-window:t]
            feat = [
                1.0,
                np.mean(recent),
                np.std(recent),
                np.sum(recent**2),
                np.mean(recent**2) - np.mean(recent)**2,
            ]
        features.append(feat)

    features = np.array(features)

    rv_window = 5
    realized_var = np.zeros(n)
    for t in range(rv_window, n):
        realized_var[t] = np.mean(returns[t-rv_window:t]**2) * 252

    X_train = features[window:train_end]
    y_train = realized_var[window:train_end]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=0.5)
    model.fit(X_train_scaled, y_train)

    X_all = features[window:]
    X_all_scaled = scaler.transform(X_all)
    v_hat = model.predict(X_all_scaled)

    pad_length = n_prices - len(v_hat)
    v_hat_full = np.concatenate([np.full(pad_length, v_hat[0]), v_hat])
    v_hat_full = np.maximum(v_hat_full, 1e-4)

    return v_hat_full


# ============================================================================
# 4. Hedging Strategies
# ============================================================================

def hedge_portfolio(S, v_used, v_true, K=100, T_initial=1.0, r=0.02, strategy='delta'):
    """
    Simulate hedging a short call position.

    KEY: The market prices the option using TRUE vol, but we hedge using
    our ESTIMATED vol. This creates hedging error when estimates are wrong.

    Args:
        S: price path
        v_used: volatility estimates for computing delta (what we use)
        v_true: true volatility for pricing (what market uses)
        K: strike
        T_initial: initial time to maturity
        r: risk-free rate
        strategy: 'delta' (standard) or 'delta_vega' (vol-adjusted)

    Returns:
        pnl: hedging P&L at each step
        hedge_error: final hedging error
    """
    n = len(S)
    dt = 1/252

    # Initial option value - market prices with TRUE vol
    sigma_true_init = np.sqrt(v_true[0])
    C0 = bs_call_price(S[0], K, T_initial, r, sigma_true_init)

    # Track hedge portfolio value
    cash = C0  # Premium received
    shares = 0.0

    pnl = np.zeros(n)
    deltas = np.zeros(n)

    for t in range(n):
        T_remaining = T_initial - t * dt

        if T_remaining <= 0:
            break

        sigma_used = np.sqrt(v_used[t])  # Our estimate for hedging
        sigma_true = np.sqrt(v_true[t])  # Market's pricing vol

        # Compute hedge ratio using OUR estimate
        delta = bs_delta(S[t], K, T_remaining, r, sigma_used)

        if strategy == 'delta_vega':
            # Adjust delta based on vol forecast change
            if t > 0:
                vol_change = sigma_used - np.sqrt(v_used[t-1])
                vega = bs_vega(S[t], K, T_remaining, r, sigma_used)
                # Adjust for expected vol mean reversion
                delta_adj = -vega * vol_change / (S[t] * sigma_used + 1e-9) * 0.1
                delta = np.clip(delta + delta_adj, 0, 1)

        deltas[t] = delta

        # Rebalance
        shares_needed = delta
        shares_change = shares_needed - shares
        cash -= shares_change * S[t]  # Buy/sell shares
        shares = shares_needed

        # Portfolio value: cash + shares * S - option liability
        # Market prices option with TRUE vol
        C_t = bs_call_price(S[t], K, T_remaining, r, sigma_true)
        portfolio_value = cash + shares * S[t] - C_t

        pnl[t] = portfolio_value

    # Final P&L at expiry
    final_payoff = max(S[-1] - K, 0)
    final_hedge_value = cash + shares * S[-1]
    hedge_error = final_hedge_value - final_payoff

    return pnl, hedge_error, deltas


# ============================================================================
# 5. Metrics
# ============================================================================

def compute_hedge_metrics(hedge_errors):
    """Compute hedging performance metrics."""
    return {
        'mean_error': np.mean(hedge_errors),
        'std_error': np.std(hedge_errors),
        'rmse': np.sqrt(np.mean(hedge_errors**2)),
        'mae': np.mean(np.abs(hedge_errors)),
        'max_error': np.max(np.abs(hedge_errors)),
    }


# ============================================================================
# 6. Main Experiment
# ============================================================================

def run_experiment(n_trials=100, n_steps=252):
    """Run hedging comparison across multiple trials."""

    strategies = {
        'oracle': {'vol': 'true', 'strategy': 'delta'},
        'sigkkf': {'vol': 'sigkkf', 'strategy': 'delta'},
        'constant': {'vol': 'constant', 'strategy': 'delta'},
        'sigkkf_vega': {'vol': 'sigkkf', 'strategy': 'delta_vega'},
    }

    hedge_errors = {k: [] for k in strategies}

    for trial in range(n_trials):
        S, v_true, returns = simulate_heston(n_steps=n_steps, seed=trial*100)

        v_estimates = {
            'true': v_true,
            'sigkkf': estimate_volatility_sigkkf(returns, n_steps),
            'constant': np.full(n_steps, np.mean(v_true)),
        }

        for name, config in strategies.items():
            v_used = v_estimates[config['vol']]
            _, error, _ = hedge_portfolio(S, v_used, v_true, strategy=config['strategy'])
            hedge_errors[name].append(error)

    # Compute metrics
    metrics = {}
    for name in strategies:
        metrics[name] = compute_hedge_metrics(np.array(hedge_errors[name]))

    return metrics, hedge_errors


def create_figure():
    """Create visualization for LQR-style hedging."""
    fig = plt.figure(figsize=(15, 10))

    # Single example
    np.random.seed(42)
    n_steps = 252  # 1 year
    K = 100
    T = 1.0

    S, v_true, returns = simulate_heston(n_steps=n_steps)
    v_sigkkf = estimate_volatility_sigkkf(returns, n_steps)
    v_constant = np.full(n_steps, np.mean(v_true))

    t = np.arange(n_steps) / 252

    # Run hedging strategies
    pnl_oracle, err_oracle, delta_oracle = hedge_portfolio(S, v_true, v_true, K=K)
    pnl_sigkkf, err_sigkkf, delta_sigkkf = hedge_portfolio(S, v_sigkkf, v_true, K=K)
    pnl_const, err_const, delta_const = hedge_portfolio(S, v_constant, v_true, K=K)

    # =========================================================================
    # Panel 1: Problem Setup
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')

    setup_text = """
    LQR-STYLE DERIVATIVES HEDGING

    Problem:
    - Short a call option
    - Hedge with underlying stock
    - Volatility is STOCHASTIC & HIDDEN

    Standard approach:
        δ = ∂C/∂S (Black-Scholes delta)

    Challenge:
        δ depends on σ, which is unknown!

    Methods:
    1. Oracle: Use true σ_t
    2. Sig-KKF: Estimate σ_t
    3. Constant: Use historical mean

    Metric: Hedging error at expiry
    """
    ax1.text(0.05, 0.5, setup_text, transform=ax1.transAxes,
             fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

    # =========================================================================
    # Panel 2: Price and Volatility
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    ax2_twin = ax2.twinx()

    ax2.plot(t, S, color='#2c3e50', linewidth=1.5, label='Price')
    ax2.axhline(K, color='gray', linestyle='--', alpha=0.5, label=f'Strike K={K}')
    ax2_twin.plot(t, np.sqrt(v_true)*100, color='#e74c3c', linewidth=1.5, alpha=0.7, label='True Vol')
    ax2_twin.plot(t, np.sqrt(v_sigkkf)*100, color='#27ae60', linewidth=1.5, alpha=0.7, label='Sig-KKF Vol')

    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Price', color='#2c3e50')
    ax2_twin.set_ylabel('Volatility (%)', color='#e74c3c')
    ax2.set_title('Stock Price & Volatility', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Delta Comparison
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, delta_oracle, 'k-', linewidth=2, label='Oracle δ')
    ax3.plot(t, delta_sigkkf, color='#27ae60', linewidth=1.5, label='Sig-KKF δ')
    ax3.plot(t, delta_const, color='#3498db', linewidth=1.5, linestyle='--', label='Constant δ')

    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Delta (hedge ratio)')
    ax3.set_title('Computed Deltas', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)

    # =========================================================================
    # Panel 4: Hedging P&L
    # =========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, pnl_oracle, 'k-', linewidth=2, label=f'Oracle (error={err_oracle:.2f})')
    ax4.plot(t, pnl_sigkkf, color='#27ae60', linewidth=2, label=f'Sig-KKF (error={err_sigkkf:.2f})')
    ax4.plot(t, pnl_const, color='#3498db', linewidth=2, linestyle='--', label=f'Constant (error={err_const:.2f})')

    ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time (years)')
    ax4.set_ylabel('Hedge Portfolio Value')
    ax4.set_title('Hedging P&L Path', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 5: Multi-trial hedge error distribution
    # =========================================================================
    ax5 = fig.add_subplot(2, 3, 5)

    print("Running multi-trial hedging experiment...")
    metrics, all_errors = run_experiment(n_trials=100)

    strategies = ['oracle', 'sigkkf', 'constant']
    colors = ['#2c3e50', '#27ae60', '#3498db']
    labels = ['Oracle', 'Sig-KKF', 'Constant']

    for strat, color, label in zip(strategies, colors, labels):
        errors = all_errors[strat]
        ax5.hist(errors, bins=25, alpha=0.5, color=color, label=label, density=True)

    ax5.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Hedging Error at Expiry')
    ax5.set_ylabel('Density')
    ax5.set_title('Hedging Error Distribution\n(100 trials)', fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Summary Statistics
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Compute improvement
    const_rmse = metrics['constant']['rmse']
    sigkkf_rmse = metrics['sigkkf']['rmse']
    oracle_rmse = metrics['oracle']['rmse']

    improvement = (const_rmse - sigkkf_rmse) / (const_rmse - oracle_rmse) * 100 if const_rmse != oracle_rmse else 0

    summary_text = f"""
    HEDGING PERFORMANCE (100 trials):

                         Oracle    Sig-KKF   Constant
    RMSE:                {metrics['oracle']['rmse']:.3f}     {metrics['sigkkf']['rmse']:.3f}     {metrics['constant']['rmse']:.3f}
    Std Error:           {metrics['oracle']['std_error']:.3f}     {metrics['sigkkf']['std_error']:.3f}     {metrics['constant']['std_error']:.3f}
    Max Error:           {metrics['oracle']['max_error']:.3f}     {metrics['sigkkf']['max_error']:.3f}     {metrics['constant']['max_error']:.3f}

    KEY INSIGHT:
    - Delta depends on unknown σ_t
    - Wrong σ → wrong δ → hedging error
    - Sig-KKF estimates σ from price path
    - Captures {improvement:.0f}% of Oracle's advantage

    This is a POMDP problem:
    - Hidden state: σ_t
    - Observation: price path
    - Control: hedge ratio δ_t
    - Objective: min E[(hedge error)²]
    """

    ax6.text(0.02, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=9, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

    plt.suptitle('Derivatives Hedging as POMDP Control',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(SCRIPT_DIR, '../docs/lqr_hedging.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Saved lqr_hedging.png")
    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print("LQR-Style Derivatives Hedging")
    print("with Hidden Stochastic Volatility")
    print("=" * 60)

    metrics = create_figure()

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    for strategy in ['oracle', 'sigkkf', 'constant']:
        print(f"\n{strategy.upper()}:")
        for metric, value in metrics[strategy].items():
            print(f"  {metric}: {value:.4f}")
