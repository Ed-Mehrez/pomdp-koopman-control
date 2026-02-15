"""
Merton Portfolio Problem with Stochastic Volatility

This experiment tests whether Sig-KKF can help with optimal portfolio
allocation when volatility is stochastic and unobserved.

Setup:
- Risky asset follows Heston model (stochastic volatility)
- Investor has CRRA utility U(W) = W^(1-γ)/(1-γ)
- Classic Merton solution: π* = (μ-r)/(γσ²)
- Challenge: σ_t is hidden!

Comparison:
1. Oracle: Knows true volatility v_t
2. Sig-KKF: Estimates v_t from price path
3. Constant: Uses long-run mean volatility

Metrics:
- Terminal wealth distribution
- Sharpe ratio
- Certainty equivalent
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# 1. Heston Model Simulation
# ============================================================================

def simulate_heston(n_steps=1000, dt=1/252, S0=100, v0=0.04,
                    mu=0.08, r=0.02, kappa=0.5, theta=0.04, xi=0.5, rho=-0.7,
                    seed=42):
    """
    Simulate Heston stochastic volatility model.

    Returns:
        S: price path
        v: variance path (hidden in practice)
        returns: log returns
    """
    np.random.seed(seed)

    S = np.zeros(n_steps)
    v = np.zeros(n_steps)
    S[0] = S0
    v[0] = v0

    for t in range(1, n_steps):
        # Correlated Brownian motions
        z1 = np.random.randn()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()

        # Variance process (ensure non-negative)
        v[t] = v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(max(v[t-1], 0) * dt) * z2
        v[t] = max(v[t], 1e-8)

        # Price process
        S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1)

    returns = np.diff(np.log(S))
    return S, v, returns


# ============================================================================
# 2. Volatility Estimation (Sig-KKF style)
# ============================================================================

def estimate_volatility_sigkkf(returns, n_prices, train_frac=0.3):
    """
    Estimate volatility using signature-based features.
    Uses rolling realized variance as training target (noisy proxy).

    Args:
        returns: log returns (length n_prices - 1)
        n_prices: number of price points (to match output length)
        train_frac: fraction of data for training
    """
    n = len(returns)
    train_end = int(n * train_frac)

    # Features: local path statistics
    window = 20
    features = []

    for t in range(n):
        if t < window:
            feat = [0, 0, 0, 0, 0]
        else:
            recent = returns[t-window:t]
            feat = [
                1.0,  # bias
                np.mean(recent),  # recent drift
                np.std(recent),  # recent vol
                np.sum(recent**2),  # realized variance
                np.mean(recent**2) - np.mean(recent)**2,  # variance of returns
            ]
        features.append(feat)

    features = np.array(features)

    # Target: realized variance (noisy proxy for true variance)
    rv_window = 10
    realized_var = np.zeros(n)
    for t in range(rv_window, n):
        realized_var[t] = np.mean(returns[t-rv_window:t]**2) * 252  # annualized

    # Train on first portion
    X_train = features[window:train_end]
    y_train = realized_var[window:train_end]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # Predict
    X_all = features[window:]
    X_all_scaled = scaler.transform(X_all)
    v_hat = model.predict(X_all_scaled)

    # Pad with initial values to match n_prices
    # returns has length n_prices - 1, X_all has length n - window = n_prices - 1 - window
    # So v_hat has length n_prices - 1 - window
    # We need to pad to get n_prices elements
    pad_length = n_prices - len(v_hat)
    v_hat_full = np.concatenate([np.full(pad_length, v_hat[0]), v_hat])

    # Ensure positive
    v_hat_full = np.maximum(v_hat_full, 1e-4)

    return v_hat_full


# ============================================================================
# 3. Portfolio Strategies
# ============================================================================

def merton_allocation(mu, r, gamma, v):
    """
    Classic Merton allocation: π* = (μ-r)/(γσ²)
    """
    return (mu - r) / (gamma * v)


def simulate_portfolio(S, v_used, mu=0.08, r=0.02, gamma=2.0,
                       rebalance_freq=1, max_leverage=2.0):
    """
    Simulate portfolio with given volatility estimates.

    Args:
        S: price path
        v_used: volatility estimates to use for allocation
        rebalance_freq: how often to rebalance (1 = daily)
        max_leverage: maximum leverage allowed

    Returns:
        W: wealth path
        allocations: portfolio weights over time
    """
    n = len(S)
    dt = 1/252

    W = np.zeros(n)
    W[0] = 1.0  # Start with $1

    allocations = np.zeros(n)

    for t in range(1, n):
        # Compute allocation based on estimated volatility
        if t % rebalance_freq == 0:
            pi = merton_allocation(mu, r, gamma, v_used[t-1])
            # Clip to avoid extreme leverage
            pi = np.clip(pi, -max_leverage, max_leverage)
            allocations[t] = pi
        else:
            allocations[t] = allocations[t-1]

        # Portfolio return
        stock_return = S[t] / S[t-1] - 1
        bond_return = r * dt

        portfolio_return = allocations[t] * stock_return + (1 - allocations[t]) * bond_return
        W[t] = W[t-1] * (1 + portfolio_return)

    return W, allocations


# ============================================================================
# 4. Evaluation Metrics
# ============================================================================

def compute_metrics(W, gamma=2.0, rf_rate=0.02, dt=1/252):
    """
    Compute portfolio performance metrics.
    """
    returns = np.diff(W) / W[:-1]

    # Terminal wealth
    terminal = W[-1]

    # Annualized return and volatility
    n_periods = len(returns)
    total_return = W[-1] / W[0] - 1
    years = n_periods * dt
    annual_return = (1 + total_return)**(1/years) - 1
    vol = np.std(returns) * np.sqrt(252)  # Annualized volatility

    # Sharpe ratio (annualized)
    sharpe = (annual_return - rf_rate) / (vol + 1e-9)

    # Certainty equivalent (for CRRA utility)
    # CE is the certain wealth that gives same utility as terminal wealth
    if gamma == 1:
        ce = terminal  # Log utility
    else:
        # For a single terminal wealth, CE = terminal wealth
        ce = terminal

    # Max drawdown
    peak = np.maximum.accumulate(W)
    drawdown = (peak - W) / peak
    max_dd = np.max(drawdown)

    return {
        'terminal_wealth': terminal,
        'sharpe': sharpe,
        'certainty_equivalent': ce,
        'max_drawdown': max_dd,
        'volatility': vol
    }


# ============================================================================
# 5. Main Experiment
# ============================================================================

def run_experiment(n_trials=50, n_steps=1000, gamma=2.0):
    """
    Run multiple trials comparing strategies.
    """
    results = {'oracle': [], 'sigkkf': [], 'constant': []}

    for trial in range(n_trials):
        # Simulate market
        S, v_true, returns = simulate_heston(n_steps=n_steps, seed=trial*100)

        # Volatility estimates
        v_constant = np.full(n_steps, np.mean(v_true))  # Long-run mean
        v_sigkkf = estimate_volatility_sigkkf(returns, n_steps)

        # Run portfolios
        W_oracle, _ = simulate_portfolio(S, v_true, gamma=gamma)
        W_sigkkf, _ = simulate_portfolio(S, v_sigkkf, gamma=gamma)
        W_constant, _ = simulate_portfolio(S, v_constant, gamma=gamma)

        # Compute metrics
        results['oracle'].append(compute_metrics(W_oracle, gamma))
        results['sigkkf'].append(compute_metrics(W_sigkkf, gamma))
        results['constant'].append(compute_metrics(W_constant, gamma))

    # Aggregate
    summary = {}
    for strategy in results:
        summary[strategy] = {
            metric: np.mean([r[metric] for r in results[strategy]])
            for metric in results[strategy][0].keys()
        }
        summary[strategy]['std'] = {
            metric: np.std([r[metric] for r in results[strategy]])
            for metric in results[strategy][0].keys()
        }

    return summary, results


def create_merton_figure():
    """
    Create visualization for the presentation.
    """
    fig = plt.figure(figsize=(15, 10))

    # Single example simulation
    np.random.seed(42)
    n_steps = 1000

    S, v_true, returns = simulate_heston(n_steps=n_steps)
    v_constant = np.full(n_steps, np.mean(v_true))
    v_sigkkf = estimate_volatility_sigkkf(returns, n_steps)

    t = np.arange(n_steps) / 252  # Years

    # Run portfolios
    gamma = 2.0
    W_oracle, alloc_oracle = simulate_portfolio(S, v_true, gamma=gamma)
    W_sigkkf, alloc_sigkkf = simulate_portfolio(S, v_sigkkf, gamma=gamma)
    W_constant, alloc_constant = simulate_portfolio(S, v_constant, gamma=gamma)

    # =========================================================================
    # Panel 1: Price and True Volatility
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    ax1_twin = ax1.twinx()

    ax1.plot(t, S, color='#2c3e50', linewidth=1.5, label='Price')
    ax1_twin.plot(t, np.sqrt(v_true), color='#e74c3c', linewidth=1.5, alpha=0.7, label='Vol')

    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Price', color='#2c3e50')
    ax1_twin.set_ylabel('Volatility', color='#e74c3c')
    ax1.set_title('Heston Model Simulation', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: Volatility Estimation
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, np.sqrt(v_true), 'k-', linewidth=2, label='True Vol', alpha=0.7)
    ax2.plot(t, np.sqrt(v_sigkkf), color='#27ae60', linewidth=1.5, label='Sig-KKF Est')
    ax2.axhline(np.sqrt(np.mean(v_true)), color='#3498db', linestyle='--', label='Constant')

    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Volatility')
    ax2.set_title('Volatility Estimation', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Portfolio Allocations
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, alloc_oracle, 'k-', linewidth=1.5, label='Oracle', alpha=0.7)
    ax3.plot(t, alloc_sigkkf, color='#27ae60', linewidth=1.5, label='Sig-KKF')
    ax3.plot(t, alloc_constant, color='#3498db', linewidth=1.5, linestyle='--', label='Constant')

    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Stock Allocation (π)')
    ax3.set_title('Portfolio Allocations\n(Merton formula)', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 2.5)

    # =========================================================================
    # Panel 4: Wealth Paths
    # =========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, W_oracle, 'k-', linewidth=2, label='Oracle')
    ax4.plot(t, W_sigkkf, color='#27ae60', linewidth=2, label='Sig-KKF')
    ax4.plot(t, W_constant, color='#3498db', linewidth=2, linestyle='--', label='Constant')

    ax4.set_xlabel('Time (years)')
    ax4.set_ylabel('Wealth')
    ax4.set_title('Portfolio Wealth', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 5: Multi-trial Results
    # =========================================================================
    ax5 = fig.add_subplot(2, 3, 5)

    print("Running multi-trial experiment...")
    summary, all_results = run_experiment(n_trials=50)

    strategies = ['oracle', 'sigkkf', 'constant']
    colors = ['#2c3e50', '#27ae60', '#3498db']
    labels = ['Oracle', 'Sig-KKF', 'Constant']

    # Terminal wealth distribution
    for i, (strat, color, label) in enumerate(zip(strategies, colors, labels)):
        terminals = [r['terminal_wealth'] for r in all_results[strat]]
        ax5.hist(terminals, bins=20, alpha=0.5, color=color, label=label, density=True)

    ax5.set_xlabel('Terminal Wealth')
    ax5.set_ylabel('Density')
    ax5.set_title('Terminal Wealth Distribution\n(50 trials)', fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Summary Statistics
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    MERTON PORTFOLIO RESULTS (50 trials, γ=2.0):

                      Oracle    Sig-KKF   Constant
    Terminal Wealth:  {summary['oracle']['terminal_wealth']:.3f}     {summary['sigkkf']['terminal_wealth']:.3f}     {summary['constant']['terminal_wealth']:.3f}
    Sharpe Ratio:     {summary['oracle']['sharpe']:.3f}     {summary['sigkkf']['sharpe']:.3f}     {summary['constant']['sharpe']:.3f}
    Max Drawdown:     {summary['oracle']['max_drawdown']:.1%}    {summary['sigkkf']['max_drawdown']:.1%}    {summary['constant']['max_drawdown']:.1%}

    INTERPRETATION:
    - Oracle: Upper bound (knows true vol)
    - Sig-KKF: Adapts to estimated vol
    - Constant: Ignores vol changes

    Sig-KKF captures {(summary['sigkkf']['sharpe'] - summary['constant']['sharpe']) / (summary['oracle']['sharpe'] - summary['constant']['sharpe']) * 100:.0f}% of the
    Oracle's advantage over Constant.
    """

    ax6.text(0.05, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=10, family='monospace', va='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

    plt.suptitle('Merton Portfolio with Hidden Stochastic Volatility',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    plt.savefig(os.path.join(SCRIPT_DIR, '../docs/merton_portfolio_comparison.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Saved merton_portfolio_comparison.png")
    print(f"\nSig-KKF captures {(summary['sigkkf']['sharpe'] - summary['constant']['sharpe']) / (summary['oracle']['sharpe'] - summary['constant']['sharpe']) * 100:.0f}% of Oracle's advantage")

    return summary


if __name__ == '__main__':
    print("="*60)
    print("Merton Portfolio with Stochastic Volatility Experiment")
    print("="*60)

    summary = create_merton_figure()

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    for strategy in ['oracle', 'sigkkf', 'constant']:
        print(f"\n{strategy.upper()}:")
        for metric, value in summary[strategy].items():
            if metric != 'std':
                print(f"  {metric}: {value:.4f}")
