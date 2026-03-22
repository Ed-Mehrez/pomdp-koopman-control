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
RKHS_KRONIC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../rkhs_kronic/src'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RKHS_KRONIC_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# 1. Heston Model Simulation
# ============================================================================

def simulate_heston(n_steps=1000, dt=1/252, S0=100, v0=0.04,
                    mu=0.08, r=0.02, kappa=0.3, theta=0.04, xi=0.6, rho=-0.7,
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
# 2. Volatility Estimation using Cumulative Signatures
# ============================================================================

class CumulativeSignatureState:
    """
    Maintains cumulative signature state using Chen's identity.

    S(path_1 ⊕ path_2) = S(path_1) ⊗ S(path_2)

    This allows O(1) updates regardless of path length.
    """

    def __init__(self, level=2):
        self.level = level
        self.S1 = np.zeros(2)
        self.S2 = np.zeros((2, 2))

    def update(self, dt, dx):
        """Update signature with increment (dt, dx) using Chen's identity."""
        inc_S1 = np.array([dt, dx])
        inc_S2 = np.outer(inc_S1, inc_S1) / 2.0
        self.S2 = self.S2 + inc_S2 + np.outer(self.S1, inc_S1)
        self.S1 = self.S1 + inc_S1

    def get_levy_area(self):
        return (self.S2[0, 1] - self.S2[1, 0]) / 2.0

    def get_feature_vector(self):
        return np.array([
            self.S1[0], self.S1[1],
            self.S2[0, 0], self.S2[0, 1], self.S2[1, 0], self.S2[1, 1],
            self.get_levy_area(),
        ])


def compute_cumulative_signature_features(returns, dt=1/252):
    """
    Compute signature DERIVATIVES (local features from cumulative signature).

    Key insight: Use dS/dt to capture LOCAL dynamics while using GLOBAL path info.
    """
    n = len(returns)
    features = []
    sig = CumulativeSignatureState(level=2)
    prev_features = None
    window = 20
    recent_sq_returns = []

    for t in range(n):
        sig.update(dt, returns[t])
        recent_sq_returns.append(returns[t]**2)
        if len(recent_sq_returns) > window:
            recent_sq_returns.pop(0)

        current_features = sig.get_feature_vector()

        if prev_features is not None and t >= window:
            dsig = current_features - prev_features
            local_qv = np.mean(recent_sq_returns) * 252
            local_std = np.std(recent_sq_returns) if len(recent_sq_returns) > 1 else 0

            feat = np.array([
                1.0,
                current_features[6],  # Lévy area
                current_features[5],  # cumulative QV
                dsig[1],  # d(return)/dt
                dsig[5],  # d(QV)/dt
                dsig[6],  # d(Lévy area)/dt
                local_qv,
                returns[t]**2 * 252,
                local_std,
                dsig[5] * np.sign(dsig[6]),
            ])
        else:
            feat = np.zeros(10)

        features.append(feat)
        prev_features = current_features.copy()

    return np.array(features)


def _estimate_volatility_simple(returns, n_prices, train_frac=0.3):
    """Simple rolling volatility estimator."""
    n = len(returns)
    window = 20
    v_hat = np.zeros(n)
    for t in range(n):
        if t < window:
            v_hat[t] = np.mean(returns[:t+1]**2) * 252 if t > 0 else 0.04
        else:
            v_hat[t] = np.mean(returns[t-window:t]**2) * 252
    pad_length = n_prices - n
    v_hat_full = np.concatenate([np.full(pad_length, v_hat[0]), v_hat])
    return np.maximum(v_hat_full, 1e-4)


def estimate_volatility_sigkkf(returns, n_prices, train_frac=0.3, use_kernel=False):
    """
    Estimate volatility using cumulative signature derivatives.

    Uses Chen's identity for O(1) signature updates, then extracts
    signature DERIVATIVES (dS/dt) as local features for regression.
    """
    n = len(returns)
    train_end = int(n * train_frac)
    window = 20

    # Compute cumulative signature features
    features = compute_cumulative_signature_features(returns)

    # Target: realized variance
    rv_window = 10
    realized_var = np.zeros(n)
    for t in range(rv_window, n):
        realized_var[t] = np.mean(returns[t-rv_window:t]**2) * 252

    # Train
    X_train = features[window:train_end]
    y_train = realized_var[window:train_end]

    valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    from sklearn.linear_model import RidgeCV
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    model.fit(X_train_scaled, y_train)

    # Predict
    X_all = features[window:]
    X_all_scaled = scaler.transform(X_all)
    v_hat = model.predict(X_all_scaled)

    # Pad
    pad_length = n_prices - len(v_hat)
    v_hat_full = np.concatenate([np.full(pad_length, max(v_hat[0], 0.01)), v_hat])
    return np.maximum(v_hat_full, 1e-4)


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

    # CRRA utility value (for aggregation across trials)
    # U(W) = W^(1-γ)/(1-γ) for γ ≠ 1, or log(W) for γ = 1
    if gamma == 1:
        utility = np.log(terminal)
    else:
        utility = (terminal ** (1 - gamma)) / (1 - gamma)

    # Max drawdown
    peak = np.maximum.accumulate(W)
    drawdown = (peak - W) / peak
    max_dd = np.max(drawdown)

    return {
        'terminal_wealth': terminal,
        'sharpe': sharpe,
        'utility': utility,  # CRRA utility value
        'max_drawdown': max_dd,
        'volatility': vol
    }


# ============================================================================
# 5. Main Experiment
# ============================================================================

def run_experiment(n_trials=50, n_steps=1000, gamma=2.0, theta=0.04):
    """
    Run multiple trials comparing strategies.

    IMPORTANT: We now use theta (the MODEL parameter) for constant,
    NOT np.mean(v_true) which would be cheating (look-ahead bias).
    """
    results = {'oracle': [], 'sigkkf': [], 'constant': [], 'constant_cheat': []}

    for trial in range(n_trials):
        # Simulate market
        S, v_true, returns = simulate_heston(n_steps=n_steps, seed=trial*100)

        # Volatility estimates
        # FAIR: Use model's long-run mean theta (what you'd estimate from history)
        v_constant = np.full(n_steps, theta)
        # CHEAT: Use in-sample mean (look-ahead bias - for comparison only)
        v_constant_cheat = np.full(n_steps, np.mean(v_true))
        v_sigkkf = estimate_volatility_sigkkf(returns, n_steps, use_kernel=True)

        # Run portfolios
        W_oracle, _ = simulate_portfolio(S, v_true, gamma=gamma)
        W_sigkkf, _ = simulate_portfolio(S, v_sigkkf, gamma=gamma)
        W_constant, _ = simulate_portfolio(S, v_constant, gamma=gamma)
        W_constant_cheat, _ = simulate_portfolio(S, v_constant_cheat, gamma=gamma)

        # Compute metrics
        results['oracle'].append(compute_metrics(W_oracle, gamma))
        results['sigkkf'].append(compute_metrics(W_sigkkf, gamma))
        results['constant'].append(compute_metrics(W_constant, gamma))
        results['constant_cheat'].append(compute_metrics(W_constant_cheat, gamma))

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
        # Compute certainty equivalent from expected utility
        # CE = ((1-γ) * E[U])^(1/(1-γ)) for γ ≠ 1
        # CE = exp(E[log(W)]) for γ = 1
        expected_utility = summary[strategy]['utility']
        if gamma == 1:
            summary[strategy]['certainty_equivalent'] = np.exp(expected_utility)
        else:
            summary[strategy]['certainty_equivalent'] = ((1 - gamma) * expected_utility) ** (1 / (1 - gamma))

    return summary, results


def create_merton_figure():
    """
    Create visualization for the presentation.
    """
    fig = plt.figure(figsize=(15, 10))

    # Single example simulation
    np.random.seed(42)
    n_steps = 1000

    theta = 0.04  # Model's long-run variance (what we'd know in practice)

    S, v_true, returns = simulate_heston(n_steps=n_steps)
    # FAIR: Use theta (known model parameter), not in-sample mean
    v_constant = np.full(n_steps, theta)
    v_sigkkf = estimate_volatility_sigkkf(returns, n_steps, use_kernel=True)

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
    summary, all_results = run_experiment(n_trials=50, theta=theta)

    strategies = ['oracle', 'sigkkf', 'constant']
    colors = ['#2c3e50', '#27ae60', '#3498db']
    labels = ['Oracle', 'Sig-KKF', 'Constant (θ)']

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

    # Compute capture percentage safely
    oracle_adv = summary['oracle']['sharpe'] - summary['constant']['sharpe']
    sigkkf_adv = summary['sigkkf']['sharpe'] - summary['constant']['sharpe']
    capture_pct = (sigkkf_adv / oracle_adv * 100) if abs(oracle_adv) > 0.01 else 0

    summary_text = f"""
    MERTON PORTFOLIO RESULTS (50 trials, γ=2.0):

                      Oracle    Sig-KKF   Const(θ)  Const(cheat)
    Terminal Wealth:  {summary['oracle']['terminal_wealth']:.3f}     {summary['sigkkf']['terminal_wealth']:.3f}     {summary['constant']['terminal_wealth']:.3f}      {summary['constant_cheat']['terminal_wealth']:.3f}
    Sharpe Ratio:     {summary['oracle']['sharpe']:.3f}     {summary['sigkkf']['sharpe']:.3f}     {summary['constant']['sharpe']:.3f}      {summary['constant_cheat']['sharpe']:.3f}
    Max Drawdown:     {summary['oracle']['max_drawdown']:.1%}    {summary['sigkkf']['max_drawdown']:.1%}    {summary['constant']['max_drawdown']:.1%}     {summary['constant_cheat']['max_drawdown']:.1%}

    NOTE:
    - Const(θ) uses model's long-run mean (fair)
    - Const(cheat) uses in-sample mean (look-ahead!)

    Sig-KKF captures {capture_pct:.0f}% of Oracle's advantage.
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
