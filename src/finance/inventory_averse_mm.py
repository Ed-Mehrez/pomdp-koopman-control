"""
Inventory-Averse Market Maker

Extension of Kyle model where MM has quadratic inventory costs.

Cost = -γ ∫₀ᵀ I_t² dt

where I_t = -Y_t is the market maker's inventory (negative of cumulative order flow).

Key insight: Path structure matters because early inventory accumulation
has higher cost (longer holding period).

This creates:
1. Time-varying liquidity (tighter spreads early, wider late)
2. Path-dependent pricing (same Y_T but different paths → different prices)
3. Natural role for signatures to capture inventory path

The equilibrium pricing rule becomes:
    P(signature) = E[v | signature] + inventory_adjustment(signature)

where the inventory adjustment depends on the path shape.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SignatureState:
    """Truncated signature with inventory cost tracking."""
    t_cumsum: float = 0.0
    Y_cumsum: float = 0.0
    levy_area: float = 0.0
    # Cumulative inventory cost: ∫ I_t² dt where I_t = -Y_t
    inventory_cost: float = 0.0
    # Time-weighted inventory: ∫ I_t * (T-t) dt (captures "how early" inventory was accumulated)
    time_weighted_inventory: float = 0.0

    def extend(self, dt: float, dY: float, T: float):
        """Extend signature and track inventory metrics."""
        current_t = self.t_cumsum
        current_Y = self.Y_cumsum

        # Lévy area
        self.levy_area += (current_Y * dt - current_t * dY) / 2

        # Inventory cost increment (using current inventory level)
        I_t = -current_Y  # Inventory = negative of order flow
        self.inventory_cost += I_t ** 2 * dt

        # Time-weighted inventory (time remaining * inventory)
        time_remaining = max(T - current_t, 0)
        self.time_weighted_inventory += I_t * time_remaining * dt

        # Update cumulative values
        self.t_cumsum += dt
        self.Y_cumsum += dY

    @property
    def Y_T(self) -> float:
        return self.Y_cumsum

    @property
    def T_elapsed(self) -> float:
        return self.t_cumsum

    def to_vector(self) -> np.ndarray:
        """Return extended signature features."""
        return np.array([
            self.T_elapsed,
            self.Y_T,
            self.levy_area,
            self.inventory_cost,
            self.time_weighted_inventory
        ])

    def copy(self) -> 'SignatureState':
        new = SignatureState()
        new.t_cumsum = self.t_cumsum
        new.Y_cumsum = self.Y_cumsum
        new.levy_area = self.levy_area
        new.inventory_cost = self.inventory_cost
        new.time_weighted_inventory = self.time_weighted_inventory
        return new


class InventoryAverseMM:
    """
    Market maker with quadratic inventory costs.

    The MM's problem (simplified):
        max E[∫(P_t - v)dY_t - γ∫I_t²dt]

    At equilibrium, the MM sets:
        P_t = E[v | F_t] + γ * marginal_inventory_cost

    where the marginal cost depends on current inventory and time remaining.
    """
    def __init__(self, dt: float, P_0: float, T: float,
                 gamma: float = 0.1,  # Inventory cost parameter
                 reg: float = 0.1,
                 max_budget: int = 500):
        self.dt = dt
        self.P_0 = P_0
        self.T = T
        self.gamma = gamma
        self.reg = reg
        self.max_budget = max_budget

        # Current state
        self.sig = SignatureState()
        self.P_t = P_0

        # Training data
        self.support_sigs: List[SignatureState] = []
        self.support_v: List[float] = []
        self.alpha = None
        self.lengthscales = np.array([1.0, 10.0, 5.0, 100.0, 50.0])

    def reset(self):
        """Reset for new episode."""
        self.sig = SignatureState()
        self.P_t = self.P_0

    def _signature_kernel(self, s1: SignatureState, s2: SignatureState) -> float:
        """RBF kernel on extended signature features."""
        x = s1.to_vector()
        y = s2.to_vector()
        diff = (x - y) / self.lengthscales
        return np.exp(-0.5 * np.sum(diff ** 2))

    def _inventory_adjustment(self, sig: SignatureState) -> float:
        """
        Compute inventory-cost-based price adjustment.

        The MM should charge more when:
        - Current inventory is large (higher marginal cost)
        - Time remaining is large (longer exposure)

        Marginal cost of taking dY more order flow:
            d/dY [γ∫(I_t + dY)²dt] ≈ 2γ * I_t * (T - t)
        """
        I_t = -sig.Y_T  # Current inventory
        time_remaining = max(self.T - sig.T_elapsed, self.dt)

        # Marginal inventory cost
        return 2 * self.gamma * I_t * time_remaining

    def predict_v(self, sig: SignatureState) -> float:
        """Predict E[v | signature] using kernel regression."""
        if self.alpha is None or len(self.support_sigs) < 5:
            return self.P_0

        k_vec = np.array([self._signature_kernel(sig, s) for s in self.support_sigs])
        return float(k_vec @ self.alpha)

    def get_price(self, sig: SignatureState) -> float:
        """
        Full pricing rule:
            P = E[v | signature] + inventory_adjustment
        """
        E_v = self.predict_v(sig)
        adjustment = self._inventory_adjustment(sig)
        return E_v + adjustment

    def filter_step(self, dY_t: float) -> float:
        """Process order flow increment."""
        self.sig.extend(self.dt, dY_t, self.T)
        self.P_t = self.get_price(self.sig)
        return self.P_t

    def end_of_episode_update(self, final_v: float):
        """Update with revealed true value."""
        self.support_sigs.append(self.sig.copy())
        self.support_v.append(final_v)

        if len(self.support_sigs) > self.max_budget:
            self.support_sigs.pop(0)
            self.support_v.pop(0)

        self._fit_krr()

    def _fit_krr(self):
        """Fit kernel ridge regression for E[v | signature]."""
        n = len(self.support_sigs)
        if n < 5:
            self.alpha = None
            return

        # Compute adaptive lengthscales
        features = np.array([s.to_vector() for s in self.support_sigs])
        stds = np.std(features, axis=0)
        self.lengthscales = np.maximum(stds * 0.5, [0.1, 1.0, 0.5, 1.0, 1.0])

        # Build kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = K[j, i] = self._signature_kernel(
                    self.support_sigs[i], self.support_sigs[j])

        # Solve KRR
        K_reg = K + self.reg * np.eye(n)
        try:
            self.alpha = np.linalg.solve(K_reg, np.array(self.support_v))
        except:
            self.alpha = np.linalg.lstsq(K_reg, np.array(self.support_v), rcond=None)[0]

    def get_effective_lambda(self, dY_test: float = 0.1) -> float:
        """
        Compute effective price impact λ = dP/dY.

        In inventory-averse model, this is:
            λ = λ_info + λ_inventory

        where λ_inventory = 2γ(T-t) is the inventory cost component.
        """
        P_curr = self.P_t

        # Peek ahead
        sig_peek = self.sig.copy()
        sig_peek.extend(self.dt, dY_test, self.T)
        P_peek = self.get_price(sig_peek)

        return max((P_peek - P_curr) / dY_test, 0.01)


class InformedTrader:
    """Insider who accounts for inventory-based pricing."""
    def __init__(self, T: float, dt: float):
        self.T = T
        self.dt = dt

    def compute_trading_rate(self, v: float, t: float, P_t: float,
                             lambda_: float) -> float:
        """
        Optimal trading rate given inventory-adjusted pricing.

        The optimal rate balances:
        - Information value: (v - P_t)
        - Execution cost: λ * θ
        """
        time_left = max(self.T - t, self.dt)
        return (v - P_t) / (lambda_ * time_left)


def compare_pricing_paths():
    """
    Demonstrate that same Y_T with different paths gets different prices.

    Key experiment:
    - Path A: Accumulate early, hold
    - Path B: Hold, accumulate late

    Both end at same Y_T, but Path A has higher inventory cost
    → should get higher price (MM requires compensation).
    """
    np.random.seed(42)

    T = 1.0
    dt = 0.01
    gamma = 0.5  # Significant inventory cost

    # Create MM for comparison
    mm = InventoryAverseMM(dt=dt, P_0=100.0, T=T, gamma=gamma)

    # Path A: Buy early (first half), hold (second half)
    sig_A = SignatureState()
    path_A = [0.0]
    target_Y = 10.0  # Target terminal order flow

    steps = int(T / dt)
    for i in range(steps):
        t = i * dt
        if t < 0.5 * T:
            dY = target_Y / (0.5 * T) * dt  # Accumulate
        else:
            dY = 0.0  # Hold
        sig_A.extend(dt, dY, T)
        path_A.append(sig_A.Y_T)

    # Path B: Hold (first half), buy late (second half)
    sig_B = SignatureState()
    path_B = [0.0]

    for i in range(steps):
        t = i * dt
        if t >= 0.5 * T:
            dY = target_Y / (0.5 * T) * dt  # Accumulate
        else:
            dY = 0.0  # Hold
        sig_B.extend(dt, dY, T)
        path_B.append(sig_B.Y_T)

    print("="*60)
    print("PATH COMPARISON: Same Y_T, Different Paths")
    print("="*60)

    print(f"\nPath A (buy early, hold):")
    print(f"  Y_T = {sig_A.Y_T:.2f}")
    print(f"  Lévy area = {sig_A.levy_area:.2f}")
    print(f"  Inventory cost = {sig_A.inventory_cost:.2f}")
    print(f"  Time-weighted inventory = {sig_A.time_weighted_inventory:.2f}")

    print(f"\nPath B (hold, buy late):")
    print(f"  Y_T = {sig_B.Y_T:.2f}")
    print(f"  Lévy area = {sig_B.levy_area:.2f}")
    print(f"  Inventory cost = {sig_B.inventory_cost:.2f}")
    print(f"  Time-weighted inventory = {sig_B.time_weighted_inventory:.2f}")

    # Inventory cost difference
    cost_ratio = sig_A.inventory_cost / max(sig_B.inventory_cost, 0.01)
    print(f"\n  → Path A inventory cost / Path B = {cost_ratio:.2f}x")

    if cost_ratio > 1.5:
        print("  ✓ Early accumulation correctly has higher inventory cost!")

    return sig_A, sig_B, path_A, path_B


def run_inventory_experiment():
    """
    Main experiment: Train inventory-averse MM and show path dependence.
    """
    np.random.seed(42)

    # Parameters
    T = 1.0
    dt = 0.005
    sigma_z = 1.0
    P_0 = 100.0
    num_episodes = 400

    # Compare different inventory cost levels
    gammas = [0.0, 0.2, 0.5]
    results = {}

    for gamma in gammas:
        print(f"\n{'='*60}")
        print(f"INVENTORY COST γ = {gamma}")
        print("="*60)

        mm = InventoryAverseMM(dt=dt, P_0=P_0, T=T, gamma=gamma, reg=0.1)
        insider = InformedTrader(T=T, dt=dt)

        Y_history = []
        v_history = []
        inv_cost_history = []
        levy_history = []
        lambda_history = []  # Track effective price impact over time

        def get_bimodal_v():
            if np.random.rand() < 0.5:
                return np.random.normal(80, 10)
            return np.random.normal(120, 10)

        for ep in tqdm(range(num_episodes), desc=f"γ={gamma}"):
            mm.reset()
            v = get_bimodal_v()

            episode_lambdas = []
            steps = int(T / dt)

            for i in range(steps):
                t = i * dt
                lam_t = mm.get_effective_lambda(0.1)
                lam_t = max(lam_t, 1.0)  # Floor
                episode_lambdas.append(lam_t)

                theta_t = insider.compute_trading_rate(v, t, mm.P_t, lam_t)
                theta_t = np.clip(theta_t, -50, 50)

                dX_t = theta_t * dt
                dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
                mm.filter_step(dX_t + dZ_t)

            Y_history.append(mm.sig.Y_T)
            v_history.append(v)
            inv_cost_history.append(mm.sig.inventory_cost)
            levy_history.append(mm.sig.levy_area)
            lambda_history.append(np.mean(episode_lambdas))

            mm.end_of_episode_update(v)

        results[gamma] = {
            'mm': mm,
            'Y': np.array(Y_history),
            'v': np.array(v_history),
            'inv_cost': np.array(inv_cost_history),
            'levy': np.array(levy_history),
            'lambda': np.array(lambda_history),
        }

        # Print statistics
        print(f"\nStatistics:")
        print(f"  Y_T: mean={results[gamma]['Y'].mean():.2f}, std={results[gamma]['Y'].std():.2f}")
        print(f"  Inventory cost: mean={results[gamma]['inv_cost'].mean():.2f}")
        print(f"  Avg λ: {results[gamma]['lambda'].mean():.2f}")
        print(f"  Corr(Y_T, v): {np.corrcoef(results[gamma]['Y'], results[gamma]['v'])[0,1]:.3f}")

    # Run path comparison
    print("\n")
    sig_A, sig_B, path_A, path_B = compare_pricing_paths()

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top left: Example paths showing early vs late accumulation
    ax = axes[0, 0]
    t_grid = np.linspace(0, T, len(path_A))
    ax.plot(t_grid, path_A, 'b-', linewidth=2, label='Early accumulation')
    ax.plot(t_grid, path_B, 'r-', linewidth=2, label='Late accumulation')
    ax.axhline(path_A[-1], color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Order Flow Y_t')
    ax.set_title('Same Y_T, Different Paths\n(Early has higher inventory cost)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top middle: Inventory cost vs Lévy area
    ax = axes[0, 1]
    for gamma in gammas:
        r = results[gamma]
        ax.scatter(r['levy'], r['inv_cost'], alpha=0.3, s=15, label=f'γ={gamma}')
    ax.set_xlabel('Lévy Area')
    ax.set_ylabel('Inventory Cost')
    ax.set_title('Inventory Cost vs Lévy Area')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Effective λ for different γ
    ax = axes[0, 2]
    for gamma in gammas:
        r = results[gamma]
        ax.hist(r['lambda'], bins=20, alpha=0.5, label=f'γ={gamma}', density=True)
    ax.set_xlabel('Average Effective λ')
    ax.set_ylabel('Density')
    ax.set_title('Price Impact Distribution\n(Higher γ → Higher λ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Y_T vs v for different γ
    ax = axes[1, 0]
    colors = ['blue', 'orange', 'green']
    for i, gamma in enumerate(gammas):
        r = results[gamma]
        ax.scatter(r['Y'], r['v'], alpha=0.3, s=15, c=colors[i], label=f'γ={gamma}')
    ax.set_xlabel('Terminal Order Flow Y_T')
    ax.set_ylabel('True Value v')
    ax.set_title('Y_T vs v\n(Higher γ → Smaller Y_T range)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom middle: Compare pricing for γ=0 vs γ=0.5
    ax = axes[1, 1]
    # Get pricing curves from trained MMs
    Y_grid = np.linspace(-30, 60, 100)

    for gamma in [0.0, 0.5]:
        r = results[gamma]
        mm = r['mm']

        # Simple pricing: just interpolate from data
        Y_data = r['Y']
        v_data = r['v']

        # Nadaraya-Watson for visualization
        bandwidth = np.std(Y_data) * 0.3
        P_grid = []
        for Y in Y_grid:
            distances = np.abs(Y_data - Y)
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            if weights.sum() > 1e-10:
                P_grid.append(weights @ v_data / weights.sum())
            else:
                P_grid.append(P_0)

        ax.plot(Y_grid, P_grid, linewidth=2, label=f'γ={gamma}')

    ax.axhline(120, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Y_T')
    ax.set_ylabel('E[v | Y_T]')
    ax.set_title('Learned Pricing Rules')
    ax.legend()
    ax.set_ylim(60, 140)
    ax.grid(True, alpha=0.3)

    # Bottom right: Price impact over time within episode
    ax = axes[1, 2]

    # Simulate one episode and track λ over time
    test_gamma = 0.5
    mm_test = InventoryAverseMM(dt=dt, P_0=P_0, T=T, gamma=test_gamma, reg=0.1)

    # Copy some training data
    mm_test.support_sigs = results[test_gamma]['mm'].support_sigs[:100]
    mm_test.support_v = results[test_gamma]['mm'].support_v[:100]
    mm_test._fit_krr()

    # Simulate with v=110 (buy pressure)
    v_test = 110
    mm_test.reset()
    t_track = []
    lambda_track = []
    inv_track = []

    steps = int(T / dt)
    for i in range(steps):
        t = i * dt
        t_track.append(t)
        lambda_track.append(mm_test.get_effective_lambda(0.1))
        inv_track.append(-mm_test.sig.Y_T)  # Inventory

        lam_t = max(mm_test.get_effective_lambda(0.1), 1.0)
        theta_t = (v_test - mm_test.P_t) / (lam_t * max(T - t, dt))
        theta_t = np.clip(theta_t, -50, 50)

        dX_t = theta_t * dt
        dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z * 0.3  # Low noise for clarity
        mm_test.filter_step(dX_t + dZ_t)

    ax.plot(t_track, lambda_track, 'b-', linewidth=2, label='Effective λ')
    ax2 = ax.twinx()
    ax2.plot(t_track, inv_track, 'r--', linewidth=2, label='Inventory')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price Impact λ', color='blue')
    ax2.set_ylabel('Inventory', color='red')
    ax.set_title(f'Intra-Episode Dynamics (γ={test_gamma})')
    ax.grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/docs/inventory_averse_mm.png', dpi=150)
    plt.close()

    print(f"\n\nPlot saved to docs/inventory_averse_mm.png")

    return results


if __name__ == "__main__":
    results = run_inventory_experiment()
