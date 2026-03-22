"""
Strategic Bluffing Equilibrium

KEY QUESTION: When is trading against your information optimal?

Standard Kyle: Never. The optimal strategy is monotonic accumulation.

BUT with:
1. MM using exponential moving average (recent flow weighted more)
2. Finite memory / belief momentum
3. Non-linear pricing with "conviction" regions

...bluffing CAN be profitable.

SETUP:
- Bimodal prior: v ∈ {80, 120}
- Insider knows v = 120 (high value)
- MM uses EMA of order flow to form beliefs

BLUFFING STRATEGY:
1. Initially SELL (trade against information) → MM thinks v = 80
2. MM lowers price toward 80
3. BUY aggressively at artificially low price
4. Profit from the price manipulation

This shows:
- Bluffing creates HIGH |Lévy area| (just like spoofing)
- But bluffing is RATIONAL given MM's learning rule
- Lévy area alone doesn't distinguish manipulation from strategic bluffing

IMPLICATION FOR DETECTION:
Must combine |Lévy area| with equilibrium analysis to determine if
the pattern is consistent with rational informed trading.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class PathState:
    """Track order flow path."""
    t: float = 0.0
    Y: float = 0.0
    levy_area: float = 0.0

    def extend(self, dt: float, dY: float):
        self.levy_area += (self.Y * dt - self.t * dY) / 2
        self.t += dt
        self.Y += dY

    def copy(self):
        new = PathState()
        new.t, new.Y, new.levy_area = self.t, self.Y, self.levy_area
        return new


class MomentumMM:
    """
    Market maker with exponential moving average beliefs.

    This MM overweights recent order flow, creating exploitable momentum.

    Belief update:
        b_t = (1 - α) * b_{t-1} + α * signal(dY_t)

    Where α controls how quickly beliefs update.
    """
    def __init__(self, P_0: float, mu_L: float, mu_H: float,
                 alpha: float = 0.1, lambda_: float = 2.0):
        self.P_0 = P_0
        self.mu_L, self.mu_H = mu_L, mu_H
        self.alpha = alpha  # Belief update speed
        self.lambda_ = lambda_

        # Belief that v = mu_H (probability)
        self.belief_H = 0.5  # Start at prior
        self.price = P_0

    def reset(self):
        self.belief_H = 0.5
        self.price = self.P_0

    def update(self, dY: float, dt: float):
        """Update beliefs based on order flow signal."""
        # Signal: positive dY suggests high value
        signal = 1 / (1 + np.exp(-self.lambda_ * dY / dt))  # Sigmoid

        # EMA update
        self.belief_H = (1 - self.alpha) * self.belief_H + self.alpha * signal

        # Price is expected value
        self.price = self.belief_H * self.mu_H + (1 - self.belief_H) * self.mu_L

        return self.price


def simulate_honest_strategy(v: float, mm: MomentumMM, T: float, dt: float,
                              sigma_z: float) -> Tuple[PathState, List[float], List[float], float]:
    """
    Honest (Kyle-optimal) strategy: always trade toward v.
    """
    steps = int(T / dt)
    mm.reset()
    path = PathState()

    Y_history = [0.0]
    P_history = [mm.price]
    total_profit = 0.0

    for i in range(steps):
        t = i * dt
        time_left = max(T - t, dt)

        # Kyle optimal: trade toward v
        theta_t = (v - mm.price) / (mm.lambda_ * time_left)
        theta_t = np.clip(theta_t, -30, 30)

        dX_t = theta_t * dt
        dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
        dY_t = dX_t + dZ_t

        # Profit from this trade
        profit = theta_t * dt * (v - mm.price)
        total_profit += profit

        path.extend(dt, dY_t)
        mm.update(dY_t, dt)

        Y_history.append(path.Y)
        P_history.append(mm.price)

    return path, Y_history, P_history, total_profit


def simulate_bluffing_strategy(v: float, mm: MomentumMM, T: float, dt: float,
                                sigma_z: float, bluff_duration: float = 0.3,
                                bluff_intensity: float = 15.0) -> Tuple[PathState, List[float], List[float], float]:
    """
    Bluffing strategy: initially trade AGAINST information, then exploit.

    Phase 1 (t < bluff_duration * T):
        Trade opposite to information to fool MM
    Phase 2:
        Trade aggressively with information at manipulated price
    """
    steps = int(T / dt)
    mm.reset()
    path = PathState()

    Y_history = [0.0]
    P_history = [mm.price]
    total_profit = 0.0

    bluff_end = bluff_duration * T

    for i in range(steps):
        t = i * dt
        time_left = max(T - t, dt)

        if t < bluff_end:
            # BLUFFING PHASE: Trade opposite to information
            # If v = 120 (high), we SELL to make MM think v = 80
            direction = -1 if v > mm.P_0 else 1
            theta_t = direction * bluff_intensity
        else:
            # EXPLOITATION PHASE: Trade with information
            # Now buy aggressively at the artificially low price
            theta_t = (v - mm.price) / (mm.lambda_ * time_left) * 1.5  # More aggressive
            theta_t = np.clip(theta_t, -50, 50)

        dX_t = theta_t * dt
        dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
        dY_t = dX_t + dZ_t

        # Profit from this trade
        profit = theta_t * dt * (v - mm.price)
        total_profit += profit

        path.extend(dt, dY_t)
        mm.update(dY_t, dt)

        Y_history.append(path.Y)
        P_history.append(mm.price)

    return path, Y_history, P_history, total_profit


def find_optimal_bluff_parameters(v: float, mm_template: MomentumMM,
                                   T: float, dt: float, sigma_z: float,
                                   n_trials: int = 50):
    """
    Find the bluffing parameters that maximize expected profit.
    """
    best_params = None
    best_profit = -np.inf

    # Grid search over bluff duration and intensity
    for bluff_duration in np.linspace(0.1, 0.5, 5):
        for bluff_intensity in np.linspace(5, 25, 5):
            profits = []
            for _ in range(n_trials):
                mm = MomentumMM(mm_template.P_0, mm_template.mu_L,
                               mm_template.mu_H, mm_template.alpha, mm_template.lambda_)
                _, _, _, profit = simulate_bluffing_strategy(
                    v, mm, T, dt, sigma_z, bluff_duration, bluff_intensity)
                profits.append(profit)

            avg_profit = np.mean(profits)
            if avg_profit > best_profit:
                best_profit = avg_profit
                best_params = (bluff_duration, bluff_intensity)

    return best_params, best_profit


def run_experiment():
    """Compare honest vs bluffing strategies."""
    np.random.seed(42)

    # Parameters
    T, dt = 1.0, 0.01
    sigma_z = 0.3
    P_0 = 100.0
    mu_L, mu_H = 80.0, 120.0

    # MM with momentum (exploitable)
    alpha = 0.15  # How fast beliefs update
    lambda_ = 2.0

    print("="*60)
    print("BLUFFING EQUILIBRIUM EXPERIMENT")
    print("="*60)
    print(f"Prior: v ∈ {{{mu_L}, {mu_H}}} with equal probability")
    print(f"MM belief update speed α = {alpha}")
    print(f"Insider knows: v = {mu_H} (high value)")

    # Create template MM
    mm_template = MomentumMM(P_0, mu_L, mu_H, alpha, lambda_)

    # Find optimal bluff parameters
    print("\nSearching for optimal bluffing strategy...")
    best_params, best_bluff_profit = find_optimal_bluff_parameters(
        mu_H, mm_template, T, dt, sigma_z, n_trials=30)

    print(f"Optimal bluff: duration={best_params[0]:.2f}T, intensity={best_params[1]:.1f}")

    # Compare strategies over many trials
    n_compare = 100
    honest_profits = []
    bluff_profits = []
    honest_levy = []
    bluff_levy = []

    for _ in range(n_compare):
        # Honest strategy
        mm = MomentumMM(P_0, mu_L, mu_H, alpha, lambda_)
        path_h, Y_h, P_h, profit_h = simulate_honest_strategy(mu_H, mm, T, dt, sigma_z)
        honest_profits.append(profit_h)
        honest_levy.append(abs(path_h.levy_area))

        # Bluffing strategy
        mm = MomentumMM(P_0, mu_L, mu_H, alpha, lambda_)
        path_b, Y_b, P_b, profit_b = simulate_bluffing_strategy(
            mu_H, mm, T, dt, sigma_z, best_params[0], best_params[1])
        bluff_profits.append(profit_b)
        bluff_levy.append(abs(path_b.levy_area))

    print("\n" + "-"*40)
    print("STRATEGY COMPARISON (v = 120)")
    print("-"*40)
    print(f"Honest strategy:  avg profit = {np.mean(honest_profits):.2f} ± {np.std(honest_profits):.2f}")
    print(f"Bluffing strategy: avg profit = {np.mean(bluff_profits):.2f} ± {np.std(bluff_profits):.2f}")

    profit_improvement = (np.mean(bluff_profits) - np.mean(honest_profits)) / abs(np.mean(honest_profits)) * 100
    print(f"\nBluffing profit improvement: {profit_improvement:+.1f}%")

    print(f"\n|Lévy area| comparison:")
    print(f"  Honest:   {np.mean(honest_levy):.3f} ± {np.std(honest_levy):.3f}")
    print(f"  Bluffing: {np.mean(bluff_levy):.3f} ± {np.std(bluff_levy):.3f}")
    print(f"  Ratio: {np.mean(bluff_levy)/np.mean(honest_levy):.1f}x")

    # Key insight
    if np.mean(bluff_profits) > np.mean(honest_profits):
        print("\n✓ BLUFFING IS OPTIMAL against this MM!")
        print("  → High |Lévy area| can be EQUILIBRIUM behavior")
        print("  → Cannot use Lévy area alone to detect manipulation")

    # Generate example trajectories for plotting
    np.random.seed(123)  # Fixed seed for reproducible plot

    mm = MomentumMM(P_0, mu_L, mu_H, alpha, lambda_)
    path_h, Y_h, P_h, _ = simulate_honest_strategy(mu_H, mm, T, dt, sigma_z * 0.3)

    mm = MomentumMM(P_0, mu_L, mu_H, alpha, lambda_)
    path_b, Y_b, P_b, _ = simulate_bluffing_strategy(mu_H, mm, T, dt, sigma_z * 0.3,
                                                      best_params[0], best_params[1])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    t_grid = np.linspace(0, T, len(Y_h))

    # Top left: Order flow paths
    ax = axes[0, 0]
    ax.plot(t_grid, Y_h, 'b-', linewidth=2, label='Honest')
    ax.plot(t_grid, Y_b, 'r-', linewidth=2, label='Bluffing')
    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(best_params[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Bluff ends (t={best_params[0]:.1f})')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Cumulative Order Flow Yₜ')
    ax.set_title('Trading Paths: Honest vs Bluffing')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top middle: Price paths
    ax = axes[0, 1]
    ax.plot(t_grid, P_h, 'b-', linewidth=2, label='Price (Honest)')
    ax.plot(t_grid, P_b, 'r-', linewidth=2, label='Price (Bluffing)')
    ax.axhline(mu_H, color='green', linestyle='--', alpha=0.7, label=f'True v = {mu_H}')
    ax.axhline(mu_L, color='gray', linestyle='--', alpha=0.5, label=f'Low mode = {mu_L}')
    ax.axvline(best_params[0], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('MM Price Pₜ')
    ax.set_title('Price Evolution\n(Bluffing drives price down, then exploits)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: Profit comparison histograms
    ax = axes[0, 2]
    bins = np.linspace(min(min(honest_profits), min(bluff_profits)),
                       max(max(honest_profits), max(bluff_profits)), 25)
    ax.hist(honest_profits, bins=bins, alpha=0.6, label='Honest', color='blue', density=True)
    ax.hist(bluff_profits, bins=bins, alpha=0.6, label='Bluffing', color='red', density=True)
    ax.axvline(np.mean(honest_profits), color='blue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(bluff_profits), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Profit')
    ax.set_ylabel('Density')
    ax.set_title(f'Profit Distribution\nBluffing: +{profit_improvement:.1f}% improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Lévy area comparison
    ax = axes[1, 0]
    ax.hist(honest_levy, bins=20, alpha=0.6, label='Honest', color='blue', density=True)
    ax.hist(bluff_levy, bins=20, alpha=0.6, label='Bluffing', color='red', density=True)
    ax.set_xlabel('|Lévy Area|')
    ax.set_ylabel('Density')
    ax.set_title(f'|Lévy Area| Distribution\nBluffing has {np.mean(bluff_levy)/np.mean(honest_levy):.1f}x higher')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom middle: Y_T vs Lévy area scatter
    ax = axes[1, 1]
    # Generate more samples for scatter
    Y_T_honest, levy_honest = [], []
    Y_T_bluff, levy_bluff = [], []
    for _ in range(50):
        mm = MomentumMM(P_0, mu_L, mu_H, alpha, lambda_)
        path, Y_path, _, _ = simulate_honest_strategy(mu_H, mm, T, dt, sigma_z)
        Y_T_honest.append(path.Y)
        levy_honest.append(path.levy_area)

        mm = MomentumMM(P_0, mu_L, mu_H, alpha, lambda_)
        path, Y_path, _, _ = simulate_bluffing_strategy(mu_H, mm, T, dt, sigma_z,
                                                         best_params[0], best_params[1])
        Y_T_bluff.append(path.Y)
        levy_bluff.append(path.levy_area)

    ax.scatter(Y_T_honest, levy_honest, alpha=0.6, s=30, c='blue', label='Honest')
    ax.scatter(Y_T_bluff, levy_bluff, alpha=0.6, s=30, c='red', label='Bluffing')
    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Terminal Order Flow Y_T')
    ax.set_ylabel('Lévy Area')
    ax.set_title('Signature Space\n(Both are RATIONAL informed strategies)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Key insight text
    ax = axes[1, 2]
    ax.axis('off')

    insight_text = f"""
    KEY INSIGHT: When Bluffing is Optimal
    ═══════════════════════════════════════

    Setup:
    • MM uses exponential moving average (α = {alpha})
    • Recent order flow weighted more heavily
    • This creates exploitable momentum

    Result:
    • Bluffing profit: +{profit_improvement:.1f}% vs honest
    • Bluffing |Lévy area|: {np.mean(bluff_levy)/np.mean(honest_levy):.1f}x higher

    IMPLICATION FOR DETECTION:
    ───────────────────────────────────────
    • High |Lévy area| ≠ always manipulation
    • May be OPTIMAL informed trading
    • Detection requires equilibrium analysis

    What Lévy area CAN detect:
    ───────────────────────────────────────
    • Uninformed manipulation (spoofing)
    • Out-of-equilibrium behavior

    What it CANNOT detect:
    ───────────────────────────────────────
    • Strategic bluffing (informed + reversals)
    • Requires knowing the equilibrium structure
    """
    ax.text(0.02, 0.98, insight_text, transform=ax.transAxes,
            fontsize=10, family='monospace', verticalalignment='top')

    plt.tight_layout()
    plt.savefig('/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/docs/bluffing_equilibrium.png', dpi=150)
    plt.close()

    print(f"\nPlot saved to docs/bluffing_equilibrium.png")

    return {
        'honest_profits': honest_profits,
        'bluff_profits': bluff_profits,
        'honest_levy': honest_levy,
        'bluff_levy': bluff_levy,
        'profit_improvement': profit_improvement,
        'best_params': best_params
    }


if __name__ == "__main__":
    results = run_experiment()
