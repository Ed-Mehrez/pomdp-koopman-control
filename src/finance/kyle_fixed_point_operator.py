"""
Kyle Equilibrium via Explicit Fixed-Point Iteration

The equilibrium pricing rule P* satisfies:
    P* = T(P*)

where T is the "rational expectations operator":
    T(P)(Y) = E[v | Y_T = Y; θ*(P)]

and θ*(P) is the insider's best response to pricing rule P.

This gives us:
1. Explicit convergence tracking
2. Contraction mapping interpretation
3. Clean theory for publication
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from tqdm import tqdm
from scipy.interpolate import interp1d


class PricingRule:
    """
    Represents a pricing rule P: R -> R via kernel ridge regression.

    This is the "function" being iterated in the fixed-point scheme.
    """
    def __init__(self, Y_grid: np.ndarray, P_values: np.ndarray):
        """Initialize from grid representation."""
        self.Y_grid = Y_grid.copy()
        self.P_values = P_values.copy()
        # Create interpolator
        self._interp = interp1d(Y_grid, P_values, kind='linear',
                                bounds_error=False, fill_value='extrapolate')

    def __call__(self, Y: float) -> float:
        """Evaluate P(Y)."""
        return float(self._interp(Y))

    def evaluate_grid(self, Y_grid: np.ndarray) -> np.ndarray:
        """Evaluate on a grid."""
        return self._interp(Y_grid)

    def derivative(self, Y: float, dY: float = 0.1) -> float:
        """Compute dP/dY via finite difference."""
        return (self(Y + dY) - self(Y)) / dY

    def distance(self, other: 'PricingRule', Y_grid: np.ndarray) -> float:
        """Compute L2 distance to another pricing rule."""
        P1 = self.evaluate_grid(Y_grid)
        P2 = other.evaluate_grid(Y_grid)
        return np.sqrt(np.mean((P1 - P2) ** 2))


def insider_best_response(v: float, P: PricingRule, T: float, dt: float,
                          sigma_z: float, lambda_floor: float = 0.1) -> Tuple[np.ndarray, float]:
    """
    Compute insider's optimal trading path given pricing rule P.

    Returns the path of order flow Y and terminal Y_T.
    """
    steps = int(T / dt)
    Y_t = 0.0

    for i in range(steps):
        t = i * dt
        time_left = max(T - t, dt)

        # Current price and price impact
        P_t = P(Y_t)
        lam_t = max(P.derivative(Y_t), lambda_floor)

        # Optimal trading rate: θ* = (v - P_t) / (λ * (T - t))
        theta_t = (v - P_t) / (lam_t * time_left)
        theta_t = np.clip(theta_t, -100, 100)

        # Execute
        dX_t = theta_t * dt
        dZ_t = np.random.normal(0, np.sqrt(dt)) * sigma_z
        Y_t += dX_t + dZ_t

    return Y_t


def rational_expectations_operator(P: PricingRule,
                                   prior_sampler: Callable[[], float],
                                   T: float, dt: float, sigma_z: float,
                                   n_samples: int,
                                   Y_grid: np.ndarray,
                                   kernel_bandwidth: float = 0.3,
                                   lambda_floor: float = 0.5) -> PricingRule:
    """
    Apply the rational expectations operator T to pricing rule P.

    T(P)(Y) = E[v | Y_T = Y; θ*(P)]

    Implementation:
    1. Sample v from prior
    2. Compute insider's optimal Y_T under P
    3. Collect (Y_T, v) pairs
    4. Fit conditional expectation E[v | Y_T]
    """
    # Generate samples
    Y_samples = []
    v_samples = []

    for _ in range(n_samples):
        v = prior_sampler()
        Y_T = insider_best_response(v, P, T, dt, sigma_z, lambda_floor)
        Y_samples.append(Y_T)
        v_samples.append(v)

    Y_samples = np.array(Y_samples)
    v_samples = np.array(v_samples)

    # Fit E[v | Y_T] using Nadaraya-Watson kernel regression
    Y_std = max(np.std(Y_samples), 1.0)
    bandwidth = Y_std * kernel_bandwidth

    P_new_values = np.zeros_like(Y_grid)

    for i, Y_target in enumerate(Y_grid):
        # Kernel weights
        distances = np.abs(Y_samples - Y_target)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights_sum = weights.sum()

        if weights_sum > 1e-10:
            P_new_values[i] = (weights @ v_samples) / weights_sum
        else:
            # Extrapolation: use nearest sample
            nearest_idx = np.argmin(distances)
            P_new_values[i] = v_samples[nearest_idx]

    return PricingRule(Y_grid, P_new_values), Y_samples, v_samples


def find_equilibrium(prior_sampler: Callable[[], float],
                     P_0: float,
                     T: float, dt: float, sigma_z: float,
                     Y_range: float = 50.0,
                     n_grid: int = 100,
                     n_samples_per_iter: int = 200,
                     max_iterations: int = 20,
                     tol: float = 0.5,
                     lambda_floor: float = 0.5,
                     damping: float = 0.0,
                     verbose: bool = True) -> Tuple[PricingRule, List[PricingRule], List[float]]:
    """
    Find equilibrium pricing rule via fixed-point iteration.

    P_{n+1} = T(P_n)

    Returns:
        P_star: Converged pricing rule
        history: List of intermediate pricing rules
        distances: Convergence history ||P_{n+1} - P_n||
    """
    # Grid for representing pricing rules
    Y_grid = np.linspace(-Y_range, Y_range, n_grid)

    # Initial guess: linear Kyle pricing
    # In linear Kyle: P(Y) = P_0 + λ * Y where λ = σ_v / (2 * σ_z * sqrt(T))
    # For simplicity, start with a gentle slope
    lambda_init = 0.5
    P_init_values = P_0 + lambda_init * Y_grid
    P_current = PricingRule(Y_grid, P_init_values)

    history = [P_current]
    distances = []
    sample_history = []  # Store samples for visualization

    if verbose:
        print("="*60)
        print("FIXED-POINT ITERATION FOR KYLE EQUILIBRIUM")
        print("="*60)
        print(f"Grid: Y ∈ [{-Y_range}, {Y_range}], {n_grid} points")
        print(f"Samples per iteration: {n_samples_per_iter}")
        print(f"Tolerance: {tol}")
        if damping > 0:
            print(f"Damping: {damping} (P_new = {1-damping:.1f}*T(P) + {damping:.1f}*P)")
        print("-"*60)

    for iteration in range(max_iterations):
        # Apply operator
        P_raw, Y_samples, v_samples = rational_expectations_operator(
            P_current, prior_sampler, T, dt, sigma_z,
            n_samples_per_iter, Y_grid,
            lambda_floor=lambda_floor
        )

        # Apply damping: P_new = (1-damping)*T(P) + damping*P
        if damping > 0:
            P_raw_values = P_raw.evaluate_grid(Y_grid)
            P_current_values = P_current.evaluate_grid(Y_grid)
            P_damped_values = (1 - damping) * P_raw_values + damping * P_current_values
            P_new = PricingRule(Y_grid, P_damped_values)
        else:
            P_new = P_raw

        # Compute distance
        dist = P_current.distance(P_new, Y_grid)
        distances.append(dist)

        # Store samples from this iteration
        sample_history.append((Y_samples.copy(), v_samples.copy()))

        if verbose:
            # Check S-curve properties
            P_low = P_new(np.percentile(Y_samples, 10))
            P_mid = P_new(np.percentile(Y_samples, 50))
            P_high = P_new(np.percentile(Y_samples, 90))
            print(f"Iter {iteration+1:2d}: ||P_new - P_old|| = {dist:.3f}, "
                  f"P(Y_10%)={P_low:.1f}, P(Y_50%)={P_mid:.1f}, P(Y_90%)={P_high:.1f}")

        history.append(P_new)

        # Check convergence
        if dist < tol:
            if verbose:
                print("-"*60)
                print(f"Converged after {iteration+1} iterations!")
            break

        P_current = P_new

    return P_new, history, distances, sample_history


def run_fixed_point_experiment():
    """Run the fixed-point iteration experiment."""
    np.random.seed(42)

    # Parameters
    T = 1.0
    dt = 0.01  # Coarser for speed
    P_0 = 100.0
    sigma_z = 1.0

    # Bimodal prior
    def bimodal_prior():
        if np.random.rand() < 0.5:
            return np.random.normal(80, 10)
        return np.random.normal(120, 10)

    print("\n" + "="*60)
    print("EXPERIMENT: Fixed-Point Iteration for Bimodal Kyle")
    print("="*60)

    # Find equilibrium with damping for stable convergence
    P_star, history, distances, sample_history = find_equilibrium(
        prior_sampler=bimodal_prior,
        P_0=P_0,
        T=T, dt=dt, sigma_z=sigma_z,
        Y_range=60.0,
        n_grid=100,
        n_samples_per_iter=400,  # More samples for stability
        max_iterations=20,
        tol=0.2,
        lambda_floor=1.0,
        damping=0.3,  # Damping for stable convergence
        verbose=True
    )

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Convergence
    ax = axes[0, 0]
    ax.semilogy(range(1, len(distances)+1), distances, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||P_{n+1} - P_n||')
    ax.set_title('Fixed-Point Convergence')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Tolerance')
    ax.legend()

    # Top right: Evolution of pricing rules
    ax = axes[0, 1]
    Y_plot = np.linspace(-50, 60, 200)

    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for i, P in enumerate(history):
        alpha = 0.3 if i < len(history) - 1 else 1.0
        lw = 1 if i < len(history) - 1 else 3
        label = f'Iter {i}' if i == 0 or i == len(history)-1 else None
        ax.plot(Y_plot, P.evaluate_grid(Y_plot), color=colors[i],
                alpha=alpha, linewidth=lw, label=label)

    ax.axhline(120, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(100, color='black', linestyle=':', alpha=0.5)
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Y_T')
    ax.set_ylabel('P(Y_T)')
    ax.set_title('Evolution of Pricing Rule P_n → P*')
    ax.set_ylim(60, 140)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Final equilibrium with data
    ax = axes[1, 0]
    Y_final, v_final = sample_history[-1]

    colors_scatter = ['blue' if v < 100 else 'red' for v in v_final]
    ax.scatter(Y_final, v_final, c=colors_scatter, alpha=0.5, s=20, label='Samples')
    ax.plot(Y_plot, P_star.evaluate_grid(Y_plot), 'k-', linewidth=3, label='P*(Y)')
    ax.axhline(120, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Y_T')
    ax.set_ylabel('v / P(Y)')
    ax.set_title('Equilibrium: P*(Y) = E[v | Y_T = Y]')
    ax.set_ylim(50, 150)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Compare iterations 1, mid, final
    ax = axes[1, 1]
    iterations_to_show = [0, len(history)//2, len(history)-1]

    for i in iterations_to_show:
        P = history[i]
        label = f'Iteration {i}'
        ax.plot(Y_plot, P.evaluate_grid(Y_plot), linewidth=2, label=label)

    ax.axhline(120, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(100, color='black', linestyle=':', alpha=0.5)
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Y_T')
    ax.set_ylabel('P(Y_T)')
    ax.set_title('Pricing Rule: Initial → Middle → Final')
    ax.set_ylim(60, 140)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Kyle Equilibrium via Fixed-Point Iteration: P* = T(P*)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/examples/finance/results/kyle_fixed_point.png', dpi=150)
    plt.close()

    print(f"\nPlot saved to examples/finance/results/kyle_fixed_point.png")

    # Quantitative analysis
    print("\n" + "="*60)
    print("EQUILIBRIUM ANALYSIS")
    print("="*60)

    Y_final, v_final = sample_history[-1]
    corr = np.corrcoef(Y_final, v_final)[0, 1]
    print(f"Correlation(Y_T, v) at equilibrium: {corr:.3f}")

    # Check S-curve
    Y_10 = np.percentile(Y_final, 10)
    Y_50 = np.percentile(Y_final, 50)
    Y_90 = np.percentile(Y_final, 90)

    P_10 = P_star(Y_10)
    P_50 = P_star(Y_50)
    P_90 = P_star(Y_90)

    print(f"\nS-curve check:")
    print(f"  P(Y_10%={Y_10:.1f}) = {P_10:.1f}")
    print(f"  P(Y_50%={Y_50:.1f}) = {P_50:.1f}")
    print(f"  P(Y_90%={Y_90:.1f}) = {P_90:.1f}")

    if P_90 > 110 and P_10 < 90:
        print("  ✓ S-curve successfully separates bimodal modes!")

    # Convergence rate analysis
    if len(distances) > 2:
        ratios = [distances[i+1]/distances[i] for i in range(len(distances)-1) if distances[i] > 0.01]
        if ratios:
            avg_ratio = np.mean(ratios)
            print(f"\nConvergence rate: ||P_{{n+1}} - P_n|| / ||P_n - P_{{n-1}}|| ≈ {avg_ratio:.2f}")
            if avg_ratio < 1:
                print("  → Contraction mapping! (ratio < 1)")

    return P_star, history, distances


if __name__ == "__main__":
    run_fixed_point_experiment()
