"""
LQG Baseline: Establish benchmark for KRONIC-POMDP

LQG = Linear-Quadratic-Gaussian:
- Linear dynamics: dx = Ax dt + Bu dt + G dW
- Quadratic cost: J = E[∫(x'Qx + u'Ru)dt + x_T'Q_f x_T]
- Gaussian noise: both process and observation

Key property: SEPARATION PRINCIPLE
- Optimal filter: Kalman filter (belief = Gaussian)
- Optimal control: LQR on estimated state x̂

This gives us ground truth to compare KRONIC-POMDP against.
"""

import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt


class LQGSystem:
    """Simple 1D LQG tracking problem."""

    def __init__(self,
                 A=-0.5,      # State dynamics (mean-reverting)
                 B=1.0,       # Control input
                 C=1.0,       # Observation matrix
                 G=0.3,       # Process noise
                 H=0.5,       # Observation noise
                 Q=1.0,       # State cost
                 R=0.1,       # Control cost
                 dt=0.01):

        self.A = A
        self.B = B
        self.C = C
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.dt = dt

        # Precompute LQR and Kalman gains
        self._compute_lqr_gain()
        self._compute_kalman_gain()

    def _compute_lqr_gain(self):
        """Solve algebraic Riccati for LQR gain."""
        # Continuous-time ARE: A'P + PA - PBR^{-1}B'P + Q = 0
        # For scalar: simplifies significantly
        # P satisfies: 2AP - P²B²/R + Q = 0
        # K = R^{-1}B'P = BP/R

        # Quadratic formula for P
        a = self.B**2 / self.R
        b = -2 * self.A
        c = -self.Q

        P = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        self.P_lqr = P
        self.K_lqr = self.B * P / self.R

        print(f"LQR gain K = {self.K_lqr:.4f}")
        print(f"Closed-loop pole: A - BK = {self.A - self.B * self.K_lqr:.4f}")

    def _compute_kalman_gain(self):
        """Solve algebraic Riccati for Kalman gain."""
        # Dual ARE: AΣ + ΣA' - ΣC'(H²)^{-1}CΣ + G² = 0
        # For scalar: 2AΣ - Σ²C²/H² + G² = 0
        # L = ΣC/H²

        a = self.C**2 / self.H**2
        b = -2 * self.A
        c = -self.G**2

        Sigma = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        self.Sigma_kalman = Sigma
        self.L_kalman = Sigma * self.C / self.H**2

        print(f"Kalman gain L = {self.L_kalman:.4f}")
        print(f"Observer pole: A - LC = {self.A - self.L_kalman * self.C:.4f}")

    def simulate(self, x0=1.0, T=10.0, seed=42):
        """Simulate closed-loop LQG system."""
        np.random.seed(seed)

        n_steps = int(T / self.dt)
        sqrt_dt = np.sqrt(self.dt)

        # Storage
        t = np.zeros(n_steps)
        x_true = np.zeros(n_steps)  # True state
        x_hat = np.zeros(n_steps)   # Estimated state
        y = np.zeros(n_steps)       # Observations
        u = np.zeros(n_steps)       # Control

        # Initial conditions
        x_true[0] = x0
        x_hat[0] = 0.0  # Start with zero estimate (uncertain)

        for k in range(n_steps - 1):
            t[k+1] = (k+1) * self.dt

            # Control based on estimate (certainty equivalence)
            u[k] = -self.K_lqr * x_hat[k]

            # True state evolution
            dW = np.random.randn() * sqrt_dt
            x_true[k+1] = x_true[k] + (self.A * x_true[k] + self.B * u[k]) * self.dt + self.G * dW

            # Observation (with noise)
            dV = np.random.randn() * sqrt_dt
            y[k+1] = self.C * x_true[k+1] + self.H * dV / sqrt_dt  # Instantaneous

            # Kalman filter update (proper discrete form)
            # 1. Predict
            x_pred = x_hat[k] + (self.A * x_hat[k] + self.B * u[k]) * self.dt
            # 2. Innovation against prediction
            innovation = y[k+1] - self.C * x_pred
            # 3. Update (note: L*dt for continuous-to-discrete conversion)
            x_hat[k+1] = x_pred + self.L_kalman * innovation * self.dt

        return {
            't': t,
            'x_true': x_true,
            'x_hat': x_hat,
            'y': y,
            'u': u
        }

    def compute_cost(self, result):
        """Compute realized LQG cost."""
        x = result['x_true']
        u = result['u']

        running_cost = np.sum(self.Q * x**2 + self.R * u**2) * self.dt
        terminal_cost = self.Q * x[-1]**2

        return running_cost + terminal_cost


class BeliefKRONIC:
    """
    KRONIC in belief space for LQG.

    For LQG, belief state is (x̂, Σ). Since Σ converges to steady-state,
    we can work with x̂ alone after transient.

    Key insight: The optimal value function for LQG is:
        V(x̂, Σ) = x̂' P x̂ + trace(P Σ) + constant

    The first term depends on x̂, the second is constant at steady-state.
    So eigenfunctions of the x̂ dynamics suffice!
    """

    def __init__(self, lqg_system):
        self.sys = lqg_system

    def learn_eigenfunctions(self, trajectories, n_eigs=5):
        """
        Learn eigenfunctions from belief trajectories.

        For 1D LQG, the generator of x̂ dynamics is:
            L_hat f(x̂) = (A - LC)x̂ · f'(x̂) + ½(LH)² · f''(x̂)

        This is an Ornstein-Uhlenbeck process!
        Eigenfunctions are Hermite polynomials.
        """
        # Collect (x̂, dx̂) pairs from trajectories
        x_hat_all = []
        dx_hat_all = []

        for traj in trajectories:
            x_hat = traj['x_hat']
            dx_hat = np.diff(x_hat) / self.sys.dt
            x_hat_all.extend(x_hat[:-1])
            dx_hat_all.extend(dx_hat)

        x_hat_all = np.array(x_hat_all)
        dx_hat_all = np.array(dx_hat_all)

        # For OU process, eigenfunctions are Hermite polynomials
        # λ_n = n(A - LC) for n-th Hermite polynomial
        # H_0 = 1, H_1 = x, H_2 = x² - σ², etc.

        A_eff = self.sys.A - self.sys.L_kalman * self.sys.C
        sigma_sq = (self.sys.L_kalman * self.sys.H)**2 / (-2 * A_eff)

        print(f"\nBelief dynamics: dx̂ = {A_eff:.3f} x̂ dt + noise")
        print(f"Stationary variance: σ² = {sigma_sq:.4f}")

        # Eigenvalues
        eigenvalues = [n * A_eff for n in range(n_eigs)]
        print(f"\nAnalytical eigenvalues: {eigenvalues}")

        return eigenvalues, sigma_sq

    def verify_value_function(self, result):
        """
        Verify that V(x̂) = P x̂² matches the cost-to-go.
        """
        x_hat = result['x_hat']

        # Predicted value (from LQR Riccati)
        V_predicted = self.sys.P_lqr * x_hat**2

        # This should track the actual cost-to-go (approximately)
        # Cost-to-go at time k = sum of future costs
        x = result['x_true']
        u = result['u']

        n = len(x)
        V_actual = np.zeros(n)
        for k in range(n-1, -1, -1):
            if k == n-1:
                V_actual[k] = self.sys.Q * x[k]**2
            else:
                V_actual[k] = (self.sys.Q * x[k]**2 + self.sys.R * u[k]**2) * self.sys.dt + V_actual[k+1]

        return V_predicted, V_actual


def main():
    print("=" * 60)
    print("LQG BASELINE FOR KRONIC-POMDP")
    print("=" * 60)

    # Create system
    lqg = LQGSystem(
        A=-0.5,   # Mean-reverting
        B=1.0,
        C=1.0,
        G=0.3,    # Process noise
        H=0.5,    # Observation noise
        Q=1.0,
        R=0.1,
        dt=0.01
    )

    # Simulate
    print("\n[1] Simulating LQG system...")
    result = lqg.simulate(x0=2.0, T=20.0)

    cost = lqg.compute_cost(result)
    print(f"Total cost: {cost:.4f}")

    # Belief KRONIC
    print("\n[2] Belief-space KRONIC analysis...")
    kronic = BeliefKRONIC(lqg)
    eigenvalues, sigma_sq = kronic.learn_eigenfunctions([result])

    # Verify value function
    print("\n[3] Verifying value function structure...")
    V_pred, V_actual = kronic.verify_value_function(result)

    corr = np.corrcoef(V_pred[100:], V_actual[100:])[0, 1]  # Skip transient
    print(f"Correlation(V_predicted, V_actual): {corr:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # State tracking
    ax = axes[0, 0]
    ax.plot(result['t'], result['x_true'], 'b-', label='True x', alpha=0.7)
    ax.plot(result['t'], result['x_hat'], 'r--', label='Estimate x̂', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('State Estimation (Kalman Filter)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Estimation error
    ax = axes[0, 1]
    error = result['x_true'] - result['x_hat']
    ax.plot(result['t'], error, 'g-', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(np.sqrt(lqg.Sigma_kalman), color='r', linestyle=':', label=f'±√Σ = ±{np.sqrt(lqg.Sigma_kalman):.2f}')
    ax.axhline(-np.sqrt(lqg.Sigma_kalman), color='r', linestyle=':')
    ax.set_xlabel('Time')
    ax.set_ylabel('Estimation Error')
    ax.set_title('Estimation Error (x - x̂)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Control
    ax = axes[1, 0]
    ax.plot(result['t'], result['u'], 'purple', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control u')
    ax.set_title(f'Control Input (K = {lqg.K_lqr:.3f})')
    ax.grid(True, alpha=0.3)

    # Value function
    ax = axes[1, 1]
    ax.plot(result['t'], V_actual, 'b-', label='Actual cost-to-go', alpha=0.7)
    ax.plot(result['t'], V_pred, 'r--', label='V(x̂) = Px̂²', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Value Function (corr = {corr:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/lqg_baseline.png', dpi=150)
    plt.close()

    print("\n[4] Saved figure to kronic_pomdp/experiments/lqg_baseline.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
LQG establishes the baseline:
1. Belief state: x̂ (Kalman estimate) + Σ (covariance, constant at steady-state)
2. Value function: V(x̂) = P·x̂² (quadratic in belief)
3. Optimal control: u = -K·x̂ (linear in belief)

For KRONIC-POMDP:
- Eigenfunctions of belief dynamics are Hermite polynomials
- Eigenvalues: λ_n = n·(A - LC) = {[f'{n*(lqg.A - lqg.L_kalman*lqg.C):.3f}' for n in range(4)]}
- V(x̂) = P·x̂² is the n=2 eigenfunction (up to scaling)

Next steps:
- Replace Kalman with signature-based belief
- Learn eigenfunctions from path signatures
- Compare control performance
""")


if __name__ == "__main__":
    main()
