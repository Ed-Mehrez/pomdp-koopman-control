r"""
Honest Benchmark: Merton-Heston POMDP
======================================

Does any data-driven method produce better portfolio outcomes than
simple plug-in baselines for the Heston POMDP?

Setup:
- Hidden state: V_t (Heston stochastic volatility)
- Observable: returns r_t = mu*dt + sqrt(V_t)*sqrt(dt)*z1_t
- Goal: Estimate V_t, compute allocation pi_t, maximize CRRA utility

Design:
- Strict train/test split (disjoint seed ranges)
- Paired comparison (all strategies share same Brownian noise)
- 5 macro-seeds for error bars
- CE = (mean(W^{1-gamma}))^{1/(1-gamma)} as primary metric
- Vary rho in {0, -0.3, -0.5, -0.7}
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    import signatory
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False
    print("[WARN] signatory not available; SigRidge filter disabled.")


# ====================================================================
# Section 1: Simulation Infrastructure
# ====================================================================

@dataclass
class HestonParams:
    mu: float = 0.08
    r: float = 0.02
    gamma: float = 2.0
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.5
    dt: float = 1 / 252


def generate_shared_noise(n_steps: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-generate iid N(0,1) draws for paired comparison.

    z1 drives stock returns, z2 is independent (combined with rho for vol).
    """
    rng = np.random.RandomState(seed)
    z1 = rng.randn(n_steps)
    z2 = rng.randn(n_steps)
    return z1, z2


def simulate_heston(p: HestonParams, z1: np.ndarray, z2: np.ndarray,
                    V0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Heston path from pre-generated noise. Returns (V, returns)."""
    n = len(z1)
    V = np.zeros(n)
    returns = np.zeros(n)
    V[0] = V0 if V0 is not None else p.theta
    sqrt_dt = np.sqrt(p.dt)

    for t in range(1, n):
        v = max(V[t - 1], 1e-8)
        sv = np.sqrt(v)
        dW_S = sqrt_dt * z1[t]
        dW_V = sqrt_dt * (p.rho * z1[t] + np.sqrt(1 - p.rho**2) * z2[t])

        returns[t] = p.mu * p.dt + sv * dW_S
        V[t] = max(v + p.kappa * (p.theta - v) * p.dt + p.xi * sv * dW_V, 1e-8)

    return V, returns


def simulate_wealth(returns: np.ndarray, pi: np.ndarray,
                    r: float, dt: float) -> np.ndarray:
    """Deterministic wealth from returns and allocations.

    W[t+1] = W[t] * (1 + r*dt + pi[t]*(returns[t] - r*dt))
    No extra randomness — the return already embeds the Brownian motion.
    """
    n = len(returns)
    W = np.ones(n)
    for t in range(1, n):
        portfolio_ret = r * dt + pi[t - 1] * (returns[t] - r * dt)
        W[t] = W[t - 1] * max(1 + portfolio_ret, 1e-10)
    return W


def analytical_hedging_demand(p: HestonParams) -> float:
    """Exact infinite-horizon hedging demand (Chacko-Viceira / Liu 2007).

    pi* = myopic + hedge_constant
    hedge_constant = rho * xi * (1-gamma) * B_inf / gamma
    where B_inf solves: 0.5*xi^2*B^2 - kappa_eff*B + gamma_coeff = 0
    """
    a = p.mu - p.r
    kappa_eff = p.kappa - p.rho * p.xi * (1 - p.gamma) * a / p.gamma
    gamma_coeff = -(1 - p.gamma) * a**2 / (2 * p.gamma**2)
    disc = kappa_eff**2 - 4 * 0.5 * p.xi**2 * gamma_coeff
    if disc < 0:
        return float('nan')
    B_inf = (kappa_eff - np.sqrt(disc)) / (p.xi**2)
    return p.rho * p.xi * (1 - p.gamma) * B_inf / p.gamma


# ====================================================================
# Section 2: Volatility Filters
# ====================================================================

class OracleFilter:
    """Returns true V. Upper bound on any filter."""
    def __init__(self, p: HestonParams):
        self.p = p

    def train(self, train_returns, train_V):
        pass

    def filter(self, returns, V_true):
        return V_true.copy()


class BPFFilter:
    """Bootstrap Particle Filter. Knows true model params."""
    def __init__(self, p: HestonParams, n_particles: int = 1000):
        self.p = p
        self.N = n_particles

    def train(self, train_returns, train_V):
        pass  # knows model

    def filter(self, returns, V_true=None):
        p = self.p
        T = len(returns)
        sqrt_dt = np.sqrt(p.dt)
        # Separate RNG so particle noise doesn't disturb shared stream
        rng = np.random.RandomState(abs(hash(tuple(returns[:5].tolist()))) % (2**31))

        particles = np.clip(
            rng.normal(p.theta, 0.01, self.N), 1e-6, 1.0
        )
        V_est = np.zeros(T)
        V_est[0] = p.theta

        for t in range(1, T):
            v = np.maximum(particles, 1e-6)
            dW = rng.randn(self.N) * sqrt_dt
            drift = p.kappa * (p.theta - v) * p.dt
            diffusion = p.xi * np.sqrt(v) * dW
            particles_pred = np.maximum(v + drift + diffusion, 1e-6)

            # Likelihood: r_t ~ N(mu*dt, V*dt)
            var_term = particles_pred * p.dt
            r_obs = returns[t]
            log_w = -0.5 * (np.log(var_term) + (r_obs - p.mu * p.dt)**2 / var_term)
            log_w -= np.max(log_w)
            weights = np.exp(log_w)
            weights /= np.sum(weights)

            V_est[t] = np.sum(particles_pred * weights)

            # Resample
            indices = rng.choice(self.N, size=self.N, p=weights)
            particles = particles_pred[indices]

        return V_est


class EWMAFilter:
    """Exponential weighted moving average of r^2/dt. No model knowledge."""
    def __init__(self, p: HestonParams, lam: float = 0.94):
        self.p = p
        self.lam = lam

    def train(self, train_returns, train_V):
        pass

    def filter(self, returns, V_true=None):
        n = len(returns)
        V_est = np.zeros(n)
        V_est[0] = self.p.theta
        for t in range(1, n):
            rv = returns[t]**2 / self.p.dt
            V_est[t] = self.lam * V_est[t - 1] + (1 - self.lam) * rv
        return V_est


class RealizedQVFilter:
    """Windowed mean of r^2/dt."""
    def __init__(self, p: HestonParams, window: int = 20):
        self.p = p
        self.window = window

    def train(self, train_returns, train_V):
        pass

    def filter(self, returns, V_true=None):
        n = len(returns)
        rv = returns**2 / self.p.dt
        V_est = np.zeros(n)
        V_est[0] = self.p.theta
        for t in range(1, n):
            start = max(1, t - self.window + 1)
            V_est[t] = np.mean(rv[start:t + 1])
        return V_est


class SigRidgeFilter:
    """Logsignature features -> Ridge regression -> V estimate.

    Trained with supervision (V_true available in training).
    Applied without V_true at test time.
    Reports held-out R^2 from internal 80/20 split.
    """
    def __init__(self, p: HestonParams, window: int = 20,
                 sig_level: int = 3, alpha: float = 1.0):
        self.p = p
        self.window = window
        self.sig_level = sig_level
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()
        self.oos_r2 = None

    def _extract_sigs_batch(self, returns: np.ndarray) -> np.ndarray:
        """Extract logsignature features from all windows."""
        n = len(returns)
        paths = []
        for t in range(self.window, n):
            window_rets = returns[t - self.window:t]
            time_steps = np.linspace(0, 1, self.window)
            cumsum = np.cumsum(window_rets)
            paths.append(np.column_stack([time_steps, cumsum]))

        if not paths:
            return np.array([])

        paths_t = torch.tensor(np.array(paths), dtype=torch.float32)
        with torch.no_grad():
            sigs = signatory.logsignature(paths_t, self.sig_level)
        return sigs.numpy()

    def train(self, train_returns: np.ndarray, train_V: np.ndarray) -> float:
        """Train sig->V mapping. Returns held-out R^2."""
        sigs = self._extract_sigs_batch(train_returns)
        if len(sigs) == 0:
            self.oos_r2 = float('nan')
            return self.oos_r2

        targets = train_V[self.window:self.window + len(sigs)]

        # 80/20 internal split for honest R^2
        n = len(sigs)
        n_train = int(0.8 * n)
        sigs_tr, sigs_val = sigs[:n_train], sigs[n_train:]
        tgt_tr, tgt_val = targets[:n_train], targets[n_train:]

        self.scaler.fit(sigs_tr)
        X_tr = self.scaler.transform(sigs_tr)
        X_val = self.scaler.transform(sigs_val)

        model = Ridge(alpha=self.alpha)
        model.fit(X_tr, tgt_tr)
        pred_val = model.predict(X_val)
        ss_res = np.sum((tgt_val - pred_val)**2)
        ss_tot = np.sum((tgt_val - np.mean(tgt_val))**2)
        self.oos_r2 = 1 - ss_res / (ss_tot + 1e-10)

        # Refit on full data for deployment
        X_full = self.scaler.fit_transform(sigs)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_full, targets)

        return self.oos_r2

    def filter(self, returns: np.ndarray, V_true=None) -> np.ndarray:
        """Apply trained model. No access to V_true."""
        n = len(returns)
        V_est = np.full(n, self.p.theta)

        if self.model is None:
            return V_est

        sigs = self._extract_sigs_batch(returns)
        if len(sigs) > 0:
            X = self.scaler.transform(sigs)
            preds = self.model.predict(X)
            V_est[self.window:self.window + len(preds)] = np.maximum(preds, 1e-6)

        return V_est


# ====================================================================
# Section 3: Controllers
# ====================================================================

class MyopicController:
    """pi = (mu-r) / (gamma * V_hat), clipped to [0, max_lev]."""
    def __init__(self, p: HestonParams, max_lev: float = 3.0):
        self.p = p
        self.max_lev = max_lev

    def allocate(self, V_hat: np.ndarray) -> np.ndarray:
        V_safe = np.maximum(V_hat, 1e-6)
        pi = (self.p.mu - self.p.r) / (self.p.gamma * V_safe)
        return np.clip(pi, 0.0, self.max_lev)


class HedgingController:
    """Myopic + analytical Chacko-Viceira hedging demand (constant offset)."""
    def __init__(self, p: HestonParams, max_lev: float = 3.0):
        self.p = p
        self.max_lev = max_lev
        self.hedge = analytical_hedging_demand(p)

    def allocate(self, V_hat: np.ndarray) -> np.ndarray:
        V_safe = np.maximum(V_hat, 1e-6)
        pi = (self.p.mu - self.p.r) / (self.p.gamma * V_safe) + self.hedge
        return np.clip(pi, 0.0, self.max_lev)


class ConstantController:
    """pi = (mu-r) / (gamma * theta). No filtering needed."""
    def __init__(self, p: HestonParams):
        self.pi_const = (p.mu - p.r) / (p.gamma * p.theta)

    def allocate(self, V_hat: np.ndarray) -> np.ndarray:
        return np.full_like(V_hat, self.pi_const)


# ====================================================================
# Section 4: Metrics
# ====================================================================

def compute_CE(W_terminals: np.ndarray, gamma: float) -> float:
    """Certainty equivalent for CRRA utility."""
    if gamma == 1.0:
        return np.exp(np.mean(np.log(np.maximum(W_terminals, 1e-20))))
    else:
        utilities = W_terminals**(1 - gamma) / (1 - gamma)
        E_U = np.mean(utilities)
        return ((1 - gamma) * E_U) ** (1 / (1 - gamma))


def compute_sharpe(W_terminals: np.ndarray, T: float) -> float:
    """Annualized Sharpe from terminal wealth distribution."""
    log_ret = np.log(np.maximum(W_terminals, 1e-20))
    mean_ann = np.mean(log_ret) / T
    std_ann = np.std(log_ret) / np.sqrt(T)
    return mean_ann / std_ann if std_ann > 1e-10 else 0.0


# ====================================================================
# Section 5: Experiment
# ====================================================================

def run_single_experiment(p: HestonParams,
                          n_train_paths: int = 50,
                          n_train_steps: int = 500,
                          n_test_paths: int = 100,
                          n_test_steps: int = 252,
                          macro_seed: int = 0) -> Dict[str, Dict]:
    """Run one complete experiment for a given macro seed."""

    base = macro_seed * 1000

    # --- Training phase ---
    all_train_returns = []
    all_train_V = []
    for i in range(n_train_paths):
        z1, z2 = generate_shared_noise(n_train_steps, seed=base + i)
        V, rets = simulate_heston(p, z1, z2)
        all_train_returns.append(rets)
        all_train_V.append(V)

    train_returns_cat = np.concatenate(all_train_returns)
    train_V_cat = np.concatenate(all_train_V)

    # Build filters
    oracle = OracleFilter(p)
    bpf = BPFFilter(p, n_particles=1000)
    ewma = EWMAFilter(p, lam=0.94)
    rqv = RealizedQVFilter(p, window=20)
    sig_filter = SigRidgeFilter(p, window=20, sig_level=3, alpha=1.0) if HAS_SIGNATORY else None

    # Train (only SigRidge needs training data)
    sig_r2 = None
    if sig_filter is not None:
        sig_r2 = sig_filter.train(train_returns_cat, train_V_cat)

    # Build controllers
    myopic = MyopicController(p)
    hedging = HedgingController(p)
    constant = ConstantController(p)

    # Define strategies: (name, filter, controller)
    strategies = [
        ("Oracle", oracle, myopic),
        ("Oracle+Hedge", oracle, hedging),
        ("BPF+Myopic", bpf, myopic),
        ("EWMA+Myopic", ewma, myopic),
        ("RQV+Myopic", rqv, myopic),
        ("Constant", oracle, constant),  # oracle unused, constant ignores V_hat
    ]
    if sig_filter is not None:
        strategies.insert(5, ("SigRidge+Myopic", sig_filter, myopic))

    # --- Testing phase ---
    results = {name: [] for name, _, _ in strategies}

    for j in range(n_test_paths):
        test_seed = base + 500 + j
        z1, z2 = generate_shared_noise(n_test_steps, seed=test_seed)
        V_true, returns = simulate_heston(p, z1, z2)

        for name, filt, ctrl in strategies:
            V_hat = filt.filter(returns, V_true=V_true)
            pi_seq = ctrl.allocate(V_hat)
            W = simulate_wealth(returns, pi_seq, p.r, p.dt)
            results[name].append(W[-1])

    # Compute metrics per strategy
    metrics = {}
    T = n_test_steps * p.dt
    for name in results:
        Ws = np.array(results[name])
        metrics[name] = {
            'CE': compute_CE(Ws, p.gamma),
            'sharpe': compute_sharpe(Ws, T),
            'mean_W': np.mean(Ws),
            'std_W': np.std(Ws),
        }
    metrics['_sig_r2'] = sig_r2

    return metrics


def main():
    rho_list = [0.0, -0.3, -0.5, -0.7]
    n_macro = 5

    print("=" * 80)
    print("HONEST BENCHMARK: Merton-Heston POMDP")
    print("=" * 80)

    base_p = HestonParams()
    print(f"Parameters: mu={base_p.mu}, r={base_p.r}, gamma={base_p.gamma}, "
          f"kappa={base_p.kappa}, theta={base_p.theta}, xi={base_p.xi}")
    print(f"Train: 50 paths x 500 steps | Test: 100 paths x 252 steps")
    print(f"Macro seeds: {n_macro} | Strategies: 7")
    print()

    # Collect all results: rho -> list of metric dicts (one per macro seed)
    all_results = {}

    for rho in rho_list:
        p = HestonParams(rho=rho)
        hedge_theory = analytical_hedging_demand(p)
        print(f"--- rho = {rho:.1f}  (hedging demand = {hedge_theory:+.4f}) ---")

        macro_results = []
        sig_r2s = []
        t0 = time.time()

        for ms in range(n_macro):
            m = run_single_experiment(p, macro_seed=ms)
            macro_results.append(m)
            if m.get('_sig_r2') is not None:
                sig_r2s.append(m['_sig_r2'])

        elapsed = time.time() - t0

        # Get strategy names from first result
        strategy_names = [k for k in macro_results[0] if not k.startswith('_')]

        # Aggregate CE across macro seeds
        print(f"{'Strategy':<20s}  {'CE (mean +/- std)':<22s}  {'Sharpe':>8s}  {'Mean W':>8s}")
        print("-" * 65)

        rho_summary = {}
        for name in strategy_names:
            ces = [m[name]['CE'] for m in macro_results]
            sharpes = [m[name]['sharpe'] for m in macro_results]
            mean_ws = [m[name]['mean_W'] for m in macro_results]
            ce_mean, ce_std = np.mean(ces), np.std(ces)
            sh_mean = np.mean(sharpes)
            w_mean = np.mean(mean_ws)
            print(f"{name:<20s}  {ce_mean:.4f} +/- {ce_std:.4f}     {sh_mean:>8.3f}  {w_mean:>8.4f}")
            rho_summary[name] = {'ce_mean': ce_mean, 'ce_std': ce_std,
                                 'sharpe': sh_mean, 'mean_W': w_mean}

        if sig_r2s:
            print(f"  [SigRidge held-out R^2 = {np.mean(sig_r2s):.3f} +/- {np.std(sig_r2s):.3f}]")

        print(f"  [{elapsed:.1f}s elapsed]")
        print()

        all_results[rho] = rho_summary

    # --- Summary table ---
    print("=" * 80)
    print("SUMMARY: CE by Strategy x Rho")
    print("=" * 80)
    strategy_names = list(next(iter(all_results.values())).keys())
    header = f"{'Strategy':<20s}" + "".join(f"  rho={r:+.1f}" for r in rho_list)
    print(header)
    print("-" * (20 + 12 * len(rho_list)))
    for name in strategy_names:
        row = f"{name:<20s}"
        for rho in rho_list:
            ce = all_results[rho][name]['ce_mean']
            row += f"  {ce:>8.4f}  "
        print(row)

    # --- Plot ---
    fig, axes = plt.subplots(1, len(rho_list), figsize=(4 * len(rho_list), 5),
                             sharey=True)
    if len(rho_list) == 1:
        axes = [axes]

    colors = {
        'Oracle': '#1f77b4',
        'Oracle+Hedge': '#2ca02c',
        'BPF+Myopic': '#ff7f0e',
        'EWMA+Myopic': '#d62728',
        'RQV+Myopic': '#9467bd',
        'SigRidge+Myopic': '#e377c2',
        'Constant': '#7f7f7f',
    }

    for ax, rho in zip(axes, rho_list):
        summary = all_results[rho]
        names = list(summary.keys())
        ces = [summary[n]['ce_mean'] for n in names]
        stds = [summary[n]['ce_std'] for n in names]
        cols = [colors.get(n, '#333333') for n in names]

        bars = ax.bar(range(len(names)), ces, yerr=stds, capsize=3,
                      color=cols, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('+', '\n+') for n in names],
                           rotation=45, ha='right', fontsize=7)
        ax.set_title(f"rho = {rho:.1f}", fontsize=10)
        ax.set_ylabel("Certainty Equivalent" if rho == rho_list[0] else "")
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Honest Benchmark: Merton-Heston POMDP", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = "kronic_pomdp/experiments/honest_benchmark.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n[Plot saved to {out_path}]")


if __name__ == "__main__":
    main()
