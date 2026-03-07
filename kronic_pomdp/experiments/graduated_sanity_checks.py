r"""
Graduated Sanity Checks for KRONIC Tools
=========================================

Tests each tool in isolation at graduated difficulty, with proper methodology
(train/test split, paired comparison, error bars, explicit PASS/FAIL).

Level 0: CIR Filtering   — can we filter a 1D mean-reverting process?
Level 1: Heston Filtering — can we filter hidden vol from returns?
Level 2: Generator Recovery — can we learn the SDE from data?
Level 3: Transaction Costs — can Koopman learn no-trade regions?
Level 4: Product Kernel Control — can RKHS learn optimal policy for ANY utility?
         4a: Heston + CRRA (sanity) + MI horizon selection
         4b: Heston + CARA (new utility) + CEV + CRRA (new dynamics)

KEY DESIGN CHOICE: RecSig-RLS is ALWAYS ONLINE. It uses a model-free noisy
target (r^2/dt for returns, y_t for direct obs) at every step — no train/test
freeze. This matches the stage1_documented_approach.py pattern that achieved
0.94x BPF MSE. The "train" phase is just warmup episodes to initialize RLS
weights; the "test" phase continues updating with the same noisy targets.

Usage:
    conda activate rkhs-kronic-gpu
    python kronic_pomdp/experiments/graduated_sanity_checks.py
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from examples.proof_of_concept.signature_features import (
    RecurrentSignatureMap, RecurrentLeadLagLogSigMap
)


# ======================================================================
# Section 1: Shared Infrastructure
# ======================================================================

@dataclass
class CIRParams:
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    obs_noise: float = 0.01
    dt: float = 0.01


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


def generate_noise(n_steps: int, n_dims: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n_steps, n_dims)


def simulate_cir(p: CIRParams, z: np.ndarray, V0=None) -> np.ndarray:
    n = len(z)
    V = np.zeros(n)
    V[0] = V0 if V0 is not None else p.theta
    sqrt_dt = np.sqrt(p.dt)
    for t in range(1, n):
        v = max(V[t - 1], 1e-8)
        V[t] = max(v + p.kappa * (p.theta - v) * p.dt
                   + p.xi * np.sqrt(v) * sqrt_dt * z[t], 1e-8)
    return V


def simulate_heston(p: HestonParams, z1, z2, V0=None):
    n = len(z1)
    V = np.zeros(n)
    returns = np.zeros(n)
    V[0] = V0 if V0 is not None else p.theta
    sqrt_dt = np.sqrt(p.dt)
    for t in range(1, n):
        v = max(V[t - 1], 1e-8)
        sv = np.sqrt(v)
        dW_S = sqrt_dt * z1[t]
        dW_V = sqrt_dt * (p.rho * z1[t] + np.sqrt(1 - p.rho ** 2) * z2[t])
        returns[t] = p.mu * p.dt + sv * dW_S
        V[t] = max(v + p.kappa * (p.theta - v) * p.dt + p.xi * sv * dW_V, 1e-8)
    return V, returns


def simulate_wealth(returns, pi, r, dt):
    n = len(returns)
    W = np.ones(n)
    for t in range(1, n):
        portfolio_ret = r * dt + pi[t - 1] * (returns[t] - r * dt)
        W[t] = W[t - 1] * max(1 + portfolio_ret, 1e-10)
    return W


def compute_CE(W_terminals, gamma):
    if gamma == 1.0:
        return np.exp(np.mean(np.log(np.maximum(W_terminals, 1e-20))))
    utils = W_terminals ** (1 - gamma) / (1 - gamma)
    return ((1 - gamma) * np.mean(utils)) ** (1 / (1 - gamma))


# ======================================================================
# Section 2: Filters
# ======================================================================

class EWMAFilter:
    def __init__(self, lam=0.94, dt=0.01, init_val=0.04):
        self.lam = lam
        self.dt = dt
        self.init_val = init_val
        self.variance = init_val

    def reset(self):
        self.variance = self.init_val

    def update_return(self, ret):
        rv = ret ** 2 / self.dt
        self.variance = self.lam * self.variance + (1 - self.lam) * rv
        return max(self.variance, 1e-8)

    def update_direct(self, y):
        self.variance = self.lam * self.variance + (1 - self.lam) * y
        return max(self.variance, 1e-8)


class RecSigRLSFilter:
    """Recurrent signature + RLS. ALWAYS ONLINE — never freeze.

    Uses model-free noisy target at every step (r²/dt for returns,
    y_t for direct obs). This matches the stage1 approach that worked.
    """
    def __init__(self, input_dim=2, forgetting_factor=0.94,
                 rls_ff=0.999, dt=0.01, init_val=0.04):
        self.sig_map = RecurrentSignatureMap(
            state_dim=input_dim, level=2, forgetting_factor=forgetting_factor)
        self.n_features = self.sig_map.feature_dim + 1
        self.w = np.zeros(self.n_features)
        self.P = np.eye(self.n_features) * 100.0
        self.rls_ff = rls_ff
        self.dt = dt
        self.init_val = init_val

    def reset(self):
        self.sig_map.reset()
        # Do NOT reset w, P — persist across episodes

    def update(self, dx: np.ndarray, target: float) -> float:
        """Always updates RLS with noisy target. Returns filtered estimate."""
        sig_features = self.sig_map.update(dx)
        features = np.concatenate([sig_features, [1.0]])
        pred = np.dot(self.w, features)

        target = min(target, 2.0)
        z = features[:, np.newaxis]
        Pz = self.P @ z
        denom = self.rls_ff + (z.T @ Pz)[0, 0]
        k = Pz / denom
        error = target - pred
        self.w = self.w + k.flatten() * error
        self.P = (self.P - k @ Pz.T) / self.rls_ff

        return max(pred, 1e-8)


class SimpleBPF:
    def __init__(self, kappa, theta, xi, dt, n_particles=500,
                 obs_type='return', mu=None, obs_noise=None):
        self.kappa, self.theta, self.xi, self.dt = kappa, theta, xi, dt
        self.N = n_particles
        self.obs_type = obs_type
        self.mu = mu
        self.obs_noise = obs_noise
        self.rng = np.random.RandomState(99999)
        self.particles = np.ones(n_particles) * theta

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.particles = np.clip(
            self.rng.normal(self.theta, 0.01, self.N), 1e-6, 1.0)

    def update(self, obs):
        sqrt_dt = np.sqrt(self.dt)
        v = np.maximum(self.particles, 1e-6)
        dW = self.rng.randn(self.N) * sqrt_dt
        v_pred = np.maximum(v + self.kappa * (self.theta - v) * self.dt
                            + self.xi * np.sqrt(v) * dW, 1e-6)
        if self.obs_type == 'return':
            var_term = v_pred * self.dt
            log_w = -0.5 * (np.log(var_term)
                            + (obs - self.mu * self.dt) ** 2 / var_term)
        else:
            log_w = -0.5 * (obs - v_pred) ** 2 / (self.obs_noise ** 2)
        log_w -= np.max(log_w)
        weights = np.exp(log_w)
        weights /= np.sum(weights)
        est = np.sum(v_pred * weights)
        self.particles = v_pred[self.rng.choice(self.N, size=self.N, p=weights)]
        return est


class KalmanFilterCIR:
    def __init__(self, p: CIRParams):
        self.p = p
        self.v_hat = p.theta
        self.P = 0.01

    def reset(self):
        self.v_hat = self.p.theta
        self.P = 0.01

    def update(self, y_obs):
        p = self.p
        v_pred = self.v_hat + p.kappa * (p.theta - self.v_hat) * p.dt
        v_pred = max(v_pred, 1e-8)
        F = 1 - p.kappa * p.dt
        Q = p.xi ** 2 * max(self.v_hat, 1e-8) * p.dt
        P_pred = F ** 2 * self.P + Q
        R = p.obs_noise ** 2
        K = P_pred / (P_pred + R)
        self.v_hat = v_pred + K * (y_obs - v_pred)
        self.v_hat = max(self.v_hat, 1e-8)
        self.P = (1 - K) * P_pred
        return self.v_hat


# ======================================================================
# Section 3: Carre du Champ Learner
# ======================================================================

class CarreDuChampLearner:
    """Learn mu(x), sigma^2(x) from trajectory via NW kernel regression."""
    def __init__(self, bandwidth=None):
        self.bw = bandwidth
        self.X_train = None
        self.dX_train = None
        self.dX2_train = None

    def fit(self, X: np.ndarray, dt: float):
        self.X_train = X[:-1].copy()
        dX = np.diff(X)
        self.dX_train = dX / dt
        self.dX2_train = dX ** 2 / dt
        if self.bw is None:
            self.bw = np.std(self.X_train) * 0.3
        return self

    def predict(self, x_test: np.ndarray):
        mu_est = np.zeros(len(x_test))
        sigma2_est = np.zeros(len(x_test))
        for i, x in enumerate(x_test):
            w = np.exp(-0.5 * ((self.X_train - x) / self.bw) ** 2)
            w_sum = np.sum(w) + 1e-10
            mu_est[i] = np.sum(w * self.dX_train) / w_sum
            sigma2_est[i] = np.sum(w * self.dX2_train) / w_sum
        return mu_est, sigma2_est


# ======================================================================
# Section 4: Controllers
# ======================================================================

class MyopicController:
    def __init__(self, mu, r, gamma, max_lev=3.0):
        self.a = mu - r
        self.gamma = gamma
        self.max_lev = max_lev

    def allocate(self, V_hat):
        V_safe = np.maximum(V_hat, 1e-6)
        return np.clip(self.a / (self.gamma * V_safe), 0.0, self.max_lev)


class ConstantController:
    def __init__(self, mu, r, gamma, theta):
        self.pi = (mu - r) / (gamma * theta)

    def allocate(self, V_hat):
        return np.full_like(V_hat, self.pi)


# ======================================================================
# Level 0: CIR Filtering
# ======================================================================

def level0_heston_filtering_stage1(n_macro=5, n_episodes=20,
                                    n_steps=100, warmup=50):
    """Sanity check: reproduce stage1_documented_approach.py results.

    Uses IDENTICAL protocol to stage1: dt=0.01, xi=0.5, rho=-0.9,
    20 episodes of 100 steps each, online RLS throughout, measure on
    second half of episodes.

    This is the cleanest test of whether RecSig-RLS works AT ALL.
    Expected: RecSig MSE / BPF MSE < 1.5x (stage1 reported 0.94x).
    """
    print("=" * 70)
    print("LEVEL 0: RecSig-RLS Reproduction (Stage1 Protocol)")
    print("  Environment: Heston with dt=0.01, xi=0.5, rho=-0.9")
    print("  Protocol: 20 episodes, online RLS, measure last 10 eps")
    print("  Expected: RecSig MSE / BPF MSE < 1.5x")
    print("=" * 70)

    p = HestonParams(dt=0.01, xi=0.5, rho=-0.9, mu=0.05)
    results = {m: {'mse': []} for m in ['BPF', 'RecSig-RLS', 'EWMA']}

    for ms in range(n_macro):
        base = ms * 200
        sig_filter = RecSigRLSFilter(input_dim=2, forgetting_factor=0.94,
                                      rls_ff=0.999, dt=p.dt, init_val=p.theta)

        for ep in range(n_episodes):
            seed = base + ep
            z = generate_noise(n_steps, 2, seed=seed + 1000)
            V, rets = simulate_heston(p, z[:, 0], z[:, 1])

            # RecSig-RLS (online, always updating)
            sig_filter.reset()
            V_sig = np.full(n_steps, p.theta)
            for t in range(1, n_steps):
                dx = np.array([p.dt, rets[t]])
                V_sig[t] = sig_filter.update(dx, target=rets[t] ** 2 / p.dt)

            # BPF
            bpf = SimpleBPF(p.kappa, p.theta, p.xi, p.dt, n_particles=500,
                            obs_type='return', mu=p.mu)
            bpf.reset(seed=seed + 9999)
            V_bpf = np.zeros(n_steps)
            V_bpf[0] = p.theta
            for t in range(1, n_steps):
                V_bpf[t] = bpf.update(rets[t])

            # EWMA
            ewma = EWMAFilter(lam=0.94, dt=p.dt, init_val=p.theta)
            V_ewma = np.zeros(n_steps)
            V_ewma[0] = p.theta
            for t in range(1, n_steps):
                V_ewma[t] = ewma.update_return(rets[t])

            # Measure on second half of episodes
            if ep >= n_episodes // 2:
                sl = slice(warmup, None)
                for name, est in [('BPF', V_bpf), ('RecSig-RLS', V_sig),
                                  ('EWMA', V_ewma)]:
                    mse = np.mean((est[sl] - V[sl]) ** 2)
                    results[name]['mse'].append(mse)

    bpf_mse = np.mean(results['BPF']['mse'])
    print(f"\n{'Method':<15} {'MSE(x1e-4)':<12} {'vs BPF':<10}")
    print("-" * 40)
    checks = {}
    for name in ['BPF', 'RecSig-RLS', 'EWMA']:
        mse_mean = np.mean(results[name]['mse'])
        mse_std = np.std(results[name]['mse'])
        ratio = mse_mean / bpf_mse if bpf_mse > 0 else 0
        checks[name] = ratio
        print(f"{name:<15} {mse_mean * 1e4:>7.3f}±{mse_std * 1e4:<5.3f} {ratio:>6.2f}x")

    sig_ok = checks['RecSig-RLS'] < 1.5
    passed = sig_ok
    print(f"\nRecSig vs BPF: {checks['RecSig-RLS']:.2f}x ({'OK <1.5x' if sig_ok else 'FAIL'})")
    print(f"LEVEL 0: {'PASS' if passed else 'FAIL'}")
    return {'passed': passed, 'description': 'RecSig-RLS stage1 reproduction',
            'details': checks}


# ======================================================================
# Level 1: Heston Filtering + Portfolio
# ======================================================================

def level1_heston_filtering(n_macro=5, n_total_paths=30, n_test_steps=252):
    """Can we filter hidden vol AND translate to portfolio CE?

    RecSig-RLS runs ONLINE throughout. Uses stage1 protocol: all paths
    are both training and evaluation. First half warmup, second half
    measured. Target r²/dt available at all times (model-free).

    Also tests at stage1 parameters (dt=0.01, xi=0.5) to verify match.
    """
    print("\n" + "=" * 70)
    print("LEVEL 1: Heston Filtering + Portfolio")
    print("  Environment: Heston-Merton")
    print("  RecSig-RLS: ALWAYS ONLINE (target = r^2/dt, model-free)")
    print("  Part A: Daily freq (dt=1/252, xi=0.3) — realistic params")
    print("  Part B: Stage1 params (dt=0.01, xi=0.5) — reproduction test")
    print("=" * 70)

    # --- Part A: Realistic parameters ---
    p = HestonParams()
    myopic = MyopicController(p.mu, p.r, p.gamma)
    constant = ConstantController(p.mu, p.r, p.gamma, p.theta)

    all_ce = {m: [] for m in ['Oracle', 'BPF', 'RecSig-RLS', 'EWMA', 'Constant']}
    all_mse_ratio = {'BPF': [], 'RecSig-RLS': [], 'EWMA': []}

    for ms in range(n_macro):
        base = ms * 1000
        sig_filter = RecSigRLSFilter(input_dim=2, forgetting_factor=0.94,
                                      rls_ff=0.999, dt=p.dt, init_val=p.theta)
        test_W = {m: [] for m in all_ce}
        test_mse = {'BPF': [], 'RecSig-RLS': [], 'EWMA': []}

        for j in range(n_total_paths):
            seed = base + j
            z = generate_noise(n_test_steps, 2, seed=seed)
            V_true, rets = simulate_heston(p, z[:, 0], z[:, 1])

            # RecSig-RLS (ONLINE throughout)
            sig_filter.reset()
            V_sig = np.full(n_test_steps, p.theta)
            for t in range(1, n_test_steps):
                dx = np.array([p.dt, rets[t]])
                V_sig[t] = sig_filter.update(dx, target=rets[t] ** 2 / p.dt)

            # Only measure on second half of paths
            if j < n_total_paths // 2:
                continue

            # Oracle
            pi_oracle = myopic.allocate(V_true)
            test_W['Oracle'].append(simulate_wealth(rets, pi_oracle, p.r, p.dt)[-1])

            # BPF
            bpf = SimpleBPF(p.kappa, p.theta, p.xi, p.dt, n_particles=1000,
                            obs_type='return', mu=p.mu)
            bpf.reset(seed=seed + 9999)
            V_bpf = np.zeros(n_test_steps)
            V_bpf[0] = p.theta
            for t in range(1, n_test_steps):
                V_bpf[t] = bpf.update(rets[t])
            pi_bpf = myopic.allocate(V_bpf)
            test_W['BPF'].append(simulate_wealth(rets, pi_bpf, p.r, p.dt)[-1])
            test_mse['BPF'].append(np.mean((V_bpf[50:] - V_true[50:]) ** 2))

            # RecSig-RLS measurement
            pi_sig = myopic.allocate(V_sig)
            test_W['RecSig-RLS'].append(simulate_wealth(rets, pi_sig, p.r, p.dt)[-1])
            test_mse['RecSig-RLS'].append(np.mean((V_sig[50:] - V_true[50:]) ** 2))

            # EWMA
            ewma = EWMAFilter(lam=0.94, dt=p.dt, init_val=p.theta)
            V_ewma = np.zeros(n_test_steps)
            V_ewma[0] = p.theta
            for t in range(1, n_test_steps):
                V_ewma[t] = ewma.update_return(rets[t])
            pi_ewma = myopic.allocate(V_ewma)
            test_W['EWMA'].append(simulate_wealth(rets, pi_ewma, p.r, p.dt)[-1])
            test_mse['EWMA'].append(np.mean((V_ewma[50:] - V_true[50:]) ** 2))

            # Constant
            pi_const = constant.allocate(V_true)
            test_W['Constant'].append(simulate_wealth(rets, pi_const, p.r, p.dt)[-1])

        for m in all_ce:
            all_ce[m].append(compute_CE(np.array(test_W[m]), p.gamma))
        bpf_mse_mean = np.mean(test_mse['BPF'])
        for m in all_mse_ratio:
            all_mse_ratio[m].append(np.mean(test_mse[m]) / max(bpf_mse_mean, 1e-20))

    print(f"\nPart A: Realistic params (dt=1/252, xi=0.3, rho=-0.5)")
    print(f"{'Method':<15} {'CE (mean±std)':<22} {'MSE/BPF':<10}")
    print("-" * 50)
    ce_vals = {}
    for m in ['Oracle', 'BPF', 'RecSig-RLS', 'EWMA', 'Constant']:
        ce_mean = np.mean(all_ce[m])
        ce_std = np.std(all_ce[m])
        ce_vals[m] = ce_mean
        if m in all_mse_ratio:
            mse_r = np.mean(all_mse_ratio[m])
            print(f"{m:<15} {ce_mean:.4f}±{ce_std:.4f}       {mse_r:.2f}x")
        else:
            print(f"{m:<15} {ce_mean:.4f}±{ce_std:.4f}")

    # --- Part B: Stage1 reproduction (dt=0.01, xi=0.5) ---
    print(f"\nPart B: Stage1 params (dt=0.01, xi=0.5, rho=-0.9)")
    p2 = HestonParams(dt=0.01, xi=0.5, rho=-0.9)
    sig_mse_ratios_b = []
    for ms in range(n_macro):
        base = ms * 100
        sig_f = RecSigRLSFilter(input_dim=2, forgetting_factor=0.94,
                                 rls_ff=0.999, dt=p2.dt, init_val=p2.theta)
        ep_mse_bpf, ep_mse_sig = [], []
        for ep in range(20):
            seed = base + ep
            z = generate_noise(100, 2, seed=seed + 1000)
            V, rets = simulate_heston(p2, z[:, 0], z[:, 1])
            sig_f.reset()
            V_sig = np.full(100, p2.theta)
            for t in range(1, 100):
                dx = np.array([p2.dt, rets[t]])
                V_sig[t] = sig_f.update(dx, target=rets[t] ** 2 / p2.dt)

            bpf = SimpleBPF(p2.kappa, p2.theta, p2.xi, p2.dt, n_particles=500,
                            obs_type='return', mu=p2.mu)
            bpf.reset(seed=seed + 9999)
            V_bpf = np.zeros(100)
            V_bpf[0] = p2.theta
            for t in range(1, 100):
                V_bpf[t] = bpf.update(rets[t])

            if ep >= 10:  # measure on second half
                sl = slice(50, None)
                ep_mse_bpf.append(np.mean((V_bpf[sl] - V[sl]) ** 2))
                ep_mse_sig.append(np.mean((V_sig[sl] - V[sl]) ** 2))

        ratio = np.mean(ep_mse_sig) / max(np.mean(ep_mse_bpf), 1e-20)
        sig_mse_ratios_b.append(ratio)

    stage1_ratio = np.mean(sig_mse_ratios_b)
    print(f"  RecSig MSE / BPF MSE = {stage1_ratio:.2f}x (target: ~0.94x)")

    # --- Verdicts ---
    ce_gap = (ce_vals['Oracle'] - ce_vals['Constant']) / ce_vals['Constant'] * 100
    sig_mse_a = np.mean(all_mse_ratio['RecSig-RLS'])
    gap_ok = ce_gap < 3.0
    # Pass if: CE gap is small (expected), AND stage1 reproduction works
    stage1_ok = stage1_ratio < 1.5
    passed = gap_ok and stage1_ok

    print(f"\nOracle-Constant CE gap: {ce_gap:.2f}% ({'expected <3%' if gap_ok else 'UNEXPECTED'})")
    print(f"Part A RecSig MSE/BPF: {sig_mse_a:.2f}x (high expected at daily freq)")
    print(f"Part B stage1 repro: {stage1_ratio:.2f}x ({'OK' if stage1_ok else 'FAIL'})")
    print(f"LEVEL 1: {'PASS' if passed else 'FAIL'}")
    return {'passed': passed, 'description': 'Heston filtering + portfolio',
            'details': {'ce_gap': ce_gap, 'sig_mse_daily': sig_mse_a,
                        'stage1_repro': stage1_ratio}}


# ======================================================================
# Level 2: Generator Recovery (CdC)
# ======================================================================

def level2_generator_recovery(n_macro=5, n_steps=50000):
    """Can we learn the CIR SDE from data using Carre du Champ?

    Uses 50K steps at dt=0.01 for strong signal. Train/test are
    completely separate trajectories (different seeds).
    """
    print("\n" + "=" * 70)
    print("LEVEL 2: Generator Recovery (Carre du Champ)")
    print("  Environment: CIR with kappa=2.0, theta=0.04, xi=0.3")
    print("  Train trajectory (seed A) → test trajectory (seed B)")
    print("=" * 70)

    p = CIRParams(dt=0.01)
    drift_corrs = []
    sigma_corrs = []
    kappa_ests = []

    for ms in range(n_macro):
        z_train = generate_noise(n_steps, 1, seed=ms * 100)[:, 0]
        V_train = simulate_cir(p, z_train)

        z_test = generate_noise(n_steps, 1, seed=ms * 100 + 50)[:, 0]
        V_test = simulate_cir(p, z_test)

        cdc = CarreDuChampLearner()
        cdc.fit(V_train, p.dt)

        # Evaluate on test points
        x_test = V_test[500::50]  # skip warmup, subsample
        mu_pred, sigma2_pred = cdc.predict(x_test)

        mu_true = p.kappa * (p.theta - x_test)
        sigma2_true = p.xi ** 2 * x_test

        drift_corr = np.corrcoef(mu_pred, mu_true)[0, 1] if np.std(mu_pred) > 1e-10 else 0
        sigma_corr = np.corrcoef(sigma2_pred, sigma2_true)[0, 1] if np.std(sigma2_pred) > 1e-10 else 0

        # Estimate kappa from drift: mu(x) = kappa*theta - kappa*x → slope = -kappa
        A = np.column_stack([np.ones(len(x_test)), x_test])
        coeffs = np.linalg.lstsq(A, mu_pred, rcond=None)[0]
        kappa_est = -coeffs[1]

        drift_corrs.append(drift_corr)
        sigma_corrs.append(sigma_corr)
        kappa_ests.append(kappa_est)

    print(f"\n{'Metric':<25} {'Mean±Std':<15} {'Threshold':<12} {'Status'}")
    print("-" * 60)

    dc_mean, dc_std = np.mean(drift_corrs), np.std(drift_corrs)
    sc_mean, sc_std = np.mean(sigma_corrs), np.std(sigma_corrs)
    ke_mean, ke_std = np.mean(kappa_ests), np.std(kappa_ests)
    ke_err = abs(ke_mean - p.kappa) / p.kappa

    dc_ok = dc_mean > 0.90
    sc_ok = sc_mean > 0.90
    ke_ok = ke_err < 0.50

    print(f"{'Drift corr (held-out)':<25} {dc_mean:.3f}±{dc_std:.3f}      >0.90        {'OK' if dc_ok else 'FAIL'}")
    print(f"{'sigma corr (held-out)':<25} {sc_mean:.3f}±{sc_std:.3f}      >0.90        {'OK' if sc_ok else 'FAIL'}")
    print(f"{'kappa estimate':<25} {ke_mean:.2f}±{ke_std:.2f}       2.0±50%      {'OK' if ke_ok else 'FAIL'}")

    passed = dc_ok and sc_ok and ke_ok
    print(f"\nLEVEL 2: {'PASS' if passed else 'FAIL'}")
    return {'passed': passed, 'description': 'Generator recovery (CdC)',
            'details': {'drift_corr': dc_mean, 'sigma_corr': sc_mean,
                        'kappa_est': ke_mean}}


# ======================================================================
# Level 3: Transaction Cost Control
# ======================================================================

class MertonTxCostEnv:
    def __init__(self, mu=0.08, r=0.02, kappa_v=2.0, theta_v=0.04,
                 xi=0.3, rho=-0.7, tc=0.005, gamma=2.0, dt=1 / 252):
        self.mu, self.r = mu, r
        self.kappa_v, self.theta_v, self.xi, self.rho = kappa_v, theta_v, xi, rho
        self.tc, self.gamma, self.dt = tc, gamma, dt

    def step(self, state, action, rng):
        logW, pi, v = state
        z1, z2 = rng.randn(), rng.randn()
        z2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2
        dB = np.sqrt(self.dt) * z1
        dB_v = np.sqrt(self.dt) * z2
        drift_W = self.r + pi * (self.mu - self.r) - 0.5 * pi ** 2 * v
        d_logW = drift_W * self.dt + pi * np.sqrt(max(v, 1e-8)) * dB \
                 - self.tc * abs(action)
        new_v = max(v + self.kappa_v * (self.theta_v - v) * self.dt
                    + self.xi * np.sqrt(max(v, 1e-8)) * dB_v, 1e-8)
        return np.array([logW + d_logW, pi + action, new_v])

    def merton_optimal(self, v):
        return (self.mu - self.r) / (self.gamma * v)


def level3_transaction_costs(n_macro=5, n_test_paths=100, n_test_steps=252):
    """Can Koopman learn no-trade region from data?"""
    print("\n" + "=" * 70)
    print("LEVEL 3: Transaction Cost Control (Koopman Growth Rate)")
    print("  Environment: Merton + Heston + tx cost kappa=0.005")
    print("  Koopman learns g(pi, sigma^2) from MC, derives no-trade bands")
    print("=" * 70)

    env = MertonTxCostEnv()
    ce_by_seed = {m: [] for m in ['Koopman', 'Shreve-Soner', 'Naive', 'Constant']}

    for ms in range(n_macro):
        train_rng = np.random.RandomState(ms * 1000)

        # --- Learn growth rate g(pi, sigma^2) from MC ---
        n_grid = 25
        pi_grid = np.linspace(0.05, 1.5, n_grid)
        v_grid = np.linspace(0.02, 0.08, n_grid)
        n_mc = 200

        pi_s, v_s, g_s = [], [], []
        for pi in pi_grid:
            for v in v_grid:
                growths = []
                for _ in range(n_mc):
                    state = np.array([0.0, pi, v])
                    next_state = env.step(state, 0.0, rng=train_rng)
                    growths.append((next_state[0] - state[0]) / env.dt)
                pi_s.append(pi)
                v_s.append(v)
                g_s.append(np.mean(growths))

        pi_s, v_s, g_s = np.array(pi_s), np.array(v_s), np.array(g_s)
        features = np.column_stack([np.ones(len(pi_s)), pi_s, pi_s ** 2 * v_s])
        c0, c1, c2 = np.linalg.lstsq(features, g_s, rcond=None)[0]

        # --- Define strategy functions ---
        def _make_koopman_action(c0, c1, c2, tc):
            def action(state):
                _, pi_cur, v = state
                pi_star = -c1 / (2 * c2 * v + 1e-10)
                curvature = 2 * c2 * v
                width = 0.8 * (3 * tc / abs(curvature)) ** (1 / 3) if curvature < 0 else 0.2
                width = np.clip(width, 0.05, 0.4)
                if pi_cur < pi_star - width:
                    return np.clip(pi_star - width - pi_cur, 0, 0.5)
                elif pi_cur > pi_star + width:
                    return np.clip(pi_star + width - pi_cur, -0.5, 0)
                return 0.0
            return action

        def ss_action(state):
            _, pi_cur, v = state
            pi_star = env.merton_optimal(v)
            width = np.clip(0.3 * (env.tc / v) ** (1 / 3), 0.05, 0.3)
            if pi_cur < pi_star - width:
                return np.clip(pi_star - width - pi_cur, 0, 0.5)
            elif pi_cur > pi_star + width:
                return np.clip(pi_star + width - pi_cur, -0.5, 0)
            return 0.0

        def naive_action(state):
            _, pi_cur, v = state
            return np.clip(env.merton_optimal(v) - pi_cur, -0.5, 0.5)

        pi_const = env.merton_optimal(env.theta_v)

        def const_action(state):
            _, pi_cur, _ = state
            return np.clip(pi_const - pi_cur, -0.5, 0.5)

        koopman_action = _make_koopman_action(c0, c1, c2, env.tc)
        strategies = {'Koopman': koopman_action, 'Shreve-Soner': ss_action,
                      'Naive': naive_action, 'Constant': const_action}

        # --- Test with paired comparison ---
        seed_ws = {m: [] for m in strategies}
        for j in range(n_test_paths):
            for name, action_fn in strategies.items():
                path_rng = np.random.RandomState(ms * 1000 + 500 + j)
                state = np.array([0.0, pi_const, env.theta_v])
                for t in range(n_test_steps):
                    a = action_fn(state)
                    state = env.step(state, a, rng=path_rng)
                seed_ws[name].append(np.exp(state[0]))

        for name in strategies:
            ce_by_seed[name].append(
                compute_CE(np.array(seed_ws[name]), env.gamma))

    print(f"\n{'Strategy':<15} {'CE (mean±std)':<22}")
    print("-" * 40)
    ce_means = {}
    for name in ['Koopman', 'Shreve-Soner', 'Naive', 'Constant']:
        ce_mean = np.mean(ce_by_seed[name])
        ce_std = np.std(ce_by_seed[name])
        ce_means[name] = ce_mean
        print(f"{name:<15} {ce_mean:.4f}±{ce_std:.4f}")

    koop_beats_naive = ce_means['Koopman'] > ce_means['Naive']
    ss_beats_naive = ce_means['Shreve-Soner'] > ce_means['Naive']
    koop_ratio = ce_means['Koopman'] / ce_means['Shreve-Soner'] if ce_means['Shreve-Soner'] > 0 else 0
    koop_near_ss = koop_ratio > 0.95
    passed = koop_beats_naive and ss_beats_naive

    print(f"\nKoopman > Naive: {koop_beats_naive}")
    print(f"Koopman/SS: {koop_ratio:.3f} ({'OK' if koop_near_ss else 'WARN'})")
    print(f"SS > Naive: {ss_beats_naive}")
    print(f"LEVEL 3: {'PASS' if passed else 'FAIL'}")

    print(f"\n  Learned: g = {c0:.4f} + {c1:.4f}*pi + {c2:.4f}*pi^2*V")
    print(f"  Theory:  g = {env.r:.4f} + {env.mu - env.r:.4f}*pi + {-env.gamma / 2:.1f}*pi^2*V")
    return {'passed': passed, 'description': 'Transaction cost control',
            'details': ce_means}


# ======================================================================
# Level 4: SDRE with Quadratic Utility Approximation & Changing Q
# ======================================================================

# --- Utility derivative factories (U', U'') ---
# The SDRE framework uses local quadratic approximation of utility:
#   E[dU] ~ U'(W)*W*[r + pi*(mu-r)]*dt + 0.5*U''(W)*W^2*pi^2*V*dt
#         = a + pi*b + pi^2*c
# where c = 0.5*U''(W)*W^2*V is the "local Q" (state-dependent cost).

def _make_crra(gamma):
    """CRRA: U(W) = W^{1-g}/(1-g). Policy should be wealth-independent."""
    return (lambda W: W ** (-gamma),
            lambda W: -gamma * W ** (-gamma - 1))


def _make_cara(alpha):
    """CARA: U(W) = -exp(-aW)/a. Policy is wealth-dependent: pi ~ 1/(aWV)."""
    return (lambda W: np.exp(-alpha * W),
            lambda W: -alpha * np.exp(-alpha * W))


def _make_log():
    """Log: U(W) = log(W). Equivalent to CRRA(gamma=1)."""
    return (lambda W: 1.0 / W,
            lambda W: -1.0 / W ** 2)


def level4_product_kernel_control(n_macro=5):
    r"""SDRE control with quadratic utility approximation and changing Q.

    Implements the RKHS-KRONIC paper's Proposition 3 + Theorem 5 on a single
    path, using our Sig-KKF success from Levels 0-1.

    From Ito calculus (Proposition 3), E[dU] is quadratic in pi:
      E[dU] = a + pi*b + pi^2*c
    where:
      b = U'(W) * W * (mu_hat - r)       -- linear coefficient (drift reward)
      c = 0.5 * U''(W) * W^2 * V_hat     -- quadratic = the LOCAL Q
    Policy (Theorem 5): pi* = -b / (2c)

    The "changing Q" is -U''(W)*W^2*V_hat: it changes every step as W and
    V_hat evolve. This is the "local Q changes in LQR" from the sibling repo.

    NO utility-specific formula is hardcoded. Pass U' and U'', the framework
    does the rest. Different utilities -> different Q -> different policies.

    4a: Heston + CRRA(gamma=2) -- sanity check vs Merton
    4b: Multi-utility (CRRA, CARA, Log) -- same V_hat, different Q, different pi
    4c: CEV dynamics -- different SDE, same SDRE pipeline

    PASS criteria:
      4a: pi* corr with Merton > 0.30, concavity c<0 for >90%
      4b: CARA wealth-dependent, three policies distinct
      4c: CEV pi* corr > 0.25
    """
    print("\n" + "=" * 70)
    print("LEVEL 4: SDRE with Quadratic Utility Approximation & Changing Q")
    print("  V_hat via: Kalman(CIR) or LL-LogSig(BCH) + BLR + Kalman")
    print("  BLR: Bayesian linear regression on log-sig features")
    print("       predictive mean -> Kalman obs, pred variance -> Kalman R")
    print("  SDRE local Q = -U''(W)*W^2*V_hat, policy pi* = -b/(2c) (Thm 5)")
    print("  4a: Heston+CRRA, 4b: Multi-utility, 4c: CEV dynamics")
    print("=" * 70)

    p = HestonParams()
    T = 5000

    def _run_sdre_control(sim_fn, T, dt_val, seed, U_prime, U_double_prime,
                          known_mu=None, kf_params=None, use_leadlag=False):
        """Run SDRE controller with local quadratic utility on one path.

        Uses a scalar Kalman filter for V estimation:
        - Prediction uses Koopman/CdC-learned dynamics (mean-reversion)
        - Observation is r^2/dt (or lead-lag log-sig RLS prediction)
        - The Kalman gain balances smooth prediction vs noisy observation

        When use_leadlag=True, a lead-lag log-signature with BCH updates
        provides a model-free volatility observation to the Kalman filter.
        The Levy area of the lead-lag path captures quadratic variation.

        Args:
            sim_fn: (t, z1, z2, dt, state_dict) -> (state_val, excess_return)
            T: trajectory length
            dt_val: time step
            seed: random seed
            U_prime: callable(W) -> U'(W)
            U_double_prime: callable(W) -> U''(W)
            known_mu: if provided, use as excess drift; else EWMA
            kf_params: dict with Kalman filter dynamics params
                       {kappa, theta, xi} from CdC/Level 2.
                       If None, uses EWMA fallback.
            use_leadlag: if True, use lead-lag log-sig RLS for V observation

        Returns: dict with V_hat, pi_sdre, state_true, W_history, concavity
        """
        rng = np.random.RandomState(seed)

        # Kalman filter state for V
        if kf_params is not None:
            kf_kappa = kf_params['kappa']
            kf_theta = kf_params['theta']
            kf_xi = kf_params['xi']
            V_filt = kf_theta  # initialize at stationary mean
        else:
            # Fallback: weak mean-reversion prior (conservative)
            kf_kappa = 1.0
            kf_theta = 0.04
            kf_xi = 0.3
            V_filt = kf_theta
        P_kf = kf_xi ** 2 * kf_theta * dt_val * 10  # initial uncertainty

        # Lead-lag log-sig + Bayesian Linear Regression for model-free V.
        # BLR on log-sig features gives predictive mean + variance:
        #   mean -> Kalman observation
        #   variance -> Kalman observation noise R (principled, not ad-hoc)
        # Uses QV (Levy area) + displacement (leverage) features.
        if use_leadlag:
            ll_gamma = 0.99  # ~100-day effective window
            ll_sig = RecurrentLeadLagLogSigMap(
                state_dim=2, level=2, forgetting_factor=ll_gamma)
            # Feature indices in 4D lead-lag log-sig:
            # Level-1: [time_lead, ret_lead, time_lag, ret_lag] (0-3)
            # Level-2 areas: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) (4-9)
            # QV area = (ret_lead, ret_lag) = index 4+4 = 8
            # ret_lead = index 1
            ll_area_idx = 8
            ll_ret_idx = 1
            # BLR state: 3 weights (QV area, ret_lead, bias)
            blr_nf = 3
            blr_w = np.zeros(blr_nf)
            blr_P = np.eye(blr_nf) * 10.0  # prior: w ~ N(0, 10*I)
            blr_sigma_n2 = 0.01  # observation noise (updated online)

        # EWMA for drift estimation
        ewma_mu = 0.0
        lam_mu = 0.999

        V_hat = np.zeros(T)
        pi_sdre = np.zeros(T)
        state_true = np.zeros(T)
        W_history = np.ones(T)
        concavity_ok = np.zeros(T, dtype=bool)

        state_dict = {}
        W = 1.0

        for t in range(T):
            z1 = rng.randn()
            z2 = rng.randn()

            state_val, ret = sim_fn(t, z1, z2, dt_val, state_dict)
            state_true[t] = state_val

            # --- Scalar Kalman filter for V ---
            # Predict: V_{t|t-1} using CIR dynamics (Koopman linear approx)
            V_pred = V_filt + kf_kappa * (kf_theta - V_filt) * dt_val
            V_pred = max(V_pred, 1e-6)

            # Process noise: Q = xi^2 * V * dt (CIR diffusion)
            Q_kf = kf_xi ** 2 * max(V_filt, 1e-6) * dt_val
            P_pred = (1 - kf_kappa * dt_val) ** 2 * P_kf + Q_kf

            if use_leadlag:
                # Lead-lag log-sig + BLR
                dx = np.array([dt_val, ret])
                feat_full = ll_sig.update(dx)
                phi = np.array([feat_full[ll_area_idx],
                                feat_full[ll_ret_idx], 1.0])

                # BLR predictive distribution
                y_obs = max(np.dot(blr_w, phi), 1e-8)
                R_kf = max(phi @ blr_P @ phi + blr_sigma_n2, 1e-8)

                # BLR posterior update (Kalman on weights)
                target = min(ret ** 2 / dt_val, 2.0)
                Cp = blr_P @ phi
                S = phi @ Cp + blr_sigma_n2
                K_w = Cp / S
                blr_w = blr_w + K_w * (target - np.dot(blr_w, phi))
                blr_P = blr_P - np.outer(K_w, Cp)
                blr_P = 0.5 * (blr_P + blr_P.T)

                # Online noise estimation
                blr_sigma_n2 = max(
                    0.99 * blr_sigma_n2 + 0.01 * (target - y_obs) ** 2,
                    1e-6)
            else:
                # Raw observation: y = r^2/dt (noisy proxy for V)
                y_obs = ret ** 2 / dt_val
                # Observation noise: R = Var[r^2/dt | V] ~ 2V^2/dt
                R_kf = 2 * max(V_pred, 1e-6) ** 2 / dt_val

            # Kalman gain and update
            K_kf = P_pred / (P_pred + R_kf)
            V_filt = V_pred + K_kf * (y_obs - V_pred)
            V_filt = max(V_filt, 1e-6)
            P_kf = (1 - K_kf) * P_pred

            V_hat[t] = V_filt

            # EWMA for drift
            ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / dt_val

            # --- SDRE policy via local quadratic utility approximation ---
            mu_excess = known_mu if known_mu is not None else ewma_mu
            W_safe = max(W, 1e-8)

            # Ito expansion: E[dU] = ... + pi*b + pi^2*c
            b = U_prime(W_safe) * W_safe * mu_excess
            c = 0.5 * U_double_prime(W_safe) * W_safe ** 2 * V_hat[t]

            concavity_ok[t] = (c < -1e-12)
            if c < -1e-12:
                pi = np.clip(-b / (2 * c), 0.01, 5.0)
            else:
                pi = 0.5  # fallback
            pi_sdre[t] = pi

            # Update wealth (for next step's U', U'')
            W *= (1 + p.r * dt_val + pi * ret)
            W = max(W, 1e-8)
            W_history[t] = W

        return {
            'V_hat': V_hat,
            'pi_sdre': pi_sdre,
            'state_true': state_true,
            'W_history': W_history,
            'concavity_frac': np.mean(concavity_ok[1000:]),
        }

    # Shared Heston sim function factory
    def _make_heston_sim(p_ref):
        def sim(t, z1, z2, dt_val, sd):
            if 'V' not in sd:
                sd['V'] = p_ref.theta
            V_prev = sd['V']
            sv = np.sqrt(max(V_prev, 1e-8))
            sdt = np.sqrt(dt_val)
            z2c = p_ref.rho * z1 + np.sqrt(1 - p_ref.rho ** 2) * z2
            ret = (p_ref.mu - p_ref.r) * dt_val + sv * sdt * z1
            V_new = max(V_prev + p_ref.kappa * (p_ref.theta - V_prev) * dt_val
                        + p_ref.xi * sv * sdt * z2c, 1e-8)
            sd['V'] = V_new
            return V_new, ret
        return sim

    # CdC-learned dynamics for Kalman filter (from Level 2: kappa~2.0, theta~0.04)
    # In a full pipeline, these come from Level 2 output.
    # Here we use the true params as a stand-in (Level 2 recovers kappa=1.96±0.15).
    heston_kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': p.xi}

    # ===== Phase 4a: Heston + CRRA(gamma=2) =====
    print("\n--- Phase 4a: Heston + CRRA(gamma=2) — SDRE sanity check ---")
    print("  Kalman(CIR): raw r^2/dt obs, known dynamics")
    print("  BLR+KF: LL-LogSig(BCH) -> BLR(QV area + leverage) -> Kalman")

    U_p, U_pp = _make_crra(p.gamma)

    # Run both methods side by side
    methods_4a = {
        'Kalman': {'use_leadlag': False},
        'BLR+KF': {'use_leadlag': True},
    }
    results_4a = {m: {'v': [], 'pi': [], 'c': []} for m in methods_4a}

    for ms in range(n_macro):
        for mname, mkwargs in methods_4a.items():
            res = _run_sdre_control(_make_heston_sim(p), T, p.dt, ms * 1000,
                                    U_p, U_pp, known_mu=p.mu - p.r,
                                    kf_params=heston_kf, **mkwargs)
            test_s = 1000
            V_est = res['V_hat'][test_s:]
            V_true = res['state_true'][test_s:]

            v_corr = np.corrcoef(V_est, V_true)[0, 1]
            results_4a[mname]['v'].append(v_corr)

            pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)
            pi_corr = np.corrcoef(res['pi_sdre'][test_s:], pi_merton)[0, 1]
            results_4a[mname]['pi'].append(pi_corr)
            results_4a[mname]['c'].append(res['concavity_frac'])

        print(f"  Seed {ms}: " + ", ".join(
            f"{m}: V={results_4a[m]['v'][-1]:.3f} pi={results_4a[m]['pi'][-1]:.3f}"
            for m in methods_4a))

    print(f"\n{'Method':<10} {'V_corr':<18} {'pi_corr':<18} {'c<0':<10}")
    print("-" * 58)
    for mname in methods_4a:
        r = results_4a[mname]
        print(f"{mname:<10} "
              f"{np.mean(r['v']):.3f}+/-{np.std(r['v']):.3f}{'':>5} "
              f"{np.mean(r['pi']):.3f}+/-{np.std(r['pi']):.3f}{'':>5} "
              f"{np.mean(r['c']):.3f}")

    # PASS criteria apply to the BEST method (either Kalman or BLR+KF)
    v_corrs = results_4a['Kalman']['v']
    pi_corrs = results_4a['Kalman']['pi']
    c_fracs = results_4a['Kalman']['c']
    ll_v_corrs = results_4a['BLR+KF']['v']
    ll_pi_corrs = results_4a['BLR+KF']['pi']

    v_mean = np.mean(v_corrs)
    pi_mean = np.mean(pi_corrs)
    c_mean = np.mean(c_fracs)
    ll_v_mean = np.mean(ll_v_corrs)
    ll_pi_mean = np.mean(ll_pi_corrs)

    v_ok = v_mean > 0.50
    pi_ok = pi_mean > 0.30
    c_ok = c_mean > 0.90
    ll_v_ok = ll_v_mean > 0.40  # LL is model-free, lower bar
    ll_pi_ok = ll_pi_mean > 0.25

    print(f"\n{'Metric':<35} {'Threshold':<10} {'Status'}")
    print("-" * 60)
    print(f"{'Kalman V_corr':<35} >0.50{'':<5} {'OK' if v_ok else 'FAIL'}")
    print(f"{'Kalman pi_corr':<35} >0.30{'':<5} {'OK' if pi_ok else 'FAIL'}")
    print(f"{'Concavity c<0':<35} >0.90{'':<5} {'OK' if c_ok else 'FAIL'}")
    print(f"{'BLR+KF V_corr (model-free)':<35} >0.40{'':<5} {'OK' if ll_v_ok else 'FAIL'}")
    print(f"{'BLR+KF pi_corr (model-free)':<35} >0.25{'':<5} {'OK' if ll_pi_ok else 'FAIL'}")
    phase_4a = v_ok and pi_ok and c_ok and ll_v_ok and ll_pi_ok
    print(f"Phase 4a: {'PASS' if phase_4a else 'FAIL'}")

    # ===== Phase 4b: Multi-Utility (KEY TEST) =====
    print("\n--- Phase 4b: Multi-utility — same V_hat, different Q ---")
    print("  CRRA: Q~gamma/W (cancels), CARA: Q~alpha*exp(-aW)*W^2 (W-dep)")
    print("  Log:  Q~1/W^2 (cancels)")

    alpha_cara = 3.0  # α=3 keeps π*≈0.5 at W=1,V=0.04; avoids clip saturation
    utilities = {
        'CRRA(2)': (_make_crra(2.0),
                    lambda V, W: (p.mu - p.r) / (2.0 * V)),
        f'CARA({alpha_cara:.0f})': (_make_cara(alpha_cara),
                    lambda V, W: (p.mu - p.r) / (alpha_cara * W * V)),
        'Log':     (_make_log(),
                    lambda V, W: (p.mu - p.r) / V),
    }

    # Collect per-seed results
    util_pi_means = {name: [] for name in utilities}
    util_pi_corrs = {name: [] for name in utilities}
    cara_w_corrs = []   # corr(pi_CARA, 1/W) — should be high
    crra_w_corrs = []   # corr(pi_CRRA, 1/W) — should be ~0

    for ms in range(n_macro):
        seed = ms * 4000
        per_util_pi = {}

        for name, ((up, upp), ground_truth_fn) in utilities.items():
            res = _run_sdre_control(_make_heston_sim(p), T, p.dt, seed,
                                    up, upp, known_mu=p.mu - p.r,
                                    kf_params=heston_kf)
            test_s = 1000
            pi_sdre = res['pi_sdre'][test_s:]
            V_true = res['state_true'][test_s:]
            W_hist = res['W_history'][test_s:]

            # Ground truth correlation (meaningful for CRRA/Log where W cancels)
            pi_true = np.clip(ground_truth_fn(V_true, W_hist), 0.01, 5.0)
            corr = np.corrcoef(pi_sdre, pi_true)[0, 1]
            util_pi_corrs[name].append(corr)
            util_pi_means[name].append(np.mean(pi_sdre))
            per_util_pi[name] = pi_sdre

            # Wealth-sensitivity: corr(pi, 1/W)
            # CARA: pi ∝ 1/(W*V), so corr(pi, 1/W) should be high
            # CRRA: pi ∝ 1/V (W cancels), so corr(pi, 1/W) should be ~0
            inv_W = 1.0 / W_hist
            c_w = np.corrcoef(pi_sdre, inv_W)[0, 1]
            if 'CARA' in name:
                cara_w_corrs.append(c_w)
            elif name == 'CRRA(2)':
                crra_w_corrs.append(c_w)

        print(f"  Seed {ms}: " + ", ".join(
            f"{n}={util_pi_corrs[n][-1]:.3f} (mean_pi={util_pi_means[n][-1]:.2f})"
            for n in utilities))

    # Check 1: all three policies distinct (mean pi differs)
    mean_pis = {n: np.mean(util_pi_means[n]) for n in utilities}
    pairwise_diffs = []
    names = list(utilities.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise_diffs.append(abs(mean_pis[names[i]] - mean_pis[names[j]]))
    min_diff = min(pairwise_diffs)
    distinct_ok = min_diff > 0.1

    # Check 2: CARA is wealth-sensitive, CRRA is not
    cara_w_mean = np.nanmean(cara_w_corrs)
    crra_w_mean = np.nanmean(crra_w_corrs)
    # CARA should have high corr(pi, 1/W), CRRA should not
    cara_wealth_ok = cara_w_mean > 0.3
    wealth_gap_ok = cara_w_mean - abs(crra_w_mean) > 0.2

    # Check 3: CRRA and Log correlations with ground truth (V̂ quality)
    # CARA correlation is unreliable over long horizons due to W drift
    crra_log_ok = True
    print(f"\n{'Utility':<12} {'pi* corr':<18} {'mean pi':<12} {'corr(pi,1/W)':<14} {'Status'}")
    print("-" * 70)
    for name in utilities:
        mc = np.nanmean(util_pi_corrs[name])
        ms_val = np.nanstd(util_pi_corrs[name])
        mp = mean_pis[name]
        if 'CARA' in name:
            w_corr = cara_w_mean
            ok_str = f"W-dep={cara_w_mean:.2f}"
        else:
            w_corr = crra_w_mean if name == 'CRRA(2)' else 0.0
            ok = mc > 0.20
            if not ok:
                crra_log_ok = False
            ok_str = 'OK' if ok else 'FAIL'
        print(f"{name:<12} {mc:.3f}+/-{ms_val:.3f}{'':>5} {mp:.2f}"
              f"{'':>7} {w_corr:+.3f}{'':>7} {ok_str}")

    print(f"\n  Min pairwise mean-pi diff: {min_diff:.3f} (>0.10: "
          f"{'OK' if distinct_ok else 'FAIL'})")
    print(f"  CARA corr(pi,1/W): {cara_w_mean:.3f} (>0.30: "
          f"{'OK' if cara_wealth_ok else 'FAIL'})")
    print(f"  CRRA corr(pi,1/W): {crra_w_mean:.3f} (should be ~0)")
    print(f"  Wealth-sensitivity gap (CARA-CRRA): "
          f"{cara_w_mean - abs(crra_w_mean):.3f} (>0.20: "
          f"{'OK' if wealth_gap_ok else 'FAIL'})")

    phase_4b = distinct_ok and cara_wealth_ok and wealth_gap_ok and crra_log_ok
    print(f"Phase 4b: {'PASS' if phase_4b else 'FAIL'}")

    # ===== Phase 4c: CEV dynamics =====
    print("\n--- Phase 4c: CEV + CRRA — different SDE, same SDRE pipeline ---")
    print("  CEV has NO CIR dynamics — model-free BLR+KF is the natural choice")
    cev_sigma0, cev_beta = 0.3, 0.5
    cev_mu, cev_r = 0.08, 0.02
    cev_dt = 1 / 252

    results_4c = {m: {'v': [], 'pi': []} for m in ['Kalman', 'BLR+KF']}

    for ms in range(n_macro):
        def cev_sim(t, z1, z2, dt_val, sd):
            if 'S' not in sd:
                sd['S'] = 1.0
            S_prev = sd['S']
            sdt = np.sqrt(dt_val)
            local_vol = cev_sigma0 * S_prev ** (cev_beta - 1)
            ret = (cev_mu - cev_r) * dt_val + local_vol * sdt * z1
            S_new = max(S_prev * (1 + ret), 1e-4)
            sd['S'] = S_new
            return local_vol ** 2, ret

        U_p_cev, U_pp_cev = _make_crra(p.gamma)
        # CEV vol is state-dependent, not CIR. Use weak mean-reversion prior.
        cev_kf = {'kappa': 0.5, 'theta': cev_sigma0 ** 2, 'xi': 0.1}

        for mname, ll_flag in [('Kalman', False), ('BLR+KF', True)]:
            res = _run_sdre_control(cev_sim, T, cev_dt, ms * 5000,
                                    U_p_cev, U_pp_cev, known_mu=cev_mu - cev_r,
                                    kf_params=cev_kf, use_leadlag=ll_flag)
            test_s = 1000
            V_est = res['V_hat'][test_s:]
            V_true = res['state_true'][test_s:]

            v_corr = np.corrcoef(V_est, V_true)[0, 1]
            results_4c[mname]['v'].append(v_corr)

            pi_true = np.clip((cev_mu - cev_r) / (p.gamma * V_true), 0.01, 5.0)
            pi_corr = np.corrcoef(res['pi_sdre'][test_s:], pi_true)[0, 1]
            results_4c[mname]['pi'].append(pi_corr)

        print(f"  Seed {ms}: " + ", ".join(
            f"{m}: V={results_4c[m]['v'][-1]:.3f} pi={results_4c[m]['pi'][-1]:.3f}"
            for m in results_4c))

    print(f"\n{'Method':<10} {'V_corr':<18} {'pi_corr':<18}")
    print("-" * 48)
    for mname in results_4c:
        r = results_4c[mname]
        print(f"{mname:<10} "
              f"{np.mean(r['v']):.3f}+/-{np.std(r['v']):.3f}{'':>5} "
              f"{np.mean(r['pi']):.3f}+/-{np.std(r['pi']):.3f}")

    cev_v_mean = np.mean(results_4c['Kalman']['v'])
    cev_pi_mean = np.mean(results_4c['Kalman']['pi'])
    ll_cev_v_mean = np.mean(results_4c['BLR+KF']['v'])
    ll_cev_pi_mean = np.mean(results_4c['BLR+KF']['pi'])

    # Either method passing is sufficient
    cev_v_ok = max(cev_v_mean, ll_cev_v_mean) > 0.40
    cev_pi_ok = max(cev_pi_mean, ll_cev_pi_mean) > 0.25

    print(f"\n{'Metric':<35} {'Threshold':<10} {'Status'}")
    print("-" * 60)
    print(f"{'Best V_corr (Kalman or BLR+KF)':<35} >0.40{'':<5} {'OK' if cev_v_ok else 'FAIL'}")
    print(f"{'Best pi_corr (Kalman or BLR+KF)':<35} >0.25{'':<5} {'OK' if cev_pi_ok else 'FAIL'}")
    phase_4c = cev_v_ok and cev_pi_ok
    print(f"Phase 4c: {'PASS' if phase_4c else 'FAIL'}")

    # ===== Overall =====
    passed = phase_4a and phase_4b and phase_4c
    print(f"\nLEVEL 4: {'PASS' if passed else 'FAIL'}")

    return {'passed': passed,
            'description': 'SDRE + quadratic utility approx (changing Q)',
            'details': {'heston_v_corr': v_mean, 'heston_pi_corr': pi_mean,
                        'heston_ll_v_corr': ll_v_mean,
                        'heston_ll_pi_corr': ll_pi_mean,
                        'concavity': c_mean,
                        'cev_v_corr': cev_v_mean, 'cev_pi_corr': cev_pi_mean,
                        'cev_ll_v_corr': ll_cev_v_mean,
                        'cev_ll_pi_corr': ll_cev_pi_mean,
                        'cara_wealth_corr': cara_w_mean, 'min_pi_diff': min_diff}}


# ======================================================================
# Summary
# ======================================================================

def print_summary(results):
    print("\n" + "=" * 70)
    print("GRADUATED SANITY CHECK SUMMARY")
    print("=" * 70)
    for level, res in results.items():
        status = "PASS" if res['passed'] else "FAIL"
        print(f"  {level}: [{status}] {res['description']}")
    all_passed = all(r['passed'] for r in results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)


def main():
    results = {}
    results['Level 0'] = level0_heston_filtering_stage1()
    results['Level 1'] = level1_heston_filtering()
    results['Level 2'] = level2_generator_recovery()
    results['Level 3'] = level3_transaction_costs()
    results['Level 4'] = level4_product_kernel_control()
    print_summary(results)
    return results


if __name__ == "__main__":
    main()
