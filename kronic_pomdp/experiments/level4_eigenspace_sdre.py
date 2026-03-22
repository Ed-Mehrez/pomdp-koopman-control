"""
Level 4 Eigenspace SDRE — KRONIC-style eigenfunction decomposition.

Key idea: The slow eigenfunctions of the signature dynamics operator
select the linear combinations of signature features that change most
slowly — these track the persistent hidden state (volatility).

Architecture:
  1. Warmup: collect signature features, learn generator A via batch LS
  2. Eigendecompose A, sort by |Re(lambda)| (slowest first)
  3. Online: project signatures to eigenspace, keep slow modes
  4. Learn V_hat = w . z_slow via online RLS (target = r^2/dt)
  5. SDRE policy from eigenspace-derived V_hat

The slow eigenfunction IS the spectral filter — no Kalman needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from examples.proof_of_concept.signature_features import RecurrentSignatureMap

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

HestonParams = namedtuple('HestonParams',
    ['mu', 'r', 'kappa', 'theta', 'xi', 'rho', 'gamma', 'dt'])

P = HestonParams(mu=0.08, r=0.02, kappa=2.0, theta=0.04, xi=0.3,
                 rho=-0.5, gamma=2.0, dt=1/252)
T = 10000


def _make_crra(gamma):
    return (lambda W: W**(-gamma), lambda W: -gamma * W**(-gamma-1))


def _make_heston_sim(p):
    def sim(t, z1, z2, dt_val, sd):
        if 'V' not in sd:
            sd['V'] = p.theta
        V_prev = sd['V']
        sv = np.sqrt(max(V_prev, 1e-8))
        sdt = np.sqrt(dt_val)
        z2c = p.rho * z1 + np.sqrt(1 - p.rho**2) * z2
        ret = (p.mu - p.r) * dt_val + sv * sdt * z1
        V_new = max(V_prev + p.kappa * (p.theta - V_prev) * dt_val
                    + p.xi * sv * sdt * z2c, 1e-8)
        sd['V'] = V_new
        return V_new, ret
    return sim


class EigenspaceSigFilter:
    """Eigenfunction-based volatility filter using signature dynamics.

    Phase 1 (warmup): Collect signature features, learn generator A,
        eigendecompose, identify slow modes.
    Phase 2 (online): Project to eigenspace, use slow modes + RLS
        to estimate V.

    The slow eigenfunctions act as a spectral filter — they select
    the persistent components of the signature that track the hidden
    volatility state.
    """
    def __init__(self, input_dim=2, sig_level=2, sig_ff=0.94,
                 n_warmup=2000, n_slow_modes=None, rls_ff=0.999):
        self.sig_map = RecurrentSignatureMap(
            state_dim=input_dim, level=sig_level, forgetting_factor=sig_ff)
        self.n_features = self.sig_map.feature_dim  # 6 for dim=2, level=2
        self.n_warmup = n_warmup
        self.n_slow_modes = n_slow_modes  # None = auto-select
        self.rls_ff = rls_ff

        # Warmup storage
        self.phi_history = []
        self.warmup_done = False

        # Eigenspace projection (set after warmup)
        self.V_proj = None  # (n_features, n_slow) projection to slow eigenspace
        self.eigenvalues = None
        self.n_slow = None

        # RLS in eigenspace: z_slow -> V_hat
        self.w_rls = None
        self.P_rls = None

        # Running estimate
        self.last_phi = None

    def reset(self):
        self.sig_map.reset()
        self.last_phi = None

    def _learn_generator(self):
        """Learn generator A from collected signature pairs.

        Model: (phi_{t+1} - phi_t) / dt ≈ A @ phi_t
        Solve via regularized least squares.
        """
        Phi = np.array(self.phi_history)  # (T, n_features)
        Phi_current = Phi[:-1]  # (T-1, n_features)
        Phi_next = Phi[1:]

        # Target: dphi/dt
        # Note: we don't know dt here, but it's constant, so A*dt is what we learn
        dPhi = Phi_next - Phi_current  # ≈ A * dt * Phi_current

        # Regularized LS: A_dt = dPhi^T Phi_current (Phi_current^T Phi_current + reg)^{-1}
        XtX = Phi_current.T @ Phi_current
        reg = 1e-4 * np.trace(XtX) / self.n_features * np.eye(self.n_features)
        XtY = Phi_current.T @ dPhi

        A_dt = np.linalg.solve(XtX + reg, XtY).T  # (n_features, n_features)
        return A_dt

    def _eigendecompose(self, A_dt):
        """Eigendecompose generator, select slow modes."""
        eigenvalues, eigenvectors = np.linalg.eig(A_dt)

        # Sort by |Re(lambda)| ascending (slowest first)
        # A_dt = A * dt, so eigenvalues are lambda * dt
        idx = np.argsort(np.abs(np.real(eigenvalues)))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Handle complex eigenvalues: take real part of eigenvectors
        # For complex conjugate pairs, use Re(v) and Im(v) as two real eigenvectors
        real_eigvecs = []
        real_eigvals = []
        i = 0
        while i < len(eigenvalues):
            if np.isreal(eigenvalues[i]) or abs(np.imag(eigenvalues[i])) < 1e-10:
                real_eigvecs.append(np.real(eigenvectors[:, i]))
                real_eigvals.append(np.real(eigenvalues[i]))
                i += 1
            else:
                # Complex pair: use Re and Im
                real_eigvecs.append(np.real(eigenvectors[:, i]))
                real_eigvecs.append(np.imag(eigenvectors[:, i]))
                real_eigvals.append(np.real(eigenvalues[i]))
                real_eigvals.append(np.real(eigenvalues[i]))
                i += 2

        V_mat = np.column_stack(real_eigvecs)  # (n_features, n_features)
        lambdas = np.array(real_eigvals)

        # Select slow modes: smallest |Re(lambda)|
        n_slow = self.n_slow_modes
        if n_slow is None:
            # Auto: keep modes with |lambda*dt| < 0.1 (half-life > 10 steps)
            # Or at least 2, at most n_features-1
            slow_mask = np.abs(lambdas) < 0.1
            n_slow = max(2, min(sum(slow_mask), self.n_features - 1))
        n_slow = min(n_slow, len(lambdas))

        self.V_proj = V_mat[:, :n_slow]  # (n_features, n_slow)
        self.eigenvalues = lambdas[:n_slow]
        self.n_slow = n_slow

        # Initialize RLS in eigenspace: (n_slow + 1) features (+ bias)
        n_rls = n_slow + 1
        self.w_rls = np.zeros(n_rls)
        self.P_rls = np.eye(n_rls) * 100.0

        return lambdas

    def update(self, dx, ret, dt, init_val=0.04):
        """Process one step: update signatures, project to eigenspace, estimate V.

        Args:
            dx: signature increment [dt, ret]
            ret: raw return
            dt: time step
            init_val: fallback V estimate during warmup

        Returns: V_hat estimate
        """
        phi = self.sig_map.update(dx)
        self.last_phi = phi.copy()

        if not self.warmup_done:
            self.phi_history.append(phi.copy())
            if len(self.phi_history) >= self.n_warmup:
                # Learn generator and eigendecompose
                A_dt = self._learn_generator()
                lambdas = self._eigendecompose(A_dt)
                self.warmup_done = True
                self.phi_history = []  # Free memory
            return init_val

        # Project to slow eigenspace
        z_slow = self.V_proj.T @ phi  # (n_slow,)
        features = np.concatenate([z_slow, [1.0]])  # + bias

        # RLS prediction
        pred = np.dot(self.w_rls, features)

        # Update RLS with noisy target r^2/dt
        target = min(ret**2 / dt, 2.0)
        z = features[:, np.newaxis]
        Pz = self.P_rls @ z
        denom = self.rls_ff + (z.T @ Pz)[0, 0]
        k = Pz / denom
        self.w_rls += k.flatten() * (target - pred)
        self.P_rls = (self.P_rls - k @ Pz.T) / self.rls_ff

        return max(pred, 1e-8)


def run_sdre_eigenspace(p, T, seed, U_prime, U_double_prime,
                        known_mu=None, n_warmup=2000, n_slow_modes=None,
                        sig_ff=0.94, rls_ff=0.999):
    """SDRE controller using eigenspace signature filter."""
    rng = np.random.RandomState(seed)

    eig_filter = EigenspaceSigFilter(
        input_dim=2, sig_ff=sig_ff, n_warmup=n_warmup,
        n_slow_modes=n_slow_modes, rls_ff=rls_ff)

    ewma_mu = 0.0
    lam_mu = 0.999

    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)

    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        dx = np.array([p.dt, ret])
        V_est = eig_filter.update(dx, ret, p.dt, init_val=p.theta)
        V_hat[t] = V_est

        # Drift
        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt

        # SDRE policy
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)

        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]

        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi

        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {
        'V_hat': V_hat, 'pi_sdre': pi_sdre, 'state_true': state_true,
        'W_history': W_history,
        'n_slow': eig_filter.n_slow,
        'eigenvalues': eig_filter.eigenvalues,
    }


def run_sdre_kalman(p, T, seed, U_prime, U_double_prime,
                    known_mu=None, xi_kf=None):
    """Kalman baseline (same as level4_generator_sdre.py)."""
    rng = np.random.RandomState(seed)
    kf_kappa, kf_theta = p.kappa, p.theta
    kf_xi = xi_kf if xi_kf is not None else p.xi
    V_filt = kf_theta
    P_kf = kf_xi**2 * kf_theta * p.dt * 10
    ewma_mu, lam_mu = 0.0, 0.999

    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        V_pred = max(V_filt + kf_kappa * (kf_theta - V_filt) * p.dt, 1e-6)
        Q_kf = kf_xi**2 * max(V_filt, 1e-6) * p.dt
        P_pred = (1 - kf_kappa * p.dt)**2 * P_kf + Q_kf
        y_obs = ret**2 / p.dt
        R_kf = 2 * max(V_pred, 1e-6)**2 / p.dt
        K_kf = P_pred / (P_pred + R_kf)
        V_filt = max(V_pred + K_kf * (y_obs - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred
        V_hat[t] = V_filt

        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)
        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi
        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {'V_hat': V_hat, 'pi_sdre': pi_sdre, 'state_true': state_true,
            'W_history': W_history}


def run_sdre_recsig_rls(p, T, seed, U_prime, U_double_prime,
                        known_mu=None, sig_ff=0.94, rls_ff=0.999):
    """RecSig-RLS baseline (Level 0-1 approach, targeting r^2/dt)."""
    rng = np.random.RandomState(seed)

    sig_map = RecurrentSignatureMap(state_dim=2, level=2, forgetting_factor=sig_ff)
    n_feat = sig_map.feature_dim + 1
    w = np.zeros(n_feat)
    P_rls = np.eye(n_feat) * 100.0

    ewma_mu, lam_mu = 0.0, 0.999
    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    state_dict = {}
    sim_fn = _make_heston_sim(p)
    W = 1.0

    for t in range(T):
        z1 = rng.randn()
        z2 = rng.randn()
        state_val, ret = sim_fn(t, z1, z2, p.dt, state_dict)
        state_true[t] = state_val

        dx = np.array([p.dt, ret])
        phi = sig_map.update(dx)
        features = np.concatenate([phi, [1.0]])
        pred = np.dot(w, features)

        target = min(ret**2 / p.dt, 2.0)
        z = features[:, np.newaxis]
        Pz = P_rls @ z
        denom = rls_ff + (z.T @ Pz)[0, 0]
        k = Pz / denom
        w += k.flatten() * (target - pred)
        P_rls = (P_rls - k @ Pz.T) / rls_ff

        V_hat[t] = max(pred, 1e-8)
        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)
        b = U_prime(W_safe) * W_safe * mu_excess
        c = 0.5 * U_double_prime(W_safe) * W_safe**2 * V_hat[t]
        if c < -1e-12:
            pi = np.clip(-b / (2 * c), 0.01, 5.0)
        else:
            pi = 0.5
        pi_sdre[t] = pi
        W *= (1 + p.r * p.dt + pi * ret)
        W = max(W, 1e-8)
        W_history[t] = W

    return {'V_hat': V_hat, 'pi_sdre': pi_sdre, 'state_true': state_true,
            'W_history': W_history}


def compare_all(p, n_seeds=5, known_mu_mode=True):
    """Compare eigenspace vs Kalman vs RecSig-RLS."""
    U_p, U_pp = _make_crra(p.gamma)
    test_s = 2500  # after warmup
    known_mu = (p.mu - p.r) if known_mu_mode else None

    methods = {
        'Kalman(xi=0.3)': lambda s: run_sdre_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, xi_kf=0.3),
        'Kalman(xi=2.0)': lambda s: run_sdre_kalman(
            p, T, s, U_p, U_pp, known_mu=known_mu, xi_kf=2.0),
        'RecSig-RLS': lambda s: run_sdre_recsig_rls(
            p, T, s, U_p, U_pp, known_mu=known_mu),
        'Eigenspace(auto)': lambda s: run_sdre_eigenspace(
            p, T, s, U_p, U_pp, known_mu=known_mu, n_slow_modes=None),
        'Eigenspace(2)': lambda s: run_sdre_eigenspace(
            p, T, s, U_p, U_pp, known_mu=known_mu, n_slow_modes=2),
        'Eigenspace(4)': lambda s: run_sdre_eigenspace(
            p, T, s, U_p, U_pp, known_mu=known_mu, n_slow_modes=4),
    }

    v_corrs = {m: [] for m in methods}
    pi_corrs = {m: [] for m in methods}
    pi_stds = {m: [] for m in methods}

    for seed in range(0, n_seeds * 1000, 1000):
        for name, fn in methods.items():
            res = fn(seed)
            V_true = res['state_true'][test_s:]
            V_h = res['V_hat'][test_s:]
            pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)

            vc = np.corrcoef(V_h, V_true)[0, 1]
            pc = np.corrcoef(res['pi_sdre'][test_s:], pi_merton)[0, 1]
            ps = np.std(res['pi_sdre'][test_s:])

            v_corrs[name].append(vc)
            pi_corrs[name].append(pc)
            pi_stds[name].append(ps)

            if 'Eigenspace' in name and 'n_slow' in res and seed == 0:
                print(f"  {name}: n_slow={res['n_slow']}, "
                      f"eigenvalues={res['eigenvalues']}")

    suffix = 'known_mu' if known_mu_mode else 'learned_mu'
    print(f"\nMulti-seed comparison ({n_seeds} seeds, {suffix}):")
    print(f"{'Method':<22} {'V_corr':>14} {'pi_corr':>14} {'pi_std':>14}")
    print("-" * 66)
    for m in methods:
        vc = f"{np.mean(v_corrs[m]):.3f}+/-{np.std(v_corrs[m]):.3f}"
        pc = f"{np.mean(pi_corrs[m]):.3f}+/-{np.std(pi_corrs[m]):.3f}"
        ps = f"{np.mean(pi_stds[m]):.3f}+/-{np.std(pi_stds[m]):.3f}"
        print(f"{m:<22} {vc:>14} {pc:>14} {ps:>14}")

    # Visualization: single seed
    seed = 42
    t_days = np.arange(T) / 252
    results = {name: fn(seed) for name, fn in methods.items()}
    V_true = results['Kalman(xi=0.3)']['state_true']
    pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(t_days[test_s:], np.sqrt(V_true[test_s:]),
            alpha=0.3, lw=0.5, color='gray', label='True $\\sqrt{V}$')
    for name in ['Kalman(xi=0.3)', 'Kalman(xi=2.0)', 'RecSig-RLS',
                 'Eigenspace(auto)', 'Eigenspace(2)']:
        res = results[name]
        corr = np.corrcoef(res['V_hat'][test_s:], V_true[test_s:])[0, 1]
        ax.plot(t_days[test_s:], np.sqrt(np.maximum(res['V_hat'][test_s:], 1e-8)),
                alpha=0.7, lw=0.8, label=f'{name} ({corr:.3f})')
    ax.set_ylabel('$\\sqrt{V}$')
    ax.set_title(f'Volatility Estimation: Eigenspace vs Baselines ({suffix})')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    ax.plot(t_days[test_s:], pi_merton[test_s:],
            alpha=0.3, lw=0.5, color='gray', label='Merton $\\pi^*$')
    for name in ['Kalman(xi=0.3)', 'Kalman(xi=2.0)', 'RecSig-RLS',
                 'Eigenspace(auto)', 'Eigenspace(2)']:
        res = results[name]
        corr = np.corrcoef(res['pi_sdre'][test_s:], pi_merton[test_s:])[0, 1]
        ax.plot(t_days[test_s:], res['pi_sdre'][test_s:],
                alpha=0.7, lw=0.8, label=f'{name} ({corr:.3f})')
    ax.set_ylabel('$\\pi$')
    ax.set_xlabel('Years')
    ax.set_title('SDRE Policy')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 3.0)

    plt.tight_layout()
    plt.savefig(f'kronic_pomdp/experiments/level4_eigenspace_{suffix}.png',
                bbox_inches='tight')
    plt.close()
    print(f"Saved: level4_eigenspace_{suffix}.png")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    method_names = list(methods.keys())
    x = np.arange(len(method_names))
    w_bar = 0.6

    for ax_idx, (metric, label) in enumerate([
            (v_corrs, 'corr($\\hat{V}$, $V_{true}$)'),
            (pi_corrs, 'corr($\\pi_{SDRE}$, $\\pi_{Merton}$)'),
            (pi_stds, 'std($\\pi_{SDRE}$)')]):
        ax = axes[ax_idx]
        means = [np.mean(metric[m]) for m in method_names]
        stds = [np.std(metric[m]) for m in method_names]
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        ax.bar(x, means, w_bar, yerr=stds, capsize=3, color=colors[:len(x)])
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=25, fontsize=8, ha='right')
        ax.set_ylabel(label)

    plt.suptitle(f'Eigenspace SDRE vs Baselines ({suffix})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'kronic_pomdp/experiments/level4_eigenspace_bars_{suffix}.png',
                bbox_inches='tight')
    plt.close()
    print(f"Saved: level4_eigenspace_bars_{suffix}.png")


if __name__ == '__main__':
    print("=== Eigenspace SDRE Diagnostics ===\n")

    print("--- Known mu (isolates V estimation) ---")
    compare_all(P, n_seeds=5, known_mu_mode=True)

    print("\n\n--- Learned mu (full pipeline) ---")
    compare_all(P, n_seeds=5, known_mu_mode=False)

    print("\n=== Done ===")
