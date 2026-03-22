"""
Level 4 Generator-based SDRE — Using the empirical generator for control.

Instead of:  EWMA → mu_hat,  Kalman(r^2/dt) → V_hat  →  SDRE
We do:       Signatures → Generator → (mu_hat, V_hat) →  SDRE

The CdC identity on the return stream:
   mu_hat(phi) = E[r/dt | phi_t]           = w_mu . phi_t
   E[r^2/dt | phi_t]                        = w_v . phi_t
   V_hat(phi)  = w_v . phi_t - (mu_hat)^2 dt   (CdC decomposition)

This unifies drift and diffusion estimation in one framework:
the Koopman generator applied to return observables through signatures.
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


class DualTargetSigRLS:
    """Signature-based RLS that learns BOTH mu and V from the generator.

    Two parallel RLS heads on the same signature features:
      Head 1: phi_t -> r_{t+1}/dt   (conditional drift = generator on r)
      Head 2: phi_t -> r_{t+1}^2/dt (conditional 2nd moment = generator on r^2)

    Then V_hat = head2 - (head1)^2 * dt  (CdC decomposition)
    """
    def __init__(self, input_dim=2, forgetting_factor=0.94,
                 rls_ff=0.999, init_mu=0.06, init_v=0.04):
        self.sig_map = RecurrentSignatureMap(
            state_dim=input_dim, level=2, forgetting_factor=forgetting_factor)
        self.n_features = self.sig_map.feature_dim + 1  # +1 for bias

        # Head 1: drift (r/dt)
        self.w_mu = np.zeros(self.n_features)
        self.P_mu = np.eye(self.n_features) * 100.0

        # Head 2: second moment (r^2/dt)
        self.w_v = np.zeros(self.n_features)
        self.P_v = np.eye(self.n_features) * 100.0

        self.rls_ff = rls_ff
        self.init_mu = init_mu
        self.init_v = init_v

    def reset(self):
        self.sig_map.reset()

    def update(self, dx, ret, dt):
        """Update both heads with new return observation.

        Args:
            dx: signature increment [dt, ret]
            ret: raw return (for computing targets)
            dt: time step

        Returns: (mu_hat, V_hat)
        """
        sig_features = self.sig_map.update(dx)
        features = np.concatenate([sig_features, [1.0]])
        z = features[:, np.newaxis]

        # Target 1: r/dt (drift)
        target_mu = ret / dt
        # Clip extreme values (fat tails)
        target_mu = np.clip(target_mu, -10.0, 10.0)
        pred_mu = np.dot(self.w_mu, features)

        Pz_mu = self.P_mu @ z
        denom_mu = self.rls_ff + (z.T @ Pz_mu)[0, 0]
        k_mu = Pz_mu / denom_mu
        self.w_mu += k_mu.flatten() * (target_mu - pred_mu)
        self.P_mu = (self.P_mu - k_mu @ Pz_mu.T) / self.rls_ff

        # Target 2: r^2/dt (second moment)
        target_v = min(ret**2 / dt, 2.0)  # clip as in RecSigRLS
        pred_v = np.dot(self.w_v, features)

        Pz_v = self.P_v @ z
        denom_v = self.rls_ff + (z.T @ Pz_v)[0, 0]
        k_v = Pz_v / denom_v
        self.w_v += k_v.flatten() * (target_v - pred_v)
        self.P_v = (self.P_v - k_v @ Pz_v.T) / self.rls_ff

        # CdC decomposition: V = E[r^2/dt] - (E[r/dt])^2 * dt
        mu_hat = pred_mu
        moment2 = max(pred_v, 1e-8)
        V_hat = max(moment2 - mu_hat**2 * dt, 1e-8)

        return mu_hat, V_hat


def run_sdre_generator(p, T, seed, U_prime, U_double_prime,
                       known_mu=None, sig_ff=0.94, rls_ff=0.999):
    """SDRE controller using signature-based generator for BOTH mu and V."""
    rng = np.random.RandomState(seed)

    gen = DualTargetSigRLS(input_dim=2, forgetting_factor=sig_ff,
                           rls_ff=rls_ff, init_mu=p.mu - p.r, init_v=p.theta)

    V_hat = np.zeros(T)
    mu_hat = np.zeros(T)
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

        # Generator: signatures -> (mu, V) via CdC
        dx = np.array([p.dt, ret])
        mu_est, V_est = gen.update(dx, ret, p.dt)

        V_hat[t] = V_est
        mu_hat[t] = mu_est

        # SDRE policy
        mu_excess = known_mu if known_mu is not None else mu_est
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
        'V_hat': V_hat, 'mu_hat': mu_hat, 'pi_sdre': pi_sdre,
        'state_true': state_true, 'W_history': W_history,
    }


def run_sdre_kalman(p, T, seed, U_prime, U_double_prime,
                    known_mu=None, xi_kf=None):
    """SDRE controller using Kalman filter (current approach)."""
    rng = np.random.RandomState(seed)

    kf_kappa, kf_theta = p.kappa, p.theta
    kf_xi = xi_kf if xi_kf is not None else p.xi
    V_filt = kf_theta
    P_kf = kf_xi**2 * kf_theta * p.dt * 10
    ewma_mu = 0.0
    lam_mu = 0.999

    V_hat = np.zeros(T)
    mu_hat_arr = np.zeros(T)
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

        # Kalman predict
        V_pred = V_filt + kf_kappa * (kf_theta - V_filt) * p.dt
        V_pred = max(V_pred, 1e-6)
        Q_kf = kf_xi**2 * max(V_filt, 1e-6) * p.dt
        P_pred = (1 - kf_kappa * p.dt)**2 * P_kf + Q_kf

        # Kalman observe
        y_obs = ret**2 / p.dt
        R_kf = 2 * max(V_pred, 1e-6)**2 / p.dt
        K_kf = P_pred / (P_pred + R_kf)
        V_filt = max(V_pred + K_kf * (y_obs - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred

        V_hat[t] = V_filt

        # EWMA drift
        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_hat_arr[t] = ewma_mu

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
        'V_hat': V_hat, 'mu_hat': mu_hat_arr, 'pi_sdre': pi_sdre,
        'state_true': state_true, 'W_history': W_history,
    }


def run_sdre_hybrid(p, T, seed, U_prime, U_double_prime,
                    known_mu=None, xi_kf=1.0, sig_ff=0.94, rls_ff=0.999):
    """Hybrid: Signature-filtered observation -> Kalman filter.

    Instead of y_t = r^2/dt (single noisy draw), we use:
        y_t = sig_rls_prediction (signature-smoothed V estimate)

    The signature RLS acts as an observation ENCODER that aggregates
    path history into a less noisy V proxy. The Kalman filter then
    applies CIR dynamics on top for temporal consistency.

    For drift: use the signature-based mu estimate (generator head 1)
    instead of EWMA — the signatures encode path structure beyond
    simple exponential averaging.
    """
    rng = np.random.RandomState(seed)

    # Signature-based observation encoder
    sig_map = RecurrentSignatureMap(state_dim=2, level=2, forgetting_factor=sig_ff)
    n_features = sig_map.feature_dim + 1
    w_v = np.zeros(n_features)
    P_rls = np.eye(n_features) * 100.0
    w_mu = np.zeros(n_features)
    P_rls_mu = np.eye(n_features) * 100.0

    # Kalman filter state
    kf_kappa, kf_theta, kf_xi = p.kappa, p.theta, xi_kf
    V_filt = kf_theta
    P_kf = kf_xi**2 * kf_theta * p.dt * 10

    V_hat = np.zeros(T)
    mu_hat_arr = np.zeros(T)
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

        # --- Signature RLS: learn observation encoder ---
        dx = np.array([p.dt, ret])
        sig_features = sig_map.update(dx)
        features = np.concatenate([sig_features, [1.0]])
        z = features[:, np.newaxis]

        # RLS head for V (target = r^2/dt)
        target_v = min(ret**2 / p.dt, 2.0)
        pred_v_sig = np.dot(w_v, features)
        Pz = P_rls @ z
        denom = rls_ff + (z.T @ Pz)[0, 0]
        k = Pz / denom
        w_v += k.flatten() * (target_v - pred_v_sig)
        P_rls = (P_rls - k @ Pz.T) / rls_ff

        # RLS head for drift (target = r/dt)
        target_mu = np.clip(ret / p.dt, -10.0, 10.0)
        pred_mu_sig = np.dot(w_mu, features)
        Pz_mu = P_rls_mu @ z
        denom_mu = rls_ff + (z.T @ Pz_mu)[0, 0]
        k_mu = Pz_mu / denom_mu
        w_mu += k_mu.flatten() * (target_mu - pred_mu_sig)
        P_rls_mu = (P_rls_mu - k_mu @ Pz_mu.T) / rls_ff

        # --- Kalman filter: use sig prediction as observation ---
        # Predict
        V_pred = V_filt + kf_kappa * (kf_theta - V_filt) * p.dt
        V_pred = max(V_pred, 1e-6)
        Q_kf = kf_xi**2 * max(V_filt, 1e-6) * p.dt
        P_pred = (1 - kf_kappa * p.dt)**2 * P_kf + Q_kf

        # Observe: signature-smoothed V estimate (much less noisy than r^2/dt)
        y_obs = max(pred_v_sig, 1e-6)
        # The observation noise is the RLS prediction error variance
        # Rough estimate: R ~ residual variance of sig -> r^2/dt regression
        # In steady state, sig prediction has variance ~V^2 * (d/N) where
        # d = n_features and N = effective sample size from forgetting
        # Simpler: use a fraction of 2V^2/dt (the raw r^2/dt noise)
        # The signature aggregates ~1/(1-gamma) = ~17 steps, so noise reduces by ~17x
        eff_window = 1.0 / (1.0 - sig_ff)  # ~17 for ff=0.94
        R_kf = 2 * max(V_pred, 1e-6)**2 / (p.dt * eff_window)

        # Update
        K_kf = P_pred / (P_pred + R_kf)
        V_filt = max(V_pred + K_kf * (y_obs - V_pred), 1e-6)
        P_kf = (1 - K_kf) * P_pred

        V_hat[t] = V_filt

        # Drift: signature-based (generator head)
        mu_hat_arr[t] = pred_mu_sig

        # SDRE policy
        mu_excess = known_mu if known_mu is not None else pred_mu_sig
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
        'V_hat': V_hat, 'mu_hat': mu_hat_arr, 'pi_sdre': pi_sdre,
        'state_true': state_true, 'W_history': W_history,
    }


def compare_approaches(p, seed=42, known_mu_mode=True):
    """Compare: Kalman(xi=true), Kalman(xi=2x), Generator-SDRE."""
    U_p, U_pp = _make_crra(p.gamma)
    test_s = 1000
    known_mu = (p.mu - p.r) if known_mu_mode else None
    label_suffix = " (known $\\mu$)" if known_mu_mode else " (learned $\\mu$)"

    configs = {
        f'Kalman ($\\xi$={p.xi})': lambda: run_sdre_kalman(
            p, T, seed, U_p, U_pp, known_mu=known_mu, xi_kf=p.xi),
        f'Kalman ($\\xi$=2.0)': lambda: run_sdre_kalman(
            p, T, seed, U_p, U_pp, known_mu=known_mu, xi_kf=2.0),
        'Generator (Sig-CdC)': lambda: run_sdre_generator(
            p, T, seed, U_p, U_pp, known_mu=known_mu),
        'Hybrid (Sig+Kalman)': lambda: run_sdre_hybrid(
            p, T, seed, U_p, U_pp, known_mu=known_mu, xi_kf=1.0),
    }

    results = {name: fn() for name, fn in configs.items()}

    # Get true V from any result (same seed → same Heston path)
    V_true = list(results.values())[0]['state_true']
    pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)
    t_days = np.arange(T) / 252

    # --- Figure 1: V_hat comparison ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(t_days[test_s:], np.sqrt(V_true[test_s:]),
            alpha=0.3, lw=0.5, color='gray', label='True $\\sqrt{V}$')
    for name, res in results.items():
        corr = np.corrcoef(res['V_hat'][test_s:], V_true[test_s:])[0, 1]
        ax.plot(t_days[test_s:], np.sqrt(res['V_hat'][test_s:]),
                alpha=0.8, lw=1.0, label=f'{name} (corr={corr:.3f})')
    ax.set_ylabel('Volatility $\\sqrt{V}$')
    ax.set_title(f'Volatility Estimation: Generator vs Kalman{label_suffix}')
    ax.legend(loc='upper right', fontsize=9)

    # Panel 2: pi comparison
    ax = axes[1]
    ax.plot(t_days[test_s:], pi_merton[test_s:],
            alpha=0.3, lw=0.5, color='gray', label='Merton $\\pi^*$')
    for name, res in results.items():
        corr = np.corrcoef(res['pi_sdre'][test_s:], pi_merton[test_s:])[0, 1]
        ax.plot(t_days[test_s:], res['pi_sdre'][test_s:],
                alpha=0.8, lw=0.8, label=f'{name} (corr={corr:.3f})')
    ax.set_ylabel('Allocation $\\pi$')
    ax.set_title('SDRE Policy Comparison')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 3.0)

    # Panel 3: mu_hat comparison (if learning mu)
    ax = axes[2]
    true_mu = p.mu - p.r
    ax.axhline(true_mu, color='gray', ls='--', alpha=0.5, label=f'True $\\mu-r$={true_mu:.3f}')
    for name, res in results.items():
        mu_h = res['mu_hat'][test_s:]
        ax.plot(t_days[test_s:], mu_h, alpha=0.7, lw=0.7,
                label=f'{name} (mean={np.mean(mu_h):.4f})')
    ax.set_ylabel('$\\hat{\\mu} - r$')
    ax.set_xlabel('Years')
    ax.set_title('Drift Estimation')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    suffix = 'known_mu' if known_mu_mode else 'learned_mu'
    plt.savefig(f'kronic_pomdp/experiments/level4_generator_vs_kalman_{suffix}.png',
                bbox_inches='tight')
    plt.close()
    print(f"Saved: level4_generator_vs_kalman_{suffix}.png")

    # Print summary stats
    print(f"\n{'Method':<28} {'V_corr':>8} {'pi_corr':>8} {'pi_std':>8} {'V_MSE':>10}")
    print("-" * 68)
    for name, res in results.items():
        Vh = res['V_hat'][test_s:]
        Vt = V_true[test_s:]
        vc = np.corrcoef(Vh, Vt)[0, 1]
        pc = np.corrcoef(res['pi_sdre'][test_s:], pi_merton[test_s:])[0, 1]
        ps = np.std(res['pi_sdre'][test_s:])
        vm = np.mean((Vh - Vt)**2)
        print(f"{name:<28} {vc:>8.3f} {pc:>8.3f} {ps:>8.3f} {vm:>10.6f}")


def multi_seed_comparison(p, n_seeds=5, known_mu_mode=True):
    """Multi-seed comparison of all approaches."""
    U_p, U_pp = _make_crra(p.gamma)
    test_s = 1000
    known_mu = (p.mu - p.r) if known_mu_mode else None

    methods = ['Kalman(xi=0.3)', 'Kalman(xi=2.0)', 'Generator', 'Hybrid']
    v_corrs = {m: [] for m in methods}
    pi_corrs = {m: [] for m in methods}
    pi_stds = {m: [] for m in methods}

    for seed in range(0, n_seeds * 1000, 1000):
        # Kalman variants
        for xi_val, label in [(0.3, 'Kalman(xi=0.3)'), (2.0, 'Kalman(xi=2.0)')]:
            res = run_sdre_kalman(p, T, seed, U_p, U_pp, known_mu=known_mu, xi_kf=xi_val)
            V_true = res['state_true'][test_s:]
            pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)
            v_corrs[label].append(np.corrcoef(res['V_hat'][test_s:], V_true)[0, 1])
            pi_corrs[label].append(np.corrcoef(res['pi_sdre'][test_s:], pi_merton)[0, 1])
            pi_stds[label].append(np.std(res['pi_sdre'][test_s:]))

        # Generator (raw)
        res = run_sdre_generator(p, T, seed, U_p, U_pp, known_mu=known_mu)
        V_true = res['state_true'][test_s:]
        pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)
        v_corrs['Generator'].append(np.corrcoef(res['V_hat'][test_s:], V_true)[0, 1])
        pi_corrs['Generator'].append(np.corrcoef(res['pi_sdre'][test_s:], pi_merton)[0, 1])
        pi_stds['Generator'].append(np.std(res['pi_sdre'][test_s:]))

        # Hybrid (sig-filtered obs -> Kalman)
        res = run_sdre_hybrid(p, T, seed, U_p, U_pp, known_mu=known_mu, xi_kf=1.0)
        v_corrs['Hybrid'].append(np.corrcoef(res['V_hat'][test_s:], V_true)[0, 1])
        pi_corrs['Hybrid'].append(np.corrcoef(res['pi_sdre'][test_s:], pi_merton)[0, 1])
        pi_stds['Hybrid'].append(np.std(res['pi_sdre'][test_s:]))

    suffix = 'known_mu' if known_mu_mode else 'learned_mu'
    print(f"\nMulti-seed comparison ({n_seeds} seeds, {'known' if known_mu_mode else 'learned'} mu):")
    print(f"{'Method':<22} {'V_corr':>12} {'pi_corr':>12} {'pi_std':>12}")
    print("-" * 60)
    for m in methods:
        vc = f"{np.mean(v_corrs[m]):.3f}+/-{np.std(v_corrs[m]):.3f}"
        pc = f"{np.mean(pi_corrs[m]):.3f}+/-{np.std(pi_corrs[m]):.3f}"
        ps = f"{np.mean(pi_stds[m]):.3f}+/-{np.std(pi_stds[m]):.3f}"
        print(f"{m:<22} {vc:>12} {pc:>12} {ps:>12}")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(methods))
    w = 0.6

    ax = axes[0]
    means = [np.mean(v_corrs[m]) for m in methods]
    stds = [np.std(v_corrs[m]) for m in methods]
    ax.bar(x, means, w, yerr=stds, capsize=4, color=['C0', 'C1', 'C2', 'C3'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=9)
    ax.set_ylabel('corr($\\hat{V}$, $V_{true}$)')
    ax.set_title('V Estimation Quality')

    ax = axes[1]
    means = [np.mean(pi_corrs[m]) for m in methods]
    stds = [np.std(pi_corrs[m]) for m in methods]
    ax.bar(x, means, w, yerr=stds, capsize=4, color=['C0', 'C1', 'C2', 'C3'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=9)
    ax.set_ylabel('corr($\\pi_{SDRE}$, $\\pi_{Merton}$)')
    ax.set_title('Policy Quality')

    ax = axes[2]
    means = [np.mean(pi_stds[m]) for m in methods]
    stds_of_stds = [np.std(pi_stds[m]) for m in methods]
    ax.bar(x, means, w, yerr=stds_of_stds, capsize=4, color=['C0', 'C1', 'C2', 'C3'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=9)
    ax.set_ylabel('std($\\pi_{SDRE}$)')
    ax.set_title('Policy Volatility')

    plt.suptitle(f'Generator vs Kalman: {n_seeds} Seeds ({suffix.replace("_", " ")})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f'kronic_pomdp/experiments/level4_generator_bars_{suffix}.png',
                bbox_inches='tight')
    plt.close()
    print(f"Saved: level4_generator_bars_{suffix}.png")


if __name__ == '__main__':
    print("=== Generator-Based SDRE Diagnostics ===\n")

    # With known mu (isolates V estimation quality)
    print("--- Known mu: isolates V estimation ---")
    compare_approaches(P, seed=42, known_mu_mode=True)

    print("\n--- Known mu: multi-seed ---")
    multi_seed_comparison(P, n_seeds=5, known_mu_mode=True)

    # With learned mu (full pipeline)
    print("\n\n--- Learned mu: full pipeline ---")
    compare_approaches(P, seed=42, known_mu_mode=False)

    print("\n--- Learned mu: multi-seed ---")
    multi_seed_comparison(P, n_seeds=5, known_mu_mode=False)

    print("\n=== Done ===")
