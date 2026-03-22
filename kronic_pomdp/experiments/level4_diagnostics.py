"""
Level 4 SDRE Diagnostics — Visualizations and tuning exploration.

Generates:
  1. V_hat vs V_true time series (Kalman filter quality)
  2. pi_SDRE vs pi_Merton scatter + time series
  3. Multi-utility panel: three pi trajectories on same Heston path
  4. CARA wealth-dependence: pi vs 1/W scatter (vs CRRA)
  5. Wealth trajectories: SDRE vs Oracle vs Constant
  6. Tuning sweep: observation noise scaling R_scale
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

HestonParams = namedtuple('HestonParams',
    ['mu', 'r', 'kappa', 'theta', 'xi', 'rho', 'gamma', 'dt'])

P = HestonParams(mu=0.08, r=0.02, kappa=2.0, theta=0.04, xi=0.3,
                 rho=-0.5, gamma=2.0, dt=1/252)
T = 10000  # ~40 years


# --- Utility factories ---
def _make_crra(gamma):
    return (lambda W: W**(-gamma), lambda W: -gamma * W**(-gamma-1))

def _make_cara(alpha):
    return (lambda W: np.exp(-alpha * W), lambda W: -alpha * np.exp(-alpha * W))

def _make_log():
    return (lambda W: 1.0 / W, lambda W: -1.0 / W**2)


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


def run_sdre(p, T, seed, U_prime, U_double_prime, known_mu=None,
             kf_params=None, R_scale=1.0):
    """Single-path SDRE controller with Kalman filter on V.

    R_scale: multiplier on observation noise R. >1 = trust model more, <1 = trust obs more.
    """
    rng = np.random.RandomState(seed)

    kf_kappa = kf_params.get('kappa', 2.0) if kf_params else 2.0
    kf_theta = kf_params.get('theta', 0.04) if kf_params else 0.04
    kf_xi = kf_params.get('xi', 0.3) if kf_params else 0.3
    V_filt = kf_theta
    P_kf = kf_xi**2 * kf_theta * p.dt * 10

    ewma_mu = 0.0
    lam_mu = 0.999

    V_hat = np.zeros(T)
    pi_sdre = np.zeros(T)
    state_true = np.zeros(T)
    W_history = np.ones(T)
    K_gain = np.zeros(T)

    state_dict = {}
    W = 1.0
    sim_fn = _make_heston_sim(p)

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
        R_kf = R_scale * 2 * max(V_pred, 1e-6)**2 / p.dt

        # Kalman update
        K_kf_val = P_pred / (P_pred + R_kf)
        V_filt = V_pred + K_kf_val * (y_obs - V_pred)
        V_filt = max(V_filt, 1e-6)
        P_kf = (1 - K_kf_val) * P_pred

        V_hat[t] = V_filt
        K_gain[t] = K_kf_val

        # Drift
        ewma_mu = lam_mu * ewma_mu + (1 - lam_mu) * ret / p.dt
        mu_excess = known_mu if known_mu is not None else ewma_mu
        W_safe = max(W, 1e-8)

        # SDRE
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
        'W_history': W_history, 'K_gain': K_gain,
    }


def fig1_filter_and_policy(p, seed=42):
    """V_hat vs V_true + pi_SDRE vs pi_Merton time series."""
    U_p, U_pp = _make_crra(p.gamma)
    kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': p.xi}
    res = run_sdre(p, T, seed, U_p, U_pp, known_mu=p.mu - p.r, kf_params=kf)

    t_days = np.arange(T) / 252
    test_s = 500
    V_true = res['state_true']
    V_hat = res['V_hat']
    pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Volatility
    ax = axes[0]
    ax.plot(t_days[test_s:], np.sqrt(V_true[test_s:]), alpha=0.6, label='True $\\sqrt{V}$', lw=0.8)
    ax.plot(t_days[test_s:], np.sqrt(V_hat[test_s:]), alpha=0.8, label='Kalman $\\sqrt{\\hat{V}}$', lw=0.8)
    corr_v = np.corrcoef(V_hat[test_s:], V_true[test_s:])[0, 1]
    ax.set_ylabel('Volatility')
    ax.set_title(f'Kalman Filter Tracking (corr = {corr_v:.3f})')
    ax.legend(loc='upper right')

    # Panel 2: Policy
    ax = axes[1]
    ax.plot(t_days[test_s:], pi_merton[test_s:], alpha=0.5, label='Merton $\\pi^*$', lw=0.8)
    ax.plot(t_days[test_s:], res['pi_sdre'][test_s:], alpha=0.7, label='SDRE $\\pi^*$', lw=0.8)
    corr_pi = np.corrcoef(res['pi_sdre'][test_s:], pi_merton[test_s:])[0, 1]
    ax.set_ylabel('Allocation $\\pi$')
    ax.set_title(f'SDRE Policy vs Merton Ground Truth (corr = {corr_pi:.3f})')
    ax.legend(loc='upper right')

    # Panel 3: Kalman gain
    ax = axes[2]
    ax.plot(t_days[test_s:], res['K_gain'][test_s:], alpha=0.7, lw=0.8, color='green')
    ax.set_ylabel('Kalman Gain $K$')
    ax.set_xlabel('Years')
    ax.set_title(f'Kalman Gain (mean = {np.mean(res["K_gain"][test_s:]):.4f})')

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_filter_policy.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_filter_policy.png")


def fig2_scatter(p, seed=42):
    """Scatter: pi_SDRE vs pi_Merton for CRRA."""
    U_p, U_pp = _make_crra(p.gamma)
    kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': p.xi}
    res = run_sdre(p, T, seed, U_p, U_pp, known_mu=p.mu - p.r, kf_params=kf)

    test_s = 1000
    V_true = res['state_true'][test_s:]
    pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)
    pi_sdre = res['pi_sdre'][test_s:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: pi
    ax = axes[0]
    ax.scatter(pi_merton, pi_sdre, alpha=0.05, s=3, rasterized=True)
    lims = [0, min(3.0, max(pi_merton.max(), pi_sdre.max()) * 1.1)]
    ax.plot(lims, lims, 'r--', lw=1.5, label='perfect')
    ax.set_xlabel('Merton $\\pi^*$ (true V)')
    ax.set_ylabel('SDRE $\\pi^*$ (Kalman $\\hat{V}$)')
    corr = np.corrcoef(pi_sdre, pi_merton)[0, 1]
    ax.set_title(f'Policy Scatter (corr = {corr:.3f})')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Scatter: V
    ax = axes[1]
    V_hat = res['V_hat'][test_s:]
    ax.scatter(V_true, V_hat, alpha=0.05, s=3, rasterized=True)
    lims_v = [0, max(V_true.max(), V_hat.max()) * 1.1]
    ax.plot(lims_v, lims_v, 'r--', lw=1.5, label='perfect')
    ax.set_xlabel('True V')
    ax.set_ylabel('Kalman $\\hat{V}$')
    corr_v = np.corrcoef(V_hat, V_true)[0, 1]
    ax.set_title(f'Volatility Scatter (corr = {corr_v:.3f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_scatter.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_scatter.png")


def fig3_multi_utility(p, seed=42):
    """Three utilities on the same Heston path: different Q -> different pi."""
    kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': p.xi}

    utilities = {
        'CRRA($\\gamma$=2)': _make_crra(2.0),
        'Log': _make_log(),
        'CARA($\\alpha$=3)': _make_cara(3.0),
    }

    results = {}
    for name, (up, upp) in utilities.items():
        results[name] = run_sdre(p, T, seed, up, upp, known_mu=p.mu - p.r, kf_params=kf)

    t_days = np.arange(T) / 252
    test_s = 500

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: All three pi trajectories
    ax = axes[0]
    colors = {'CRRA($\\gamma$=2)': 'C0', 'Log': 'C1', 'CARA($\\alpha$=3)': 'C2'}
    for name, res in results.items():
        pi = res['pi_sdre'][test_s:]
        ax.plot(t_days[test_s:], pi, alpha=0.6, lw=0.7, label=f'{name} (mean={np.mean(pi):.2f})',
                color=colors[name])
    ax.set_ylabel('Allocation $\\pi$')
    ax.set_title('Same Heston Path, Three Utilities, Three Different Policies')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 3.5)

    # Panel 2: Wealth trajectories
    ax = axes[1]
    for name, res in results.items():
        ax.plot(t_days[test_s:], res['W_history'][test_s:], alpha=0.7, lw=0.8,
                label=name, color=colors[name])
    ax.set_ylabel('Wealth $W$')
    ax.set_title('Wealth Evolution Under Different Utilities')
    ax.legend(loc='upper left')
    ax.set_yscale('log')

    # Panel 3: True V (shared) + V_hat from one run
    ax = axes[2]
    res0 = list(results.values())[0]
    ax.plot(t_days[test_s:], np.sqrt(res0['state_true'][test_s:]),
            alpha=0.5, lw=0.7, color='gray', label='True $\\sqrt{V}$')
    ax.plot(t_days[test_s:], np.sqrt(res0['V_hat'][test_s:]),
            alpha=0.7, lw=0.7, color='black', label='Kalman $\\sqrt{\\hat{V}}$')
    ax.set_ylabel('Volatility')
    ax.set_xlabel('Years')
    ax.set_title('Shared Volatility Estimate (same for all utilities)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_multi_utility.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_multi_utility.png")


def fig4_wealth_dependence(p, seed=42):
    """CARA vs CRRA: scatter pi vs 1/W showing wealth-dependence."""
    kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': p.xi}

    res_crra = run_sdre(p, T, seed, *_make_crra(2.0), known_mu=p.mu - p.r, kf_params=kf)
    res_cara = run_sdre(p, T, seed, *_make_cara(3.0), known_mu=p.mu - p.r, kf_params=kf)

    test_s = 1000

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CRRA: pi vs 1/W (should be flat — no wealth dependence)
    ax = axes[0]
    W_crra = res_crra['W_history'][test_s:]
    pi_crra = res_crra['pi_sdre'][test_s:]
    ax.scatter(1.0 / W_crra, pi_crra, alpha=0.03, s=3, rasterized=True)
    corr_crra = np.corrcoef(pi_crra, 1.0 / W_crra)[0, 1]
    ax.set_xlabel('1/W')
    ax.set_ylabel('$\\pi_{CRRA}$')
    ax.set_title(f'CRRA: corr($\\pi$, 1/W) = {corr_crra:.3f}\n(wealth-independent)')

    # CARA: pi vs 1/W (should show positive relationship)
    ax = axes[1]
    W_cara = res_cara['W_history'][test_s:]
    pi_cara = res_cara['pi_sdre'][test_s:]
    ax.scatter(1.0 / W_cara, pi_cara, alpha=0.03, s=3, rasterized=True, color='C2')
    corr_cara = np.corrcoef(pi_cara, 1.0 / W_cara)[0, 1]
    # Fit line
    z = np.polyfit(1.0 / W_cara, pi_cara, 1)
    x_fit = np.linspace(min(1.0 / W_cara), max(1.0 / W_cara), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=2, alpha=0.8)
    ax.set_xlabel('1/W')
    ax.set_ylabel('$\\pi_{CARA}$')
    ax.set_title(f'CARA: corr($\\pi$, 1/W) = {corr_cara:.3f}\n(wealth-dependent)')

    plt.suptitle('Wealth-Dependence Test: Changing Q Captures Utility Differences',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_wealth_dependence.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_wealth_dependence.png")


def fig5_performance(p, n_seeds=5):
    """Wealth trajectories: SDRE vs Oracle vs Constant (5 seeds)."""
    kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': p.xi}
    U_p, U_pp = _make_crra(p.gamma)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    terminal_wealth = {'SDRE-KF': [], 'Oracle': [], 'Constant': [], 'EWMA': []}

    for si, seed in enumerate(range(0, n_seeds * 1000, 1000)):
        # SDRE with Kalman filter
        res = run_sdre(p, T, seed, U_p, U_pp, known_mu=p.mu - p.r, kf_params=kf)

        # Oracle and Constant: replay same returns
        rng = np.random.RandomState(seed)
        W_oracle = 1.0
        W_const = 1.0
        W_ewma = 1.0
        pi_const = (p.mu - p.r) / (p.gamma * p.theta)

        W_o_hist = np.ones(T)
        W_c_hist = np.ones(T)
        W_e_hist = np.ones(T)
        ewma_v = p.theta

        state_dict = {}
        sim_fn = _make_heston_sim(p)
        rng2 = np.random.RandomState(seed)

        for t in range(T):
            z1 = rng2.randn()
            z2 = rng2.randn()
            V_true_t, ret = sim_fn(t, z1, z2, p.dt, state_dict)

            # Oracle: knows true V
            pi_oracle = np.clip((p.mu - p.r) / (p.gamma * V_true_t), 0.01, 5.0)
            W_oracle *= (1 + p.r * p.dt + pi_oracle * ret)
            W_oracle = max(W_oracle, 1e-8)
            W_o_hist[t] = W_oracle

            # Constant
            W_const *= (1 + p.r * p.dt + pi_const * ret)
            W_const = max(W_const, 1e-8)
            W_c_hist[t] = W_const

            # EWMA
            ewma_v = 0.99 * ewma_v + 0.01 * ret**2 / p.dt
            pi_ewma = np.clip((p.mu - p.r) / (p.gamma * max(ewma_v, 1e-6)), 0.01, 5.0)
            W_ewma *= (1 + p.r * p.dt + pi_ewma * ret)
            W_ewma = max(W_ewma, 1e-8)
            W_e_hist[t] = W_ewma

        terminal_wealth['SDRE-KF'].append(res['W_history'][-1])
        terminal_wealth['Oracle'].append(W_o_hist[-1])
        terminal_wealth['Constant'].append(W_c_hist[-1])
        terminal_wealth['EWMA'].append(W_e_hist[-1])

        if si < 6:
            ax = axes[si // 3, si % 3]
            t_days = np.arange(T) / 252
            ax.plot(t_days, W_o_hist, alpha=0.7, lw=0.8, label='Oracle')
            ax.plot(t_days, res['W_history'], alpha=0.7, lw=0.8, label='SDRE-KF')
            ax.plot(t_days, W_e_hist, alpha=0.7, lw=0.8, label='EWMA')
            ax.plot(t_days, W_c_hist, alpha=0.7, lw=0.8, label='Constant')
            ax.set_yscale('log')
            ax.set_xlabel('Years')
            ax.set_ylabel('Wealth')
            ax.set_title(f'Seed {seed}')
            if si == 0:
                ax.legend(fontsize=8)

    # Last panel: bar chart of CE
    ax = axes[1, 2]
    ce = {}
    for name, ws in terminal_wealth.items():
        # CE = ((1-gamma)*E[U])^(1/(1-gamma)) where U = W^(1-gamma)/(1-gamma)
        utils = [w**(1 - p.gamma) / (1 - p.gamma) for w in ws]
        ce[name] = ((1 - p.gamma) * np.mean(utils))**(1 / (1 - p.gamma))

    bars = list(ce.keys())
    vals = [ce[b] for b in bars]
    colors = ['C0', 'C1', 'C2', 'C3']
    ax.bar(bars, vals, color=colors, alpha=0.7)
    ax.set_ylabel('Certainty Equivalent')
    ax.set_title(f'CE ({n_seeds} seeds)')
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('SDRE-Kalman Performance vs Baselines (CRRA $\\gamma$=2)', fontsize=13)
    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_performance.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_performance.png")

    print("\nTerminal CE:")
    for name, v in ce.items():
        print(f"  {name}: {v:.4f}")


def fig6_tuning_xi(p, seed=42):
    """Sweep xi_kf (process noise / prior uncertainty) to see its effect.

    Higher xi_kf = "I'm less sure about my CIR dynamics" = higher Kalman gain.
    The true xi is 0.3. We test inflating it to make the filter more responsive.
    """
    U_p, U_pp = _make_crra(p.gamma)
    test_s = 1000

    xi_vals = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    v_corrs = []
    pi_corrs = []
    pi_stds = []
    k_gains = []
    v_mses = []

    for xi_kf in xi_vals:
        kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': xi_kf}
        res = run_sdre(p, T, seed, U_p, U_pp, known_mu=p.mu - p.r, kf_params=kf)
        V_true = res['state_true'][test_s:]
        V_hat = res['V_hat'][test_s:]
        pi_merton = np.clip((p.mu - p.r) / (p.gamma * V_true), 0.01, 5.0)

        v_corrs.append(np.corrcoef(V_hat, V_true)[0, 1])
        v_mses.append(np.mean((V_hat - V_true)**2))
        pi_corrs.append(np.corrcoef(res['pi_sdre'][test_s:], pi_merton)[0, 1])
        pi_stds.append(np.std(res['pi_sdre'][test_s:]))
        k_gains.append(np.mean(res['K_gain'][test_s:]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.semilogx(xi_vals, v_corrs, 'o-', lw=2)
    ax.axvline(p.xi, color='gray', ls='--', alpha=0.5, label=f'true $\\xi$={p.xi}')
    ax.set_xlabel('$\\xi_{KF}$ (process noise prior)')
    ax.set_ylabel('corr($\\hat{V}$, $V_{true}$)')
    ax.set_title('V Estimation: Correlation')
    ax.legend()

    ax = axes[0, 1]
    ax.semilogx(xi_vals, pi_corrs, 's-', lw=2, color='C1')
    ax.axvline(p.xi, color='gray', ls='--', alpha=0.5, label=f'true $\\xi$={p.xi}')
    ax.set_xlabel('$\\xi_{KF}$')
    ax.set_ylabel('corr($\\pi_{SDRE}$, $\\pi_{Merton}$)')
    ax.set_title('Policy Quality')
    ax.legend()

    ax = axes[1, 0]
    ax.semilogx(xi_vals, pi_stds, '^-', lw=2, color='C2')
    # Reference: oracle pi_std
    kf_true = {'kappa': p.kappa, 'theta': p.theta, 'xi': p.xi}
    res_ref = run_sdre(p, T, seed, U_p, U_pp, known_mu=p.mu - p.r, kf_params=kf_true)
    V_true_ref = res_ref['state_true'][test_s:]
    pi_oracle_std = np.std(np.clip((p.mu - p.r) / (p.gamma * V_true_ref), 0.01, 5.0))
    ax.axhline(pi_oracle_std, color='red', ls=':', alpha=0.7, label=f'Oracle $\\pi$ std={pi_oracle_std:.2f}')
    ax.axvline(p.xi, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('$\\xi_{KF}$')
    ax.set_ylabel('std($\\pi_{SDRE}$)')
    ax.set_title('Policy Volatility')
    ax.legend()

    ax = axes[1, 1]
    ax.semilogx(xi_vals, k_gains, 'D-', lw=2, color='C3')
    ax.axvline(p.xi, color='gray', ls='--', alpha=0.5, label=f'true $\\xi$={p.xi}')
    ax.set_xlabel('$\\xi_{KF}$')
    ax.set_ylabel('Mean Kalman Gain')
    ax.set_title('Kalman Gain (higher = trusts obs more)')
    ax.legend()

    plt.suptitle('Tuning: Process Noise Prior $\\xi_{KF}$ (inflating = less conservative filter)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_tuning_xi.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_tuning_xi.png")

    print(f"\nxi_kf sweep (true xi = {p.xi}):")
    print(f"{'xi_kf':>8} {'V_corr':>8} {'V_MSE':>10} {'pi_corr':>8} {'pi_std':>8} {'K_gain':>8}")
    for i, xi in enumerate(xi_vals):
        print(f"{xi:>8.2f} {v_corrs[i]:>8.3f} {v_mses[i]:>10.6f} {pi_corrs[i]:>8.3f} "
              f"{pi_stds[i]:>8.3f} {k_gains[i]:>8.4f}")

    # Find the best xi by pi_corr
    best_idx = np.argmax(pi_corrs)
    print(f"\nBest pi_corr: xi_kf={xi_vals[best_idx]:.2f} -> "
          f"V_corr={v_corrs[best_idx]:.3f}, pi_corr={pi_corrs[best_idx]:.3f}")


def fig7_xi_timeseries(p, seed=42):
    """Compare V_hat and pi time series for conservative vs tuned xi."""
    U_p, U_pp = _make_crra(p.gamma)
    test_s = 500
    t_days = np.arange(T) / 252

    xi_configs = {
        f'$\\xi_{{KF}}$={p.xi} (true)': p.xi,
        '$\\xi_{KF}$=1.0': 1.0,
        '$\\xi_{KF}$=2.0': 2.0,
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Run all configs
    results = {}
    for label, xi_kf in xi_configs.items():
        kf = {'kappa': p.kappa, 'theta': p.theta, 'xi': xi_kf}
        results[label] = run_sdre(p, T, seed, U_p, U_pp, known_mu=p.mu - p.r, kf_params=kf)

    # Panel 1: V_hat comparison
    ax = axes[0]
    res0 = list(results.values())[0]
    ax.plot(t_days[test_s:], np.sqrt(res0['state_true'][test_s:]),
            alpha=0.3, lw=0.5, color='gray', label='True $\\sqrt{V}$')
    for label, res in results.items():
        corr = np.corrcoef(res['V_hat'][test_s:], res['state_true'][test_s:])[0, 1]
        ax.plot(t_days[test_s:], np.sqrt(res['V_hat'][test_s:]),
                alpha=0.8, lw=1.2, label=f'{label} (corr={corr:.3f})')
    ax.set_ylabel('Volatility $\\sqrt{V}$')
    ax.set_title('Kalman Filter: Effect of Process Noise Prior on Tracking')
    ax.legend(loc='upper right')

    # Panel 2: pi comparison
    ax = axes[1]
    pi_oracle = np.clip((p.mu - p.r) / (p.gamma * res0['state_true'][test_s:]), 0.01, 5.0)
    ax.plot(t_days[test_s:], pi_oracle, alpha=0.3, lw=0.5, color='gray', label='Merton $\\pi^*$')
    for label, res in results.items():
        corr = np.corrcoef(res['pi_sdre'][test_s:], pi_oracle)[0, 1]
        ax.plot(t_days[test_s:], res['pi_sdre'][test_s:],
                alpha=0.8, lw=1.0, label=f'{label} (corr={corr:.3f})')
    ax.set_ylabel('Allocation $\\pi$')
    ax.set_xlabel('Years')
    ax.set_title('SDRE Policy: More Responsive Filter = Better Tracking')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 3.0)

    plt.tight_layout()
    plt.savefig('kronic_pomdp/experiments/level4_xi_comparison.png', bbox_inches='tight')
    plt.close()
    print("Saved: level4_xi_comparison.png")


if __name__ == '__main__':
    print("=== Level 4 SDRE Diagnostics ===\n")

    print("--- Fig 1: Filter + Policy Time Series ---")
    fig1_filter_and_policy(P)

    print("\n--- Fig 2: Scatter Plots ---")
    fig2_scatter(P)

    print("\n--- Fig 3: Multi-Utility Comparison ---")
    fig3_multi_utility(P)

    print("\n--- Fig 4: Wealth Dependence ---")
    fig4_wealth_dependence(P)

    print("\n--- Fig 5: Performance vs Baselines ---")
    fig5_performance(P)

    print("\n--- Fig 6: xi_kf Tuning Sweep ---")
    fig6_tuning_xi(P)

    print("\n--- Fig 7: xi Time Series Comparison ---")
    fig7_xi_timeseries(P)

    print("\n=== Done ===")
