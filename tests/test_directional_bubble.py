"""
Directional Bubble Detection Test.

Tests whether the directional α test can detect bubbles that are hidden
in a portfolio direction, not visible per-asset.

Key DGP: Rotated bubble.
  Y1 = CEV β=2.5 (bubble), Y2 = GBM (no bubble)
  X = R(θ) · Y  →  neither X1 nor X2 is obviously a bubble
  But the directional test along w* ≈ R^{-1} e1 should recover α > 2.

Theory (§8 of theory_eigenfunction_bubble_detection.md):
  For portfolio w, σ²_w(z) = w^T Σ(x) w where z = w^T x.
  Feller: bubble along w ⟺ ∫^∞ z/σ²_w(z) dz < ∞ ⟺ α_w > 2.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kronic_pomdp', 'experiments'))
from cdc_kernel_estimators import MultivariateCdCEstimator


# --- Multi-asset DGP simulators (exponential Euler) ---

def simulate_correlated_gbm(S0, mu, sigma, rho, T, dt, seed=42):
    """2D correlated GBM. No bubble (α=2 for both assets)."""
    rng = np.random.RandomState(seed)
    n_steps = int(T / dt)
    d = len(S0)
    S = np.zeros((n_steps + 1, d))
    S[0] = S0
    sqrt_dt = np.sqrt(dt)

    # Cholesky of correlation matrix
    C = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(C)

    for t in range(n_steps):
        Z = L @ rng.randn(d)
        for i in range(d):
            S[t+1, i] = S[t, i] * np.exp(
                (mu[i] - 0.5 * sigma[i]**2) * dt + sigma[i] * sqrt_dt * Z[i])
    return S


def simulate_multi_cev(S0, mu, c, beta, rho, T, dt, seed=42):
    """2D correlated CEV. Bubble if any β_i > 2."""
    rng = np.random.RandomState(seed)
    n_steps = int(T / dt)
    d = len(S0)
    S = np.zeros((n_steps + 1, d))
    S[0] = S0
    sqrt_dt = np.sqrt(dt)

    C = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(C)

    for t in range(n_steps):
        Z = L @ rng.randn(d)
        for i in range(d):
            s = S[t, i]
            vol_log = c[i] * s ** (beta[i] / 2 - 1)
            drift_log = mu[i] - 0.5 * c[i]**2 * s ** (beta[i] - 2)
            S[t+1, i] = s * np.exp(drift_log * dt + vol_log * sqrt_dt * Z[i])
    return S


def simulate_separable_sv_cev(X0=100.0, Y0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                               c=0.02, beta_slope=25.0, rho=-0.5,
                               T=100.0, dt=0.01, seed=42):
    """Separable SV where the CEV exponent depends on vol level.

    dY = kappa*(theta - Y)*dt + xi*sqrt(Y)*dW2       (CIR vol)
    dX = mu*X*dt + c * X^(beta(Y)/2) * dW1           (CEV with level-dependent β)

    where beta(Y) = 2 + beta_slope * max(0, Y - theta)

    At Y = theta: β = 2.0 (Feller boundary)
    At Y > theta: β > 2 (bubble regime — strict local martingale)
    At Y < theta: β = 2.0 (no bubble)

    The CIR spends ~50% of time above theta, so:
    - Marginal test averages across regimes → α ≈ 2 + noise (ambiguous)
    - Conditional Feller in high-Y bins → clear α > 2

    When beta_slope = 0: no bubble at any vol level (control).

    Returns (n_steps+1, 2) array with columns [X, Y].
    """
    rng = np.random.RandomState(seed)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    X = np.zeros((n_steps + 1, 2))
    X[0, 0] = X0
    X[0, 1] = Y0

    C = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(C)

    mu = 0.02

    for t in range(n_steps):
        Z = L @ rng.randn(2)
        x, y = X[t, 0], X[t, 1]
        y = max(y, 1e-8)

        # CIR vol process
        y_new = y + kappa * (theta - y) * dt + xi * np.sqrt(y) * sqrt_dt * Z[1]
        y_new = max(y_new, 1e-8)

        # Level-dependent CEV exponent
        beta_y = 2.0 + beta_slope * max(0.0, y - theta)

        # CEV price (exponential Euler)
        vol_log = c * x ** (beta_y / 2 - 1)
        drift_log = mu - 0.5 * c**2 * x ** (beta_y - 2)
        x_new = x * np.exp(drift_log * dt + vol_log * sqrt_dt * Z[0])

        X[t + 1, 0] = x_new
        X[t + 1, 1] = y_new

    return X


def simulate_rotated_bubble(S0_bubble, S0_gbm, mu_b, c_b, beta_b,
                             mu_g, sigma_g, theta, rho, T, dt, seed=42):
    """
    Rotated bubble: Y1=CEV (bubble), Y2=GBM (no bubble), X = R(θ)·Y.

    In the X basis, neither asset is obviously a bubble.
    The directional test along w* ≈ R^{-1} e1 should recover α > 2.

    Args:
        theta: Rotation angle in radians (0 = no rotation = trivial)
        rho: Correlation between the two Brownian motions
    """
    rng = np.random.RandomState(seed)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    # Simulate Y (natural basis)
    Y = np.zeros((n_steps + 1, 2))
    Y[0, 0] = S0_bubble
    Y[0, 1] = S0_gbm

    C = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(C)

    for t in range(n_steps):
        Z = L @ rng.randn(2)
        # Y1: CEV (bubble)
        s = Y[t, 0]
        vol_log = c_b * s ** (beta_b / 2 - 1)
        drift_log = mu_b - 0.5 * c_b**2 * s ** (beta_b - 2)
        Y[t+1, 0] = s * np.exp(drift_log * dt + vol_log * sqrt_dt * Z[0])

        # Y2: GBM (no bubble)
        Y[t+1, 1] = Y[t, 1] * np.exp(
            (mu_g - 0.5 * sigma_g**2) * dt + sigma_g * sqrt_dt * Z[1])

    # Rotate: X = R(θ) · Y
    R = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
    X = Y @ R.T  # (n_steps+1, 2) @ (2, 2) = (n_steps+1, 2)

    return X, Y, R


# --- Main test ---

def run_dgp(name, X, dt, true_bubble, true_bubble_direction=None, verbose=True):
    """Run per-asset and directional α tests."""
    print(f"\n  [{name}] N={len(X)-1}, d={X.shape[1]}, TRUE: {'BUBBLE' if true_bubble else 'NO BUBBLE'}")

    est = MultivariateCdCEstimator(n_landmarks=100)
    est.fit(X, dt=dt)

    # Per-asset test
    per_asset = est.alpha_per_asset()
    if verbose:
        for i in range(X.shape[1]):
            print(f"    Asset {i}: α = {per_asset['alpha_means'][i]:.2f} "
                  f"± {per_asset['alpha_sds'][i]:.2f}, "
                  f"P(bubble) = {per_asset['p_bubbles'][i]:.3f}")

    # Directional test
    dir_res = est.directional_alpha_test(n_directions=36)
    if verbose:
        print(f"    Directional: α_max = {dir_res['alpha_max']:.2f} "
              f"± {dir_res['alpha_max_sd']:.2f}, "
              f"w* = [{dir_res['w_star'][0]:.3f}, {dir_res['w_star'][1]:.3f}], "
              f"P(bubble) = {dir_res['p_bubble']:.3f}")
        if true_bubble_direction is not None:
            alignment = abs(np.dot(dir_res['w_star'], true_bubble_direction))
            print(f"    Alignment with true bubble direction: {alignment:.3f}")

    # Classification — both use theoretical threshold P(bubble) > 0.5
    any_marginal_bubble = np.any(per_asset['p_bubbles'] > 0.5)
    directional_bubble = dir_res['p_bubble'] > 0.5

    correct_marginal = any_marginal_bubble == true_bubble
    correct_directional = directional_bubble == true_bubble

    print(f"    → Marginal: {'CORRECT' if correct_marginal else 'WRONG'} | "
          f"Directional: {'CORRECT' if correct_directional else 'WRONG'}")

    return {
        'name': name,
        'true_bubble': true_bubble,
        'per_asset': per_asset,
        'directional': dir_res,
        'correct_marginal': correct_marginal,
        'correct_directional': correct_directional,
    }


def main():
    print("=" * 70)
    print("Directional Bubble Detection Test")
    print("Feller along portfolio w: bubble ⟺ α_w > 2")
    print("=" * 70)

    dt = 0.01
    T = 100.0
    seeds = [42, 123, 7]

    all_results = []

    # --- DGP 1: Correlated GBM (no bubble) ---
    dgp_name = "Corr GBM"
    print(f"\n{'='*70}\nDGP: {dgp_name} — NO BUBBLE\n{'='*70}")
    for seed in seeds:
        X = simulate_correlated_gbm(
            S0=[100, 100], mu=[0.05, 0.03], sigma=[0.3, 0.2],
            rho=0.5, T=T, dt=dt, seed=seed)
        res = run_dgp(f"{dgp_name} s={seed}", X, dt, true_bubble=False)
        all_results.append(res)

    # --- DGP 2: CEV β=2.5 on asset 1 (marginal bubble, easy) ---
    dgp_name = "CEV β=2.5 + GBM"
    print(f"\n{'='*70}\nDGP: {dgp_name} — BUBBLE (marginal)\n{'='*70}")
    for seed in seeds:
        X = simulate_multi_cev(
            S0=[100, 100], mu=[0.02, 0.03], c=[0.02, 0.3],
            beta=[2.5, 2.0], rho=0.3, T=T, dt=dt, seed=seed)
        res = run_dgp(f"{dgp_name} s={seed}", X, dt, true_bubble=True,
                       true_bubble_direction=np.array([1.0, 0.0]))
        all_results.append(res)

    # --- DGP 3: Rotated bubble θ=π/4 (hidden, the key test) ---
    dgp_name = "Rotated θ=π/4"
    theta = np.pi / 4
    R_inv_e1 = np.array([np.cos(theta), np.sin(theta)])  # true bubble direction in X space
    print(f"\n{'='*70}\nDGP: {dgp_name} — BUBBLE (hidden in direction [{R_inv_e1[0]:.2f}, {R_inv_e1[1]:.2f}])\n{'='*70}")
    for seed in seeds:
        X, Y, R = simulate_rotated_bubble(
            S0_bubble=100, S0_gbm=100,
            mu_b=0.02, c_b=0.02, beta_b=2.5,
            mu_g=0.03, sigma_g=0.3,
            theta=theta, rho=0.0, T=T, dt=dt, seed=seed)
        res = run_dgp(f"{dgp_name} s={seed}", X, dt, true_bubble=True,
                       true_bubble_direction=R_inv_e1)
        all_results.append(res)

    # --- DGP 4: Rotated bubble θ=π/6 (less rotation) ---
    dgp_name = "Rotated θ=π/6"
    theta = np.pi / 6
    R_inv_e1 = np.array([np.cos(theta), np.sin(theta)])
    print(f"\n{'='*70}\nDGP: {dgp_name} — BUBBLE (hidden in direction [{R_inv_e1[0]:.2f}, {R_inv_e1[1]:.2f}])\n{'='*70}")
    for seed in seeds:
        X, Y, R = simulate_rotated_bubble(
            S0_bubble=100, S0_gbm=100,
            mu_b=0.02, c_b=0.02, beta_b=2.5,
            mu_g=0.03, sigma_g=0.3,
            theta=theta, rho=0.0, T=T, dt=dt, seed=seed)
        res = run_dgp(f"{dgp_name} s={seed}", X, dt, true_bubble=True,
                       true_bubble_direction=R_inv_e1)
        all_results.append(res)

    # --- DGP 5: Rotated bubble θ=π/4, stronger bubble β=3.0 ---
    dgp_name = "Rotated β=3 θ=π/4"
    theta = np.pi / 4
    R_inv_e1 = np.array([np.cos(theta), np.sin(theta)])
    print(f"\n{'='*70}\nDGP: {dgp_name} — BUBBLE (strong, hidden)\n{'='*70}")
    for seed in seeds:
        X, Y, R = simulate_rotated_bubble(
            S0_bubble=100, S0_gbm=100,
            mu_b=0.02, c_b=0.01, beta_b=3.0,
            mu_g=0.03, sigma_g=0.3,
            theta=theta, rho=0.0, T=T, dt=dt, seed=seed)
        res = run_dgp(f"{dgp_name} s={seed}", X, dt, true_bubble=True,
                       true_bubble_direction=R_inv_e1)
        all_results.append(res)

    # --- DGP 6: Separable SV bubble (level-dependent α) ---
    # CEV exponent β(Y) = 2 + 30·max(0, Y-θ). Above θ: bubble regime.
    # Marginal test sees mixture → ambiguous. Conditional Feller bins by Y → catches it.
    dgp_name = "SepSV bubble"
    print(f"\n{'='*70}\nDGP: {dgp_name} — BUBBLE (α > 2 at high vol)\n{'='*70}")
    for seed in seeds:
        X_sv = simulate_separable_sv_cev(
            beta_slope=30.0, T=T, dt=dt, seed=seed)
        res = run_dgp(f"{dgp_name} s={seed}", X_sv, dt, true_bubble=True)

        # Conditional Feller test — should detect α > 2 in high-vol bins
        est = MultivariateCdCEstimator(n_landmarks=100)
        est.fit(X_sv, dt=dt)
        cond_res = est.conditional_feller_test(
            price_idx=0, vol_proxy_idx=1, n_vol_bins=5, n_landmarks_1d=80)
        print(f"    Conditional Feller: α_max = {cond_res['alpha_max']:.2f} "
              f"± {cond_res['alpha_max_sd']:.2f}, "
              f"P(bubble) = {cond_res['p_bubble']:.3f}, "
              f"α_weighted = {cond_res['alpha_weighted']:.2f} "
              f"± {cond_res['sd_weighted']:.2f}")
        for b in range(len(cond_res['alpha_per_bin'])):
            print(f"      Bin {b}: α = {cond_res['alpha_per_bin'][b]:.2f} "
                  f"± {cond_res['sd_per_bin'][b]:.2f}, "
                  f"P(α>2) = {cond_res['p_bubble_per_bin'][b]:.3f}, "
                  f"n = {cond_res['vol_bin_counts'][b]}")

        cond_correct = (cond_res['p_bubble'] > 0.5) == True
        res['cond_bubble'] = cond_res['p_bubble'] > 0.5
        res['correct_conditional'] = cond_correct
        print(f"    → Conditional: {'CORRECT' if cond_correct else 'WRONG'}")
        all_results.append(res)

    # --- DGP 7: Separable SV no bubble (constant β=2) ---
    dgp_name = "SepSV no bub"
    print(f"\n{'='*70}\nDGP: {dgp_name} — NO BUBBLE (β=2 at all vol levels)\n{'='*70}")
    for seed in seeds:
        X_sv = simulate_separable_sv_cev(
            beta_slope=0.0, T=T, dt=dt, seed=seed)
        res = run_dgp(f"{dgp_name} s={seed}", X_sv, dt, true_bubble=False)

        # Conditional Feller test — should see α ≈ 2 in all bins
        est = MultivariateCdCEstimator(n_landmarks=100)
        est.fit(X_sv, dt=dt)
        cond_res = est.conditional_feller_test(
            price_idx=0, vol_proxy_idx=1, n_vol_bins=5, n_landmarks_1d=80)
        print(f"    Conditional Feller: α_max = {cond_res['alpha_max']:.2f} "
              f"± {cond_res['alpha_max_sd']:.2f}, "
              f"P(bubble) = {cond_res['p_bubble']:.3f}, "
              f"α_weighted = {cond_res['alpha_weighted']:.2f} "
              f"± {cond_res['sd_weighted']:.2f}")
        for b in range(len(cond_res['alpha_per_bin'])):
            print(f"      Bin {b}: α = {cond_res['alpha_per_bin'][b]:.2f} "
                  f"± {cond_res['sd_per_bin'][b]:.2f}, "
                  f"P(α>2) = {cond_res['p_bubble_per_bin'][b]:.3f}, "
                  f"n = {cond_res['vol_bin_counts'][b]}")

        cond_correct = (cond_res['p_bubble'] > 0.5) == False
        res['cond_bubble'] = cond_res['p_bubble'] > 0.5
        res['correct_conditional'] = cond_correct
        print(f"    → Conditional: {'CORRECT' if cond_correct else 'WRONG'}")
        all_results.append(res)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'DGP':<25} {'True':>6} {'Marginal':>9} {'Direction':>10} {'Conditional':>12}")
    print("-" * 67)

    # Group by DGP name (3 seeds each)
    dgp_groups = {}
    for r in all_results:
        base_name = r['name'].rsplit(' s=', 1)[0]
        if base_name not in dgp_groups:
            dgp_groups[base_name] = []
        dgp_groups[base_name].append(r)

    for dgp_name, results in dgp_groups.items():
        label = "BUBBLE" if results[0]['true_bubble'] else "no bub"
        n_m = sum(r['correct_marginal'] for r in results)
        n_d = sum(r['correct_directional'] for r in results)
        n = len(results)
        has_cond = 'correct_conditional' in results[0]
        if has_cond:
            n_c = sum(r['correct_conditional'] for r in results)
            cond_str = f"   {n_c}/{n}"
        else:
            cond_str = "       —"
        print(f"{dgp_name:<25} {label:>6} {n_m}/{n:>7} {n_d}/{n:>9} {cond_str:>12}")

    # Overall
    n_total = len(all_results)
    n_m_total = sum(r['correct_marginal'] for r in all_results)
    n_d_total = sum(r['correct_directional'] for r in all_results)
    has_cond_results = [r for r in all_results if 'correct_conditional' in r]
    n_c_total = sum(r['correct_conditional'] for r in has_cond_results)
    n_c_denom = len(has_cond_results)
    cond_summary = f", Conditional {n_c_total}/{n_c_denom}" if n_c_denom > 0 else ""
    print(f"\n  Total: Marginal {n_m_total}/{n_total}, Directional {n_d_total}/{n_total}{cond_summary}")

    return all_results


if __name__ == "__main__":
    results = main()
