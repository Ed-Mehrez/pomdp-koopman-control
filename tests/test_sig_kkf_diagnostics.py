"""
Diagnostics for SigKKF vs EDMD on SABR.
Key questions:
1. Why does EDMD fail? (R²(S,V) too low → noisy σ̂²)
2. Why is KKF α biased low? (2.02 vs 3.0 for CEV β=1.5)
3. Does the 2D conditioning actually help vs 1D?
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kronic_pomdp', 'experiments'))
from gp_bubble_detector import SigKKFFellerGP, EDMDSigFellerGP, FellerGP


def simulate_cev(beta, n_steps=15000, dt=0.01, sigma0=0.3, S0=1.0,
                 seed=42):
    rng = np.random.RandomState(seed)
    S = np.zeros(n_steps)
    S[0] = S0
    mu = 0.05
    for t in range(1, n_steps):
        vol = sigma0 * max(S[t-1], 1e-6) ** beta
        S[t] = S[t-1] + mu * S[t-1] * dt + vol * np.sqrt(dt) * rng.randn()
        S[t] = max(S[t], 1e-6)
    dS = np.diff(S)
    dS = np.append(dS, dS[-1])
    return S, dS, dt


def simulate_sabr(beta, n_steps=15000, dt=0.01, sigma0=0.3, nu=0.5,
                  rho=-0.5, S0=1.0, seed=42):
    rng = np.random.RandomState(seed)
    S = np.zeros(n_steps)
    V = np.zeros(n_steps)
    S[0] = S0
    V[0] = sigma0

    for t in range(1, n_steps):
        z1 = rng.randn()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.randn()
        v = V[t-1]
        s = max(S[t-1], 1e-6)
        S[t] = s + v * s**beta * np.sqrt(dt) * z1
        V[t] = v * np.exp(-0.5 * nu**2 * dt + nu * np.sqrt(dt) * z2)
        S[t] = max(S[t], 1e-6)

    dS = np.diff(S)
    dS = np.append(dS, dS[-1])
    return S, dS, dt, V


def main():
    print("=" * 70)
    print("  Diagnostics: SigKKF vs EDMD")
    print("=" * 70)

    # ── Test 1: Why is CEV α biased low? ──
    print("\n── CEV β=1.5: comparing SigKKF vs FellerGP ──")
    S, dS, dt = simulate_cev(1.5, n_steps=15000, seed=42)

    # Standard FellerGP (known good)
    fgp = FellerGP(n_landmarks=80)
    fgp.fit(S, dS, dt)
    print(f"  FellerGP:    α={fgp.alpha_mean:.3f}±{fgp.alpha_sd:.3f}")

    # SigKKF
    skkf = SigKKFFellerGP(n_nystrom=50, n_gp_landmarks=60)
    skkf.fit(S, dS, dt)
    print(f"  SigKKF:      α={skkf.alpha_mean:.3f}±{skkf.alpha_sd:.3f}")
    print(f"    has_sv={skkf._has_sv}, R²={skkf._r2_sv:.3f}")

    # Check: is the KKF σ̂² correct?
    print(f"    σ̂² range: [{skkf._sigma2_hat.min():.4f}, "
          f"{skkf._sigma2_hat.max():.4f}]")
    print(f"    σ²_log range: [{skkf._sigma2_log.min():.6f}, "
          f"{skkf._sigma2_log.max():.6f}]")

    # True σ² for CEV: σ₀² S^{2β}
    sigma0 = 0.3
    beta = 1.5
    true_sigma2 = sigma0**2 * S[200:]**( 2*beta)
    kkf_sigma2 = skkf._sigma2_hat
    # Align lengths
    n_cmp = min(len(true_sigma2), len(kkf_sigma2))
    corr = np.corrcoef(true_sigma2[:n_cmp], kkf_sigma2[:n_cmp])[0, 1]
    ratio = np.median(kkf_sigma2[:n_cmp]) / np.median(true_sigma2[:n_cmp])
    print(f"    σ̂² vs true: corr={corr:.3f}, median ratio={ratio:.3f}")

    # ── Test 2: EDMD failure analysis ──
    print("\n── SABR β=1.5: EDMD failure analysis ──")
    S, dS, dt, V = simulate_sabr(1.5, n_steps=15000, seed=42)

    edmd = EDMDSigFellerGP(n_nystrom=50, n_gp_landmarks=60)
    edmd.fit(S, dS, dt)

    print(f"  EDMD σ̂² range: [{edmd._sigma2_hat.min():.4f}, "
          f"{edmd._sigma2_hat.max():.4f}]")
    print(f"  EDMD σ̂² > 0 frac: "
          f"{np.mean(edmd._sigma2_hat > 1e-10):.3f}")

    # Compare KKF
    skkf2 = SigKKFFellerGP(n_nystrom=50, n_gp_landmarks=60)
    skkf2.fit(S, dS, dt)
    print(f"\n  KKF σ̂² range: [{skkf2._sigma2_hat.min():.4f}, "
          f"{skkf2._sigma2_hat.max():.4f}]")
    print(f"  KKF σ̂² > 0 frac: "
          f"{np.mean(skkf2._sigma2_hat > 1e-10):.3f}")

    # True σ² for SABR: V² S^{2β}
    true_sigma2 = V[200:]**2 * S[200:]**(2*1.5)
    n_cmp = min(len(true_sigma2), len(skkf2._sigma2_hat))
    corr_kkf = np.corrcoef(
        true_sigma2[:n_cmp], skkf2._sigma2_hat[:n_cmp])[0, 1]
    n_cmp2 = min(len(true_sigma2), len(edmd._sigma2_hat))
    corr_edmd = np.corrcoef(
        true_sigma2[:n_cmp2], edmd._sigma2_hat[:n_cmp2])[0, 1]
    print(f"\n  σ̂² vs true σ²(S,V): KKF corr={corr_kkf:.3f}, "
          f"EDMD corr={corr_edmd:.3f}")

    # ── Test 3: More Nyström landmarks for EDMD ──
    print("\n── SABR β=1.5: EDMD with more landmarks ──")
    for n_nyst in [50, 100, 150]:
        edmd_big = EDMDSigFellerGP(
            n_nystrom=n_nyst, n_gp_landmarks=80)
        edmd_big.fit(S, dS, dt)
        print(f"  m={n_nyst}: α={edmd_big.alpha_mean:.2f}±"
              f"{edmd_big.alpha_sd:.2f}, has_sv={edmd_big._has_sv}, "
              f"R²={edmd_big._r2_sv:.3f}")

    # ── Test 4: More KKF seeds ──
    print("\n── SABR β=1.5: KKF across 5 seeds ──")
    alphas = []
    for seed in range(5):
        S, dS, dt, V = simulate_sabr(1.5, n_steps=15000, seed=seed)
        skkf3 = SigKKFFellerGP(n_nystrom=50, n_gp_landmarks=60)
        skkf3.fit(S, dS, dt)
        alphas.append(skkf3.alpha_mean)
        has_sv = getattr(skkf3, '_has_sv', None)
        print(f"  seed={seed}: α={skkf3.alpha_mean:.2f}±"
              f"{skkf3.alpha_sd:.2f}, P={skkf3.p_bubble:.3f}, "
              f"has_sv={has_sv}")

    print(f"  Mean α={np.mean(alphas):.2f}±{np.std(alphas):.2f} "
          f"(expected 3.0, marginal ~6.0)")

    # ── Test 5: CEV across seeds ──
    print("\n── CEV β=1.5: KKF across 5 seeds ──")
    alphas_cev = []
    for seed in range(5):
        S, dS, dt = simulate_cev(1.5, n_steps=15000, seed=seed)
        skkf4 = SigKKFFellerGP(n_nystrom=50, n_gp_landmarks=60)
        skkf4.fit(S, dS, dt)
        alphas_cev.append(skkf4.alpha_mean)
        print(f"  seed={seed}: α={skkf4.alpha_mean:.2f}±"
              f"{skkf4.alpha_sd:.2f}, P={skkf4.p_bubble:.3f}")

    print(f"  Mean α={np.mean(alphas_cev):.2f}±{np.std(alphas_cev):.2f} "
          f"(expected 3.0)")

    print(f"\n{'=' * 70}")


if __name__ == '__main__':
    main()
