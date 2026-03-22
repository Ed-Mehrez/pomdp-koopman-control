"""
Test SigKKFFellerGP and EDMDSigFellerGP on CEV and SABR processes.

Key question: does the KKF/EDMD + 2D GP approach give correct structural
elasticity for SABR (where marginal NW gives inflated α)?

Expected results:
  CEV β=1.5: α≈3.0 (no bubble), β=2.5: α≈5.0 (bubble)
  SABR β=1.5: α≈3.0 (NOT ≈6.0 which is the marginal bias)
  SABR β=2.5: α≈5.0 (bubble)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'kronic_pomdp', 'experiments'))
from gp_bubble_detector import (SigKKFFellerGP, EDMDSigFellerGP,
                                MarginalLikelihoodFellerGP)


def simulate_cev(beta, n_steps=10000, dt=0.01, sigma0=0.3, S0=1.0,
                 seed=42):
    """CEV: dS = μS dt + σ₀ S^β dW"""
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


def simulate_sabr(beta, n_steps=10000, dt=0.01, sigma0=0.3, nu=0.2,
                  rho=-0.5, S0=1.0, seed=42):
    """SABR: dS = V S^β dW₁, dV = nu V dW₂, corr(W₁,W₂)=ρ

    Uses exact lognormal solution for V to avoid absorption at zero.
    V_t = V_{t-1} · exp(-½ν²Δt + ν√Δt·z₂)

    Note: ν=0.2 (not 0.5) so V stays alive over T=150 time units.
    With ν=0.5, lognormal V → 0 for most of the simulation.
    """
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
        # Exact lognormal for V (never touches zero)
        V[t] = v * np.exp(-0.5 * nu**2 * dt + nu * np.sqrt(dt) * z2)

        S[t] = max(S[t], 1e-6)

    dS = np.diff(S)
    dS = np.append(dS, dS[-1])
    return S, dS, dt, V


def test_one(name, estimator_cls, z, dz, dt, expected_alpha, tol=1.5,
             **kwargs):
    """Run estimator and check α estimate."""
    est = estimator_cls(**kwargs)
    est.fit(z, dz, dt)

    alpha = est.alpha_mean
    sd = est.alpha_sd
    p = est.p_bubble

    has_sv = getattr(est, '_has_sv', None)
    r2_sv = getattr(est, '_r2_sv', None)
    beta_v = getattr(est, '_beta_v', None)

    ok = abs(alpha - expected_alpha) < tol
    status = "PASS" if ok else "FAIL"

    print(f"  [{status}] {name}: α={alpha:.2f}±{sd:.2f}, "
          f"P(bubble)={p:.3f}, expected α≈{expected_alpha:.1f}")
    if has_sv is not None:
        print(f"         has_sv={has_sv}, R²(S,V)={r2_sv:.3f}", end="")
        if beta_v is not None:
            print(f", β_v={beta_v:.2f}", end="")
        print()

    return ok, alpha, sd


def main():
    print("=" * 70)
    print("  SigKKFFellerGP and EDMDSigFellerGP Tests")
    print("=" * 70)

    results = []
    n_steps = 15000

    # ── CEV tests ──
    print("\n── CEV (no stochastic vol) ──")
    for beta in [1.5, 2.5]:
        expected_alpha = 2 * beta  # σ² = σ₀² S^{2β}
        is_bubble = beta > 2
        S, dS, dt = simulate_cev(beta, n_steps=n_steps, seed=42)

        print(f"\n  CEV β={beta} (bubble={is_bubble})")
        for cls_name, cls in [("SigKKF", SigKKFFellerGP),
                               ("EDMD", EDMDSigFellerGP)]:
            ok, a, sd = test_one(
                f"{cls_name} CEV β={beta}", cls, S, dS, dt,
                expected_alpha, tol=2.0,
                n_nystrom=50, n_gp_landmarks=60)
            results.append((f"{cls_name} CEV β={beta}", ok, a, sd))

    # ── SABR tests ──
    # ν=0.2 (default) keeps V alive; ν=0.5 makes V→0 over T=150
    print("\n── SABR (stochastic vol, ρ=-0.5, ν=0.2) ──")
    print("  Marginal NW has inflated α for stochastic vol")

    for beta in [1.5, 2.5]:
        expected_alpha = 2 * beta
        is_bubble = beta > 2
        S, dS, dt, V = simulate_sabr(
            beta, n_steps=n_steps, rho=-0.5, seed=42)

        print(f"\n  SABR β={beta}, ρ=-0.5 (bubble={is_bubble})")
        for cls_name, cls in [("SigKKF", SigKKFFellerGP),
                               ("EDMD", EDMDSigFellerGP)]:
            ok, a, sd = test_one(
                f"{cls_name} SABR β={beta}", cls, S, dS, dt,
                expected_alpha, tol=2.0,
                n_nystrom=50, n_gp_landmarks=60)
            results.append((f"{cls_name} SABR β={beta}", ok, a, sd))

    # ── SABR with stronger correlation ──
    print("\n── SABR (ρ=-0.7, stronger vol-price coupling) ──")
    for beta in [1.5]:
        expected_alpha = 2 * beta
        S, dS, dt, V = simulate_sabr(
            beta, n_steps=n_steps, rho=-0.7, seed=42)

        print(f"\n  SABR β={beta}, ρ=-0.7 (no bubble)")
        for cls_name, cls in [("SigKKF", SigKKFFellerGP),
                               ("EDMD", EDMDSigFellerGP)]:
            ok, a, sd = test_one(
                f"{cls_name} SABR β={beta} ρ=-0.7", cls, S, dS, dt,
                expected_alpha, tol=2.0,
                n_nystrom=50, n_gp_landmarks=60)
            results.append((f"{cls_name} SABR β={beta} ρ=-0.7",
                           ok, a, sd))

    # ── Marginal Likelihood Feller (KF marginalizing V) ──
    print("\n── MarginalLikelihoodFellerGP ──")
    print("  Marginalizes V via 1D KF. Conservative when ρ<0 (inflates α).")
    print("  CEV: near-exact. SABR: α inflated (upper bound on true α).")

    # CEV
    for beta in [1.5, 2.5]:
        expected_alpha = 2 * beta
        is_bubble = beta > 2
        S, dS, dt = simulate_cev(beta, n_steps=n_steps, seed=42)
        print(f"\n  CEV β={beta} (bubble={is_bubble})")
        est = MarginalLikelihoodFellerGP()
        est.fit(S, dS, dt)
        ok = abs(est.alpha_mean - expected_alpha) < 0.5  # tight for CEV
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] MargLik CEV β={beta}: α={est.alpha_mean:.3f}"
              f"±{est.alpha_sd:.3f}, P(bubble)={est.p_bubble:.4f}, "
              f"q={est._q_logvol:.4f}")
        results.append((f"MargLik CEV β={beta}", ok,
                        est.alpha_mean, est.alpha_sd))

    # SABR (document conservatism)
    for beta in [1.5, 2.5]:
        for rho in [-0.5, -0.7]:
            expected_alpha = 2 * beta
            S, dS, dt, V = simulate_sabr(
                beta, n_steps=n_steps, rho=rho, seed=42)
            print(f"\n  SABR β={beta}, ρ={rho}")
            est = MarginalLikelihoodFellerGP()
            est.fit(S, dS, dt)
            # For SABR: α is inflated by ρ. Accept if α ≥ expected
            # (conservative: test overestimates risk when ρ<0)
            ok = est.alpha_mean >= expected_alpha - 0.5
            status = "PASS" if ok else "FAIL"
            inflate = est.alpha_mean - expected_alpha
            print(f"  [{status}] MargLik SABR β={beta} ρ={rho}: "
                  f"α={est.alpha_mean:.3f}±{est.alpha_sd:.3f}, "
                  f"inflate={inflate:+.2f}, "
                  f"P(bubble)={est.p_bubble:.4f}")
            results.append((f"MargLik SABR β={beta} ρ={rho}",
                           ok, est.alpha_mean, est.alpha_sd))

    # ── Summary ──
    print(f"\n{'=' * 70}")
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_total = len(results)
    print(f"  Results: {n_pass}/{n_total} passed")
    print(f"{'=' * 70}")

    print(f"\n  {'Test':<35} {'α':>6} {'±SD':>6} {'Pass':>5}")
    print(f"  {'-' * 55}")
    for name, ok, a, sd in results:
        status = "OK" if ok else "FAIL"
        print(f"  {name:<35} {a:>6.2f} {sd:>6.2f} {status:>5}")


if __name__ == '__main__':
    main()
