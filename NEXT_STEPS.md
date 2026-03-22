# POMDP-Koopman-Control: Recommended Next Steps

## Paper Landscape

This repo supports **two papers**, with a possible third:

### Paper 1: "Signature-Based Filtering and SDRE Control for POMDP Portfolio Optimization"

**Status**: All 5 graduated sanity checks pass. Honest benchmark done.

**Core contribution**: A model-free POMDP portfolio optimization framework using:
1. Recurrent lead-lag log-signatures for online filtering of latent volatility
2. SDRE (State-Dependent Riccati Equation) with Itô quadratic structure
3. BLR (Bayesian Linear Regression) on signature features for V̂ with principled uncertainty

The key insight: at daily frequency, filtering barely matters (CE gap ~2.4%), but the
framework becomes valuable when the SDE form is unknown (model-free) or when observations
are nonlinear. SDRE avoids regression entirely (SNR=0.02 at daily freq) by using
Itô expansion: E[ΔU] = a + π·b + π²·c, π* = -b/(2c).

**What's done**:
- Graduated sanity checks (5 levels, all pass, 5 macro seeds):
  - L0: RecSig-RLS reproduction (1.27x BPF MSE)
  - L1: Heston filtering + portfolio (CE gap 2.38%)
  - L2: CdC generator recovery (held-out drift corr 0.96)
  - L3: Koopman no-trade region learning (0.993 of Shreve-Soner)
  - L4: SDRE + BLR+KF (V̂ corr 0.78 vs Kalman 0.68)
- Honest benchmark: BPF captures ~50% of oracle-to-constant gap, EWMA ~30%
- Lead-lag log-sig with BLR: V_corr=0.78, beats RLS (0.76), EWMA area (0.76)
- Level 5 hedging demand: KRR in signature RKHS, 0.28-0.33x of ground truth
- Koopman growth rate: correct no-trade region (0.7% of Shreve-Soner CE)

**What's needed for submission**:
1. Higher-frequency validation: intraday data where filtering value is larger
2. Multi-asset extension: signatures of multi-asset paths, cross-impact
3. Transaction costs: integrate Koopman no-trade region with SDRE
4. Compare with:
   - Model-based: Kalman/EKF + Merton
   - Deep RL: DDPG/PPO on portfolio POMDP
   - Particle filter: BPF + Merton
5. Regret analysis: bound on CE loss vs oracle
6. Write up:
   - Framework diagram: observations → signatures → BLR → V̂ → SDRE → π*
   - Table: CE comparison across methods × DGPs (Heston, CEV, SABR)
   - Theory: Itô expansion for SDRE, signature universality (Hida-Malliavin)

### Paper 2: "Online Koopman Operator Learning for Continuous-Time Stochastic Control"

**Status**: Theory established (generator structure L_π = L₀ + πL₁ + π²L₂), experiments done.

**Core contribution**: Learning the Koopman generator from trajectory data for
continuous-time stochastic control. The generator's bilinear structure in the control
variable π enables SDRE without value function regression.

**What's done**:
- Generator L_π = L₀ + πL₁ + π²L₂ from Itô calculus (exact)
- Separable CRRA: V(W,V) = W^{1-γ}·h(V)/(1-γ) (exact)
- CdC identity: μ(x) = Lx, σ²(x) = Lx² - 2xLx (verified)
- Koopman growth rate for no-trade region
- KGEDMD beats A&J for σ² estimation

**What's needed**:
1. Online Koopman: streaming generator updates from new data
2. Convergence guarantees: consistency of EDMD/gEDMD estimator
3. Extend to multi-dimensional state (wealth × vol × signal)
4. Compare with fitted value iteration, policy gradient
5. Numerical stability analysis for large eigenvalue ratios

### Paper 3 (Optional): "Hedging Demand Estimation via Kernel Ridge Regression in Signature RKHS"

**Status**: Level 5 shows 0.28-0.33x of ground truth hedging demand.

**Core contribution**: Nonparametric estimation of the value function gradient
(hedging demand) using KRR in signature RKHS. The kernel gradient gives the
hedge direction: ∇h(sig)/h(sig).

**What's done**:
- FreshSig (per-block) best: 0.28-0.33x of ground truth
- All signs correct, all CE improvements positive
- Correct Riccati for ground truth (verified by 2M-step simulation)

**What's needed**:
1. Attenuation bias correction (currently ~0.3x)
2. Longer block length k or debiased KRR
3. Compare with parametric hedging demand (Kraft 2005)
4. Multi-asset hedging demand

---

## Immediate Technical TODOs

### Priority 1: Paper 1 readiness
- [ ] Higher-frequency experiments (dt=1/252/6.5 for hourly)
- [ ] Multi-asset signatures (2-3 correlated assets)
- [ ] Transaction cost integration with Koopman no-trade
- [ ] Deep RL comparison (DDPG baseline)
- [ ] Clean CE comparison table

### Priority 2: Code quality
- [ ] Package filtering code as installable module
- [ ] Separate filtering (src/sskf/) from control (src/control/)
- [ ] Remove stale experiment scripts from finance/experiments/
- [ ] Document the graduated sanity checks as a testing framework

### Priority 3: Hedging demand
- [ ] Diagnose attenuation bias (0.3x)
- [ ] Try debiased/longer blocks
- [ ] Validate on known parametric cases

---

## Key Files

| File | Purpose |
|------|---------|
| `kronic_pomdp/experiments/graduated_sanity_checks.py` | All 5 sanity check levels |
| `kronic_pomdp/experiments/honest_benchmark.py` | Heston POMDP benchmark |
| `kronic_pomdp/experiments/level5_hedging_demand.py` | KRR hedging demand |
| `examples/proof_of_concept/signature_features.py` | RecurrentLeadLagLogSigMap |
| `src/sskf/streaming_sig_kkf.py` | Streaming signature KKF |
| `finance/experiments/merton_kronic_bilinear.py` | HestonMertonEnv |
| `finance/experiments/verify_heston_baseline.py` | Correct hedging demand formula |

## Known Gotchas

1. **Hedging demand formula**: Do NOT divide by theta. Use Kraft (2005) eq. 14.
2. **RecSig-RLS must be always-online**: Freezing weights → 13x worse.
3. **Product kernel SNR at daily freq**: ~0.0002 signal, ~0.013 noise. Use SDRE instead.
4. **Discrete vs Continuous Kalman**: Euler L·dt is 19x too small vs proper discrete.
5. **In-sample R² means nothing**: Always hold out 20% minimum.
6. **Batch SigRidge is weak**: Use online RecSig-RLS instead (stage1 pattern).
7. **Lévy area ≠ QV for (time, price)**: Must use LEAD-LAG transform.
