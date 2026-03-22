# POMDP-Koopman-Control: Recommended Next Steps

## Paper Landscape

This repo supports **three papers**:

### Paper 1: "Koopman Generator Learning for Partially Observed Stochastic Control"

**Target audience**: Dynamical systems / data-driven control (Brunton, Mezić, Peitz, Klus).

**Status**: Core theory and methodology established. Needs science/engineering benchmark systems.

**Core contribution**: A data-driven framework for continuous-time stochastic optimal
control under partial observations, combining:
1. Koopman generator estimation via kernel gEDMD (KGEDMD) from trajectory data
2. Generator bilinear structure: L_u = L₀ + u·L₁ + u²·L₂ (exact for control-affine SDEs)
3. Carré du Champ (CdC) identity for diffusion coefficient recovery: σ²(x) = Lx² - 2xLx
4. SDRE (State-Dependent Riccati Equation) via Itô quadratic expansion — no value function regression
5. Signature-based nonlinear filtering for latent state reconstruction

The key insight: the generator's bilinear structure in the control variable
enables SDRE without fitting a global value function. The Itô expansion
E[ΔJ] = a + u·b + u²·c gives u* = -b/(2c) locally, avoiding the curse of
dimensionality in Bellman regression.

**What's established**:
- Generator L_u = L₀ + u·L₁ + u²·L₂ from Itô calculus (exact derivation)
- CdC identity verified: drift corr=0.96, σ corr=0.995 on held-out data (CIR)
- KGEDMD-direct beats Ait-Sahalia & Jacod for σ² estimation (CIR: 1.8% vs 2.4%)
- SDRE with Itô quadratic: model-free V̂ corr 0.78 vs model-based Kalman 0.68
- Signature features for nonlinear filtering: 1.27x BPF MSE (model-free vs model-based)
- Koopman growth rate learns correct optimal control regions (0.7% of analytic solution)
- Graduated sanity checks: 5 levels, all pass, 5 macro seeds

**What's needed for submission**:
1. **Benchmark systems from dynamical systems literature**:
   - Lorenz-63/96 with partial observations (filtering + control)
   - Duffing oscillator with stochastic forcing (nonlinear control)
   - Fluid flow control: cylinder wake stabilization (Brunton & Noack 2015)
   - Reaction-diffusion system (Peitz & Klus 2019 benchmark)
   - Double-well potential with noise (standard Koopman test problem)
2. **Comparison with established methods**:
   - EDMD + LQR (Korda & Mezić 2018)
   - Deep Koopman (Lusch, Wehmeyer & Clementi 2018)
   - Kernel EDMD (Williams, Kevrekidis & Rowley 2015)
   - MPC with learned models (data-driven MPC)
   - Fitted value iteration / policy gradient (RL baselines)
3. **Theory**:
   - Convergence guarantees for KGEDMD estimator (consistency, rates)
   - Error bounds: generator approximation error → control suboptimality
   - Connection to Peitz & Klus (2019) generator EDMD framework
   - Relationship between CdC and diffusion maps (Coifman & Lafon 2006)
4. **Scalability**:
   - Nyström approximation for large state spaces
   - Online/streaming generator updates
   - Multi-dimensional state: demonstrate on d=3-10 systems
5. **Write up**:
   - Framework diagram: observations → signatures → KGEDMD → L_u → SDRE → u*
   - Table: control cost comparison across methods × benchmark systems
   - Theory: bilinear generator structure, CdC identity, SDRE via Itô expansion

### Paper 2: "Signature-Based Filtering for Partially Observed Stochastic Systems"

**Target audience**: Signal processing / filtering / stochastic systems.

**Status**: All 5 graduated sanity checks pass. Honest benchmark done.

**Core contribution**: Recurrent lead-lag log-signatures as a model-free alternative
to particle filters and Kalman filters for latent state estimation in SDEs. BLR
(Bayesian Linear Regression) on signature features provides principled uncertainty
quantification for the filtered state.

**What's established**:
- RecSig-RLS with BLR: V̂ corr=0.78 (vs Kalman(CIR) 0.68 on Heston, 0.63 on CEV)
- Lead-lag log-signature captures QV through Lévy area (Lévy area = QV/2)
- Model-free: beats misspecified Kalman on CEV (V̂=0.77 vs 0.63)
- O(1) per step vs O(N) for particle filters
- BLR provides principled observation noise R for outer Kalman loop
- Honest benchmark: BPF captures ~50% oracle-to-constant gap, EWMA ~30%

**What's needed**:
1. Benchmark on standard filtering problems (bearings-only tracking, etc.)
2. Higher-frequency validation where filtering value is larger
3. Multi-sensor / multi-asset extension
4. Comparison with EKF, UKF, BPF, EnKF on nonlinear systems
5. Theoretical analysis: signature universality → consistency of filter

### Paper 3: "Value Function Gradient Estimation via Kernel Ridge Regression in Signature RKHS"

**Status**: Level 5 shows 0.28-0.33x of ground truth. Proof of concept.

**Core contribution**: Nonparametric estimation of the value function gradient
(hedging demand / adjoint variable) using KRR in signature RKHS. The kernel
gradient gives the control correction: ∇h(sig)/h(sig).

**What's established**:
- FreshSig (per-block) best: 0.28-0.33x of ground truth
- All signs correct, all cost improvements positive
- Correct Riccati for ground truth (verified by 2M-step simulation)

**What's needed**:
1. Attenuation bias correction (currently ~0.3x)
2. Longer block length k or debiased KRR
3. Validate on Brunton-style benchmark systems (not just finance)
4. Compare with adjoint-based methods, backpropagation through dynamics

---

## Immediate Technical TODOs

### Priority 1: Paper 1 — benchmark systems
- [ ] Implement Lorenz-63 POMDP (observe x₁, control forcing on x₂)
- [ ] Implement Duffing oscillator with stochastic forcing
- [ ] Double-well potential: KGEDMD generator vs analytic
- [ ] EDMD + LQR baseline (Korda & Mezić 2018)
- [ ] Control cost comparison table across methods × systems
- [ ] CdC identity verification on each benchmark system

### Priority 2: Paper 2 — filtering benchmarks
- [ ] Bearings-only tracking benchmark
- [ ] Higher-frequency experiments (dt=1/252/6.5 for hourly)
- [ ] Multi-dimensional filtering (d=3-5 latent states)
- [ ] EKF/UKF/EnKF comparison on nonlinear systems

### Priority 3: Code quality
- [ ] Package filtering code as installable module
- [ ] Separate filtering (src/sskf/) from control (src/control/)
- [ ] Remove stale experiment scripts from finance/experiments/
- [ ] Document the graduated sanity checks as a testing framework

### Priority 4: Value function gradients
- [ ] Diagnose attenuation bias (0.3x)
- [ ] Try debiased/longer blocks
- [ ] Validate on known analytic cases (LQR, double-well)

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
