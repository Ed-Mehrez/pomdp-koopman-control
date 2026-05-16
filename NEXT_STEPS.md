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
1. **Benchmark systems from dynamical systems literature** (start with ONE for v1, expand only after it ships):
   - **v1 starting point**: Double-well potential with noise — simplest end-to-end
     test of CdC + KGEDMD + SDRE, has analytic ground truth, standard Koopman benchmark
   - Deferred to v2+ (do not build until double-well is clean):
     - Lorenz-63/96 with partial observations (filtering + control)
     - Duffing oscillator with stochastic forcing
     - Fluid flow: cylinder wake stabilization (Brunton & Noack 2015)
     - Reaction-diffusion system (Peitz & Klus 2019 benchmark)
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

6. **Finance application slot** (one application section, NOT the headline, NOT a separate paper):
   - Option market-making v1: pure Heston simulator, low-dim state `[q, h, V_hat, tau]`
     where `q` = option inventory, `h` = net delta exposure, `V_hat` = filtered variance,
     `tau` = time to maturity (decided 2026-04-06 to include tau explicitly so the
     controller never has its non-stationarity blamed on the policy structure; the
     env already exposes `time_to_maturity` on `OptionMMState`).
   - Contract convention: one fixed-strike European call, struck ATM at reset. Episode
     horizon must be shorter than maturity. No expiry, roll, or re-strike in v1.
     Default maturity is 1Y (codex's choice); with `tau` in the controller state this
     is an independent parameter and can be shortened later if a more dynamic `tau`
     is desired without changing the state spec.
   - Fill convention: exponential Poisson fills with frozen parameters, no queue
     dynamics, no market impact. Default same-step bid+ask fills use a conservative
     `mid_drift` policy to avoid a zero-risk full-spread subsidy. Market-path RNG
     is separate from fill/tie-break RNGs so paired controllers share the same
     exogenous Heston path.
   - CE convention: benchmark configs use positive `initial_cash` (e.g. 100,000)
     rather than shifting wealth inside the metrics module.
   - **Gating baseline**: Avellaneda–Stoikov closed form with same `V_hat`. SDRE must
     beat A-S-with-V_hat under the Bayesian gating rule below or the control framing
     adds nothing.
   - **Critical ablation**: SDRE-on-`(q,h,V_hat,tau)` vs linear-inventory-rule-on-`(q,h,V_hat,tau)`
     (identical state). Isolates control structure from filter contribution.
   - **Pre-registered primary metric (Bayesian, no frequentist tests per repo rule)**:
     posterior on `ΔCE = CE_candidate − CE_baseline` from a Bayesian bootstrap (Rubin 1981)
     over paired seeds, with Dirichlet(1,…,1) weights and the CE functional reweighted
     directly. **Ship condition**: `P(ΔCE > 0 | data) ≥ 0.95`, equivalently 95%
     credible interval strictly above zero. Report posterior mean, posterior SD
     (`sd_post`, NOT SE — see `feedback_bayesian_naming.md`), 95% CrI, and `P(ΔCE > 0)`.
     Secondary: inventory variance, max DD, net delta exposure RMS, time at inventory limit.
   - Use the SAME SDRE machinery as the science benchmarks — no finance-specific tuning.
   - **Honest framing rule**: do not advertise pricing accuracy or IV RMSE. The result
     is "wealth/inventory control under partial observation," and a clean null
     ("framework recovers A-S, no improvement at daily freq") is publishable.
   - **Excluded from v1**: VRP factor (answer-in-the-basis trap), multi-strike,
     Alpaca replay, dual-control `P·q²` term, learned Koopman lift.
   - **Fork decision** (revisit later, NOT now): split into a separate repo only if
     OMM v1 produces a paper-worthy standalone result OR develops genuinely separate
     infrastructure (Alpaca replay adapters, fill calibration, broker glue, data
     manifests). Until then OMM lives in `src/applications/option_mm/`.

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

### Priority 0: Minimal shared core (extract on demand, NOT a framework up front)
- [ ] `src/control/sdre.py` — tiny local Riccati / Itô-quadratic helper. Only what
      double-well AND OMM both need. Do NOT design a general SDRE framework.
- [ ] `src/eval/paired.py` — paired-seed / bootstrap / CE helpers extracted from
      `kronic_pomdp/experiments/honest_benchmark.py`. Only what both tracks need.
- [ ] If only one track ends up using a piece, leave it inline in that track. Generalize
      ONLY when both tracks demonstrably share the same call pattern.
- [ ] **Quarantine** the broken Kyle / `SignatureState` modules
      (`src/finance/adaptive_kyle.py`, `adaptive_kyle_kernel.py`,
      `kronic_pomdp/experiments/online_rbf_sig_belief.py`,
      `kronic_pomdp/experiments/signature_kronic_control.py`). Mine for formulas only;
      do NOT put them on the critical path of either track.

### Priority 1: Paper 1 — v1 science benchmark (double-well first)
- [ ] Keep Brunton/Klus-facing POC work in `experiments/science_poc/` and
      `docs/poc_brunton_klus_2026_05/`; finance remains a stress-test lane, not
      the lead narrative.
- [ ] Double-well potential POMDP: KGEDMD generator vs analytic
- [ ] CdC identity verification on double-well
- [ ] EDMD + LQR baseline (Korda & Mezić 2018) on double-well
- [ ] SDRE controller on double-well using `src/control/sdre.py`
- [ ] Paired-seed comparison table (KGEDMD-SDRE vs EDMD-LQR vs analytic)
- [ ] **Stop condition**: ship double-well figure before touching Lorenz/Duffing/fluid

### Priority 1b: Paper 1 — finance application slot (OMM v1, pure simulator)

**Stage 1 — env + smoke (DONE 2026-04-07)**
- [x] `src/applications/option_mm/env.py` — Heston-only sim, fixed-strike ATM call,
      `mid_drift` same-step both-fill default, split RNG (`path_rng`/`fill_rng`/`tie_rng`)
      for paired-seed honesty, censoring + variance-floor monitoring, `net_delta` field.
- [x] `src/applications/option_mm/metrics.py` — `UtilitySpec` + `crra_utility` +
      `cara_utility`; `paired_ce_posterior(method="delta"|"mc"|"bootstrap")`;
      `paired_mean_difference_posterior` (Student-t conjugate);
      `paired_bayesian_bootstrap_posterior` (general Dirichlet weights, fallback only).
- [x] `finance/experiments/option_mm_smoke.py` at N=500 — all 10 stage-1 checks PASS.
      Inventory std grows √T to within 2%. Censoring rate 0. Variance floor binding 0.02%.

**Stage 2 — beliefs (EWMA) + A-S gating (DONE 2026-04-07)**
- [x] `src/applications/option_mm/beliefs.py` — `EWMAVarianceFilter` only.
- [x] `src/applications/option_mm/controllers.py` — `no_quote`, `constant_spread`,
      `avellaneda_stoikov` (strict textbook, no tuning).
- [x] `finance/experiments/option_mm_gating.py` at **N=5000** (re-spec'd from 500
      after underpowered initial gate; per-seed SNR ≈ 0.058 needs N ≥ ~810 for
      `P ≥ 0.95`, N=5000 gives ~4σ margin).
- [x] **Stage 2 result (locked)**: A-S-with-EWMA beats constant-spread.
      CRRA(γ=2): ΔCE = 26.97, sd_post = 6.60, CrI [14.03, 39.90], P(>0) = 0.99998.
      CARA(α=0.001): ΔCE = 84.55, sd_post = 12.30, P(>0) ≈ 1.0 (curvature-driven larger
      magnitude — α=0.001 puts CARA in a tail-CE regime; for matched Arrow-Pratt to
      CRRA(γ=2) at W≈1e5 use α≈2e-5 next time). Delta vs MC agreement <1% under
      both utilities. Spread capture 1.92×, net delta RMS 0.67×, MTM noise 0.68×.

**Stage 3 — filter ablation (DONE 2026-04-07)**
- [x] Extended `beliefs.py` with `OracleVarianceFilter`, `BootstrapParticleFilter`
      (200 particles), `RecursiveSigRLSFilter` (lead-lag log-sig + Bayesian RLS).
- [x] `finance/experiments/option_mm_filter_ablation.py` — A-S with each of
      {oracle, BPF, RecSig-RLS, EWMA} on the same N=5000 paired seeds and the same
      `SeedSequence(20260407)` as Stage 2. EWMA−constant contrast reproduces Stage 2
      number exactly (26.967789 to the digit), confirming wiring correctness.
- [x] **Stage 3 result (locked)**: filter quality is essentially **saturated**.
      Total spread among the four filters is 0.138 CE units; total controller gap
      (EWMA−constant) is 26.97. Filter quality accounts for **0.5%** of the controller
      advantage. EWMA captures 100.17% of the oracle gap, BPF captures 99.92%,
      RecSig captures 100.43%. Reproduces the `honest_benchmark.py` finding under
      controller evaluation: at daily frequency under default Heston params,
      sophisticated filtering buys essentially nothing in CE terms.
- [x] **Interesting sub-finding (publishable, not a bug)**: RecSig and EWMA slightly
      *exceed* oracle in CE (RecSig − BPF = +0.138 ± 0.059, P = 0.99). This is because
      textbook A-S is mis-specified for stochastic vol — A-S assumes σ² is constant
      over (T−t), but the optimal σ² to plug in is `E[avg V over (t, T) | F_t]`, a
      forward-averaged variance. Smoothed filters (RecSig, EWMA) are *closer* to
      forward-averaged V than instantaneous V_t, so they accidentally compensate
      for A-S's mis-specification. Cite Cartea & Jaimungal (2015) §5. Not worth
      chasing for Stage 4 — magnitude is 0.5% of the controller gap.

**Stage 4 — control structure ablation: SDRE vs linear rule (NEXT)**
- [x] **Filter decision**: Stage 4 uses **EWMA**. Justified by Stage 3 — no measurable
      filter advantage to anything fancier, and using the simplest filter means
      SDRE-vs-linear-rule cannot be confounded by filter sophistication.
- [ ] `src/applications/option_mm/controllers.py` add (d) `linear_inventory_rule`
      and (e) `sdre_controller` on `(q, h, V_hat, tau)`. Both consume the same EWMA
      `V_hat`. SDRE uses the Itô-quadratic expansion `E[ΔU] = a + π·b + π²·c`,
      `π* = -b/(2c)` derived in `kronic_pomdp/experiments/level4_generator_sdre.py`
      — port the math, do NOT extend the prototype as a framework.
- [ ] `src/control/sdre.py` (Priority 0 from the framework cleanup) — extract the
      tiny local Riccati / Itô-quadratic helper from level4 ONLY if both
      double-well and OMM v4 will call it. Otherwise leave inline in
      `controllers.py`. Per `feedback_no_framework_up_front.md`.
- [ ] `finance/experiments/option_mm_ablation.py` — paired bootstrap of SDRE vs
      linear-rule vs A-S, all consuming EWMA, on the same N=5000 paired seeds and
      same `SeedSequence(20260407)` as Stages 2 and 3. **Run a power-calc pilot
      first** at N=100 to estimate per-seed SNR for the SDRE−linear contrast.
      Per `feedback_power_calc_discipline.md`, do not reuse N from Stage 2/3 blindly.
- [ ] **Pre-registered Stage 4 ship rule**: `P(ΔCE_SDRE−linear > 0 | data) ≥ 0.95`
      under CRRA(γ=2) AND CARA(α=2e-5, matched Arrow-Pratt). delta+MC agreement
      ≤ 5% relative under both utilities. If the gate fails, **do NOT silently
      bump N** per `feedback_no_silent_n_changes.md` — diagnose, recommend, wait.
- [ ] **Outcome interpretations**:
      - SDRE clears gate AND beats A-S in paired ΔCE → control structure adds value.
      - SDRE ties linear-rule but both beat A-S → win lives in the (q,h,V̂,τ) state
        space, not the policy structure. **Ship the linear rule.** Publishable null.
      - SDRE ties A-S → control structure with the same state space adds nothing.
        Stage 4 is null. Publishable.
- [ ] **Gaussian-fill validation**: in sim, compare SDRE policy vs exact-Poisson-fill
      optimum on a smaller scenario set. If gap is large, the diffusion approximation
      underlying SDRE is solving the wrong problem and SDRE results don't generalize.

**Stage 4 — control structure ablation (DEFERRED until Stage 3 lands)**
- [ ] `src/applications/option_mm/controllers.py` add (d) `linear_inventory_rule`
      and (e) `sdre_controller` on `(q,h,V_hat,tau)`. Both consume the *same* filter
      chosen in Stage 3.
- [ ] `finance/experiments/option_mm_ablation.py` — paired bootstrap of
      SDRE vs linear-rule vs A-S, all at N=5000 (or larger if Stage 3 power calc
      indicates).
- [ ] **Pre-registered Stage 4 ship rule**: `P(ΔCE_SDRE−linear > 0 | data) ≥ 0.95`
      under CRRA(γ=2) AND CARA(α=2e-5). If only A-S vs constant_spread shows a gap
      (i.e., SDRE ties with linear-rule), the win lives in the state space, not the
      policy structure — that's "ship the linear rule", a clean negative result that
      is still publishable.
- [ ] **Gaussian-fill validation**: in sim, compare SDRE policy vs exact-Poisson-fill
      optimum. If gap is large, the diffusion approximation is solving the wrong problem.

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
