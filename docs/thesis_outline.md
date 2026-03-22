# Thesis Outline: Bayesian Kernel Methods for Market Making Under Uncertainty

## Unifying Theme

**How do market makers set prices when key state variables are unobservable?**

Three instantiations of this problem, addressed with a unified toolkit:
1. **Volatility** is latent → estimate from price dynamics (Ch. 1)
2. **Arbitrage constraints** must be satisfied → bake into kernel structure (Ch. 2)
3. **Informed order flow** is hidden → filter via POMDP (Ch. 3)

---

## Chapter 1: Bayesian Kernel Methods for Model-Free Option Pricing

### 1.1 Introduction
- Problem: Price options when no options market exists
- Traditional approaches require parametric models (BS, Heston, SABR)
- Our contribution: Model-free volatility estimation with uncertainty quantification

### 1.2 Mathematical Framework
- **Carré du Champ Identity**: σ²(x) = L(x²) - 2x·L(x)
- **KGEDMD**: Kernel Generator EDMD for nonparametric generator learning
- **Bayesian Interpretation**: KRR = Bayesian Linear Regression
  - Posterior: w|y ~ N(m_w, Σ_w)
  - Predictive variance decomposes into epistemic + aleatoric

### 1.3 Option Pricing Under Volatility Uncertainty
- Monte Carlo propagation of posterior through Black-Scholes
- Bid-Ask from percentiles: Bid = P₂₅, Ask = P₇₅
- Automatic spread adjustment based on posterior width

### 1.4 Eigenfunction Pricing
- Variance swaps: E[∫V dt | V₀] via eigenfunction expansion
- VIX computation without derivatives data
- European options on volatility

### 1.5 Empirical Results
- Walk-forward CV on simulated CIR: KGEDMD best RMSE/correlation
- Bitcoin options: 3-5% error using only underlying prices
- Posterior calibration analysis

### 1.6 Extensions
- Log-signature features for order flow integration
- Multi-asset covariance via multivariate CdC

**Key Files**:
- `kronic_pomdp/experiments/mvp_benchmark.py`
- `kronic_pomdp/experiments/eigenfunction_pricing.py`
- `docs/posterior_market_making.qmd`

---

## Chapter 2: Arbitrage-Free Kernel Construction for Option Surfaces

### 2.1 Introduction
- Problem: Ensure implied vol surfaces satisfy no-arbitrage conditions
- SVI/SSVI: Only slice-wise arb-free (single expiry)
- SANOS (Buehler 2025): LP constraints
- Our contribution: **Structural constraints via kernel design**

### 2.2 Arbitrage-Free Conditions
- **Butterfly (Durrleman)**: ∂²C/∂K² ≥ 0
- **Calendar Spread**: ∂w/∂T ≥ 0 (total variance monotone in maturity)
- **Call Spread**: ∂C/∂K ∈ [-1, 0]

### 2.3 Monotone-Convex Feature Maps
- Key insight: If φⱼ(k,T) is convex in k and monotone in T, and wⱼ ≥ 0, then
  w(k,T) = Σ wⱼ φⱼ(k,T) is automatically arb-free

- **In T direction**: CDF basis ψₘ(T) = Φ((T - Tₘ)/ℓ_T)
  - Monotonically increasing by construction

- **In k direction**: Double-integrated Gaussian
  - χₙ(k) = ∫∫ exp(-(k'' - kₙ)²/2ℓ²) dk'' dk'
  - Convex by construction (second derivative = Gaussian ≥ 0)

- **Product basis**: φₘₙ(k,T) = ψₘ(T) · χₙ(k)

### 2.4 Bayesian Formulation
- Prior: log wₘₙ ~ N(μ₀, σ₀²) ensures positivity
- Posterior via MAP + Laplace approximation
- Sample posterior surfaces → all samples arb-free by construction

### 2.5 Full Durrleman Condition
- Simple convexity ∂²w/∂k² ≥ 0 is necessary but not sufficient
- Full condition: g(k,T) ≥ 0 where g involves w, w', w''
- Soft penalty or constraint: |w'| ≤ 4w^{1/2}/(T^{1/2}(1+|k|))

### 2.6 Comparison with Existing Methods

| Aspect | SVI/SSVI | SANOS (LP) | Our Approach |
|--------|----------|------------|--------------|
| Calendar arb-free | No (slice-wise) | Yes (constraint) | Yes (structural) |
| Butterfly | Partial | Yes (constraint) | Yes (structural) |
| Uncertainty | None | None | Full posterior |
| Bid-Ask | Ad-hoc | LP bounds | Percentiles |

### 2.7 Empirical Results
- Synthetic Heston: 100% of posterior samples pass arb-free tests
- Standard RBF: Fails calendar (min ∂w/∂T = -0.12) and butterfly
- Arb-free kernel: Passes all conditions by construction

### 2.8 Integration with Chapter 1
- Use arb-free implied vol surface as input for local vol via Dupire
- Posterior on w propagates to posterior on local vol
- Connection to KGEDMD: Same Bayesian KRR machinery

**Key Files**:
- `kronic_pomdp/experiments/arb_free_kernel_poc.py`
- `docs/theory_arb_free_kernel.md`

---

## Chapter 3: Market Making with Insider Information (Future Work)

*Note: This chapter outlines future research directions and is not being actively developed.*

### 3.1 Motivation
- Kyle (1985): Insider has private signal about asset value
- Market maker infers information from order flow
- Price impact = permanent (information) + transient (inventory)

### 3.2 Multi-Asset Kyle Model
- N correlated assets with common and idiosyncratic factors
- Insider observes subset of factors
- Market maker must jointly estimate:
  - Asset values (latent)
  - Insider's private signal (latent)
  - Noise trader intensity (observable)

### 3.3 POMDP Formulation
- **State**: (V₁,...,Vₙ, I₁,...,Iₖ) where I = insider signals
- **Observation**: Order flow Qₜ = insider + noise
- **Belief**: p(V, I | Q₁:ₜ)
- **Action**: Bid-ask quotes (δᵇⁱᵈ, δᵃˢᵏ) for each asset

### 3.4 Connection to Chapters 1-2
- **Chapter 1 tools**: KGEDMD for learning price dynamics from order flow
- **Chapter 2 tools**: Arbitrage-free constraints on cross-asset spreads
- **New tools needed**:
  - Belief filtering in multi-agent setting
  - Game-theoretic equilibrium (insider responds to MM strategy)
  - Signature features for order flow patterns

### 3.5 Proposed Methodology

1. **Belief Filtering via Signatures**
   - Lead-lag log-signature of order flow captures cross-asset lead-lag
   - BLR + Kalman for belief update (extends Level 4 machinery)

2. **SDRE for Optimal Quoting**
   - Proposition 3 (Itô quadratic): E[ΔProfit] = a + π·b + π²·c
   - Local quadratic approximation → state-dependent Riccati
   - Extends to multi-asset via tensor product kernels

3. **Equilibrium Computation**
   - Fixed-point iteration: MM strategy → insider best response → update MM
   - Kernel value function for continuation value

### 3.6 Expected Contributions
- First application of KGEDMD to multi-asset market making
- Signature-based belief filtering for order flow
- Arbitrage-free cross-asset spread construction

### 3.7 Theoretical Questions
- Does equilibrium exist? (Fixed-point theorem conditions)
- How does correlation structure affect information leakage?
- When does signature filter outperform parametric Kalman?

**Relevant Existing Work**:
- `docs/plan_kkf_insider.md` (preliminary notes)
- `docs/theory_kyle_fixed_point.md` (equilibrium theory)
- `kronic_pomdp/experiments/graduated_sanity_checks.py` Level 4 (BLR+KF machinery)

---

## Thesis Contributions Summary

| Chapter | Problem | Method | Key Innovation |
|---------|---------|--------|----------------|
| 1 | Vol estimation without options | KGEDMD + Bayesian KRR | Posterior uncertainty → bid-ask |
| 2 | Arb-free vol surfaces | Monotone-convex features | Structural (not constraint) guarantees |
| 3 | Insider information filtering | Signature POMDP + SDRE | Multi-asset Kyle equilibrium |

## Unifying Mathematical Framework

All three chapters use:
1. **Koopman/Generator operators** for dynamics learning
2. **Kernel ridge regression** with Bayesian interpretation
3. **Signature features** for path-dependent information
4. **Carré du Champ** for volatility extraction

The progression:
- Ch. 1: Learn σ²(S) from underlying → price options
- Ch. 2: Ensure σ(K,T) surface is arb-free → structural constraints
- Ch. 3: Filter latent information from order flow → optimal market making

---

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| CdC identity + KGEDMD | ✅ Complete | `cdc_kernel_estimators.py` |
| Bayesian KRR posterior | ✅ Complete | `mvp_benchmark.py` |
| Walk-forward CV | ✅ Complete | `mvp_benchmark.py` |
| Eigenfunction pricing | ✅ Complete | `eigenfunction_pricing.py` |
| Arb-free kernel (basic) | ✅ Complete | `arb_free_kernel_poc.py` |
| Full Durrleman penalty | 🔄 In progress | `arb_free_kernel_poc.py` |
| Multi-asset CdC | ✅ Complete | `cdc_kernel_estimators.py` |
| Lead-lag log-signature | ✅ Complete | `signature_features.py` |
| BLR + Kalman filter | ✅ Complete | `graduated_sanity_checks.py` |
| Kyle POMDP formulation | 📋 Planned | Ch. 3 future work |
| Multi-asset equilibrium | 📋 Planned | Ch. 3 future work |
