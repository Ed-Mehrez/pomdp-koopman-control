# KRONIC for POMDPs via Signature Belief Representations

## Core Idea

KRONIC (Koopman-based Robust Optimal Nonlinear Control) assumes full state observability to compute eigenfunctions ψ(x). In POMDPs, we only observe y = h(x) + noise, so we can't compute ψ(x) directly.

**Key insight**: Learn eigenfunctions of the **belief state**, not the true state. Signatures of the observation history provide a finite-dimensional sufficient statistic for belief.

## Research Questions

### 1. Belief Representation via Signatures
- Can signatures of observation history serve as belief sufficient statistics?
- Connection: Lead-lag Lévy area ≈ QV/2 (captures volatility from path)
- For LQG: belief is (x̂, P). For nonlinear: signatures compress history.

### 2. Generator in Belief Space
- True state generator: Lf = b·∇f + ½tr(Σ·∇²f)
- Belief generator: L_b f(b) = ... (depends on observation model)
- Eigenfunctions of belief dynamics → linear control in belief space

### 3. LP vs LQR Structure
- **LQR**: Quadratic running cost → Riccati equation
- **Terminal utility**: max E[U(W_T)] with U ≈ Σ wᵢ ψᵢ (linear in eigenfunctions)
- With control bounds → **LP** structure, not LQR!
- Transaction costs (L1) → bang-bang / LP

## Test Environments (from finance/)

| Problem | Hidden State | Observations | POMDP Structure |
|---------|-------------|--------------|-----------------|
| Merton-Heston | Volatility V | Returns dW/W | Filtering + control |
| Kyle Model | Insider value v | Order flow Y | Information asymmetry |
| Transaction Costs | Optimal π* | Past allocations | History-dependent |

## Key Files (from parent repo)

### Signatures & Filtering
- `../src/sskf/streaming_sig_kkf.py` - Signature-based Kalman filter
- `../src/finance/signature_volatility.py` - Vol estimation from signatures

### KGEDMD & Control
- `../finance/experiments/merton_kgedmd_utility.py` - Eigenfunction learning
- `../rkhs_kronic/` - Original KRONIC implementation

### Theory
- `../docs/theory_crra_eigenfunction.md` - CRRA utility eigenfunctions
- `../docs/gedmd_ito_correction.md` - Full generator with Itô terms

## Proposed Contributions

1. **Theoretical**: Extend KRONIC to POMDPs via belief-space eigenfunctions
2. **Practical**: Signatures as universal belief representation
3. **Algorithmic**: LP formulation for terminal utility + transaction costs
4. **Applications**: Stochastic volatility portfolio optimization

## Experimental Results

### LQG Baseline (A=-0.5, G=0.3, H=0.5)

**CRITICAL: Must use proper discrete Kalman filter!**

Initial results showed signatures "beating" Kalman - this was due to **buggy Euler discretization**:
- Continuous Kalman gain: L = 0.281
- Used: L·dt = 0.003 (too small!)
- Correct discrete gain: L_d = 0.054 (**19x larger!**)

**With PROPER discrete Kalman (fair comparison):**

| Method | RMSE | vs Kalman | Notes |
|--------|------|-----------|-------|
| **Discrete Kalman** | **0.116** | **1.00x** | Optimal baseline |
| RBF-Signature (tuned) | 0.114 | 0.98x | Matches Kalman! |
| EMA (fair decay) | 0.124 | 1.07x | Worse than Kalman |

**Key insight**: Signatures MATCH Kalman (within 2%) when:
1. Using proper **discrete** Kalman baseline (not Euler-discretized continuous)
2. Using **fair decay** = exp((A - L·C)·dt) to match Kalman's effective memory

### Hida-Malliavin Isomorphism: CONFIRMED (with caveats)

**Theorem**: Signatures are a SUFFICIENT REPRESENTATION for Bayesian inference.

| Quantity | Kalman | Signature | Match? |
|----------|--------|-----------|--------|
| Posterior mean E[x\|y] | x̂ (exact formula) | regression #1 | ✓ 2% |
| Posterior var Var[x\|y] | P (Riccati) | regression #2 | ✓ 10% |
| GP variance Var[f(sig)] | N/A | from kernel | ✗ 3.5x (wrong concept!) |

**Critical distinction:**
- **GP variance** = uncertainty about regression function f (→ 0 with more data)
- **Posterior variance** = uncertainty about x given observations (→ P_kalman)

**Caveats:**
1. Isomorphism says info IS THERE, but you must LEARN mappings separately
2. Need regression #1 for mean, regression #2 for variance
3. Fair decay = exp((A-LC)·dt) required for fair comparison
4. Level ≥ 2 signatures needed (Lévy area captures variance)

### When Signatures Add Value

For **known linear Gaussian** systems: Kalman is optimal, signatures match but don't beat.

Signatures should help when:
1. **Model misspecification**: true dynamics unknown
2. **Nonlinear observations**: y = h(x) + noise
3. **Non-Gaussian noise**: higher moments matter
4. **Stochastic volatility**: volatility is hidden state

## Implementation Plan

### Phase 1: LQG Baseline ✅
- [x] Implement standard LQG (Kalman + LQR)
- [x] Validate on simple 1D tracking problem
- [x] Document LP vs LQR theory

### Phase 2: Signature Belief ✅
- [x] Online signature filter with Chen's identity
- [x] Compare cumulative vs decay modes
- [x] RBF kernel regression: sig → belief
- [x] Compare to LQG on linear problem

### Phase 3: Signature-KRONIC Control ✅
- [x] Direct sig → control mapping via kernel regression
- [x] Demonstrate model-free control matches LQR
- [x] O(1) online updates via Chen's identity

### Phase 4: Nonlinear POMDP (TODO)
- [ ] Merton-Heston with signature belief
- [ ] Kyle model with signature-based pricing
- [ ] Benchmark against particle filters

### Phase 5: LP Formulation (TODO)
- [ ] Discretize eigenfunction dynamics
- [ ] Formulate LP for terminal utility
- [ ] Add transaction cost constraints

## Related Work

- **KRONIC**: Kaiser et al. (2021) - Koopman control for fully observed systems
- **Signature Methods**: Lyons (1998), Chevyrev-Oberhauser (2022)
- **POMDP Control**: Kaelbling et al. (1998), belief MDP approach
- **LQG**: Classical separation principle
