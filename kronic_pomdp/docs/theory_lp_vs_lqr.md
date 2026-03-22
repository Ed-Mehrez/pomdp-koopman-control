# LP vs LQR: When Eigenfunction Control Becomes Linear Programming

## Overview

Standard KRONIC uses LQR because of quadratic running costs. But for **terminal utility maximization** with linear eigenfunction representation, the problem becomes a **Linear Program** (LP).

This is potentially a significant simplification!

---

## Standard KRONIC: LQR Structure

### Setup
- State dynamics: dx = f(x)dt + g(x)u dt (control-affine)
- Koopman lift: dψ = Λψ dt + Bu dt (linear in eigenfunctions)
- Running cost: J = ∫(ψ'Qψ + u'Ru)dt

### Solution
Riccati equation gives optimal gain K:
```
u* = -K ψ(x)
```

The quadratic cost → quadratic value function → Riccati ODE.

---

## Terminal Utility: LP Structure

### Setup
For terminal wealth utility (e.g., Merton portfolio):
- Objective: max E[U(W_T)]
- Eigenfunction representation: U(x) ≈ Σᵢ wᵢ ψᵢ(x)
- Linear dynamics: ψ(T) = e^{ΛT} ψ(0) + ∫₀ᵀ e^{Λ(T-s)} Bu(s) ds

### Key Insight: Linear in Control

Discretizing with time steps Δt:
```
ψ_{k+1} = A_d ψ_k + B_d u_k

where A_d = e^{ΛΔt}, B_d = (∫₀^{Δt} e^{Λs} ds) B
```

Terminal utility becomes:
```
E[U(W_T)] ≈ Σᵢ wᵢ ψᵢ(T) = w' ψ_T
```

Propagating dynamics:
```
ψ_T = A_d^N ψ_0 + Σₖ A_d^{N-k-1} B_d u_k
```

This is **affine in the control sequence** {u_0, ..., u_{N-1}}!

### LP Formulation

**Objective**:
```
max  w' ψ_T  =  w' A_d^N ψ_0 + w' Σₖ A_d^{N-k-1} B_d u_k
```

**Constraints**:
- Control bounds: u_min ≤ u_k ≤ u_max
- State constraints: Can add ψ_k bounds if needed

**This is an LP!**
```
max   c' u
s.t.  u_min ≤ u ≤ u_max
```

where c encodes the eigenfunction propagation.

---

## Transaction Costs: Still LP (or L1 Regularization)

With transaction costs κ|Δπ|:

**Objective**:
```
max  E[U(W_T)] - κ Σₖ |u_k|
```

This is still LP! Introduce slack variables:
```
max  w' ψ_T - κ Σₖ (u_k⁺ + u_k⁻)
s.t. u_k = u_k⁺ - u_k⁻
     u_k⁺, u_k⁻ ≥ 0
```

---

## When Does LP Apply?

| Criterion | LQR | LP |
|-----------|-----|-----|
| Running cost | Quadratic | None (or linear) |
| Terminal cost | Quadratic | Linear in eigenfunctions |
| Control cost | Quadratic | Linear (L1) or box constraints |
| Solution method | Riccati | Standard LP solver |
| Bang-bang? | No | Yes (at corners) |

---

## Implications for POMDP-KRONIC

In the POMDP setting with belief state b:
1. Learn eigenfunctions ψ(b) of belief dynamics
2. Represent terminal utility as U ≈ w' ψ(b)
3. Solve LP for optimal control sequence

**Key advantage**: LP is convex, globally optimal, and scales well!

---

## Connection to Model Predictive Control (MPC)

This is essentially MPC with:
- Eigenfunction-based prediction model
- LP instead of QP for optimization
- Receding horizon implementation

Each step:
1. Observe y, update belief b
2. Compute ψ(b)
3. Solve LP for {u_0, ..., u_{H-1}}
4. Apply u_0, repeat

---

## Example: Merton with Transaction Costs

State: (W, π, V) → Belief: (Ŵ, π, V̂) or signature-based

**Dynamics** (in eigenfunction space):
```
ψ_{k+1} = A(π) ψ_k + B Δπ
```

**Objective**:
```
max  E[W_T^{1-γ}/(1-γ)] - κ Σ|Δπ_k|
    ≈ w' ψ_T - κ Σ|Δπ_k|
```

**LP**:
```
max   w' (A^N ψ_0 + Σ A^{N-k-1} B Δπ_k) - κ Σ(Δπ_k⁺ + Δπ_k⁻)
s.t.  Δπ_k = Δπ_k⁺ - Δπ_k⁻
      Δπ_k⁺, Δπ_k⁻ ≥ 0
      π_min ≤ π_k ≤ π_max
```

This directly gives the no-trade region as the LP solution!

---

## Open Questions

1. **Eigenvalue dependence on control**: If λᵢ = λᵢ(π), dynamics aren't quite linear. Need local linearization or multiple LPs?

2. **Continuous time**: LP is natural for discrete time. What's the continuous-time analog? (Impulse control → singular LP?)

3. **Belief uncertainty**: How does prediction uncertainty affect LP feasibility?

4. **Comparison to HJB**: When does LP give same answer as solving HJB directly?

---

## References

- Kaiser et al. (2021) - KRONIC: Koopman control
- Shreve & Soner (1994) - Transaction costs asymptotics
- Bemporad & Morari (1999) - MPC with constraints
- Boyd & Vandenberghe (2004) - Convex optimization
