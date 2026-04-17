# Signature-Based Filtering and Control: A Rigorous Synthesis of the Three Routes

## Abstract

This note gives a unified theoretical account of the three control routes now
used in the repo:

1. **Approach I — factor-reduced / homothetic control**:
   exploit scale invariance to reduce the control problem to a latent factor.
2. **Approach II — stationary transformed-state control**:
   transform a non-ergodic observation path into a stationary lifted state and
   learn/control there.
3. **Approach III — finite-horizon local semigroup control**:
   avoid invariant-measure arguments and work directly with short-horizon local
   generator expansions around a reference policy.

Approach I is the mathematically clean Heston/CRRA benchmark.
Approach II is the best current route to a reusable general architecture.
Approach III is the endgame: the most general control theory, but also the most
technically delicate.

This document focuses on the rigorous structure of Approach III and explains
how Approaches I and II embed into it.  Detailed supporting notes for the first
two routes are:

- [theory_crra_eigenfunction.md](/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/docs/theory_crra_eigenfunction.md:1)
- [theory_ergodic_signatures_and_horizon_selection.md](/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/docs/theory_ergodic_signatures_and_horizon_selection.md:1)

---

## 1. Unified POMDP Setup

Let $(\Omega, \mathcal F, (\mathcal F_t)_{t\ge 0}, \mathbb P)$ be a filtered
probability space supporting a hidden Markov state $X_t$, an observation
process $Y_t$, and an admissible control process $a_t$.

The partially observed control problem is:

$$
J^\pi(t, y_{(-\infty,t]})
:=
\mathbb E^\pi\!\left[
\int_t^T \ell(s, X_s, a_s)\,ds + g(X_T)
\;\middle|\;
Y_{(-\infty,t]} = y_{(-\infty,t]}
\right],
$$

and the objective is to optimize over admissible observation-adapted policies
$\pi$.

The repo’s general strategy is to replace the full observation history by a
lifted state

$$
S_t = \mathcal T_t\!\big(Y_{(-\infty,t]}\big),
$$

where $\mathcal T_t$ may be:

- a classical filter (Kalman, particle filter, etc.),
- a signature / fading-memory transform,
- or any learned belief-state map.

The key question is: when is control on $S_t$ mathematically justified?

### Definition 1.1 (Control-Sufficient Lifted State)

A lifted state $S_t$ is **control-sufficient** if for every admissible future
control sequence, the conditional law of all future rewards and future
observations given the observation history depends on the past only through
$S_t$.

Equivalently, for every bounded measurable future functional $F$,

$$
\mathbb E^\pi\!\left[
F \,\middle|\, Y_{(-\infty,t]}
\right]
=
\mathbb E^\pi\!\left[
F \,\middle|\, S_t
\right].
$$

### Proposition 1.2 (Reduction to a Fully Observed Control Problem)

If $S_t$ is control-sufficient and Markov under admissible controls, then the
original POMDP is equivalent to a fully observed control problem on $S_t$.

#### Proof

By control sufficiency, for every admissible policy the continuation value at
time $t$ depends on the observation history only through $S_t$.  Therefore the
Bellman value function may be written as

$$
V(t, S_t)
=
\sup_{\pi \in \mathcal A}
\mathbb E^\pi\!\left[
\int_t^T \ell(s, X_s, a_s)\,ds + g(X_T)
\;\middle|\;
S_t
\right].
$$

Since $S_t$ is Markov under admissible controls, the family of conditional laws
needed for dynamic programming closes on $S_t$.  Standard controlled Markov
process arguments therefore apply, and the problem reduces to a fully observed
control problem with state variable $S_t$. $\square$

#### Remark 1.3

This proposition is the common foundation of all three routes.

- In Approach I, $S_t$ is the latent factor itself.
- In Approach II, $S_t$ is a stationary transformed path state.
- In Approach III, $S_t$ is whatever lifted state supports a local semigroup
  expansion, even if no invariant measure is available.

---

## 2. Why a Third Route Is Needed

Approaches I and II both lean on structural simplification:

- exact homothetic reduction in Approach I,
- stationary transformed-state theory in Approach II.

But finance control problems often violate at least one of those conveniences:

- raw price or wealth levels are non-ergodic;
- utility need not be homothetic;
- the relevant control objective is finite-horizon rather than stationary;
- the best local coordinate may depend on the current state.

So we need a theory that is:

1. **finite-horizon** rather than invariant-measure by default;
2. **local-around-reference** rather than global in action;
3. valid on a lifted state $S_t$ without requiring exact stationary closure.

That is the role of Approach III.

---

## 3. Finite-Horizon Local Semigroup Theory

Fix a lifted state space $\mathcal S$ and a control set $\mathcal A$.  For a
frozen control $a \in \mathcal A$, let

$$
P_{t,t+h}^a f(s)
:=
\mathbb E\!\left[
f(S_{t+h})
\;\middle|\;
S_t = s,\ a_u \equiv a \text{ on } [t,t+h]
\right]
$$

be the controlled semigroup acting on test functions $f$ in the domain of the
generator.

Assume the frozen-control generator exists:

$$
L^a f(s)
:=
\lim_{h \downarrow 0}
\frac{P_{t,t+h}^a f(s)-f(s)}{h}.
$$

### Proposition 3.1 (Short-Horizon Dynkin Expansion)

Let $f$ lie in the domain of $L^a$.  Then

$$
P_{t,t+h}^a f(s)
=
f(s) + h\,L^a f(s) + o(h)
\qquad \text{as } h \downarrow 0.
$$

#### Proof

This is the definition of the infinitesimal generator.  More explicitly, by
Dynkin’s formula,

$$
\mathbb E\!\left[f(S_{t+h}) \mid S_t=s\right]
=
f(s) + \mathbb E\!\left[\int_t^{t+h} L^a f(S_u)\,du \,\middle|\, S_t=s\right].
$$

If $L^a f$ is continuous at $s$, then the integral equals
$hL^a f(s)+o(h)$, giving the expansion. $\square$

### Definition 3.2 (Reference Policy and Local Coordinate)

Let $a_{\mathrm{ref}}: \mathcal S \to \mathcal A$ be a nominal policy.  A
**local control coordinate** is a map

$$
\Delta a = \mathcal N_s(u, a_{\mathrm{ref}}(s)),
$$

where $u$ is a local coordinate near zero and
$a = a_{\mathrm{ref}}(s) + \Delta a$ is the executed action.

The local coordinate may be:

- additive,
- proportional to the reference action,
- or state-scaled by a local amplitude.

### Proposition 3.3 (Local Quadratic Expansion Around a Reference Policy)

Assume that for each fixed test function $f$ and state $s$,
$a \mapsto L^a f(s)$ is twice continuously differentiable near
$a_{\mathrm{ref}}(s)$.  Then for $\delta a$ small,

$$
P_{t,t+h}^{a_{\mathrm{ref}}+\delta a} f(s)
=
f(s)
+
h\,L^{a_{\mathrm{ref}}}f(s)
+
h\,B_f(s)\cdot \delta a
+
\frac{h}{2}\,\delta a^\top C_f(s)\,\delta a
+
o\!\left(h\|\delta a\|^2 + h\|\delta a\|\right),
$$

where

$$
B_f(s) := \partial_a L^a f(s)\big|_{a=a_{\mathrm{ref}}(s)},
\qquad
C_f(s) := \partial^2_{aa} L^a f(s)\big|_{a=a_{\mathrm{ref}}(s)}.
$$

#### Proof

By Proposition 3.1,

$$
P_{t,t+h}^{a_{\mathrm{ref}}+\delta a} f(s)
=
f(s) + h\,L^{a_{\mathrm{ref}}+\delta a}f(s) + o(h).
$$

Now apply the ordinary second-order Taylor expansion of
$a \mapsto L^a f(s)$ around $a_{\mathrm{ref}}(s)$:

$$
L^{a_{\mathrm{ref}}+\delta a}f(s)
=
L^{a_{\mathrm{ref}}}f(s)
+
\partial_a L^a f(s)\big|_{a=a_{\mathrm{ref}}}\cdot \delta a
+
\frac{1}{2}\delta a^\top
\partial^2_{aa}L^a f(s)\big|_{a=a_{\mathrm{ref}}}
\delta a
+
o(\|\delta a\|^2).
$$

Substituting into the semigroup expansion gives the result. $\square$

#### Remark 3.4

This proposition is the precise finite-horizon justification for fitting a
local quadratic action-response model.  No invariant measure is used.  No
stationary eigenfunction is required.  Only local generator regularity is
required.

---

## 4. Bilinear and Quadratic Lifted Dynamics

Let $\psi: \mathcal S \to \mathbb R^m$ be a vector of lifted observables.
Approach III becomes especially useful when the generator of $\psi$ has a
structured action dependence.

### Proposition 4.1 (Local Bilinear Generator Structure)

Suppose

$$
L^a \psi(s) = A_0(s)\psi(s) + \sum_{i=1}^q a_i A_i(s)\psi(s),
$$

with $a = (a_1,\ldots,a_q)$.  Then around a reference action
$a_{\mathrm{ref}}(s)$ and local deviation $\delta a$,

$$
\mathbb E\!\left[\psi(S_{t+h})-\psi(S_t)\mid S_t=s\right]
=
h\left(
A_{\mathrm{ref}}(s)\psi(s)
+
\sum_{i=1}^q \delta a_i A_i(s)\psi(s)
\right)
+
o(h),
$$

where

$$
A_{\mathrm{ref}}(s) := A_0(s) + \sum_{i=1}^q a_{\mathrm{ref},i}(s)A_i(s).
$$

#### Proof

Apply Proposition 3.1 componentwise to $\psi$:

$$
\mathbb E[\psi(S_{t+h}) \mid S_t=s]
=
\psi(s) + h L^a \psi(s) + o(h).
$$

Substitute the assumed generator form and split
$a = a_{\mathrm{ref}} + \delta a$. $\square$

#### Remark 4.2

This is the rigorous meaning of the phrase “local bilinear control” in the
repo.  It is not a global claim about all actions.  It is a local first-order
claim around a nominal controller.

### Proposition 4.3 (Quadratic Residual as the Second-Order Correction)

Suppose instead that the generator has the local expansion

$$
L^{a_{\mathrm{ref}}+\delta a}\psi(s)
=
A_{\mathrm{ref}}(s)\psi(s)
+
\sum_{i=1}^q \delta a_i B_i(s)\psi(s)
+
\frac{1}{2}
\sum_{i,j=1}^q \delta a_i \delta a_j C_{ij}(s)\psi(s)
+
o(\|\delta a\|^2).
$$

Then:

1. the bilinear model is first-order accurate in $\delta a$;
2. the quadratic residual is the leading second-order correction;
3. estimating only the bilinear term is justified as a Phase 1 baseline.

#### Proof

The stated expansion implies

$$
L^{a_{\mathrm{ref}}+\delta a}\psi
-
\left(
A_{\mathrm{ref}}\psi + \sum_i \delta a_i B_i\psi
\right)
=
O(\|\delta a\|^2).
$$

Applying Proposition 3.1 to both sides shows that the one-step prediction error
of the bilinear model is $O(h\|\delta a\|^2)$, so the bilinear model is
first-order accurate.  The quadratic term is exactly the first missing term.
$\square$

#### Remark 4.4

This proposition is why the finance line should not jump straight to a global
control-quadratic fit.  The mathematically disciplined order is:

1. fit the local bilinear part,
2. verify the first-order signal exists,
3. only then add the quadratic residual.

---

## 5. The Natural Local Coordinate Need Not Be Additive

The current finance debugging made one point unavoidable: the correct local
coordinate is part of the theory, not just implementation detail.

### Proposition 5.1 (Reference-Relative Coordinates)

Suppose the exact optimal control has the form

$$
a^*(s)=a_{\mathrm{ref}}(s)\big(1+u^*(s)\big),
\qquad
a_{\mathrm{ref}}(s)\neq 0.
$$

Then the coordinate

$$
u = \frac{a-a_{\mathrm{ref}}(s)}{a_{\mathrm{ref}}(s)}
$$

is the natural local control variable.  In this coordinate, the optimal action
is represented exactly by $u^*(s)$.

#### Proof

By definition,

$$
a = a_{\mathrm{ref}}(s)(1+u)
\quad \Longleftrightarrow \quad
u = \frac{a-a_{\mathrm{ref}}(s)}{a_{\mathrm{ref}}(s)}.
$$

Substituting $a=a^*(s)$ gives

$$
u = \frac{a_{\mathrm{ref}}(s)(1+u^*(s)) - a_{\mathrm{ref}}(s)}{a_{\mathrm{ref}}(s)}
=
u^*(s).
$$

So the representation is exact in the chosen coordinate. $\square$

### Corollary 5.2 (Stationary Heston-CRRA Relative Overlay)

In the stationary Heston-CRRA benchmark,

$$
\pi^*(v)
=
\pi_{\mathrm{myopic}}(v)\left(1+\frac{\rho\xi p}{\mu-r}\right),
$$

so the natural local coordinate is

$$
u
=
\frac{\pi-\pi_{\mathrm{myopic}}(v)}{\pi_{\mathrm{myopic}}(v)},
$$

and the target overlay is the state-independent constant

$$
u^*=\frac{\rho\xi p}{\mu-r}.
$$

#### Proof

Apply Proposition 5.1 to the multiplicative representation proved in
[theory_crra_eigenfunction.md](/home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control/docs/theory_crra_eigenfunction.md:452). $\square$

#### Remark 5.3

This is the precise theory behind the recent additive-versus-multiplicative
debugging discussion.  If the reference policy is state-dependent, then the
local coordinate must be interpreted **at the current state**, not frozen once
at the initial state unless the benchmark itself is additive.

---

## 6. Relationship Between the Three Approaches

### Proposition 6.1 (Approach I Is a Special Case of Approach III)

Suppose the partially observed control problem admits an exact homothetic
reduction

$$
V(t,w,x) = \frac{w^{1-\gamma}}{1-\gamma}\,\Phi(t,x).
$$

Then the finite-horizon local semigroup theory of Sections 3–5 applied to the
reduced factor state $S_t = X_t$ reproduces the reduced HJB of
Approach I.

#### Proof

The reduced factor process $X_t$ is a fully observed Markov state after the
homothetic reduction.  The controlled semigroup acting on $\Phi(t,\cdot)$ is
exactly the reduced semigroup of the factor process.  Applying the short-time
Dynkin expansion to $\Phi$ yields the reduced HJB from the factor-reduced
problem.  The continuation-value term in Approach I is therefore simply the
finite-horizon local generator term on the reduced state. $\square$

### Proposition 6.2 (Approach II Supplies the State on Which Approach III Acts)

Suppose a transformed state $S_t=\mathcal T_t(Y_{(-\infty,t]})$ is
control-sufficient and stationary.  Then:

1. invariant-measure regression and horizon-selection arguments may be carried
   out on $S_t$ (Approach II);
2. finite-horizon local semigroup control may be carried out on the same state
   $S_t$ (Approach III).

#### Proof

Part 1 follows from control sufficiency plus stationarity, as in the stationary
transform theory.  Part 2 follows from Proposition 1.2 and the local semigroup
expansions of Sections 3–5.  Thus Approach II provides the representation
layer, while Approach III provides the finite-horizon control theory on top of
that representation. $\square$

#### Remark 6.3

This is the main architectural conclusion for the repo:

- Approach I is the exact benchmark when symmetry closes the problem;
- Approach II is the best current candidate for a general reusable state
  representation;
- Approach III is the right long-run controller theory on that state.

---

## 7. Practical Consequences for the Repo

The three-route program should therefore be interpreted as:

1. **Benchmark first with Approach I**.
   Use exact homothetic cases such as Heston-CRRA to verify signs, units, and
   local coordinates.

2. **Generalize the representation via Approach II**.
   Build stationary transformed states using fading-memory/signature machinery,
   because raw finance levels are usually non-ergodic.

3. **Complete the control theory via Approach III**.
   Once the representation is stable, move from invariant-measure heuristics to
   finite-horizon local semigroup control.

### Remark 7.1 (What the Current Heston Gate Does and Does Not Prove)

If a gate fixes an additive overlay once at the initial state and then rolls it
out around a state-adaptive reference policy, it is testing a restricted class
of local controls.  A failure of that gate means:

- the restricted class failed,

not:

- the local-semigroup program is false.

To be faithful to the theory, the local coordinate must be applied through the
current state whenever the benchmark target itself is reference-relative.

### Remark 7.2 (Why This Route Is Still Generalizable)

Approach III does not require:

- homothetic utility,
- an invariant measure on raw levels,
- or a closed-form analytic benchmark.

It requires only:

1. a lifted state $S_t$ on which short-horizon prediction is meaningful;
2. a nominal controller $a_{\mathrm{ref}}(s)$;
3. a local coordinate $u$ around that controller;
4. enough regularity to estimate the first and second local generator response.

That is exactly the level of abstraction needed if the repo is to support both
finance and non-finance partially observed control problems.
