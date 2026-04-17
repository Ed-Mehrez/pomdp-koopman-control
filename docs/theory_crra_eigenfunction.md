# CRRA Utility as an Eigenfunction of the Koopman Generator

## Setup

Consider a wealth process with allocation $\pi$ to a risky asset:

$$dW = W\left[r + \pi(\mu - r)\right]dt + W\pi\sigma \, dB_t$$

where:
- $W$ = wealth
- $r$ = risk-free rate
- $\mu$ = expected return of risky asset
- $\sigma$ = volatility
- $\pi$ = fraction allocated to risky asset

## The Infinitesimal Generator

For a diffusion process $dX = b(X)dt + a(X)dB$, the **infinitesimal generator** $L$ acts on smooth functions $f$ as:

$$Lf = b(X) \cdot \nabla f + \frac{1}{2}a(X)^2 \cdot \nabla^2 f$$

For scalar wealth dynamics:

$$Lf = b(W) \frac{\partial f}{\partial W} + \frac{1}{2}a(W)^2 \frac{\partial^2 f}{\partial W^2}$$

where:
- Drift: $b(W) = W[r + \pi(\mu-r)]$
- Diffusion: $a(W) = W\pi\sigma$

---

## Theorem: $W^{1-\gamma}$ is an Eigenfunction

**Claim:** The function $f(W) = W^{1-\gamma}$ satisfies the eigenvalue equation:

$$L[W^{1-\gamma}] = \lambda \cdot W^{1-\gamma}$$

**Proof:**

### Step 1: Compute derivatives

$$f(W) = W^{1-\gamma}$$

$$f'(W) = (1-\gamma)W^{-\gamma}$$

$$f''(W) = (1-\gamma)(-\gamma)W^{-\gamma-1} = -\gamma(1-\gamma)W^{-\gamma-1}$$

### Step 2: Apply the generator

$$Lf = W[r + \pi(\mu-r)] \cdot (1-\gamma)W^{-\gamma} + \frac{1}{2}(W\pi\sigma)^2 \cdot \left(-\gamma(1-\gamma)W^{-\gamma-1}\right)$$

### Step 3: Simplify first term

$$\text{Term 1} = (1-\gamma)[r + \pi(\mu-r)] \cdot W^{1-\gamma}$$

### Step 4: Simplify second term

$$\text{Term 2} = \frac{1}{2} W^2 \pi^2 \sigma^2 \cdot \left(-\gamma(1-\gamma)\right) W^{-\gamma-1}$$

$$= -\frac{\gamma(1-\gamma)}{2} \pi^2 \sigma^2 \cdot W^{1-\gamma}$$

### Step 5: Combine

$$Lf = (1-\gamma)W^{1-\gamma} \left[ r + \pi(\mu-r) - \frac{\gamma}{2}\pi^2\sigma^2 \right]$$

### Step 6: Identify the eigenvalue

$$\boxed{L[W^{1-\gamma}] = \underbrace{(1-\gamma) \cdot g(\pi, \sigma^2)}_{\lambda} \cdot W^{1-\gamma}}$$

where the **certainty-equivalent growth rate** is:

$$g(\pi, \sigma^2) = r + \pi(\mu-r) - \frac{\gamma}{2}\pi^2\sigma^2$$

$\square$

---

## Interpretation

### The Eigenvalue Equation

We have proven that $\psi(W) = W^{1-\gamma}$ satisfies:

$$L\psi = \lambda \psi$$

with eigenvalue $\lambda = (1-\gamma) \cdot g(\pi, \sigma^2)$.

### Physical Meaning

The eigenvalue $\lambda$ represents the **instantaneous growth rate** of the eigenfunction:

$$\frac{d}{dt}\mathbb{E}[\psi(W_t)] = \lambda \cdot \mathbb{E}[\psi(W_t)]$$

This means:

$$\mathbb{E}[W_t^{1-\gamma}] = W_0^{1-\gamma} \cdot e^{\lambda t}$$

### Connection to Utility Maximization

CRRA utility is $U(W) = \frac{W^{1-\gamma}}{1-\gamma}$, so:

$$\mathbb{E}[U(W_T)] = \frac{W_0^{1-\gamma}}{1-\gamma} \cdot e^{\lambda T}$$

To maximize expected utility, we maximize $\lambda$ (for $\gamma > 1$, $(1-\gamma) < 0$, but maximizing $\lambda$ still maximizes expected utility since utility is negative and we want it "less negative").

---

## Optimal Control: Merton's Formula

To find the optimal allocation $\pi^*$, maximize $g(\pi, \sigma^2)$:

$$\frac{\partial g}{\partial \pi} = (\mu - r) - \gamma \pi \sigma^2 = 0$$

Solving:

$$\boxed{\pi^* = \frac{\mu - r}{\gamma \sigma^2}}$$

This is **Merton's optimal portfolio allocation**.

---

## Why This Matters for Koopman Methods

### Key Insight

KGEDMD learns eigenfunctions $\psi$ and eigenvalues $\lambda$ from data. If we find an eigenfunction that correlates highly with $W^{1-\gamma}$, we have **discovered the utility function without knowing its form**.

### Eigenvalue-Based Control

Once we have the utility eigenfunction $\psi \approx W^{1-\gamma}$:

1. The eigenvalue $\lambda(\pi) = L\psi / \psi$ varies with allocation $\pi$
2. Find $\pi^* = \arg\max_\pi \lambda(\pi)$
3. This recovers Merton's optimal allocation!

### Why Weighted Combinations Fail

If we combine eigenfunctions: $\hat{U} = \sum_i w_i \psi_i$

Then:
$$L\hat{U} = \sum_i w_i L\psi_i = \sum_i w_i \lambda_i \psi_i$$

But:
$$\frac{L\hat{U}}{\hat{U}} = \frac{\sum_i w_i \lambda_i \psi_i}{\sum_i w_i \psi_i} \neq \sum_i w_i \lambda_i$$

The ratio is **not** a constant, so $\hat{U}$ is **not** an eigenfunction. This is why Ridge regression can reconstruct utility values (high $R^2$) but fails for control (wrong $\pi^*$).

---

## Transaction Costs Extension

With transaction cost $\kappa$, we can derive the no-trade region width from the eigenvalue curvature:

$$g''(\pi^*) = -\gamma \sigma^2$$

Using Shreve-Soner asymptotic analysis:

$$\Delta\pi = \left(\frac{3\kappa}{2|g''(\pi^*)|}\right)^{1/3} = \left(\frac{3\kappa}{2\gamma\sigma^2}\right)^{1/3}$$

This gives the width of the no-trade region around $\pi^*$.

---

## Extension: Stochastic Volatility (Heston)

### The Problem

In the Heston model, volatility is itself stochastic:

$$dW = W[r + \pi(\mu-r)]dt + W\pi\sqrt{V_t} \, dB^W_t$$

$$dV = \kappa(\theta - V)dt + \xi\sqrt{V} \, dB^V_t$$

with $\langle dB^W, dB^V \rangle = \rho \, dt$.

**Question:** Is $W^{1-\gamma}$ still an eigenfunction?

### The Generator for $(W, V)$

For the joint process, the generator acts on $f(W, V)$:

$$Lf = b_W \frac{\partial f}{\partial W} + b_V \frac{\partial f}{\partial V} + \frac{1}{2}a_W^2 \frac{\partial^2 f}{\partial W^2} + \frac{1}{2}a_V^2 \frac{\partial^2 f}{\partial V^2} + \rho \, a_W a_V \frac{\partial^2 f}{\partial W \partial V}$$

where:
- $b_W = W[r + \pi(\mu-r)]$, $a_W = W\pi\sqrt{V}$
- $b_V = \kappa(\theta - V)$, $a_V = \xi\sqrt{V}$

### Applying to $f(W,V) = W^{1-\gamma}$

Since $f$ depends only on $W$ (not $V$):

$$\frac{\partial f}{\partial V} = 0, \quad \frac{\partial^2 f}{\partial V^2} = 0, \quad \frac{\partial^2 f}{\partial W \partial V} = 0$$

Therefore:

$$Lf = W[r + \pi(\mu-r)] \cdot (1-\gamma)W^{-\gamma} + \frac{1}{2}(W\pi)^2 V \cdot (-\gamma(1-\gamma)W^{-\gamma-1})$$

$$= (1-\gamma)W^{1-\gamma}\left[r + \pi(\mu-r) - \frac{\gamma}{2}\pi^2 V\right]$$

### The Problem: State-Dependent "Eigenvalue"

We get:

$$L[W^{1-\gamma}] = \underbrace{(1-\gamma) \cdot g(\pi, V)}_{"\lambda(V)"} \cdot W^{1-\gamma}$$

where $g(\pi, V) = r + \pi(\mu-r) - \frac{\gamma}{2}\pi^2 V$.

**This is NOT a proper eigenvalue equation** because $\lambda$ depends on the state variable $V$!

### Resolution: Conditional Eigenfunction

$W^{1-\gamma}$ is an eigenfunction **conditional on $V$**:

- For fixed $V = \sigma^2$, eigenvalue is $\lambda = (1-\gamma)g(\pi, \sigma^2)$
- The optimal allocation $\pi^*(V) = \frac{\mu-r}{\gamma V}$ depends on current volatility

### The Full Value Function

For the HJB equation with stochastic volatility, the value function has the form:

$$V(W, V, t) = \frac{W^{1-\gamma}}{1-\gamma} \cdot h(V, t)$$

where $h(V, t)$ solves a PDE in volatility space. This is the **separation of variables** property of CRRA utility.

**Key insight:** The utility part $W^{1-\gamma}$ factors out, but the "correction" $h(V,t)$ accounts for:
1. Future volatility uncertainty
2. Hedging demand from correlation $\rho$
3. The value of waiting for volatility mean-reversion

### What KGEDMD Learns

When KGEDMD finds an eigenfunction $\psi \approx W^{1-\gamma}$:

1. It discovers the **wealth-dependent part** of the value function
2. The eigenvalue $\lambda(\pi, V) = L\psi/\psi$ is state-dependent
3. Maximizing $\lambda(\pi, V)$ at current $V$ gives the myopic Merton allocation

**This is still useful** because:
- The myopic policy $\pi^* = (\mu-r)/(\gamma V)$ is optimal in many cases
- Hedging demand corrections are often small for typical parameters
- We learn the structure without knowing the utility form

### When Does Myopic = Optimal?

The myopic policy equals the true optimal when:

1. **$\rho = 0$**: No correlation between wealth and volatility shocks
2. **Log utility ($\gamma = 1$)**: Myopia is always optimal
3. **IID volatility**: No predictability to exploit

For $\rho \neq 0$, there's an additional **hedging demand**:

$$\pi^*_{hedge} = -\frac{\rho \xi}{\gamma \sqrt{V}} \cdot \frac{\partial h / \partial V}{h}$$

This requires knowing $h(V,t)$, which KGEDMD does not directly provide.

---

## Proper Eigenfunctions of the Full System

### The Key Insight

If KGEDMD is trained on the **full joint state** $(W, V)$ with correct generator terms for both, it should find eigenfunctions of the form:

$$\psi(W, V) = W^{1-\gamma} \cdot f(V)$$

where $f(V)$ satisfies an ODE that makes $\psi$ a **proper eigenfunction** with constant $\lambda$.

### Derivation

For $\psi(W,V) = W^{1-\gamma} f(V)$, compute derivatives:

$$\frac{\partial \psi}{\partial W} = (1-\gamma)W^{-\gamma} f(V), \quad \frac{\partial \psi}{\partial V} = W^{1-\gamma} f'(V)$$

$$\frac{\partial^2 \psi}{\partial W^2} = -\gamma(1-\gamma)W^{-\gamma-1} f(V), \quad \frac{\partial^2 \psi}{\partial V^2} = W^{1-\gamma} f''(V)$$

$$\frac{\partial^2 \psi}{\partial W \partial V} = (1-\gamma)W^{-\gamma} f'(V)$$

Apply the full generator:

$$L\psi = W^{1-\gamma} \Big[ (1-\gamma)g(\pi,V) f(V) + \kappa(\theta-V) f'(V) + \frac{1}{2}\xi^2 V f''(V) + \rho\xi\pi(1-\gamma)V f'(V) \Big]$$

### The Eigenvalue ODE

For $L\psi = \lambda \psi$, we need $f(V)$ to satisfy:

$$\frac{1}{2}\xi^2 V f''(V) + \Big[\kappa(\theta-V) + \rho\xi\pi(1-\gamma)V\Big] f'(V) + \Big[(1-\gamma)g(\pi,V) - \lambda\Big] f(V) = 0$$

This is exactly the **Riccati-type ODE** from the HJB equation for Heston!

### What This Means for KGEDMD

If we train KGEDMD with:
1. **State**: $(W, V)$ or $(log W, \pi, V)$
2. **Generator terms**: Include BOTH wealth drift/diffusion AND volatility drift/diffusion
3. **Cross terms**: Include the correlation $\rho$ term in the generator

Then KGEDMD should learn eigenfunctions $\psi(W, V)$ that:
- Have **constant eigenvalues** $\lambda$ (not state-dependent!)
- Capture the full value function structure $W^{1-\gamma} f(V)$
- Include the hedging demand information in $f(V)$

### Current Implementation Gap

In `merton_kgedmd_utility.py`, we compute the drift as:

```python
def compute_drift(self, state):
    log_W, pi, v = state
    drift_log_W = self.r + pi * (self.mu - self.r) - 0.5 * pi**2 * v
    drift_v = self.kappa_v * (self.theta_v - v)
    return np.array([drift_log_W, 0.0, drift_v])
```

**Issue**: We include drift for V but not the diffusion terms! The KGEDMD generator matrix $G_{10}$ is computed using only the drift, missing:

1. **Diffusion term for V**: $(1/2)\xi^2 V \cdot \partial^2/\partial V^2$
2. **Cross term**: $\rho \cdot a_W a_V \cdot \partial^2/\partial W \partial V$

### How to Fix

For proper KGEDMD on Heston, we need to compute:

$$G_{10}[i,j] = \dot{X}_i \cdot \nabla_x k(X_i, X_j) + \frac{1}{2}\text{tr}\left(\Sigma(X_i) \cdot \nabla^2_x k(X_i, X_j)\right)$$

where $\Sigma(X) = a(X) a(X)^T$ is the diffusion matrix:

$$\Sigma = \begin{pmatrix} (W\pi)^2 V & \rho W\pi\sqrt{V} \cdot \xi\sqrt{V} \\ \rho W\pi\sqrt{V} \cdot \xi\sqrt{V} & \xi^2 V \end{pmatrix}$$

The second-order term (Itô correction) is what's missing!

### Connection to Signatures

This is exactly where **signatures and lead-lag** come in:
- Lead-lag Lévy area captures quadratic variation (diffusion)
- Level 2 signature terms encode the $\Sigma$ information
- RBF on signatures implicitly includes the Itô correction

**See**: `docs/gedmd_ito_correction.md` and `rkhs_kronic/src/generator_estimator.py`

---

## Eigenvalue Averaging: Why L(U)/U Fails but Separate λᵢ Works

### The Problem

Suppose we reconstruct utility as a weighted combination of eigenfunctions:

$$U(x) \approx \sum_i w_i \psi_i(x)$$

where each $\psi_i$ satisfies $L\psi_i = \lambda_i \psi_i$ (true eigenfunction with constant eigenvalue).

**Question**: How do we maximize $\mathbb{E}[U(W_T)]$ over allocation $\pi$?

### The Wrong Approach: Effective Eigenvalue

A natural but **incorrect** approach is to define an "effective eigenvalue":

$$\lambda_{\text{eff}} = \frac{L(U)}{U} = \frac{L(\sum_i w_i \psi_i)}{\sum_i w_i \psi_i} = \frac{\sum_i w_i \lambda_i \psi_i}{\sum_i w_i \psi_i}$$

**This is NOT constant!** It varies across the state space because the $\psi_i(x)$ have different spatial profiles.

If we average over samples:

$$\bar{\lambda}_{\text{eff}} = \frac{1}{N}\sum_{n=1}^N \frac{L(U)(x_n)}{U(x_n)}$$

This is a **biased estimator** with high variance because we're averaging a ratio of sums.

**Empirical result**: Using $\bar{\lambda}_{\text{eff}}$ to find optimal $\pi^*$ gives 49% error!

### The Correct Approach: Separate Eigenvalue Estimation

The correct approach uses the **linearity of expectation** properly.

**Step 1**: For each eigenfunction separately, estimate $\lambda_i(\pi)$:

$$\lambda_i(\pi) = \frac{1}{N}\sum_{n=1}^N \frac{L\psi_i(x_n)}{\psi_i(x_n)}$$

For true eigenfunctions, $L\psi_i/\psi_i = \lambda_i$ is constant, so this averaging is valid and low-variance.

**Step 2**: Use the exact evolution formula:

$$\mathbb{E}[\psi_i(W_T) | W_0] = e^{\lambda_i T} \psi_i(W_0)$$

**Step 3**: Combine using Ridge weights:

$$\mathbb{E}[U(W_T)] = \sum_i w_i \mathbb{E}[\psi_i(W_T)] = \sum_i w_i e^{\lambda_i(\pi) T} \psi_i(W_0)$$

**Step 4**: Maximize over $\pi$:

$$\pi^* = \arg\max_\pi \sum_i w_i e^{\lambda_i(\pi) T} \psi_i(W_0)$$

### Theorem: Consistency of Separate Estimation

**Proposition**: Let $\{\psi_i\}$ be eigenfunctions of the generator $L$ with eigenvalues $\{\lambda_i\}$. Let $U = \sum_i w_i \psi_i$ be a weighted reconstruction of the value function. Then:

1. **Separate estimation is unbiased**: For each $i$, $\hat{\lambda}_i = \frac{1}{N}\sum_n \frac{L\psi_i(x_n)}{\psi_i(x_n)} \to \lambda_i$ as $N \to \infty$.

2. **Combined estimation is biased**: $\frac{1}{N}\sum_n \frac{L(U)(x_n)}{U(x_n)} \not\to$ any constant in general.

3. **Correct optimization**: $\arg\max_\pi \sum_i w_i e^{\hat{\lambda}_i(\pi) T} \psi_i(W_0) \to \arg\max_\pi \mathbb{E}[U(W_T)]$.

**Proof sketch**:
1. For true eigenfunctions, $L\psi_i/\psi_i = \lambda_i$ everywhere, so averaging any samples gives $\lambda_i$.
2. The ratio $\sum w_i\lambda_i\psi_i / \sum w_i\psi_i$ depends on the sample distribution of $\psi_i$ values, introducing bias.
3. By dominated convergence and continuity of the objective. $\square$

### Empirical Results (vs TRUE Heston Optimal)

**Critical point**: The Merton formula $\pi^* = (\mu-r)/(\gamma\theta_v)$ is **wrong** for Heston!

Under Heston with $\rho < 0$:
- High volatility correlates with price drops
- During high-vol periods, optimal $\pi$ is lower
- Jensen's inequality: $\mathbb{E}[\pi^*(V)] < \pi^*(\mathbb{E}[V])$ due to convexity of $1/V$

**True optimal** (via Monte Carlo): $\pi^*_{\text{Heston}} = 0.70$, not Merton's $0.75$!

| Method | Learned $\pi^*$ | True Heston $\pi^*$ | Error |
|--------|-----------------|---------------------|-------|
| Single eigenfunction | 0.87 | 0.70 | 24.8% |
| Averaged ratio (broken) | 1.12 | 0.70 | 71.4% |
| **Separate + combine** | **0.65** | **0.70** | **7.3%** |
| Merton formula | 0.75 | 0.70 | 7.1% |

**Key finding**: KGEDMD matches Merton accuracy (7.3% vs 7.1%) **without knowing**:
- The utility function form
- The Merton formula
- The model structure

### Practical Implementation

```python
# WRONG: Averaging the ratio
U_approx = sum(w[i] * psi[i] for i in range(n_eigs))
L_U_approx = sum(w[i] * L_psi[i] for i in range(n_eigs))
lambda_eff = mean(L_U_approx / U_approx)  # BIASED!

# CORRECT: Separate estimation, then combine
lambda_i = [mean(L_psi[:, i] / psi[:, i]) for i in range(n_eigs)]
E_U_T = sum(w[i] * exp(lambda_i[i] * T) * psi_0[i] for i in range(n_eigs))
pi_opt = argmax_pi(E_U_T)  # UNBIASED!
```

### Key Insight

The mathematical principle: **eigenvalue estimation must be done per-eigenfunction**.

Once you have $\lambda_i(\pi)$ for each $i$, you can combine them however you want (Ridge weights, etc.) in the **expected utility formula**. But you cannot combine first and then estimate an "effective eigenvalue" - that introduces bias from the nonlinear ratio.

This is analogous to why $\mathbb{E}[X/Y] \neq \mathbb{E}[X]/\mathbb{E}[Y]$ in general.

---

## Part III: Approach I — Factor-Reduced Control Under Homothetic Preferences

The preceding sections explain why CRRA utility naturally interacts with
generator methods.  This section isolates the first of the three theory routes
now guiding the repo:

1. **Factor-reduced / homothetic control**: exploit scale invariance to reduce
   the control problem to the latent factor.
2. **Stationary transformed-state control**: transform a non-ergodic observable
   path into a stationary lifted state and learn/control there.
3. **Finite-horizon local semigroup control**: abandon invariant-measure
   arguments and work directly with short-horizon controlled semigroups.

Approach I is the cleanest Heston benchmark because it produces exact
structure.  Its weakness is equally clear: it depends on homothetic
preferences and multiplicative wealth dynamics.

### 13. General Setup

Let $X_t$ be a latent Markov factor taking values in a state space
$\mathcal{X}$, and let wealth evolve as

$$
dW_t
=
W_t \Big(r(X_t, t) + \pi_t^\top b(X_t, t)\Big)dt
+
W_t \,\pi_t^\top \Sigma(X_t, t)\, dB_t,
$$

where $B_t$ is Brownian motion, $\pi_t$ is an admissible control, and the
factor dynamics are autonomous:

$$
dX_t = \beta(X_t, t)dt + \Gamma(X_t, t)d\widetilde B_t.
$$

The key structural assumption is that the wealth equation is
**multiplicative in wealth** and the factor dynamics do **not** depend on the
wealth level.  Consider CRRA utility

$$
U(w)=\frac{w^{1-\gamma}}{1-\gamma}, \qquad \gamma \neq 1.
$$

Define the finite-horizon value function

$$
V(t,w,x)
:=
\sup_{\pi \in \mathcal{A}}
\mathbb{E}_{t,w,x}\big[U(W_T^\pi)\big].
$$

### Proposition 13.1 (Homothetic Reduction)

Under the multiplicative wealth dynamics above and CRRA utility,

$$
V(t,\lambda w,x)=\lambda^{1-\gamma}V(t,w,x)
\qquad
\text{for every } \lambda > 0.
$$

Consequently there exists a reduced factor value function $\Phi$ such that

$$
V(t,w,x)=\frac{w^{1-\gamma}}{1-\gamma}\,\Phi(t,x).
$$

#### Proof

Fix an admissible control $\pi$.  Because the wealth SDE is linear in the
current wealth level, if $W^{\pi, w}$ denotes the wealth process started at
$W_t = w$, then pathwise

$$
W_s^{\pi, \lambda w} = \lambda W_s^{\pi, w}
\qquad \text{for all } s \in [t,T].
$$

Therefore

$$
U\!\left(W_T^{\pi, \lambda w}\right)
=
\frac{(\lambda W_T^{\pi, w})^{1-\gamma}}{1-\gamma}
=
\lambda^{1-\gamma} U\!\left(W_T^{\pi,w}\right).
$$

Taking conditional expectation and then the supremum over controls yields

$$
V(t,\lambda w,x)
=
\sup_{\pi \in \mathcal A}
\mathbb E_{t,w,x}\!\left[
\lambda^{1-\gamma}U(W_T^{\pi,w})
\right]
=
\lambda^{1-\gamma}V(t,w,x).
$$

Set

$$
\Phi(t,x) := (1-\gamma)V(t,1,x),
$$

and the representation follows by choosing $\lambda = w$. $\square$

#### Remark 13.2

This is the exact mathematical reason Heston-CRRA is such a convenient
benchmark.  The wealth scale disappears and the control problem closes on the
latent factor.  That makes Approach I both rigorous and narrow: the whole
reduction hinges on the homogeneity of CRRA.

### Corollary 13.3 (Reduced HJB on the Latent Factor)

Assume $\Phi$ is smooth enough for dynamic programming.  Then the full HJB for
$V$ reduces to a lower-dimensional HJB for $\Phi$ on factor space:

$$
\partial_t \Phi
+
\mathcal L^X \Phi
+
\sup_{\pi}
\left\{
(1-\gamma)\Phi \,\pi^\top b
-\frac{\gamma(1-\gamma)}{2}\Phi \,\pi^\top \Sigma \Sigma^\top \pi
+
(1-\gamma)\,\pi^\top \Sigma \Gamma^\top \nabla_x \Phi
\right\}
+
(1-\gamma)r\,\Phi
=0,
$$

with terminal condition $\Phi(T,x)=1$.

Here $\mathcal L^X$ is the generator of the latent factor process $X_t$.

#### Proof

Write

$$
V(t,w,x)=\frac{w^{1-\gamma}}{1-\gamma}\Phi(t,x),
$$

compute the derivatives

$$
V_w = w^{-\gamma}\Phi,
\qquad
V_{ww} = -\gamma w^{-\gamma-1}\Phi,
\qquad
\nabla_x V = \frac{w^{1-\gamma}}{1-\gamma}\nabla_x \Phi,
\qquad
\nabla_{wx} V = w^{-\gamma}\nabla_x \Phi,
$$

and substitute into the full HJB for the controlled pair $(W_t, X_t)$.
Every term contains a common factor $w^{1-\gamma}/(1-\gamma)$, which may be
divided out.  The displayed reduced HJB remains. $\square$

#### Remark 13.4

The extra term

$$
\pi^\top \Sigma \Gamma^\top \nabla_x \Phi
$$

is the continuation-value coupling.  This is the precise place where
intertemporal hedging demand enters.  Any controller based only on the
instantaneous one-step wealth quadratic ignores this term and therefore cannot
recover non-myopic demand except by accident.

### 14. Specialization to Heston

For the Heston portfolio problem,

$$
\begin{aligned}
dW_t
&=
W_t\left[r + \pi_t(\mu-r)\right]dt
+
W_t \pi_t \sqrt{V_t}\,dB_t^W, \\
dV_t
&=
\kappa(\theta-V_t)dt + \xi \sqrt{V_t}\,dB_t^V,
\qquad
d\langle B^W, B^V\rangle_t = \rho\,dt.
\end{aligned}
$$

The reduced HJB becomes

$$
\partial_t \Phi
+
\kappa(\theta-v)\,\partial_v \Phi
+
\frac{1}{2}\xi^2 v\,\partial_{vv}\Phi
+
(1-\gamma)r\,\Phi
+
\sup_\pi
\left\{
(1-\gamma)(\mu-r)\pi \Phi
-\frac{\gamma(1-\gamma)}{2}\pi^2 v \Phi
+
(1-\gamma)\rho\xi v \pi \partial_v \Phi
\right\}
=0.
$$

### Proposition 14.1 (Optimal Heston Control in Reduced Form)

Whenever $\Phi(t,v) > 0$ and the HJB is smooth enough, the optimizer is

$$
\pi^*(t,v)
=
\frac{\mu-r}{\gamma v}
+
\frac{\rho \xi}{\gamma}\,\frac{\partial_v \Phi(t,v)}{\Phi(t,v)}.
$$

The first term is the myopic Merton demand.  The second term is the
intertemporal hedging demand.

#### Proof

The term to be maximized over $\pi$ is a concave quadratic:

$$
Q(\pi)
=
(1-\gamma)(\mu-r)\pi \Phi
-\frac{\gamma(1-\gamma)}{2}\pi^2 v \Phi
+
(1-\gamma)\rho\xi v \pi \partial_v \Phi.
$$

Differentiate with respect to $\pi$:

$$
Q'(\pi)
=
(1-\gamma)(\mu-r)\Phi
-\gamma(1-\gamma)\pi v \Phi
+
(1-\gamma)\rho\xi v \partial_v \Phi.
$$

Set $Q'(\pi)=0$ and divide by $(1-\gamma)\Phi$:

$$
(\mu-r) - \gamma \pi v + \rho\xi v \frac{\partial_v \Phi}{\Phi} = 0.
$$

Rearranging yields the formula.  Concavity follows because

$$
Q''(\pi) = -\gamma(1-\gamma) v \Phi < 0
$$

when $\gamma > 1$, $v>0$, and $\Phi > 0$. $\square$

### Proposition 14.2 (Stationary Multiplicative Overlay)

Under the stationary ansatz

$$
\Phi(v)=v^p
$$

for some constant exponent $p$, the optimal control becomes

$$
\pi^*(v)
=
\frac{\mu-r}{\gamma v}
+
\frac{\rho\xi p}{\gamma v}
=
\pi_{\mathrm{myopic}}(v)\left(
1+\frac{\rho\xi p}{\mu-r}
\right).
$$

Hence the natural local control coordinate is the **relative overlay**

$$
u^*
:=
\frac{\pi^*(v)-\pi_{\mathrm{myopic}}(v)}{\pi_{\mathrm{myopic}}(v)}
=
\frac{\rho\xi p}{\mu-r},
$$

which is independent of $v$.

#### Proof

If $\Phi(v)=v^p$, then

$$
\frac{\partial_v \Phi(v)}{\Phi(v)}
=
\frac{p}{v}.
$$

Substituting this into Proposition 14.1 gives

$$
\pi^*(v)
=
\frac{\mu-r}{\gamma v} + \frac{\rho\xi p}{\gamma v}.
$$

Factor out $\pi_{\mathrm{myopic}}(v)=\frac{\mu-r}{\gamma v}$ to obtain the
multiplicative representation. $\square$

#### Remark 14.3

This proposition is exactly why a **state-adaptive proportional overlay**
matches the stationary Heston theory while a **frozen additive overlay** does
not.  If the benchmark target is Proposition 14.2, then the local coordinate
should be

$$
u_t = \frac{\pi_t - \pi_{\mathrm{ref}}(V_t)}{\pi_{\mathrm{ref}}(V_t)},
$$

not a state-independent additive $\Delta \pi$.

### 15. Why Approach I Is Powerful and Why It Is Narrow

### Proposition 15.1 (Failure of the Homothetic Reduction Outside CRRA-Type Symmetry)

Suppose the utility function is not positively homogeneous, or suppose the
wealth dynamics are not multiplicative in wealth.  Then in general there does
not exist a factorization

$$
V(t,w,x) = G(w)\Phi(t,x)
$$

with $G$ depending only on wealth and $\Phi$ depending only on the latent
factor.

#### Proof

The factorization in Proposition 13.1 relied on the exact scaling identity

$$
U(\lambda w) = \lambda^{1-\gamma} U(w)
$$

and the pathwise multiplicative scaling

$$
W^{\pi,\lambda w}_T = \lambda W^{\pi,w}_T.
$$

If either identity fails, then

$$
V(t,\lambda w,x)
=
\sup_\pi
\mathbb E[U(W_T^{\pi,\lambda w})]
$$

cannot be reduced to a scalar multiple of $V(t,w,x)$ by the same argument.
Hence no such separation can be guaranteed in general. $\square$

#### Remark 15.2

This is why Approach I should be treated as:

- a **benchmark route** when homothetic structure exists,
- not the universal architecture for the repo.

It is mathematically exact and computationally attractive, but it excludes:

- CARA utility,
- wealth-dependent frictions,
- state constraints that scale non-homogeneously,
- and many non-finance control problems where there is no wealth homogeneity to
  exploit.

In the project’s current three-route theory:

- Approach I gives the **cleanest validation benchmark**;
- Approach II is the **general stationary-representation route**;
- Approach III is the **general finite-horizon control route**.
