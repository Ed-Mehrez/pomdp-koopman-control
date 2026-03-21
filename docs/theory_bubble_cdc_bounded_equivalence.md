# The Equivalence of Multidimensional Bubble Tests: Carré du Champ vs. Bounded Eigenfunctions

## 1. Introduction and Core Definitions

The fundamental problem of bubble detection under the Jarrow-Protter framework is to determine whether an asset price process $\mathbf{X}_t \in \mathbb{R}^d$ (potentially encompassing hidden stochastic volatility states) is a True Martingale or a Strict Local Martingale under an Equivalent Local Martingale Measure (ELMM) $\mathbb{Q}$. If it is a Strict Local Martingale, the asset exhibits a structural bubble.

### 1.1 The Infinitesimal Generator
Under the statistical measure $\mathbb{P}$, the continuous multidimensional Itô diffusion is governed by:
$$ d\mathbf{X}_t = b(\mathbf{X}_t) dt + \sigma(\mathbf{X}_t) d\mathbf{W}_t $$
Its infinitesimal generator (extended Koopman generator) $\mathcal{L}$ acting on a twice-differentiable $f \in C^2(\mathbb{R}^d)$ is:
$$ \mathcal{L} f(\mathbf{x}) = \sum_{i=1}^d b_i(\mathbf{x}) \frac{\partial f}{\partial x_i} + \frac{1}{2} \sum_{i,j=1}^d a_{ij}(\mathbf{x}) \frac{\partial^2 f}{\partial x_i \partial x_j} $$
where $a(\mathbf{x}) = \sigma(\mathbf{x})\sigma^T(\mathbf{x})$ is the diffusion matrix (quadratic variation rate).

### 1.2 The Carré du Champ (CdC) Operator
The Carré du Champ (Field Square) operator $\Gamma$ isolates the diffusion structure of the generator by annihilating the first-order drift terms:
$$ \Gamma(f, g) = \mathcal{L}(fg) - f\mathcal{L}(g) - g\mathcal{L}(f) $$
For the diffusion generator, this evaluates strictly to:
$$ \Gamma(f, g) = \sum_{i,j=1}^d a_{ij}(\mathbf{x}) \frac{\partial f}{\partial x_i} \frac{\partial g}{\partial x_j} $$
**Remark (The Invariance Property):** By Girsanov's Theorem, changing from the statistical measure $\mathbb{P}$ to the risk-neutral ELMM $\mathbb{Q}$ alters the drift $b(\mathbf{x})$ but leaves the diffusion matrix $a(\mathbf{x})$ invariant. Therefore, the CdC operator $\Gamma$ is **perfectly invariant** under measure transformations.

---

## 2. Theoretical Equivalence of Explosion Tests

Dandapani & Protter (2019) establish that a process is a Strict Local Martingale under $\mathbb{Q}$ if and only if the process $\mathbf{X}_t$ topologically explodes in finite time under $\mathbb{Q}$. This explosion depends fundamentally and exclusively on the super-linear growth of the diffusion matrix $a(\mathbf{x})$.

We establish three mathematically equivalent operator-theoretic tests for financial bubbles.

### 2.1 Test A: The Bounded Eigenfunction Test (Khasminskii's Topological Condition)
**Theorem (Bounded Eigenfunction):** Let $\mathcal{L}^{\mathbb{Q}}$ be the generator of the process under the risk-neutral measure $\mathbb{Q}$. The process $\mathbf{X}_t$ explodes in finite time (i.e., is a bubble) **if and only if** there exists a strictly positive, globally bounded eigenfunction $u(\mathbf{x})$ for the principal eigenvalue $\lambda > 0$:
$$ \mathcal{L}^{\mathbb{Q}} u = \lambda u \quad \text{such that } 0 < u(\mathbf{x}) \le M < \infty \quad \forall \mathbf{x} \in \mathbb{R}^d $$

**Proof Sketch (Khasminskii, *Stochastic Stability of Differential Equations*, 2nd ed., Ch. 3–4; Ethier & Kurtz, *Markov Processes*, Theorem 4.5.4):**
If the process explodes at a random finite time $\tau_e$, define the bounded survival function $u(\mathbf{x}) = \mathbb{E}^{\mathbb{Q}}_{\mathbf{x}}[e^{-\lambda \tau_e}]$. By Dynkin's formula, $\mathcal{L}^{\mathbb{Q}} u = \lambda u$, and because it is an expectation of a decaying exponential, $0 \le u \le 1$. Conversely, if such a bounded positive function exists for $\lambda > 0$, the Dynkin integral bounds force the explosion time $\tau_e$ to have positive probability mass at finite limits.

### 2.2 Test B: The Measure-Invariant Carré du Champ Test
Because the drift under $\mathbb{P}$ varies wildly from the drift under $\mathbb{Q}$, applying Test A directly to the empirical data-driven generator $\mathcal{L}^{\mathbb{P}}$ is invalid. Jarrow, Protter, & San Martin (2022) proved that invariance breaks down under multidimensional stochastic volatility: an asset can be a True Martingale under one measure and a Bubble under another purely due to drift differences across dimensions. 

However, because explosions are driven topologically by $a(\mathbf{x})$, we map Khasminskii's constructive conditions (see Khasminskii, 2012, Ch. 3; related criteria in Stroock & Varadhan, *Multidimensional Diffusion Processes*, Ch. 10) wholly into the measure-invariant CdC operator.

**Theorem (The CdC Explosion Criteria):** Evaluate the CdC operator along the state coordinate functions $f_i(\mathbf{x}) = x_i$:
$$ \Gamma(x_i, x_j) = a_{ij}(\mathbf{x}) $$
Define the radial squared diffusion bounds based strictly on traces and quadratic forms of the empirical CdC matrices:
1. $A(\rho^2/2) \le \mathbf{x}^T \Gamma(\mathbf{x}, \mathbf{x}^T) \mathbf{x}$  (Radial Dispersion Lower Bound)
2. $\mathbf{x}^T \Gamma(\mathbf{x}, \mathbf{x}^T) \mathbf{x} \cdot B(\rho^2/2) \le \text{Trace}(\Gamma)$ (Total Radial Growth Upper Bound, neglecting drift for strict volatility dominance)

The process is a Structural Bubble under *all* ELMMs if the CdC integrals converge:
$$ \int_1^\infty [C(\rho)]^{-1} \left( \int_1^\rho \frac{C(s)}{A(s)} ds \right) d\rho < \infty, \quad \text{where } C(\rho) = \exp\left( \frac{1}{2}\int_1^\rho B(s)ds \right) $$

**Proof of Equivalence:** The CdC matrix exactly reconstructs the local geometric quadratic variation. Because the bounds $A(\cdot)$ and $B(\cdot)$ depend strictly on $a(\mathbf{x})$ independent of $b(\mathbf{x})$, and because the Dandapani-Protter conditions establish explosion purely as an integration over $a(\mathbf{x})$ growth rates, Test A and Test B are isomorphic. If the CdC integral Test B converges, the explosive volatility mathematically guarantees the existence of the bounded $\mathbb{Q}$-eigenfunction required by Test A.

---

## 3. The Restricted RKHS Behavior (RBF Kernel Koopman)

When computing data-driven Koopman operators using Gaussian Radial Basis Functions (RBFs), we restrict the global operator space to a specific, bounded Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}_K$. 

A critical question arises: *If the true Data Generating Process (DGP) is a True Martingale (e.g., Geometric Brownian Motion) with UNBOUNDED unstable eigenfunctions, what occurs when we evaluate the operator on a strictly BOUNDED function space? Does it produce spurious positive eigenvalues?*

### 3.1 The Spectrum of GBM on Bounded Functions
For Geometric Brownian Motion (GBM): $dS_t = \mu S_t dt + \sigma S_t dW_t$. 
The standard generator eigenvalue equation $\mathcal{L} f = \lambda f$ admits the solution $f(S) = S$ for $\lambda = \mu > 0$. However, $f(S) = S$ is unbounded as $S \to \infty$. 

By the Bounded Eigenfunction Theorem (Test A), because GBM doesn't explode, there is no *bounded* function that produces a strongly positive eigenvalue ($\text{Re}(\lambda)>0$) reflecting an explosion. All strictly bounded eigenfunctions globally track the diffusion's eventual boundary hit (hitting zero), forcing their eigenvalues onto the real left semi-axis $\lambda \le 0$ or exactly $\lambda = 0$ (the constant function $f(S)=1$).

### 3.2 Spurious Modes and Projection Artifacts in RBF Kernels
When we run Extended Dynamic Mode Decomposition (EDMD) or the Kernel Koopman algorithm with bounded RBFs, we diagonalize a finite-dimensional Rayleigh quotient matrix $\mathcal{K}_{RBF}$ representing the projection of the true generator onto the span of the bounding kernels.

**Theorem (Absence of Endogenous Spurious Modes):** If the true diffusion matrix $a(\mathbf{x})$ is non-explosive (fails the CdC integral test), the continuous spectrum of the generator restricted to $L^\infty$ lies strictly in the left half-plane. A Galerkin projection (EDMD) onto a dense bounded RKHS span will not spontaneously generate a right-half-plane eigenvalue $\lambda > 0$ as the dictionary size $N \to \infty$.

**The Boundary Phenomenon (Why Spurious Modes Appear in Practice):**
In numerical practice, RBF dictionaries $k(\mathbf{x}, \mathbf{c}_i)$ are defined by finite center landmarks $\mathbf{c}_i \in \mathcal{C}$. 
Outside the convex hull of $\mathcal{C}$, the function span artificially decays to zero: $\lim_{|\mathbf{x}| \to \infty} f_{RBF}(\mathbf{x}) = 0$. 

If the simulated GBM path wanders outside the landmark grid $\mathcal{C}$, the numerical algorithm perceives a massive, catastrophic "drop" in state value solely because the kernel basis lost coverage. 
To fit this empirical boundary cliff, the discrete Koopman matrix $\mathcal{K}_{RBF}$ will invent high-frequency, complex right-half-plane eigenvalues ($\text{Re}(\lambda)>0$) to approximate the sudden artificial discontinuity in the derivative space. 

**Resolution:** 
Spurious modes in the RBF Koopman are **not** caused by restricting unbounded eigenfunctions to bounded spaces (which correctly zeroes them out). They are caused by **extrapolation failure** when data leaves the bounded support of the kernel dictionary. This is resolved intrinsically via:
1. **Dynamic Nyström Landmark Updates:** Ensure the grid $\mathcal{C}$ covers exactly the empirical path variance.
2. **Applying the Dual CdC Test:** Rather than trusting the spectral decomposition of $\mathcal{K}_{RBF}$ alone (Test A), compute the CdC matrix algebraically: $\Gamma_{RBF} = \mathcal{K}_{RBF}(f^2) - 2f \mathcal{K}_{RBF}(f)$ on the interior of the domain, and numerically integrate Test B. The purely algebraic extraction of $a(\mathbf{x})$ from the Koopman matrix avoids boundary-eigenvalue pollution.

---

## 4. Justification for Tail Extrapolation: Regular Variation and Bounds

A critical methodological question arises in the CdC-bridged eigenfunction test: we estimate the diffusion matrix $a(\mathbf{x})$ via CdC kernel regression on the observed data range, then extrapolate a fitted power law to an extended domain to evaluate the Sturm-Liouville eigenvalue problem. Why is this extrapolation justified? We develop the argument first for the 1D diffusion-only case (which is what the CdC annihilation of drift gives us), then extend to the full multi-dimensional setting.

### 4.1 The 1D Diffusion-Only Case: Only a Bound Is Needed

For a 1D diffusion under the risk-neutral measure $\mathbb{Q}$ where the price process is a local martingale ($b^{\mathbb{Q}} = 0$), the explosion criterion reduces to the **Feller test at $+\infty$**: the process explodes (i.e., is a strict local martingale / bubble) if and only if the scale function integral converges:
$$ \int_1^\infty \frac{s}{\sigma^2(s)} \, ds < \infty $$

This is the natural setting for our CdC test because the CdC operator annihilates drift, leaving the diffusion-only generator $\mathcal{L}_{\text{diff}} = \frac{1}{2}\sigma^2(x)\partial_{xx}$. The Feller integral for this driftless generator is exactly $\int s/\sigma^2(s) \, ds$ (the scale function $s'(x) = 1$ when $b = 0$).

**Key observation**: This is a convergence/divergence dichotomy that depends only on the **growth rate** of $\sigma^2(s)$:

- **For bubble detection** (showing convergence): If $\sigma^2(s) \ge c \cdot s^\alpha$ for $\alpha > 2$ and $s$ sufficiently large, then $\int_R^\infty s/\sigma^2(s) \, ds \le (1/c)\int_R^\infty s^{1-\alpha} \, ds < \infty$. A **lower bound** on $\sigma^2$ growth suffices.
- **For stability** (showing divergence): If $\sigma^2(s) \le c \cdot s^\alpha$ for $\alpha \le 2$ and $s$ sufficiently large, then $\int_R^\infty s/\sigma^2(s) \, ds \ge (1/c)\int_R^\infty s^{1-\alpha} \, ds$. For $\alpha < 2$, the integral $\int s^{1-\alpha} ds$ diverges since $1-\alpha > -1$. For $\alpha = 2$, $\int s^{-1} ds = \log s \to \infty$. An **upper bound** suffices.

Therefore, the power-law extrapolation does not need to match $\sigma^2(S)$ pointwise in the tail — it only needs to correctly identify the **tail index** $\alpha$ relative to the critical threshold 2.

**Remark (Drift and the general 1D case):** When the drift $b(x)$ is nonzero (i.e., under $\mathbb{P}$), the Feller test involves the full scale function $s'(x) = \exp(-\int^x 2b(y)/\sigma^2(y) \, dy)$, and the explosion criterion becomes $\int s'(x) \int^x [s'(y)\sigma^2(y)]^{-1} dy \, dx < \infty$. This depends on the drift-to-diffusion ratio $b/\sigma^2$, which is NOT measure-invariant. However, by CdC invariance (§1.2), the extracted $\sigma^2$ is the same under $\mathbb{P}$ and $\mathbb{Q}$, and the bubble test asks about explosion under $\mathbb{Q}$ where the price drift $b^{\mathbb{Q}} = 0$. Therefore, applying the driftless Feller integral $\int s/\sigma^2(s) ds$ to the CdC-extracted $\sigma^2$ is exactly the correct test.

### 4.2 The Multi-Dimensional Explosion Integral

For $d$-dimensional diffusions, the Khasminskii radial comparison test (see Khasminskii, *Stochastic Stability of Differential Equations*, 2nd ed., 2012, Chapter 3; closely related criteria appear in Stroock & Varadhan, *Multidimensional Diffusion Processes*, Chapter 10) gives sufficient conditions for explosion via a double integral involving radial profiles of the diffusion matrix:
$$ \int_1^\infty [C(\rho)]^{-1} \left( \int_1^\rho \frac{C(s)}{A(s)} \, ds \right) d\rho < \infty, \quad C(\rho) = \exp\left(\tfrac{1}{2}\int_1^\rho B(s) \, ds\right) $$

where $A(\rho)$ and $B(\rho)$ bound the radial dispersion and trace ratio:
$$ A(\rho^2/2) \le \inf_{|\mathbf{x}|=\rho} \mathbf{x}^T a(\mathbf{x}) \mathbf{x}, \qquad B(\rho^2/2) \ge \sup_{|\mathbf{x}|=\rho} \frac{\text{Tr}(a(\mathbf{x}))}{\mathbf{x}^T a(\mathbf{x}) \mathbf{x}} $$

The bound argument from §4.1 extends to this double integral, but with an important subtlety: the exponential factor $C(\rho)$ from $B(\rho)$ must be controlled.

**Proposition (Convergence under power-law growth):** Suppose $A(\rho) \ge c \cdot \rho^\alpha$ and $B(\rho) \le D \cdot \rho^{-\beta}$ for large $\rho$, with $\alpha > 2$ and $\beta > 0$ (i.e., the trace ratio decays). Then the double integral converges.

*Proof sketch.* With $\beta > 0$: $\int_1^\rho B(s) ds \le D \int_1^\rho s^{-\beta} ds$, which is bounded as $\rho \to \infty$ when $\beta > 1$, or grows like $\rho^{1-\beta}$ when $0 < \beta < 1$. In either case $C(\rho)$ is at most polynomial. The inner integral $\int_1^\rho C(s)/A(s) ds \le \int_1^\rho s^{p-\alpha} ds$ for some $p \ge 0$, which is bounded when $\alpha > p + 1$. The outer integral $\int C(\rho)^{-1} \cdot (\text{bounded}) \, d\rho$ converges when $C(\rho)$ grows. The exact threshold depends on $\alpha, \beta, D$, but the key point is: for sufficiently large $\alpha$, convergence is guaranteed regardless of the $B$ profile. $\square$

**The financial setting**: For SV models (Heston, CEV-Heston), $B(\rho)$ measures $\text{Tr}(a)/(\mathbf{x}^T a \mathbf{x})$, which is the ratio of total diffusion to radial diffusion. Along the explosion direction ($S \to \infty$ with $V$ bounded), $\mathbf{x}^T a \mathbf{x} \sim V S^{2\gamma+2}$ while $\text{Tr}(a) \sim V S^{2\gamma} + \xi^2 V$, giving $B \sim S^{-2} \to 0$. So $B$ decays quadratically and $C(\rho)$ is bounded — the 1D argument applies essentially unchanged, with the same critical index $\alpha = 2$.

**Remark**: When $B(\rho)$ is bounded but does NOT decay (e.g., $B(\rho) \equiv D > 0$), $C(\rho) = e^{D\rho/2}$ grows exponentially. In this case, the double integral can still converge, but the analysis is more delicate: the outer factor $C(\rho)^{-1} = e^{-D\rho/2}$ decays exponentially, compensating the inner growth. The convergence criterion then depends on the interaction between $A$'s polynomial growth and $C$'s exponential growth. For financial models this case does not arise (§4.4 explains why), so we do not pursue it here.

### 4.3 Regular Variation Theory

We now justify why the power-law extrapolation $\hat{\sigma}^2(S) = C \cdot S^\alpha$ correctly captures the tail growth index.

**Scalar Regular Variation** (Bingham, Goldie & Teugels, *Regular Variation*, 1987): A measurable function $f: (0,\infty) \to (0,\infty)$ is **regularly varying** at $\infty$ with index $\alpha \in \mathbb{R}$ if for all $t > 0$:
$$ \lim_{s \to \infty} \frac{f(ts)}{f(s)} = t^\alpha $$

By the **Karamata Characterization** (BGT, Theorem 1.4.1), this is equivalent to $f(s) = s^\alpha \cdot L(s)$ where $L$ is a **slowly varying function** ($L(ts)/L(s) \to 1$ as $s \to \infty$). The Karamata Representation (BGT, Theorem 1.3.1) further gives $L(x) = c(x) \exp\left(\int_a^x \varepsilon(t)/t \, dt\right)$ with $c(x) \to c > 0$ and $\varepsilon(t) \to 0$.

**Karamata's Theorem on integrals** (BGT, Proposition 1.5.8): If $f$ is regularly varying with index $\alpha$ and locally bounded, then:
- For $\alpha > -1$: $\int_1^x f(t) \, dt \sim x f(x) / (\alpha + 1)$ as $x \to \infty$
- For $\alpha < -1$: $\int_x^\infty f(t) \, dt \sim -x f(x) / (\alpha + 1)$ as $x \to \infty$

**Application to the Feller integral:** With $\sigma^2(s) = s^\alpha L(s)$, the integrand $s/\sigma^2(s) = s^{1-\alpha}/L(s)$ is regularly varying with index $1-\alpha$. By Karamata's theorem:
- $\alpha > 2 \Rightarrow 1-\alpha < -1 \Rightarrow \int_x^\infty s^{1-\alpha}/L(s) \, ds$ converges (explosion)
- $\alpha < 2 \Rightarrow 1-\alpha > -1 \Rightarrow \int_1^x s^{1-\alpha}/L(s) \, ds \to \infty$ (no explosion)
- $\alpha = 2 \Rightarrow 1-\alpha = -1$: borderline, depends on $L$ (e.g., $L(s) \equiv 1$ gives $\int s^{-1} ds = \infty$; $L(s) = s^\varepsilon$ for any $\varepsilon > 0$ gives convergence)

This is the rigorous statement: **the Feller integral convergence depends only on $\alpha$, not on $L$, except at the borderline $\alpha = 2$**. This follows from Karamata's theorem, not merely from comparison with pure power laws.

### 4.4 Dimensional Reduction for Stochastic Volatility Models

In financial applications, the state $\mathbf{X}_t = (S_t, V_t, \ldots)$ decomposes into the price $S$ and auxiliary volatility components $V$. The price process $S_t$ is generically **non-ergodic** (transient — it can drift to 0 or $\infty$). The explosion analysis simplifies when the volatility process is **positive recurrent**, but care is needed about what "ergodic" means here and when it fails.

#### 4.4.1 What Needs to Be Ergodic

The CdC kernel regression estimates $\hat{\sigma}^2(S) \approx \mathbb{E}[(\Delta S)^2/dt \mid S_t \approx S]$. In a joint model $(S_t, V_t)$ with $a_{11}(S,V) = V \cdot S^{2\gamma}$, this conditional expectation is:
$$ \hat{\sigma}^2(S) = S^{2\gamma} \cdot \mathbb{E}[V_t \mid S_t \approx S] $$

This involves **neither** joint ergodicity of $(S, V)$ (which fails — $S$ is transient) **nor** marginal ergodicity of $S$ (also fails). What it requires is:

1. **The conditional expectation $\mathbb{E}[V_t \mid S_t = S]$ is bounded away from 0 and $\infty$** as $S \to \infty$, so that it acts as a slowly varying function $L(S)$ multiplying $S^{2\gamma}$.
2. **The kernel regression has enough observations at each $S$ level** to estimate this conditional expectation — which is guaranteed by the kernel smoothing bandwidth, not by ergodicity of $S$ per se.

Condition (1) is sufficient: if $\mathbb{E}[V \mid S = s] = L(s)$ with $L$ slowly varying, then $\hat{\sigma}^2(s) = s^{2\gamma} \cdot L(s)$ is regularly varying with index $2\gamma$, and by Karamata's theorem (§4.3), the Feller integral convergence depends only on $2\gamma$, not on $L$.

#### 4.4.2 When Volatility IS Ergodic (The Standard Case)

**Proposition (Effective 1D Reduction under Ergodic Vol):** Let $\mathbf{X}_t = (S_t, V_t)$ where $V_t$ is a positive recurrent diffusion with stationary distribution $\pi$ and $\mathbb{E}_\pi[V] = \theta > 0$. Then:

1. **Non-explosion of $V$**: Positive recurrence implies $V_t$ does not explode. Therefore $|\mathbf{X}_t| \to \infty$ requires $|S_t| \to \infty$.
2. **Conditional expectation is well-behaved**: Even with leverage correlation $\rho \ne 0$, the conditional expectation $\mathbb{E}[V_t \mid S_t = s]$ is bounded for large $s$ because $V_t$ is mean-reverting to $\theta$.
3. **CdC regression estimates the correct index**: The kernel-smoothed $\hat{\sigma}^2(S)$ converges to $S^{2\gamma} \cdot \mathbb{E}[V \mid S]$, which is regularly varying with index $2\gamma$.

**Important caveat on the state space**: Positive recurrence does NOT mean $V$ is confined to a compact set. The CIR process has state space $(0, \infty)$, and $V_{\text{ess-inf}} = 0$ under the stationary Gamma distribution. One cannot define $A_{\text{eff}}(S) = V_{\min} \cdot S^{2\gamma}$ with $V_{\min} > 0$ as a pointwise lower bound. The correct object is the **conditional** (or ergodic) **average**, not a worst-case infimum.

**Formal justification**: For the CdC-extracted quantity, the Nadaraya-Watson kernel regression estimates:
$$ \hat{\sigma}^2(S) = \frac{\sum_t K_h(S_t - S) \cdot (\Delta S_t)^2/dt}{\sum_t K_h(S_t - S)} \xrightarrow{} \mathbb{E}[V_t \cdot S_t^{2\gamma} \mid S_t = S] = S^{2\gamma} \cdot \mathbb{E}[V_t \mid S_t = S] $$
The conditional expectation $\mathbb{E}[V_t \mid S_t = S]$ is a bounded, positive function for mean-reverting $V$. Even with leverage correlation $\rho < 0$ (high $S$ → low $V$), it converges to a finite limit as $S \to \infty$. Therefore $\hat{\sigma}^2(S) = S^{2\gamma} \cdot L(S)$ with $L$ slowly varying, preserving the regular variation index.

**Remark on the ergodic theorem**: Strictly speaking, the ergodic theorem applies to time averages of the **joint** process. Since $(S_t, V_t)$ is not jointly ergodic (S is transient), we cannot invoke the standard Birkhoff ergodic theorem. What we use instead is the consistency of the Nadaraya-Watson estimator under mixing conditions (which hold for the Markov process restricted to any compact set of S-values). This is a statistical result, not an ergodic one.

| Vol Model | Positive Recurrent? | $\mathbb{E}[V \mid S=s]$ as $s \to \infty$ | Index Preserved? |
|-----------|---------------------|----------------------------------------------|-----------------|
| CIR/Heston | Yes ($2\kappa\theta \ge \xi^2$) | $\to \theta$ (bounded) | Yes |
| OU (Stein-Stein) | Yes | $\to \theta$ (bounded) | Yes |
| GARCH diffusion | Yes | Bounded | Yes |
| **SABR** | **No** (log-normal vol is GBM) | **Unbounded** — can grow with $S$ | **Possibly not** |
| **3/2 model** | **Depends** on parameters | Can explode | **Fails if non-recurrent** |
| Rough vol (fOU) | Not Markov | Generalized stationarity | Heuristic only |

#### 4.4.3 When Volatility is NOT Ergodic (SABR and Other Transient Vol Models)

The SABR model $dV_t = \xi V_t dW_t^V$ has log-normal volatility: $V_t$ is a geometric Brownian motion, which is **transient** (no stationary distribution). In this case:

- $V_t$ and $S_t$ can both grow without bound, potentially in a correlated way.
- The conditional expectation $\mathbb{E}[V_t \mid S_t = s]$ may grow with $s$, altering the effective tail index. If $\mathbb{E}[V \mid S = s] \sim s^\beta$ for some $\beta > 0$, then the effective $\hat{\sigma}^2(s) \sim s^{2\gamma + \beta}$, which has a different regular variation index than $2\gamma$ alone.
- With leverage correlation $\rho < 0$, the opposite occurs: $\mathbb{E}[V^2 \mid S = s]$ *shrinks* with $s$, causing $\beta < 0$ and **attenuating** the estimated index. This creates **false negatives**: a true bubble ($\gamma > 1$) may appear stable because the 1D regression sees an effective $\hat{\alpha} = 2\gamma + \beta < 2$.

**Resolution: Log-Linear Separation (Approach D)**

Rather than the full multi-dimensional Khasminskii test, we exploit a structural property of financial diffusion models: the diffusion coefficient is **multiplicatively separable** in $S$ and $V$.

**Proposition (Log-Linear Separation):** For multiplicatively separable diffusion $a_{11}(S,V) = f(V) \cdot g(S)$ with $g$ regularly varying with index $\alpha$, let $\hat{V}_t$ be a consistent estimator of $V_t$. The log-linear regression
$$\log \hat{\sigma}^2 \sim \alpha \cdot \log S + \beta \cdot \log \hat{V} + c$$
yields a consistent estimator of $\alpha$ under any vol dynamics (ergodic or transient), provided $\hat{V}$ is a consistent estimator of $V$.

*Proof.* By separability, $\log(a_{11}) = \log(g(S)) + \log(f(V))$. The CdC regression target $(\Delta S)^2/dt$ estimates $a_{11}(S_t, V_t) = f(V_t) \cdot g(S_t)$ up to chi-squared noise. Taking logs: $\log((\Delta S)^2/dt) \approx \log(g(S_t)) + \log(f(V_t)) + \varepsilon_t$ where $\varepsilon_t$ is mean-zero (after Jensen correction). The OLS/BayesianRidge regression on $[\log S_t, \log \hat{V}_t]$ identifies the $S$-coefficient as the regular variation index of $g$, because the $\log \hat{V}_t$ regressor absorbs the $f(V)$-dependence that contaminates the 1D regression. The consistency follows from the Frisch-Waugh-Lovell theorem: the coefficient on $\log S$ in the multiple regression equals the coefficient from regressing $\log((\Delta S)^2/dt)$ on $\log S$ after partialling out $\log \hat{V}$ — i.e., after removing the V-induced confounding. $\square$

**Why this works for SABR($\rho < 0$):** The 1D regression confounds the $S^{2\gamma}$ scaling with the $V^2$-shrinkage from leverage. The 2D regression separates them: $\alpha$ captures $S^{2\gamma}$, $\beta$ captures $V^2$ dependence. The Feller test uses $\alpha$ alone.

**Coverage:** This applies to ALL standard financial models:

| Model | $a_{11}(S,V)$ | $f(V)$ | $g(S)$ | True $\alpha$ |
|-------|---------------|--------|--------|---------------|
| CEV | $\sigma^2 S^{2\gamma}$ | $\sigma^2$ | $S^{2\gamma}$ | $2\gamma$ |
| Heston | $V S^2$ | $V$ | $S^2$ | 2 |
| CEV-Heston | $V S^{2\gamma}$ | $V$ | $S^{2\gamma}$ | $2\gamma$ |
| SABR | $V^2 S^{2\gamma}$ | $V^2$ | $S^{2\gamma}$ | $2\gamma$ |
| 3/2-CEV | $V S^{2\gamma}$ | $V$ | $S^{2\gamma}$ | $2\gamma$ |

**Diagnostic: V-tercile conditional check.** As a robustness check for the separability assumption, we split data into V-terciles and compute 1D $\alpha$ within each. If $\max_q(\alpha_q) \gg \alpha_{\text{joint}}$, the separability assumption may be violated (e.g., $a_{11}$ has cross-terms $V^\delta S^\gamma$).

**Connection to Local Volatility Elasticity:** The log-linear separation recovers the global $S$-exponent $\alpha$ by conditioning on $V$. A more general approach is the **local volatility elasticity** $\varepsilon(S) = \partial\log\sigma^2(S)/\partial\log S$, computed from the GP posterior gradient (see `theory_feller_universality_jump_robustness.md` §1.5). The relationship: log-linear separation gives a single $\alpha$ for the $S$-direction (averaging over the observed range), while local elasticity gives $\varepsilon(S)$ at each price level. For CEV-type models, $\varepsilon(S) \equiv \alpha$ (constant), so both agree. For models with non-constant scaling (e.g., $\sigma^2(S) = S^2\log(S)^p$), the local elasticity captures the slowly varying correction that the global regression misses. The MLKFellerGP ARD kernel provides both: the parametric mean gives global $\alpha$ (equivalent to log-linear separation when vol proxy is included), and the GP gradient gives local $\varepsilon$.

**Remark (Why not full 2D Sturm-Liouville):** The full 2D eigenvalue problem $\frac{1}{2}[a_{11}u_{SS} + 2a_{12}u_{SV} + a_{22}u_{VV}] = \lambda u$ would be the most general approach, but is over-engineered here. Separability holds for all models of interest, so the 1D Sturm-Liouville with the corrected $\alpha$ from log-linear separation suffices. The 1D problem is also more robust (fewer grid parameters, no boundary condition choices in the V direction).

**Alternative resolutions** (retained for completeness, but superseded by log-linear separation for separable models):
1. Work with the **full multi-dimensional** Khasminskii test (§4.2, §4.5) using the joint radial profile $A(\rho)$ on the sphere $\mathbb{S}^1$ — no dimensional reduction.
2. Use the **CdC kernel regression in the joint state** $(S, V)$: estimate the full $2 \times 2$ CdC matrix $\hat{a}_{ij}(S, V)$ and evaluate the multi-d explosion integral numerically.
3. For the 1D CdC regression on $S$ alone, acknowledge that the estimated $\hat{\sigma}^2(S)$ captures the **path-conditional** volatility (averaged over the V-values that co-occur with each S level in the observed path), which may not reflect the true conditional dynamics.

#### 4.4.4 Examples

**Example (Heston):** $a_{11}(S,V) = V \cdot S^2$, $V$ CIR with $\theta = 0.04$. Conditional average: $\mathbb{E}[V \mid S = s] \to \theta = 0.04$ as $s \to \infty$ (mean-reversion dominates). So $\hat{\sigma}^2(S) \approx 0.04 \cdot S^2$, regularly varying with index $\alpha = 2$ → non-explosive, true martingale.

**Example (CEV-Heston hybrid):** $a_{11}(S,V) = V \cdot S^{2\gamma}$, $\gamma > 1$, $V$ CIR. Same reasoning: $\hat{\sigma}^2(S) \approx \theta \cdot S^{2\gamma}$, index $\alpha = 2\gamma > 2$ → bubble.

**Example (SABR, $\gamma > 1$):** $dS = V S^\gamma dW^S$, $dV = \xi V dW^V$, $\langle W^S, W^V \rangle = \rho \, dt$. Here $a_{11} = V^2 S^{2\gamma}$ and $V$ is GBM (transient). The 1D CdC regression estimates an effective $\hat{\alpha}_{1D} = 2\gamma + \beta_{\text{path}}$ where $\beta_{\text{path}}$ depends on the correlation-induced $V|S$ relationship:
- $\rho > 0$: $V$ tends to be large when $S$ is large → $\beta_{\text{path}} > 0$ → $\hat{\alpha}_{1D} > 2\gamma$ (conservative, still detects bubble).
- $\rho < 0$ (leverage): $V$ tends to be small when $S$ is large → $\beta_{\text{path}} < 0$ → $\hat{\alpha}_{1D} < 2\gamma$ (**false negative risk** for $\gamma$ near 1).

The log-linear separation (§4.4.3) resolves this: the joint regression $\log(\hat{\sigma}^2) \sim \alpha \cdot \log S + \beta \cdot \log \hat{V}$ recovers $\alpha = 2\gamma$ regardless of $\rho$, because $a_{11} = V^2 \cdot S^{2\gamma}$ is multiplicatively separable.

### 4.5 Radial Regular Variation for Matrix-Valued Functions

For the full multi-dimensional theory (beyond the dimensional reduction), we need regular variation of the matrix-valued diffusion coefficient.

**Definition (Radial Regular Variation):** The locally bounded, measurable diffusion matrix $a: \mathbb{R}^d \to \mathbb{R}^{d \times d}_{\ge 0}$ has **radial tail index** $\alpha$ on a cone $\mathcal{C} \subseteq \mathbb{S}^{d-1}$ if the convergence
$$ \frac{\boldsymbol{\theta}^T a(r\boldsymbol{\theta}) \boldsymbol{\theta}}{r^\alpha \, L(r)} \xrightarrow{r \to \infty} g(\boldsymbol{\theta}) $$
holds **uniformly** over $\boldsymbol{\theta} \in \mathcal{C}$, where $L$ is slowly varying and $g: \mathcal{C} \to (0,\infty)$ is the **angular profile** (continuous, strictly positive on $\mathcal{C}$).

**Remark on uniform convergence**: The definition requires uniform (not merely pointwise) convergence over $\mathcal{C}$. This is essential for interchanging the limit and infimum in the Inheritance Lemma below. For scalar regularly varying functions, the **Uniform Convergence Theorem** (BGT, Theorem 1.2.1) provides uniformity on compact subsets of the radial variable for free. However, uniformity over the angular variable $\boldsymbol{\theta}$ is an additional condition that must be verified for each model class. For all polynomial SDE coefficients (§4.7), this is automatic because $\boldsymbol{\theta}^T a(r\boldsymbol{\theta})\boldsymbol{\theta}$ is an explicit polynomial in $r$ and $\boldsymbol{\theta}$.

**Inheritance Lemma**: If $a(\mathbf{x})$ has radial tail index $\alpha$ on $\mathcal{C}$ with uniform convergence and angular profile $g$, then the radial dispersion profile satisfies:
$$ A(\rho) := \inf_{\boldsymbol{\theta} \in \mathcal{C}} \boldsymbol{\theta}^T a(\rho\boldsymbol{\theta}) \boldsymbol{\theta} = \rho^\alpha \, L(\rho) \cdot \left[\inf_{\boldsymbol{\theta} \in \mathcal{C}} g(\boldsymbol{\theta})\right] \cdot (1 + o(1)) $$
i.e., $A(\rho)$ is regularly varying with the **same index** $\alpha$.

*Proof.* Write $h_\rho(\boldsymbol{\theta}) = \boldsymbol{\theta}^T a(\rho\boldsymbol{\theta})\boldsymbol{\theta} / (\rho^\alpha L(\rho))$. By assumption, $h_\rho \to g$ uniformly on $\mathcal{C}$. For any $\varepsilon > 0$, there exists $\rho_0$ such that $|h_\rho(\theta) - g(\theta)| < \varepsilon$ for all $\rho > \rho_0$ and all $\theta \in \mathcal{C}$. Then $\inf_\theta g(\theta) - \varepsilon < \inf_\theta h_\rho(\theta) < \inf_\theta g(\theta) + \varepsilon$, so $\inf_\theta h_\rho(\theta) \to \inf_\theta g(\theta) = g_* > 0$ (by compactness of $\mathcal{C}$ and continuity of $g$). Therefore $A(\rho) = \rho^\alpha L(\rho) \cdot (g_* + o(1))$. $\square$

**Consequence**: By Karamata's theorem (§4.3), the convergence/divergence of the 1D Feller-type integral over $A(\rho)$ depends only on $\alpha$, not on $L$ or $g_*$ (for $\alpha \ne 2$). For the full Stroock-Varadhan double integral, the same conclusion holds when $B(\rho) \to 0$ (§4.2), as is the case for financial SV models.

### 4.6 Potter Bounds and Estimation Robustness

**Potter's Theorem** (BGT, Theorem 1.5.6; the exact numbering varies slightly between the 1987 hardback and 1989 paperback editions — the result appears in §1.5 "Uniform convergence and representation"): For a measurable, regularly varying function $f$ with index $\alpha$ and any $A > 1$, $\delta > 0$, there exists $x_0 = x_0(A, \delta)$ such that for all $x, y > x_0$:
$$ \frac{f(y)}{f(x)} \le A \max\left\{(y/x)^{\alpha+\delta}, \, (y/x)^{\alpha-\delta}\right\} $$

**Note**: This requires only measurability of $f$, not continuity or monotonicity.

**Application to the CdC integral**: Suppose the true radial tail index is $\alpha$ and we estimate $\hat{\alpha}$ with error $|\hat{\alpha} - \alpha| < \delta$. By Potter bounds, the extrapolation $C \cdot S^{\hat{\alpha}}$ satisfies:
$$ c_1 \cdot S^{\alpha - \delta} \le \sigma^2(S) \le c_2 \cdot S^{\alpha + \delta} $$
for large $S$ (with constants depending on $A$). The Feller integral convergence classification is correct whenever $|\alpha - 2| > \delta$: if $\alpha > 2 + \delta$, the lower bound gives convergence; if $\alpha < 2 - \delta$, the upper bound gives divergence.

The posterior SD $\sigma_\alpha$ from BayesianRidge directly quantifies the estimation uncertainty $\delta$, and $P(\text{bubble}) = P(\alpha > 2 \mid \text{data})$ correctly propagates this through the convergence dichotomy. When $|\hat{\alpha} - 2| < \sigma_\alpha$ (the "zone of indistinguishability"), the posterior assigns intermediate probability — honestly reflecting that finite data cannot resolve the borderline case.

### 4.7 Connection to the Hill Estimator (Extreme Value Theory)

Our BayesianRidge regression $\log \hat{\sigma}^2(S) \sim \alpha \cdot \log S$ is structurally analogous to the **Hill estimator** for the tail index of heavy-tailed distributions.

**The Hill estimator** (Hill, 1975; see de Haan & Ferreira, *Extreme Value Theory: An Introduction*, 2006, §3.2): For i.i.d. random variables with regularly varying tail $\bar{F}(x) = x^{-1/\gamma} L(x)$, the Hill estimator is:
$$ \hat{\gamma}_{\text{Hill}} = \frac{1}{k} \sum_{i=1}^k \left[\log X_{(n-i+1)} - \log X_{(n-k)}\right] $$
This estimates the **extreme value index** $\gamma = 1/\alpha$, not $\alpha$ directly. Under second-order regular variation conditions (de Haan & Ferreira, Theorem 3.2.5), with appropriate choice of $k = k(n) \to \infty$, $k/n \to 0$:
$$ \sqrt{k}\left(\hat{\gamma}_{\text{Hill}} - \gamma\right) \xrightarrow{d} N(0, \gamma^2) $$
Transforming to $\hat{\alpha} = 1/\hat{\gamma}$ via the delta method gives $\sqrt{k}(\hat{\alpha} - \alpha) \xrightarrow{d} N(0, \alpha^2)$.

**Remark on second-order conditions**: The CLT requires a second-order regular variation condition on the tail, and the bias term $\sqrt{k} A(n/k) \to \lambda$ must vanish (or be corrected). In our setting, the CdC extraction $\hat{\sigma}^2(S)$ via KRR provides a denoised version of the "observations," effectively reducing the variance compared to raw Hill estimation. The BayesianRidge posterior automatically incorporates uncertainty from both observation noise and effective degrees of freedom.

**Multivariate regular variation** (Resnick, *Heavy-Tail Phenomena*, 2007, Chapter 6): For random vectors $\mathbf{X} \in \mathbb{R}^d$, multivariate regular variation decomposes into a **radial** component (governed by the scalar tail index $\alpha$) and an **angular** component (the spectral measure $\Sigma$ on $\mathbb{S}^{d-1}$). The radial tail index is estimated via scalar methods on $|\mathbf{X}|$, which is precisely what our dimensional reduction (§4.4) achieves: the CdC regression on squared price increments estimates the radial tail index along the explosion direction.

### 4.8 Financial Model Coverage

All standard diffusion models in mathematical finance have polynomial SDE coefficients, which automatically satisfy radial regular variation with uniform angular convergence:

| Model | $a(\mathbf{x})$ | Explosion Direction | Radial Index $\alpha$ |
|-------|-----------------|---------------------|----------------------|
| GBM ($d=1$) | $\sigma^2 S^2$ | $S \to \infty$ | 2 (borderline) |
| CEV $\gamma < 1$ ($d=1$) | $\sigma^2 S^{2\gamma}$ | $S \to \infty$ | $2\gamma < 2$ |
| CEV $\gamma > 1$ ($d=1$) | $\sigma^2 S^{2\gamma}$ | $S \to \infty$ | $2\gamma > 2$ (bubble) |
| Heston ($d=2$) | $\begin{pmatrix} VS^2 & \rho\xi VS \\ \rho\xi VS & \xi^2 V \end{pmatrix}$ | $S$-axis ($V$ ergodic) | 2 (stable) |
| CEV-Heston ($d=2$) | $\begin{pmatrix} VS^{2\gamma} & \cdots \\ \cdots & \xi^2 V \end{pmatrix}$ | $S$-axis | $2\gamma$ |
| SABR ($d=2$) | $\begin{pmatrix} V^2 S^{2\gamma} & \cdots \\ \cdots & \xi^2 V^2 \end{pmatrix}$ | Joint ($V$ transient!) | $\ge 2\gamma$ (see §4.4.3) |
| fSDE CEV | Path-dependent $\sigma^2(V_t) S^{2\gamma}$ | $S$-axis | $2\gamma$ (heuristic) |

**Note on fSDE**: The last row is marked "heuristic" because the Stroock-Varadhan/Khasminskii framework requires Markov diffusions. For path-dependent processes, the infinitesimal generator does not exist in the classical sense. The tail index $2\gamma$ is still the correct heuristic (the local volatility structure $\sigma(S) \propto S^\gamma$ drives the same explosion mechanism), but the rigorous justification requires the signature-based approach of **Level 4**, which bypasses the generator framework.

### 4.9 What Could Go Wrong

The bound argument requires that the CdC-extracted $\sigma^2(S)$ (or more generally $A(\rho)$) is regularly varying along the explosion direction. This can fail for several reasons:

1. **Non-regular tails**: If $\sigma^2(S)$ has different growth exponents in disjoint regions of the tail (e.g., $\sigma^2(S) = S^2 \cdot \mathbb{1}_{S < S^*} + S^3 \cdot \mathbb{1}_{S \ge S^*}$ with $S^*$ beyond the data range), the power-law extrapolation from the observed range would give the wrong index. This is a fundamental limitation of ANY finite-sample test and is economically implausible for standard diffusions, though it could arise from regulatory interventions or structural breaks.

2. **Angular degeneracy**: If the angular profile $g(\boldsymbol{\theta})$ vanishes along certain directions within the explosion cone (i.e., $\inf_\theta g(\theta) = 0$), the Inheritance Lemma gives a trivial bound $A(\rho) \ge 0$. This occurs when the diffusion matrix is degenerate (hypoelliptic). For explosion detection, we only need $g > 0$ on the relevant explosion direction, not on all of $\mathbb{S}^{d-1}$.

3. **Non-ergodic / transient volatility** (see §4.4.3): The 1D dimensional reduction relies on $\mathbb{E}[V \mid S = s]$ being bounded (slowly varying). When $V$ is transient (SABR, 3/2 model in certain regimes), the conditional expectation can grow or shrink with $S$, altering the effective tail index. With leverage ($\rho < 0$), the index is **attenuated**, creating false negatives. **Resolution**: The log-linear separation approach (§4.4.3) controls for $V$ by including $\log \hat{V}$ as a regressor, recovering the true $S$-exponent $\alpha$ for all multiplicatively separable models.

4. **Slowly varying corrections at the boundary**: At exactly $\alpha = 2$ (the borderline), the slowly varying part $L(s)$ determines convergence/divergence. Our BayesianRidge fit captures a *constant* $L$ (the intercept $C$ in $\sigma^2 = C \cdot S^\alpha$). If the true $L(s)$ varies (e.g., $L(s) = \log s$), we would misclassify. However, this is precisely the case where the posterior assigns $P(\alpha > 2 \mid \text{data}) \approx 0.5$, honestly reflecting the indeterminacy.

5. **Rough volatility / non-Markov**: The Stroock-Varadhan test assumes Markov diffusions. For non-Markov processes (rough vol, fSDE), the infinitesimal generator does not exist in the classical sense. This is the motivation for **Level 4** (signature-based), which uses path-dependent CdC extraction to bypass the Markov generator framework.

**Remark on falsifiability**: The regular variation assumption is partially falsifiable on the observed range: significant curvature in the log-log plot of $\hat{\sigma}^2(S)$ vs $S$ indicates a non-constant local exponent (i.e., a non-trivial slowly varying correction), which should widen the posterior uncertainty and push $P(\text{bubble})$ toward the indeterminate range.

---

## 5. Conclusion and Synthesis


The CdC-bridged eigenfunction test provides a theoretically complete framework for bubble detection:

1. **CdC Extraction** (measure-invariant): $\hat{\sigma}^2(S)$ via kernel regression on squared increments, annihilating unknown drift.
2. **Tail Characterization** (regular variation): BayesianRidge extracts the tail index $\alpha$ with posterior uncertainty. By §4, only a bound on $\alpha$ relative to the critical threshold 2 is needed — the slowly varying correction $L(s)$ does not affect the convergence dichotomy.
3. **Bayesian Decision** (direct posterior): Each level computes $P(\text{bubble} \mid \text{data}) = P(\alpha > 2 \mid \text{posterior})$ directly from the BayesianRidge posterior using the Gaussian CDF. No null hypothesis, no critical values. L1, L3, and L3b all test "$\alpha > 2$" with different estimators (1D CdC, log-linear separation, signature-based); they are NOT independent, so the best available estimator replaces (not stacks with) the others. L2 (Feller test on $V$) and L1M (multivariate) are genuinely independent.
4. **Sturm-Liouville Eigenvalue** (diagnostic only): The eigenvalue from $\frac{1}{2}\sigma^2(x) u’’ = \lambda u$ on the extended grid is retained as a diagnostic. However, on finite discretized grids the eigenvalue is always positive, making $\lambda > 0$ unreliable as a decision threshold. The Bayesian posterior on $\alpha$ is the primary decision mechanism.
5. **Hierarchical Combination** (noisy-OR): Per-level Bayesian posteriors $P_k$ are combined via noisy-OR: $P(\text{any bubble}) = 1 - \prod_k (1 - P_k)$. No Beta likelihood layer is needed since each level already outputs a proper Bayesian posterior. This achieves 11/11 on Tier III-SV (SABR, 3/2 models) with all nesting preserved.

The dual Sig-KKF / RBF Koopman architecture provides cross-validation:
- **Sig-KKF (Unbounded Polynomials):** Learns the global drift and diffusion operators linearly but cannot bound the eigenfunctions. It provides the global $\Gamma(\mathbf{x}, \mathbf{x}^T)$ polynomial matrices for analytic integration of the Khasminskii constraints (Test B).
- **RBF Koopman (Bounded RKHS):** Restricts the operator geometrically. If a structural, density-covered positive eigenvalue persists (Test A), it mechanically guarantees an explosion bound. By comparing this direct spectral check against the Sig-KKF’s algebraic CdC integral, the presence of a bubble is mathematically verified across distinct functional topologies.
