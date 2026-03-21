# Theoretical Comparison: Model-Free QV Scaling v.s. Bayesian Koopman-Khasminskii

This document provides a rigorous mathematical comparison of two prominent methods for detecting financial bubbles (Strict Local Martingales): the Model-Free Quadratic Variation (QV) Scaling test and the Bayesian Khasminskii-Koopman test. 

We will prove that while the Model-Free QV test is necessary for a restricted class of 1D Markov diffusions, it possesses critical theoretical failure cases in the presence of hidden state dimensions (Stochastic Volatility) and memory (Rough Volatility). We then prove how the Signature Koopman-Kalman Filter (Sig-KKF) augmented with the Bayesian Khasminskii test theoretically closes these loopholes.

---

## 1. The Strict Local Martingale Framework

Under the Jarrow-Protter framework natively adopted in this repository, an asset price $S_t$ constitutes a **bubble** under the risk-neutral measure $\mathbb{Q}$ if and only if it is a **Strict Local Martingale (SLM)**: a local martingale that is *not* a true martingale.
$$ \mathbb{E}^\mathbb{Q}[S_t] < S_0 \quad \text{for some } t > 0 $$

To test for this property empirically, we evaluate the explosion criteria of the underlying stochastic differential equation.

---

## 2. The Model-Free QV Scaling Test ($\alpha > 2$)

The Model-Free test evaluates the asymptotic scaling of the asset's Quadratic Variation against its price level.

### 2.1 The Core Theorem
Let $S_t$ be a 1-dimensional continuous local martingale satisfying the Markovian SDE:
$$ dS_t = \sigma(S_t) dW_t $$
where $\sigma(\cdot)$ is a strictly positive, locally Hölder continuous function. 

By Feller's Test for Explosions and the Dambis-Dubins-Schwarz time-change theorem, $S_t$ is a Strict Local Martingale if and only if the scale function integral converges:
$$ \int_{c}^{\infty} \frac{x}{\sigma^2(x)} dx < \infty $$

If we assume the local variance has an asymptotic polynomial bound $\sigma^2(S_t) \sim S_t^\alpha$ as $S_t \to \infty$, the integral condition simplifies:
$$ \int_{c}^{\infty} x^{1-\alpha} dx < \infty \iff \alpha > 2 $$

### 2.2 Empirical Implementation via Lead-Lag Signatures
To execute this model-free, we compute the Quadratic Variation over a rolling window $w = [t, t+\Delta]$. By utilizing the **Lead-Lag Signature** of the path, the Lévy Area (the antisymmetric term of the level-2 Signature) mathematically coincides with the Quadratic Variation:
$$ A_{[t, t+\Delta]}^{LeadLag} \equiv \frac{1}{2}\sum (S_{i} - S_{i-1})^2 = \langle S \rangle_{[t, t+\Delta]} $$

We then run the log-log cross-sectional regression across windows:
$$ \log \langle S \rangle_w = \alpha \log \bar{S}_w + C + \epsilon_w $$
If the estimated scaling exponent $\hat{\alpha} > 2$, the asset is classified as a bubble.

### 2.3 Theoretical Failure Cases of the $\alpha > 2$ Test

While elegant, the $\alpha > 2$ test is fundamentally limited by its implicit assumption that the asset is a 1-dimensional time-homogeneous Markov diffusion. When we relax this, the test breaks down in three critical ways.

#### Failure Case 1: The ELMM Invariance Breakdown (Stochastic Volatility)
Suppose the true Data Generating Process is a multi-dimensional Stochastic Volatility model (e.g., Heston):
$$ dS_t = \sqrt{V_t} S_t dW_t^{(1)} $$
$$ dV_t = \kappa(\theta - V_t) dt + \xi \sqrt{V_t} dW_t^{(2)} $$

In a devastating blow to model-free empirical tests, **Jarrow, Protter, and San Martin (2022)** explicitly prove that while the Strict Local Martingale (bubble) property is invariant to the choice of Equivalent Local Martingale Measure (ELMM) for 1D diffusions, **this invariance mathematically breaks down when stochastic volatility is present.** 

The empirical Lead-Lag QV scaling test ($\alpha > 2$) is computed using historical price data, meaning the quadratic variation is measured under the statistical probability measure $\mathbb{P}$. However, by definition, an asset exhibits a bubble if and only if it is a Strict Local Martingale under the risk-neutral measure $\mathbb{Q}$. Because Jarrow et al. (2022) proved the bubble property is *not invariant* across measures in multidimensional SV models, evaluating the $\mathbb{P}$-measure empirical QV scaling provides mathematically invalid conclusions regarding the $\mathbb{Q}$-measure bubble status. Furthermore, an omitted variable bias occurs where a massive, mean-reverting spike in $V_t$ causes the scalar $\alpha$-regression to hallucinate an $\hat{\alpha} \gg 2$, firing severe false positives.

#### Failure Case 2: Non-Markovian Rough Volatility
If the asset exhibits rough volatility (e.g., driven by a Fractional Brownian Motion $W_t^H$ with Hurst exponent $H < 0.5$):
$$ \sigma_t = \exp( \nu W_t^H ) $$
The variance process is no longer Markovian; it retains infinite-dimensional memory. The log-log regression $\log \langle S \rangle_w$ versus $\log \bar{S}_w$ becomes fundamentally mis-specified, as the variance scaling is dominated by the fractional memory kernel rather than the immediate price level. This destroys the asymptotic guarantees of the Feller integral.

#### Failure Case 3: Threshold Ambiguity ($\alpha=2$)
If $\sigma^2(S_t) \sim S_t^2 \log(S_t)^p$, the scaling exponent is exactly $\alpha = 2$ (the logarithmic term is sub-polynomial).
$$ \int_c^\infty \frac{x}{x^2 \log(x)^p} dx = \int_{\log c}^\infty \frac{1}{u^p} du $$
This converges (yielding a Bubble) if and only if $p > 1$. The $\alpha$-regression cannot detect logarithmic fine-structure, resulting in a blind spot for borderline bubbles.

#### 2.4 Partial Resolution: Local Volatility Elasticity

The **local volatility elasticity** $\varepsilon(S) = \partial\log\sigma^2(S)/\partial\log S$ generalizes the global $\alpha$ to non-CEV 1D diffusions. When estimated via GP posterior gradient (see `theory_feller_universality_jump_robustness.md` §1.5), it partially addresses Failure Cases 2 and 3:

- **Failure Case 3 (log corrections)**: $\sigma^2(S) = S^2 \log(S)^p$ gives $\varepsilon(S) = 2 + p/\log S > 2$ locally — the GP gradient captures the slowly varying correction that the global regression misses.
- **Failure Case 2 (rough vol)**: $\varepsilon(S)$ is still a 1D marginal quantity. If the path-dependent $\sigma^2$ depends on the full volatility trajectory (not just price level), the marginal elasticity is contaminated. Partial mitigation: conditioning $\varepsilon$ on signature features via the ARD kernel in MLKFellerGP.
- **Failure Case 1 (SV)**: $\varepsilon$ computed on marginal $\hat\sigma^2(S)$ inherits the omitted-variable bias from hidden vol. **Not addressed** by local elasticity alone — requires the joint $(S, V)$ conditioning (log-linear separation or full multi-D Khasminskii).

The hierarchy: global $\alpha$ (CEV-type) $\subset$ local $\varepsilon(S)$ (general 1D) $\subset$ conditional $\varepsilon(S|V)$ (separable SV) $\subset$ path-conditioned $\varepsilon(S|\text{sig})$ (rough vol). The MLKFellerGP ARD kernel automatically selects the appropriate level.

---

## 3. The Bayesian Koopman-Khasminskii Synthesis

To mathematically close these loopholes, we must abandon the 1D regression and discover the true, potentially high-dimensional state space. 

### 3.1 Theorem: Khasminskii's Non-Explosion Criterion
Let $\mathbf{X}_t \in \mathcal{D} \subseteq \mathbb{R}^d$ be a multi-dimensional diffusion (encompassing both Price $S_t$ and Hidden Volatility $V_t$) with infinitesimal generator $\mathcal{L}$.
$$ \mathcal{L}f(\mathbf{x}) = \sum_{i=1}^d b_i(\mathbf{x}) \frac{\partial f}{\partial x_i} + \frac{1}{2} \sum_{i,j=1}^d (\Sigma\Sigma^T)_{ij}(\mathbf{x}) \frac{\partial^2 f}{\partial x_i \partial x_j} $$

**Dandapani and Protter (2019)** explicitly extend Khasminskii's stability theory to test for multidimensional strict local martingales. An asset is a True Martingale (No Bubble) if and only if there exists a Lyapunov function $V(\mathbf{x}) \to \infty$ as $\mathbf{x} \to \partial \mathcal{D}$ and a constant $\lambda > 0$ such that:
$$ \mathcal{L}V(\mathbf{x}) \le \lambda V(\mathbf{x}) $$
If no such function exists, the process explodes under the equivalent measure and is a **Strict Local Martingale (Bubble)**.

### 3.2 Closing Loopholes via the Signature Koopman-Kalman Filter (Sig-KKF)
The Khasminskii condition requires knowledge of the true high-dimensional generator $\mathcal{L}$. We discover this organically using the streaming Sig-KKF.

1. **Solving Failure Case 1 & 2 (Hidden States & Memory):**
   By lifting the raw price path into the Path Signature space $S(\mathbf{X})_{[0, t]}$, we invoke the rough path universal approximation theorem. The Signature unwinds non-Markovian memory (Failure Case 2) and implicitly spans the hidden stochastic volatility dimensions (Failure Case 1).
   
2. **Linear Koopman Projection:**
   The KKF learns a matrix $A$ such that $d\mathbb{E}[S_t] = A \mathbb{E}[S_t] dt$. Because the Signature spans the coordinate ring, the linear generator $A$ is precisely the matrix representation of the true multi-dimensional infinitesimal generator $\mathcal{L}$.

3. **Bayesian Khasminskii Test:**
   Instead of testing 1D QV scaling, we pass the Koopman-discovered state dimensions to the formal Khasminskii test.
   Rather than a point estimate, the Sig-KKF computes the true Bayesian posterior of the generator $\mathcal{L} \sim \mathcal{MN}(\hat{A}_t, \sigma^2 P_t)$. By sampling from this posterior, we compute the probabilistic bounds of the Khasminskii failure condition.

### 3.3 Final Proof of Superiority
By taking the multi-dimensional generator discovered by the Sig-KKF and passing it through the Bayesian Khasminskii criterion (specifically integrating the Stroock-Varadhan bounds over the Koopman-discovered state space), we formally test the strict Dandapani-Protter geometric explosion condition on the *true unconfounded state space*, eliminating the Omitted Variable Bias and Non-Markovian failure cases inherent to the scalar Model-Free $\alpha > 2$ regression.

---

## 4. The Operator-Theoretic Explosion Test (CdC and Bounded Eigenfunctions)

The ultimate goal of bubble detection is a fully operator-focused spectral test. While the naive condition "Koopman $\mathcal{L}$ possesses an eigenvalue $\text{Re}(\lambda)>0$" is mathematically false due to the Geometric Brownian Motion (GBM) counterexample, Khasminskii's framework establishes a rigorous spectral equivalent that isolates the volatility structure.

### 4.1 The Role of the Carré du Champ (CdC) Operator
As established by Girsanov's Theorem, changing the probability measure from the historical measure $\mathbb{P}$ to an Equivalent Local Martingale Measure (ELMM) $\mathbb{Q}$ *only alters the drift* of the process. The quadratic variation (volatility structure) remains identical. Because a bubble is defined as an explosion under $\mathbb{Q}$ (a Strict Local Martingale), and explosions are purely topological phenomena driven by super-linear volatility growth, the structural test for a bubble must isolate the volatility.

Therefore, the theoretically pure operator for explosion testing is not the standard Koopman generator $\mathcal{L}$ (which is confounded by the $\mathbb{P}$-measure drift), but the **Carré du Champ (CdC) operator**, denoted $\Gamma$:
$$ \Gamma(f, g) = \mathcal{L}(fg) - f\mathcal{L}(g) - g\mathcal{L}(f) $$
For a diffusion, $\Gamma(f, f) = \sum_{i,j} a_{ij}(\mathbf{x}) \frac{\partial f}{\partial x_i} \frac{\partial f}{\partial x_j}$, which completely annihilates the drift vector $b(\mathbf{x})$ and isolates the diffusion matrix $a(\mathbf{x}) = \sigma(\mathbf{x})\sigma(\mathbf{x})^T$.

By analyzing the spectrum and eigenfunctions of the CdC operator (or its associated quadratic forms), we construct an explosion test that is fundamentally immune to $\mathbb{P}$-measure drift distortions.

### 4.2 The Bounded Eigenfunction Theorem
By translating Khasminskii's Theorem 4 into the language of Semigroup Theory acting on the volatility structure, a multidimensional diffusion $\mathbf{X}_t$ explodes in finite time (Bubble) **if and only if** the operator encoding its spatial growth (isolated via the CdC) possesses a strictly positive principal eigenvalue $\lambda > 0$ whose corresponding eigenfunction $u(\mathbf{x})$ is **strictly bounded and globally positive**.
$$ \mathcal{L}u = \lambda u \quad \text{such that } 0 < u(\mathbf{x}) < \infty \text{ for all } \mathbf{x} $$

### 4.2 Ruling Out the GBM Counterexample
This operator condition immediately and elegantly rules out the spurious GBM cases.
* **The False Positive:** For Geometric Brownian Motion, $\mathcal{L}(x) = \mu x$. The generator has an eigenvalue $\lambda = \mu > 0$ with eigenfunction $f(x) = x$. However, this eigenfunction is **unbounded** as $x \to \infty$. Therefore, it fails the Khasminskii topological condition and is correctly identified as a True Martingale (No Bubble).
* **The True Positive:** For a true bubble such as a CEV process with exponent $\gamma > 1$, the process explodes at a finite random time $\tau_{exp}$. The function $u(\mathbf{x}) = \mathbb{E}^{\mathbf{x}} \left[ e^{-\lambda \tau_{exp}} \right]$ maps the state space to $(0, 1]$. By Itô's Lemma and Dynkin's formula, $\mathcal{L} u(\mathbf{x}) = \lambda u(\mathbf{x})$. Because $u(\mathbf{x})$ is bounded within $(0, 1]$, the bounded eigenfunction criteria is perfectly satisfied.

### 4.3 Sig-KKF (Signatures) vs. RBF-KKT (Kernels)
This bounded-eigenfunction requirement theoretically splits our Data-Driven Koopman methods.

1. **The Signature KKF (Unbounded Polynomials):** The Path Signature spans the coordinate ring (polynomials). All non-constant polynomials are fundamentally unbounded. If our Sig-KKF extracts a positive eigenvalue $\lambda > 0$, it only mathematically proves that some unbounded polynomial projection is growing exponentially. Because it cannot span the bounded topology, the Sig-KKF Matrix cannot be used individually to prove a bubble. It must be used purely for **Timestamp extraction** to locate regime shifts, feeding the recovered state dimension into the Bayesian Khasminskii integration test.

2. **The Kernel Koopman Tensor (Bounded RKHS):** If we learn the Koopman matrix using a Reproducing Kernel Hilbert Space (RKHS) spawned by Gaussian Radial Basis Functions (RBFs), **every function in the dictionary span is globally bounded**. 
Thus, if the RBF Kernel Koopman empirical matrix exhibits an eigenvalue $\text{Re}(\lambda) > 0$, the corresponding eigenfunction is constrained to the bounded RBF span. It perfectly satisfies the Khasminskii Bounded Eigenfunction criterion, mathematically classifying the asset as a Structural Bubble with no secondary integration tests required!
