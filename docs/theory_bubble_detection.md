# Unified Theory of Operator-Theoretic Bubble Detection

---

## Table of Contents

1. [Strict Local Martingales and Financial Bubbles](#1-strict-local-martingales-and-financial-bubbles)
2. [The α > 2 Test: Regular Variation and Local Elasticity](#2-the-α--2-test-regular-variation-and-local-elasticity)
3. [Measure Invariance and the Carré du Champ](#3-measure-invariance-and-the-carré-du-champ)
4. [Multi-Dimensional Theory: Khasminskii and Bounded Eigenfunctions](#4-multi-dimensional-theory-khasminskii-and-bounded-eigenfunctions)
5. [Stochastic Volatility: When 1D Works and Fails](#5-stochastic-volatility-when-1d-works-and-fails)
6. [Jump Robustness and Universality](#6-jump-robustness-and-universality)
7. [Why Eigenvalue Approaches Fail](#7-why-eigenvalue-approaches-fail)
8. [Tiered Detection Architecture](#8-tiered-detection-architecture)
9. [The GP Pipeline](#9-the-gp-pipeline)
10. [Experimental Results and Dead Ends](#10-experimental-results-and-dead-ends)
11. [References](#11-references)

---

## 1. Strict Local Martingales and Financial Bubbles

### 1.1 The Jarrow-Protter-Shimbo Framework

Under the Jarrow-Protter-Shimbo (2010) framework, an asset price $S_t$ constitutes a **bubble** under the risk-neutral measure $\mathbb{Q}$ if and only if the discounted price is a **strict local martingale (SLM)**: a local martingale that is not a true martingale.

$$\mathbb{E}^{\mathbb{Q}}[S_T] < S_0 \quad \text{for some } T > 0$$

For a 1D continuous local martingale $dS_t = \sigma(S_t)\,dW_t^{\mathbb{Q}}$ on $(0,\infty)$, the fundamental characterization is:

**Theorem (Delbaen-Shirakawa 2002, Mijatović-Urusov 2012).** The process $S_t$ is a strict local martingale if and only if the Feller scale function integral converges:

$$\int_c^\infty \frac{x}{\sigma^2(x)}\,dx < \infty \quad \text{for some (equivalently, any) } c > 0 \tag{F}$$

This criterion depends only on $\sigma^2(x)$, not on the drift. The bubble property is therefore determined by the local volatility function alone.

**Proof sketch.** The scale function of $dS = \sigma(S)\,dW$ is $s(x) = x$ (no drift). The boundary $x = \infty$ is accessible (exit) if and only if $\int_c^\infty s'(x)\,m(dx) < \infty$ where $m(dx) = 2\,dx/(\sigma^2(x)\,s'(x))$ is the speed measure. With $s'(x) = 1$, this reduces to $2\int_c^\infty dx/\sigma^2(x)$. The weighted version $\int x/\sigma^2(x)\,dx$ arises from the Engelbert-Schmidt characterization via the function $v(x) = \int_c^x y/\sigma^2(y)\,dy$: the process explodes iff $v(\infty) < \infty$.

### 1.2 Type I vs. Type II Bubbles

Consider a stochastic volatility model under $\mathbb{Q}$:
$$dX_t = r X_t\,dt + \sigma(X_t, Y_t) X_t\,dW_t^1, \quad dY_t = \mu_Y(Y_t)\,dt + \sigma_Y(Y_t)\,d\tilde{W}_t$$
where $d\langle W^1, \tilde{W} \rangle_t = \rho\,dt$.

Let $s_Y(\infty) = \int^\infty \exp\!\left(-\int^v \frac{2\mu_Y(u)}{\sigma_Y^2(u)}\,du\right)dv$ be the Feller scale function for $Y$.

**Definition (Type I / Type II).** A bubble is:

- **Type I (Pathwise):** $s_Y(\infty) < \infty$. The vol process $Y$ reaches $+\infty$ in finite time with positive probability. The bubble manifests on observable paths.

- **Type II (Measure-Theoretic):** $s_Y(\infty) = \infty$ but $X$ is still a strict local martingale. $Y \to 0$ or $Y \to Y_\infty < \infty$ a.s., but moments $\mathbb{E}[Y_t^k]$ diverge for some $k$. The bubble exists only in expectation.

**Example (JPS).** For $dY = KY^2\,dt + Y\,d\tilde{W}$, the bubble condition is $2K + \rho^2 \geq 0$:

| $K$ | $s_Y(\infty)$ | $Y$ behavior | Type |
|-----|--------------|-------------|------|
| $K > 0$ | $< \infty$ | Explodes a.s. | I |
| $K = 0$ | $= \infty$ | $\to 0$ a.s. | II (iff $\rho \neq 0$) |
| $-\rho^2/2 \leq K < 0$ | $= \infty$ | $\to 0$ a.s. | II |
| $K < -\rho^2/2$ | $= \infty$ | $\to 0$ a.s. | No bubble |

### 1.3 Single-Path Detectability

**Theorem 1 (Single-Path Detectability).**

**(i) Type I bubbles are detectable.** If $s_Y(\infty) < \infty$, there exists a consistent test $\phi_T$ based on a single path such that under $H_0$ (no bubble) $\mathbb{E}[\phi_T] \leq \alpha$ for all $T$, and under $H_1$ (Type I) $\mathbb{E}[\phi_T] \to 1$ as $T \to \infty$.

The test: estimate $\hat\sigma^2_Y(y)$ and $\hat\mu_Y(y)$ via Nadaraya-Watson, compute $\hat g(y) = 2\hat\mu_Y/\hat\sigma^2_Y$, and test $\hat s_Y(\infty) < \infty$ via the ScaleFunctionGP posterior.

**(ii) Type II bubbles are undetectable from a single path.** If $s_Y(\infty) = \infty$ (Type II), then for any single-path test $\phi_T$:

$$\limsup_{T \to \infty} \mathbb{E}_{H_1}[\phi_T] \leq \limsup_{T \to \infty} \mathbb{E}_{H_0}[\phi_T]$$

No single-path test can distinguish a Type II bubble from a non-bubble.

**Proof sketch of (ii).** Under Type II, $Y_t \to 0$ a.s. The path-wise law of $(X, Y)$ restricted to any compact $[0, T]$ is absolutely continuous with respect to the law under a non-bubble parameterization. Both produce $Y \to 0$ at the same rate — the distinction is in the tail of $Y$'s distribution across paths, not in the typical path. The Radon-Nikodym derivative $d\mathbb{P}_{H_1}/d\mathbb{P}_{H_0}|_{\mathcal{F}_T}$ concentrates around 1 for typical paths, making the hypotheses mutually contiguous. By Le Cam's third lemma, no test can separate them. $\square$

### 1.4 Portfolio Irrelevance of Type II Bubbles

**Theorem 2 (Portfolio Irrelevance).** Consider an investor with utility $U$ satisfying $U'(0+) < \infty$. Under a Type II bubble:

**(i)** For any bounded admissible strategy $\pi$, $W_T^\pi / (W_0 e^{rT}) \to 1$ a.s. as $T \to \infty$ (vol dies, risky asset becomes effectively risk-free).

**(ii)** The bubble component $B_t = X_t - \mathbb{E}_\mathbb{Q}[e^{-r(T-t)}X_T | \mathcal{F}_t]$ satisfies $B_t/X_t \to 0$ in probability.

**(iii)** CE loss decays: $\text{CE}_\text{aware} - \text{CE}_\text{unaware} = O(e^{-cT})$ for some $c > 0$.

**Implication.** Single-path detection detects exactly the portfolio-relevant class: Type I bubbles (detectable, actionable) and correctly ignores Type II (undetectable, no CE consequence). This is the best possible result for single-path detection.

For completeness: Type II bubbles ARE detectable with multiple independent paths — $\hat B_T = X_0 - N^{-1}\sum_i e^{-rT}X_T^{(i)}$ is consistent — but detection requires $N = \Omega(e^{\lambda T})$ paths (exponentially many in the horizon), which is impractical.

---

## 2. The α > 2 Test: Regular Variation and Local Elasticity

### 2.1 The Power-Law Criterion

The Feller integral (F) is determined by the tail growth of $\sigma^2(x)$. If $\sigma^2(x) \sim Cx^\alpha$ as $x \to \infty$:

$$\int_c^\infty \frac{x}{Cx^\alpha}\,dx = \frac{1}{C}\int_c^\infty x^{1-\alpha}\,dx \quad \text{converges iff} \quad \alpha > 2$$

Therefore: **the asset is a strict local martingale (bubble) if and only if $\alpha > 2$.**

This is the theoretical foundation for all detection tiers. The equivalence chain is:

$$\underbrace{\hat\sigma^2(S)}_{\text{KGEDMD}} \xrightarrow{\text{regression}} \underbrace{\hat\alpha}_{\text{tail exponent}} \xrightarrow{\hat\alpha > 2?} \underbrace{\lambda_c > 0}_{\text{Engländer-Pinsky}} \iff \underbrace{\text{explosion}}_{\text{Feller}} \iff \underbrace{\text{bubble}}_{\text{Jarrow-Protter}}$$

Each arrow is a theorem, not a heuristic. The only approximation is in the KGEDMD regression (finite sample, finite landmarks), controlled by standard kernel regression theory.

### 2.2 Regular Variation is Generic

**Definition 2.1 (Regular variation).** A measurable function $L : (0,\infty) \to (0,\infty)$ is **regularly varying at infinity with index $\rho$**, written $L \in RV_\rho$, if for all $\lambda > 0$:

$$\lim_{x \to \infty} \frac{L(\lambda x)}{L(x)} = \lambda^\rho$$

Every $L \in RV_\rho$ admits $L(x) = x^\rho \ell(x)$ where $\ell$ is slowly varying (Karamata Characterization, BGT Theorem 1.4.1).

**Definition 2.2 (Financially relevant diffusion).** A SDE $dS = \mu(S)dt + \sigma(S)dW$ is **financially relevant** if:

- (FR1) $\sigma \in C^1(0,\infty)$ with $\sigma(s) > 0$ (non-degeneracy)
- (FR2) $\sigma^2$ is **eventually monotone** (there exists $M$ such that $\sigma^2$ is monotone on $[M,\infty)$)
- (FR3) The process admits a stationary distribution on compact subsets when $\alpha \leq 2$

**Theorem 2.3 (Regular variation of financial diffusions).** Under (FR1)–(FR2), $\sigma^2 \in RV_\alpha$ for some finite $\alpha$.

*Proof.* By (FR1), $\phi(t) = \log\sigma^2(e^t)$ is $C^1$. By (FR2), $\phi$ is eventually monotone on $[\log M, \infty)$. A continuous, eventually monotone function has a limit $\alpha = \lim_{t\to\infty} \phi(t)/t \in (-\infty, \infty]$. For any $\lambda > 0$, if $\phi(t)/t \to \alpha$, then $\phi(t + \log\lambda) - \phi(t) \to \alpha\log\lambda$, so $\sigma^2(\lambda x)/\sigma^2(x) \to \lambda^\alpha$. Finiteness of $\alpha$: (FR2) excludes super-polynomial growth like $\sigma^2(x) = e^x$, which would be ruled out by Proposition 2.5 (return tails). $\square$

**Proposition 2.4 (Leverage effect implies eventual monotonicity).** If $\sigma^2(S) = g(S)$ and the leverage effect $\text{Corr}(dS_t, d\sigma^2_t) \neq 0$ is persistent, then $g$ is eventually monotone (either eventually non-decreasing for inverse leverage or eventually non-increasing for the standard equity leverage effect). Oscillating $\sigma^2$ would produce periodically inverting implied volatility smiles — never observed empirically.

**Proposition 2.5 (Power-law return tails imply finite $\alpha$).** Empirically documented Pareto return tails $\mathbb{P}(|r_t| > x) \sim Cx^{-\xi}$ with $\xi \in (2,\infty)$ (typically $\xi \approx 3$–$5$ for equities) imply $\sigma^2 \in RV_\alpha$ with finite $\alpha \leq \xi + 1 < \infty$. This rules out $\sigma^2(x) = e^x$ or other super-polynomial forms.

### 2.3 Karamata's Theorem: Only α Matters

**Karamata's Theorem (BGT, Proposition 1.5.8).** If $f \in RV_\alpha$ is locally bounded:
- For $\alpha > -1$: $\int_1^x f(t)\,dt \sim xf(x)/(\alpha+1)$ as $x \to \infty$
- For $\alpha < -1$: $\int_x^\infty f(t)\,dt \sim -xf(x)/(\alpha+1)$ as $x \to \infty$

**Application.** With $\sigma^2(x) = x^\alpha L(x)$, the Feller integrand $x/\sigma^2(x) = x^{1-\alpha}/L(x)$ is in $RV_{1-\alpha}$. By Karamata:
- $\alpha > 2 \Rightarrow 1-\alpha < -1$: integral converges (explosion, bubble)
- $\alpha < 2 \Rightarrow 1-\alpha > -1$: integral diverges (no explosion)
- $\alpha = 2$: borderline, depends on the slowly varying part $L$

**Critical consequence**: The Feller integral convergence depends only on $\alpha$, not on the slowly varying correction $L(x)$, except at the borderline $\alpha = 2$. This is why the log-log regression on $\sigma^2$ vs. $S$ works: it correctly identifies the tail index regardless of the slowly varying correction.

### 2.4 Local Volatility Elasticity

The global index $\alpha$ summarizes tail behavior. The more refined quantity is the **local volatility elasticity**:

**Definition 2.6.** For $\sigma^2 \in C^1(0,\infty)$, the local volatility elasticity at price level $S$ is:

$$\varepsilon(S) = \frac{\partial\log\sigma^2(S)}{\partial\log S} = \frac{S\cdot(\sigma^2)'(S)}{\sigma^2(S)}$$

**Proposition 2.7 (Relationship to Feller test).** The Feller integral converges iff $\liminf_{S\to\infty}\varepsilon(S) > 2$.

*Proof.* Write $\sigma^2(S) = \exp(\int_1^S \varepsilon(u)/u\,du + C)$. If $\varepsilon(S) \geq 2 + \delta$ for $S > M$, then $\int_1^x \varepsilon(u)/u\,du \geq (2+\delta)\log(x/M) + O(1)$, so the integrand decays as $x^{-(1+\delta)}$ and the integral converges. Conversely, if $\varepsilon(S) \leq 2$ on a set of positive log-measure, the integral diverges. $\square$

**Hierarchy of generality:**

| Estimator | Assumes | Handles |
|-----------|---------|---------|
| Global $\alpha$ (log-log OLS) | $\sigma^2 \in RV_\alpha$ | CEV, GBM, CIR, standard diffusions |
| Local $\varepsilon(S)$ (GP gradient) | $\sigma^2$ smooth in $S$ | Above + log corrections, non-monotone $\sigma^2$ |
| Conditional $\varepsilon(S|V)$ (MLK + vol proxy) | $\sigma^2(S,V)$ smooth, $V$ observable | Above + SABR, Heston, CEV-SV |
| Path-conditioned $\varepsilon(S|\text{sig})$ (MLK + signatures) | $\sigma^2$ depends on path | Above + rough vol |

**Failure cases addressed by $\varepsilon(S)$ but not by global $\alpha$:**
- *Logarithmic corrections*: $\sigma^2(S) = CS^2(\log S)^p$ gives global $\hat\alpha \approx 2$ but local $\varepsilon(S) = 2 + p/\log S > 2$ (for $p > 0$), correctly identifying the bubble.
- *Non-monotone $\sigma^2$*: Averaging across regimes attenuates $\alpha$; local $\varepsilon(S)$ captures the high-$S$ regime.

**Failure cases NOT addressed by $\varepsilon(S)$:**
- *SABR with leverage*: $\varepsilon$ computed on marginal $\hat\sigma^2(S)$ inherits the omitted-variable bias from hidden $V$. Requires the joint $(S,V)$ conditioning (§5).
- *Rough volatility*: Path-dependent $\sigma^2$ requires signature-conditioned estimation.

**GP gradient computation.** The GP posterior mean is $\hat f(S) = \alpha\log S + c + g(S)$ where $g$ is the nonparametric GP residual. The local elasticity is:

$$\hat\varepsilon(S) = \alpha + \frac{\partial g}{\partial\log S} = \alpha + \sum_j \frac{-(\log S - \log S_j)}{\ell^2}\,k_*(S, S_j)\cdot(C^{-1}\mathbf{r})_j$$

This is an $O(m)$ computation (dot product with $m$ landmarks) requiring no additional GP inference.

**Role separation.** For bubble detection (Feller test), $\alpha_\infty = \lim_{S\to\infty}\varepsilon(S)$ is the correct criterion. For bubble dynamics (tracking evolution, hazard rates), the local $\varepsilon(S_t, t)$ at the current price is more informative.

### 2.5 Regime-Switching Regular Variation

**Proposition 2.8.** Let $S$ follow a Markov-switching diffusion $dS = \sigma_{\theta(t)}(S)dW$ with $\theta(t) \in \{1,\ldots,K\}$ and $\sigma_k^2 \in RV_{\alpha_k}$. Then:

(a) The process is a strict local martingale iff it spends positive Lebesgue-measure time in regimes with $\alpha_k > 2$.

(b) The time-local MLKFellerGP test consistently estimates $\alpha_{\theta(T)}$ as $n \to \infty$, $\Delta t \to 0$.

### 2.6 The Generalized Principal Eigenvalue Connection

**Theorem (Engländer-Pinsky 1999; Pinsky 1995).** The generalized principal eigenvalue:

$$\lambda_c = \sup\{\lambda \in \mathbb{R} : \exists u > 0 \text{ with } Lu = \lambda u \text{ on } (0,\infty)\}$$

satisfies $\lambda_c > 0$ iff the process is explosive (reaches the boundary in finite time). Combined with the Feller test:

$$\lambda_c > 0 \iff \int_c^\infty \frac{x}{\sigma^2(x)}\,dx < \infty \iff \alpha_\infty > 2$$

The three forms of the criterion — explosion, Feller integral, $\alpha > 2$ — are mathematically equivalent. Only the $\alpha$ form is numerically accessible from discrete price data (§7 explains why direct eigenvalue computation fails).

**Large deviations interpretation.** $\lambda_c = \lim_{t\to\infty} t^{-1}\log\|T_t\|_{L^2 \to L^2}$ is the exponential growth rate of the semigroup operator norm. When $\lambda_c > 0$, the semigroup grows exponentially. Equivalently, $\lambda_c = \lim_{D \uparrow (0,\infty)} \lambda_1(D)$ where $\lambda_1(D)$ is the principal Dirichlet eigenvalue on bounded $D$ — connecting to the Sturm-Liouville approach that fails in practice (§7.4.3).

---

## 3. Measure Invariance and the Carré du Champ

### 3.1 CdC Definition and Invariance

**Definition 3.1.** For a diffusion $d\mathbf{X} = b(\mathbf{X})dt + \sigma(\mathbf{X})d\mathbf{W}$ with generator $\mathcal{L}f = \sum_i b_i \partial_i f + \frac{1}{2}\sum_{i,j} a_{ij}\partial_i\partial_j f$, the **Carré du Champ (CdC) operator** is:

$$\Gamma(f, g) = \mathcal{L}(fg) - f\mathcal{L}(g) - g\mathcal{L}(f) = \sum_{i,j} a_{ij}(\mathbf{x})\frac{\partial f}{\partial x_i}\frac{\partial g}{\partial x_j}$$

The CdC completely annihilates the drift vector $b(\mathbf{x})$ and isolates the diffusion matrix $a(\mathbf{x}) = \sigma(\mathbf{x})\sigma(\mathbf{x})^\top$.

**Measure invariance.** By Girsanov's theorem, changing from $\mathbb{P}$ to $\mathbb{Q}$ alters the drift $b(\mathbf{x})$ but leaves $a(\mathbf{x})$ invariant. Therefore $\Gamma$ is **invariant under measure transformations**. This is the key property enabling bubble detection from $\mathbb{P}$-data for a $\mathbb{Q}$-defined property.

### 3.2 Why the α Test Works from P-Data

Since:
1. The SLM criterion depends only on $\sigma^2(S)$ (§1.1)
2. The Feller/Khasminskii explosion criterion depends only on $\sigma^2(S)$
3. $\sigma^2(S)$ is measure-invariant (CdC invariance)

Estimating $\alpha$ from $\mathbb{P}$-data gives the correct $\mathbb{Q}$-answer. Moreover, squared increments are measure-invariant:
$$(\Delta S)^2 = (\mu\,dt + \sigma\,dW)^2 = \sigma^2\,dt + O(dt^{3/2})$$
The drift term $\mu^2 dt^2$ is negligible. The regression on $(\Delta S)^2/dt$ vs. $S$ directly estimates the $\mathbb{Q}$-quantity.

**Caveat on SV models.** Jarrow, Protter, and San Martín (2022) proved that the bubble property is NOT invariant across ELMMs in multi-dimensional SV models. An asset can be a true martingale under one measure and a bubble under another purely due to drift differences across dimensions. However, this does not undermine the $\alpha$ test when properly conditioned on the volatility state (§5): the ELMM invariance breakdown concerns the drift of the volatility process, and conditioning on the vol proxy (ARD kernel in MLKFellerGP) correctly separates the $S$-exponent from vol-induced confounding.

### 3.3 Multi-Dimensional CdC Explosion Criteria

For a $d$-dimensional diffusion, Khasminskii's radial comparison test (2012, Chapter 3) gives sufficient conditions for explosion via radial profiles of $a(\mathbf{x})$:

$$\int_1^\infty [C(\rho)]^{-1}\left(\int_1^\rho \frac{C(s)}{A(s)}\,ds\right)d\rho < \infty, \quad C(\rho) = \exp\!\left(\tfrac{1}{2}\int_1^\rho B(s)\,ds\right)$$

where:
$$A(\rho^2/2) \leq \inf_{|\mathbf{x}|=\rho}\mathbf{x}^\top a(\mathbf{x})\mathbf{x}, \qquad B(\rho^2/2) \geq \sup_{|\mathbf{x}|=\rho}\frac{\text{Tr}(a(\mathbf{x}))}{\mathbf{x}^\top a(\mathbf{x})\mathbf{x}}$$

**Proposition (Convergence under power-law growth).** If $A(\rho) \geq c\rho^\alpha$ and $B(\rho) \leq D\rho^{-\beta}$ for large $\rho$, with $\alpha > 2$ and $\beta > 0$, then the double integral converges (explosion occurs).

**Financial setting.** For SV models (Heston, CEV-Heston), along the explosion direction $S \to \infty$ with $V$ bounded: $\mathbf{x}^\top a\mathbf{x} \sim VS^{2\gamma+2}$ while $\text{Tr}(a) \sim VS^{2\gamma} + \xi^2 V$, giving $B \sim S^{-2} \to 0$. So $B$ decays quadratically and $C(\rho)$ is bounded — the 1D argument applies essentially unchanged with critical index $\alpha = 2$.

### 3.4 Radial Regular Variation for Matrix-Valued Functions

**Definition 3.2 (Radial Regular Variation).** The diffusion matrix $a : \mathbb{R}^d \to \mathbb{R}^{d\times d}_{\geq 0}$ has **radial tail index** $\alpha$ on a cone $\mathcal{C} \subseteq \mathbb{S}^{d-1}$ if:

$$\frac{\boldsymbol\theta^\top a(r\boldsymbol\theta)\boldsymbol\theta}{r^\alpha L(r)} \xrightarrow{r\to\infty} g(\boldsymbol\theta)$$

uniformly over $\boldsymbol\theta \in \mathcal{C}$, where $L$ is slowly varying and $g : \mathcal{C} \to (0,\infty)$ is the angular profile.

**Inheritance Lemma.** Under uniform convergence, the radial dispersion profile $A(\rho) := \inf_{\boldsymbol\theta\in\mathcal{C}}\boldsymbol\theta^\top a(\rho\boldsymbol\theta)\boldsymbol\theta$ is regularly varying with the same index $\alpha$:

$$A(\rho) = \rho^\alpha L(\rho)\cdot[\inf_{\boldsymbol\theta\in\mathcal{C}} g(\boldsymbol\theta)]\cdot(1 + o(1))$$

*Proof.* Let $h_\rho(\boldsymbol\theta) = \boldsymbol\theta^\top a(\rho\boldsymbol\theta)\boldsymbol\theta/(\rho^\alpha L(\rho))$. By assumption, $h_\rho \to g$ uniformly. For any $\varepsilon > 0$, for all $\rho > \rho_0(\varepsilon)$ and all $\boldsymbol\theta$: $|h_\rho(\boldsymbol\theta) - g(\boldsymbol\theta)| < \varepsilon$. Then $\inf_\theta h_\rho(\theta) \to g_* = \inf_\theta g(\theta) > 0$ (by compactness and continuity). $\square$

**Consequence.** By Karamata's theorem, the convergence/divergence of the Feller integral over $A(\rho)$ depends only on $\alpha$, not on $L$ or $g_*$. The $\alpha > 2$ criterion transfers to the multi-dimensional case unchanged.

**Potter's Theorem (BGT, Theorem 1.5.6).** For a regularly varying $f$ with index $\alpha$, any $A > 1$, $\delta > 0$, there exists $x_0$ such that for all $x, y > x_0$:

$$\frac{f(y)}{f(x)} \leq A\max\{(y/x)^{\alpha+\delta},\,(y/x)^{\alpha-\delta}\}$$

**Application.** If the true $\alpha$ is estimated with error $|\hat\alpha - \alpha| < \delta$, the Feller integral classification is correct whenever $|\alpha - 2| > \delta$. The BayesianRidge posterior SD $\sigma_\alpha$ quantifies $\delta$, and $P(\text{bubble}) = P(\alpha > 2 | \text{data}) = \Phi((\hat\alpha - 2)/\hat\sigma_\alpha)$ correctly propagates this through the convergence dichotomy.

---

## 4. Multi-Dimensional Theory: Khasminskii and Bounded Eigenfunctions

These two theorems provide dual operator-theoretic characterizations of explosion. They are proved here in full because they underlie the entire detection framework.

### 4.1 Theorem A: Khasminskii Non-Explosion via Lyapunov Functions

**Source:** Khasminskii (2012), Theorem 3.5.

**Theorem A.** Let $\mathbf{X}_t \in \mathcal{D} \subseteq \mathbb{R}^d$ be a diffusion with generator $\mathcal{L}f = \sum_i b_i\partial_i f + \frac{1}{2}\sum_{i,j} a_{ij}\partial_i\partial_j f$. If there exists a **Lyapunov function** $V : \mathcal{D} \to [1,\infty)$ with $V(\mathbf{x}) \to \infty$ as $\|\mathbf{x}\| \to \infty$ and a constant $\lambda > 0$ such that:

$$\mathcal{L}V(\mathbf{x}) \leq \lambda V(\mathbf{x}) \quad \forall\,\mathbf{x} \in \mathcal{D}$$

then the process does not explode in finite time ($\tau_\infty = \infty$ a.s.), i.e., there is **no bubble**.

**Proof.** Define $D_n = \{\mathbf{x} : V(\mathbf{x}) < n\}$ and $\tau_n = \inf\{t \geq 0 : \mathbf{X}_t \notin D_n\}$. Since $V \to \infty$, $D_n \uparrow \mathcal{D}$ and $\tau_n \uparrow \tau_\infty$.

By Dynkin's formula applied to the stopped process:
$$\mathbb{E}^{\mathbf{x}}[V(\mathbf{X}_{t\wedge\tau_n})] = V(\mathbf{x}) + \mathbb{E}^{\mathbf{x}}\!\int_0^{t\wedge\tau_n}\mathcal{L}V(\mathbf{X}_s)\,ds \leq V(\mathbf{x}) + \lambda\!\int_0^t\mathbb{E}^{\mathbf{x}}[V(\mathbf{X}_{s\wedge\tau_n})]\,ds$$

By Grönwall's inequality: $\mathbb{E}^{\mathbf{x}}[V(\mathbf{X}_{t\wedge\tau_n})] \leq V(\mathbf{x})e^{\lambda t}$.

On $\{\tau_n \leq t\}$, $V(\mathbf{X}_{\tau_n}) \geq n$, so by Markov's inequality:
$$n\cdot\mathbb{P}^{\mathbf{x}}(\tau_n \leq t) \leq \mathbb{E}^{\mathbf{x}}[V(\mathbf{X}_{t\wedge\tau_n})] \leq V(\mathbf{x})e^{\lambda t}$$

Therefore $\mathbb{P}^{\mathbf{x}}(\tau_\infty \leq t) = \lim_{n\to\infty}\mathbb{P}^{\mathbf{x}}(\tau_n \leq t) \leq \lim_{n\to\infty} V(\mathbf{x})e^{\lambda t}/n = 0$ for every $t > 0$. $\blacksquare$

**Contrapositive (bubble test).** If no such Lyapunov function exists, the process explodes — it is a strict local martingale.

**Computational implementation (LP feasibility).** On an EDMD-estimated generator $\hat{\mathcal{L}}$ with polynomial dictionary $\Psi = [S^2, S^4]$:

$$\text{Find}\,\mathbf{w} \geq \mathbf{1} \text{ s.t. } (\hat{\mathcal{L}} - \lambda I)\mathbf{w}^\top\Psi(S) \leq 0 \quad \forall S \in \mathcal{S}_\text{grid}$$

- **Feasible** $\Rightarrow$ Lyapunov function exists $\Rightarrow$ **No bubble**
- **Infeasible** $\Rightarrow$ **Bubble** (Khasminskii condition fails)

**Critical requirement.** Basis functions must be **unbounded** (polynomials). RBFs decay to zero at infinity and cannot serve as Lyapunov functions. LP infeasibility with a small basis does not prove explosion — the basis may be too small.

### 4.2 Theorem B: Ethier-Kurtz Bounded Eigenfunction Explosion

**Source:** Ethier and Kurtz (1986), Chapter 4, Theorem 4.5.4.

**Theorem B.** Let $\mathbf{X}_t$ be a Markov process with generator $\mathcal{L}$. Suppose there exists $u \in \mathcal{D}(\mathcal{L})$ with $0 < u(\mathbf{x}) \leq M < \infty$ for all $\mathbf{x}$, and $\lambda > 0$, such that:

$$\mathcal{L}u(\mathbf{x}) = \lambda\,u(\mathbf{x})$$

Then $\mathbb{P}^{\mathbf{x}}(\tau_\infty < \infty) > 0$ for all $\mathbf{x}$: the process **explodes** (is a bubble).

**Proof.** Assume for contradiction that $\tau_\infty = \infty$ a.s. and derive $u \equiv 0$.

Define $M_t^{(n)} = e^{-\lambda(t\wedge\tau_n)}u(\mathbf{X}_{t\wedge\tau_n})$. By Itô's formula:
$$dM_t^{(n)} = e^{-\lambda t}[-\lambda u(\mathbf{X}_t) + \mathcal{L}u(\mathbf{X}_t)]\,dt + (\text{martingale term})$$

By $\mathcal{L}u = \lambda u$, the drift vanishes: $M_t^{(n)}$ is a local martingale. Since $|M_t^{(n)}| \leq M$, it is a bounded (hence true) martingale, so:

$$u(\mathbf{x}) = M_0^{(n)} = \mathbb{E}^{\mathbf{x}}[M_t^{(n)}] = \mathbb{E}^{\mathbf{x}}\bigl[e^{-\lambda(t\wedge\tau_n)}u(\mathbf{X}_{t\wedge\tau_n})\bigr]$$

As $n \to \infty$, by DCT: $u(\mathbf{x}) = \mathbb{E}^{\mathbf{x}}\bigl[e^{-\lambda(t\wedge\tau_\infty)}u(\mathbf{X}_{t\wedge\tau_\infty})\bigr]$.

If $\tau_\infty = \infty$ a.s., then $u(\mathbf{x}) = \mathbb{E}^{\mathbf{x}}[e^{-\lambda t}u(\mathbf{X}_t)] \to 0$ as $t \to \infty$ by DCT (since $|e^{-\lambda t}u| \leq Me^{-\lambda t} \to 0$). This contradicts $u(\mathbf{x}) > 0$. $\blacksquare$

**Remark.** If $u$ is additionally strictly positive and continuous with $u(\mathbf{x}) \to 0$ as $\|\mathbf{x}\| \to \infty$, the strong Markov property gives a.s. explosion (Ethier-Kurtz, Corollary 4.5.5).

**Constructing the bounded eigenfunction.** When explosion occurs, define $u(\mathbf{x}) = \mathbb{E}^{\mathbf{x}}[e^{-\lambda\tau_\infty}]$ for any $\lambda > 0$. Then $0 < u \leq 1$ and $\mathcal{L}u = \lambda u$ (by Dynkin's formula + strong Markov). For CEV $\beta > 2$: this $u$ exists and is bounded $\in (0,1]$. For GBM ($\beta = 2$): $\tau_\infty = \infty$ a.s., so no such bounded eigenfunction with $\lambda > 0$ exists.

**Connection to SLMs (Dandapani-Protter 2019, Theorem 2.1).** A continuous non-negative local martingale is a strict local martingale iff there exists an ELMM $\tilde{\mathbb{Q}}$ under which it explodes in finite time. (Forward direction via Fatou: $\mathbb{E}^{\tilde{\mathbb{Q}}}[S_T] \leq \liminf_n \mathbb{E}^{\tilde{\mathbb{Q}}}[S_{T\wedge\tau_n}] = S_0$, with strict inequality when $\tau_\infty \leq T$ with positive probability.)

### 4.3 Equivalence: Test A ↔ Test B ↔ CdC Integral

| Property | Khasminskii (Thm A) | Ethier-Kurtz (Thm B) |
|----------|--------------------|--------------------|
| Tests for | Non-explosion (safety) | Explosion (bubble) |
| Direction | Sufficient for NO bubble | Sufficient for bubble |
| Basis requirement | **Unbounded** (polynomials) | **Bounded** (RBFs) |
| Test type | LP feasibility | Spectral: $\text{Re}(\lambda) > 0$ |
| Failure mode | LP infeasible $\not\Rightarrow$ bubble (basis too small) | $\text{Re}(\lambda) > 0$ may be RMT noise |

**Complete test strategy.** Use both: if Khasminskii LP is feasible $\Rightarrow$ no bubble. If LP is infeasible AND Ethier-Kurtz finds $\text{Re}(\lambda) > 0$ with null-calibrated significance $\Rightarrow$ bubble.

In practice, this three-way equivalence (explosion $\iff \lambda_c > 0 \iff \alpha > 2$) has a single numerically accessible form: the $\alpha$ test. See §7 for why the eigenvalue computation fails in practice despite the exact theoretical equivalence.

### 4.4 Connection to Qin-Linetsky Spectral Theory (Post-Detection)

Qin-Linetsky (2015) assume the state process is **conservative** (non-explosive). Their recurrence/transience distinction operates within the non-explosive regime. Q&L's Theorem 3.1 (uniqueness of the recurrent eigenfunction) applies when $\alpha \leq 2$. The relationship is:

```
σ²(S) ~ c²S^α
     │
     ├── α > 2: EXPLOSIVE → strict local martingale → BUBBLE
     │   (Outside Q&L framework)
     │
     └── α ≤ 2: NON-EXPLOSIVE → true martingale → NO BUBBLE
         (Q&L framework applies)
         │
         ├── Process recurrent → recurrent eigenfunction π_R exists (unique)
         │   → Hansen-Scheinkman factorization
         │   → Ross Recovery possible
         │
         └── Process transient → no recurrent eigenfunction
             → Hansen-Scheinkman factorization not unique
             → Ross Recovery fails
```

Bubble detection ($\alpha > 2$) and Q&L spectral theory ($\alpha \leq 2$) are complementary: the former determines whether the process is explosive; the latter characterizes long-term behavior of non-explosive processes.

---

## 5. Stochastic Volatility: When 1D Works and Fails

### 5.1 Effective 1D Diffusion Under Ergodic Volatility

For a general SV model $dS = \mu(S,V)S\,dt + \sigma_S(S,V)\,dW_S$, the 1D KGEDMD estimates the **effective diffusion coefficient**:

$$\sigma^2_\text{eff}(S) = \mathbb{E}\!\left[\sigma_S^2(S, V) \mid \text{process visits } S\right]$$

This is the time-averaged conditional expectation — kernel regression on $(\Delta S)^2/\Delta t$ given $S$ estimates this quantity.

**When it works — fast-mixing volatility (Heston, CIR):** If $V$ is positive recurrent with stationary mean $\theta$, then $\mathbb{E}[V | S = s] \to \theta$ as $s \to \infty$ (mean-reversion dominates regardless of $S$ level). So $\sigma^2_\text{eff}(S) \approx \theta S^2$, giving $\alpha_\text{eff} = 2$ (no bubble, correct).

**When it fails — correlated vol with transient price-vol dependence (SABR $\gamma > 1$):** For SABR, $\sigma^2_S = V^2 S^{2\gamma}$, so $\sigma^2_\text{eff}(S) = \mathbb{E}[V^2 | S = s]\cdot S^{2\gamma}$. With leverage $\rho < 0$, paths reaching high $S$ required sustained high $V$, which is stochastic and mean-reverting. The conditioning induces **negative correlation** between $\mathbb{E}[V^2|S]$ and $S$, pulling the effective $\alpha$ well below $2\gamma$.

**Experimental confirmation.** SABR $\gamma = 1.5$ gives $\hat\alpha \approx 0.2$ instead of $2\gamma = 3.0$. The marginal 1D approach completely fails because the $V$-$S$ correlation dominates.

**Model coverage table:**

| Vol Model | Positive Recurrent? | $\mathbb{E}[V|S=s]$ as $s\to\infty$ | Index Preserved? |
|-----------|--------------------|------------------------------------|-----------------|
| CIR/Heston | Yes ($2\kappa\theta \geq \xi^2$) | $\to \theta$ (bounded) | Yes |
| OU (Stein-Stein) | Yes | $\to \theta$ (bounded) | Yes |
| GARCH diffusion | Yes | Bounded | Yes |
| **SABR** | **No** (log-normal vol is GBM) | **Unbounded** — grows with $S$ | **Possibly not** |
| **3/2 model** | **Depends** on parameters | Can explode | **Fails if non-recurrent** |
| Rough vol (fOU) | Not Markov | Generalized stationarity | Heuristic only |

### 5.2 Log-Linear Separation for Multiplicatively Separable SV

**Proposition (Log-Linear Separation).** For $a_{11}(S,V) = f(V)\cdot g(S)$ with $g \in RV_\alpha$, let $\hat V_t$ be a consistent estimator of $V_t$. The log-linear regression:

$$\log\hat\sigma^2 \sim \alpha\cdot\log S + \beta\cdot\log\hat V + c$$

yields a consistent estimator of $\alpha$ under any vol dynamics (ergodic or transient), provided $\hat V$ is consistent.

*Proof.* By separability, $\log a_{11} = \log g(S) + \log f(V)$. The regression target $(\Delta S)^2/dt$ estimates $f(V_t)g(S_t)$. Taking logs: $\log((\Delta S)^2/dt) \approx \log g(S_t) + \log f(V_t) + \varepsilon_t$ (mean-zero after Jensen correction). OLS on $[\log S_t, \log\hat V_t]$ identifies the $S$-coefficient as the regular variation index of $g$ by the Frisch-Waugh-Lovell theorem: the $\log S$ coefficient equals the coefficient from regressing the response on $\log S$ after partialling out $\log\hat V$. $\square$

**Why this works for SABR ($\rho < 0$).** The 1D regression confounds $S^{2\gamma}$ scaling with $V^2$-shrinkage from leverage. The 2D regression separates them: $\alpha$ captures $S^{2\gamma}$, $\beta$ captures $V^2$ dependence. The Feller test uses $\alpha$ alone.

**Coverage table:**

| Model | $a_{11}(S,V)$ | $f(V)$ | $g(S)$ | True $\alpha$ |
|-------|--------------|--------|--------|--------------|
| CEV | $\sigma^2 S^{2\gamma}$ | $\sigma^2$ | $S^{2\gamma}$ | $2\gamma$ |
| Heston | $VS^2$ | $V$ | $S^2$ | 2 |
| CEV-Heston | $VS^{2\gamma}$ | $V$ | $S^{2\gamma}$ | $2\gamma$ |
| SABR | $V^2 S^{2\gamma}$ | $V^2$ | $S^{2\gamma}$ | $2\gamma$ |
| 3/2-CEV | $VS^{2\gamma}$ | $V$ | $S^{2\gamma}$ | $2\gamma$ |

### 5.3 The JPS 2022 Counter-Example: Non-Separable SV

JPS (2022, Remark 6) construct a counter-example where no Feller-based test can detect the bubble:

$$dX = X\cdot Y\,dW_1, \quad dY = KY^2\,dt + Y(\rho\,dW_1 + \sqrt{1-\rho^2}\,dW_2)$$

Here bubble $\iff K \geq -\rho$. At any fixed $Y = y$:

$$\sigma^2(X | Y = y) = X^2\cdot y^2 \implies \alpha = 2 \text{ (exactly, for all } y\text{)}$$

The bubble arises NOT from $\alpha > 2$ but from the coupled $Y$ dynamics ($KY^2$ drift) feeding back through correlation. No Feller-based test can detect this — it requires the joint generator spectrum (positive eigenvalue $\Rightarrow$ explosion). This motivates Level L3 in the tiered architecture (§8), which applies the Koopman generator eigenvalue test on the 2D state $(X, Y)$.

### 5.4 The Conditional Feller Approach (L2-SV)

When $\sigma^2(S, V) = g(V)\cdot S^{\alpha(V)}$ with level-dependent exponent $\alpha(V)$, the marginal test averages across vol regimes and may miss bubbles appearing only at high vol:

1. Bin observations by vol proxy $V$ into quantiles
2. Run Feller $\alpha$ test on price sub-series within each bin
3. Aggregate with Šidák correction: $P(\text{bubble}) = 1 - \prod_{q=1}^Q(1 - P_q)$

This detects regimes where $\alpha(V) > 2$ only at certain vol levels while the marginal $\alpha \approx 2$ due to averaging. Validated on CIR-modulated CEV: conditional Feller 6/6, marginal 3/6.

---

## 6. Jump Robustness and Universality

### 6.1 The Semimartingale Setting

Consider the general Itô semimartingale:

$$S_t = S_0 + \int_0^t \mu_s\,ds + \int_0^t \sigma_s\,dW_s + J_t$$

where $J_t = \sum_{i=1}^{N_t}\Delta J_i$ (finite-activity jumps).

**Assumption 2.1 (Standard conditions).** (FA1) $N_t \sim \text{Poisson}(\lambda t)$, $\lambda < \infty$; (FA2) $\mathbb{E}[|\Delta J|^{2+\delta}] < \infty$ for some $\delta > 0$. This covers Merton (1976), Bates (1996), Kou (2002).

### 6.2 Bipower Variation

**Definition (Barndorff-Nielsen & Shephard 2004).** The realized bipower variation is:

$$BV_n = \frac{\pi}{2}\sum_{i=2}^n |\Delta S_i|\cdot|\Delta S_{i-1}|$$

**Theorem 2.3 (BNS 2004, 2006).** Under Assumption 2.1: $BV_n \xrightarrow{p} \int_0^T \sigma^2_s\,ds$ as $n \to \infty$.

*Proof sketch.* Consecutive increments $|\Delta S_i|\cdot|\Delta S_{i-1}|$ involve adjacent intervals. For the continuous component, both factors are $O(\sqrt{\Delta_n})$, giving $O(\Delta_n)$ product (correct order). For jumps (finite activity), a jump of size $O(1)$ appears in at most one interval; the adjacent factor is $O(\sqrt{\Delta_n})$, so each cross-term contributes $O(\sqrt{\Delta_n})$. Summed over $O(\lambda T)$ jumps: total jump bias is $O(\lambda T\sqrt{\Delta_n}) \to 0$. The continuous part converges to $\int\sigma^2_s\,ds$ by standard BPV theory. $\square$

**Bipower Nadaraya-Watson estimator.** The localized version:

$$\hat\sigma^2_{BV}(x) = \frac{\pi}{2}\cdot\frac{\sum_{i=2}^n K_h(S_{i-1}-x)\cdot|\Delta S_i|\cdot|\Delta S_{i-1}|}{\sum_{i=2}^n K_h(S_{i-1}-x)}$$

**Theorem 2.5 (Consistency).** Under Assumption 2.1, if $\sigma^2_c$ is continuous and the process is positive recurrent, then as $n\to\infty$, $\Delta_n\to 0$, $h\to 0$, $nh\Delta_n\to\infty$: $\hat\sigma^2_{BV}(x) \xrightarrow{p} \sigma^2_c(x)$.

**Theorem 2.6 (Jump-robust Feller test).** Under Assumption 2.1 with $\sigma^2_c \in RV_\alpha$: (a) $\hat\alpha_{BV} \xrightarrow{p} \alpha$; (b) $P(\alpha > 2 | \text{data}) = \Phi((\hat\alpha_{BV}-2)/\hat\sigma_\alpha)$ is a consistent test for the SLM property of the continuous component; (c) jumps do not affect the test.

### 6.3 The Resolution Principle: Finite vs. Infinite Activity

**Theorem 3.1 (Effective finite activity).** Set truncation $\varepsilon = C\sqrt{\Delta_n}$. Then:

(a) The compensated small-jump component $M^\varepsilon_t = \int_0^t\int_{|x|\leq\varepsilon} x\,\tilde\mu(ds,dx)$ has per-step increments that are $O_p(\sqrt{\Delta_n})$ — asymptotically Gaussian, absorbed into the diffusion.

(b) Large jumps have finite activity: $\nu(\{|x| > \varepsilon\}) < \infty$ for all $\varepsilon > 0$ (definition of Lévy measure).

(c) At sampling frequency $\Delta_n$, the observed process is statistically indistinguishable from a diffusion + compound Poisson model satisfying Assumption 2.1.

### 6.4 Market Microstructure as Resolution Bound

- **Tick size** (Proposition 3.2): All observed price changes satisfy $|\Delta S| \in \{0, \delta, 2\delta, \ldots\}$ where $\delta$ is the minimum price increment. The effective Lévy measure has finite activity: $\nu_\text{obs}(\mathbb{R}\setminus\{0\}) = \nu(\{|x| \geq \delta\}) < \infty$.

- **Circuit breakers** (Proposition 3.3): Trading halts at thresholds (NYSE Level 1 at $-7\%$) imply $|\Delta S/S| \leq \bar r$, so $\mathbb{E}[|\Delta J|^p] < \infty$ for all $p > 0$.

- **Finite order flow** (Proposition 3.4): Price changes occur through discrete order book events with finite event rate $\lambda_\text{events} < \infty$. The physical mechanism ensures finite-activity jumps in practice.

### 6.5 Jump Bubbles are Impossible Under Standard Conditions

**Theorem (Independent bubble sources; Protter 2013).** Write the multiplicative decomposition:
$$S_t = S_0\cdot\mathcal{E}(M^c)_t\cdot\mathcal{E}(M^d)_t$$
A continuous bubble exists iff $\mathcal{E}(M^c)$ is a SLM (i.e., $\alpha > 2$). A jump bubble exists iff $\mathcal{E}(M^d)$ is a SLM, i.e., $\int_{|x|>1}|x|\,\nu(dx) = \infty$.

**Theorem 4.3 (No jump bubbles under finite moments).** Under Assumption 2.1:
$$\int_{|x|>1}|x|\,\nu(dx) = \lambda\int_{|x|>1}|x|\,F(dx) \leq \lambda\,\mathbb{E}[|\Delta J|] < \infty$$

Jump bubbles are impossible. By Propositions 3.3 and 3.4, this holds for all observed financial price processes.

**Jump bubbles for infinite-activity processes (theoretical completeness).** Only pure stable processes with $\nu(dx) \sim |x|^{-1-\alpha}dx$ (no exponential tempering) and $\alpha \geq 1$ can produce jump bubbles. Standard models (NIG, CGMY, VG) have exponentially tempered tails, ensuring $\int_{|x|>1}|x|\,\nu(dx) < \infty$. Pure stable processes with $\alpha \geq 1$ violate finite variance and are not used as asset price models.

### 6.6 The Blumenthal-Getoor Index and Power Variation

**Definition 4.5 (BG index).** $\beta_{BG} = \inf\{p \geq 0 : \int_{|x|\leq 1}|x|^p\,\nu(dx) < \infty\}$.

**Theorem (Todorov-Tauchen 2011; Aït-Sahalia-Jacod 2009).** For $p$-th power variation $V^p_n = \sum_{i=1}^n|\Delta S_i|^p$:

$$\Delta_n^{1-p/2}V^p_n \xrightarrow{p} \begin{cases} m_p\int_0^T|\sigma_s|^p\,ds & p < \beta_{BG} \\ \sum_{s\leq T}|\Delta S_s|^p & p > \beta_{BG} \end{cases}$$

where $m_p = \mathbb{E}[|Z|^p]$ for $Z\sim\mathcal{N}(0,1)$. The slope change in $\log V^p_n$ vs. $p$ identifies $\beta_{BG}$.

### 6.7 Universality Theorem

**Theorem 5.1 (Universality).** Let $S$ be a positive semimartingale satisfying NFLVR with:
- (U1) $\sigma^2_c$ smooth and eventually monotone (implied by leverage effect)
- (U2) $\nu(\{|x|>\varepsilon\}) < \infty$ for all $\varepsilon > 0$ (Lévy measure definition)
- (U3) $\mathbb{E}[|\Delta J|^{2+\delta}] < \infty$ (implied by circuit breakers)

Then: (a) $\sigma^2_c \in RV_\alpha$ (Theorem 2.3); (b) $\hat\sigma^2_{BV}$ consistently estimates $\sigma^2_c$ (Theorem 2.5); (c) the Feller test $\hat\alpha_{BV} > 2$ is consistent for continuous bubbles (Theorem 2.6); (d) jump bubbles are impossible (Theorem 4.3).

**The BV-robust Feller test is necessary and sufficient for bubble detection in the class of NFLVR semimartingales consistent with (U1)–(U3).**

---

## 7. Why Eigenvalue Approaches Fail

This section documents four operator-based approaches that were tested and failed. The failures are instructive: they explain why the theoretically equivalent eigenvalue form $\lambda_c > 0$ cannot be numerically accessed from discrete data, while the $\alpha > 2$ form can.

### 7.1 CdC via Generator: Catastrophic Cancellation

The CdC identity $\sigma^2(S) = \Gamma(S,S) = L(S^2) - 2S\cdot L(S)$ suffers catastrophic cancellation. At $S = 100$: $L(S^2) \approx 10^3$ and $2S\cdot L(S) \approx 10^3$, while $\sigma^2 \approx 9$. Relative error amplified $\sim 100\times$, resulting in $\sim 0.4$ upward bias on $\alpha$.

**Resolution.** Direct KRR on $(\Delta S)^2/\Delta t$ avoids the subtraction entirely.

### 7.2 Eigenfunction Growth Reconstruction

RBF/Nyström basis functions are bounded by construction. All reconstructed eigenfunctions $\hat\pi(x) = \sum_i c_i\phi_i(x)$ decay at the boundary regardless of the true behavior. Cannot distinguish bounded vs. unbounded eigenfunctions. Note: this tests the wrong criterion — boundedness of the Khasminskii eigenfunction is automatic in RBF RKHS, so growth reconstruction is incoherent.

### 7.3 Multi-Step Koopman Propagation

$K^n$ amplifies approximation error exponentially. The martingale defect per step is $O(\Delta t^2) \approx 10^{-4}$, smaller than the Koopman approximation error. Signal drowned by noise at all useful horizons.

### 7.4 Eigenvalue Sign Test: Theory vs. Practice

**Theory.** The bounded eigenfunction theorem (Ethier-Kurtz 4.5.4; Engländer-Pinsky 1999) states: explosion $\iff \exists$ bounded positive $u$ with $Lu = \lambda u$, $\lambda > 0$. For RBF EDMD, all RKHS functions are bounded by construction. If the learned generator $L$ has an eigenvalue with $\text{Re}(\lambda) > 0$ and positive eigenfunction, the Khasminskii criterion is satisfied.

**Experimental results** (6 DGPs × 3 seeds):

| DGP | True | max Re($\lambda$) | P(bub|$\lambda$) | $\hat\alpha$ | Eigen correct | $\alpha$ correct |
|-----|------|-------------------|------------------|------------|--------------|-----------------|
| GBM $\sigma=0.3$ | no bubble | $-0.0005\pm0.000$ | 0.000 | 2.04 | 3/3 | 1/3 |
| CEV $\beta=1.5$ | no bubble | $-0.0107\pm0.004$ | 0.000 | 1.52 | 3/3 | 3/3 |
| CEV $\beta=2.5$ | **bubble** | $-0.0016\pm0.002$ | 0.000 | 2.51 | **0/3** | 3/3 |
| CEV $\beta=3.0$ | **bubble** | $-0.0028\pm0.003$ | 0.000 | 2.96 | **0/3** | 3/3 |
| Heston | no bubble | $-0.0083\pm0.006$ | 0.000 | 1.77 | 3/3 | 3/3 |
| SABR $\gamma=1.5$ | **bubble** | $0.0007\pm0.008$ | 0.000 | 0.15 | **0/3** | 0/3 |

**Result: FAILS.** Correctly identifies all non-bubble DGPs, misses every bubble DGP.

**Why it fails — signal-to-noise.** For $\Delta t = 0.01$, a true generator eigenvalue $\lambda = 0.5$ maps to Koopman eigenvalue $\mu = 1 + \lambda\cdot\Delta t = 1.005$. With $m = 80$ landmarks and regularization $\lambda_\text{reg} = 10^{-3}$, this 0.5% deviation from 1 is below the noise floor. Regularization shrinks Koopman eigenvalues toward 0 (generator eigenvalues toward $-\infty$), burying the signal. Even for CEV $\beta = 3.0$ (strong bubble): max Re($\lambda$) $\approx -0.003$.

### 7.4.1 Multi-Horizon Direct Fitting: Also Fails

Instead of $K^n$, fit the Koopman operator directly at longer horizons using $(X_t, X_{t+k\Delta t})$ pairs. At $\Delta t = k\cdot dt$, the Koopman eigenvalue $\mu = e^{\lambda\Delta t}$ is amplified: for $\lambda = 0.5$, $\mu(\Delta t=0.01) = 1.005$ vs. $\mu(\Delta t=1.0) = 1.65$.

**Multi-horizon results** (horizons $k \in \{1, 10, 50, 100\}$, 6 DGPs × 3 seeds):

| DGP | True | Single P(bub) | Multi P(bub) | Single | Multi | $\alpha$ test |
|-----|------|--------------|-------------|--------|-------|--------------|
| CEV $\beta=2.5$ | bubble | 0.000 | 0.000 | **0/3** | **0/3** | 3/3 |
| CEV $\beta=3.0$ | bubble | 0.000 | 0.000 | **0/3** | **0/3** | 3/3 |
| SABR $\gamma=1.5$ | bubble | 0.000 | 0.000 | **0/3** | **0/3** | 0/3 |

**Result: Multi-horizon ALSO FAILS.** No improvement over single-step.

### 7.4.2 Root Cause: Discrete-Time Processes Cannot Explode

Initially attributed to conservative Euler clamping. Tested exponential Euler (log-space, no clamping) — results identical. Also tested Sturm-Liouville eigenvalue computation on the estimated generator — 1/6 DGPs correct (worse than Koopman due to noisy estimates at grid boundaries).

**The true root cause is fundamental:** Discrete-time Markov chains cannot explode in finite time, regardless of simulation scheme. The explosive eigenvalue $\lambda_c > 0$ is a continuous-time phenomenon — it reflects accumulation of infinitely many infinitesimal steps. At any finite $\Delta t$, the Koopman transition operator is a stochastic kernel with $|\mu| \leq 1$, and $\lambda_\text{gen} = \log(\mu)/\Delta t$ is biased toward $\leq 0$ by regularization noise. This applies equally to real data — prices observed at discrete times never literally reach infinity.

### 7.4.3 Sturm-Liouville Attempt: Also Fails

Discretize $L = \frac{1}{2}\hat\sigma^2\partial^2 + \hat\mu\partial$ on a grid with Dirichlet BCs. Theory: as domain $D \to (0,\infty)$, $\lambda_1(D) \to \lambda_c$. Practice: noise in $\hat\sigma^2$ and $\hat\mu$ at the grid boundaries dominates, creating spurious positive eigenvalues for non-bubble DGPs (GBM false positive rate 33%, Heston 67%) while still missing bubble DGPs. Sturm-Liouville is also 1D-specific and cannot scale to multivariate.

### 7.5 Why Bounded RKHS Functions Cannot Detect Bubbles

The Khasminskii eigenfunction $u(x) = \mathbb{E}[e^{-\lambda\tau_\text{exp}}]$ is theoretically bounded in $(0,1]$, but it is nearly constant ($\approx 1$) everywhere in the observed price range and only deviates near the explosion boundary (at infinity). RBF kernel functions are localized — they cannot resolve a function that is essentially flat except at infinity. The explosive signal lives in **unbounded** test functions ($S^p$ for large $p$), which are outside the RBF RKHS.

**Spurious modes from extrapolation failure.** Outside the convex hull of the landmark grid $\mathcal{C}$, RBF functions artificially decay to zero. If data wanders outside $\mathcal{C}$, the Koopman matrix invents right-half-plane eigenvalues to fit the artificial boundary cliff. These are not true explosion signals — they are extrapolation artifacts. They are resolved by dynamic landmark updates or by the dual CdC test (which is algebraic and immune to boundary pollution).

### 7.6 Conclusion: Why the α Test Succeeds

The three equivalent forms — $\text{explosion} \iff \lambda_c > 0 \iff \alpha > 2$ — have radically different numerical properties:

- $\lambda_c > 0$: requires resolving the near-constant survival function $u(x)$ from discrete observations; impossible with bounded RKHS functions
- $\alpha > 2$: requires asking whether $\sigma^2(S)$ grows faster than $S^2$; this is a **local** property of the generator coefficients, not a **global** spectral property

The $\alpha$ test succeeds because the Feller integral characterizes explosion through local behavior of $\sigma^2(S)$ in the observed range, whereas the eigenvalue characterization requires global spectral information about the entire half-line. The log-log regression $\log\hat\sigma^2 \sim \alpha\log S$ directly estimates the quantity that determines explosion, without passing through the theoretically equivalent but numerically inaccessible eigenvalue form.

---

## 8. Tiered Detection Architecture

### 8.1 Three Tiers of Generality

The architecture is structured as a progressive relaxation of assumptions, with each tier adding one layer of the signature toolkit:

| Tier | SDE Class | Assumptions | Active Levels | Detection Floor |
|------|-----------|-------------|--------------|----------------|
| I | Autonomous Markov | $dS = \sigma(S)dW$, time-invariant | L1 + L2 | ~2-3 months (5-min) |
| II | Non-autonomous Markov | $dS = \sigma(t,S)dW$, time-varying | L1 (windowed) | ~6 months |
| III | fSDE (non-Markov) | $dS = \sigma(S)dW^H$, $H \neq 0.5$ | L1 + L3, fGN whitening | ~6-12 months |

**Tier I** (time-invariant Markov) yields the sharpest results: the Feller test is exact, ergodic recurrence gives optimal sample efficiency, and signatures reduce to QV estimation. With 5-minute bars, $\text{SE}(\hat\alpha) \sim O(1/\sqrt{N})$ gives $2\sigma$ separation in 2-3 months for $\alpha \geq 3$.

**Tier II** relaxes time-invariance. $\sigma(t, S)$ drifts, so L2 becomes unreliable. L1 adapts via rolling windows with fewer effective samples per window. Signatures capture instantaneous QV without assuming stationarity.

**Tier III** drops the Markov assumption. The process has memory ($H \neq 0.5$), so pointwise methods are misspecified. fGN whitening corrects spurious $dt^{2H}$ scaling causing false positives on rough paths.

### 8.2 Detection Level Table

| Level | Test | Handles | Method | Tier |
|-------|------|---------|--------|------|
| L1 | Sig QV scaling ($\alpha > 2$) | 1D CEV-type | $\log(\text{QV}) \sim \alpha\log\bar S$ | I, II, III* |
| L2 | Nonparametric Feller + GP | 1D + directional portfolios | NW $\hat\sigma^2(z)$ + GP posterior | I only |
| L2-SV | Conditional Feller | Separable SV: $\sigma^2(S,V) = f(V)S^{\alpha(V)}$ | Bin by $V$, Feller per bin | I |
| L3 | Koopman generator eig. | Non-separable SV (JPS 2022) | 2D generator eigenvalue | Future |

*With fGN whitening for Tier III.

### 8.3 Per-Level Bayesian Noisy-OR Combination

Each level detects a **different bubble mechanism**: L1 detects CEV-type scaling, L2-SV detects Feller violations in stochastic volatility, L3 detects general explosion. These are **orthogonal** — a CEV bubble does not trigger the Feller test, and vice versa.

A shared-$\theta$ hierarchical model (where all levels respond to the same latent bubble state) is **structurally wrong**: when L2 is silent for a CEV bubble, this is irrelevant, not evidence against a bubble.

Instead, we use **per-level Bayesian noisy-OR**. Each level has its own latent bubble indicator:

$$\theta_k \sim \text{Bernoulli}(\pi_0), \quad P_k | \theta_k = 1 \sim \text{Beta}(a_1^{(k)}, b_1^{(k)}), \quad P_k | \theta_k = 0 \sim \text{Beta}(a_0^{(k)}, b_0^{(k)})$$

Each level computes its own posterior:

$$P(\theta_k = 1 | P_k) = \frac{\pi_0 f_1^{(k)}(P_k)}{\pi_0 f_1^{(k)}(P_k) + (1-\pi_0)f_0^{(k)}(P_k)}$$

Combined via noisy-OR (any mechanism suffices):

$$P(\text{Bubble}) = 1 - \prod_{k=1}^K\bigl(1 - P(\theta_k = 1 | P_k)\bigr)$$

A level that is silent ($P_k \approx 0$) contributes $(1-0) = 1$ to the product — no effect on the overall posterior.

**Default priors:** All levels: Bubble $\text{Beta}(5, 1.5)$, Null $\text{Beta}(1.5, 5)$; L3: Bubble $\text{Beta}(3, 2)$, Null $\text{Beta}(1.5, 5)$ (weaker signal); Prior: $\pi_0 = 0.1$ (bubbles are rare).

### 8.4 Roughness Handling: fGN Whitening

When $H \neq 0.5$, Levels 1-2 (which assume Markov dynamics) become misspecified. Rather than applying an ad-hoc multiplicative "roughness discount" $\eta = H/0.5$ (no theoretical justification), apply **fGN whitening**:

1. Estimate $H$ from the price path (DFA, variogram, periodogram, or R/S)
2. Build the fGN correlation matrix $\Sigma_H$ with $r(k) = \frac{1}{2}(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})$
3. Cholesky: $\Sigma_H = LL^\top$
4. Whiten increments: $\Delta\Phi_w = L^{-1}\Delta\Phi$
5. Run all tests on whitened data

This removes spurious $dt^{2H}$ divergence without discarding statistical power through arbitrary discounting. For Level 3 (GP Koopman), roughness is handled automatically: the GP posterior covariance absorbs misspecification as increased noise variance.

### 8.5 DGP Benchmark Suite (10 Cases, Tiered)

Each tier's detector is tested on all DGPs from current and previous tiers (**nesting property**: adding levels must not break simpler case detection).

**Tier I: Autonomous Markov ($dS = \sigma(S)dW$)**

| # | DGP | Truth | Challenge |
|---|-----|-------|-----------|
| 1 | GBM $\sigma = 0.20$ | STABLE | Baseline |
| 2 | CEV $\gamma = 0.8$ | STABLE | Below threshold |
| 3 | CEV $\gamma = 1.5$ | BUBBLE | Moderate bubble |
| 4 | CEV $\gamma = 2.0$ | BUBBLE | Strong bubble |

**Tier II: Markov Stochastic Volatility**

| # | DGP | Truth | Challenge |
|---|-----|-------|-----------|
| 5 | Heston standard $\xi = 0.3$ | STABLE | Stochastic vol |
| 6 | Heston high-vol $\xi = 0.5$ | STABLE | High vol-of-vol |
| 7 | Feller Heston $\xi = 3.0$ | BUBBLE | Vol-driven bubble |

**Tier III: Non-Markov (Rough/fSDE)**

| # | DGP | Truth | Challenge |
|---|-----|-------|-----------|
| 8 | Rough vol $H = 0.3$ | STABLE | Rough Heston |
| 9 | Rough vol $H = 0.1$ | STABLE | Very rough |
| 10 | fSDE CEV $H=0.1$, $\gamma=2.0$ | BUBBLE | Rough + bubble |

**Validated Results (5-min bars, $T = 0.5$ yr, $N \approx 9828$):**

| Tier | Score | Nesting |
|------|-------|---------|
| I (L1 only) | 4/4 | — |
| II (L1+L2) | 7/7 | T1 ✓ (4/4) |
| III (L1+L2+L3) | 10/10 | T1 ✓ (4/4), T2 ✓ (3/3) |

All uncertainty estimates are Bayesian: L1 via BayesianRidge posterior $\alpha \sim \mathcal{N}(\hat\alpha, \sigma^2_\alpha)$, L2 via BayesianRidge posterior sampling of CIR parameters, L3 via GP posterior with null-calibrated Bayesian Tracy-Widom test.

---

## 9. The GP Pipeline

### 9.1 GP-KRR Equivalence

Kernel ridge regression (KRR) with regularization $\lambda$ and GP regression with noise variance $\sigma^2_n$ are identical (Rasmussen & Williams 2006, §6.2):

$$\hat f(x) = \mathbf{k}(x, X)(\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y} = \text{GP posterior mean with } \sigma^2_n = \lambda$$

The GP additionally provides the posterior variance:

$$\text{Var}[f(x)] = k(x,x) - \mathbf{k}(x,X)(\mathbf{K} + \sigma^2_n\mathbf{I})^{-1}\mathbf{k}(X,x)$$

This variance is the unified uncertainty quantification across all tiers.

### 9.2 Stage 1: Diffusion Coefficient Estimation

Given observations $(X_t, \Delta X_t)$, the squared increments $y_t = (\Delta X_t)^2/\Delta t$ are noisy observations of $\sigma^2(X_t)$. The GP model:

$$\sigma^2(x) \sim \text{GP}(0, k_\text{RBF}), \quad y_t = \sigma^2(X_t) + \varepsilon_t$$

is exactly the Nadaraya-Watson estimator (= KRR with specific kernel). The Nyström approximation (landmark basis) is a standard GP inducing-point method (Quiñonero-Candela & Rasmussen 2005).

**KGEDMD consistency theorem** (Steinwart & Christmann 2008, Theorem 6.23): Under the RBF kernel (universal), stationary ergodic data, $\lambda_N \to 0$ with $\lambda_N N \to \infty$, and landmarks spanning the data support: $\hat\sigma^2(S) \to \sigma^2(S)$ in probability pointwise. Finite-sample rates: $O(m^{-1/2}) + O(N^{-1/2})$ for $m$ Nyström landmarks and $N$ data points.

**Practical performance.** With $m = 80$, $N = 10{,}000$: $\sim 10\%$ pointwise error, consistent with observed 1.8–5.1% RMSE on ergodic processes (CdC kernel estimators experiment).

### 9.3 Stage 2: Feller Exponent Estimation

Given GP estimates $\hat\sigma^2(z_j)$ at landmarks, the Feller test fits a GP with parametric mean (R&W §2.7):

$$\log\hat\sigma^2(z_j) = \underbrace{\alpha\log|z_j| + c}_{\text{parametric mean}} + \underbrace{f(z_j)}_{\text{GP residual}} + \varepsilon_j$$

The posterior on $\beta = (\alpha, c)$ is:

$$\bar\beta = (H^\top C^{-1}H)^{-1}H^\top C^{-1}\mathbf{y}, \quad \text{Cov}(\beta) = (H^\top C^{-1}H)^{-1}$$

where $C = \sigma_f^2 K_\text{SE} + \Sigma_n$. The bubble probability:

$$P(\text{bubble} | \text{data}) = P(\alpha > 2 | \text{data}) = \Phi\!\left(\frac{\hat\alpha - 2}{\hat\sigma_\alpha}\right)$$

When $\sigma_f$ is selected via blocked CV, the GP automatically widens the posterior at the Feller boundary ($\alpha \approx 2$ for GBM), giving calibrated $P \approx 0.5$ — honest UQ for borderline cases.

### 9.4 Stage 3: Eigenfunction Pricing (Post-Detection)

The KGEDMD Koopman solve $\hat K = (\mathbf{K}_{nM}^\top\mathbf{K}_{nM} + \lambda\mathbf{I})^{-1}\mathbf{K}_{nM}^\top\mathbf{K}_{n+1,M}$ is $d$ parallel GP regressions. The generator $\hat L = (\hat K - I)/\Delta t$ gives:

$$\mathbb{E}[f(X_T) | X_0] = \sum_k c_k\cdot\lambda_k^{T/\Delta t}\cdot v_k(X_0)$$

When $\alpha \leq 2$ (no bubble), this gives the Hansen-Scheinkman factorization and Q&L eigenfunction pricing.

### 9.5 Kernel Specialization Across Tiers

| Tier | GP Observation Model | Kernel $k$ | Mean Function $m$ |
|------|---------------------|------------|-------------------|
| L1 | $\log\hat\sigma^2 = \alpha\log|S| + c + \varepsilon$ | $k = 0$ (degenerate) | $\alpha\log|S| + c$ |
| L2 | $\log\hat\sigma^2_w(z) = \alpha_w\log|z| + c + f(z) + \varepsilon$ | $k_\text{SE}(\log z)$ | $\alpha\log|z| + c$ |
| L2-SV | Same, per V-bin | $k_\text{SE}(\log z) \otimes k_\text{SE}(V)$ | $\alpha(V)\log|z| + c(V)$ |
| L3 | $g(y) = 2\mu(y)/\sigma^2(y)$ (scale integrand) | $k_\text{SE}(y)$ | 0 |
| Pricing | $\phi_j(X_{t+\Delta t}) = \sum_i K_{ij}\phi_i(X_t) + \varepsilon$ | $k_\text{RBF}(X)$ | 0 |

L1 IS L2 with $\sigma_f = 0$. L2-SV IS L2 with product kernel. GCV model selection over $\sigma_f$ and kernel weights automatically selects the appropriate tier from data.

### 9.6 Level 3 as GP on the Scale Function Integrand

For non-separable SV where the Feller $\alpha$ test gives $\alpha = 2$ at every vol level (JPS 2022 case), the bubble mechanism operates through the vol process drift. The full Feller boundary classification uses:

$$s(y) = \int^y \exp\!\left(-\int^u \frac{2\mu(v)}{\sigma^2(v)}\,dv\right)du$$

Bubble (vol explosion) $\iff s(\infty) < \infty$ $\iff$ the integrand $g(y) = 2\mu(y)/\sigma^2(y)$ is sufficiently positive at large $y$.

The GP model $g(y) \sim \text{GP}(0, k_\text{SE}(y))$ with $g(y_j)$ estimated at landmarks from NW estimates of $\mu(y_j)$ and $\sigma^2(y_j)$. When $y$ is small (uninformative data), the posterior reverts to the prior $\Rightarrow P(g > 0) \approx 0.5$ $\Rightarrow$ calibrated uncertainty at the detection boundary.

### 9.7 Connection to Existing Methods

The GP framework subsumes existing approaches:

- **Nadaraya-Watson** (Aït-Sahalia & Jacod): GP MAP with bandwidth-matched kernel, no UQ
- **PSY/GSADF**: Tests drift (measure-dependent), not $\sigma^2$ (measure-invariant); no theoretical connection to SLM
- **JPS 2022 invariance**: Correctly identifies $\sigma^2$ as the invariant quantity; our framework adds GP UQ
- **Parametric models** (CEV, SABR): GP with degenerate kernel ($\sigma_f = 0$) and specific mean function

---

## 10. Experimental Results and Dead Ends

### 10.1 What Works

**Direct KRR on $(\Delta S)^2/\Delta t$ (KGEDMD-direct).** Beats the CdC generator approach:

| Model | KGEDMD-direct RMSE | Aït-Sahalia & Jacod (NW) RMSE |
|-------|-------------------|-------------------------------|
| CIR | 1.8% | 2.4% |
| CEV | 2.1% | 2.9% |
| OU | 5.1% | 7.5% |

The A&J estimator is equivalent to CdC-NW (proved identical with same kernel/bandwidth). The KGEDMD-direct method beats it because it avoids the cancellation in $L(S^2) - 2S\cdot L(S)$.

**Signature QV scaling (L1 level).** Lead-lag Lévy area scales with price level (validated):

| $\beta$ | True Status | $\hat\alpha$ | Detection rate |
|---------|------------|-------------|----------------|
| 1.5 | No bubble | $1.49\pm0.08$ | 0% |
| 2.0 | No bubble | $2.00\pm0.09$ | 0% |
| 2.5 | Bubble | $2.52\pm0.02$ | 100% |
| 3.0 | Bubble | $3.07\pm0.03$ | 100% |

R² > 0.9, perfect classification. The lead-lag Lévy area = QV: $A^{\text{LeadLag}}_{[t,t+\Delta]} = \frac{1}{2}\sum(\Delta S_i)^2 = \langle S\rangle_{[t,t+\Delta]}$. **Critical**: plain (time, price) Lévy area does NOT equal QV — must use the lead-lag transform.

### 10.2 Dead Ends

**CdC via generator.** $\Gamma(S,S) = L(S^2) - 2S\cdot L(S)$: catastrophic cancellation at moderate $S$ (§7.1). Not recommended.

**Eigenfunction growth reconstruction.** Incoherent in RBF RKHS (§7.2).

**Multi-step Koopman propagation.** Signal $O(\Delta t^2)$ smaller than noise at all horizons (§7.3).

**Eigenvalue sign test (single-step and multi-horizon).** Fundamental impossibility: discrete-time Markov chains cannot explode; regularization buries the $\lambda > 0$ signal (§7.4, §7.4.1).

**Sturm-Liouville approach.** Noisy boundary conditions produce false positives on GBM (33%) and Heston (67%) while still missing bubble DGPs. 1D-specific; cannot scale (§7.4.3).

### 10.3 Why the Dead Ends Are Dead

All eigenvalue/eigenfunction approaches fail for the same fundamental reason: the equivalent characterization $\lambda_c > 0$ requires resolving behavior near the explosion boundary (at $S \to \infty$), which is outside the data support and outside the function space spanned by bounded kernels. The $\alpha > 2$ characterization answers a strictly easier question — how does $\sigma^2(S)$ grow in the observed range? — which is both locally testable and numerically accessible.

The SABR failure ($\hat\alpha \approx 0.2$ instead of 3.0) is not an eigenvalue failure but a conditional-expectation failure: the 1D marginal $\sigma^2_\text{eff}(S)$ conflates the price-level scaling with the leverage-induced vol-price correlation. The resolution (log-linear separation, L2-SV conditional Feller) operates at the $\sigma^2$ level, not the eigenvalue level.

---

## 11. References

1. Aït-Sahalia, Y. & Jacod, J. (2009). Testing for jumps in a discretely observed process. *Annals of Statistics*, 37(1), 184–222.

2. Aït-Sahalia, Y. & Jacod, J. (2014). *High-Frequency Financial Econometrics*. Princeton University Press.

3. Arlot, S. & Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statist. Surv.*, 4, 40–79.

4. Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. *Journal of Financial Econometrics*, 2(1), 1–37.

5. Barndorff-Nielsen, O. E. & Shephard, N. (2006). Econometrics of testing for jumps in financial economics using bipower variation. *Journal of Financial Econometrics*, 4(1), 1–30.

6. Bates, D. S. (1996). Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche Mark options. *Review of Financial Studies*, 9(1), 69–107.

7. Berlinet, A. & Thomas-Agnan, C. (2004). *Reproducing Kernel Hilbert Spaces in Probability and Statistics*. Kluwer.

8. Bingham, N. H., Goldie, C. M. & Teugels, J. L. (1987). *Regular Variation*. Cambridge University Press.

9. Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.

10. Dandapani, A. & Protter, P. (2019). Strict local martingales via filtration enlargement. *Stochastic Processes and their Applications*, 129(7), 2519–2539.

11. Delbaen, F. & Shirakawa, H. (2002). No arbitrage condition for positive diffusion price processes. *Asia-Pacific Financial Markets*, 9(3–4), 159–168.

12. de Haan, L. & Ferreira, A. (2006). *Extreme Value Theory: An Introduction*. Springer.

13. Donsker, M. D. & Varadhan, S. R. S. (1975). Asymptotic evaluation of certain Markov process expectations for large time. *Comm. Pure Appl. Math.*, 28, 1–47.

14. Ekström, E. & Tysk, J. (2009). Bubbles, convexity and the Black-Scholes equation. *Ann. Appl. Probab.*, 19(4), 1369–1384.

15. Engelbert, H. J. & Schmidt, W. (1991). Strong Markov continuous local martingales and solutions of one-dimensional stochastic differential equations. *Mathematische Nachrichten*, 143(1), 167–184.

16. Engländer, J. & Pinsky, R. G. (1999). On the construction and support properties of measure-valued diffusions on $D \subseteq \mathbb{R}^d$ with spatially dependent branching. *Ann. Probab.*, 27(2), 684–730.

17. Ethier, S. N. & Kurtz, T. G. (1986). *Markov Processes: Characterization and Convergence*. Wiley. Theorem 4.5.4.

18. Feller, W. (1952). The parabolic differential equations and the associated semi-groups of transformations. *Annals of Mathematics*, 55(3), 468–519.

19. Florens-Zmirou, D. (1993). On estimating the diffusion coefficient from discrete observations. *Journal of Applied Probability*, 30(4), 790–804.

20. Hansen, L. P. & Scheinkman, J. A. (2009). Long-term risk: An operator approach. *Econometrica*, 77(1), 177–234.

21. Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. *Annals of Statistics*, 3(5), 1163–1174.

22. Jarrow, R., Protter, P. & Shimbo, K. (2010). Asset price bubbles in incomplete markets. *Mathematical Finance*, 20(2), 145–185.

23. Jarrow, R., Protter, P. & San Martín, F. (2022). Bubble invariance. Working paper.

24. Kanagawa, M., Hennig, P., Sejdinovic, D. & Sriperumbudur, B. K. (2018). Gaussian processes and kernel methods: A review on connections and differences. *arXiv:1807.02582*.

25. Karlin, S. & Taylor, H. M. (1981). *A Second Course in Stochastic Processes*. Academic Press. Chapter 15.

26. Khasminskii, R. Z. (1960). Ergodic properties of recurrent diffusion processes. *Theory Probab. Appl.*, 5(2), 179–196.

27. Khasminskii, R. Z. (2012). *Stochastic Stability of Differential Equations*, 2nd ed. Springer. Theorems 3.5, Ch. 4.

28. Klus, S., Nüske, F., Peitz, S., Niemann, J.-H., Clementi, C. & Schütte, C. (2020). Data-driven approximation of the Koopman generator. *Multiscale Model. Simul.*, 18(4), 1532–1959.

29. Kou, S. G. (2002). A jump-diffusion model for option pricing. *Management Science*, 48(8), 1086–1101.

30. Kraft, H. (2005). Optimal portfolios and Heston's stochastic volatility model. *Quantitative Finance*, 5(3), 303–313.

31. Mancini, C. (2009). Non-parametric threshold estimation for models with stochastic diffusion coefficient and jumps. *Scandinavian Journal of Statistics*, 36(2), 270–296.

32. Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.

33. Mijatović, A. & Urusov, M. (2012). On the martingale property of certain local martingales. *Probability Theory and Related Fields*, 152(1–2), 1–30.

34. Pinsky, R. G. (1995). *Positive Harmonic Functions and Diffusion*. Cambridge University Press. Chapter 4.

35. Protter, P. (2013). A mathematical theory of financial bubbles. In *Paris-Princeton Lectures on Mathematical Finance 2013*. Springer, 1–108.

36. Qin, L. & Linetsky, V. (2015). Positive eigenfunctions of Markovian pricing operators: Hansen-Scheinkman factorization, Ross Recovery and long-term pricing. *Operations Research*, 64(1), 99–117.

37. Quiñonero-Candela, J. & Rasmussen, C. E. (2005). A unifying view of sparse approximate Gaussian process regression. *JMLR*, 6, 1939–1959.

38. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. §2.7, §6.2.

39. Resnick, S. I. (2007). *Heavy-Tail Phenomena: Probabilistic and Statistical Modeling*. Springer. Chapter 6.

40. Steinwart, I. & Christmann, A. (2008). *Support Vector Machines*. Springer. Theorem 6.23.

41. Stroock, D. W. & Varadhan, S. R. S. (1979). *Multidimensional Diffusion Processes*. Springer. Chapter 10.

42. Todorov, V. & Tauchen, G. (2011). Volatility jumps. *Journal of Business & Economic Statistics*, 29(3), 356–371.
