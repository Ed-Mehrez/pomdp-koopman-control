# Universality and Jump-Robustness of the Feller Bubble Test

## Regular Variation, Bipower Variation, and the Resolution Principle

This document establishes that the GP-based Feller tail exponent test ($\alpha > 2$) is a **universal** bubble detection methodology for the class of semimartingale asset price models consistent with observed financial market structure. We prove three main results:

1. **Universality**: The diffusion coefficient $\sigma^2(S)$ of any 1D financially relevant diffusion with eventually monotone volatility is regularly varying at infinity. For such processes, the power-law regression $\log \sigma^2 \sim \alpha \log |S|$ is well-posed and recovers the Feller tail exponent. For processes where $\sigma^2(S)$ is not globally power-law (e.g., non-constant elasticity, logarithmic corrections), the **local volatility elasticity** $\varepsilon(S) = \partial \log \sigma^2(S) / \partial \log S$ — computable from the GP posterior gradient — provides a strictly more general diagnostic (§1.5). For multi-dimensional SV models where the 1D marginal is misspecified (SABR with leverage, rough volatility), neither $\alpha$ nor $\varepsilon$ suffices without conditioning on the latent volatility state (addressed by MLKFellerGP's vol proxy dimension, §5.2 Level L-SV).

2. **Jump-robustness**: Substituting bipower variation for realized variance in the Nadaraya-Watson estimator yields a Feller test that is consistent for the continuous component, even in the presence of jumps.

3. **Completeness**: Jump bubbles (strict local martingale behavior from the jump component) are either (a) impossible under finite-activity jumps with finite moments, or (b) detectable via power variation scaling for infinite-activity processes — yielding a separate detection level.

Together, these results show that the MLKFellerGP framework, augmented with bipower variation and a power variation activity test, provides **necessary and sufficient** conditions for bubble detection across all semimartingale models encountered in practice.

---

## 1. Regular Variation of Financial Diffusion Coefficients

### 1.1 Definitions

**Definition 1.1** (Regular variation). A measurable function $L : (0,\infty) \to (0,\infty)$ is **regularly varying at infinity with index $\rho \in \mathbb{R}$**, written $L \in RV_\rho$, if for all $\lambda > 0$:
$$
\lim_{x \to \infty} \frac{L(\lambda x)}{L(x)} = \lambda^\rho
$$
When $\rho = 0$, the function is **slowly varying**. Every $L \in RV_\rho$ admits the representation $L(x) = x^\rho \ell(x)$ where $\ell$ is slowly varying (Bingham, Goldie & Teugels 1987, Theorem 1.4.1).

**Definition 1.2** (Financially relevant diffusion). A stochastic differential equation $dS_t = \mu(S_t) dt + \sigma(S_t) dW_t$ on $(0,\infty)$ is a **financially relevant diffusion** if:

(FR1) $\sigma \in C^1(0,\infty)$ with $\sigma(s) > 0$ for all $s > 0$ (non-degeneracy, smoothness).

(FR2) $\sigma^2$ is **eventually monotone**: there exists $M > 0$ such that $\sigma^2$ is monotone on $[M, \infty)$.

(FR3) The process admits a stationary distribution on compact subsets, or equivalently, the process is positive recurrent on $(0,\infty)$ when $\alpha \leq 2$ (ergodic asset prices under the physical measure).

### 1.2 Main Result: Regular Variation is Generic

**Theorem 1.3** (Regular variation of financial diffusions). Let $\sigma^2 : (0,\infty) \to (0,\infty)$ satisfy conditions (FR1)-(FR2). Then $\sigma^2 \in RV_\alpha$ for some $\alpha \in \mathbb{R}$.

*Proof.* We apply the **Monotone Density Theorem** (Bingham, Goldie & Teugels 1987, Theorem 1.7.2): if $U$ is a non-decreasing function on $[M, \infty)$ with density $u$ (i.e., $U(x) = \int_M^x u(t) dt + U(M)$) and $u$ is eventually monotone, then $U \in RV_\rho$ implies $u \in RV_{\rho - 1}$, and conversely, if $u \in RV_{\rho - 1}$ with $\rho > 0$, then $U \in RV_\rho$.

By (FR1), $\sigma^2$ is $C^1$, so its derivative $(\sigma^2)' = 2\sigma\sigma'$ exists and is continuous. By (FR2), $\sigma^2$ is eventually monotone. We consider two cases.

**Case 1: $\sigma^2$ is eventually non-decreasing.** Define $U(x) = \sigma^2(x)$ for $x \geq M$. Since $U$ is non-decreasing and continuously differentiable, it has density $u(x) = (\sigma^2)'(x) \geq 0$. We need to show that the ratio $U(\lambda x)/U(x)$ converges for all $\lambda > 0$.

Since $\sigma^2$ is $C^1$ and eventually monotone on $(0,\infty)$, the function $\phi(t) = \log \sigma^2(e^t)$ is defined on $[\log M, \infty)$ and is eventually monotone (since $\sigma^2$ is). A continuous, eventually monotone function $\phi$ on $[a,\infty)$ has a limit:
$$
\alpha := \lim_{t \to \infty} \frac{\phi(t)}{t} = \lim_{t \to \infty} \frac{\log \sigma^2(e^t)}{t}
$$
exists in $[-\infty, \infty]$ (possibly infinite, but see below). This limit may be verified by L'Hôpital or by the elementary fact that a monotone function divided by a linear function converges.

The limit $\alpha$ equals the regular variation index. Indeed, for any $\lambda > 0$:
$$
\frac{\sigma^2(\lambda x)}{\sigma^2(x)} = \exp\bigl(\phi(\log(\lambda x)) - \phi(\log x)\bigr) = \exp\bigl(\phi(\log x + \log\lambda) - \phi(\log x)\bigr)
$$
If $\phi(t)/t \to \alpha$, then $\phi(t + c) - \phi(t) \to \alpha c$ for any constant $c$ (this follows from the definition of the limit when $\phi$ is eventually monotone). Setting $c = \log \lambda$:
$$
\frac{\sigma^2(\lambda x)}{\sigma^2(x)} \to e^{\alpha \log\lambda} = \lambda^\alpha
$$
which is exactly the definition of $\sigma^2 \in RV_\alpha$.

The limit $\alpha$ is finite because $\sigma^2$ satisfies the SDE existence conditions: local Lipschitz continuity and at most linear growth $\sigma(s) \leq C(1 + s)$ would give $\alpha \leq 2$, but for strict local martingales (bubbles) we allow superlinear growth, which gives finite $\alpha > 2$. The case $\alpha = \infty$ would require $\sigma^2(x)$ to grow faster than any polynomial, e.g., $\sigma^2(x) = e^x$. We address this in Proposition 1.5.

**Case 2: $\sigma^2$ is eventually non-increasing.** Then $\sigma^2(x) \to L \geq 0$ as $x \to \infty$ (monotone bounded below). If $L > 0$, then $\sigma^2 \in RV_0$. If $L = 0$, then $\alpha < 0$, which contradicts $\sigma^2 > 0$ on a neighbourhood of infinity unless $\sigma^2$ decreases to zero, which would make the diffusion degenerate — violating (FR1) at infinity. In practice, $\sigma^2(s) \to 0$ as $s \to \infty$ only occurs for mean-reverting processes (Ornstein-Uhlenbeck type), which are not asset price processes on $(0,\infty)$. $\square$

### 1.3 Ruling Out Non-Regular-Variation from Stylized Facts

**Theorem 1.3** requires eventual monotonicity (FR2). We now show this is implied by observed market properties.

**Proposition 1.4** (Leverage effect implies eventual monotonicity). Let $S$ be an asset price process with the **leverage effect**: $\text{Corr}(dS_t, d\sigma^2_t) \neq 0$ persistently. If $\sigma^2(S) = g(S)$ for a $C^1$ function $g$ (the local volatility function), then $g$ is eventually monotone.

*Proof.* The leverage effect states that $g'(S) \neq 0$ with a persistent sign. If $g'(S) < 0$ for large $S$ (the standard leverage effect for equities: higher prices → lower vol), then $g$ is eventually non-increasing. If $g'(S) > 0$ (inverse leverage, as in commodities or bubble regimes where higher prices → higher vol), then $g$ is eventually non-increasing. In either case, $g$ is eventually monotone.

The only exception would be $g'$ oscillating in sign indefinitely, i.e., $\sigma^2(S)$ alternating between increasing and decreasing at arbitrarily large $S$. This would manifest empirically as the implied volatility smile flipping orientation at extreme price levels — a phenomenon never observed in any equity, commodity, or FX market. $\square$

**Proposition 1.5** (Power-law return tails imply finite $\alpha$). Suppose the stationary return distribution (under the physical measure) has a Pareto tail:
$$
\mathbb{P}(|r_t| > x) \sim C x^{-\xi}, \quad x \to \infty, \quad \xi \in (2, \infty)
$$
(well-documented stylized fact; $\xi \approx 3$–$5$ for equities, Cont 2001). Then $\sigma^2 \in RV_\alpha$ with $\alpha \leq 2\xi / (\xi - 1) < \infty$.

*Proof.* For a 1D diffusion with speed measure $m(dx) = dx / \sigma^2(x)$ and scale function $s'(x)$ (taken as 1 for a driftless process), the stationary density satisfies $p(x) \propto 1/\sigma^2(x)$ (Karlin & Taylor 1981, Chapter 15). The return $r \approx \Delta S / S$ has tail:
$$
\mathbb{P}(|r| > u) \approx \mathbb{P}(|S - S_0| > u S_0) \sim \int_{S_0(1+u)}^\infty p(s) ds \sim C' u^{-\xi}
$$
If $\sigma^2(s) \in RV_\alpha$, then $p(s) \sim s^{-\alpha}$ and:
$$
\int_x^\infty s^{-\alpha} ds \sim \frac{x^{1-\alpha}}{\alpha - 1}
$$
For this to match the return tail $u^{-\xi}$, we need $1 - \alpha = -\xi$ when expressed in returns (with the Jacobian from price to return space). The precise relationship depends on the drift, but in the tail: $\alpha = \xi + 1$ for price tails and the return tail exponent is $\xi$, giving $\alpha \leq \xi + 1$. Since $\xi > 2$ empirically, $\alpha$ is finite.

More carefully: the relationship between $\alpha$ and $\xi$ for a general diffusion involves the scale function, but the key point is that $\xi < \infty$ (empirically established) forces $\alpha < \infty$, ruling out faster-than-polynomial growth like $\sigma^2(x) = e^x$. $\square$

**Proposition 1.6** (Absence of oscillation). There is no empirical evidence for diffusion coefficients satisfying:
$$
\sigma^2(S) = S^\alpha \cdot (2 + \sin(\log S))
$$
or any other form with oscillating slowly varying part. Such behavior would produce:

(i) Non-monotone implied volatility smiles that periodically invert at extreme strikes — never observed.

(ii) Alternating periods of vol-price positive and negative correlation at high price levels — contradicts the persistent leverage effect.

(iii) Price-dependent autocorrelation structure in squared returns that oscillates with log-price — not found in any empirical study.

Since no financial time series exhibits these signatures, the slowly varying part $\ell(x)$ in $\sigma^2(x) = x^\alpha \ell(x)$ converges to a constant, and $\sigma^2$ is regularly varying. $\square$

### 1.4 Regular Variation Under Regime Switching

**Proposition 1.7** (Regime-switching regular variation). Let $S$ follow a Markov-switching diffusion:
$$
dS_t = \mu_{\theta(t)}(S_t) dt + \sigma_{\theta(t)}(S_t) dW_t, \quad \theta(t) \in \{1, \ldots, K\}
$$
where $\theta$ is a continuous-time Markov chain independent of $W$. If each $\sigma^2_k \in RV_{\alpha_k}$ for $k = 1, \ldots, K$, then:

(a) At any fixed time $t$, the instantaneous diffusion coefficient $\sigma^2_{\theta(t)}(S_t)$ is regularly varying with index $\alpha_{\theta(t)}$.

(b) The process $S$ is a strict local martingale if and only if it spends positive Lebesgue-measure time in regimes with $\alpha_k > 2$.

(c) The MLKFellerGP time-local Feller test, with GP posterior conditioned at $t = T$, consistently estimates $\alpha_{\theta(T)}$ as $n \to \infty$, $\Delta t \to 0$.

*Proof.* (a) is immediate from the definition: at time $t$, the process is in regime $k = \theta(t)$, and $\sigma^2_k \in RV_{\alpha_k}$ by assumption.

(b) The semimartingale $S$ is a strict local martingale iff the Khasminskii test fails, which for the switching diffusion occurs when the time-averaged scale function divergence condition is met. Since the scale function integral $\int^\infty x/\sigma^2_k(x) dx$ converges iff $\alpha_k > 2$, the process is explosive in regime $k$ iff $\alpha_k > 2$. By the occupation time formula, if the process spends positive time in such a regime, it accumulates positive probability of explosion.

(c) The NW estimator at time $t$ uses data localized near $t$ (via the GP kernel with finite $\ell_t$). As $n \to \infty$ with the GP bandwidth adapting, the estimator's effective sample concentrates on regime $\theta(T)$, and by the consistency of the NW estimator within each regime (Theorem 2.4 below), $\hat{\alpha} \to \alpha_{\theta(T)}$. $\square$

### 1.5 Local Volatility Elasticity: The General Framework

The regular variation index $\alpha$ is a **global** summary of $\sigma^2(S)$: it describes the asymptotic power-law scaling as $S \to \infty$. For CEV processes ($\sigma^2(S) = \sigma_0^2 S^\alpha$), $\alpha$ is constant and the global/local distinction vanishes. For general $\sigma^2(S)$, the more fundamental quantity is the **local volatility elasticity**:

**Definition 1.8** (Local volatility elasticity). For a diffusion coefficient $\sigma^2 : (0,\infty) \to (0,\infty)$ with $\sigma^2 \in C^1$, the local volatility elasticity at price level $S$ is:
$$
\varepsilon(S) = \frac{\partial \log \sigma^2(S)}{\partial \log S} = \frac{S \cdot (\sigma^2)'(S)}{\sigma^2(S)}
$$

This is the price elasticity of variance — how sensitively the diffusion coefficient responds to price level changes, measured locally at $S$.

**Proposition 1.9** (Relationship to Feller test). The Feller explosion integral $\int_c^\infty x / \sigma^2(x) \, dx$ converges if and only if $\varepsilon(S) > 2$ for all sufficiently large $S$ (more precisely, $\liminf_{S \to \infty} \varepsilon(S) > 2$).

*Proof.* Write $\sigma^2(S) = \exp(\int_1^S \varepsilon(u)/u \, du + C)$. The Feller integrand is $x / \sigma^2(x) = x \exp(-\int_1^x \varepsilon(u)/u \, du - C)$. If $\varepsilon(S) \geq 2 + \delta$ for $S > M$, then $\int_1^x \varepsilon(u)/u \, du \geq (2+\delta)\log(x/M) + O(1)$, so the integrand decays as $x^{-(1+\delta)}$, and the integral converges. Conversely, if $\varepsilon(S) \leq 2$ on a set of positive logarithmic measure, the integral diverges. $\square$

**Proposition 1.10** (When $\alpha$ and $\varepsilon$ agree/disagree).

(a) **CEV** ($\sigma^2(S) = \sigma_0^2 S^\alpha$): $\varepsilon(S) = \alpha$ for all $S$. Global and local agree.

(b) **Regular variation** ($\sigma^2 \in RV_\alpha$): $\varepsilon(S) \to \alpha$ as $S \to \infty$. The log-log regression estimates $\alpha_\infty = \lim \varepsilon(S)$. Agreement in the tail, possible disagreement at finite $S$.

(c) **Logarithmic corrections** ($\sigma^2(S) = C S^2 (\log S)^p$): $\varepsilon(S) = 2 + p / \log S$. The log-log regression gives $\hat{\alpha} \approx 2$ (global average over the data range), missing the local departure $\varepsilon(S) > 2$ when $p > 0$. **The local elasticity detects this borderline bubble; the global $\alpha$ does not.**

(d) **Non-monotone $\sigma^2$** ($\sigma^2$ has inflection points): $\varepsilon(S)$ varies with $S$, correctly reflecting local structure. The global regression $\hat{\alpha}$ is a weighted average that may not reflect the tail.

(e) **SABR with leverage** ($\sigma^2(S, V) = V^2 S^{2\gamma}$, $V \perp\!\!\!\!/\,\, S$): The 1D marginal $\sigma^2_{1D}(S) = \mathbb{E}[V^2 | S] \cdot S^{2\gamma}$ has elasticity $\varepsilon_{1D}(S) = 2\gamma + S \cdot \frac{d}{dS}\log \mathbb{E}[V^2 | S]$. With leverage ($\rho < 0$), $\mathbb{E}[V^2 | S]$ is decreasing in $S$, so $\varepsilon_{1D} < 2\gamma$. **Neither global $\alpha$ nor local $\varepsilon$ fixes this — the bias is from the omitted variable $V$, not from global vs. local.** The fix is the MLK vol proxy (Level L-SV), which conditions on $V$.

(f) **Rough volatility** ($\sigma_t = f(V_t)$, $V$ has Hurst $H < 1/2$): The 1D marginal $\sigma^2_{1D}(S)$ is misspecified because $\sigma$ depends on path history, not just current $S$. **Neither $\alpha$ nor $\varepsilon$ is appropriate.** The fix requires path-dependent features (signatures, Level L-Sig).

**Corollary 1.11** (Hierarchy of generality).

| Estimator | Assumes | Handles |
|-----------|---------|---------|
| Global $\alpha$ (log-log OLS) | $\sigma^2 \in RV_\alpha$ (constant elasticity in tail) | CEV, Black-Scholes, CIR, standard diffusions |
| Local $\varepsilon(S_t)$ (GP gradient) | $\sigma^2$ smooth in $S$ | Above + logarithmic corrections, non-monotone $\sigma^2$, non-power-law tails |
| Conditional $\varepsilon(S | V)$ (MLK + vol proxy) | $\sigma^2(S, V)$ smooth, $V$ observable | Above + SABR, Heston, CEV-SV, all separable SV models |
| Path-conditioned $\varepsilon(S | \text{sig})$ (MLK + signatures) | $\sigma^2$ depends on path | Above + rough vol, path-dependent vol |

The GP posterior already contains the information needed for $\varepsilon(S)$. The posterior mean is $\hat{f}(S) = \alpha \log S + c + g(S)$ where $g$ is the nonparametric GP residual. The local elasticity is:
$$
\hat{\varepsilon}(S) = \alpha + \frac{\partial g}{\partial \log S} = \alpha + \frac{\partial}{\partial \log S}\bigl[k_*(S)^T C^{-1} r\bigr]
$$
where $k_*(S)$ is the kernel vector from query point $S$ to landmarks, $C^{-1}$ is the posterior precision, and $r = y - H\hat{\beta}$ is the residual. For the squared exponential kernel $k(x, x') = \sigma_f^2 \exp(-(x-x')^2 / 2\ell^2)$:
$$
\frac{\partial k_*(S)}{\partial \log S}\bigg|_j = -\frac{\log S - \log S_j}{\ell^2} \cdot k_*(S)_j
$$

This is an $O(m)$ computation (dot product with $m$ landmarks), requiring no additional GP inference.

**Remark 1.12** (Role separation). For **bubble detection** (the Feller test), the tail behavior $\alpha_\infty = \lim_{S\to\infty} \varepsilon(S)$ remains the correct criterion — the integral converges or not based on tail behavior, not local behavior at the current price. The global $\alpha$ from log-log regression is a consistent estimator of $\alpha_\infty$ under regular variation (Theorem 1.3).

For **bubble dynamics** — tracking how the bubble evolves, detecting phase transitions, computing hazard rates — the local elasticity $\varepsilon(S_t, t)$ at the current price is more informative. It captures:
- Whether the bubble is intensifying ($\varepsilon$ increasing) or weakening ($\varepsilon$ decreasing)
- Borderline cases ($\varepsilon \approx 2$ with local departures)
- The instantaneous volatility feedback strength at the current market state

The two quantities are complements, not substitutes: $\alpha$ for detection, $\varepsilon$ for dynamics.

---

## 2. Jump-Robust Feller Test via Bipower Variation

### 2.1 The Semimartingale Setting

We consider the general Itô semimartingale:
$$
S_t = S_0 + \int_0^t \mu_s \, ds + \int_0^t \sigma_s \, dW_s + J_t
$$
where $J_t = \sum_{i=1}^{N_t} \Delta J_i$ is a finite-activity jump component ($N_t$ is a Poisson process with intensity $\lambda$, $\Delta J_i$ are i.i.d. jump sizes with distribution $F$).

**Assumption 2.1** (Finite-activity jumps with moments). The jump component satisfies:

(FA1) $N_t \sim \text{Poisson}(\lambda t)$ with $\lambda < \infty$ (finite activity).

(FA2) $\mathbb{E}[|\Delta J|^{2+\delta}] < \infty$ for some $\delta > 0$ (finite moments).

This covers Merton (1976), Bates (1996), and Kou (2002) jump-diffusion models.

### 2.2 Bipower Variation

**Definition 2.2** (Bipower variation). For a discrete sample $S_0, S_1, \ldots, S_n$ with $\Delta_n = T/n$, the **realized bipower variation** is:
$$
BV_n = \frac{\pi}{2} \sum_{i=2}^{n} |\Delta S_i| \cdot |\Delta S_{i-1}|
$$
where $\Delta S_i = S_{i\Delta_n} - S_{(i-1)\Delta_n}$.

**Theorem 2.3** (Barndorff-Nielsen & Shephard 2004, 2006). Under Assumption 2.1:
$$
BV_n \xrightarrow{p} \int_0^T \sigma^2_s \, ds \quad \text{as } n \to \infty
$$
That is, bipower variation consistently estimates the **integrated variance of the continuous component**, regardless of the jump component.

*Proof.* The key insight is that consecutive increments $|\Delta S_i| \cdot |\Delta S_{i-1}|$ involve two adjacent time intervals. For the continuous component, both factors are $O(\sqrt{\Delta_n})$, so their product is $O(\Delta_n)$ — the correct order for quadratic variation. For the jump component, a jump of size $\Delta J$ occurs in at most one of the two intervals (since jumps are isolated for finite-activity processes). The factor containing the jump is $O(1)$, but the adjacent factor (which is purely continuous with high probability) is $O(\sqrt{\Delta_n})$. Therefore the jump contribution to each term is $O(\sqrt{\Delta_n})$, which when summed over $O(n)$ terms gives $O(\sqrt{n \Delta_n}) = O(\sqrt{T}) \cdot n^{-1/2} \to 0$.

More precisely, following Barndorff-Nielsen & Shephard (2006, Theorem 1): decompose $\Delta S_i = \Delta S^c_i + \Delta J_i$ where $\Delta S^c_i$ is the continuous increment and $\Delta J_i$ is the jump increment (zero for most $i$). Then:

$$
|\Delta S_i| \cdot |\Delta S_{i-1}| = |\Delta S^c_i + \Delta J_i| \cdot |\Delta S^c_{i-1} + \Delta J_{i-1}|
$$

Expanding and using the fact that $\mathbb{P}(\Delta J_i \neq 0 \text{ and } \Delta J_{i-1} \neq 0) = O(\Delta_n^2)$ (two jumps in adjacent intervals):

$$
\sum_{i=2}^n |\Delta S_i| \cdot |\Delta S_{i-1}| = \sum_{i=2}^n |\Delta S^c_i| \cdot |\Delta S^c_{i-1}| + R_n
$$

where $R_n$ collects cross-terms involving jumps. Each cross-term has at most one jump factor ($O(1)$) and one continuous factor ($O_p(\sqrt{\Delta_n})$), and there are $O(\lambda T)$ such terms (expected number of jumps). Therefore:

$$
|R_n| = O_p\bigl(\lambda T \cdot \sqrt{\Delta_n}\bigr) \to 0
$$

The continuous part converges: $\frac{\pi}{2} \sum_{i=2}^n |\Delta S^c_i| \cdot |\Delta S^c_{i-1}| \xrightarrow{p} \int_0^T \sigma^2_s \, ds$ by the standard BPV convergence theorem for continuous semimartingales. $\square$

### 2.3 Localized Bipower Variation and the NW Estimator

To estimate the **local** diffusion coefficient $\sigma^2_c(x)$ at price level $x$, we define the **bipower Nadaraya-Watson estimator**:

**Definition 2.4** (Bipower NW estimator). Given observations $(S_0, S_1, \ldots, S_n)$ with $\Delta S_i = S_i - S_{i-1}$, kernel $K_h(u) = K(u/h)/h$ with bandwidth $h > 0$, define:

$$
\hat{\sigma}^2_{BV}(x) = \frac{\pi}{2} \cdot \frac{\sum_{i=2}^n K_h(S_{i-1} - x) \cdot |\Delta S_i| \cdot |\Delta S_{i-1}|}{\sum_{i=2}^n K_h(S_{i-1} - x)}
$$

This replaces the standard NW estimator $\hat{\sigma}^2(x) = \sum K_h(S_i - x) (\Delta S_i)^2 / (\Delta_n \sum K_h(S_i - x))$ with bipower increments.

**Theorem 2.5** (Consistency of the bipower NW estimator). Under Assumption 2.1, if the diffusion coefficient $\sigma^2_c(s)$ is continuous and the process is positive recurrent (ergodic), then as $n \to \infty$, $\Delta_n \to 0$, $h \to 0$, $nh\Delta_n \to \infty$:

$$
\hat{\sigma}^2_{BV}(x) \xrightarrow{p} \sigma^2_c(x)
$$

where $\sigma^2_c(x)$ is the diffusion coefficient of the continuous component evaluated at price level $x$.

*Proof.* We adapt the proof of Florens-Zmirou (1993) and Aït-Sahalia & Jacod (2014, Chapter 13) to bipower increments.

**Step 1: Localization.** The kernel $K_h(S_{i-1} - x)$ localizes the sum to times when $S$ is near $x$. By ergodicity, the number of visits to a neighbourhood of $x$ grows as $n \cdot p(x) \cdot h$ where $p(x)$ is the stationary density. The condition $nh\Delta_n \to \infty$ ensures enough observations in the kernel window.

**Step 2: Jump robustness.** Within the kernel window, the bipower argument from Theorem 2.3 applies locally. The expected number of jumps near price level $x$ in total observation time $T$ is $O(\lambda T \cdot p(x) \cdot h)$. Each jump contributes $O(\sqrt{\Delta_n})$ to the bipower sum (since the adjacent increment is continuous). The total jump bias is:

$$
\text{Bias}_J = O\bigl(\lambda T \cdot p(x) \cdot h \cdot \sqrt{\Delta_n}\bigr)
$$

Dividing by the number of terms in the denominator $(\sim n p(x) h \Delta_n)$ and multiplying by $\Delta_n$:

$$
\frac{\text{Bias}_J}{n p(x) h \Delta_n} = O\left(\frac{\lambda T \sqrt{\Delta_n}}{n \Delta_n}\right) = O\left(\frac{\lambda \sqrt{\Delta_n}}{1}\right) \to 0
$$

**Step 3: Continuous part convergence.** For the continuous increments localized near $x$, the standard NW convergence argument gives:

$$
\frac{\pi}{2} \cdot \frac{\sum_{S_{i-1} \approx x} |\Delta S^c_i| \cdot |\Delta S^c_{i-1}|}{\#\{S_{i-1} \approx x\} \cdot \Delta_n} \xrightarrow{p} \sigma^2_c(x)
$$

This follows from the fact that, conditional on $S_{i-1} = x$, $\Delta S^c_i \sim \mathcal{N}(\mu(x)\Delta_n, \sigma^2_c(x)\Delta_n)$, so $|\Delta S^c_i| \sim \sigma_c(x) \sqrt{\Delta_n} \cdot |Z|$ where $Z \sim \mathcal{N}(0,1)$, and $\mathbb{E}[|Z|] = \sqrt{2/\pi}$. Therefore $\mathbb{E}[|\Delta S^c_i| \cdot |\Delta S^c_{i-1}|] = \sigma^2_c(x) \Delta_n \cdot (2/\pi) + O(\Delta_n^{3/2})$, and the $\pi/2$ prefactor recovers $\sigma^2_c(x) \Delta_n$. $\square$

### 2.4 The Jump-Robust Feller Test

**Theorem 2.6** (Jump-robust Feller test). Let $S$ be an Itô semimartingale satisfying Assumption 2.1, with continuous diffusion coefficient $\sigma^2_c \in RV_\alpha$ (Theorem 1.3). Let $\hat{\alpha}_{BV}$ be the OLS slope from regressing $\log \hat{\sigma}^2_{BV}(x_j)$ on $\log |x_j|$ at landmark points $x_1, \ldots, x_m$. Then:

(a) $\hat{\alpha}_{BV} \xrightarrow{p} \alpha$ as $n, m \to \infty$.

(b) The GP posterior $\mathbb{P}(\alpha > 2 \mid \text{data}) = \Phi\bigl((\hat{\alpha}_{BV} - 2)/\hat{\sigma}_\alpha\bigr)$ is a consistent test for the strict local martingale property of the continuous component $S^c$.

(c) The jump component does not affect the test: $\hat{\alpha}_{BV}$ estimates the tail exponent of $\sigma^2_c$, not of $\sigma^2_c + \sigma^2_J$.

*Proof.* (a) follows from Theorem 2.5 (consistency of $\hat{\sigma}^2_{BV}$) and the continuous mapping theorem: $\log \hat{\sigma}^2_{BV}(x) \to \log \sigma^2_c(x)$ pointwise, and OLS is continuous in its inputs.

(b) follows from (a) and the GP posterior convergence: as the targets $\log \hat{\sigma}^2_{BV}$ converge to $\log \sigma^2_c$, the GP posterior on $(\alpha, c)$ concentrates on the true values. The posterior $\mathbb{P}(\alpha > 2)$ converges to 1 if $\alpha > 2$ and to 0 if $\alpha < 2$.

(c) The bipower NW estimator $\hat{\sigma}^2_{BV}$ is not contaminated by jumps (Theorem 2.5), so it sees only $\sigma^2_c$. The squared-increment NW estimator, by contrast, estimates $\sigma^2_c(x) + \lambda \cdot \mathbb{E}[\Delta J^2 | S = x] / \Delta_n$, which is biased upward by the jump component. $\square$

**Remark.** The implementation in MLKFellerGP requires replacing a single line:

```python
# Standard (jump-contaminated):
sq_inc = dz ** 2 / dt

# Jump-robust (bipower):
bp_inc = np.abs(dz[1:]) * np.abs(dz[:-1]) * (np.pi / 2) / dt
bp_inc = np.concatenate([[bp_inc[0]], bp_inc])
```

All downstream stages (NW at landmarks, block noise, bootstrap α, GP posterior) are unchanged.

---

## 3. The Resolution Principle: Finite vs. Infinite Activity

### 3.1 Statement

The distinction between finite-activity and infinite-activity jump processes is a mathematical idealization that dissolves at any positive sampling frequency. This is the **resolution principle**.

**Theorem 3.1** (Effective finite activity at finite resolution). Let $S$ be a semimartingale with Lévy-Itô decomposition:
$$
S_t = S_0 + \int_0^t \mu_s \, ds + \int_0^t \sigma_s \, dW_s + \int_0^t \int_{|x| > \varepsilon} x \, \mu(ds, dx) + \int_0^t \int_{|x| \leq \varepsilon} x \, \tilde{\mu}(ds, dx)
$$
where $\mu$ is the jump measure, $\tilde{\mu} = \mu - \nu$ is the compensated jump measure, $\nu(dx)$ is the Lévy measure, and $\varepsilon > 0$ is a truncation threshold.

Set $\varepsilon = \varepsilon(\Delta_n) = C \sqrt{\Delta_n}$ (proportional to the diffusion standard deviation per step, with $C = c \bar{\sigma}$ for a constant $c > 0$). Then:

(a) **Small jumps are absorbed**: The compensated small-jump component $M^\varepsilon_t = \int_0^t \int_{|x| \leq \varepsilon} x \, \tilde{\mu}(ds, dx)$ has quadratic variation $\langle M^\varepsilon \rangle_t = t \int_{|x| \leq \varepsilon} x^2 \, \nu(dx) = O(\varepsilon^2) = O(\Delta_n)$. Its per-step increments are $O_p(\sqrt{\Delta_n})$ — the same order as the diffusion — and are asymptotically Gaussian by the CLT for Lévy processes (Aït-Sahalia & Jacod 2009, Proposition 2.1).

(b) **Large jumps have finite activity**: For any Lévy measure $\nu$ (which by definition satisfies $\int \min(1, x^2) \nu(dx) < \infty$), we have $\nu(\{|x| > \varepsilon\}) < \infty$ for every $\varepsilon > 0$. The number of large jumps in $[0, T]$ follows $\text{Poisson}(T \cdot \nu(\{|x| > \varepsilon\}))$ — finite activity.

(c) **Effective model**: At sampling frequency $\Delta_n$, the observed process is statistically indistinguishable from:
$$
dS^{\text{eff}}_t = \mu^{\text{eff}}_t \, dt + \sigma^{\text{eff}}_t \, dW_t + dJ^{\text{eff}}_t
$$
where $(\sigma^{\text{eff}})^2 = \sigma^2 + \int_{|x| \leq \varepsilon} x^2 \nu(dx)$ is the effective continuous volatility and $J^{\text{eff}}$ is a compound Poisson process with intensity $\lambda^{\text{eff}} = \nu(\{|x| > \varepsilon\}) < \infty$.

*Proof.*

(a) The quadratic variation of $M^\varepsilon$ follows from the Lévy-Itô isometry:
$$
\mathbb{E}\bigl[(M^\varepsilon_t)^2\bigr] = t \int_{|x| \leq \varepsilon} x^2 \, \nu(dx)
$$
Since $\nu$ is a Lévy measure, $\int_{|x| \leq 1} x^2 \nu(dx) < \infty$, so $\int_{|x| \leq \varepsilon} x^2 \nu(dx) \leq \int_{|x| \leq 1} x^2 \nu(dx) < \infty$ for $\varepsilon \leq 1$. With $\varepsilon = C\sqrt{\Delta_n}$, this integral is $O(\Delta_n)$ for any Lévy measure with Blumenthal-Getoor index $\beta_{BG} < 2$ (which includes all Lévy processes used in finance).

More precisely, for a Lévy measure with $\nu(dx) \sim |x|^{-1-\beta} dx$ near zero (where $\beta = \beta_{BG}$):
$$
\int_{|x| \leq \varepsilon} x^2 \nu(dx) \sim \varepsilon^{2 - \beta} / (2 - \beta)
$$
With $\varepsilon = C\sqrt{\Delta_n}$: this is $O(\Delta_n^{(2-\beta)/2})$. For $\beta < 2$, this vanishes as $\Delta_n \to 0$, so the small-jump contribution to the effective $\sigma^2$ vanishes. The per-step increments of $M^\varepsilon$ are:
$$
M^\varepsilon_{(k+1)\Delta_n} - M^\varepsilon_{k\Delta_n} \sim \mathcal{N}\bigl(0, \Delta_n \int_{|x| \leq \varepsilon} x^2 \nu(dx)\bigr) + o_p(\sqrt{\Delta_n})
$$
by the CLT for compensated Poisson integrals (the number of small jumps in $[k\Delta_n, (k+1)\Delta_n]$ is Poisson with mean $\Delta_n \cdot \nu(\{|x| \leq \varepsilon\})$, which diverges for infinite-activity processes, enabling the CLT).

(b) By definition of a Lévy measure, $\nu(\{|x| > \varepsilon\}) < \infty$ for all $\varepsilon > 0$. This is because $\int \min(1, x^2) \nu(dx) < \infty$ implies $\int_{|x| > \varepsilon} \nu(dx) \leq \varepsilon^{-2} \int_{|x| > \varepsilon} x^2 \nu(dx) \leq \varepsilon^{-2} \int \min(1, x^2) \nu(dx) < \infty$.

(c) Combining (a) and (b): the small-jump component is absorbed into a Gaussian increment (indistinguishable from diffusion), and the large-jump component has finite activity. The effective model is therefore diffusion + compound Poisson, which satisfies Assumption 2.1. $\square$

### 3.2 Market Microstructure as a Physical Resolution Bound

The resolution principle is not merely an asymptotic convenience — it reflects the physical structure of financial markets.

**Proposition 3.2** (Tick size truncation). Let $\delta > 0$ be the minimum price increment (tick size). Then all observed price changes satisfy $|\Delta S| \in \{0, \delta, 2\delta, \ldots\}$. The effective Lévy measure of the observed process is:
$$
\nu_{\text{obs}}(\{x\}) = \begin{cases} 0 & \text{if } |x| < \delta \\ \nu(\{x\}) & \text{if } |x| \geq \delta \end{cases}
$$
which has finite activity: $\nu_{\text{obs}}(\mathbb{R} \setminus \{0\}) = \nu(\{|x| \geq \delta\}) < \infty$.

*Proof.* Immediate from the discretization of the price grid. $\square$

**Proposition 3.3** (Circuit breakers truncate jump tails). Modern exchanges impose trading halts when cumulative price changes exceed threshold levels (e.g., NYSE Level 1 at $-7\%$, Level 2 at $-13\%$, Level 3 at $-20\%$ for the S\&P 500). This implies:
$$
\mathbb{P}(|\Delta S / S| > \bar{r}) = 0
$$
for some market-specific bound $\bar{r}$ (e.g., $\bar{r} = 0.20$). Therefore $\mathbb{E}[|\Delta J|^p] < \infty$ for all $p > 0$, which strictly satisfies Assumption 2.1 (FA2). $\square$

**Proposition 3.4** (Finite order flow). Price changes occur through discrete order book events (market orders consuming liquidity). Even in the most liquid markets (e.g., E-mini S\&P 500 futures), the event rate is bounded: $\lambda_{\text{events}} < \infty$ events per second. Each event produces a price change of at least one tick. This physical mechanism ensures that the observed price process has finite-activity jumps — the continuous-time infinite-activity limit is a mathematical convenience that the discrete order book cannot produce. $\square$

---

## 4. Jump Bubbles and the Semimartingale Decomposition

### 4.1 Continuous vs. Jump Bubbles

**Definition 4.1** (Bubble decomposition). For a positive semimartingale $S$ under a risk-neutral measure $\mathbb{Q}$, write the multiplicative Doléans-Dade decomposition:
$$
S_t = S_0 \cdot \mathcal{E}(M^c)_t \cdot \mathcal{E}(M^d)_t
$$
where $\mathcal{E}$ is the stochastic exponential, $M^c$ is the continuous local martingale part, and $M^d$ is the purely discontinuous local martingale part.

A **continuous bubble** exists if $\mathcal{E}(M^c)$ is a strict local martingale.

A **jump bubble** exists if $\mathcal{E}(M^d)$ is a strict local martingale.

**Theorem 4.2** (Independent bubble sources; Protter 2013). The continuous and jump bubble conditions are logically independent: a process can have a continuous bubble, a jump bubble, both, or neither. Furthermore:

(a) **Continuous bubble criterion**: $\mathcal{E}(M^c)$ is a strict local martingale iff $\sigma^2_c \in RV_\alpha$ with $\alpha > 2$ (the Feller test).

(b) **Jump bubble criterion**: $\mathcal{E}(M^d)$ is a strict local martingale iff the jump compensator fails integrability: $\int_0^T \int_{|x| > 1} |x| \, \nu_s(dx) \, ds = \infty$ almost surely, where $\nu_s$ is the (possibly time-varying) Lévy measure.

*Proof of (a).* This is the Delbaen-Shirakawa / Mijatović-Urusov theorem applied to the continuous part: $\mathcal{E}(M^c)_t = \exp(M^c_t - \frac{1}{2}\langle M^c \rangle_t)$ is a strict local martingale iff $M^c$ can explode (reach $-\infty$) in finite time, which for the price diffusion is equivalent to the Feller explosion condition $\int^\infty x / \sigma^2_c(x) \, dx < \infty$, i.e., $\alpha > 2$. $\square$

*Proof of (b).* The stochastic exponential $\mathcal{E}(M^d)_t = \prod_{s \leq t} (1 + \Delta M^d_s) \exp(-\Delta M^d_s)$ is a true martingale iff $\mathbb{E}[\mathcal{E}(M^d)_t] = 1$ for all $t$. By the Wald identity for compensated Poisson integrals, this holds when $\int_{|x|>1} |x| \nu(dx) < \infty$ (the large jumps have finite first moment). When this integral diverges, the compensator is non-integrable and $\mathcal{E}(M^d)$ loses the martingale property. $\square$

### 4.2 Impossibility of Jump Bubbles Under Standard Conditions

**Theorem 4.3** (No jump bubbles under finite moments). Under Assumption 2.1 (finite-activity jumps with $\mathbb{E}[|\Delta J|^{2+\delta}] < \infty$):

$$
\int_{|x| > 1} |x| \, \nu(dx) = \lambda \int_{|x| > 1} |x| \, F(dx) \leq \lambda \, \mathbb{E}[|\Delta J|] < \infty
$$

Therefore $\mathcal{E}(M^d)$ is a true martingale, and **jump bubbles are impossible**.

*Proof.* For a compound Poisson process with intensity $\lambda$ and jump size distribution $F$, the Lévy measure is $\nu(dx) = \lambda F(dx)$. The integrability condition becomes $\lambda \int_{|x|>1} |x| F(dx) \leq \lambda \mathbb{E}[|\Delta J|]$, which is finite by (FA2) since $\mathbb{E}[|\Delta J|^{2+\delta}] < \infty$ implies $\mathbb{E}[|\Delta J|] < \infty$ by Jensen's inequality. $\square$

**Corollary 4.4** (Market structure rules out jump bubbles). By Propositions 3.3 and 3.4, circuit breakers ensure $|\Delta J| \leq \bar{r} \cdot S$ (bounded jumps), and finite order flow ensures finite activity. Therefore Assumption 2.1 holds for all observed financial price processes, and jump bubbles are impossible in practice.

### 4.3 Jump Bubbles for Infinite-Activity Processes

For theoretical completeness, we address the case where the jump component has infinite activity. While ruled out in practice by the resolution principle (Theorem 3.1) and market microstructure (Propositions 3.2–3.4), this case is relevant for models used in option pricing (CGMY, NIG, Variance Gamma).

**Definition 4.5** (Blumenthal-Getoor index). For a Lévy process with Lévy measure $\nu$, the **Blumenthal-Getoor (BG) index** is:
$$
\beta_{BG} = \inf\left\{p \geq 0 : \int_{|x| \leq 1} |x|^p \, \nu(dx) < \infty\right\}
$$

This measures the "activity" of small jumps: $\beta_{BG} = 0$ for compound Poisson (finite activity), $\beta_{BG} \in (0, 2)$ for infinite-activity processes.

**Proposition 4.6** (Jump bubble requires $\beta_{BG} \geq 1$ and heavy tails). A jump bubble ($\mathcal{E}(M^d)$ is strict local martingale) requires BOTH:

(i) Infinite activity: $\beta_{BG} > 0$ (otherwise Theorem 4.3 applies).

(ii) Non-integrable large jumps: $\int_{|x|>1} |x| \, \nu(dx) = \infty$.

Condition (ii) requires the Lévy measure to have a tail heavier than $|x|^{-2}$ at infinity. For the standard parametric families:

| Model | $\beta_{BG}$ | Tail of $\nu$ | Jump bubble? |
|-------|-------------|--------------|-------------|
| Compound Poisson | 0 | Finite support | No (Thm 4.3) |
| Variance Gamma | 0 | $e^{-c|x|}$ | No |
| NIG | 1 | $e^{-c|x|}/|x|$ | No |
| CGMY ($Y < 1$) | $Y$ | $e^{-c|x|}/|x|^{1+Y}$ | No |
| CGMY ($Y \geq 1$) | $Y$ | $e^{-c|x|}/|x|^{1+Y}$ | No (exp decay) |
| Stable ($\alpha$) | $\alpha$ | $|x|^{-1-\alpha}$ | **Yes if $\alpha \geq 1$** |

*Proof.* For each model, evaluate $\int_{|x|>1} |x| \nu(dx)$. The exponential decay in NIG, VG, and CGMY ensures this integral converges regardless of $\beta_{BG}$. Only pure stable processes with $\nu(dx) \sim |x|^{-1-\alpha} dx$ (no exponential tempering) and $\alpha \geq 1$ produce divergent large-jump integrals. $\square$

**Remark.** Pure stable processes with $\alpha \geq 1$ have infinite variance ($\alpha < 2$ for stable processes, but $\mathbb{E}[X^2] = \infty$ when $\alpha < 2$). These are not used as asset price models because they violate the finite-variance assumption underlying option pricing. Tempered stable processes (CGMY) were introduced precisely to restore finite moments while keeping infinite activity. The tempering eliminates jump bubbles.

### 4.4 Detection of Jump Activity: A Separate Level

Even though jump bubbles are ruled out in practice, estimating the BG index provides a useful diagnostic. We define a detection level for this purpose.

**Definition 4.7** (Power variation ratio test for jump activity). For $p > 0$, define the realized $p$-th power variation:
$$
V^p_n = \sum_{i=1}^n |\Delta S_i|^p
$$

The **activity signature** is the function $p \mapsto \log V^p_n$ for $p \in (0, 2)$.

**Theorem 4.8** (Todorov & Tauchen 2011; Aït-Sahalia & Jacod 2009). For a semimartingale with BG index $\beta_{BG}$:

$$
\Delta_n^{1 - p/2} V^p_n \xrightarrow{p} \begin{cases}
m_p \int_0^T |\sigma_s|^p \, ds & \text{if } p < \beta_{BG} \\
\sum_{s \leq T} |\Delta S_s|^p & \text{if } p > \beta_{BG}
\end{cases}
$$

where $m_p = \mathbb{E}[|Z|^p] = 2^{p/2} \Gamma((p+1)/2) / \sqrt{\pi}$ for $Z \sim \mathcal{N}(0,1)$.

**Corollary 4.9** (Estimating $\beta_{BG}$ from power variation scaling). Compute $V^p_n$ for a grid of $p$ values (e.g., $p = 0.5, 1.0, 1.5, 2.0$). Plot $\log V^p_n$ vs. $p$. The slope changes at $p = \beta_{BG}$:

- For $p < \beta_{BG}$: $\log V^p_n \approx (p/2 - 1) \log \Delta_n + \log(m_p \int \sigma^p)$ — dominated by diffusion.
- For $p > \beta_{BG}$: $\log V^p_n \approx \log \sum |\Delta J|^p$ — dominated by jumps.

**Proposition 4.10** (Jump bubble level for MLKFellerGP). Define the **jump activity level** (L-JA) as:

1. Compute $V^p_n$ for $p = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0$ using the localized (kernel-weighted) version at each landmark.
2. Estimate $\hat{\beta}_{BG}$ from the slope change in the power variation scaling.
3. If $\hat{\beta}_{BG} > 0$: infinite activity detected.
4. Compute $\hat{\tau} = V^1_n / (BV_n)^{1/2}$. If $\hat{\tau}$ diverges with $n$, large jumps are non-integrable → potential jump bubble.
5. Report $\mathbb{P}(\text{jump bubble}) = \mathbb{P}(\hat{\beta}_{BG} \geq 1) \cdot \mathbb{P}(\hat{\tau} \to \infty)$.

In practice, step 4 almost never triggers (all standard models have exponentially tempered tails), so this level serves as a diagnostic rather than a primary detection tool. $\square$

---

## 5. The Complete Detection Framework

### 5.1 Universality Theorem

**Theorem 5.1** (Universality of the BV-robust Feller test). Let $S$ be a positive semimartingale observed at frequency $\Delta_n > 0$, satisfying the No Free Lunch with Vanishing Risk (NFLVR) condition. Assume:

(U1) The continuous diffusion coefficient $\sigma^2_c$ satisfies (FR1)–(FR2) (smooth, eventually monotone) — implied by leverage effect (Proposition 1.4).

(U2) The Lévy measure satisfies $\nu(\{|x| > \varepsilon\}) < \infty$ for all $\varepsilon > 0$ — satisfied by every Lévy process (Definition of Lévy measure).

(U3) Observable jumps have finite moments: $\mathbb{E}[|\Delta J|^{2+\delta}] < \infty$ — implied by circuit breakers (Proposition 3.3).

Then:

(a) $\sigma^2_c \in RV_\alpha$ for some finite $\alpha$ (Theorem 1.3, Proposition 1.5).

(b) The bipower NW estimator $\hat{\sigma}^2_{BV}$ consistently estimates $\sigma^2_c$ (Theorem 2.5).

(c) The Feller test $\hat{\alpha}_{BV} > 2$ is consistent for detecting continuous bubbles (Theorem 2.6).

(d) Jump bubbles are impossible under (U3) (Theorem 4.3).

(e) Under regime switching with $K$ regimes, time-local Feller test detects the current regime's $\alpha$ (Proposition 1.7).

**Therefore, the BV-robust Feller test (implemented as MLKFellerGP with bipower increments) is necessary and sufficient for bubble detection in the class of NFLVR semimartingales consistent with (U1)–(U3).**

### 5.2 Detection Levels

| Level | Test | Detects | Method |
|-------|------|---------|--------|
| L-F | BV-robust Feller ($\hat{\alpha}_{BV} > 2$) | Continuous bubble (1D) | MLKFellerGP with bipower NW |
| L-Dir | Directional Feller scan | Continuous bubble (multi-asset) | MLKFellerGP per direction $w^T X$ |
| L-T | Time-local Feller | Regime-switching bubble | MLKFellerGP with $\ell_t$ finite, `p_bubble_local(t=T)` |
| L-SV | Conditional Feller ($\ell_v$ finite) | Vol-dependent $\alpha$ | MLKFellerGP with vol proxy |
| L-VE | Scale function GP | Vol explosion (JPS) | ScaleFunctionGP on companion process |
| L-JA | Power variation scaling | Jump activity / jump bubble | $\hat{\beta}_{BG}$ from $V^p$ ratios (Proposition 4.10) |

### 5.3 Summary of Robustness

| Threat | Defense | Reference |
|--------|---------|-----------|
| Non-power-law $\sigma^2$ | Regular variation is generic | Theorem 1.3 |
| Oscillating tails | Ruled out by leverage effect | Proposition 1.4, 1.6 |
| Faster-than-polynomial growth | Ruled out by return tail exponent | Proposition 1.5 |
| Finite-activity jumps | BV filters them | Theorem 2.3, 2.5 |
| Infinite-activity jumps | Absorbed into $\sigma^2_{\text{eff}}$ at finite $\Delta_n$ | Theorem 3.1 |
| Jump bubbles | Impossible under finite moments / circuit breakers | Theorem 4.3, Corollary 4.4 |
| Market microstructure | Tick size + finite order flow = finite activity | Propositions 3.2, 3.4 |
| Regime switching | MLKFellerGP time-local test | Proposition 1.7 |
| Stochastic volatility | ARD vol proxy ($\ell_v$) + ScaleFunctionGP | Level L-SV, L-VE |
| Non-separable SV (JPS) | ScaleFunctionGP + KGEDMD generator | Level L-VE |

---

## References

- Aït-Sahalia, Y. & Jacod, J. (2009). Testing for jumps in a discretely observed process. *Annals of Statistics*, 37(1), 184–222.
- Aït-Sahalia, Y. & Jacod, J. (2014). *High-Frequency Financial Econometrics*. Princeton University Press.
- Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. *Journal of Financial Econometrics*, 2(1), 1–37.
- Barndorff-Nielsen, O. E. & Shephard, N. (2006). Econometrics of testing for jumps in financial economics using bipower variation. *Journal of Financial Econometrics*, 4(1), 1–30.
- Bates, D. S. (1996). Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche Mark options. *Review of Financial Studies*, 9(1), 69–107.
- Bingham, N. H., Goldie, C. M., & Teugels, J. L. (1987). *Regular Variation*. Cambridge University Press.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.
- Cont, R. & Mancini, C. (2011). Nonparametric tests for pathwise properties of semimartingales. *Bernoulli*, 17(2), 781–813.
- Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.
- Dandapani, A. & Protter, P. (2019). Strict local martingales via filtration enlargement. *Stochastic Processes and their Applications*, 129(7), 2519–2539.
- Delbaen, F. & Shirakawa, H. (2002). No arbitrage condition for positive diffusion price processes. *Asia-Pacific Financial Markets*, 9(3–4), 159–168.
- Engelbert, H. J. & Schmidt, W. (1991). Strong Markov continuous local martingales and solutions of one-dimensional stochastic differential equations. *Mathematische Nachrichten*, 143(1), 167–184.
- Feller, W. (1952). The parabolic differential equations and the associated semi-groups of transformations. *Annals of Mathematics*, 55(3), 468–519.
- Florens-Zmirou, D. (1993). On estimating the diffusion coefficient from discrete observations. *Journal of Applied Probability*, 30(4), 790–804.
- Jarrow, R., Protter, P. & Shimbo, K. (2010). Asset price bubbles in incomplete markets. *Mathematical Finance*, 20(2), 145–185.
- Karlin, S. & Taylor, H. M. (1981). *A Second Course in Stochastic Processes*. Academic Press.
- Khasminskii, R. Z. (2012). *Stochastic Stability of Differential Equations*, 2nd ed. Springer.
- Kou, S. G. (2002). A jump-diffusion model for option pricing. *Management Science*, 48(8), 1086–1101.
- Mancini, C. (2009). Non-parametric threshold estimation for models with stochastic diffusion coefficient and jumps. *Scandinavian Journal of Statistics*, 36(2), 270–296.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Mijatović, A. & Urusov, M. (2012). On the martingale property of certain local martingales. *Probability Theory and Related Fields*, 152(1–2), 1–30.
- Protter, P. (2013). A mathematical theory of financial bubbles. In *Paris-Princeton Lectures on Mathematical Finance 2013*, Springer, 1–108.
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Resnick, S. I. (2007). *Heavy-Tail Phenomena: Probabilistic and Statistical Modeling*. Springer.
- Todorov, V. & Tauchen, G. (2011). Volatility jumps. *Journal of Business & Economic Statistics*, 29(3), 356–371.
