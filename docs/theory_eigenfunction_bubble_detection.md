# Eigenfunction Theory of Bubble Detection

## Connecting Feller's Test, Khasminskii, and Qin-Linetsky

### 1. Bubbles as Strict Local Martingales

A bubble exists when the discounted asset price $S_t/B_t$ is a **strict local martingale** under the risk-neutral measure $\mathbb{Q}$ — a local martingale that is NOT a true martingale (Jarrow, Protter & Shimbo 2010).

For a 1D diffusion $dS = rS\,dt + \sigma(S)\,dW^{\mathbb{Q}}$ (under $\mathbb{Q}$), the fundamental result is:

**Theorem** (Delbaen-Shirakawa 2002, Mijatović-Urusov 2012):
$$
S \text{ is a strict local martingale} \iff \int^\infty \frac{x}{\sigma^2(x)} dx < \infty
$$

For $\sigma^2(S) \sim c^2 S^\alpha$ as $S \to \infty$:
$$
\int^\infty \frac{x}{c^2 x^\alpha} dx = \int^\infty \frac{1}{c^2 x^{\alpha-1}} dx < \infty \iff \alpha > 2
$$

This is the **α > 2 test**: the growth rate of the diffusion coefficient determines whether the process can reach infinity in finite time (explosion), which is equivalent to being a strict local martingale.

### 2. Where Feller/Khasminskii Fit

**Feller's boundary classification** (for 1D diffusions): determines whether a boundary point (here, $S = \infty$) is:
- **Natural** (inaccessible, but process "tries" to reach it)
- **Entrance** (process can start from there but never returns)
- etc.

For $\sigma^2(S) \sim c^2 S^\alpha$:
- $\alpha \leq 2$: infinity is a natural boundary, process is **non-explosive**
- $\alpha > 2$: infinity is **accessible**, process can explode → strict local martingale → bubble

**Khasminskii's Lyapunov criterion**: A diffusion is non-explosive if there exists a positive function $u$ with $Lu \leq -1$ outside a compact set. This is equivalent to the Feller test for 1D but extends to higher dimensions. For 1D with $\sigma^2(S) \sim c^2 S^\alpha$: such a function exists iff $\alpha \leq 2$.

### 3. Where Qin-Linetsky (2015) Fits — And Doesn't

**Critical assumption**: Q&L assume the state process X is **conservative** (non-explosive):
> "$\mathbb{P}_x(X_t \in E) = 1$ for each initial $x \in E$ and all $t \geq 0$ (no killing or explosion)" — Q&L p.5

Their recurrence/transience distinction is about whether a **non-explosive** process returns to compact sets:
- **Recurrent**: process returns to every compact set a.s.
- **Transient**: process escapes to infinity without returning

Q&L's Theorem 3.1 (uniqueness of recurrent eigenfunction) says: there is at most one positive eigenfunction $\pi_R$ of the pricing operator such that X is recurrent under the eigen-measure $\mathbb{Q}^{\pi_R}$. This eigenfunction is the Perron-Frobenius eigenvector of the pricing semigroup.

**This does NOT directly address bubble detection** because:
1. Bubbles involve **explosion** (process exits the state space in finite time)
2. Q&L's framework excludes explosive processes by assumption
3. Q&L's recurrence/transience is a FINER distinction within non-explosive processes

### 4. The Complementary Relationship

The correct picture for a 1D diffusion $dS = rS\,dt + \sigma(S)\,dW^{\mathbb{Q}}$:

```
σ²(S) ~ c²S^α
     │
     ├── α > 2: EXPLOSIVE → strict local martingale → BUBBLE
     │   (Outside Q&L framework — process is not conservative)
     │
     └── α ≤ 2: NON-EXPLOSIVE → true martingale → NO BUBBLE
         (Q&L framework applies)
         │
         ├── Process recurrent → recurrent eigenfunction π_R exists (unique)
         │   → Hansen-Scheinkman factorization
         │   → Ross Recovery possible
         │   → Long-term pricing asymptotics
         │
         └── Process transient → no recurrent eigenfunction
             → Hansen-Scheinkman factorization not unique
             → Ross Recovery fails
```

So Q&L and bubble detection are **complementary**:
- **Bubble detection** (α > 2): Determines whether the process is explosive
- **Q&L spectral theory** (α ≤ 2): Characterizes the non-explosive process's long-term behavior

### 5. What Q&L IS Useful For (Post-Bubble-Detection)

Once we've established no bubble (α ≤ 2), Q&L gives us:

1. **Eigenfunction-based pricing**: $\mathscr{P}_t f(x) = c_f e^{-\lambda_R t} \pi_R(x) + O(e^{-(\lambda_R + a)t})$
   - Long-maturity bond/option pricing dominated by the principal eigenfunction
   - Spectral gap $a$ controls convergence rate

2. **Hansen-Scheinkman factorization**: $S_t = e^{-\lambda t} \frac{\pi(X_0)}{\pi(X_t)} M_t^\pi$
   - Decomposes the SDF into discount + eigenfunction ratio + martingale
   - Risk premium decomposition via CdC: $G^{\mathbb{Q}}f = G^\pi f + \Gamma^\pi(h,f)/h$ (Thm 4.1)

3. **Ross Recovery**: Under transition independence + recurrence, physical measure $\mathbb{P}$ can be recovered from state prices via $\pi_R$

4. **Explicit examples** (Section 6 + Appendices F-G):
   - CIR: $\pi_R(x) = e^{-(b+\gamma)x/\sigma^2}$ (bounded, decaying)
   - 3/2 model: $\pi_0(x) = x^{-1/\kappa}$ (power-law, depends on parameters)
   - Affine models: $\pi(x) = e^{u^\top x}$ via quadratic vector equations

### 6. Measure Invariance and the α Test

**Why the α test works from P-data for a Q-defined property:**

Girsanov's theorem changes the drift: $\mu^{\mathbb{Q}}(S) = rS \neq \mu^{\mathbb{P}}(S)$.
But the diffusion coefficient is **invariant**: $\sigma^2(S)$ is the same under $\mathbb{P}$ and $\mathbb{Q}$.

Since:
- The strict local martingale criterion depends only on $\sigma^2(S)$ (Section 1)
- The Feller/Khasminskii explosion criterion depends only on $\sigma^2(S)$
- The α parameter is determined entirely by $\sigma^2(S)$

Estimating α from $\mathbb{P}$-data gives the correct $\mathbb{Q}$-answer.

Moreover, **squared increments are already measure-invariant**:
$$
(\Delta S)^2 = (\mu\,dt + \sigma\,dW)^2 = \sigma^2\,dt + O(dt^{3/2})
$$
The drift term $\mu^2 dt^2$ is negligible relative to $\sigma^2 dt$.

### 7. Operator-Based Bubble Detection: What Works and What Doesn't

Four operator-based approaches were tested. All fail for different reasons:

#### 7.1 CdC via Generator
$\sigma^2(S) = \Gamma(S, S) = L(S^2) - 2S \cdot L(S)$ — catastrophic cancellation.
At $S = 100$: $L(S^2) \approx 10^3$ and $2S \cdot L(S) \approx 10^3$, while $\sigma^2 \approx 9$.
Relative error amplified ~100×. Results in ~0.4 upward bias on α.

#### 7.2 Eigenfunction Growth Reconstruction
RBF/Nyström basis functions are bounded by construction. All reconstructed eigenfunctions $\hat{\pi}(x) = \sum_i c_i \phi_i(x)$ decay at the boundary regardless of true behavior. Cannot distinguish bounded vs unbounded eigenfunctions.

**Note**: This tests the *wrong criterion*. The correct Khasminskii criterion is the eigenvalue sign (§7.4), not eigenfunction growth — in RBF RKHS, all functions are bounded, so boundedness is automatic.

#### 7.3 Multi-step Koopman Propagation
$K^n$ amplifies approximation error exponentially. The martingale defect at one step is $O(dt^2) \approx 10^{-4}$, smaller than the Koopman approximation error per step. Signal drowned by noise at all useful horizons.

#### 7.4 Eigenvalue Sign Test (the correct Khasminskii criterion)

**Theory**: The bounded eigenfunction theorem (Khasminskii Ch 3-4; Ethier & Kurtz Thm 4.5.4; Engländer & Pinsky) states:
> Explosion (bubble) ⟺ ∃ bounded, positive eigenfunction $u$ with $Lu = \lambda u$, $\lambda > 0$

For RBF Koopman, all RKHS functions are bounded by construction, so if the learned generator $L$ has an eigenvalue with $\text{Re}(\lambda) > 0$ and the corresponding eigenfunction is positive, the Khasminskii criterion is satisfied → bubble.

**Connection**: The survival function $u(x) = \mathbb{E}[e^{-\lambda \tau_{\text{exp}}}]$ is bounded in $(0,1]$, satisfies $Lu = \lambda u$. The generalized principal eigenvalue $\lambda_c$ (Engländer & Pinsky 1999) equals $\sup\{\lambda : \exists u > 0, Lu = \lambda u\}$; explosion ⟺ $\lambda_c > 0$.

**Spurious mode concern**: Extrapolation failure when data leaves landmark support (§3.2 of `theory_bubble_cdc_bounded_equivalence.md`) can create positive eigenvalues. Four filters: (i) eigenfunction positivity, (ii) imaginary part ratio, (iii) boundary mass concentration, (iv) multi-seed stability.

**Experimental results** (`tests/test_eigenvalue_bubble_sign.py`, 6 DGPs × 3 seeds):

| DGP | True | max Re(λ) | P(bub\|λ) | α̂ | Eigen | α test |
|-----|------|-----------|-----------|------|-------|--------|
| GBM σ=0.3 | no bubble | −0.0005±0.000 | 0.000 | 2.04 | 3/3 | 1/3 |
| CEV β=1.5 | no bubble | −0.0107±0.004 | 0.000 | 1.52 | 3/3 | 3/3 |
| CEV β=2.5 | **bubble** | −0.0016±0.002 | 0.000 | 2.51 | **0/3** | 3/3 |
| CEV β=3.0 | **bubble** | −0.0028±0.003 | 0.000 | 2.96 | **0/3** | 3/3 |
| Heston | no bubble | −0.0083±0.006 | 0.000 | 1.77 | 3/3 | 3/3 |
| SABR γ=1.5 | **bubble** | 0.0007±0.008 | 0.000 | 0.15 | **0/3** | 0/3 |

**Result: FAILS.** The eigenvalue sign test correctly identifies all *non-bubble* DGPs (no false positives — all Re(λ) ≤ 0) but **misses every bubble DGP** (no true positives).

**Why it fails — signal-to-noise**: For dt=0.01, a true generator eigenvalue $\lambda = 0.5$ maps to Koopman eigenvalue $\mu = 1 + \lambda \cdot dt = 1.005$. With $m=80$ landmarks and regularization $\lambda_{\text{reg}} = 10^{-3}$, this 0.5% deviation from 1 is below the noise floor. Regularization systematically shrinks Koopman eigenvalues toward 0 (generator eigenvalues toward $-\infty$), burying the positive eigenvalue signal. Even for CEV β=3.0 (strong bubble), max Re(λ) ≈ −0.003 — the regularization wins.

**Path forward attempted — multi-horizon direct fitting (§7.4.1)**: Instead of $K^n$ (§7.3, which compounds error), fit the Koopman operator DIRECTLY at longer horizons using $(X_t, X_{t+\Delta t})$ pairs. At $\Delta t = k \cdot dt$, the Koopman eigenvalue $\mu = e^{\lambda \cdot \Delta t}$ is amplified: for $\lambda = 0.5$, $\mu(dt=0.01) = 1.005$ vs $\mu(\Delta t=1.0) = 1.65$.

**Implementation**: `SigKGEDMDCdCEstimator.fit_koopman_at_horizon(k)` — reuses landmarks/bandwidth from `fit()`, solves regression $K_{\text{horizon}} = (K_t^T K_t + \lambda_{\text{reg}} I)^{-1} K_t^T K_{\text{next}}$ on $(X[:-k], X[k:])$ pairs. Generator eigenvalues recovered via $\lambda_{\text{gen}} = \log(\mu_{\Delta t}) / \Delta t$ (not $(μ-1)/Δt$). Diagnostics: kernel coverage at targets, semigroup consistency ratio, explosion probability.

**Multi-horizon results** (`tests/test_eigenvalue_bubble_sign.py`, horizons $k \in \{1, 10, 50, 100\}$, 6 DGPs × 3 seeds):

| DGP | True | Single P(bub) | Multi P(bub) | α̂ | Single | Multi | α test |
|-----|------|--------------|-------------|------|--------|-------|--------|
| GBM σ=0.3 | no bubble | 0.000 | 0.000 | 1.87 | 3/3 | 3/3 | 3/3 |
| CEV β=1.5 | no bubble | 0.000 | 0.000 | 1.53 | 3/3 | 3/3 | 3/3 |
| CEV β=2.5 | **bubble** | 0.000 | 0.000 | 2.50 | **0/3** | **0/3** | 3/3 |
| CEV β=3.0 | **bubble** | 0.000 | 0.000 | 2.98 | **0/3** | **0/3** | 3/3 |
| Heston | no bubble | 0.000 | 0.092 | 1.74 | 3/3 | 3/3 | 3/3 |
| SABR γ=1.5 | **bubble** | 0.000 | 0.000 | 0.13 | **0/3** | **0/3** | 0/3 |

**Result: Multi-horizon ALSO FAILS.** No improvement over single-step: 3/6 DGPs correct (all non-bubble), 0/3 bubble DGPs detected at any horizon.

**Root cause — discrete-time processes cannot explode (March 2026 §7.4.2)**: Initially attributed to conservative Euler clamping (`max(S, 1e-4)`). Tested exponential Euler (log-space, S always positive, no clamping) — results identical. Also tested Sturm-Liouville eigenvalue computation on the estimated generator (discretize $L = \frac{1}{2}\hat\sigma^2\partial^2 + \hat\mu\partial$ on a grid with Dirichlet BCs) — 1/6 DGPs correct (worse than Koopman due to noisy coefficient estimates at grid boundaries).

**The true root cause is fundamental**: discrete-time Markov chains cannot explode in finite time, regardless of simulation scheme. The explosive eigenvalue $\lambda_c > 0$ is a continuous-time phenomenon: it reflects accumulation of infinitely many infinitesimal steps. At any finite $dt$, the Koopman transition operator is a stochastic kernel with $|\mu| \leq 1$, and $\lambda_{\text{gen}} = \log(\mu)/dt$ is biased toward $\leq 0$ by the regularization noise floor. Even without regularization, the discrete-time transition contains no explosive signal. **This applies equally to real data** — prices observed at discrete times never literally reach infinity.

**Why bounded RKHS functions cannot detect bubbles**: The Khasminskii eigenfunction $u(x) = \mathbb{E}[e^{-\lambda\tau_{\exp}}]$ is theoretically bounded in $(0,1]$, but it is nearly constant ($\approx 1$) everywhere in the observed price range and only deviates near the explosion boundary (at infinity). RBF kernel functions are localized — they cannot resolve a function that is essentially flat except at infinity. The explosive signal lives in *unbounded* test functions ($S^p$ for large $p$), which are outside the RKHS.

**Sturm-Liouville attempt (§7.4.3)**: `SigKGEDMDCdCEstimator.sturm_liouville_eigenvalues()` discretizes the generator on a 1D grid using estimated $\hat\sigma^2(S)$ and $\hat\mu(S)$. Theory: Dirichlet BCs implement the killed semigroup; as domain $D \to (0,\infty)$, $\lambda_1(D) \to \lambda_c$. Practice: noise in $\hat\sigma^2$ and $\hat\mu$ at the grid boundaries dominates, creating spurious positive eigenvalues for non-bubble DGPs (GBM false positive rate 33%, Heston 67%) while still missing bubble DGPs. SL is 1D-specific and cannot scale to multivariate.

**Conclusion — eigenvalue approaches are a dead end for bubble detection.** Three methods tested (Koopman eigenvalue, multi-horizon Koopman, Sturm-Liouville), all fail for the same fundamental reason: they require resolving a near-constant bounded eigenfunction from discrete observations. The α test succeeds because it answers a different, easier question: does $\sigma^2(S)$ grow faster than $S^2$? This is a *local* property of the generator coefficients, not a *global* spectral property. The equivalence chain $\text{explosion} \Leftrightarrow \alpha > 2 \Leftrightarrow \lambda_c > 0$ is exact in theory, but the $\alpha > 2$ form is the only one that is numerically accessible from discrete price data.

**References**: Khasminskii, *Stochastic Stability*, Ch 3-4; Ethier & Kurtz, Thm 4.5.4; Engländer & Pinsky (1999); Ekström & Tysk (2009).

### 8. Theoretical Justification of the KGEDMD α Test for General DGPs

The pipeline uses KGEDMD to estimate $\sigma^2(S)$, then tests $\alpha > 2$ via Bayesian regression on $\log\hat\sigma^2 \sim \alpha \log S$. This section rigorously justifies each step for DGPs beyond the CEV toy model.

#### 8.1 The General Feller Integral Test

For a 1D diffusion $dS = b(S)\,dt + \sigma(S)\,dW$ on $(0, \infty)$, the **Feller test for explosion at $+\infty$** is:

$$
\text{Explosion at } +\infty \iff \int_c^\infty s'(x)^{-1}\,dx < \infty
$$

where $s'(x) = \exp\left(-\int^x \frac{2b(y)}{\sigma^2(y)}\,dy\right)$ is the scale density. Under the risk-neutral measure $\mathbb{Q}$ where $b(S) = rS$ (martingale pricing), a simpler characterization applies (Delbaen-Shirakawa 2002, Mijatović-Urusov 2012):

$$
S \text{ is a strict local martingale} \iff \int_c^\infty \frac{x}{\sigma^2(x)}\,dx < \infty \tag{F}
$$

**This criterion depends only on $\sigma^2(S)$, not on drift.** This is the master theorem — everything else follows.

#### 8.2 From Feller to α: The Tail Exponent

Define the **local volatility exponent**:
$$
\alpha(S) = \frac{d \log \sigma^2(S)}{d \log S}
$$

If $\alpha(S) \to \alpha_\infty$ as $S \to \infty$ (the tail exponent exists), then $\sigma^2(S) \sim C \cdot S^{\alpha_\infty}$ for large $S$, and:
$$
\int_c^\infty \frac{x}{\sigma^2(x)}\,dx \sim \int_c^\infty \frac{x}{C \cdot x^{\alpha_\infty}}\,dx = \int_c^\infty C^{-1} x^{1-\alpha_\infty}\,dx
$$

This converges iff $\alpha_\infty > 2$. Therefore:

$$
\boxed{\text{Explosion} \iff \alpha_\infty > 2}
$$

**Important**: The α test estimates $\alpha_\infty$ from finite data. The regression $\log\hat\sigma^2 \sim \alpha \log S + C$ yields $\hat\alpha \approx \alpha_\infty$ when:

1. The power-law tail behavior sets in within the observed price range (not just at $S = \infty$)
2. The σ² estimates are accurate in this range (see §8.3)

For processes where $\alpha(S)$ is NOT constant (e.g., sigmoid-like $\sigma^2$ that transitions from one regime to another), the regression gives a weighted average over the observed range, which may differ from $\alpha_\infty$. **This is a genuine limitation** — the test is designed for processes with regular tail behavior.

#### 8.3 Consistency of KGEDMD σ² Estimation

The KGEDMD `sigma_squared_direct` method performs kernel ridge regression:
$$
\hat\sigma^2(S) = \arg\min_{f \in \mathcal{H}_K} \sum_{i=1}^N \left(\frac{(\Delta S_i)^2}{\Delta t} - f(S_i)\right)^2 + \lambda \|f\|_{\mathcal{H}_K}^2
$$

**Theorem** (kernel regression consistency; Steinwart & Christmann 2008, Theorem 6.23): Under the RBF kernel (which is universal), if:

- (C1) The data $(S_i, (\Delta S_i)^2/\Delta t)$ are drawn from a stationary ergodic process (or have sufficient mixing)
- (C2) The regularization $\lambda_N \to 0$ and $\lambda_N N \to \infty$ as $N \to \infty$
- (C3) The landmarks span the support of the data distribution

then $\hat\sigma^2(S) \to \sigma^2(S)$ in probability at each point $S$ in the interior of the data support.

**Finite-sample rates**: For $m$ Nyström landmarks and $N$ data points, the KGEDMD approximation error is $O(m^{-1/2}) + O(N^{-1/2})$ (Nyström approximation + statistical error). With $m = 80$ and $N = 10{,}000$, this gives ~10% pointwise error, consistent with the observed 1.8–5.1% RMSE on ergodic processes (§CdC Kernel Estimators).

**Key requirement**: the data must **visit the range** where σ²(S) exhibits its tail behavior. For non-explosive processes, ergodicity guarantees this. For near-explosive processes, the paths reach high S values with positive probability, providing data in the tail.

#### 8.4 The Equivalence: α > 2 ⟺ λ_c > 0

The **generalized principal eigenvalue** (Engländer-Pinsky 1999, Pinsky 1995) is:
$$
\lambda_c = \sup\{\lambda \in \mathbb{R} : \exists u > 0 \text{ with } Lu = \lambda u \text{ on } (0,\infty)\}
$$

**Theorem** (Pinsky 1995, Ch. 4): For a 1D diffusion with generator $L = \frac{1}{2}\sigma^2(x)\partial_{xx} + b(x)\partial_x$:
$$
\lambda_c > 0 \iff \text{the process is explosive (reaches } \partial \text{ in finite time)}
$$

Combined with the Feller integral test (§8.1):
$$
\lambda_c > 0 \iff \int_c^\infty \frac{x}{\sigma^2(x)}\,dx < \infty \iff \alpha_\infty > 2
$$

This closes the chain: the KGEDMD α test is testing $\lambda_c > 0$ via the **operationally equivalent** Feller criterion, bypassing the numerically intractable spectral computation (see §7.4 for why direct eigenvalue estimation fails).

The three forms of the criterion — explosion, Feller integral, α > 2 — are mathematically equivalent for 1D diffusions with regular tail behavior. The α test is the only numerically accessible one.

#### 8.5 Stochastic Volatility: When the 1D Test Works and Fails

For a general SV model:
$$
dS = \mu(S,V)S\,dt + \sigma_S(S,V)\,dW_S, \quad dV = \mu_V(S,V)\,dt + \sigma_V(S,V)\,dW_V
$$

the KGEDMD operates on S **marginally**, estimating the **effective 1D diffusion coefficient**:
$$
\sigma^2_{\text{eff}}(S) = \mathbb{E}\left[\sigma_S^2(S, V) \,\Big|\, \text{process visits } S\right]
$$

This is the time-averaged conditional expectation — the NW/KRR regression on $(\Delta S)^2/\Delta t$ given $S$ naturally estimates this quantity.

**When it works** — fast-mixing volatility (Heston): If V is mean-reverting with rate $\kappa$ and the observation timescale $T$ satisfies $T \gg 1/\kappa$, then the marginal effective diffusion is:
$$
\sigma^2_{\text{eff}}(S) \approx \mathbb{E}[V] \cdot S^2 = \theta \cdot S^2 \quad \Rightarrow \quad \alpha_{\text{eff}} = 2 \quad (\text{no bubble, correct})
$$

More precisely, $\sigma^2_{\text{eff}}(S) = \mathbb{E}[V | \text{process at } S] \cdot S^2$. For Heston with moderate leverage ($\rho$), $\mathbb{E}[V|S]$ varies weakly with $S$ (V mean-reverts regardless of S level), so $\alpha_{\text{eff}} \approx 2$.

**When it fails** — correlated vol with non-Markov price-vol dependence (SABR γ > 1): For SABR, $\sigma_S = V \cdot S^{\gamma-1} \cdot S = V S^\gamma$, so the true α should be $2\gamma$. But:
$$
\sigma^2_{\text{eff}}(S) = \mathbb{E}[V^2 | \text{process at } S] \cdot S^{2\gamma}
$$

The conditional $\mathbb{E}[V^2 | \text{process at } S]$ is **not constant** — it depends strongly on the path history. Paths that reach high $S$ required sustained high $V$ (which is stochastic and mean-reverting in the log-normal SABR vol). The conditioning induces a **negative correlation** between $\mathbb{E}[V^2 | S]$ and $S$, pulling the effective α well below $2\gamma$.

**Experimental confirmation**: SABR γ=1.5 gives $\hat\alpha \approx 0.2$ instead of $2\gamma = 3.0$ — the 1D marginal approach completely fails because the V-S correlation dominates the pure $S^\gamma$ scaling.

**Theoretical fix**: For SV models, bubble detection requires the JOINT (S, V) process, not the marginal. Options:
1. **If V is observable**: Condition on V and test the conditional σ²(S | V) scaling → same α test but in the (S, V) space
2. **If V is latent**: Use the signature-augmented state (log S, QV) — this is exactly what `SigKGEDMDCdCEstimator` does with ARD bandwidth and GCV model selection. When GCV selects $w_{qv} > 0$, the QV dimension proxies for V, and the α test operates on the augmented state
3. **Theoretical criterion for the joint process**: A multivariate diffusion on $\mathbb{R}^d$ is explosive iff there exists no Lyapunov function $V(x) \to \infty$ with $LV \leq cV$ outside a compact set (Khasminskii generalization). For SV models, this reduces to checking whether the joint drift + diffusion can push (S, V) to infinity. This is model-specific and does not reduce to a simple α test.

**Current status**: The SV bubble detection works when GCV auto-selects the QV dimension (11/11 on Tier III-SV holdout v2). When the process is Markov in S alone (CEV, GBM), GCV selects $w_{qv} = 0$ (1D test). When non-Markov features help (SV models where V is slow-mixing), GCV selects $w_{qv} > 0$. The ARD kernel adapts the effective dimensionality.

#### 8.6 Connection to Large Deviations

The generalized principal eigenvalue $\lambda_c$ has a **large deviations interpretation** (Donsker-Varadhan theory):

For the semigroup $T_t f(x) = \mathbb{E}_x[f(X_t)]$:
$$
\lambda_c = \lim_{t \to \infty} \frac{1}{t} \log \|T_t\|_{L^2 \to L^2}
$$

This is the exponential growth rate of the semigroup operator norm. When $\lambda_c > 0$, the semigroup grows exponentially — there exist initial conditions from which $\mathbb{E}_x[f(X_t)]$ grows without bound, i.e., the process escapes to infinity.

For the killed semigroup $T_t^D f(x) = \mathbb{E}_x[f(X_t)\mathbf{1}_{t < \tau_D}]$ on bounded $D$:
$$
\lambda_c = \lim_{D \uparrow (0,\infty)} \lambda_1(D)
$$

where $\lambda_1(D)$ is the principal Dirichlet eigenvalue on $D$. This connects to our Sturm-Liouville attempt (§7.4.3): the SL approach computes $\lambda_1(D)$ for a specific bounded $D$, which should converge to $\lambda_c$ as $D$ grows. In practice, it fails because $\lambda_1(D)$ converges slowly and is sensitive to boundary effects in the estimated coefficients.

The α test bypasses this entirely by characterizing $\lambda_c$ through its connection to the Feller integral (§8.4), which depends only on **local** properties of $\sigma^2(S)$ — specifically, its tail growth rate.

**Summary of the theoretical chain**:
$$
\underbrace{\hat\sigma^2(S)}_{\text{KGEDMD (§8.3)}} \xrightarrow{\text{regression}} \underbrace{\hat\alpha}_{\text{tail exponent (§8.2)}} \xrightarrow{\hat\alpha > 2 ?} \underbrace{\lambda_c > 0}_{\text{Engländer-Pinsky (§8.4)}} \iff \underbrace{\text{explosion}}_{\text{Feller (§8.1)}} \iff \underbrace{\text{bubble}}_{\text{Jarrow-Protter (§1)}}
$$

Each arrow is a theorem, not a heuristic. The only approximation is in the KGEDMD regression (finite sample, finite landmarks), which is controlled by standard kernel regression theory.

### 9. The Pipeline

```
Theory:     Explosion ⟺ strict local martingale ⟺ α > 2    [Feller/Khasminskii]
            σ²(S) is measure-invariant → test valid from P-data
            Each arrow is a theorem (see §8.1–8.6)

Data:       Observed prices S under P (possibly with latent stochastic vol)
            ↓
Features:   Lead-lag signature → QV augmentation (for non-Markov SV)
            GCV auto-selects: w_qv = 0 (Markov) or w_qv > 0 (SV)
            ↓
σ² est:     Direct KRR on (ΔS)²/dt with ARD kernel + Nyström (§8.3)
            Consistency: universal kernel + ergodic data → σ̂²(S) → σ²(S)
            ↓
α test:     BayesianRidge: log(σ̂²) ~ α·log(S) + C
            Estimates tail exponent α_∞ (§8.2)
            ↓
Decision:   P(bubble) = P(α > 2 | posterior)
            Threshold at α = 2 from Feller integral (§8.1)

Caveat:     SV with strong leverage (SABR): marginal α underestimates
            true 2γ — need QV augmentation or joint (S,V) test (§8.5)

Post-test:  IF no bubble (α ≤ 2):
            → Q&L spectral theory applies
            → Koopman eigenfunctions for pricing
            → Hansen-Scheinkman factorization
            → Ross Recovery (if recurrent + transition independent)
```

### 10. Connection to Existing Khasminskii Framework

Our codebase already has the Khasminskii test via a different route:
- `stationarity_transforms.py`: Lead-lag signature Lévy area scales with price level
- Regress $\log(\text{QV}) \sim \alpha \cdot \log(\bar{S})$ to get α
- Bubble ⟺ α > 2

This is mathematically equivalent to the direct KRR σ² regression but uses signature features to estimate QV (realized variance) rather than kernel regression on $(ΔS)^2/dt$.

The KGEDMD approach in `SigKGEDMDCdCEstimator` provides:
1. **Same α test** via direct σ² regression (primary)
2. **Generator + CdC** as diagnostic (validates operator quality)
3. **Pricing pipeline** when α ≤ 2 (eigenfunction decomposition)

### 11. Multi-Asset and Stochastic Volatility: Tiered Architecture

Our bubble detection framework is organized in tiers of increasing generality:

```
Tier    Test                      Handles                           Method
────    ────                      ───────                           ──────
L1      Sig QV scaling            1D CEV-type                       log(QV) ~ α·log(S̄)
L2      Nonparametric Feller+GP   1D + directional portfolios       NW σ̂²(z) + GP posterior
L2-SV   Conditional Feller        Separable SV: σ²(S,V)=f(V)·Sᵅ⁽ⱽ⁾  Bin by V, Feller per bin
L3      Koopman generator eig     Non-separable SV (JPS 2022)       2D generator eigenvalue
```

**L2: Directional Feller with GP (§8 extension to $\mathbb{R}^d$)**

For a d-dimensional price process, we test bubble along portfolio direction $w$:
$$
z = w^T X, \quad \sigma^2_w(z) = w^T \Sigma(X) w \big|_{w^T X = z}
$$
Bubble along $w$ ⟺ $\int^\infty z/\sigma^2_w(z)\,dz < \infty$ ⟺ $\alpha_w > 2$.

The directional test scans $n$ directions uniformly on the unit sphere, estimates $\alpha_w$ per direction via:
1. Nadaraya-Watson kernel regression for $\hat\sigma^2_w(z)$ at quantile landmarks
2. GP with parametric mean (R&W §2.7): $\log\hat\sigma^2(z) = \alpha\cdot\log|z| + c + f(z)$, $f \sim \text{GP}(0, k_{\text{SE}})$
3. Blocked time-series CV for GP flexibility $\sigma_f$ (prevents SD collapse)
4. Blocked bootstrap SE for temporal correlation correction
5. Šidák correction: $P(\text{bubble}) = \Phi(z_{\max})^{n_{\text{eff}}}$

This connects to **multivariate regular variation** (Resnick 2007): our directional scan IS the empirical estimation of the spectral measure on the unit sphere. The bubble directions are exactly those where the spectral measure concentrates with tail index > 2.

**L2-SV: Conditional Feller for Separable SV**

When $\sigma^2(S, V) = g(V) \cdot S^{\alpha(V)}$ with level-dependent exponent $\alpha(V)$, the marginal test averages across vol regimes and may miss bubbles that only appear at high vol:

1. Bin observations by vol proxy $V$ into quantiles
2. Run Feller α test on price sub-series within each bin
3. Aggregate with Šidák correction: $P(\text{bubble}) = \Phi(z_{\max})^{n_{\text{bins}}}$

This detects regimes where $\alpha(V) > 2$ only at certain vol levels, while the marginal $\alpha \approx 2$ due to averaging. Validated on CIR-modulated CEV: conditional Feller 6/6, marginal 3/6.

**L3: Non-Separable SV (JPS 2022 Counter-Example)**

The Feller test (all tiers above) assumes the bubble mechanism is through $\sigma^2(S)$ growing super-quadratically: $\alpha > 2$. However, JPS 2022 (Remark 6) construct a counter-example:
$$
dX = X\cdot Y\,dW_1, \quad dY = K\cdot Y^2\,dt + Y(\rho\,dW_1 + \sqrt{1-\rho^2}\,dW_2)
$$
where bubble ⟺ $K \geq -\rho$. At any fixed $Y = y$:
$$
\sigma^2(X|Y=y) = X^2 \cdot y^2 \implies \alpha = 2 \text{ (exactly, for all } y\text{)}
$$

The bubble arises NOT from $\alpha > 2$ but from the coupled Y dynamics ($K\cdot Y^2$ drift) feeding back through correlation. **No Feller-based test can detect this** — it requires the joint generator spectrum (positive eigenvalue ⟹ explosion).

This is where the Koopman eigenvalue approach (existing KGEDMD infrastructure) becomes necessary: learn the 2D generator $L$ on $(X, Y)$ and check if it has a positive eigenvalue. This tier remains future work.

**Comparison with Existing Literature**

| Method | Theoretical Basis | What It Tests | Limitation |
|--------|------------------|---------------|------------|
| PSY/GSADF | Unit root / explosive AR | Drift > 0 in log prices | Ad-hoc, not SLM |
| JPS 2022 invariance | QV scaling under measure change | $\sigma^2(S)$ via QV | Assumes separable σ |
| Our L1 (sig QV) | Lead-lag Lévy area = QV | Same as JPS, via signatures | 1D only |
| Our L2 (GP Feller) | Feller integral + GP posterior | Directional α with UQ | Separable σ only |
| Our L2-SV | Conditional Feller | Level-dependent α(V) | Still needs α > 2 |
| Our L3 (Koopman gen.) | Joint generator eigenvalue | Non-separable SV | Future work |

Our contribution beyond JPS: (1) principled UQ via GP posterior, (2) directional extension via spectral measure, (3) conditional Feller for level-dependent SV, (4) roadmap for non-separable SV via Koopman generator.

### References

- Delbaen, F., & Shirakawa, H. (2002). No arbitrage condition for positive diffusion price processes.
- Mijatović, A., & Urusov, M. (2012). On the martingale property of certain local martingales.
- Jarrow, R., Protter, P., & Shimbo, K. (2010). Asset price bubbles in incomplete markets.
- Qin, L., & Linetsky, V. (2015). Positive Eigenfunctions of Markovian Pricing Operators: Hansen-Scheinkman Factorization, Ross Recovery and Long-Term Pricing.
- Hansen, L. P., & Scheinkman, J. A. (2009). Long-term risk: An operator approach.
- Khasminskii, R. Z. (1960). Ergodic properties of recurrent diffusion processes.
- Khasminskii, R. Z. (2012). *Stochastic Stability of Differential Equations*, 2nd ed. Springer.
- Feller, W. (1952). The parabolic differential equations and the associated semi-groups of transformations.
- Engländer, J., & Pinsky, R. G. (1999). On the construction and support properties of measure-valued diffusions on $D \subseteq \mathbb{R}^d$ with spatially dependent branching. *Ann. Probab.*, 27(2), 684-730.
- Pinsky, R. G. (1995). *Positive Harmonic Functions and Diffusion*. Cambridge University Press.
- Steinwart, I., & Christmann, A. (2008). *Support Vector Machines*. Springer. [Theorem 6.23: kernel regression consistency]
- Ekström, E., & Tysk, J. (2009). Bubbles, convexity and the Black-Scholes equation. *Ann. Appl. Probab.*, 19(4), 1369-1384.
- Donsker, M. D., & Varadhan, S. R. S. (1975). Asymptotic evaluation of certain Markov process expectations for large time. *Comm. Pure Appl. Math.*, 28, 1-47.
- Jarrow, R., Protter, P., & San Martín, F. (2022). Bubble Invariance. *Working paper*.
- Resnick, S. I. (2007). *Heavy-Tail Phenomena: Probabilistic and Statistical Modeling*. Springer.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statist. Surv.*, 4, 40-79.
