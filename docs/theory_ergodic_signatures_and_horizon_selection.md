# Ergodic Signatures and Information-Theoretic Horizon Selection

## Abstract

We develop a rigorous theoretical framework connecting three ideas:
1. **Exponentially Fading Memory (EFM) Signatures** as optimal ergodic transforms for non-stationary financial processes
2. **Mutual-information-based horizon selection** via held-out R² as a proxy for MI
3. The **Bakry-Émery Gamma calculus** linking the Carré du Champ operator (already used for drift/diffusion extraction) to spectral gaps, mixing rates, and horizon bounds

Together these provide a principled, model-free approach to:
- Transform arbitrary (non-ergodic) financial processes into stationary feature spaces
- Select the optimal lookahead horizon H for multi-step utility learning
- Quantify when the method can and cannot work (signal-to-noise bounds)

---

## Part I: The Stationarity Problem in Financial Koopman Learning

### 1.1 Why Ergodicity Matters

Historically, the repo studied a **global product-kernel Koopman control**
framework that learns the growth-rate function:

$$g(\pi, x) = a(x) + \pi \cdot b(x) + \pi^2 \cdot c(x)$$

by regressing H-step utility outcomes $y_H = U(W_{t+H}/W_t)$ onto product
features $\Psi(x_t) \otimes [1, \pi, \pi^2]$. This regression is well-posed
only if the joint distribution $(x_t, \pi_t, y_{H,t})$ is (approximately)
stationary. Otherwise:

- The regression coefficients $\mathbf{w}$ are time-dependent
- Train/test splits are confounded by distributional shift
- The R² metric is meaningless (high R² from spurious trends)

**The core problem**: Most financial state variables are non-stationary.

| Process | Stationary? | Transform needed |
|---------|-------------|-----------------|
| CIR (variance) | Yes (mean-reverting) | None |
| GBM (price) | No (random walk with drift) | Log-returns or realized vol |
| Heston (price + vol) | Price: No, Vol: Yes | Price → returns; vol as-is |
| Regime-switching | Conditionally | EWMA or EFM of observables |

### 1.2 The Insufficiency of Ad-Hoc Transforms

Common approaches — log-returns, rolling variance, EWMA — work for specific cases but lack:

1. **Universality**: Log-returns lose path information; EWMA is linear
2. **Optimality**: No principle for choosing the EWMA span or window
3. **Theoretical guarantees**: No formal ergodicity/mixing rate results

We need a transform $\phi: \text{paths} \to \text{features}$ that is:
- **Ergodic**: $\phi(X)_t$ is stationary for any underlying process $X$
- **Universal**: $\phi$ can approximate any continuous path-functional
- **Tunable**: A single parameter controls the memory/stationarity tradeoff
- **Computable**: O(1) per time step (for online deployment)

The later parts of this note keep the transform theory, but they refine the
controller recommendation.  The current view of the repo is:

1. the **stationary transformed-state** story remains the right representation
   layer;
2. the earlier **global product-kernel controller** is no longer the preferred
   control architecture for finance;
3. the recommended kernel controller is now a **reference-conditioned local
   residual kernel / GP head** acting on a stationary transformed state.

---

## Part II: Exponentially Fading Memory Signatures

### 2.1 Definition

**Definition 2.1** (Abi Jaber & Sotnikov, 2025). Let $X = (X_t)_{t \in \mathbb{R}}$ be a continuous path in $\mathbb{R}^d$, and let $\boldsymbol{\lambda} = (\lambda^1, \ldots, \lambda^d) \in \mathbb{R}_{>0}^d$ be a vector of positive decay rates. The **$\boldsymbol{\lambda}$-exponentially fading memory (EFM) signature** of $X$ at time $t$ is defined level-by-level:

$$\mathbb{X}^{\boldsymbol{\lambda}, n}_{-\infty, t} := \int_{-\infty < u_1 < \cdots < u_n < t} \prod_{k=1}^{n} e^{-\lambda^{i_k}(t - u_k)} \, dX_{u_1}^{i_1} \cdots dX_{u_n}^{i_n}$$

for multi-indices $(i_1, \ldots, i_n) \in \{1, \ldots, d\}^n$.

**Notation**: We write $\mathbb{X}^{\boldsymbol{\lambda}}_t = (1, \mathbb{X}^{\boldsymbol{\lambda},1}_t, \mathbb{X}^{\boldsymbol{\lambda},2}_t, \ldots)$ for the full EFM signature at time $t$.

### 2.2 Connection to Existing Practice

**Remark 2.2** (Level 1 = Exponential Moving Average). At level 1, the EFM signature reduces to the continuous-time EMA:

$$\mathbb{X}^{\lambda^i, 1}_t = \int_{-\infty}^t e^{-\lambda^i(t-s)} \, dX_s^i$$

which is exactly the Ornstein-Uhlenbeck integral of $dX^i$. This is the continuous analogue of our `RecurrentSignatureMap` with forgetting factor $\gamma = e^{-\lambda \cdot dt}$.

**Remark 2.3** (Level 2 = Fading Lévy Area). At level 2, the anti-symmetric component captures the fading Lévy area:

$$\mathbb{X}^{\boldsymbol{\lambda}, ij}_t - \mathbb{X}^{\boldsymbol{\lambda}, ji}_t = \text{exponentially-weighted signed area}$$

This connects directly to our quadratic variation / Itô correction extraction (see `docs/gedmd_ito_correction.md`). The fading ensures stationarity of the QV estimate.

### 2.3 Key Properties

**Theorem 2.4** (Properties of EFM Signatures; Abi Jaber & Sotnikov 2025).
The EFM signature satisfies:

1. **Modified Chen Identity**: For $s < u < t$,

$$\mathbb{X}^{\boldsymbol{\lambda}}_{s,t} = \mathbb{X}^{\boldsymbol{\lambda}}_{s,u} \otimes_{\boldsymbol{\lambda}, u, t} \mathbb{X}^{\boldsymbol{\lambda}}_{u,t}$$

where $\otimes_{\boldsymbol{\lambda}, u, t}$ is the shuffle product modified by exponential decay applied to the left factor.

2. **Path Determinacy**: The EFM signature $\mathbb{X}^{\boldsymbol{\lambda}}_t$ determines the path $X$ uniquely (up to tree-like equivalence).

3. **Universal Approximation**: For any continuous fading-memory functional $F$ on paths, there exists a linear functional $\ell$ on the tensor algebra such that $\langle \ell, \mathbb{X}^{\boldsymbol{\lambda}}_t \rangle$ approximates $F(X_{(-\infty, t]})$ uniformly on compacts.

4. **Time Invariance**: $\mathbb{X}^{\boldsymbol{\lambda}}_t$ depends on the path only through relative times $(t - u_k)$, not absolute time $t$.

### 2.4 Stationarity and Ergodicity

**Theorem 2.5** (Stationarity; Abi Jaber & Sotnikov 2025, Theorem 3.16).
Let $\widehat{W}_t = (t, W_t)$ be time-augmented Brownian motion. Then $\mathbb{X}^{\boldsymbol{\lambda}}_t := \mathbb{X}^{\boldsymbol{\lambda}}_{-\infty, t}(\widehat{W})$ is a **stationary** process. More precisely, it evolves as a group-valued Ornstein-Uhlenbeck process on the tensor algebra $T((\mathbb{R}^{d+1}))$.

**Theorem 2.6** (Exponential Ergodicity; Abi Jaber & Sotnikov 2025, Theorem 5.3).
The EFM-signature process $\mathbb{X}^{\boldsymbol{\lambda}}_t$ of time-augmented Brownian motion is **exponentially ergodic** in the Wasserstein distance:

$$\mathcal{W}_p\left(\text{Law}(\mathbb{X}^{\boldsymbol{\lambda}}_t), \mu_{\boldsymbol{\lambda}}\right) \leq C \cdot e^{-\rho t}$$

where $\mu_{\boldsymbol{\lambda}}$ is the unique invariant measure and $\rho > 0$ depends on $\boldsymbol{\lambda}$.

**Theorem 2.7** (Markov Property; Abi Jaber & Sotnikov 2025, Theorem 5.2).
$\mathbb{X}^{\boldsymbol{\lambda}}_t$ is Markov. Its transition semigroup is that of a group-valued OU process, with mean-reversion at rate governed by $\boldsymbol{\lambda}$.

### 2.5 The Decay Parameter and Its Interpretation

The vector $\boldsymbol{\lambda}$ controls the bias-variance tradeoff:

| $\lambda$ | Memory | Stationarity | Information |
|-----------|--------|--------------|-------------|
| Large | Short (fast decay) | Strong (fast mixing) | Less (ignores history) |
| Small | Long (slow decay) | Weak (slow mixing) | More (captures long-range) |

**For level-$n$ terms**, the effective decay rate is $\lambda_{\mathbf{v}} = \sum_{k=1}^n \lambda^{i_k}$, so higher-order terms decay faster. This is a natural regularization: the most complex path-interactions are the most aggressively forgotten.

### 2.6 Discrete-Time Implementation

For numerical computation with time step $dt$, the EFM signature reduces to a recurrence. Define the forgetting factor $\gamma^i = e^{-\lambda^i \cdot dt}$. Then:

**Level 1**: $Z^i_{t+dt} = \gamma^i \cdot Z^i_t + \Delta X^i_{t+dt}$

**Level 2**: $Z^{ij}_{t+dt} = \gamma^i \gamma^j \cdot Z^{ij}_t + Z^i_t \cdot \Delta X^j_{t+dt} \cdot \gamma^i + \frac{1}{2} \Delta X^i_{t+dt} \cdot \Delta X^j_{t+dt}$

This is precisely the `RecurrentSignatureMap` in `examples/proof_of_concept/signature_features.py` with forgetting factor $\gamma$. The EFM theory provides the theoretical justification for what was implemented empirically.

**Proposition 2.8** (Discrete-Continuous Correspondence). Let $\gamma = e^{-\lambda \cdot dt}$ for scalar $\lambda > 0$. The discrete recurrence $Z_{n+1} = \gamma Z_n + \Delta X_{n+1}$ converges to the continuous EFM level-1 integral as $dt \to 0$:

$$Z_{n} \xrightarrow{dt \to 0} \int_{-\infty}^{t_n} e^{-\lambda(t_n - s)} dX_s$$

*Proof*. By Euler-Maruyama. The discrete sum $Z_n = \sum_{k=-\infty}^{n} \gamma^{n-k} \Delta X_k$ is a Riemann sum for the integral with exponential kernel. Standard convergence results for Riemann-Stieltjes integrals apply. $\square$

---

## Part III: Bakry-Émery Gamma Calculus and Spectral Gaps

### 3.1 The Carré du Champ and Mixing Rates

The Bakry-Émery framework provides a calculus-based approach to quantifying ergodicity. Our codebase already uses the Carré du Champ for drift/diffusion extraction (Level 2 sanity check). Here we show it also controls the mixing rate.

**Definition 3.1** (Carré du Champ). For a Markov generator $L$, the Carré du Champ operator is:

$$\Gamma(f, g) = \frac{1}{2}\left[L(fg) - f \cdot Lg - g \cdot Lf\right]$$

We write $\Gamma(f) = \Gamma(f, f)$. For an Itô diffusion $dX_t = \mu(X_t)dt + \sigma(X_t)dW_t$:

$$\Gamma(f)(x) = \frac{1}{2}\sigma^2(x) |f'(x)|^2$$

**Definition 3.2** (Iterated Carré du Champ). The $\Gamma_2$ operator is the Carré du Champ of $\Gamma$:

$$\Gamma_2(f) = \frac{1}{2}\left[L\Gamma(f) - 2\Gamma(f, Lf)\right]$$

### 3.2 The Bakry-Émery Criterion

**Theorem 3.3** (Bakry-Émery, 1985). If $\Gamma_2(f) \geq \rho \cdot \Gamma(f)$ for all smooth $f$ and some $\rho > 0$, then:

1. **Poincaré Inequality**: $\text{Var}_\mu(f) \leq \frac{1}{\rho} \mathbb{E}_\mu[\Gamma(f)]$

2. **Exponential Mixing**: $\text{Var}_\mu(P_t f) \leq e^{-2\rho t} \cdot \text{Var}_\mu(f)$

3. **Spectral Gap**: The generator $L$ has spectral gap $\geq \rho$ in $L^2(\mu)$

where $\mu$ is the invariant measure and $P_t$ is the semigroup.

### 3.3 Computing $\Gamma_2$ for Financial Diffusions

For a 1D diffusion $dX_t = \mu(x)dt + \sigma(x)dW_t$, explicit computation gives:

$$\Gamma_2(f)(x) = \frac{1}{2}\sigma^2(x)\left[\sigma^2(x)(f'')^2 + \left(\mu'(x)\sigma(x) + \mu(x)\sigma'(x) + \frac{1}{2}\sigma(x)\sigma''(x)\right)(f')^2\right]$$

The Bakry-Émery condition $\Gamma_2 \geq \rho \Gamma$ reduces (for 1D) to:

$$\mu'(x) + \frac{\mu(x)\sigma'(x)}{\sigma(x)} + \frac{1}{2}\sigma''(x) \geq \rho$$

**Example 3.4** (CIR Process: $dV_t = \kappa(\theta - V_t)dt + \xi\sqrt{V_t}dW_t$).

With $\mu(V) = \kappa(\theta - V)$ and $\sigma(V) = \xi\sqrt{V}$:

$$\mu'(V) = -\kappa, \quad \frac{\mu(V)\sigma'(V)}{\sigma(V)} = \frac{\kappa(\theta - V)\xi/(2\sqrt{V})}{\xi\sqrt{V}} = \frac{\kappa(\theta - V)}{2V}$$

For large $V$: $\mu'(V) \approx -\kappa$, so $\rho \approx \kappa$. This gives the **spectral gap $\kappa$** and mixing time $\tau_{\text{mix}} \approx 1/(2\kappa)$.

**Example 3.5** (GBM: $dS_t = \mu S_t dt + \sigma S_t dW_t$).

$$\mu'(S) = 0, \quad \sigma''(S) = 0 \implies \Gamma_2 = 0$$

There is NO spectral gap — GBM is non-ergodic. This confirms that raw prices cannot be used as features.

### 3.4 Connection to Horizon Selection

**Proposition 3.6** (Spectral Gap Bounds the Useful Horizon). Let $L$ have spectral gap $\rho > 0$ with invariant measure $\mu$. For any feature $f$ with $\mathbb{E}_\mu[f] = 0$:

$$\text{Corr}(f(X_0), f(X_H)) \leq e^{-\rho H}$$

Therefore, any H-step regression using features of $X_0$ to predict labels depending on $X_H$ has:

$$R^2(H) \leq C \cdot e^{-2\rho H}$$

for some $C > 0$ (depending on the label-feature relationship).

*Proof*. By the spectral decomposition of $P_t$ and the definition of spectral gap:
$$\text{Cov}_\mu(f(X_0), f(X_H)) = \langle f, P_H f \rangle_{L^2(\mu)} \leq e^{-\rho H} \|f\|^2_{L^2(\mu)}$$

Dividing by variances gives the correlation bound. The R² bound follows since R² ≤ Corr² for simple regression, with the factor of 2 from squaring. $\square$

**Corollary 3.7** (Upper Bound on Useful Horizon). The mutual information $I(X_0; Y_H) \to 0$ as $H \to \infty$ at rate $\rho$. Therefore, there exists $H_{\max} \sim 1/\rho$ beyond which no regression can extract signal.

For the Heston model: $\rho \approx \kappa = 2.0$, so $H_{\max} \approx 1/2 = 0.5$ years $\approx 126$ days.

---

## Part IV: Information-Theoretic Horizon Selection

### 4.1 Motivation

Given a historical global product-kernel regression:

$$y_H = g_H(\pi, X_0; \mathbf{w}) + \varepsilon_H$$

where $y_H = U(W_{t+H}/W_t)$ and
$g_H(\pi, x) = a_H(x) + \pi b_H(x) + \pi^2 c_H(x)$, the horizon $H$ controls a
bias-variance tradeoff:

- **H too small**: The label $y_H$ is dominated by single-step noise. SNR ∝ $\sqrt{H}$ for i.i.d. returns, so R² ∝ H for small H.
- **H too large**: The state features $\Psi(X_0)$ are decorrelated from $y_H$ due to mixing. R² ∝ $e^{-2\rho H}$ for large H.

The optimal H* balances these two forces.

### 4.2 R² as a Proxy for Mutual Information

**Proposition 4.1** (R² and MI for Gaussian Errors). If the regression residuals are approximately Gaussian, then:

$$I(\Psi(X_0), \pi; Y_H) = -\frac{1}{2}\log(1 - R^2)$$

where $R^2$ is the population coefficient of determination. Therefore:

$$H^* = \arg\max_H R^2_{\text{held-out}}(H)$$

is equivalent to maximizing $I(\Psi(X_0), \pi; Y_H)$ under the Gaussian assumption.

*Proof*. For a linear regression model $Y = X\beta + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2_\varepsilon)$:

$$I(X; Y) = h(Y) - h(Y|X) = \frac{1}{2}\log\frac{\text{Var}(Y)}{\text{Var}(\varepsilon)} = -\frac{1}{2}\log\left(1 - \frac{\text{Var}(\hat{Y})}{\text{Var}(Y)}\right) = -\frac{1}{2}\log(1 - R^2)$$

where $h(\cdot)$ denotes differential entropy. $\square$

**Remark 4.2**. The Gaussian assumption is not restrictive. For non-Gaussian errors, R² remains a monotone proxy for MI under mild regularity conditions (the KL divergence between $(X,Y)$ and $X \otimes Y$ is bounded below by functions of R²). The key insight is that R² captures the explained fraction of variance, which is the primary quantity of interest regardless of distributional assumptions.

### 4.3 The Bias-Variance Decomposition in H

**Theorem 4.3** (Asymptotic R² Profile). Under the Heston model with CRRA
utility and stationary transformed features with spectral gap $\rho = \kappa$:

$$R^2(H) \sim \begin{cases} \alpha \cdot H \cdot dt & \text{for } H \ll 1/(2\kappa) \\ \beta \cdot e^{-2\kappa H \cdot dt} & \text{for } H \gg 1/(2\kappa) \end{cases}$$

where $\alpha$ depends on the risk premium signal-to-noise ratio and $\beta$ on the feature-label coupling strength.

The maximum occurs near:

$$H^* \approx \frac{1}{2\kappa \cdot dt} \cdot \frac{\log(\beta / \alpha)}{1 + \log(\beta / \alpha) / (2\kappa H^*)}$$

which for typical parameters gives $H^* \sim \frac{1}{4\kappa \cdot dt}$ (a quarter of the mixing time in steps).

*Proof sketch*. For small H: the utility label $y_H \approx 1 + (1-\gamma)[\pi(\mu-r) - \frac{\gamma}{2}\pi^2 V] H \cdot dt + O(\sqrt{H \cdot dt})$ noise. The signal scales as $H \cdot dt$ while the noise scales as $\sqrt{H \cdot dt}$, so SNR² ∝ $H \cdot dt$ and R² ∝ $H \cdot dt$.

For large H: the feature decorrelation $\text{Corr}(V_0, V_H) \sim e^{-\kappa H \cdot dt}$ means the explained variance decays as $e^{-2\kappa H \cdot dt}$. $\square$

### 4.4 Algorithm: MI-Based Horizon Selection

```python
def select_horizon_mi(dynamics_fn, utility_fn, feature_fn,
                      H_candidates, n_rollouts, n_landmarks, seed,
                      alpha_ridge=0.01):
    """Select H by maximizing held-out R² ≈ proxy for MI(features; label | π).

    Args:
        dynamics_fn: (state, action, noise) → new_state
        utility_fn: wealth_ratio → utility label
        feature_fn: state → feature vector (must produce stationary features)
        H_candidates: list of horizons to test (in steps)
        n_rollouts: number of (state, pi, outcome) triplets per H
        n_landmarks: number of RBF landmarks
        seed: random seed for reproducibility

    Returns:
        best_H: horizon maximizing held-out R²
        R2_scores: dict mapping H → held-out R²
    """
    R2_scores = {}

    for H in H_candidates:
        # 1. Generate rollouts with constant π held for H steps
        states, pis, wealth_ratios = generate_rollouts(
            dynamics_fn, n_rollouts, H, seed)

        # 2. Compute utility labels
        y = utility_fn(wealth_ratios)

        # 3. Extract stationary features
        features = feature_fn(states)

        # 4. Train/test split (80/20)
        n_train = int(0.8 * len(y))
        idx = np.random.RandomState(seed + 1).permutation(len(y))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        # 5. Product kernel features
        landmarks = np.quantile(features[train_idx],
                               np.linspace(0, 1, n_landmarks), axis=0)
        sigma = 1.5 * np.median(np.diff(landmarks, axis=0), axis=0)

        Psi_train = rbf_features(features[train_idx], landmarks, sigma)
        Psi_test = rbf_features(features[test_idx], landmarks, sigma)

        Phi_train = product_features(Psi_train, pis[train_idx])
        Phi_test = product_features(Psi_test, pis[test_idx])

        # 6. Ridge regression
        w = np.linalg.solve(
            Phi_train.T @ Phi_train + alpha_ridge * np.eye(Phi_train.shape[1]),
            Phi_train.T @ y[train_idx])

        # 7. Held-out R²
        y_pred = Phi_test @ w
        ss_res = np.var(y[test_idx] - y_pred)
        ss_tot = np.var(y[test_idx])
        R2_scores[H] = 1 - ss_res / ss_tot

    best_H = max(R2_scores, key=R2_scores.get)
    return best_H, R2_scores
```

### 4.5 Diagnostic: When the Method Fails Honestly

**Proposition 4.4** (Failure Detection). If $\max_H R^2(H) < \epsilon$ for a threshold $\epsilon > 0$ (e.g., $\epsilon = 0.05$), then either:

1. The feature transform $\phi$ is insufficient (wrong features)
2. The utility function has no state dependence (constant $\pi^*$ is optimal)
3. The signal is below the noise floor for the given sample size

In all cases, the method honestly reports "insufficient signal" rather than producing overfit artifacts.

*Proof*. If R² < ε for all H, then $I(\text{features}; \text{label} | \pi) < -\frac{1}{2}\log(1-\epsilon) \approx \epsilon/2$ nats. The MI bound on estimation error (via Fano's inequality) shows that no estimator can achieve relative error below $\sqrt{2/\epsilon}$ with high probability. $\square$

---

## Part V: The Optimal Ergodic Transform

### 5.1 The Variational Problem

Given non-stationary observations $Y_t$ (e.g., prices), we seek a transform $\phi$ that:
- Produces stationary features: $\phi(Y_{(-\infty,t]})$ has a well-defined invariant measure
- Maximizes information: $I(\phi(Y_{(-\infty,t]}); V_t)$ is maximized, where $V_t$ is the latent state

**Definition 5.1** (Optimal Ergodic Transform). The optimal ergodic transform $\phi^*$ solves:

$$\phi^* = \arg\max_{\phi \in \mathcal{E}} I(\phi(Y_{(-\infty,t]}); V_t)$$

where $\mathcal{E} = \{\phi : \phi(Y)_t \text{ is ergodic}\}$ is the class of ergodic transforms.

### 5.2 EFM Signatures as a Solution

**Theorem 5.2** (EFM Signatures Are Universal Ergodic Features). Let $Y_t$ be a continuous semimartingale with bounded moments. Let $\mathbb{Y}^{\boldsymbol{\lambda}}_t$ be the truncated EFM signature (up to level $N$) of the time-augmented path $(t, Y_t)$. Then:

1. $\mathbb{Y}^{\boldsymbol{\lambda}}_t$ is ergodic for any $\boldsymbol{\lambda} \succ 0$ (by Theorem 2.6)
2. For $N$ sufficiently large, $\phi^*$ can be approximated by a linear functional on $\mathbb{Y}^{\boldsymbol{\lambda}}_t$ (by Theorem 2.4, property 3)
3. The approximation error vanishes as $N \to \infty$ and $\boldsymbol{\lambda} \to 0$

*Proof sketch*.

Step 1: By the universal approximation property (Theorem 2.4.3), any continuous fading-memory functional $F(Y_{(-\infty,t]})$ can be approximated by $\langle \ell, \mathbb{Y}^{\boldsymbol{\lambda}}_t \rangle$ for some linear functional $\ell$ on the truncated tensor algebra.

Step 2: The optimal filter $\mathbb{E}[V_t | Y_{(-\infty,t]}]$ is a continuous functional of the observation path (under standard regularity: Lipschitz coefficients, non-degenerate observations). By Step 1, it is approximable by EFM signature features.

Step 3: The MI $I(\phi(Y); V_t)$ is maximized when $\phi(Y) = \mathbb{E}[V_t | Y_{(-\infty,t]}]$ (data processing inequality: any function of the observations has MI ≤ the conditional expectation). Since EFM signatures can approximate this conditional expectation, they achieve near-optimal MI. $\square$

**Remark 5.3** (The $\boldsymbol{\lambda}$ Selection Problem). The decay parameter $\boldsymbol{\lambda}$ controls:
- $\boldsymbol{\lambda}$ large → fast forgetting → strongly ergodic but low MI
- $\boldsymbol{\lambda}$ small → slow forgetting → high MI but weak ergodicity

The optimal $\boldsymbol{\lambda}^*$ can itself be selected by the MI/R² criterion of Section IV. In practice, $\gamma = e^{-\lambda \cdot dt} \in [0.90, 0.99]$ gives a manageable search range.

### 5.3 Connection to Bakry-Émery via the Generator

**Proposition 5.4** (Spectral Gap of EFM Process). The generator of the level-1 EFM signature $Z_t = \int_{-\infty}^t e^{-\lambda(t-s)} dX_s$ is that of an OU process:

$$L_Z f(z) = -\lambda z \cdot f'(z) + \frac{\sigma^2}{2\lambda} f''(z)$$

This has spectral gap $\rho = \lambda$ and invariant measure $\mu_Z = \mathcal{N}(0, \sigma^2/(2\lambda))$.

*Proof*. Direct computation. The Bakry-Émery criterion gives $\Gamma_2(f) = \lambda \Gamma(f) + \frac{\sigma^2}{2\lambda}(f'')^2 \geq \lambda \Gamma(f)$. $\square$

**Corollary 5.5** (Mixing Time ↔ Decay Rate). The EFM signature process mixes at rate $\lambda$:

$$\text{Var}_{\mu_Z}(P_t f) \leq e^{-2\lambda t} \text{Var}_{\mu_Z}(f)$$

Therefore, the horizon selection of Part IV with EFM features has:
- Maximum useful horizon: $H_{\max} \sim 1/(2\lambda \cdot dt)$ steps
- Optimal horizon: $H^* \sim 1/(4\lambda \cdot dt)$ steps (from Theorem 4.3)

This provides a **closed-form initial estimate** for the horizon search grid:

$$H_{\text{candidates}} = \left\{\frac{1}{8\lambda \cdot dt}, \frac{1}{4\lambda \cdot dt}, \frac{1}{2\lambda \cdot dt}, \frac{1}{\lambda \cdot dt}\right\}$$

### 5.4 The Hida-Malliavin Connection

The EFM signature connects to Hida white noise analysis (see `docs/hida_malliavin_signature_unification.md`) through the chaos expansion. At level $n$, the EFM signature is:

$$\mathbb{X}^{\boldsymbol{\lambda}, n}_t = I_n(f^{\boldsymbol{\lambda}}_n(\cdot, t))$$

where $I_n$ is the $n$-th iterated Itô integral and $f^{\boldsymbol{\lambda}}_n(u_1, \ldots, u_n, t) = \prod_{k=1}^n e^{-\lambda^{i_k}(t - u_k)} \mathbf{1}_{u_k < t}$ is the exponentially decaying kernel.

**Proposition 5.6** (Chaos Expansion of EFM). The EFM signature lives in the Wiener chaos:

$$\mathbb{X}^{\boldsymbol{\lambda}}_t \in \bigoplus_{n=0}^{\infty} \mathcal{H}_n$$

where $\mathcal{H}_n$ is the $n$-th Wiener chaos. Each level-$n$ component is orthogonal to all other levels (in $L^2$), which is why ridge regression on EFM features does not suffer from multicollinearity between levels.

**Remark 5.7** (Malliavin Derivative and Mixing). The Malliavin derivative $D_s \mathbb{X}^{\boldsymbol{\lambda}, n}_t$ decays as $e^{-\lambda_{\mathbf{v}}(t-s)}$ where $\lambda_{\mathbf{v}} = \sum_k \lambda^{i_k}$. This exponential decay of the Malliavin derivative is precisely the Hairer-Mattingly condition for unique ergodicity of the EFM process. The rate of decay equals the spectral gap $\rho = \min(\lambda^1, \ldots, \lambda^d)$, closing the loop between:

1. **Bakry-Émery** (spectral gap from $\Gamma_2 \geq \rho \Gamma$)
2. **Malliavin** (decay of $D_s X_t$ as $|t-s| \to \infty$)
3. **EFM parameter** (the forgetting rate $\boldsymbol{\lambda}$)

---

## Part VI: Practical Protocol

### 6.1 Complete Pipeline

Given observations $Y_t$ from an unknown process:

1. **Transform**: Compute EFM signature features $\mathbb{Y}^{\boldsymbol{\lambda}}_t$ at level $N=2$ with initial $\lambda = -\log(0.94)/dt$ (from empirical RecSig-RLS calibration)

2. **Select $\lambda$**: Grid search over $\gamma \in \{0.90, 0.92, 0.94, 0.96, 0.98\}$, select by held-out R² on a pilot regression

3. **Select $H$**: Using the selected $\lambda$, run MI-based horizon selection (Algorithm 4.4) over $H \in \{H^*/4, H^*/2, H^*, 2H^*\}$ where $H^* = 1/(4\lambda \cdot dt)$

4. **Learn**: fit a low-dimensional predictive head on the EFM state at the
   selected $H$

5. **Control**: for current applications, prefer a reference-conditioned local
   residual controller rather than a global product-kernel policy fit

6. **Validate**: check posterior support for improvement / concavity on the
   chosen trust region, rather than only a global point estimate

### 6.2 Computational Complexity

| Operation | Cost per step | Notes |
|-----------|--------------|-------|
| EFM level-1 ($d$ channels) | $O(d)$ | Just exponential decay + add |
| EFM level-2 ($d^2$ terms) | $O(d^2)$ | Update matrix + outer product |
| RBF features ($K$ landmarks) | $O(Kd)$ | Distance computation |
| Product features | $O(3K)$ | Multiply by $\pi, \pi^2$ |
| Policy extraction | $O(K)$ | Inner product + division |
| **Total per step** | $O(Kd + d^2)$ | Dominated by RBF if $K > d$ |

For $d=2$ (price + time), $K=30$ landmarks: ~120 FLOPs per step. Compare to BPF: $O(N_{\text{particles}}) \approx 500{,}000$ FLOPs.

---

## References

1. **Abi Jaber, E. & Sotnikov, D.** (2025). "Exponentially Fading Memory Signature." arXiv:2507.03700.
2. **Bakry, D. & Émery, M.** (1985). "Diffusions hypercontractives." Séminaire de probabilités XIX, LNM 1123, 177–206.
3. **Chevyrev, I. & Oberhauser, H.** (2022). "Signature moments to characterize laws of stochastic processes." JMLR 23(176), 1–42.
4. **Hairer, M. & Mattingly, J.** (2006). "Ergodicity of the 2D Navier-Stokes equations with degenerate stochastic forcing." Ann. Math. 164, 993–1032.
5. **Bonnier, P. & Oberhauser, H.** (2020). "Signature cumulants, ordered partitions, and independence of stochastic processes." Bernoulli 26(4), 2452–2486.
6. **Nourdin, I. & Peccati, G.** (2012). *Normal Approximations with Malliavin Calculus.* Cambridge University Press.
7. **Jordan, R., Kinderlehrer, D. & Otto, F.** (1998). "The variational formulation of the Fokker-Planck equation." SIAM J. Math. Anal. 29(1), 1–17.

---

## Part VII: Approach II — Stationary Transformed-State Control for Non-Ergodic Finance Data

The earlier parts of this document established the rigorous transform theory.
This final part makes the control-theoretic point explicit.

Approach II in the repo is:

1. raw price/wealth levels may be non-ergodic and unsuitable for
   invariant-measure regression;
2. choose a transform of the observation history that is stationary or
   approximately stationary;
3. learn prediction and local control in that transformed state.

This is the route that should scale beyond homothetic finance.

### 7.1 Why Raw Heston Levels Are the Wrong Invariant-Measure Coordinates

### Proposition 7.1 (Heston Splits into an Ergodic Factor and a Non-Ergodic Level)

Consider the Heston system

$$
\begin{aligned}
dX_t &= \left(\mu - \frac{1}{2}V_t\right)dt + \sqrt{V_t}\,dB_t^1, \\
dV_t &= \kappa(\theta - V_t)dt + \xi\sqrt{V_t}\,dB_t^2,
\qquad
d\langle B^1, B^2\rangle_t = \rho\,dt,
\end{aligned}
$$

where $X_t = \log S_t$.

Assume $\kappa > 0$, $\theta > 0$, and the standard positivity condition
$2\kappa\theta \ge \xi^2$.

Then:

1. the variance factor $V_t$ is positive recurrent and admits a unique
   invariant Gamma law;
2. the log-price $X_t$ has no invariant probability measure on $\mathbb{R}$;
3. therefore the joint level process $(X_t, V_t)$ has no invariant
   probability measure on $\mathbb{R}\times \mathbb{R}_+$.

#### Proof

For the CIR variance factor,

$$
dV_t = \kappa(\theta - V_t)dt + \xi\sqrt{V_t}\,dB_t^2,
$$

the drift is restoring and the diffusion coefficient is sublinear in $V$.
Classical CIR theory gives a unique invariant Gamma distribution with mean
$\theta$ and exponential convergence to equilibrium under the stated
parameters.

Now consider $X_t$.  By integrating the SDE,

$$
X_t - X_0
=
\int_0^t \left(\mu - \frac{1}{2}V_s\right)ds
+
\int_0^t \sqrt{V_s}\,dB_s^1.
$$

The martingale term has quadratic variation

$$
\left\langle \int_0^\cdot \sqrt{V_s}\,dB_s^1 \right\rangle_t
=
\int_0^t V_s\,ds.
$$

Since $V_t$ is ergodic with stationary mean $\theta > 0$, the ergodic theorem
implies

$$
\frac{1}{t}\int_0^t V_s\,ds \to \theta
\qquad \text{a.s.}
$$

Hence the quadratic variation grows asymptotically like $\theta t$, so
$X_t$ acquires diffusive spread of order $\sqrt{t}$.  In particular,
$\mathrm{Var}(X_t)$ is unbounded as $t \to \infty$, which rules out tightness
of the marginal laws of $X_t$ and therefore rules out an invariant probability
measure on $\mathbb{R}$.

If $(X_t, V_t)$ admitted an invariant law on
$\mathbb{R}\times\mathbb{R}_+$, then its $X$-marginal would be invariant for
$X_t$, contradicting the previous conclusion. $\square$

#### Remark 7.2

This proposition is the clean mathematical answer to the “is Heston ergodic?”
question:

- the latent variance factor is ergodic;
- the raw log-price level is not;
- the full level state is therefore not.

So any invariant-measure theory applied directly to raw price or wealth levels
is mis-specified from the outset.

### 7.2 A General Stationary-Transform Principle

Approach II does not ask raw levels to be ergodic.  It asks whether the
**transformed state** can be made ergodic.

### Proposition 7.3 (Exponentially Fading Transform of a Stationary-Increment Observable)

Let $R_t$ be a square-integrable semimartingale with stationary increments and
zero mean increments.  Fix $\lambda > 0$ and define

$$
Z_t := \int_{-\infty}^{t} e^{-\lambda (t-s)}\, dR_s.
$$

Then:

1. $Z_t$ is strictly stationary;
2. $Z_t$ is Markov whenever $R_t$ is an Itô process driven by white noise;
3. $Z_t$ solves the linear stochastic evolution

$$
dZ_t = -\lambda Z_t\,dt + dR_t.
$$

#### Proof

For stationarity, let $\tau_h$ denote time-shift by $h$.  Then

$$
Z_{t+h}
=
\int_{-\infty}^{t+h} e^{-\lambda(t+h-s)}\,dR_s.
$$

By the substitution $u=s-h$,

$$
Z_{t+h}
=
\int_{-\infty}^{t} e^{-\lambda(t-u)}\, dR_{u+h}.
$$

Since $R$ has stationary increments, the law of the shifted increment field
$dR_{u+h}$ equals that of $dR_u$.  Hence the finite-dimensional distributions
of $(Z_{t_1+h},\ldots,Z_{t_n+h})$ equal those of $(Z_{t_1},\ldots,Z_{t_n})$.

To obtain the dynamics, differentiate under the integral sign:

$$
\begin{aligned}
Z_{t+dt}
&=
\int_{-\infty}^{t} e^{-\lambda(t+dt-s)}\,dR_s
+
\int_t^{t+dt} e^{-\lambda(t+dt-s)}\,dR_s \\
&=
(1-\lambda dt)Z_t + dR_t + o(dt),
\end{aligned}
$$

which yields

$$
dZ_t = -\lambda Z_t\,dt + dR_t.
$$

When $dR_t$ is an Itô increment driven by white noise, this is a Markov SDE in
$Z_t$. $\square$

#### Remark 7.4

Proposition 7.3 is the level-1 version of the EFM story in Parts II and V.
The point is not that every transform is stationary.  The point is that there
exists a mathematically controlled **family** of transforms with:

- explicit memory parameter $\lambda$,
- explicit Markov realization,
- and explicit spectral gap $\rho = \lambda$ in the simplest case.

### Corollary 7.5 (Why EFM Signatures Are the Natural Generalization)

For time-augmented Brownian input, the full EFM signature process is
stationary, Markov, and exponentially ergodic by Theorems 2.5–2.7.  Therefore
EFM signatures are the higher-order nonlinear analogue of the linear fading
transform in Proposition 7.3.

#### Proof

This is exactly the content of Theorems 2.5–2.7.  Proposition 7.3 identifies
the level-1 coordinate as a scalar OU-type process.  The cited theorems extend
the same stationarity/Markov picture to the tensor-valued EFM signature.
$\square$

#### Remark 7.6

For general financial semimartingales, the exact theorem may require extra
assumptions not yet proved in this repo.  But the design principle remains:
construct a transformed state with a controlled forgetting kernel and learn the
controller there, not on the raw non-ergodic level coordinates.

### 7.3 Control Reduction on the Transformed State

### Proposition 7.7 (Control on a Stationary Sufficient Transform)

Let $Y_{(-\infty,t]}$ be the observation history and let

$$
S_t = \mathcal T(Y_{(-\infty,t]})
$$

be a transformed state.  Assume:

1. $S_t$ is Markov;
2. $S_t$ is stationary under the uncontrolled or reference dynamics;
3. for every admissible future control sequence, the conditional law of future
   rewards and future observations given the observation history depends on the
   past only through $S_t$.

Then the original partially observed control problem is equivalent to a fully
observed control problem on $S_t$.

#### Proof

Assumption 3 says that $S_t$ is a sufficient statistic for control.  Therefore
for any admissible policy and any measurable future reward functional $G$,

$$
\mathbb E\!\left[
G \,\middle|\, Y_{(-\infty,t]}
\right]
=
\mathbb E\!\left[
G \,\middle|\, S_t
\right].
$$

Consequently, the continuation value at time $t$ depends on the full
observation history only through $S_t$.  Dynamic programming may therefore be
written with state variable $S_t$ rather than the entire path history.
Assumption 1 provides the Markov property needed for the semigroup/HJB
formulation, while Assumption 2 provides the invariant-measure framework needed
for stationary regression or horizon-selection arguments. $\square$

#### Remark 7.8

This is the precise mathematical role of Approach II:

- it is **not** merely a better feature engineering trick;
- it is the route that turns a non-ergodic observed path into a state on which
  invariant-measure learning is legitimate.

### 7.4 How Approach II Fits the Three-Route Program

The three routes now used in the repo should be interpreted as follows.

1. **Approach I**: exploit exact homothetic structure and reduce to an ergodic
   latent factor whenever possible.  This is the clean Heston/CRRA benchmark.
2. **Approach II**: when raw levels are non-ergodic, learn or construct a
   stationary transformed state and do prediction/control there.
3. **Approach III**: if invariant-measure theory is too rigid, move to a
   finite-horizon local semigroup theory on the transformed or belief state.

Approach II is therefore the most general **representation layer** in the
current theory stack.

### Remark 7.9 (Why Approach II Is the Best Generalization Target)

Approach I is exact but structurally narrow.  Approach III is the likely
long-run endgame but requires delicate local semigroup arguments and good local
response estimators.  Approach II sits between them:

- broad enough to apply beyond homothetic finance,
- structured enough to exploit stationarity and spectral-gap arguments,
- and implementable with existing EFM / recurrent-signature machinery.

That is why the current recommended program is:

- use Approach I as the fast benchmark,
- build reusable tooling around Approach II,
- and treat Approach III as the later theoretical completion of the stack.

---

## Part VIII: Kernel Controller Review and Current Recommendation

The repo now contains several distinct kernel-control patterns.  They should
not be treated as interchangeable.

### 8.1 Three Kernel Architectures in the Repo

The implemented kernel routes are:

1. **Global joint state-action kernel tensor**
   - historical finance line;
   - examples: `finance/experiments/merton_kronic_kernel_tensor.py`,
     the older product-kernel discussion in this note, and earlier
     `\Psi(x) \otimes [1,\pi,\pi^2]` regressions;
   - main idea: fit a single global action-value surface over the raw action
     coordinate.

2. **Pure local kernel control from scratch**
   - current option-market-making local controller line;
   - example: `src/applications/option_mm/local_kernel_controller.py`;
   - main idea: fit a one-step reward landscape directly on
     `(z_t, u_t) \mapsto r_t` with no strong prior controller.

3. **Reference-conditioned local residual kernel control**
   - current most promising architecture for transfer to Heston and other
     partially observed finance problems;
   - clearest implemented template:
     `src/applications/option_mm/hybrid_residual_controller.py`;
   - main idea: choose a strong nominal controller $a_{\mathrm{ref}}(z)$,
     define a low-dimensional local overlay $u$, and learn only the residual
     improvement around that controller on a compact trust region.

The first route is the most expressive globally, but it is also the most
exposed to extrapolation and weak-signal failure.  The second route is the
cleanest conceptually, but it is data-hungry because the kernel must discover
both the baseline and the correction.  The third route inherits the useful
part of the second route while avoiding the worst failure mode of the first.

### Proposition 8.1 (Why the Global Joint Kernel Is the Wrong Default)

Suppose the true action-value or continuation map has a decomposition

$$
Q(z,a) = Q_{\mathrm{dom}}(z,a) + R(z,a),
$$

where:

1. $Q_{\mathrm{dom}}$ contains the dominant global geometry in the action
   variable (for example, the myopic Merton curvature in Heston, or the BBG
   quote geometry in OMM),
2. $R$ is the residual correction of actual interest.

If a kernel learner is asked to fit $Q$ directly over an unbounded or very
wide raw action domain, then the learner must simultaneously recover both the
dominant global curvature and the small residual correction.  This is strictly
harder than learning $R$ on a compact local overlay domain.

#### Proof

Let $\mathcal A$ denote the raw action domain and suppose the controller is
fit directly as a nonparametric map on $(z,a) \in \mathcal Z \times \mathcal
A$.  Then any approximation guarantee must control the full function
$Q_{\mathrm{dom}} + R$ on the relevant action region.  In finance problems of
the type considered in this repo, $Q_{\mathrm{dom}}$ often contains the large
global curvature that prevents pathological leverage or quoting behavior.  A
global kernel fit therefore spends most of its capacity recovering geometry
that is already known analytically or semi-analytically.

Now instead write

$$
a = a_{\mathrm{ref}}(z) + \Delta a(z,u),
\qquad
u \in \mathcal U,
$$

with compact $\mathcal U$, and define the residual objective

$$
\Delta Q(z,u)
:=
Q\!\big(z, a_{\mathrm{ref}}(z) + \Delta a(z,u)\big)
-
Q\!\big(z, a_{\mathrm{ref}}(z)\big).
$$

Since $\mathcal U$ is compact, the learner is now only asked to approximate
the residual map on a compact domain.  The dominant global curvature has been
factored into the reference controller, so the kernel head no longer needs to
reconstruct it from sparse data.  Therefore the residual problem is strictly
better conditioned for the current applications. $\square$

#### Remark 8.2

This proposition is the practical lesson of the old Merton kernel-tensor line.
The issue was not that kernels are inappropriate.  The issue was that the
kernel was asked to represent the **entire raw action geometry**, including
the part that should have been carried by the reference policy.

### Definition 8.3 (Reference-Conditioned Residual Kernel Controller)

Let $z_t$ denote a stationary or approximately stationary transformed state
from Approach II.  Let $a_{\mathrm{ref}}(z_t)$ be a nominal controller.  Let
$u \in \mathcal U \subset \mathbb R^q$ be a low-dimensional local coordinate,
and let

$$
a_t(u) = a_{\mathrm{ref}}(z_t) + \Delta a(z_t, u).
$$

A **reference-conditioned residual kernel controller** models either

$$
R(z_t, u) := \mathbb E[\Delta Y_t(u) \mid z_t]
$$

for a paired local response label $\Delta Y_t(u)$, or the full posterior law
of $\Delta Y_t(u)$, and then chooses $u$ on the trust region $\mathcal U$.

The key modeling choice is that the kernel head acts on the residual
coordinate $u$, not on the raw action itself.

### Proposition 8.4 (Paired Residual Labels Preserve the Control Objective)

Fix a transformed state $z$ and a common random input $\omega$.  Let
$Y(z,u,\omega)$ denote the short-horizon objective under local overlay $u$, and
define the paired residual label

$$
\Delta Y(z,u,\omega) := Y(z,u,\omega) - Y(z,0,\omega).
$$

Then:

1. the population residual target equals the improvement over the reference,
   i.e.

   $$
   \mathbb E[\Delta Y(z,u,\omega) \mid z]
   =
   \mathbb E[Y(z,u,\omega) \mid z]
   -
   \mathbb E[Y(z,0,\omega) \mid z];
   $$

2. the paired-label variance satisfies

   $$
   \operatorname{Var}(\Delta Y \mid z)
   =
   \operatorname{Var}(Y(z,u,\omega)\mid z)
   +
   \operatorname{Var}(Y(z,0,\omega)\mid z)
   -
   2\operatorname{Cov}(Y(z,u,\omega),Y(z,0,\omega)\mid z).
   $$

Hence common random numbers reduce residual-label variance whenever the two
outcomes are positively correlated.

#### Proof

The mean identity follows immediately from linearity of conditional
expectation.  The variance identity is the standard variance formula for a
difference of two random variables.  If the same noise is used for both
evaluations, the covariance term is typically positive in controlled diffusion
and market-making settings, so the variance of the paired difference is
smaller than the variance under independent rollout differences. $\square$

#### Remark 8.5

This is why the current Bayesian local-response work should keep common random
numbers fixed across the local overlay grid.  The goal is to estimate the
action-dependent correction, not to pay avoidable Monte Carlo noise.

### Proposition 8.6 (KRR / GP Residual Heads Are the Right Bayesian Kernel Form)

Let $f(z,u)$ denote the residual response map on the compact domain
$\mathcal Z \times \mathcal U$.  Put a Gaussian-process prior

$$
f \sim \mathcal{GP}(0, k((z,u),(z',u')))
$$

and observe noisy labels

$$
\Delta Y_i = f(z_i,u_i) + \varepsilon_i,
\qquad
\varepsilon_i \sim \mathcal N(0,\sigma_n^2).
$$

Then:

1. the posterior mean equals kernel ridge regression with ridge
   $\lambda = \sigma_n^2$;
2. the posterior variance supplies a principled abstention or risk-sensitive
   action rule on the local grid;
3. this is the correct Bayesian interpretation of the kernel residual head in
   the repo.

#### Proof

This is the standard GP/KRR equivalence: the posterior mean under a zero-mean
GP prior with kernel $k$ and homoskedastic Gaussian noise is

$$
\hat f(x)
=
k(x,X)\big(K(X,X)+\sigma_n^2 I\big)^{-1} y,
$$

which is exactly the kernel-ridge predictor with ridge $\lambda=\sigma_n^2$.
The posterior variance is also explicit:

$$
\operatorname{Var}(f(x)\mid D)
=
k(x,x) - k(x,X)\big(K+\sigma_n^2 I\big)^{-1}k(X,x),
$$

so posterior probability of improvement, lower credible bounds, and abstention
to the reference action are immediate. $\square$

#### Remark 8.7

This proposition matters for the repo’s current standards.  Kernel methods in
the controller should be framed as **Bayesian residual models**, not just as
point-estimate regressors with ad hoc regularization.

### 8.2 Current Recommendation for the Repo

For the current Heston and transformed-state lines, the most promising kernel
architecture is:

1. **Representation**: build a stationary transformed state $z_t$ using
   Approach II machinery (EFM, recurrent signatures, Kalman-like filters, or
   other stationary lifted states).
2. **Reference controller**: choose a strong nominal controller
   $a_{\mathrm{ref}}(z_t)$.
   - Heston benchmark: myopic Merton action;
   - OMM benchmark: BBG-style controller.
3. **Low-dimensional overlay**: define a compact local coordinate $u$ around
   the reference policy.
4. **Residual label**: fit paired local residual improvement, not the raw
   global objective.
5. **Kernel head**: use GP/KRR on $(z_t,u)$, interpreted Bayesianly.
6. **Decision rule**: choose the overlay by posterior improvement /
   lower-credible-bound logic, with abstention to the reference action as a
   first-class outcome.

### Remark 8.8 (What This Means for Current Files)

The codebase review supports the following interpretation:

- `finance/experiments/merton_kronic_kernel_tensor.py`
  is an important historical prototype, but it should no longer be treated as
  the mainline kernel architecture for finance control.
- `src/applications/option_mm/local_kernel_controller.py`
  remains useful as a pure from-scratch benchmark and as a negative-control
  reference for data hunger.
- `src/applications/option_mm/hybrid_residual_controller.py`
  is the clearest existing implementation pattern for the recommended route:
  transformed state, strong prior controller, low-dimensional perturbation,
  local kernel residual.

### Remark 8.9 (Heston Instantiation)

For Heston/CRRA, the recommendation becomes:

$$
\pi_t = \pi_{\mathrm{myopic}}(\hat V_t)\big(1 + u_t\big),
$$

with $\pi_{\mathrm{myopic}}(\hat V_t) = (\mu-r)/(\gamma \hat V_t)$,
$u_t$ low-dimensional, and the kernel head learning the residual value or
short-horizon improvement over the myopic policy on a compact overlay set.

This is more faithful to the current theory stack than the earlier global raw
action-kernel fit.
