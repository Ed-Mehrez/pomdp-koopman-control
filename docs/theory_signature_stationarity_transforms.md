# Signature-Based Stationarity Transforms

## Problem Statement

Financial processes (prices, volatility) are often non-ergodic:
- **GBM**: Prices grow exponentially → no stationary distribution
- **Trending processes**: Drift dominates → running mean diverges
- **Regime-switching**: Parameters change → mixing fails

The Carré du Champ identity requires ergodicity for consistent generator estimation.
Rather than chopping paths into segments, we seek **transformations** g(X) such that
g(X_t) is approximately ergodic.

## Key Insight: Signatures Encode Transform Information

For a stochastic process X_t, the signature Sig(X_{[0,T]}) encodes all path information.
Under transformation Y = g(X), by Itô's lemma:

```
dY = g'(X)dX + ½g''(X)d⟨X⟩
```

The signature of Y relates to that of X through the chain rule structure.
**Crucially**: The growth pattern of signature terms reveals what g is needed.

## Diagnostic Framework

### Level-1 Signature Growth

For process X over window [0, T]:
- Sig^1(X) = X_T - X_0 (displacement)

| Growth Pattern | Interpretation | Transform |
|---------------|----------------|-----------|
| Sig^1 ~ O(√T) | Brownian-like | None (stationary increments) |
| Sig^1 ~ O(T) | Linear drift | Detrend: X - μt |
| Sig^1 ~ O(e^{λT}) | Exponential | Log: log(X) |
| Sig^1 ~ O(T^β), β>1 | Superlinear | Power: X^{1/β} |

### Level-2 Signature / Lévy Area Growth

For 2D process (t, X):
- Lévy area A = ½∫(t dX - X dt) captures "enclosed area"

| Area Growth | Interpretation | Transform |
|-------------|----------------|-----------|
| A ~ O(T^{3/2}) | Standard diffusion | None |
| A ~ O(e^{2λT}) | Exponential + drift | Log |
| A/Sig^1² bounded | Good scaling | None |
| A/Sig^1² → ∞ | Bad scaling | Rescale |

### Signature Variance Across Windows

Compute signatures on rolling windows [kW, (k+1)W]:
- Var(Sig^1_k) should be O(1) for stationary increments
- If Var(Sig^1_k) grows → non-stationarity

## The Hida Calculus Connection

In Hida calculus, the **S-transform** maps a random variable F to:
```
SF(ξ) = E[F · exp(∫ξ_t dW_t - ½∫ξ_t² dt)]
```

The expected signature E[Sig(X) | X_0 = x] is the path-space analog.
For a diffusion dX = μ(X)dt + σ(X)dW, the expected signature satisfies:
```
∂/∂t E[Sig(X_{[0,t]})] = L · E[Sig] + signature_drift_terms
```
where L is the generator.

**Key property**: If X is non-ergodic, E[Sig] diverges. But:
- E[Sig(log X)] converges for GBM
- E[Sig(X - μt)] converges for drifted BM

## Automatic Transform Detection Algorithm

```python
def detect_transform(X, window_size=252, n_windows=10):
    """
    Analyze signature growth to detect appropriate transform.

    Returns: ('none', 'log', 'sqrt', 'detrend', transform_params)
    """
    T = len(X)
    windows = np.array_split(X, n_windows)

    # 1. Compute level-1 signatures (displacements) per window
    displacements = [w[-1] - w[0] for w in windows]

    # 2. Check for exponential growth via log-displacement regression
    log_X = np.log(np.abs(X) + 1e-10)
    log_displacements = [np.log(w[-1]/w[0]) for w in windows if w[0] > 0]

    # 3. Variance of increments vs variance of log-increments
    var_dX = np.var(np.diff(X))
    var_d_logX = np.var(np.diff(log_X))

    # 4. Check if log stabilizes variance
    # For GBM: Var(dX) ~ X² but Var(d log X) ~ const
    cv_dX = np.std([np.var(np.diff(w)) for w in windows]) / np.mean([np.var(np.diff(w)) for w in windows])
    cv_d_logX = np.std([np.var(np.diff(np.log(w+1e-10))) for w in windows]) / np.mean([np.var(np.diff(np.log(w+1e-10))) for w in windows])

    if cv_d_logX < cv_dX * 0.5 and np.all(X > 0):
        return 'log', {'base': np.mean(X)}

    # 5. Check for linear trend
    t = np.arange(len(X))
    slope = np.polyfit(t, X, 1)[0]
    detrended = X - slope * t
    var_detrended = np.var(np.diff(detrended))

    if var_detrended < var_dX * 0.8:
        return 'detrend', {'slope': slope}

    return 'none', {}
```

## Signature-Based Transform Selection

More sophisticated approach using full signature structure:

```python
def signature_transform_score(X, transform='log'):
    """
    Score a transform by how well it stabilizes signature statistics.
    Lower score = more stationary.
    """
    if transform == 'log':
        Y = np.log(X + 1e-10)
    elif transform == 'sqrt':
        Y = np.sqrt(np.abs(X))
    elif transform == 'none':
        Y = X
    else:
        Y = X

    # Compute signatures on rolling windows
    window = 63  # quarterly
    n_windows = len(Y) // window

    sigs = []
    for i in range(n_windows):
        path = np.column_stack([np.arange(window), Y[i*window:(i+1)*window]])
        sig = compute_log_signature(path, level=2)
        sigs.append(sig)

    sigs = np.array(sigs)

    # Score = coefficient of variation of signature norms + trend in norms
    norms = np.linalg.norm(sigs, axis=1)
    cv = np.std(norms) / (np.mean(norms) + 1e-10)
    trend = np.abs(np.polyfit(np.arange(len(norms)), norms, 1)[0])

    return cv + trend * 10
```

## Theoretical Justification

### Theorem (Informal)
For a diffusion X_t with generator L, the expected signature satisfies:
```
E[Sig^{(n)}(X_{[0,t]}) | X_0 = x] = e^{tL} · initial_signature_terms + corrections
```

If L has no stationary distribution (non-ergodic), the exponential e^{tL} grows.
For the transformed process Y = g(X) with generator L_Y:
```
L_Y f = g'^2(x) · ½σ²(x) f'' + [g'(x)μ(x) + ½g''(x)σ²(x)] f'
```

**The transform g that makes L_Y have a stationary distribution is the one that
makes the expected signature of Y bounded.**

### Example: GBM
- X_t follows dX = μX dt + σX dW (no stationary dist)
- Y_t = log X_t follows dY = (μ - σ²/2)dt + σdW (Brownian motion)
- E[Sig^1(Y)] = (μ - σ²/2)t (linear, not exponential)
- After detrending: Z_t = Y_t - (μ - σ²/2)t is stationary

## Implementation Notes

1. **Positivity**: Log transform requires X > 0; use shifted log or sqrt for negative values
2. **Parameter estimation**: The detrending slope μ should be estimated robustly (median regression)
3. **Regime-awareness**: If signature statistics change abruptly, segment first then transform
4. **Validation**: After transform, verify that CdC generator extraction works

## Connection to Resolvent

When no transform makes the process ergodic (true regime-switching), use the resolvent:
```
R_λ f(x) = E[∫_0^∞ e^{-λt} f(X_t) dt | X_0 = x]
```

This exists even for non-ergodic processes (for λ > growth rate).
The signature analog is the **Laplace-signature**:
```
LS_λ(X) = ∫_0^∞ e^{-λt} Sig(X_{[0,t]}) dt
```

This provides a path-space "regularized" feature that's bounded even for growing processes.

## Theoretical Foundation: Ergodicity and Signature Growth

### From Lyapunov Functions to Signatures

A diffusion dX = μ(X)dt + σ(X)dW is **ergodic** if and only if there exists a
Lyapunov function V: R → R⁺ such that outside a compact set:

```
LV(x) ≤ -c·V(x) + K    for some c > 0, K > 0
```

where L is the generator: Lf = μf' + ½σ²f''.

**Example (OU is ergodic)**: For dX = -κX dt + σdW, take V(x) = x²:
```
LV = -2κx² + σ² ≤ -κx² + σ²  for |x| > σ/√κ  ✓
```

**Example (GBM is NOT ergodic)**: For dX = μX dt + σX dW, take V(x) = x²:
```
LV = 2μx² + σ²x² = (2μ + σ²)x²  (no upper bound)  ✗
```

### The Key Theorem: Signature Growth Characterizes Ergodicity

**Theorem** (Signature Growth and Mixing):
Let X_t be a Markov diffusion with generator L. Define the signature growth rate:
```
γ(X) := lim sup_{T→∞} E[||Sig(X_{[0,T]})||²] / T
```

Then:
1. If X is ergodic with spectral gap λ₁ > 0, then γ(X) < ∞
2. If X is transient or null-recurrent, then γ(X) = ∞

**Proof sketch**:
For level-1 signature (displacement), E[X_T - X_0 | X_0 = x] satisfies:
```
∂/∂T E[X_T | X_0] = E[μ(X_T) | X_0]
```

For ergodic X with stationary distribution π:
```
E[μ(X_T) | X_0] → ∫ μ(y) π(dy) = μ̄  as T → ∞
```
So E[X_T - X_0] ~ μ̄·T (linear growth), giving ||Sig¹||²/T ~ μ̄² = O(1).

For GBM: E[X_T | X_0 = x] = x·e^{μT} (exponential), so ||Sig¹||²/T → ∞.

Higher signature levels follow similarly via iterated expectations. ∎

### Optimal Transform via Spectral Gap Maximization

The **spectral gap** of generator L is:
```
λ₁(L) = inf_{f: Ef=0, Var(f)=1} ⟨-Lf, f⟩_{L²(π)}
```

Larger spectral gap = faster mixing = smaller γ(X).

For transformed process Y = g(X), the new generator is (by Itô):
```
L_g f(y) = ½[g'(g⁻¹(y))·σ(g⁻¹(y))]² f''(y)
         + [g'·μ + ½g''·σ²](g⁻¹(y)) f'(y)
```

**Variational Principle**: The optimal transform maximizes the spectral gap:
```
g* = argmax_g λ₁(L_g)
```

Since λ₁ is hard to compute, we use γ(g(X)) as a **computable proxy**:
```
g* = argmin_g γ(g(X)) = argmin_g lim sup_{T→∞} E[||Sig(g(X)_{[0,T]})||²] / T
```

This is exactly minimizing signature growth rate — now theoretically justified.

### MDL Derivation of the Joint Objective

For segmentation, we use the **Minimum Description Length** principle.
The description length of data X under model M = (g, τ₁,...,τ_K) is:

```
DL(X | M) = -log P(X | M) + log |M|
```

**Data term**: Under ergodicity on each segment, the log-likelihood scales as:
```
-log P(X | M) ∝ Σᵢ (τᵢ₊₁ - τᵢ) · γ(g(X) on [τᵢ, τᵢ₊₁])
```
This is because the transition density concentrates around the stationary
distribution at rate e^{-λ₁·t}, and γ ~ 1/λ₁.

**Model complexity term**:
```
log |M| = log(K) + log(complexity of g)
```

Minimizing DL gives:
```
J(g, τ, K) = Σᵢ T_i · γᵢ(g) + λ₁·K + λ₂·complexity(g)
```

where T_i = τᵢ₊₁ - τᵢ and γᵢ is the growth rate on segment i.

**Normalizing by segment length** (since we measure γ = ||Sig||²/T):
```
J(g, τ, K) = Σᵢ ||Sig(g(X)_{[τᵢ,τᵢ₊₁]})||² / T_i + λ₁·K + λ₂·complexity(g)
```

This is our objective, now derived from first principles via MDL.

## Hida-Malliavin Interpretation

### S-Transform and Expected Signature

In Hida's white noise analysis, the **S-transform** of a random variable F is:
```
SF(ξ) = E[F · exp(∫ξ_t dW_t - ½∫ξ_t² dt)]
```

The expected signature E[Sig(X_{[0,T]})] relates to the S-transform through:
```
E[Sig(X)] = S[Sig(X)](ξ=0)
```

For a diffusion, E[Sig] satisfies a PDE driven by the generator:
```
∂/∂T E[Sig(X_{[0,T]}) | X_0 = x] = L_x · E[Sig] + boundary terms
```

**Stationarity condition**: E[Sig]/T → const requires L to have spectral gap > 0.

### Wick Renormalization and Log-Signatures

**Wick products** :·: subtract the "expected pairing":
```
:W_t · W_s: = W_t · W_s - min(t,s)
E[:F:] = 0  for F ≠ constant (chaos decomposition)
```

The signature's iterated integrals decompose as:
```
∫∫_{s<t} dX_s dX_t = ½(X_T - X_0)² - ½⟨X⟩_T
                    = symmetric part + Lévy area
```

The **log-signature** extracts the Lévy area (antisymmetric part), which is
already "Wick-renormalized" in the sense that:
```
E[Lévy area | X stationary] = 0
```

**Insight**: The log-signature is the natural object for stationarity testing
because its expectation vanishes for stationary processes, while growing
signatures indicate non-stationarity.

### Malliavin Derivative and Optimal Transforms

The **Malliavin derivative** D_t F measures sensitivity to Brownian perturbation:
```
D_t X_T = σ(X_t) · ∂X_T/∂W_t
```

For the transformed process Y = g(X):
```
D_t Y_T = g'(X_T) · D_t X_T
```

The optimal transform g satisfies a variational equation:
```
δ/δg E[||Sig(g(X))||²/T] = 0
```

Using Malliavin calculus, this becomes:
```
E[Sig' · D(Sig)] · δg = 0  for all variations δg
```

This gives a PDE for g in terms of the process statistics — though in practice
we solve it numerically via grid search over Box-Cox family.

### Parameterized Transform Families

**Box-Cox family** (1 parameter):
```
g_λ(x) = (x^λ - 1)/λ  for λ ≠ 0
       = log(x)        for λ = 0
```
Includes log (λ→0), sqrt (λ=0.5), identity (λ=1).

**Power transform** (2 parameters):
```
g_{α,β}(x) = sign(x - β)|x - β|^α
```
Handles shifts and different power laws.

**Sinh-arcsinh transform** (2 parameters, handles heavy tails):
```
g_{ε,δ}(x) = sinh(δ · arcsinh(x) - ε)
```

### Practical Algorithm

```python
def joint_optimize(X, lambda_seg=1.0, lambda_complexity=0.1):
    """
    Joint optimization over transform and segmentation.

    Stage 1: Grid search over Box-Cox λ
    Stage 2: For each λ, find optimal segmentation via DP
    Stage 3: Select (λ*, τ*) minimizing total objective
    """
    best_obj = float('inf')
    best_lambda = 1.0
    best_segments = [(0, len(X))]

    # Stage 1: Transform candidates
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        Y = box_cox_transform(X, lam)

        # Stage 2: Optimal segmentation via dynamic programming
        segments, seg_obj = optimal_segmentation_dp(
            Y,
            cost_fn=signature_growth_rate,
            penalty=lambda_seg
        )

        # Total objective
        complexity = abs(lam - 1.0)  # Distance from identity
        obj = seg_obj + lambda_complexity * complexity

        if obj < best_obj:
            best_obj = obj
            best_lambda = lam
            best_segments = segments

    return best_lambda, best_segments

def optimal_segmentation_dp(Y, cost_fn, penalty, min_seg_len=50):
    """
    Dynamic programming for optimal segmentation.

    DP[t] = min cost to segment Y[0:t]
    DP[t] = min_{s < t} { DP[s] + cost(Y[s:t]) + penalty }
    """
    T = len(Y)
    DP = np.full(T + 1, np.inf)
    DP[0] = 0
    parent = np.zeros(T + 1, dtype=int)

    for t in range(min_seg_len, T + 1):
        for s in range(0, t - min_seg_len + 1):
            seg_cost = cost_fn(Y[s:t])
            total = DP[s] + seg_cost + penalty
            if total < DP[t]:
                DP[t] = total
                parent[t] = s

    # Backtrack to get segments
    segments = []
    t = T
    while t > 0:
        s = parent[t]
        segments.append((s, t))
        t = s

    return segments[::-1], DP[T]

def signature_growth_rate(Y):
    """Cost function: signature norm / sqrt(length)."""
    if len(Y) < 10:
        return float('inf')
    path = np.column_stack([np.linspace(0, 1, len(Y)), Y])
    sig = compute_log_signature(path, level=2)
    # For stationary process, ||Sig|| ~ O(sqrt(T))
    # So ||Sig||² / T should be O(1)
    return np.linalg.norm(sig)**2 / len(Y)
```

### Theoretical Guarantee (Informal)

**Theorem** (Existence of optimal transform-segmentation):
For a continuous semimartingale X with bounded p-variation on [0,T],
there exists an optimal (g*, τ*) minimizing J(θ, τ, K) such that:
1. On each segment [τ_i, τ_{i+1}], g*(X) is ergodic
2. The number of segments K* ≤ C · (variation of X) / λ_1

The proof uses compactness of the transform family and the signature's
universal approximation property.

### When Each Approach Dominates

| Scenario | Transform Only | Segment Only | Joint |
|----------|---------------|--------------|-------|
| GBM (constant drift) | log ✓ | ✗ | log, 1 seg |
| OU (stationary) | none ✓ | ✗ | none, 1 seg |
| Regime-switching OU | ✗ | ✓ | none, K segs |
| GBM with drift change | ✗ | ✗ | log, K segs ✓ |
| CEV with β change | sqrt? | ✓ | power, K segs ✓ |

The joint approach is most valuable for **GBM-like processes with regime changes**
where neither pure transform nor pure segmentation works.

## References

- Hambly & Lyons (2010): Uniqueness for the signature of a path
- Chevyrev & Lyons (2016): Characteristic functions of measures on geometric rough paths
- Hida (1980): Brownian Motion (S-transform and white noise analysis)
- Kraft (2005): Optimal portfolios with stochastic short rate
- Nualart (2006): Malliavin Calculus and Related Topics (variational methods)
- Box & Cox (1964): An analysis of transformations (power transforms)

