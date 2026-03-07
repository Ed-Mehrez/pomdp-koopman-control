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

## References

- Hambly & Lyons (2010): Uniqueness for the signature of a path
- Chevyrev & Lyons (2016): Characteristic functions of measures on geometric rough paths
- Hida (1980): Brownian Motion (S-transform and white noise analysis)
- Kraft (2005): Optimal portfolios with stochastic short rate

