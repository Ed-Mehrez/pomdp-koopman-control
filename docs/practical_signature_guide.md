# Practical Guide: Applying Signature Theory

This guide connects the rigorous theory in `theory_signature_characteristic_functions.md` to practical implementations.

## Key Theoretical Results → Practical Implications

| Theorem | Statement | Practical Implication |
|---------|-----------|----------------------|
| **Thm 6.1** | RBF on robust signatures is characteristic | Use RBF kernel - it captures ALL distributional info |
| **Thm 5.2** | Lead-lag Lévy area = QV/2 | Direct variance estimation without model fitting |
| **Thm 6.5** | Gaussian-averaged char. func. = RBF | Theoretical justification for RBF choice |
| **Cor 1.5** | GARCH/ARCH are covered | No multi-lag embedding needed with RBF |

---

## Application 1: Volatility Estimation

### Direct Method (No Training Needed)

```python
def estimate_qv_direct(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Direct QV estimation using Theorem 5.2: |Lévy area| = QV/2

    This is NOT a model - it's a deterministic function of the path.
    """
    qv = np.full(len(returns), np.nan)
    for t in range(window, len(returns)):
        qv[t] = np.sum(returns[t-window:t]**2)  # = 2 * |Lévy area|
    return qv
```

**When to use**: Real-time variance monitoring, initial estimates, baseline comparisons.

### Learned Method (For Prediction)

```python
from sklearn.kernel_ridge import KernelRidge

def fit_signature_vol_model(train_returns, train_qv, window=20, gamma=0.5):
    """
    Fit RBF kernel model on signatures.

    Theory: RBF is characteristic (Thm 6.1) → captures all vol dynamics
    """
    # Extract signatures
    sigs = extract_signatures(train_returns, window)

    # Fit with RBF kernel (Theorem 6.5 justifies this choice)
    model = KernelRidge(kernel='rbf', gamma=gamma, alpha=0.1)
    model.fit(sigs, train_qv[window:window+len(sigs)])

    return model
```

**When to use**: Forward volatility prediction, option pricing, risk management.

---

## Application 2: Generator Estimation (kGEDMD)

### The Problem
For SDE `dX = b(X)dt + σ(X)dW`, the Koopman generator is:
```
L = b(x)∂/∂x + ½σ²(x)∂²/∂x²
```

Traditional methods need ensemble data to estimate `E[dX|X=x]`.

### Signature Solution

Signatures capture BOTH drift and diffusion from a single path:
- **Level 1** (displacement) → drift `b`
- **Level 2 Lévy area** → diffusion `σ²` (via QV/2)

```python
def fit_generator_from_single_path(X, dt, window=30, gamma=0.1):
    """
    Estimate generator without ensemble using signatures.

    Theory: Lead-lag signature provides both drift and diffusion.
    """
    features = []
    drift_targets = []
    diff_targets = []

    for t in range(window, len(X) - 1):
        # Signature captures path geometry
        path = np.column_stack([
            np.linspace(0, 1, window + 1),
            X[t-window:t+1]
        ])
        sig = compute_signature(path, level=2)
        features.append(sig)

        # Drift: first-order increment
        drift_targets.append((X[t+1] - X[t]) / dt)

        # Diffusion: squared increment (local QV)
        diff_targets.append((X[t+1] - X[t])**2 / dt)

    # Fit both with RBF kernel
    drift_model = KernelRidge(kernel='rbf', gamma=gamma).fit(features, drift_targets)
    diff_model = KernelRidge(kernel='rbf', gamma=gamma).fit(features, diff_targets)

    return drift_model, diff_model
```

---

## Application 3: When Path Shape Matters

The theory shows signatures are **redundant** when terminal value is sufficient (e.g., standard Kyle model where Y_T determines pricing).

**Signatures ADD VALUE when**:
1. **Temporary market impact**: Execution path affects cost
2. **Spoofing detection**: Order pattern reveals intent
3. **Stochastic volatility**: Vol path affects optimal strategy
4. **Regime detection**: Path shape predicts regime changes

### Example: Spoofing Detection

```python
def detect_spoofing(order_flow_path, threshold=2.0):
    """
    Detect spoofing using signature shape features.

    Spoofers create distinctive path patterns:
    - Large orders followed by cancellations
    - Symmetric up-down patterns

    Signatures capture these shapes naturally.
    """
    # Time-augmented path
    n = len(order_flow_path)
    path = np.column_stack([np.linspace(0, 1, n), order_flow_path])

    sig = compute_signature(path, level=3)

    # Lévy area captures "looping" behavior
    # High |Lévy area| with low displacement → suspicious
    displacement = sig[1:3]  # Level 1
    levy_area = sig[5]       # Level 2 antisymmetric part (for 2D)

    # Spoofing score: high area, low net displacement
    if np.linalg.norm(displacement) < 0.1:
        score = np.abs(levy_area) / (np.linalg.norm(displacement) + 0.01)
    else:
        score = np.abs(levy_area) / np.linalg.norm(displacement)

    return score > threshold
```

---

## Choosing the Right Kernel Parameter γ

The RBF kernel `k(x,y) = exp(-γ||S(x) - S(y)||²)` has one hyperparameter.

### Rules of Thumb

| Data Scale | Suggested γ | Reasoning |
|------------|-------------|-----------|
| Normalized returns | 0.1 - 1.0 | Moderate smoothing |
| Raw prices | 0.001 - 0.01 | Larger values, need more smoothing |
| High-frequency | 1.0 - 10.0 | Small increments, less smoothing |

### Cross-Validation Approach

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'gamma': [0.01, 0.1, 0.5, 1.0, 5.0]}
cv_model = GridSearchCV(
    KernelRidge(kernel='rbf', alpha=0.1),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
cv_model.fit(signatures, targets)
best_gamma = cv_model.best_params_['gamma']
```

---

## Common Pitfalls and Solutions

### 1. Signature Explosion
**Problem**: High truncation levels create very high-dimensional features.

**Solution**: Use level 2-3 for most applications. Theory shows this captures essential structure.

```python
# Good: Low level with RBF kernel provides sufficient expressiveness
sig = compute_signature(path, level=2)  # 1 + d + d² terms

# Bad: High level creates curse of dimensionality
sig = compute_signature(path, level=6)  # Millions of terms for d > 3
```

### 2. Normalization
**Problem**: Unnormalized signatures can fail to be characteristic for unbounded paths.

**Solution**: For potentially unbounded data (e.g., cumulative returns), use robust signatures or normalize:

```python
def normalize_path(path):
    """Normalize to bounded range."""
    return (path - path.mean(axis=0)) / (path.std(axis=0) + 1e-8)
```

### 3. Time Augmentation
**Problem**: Pure price signatures lose timing information.

**Solution**: Always include time as first coordinate:

```python
# Good: Time-augmented captures variable observation intervals
path = np.column_stack([time, price])

# Bad: Loses timing info
path = price.reshape(-1, 1)
```

---

## Summary: Decision Tree

```
Is terminal value a sufficient statistic?
├── YES → Use simple methods (regression on Y_T)
│         Examples: Standard Kyle model, option payoffs
│
└── NO → Use signatures
         │
         ├── Need variance estimate?
         │   └── Direct QV = Σ(Δr)² (Theorem 5.2)
         │
         ├── Need forward prediction?
         │   └── RBF kernel on signatures (Theorem 6.1)
         │
         └── Need generator (drift + diffusion)?
             └── Unified signature approach (single path)
```

---

## References

- Theory document: `docs/theory_signature_characteristic_functions.md`
- Implementation: `src/finance/signature_volatility.py`
- Tests: `tests/test_signature_sanity.py`, `tests/test_signature_multi_lag.py`
