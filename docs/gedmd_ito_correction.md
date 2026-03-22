# GEDMD for Stochastic Processes: The Ito Correction

## The Problem

When learning the Koopman operator/generator from stochastic process data, we must account for the **Ito correction term**. The generator for a diffusion:

$$dX_t = b(X_t)dt + \sigma(X_t)dW_t$$

is NOT just the drift:

$$\mathcal{L}f = \underbrace{b \cdot \nabla f}_{\text{drift}} + \underbrace{\frac{1}{2}\sigma\sigma^T : \nabla\nabla f}_{\text{Ito correction}}$$

**Key issue**: We don't have oracle access to $b$ or $\sigma$. Both must be estimated from data.

---

## Kramers-Moyal Formulae (What Klus Recommends)

Klus (2020) explicitly suggests the **Kramers-Moyal formulae** for estimating drift and diffusion from data. These are:

**First Kramers-Moyal coefficient (drift)**:
$$D^{(1)}(x) = \lim_{\tau \to 0} \frac{1}{\tau} \mathbb{E}[X_{t+\tau} - X_t \,|\, X_t = x] = b(x)$$

**Second Kramers-Moyal coefficient (diffusion)**:
$$D^{(2)}(x) = \lim_{\tau \to 0} \frac{1}{2\tau} \mathbb{E}[(X_{t+\tau} - X_t)^2 \,|\, X_t = x] = \frac{1}{2}a(x) = \frac{1}{2}\sigma(x)\sigma(x)^T$$

These are exactly the conditional moments of the increments!

---

## The Single-Path Problem (CRITICAL)

**Kramers-Moyal has a fundamental issue**: It requires conditional expectations $\mathbb{E}[\cdot | X_t = x]$, but with a SINGLE PATH we only observe each state value once (or rarely).

**With 1 trajectory:**
- Can't compute $\mathbb{E}[dX | X = x]$ directly
- Each state value appears at most a few times
- Conditional expectation is undefined without ensemble

**Workarounds:**

### 1. Local Kernel Averaging (What Klus Implicitly Assumes)
Group nearby states and assume smoothness:
$$\hat{b}(x) \approx \frac{\sum_i K_h(X_{t_i} - x) \cdot \frac{X_{t_{i+1}} - X_{t_i}}{\Delta t}}{\sum_i K_h(X_{t_i} - x)}$$

This works if:
- Process revisits similar regions (ergodicity)
- Drift/diffusion are smooth functions
- Enough data to populate kernel neighborhoods

### 2. Lead-Lag Lévy Area (Why Signatures Solve This!)

The lead-lag signature provides **path-by-path** diffusion estimation:
$$|\text{Lévy area}| = \frac{1}{2}\sum_i (\Delta X_i)^2 = \frac{QV}{2}$$

**Key insight**: This is just a SUM over the path increments!
- NO conditional expectation needed
- Works with a SINGLE path
- Computes σ² directly from path geometry
- This is why |Lévy area| correlates r=0.84 with true variance

### 3. Parametric Estimation
Assume functional form and estimate parameters:
- Heston: $b(v) = \kappa(\theta - v)$, $\sigma(v) = \xi\sqrt{v}$
- Estimate $(\kappa, \theta, \xi)$ via MLE or GMM
- Then plug into generator formula

---

## Finite-Difference Estimation

### 1. Drift Estimation

The conditional mean of increments gives the drift:

$$\hat{b}(x) = \lim_{\Delta t \to 0} \frac{\mathbb{E}[X_{t+\Delta t} - X_t \,|\, X_t = x]}{\Delta t}$$

**Practical estimator** (kernel regression):
$$\hat{b}(x) = \frac{\sum_i K_h(X_{t_i} - x) \cdot \frac{X_{t_i + \Delta t} - X_{t_i}}{\Delta t}}{\sum_i K_h(X_{t_i} - x)}$$

where $K_h$ is a kernel with bandwidth $h$.

### 2. Diffusion Coefficient Estimation

The conditional second moment gives the diffusion:

$$\hat{\sigma\sigma^T}(x) = \lim_{\Delta t \to 0} \frac{\mathbb{E}[(X_{t+\Delta t} - X_t)(X_{t+\Delta t} - X_t)^T \,|\, X_t = x]}{\Delta t}$$

**Practical estimator** (realized variance):
$$\hat{\sigma^2}(x) = \frac{\sum_i K_h(X_{t_i} - x) \cdot \frac{(X_{t_i + \Delta t} - X_{t_i})^2}{\Delta t}}{\sum_i K_h(X_{t_i} - x)}$$

---

## Why Standard GEDMD Fails for Stochastic Processes

Standard GEDMD fits:
$$g(X_{t+\Delta t}) \approx (I + \Delta t \cdot L) g(X_t)$$

This only captures the **expected evolution**, missing the diffusion term unless:
1. Observables $g$ include second-order features (QV, signature level 2)
2. The generator matrix explicitly includes the Ito correction

### The Heston Example

For Heston variance: $dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t$

- Generator: $\mathcal{L} = \kappa(\theta - v)\frac{\partial}{\partial v} + \frac{1}{2}\xi^2 v \frac{\partial^2}{\partial v^2}$
- Standard DMD only sees drift: $\kappa(\theta - v)$
- Missing diffusion term: $\frac{1}{2}\xi^2 v \frac{\partial^2}{\partial v^2}$

**Implication**: Eigenvalues from standard GEDMD are WRONG for stochastic systems!

---

## Corrected GEDMD for Diffusions

### Option 1: Augmented Feature Space

Include second-order features that capture diffusion:
```python
# Bad: Only first-order features
features = [1, v, v**2]

# Good: Include QV-sensitive features
features = [1, v, v**2,
            np.sum(dv**2),  # Realized QV
            levy_area,       # From lead-lag signature
           ]
```

### Option 2: Explicit Generator Fitting

Fit both drift and diffusion separately:

```python
def fit_generator(X, dt, kernel_bandwidth):
    """
    Fit generator L = b*d/dx + (1/2)*sigma^2*d^2/dx^2
    without oracle access to b or sigma.
    """
    # Increments
    dX = np.diff(X)

    # Drift estimate (local mean)
    b_hat = local_mean(X[:-1], dX / dt, bandwidth=kernel_bandwidth)

    # Diffusion estimate (local variance)
    sigma2_hat = local_var(X[:-1], dX**2 / dt, bandwidth=kernel_bandwidth)

    return b_hat, sigma2_hat
```

### Option 3: Signature Features with Lead-Lag

The lead-lag signature automatically captures both:
- Level 1: Drift (displacement)
- Level 2 Lévy area: $\frac{1}{2}QV$ (diffusion!)

This is why the corrected ablation study showed |Lévy area| = QV/2.

---

## Practical Implementation for Heston/Bates

For volatility estimation from price returns:

```python
def estimate_generator_features(returns, window=50):
    """
    Extract features that capture both drift and diffusion
    without oracle access.
    """
    features = []

    for t in range(window, len(returns)):
        w = returns[t-window:t]

        # Drift proxy (mean return)
        drift = np.mean(w)

        # Diffusion proxy (realized variance)
        rv = np.sum(w**2)

        # Signature features (captures both!)
        x = np.cumsum(w)
        ll_path = lead_lag_embed(x)
        levy_area = compute_levy_area(ll_path)

        features.append([
            drift,              # First moment
            rv,                 # Second moment (≈ σ²Δt)
            np.abs(levy_area),  # QV/2 from geometry
            levy_area,          # Signed (momentum)
        ])

    return np.array(features)
```

---

## Key Theoretical References

1. **Klus et al. (2020)**: "Kernel-Based Approximation of the Koopman Generator and Schrödinger Operator"
   - **Explicitly states** gEDMD requires estimating b and σ (page 4)
   - Suggests **Kramers-Moyal formulae** for FD estimation
   - Shows $d\phi_n(x) = \sum_i b_i \partial_i\phi_n + \frac{1}{2}\sum_{ij} a_{ij}\partial_{ij}\phi_n$
2. **Florens-Zmirou (1989)**: "Approximate discrete-time schemes for statistics of diffusion processes"
3. **Fan & Zhang (2003)**: "Estimation of diffusion processes via local polynomial regression"
4. **Hambly-Lyons (2010)**: "Uniqueness for the signature of a path of bounded variation"
   - Lead-lag Lévy area = QV/2

---

## Summary: Checklist for GEDMD on Stochastic Data

- [ ] **Don't assume oracle access** to drift b or diffusion σ
- [ ] **Estimate drift** from conditional mean of increments
- [ ] **Estimate diffusion** from conditional second moment
- [ ] **Include QV-sensitive features** (signature level 2, realized variance)
- [ ] **Validate eigenvalues** against known analytical results
- [ ] **Use lead-lag embedding** for automatic QV capture

---

## Connection to Current Implementation

The signature ablation study showed:
- |Lévy area| correlates r=0.84 with true variance
- This is BECAUSE |Lévy area| = QV/2 (proven)
- Lead-lag signature automatically estimates σ² without oracle access

**This is why signatures work**: They provide a principled way to estimate the diffusion coefficient from path data, which is exactly what GEDMD needs for the Ito correction.

---

## Why Signatures Solve the Single-Path Problem

The fundamental insight is that **signatures don't need conditional expectations**:

| Approach | Requires | Single Path? |
|----------|----------|--------------|
| Kramers-Moyal | $\mathbb{E}[dX^2 \| X=x]$ | ❌ No (need ensemble) |
| Kernel regression | Ergodicity + smoothness | ⚠️ Maybe (if enough data) |
| Lead-lag signature | Just $\sum (\Delta X)^2$ | ✅ Yes! |

**The lead-lag Lévy area**:
$$|\text{Lévy area}| = \frac{1}{2}\sum_{i=1}^{n} (\Delta X_i)^2$$

This is a **deterministic function of the path**:
- Input: Single price path $\{X_0, X_1, ..., X_n\}$
- Output: Estimate of $\int_0^T \sigma^2(X_s) ds$ (integrated variance)

For stationary processes, this gives us a proxy for $\sigma^2$ without needing:
- Multiple paths
- Conditional expectations
- Ergodic assumptions (beyond stationarity)

**This is the key theoretical justification for using signatures in the Sig-KKF framework!**

---

## Existing Implementation: GeneratorEstimator

The `GeneratorEstimator` class in `rkhs_kronic/src/generator_estimator.py` already implements the full two-step approach for GEDMD:

### Step 1: Drift Estimation via Log-Signatures
```python
# Log-signature features capture path geometry compactly
logsig = compute_log_signature(path_2d, level=2)  # [displacement, Lévy_area]

# Project to x-space via NW kernel averaging (fixes negative correlation)
Psi_x = project_to_x_space(Psi_raw, x_data)

# Ridge regression in whitened space gives drift
drift_pred = Psi_x @ alpha / dt
```

### Step 2: Diffusion Estimation from Residuals
```python
# Residuals after drift subtraction
residuals = dX - drift_pred * dt

# Second-order increments for robust sigma estimation
d2_res = np.diff(residuals)
d2_sq = d2_res ** 2

# Scale factor for fGN correlation
scale_sq = 2 * g0 - 2 * g1  # g0 = dt^(2H)

# Local regression or binning gives sigma(x)
sigma_profile = sqrt(local_mean(d2_sq) / scale_sq)
```

### Theory from GeneratorEstimator docstring:
> "The generator L of dX = mu(X)dt + sigma(X)dW acts on test functions as:
>   L phi(x) = mu(x) * phi'(x) + (1/2) sigma^2(x) * phi''(x)
>
> After whitening by L^{-1} (fGN Cholesky), the learned generator encodes drift.
> Diffusion is recovered from residual variance (second Wiener chaos)."

This confirms that the unified signature approach captures both terms of the Koopman generator without requiring oracle access to b or σ

---

## Signatures Already Characterize GARCH/ARCH (Chevyrev-Oberhauser 2022)

### Key Theoretical Result

**Corollary 23**: Expected robust signature moments uniquely characterize the law of stochastic processes.
- Paper explicitly states (Section 8): "classical time series such as (G)ARCH, ARMA" are covered
- **Multi-lag embedding is NOT theoretically necessary**

### Multi-Lag: Practical Shortcut, Not Theoretical Necessity

| Setting | Multi-lag needed? | Why |
|---------|-------------------|-----|
| Distribution learning (MMD) | **NO** | E[S(X)] captures all dependencies |
| Kernel methods (RBF on sig) | **NO** | RBF provides infinite expansion |
| Linear methods, low truncation | Maybe | Practical efficiency only |

### Empirical Validation

```
RBF kernel regression for σ² prediction:
  RBF(standard_sig):    r = 0.991   ← No multi-lag needed!
  Linear(standard_sig): r = 0.40    ← Multi-lag helps only here
```

### Recommended Approach for kGEDMD

**Standard (time, X) path + RBF kernel is SUFFICIENT**:

```python
# Standard time-augmented path (no multi-lag complexity needed)
path = np.column_stack([time, X])
sig = compute_path_signature(path, level=2)
K = rbf_kernel(sig, sig, gamma=0.1)  # Captures all higher-order deps
```

The RBF kernel provides the nonlinear expansion that captures ARCH/GARCH structure
