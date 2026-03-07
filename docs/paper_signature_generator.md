# Paper: Signature Kernel Methods for SDE Generator Learning

## Working Title
**Learning Infinitesimal Generators of Markov Diffusions via Signature Kernels**

## Core Contribution
Show that signature kernel regression can learn the infinitesimal generator L of a Markov diffusion dX = μ(X)dt + σ(X)dW from trajectory data, enabling model-free recovery of drift μ(x) and diffusion σ²(x) via the Carré du Champ identity.

## Key Results

### Theoretical
1. **Generator as regression target**: For observable f, E[f(X_{t+dt}) - f(X_t) | X_t=x]/dt → Lf(x)
2. **CdC identity**: μ(x) = L(x), σ²(x) = L(x²) - 2xL(x) for 1D diffusions
3. **Signature universality**: Signatures provide universal nonlinear features (Stone-Weierstrass in path space)
4. **Consistency**: Under ergodicity, signature kernel regression of Lf is consistent

### Empirical (from graduated_sanity_checks.py Level 2)
- CIR process (50K steps, dt=0.01): drift corr=0.96, σ corr=0.995
- Kappa recovery: 1.96±0.15 (true=2.0)
- Held-out validation (not in-sample)
- Works for ergodic processes; fails for GBM (non-ergodic)

## Paper Structure

### 1. Introduction
- Problem: Learning SDE dynamics from trajectory data
- Why it matters: Model calibration, system identification, simulation
- Our approach: Signature kernels + generator regression + CdC extraction
- Main result: Model-free μ, σ² recovery competitive with parametric methods

### 2. Background
- Infinitesimal generators and Kolmogorov equations
- Carré du Champ operator: Γ(f,g) = L(fg) - fL(g) - gL(f)
- Path signatures and rough path theory (brief)
- Kernel methods and RKHS

### 3. Method
- **3.1 Generator as regression target**
  - Finite-difference approximation: (f(X_{t+dt}) - f(X_t))/dt
  - Target observables: f(x) = x, f(x) = x² suffice for 1D
  - Extension to higher dimensions

- **3.2 Signature kernel regression**
  - Lead-lag log-signature for path embedding
  - RBF kernel on signature features (or direct signature kernel)
  - Nystrom approximation for scalability
  - Choice of signature truncation level

- **3.3 CdC extraction**
  - From L̂(x) and L̂(x²): reconstruct μ̂(x), σ̂²(x)
  - Error propagation analysis
  - Regularization considerations

### 4. Theoretical Analysis
- Consistency under ergodicity
- Rate of convergence (depends on mixing time, signature level)
- Comparison to kernel DMD, neural SDE
- When it fails: non-ergodic processes

### 5. Experiments

#### 5.1 Synthetic SDEs
- **OU process**: Linear drift, constant diffusion
- **CIR process**: Mean-reverting, state-dependent diffusion
- **CEV process**: Nonlinear diffusion, non-Gaussian
- **Double-well potential**: Multimodal stationary distribution

Metrics: Held-out correlation, parameter recovery error, KL divergence of stationary distributions

#### 5.2 Sample Efficiency Comparison
**Key comparison: Signature-KRR vs model-based Kalman**

| Method | Model Required | Data Needed | μ Recovery | σ² Recovery |
|--------|---------------|-------------|------------|-------------|
| Oracle (analytic) | Full | 0 | Perfect | Perfect |
| Kalman (known model) | Drift+Diffusion | ~500 (warmup) | - | Perfect |
| **Sig-KRR (ours)** | None | ~5000 | 0.96 corr | 0.995 corr |
| Neural SDE | None | ~50000 | ? | ? |
| Kernel DMD | None | ~10000 | ? | ? |

The comparison to "model-known" BPF/Kalman shows the cost of model-free estimation. We pay ~10x data for not needing to specify the model.

#### 5.3 Ablations
- Signature level (1, 2, 3)
- Forgetting factor (cumulative vs decay)
- Kernel bandwidth selection
- Nystrom landmark count

### 6. Extensions
- **Multidimensional SDEs**: CdC gives full diffusion matrix
- **Partial observations**: Connection to filtering (future work)
- **Control-affine systems**: L_π = L_0 + πL_1 + π²L_2 decomposition
- **Jump-diffusions**: Lévy-Khintchine extension

### 7. Discussion
- Limitations: Requires ergodicity, sufficient mixing
- Computational cost: O(M³) for M Nystrom landmarks
- When to use: Unknown dynamics, model-free required, moderate data
- Future: Connection to characteristic function estimation

### 8. Conclusion

## Key Equations

**Generator definition:**
```
Lf(x) = lim_{dt→0} E[f(X_{t+dt}) - f(X_t) | X_t=x] / dt
      = μ(x)f'(x) + ½σ²(x)f''(x)
```

**CdC identity (1D):**
```
μ(x) = L(x)
σ²(x) = L(x²) - 2x·L(x)
```

**Signature kernel:**
```
k(path_1, path_2) = ⟨Sig(path_1), Sig(path_2)⟩
```

**Lead-lag embedding:**
```
(X_t) → ((X_t, X_{t-lag})) → Sig((X_t, X_{t-lag}))
```

## Related Work
- Kernel DMD / gEDMD (Williams, Rowley, Kevrekidis)
- Neural SDEs (Kidger, Lyons)
- Signature methods for time series (Chevyrev, Kormilitzin)
- Generator learning (Bittracher, Klus, Schütte)

## Figures Needed
1. Method schematic: path → signature → kernel regression → L̂ → μ̂, σ̂²
2. CIR recovery: true vs learned μ(x), σ²(x) curves
3. Sample efficiency: recovery error vs data size for different methods
4. Ablation: signature level impact on recovery quality
5. Failure case: GBM (non-ergodic) showing divergence

## Code
Primary implementation: `kronic_pomdp/experiments/graduated_sanity_checks.py` (Level 2)
Signature features: `examples/proof_of_concept/signature_features.py`
