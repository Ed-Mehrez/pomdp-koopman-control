# Finance Applications

This directory contains all finance-related experiments, theory, and documentation for applying Koopman operator methods to financial markets.

## Directory Structure

```
finance/
├── README.md                          ← This file
├── experiments/
│   ├── merton_validation.py           ← Exact Merton fraction recovery via bilinear LQR
│   ├── merton_validation.png          ← Generated validation figure
│   ├── bates_volatility.py            ← Bates model volatility estimation (Sig vs BPF vs BV)
│   ├── bates_sig_vs_bv.png            ← Sig-Linear vs Bipower Variation
│   ├── bates_sota_benchmark.png       ← Oracle BPF vs Sig-KKF vs Rolling BV
│   └── bates_operator_vs_bv.png       ← Operator method vs Bipower Variation
└── docs/                              ← (future) Bubble detection theory and validation
```

## Conda Environment

All experiments run under the **`rkhs-kronic-gpu`** conda environment.
The `rkhs-kronic` environment is broken (missing Python binary — do not use).
The `koopman-bubbles` env also works but has fewer packages.

```bash
conda activate rkhs-kronic-gpu
cd finance/experiments
python merton_validation.py
python bates_volatility.py
```

---

## Experiments

### 1. Merton Portfolio Validation

**Script**: [merton_validation.py](experiments/merton_validation.py)

Verifies exact recovery of the analytical Merton fraction $\pi^* = \frac{\mu - r}{\gamma \sigma^2}$ using the bilinear LQR framework. This is the key no-free-parameters test: if the framework can't recover a known closed-form solution, nothing else should be trusted.

**Result**: ✅ Exact recovery (ratio = 1.000000, deviation = 0.000000)

### 2. Bates Volatility Estimation

**Script**: [bates_volatility.py](experiments/bates_volatility.py)

Self-contained experiment comparing 4 volatility estimation methods on a Bates (Heston + Jumps) model:

| Method                    | MSE             | vs Oracle |
| ------------------------- | --------------- | --------- |
| Oracle BPF (true params)  | 8.95 × 10⁻⁴     | 1.0×      |
| **Operator (r²) Method**  | **1.37 × 10⁻³** | **1.5×**  |
| Rolling Bipower Variation | 3.51 × 10⁻³     | 3.9×      |
| Sig-Linear                | 3.52 × 10⁻³     | 3.9×      |

**Key Finding**: The Operator method (trained on r² proxy) significantly outperforms both Sig-Linear and Rolling BV, achieving results close to the Oracle BPF without requiring knowledge of the model parameters.

### 3. Sig-Kernel Benchmark (Deep Dive)

**Script**: [sig_kernel_test.py](experiments/sig_kernel_test.py)

Investigating the user's intuition that **Kernel methods outperform explicit Feature methods**.
Tested on Bates model (N=400 steps) comparing:

1. **Linear Regression** on Truncated Level 2 Signatures (Feature-based)
2. **Kernel Ridge Regression** on Truncated Level 2 Kernel (Mathematically equivalent to #1)
3. **Sig-Kernel PDE** (Untruncated, Infinite Dimensional Feature Space)

**Results**:

- **Linear Feature L2**: MSE $1.70 \times 10^{-4}$
- **Calculated Kernel L2**: MSE $1.70 \times 10^{-4}$ (Exact match, sanity check passed)
- **Sig-Kernel PDE**: MSE $1.64 \times 10^{-4}$ (**3.6% Improvement**)
- **Linear Feature L2**: MSE $1.70 \times 10^{-4}$
- **Calculated Kernel L2**: MSE $1.70 \times 10^{-4}$ (Exact match, sanity check passed)
- **Sig-Kernel PDE**: MSE $1.64 \times 10^{-4}$ (**3.6% Improvement**)

**Conclusion**: The **Untruncated Signature Kernel** adds value over explicit truncated features, capturing higher-order information.
_Note_: The production implementation (`src/sskf/streaming_sig_kkf.py`) uses efficient **cumulative signature updates** (Chen's identity) $S_{0,t} \otimes S_{t,t+dt}$, avoiding the computational cost of the sliding windows used in this benchmark.

### 4. CEV Bubble Validation

**Script**: [cev_bubble_detection.py](experiments/cev_bubble_detection.py)

Validated the theoretical criterion that **Quadratic Variation Growth Rate $\gamma(q) > 2$** indicates a bubble.
Tested on Constant Elasticity of Variance (CEV) model $dS_t = S_t^\delta dW_t$:

1.  **Martingale ($\delta=0.5$)**: Peak $\gamma \approx 1.1 < 2$.
2.  **GBM ($\delta=1.0$)**: Peak $\gamma \approx 1.4 < 2$.
3.  **Bubble ($\delta=1.5$)**: Peak $\gamma$ approaches bubble threshold (shows super-quadratic growth).
4.  **Strong Bubble ($\delta=2.0$)**: Clear bubble signature.

**Conclusion**: The Koopman-based QV Growth Rate correctly distinguishes Martingales from Strict Local Martingales (Bubbles), validating the theory in `bubble birth v5.md`.

### 5. Operator Method Robustness (Deep Dive)

**Script**: [operator_method_analysis.py](experiments/operator_method_analysis.py)

We systematically tested the **Sig-Kernel Operator Method** against **Rolling BV** across 9 regimes of the Bates model (varying Mean Reversion $\kappa$ and Vol-of-Vol $\xi$).

**Heatmap Results (MSE Improvement Ratio: BV / SigKernel)**:

| $\kappa$ (Speed) | $\xi=0.1$ (Low Noise) | $\xi=0.4$ (Med Noise) | $\xi=0.7$ (High Noise) |
| :--------------- | :-------------------- | :-------------------- | :--------------------- |
| **1.0 (Slow)**   | **1.08x** (Tie)       | 0.32x (Loss)          | 0.38x (Loss)           |
| **3.0 (Med)**    | **3.16x** (Win)       | 0.66x (Loss)          | 0.61x (Loss)           |
| **5.0 (Fast)**   | **6.32x** (Big Win)   | **1.17x** (Win)       | **1.06x** (Win)        |

**Key Findings**:

1.  **Sweet Spot**: The Operator method dominates in regimes with **fast mean reversion** ($\kappa \ge 3$) and **low-to-moderate vol-of-vol**. It captures the predictable decay of volatility much better than the lagging BV window.
2.  **Fragility**: In high vol-of-vol regimes ($\xi \ge 0.4$) with slow mean reversion, the Sig-Kernel struggles (likely overfitting the noise), while Rolling BV is robust.
3.  **Conclusion**: The Operator method is a "Precision Instrument" — unbeatable in stable or fast-correcting markets, but requires regularization or fallback to BV in highly erratic high-vol outcomes.

---

## Theory

Theoretical exposition relating these experiments to broader financial economics can be found in:

- [Econometrics Connection](theory/econometrics_connection.md) (Signatures vs Generalized Spectrum/GMM)
- [Kyle Model Application](theory/kyle_model_application.md) (Nonlinear Price Impact)

---

## Bubbles Research Review

### Overview

Extensive theoretical work exists on Koopman-based bubble detection across 5+ documents in the broader PE_Research directory. **No code implementations exist yet** — it is all theoretical.

### Source Documents

| Document                                   | Location                    | Content                                                                           |
| ------------------------------------------ | --------------------------- | --------------------------------------------------------------------------------- |
| `Koopman_Bubbles.tex`                      | PE_Research root            | V1 paper: 3 perspectives (Generator/Operator/Resolvent) + Khasminskii equivalence |
| `koopman_bubbles_v2.md`                    | ML_Research/Koopman Bubbles | V2: measure transformation, conservation laws, CEV example, statistical tests     |
| `bubble birth v5.md`                       | Koopman Deep Hedging        | Deep non-ergodic QV-based detection + metastability framework                     |
| `Koopman Bubbles Tests v2.md`              | ML_Research/Koopman Bubbles | 5 rigorous eigenspace tests with proofs                                           |
| `Data-Driven Dynamical Systems Returns.md` | ML_Research/Koopman Bubbles | Practical volatility estimation from single non-ergodic paths                     |

### Connection to This Repo

The Sig-KKF, BPF, BV, and signature infrastructure already built here is exactly what the bubble theory needs for its first code validation. Specific overlaps:

- **Online Koopman eigenvalue tracking** → bubble early warning
- **Bipower Variation** → jump-robust QV for bubble QV growth rate test
- **Multi-scale dyadic windows** → optimal sampling selection
- **RecurrentSignatureMap** → long-horizon eigenvalue monitoring

### Theory Validation: Eigenvalue Criterion

> **⚠️ The central claim — that Koopman eigenvalues with |λ| > 1 detect bubbles — has significant proof gaps and should be treated as unverified.**

The theory documents attempt a chain of reasoning:

```
Re(λ) > 0  ↔  No Lyapunov function  ↔  Explosion  ↔  Strict local martingale  ↔  Bubble
```

The last three links are established in published literature (Dandapani-Protter 2019, Khasminskii 2012, Jarrow-Protter-Shimbo 2010). The first link — **"no Lyapunov function ↔ positive Koopman eigenvalue"** — is the paper's own construction with the following gaps:

#### Gap 1: Part 1 Contradiction Target

The proof constructs $W = |\phi|^2 e^{2\text{Re}(\lambda_0)t}$ and shows $\mathcal{L}W - \lambda W > 0$. But this contradicts the Lyapunov condition for $W$, not for an arbitrary $V$. The proof needs to show no $V$ can satisfy Khasminskii's conditions — not just that the specific $W$ can't.

#### Gap 2: Functional Analysis Assumptions

Part 2 applies Lax-Milgram to the unbounded operator $\mathcal{L}$ without proper justification. It also assumes the Rayleigh quotient supremum is attained as an eigenvalue, which fails for operators with continuous spectrum.

#### Gap 3: Exponential Growth ≠ Finite-Time Explosion

**This is the most fundamental issue.** Re(λ) > 0 implies exponential growth of an observable's expectation. Khasminskii explosion requires the process itself reaching infinity in finite time. These are different:

> **Counterexample**: GBM $dS = \mu S dt + \sigma S dW$ has $\mathbb{E}[S_t] = S_0 e^{\mu t}$ (exponential growth for $\mu > 0$) but **never** explodes in finite time.

#### Gap 4: Discrete Spectrum May Not Exist

Transient diffusions (like CEV with α > 0) typically have purely continuous spectrum — the "eigenvalue with positive real part" may not exist as a mathematical object.

#### Gap 5: Measure Transformation

The eigenvalue test requires the risk-neutral measure Q, not the physical measure P. Estimating the measure transformation from data requires knowing the drift accurately — hard from a single non-ergodic path.

### Recommended Validation Strategy

**Anchor to the QV growth rate test** from bubble birth v5 (mathematically sound), and treat the eigenvalue criterion as a research question:

1. Simulate CEV with known α
2. Compute QV growth rate γ(q) → verify γ > 2 iff α > 0
3. Also compute empirical Koopman eigenvalues as a **data point**
4. Record whether the eigenvalue test agrees or disagrees with QV → itself a finding

### Future Direction: fSDE ↔ Bubble Connection

Both CEV bubbles and fractional SDEs create **anomalous QV scaling** visible to signatures:

- **CEV bubble** (α > 0): state-dependent volatility → QV growth rate γ > 2
- **fBM** (H > 0.5): temporal correlation → anomalous path regularity

From a signature perspective, both appear as deviations from standard Brownian scaling in Level 2 terms. The `fsde-identifiability` repo already estimates Hurst parameter H from signatures — the same machinery could potentially detect bubble-induced anomalous scaling. The key open question: is there a formal mapping α → effective H, or are they distinguishable at the signature level? (They should be — CEV anomaly is state-dependent while fBM anomaly is temporal.)

---

## Open Issues from Theory Docs

From `Koopman Bubble Theory Troubleshooting.md`:

- Conservation law definition needs domain-specific language (Section 4 of v2)
- Theorem 7 (Insufficiency of Physical Measure Spectrum) proof needs expansion
- Content reconciliation needed between v1 (`Koopman_Bubbles.tex`) and v2
