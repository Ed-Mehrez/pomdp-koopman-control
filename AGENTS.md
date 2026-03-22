# AGENTS.md

Context for AI coding agents working on this repository.

## Environment

**Use the shared conda environment** - do NOT create a new one:

```bash
conda activate rkhs-kronic-gpu
```

The `environment.yml` in this repo is for CI/collaborators only.

## Project Overview

POMDP Sensor-Controller Synthesis via Koopman operators.

**Key Innovation**: Dual Adaptive architecture

- **Sensor**: Unsupervised Koopman eigenfunction extraction → hidden state estimate
- **Controller**: Supervised policy in eigenfunction coordinates → optimal action

## Key Files

| File                                                         | Purpose                                      |
| ------------------------------------------------------------ | -------------------------------------------- |
| `finance/README.md`                                          | Finance applications overview & theory notes |
| `finance/experiments/merton_validation.py`                   | Merton exact recovery sanity check           |
| `finance/experiments/bates_volatility.py`                    | Bates model volatility estimation benchmark  |
| `examples/proof_of_concept/poc_heston_hedging.py`            | Core hedging experiments                     |
| `examples/proof_of_concept/poc_heston_integrated_control.py` | Dual Adaptive demo                           |
| `src/sskf/streaming_sig_kkf.py`                              | Streaming signature Kalman filter            |
| `src/sskf/online_path_features.py`                           | Online path feature computation              |

## Build & Test

**Always use `conda activate rkhs-kronic-gpu`** (not `rkhs-kronic`, which is broken).

```bash
# Run Merton validation (should print "EXACT RECOVERY")
python finance/experiments/merton_validation.py

# Run Bates volatility benchmark (generates 3 figures)
python finance/experiments/bates_volatility.py

# Run Heston hedging example
python examples/proof_of_concept/poc_heston_hedging.py

# Run integrated control demo
python examples/proof_of_concept/poc_heston_integrated_control.py
```

## Critical Knowledge

### Dual Adaptive Results (Heston)

- 55% variance reduction with ZERO model knowledge
- O(1) cost vs O(N) for particle filters
- Sensor achieves 0.91x MSE of 1000-particle BPF

### Eigenfunction Stability

Error propagation is limited because Koopman eigenfunctions are stable.
Sensor errors don't catastrophically affect controller.

### Optimal Window = Decorrelation Time

For Heston: τ ≈ 1/κ (mean-reversion timescale)

## Conventions

- Python 3.10+
- NumPy/SciPy for numerics
- `iisignature` for signature computation
- `torch` for online learning

## Related Repositories

- Parent: [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC)
- Sibling: [fsde-identifiability](https://github.com/Ed-Mehrez/fsde-identifiability)
- Sibling: [rkhs-koopman-control](https://github.com/Ed-Mehrez/rkhs-koopman-control)

## Documentation Standards (Strict)

### Latex Syntax Rules

1. **NEVER use `*` for subscripts**. Use `_`. (e.g., `P_t` NOT `P*t`).
2. **NEVER use `\*` or `\_` in math blocks**. Use `*` and `_`. (e.g., `\pi^*` NOT `\pi^\*`).
3. **CHECK for escaped characters** like `_\t` or `\^`.
4. **NO text-mode asterisks** in equations. Use `\cdot` or nothing.

## Bayesian Framework Consistency (Strict)

**The bubble detection framework is a GP/Bayesian framework. ALL methods must be consistently Bayesian.**

1. **No frequentist shortcuts**: Do NOT use OLS, t-tests, p-values, ADF tests, or Mann-Kendall tests. Use Bayesian equivalents (GP regression, BLR, posterior probabilities).
2. **KRR ≡ GP MAP**: Frame all kernel ridge regression as GP posterior mean (R&W §6.2). σ²_n = λ.
3. **Model selection**: Use marginal likelihood (R&W §5.4) or blocked time-series CV, NOT AIC/BIC.
4. **Uncertainty quantification**: Use GP posterior variance for UQ, not bootstrap SE (unless as a robustness check).
5. **P(bubble)**: Must come from posterior CDF, e.g. P(α > 2 | data) = Φ((α̂ - 2) / σ_post).
6. **Trend detection**: Use GP with linear mean function (R&W §2.7), NOT OLS trend tests.
7. **References**: Rasmussen & Williams (2006), Kanagawa et al. (2018), Berlinet & Thomas-Agnan (2004).

## Empirical Validation Standards (Strict)

1. **Verify Numerics Extensively**: Do not claim success just because a boolean flag flips correctly. You must log the exact values, bounds, and extracted properties (e.g., eigenvalues, predicted variance growth rates) to ensure the math actually checks out and is not a fragile numerical artifact (like extrapolation failure or matrix singularity).
2. **Visualize Findings**: When proposing a new detector or property, generate plots of the internal state (e.g., predicted variance vs. true variance over the state space) to visually and definitively prove the algorithm works as claimed.
