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

| File | Purpose |
|------|---------|
| `examples/proof_of_concept/poc_heston_hedging.py` | Core hedging experiments |
| `examples/proof_of_concept/poc_heston_integrated_control.py` | Dual Adaptive demo |
| `src/sskf/streaming_sig_kkf.py` | Streaming signature Kalman filter |
| `src/sskf/online_path_features.py` | Online path feature computation |

## Build & Test

```bash
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
