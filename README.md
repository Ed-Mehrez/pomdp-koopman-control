# POMDP Koopman Control

POMDP Sensor-Controller Synthesis via Koopman operators.

## The Problem

In partially observable systems, we need to:
1. **Estimate hidden state** from observations (filtering)
2. **Control optimally** given uncertain beliefs

Traditional approaches require full model knowledge. We achieve this **model-free**.

## Key Innovation: Dual Adaptive Architecture

Principled separation into two components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dual Adaptive Control                     │
├──────────────────────────┬──────────────────────────────────┤
│       SENSOR             │         CONTROLLER               │
│   (Unsupervised)         │         (Supervised)             │
├──────────────────────────┼──────────────────────────────────┤
│ Koopman eigenfunction    │ Policy in eigenfunction          │
│ extraction               │ coordinates                      │
│                          │                                  │
│ Input: Observations      │ Input: Eigenfunction projection  │
│ Output: Belief state     │ Output: Optimal action           │
│                          │                                  │
│ Method: Nyström-Koopman  │ Method: Online policy gradient   │
│ Cost: O(m³)              │ Cost: O(1) per step              │
└──────────────────────────┴──────────────────────────────────┘
```

**Why this works**: Eigenfunction stability limits error propagation from sensor to controller.

## Key Results (Heston Hedging)

| Method | Variance Reduction | Model Knowledge | Cost |
|--------|-------------------|-----------------|------|
| BPF (SOTA) | 75% | Full (κ,θ,ξ,ρ) | O(N) particles |
| **Dual Adaptive** | **55%** | **Zero** | **O(1)** |
| Black-Scholes | 0% | Partial | O(1) |

- Unsupervised sensor achieves **0.91x** MSE of 1000-particle BPF
- Online learning converges to stable hedging from scratch
- Optimal window = decorrelation time (τ ≈ 1/κ)

## Quick Start

```bash
# Setup
conda env create -f environment.yml
conda activate pomdp-koopman-control

# Run Heston hedging example
python examples/proof_of_concept/poc_heston_hedging.py
```

## Project Structure

```
pomdp-koopman-control/
├── src/
│   └── sskf/
│       ├── streaming_sig_kkf.py     # Streaming signature Kalman filter
│       └── online_path_features.py  # Online path feature computation
├── examples/
│   └── proof_of_concept/
│       ├── poc_heston_integrated_control.py  # Main Dual Adaptive demo
│       ├── poc_heston_hedging.py             # Hedging experiments
│       ├── poc_heston_filtering.py           # Volatility filtering
│       ├── experiment_sig_kkf_pomdp.py       # POMDP control
│       └── signature_features.py             # Log-signature utilities
└── docs/
    └── (theory documentation)
```

## Theoretical Foundation

### POMDP Belief State = Koopman Eigenfunction Projection

For a POMDP with hidden state Z and observations Y:
- Traditional: Maintain P(Z|Y_{1:t}) via particle filter
- Our approach: Project onto Koopman eigenfunctions φ_k(Y_{1:t})

The eigenfunctions encode sufficient statistics of the belief.

### Sensor-Controller Separation Theorem

Related to Hida-Malliavin filtering theory:
- Sensor computes E[f(Z)|Y_{1:t}] for eigenfunction f
- Controller optimizes over belief space (eigenfunction coordinates)
- Separation preserves optimality under mild conditions

### Why Signatures?

Path signatures capture the **temporal ordering** that standard methods miss:
- Lévy area encodes direction of evolution
- Higher levels capture path roughness
- Chen's identity enables O(1) online updates

## Applications

### 1. Option Hedging (Heston Model)
Hidden volatility, observable price. Hedge without knowing vol dynamics.

### 2. Regime-Switching Control
Hidden Markov state, observable system output. Control adaptively.

### 3. Robotics with Partial Observation
Hidden pose/velocity, noisy sensors. Navigate optimally.

## Citation

```bibtex
@article{pomdp-koopman-control,
  title={POMDP Sensor-Controller Synthesis via Koopman Operators},
  author={Mehrez, Edward},
  journal={In preparation for NeurIPS/ICML},
  year={2026}
}
```

## License

MIT License

---
*Split from [RKHS-KRONIC](https://github.com/Ed-Mehrez/RKHS-KRONIC) repository.*
