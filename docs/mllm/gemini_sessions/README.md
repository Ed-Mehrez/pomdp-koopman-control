# Gemini Brain Sessions

AI assistant session exports relevant to POMDP Koopman Control.

## Sessions Included

### a2555163 - Heston Hedging + Dual Adaptive Control (PRIMARY)
- `walkthrough_heston_hedging.md` - Complete Dual Adaptive walkthrough
- `walkthrough_heston_final.md` - Final results
- `model_scope.md` - Model definitions
- `implementation_plan.md` - Implementation steps
- Key figures: `heston_hedge_results.png`, `heston_online_learning.png`

**Key Results**:
- 55% variance reduction with ZERO model knowledge
- O(1) cost vs O(N) for particle filters
- Sensor achieves 0.91x MSE of 1000-particle BPF

### 6691c53e - Sig-KKF Adaptive Control
- `track_b_sig_kkf_validation.md` - Validation results
- CartPole swingup with signatures

### 8c645575 - Online Adaptive Discovery
- Online learning convergence analysis

## Figures

| File | Description |
|------|-------------|
| `heston_hedge_results.png` | Hedging performance comparison |
| `heston_online_learning.png` | Online learning convergence |
| `heston_window_convergence.png` | Optimal window analysis |
| `cartpole_swingup_signature_result.png` | CartPole with signatures |
| `augmented_control_*.png` | Augmented state control results |

## Source

Exported from `~/.gemini/antigravity/brain/` on Feb 14, 2026.
