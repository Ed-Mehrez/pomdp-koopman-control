# Cleanup Plan: Track A (Control & Scaling)

We are finalizing the `track-a-control` branch for "Paper 1". This branch focuses on:

1.  **High-Dimensional Scaling** (Local Koopman SDRE on Kuramoto).
2.  **Robust Control** (Local Koopman SDRE on CartPole).
3.  **Baselines** (MPPI).

We will remove all experimental artifacts related to "Paper 2" (Track B: Sig-KKF, POMDPs, Stochasticity) to ensure a clean, publishable codebase.

## Files to KEEP ✅

These files form the core of Paper 1:

| File                                                             | Purpose                                                              |
| :--------------------------------------------------------------- | :------------------------------------------------------------------- |
| `examples/proof_of_concept/experiment_local_koopman_sdre.py`     | **Main Result (CartPole)**: Robust stabilization.                    |
| `examples/proof_of_concept/experiment_kuramoto_scaling.py`       | **Main Result (Scaling)**: 20D Synchronization.                      |
| `examples/proof_of_concept/experiment_lorenz_control.py`         | **Main Result (Chaos)**: Stabilizing Lorenz to origin.               |
| `examples/proof_of_concept/solid_mppi_cartpole.py`               | **Baseline**: Ground truth MPPI for comparison.                      |
| `examples/proof_of_concept/experiment_mppi_robustness.py`        | **Baseline**: MPPI robustness tests.                                 |
| `examples/proof_of_concept/experiment_prediction_benchmark.py`   | **Validation**: Comparing RBF vs Poly vs Local.                      |
| `examples/proof_of_concept/experiment_analytic_rkhs_cartpole.py` | **Method A**: Analytic stabilization (precursor to Local SDRE).      |
| `examples/proof_of_concept/experiment_snake_robot.py`            | **Robotics Demo**: Stabilizing 2-Link Manipulator (Double Pendulum). |

## Files to DELETE ❌

These files are either obsolete, belong to Track B, or are failed experiments:

- `examples/proof_of_concept/experiment_sig_kkf_*` (Track B: Signatures/Kalman)
- `examples/proof_of_concept/poc_*` (Temporary Proof of Concepts - clutter)
- `examples/proof_of_concept/signature_features.py` (Track B utility)
- `examples/proof_of_concept/general_koopman_mpc.py` (Superseded by Local SDRE)
- `examples/proof_of_concept/general_spectral_control.py` (Superseded)
- `examples/proof_of_concept/general_spectral_potential.py` (Superseded)
- `examples/proof_of_concept/marcus_research.py` (Misc)
- `examples/proof_of_concept/optimize_hyperparams.py` (Old)
- `examples/proof_of_concept/test_heston_robustness.py` (Track B - Finance)
- `examples/proof_of_concept/demonstrate_virtues.py` (Old)

## Execution Plan

1.  **Delete** the listed files.
2.  **Verify** `experiment_local_koopman_sdre.py` and `experiment_kuramoto_scaling.py` still run.
3.  **Consolidate** documentation (Update `walkthrough.md` to focus only on Paper 1 results).

**Approval Required:** Shall I proceed with the deletion?
