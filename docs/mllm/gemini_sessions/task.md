# Task: BPF-inspired Sig-KKF Control for CartPole

- [x] Research and Definition
  - [x] Understand "BPF" in the context of CartPole control (Particle MPC)
  - [x] Create `documentation/mpc_koopman_bridge.md`
- [x] Implementation (Baselines)
  - [x] Implementing `poc_particle_mpc.py` (Shooting + MPPI)
  - [x] Verify Baselines (Success)
- [ ] Tuning Recurrent Sig-KKF [/]
  - [x] **Data**: Use MPPI to generate expert training trajectories -> **CANCELLED**
  - [x] **Pivot 1**: Offline Policy Iteration -> **Superseded by Online Request**
  - [ ] **Methodology**: Implement **Online Adaptive Sig-KKF** (RLS + Tensor-SDRE).
  - [ ] **Implementation**:
    - [ ] Add `RecursiveTensorLeastSquares` class.
    - [ ] Implement `run_online_adaptive_experiment`.
  - [ ] **Verification**: Show real-time adaptation and swing-up.
- [ ] Final Comparison
  - [ ] Update `walkthrough.md` with final tuned results
