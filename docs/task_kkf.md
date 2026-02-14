# Task: Investigate N=3 KRR-SDRE to Unlock N=5

The N=5 Snake robot control with KRR-SDRE has been problematic (likely physics or tuning). We are backing off to N=3 to isolate the issue.

## Todos

- [x] **Combine Fixed Physics with N=3 Controller**: Create `experiment_snake_krr_n3_fixed.py` using `NLinkArmEnvFast` (proven correct) and `LocalKoopmanSDRE_Snake`. <!-- id: 0 -->
- [x] **Validate N=3 Stability**: Run the new experiment. Expect robust stabilization. <!-- id: 1 -->
- [x] **Analyze N=3 Performance**: Check if "thrashing" is reduced and holding is stable. <!-- id: 2 -->
- [x] **Attempt N=5 Fix**: If N=3 works, apply the exact same physics/controller structure to N=5 (create `experiment_snake_krr_n5_fixed.py`). <!-- id: 3 -->
  - [x] **Visualize Baseline**: Generated `snake_n5_stabilization.gif` (Shows instability with manual gains).
- [x] **Automate N=5 Tuning**: Used `tune_snake_bayes.py` to find optimal params ($P=55k, D=9.7k, R=1.5$). Result: **Bounded Stability** (Dist $\approx 0.3-0.5$, w/ oscillation). <!-- id: 4 -->
  - [x] **Verified N=50**: Result: Unstable (Dist ~2.0). Density matters.
  - [x] **tested Global EDMD**: Result: Catastrophic Failure (Dist > 5.0). Global features lack local precision.
  - [x] **Tested True KGEDMD**: Result: Immediate Failure (Dist > 2.0). Dynamics-aligned features still insufficient for local stabilization.

## Phase 3: Universal Application (Hydrogym)

- [/] **Setup Hydrogym**: Install and verify environment. <!-- id: 5 -->
  - [x] **Attempt Pip Install**: Failed (requires Firedrake).
  - [ ] **Install Firedrake**: User action required (complex dependency).
  - [ ] **Verify Hydrogym**: Import successful after Firedrake install.
- [ ] **Baseline Control**: Run standard LQR/PPO on a Hydrogym env (e.g., Cylinder). <!-- id: 6 -->
- [ ] **Apply Koopman SDRE**: Port the `LocalKoopmanSDRE` controller to Hydrogym interface. <!-- id: 7 -->
- [ ] **Verify & Tune**: Demonstrate stabilization on fluid flow. <!-- id: 8 -->

## Phase 2 Revision: KKF for Insider Trading (Multiasset Kyle)

- [ ] **Implement Finance Environment**: Create `FinanceKyleEnv` with bimodal asset priors. <!-- id: 9 -->
- [ ] **Baseline Comparison**: Implement standard Linear Kyle pricing baseline. <!-- id: 10 -->
- [ ] **Apply Sig-KKF**: Train Signature-based Koopman to learn the nonlinear pricing rule. <!-- id: 11 -->
- [ ] **Manipulation Defense**: Test robustness against cross-asset bluffing strategies. <!-- id: 13 -->
- [ ] **Evaluation**: Compare MSE and visualize the learned pricing manifold. <!-- id: 12 -->
