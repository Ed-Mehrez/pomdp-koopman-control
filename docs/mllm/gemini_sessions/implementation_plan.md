# Theory Bridge Implementation: Koopman-Terminated BPF

## The Concept

We previously established that:

1.  **BPF (MPPI)**: Good at local trajectory optimization ($H=60$), but computationally expensive and myopic if $H$ is small.
2.  **Sig-KKF**: Good at global value estimation ($P$ matrix from ARE), but suffers from drift in open-loop rollout ($z_{t+k} \approx A^k z_t$).

**The Connection**: Combine them.
Use **Short-Horizon MPPI ($H=10$)** for local control, and terminate the horizon with the **Sig-KKF Value Function** ($V(x) = z(x)^T P z(x)$).

$$ J(u*{0:H}) = \sum*{t=0}^{H-1} C(x*t, u_t) + \underbrace{z(x_H)^T P z(x_H)}*{\text{Koopman Value}} $$

## Implementation Steps

### [MODIFY] `poc_cartpole_recurrent_tensor.py`

1.  **Data**: Keep the "Random + Energy Pumping" collection (No Cheating).
2.  **Learning**: Train the Koopman Tensor Model ($A, B, N$) and $C$ (Decoder) as before.

### 3. Sig-KKF Implementation: Online Adaptive (The Real-Time Analog)

#### [MODIFY] [poc_cartpole_recurrent_tensor.py](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/examples/proof_of_concept/poc_cartpole_recurrent_tensor.py)

- **The Analog**:
  - **MPPI**: Re-optimizes the _Control Sequence_ online at every step.
  - **Online Sig-KKF**: Re-optimizes the _System Model & Value_ online at every step.
- **Implementation**:
  1.  **Recursive Learning**: Implement **RLS for Tensors**. Update $A, B, N$ with every new $(z_t, u_t, z_{t+1})$ tuple.
  2.  **Adaptive Control**:
      - **Option A (Online SDRE)**: Re-solve ARE at every step (or every 10 steps) with the new model.
      - **Option B (Online Tensor-MPPI)**: Use the constantly-refreshing model for MPPI rollout. (Mitigates drift).
  3.  **Benefit**: The controller "learns to swing up" in real-time during the episode, adjusting for local model errors immediately.

This fulfills the usage of `OnlineKoopman` mentioned in earlier context. 4. **Synthesis**: Solve the Tensor ARE to get matrix $P$ (The Value Function). 5. **Control**: Implement `KoopmanValueMPPI`. - Rollout: Use the **Learned Tensor Model** for $H=10$ steps (Reduced drift risk). - Cost: Standard running cost + Terminal Koopman Cost. - State: Map physical start state ->### 4. Principled Exploration: "Koopman variance"

#### [MODIFY] [kronic_online_tensor.py](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/src/kronic_online_tensor.py)

- **Theory**: The RLS algorithm maintains an inverse covariance matrix $P$. The quantity $\sigma^2 = \phi(z,u)^T P \phi(z,u)$ represents the **Predictive Variance** (Uncertainty) of the model for that state-action pair.
- **Implementation**: Add `get_variance(z, u)` to `RecursiveTensorRLS`.

#### [MODIFY] [poc_online_sig_kkf.py](file:///home/ed/SynologyDrive/Documents/Research/P&E_Research/rkhs_kronic/examples/proof_of_concept/poc_online_sig_kkf.py)

- **Method**: **Thompson Sampling** (Posterior Sampling).
- **Implementation**: Remove the Cost Bonus. Isolate the Uncertainty ($\sigma$) and use it to perturb the dynamics in the MPPI rollout.
  $$ z\_{t+1} \sim \mathcal{N}(K z_t, \Sigma(z)) \quad \text{where } \Sigma \propto \phi^T P \phi $$
- **Why Principled?**: This is standard Bayesian Decision Making. In uncertain regions, the "Optimism in the Face of Uncertainty" emerges because some samples will be favorable, and MPPI's exponential weighting will select them. In certain regions (stabilization), $\Sigma \to 0$, realizing robust control. No heuristics.

## Expected Result

- Success where pure SDRE failed (because MPPI handles constraints/non-convexity better locally).
- Success where pure Tensor-MPPI failed (because $H=10$ drift is negligible compared to $H=60$).
