# Model Scope and Theoretical Limits

## 1. Currently Supported Regime: "Diffuse Volatility"

The current **Signature-based Volatility Sensor** is highly effective (universally robust) for processes that behave like **Continuous Semimartingales** or their discrete approximations.

### Supported Models

| Model Type              | Examples               | Performance   | Why?                                                                                                             |
| :---------------------- | :--------------------- | :------------ | :--------------------------------------------------------------------------------------------------------------- |
| **Affine Volatility**   | Heston, SABR           | **Excellent** | Signatures capture the path geometry $\int P dP$ which encodes Quad Var.                                         |
| **Rough Volatility**    | rBergomi, Rough Heston | **Excellent** | Signatures naturally capture "Roughness" (High variation) without needing Markov assumptions.                    |
| **Discrete Volatility** | GARCH(1,1), COGARCH    | **Excellent** | Convergence to diffusion limits makes them "look like" rough paths to the integration kernel.                    |
| **Path-Dependent Vol**  | Hobby-Rice, Vol-of-Vol | **Excellent** | The infinite-dimensional feature state learns the history dependence.                                            |
| **Jump Diffusion**      | Bates, Merton, NIG     | **Verified**  | Operationalized Marcus Signatures (Power Variations) allow robust separation of Diffusive Volatility from Jumps. |

### Theoretical Basis

The core feature is the **Signature** of the path $X_t$. For continuous paths, the signature approximates the **Quadratic Variation** ($[X,X]_t$). $$ \text{Sig}^{(2)}(X) \approx \text{Cov}(X) \propto \int \sigma_t^2 dt $$ Since our target is $v_t$ (or $\sigma^2$), the linear regression $w^T \text{Sig}(X)$ effectively learns to extract the Quadratic Variation.

---

## 2. Current Limitation: "The Jump Barrier"

The standard implementation struggles with **Jump Processes** (Levy Processes).

### Previously Unhandled Models (Now Solved)

| Model Type                  | Examples              | Old Performance | Solution Strategy                                                                                              |
| :-------------------------- | :-------------------- | :-------------- | :------------------------------------------------------------------------------------------------------------- |
| **Finite Activity Jumps**   | Bates, Merton         | **Verified**    | Use **Bipower Variation** to track Diffusive Vol and **Realized Variance** to track Total Quadratic Variation. |
| **Infinite Activity Jumps** | VG, NIG, Alpha-Stable | **Theoretical** | Higher-order Power Variations allow separation of continuous parts from Levy/jump parts.                       |

### The Theoretical Gap

Standard **Chen's Signatures** are defined for continuous paths. When a path jumps, the "Linear Interpolation" (used implicitly by discrete signatures) creates a fictitious "steep slope".

- **Diffusion**: $\sum (\Delta X)^2 \sim O(t)$
- **Jump**: $(\Delta X)^2 \sim O(1)$ The Jump dominates the signal, masking the diffusion volatility we want to measure.

---

### 3. Path Forward: Operationalizing Marcus Signatures

To achieve True Universality (including Jumps), we must operationalize **Jump-Robust Features**.
The user asked: _"How can we operationalize Marcus signatures in a nonparametric way?"_

### The Solution: Augmented Feature State (Power Variations)

Ait-Sahalia and Jacod (2009) provide the roadmap in _Testing for Jumps in a Discretely Observed Process_.
They show that **Power Variations** of different orders $p$ separate the Continuous and Discontinuous components of the process:

1.  **Small Powers ($p < 2$) / Bipower Variation**: Dominated by the continuous diffusion.

    - Specifically, **Multipower Variation** (Eq 22) converges to the Integrated Volatility $\int \sigma_t^2 dt$ regardless of jumps.
    - Operational Feature: **Bipower Variation (BV)** $\approx \frac{\pi}{2} \sum |\Delta X_i| |\Delta X_{i-1}|$.
    - This serves as the **Robust Volatility Feature**.

2.  **Large Powers ($p > 2$)**: Dominated by Jumps.
    - $\sum |\Delta X|^p$ converges to the sum of jumps $\sum |\Delta J|^p$.
    - Operational Feature: **Realized Variance (RV)** (p=2) or higher moments.
    - These serve as the **Jump Correction Features** (Proxy for Marcus terms $\Delta X^2, \Delta X^3 \dots$).

### Implementation Strategy

By augmenting the feature state with these moments, we allow the Linear Learner to implement the **Marcus Correction** data-drivenly:

$$ \text{Features} = [ \Phi_{\text{Chen}}(X), \text{BV}(X), \text{RV}(X), \text{PV}_{p=4}(X) ] $$

- **Regime Heston (Continuous)**: Checks $RV \approx BV$. Uses $\Phi_{\text{Chen}}$ for geometry and $BV$ for scale.
- **Regime Bates (Jumps)**: Checks $RV \gg BV$.
  - Uses $BV$ to estimate the Diffusive Volatility (Robustness).
  - Uses $RV - BV$ to estimate the Jump intensity/size.
  - Can explicitly construct the Level 2 Marcus Term: $\int dX^{\otimes 2} + \frac{1}{2}\sum (\Delta X)^2 \approx \Phi_{\text{Chen}}^{(2)} + \frac{1}{2}(RV - BV)$.

This approach "Operationalizes" the Marcus Signature without requiring explicit jump detection algorithms, relying on the **Universal Approximation** properties of linear functionals on this augmented feature space.
