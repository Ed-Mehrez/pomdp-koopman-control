# KKF for Insider Trading (Multiasset Kyle)

## Goal

Demonstrate the efficacy of **Signature-based Koopman (Sig-KKF)** in a quantitative finance setting. Specifically, we will address the **Market Maker's Filtering Problem** in a Multiasset Kyle framework where standard linear filtering (Kalman) fails.

## Theoretical Motivation

In the classic Kyle/Back models, if the asset value $\tilde{v}$ and noise $Z$ are Gaussian, the equilibrium pricing rule $P_t = E[\tilde{v} | Y_t]$ is linear in the order flow $Y_t$.
However, if the prior distribution of asset values is **Non-Gaussian** (e.g., bimodal "merger vs bankruptcy", or fat-tailed), the optimal pricing rule becomes highly nonlinear.

**Hypothesis**: Sig-KKF can learn this nonlinear pricing function $H(Y_t)$ purely from data, without knowing the underlying distributions, outperforming linear models (standard Kyle).

## Theoretical Connection: Back (2004)

As detailed in "Incomplete and Asymmetric Information in Asset Pricing Theory" (Back, 2004), the Market Maker's problem is formally a **Stochastic Filtering Problem**:

- **Signal**: $Y_t$ (Order flow), where $dY_t = h_t dt + dW_t$.
- **Filter**: $\hat{v}_t = E[\tilde{v} | \mathcal{F}^Y_t]$.
- **Innovation**: $dZ_t = dY_t - \hat{h}_t dt$.
  Our approach effectively uses **Sig-KKF as a Universal Nonlinear Filter** to solve this problem without prescribing the model dynamics ($h_t$).

## comparison to Traditional Bayesian Methods

The user asks: "How is this better than traditional Bayesian methods?"

1.  **Model-Free vs. Model-Based**:
    - **Bayesian (Particle Filter)**: Requires knowing the _exact_ SDEs and likelihoods $P(Y|\tilde{v})$. If your model of the correlation $\rho$ is slightly wrong, the filter fails.
    - **Sig-KKF**: Learns the filter operator directly from data. It doesn't need to know _how_ $\tilde{v}$ affects $Y$, only that it _does_.
2.  **Path Dependency**:
    - **Bayesian**: Hard to handle path-dependent pricing rules without massive state augmentation.
    - **Sig-KKF**: Signatures ($\Psi(Y_{[0,t]})$) natively encode the entire path history, making "Meta-Learning" of the prior straightforward.

## Reference System (@research/Multiasset Kyle)

Based on `MultiassetDynamicKyleModel_Draft3.tex`:

- **Assets**: $K$ risky assets.
- **Noise Process**: $dZ_t = \Sigma_z dW_t$, where $\Sigma_z$ captures cross-asset noise correlations (Eq 46).
- **Insider**: Observes $\tilde{v}$ and $Z_t$. Optimal strategy involves hiding trades within the noise covariance structure.
- **Market Maker**: Sets $P_t = E[\tilde{v} | Y_{[0,t]]$.

## The "Sig-KKF" Opportunity

The paper derives an analytical pricing rule (Eq 138) which is explicit _only_ when the prior $F(\tilde{v})$ and noise distributions work out nicely (e.g. Gaussian).
**We will simulate a case where the analytical solution is intractable:**

- **Asset Priors**: $\tilde{v} \sim \text{Mixture of Gaussians}$ (Binomial/Regime outcome), correlated across assets.
- **Goal**: Show Sig-KKF learns the correct nonlinear manifold $P(Y)$ without knowing the mixture weights or parameters.

### Economic Hypothesis: Cross-Asset Manipulation

The user hypothesizes that **nonlinear equilibria** allow for (or prevent) complex manipulation strategies that linear models miss.

- **Scenario**: Non-Gaussian liquidity shocks or value priors allow an insider to "bluff" in Asset A to manipulate Asset B via correlation $\rho$.
- **Test**: Compare a **Linear Market Maker** (vulnerable to bluff) vs. **Sig-KKF Market Maker** (detects nonlinear bluff pattern).

## Implementation Steps

### 1. `FinanceEnvironment` Setup

Create `src/environments/finance_kyle.py`:

- **Class**: `KyleMultiAssetEnv`
- **Params**: $K$ (assets), $\Sigma_z$ (noise cov), $\Sigma_v$ (value cov), Mixture Weights.
- **Dynamics**:
  - Generate $\tilde{v}$ from GMM.
  - Generate Noise Path $Z_t$.
  - **Insider Strategy**: Use a "Suboptimal" rational strategy (e.g. Linear TWAP scaled by signal) to generate data $X_t$, since deriving the true equilibrium optimal control for the non-Gaussian case is effectively solving the HJB we are trying to bypass.
- **Output**: Episodes of $(t, Y_t, \tilde{v})$.

### Experiment 2: Online Learning (The "Repeated Game")

To answer "How does the MM know the prior?", we simulate a **Repeated Game** over $N$ days.

- **Setup**: MM starts with a "Blank" (uninformed) model.
- **Loop (Day $k=1 \dots N$)**:
  1.  Nature draws $\tilde{v}_k$ from the unknown prior.
  2.  Trading occurs (Insider knows $\tilde{v}_k$, MM observes $Y_t$).
  3.  At $T=1$, $\tilde{v}_k$ is revealed (e.g., earnings announcement).
  4.  MM updates Sig-KKF weights using $(Y^{(k)}_{0:1}, \tilde{v}_k)$.
- **Metric**: Plot **Pricing Error vs. Day**. We expect Sig-KKF to converge to the optimal nonlinear filter, effectively "learning the prior" from history.

### 3. Execution (`examples/proof_of_concept/experiment_kyle_kkf.py`)

- **Data Gen**: Simulate 1000 "days" (episodes) of trading.
  - Regime A (Prob 0.5): High Correlation, High Value.
  - Regime B (Prob 0.5): Low Correlation, Low Value.
- **Train**:
  - Fit Linear Regression (scalar $\lambda$) - Baseline.
  - Fit Sig-KKF (Kernelized/Signature features) - Online Update.
- **Eval**: Compare MSE of price discovery ($|P_t - \tilde{v}|^2$).
- **Viz**: Plot the learned Pricing Rule $P(Y)$ vs the theoretical optimal.

## Deliverables

- `experiment_kyle_kkf.py`: The executable script.
- `kyle_pricing_rule.png`: Visualization of the nonlinear pricing manifold discovered by KKF.
- `kyle_kkf_comparison.png`: MSE comparison vs time.
