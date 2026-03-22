---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 36px;
  }
  h2 {
    font-size: 30px;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
---

# Path Signatures in Market Microstructure

## Equilibrium Learning and Manipulation Detection

**Computational Methods for Path-Dependent Kyle Models**

---

# Motivation: When Does the Trading Path Matter?

Standard Kyle (1985): Terminal order flow $Y_T$ is a **sufficient statistic** for $v$

**But path structure matters when:**

1. **Manipulation detection** — Spoofers create distinctive path patterns
2. **Inventory costs** — _When_ inventory is held affects MM's cost
3. **Stochastic volatility** — Order flow during high-vol is less informative
4. **Multiple insiders** — Path reveals _who_ traded _when_

**This talk:** Develop operator-theoretic framework for path-dependent equilibria

---

# Outline

1. **Theoretical Framework**
   - Rational expectations operator $\mathcal{T}$
   - Fixed-point characterization of equilibrium
   - Contraction mapping and convergence

2. **Path Signatures**
   - Mathematical definition and properties
   - Lévy area and path information
   - Signature kernel methods

---

3. **Application I: Spoofing Detection**
   - Why manipulators have distinctive paths
   - Signature-based detection (AUC = 0.995)

4. **Application II: Strategic Bluffing**
   - When reversals are equilibrium behavior
   - Momentum MM exploitation (+34% profit)

5. **Application III: Inventory-Averse MM**
   - Path-dependent inventory costs
   - Time-varying liquidity emergence

---

# Part I: Theoretical Framework

---

# The Kyle Model

**Setup:** Single asset with true value $v \sim \pi$

**Agents:**

- **Informed insider:** Observes $v$ at $t=0$, trades $dX_t = \theta_t \, dt$
- **Noise traders:** Random orders $dZ_t = \sigma_z \, dW_t$
- **Market maker:** Sets price $P_t$ to break even

**Order flow dynamics:**
$$dY_t = \theta_t \, dt + \sigma_z \, dW_t$$

<!-- this is only from the information set of the insider right? -->

**Insider's problem:**
$$\max_{\{\theta_t\}} \mathbb{E}\left[ \int_0^T \theta_t (v - P_t) \, dt \right]$$

---

# Pricing Rules and Best Response

**Definition (Pricing Rule):** A measurable function $P: \mathbb{R} \to \mathbb{R}$

**Lemma (Insider's Optimal Strategy):**
If $P$ is differentiable with $\lambda(Y) := P'(Y) > 0$:
$$\theta^*(v, t, Y_t; P) = \frac{v - P(Y_t)}{\lambda(Y_t) \cdot (T - t)}$$

_Proof sketch:_ HJB equation with linear value function ansatz yields the result. ∎

<!-- do we validate that the ansatz is correct??? -->

**Key insight:** Insider's behavior depends on pricing rule $P$

---

# The Rational Expectations Operator

**Definition:** Given pricing rule $P$, insider's best response $\theta^*(P)$ induces a distribution over $(Y_T, v)$

**Definition (Rational Expectations Operator):**
$$[\mathcal{T}(P)](Y) := \mathbb{E}[v \mid Y_T = Y; \theta^*(P)]$$

**Interpretation:** $\mathcal{T}(P)$ is the optimal MM pricing if insider plays best response to $P$

**Definition (Equilibrium):**
$$P^* = \mathcal{T}(P^*)$$

Equilibrium is a **fixed point** of the operator

---

# Why Fixed Points?

**Traditional approach:** Guess-and-verify (works for Gaussian case)

**Our approach:** Iterate $P_{n+1} = \mathcal{T}(P_n)$

**Advantages:**

1. Handles arbitrary priors (multimodal, heavy-tailed)
2. Explicit convergence tracking
3. Natural extension to path-dependent settings
4. Computational tractability

**The key question:** Is $\mathcal{T}$ a contraction?

---

# Regularity Conditions

**Assumption 3.1 (Prior Regularity):**
$\pi$ is **sub-Gaussian**: $\exists \sigma_\pi > 0$ such that
$$\mathbb{E}_\pi\left[e^{\lambda(v - \mathbb{E}[v])}\right] \leq e^{\lambda^2 \sigma_\pi^2 / 2}$$

_Satisfied by:_ Gaussians, finite Gaussian mixtures, bounded support distributions

<!-- can we loosen this to a tail integrability condition?  -->

**Assumption 3.2 (Price Impact):**
$\exists \, 0 < \underline{\lambda} \leq \overline{\lambda} < \infty$: $\underline{\lambda} \leq P'(Y) \leq \overline{\lambda}$

**Assumption 3.3:** $\sigma_z > 0$ (non-degenerate noise)

---

# Main Contraction Result

**Theorem (Contraction Property):**
Under Assumptions 3.1-3.3, $\exists \kappa \in (0,1)$ such that:
$$\|\mathcal{T}(P_1) - \mathcal{T}(P_2)\|_\infty \leq \kappa \|P_1 - P_2\|_\infty$$

**Proof sketch:**

1. Best response $\theta^*(P)$ is Lipschitz in $P$ (Lemma 3.1)
2. Induced distribution $\mu_P$ is Wasserstein-stable (Lemma 3.2)
3. Conditional expectation is stable under distribution perturbation
4. Contraction constant:
   $$\kappa = \frac{L_\mu \cdot \sigma_\pi}{\sigma_z \sqrt{2\pi T}} < 1$$

when noise trading $\sigma_z$ is sufficiently large. ∎

---

# Corollaries

**Corollary (Existence & Uniqueness):**
Under the contraction condition, there exists a unique equilibrium $P^* \in \mathcal{P}$.

_Proof:_ Banach fixed-point theorem on complete metric space. ∎

**Corollary (Geometric Convergence):**
$$\|P_n - P^*\|_\infty \leq \kappa^n \|P_0 - P^*\|_\infty$$

**Practical implication:** $O(\log(1/\epsilon))$ iterations for $\epsilon$-accuracy

---

# Damped Iteration for Stability

**Algorithm:** $P_{n+1} = (1-\alpha)\mathcal{T}(P_n) + \alpha P_n$

**Proposition (Damped Contraction):**
Effective contraction constant: $\kappa_\alpha = (1-\alpha)\kappa + \alpha$

For Monte Carlo estimation with variance $\sigma^2_{\text{MC}}$, optimal damping:
$$\alpha^* = \frac{\sigma^2_{\text{MC}}}{\sigma^2_{\text{MC}} + (1-\kappa)^2}$$

**Empirical result:** With $\alpha = 0.3$, convergence in ~16 iterations
Contraction ratio $\approx 0.90$

---

# Part II: Path Signatures

---

# When Paths Matter: Beyond $Y_T$

**Standard result:** In Kyle, $Y_T$ is sufficient for $v$

**But this assumes:**

- No inventory costs
- No manipulation
<!-- i thought kyle showed that no manipulation was a result of the model -->
- Constant volatility
- Single insider

**With path dependence:** Need to condition on full path $\{Y_t\}_{t \in [0,T]}$

**Problem:** Infinite-dimensional! How to represent compactly?

---

# Path Signatures: Definition

**Definition (Signature):**
For a path $\gamma: [0,T] \to \mathbb{R}^d$, the signature is:
$$S(\gamma) = \left(1, \int_0^T d\gamma_t, \int_0^T \int_0^s d\gamma_s \otimes d\gamma_t, \ldots \right)$$

For augmented path $(t, Y_t) \in \mathbb{R}^2$:

| Level | Components              | Interpretation         |
| ----- | ----------------------- | ---------------------- |
| 0     | 1                       | Constant               |
| 1     | $(T, Y_T)$              | Time and terminal flow |
| 2     | Lévy area $\mathcal{A}$ | Path "shape"           |

---

# Lévy Area: The Key Level-2 Feature

**Definition:**
$$\mathcal{A} = \frac{1}{2}\int_0^T (Y_t \, dt - t \, dY_t)$$

**Geometric interpretation:** Signed area between path and its chord

**Properties:**

- $\mathcal{A} = 0$ for linear paths
- $\mathcal{A} \neq 0$ for "wiggly" or reversing paths
- Sensitive to _when_ trading occurs

**Key insight:** Lévy area captures information not in $(T, Y_T)$ alone

---

# Signature Kernel Methods

**Signature kernel:**
$$K(S_1, S_2) = 1 + \langle \text{sig}^{(1)}_1, \text{sig}^{(1)}_2 \rangle + \langle \text{sig}^{(2)}_1, \text{sig}^{(2)}_2 \rangle + \cdots$$

**For truncated level-2:**
$$K(S_1, S_2) = 1 + T_1 T_2 + Y_{T,1} Y_{T,2} + \mathcal{A}_1 \mathcal{A}_2$$

**Kernel Ridge Regression:**
$$\hat{P}(S) = \sum_{i=1}^n \alpha_i K(S, S_i)$$

where $\alpha = (K + \lambda I)^{-1} \mathbf{v}$

---

# Signature-Extended Equilibrium

**Definition (Signature Pricing Rule):**
$$P: S_{\leq k}(\gamma) \to \mathbb{R}$$

**Definition (Path-Dependent Operator):**
$$[\mathcal{T}_{\text{sig}}(P)](\sigma) := \mathbb{E}[v \mid S_{\leq k}(\gamma) = \sigma; \theta^*(P)]$$

**Theorem (Contraction for Signatures):**
Under regularity conditions on the signature kernel, $\mathcal{T}_{\text{sig}}$ is a contraction.

_Proof uses:_ Universal approximation of signatures (Lyons 2014)

---

# Part III: Application — Spoofing Detection

---

# The Spoofing Problem

**Spoofing/Layering strategy:**

1. Place large buy orders → MM learns "buying pressure" → raises price
2. Cancel orders, sell at inflated price
3. Profit from artificial markup

**SEC enforcement examples:**

- Navinder Sarao (2015 Flash Crash)
- Tower Research Capital ($67M fine, 2021)
- Numerous HFT cases

**Challenge:** Distinguish manipulation from legitimate informed trading

---

# Why Spoofing Creates Distinctive Paths

<div class="columns">
<div>

**Informed trader:**

- Knows $v$, trades toward target
- Monotonic accumulation
- Low |Lévy area|

**Manipulator:**

- No information about $v$
- Creates artificial pressure, then reverses
- High |Lévy area| due to roundtrip

</div>
<div>

![width:500px](../docs/spoofing_detection_v2.png)

</div>
</div>

---

# Mathematical Analysis

**Informed trader path:**
$$Y_t^{\text{inf}} \approx \frac{v - P_0}{\lambda} \cdot \frac{t}{T} + \sigma_z W_t$$

Approximately linear drift → $|\mathcal{A}| \approx O(\sigma_z)$

**Manipulator path:**

$$
Y_t^{\text{man}} = \begin{cases}
\theta_{\text{up}} \cdot t & t < \tau \\
\theta_{\text{up}} \cdot \tau - \theta_{\text{down}} \cdot (t-\tau) & t \geq \tau
\end{cases}
$$

Reversal creates large area: $|\mathcal{A}| \approx O(\theta_{\text{up}} \cdot \tau^2)$

---

# Empirical Results

| Metric | Informed  | Manipulator | Ratio |      |           |     |
| ------ | --------- | ----------- | ----- | ---- | --------- | --- |
| Mean   | Lévy area |             | 0.07  | 4.19 | **59.7×** |     |
| Std    | Lévy area |             | 0.06  | 1.27 | —         |     |

<!-- the table formatting here is messed up -->

**Detection performance:**

- **AUC = 0.995** (near-perfect classification)
- Simple threshold on |Lévy area| suffices

**Economic interpretation:**
Signature-aware pricing should **discount** high-|Lévy area| order flow

---

# Equilibrium with Manipulation Deterrence

**Robust pricing rule:**
$$P^*(S) = \mathbb{E}[v \mid S] \cdot \mathbb{1}_{|\mathcal{A}| < \tau} + P_0 \cdot \mathbb{1}_{|\mathcal{A}| \geq \tau}$$

**Effect:** Manipulator's expected profit becomes zero:

- Reversal detected → price reverts to prior
- No profit from artificial markup

**Policy implication:** Regulators could require signature-based surveillance

---

# Economic Critique: Addressed

**Q: "Why would manipulators use detectable patterns?"**
A: They must execute trades to move prices. Spoofing works against _naive_ sequential learners. Signature-aware MM makes it unprofitable.

**Q: "Could informed traders have reversals too?"**
A: Yes — and this is the key insight leading to Part IV.

**Q: "What about partial execution / queue position?"**
A: Model assumes all orders execute. Extension to LOB is future work.

---

# Part IV: Strategic Bluffing — When Reversals Are Optimal

---

# The Bluffing Question

**Critical observation:** If reversals always signal manipulation, why would anyone do it?

**Answer:** Reversals can be **equilibrium behavior** for _informed_ traders when:

1. MM uses **momentum-based** beliefs
2. MM extrapolates recent order flow direction
3. Informed trader can exploit MM's extrapolation bias

**Key distinction:**

- **Spoofing:** Uninformed manipulation (no private signal)
- **Bluffing:** Informed strategic reversal (has private signal)

---

# When Bluffing Is Equilibrium

**Momentum Market Maker:**
$$b_t = (1-\alpha) b_{t-1} + \alpha \cdot \text{sign}(dY_t)$$

MM forms beliefs using exponential moving average of order flow direction

<!-- Is this a standard MM belief update rule? It seems 'irrational' -->

**Bluffing strategy (for $v > P_0$):**

1. **Phase 1 ($t < \tau$):** Trade _against_ information
   - Sell when you know $v$ is high
   - MM's belief drifts: $b_t \to -1$ (bearish)
   - Price falls artificially

2. **Phase 2 ($t \geq \tau$):** Exploit mispricing
   - Buy aggressively at artificially low price
   - Profit from the reversal

---

# Bluffing vs Honest: Path Comparison

<div class="columns">
<div>

**Honest strategy:**

- Immediate accumulation toward $v$
- Monotonic path
- Low |Lévy area|

**Bluffing strategy:**

- Initial misdirection
- Sharp reversal at $\tau$
- High |Lévy area| (1.3× honest)

**Profit comparison:**

- **+34.1%** from bluffing vs honest

---

</div>
<div>

![width:1300px](../../docs/bluffing_equilibrium.png)

</div>
</div>

---

# When Is Bluffing NOT Optimal?

**Bluffing fails when MM uses:**

1. **Bayesian updating** — Properly conditions on $Y_T$
2. **Signature-aware pricing** — Detects reversal patterns
3. **Full-information filtering** — Uses entire path history

**Key insight:**

| MM Type          | Bluffing Profitable? | Detection Possible?       |
| ---------------- | -------------------- | ------------------------- |
| Naive (momentum) | Yes (+34%)           | No (extrapolates)         |
| Bayesian         | No                   | N/A (no bluffing)         |
| Signature-aware  | No                   | Yes (discounts reversals) |

---

# Implications for Detection

**The detection problem is nuanced:**

1. **Against naive MM:** Both bluffing and spoofing are profitable
   - But bluffing uses _real information_
   - Spoofing is pure noise exploitation

2. **Against signature-aware MM:** Neither is profitable
   - High |Lévy area| → discounted pricing
   - Manipulation becomes unprofitable

**Policy implication:**

Signature-based surveillance deters _both_ spoofing and strategic bluffing, aligning incentives toward honest trading.

---

# Part V: Inventory-Averse Market Maker

---

# Inventory Costs: Economic Motivation

**Standard Kyle:** MM has no holding costs (unrealistic)

**Reality:**

- Capital tied up in positions
- Risk exposure (VaR limits)
- Opportunity cost
- Regulatory requirements

**Model:** MM incurs cost $\gamma \int_0^T I_t^2 \, dt$ where $I_t = -Y_t$

---

# Modified Equilibrium Pricing

**MM's problem:**
$$\max_P \mathbb{E}\left[ \int_0^T (P_t - v) dY_t - \gamma \int_0^T I_t^2 \, dt \right]$$

**Solution:**
$$P_t = \underbrace{\mathbb{E}[v \mid \mathcal{F}_t]}_{\text{Information}} + \underbrace{2\gamma \cdot I_t \cdot (T-t)}_{\text{Inventory adjustment}}$$

<!-- I do not understand how this solution was derived and if it's correct under the conditioning information being signatures -->

**Interpretation:**

- Long inventory ($I > 0$) → offer discount to attract sellers
- Short inventory ($I < 0$) → charge premium

---

# Path Dependence Emerges

**Key insight:** Same $Y_T$, different paths → different costs

**Example:**

| Path | Strategy        | $Y_T$ | Inventory Cost |
| ---- | --------------- | ----- | -------------- |
| A    | Buy early, hold | 10    | 79.5           |
| B    | Hold, buy late  | 10    | 9.5            |

<!-- should there be some figure that goes along with this? I don't quite see how the inventory cost was calculated qualitatively even -->

**Cost ratio: 8.4×**

Early accumulation is costlier (longer holding period)

---

# Signature Captures Timing

**Time-weighted inventory:**
$$\int_0^T I_t \cdot (T-t) \, dt$$

**Connection to Lévy area:**
$$\int_0^T Y_t \, dt - T \cdot Y_T / 2 \approx \mathcal{A}$$

**Result:** Lévy area variant captures "when" inventory was held

Signature-based pricing naturally incorporates this

---

# Empirical Results

![width:900px](../docs/inventory_averse_mm_v2.png)

---

# Time-Varying Liquidity

**Effect of inventory costs on trading:**
V
| $\gamma$ | Y_T Std | Avg λ |
| -------- | ------- | -------- |
| 0.0 | 11.10 | baseline |
| 0.2 | 11.03 | +10% |
| 0.5 | 11.01 | +25% |

**Within episode:**

- λ increases as inventory builds
- Creates natural time-variation in liquidity

**Matches empirical patterns:** Bid-ask spreads widen with inventory

---

# Part VI: Optimal Filtering and Residual Whitening

---

# The Problem with Ad-Hoc Filtering Rules

**Standard Econometric Approaches:**

- **Microstructure Noise ($R$)**: Roll's (1984) estimator uses lag-1 autocorrelation $Cov(\Delta X_t, \Delta X_{t-1})$
- **Volatility Regimes ($Q$)**: Innovation matching uses EWMA of prediction errors

**Why this fails in continuous time:**

- Ad-hoc block sizes and rolling windows ($k_n$) break down under true regime shifts
- Empirical tuning leads to overfitting local noise artifacts rather than structural shifts
- They assume the underlying dynamics are locally linear

---

# The Koopman Approach: Residual Whitening

**The Core Insight:**
We do not need to "guess" tracking parameters. The mathematics of the Koopman Operator strictly define them through **Residual Whitening**.

**The Koopman Lifting:**
$$ dZ_t = A Z_t dt + C \, dW_t $$
where $Z_t$ is the infinite-dimensional Signature state.

**The Whitening Theorem:**
If the chosen basis (e.g., Nyström landmarks) is sufficiently rich, the residual innovation $v_t = Z_{t+dt} - (I + A dt)Z_t$ is forced by construction to be a **Martingale Difference Sequence**. State dynamics are structurally completely captured by $A$.

---

# Setting Q and R Principally

**Process Noise ($Q$):**
Because $A$ captures all deterministic non-linear regime shifts globally, $Q$ is strictly the stationary irreducible error covariance of the projection.
$$ Q = \text{Cov}(Z\_{t+dt} - (I + Adt)Z_t) $$

**Measurement Noise ($R$):**
The observable price tick error is strictly the projection residual from the lifted state back to the observation manifold.
$$ R = \text{Cov}(Y_t - H Z_t) $$

**Consequence:** By calibrating $Q$ and $R$ exactly to the orthogonal residuals of the Koopman fit over a training set, the Kalman Filter mathematically tracks the geometry optimally without _any_ required online hyperparameter auto-tuning.

---

# Connection to Hida Calculus

**Hida (White Noise) Calculus Interpretation:**
Standard Itô calculus struggles with integrating over limit-order book ticks because empirical paths have ill-defined infinite variation "jumps" (microstructure noise).

When we compute Signatures over these raw paths, the Koopman Operator naturally performs a **Hida Projection**:

1. The operator $A$ isolates the smooth, differentiable drift component of the stochastic path.
2. The orthogonal residual covariance $Q$ catches the generalized white noise distribution $\dot{W}_t$ energy.

This mathematically justifies time-invariant $Q$ and $R$ matrices: the operator structurally separates the regime vector field from the generalized white noise.

---

# Summary: Theoretical Contributions

1. **Operator-theoretic framework:** Equilibrium = fixed point of $\mathcal{T}$

2. **Contraction mapping:** Existence, uniqueness, geometric convergence

3. **Signature extension:** Handles path-dependent settings

4. **Computational tractability:** $O(\log(1/\epsilon))$ iterations

5. **Optimal Filtration:** Geometric residual whitening explicitly replaces empirical tuning

---

# Summary: Applications

**Spoofing Detection:**

- Lévy area discriminates manipulators (59.7× ratio)
- Near-perfect detection (AUC = 0.995)
- Policy-relevant for market surveillance

**Strategic Bluffing:**

- Informed traders may optimally reverse against momentum MM
- +34% profit improvement over honest strategy
- Signature-aware pricing deters both spoofing AND bluffing

**Inventory-Averse MM:**

- Path structure affects inventory costs (8.4× ratio)
- Time-varying liquidity emerges endogenously
- Signatures capture "when" inventory was held

---

# Directions for Future Work

1. **Limit order book extension:** Partial execution, queue priority

2. **Multiple insiders:** Strategic interaction, coordination

3. **Stochastic volatility:** Path-dependent information content

4. **Empirical validation:** Apply to TAQ/LOBSTER data

5. **Welfare analysis:** Optimal market design with signatures

---

# References

- Kyle, A.S. (1985). Continuous Auctions and Insider Trading. _Econometrica_
- Back, K. (1992). Insider Trading in Continuous Time. _RFS_
- Lyons, T.J. (2014). Rough Paths and Signatures. _ICM Proceedings_
- Avellaneda, M. & Stoikov, S. (2008). High-Frequency Trading in a Limit Order Book. _QF_
- Banach, S. (1922). Sur les opérations dans les ensembles abstraits. _Fund. Math._

---

# Thank You

**Code:** Available in `src/finance/` directory

- `kyle_fixed_point_operator.py` — Main theoretical implementation
- `spoofing_detection_v2.py` — Manipulation detection
- `bluffing_equilibrium.py` — Strategic bluffing analysis
- `inventory_averse_mm_v2.py` — Inventory costs

**Figures:** Generated from experimental runs

- All results reproducible with fixed seeds

**Questions?**
