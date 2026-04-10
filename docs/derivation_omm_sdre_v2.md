# Derivation note — OMM SDRE v2 (dynamics-agnostic)

**Status**: AUDIT-RESOLVED DRAFT (Claude, revised 2026-04-08 after codex audit pass). Items 1-6 of the math audit are resolved and applied below; the magnitude prediction in §6 has been recomputed with the corrected per-contract vega convention; the CRRA(γ=2) ship rule is updated. Ready for codex implementation. Not a thesis appendix.
**Scope**: Single-strike, single-option, single-underlying option market making under **general continuous-time stochastic dynamics** — Markovian (Heston, Bergomi, SABR) or path-dependent / fractional (rough Heston, rough Bergomi, rough Hawkes Heston, fBm log-spot). The agent is a POMDP-style controller observing the price path, with a dynamics-agnostic interface to the underlying.
**Sources**: Bergault & Guéant (2019) `[BG19]` for the HJB structure under classical SV, Cuchiero-Gazzani-Möller (2024) `[CGM24]` for the signature volatility model framework, Avellaneda & Stoikov (2008) `[AS08]` for the exponential intensity Hamiltonian, Davis & Lleo (2014) `[DL14]` for the general-utility extension via local quadratic approximation, Çimen (2008) `[C08]` for the SDRE methodology, Bayer-Friz-Gatheral (2016) `[BFG16]` for rough volatility background, Cont-Das (2022) `[CD22]` for the skeptical view. **Full reference list with annotations**: `docs/refs/fractional_methods_references.md`.
**This is NOT new mathematics.** It is a specialization and combination of published results, with explicit substitutions to make the controller dynamics-agnostic. Any "novelty" claim in the eventual thesis chapter is in the *combination* (BG HJB + signature dynamics + Davis-Lleo utility + disciplined evaluation), not in any individual derivation step.

---

## 0. Audit checklist (READ THIS FIRST if you're codex)

Before implementing anything, verify each of these items against the math below. If any item is unclear, push back BEFORE writing code. The Stage 4 v1 audit revealed that "implement against vibes" is exactly how methodology nulls happen.

### 0.1 Sign and unit conventions

- [ ] **Quote distance**: `δ⁻ ≥ 0` is the bid distance below the mid (so `bid_price = mid − δ⁻`) and `δ⁺ ≥ 0` is the ask distance above the mid (so `ask_price = mid + δ⁺`). **Confirm this matches `env.py`.**
- [ ] **Inventory**: `q > 0` means net long the option. **Confirm this matches `env.py`.**
- [ ] **Net delta `h`**: `h = stock_position + q · multiplier · Δ`, so `h > 0` means net long delta. **Confirm this matches `OptionMMState.net_delta`.**
- [ ] **Wealth**: `W = cash + q · multiplier · option_mid + stock_position · S`. **Confirm.**
- [ ] **Hedge convention**: For v2, `hedge_trade = −h_t` (perfect re-hedging at the start of each step). The 2D action reduces to `(δ⁻, δ⁺)`.

### 0.2 Risk-neutral closed-form recovery

- [ ] At `γ_local = 0` (risk-neutral), the optimal quotes are `δ⁻* = δ⁺* = 1/k`. With env `k = 5`, this gives half-spread = 0.20 in option-mid units.
- [x] **Resolved (codex audit, item 2)**: the env uses the BG/AS convention directly (`λ_bid = Λ_0 exp(−k(mid − bid))`, `λ_ask = Λ_0 exp(−k(ask − mid))`, see `env.py:149-150`). With `k = 5`, the BG/AS risk-neutral optimum is `1/k = 0.20`. The `option_mm_smoke.py` `constant_spread` baseline uses `half_spread = 0.05`, which is a **heuristic 4×-tight constant-spread baseline, NOT the BG risk-neutral optimum**. Stage 2's locked numbers (ΔCE = 26.97 ± 6.60) remain valid numerically but should be framed as "A-S beats a heuristic tighter constant-spread," not "A-S beats the risk-neutral baseline." Stage 4 v2's `bergault_gueant_closed_form` controller is the correct risk-neutral analytic baseline.

### 0.3 Inventory skew sign

- [ ] When long inventory (`q > 0`), the agent should quote tighter on the ask (`δ⁺ < 1/k`) and wider on the bid (`δ⁻ > 1/k`). This reduces inventory.
- [ ] Equivalent statement: the inventory-skew coefficient enters as `+inventory_skew` on the bid distance and `−inventory_skew` on the ask distance.

### 0.4 The dynamics-agnostic interface

- [ ] **The controller does NOT compute its quote formula from a closed form derived assuming Heston.** It calls an `inventory_variance_estimator(state, history) -> float` callable, which can be:
  - **Heston-specific**: `σ²_inv(s_t) = 𝒱²_contract · ξ² / 4` from the BG closed form (V-independent at leading order — see §3 and codex audit item 3)
  - **Empirical sliding-window**: realized variance of the option mid over a recent window
  - **Koopman-CdC on signatures**: `(L̂ O²)(s_t) − 2 O(s_t) (L̂ O)(s_t)` from a learned Koopman generator
- [x] **Resolved (codex audit, item 1)**: the inventory-skew coefficient is **unity, not 1/2**. The universal quote formula is `δ⁻* = 1/k + γ_local · σ²_inv · (T−t) · q / m`, `δ⁺* = 1/k − γ_local · σ²_inv · (T−t) · q / m`, where `m` is the option contract multiplier (resolved per item 4 below — `q` is in contracts, quotes are in per-share dollars, so the per-contract marginal value `p_contract = γ_local · q · σ²_inv · (T−t)` is converted to quote units by dividing by `m`).
- [ ] **The controller is identical for Heston, rough Heston, rough Bergomi, and any other dynamics class.** Only the estimator changes.

### 0.5 General utility via local Arrow-Pratt

- [ ] `γ_local(W) = −U''(W)/U'(W)`, evaluated at the current wealth at each step. CRRA(γ) gives `γ_local = γ/W`; CARA(α) gives `γ_local = α`; other utilities give whatever their Arrow-Pratt computation gives.
- [x] **Updated (codex audit, item 5)**: at typical OMM wealth (`W ≈ 1e5`), CRRA(γ=2) gives `γ_local = 2e−5`, which produces inventory skew of `~0.0156 $/share per unit q` (≈7.8% of `1/k = 0.20`) under default Heston params per the corrected §6 magnitude check. **Small but NOT "essentially zero"** — the previous "essentially zero" claim was downstream of the V̂ and multiplier bugs the codex audit caught and was off by ~10⁴. CARA(α=1e−3) gives `γ_local = 1e−3`, which produces a skew of `~0.78 $/share per unit q` (~390% of `1/k`) — way too aggressive and clips both quotes for `q ≥ 1`. Use matched Arrow-Pratt CARA(α = γ/W = 2e−5) instead for the CARA contrast.

### 0.6 The leading-order BG recovery

- [ ] **The "SDRE solver" for v2 is NOT a Riccati solve at runtime.** The leading-order BG closed form is itself a closed-form formula. The SDRE controller is just the formula evaluated at each state with `γ_local` and `σ²_inv` plugged in.
- [ ] The "SDRE methodology" claim earns its keep at higher orders, multi-strike, or when the closed-form linearization fails — not at the leading order. This is fine; recovering the analytical optimum on a published reference is a publishable validation result.

---

## 1. Notation

| Symbol | Meaning | Units |
|---|---|---|
| `t` | Current time | years |
| `T` | End of trading horizon (NOT option maturity) | years |
| `τ = T − t` | Time remaining in trading horizon | years |
| `T_M` | Option maturity | years |
| `S_t` | Spot price | $ |
| `s_t` | **Observable state representation**: in v2 this is `(q, h, V̂, τ, W)` for Heston; in Stage 5+ it could be `(q, h, Sig(price path), τ, W)` for rough vol | various |
| `V̂_t` | Filtered or empirical local variance estimate (EWMA in v2) | unitless |
| `μ` | Real-world drift on spot (default 0) | per year |
| `q_t` | Option inventory (signed) | contracts |
| `h_t` | Net delta exposure | shares |
| `m` | Option contract multiplier | 100 (default) |
| `Δ_t` | BS option delta at current state | dimensionless |
| `𝒱` | Option vega in BG convention `𝒱 = ∂_{√ν} O`. **Per-contract** throughout this note (i.e., `𝒱_contract = m · 𝒱_share`); convert per-share BS Greeks to per-contract once at the source so the rest of the math is multiplier-free. For Heston, computed from BS; for rough vol, from a model-free path estimator | $ · year^{1/2} per contract |
| `cash_t`, `W_t` | Cash, total wealth | $ |
| `δ⁻_t`, `δ⁺_t` | Bid distance, ask distance from option mid | $ |
| `bid_t`, `ask_t` | Bid price, ask price | $ |
| `Λ_0`, `k` | A-S 2008 exponential intensity params: `λ(δ) = Λ_0 exp(−k δ)` | various |
| `C_Q(s_t)` | Option mid (= BS price using `V̂_t` as σ² in v2; would be a more general path-functional in Stage 5+) | $ |
| `σ²_inv(s_t)` | **Local inventory variance estimator** — the dynamics-specific scalar that the SDRE controller consumes. With `q` in contracts, units are `$² · contract^{-2} · year^{-1}` so that `(γ_local/2) · q² · σ²_inv` has units `$/year` | $² · contract^{-2} · year^{-1} |
| `H^b(p)`, `H^a(p)` | Bid/ask Hamiltonians under exponential intensity | $ per year |
| `v(t, s_t)` | Reduced value function on the observable state | $ |
| `γ_local(W)` | Local Arrow-Pratt absolute risk aversion: `−U''(W)/U'(W)` | per $ |

**Sign convention summary:**
- `δ⁻, δ⁺ ≥ 0`: distances are non-negative.
- `q > 0`: net long option.
- `h > 0`: net long delta exposure.
- `γ_local > 0`: risk averse.

---

## 2. Setup: state, action, dynamics, and the dynamics-agnostic interface

### 2.1 The observable state representation `s_t`

The agent observes some finite-dimensional summary of the current state. **The agent does NOT have access to the underlying SDE.** It only sees what the env exposes plus what it can compute from past observations.

For Stage 4 v2 (Heston env, where we have a known SDE for validation purposes):
```
s_t = (q_t, h_t, V̂_t, τ_t, W_t)
```
where `V̂_t` is computed by the EWMA filter on observed log-returns of the spot path.

For Stage 5+ (rough vol or unknown dynamics):
```
s_t = (q_t, h_t, Sig(price path)_t, τ_t, W_t)
```
where `Sig(price path)_t` is a vector of signature features computed from the price path. This is what `signature_features.py` and `streaming_sig_kkf.py` already produce in the existing repo.

For real Alpaca data (Stage 7+):
```
s_t = (q_t, h_t, Sig(price + order book features)_t, τ_t, W_t)
```

**The controller's structure is identical across all three.** It computes a single scalar `σ²_inv(s_t)` from the state and plugs it into the universal quote formula.

### 2.2 Action

```
u_t = (δ⁻_t, δ⁺_t)
```

The third component of the env action (`hedge_trade`) is determined by the perfect-re-hedging rule (`hedge_trade_t = −h_t`) given `h_t`. So the SDRE solver only optimizes `(δ⁻, δ⁺)`. The env still receives the 3D `(bid, ask, hedge_trade)` action.

### 2.3 Dynamics (general)

We assume the price `S_t` follows some continuous-time stochastic process. **Crucially, we do not assume Markov property in `(S_t, V_t)`.** The process may be:

- Markovian in `(S, ν)` (Heston, Bergomi, SABR, etc.)
- Markovian in lifted state `(S, ν, additional factors)` (jump-diffusions, regime-switching)
- Path-dependent / non-Markov in any finite state (rough Heston, rough Bergomi, rough Hawkes Heston, fBm log-spot)

The Koopman generator is well-defined on observables `f: s ↦ ℝ` regardless of which class the underlying belongs to:

```
L f(s) := lim_{h→0} (E[f(s_{t+h}) | s_t = s] − f(s)) / h
```

For Markov diffusions this is the standard infinitesimal generator. For non-Markov dynamics (rough vol), the generator is well-defined on the *lifted* state space (signatures) where the lifted state IS Markov by construction.

**The controller does not need to know which class the underlying belongs to.** It just needs the local moments — specifically the inventory variance `σ²_inv(s_t)` — which can be estimated from the learned Koopman generator (Section 7.5 below) or from any other model-free moment estimator.

### 2.4 Inventory dynamics from fills

```
dq_t = dN^b_t − dN^a_t
```
with intensities `λ^b(δ⁻_t) = Λ_0 exp(−k δ⁻_t)` and `λ^a(δ⁺_t) = Λ_0 exp(−k δ⁺_t)`. **Bid fill increases inventory; ask fill decreases inventory.**

### 2.5 Wealth dynamics

When a bid fills (one contract), wealth changes by `δ⁻ · m` (modulo multiplier reconciliation). When an ask fills, wealth changes by `δ⁺ · m`. The continuous trading in the underlying for hedging contributes the standard delta-hedge P&L. **Codex must reconcile multiplier units carefully** — see Section 0 audit item.

### 2.6 Hedging rule (v2: perfect re-hedging)

For v2 we adopt the **continuous delta-hedging assumption** from `[BG19]`, modulo our discrete time stepping. Specifically: at the start of each step, the agent's hedge action is

```
hedge_trade_t = −h_t
```

This sets the net delta to zero BEFORE the step's fills happen. After the fills, `h` drifts to `h_post = m · Δ_t · (bid_fills − ask_fills)`, which is small for typical fill counts.

**Why this is the right v2 choice**: it matches `[BG19]`'s formulation exactly, eliminates `h` as an SDRE state variable (it's always 0 at the start of each step), and reduces the action dimension from 3 to 2. **Restore as a free state variable in Stage 5+ if needed.**

---

## 3. The HJB structure (dynamics-agnostic form)

We adopt the Bergault-Guéant (2019) HJB structure but write it in dynamics-agnostic form. The value function `v(t, s_t, q)` satisfies:

$$
0 = \partial_t v + \mathcal{L}_{\text{state}} v - \frac{\gamma_{\text{local}}(W)}{2} \cdot q^2 \cdot \sigma^2_{\text{inv}}(s_t) + \mathcal{H}(s_t, q)
$$

with terminal condition `v(T, s_T, q) = 0`, where:

- `𝓛_state` is the Koopman generator of the *observable state process* (excluding inventory `q`, which is controlled). For Heston this is the standard Heston generator on `(S, ν)`. For rough vol this is the lifted Koopman generator on signature features. **The controller doesn't need to know `𝓛_state` explicitly** — it just needs `σ²_inv(s_t)`, which is one of `𝓛_state`'s outputs (Section 7.5).

- `−(γ_local/2) · q² · σ²_inv(s_t)` is the **inventory penalty term**. This is the dynamics-agnostic generalization of the Bergault-Guéant `−(γξ²/8)(q𝒱)²` term. The Heston-specific quantities `ξ²` and `𝒱²` are absorbed into `σ²_inv(s_t)`, which is now a generic scalar local variance. **For Heston with the BG vega convention `𝒱 = ∂_{√ν} O`, this term is V-independent at leading order** — the `√ν` in the diffusion `ξ√ν dW^ν` is absorbed by the change of variable to `√ν` (see §3 reconciliation below and codex audit item 3).

- `𝓗(s_t, q)` is the contribution from optimal quoting:

$$
\mathcal{H}(s_t, q) = H^b(p^b(s_t, q)) + H^a(p^a(s_t, q))
$$

with `H^b`, `H^a` the bid/ask Hamiltonians under exponential intensity (Section 4), and `p^b = p^a = −∂_q v` to leading order.

**Critical observation**: this PDE has the same *structure* as the BG (2019) HJB, but with the Heston-specific term `(γξ²/8)(q𝒱)²` replaced by the dynamics-agnostic term `(γ_local/2) · q² · σ²_inv(s_t)`. **The substitution is exact** — for Heston, `σ²_inv = 𝒱² · ξ² / 4` (Section 7.5.1 below — V-independent because the `√ν` factor in the vol diffusion is absorbed by the BG change of variable to `√ν`), which gives:

$$
\frac{\gamma_{\text{local}}}{2} \cdot q^2 \cdot \frac{\mathcal{V}^2 \cdot \xi^2}{4} = \frac{\gamma_{\text{local}} \cdot \xi^2 \cdot \mathcal{V}^2}{8} \cdot q^2
$$

For `γ_local = γ` (constant CARA), this matches the BG `(γξ²/8)(q𝒱)²` term **exactly**, no missing factor. **Reconciliation (codex audit, item 3)**: BG's penalty is genuinely V-independent because they use `𝒱 = ∂_{√ν} O`, and the chain rule absorbs `√ν` from the diffusion: `∂_ν O · ξ√ν dW = (ξ/2) · ∂_{√ν} O · dW = (ξ/2) · 𝒱 · dW`. Squaring gives the BG inventory cost with no remaining `ν_t` factor. The previous "V̂-weighted" form in this note was a bug from confusing `∂_ν O` with `∂_{√ν} O`. Cite BG (2019) Eq. (2). If BG's orthogonalized-vol-risk form is used explicitly (post-delta-hedging), multiply by `(1−ρ²)`.

---

## 4. Hamiltonians under exponential intensity (Avellaneda-Stoikov 2008)

This section is unchanged from the previous version of the note. The exponential intensity Hamiltonian gives a closed form regardless of the dynamics class.

The bid Hamiltonian:
$$H^b(p) = \sup_{\delta^- \geq 0} \Lambda_0 e^{-k \delta^-} (\delta^- - p)$$

First-order condition: `−k(δ⁻ − p) + 1 = 0` ⇒ `δ⁻* = (1/k) + p`.

Hamiltonian value at the optimum: `H^b(p) = (Λ_0 / (k e)) · e^{-k p}`.

Symmetrically, `δ⁺* = (1/k) − p`, `H^a(p) = (Λ_0 / (k e)) · e^{+k p}`.

**The total spread** `δ⁻* + δ⁺* = 2/k` is constant in `p`. Only the *skew* depends on inventory. **This is structural to exponential intensity and does not generalize to power-law intensity.**

---

## 5. Risk-neutral special case (closed form for the linear-rule baseline)

When `γ_local = 0` (no inventory penalty), the value function is independent of `q` (assuming no P-vs-Q misspecification, which we adopt for v2). To leading order:

$$
v(t, s, q) = \theta_0(t, s)
$$

The marginal value `p^* = −∂_q v = 0`, giving:

$$
\boxed{\delta^{-*} = \delta^{+*} = \frac{1}{k} \quad \text{(risk-neutral, dynamics-agnostic)}}
$$

This is the **BG/AS risk-neutral analytic baseline** for Stage 4 v2. It's mathematically identical to a constant-spread controller with `half_spread = 1/k`. **It's also the same answer regardless of whether the underlying is Heston, rough Heston, or anything else** — the inventory term is the only place the dynamics enter, and at `γ_local = 0` there's no inventory term.

**Codex audit (item 2) confirmed**: `option_mm_smoke.py` uses `half_spread = 0.05`, which is a **heuristic 4×-tight constant-spread baseline, NOT the BG/AS risk-neutral optimum** `1/k = 0.20`. Stage 4 v2's `bergault_gueant_closed_form` controller is the proper risk-neutral analytic baseline; the existing `constant_spread` controller stays as a separate heuristic anchor. Stage 2's locked numbers are valid but the framing changes from "A-S beats the risk-neutral baseline" to "A-S beats a heuristic tighter constant-spread baseline."

---

## 6. Risk-averse case (the SDRE quote formula)

When `γ_local > 0`, the inventory penalty contributes to the value function. The natural ansatz is quadratic-in-`q`:

$$
v(t, s, q) = \theta_0(t, s) + \tfrac{1}{2} q^2 \cdot \theta_2(t, s)
$$

(The linear-in-`q` term `θ_1` vanishes under our `μ = 0`, no-misspec setting, as in BG.)

Substituting into the HJB and matching powers of `q`:

**Order 2** (quadratic in `q`):

$$
0 = \partial_t \theta_2 + \mathcal{L}_{\text{state}} \theta_2 - \gamma_{\text{local}}(W) \cdot \sigma^2_{\text{inv}}(s_t)
$$

with terminal condition `θ_2(T, s) = 0`.

If `θ_2` is approximately independent of `s` over the trading horizon (the leading-order approximation, valid when `T = T_horizon ≪ T_relaxation`), the equation reduces to:

$$
\dot{\theta}_2 = \gamma_{\text{local}}(W) \cdot \sigma^2_{\text{inv}}(s_t)
$$

with terminal `θ_2(T) = 0`, giving:

$$
\theta_2(t) = -\gamma_{\text{local}}(W) \cdot \sigma^2_{\text{inv}}(s_t) \cdot (T - t)
$$

(The `s_t`-dependence on the right-hand side is held fixed at the current state — this is the SDRE local approximation: re-evaluate `θ_2` at each step using the current state's `σ²_inv`.)

The marginal value (in per-contract dollars, since `q` is in contracts):
$$
p^*_{\text{contract}} = -\partial_q v = -q \cdot \theta_2 = +\gamma_{\text{local}}(W) \cdot \sigma^2_{\text{inv}}(s_t) \cdot (T - t) \cdot q
$$

To enter the quote formula (which is in per-share dollars), convert via the multiplier `m`:
$$
p^*_{\text{quote}} = \frac{p^*_{\text{contract}}}{m}
$$

**Optimal quotes:**

$$
\boxed{\delta^{-*}(s_t) = \frac{1}{k} + \frac{\gamma_{\text{local}}(W) \cdot \sigma^2_{\text{inv}}(s_t) \cdot (T - t) \cdot q}{m}}
$$

$$
\boxed{\delta^{+*}(s_t) = \frac{1}{k} - \frac{\gamma_{\text{local}}(W) \cdot \sigma^2_{\text{inv}}(s_t) \cdot (T - t) \cdot q}{m}}
$$

**Coefficient resolution (codex audit, item 1)**: the inventory-skew coefficient is **unity, not 1/2**. The earlier draft had `(γ_local/2)` in the boxed formula, which contradicted the derivation immediately above. The `1/2` from the quadratic ansatz `v = θ_0 + (1/2) q² θ_2` is consumed by `∂_q v = q · θ_2`; it does not survive into `p*`. Cross-checked against BG (2019) Eq. for `δ*` under exponential intensity.

**Sign check:**
- `q > 0` (long): `δ⁻* > 1/k` (wider bid → fewer purchases) and `δ⁺* < 1/k` (tighter ask → more sales). ✓
- `q < 0` (short): `δ⁻* < 1/k` and `δ⁺* > 1/k`. ✓
- `q = 0`: recovers risk-neutral case. ✓

**Magnitude check** (corrected after codex audit items 3 and 4 — uses per-contract vega throughout, V-independent Heston `σ²_inv`, and the multiplier conversion in the boxed formula).

Heston defaults: `S = 100`, `K = 100`, `T_M = 1` year, `r = 0`, `σ = 0.2` (`V = 0.04`), `ξ = 0.5`, `m = 100`, `k = 5` (so `1/k = 0.20`), trading horizon `T − t = 20 · (1/252) ≈ 0.0794` year.

**Step 1 — BS vega per share**, `𝒱 = ∂O/∂σ = S · √(T_M) · φ(d₁)`:
```
d₁ = (ln(S/K) + (r + σ²/2)·T_M) / (σ · √T_M) = (0 + 0.02·1) / (0.2·1) = 0.1
φ(0.1) ≈ 0.397
𝒱_share = 100 · 1 · 0.397 ≈ 39.7   [$ · year^{1/2} per share]
```

**Step 2 — convert to per-contract** (item 4 convention, applied once at the source):
```
𝒱_contract = m · 𝒱_share = 100 · 39.7 = 3970   [$ · year^{1/2} per contract]
```

**Step 3 — Heston `σ²_inv`** (item 3, V-independent):
```
σ²_inv = ξ² · 𝒱²_contract / 4 = 0.25 · 3970² / 4 ≈ 9.85e5   [$² · year^{-1} · contract^{-2}]
```

**Step 4 — quote-unit skew per unit of inventory** (item 1, unity coefficient; item 4, divide by `m`):
```
skew_quote / q = γ_local · σ²_inv · (T−t) / m
              = γ_local · 9.85e5 · 0.0794 / 100
              ≈ γ_local · 782    [$/share per contract of inventory]
```

**Plug in three utilities:**

| Utility | `γ_local` | skew per `q=1` ($/share) | as % of `1/k = 0.20` | comment |
|---|---|---|---|---|
| **CRRA(γ=2) at `W = 1e5`** | `2e−5` | `≈ 0.0156` | **~7.8%** | Small but **not** "essentially zero" — the previous note had this off by ~10⁴ due to the V̂ and multiplier bugs |
| Matched CARA(α = γ/W = 2e−5) | `2e−5` | `≈ 0.0156` | ~7.8% | Same as CRRA at this wealth, by Arrow-Pratt construction |
| CARA(α = 1e−3) | `1e−3` | `≈ 0.78` | **390%** | Way too aggressive — clips both quotes for `q ≥ 1`. Do not use as default |

So under CRRA(γ=2) at `W = 1e5`, the per-contract skew is **about 1.6 cents on a 20 cent half-spread**. For typical inventories in the range `|q| ∈ [0, 5]`, the skew is `0–8 cents`, i.e., `0–40%` of the half-spread. That is **measurable but small**. CRRA(γ=2) is not "essentially zero" and the Stage 4 v2 SDRE controller is not mathematically forced to recover A-S to within float precision.

**Whether `sdre v2 − A-S` clears the `P(>0) ≥ 0.95` ship rule under CRRA depends on what risk aversion the existing `avellaneda_stoikov` controller is using.** Stage 2/3 used `gamma_inv = 0.1` (see `option_mm_gating.py:50`), which corresponds to an A-S inventory penalty of `risk_term = 0.1 · V̂ · (T−t) ≈ 0.1 · 0.04 · 0.0794 ≈ 3.18e−4` per unit of `q`. The dimensional unit of A-S's `gamma_inv` is unclear from the controller code (it may not be directly comparable to `γ_local` in `1/$`), and the existing A-S baseline appears to use `V̂` (Heston instantaneous variance, dimensionless / per-year) where the BG formula expects an option-mid variance in `$²/year/contract²`. **Codex should sanity-check the existing `avellaneda_stoikov` controller's units as part of the v2 implementation pre-flight**, but should not "fix" anything without a separate re-spec — the Stage 2/3 numbers are locked.

**Pre-registered ship rule update**: see §9 implementation map and `plan_omm_research.md` §5.4 — the CRRA(γ=2) gate is downgraded from `P(>0) ≥ 0.95` to a directional report, with the headline ship rule moving to validation framing (`|ΔCE_SDRE − BG closed form| / sd_post < 2`). The "improvement" gate moves to a higher-`γ_inv` regime per Plan B in `plan_omm_research.md` §6.

---

## 7. General utility extension via local Arrow-Pratt (Davis-Lleo construction)

For an arbitrary smooth concave utility `U(W)`, the local mean-variance approximation around the current wealth `W_t` gives:

$$\mathbb{E}[U(W + \delta W)] \approx U(W) + U'(W) \cdot \mathbb{E}[\delta W] - \tfrac{1}{2} |U''(W)| \cdot \mathbb{E}[(\delta W)^2]$$

Maximizing this is equivalent to maximizing `mean − ½ γ_local(W) · variance`, with the **local Arrow-Pratt absolute risk aversion** `γ_local(W) = −U''(W)/U'(W)`.

**The substitution rule is just**: in Section 6, `γ_local` is evaluated at the current wealth `W_t` from the `UtilitySpec`.

| Utility | `γ_local(W)` | Notes |
|---|---|---|
| CRRA `U(W) = W^(1−γ)/(1−γ)` | `γ/W` | Wealth-dependent |
| CARA `U(W) = −exp(−αW)` | `α` | Constant |
| Log `U(W) = log(W)` | `1/W` | CRRA(γ=1) limit |
| Quadratic `U(W) = W − (k/2)W²` | `k/(1−kW)` | Markowitz |
| HARA | `aγ/(aW+b)` | Two-parameter |
| OCE entropic | `λ` | Same as CARA |

---

## 7.5 Three estimators for `σ²_inv(s_t)` — the dynamics-specific input

The controller's *only* dynamics-specific input is the local inventory variance `σ²_inv(s_t)`. Different estimators correspond to different dynamics assumptions.

### 7.5.1 Heston-specific estimator (validation only)

For Heston dynamics, the inventory variance is the variance of a delta-hedged option position over a small time interval. From the Bergault-Guéant decomposition (with `𝒱 = ∂_{√ν} O` per-contract):

$$
\sigma^2_{\text{inv,Heston}} = \frac{\mathcal{V}^2 \cdot \xi^2}{4}
$$

(V-independent at leading order — see §3 reconciliation; codex audit item 3. If using BG's orthogonalized-vol-risk form post-delta-hedging, multiply by `(1 − ρ²)`.)

**Inputs**: option vega `𝒱_contract = m · 𝒱_share` (constant under BG vega approximation), Heston vol-of-vol `ξ`. **No `V̂_t` factor** appears in the leading-order BG penalty.

**Use case**: validation against the BG closed form. If we plug in this estimator, the SDRE controller should recover the BG analytical optimum to within numerical precision. **Test 1 in Stage 4 v2.**

**Limitation**: requires knowing it's Heston and knowing `ξ`. Not deployable on real data.

### 7.5.2 Empirical sliding-window estimator (model-free, simplest)

```
σ²_inv,empirical(s_t) = Var[ΔC_Q over a sliding window] / dt
```

where `ΔC_Q` is the change in option mid (or in the delta-hedged position value) over each step in a recent window. **No model assumptions; just compute the sample variance of the option mid changes from observed data.**

**Inputs**: window length, observed option mid history.

**Use case**: model-free claim. If the empirical estimator gives results comparable to the Heston-specific estimator on Heston data, the model-free claim is validated. **Test 2 in Stage 4 v2.**

**Limitation**: at H ≠ 1/2 (rough vol), the naive sliding-window estimator is biased due to long-memory correlations. The fix is fGN whitening (per `fsde-identifiability/README.md`), which is a Stage 5+ extension.

### 7.5.3 Other estimators (Stage 5+ extensions, NOT in v2 scope)

Several more sophisticated estimators are out of scope for v2 but worth flagging here so codex understands what the dynamics-agnostic interface is designed to support eventually:

- **Other Markovian SV models** (Stage 5): `three_halves_estimator`, `sabr_estimator`, etc. — same interface, different model-specific math. See Section 11 below.
- **fGN-whitened empirical** (Stage 5+, fSDE-dependent): for rough vol envs, the naive sliding-window estimator is biased due to long-memory correlations; the fix is fGN whitening. **Requires the `fsde-identifiability` sibling repo to be audited and partially imported. See `docs/fsde_audit_task.md` for the audit task spec.**
- **Koopman Carré-du-Champ on signature features** (Stage 5+, fSDE-dependent): the principled model-free estimator that uses `σ²(s_t) = (L̂ O²)(s_t) − 2 O(s_t) · (L̂ O)(s_t)` from a learned Koopman generator on path signatures. **Same dependency on the `fsde-identifiability` audit. See `docs/fsde_audit_task.md` Section 2.1 for the intended usage.**

**For Stage 4 v2**: implement only the Heston-specific and empirical sliding-window estimators above. Defer all signature-based and fSDE-dependent estimators until the audit task clears. **Codex should NOT import anything from `fsde-identifiability` for Stage 4 v2.**

### 7.5.4 Summary table

| Estimator | Dynamics knowledge | Stage | Use case |
|---|---|---|---|
| `bergault_gueant_heston` | Heston (parametric) | **v2** | Validation against BG analytical optimum |
| `empirical_sliding_window` | None | **v2** | Model-free claim on Heston |
| `three_halves_estimator` | 3/2 model (parametric) | Stage 5 | Demonstrate dynamics generalization |
| `sabr_estimator` | SABR (parametric) | Stage 5 | Demonstrate dynamics generalization |
| `fgn_whitened_empirical` | Hurst H from data | Stage 5+ (post-audit) | Model-free claim on rough vol |
| `koopman_cdc_signatures` | None (learned generator) | Stage 5+ (post-audit) | Principled model-free, ties to Paper 1 |

**For Stage 4 v2**: implement only `bergault_gueant_heston` and `empirical_sliding_window`. Defer the rest.

---

## 8. The "SDRE solver" — what's actually being solved

Given the closed-form expression in Section 6, **the "SDRE solver" for Stage 4 v2 is a closed-form formula, not a runtime Riccati solve**. The "SDRE methodology" claim earns its keep at higher orders, multi-strike, or constraints — not at the leading order. For v2:

```python
def sdre_optimal_action_v2(
    state: OptionMMState,
    inventory_variance_estimator: Callable[[OptionMMState, History], float],
    utility: UtilitySpec,
    horizon_remaining: float,
    distance_slope: float,
    multiplier: float,            # NEW: required for the per-share quote conversion (item 4)
    history: History,
) -> OptionMMAction:
    """Dynamics-agnostic SDRE controller for OMM v2.

    Computes the optimal quotes via the closed-form Bergault-Guéant (2019)
    expression, generalized to arbitrary smooth utility via the Davis-Lleo
    (2014) local Arrow-Pratt construction, with the dynamics entering only
    through the `inventory_variance_estimator` callable.

    Per audit items 1 and 4: the inventory-skew coefficient is unity (NOT 1/2),
    and the per-contract marginal value `p_contract` is converted to per-share
    quote units by dividing by the multiplier `m`.
    """
    sigma_sq_inv = inventory_variance_estimator(state, history)
    gamma_local = utility.arrow_pratt(state.wealth)
    half_spread_base = 1.0 / distance_slope
    # Per-contract marginal value (units: $/contract)
    p_contract = gamma_local * sigma_sq_inv * horizon_remaining * state.option_inventory
    # Per-share quote conversion (item 4: divide by multiplier)
    p_quote = p_contract / multiplier
    bid_distance = max(0.0, half_spread_base + p_quote)
    ask_distance = max(0.0, half_spread_base - p_quote)
    bid = state.option_mid - bid_distance
    ask = state.option_mid + ask_distance
    hedge_trade = -state.net_delta
    return OptionMMAction(bid_price=bid, ask_price=ask, hedge_trade=hedge_trade)
```

**That's the entire controller.** The dynamics-agnostic part is the `inventory_variance_estimator` callable, which is passed in. Different estimators (Heston-specific, empirical, Koopman-CdC, etc.) give different controllers without changing the math.

**Stage 5+ extensions** add complexity to the SDRE solver:
1. **Higher-order corrections**: solve for `θ_3`, `θ_4`, ... and add cubic / quartic terms in `q`. Where genuine Riccati machinery enters.
2. **Multi-strike**: the closed form generalizes to a *system* of Riccati equations, exactly as in `[BEGV21]`. The 2D case is closed form; the multi-asset case requires a matrix Riccati solver.
3. **State-dependent / non-Markov dynamics**: the local linearization at each state uses signature features as the state representation; the local Riccati is solved on the lifted state.
4. **Constraints** (e.g., bid ≥ 0): use a quadratic program at each state instead of the closed form.

---

## 9. Implementation map

| Math | Code |
|---|---|
| `q_t` | `state.option_inventory` |
| `h_t` | `state.net_delta` |
| `V̂_t` (filtered, Heston) | output of `EWMAVarianceFilter.update(state.spot)` |
| `s_t` (general state) | `state` plus any additional features (signatures, etc.) |
| `S_t`, `cash_t`, `W_t` | `state.spot`, `state.cash`, `state.wealth` |
| `t`, `τ = T − t` | `state.step_index * env.dt`, `(env.horizon_steps − state.step_index) * env.dt` |
| `Δ_t` | `state.option_delta` |
| `𝒱` (constant vega, Heston, **per-contract**) | precomputed once at episode start as `m * BS_vega_per_share(...)` — multiplier applied at the source, multiplier-free downstream |
| `Λ_0`, `k` | `env.fills.base_intensity`, `env.fills.distance_slope` |
| `C_Q(s_t)` | `state.option_mid` (BS price using `V̂_t`) |
| `σ²_inv(s_t)` | output of `inventory_variance_estimator(state, history)` callable |
| `γ_local(W)` | `utility.arrow_pratt(W)` (NEW field in `UtilitySpec`) |
| `hedge_trade` | `−state.net_delta` (perfect re-hedging rule) |

**New code needed:**

1. **`metrics.py`**: add `arrow_pratt: Callable[[float], float]` to `UtilitySpec`. Update existing factories (`crra_utility`, `cara_utility`). Add `quadratic_utility(k)` factory for sanity check.

2. **New module `src/applications/option_mm/inventory_variance.py`**: holds the **two estimator factories** only (`bergault_gueant_heston_estimator`, `empirical_sliding_window_estimator`). The two new controller factories (`bergault_gueant_closed_form`, `sdre_controller_v2`) live in `controllers.py`. **Pinned (Phase A response, 2026-04-08)**: the split keeps `controllers.py` focused on policy logic and `inventory_variance.py` focused on the dynamics-specific moment estimators, and avoids stuffing four new functions into one file. The "no framework up front" rule is about not creating unused helpers, not about jamming everything into one file. **All stateful closures must be instantiated fresh per episode in the runner** — the empirical sliding-window deque must never be shared across seeds or strategies.
   ```python
   def bergault_gueant_heston_estimator(env_params) -> Callable:
       """Returns σ²_inv estimator that uses Heston-specific BS vega (per-contract) and ξ.

       Per the codex audit (items 3 + 4): vega is per-contract (`m * vega_per_share`),
       computed once at episode start; the inventory variance is V-independent at
       leading order (BG change of variable to √ν absorbs the √ν in the vol diffusion).
       """
       vega_per_share = compute_constant_vega_per_share(env_params)
       vega_contract = env_params.multiplier * vega_per_share
       xi = env_params.heston_xi
       def estimator(state, history):
           return (vega_contract**2 * xi**2) / 4.0   # NOTE: no state.variance factor
       return estimator

   def empirical_sliding_window_estimator(window_length: int = 10) -> Callable:
       """Returns σ²_inv estimator using sliding-window realized variance of option mid."""
       history_buffer = deque(maxlen=window_length)
       def estimator(state, history):
           history_buffer.append(state.option_mid)
           if len(history_buffer) < 2:
               return 0.0
           returns = np.diff(np.log(np.array(history_buffer)))  # or absolute diffs
           return float(np.var(returns) / dt)
       return estimator
   ```

3. **`controllers.py`**: add `bergault_gueant_closed_form` (the risk-neutral baseline at `1/k`) and `sdre_controller_v2` (the dynamics-agnostic full controller). Both consume an `inventory_variance_estimator` callable.

4. **`option_mm_ablation_v2.py`**: gating runner with seven controllers (no_quote, constant_spread, A-S(EWMA), bergault_gueant_closed_form, sdre_controller_v2 + Heston estimator, sdre_controller_v2 + empirical estimator, heuristic_linear v1). Compare via `paired_ce_posterior(method="delta")` from `metrics.py`, with `method="mc"` as the agreement check (per the existing Bayesian metrics layer; `paired_bayesian_bootstrap_posterior` is the fallback for non-smooth functionals only).

5. **Tests** in `test_option_mm_controllers.py`:
   - `bergault_gueant_closed_form` returns `bid = mid − 1/k` and `ask = mid + 1/k` for any state.
   - `sdre_controller_v2(estimator=heston_estimator, utility=crra_utility(2.0))` at `W = 1e5` produces an inventory skew on the order of `1.6 cents/share per unit of q` (≈ 7.8% of `1/k = 0.20`); the test asserts the skew is in this magnitude range and has the correct sign per §6, NOT "within `1e-6` of `bergault_gueant_closed_form`" — the previous draft's tolerance was downstream of the V̂ and multiplier bugs and is no longer supported.
   - `sdre_controller_v2(estimator=heston_estimator, utility=cara_utility(1.0))` produces measurable inventory skew at non-zero `q`.
   - **Cross-estimator agreement check**: for an env where we know it's Heston, both `heston_estimator` and `empirical_sliding_window_estimator` should give similar `σ²_inv` values to within MC noise. Test on a long synthetic trajectory.

---

## 10. The POMDP framing (one paragraph)

For Stage 4 v2 (Markovian classical SV: Heston, 3/2, SABR), the agent observes spot but not the latent variance — strict POMDP, handled by EWMA / Kalman / BPF filtering as in Stage 1-3. The "POMDP" in `pomdp-koopman-control` covers this case directly. The richer framing — where the dynamics themselves are non-Markov (rough vol, fSDEs) and the "hidden state" is the path history that signatures must lift — is documented in `docs/fsde_audit_task.md` Section 2 as part of the Stage 5+ intended-usage context. **For v2 we stay in the classical Markov POMDP regime; the non-Markov POMDP discussion is not on the v2 critical path.**

---

## 11. Stage 5 — alternative classical SV envs (preview, NOT v2 work)

The dynamics-agnostic interface designed in Sections 6-9 enables Stage 5 to test the same SDRE controller on additional Markovian stochastic vol models without re-derivation. **None of this is v2 work.** Codex implements only the Heston pieces in v2. The Stage 5 envs are previewed here so the v2 interface design can be sanity-checked against them.

### 11.1 The 3/2 model

```
dS_t = μ S_t dt + √V_t S_t dW^S_t
dV_t = κ V_t (θ − V_t) dt + ξ V_t^{3/2} dW^V_t
```

Same `(S, V)` Markov state as Heston; vol-of-vol is `ξ V^{3/2}` instead of Heston's `ξ √V`. Empirically argued to fit short-dated SPX implied vol better than Heston (Carr-Sun 2007, Lewis 2000). **No published OMM closed form** — Bergault-Guéant doesn't apply because the vol-of-vol structure breaks the BG ansatz.

For the SDRE inventory variance estimator, we'd need to derive `σ²_inv,3/2(V̂)` analogous to `σ²_inv,Heston = 𝒱²_contract · ξ² / 4` (V-independent at leading order under the BG `√ν` change of variable; see §3 and §7.5.1). The 3/2 model's `ξ V^{3/2}` diffusion will introduce a `V̂`-dependence here that Heston doesn't have. Or use the empirical sliding-window estimator (which works on any Markov SV without re-derivation).

### 11.2 SABR

```
dF_t = α_t F_t^β dW^F_t
dα_t = ν α_t dW^α_t
d⟨W^F, W^α⟩_t = ρ dt
```

Different state structure than Heston: `(F, α)` instead of `(S, V)`, with CEV-style spot dynamics (the `F^β` term). Industry standard for FX and rates options. Hagan et al. (2002) provides an asymptotic option pricing formula. **No published OMM closed form.**

The SDRE controller's empirical sliding-window estimator works without modification on SABR data — it just measures `Var[ΔC] / dt` from observed option mid changes, regardless of whether the underlying is Heston or SABR.

### 11.3 Stage 5 experimental design

Three envs (Heston, 3/2, SABR), same controllers, same paired Bayesian posterior inference discipline. Pre-registered ship rule: SDRE with the empirical estimator beats `avellaneda_stoikov_ewma` on each non-Heston env at `P(>0) ≥ 0.95`. The Heston env continues to validate against the BG closed form.

**Codex action for v2**: only implement Heston. Verify that the `inventory_variance_estimator` callable interface is general enough that adding 3/2 and SABR estimators in Stage 5 is purely additive (new functions, no changes to the controller). If the interface needs adjustment to accommodate Stage 5 envs, raise it during the audit before implementing.

---

## 12. Audit checklist for codex (re-stated explicitly)

Before implementing Stage 4 v2, codex should:

1. **Read this entire note end-to-end.** Push back on anything unclear.
2. **Verify each item in Section 0** against `env.py`, `metrics.py`, and the existing controllers.
3. **Re-derive Section 4 (the exponential intensity Hamiltonian) from scratch** on a scratch sheet. The first-order condition `−k(δ − p) + 1 = 0` gives `δ* = p + 1/k`. Confirm the sign and the prefactor.
4. ✓ **RESOLVED (audit item 2)**: §5 collapses to `δ⁻* = δ⁺* = 1/k`. The `option_mm_smoke.py` `half_spread = 0.05` is a heuristic 4×-tight constant-spread baseline, NOT the BG/AS optimum. Stage 4 v2's `bergault_gueant_closed_form` is the proper analytic baseline.
5. ✓ **RESOLVED (audit item 1)**: §6 inventory-skew coefficient is **unity, not 1/2**. The `θ_2` ansatz is independent of `s` to leading order (the SDRE local approximation, valid when `T_horizon ≪ T_relaxation`). The BG `(γξ²/8)(q𝒱)²` term and our `(γ_local/2)·q²·σ²_inv` term match exactly under `σ²_inv = 𝒱²ξ²/4` (V-independent — see resolved item 7).
6. ✓ **RESOLVED (audit item 5)**: §6 magnitude check recomputed with corrected per-contract scaling. Under CRRA(γ=2) at `W=1e5`, the inventory skew is `~0.0156 $/share per unit q` (~7.8% of `1/k`). Not "essentially zero." Ship rule downgraded to validation framing per `plan_omm_research.md` §5.4 update.
7. ✓ **RESOLVED (audit items 3 + 4)**: vega is `𝒱 = ∂_{√ν} O` per BG convention, **per-contract** (`m · vega_per_share`) throughout this note, multiplier-free downstream. Heston-specific estimator is `σ²_inv = 𝒱²·ξ²/4` (V-independent — the `√ν` from the vol diffusion is absorbed by the change of variable). The `q` in the inventory penalty is in contracts, the marginal value `p_contract = γ_local·q·σ²_inv·(T−t)` is in `$/contract`, and `p_quote = p_contract/m` for the per-share quote formula. **Triple-check this in the env's `_price_and_delta` function or the new BS vega helper. A factor of 100 error here changes the inventory skew by 10000×.**
8. **Verify the empirical estimator** gives sensible values on a long Heston trajectory: `Var[ΔC_Q] / dt` should be approximately equal to `𝒱²_contract · ξ² / 4` on average (up to MC noise — V-independent at leading order; see §7.5.1 and codex audit item 3).
9. **Read `docs/refs/fractional_methods_references.md` Sections 1, 4, and 5** for context on the rough vol debate and the sibling repo connections.
10. **Report findings** to Ed before writing code. If everything checks out, proceed to implementation. If anything is unclear, list the concerns and wait.

The audit is not optional. **Stage 4 v1 implemented "OMM-shaped" SDRE without a math spec and ended up with a controller missing the spread-capture revenue term.** The audit prevents that failure mode by forcing codex to read and validate the math before translating it to code.

---

## 13. Out-of-scope for v2 (deferred items)

The following items are deliberately NOT in v2 scope. They are documented in `docs/fsde_audit_task.md` Section 2 (intended usage) so codex understands what the dynamics-agnostic interface is designed to support eventually. **Codex should NOT implement any of them in v2.**

- **Koopman Carré-du-Champ inventory variance estimator on signature features** — depends on the `fsde-identifiability` audit clearing.
- **Dynamics regime detection via Koopman spectral roughness** (Paper 6a) — Stage 7+ deployment-time concern.
- **Signature features as a lifted Markov state representation** for non-Markov dynamics — Stage 5+, post-audit.
- **fGN whitening for rough vol settings** — Stage 5+, post-audit.
- **Rough volatility envs** (rough Heston, rBergomi, fBm log-spot, rough Hawkes Heston) — Stage 5+, post-audit.
- **Real Alpaca options data deployment** — Stage 7+.
- **The expanded POMDP framing for non-Markov dynamics** — captured in `docs/refs/fractional_methods_references.md` Section 5 and `docs/fsde_audit_task.md` Section 2; not on the v2 critical path.

If codex feels the urge to implement any of these during v2 work, **stop and re-read this section**. They are deferred for good reasons (scope discipline, fSDE audit dependency, thesis critical path). Bringing them into v2 scope would repeat the scope-creep failure mode that the `feedback_no_overscope_thesis_critical_path.md` memory pins.

---

## 14. References

See `docs/refs/fractional_methods_references.md` for the full annotated reference list. Quick lookup:

- `[BG19]` Bergault, P. & Guéant, O. (2019). *Algorithmic market making for options*. arXiv:1907.12433.
- `[BEGV21]` Bergault, P., Evangelista, D., Guéant, O. & Vieira, D. (2018/2021). *Closed-form approximations in multi-asset market making*. arXiv:1810.04383.
- `[AS08]` Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*.
- `[DL14]` Davis, M. H. A. & Lleo, S. (2014). *Risk-Sensitive Investment Management*.
- `[C08]` Çimen, T. (2008). *State-Dependent Riccati Equation (SDRE) Control: A Survey*.
- `[CGM24]` Cuchiero, C., Gazzani, G. & Möller, J. (2024). *Signature Volatility Models: Pricing and Hedging with Fourier*. arXiv:2402.01820.
- `[BFG16]` Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing under rough volatility*.
- `[GJR18]` Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). *Volatility is rough*.
- `[CD22]` Cont, R. & Das, P. (2022). *Rough volatility: fact or artefact?*. arXiv:2203.13820.
- `[BD20]` Bäuerle, N. & Desmettre, S. (2020). *Portfolio Optimization in Fractional and Rough Heston Models*.
- `[CJP15]` Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*.
- `[BMW24]` Buehler, H., Murray, P. & Wood, B. (2024). *Deep Bellman Hedging*.

Sibling repos:
- `fsde-identifiability/papers/jmlr_fsde_identifiability_outline.md` — JMLR-target paper on fSDE identifiability via signatures + Koopman.
- `fsde-identifiability/docs/spectral_roughness_theory.md` — Paper 6a on Koopman spectral roughness estimation.
- `fSDE_video_gen/docs/paper2_fsde_timeseries.md` — Paper 2 on fSDE generative models for time series.

---

## 15. Changelog

| Date | Change |
|---|---|
| 2026-04-08 | Initial draft (Claude). Heston-specific, single-utility. |
| 2026-04-08 | **Major revision**: dynamics-agnostic via the `inventory_variance_estimator` interface. Added Section 7.5 on the three estimators, Section 10 on the POMDP question, Section 12 on dynamics regime detection. Cross-references the new `docs/refs/fractional_methods_references.md`. |
| 2026-04-08 | **Scope-narrowing revision** (after user feedback on overscope): trimmed Section 7.5.3 (Koopman CdC estimator) and Section 12 (regime detection) — moved both to `docs/fsde_audit_task.md` Section 2 as "intended usage" context. Tightened Section 10 (POMDP question) to a single paragraph. Added new Section 11 previewing Stage 5 alternative classical SV envs (3/2, SABR). Added Section 13 listing out-of-scope items for v2. **The v2 spec is now Markov-only, with the dynamics-agnostic interface as the only structural innovation.** All fSDE / signature / Koopman CdC work is deferred to Stage 5+ pending the `docs/fsde_audit_task.md` audit. |
| 2026-04-08 | **Math audit pass complete** (codex audit + Claude resolution). Items resolved: (1) inventory-skew coefficient is unity, not 1/2; (2) `option_mm_smoke.py` `half_spread = 0.05` is a heuristic 4×-tight baseline, NOT the BG/AS optimum (Stage 2 framing updated); (3) Heston `σ²_inv = 𝒱²·ξ²/4` is V-independent at leading order — the `V̂` factor in the previous draft was a bug from confusing `∂_ν O` with `∂_{√ν} O`; (4) vega is per-contract throughout (multiplier applied once at the source), the inventory penalty `p_contract` is in `$/contract`, and the quote formula divides by `m`; (5) §6 magnitude check recomputed — CRRA(γ=2) at `W=1e5` produces `~0.0156 $/share per unit q` skew (~7.8% of `1/k`), not "essentially zero." Ship rule downgraded to validation framing under CRRA, with the improvement gate moving to a higher-`γ_inv` regime per Plan B. Stale references to "paired bootstrap" and the `1e-6` SDRE-vs-BG tolerance corrected. Note status updated to AUDIT-RESOLVED DRAFT, ready for codex implementation. |
| 2026-04-08 | **Phase A pre-flight fix** (codex pre-flight verification caught one stale spot): the §8 reference pseudocode at line 441 was missing the `/m` multiplier conversion in the `inventory_skew` line — inconsistent with the §0.4, §6, and §12 audit-resolved formulas. Fixed: the pseudocode now takes `multiplier` as a parameter, splits the marginal value into `p_contract` and `p_quote = p_contract / multiplier`, and applies `p_quote` to the bid/ask distances. **Codex Phase A also confirmed the legacy `avellaneda_stoikov` controller has a real units bug** (`risk_term = gamma_inv * v_hat * horizon_remaining` is dimensionally inconsistent with a quote-distance shift; `v_hat` is Heston instantaneous variance, not option-mid variance). Stage 2/3 numbers stay locked; the v2 writeup will note the A-S baseline is "the controller as actually implemented in `controllers.py:139`," not "textbook A-S." Codex also flagged that `OptionMMState` does not expose `multiplier` or `dt` directly — the v2 estimators and controller will need a factory pattern that closes over env params. Cross-stage wiring check pattern is established in `option_mm_ablation.py:50` and `:469` and will be reused in v2. |
| 2026-04-09 | **Phase A implementation clarification**: pinned Stage 4 v2 file split (`inventory_variance.py` for estimator factories, `controllers.py` for controller factories) and required episode-local estimator/controller closures to prevent cross-seed leakage from stateful sliding-window buffers. |
