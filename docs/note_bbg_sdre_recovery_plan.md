# BBG SDRE Recovery Plan

## Level-2 de-cheating line (2026-04-14)

Earlier recovery experiments supervise on BBG actions via
`build_sdre_from_bbg` (bbg_kernelized_recovery.py and siblings) or project
BBG coordinates onto the reduced basis. That line is now relabeled **oracle
reference only**, not a training target.

A new de-cheated Level-2 line replaces it:

- **Code**: `src/applications/option_mm_bbg/state_conditional_sdre.py`
- **Runner**: `finance/experiments/bbg_state_conditional_level2.py` (smoke/dev/formal)
- **No BBG supervision**: exploration is log-normal noise around the RN
  baseline (reused from `sdre_recovery`). Basis `U_r` is from the exploration-only
  bilinear/2-stage pipeline. Head targets are per-step spread capture and
  per-step ΔV^pi — no BBG action targets, no BBG coordinate targets, no
  BBG solver at training time.
- **New hypothesis (Stage A)**: state conditioning of the reduced control
  coefficients `(rev_lin_r, rev_quad_diag_r, vega_r)` via a Bayesian-ridge
  head on an 11-D hand summary of state. `Q_r(z)` is diagonal at Stage A,
  plus the analytic rank-1 vega-penalty outer product assembled inside the
  controller.
- **Comparison ladder**:
  1. primary A/B: state_cond vs current global SDRE (`make_sdre_recovery_controller`)
  2. floor: state_cond vs risk-neutral
  3. oracle reference (not claim target): state_cond vs BBG numerical
  4. also-oracle reference: BBG-supervised kernelized recovery (bbg_kernelized_recovery.py)
- **Stage A done-definition**: infrastructure correct, controller stable
  under trust region + data-scale concavity floor, suggestive signal (P(>0))
  at smoke/dev scale. No formal claim from smoke/dev.
- **Smoke result (2026-04-14, 200 explore + 50 eval seeds)**:
  state_cond − sdre_global = +25k, P(>0) = 0.88 (suggestive, wide CrI);
  state_cond − RN ≈ 0 (floor recovered); sdre_global − RN = −26k,
  P(>0) = 0.024 (the known SDRE-vs-RN regression persists under the new
  env realization).
- **Open for Stage A formal run**: pilot-based power calc (not yet run),
  seed stability check, variance audit of head predictions.
- **Stage B (not yet implemented)**: swap hand summary for DeepSets encoder
  from `state_encoder.py`; same head + controller. Only after Stage A is
  behaving sensibly.

## Theory separation

The BBG benchmark and the SDRE candidate controller are strictly separate:

**Benchmark** (src/applications/option_mm_bbg/solver.py):
- Solved reduced HJB on (t, nu, V^pi)
- Uses exact BBG intensity formulas, grid solver, Hamiltonian optimization
- Produces optimal quote tables given full model knowledge
- This is the **ground truth** we compare against

**Candidate** (src/applications/option_mm_bbg/sdre_recovery.py):
- Learns action-state dynamics from simulated transitions (exploration data)
- Forms a local quadratic Hamiltonian from estimated quantities
- Solves for quotes via local SDRE in a **learned reduced action subspace**
- **No BBG solver or value function inside the policy at action time**
- Risk-neutral distances used as baseline (competitive MM spread, not HJB)
- BBG appears only as an external benchmark for comparison

## Action reduction is data-driven

The full action space is 40D (20 bid + 20 ask distances). Action reduction
is learned from exploration data, NOT hand-designed from OMM heuristics.

Heuristic directions (global width, vega skew, maturity tilt, moneyness tilt)
are used ONLY for post-hoc interpretability validation.

## Method priority

### Option 2 — Reduced bilinear control coordinates (PRIMARY)

The main thesis-aligned route, implemented as a **two-stage** approach
(`bilinear_2stage`) after discovering that plain bilinear SVD suffers from
a dynamics-vs-objective orientation mismatch:

1. **Collect exploration data**: run episodes with log-normal noise on
   risk-neutral quotes, recording (state, action, inventory_changes,
   spread_capture, V^pi changes) per step.

2. **Fit bilinear dynamics**: ridge regression of value-weighted response
   (spread capture + z_i V_i weighted inventory changes) on normalized
   action features. Control channel B ∈ R^{40 × 21}.

3. **Stage 1 — SVD of B**: Find a dynamics-relevant **overspace** (rank ~10).
   The SVD captures the action directions that most affect value-relevant
   state changes, but these directions may not be optimally oriented for
   the control objective.

4. **Stage 2 — Hessian eigendecomposition within the overspace**: Construct
   the risk-adjusted action-value Hessian projected into the overspace, then
   extract the top-k eigenvectors. This combines dynamics awareness with
   objective-oriented basis selection.

5. **Local quadratic SDRE in reduced space**: combine learned revenue
   curvature (diagonal, from regression) with analytical vega penalty
   (rank-1, from learned vega channel) to form the Hessian. Solve
   the local quadratic for optimal reduced action per step.

6. **Map back**: u* = u_baseline + clip(U_r @ a*, ±0.8 × baseline).

**Key finding**: Plain bilinear SVD (dynamics basis) achieves CE ≈ 76K.
The two-stage approach (dynamics + objective) achieves CE ≈ 154K, on par
with ActionPCA and beating both baselines (RN=117K, BBG=79K).

Why primary: closest to the repo's bilinear/KRONIC/local-operator methodology.
The two-stage fix preserves the bilinear dynamics story while making the
optimization well-conditioned.

### Option 1 — Action PCA / Hessian eigendecomposition (SECONDARY)

Simpler fallback and comparison route:

1. Use the same learned revenue curvature and vega channel from the
   bilinear model's regressions.

2. Construct the full 40×40 action-value Hessian:
   H = diag(rev_quad) - c_pen * outer(vega_channel, vega_channel)

3. Eigendecomposition of H gives principal action directions ranked by
   objective curvature (most important = most negative eigenvalue).

4. Same SDRE optimization in the reduced eigenbasis.

Role: diagnostic baseline against Option 2. Tests whether the dynamics-based
basis (SVD of B) outperforms the objective-based basis (eigenvalues of H).

### Option 3 — RKHS / kernel active subspace (DEFERRED)

Not implemented in this round. Reserved as escalation path if Options 1 and 2
clearly fail.

## Formal evaluation outcome: C (mostly retuning)

### What was tested

Pre-registered anti-triviality and robustness checks:
1. Simple baselines: global_width (1 param) and global_width_skew (2 params)
2. Larger paired evaluation: 500 test episodes, disjoint from training
3. Grid refinement: BBG at medium/fine/finer grids
4. Interpretability: direction heatmaps, heuristic alignment, BBG reconstruction

### Key findings

| Controller | CARA CE (500 test eps) |
|---|---|
| risk_neutral | 104,985 |
| bbg_finer | 89,320 |
| global_width_skew (2 params) | 100,643 |
| action_pca_r1 (learned) | 106,126 |
| bilinear_2stage_r1 (learned) | 112,687 |

- The 2-parameter width+skew baseline reaches CE=100,643
- The learned controllers reach CE=106-113K
- The marginal improvement is +5-12K with P(>0)=0.65-0.77 (not decisive)
- The BBG grid is converged (CE stable at ~89K across all three resolutions)
- The learned direction at rank 1 targets ATM short-dated options specifically
- Heuristic projection fractions are low (<7%) — the direction is NOT simple width/skew

### Interpretation

The learned low-rank action structure is real: explained variance shows a clear
rank-2 elbow at 91%, and the learned directions have specific strike-maturity
targeting. But the CE improvement over a trivial 2-parameter baseline is marginal.

The main driver of the apparent "BBG-beating" result was the CARA CE metric's
preference for RN-like controllers at γσ ≈ 130. Under mean-variance, BBG
clearly wins over all data-driven controllers.

### What is established
- Low-rank action structure in 40D option MM (rank-2 captures 91% of dynamics)
- Learned directions are interpretable (target ATM short-dated, not global width)
- Two-stage bilinear method resolves the dynamics-vs-objective orientation mismatch
- BBG benchmark is grid-converged and numerically stable

### What is not established
- Clear superiority of learned controllers over trivial baselines
- Recovery of the BBG value function from data (learned controllers stay in RN regime)

## Paired recovery / equivalence test (Outcome R2)

**Date**: 2026-04-10
**Script**: `finance/experiments/bbg_recovery_equivalence.py`

### Recovery gate definition

For the paired difference Δ = CE_candidate − CE_BBG:
- **Gate A (ROPE)**: P(|Δ| ≤ h) ≥ 0.95
- **Gate B (precision)**: sd_post(Δ) ≤ s_max

Calibration from a 500-episode pilot:
- BBG-RN gap (pilot) = +18,370
- **h = 7,348** (40% of |gap|)
- **s_max = 7,348**

### Recovery results (2000 test episodes)

| Candidate | mean Δ | sd_post | P(ROPE) | Gate A | Gate B |
|---|---|---|---|---|---|
| demo_r1 (supervised) | -80,904 | 9,168 | 0.000 | FAIL | FAIL |
| demo_r3 (supervised) | -89,937 | 5,871 | 0.000 | FAIL | PASS |
| sdre_action_pca_r1 | **-8,194** | **6,289** | **0.537** | FAIL | **PASS** |
| sdre_action_pca_r2 | -8,069 | 6,549 | 0.543 | FAIL | PASS |
| sdre_action_pca_r3 | -9,011 | 6,071 | 0.366 | FAIL | PASS |
| sdre_bilinear_2stage_r2 | -3,454 | 20,240 | 0.005 | FAIL | FAIL |
| global_width_skew | +16,032 | 11,568 | 0.199 | FAIL | FAIL |

### BBG action surface rank structure

SVD of the BBG action perturbation matrix (from 15K state-action pairs):

| Directions | Cumulative variance |
|---|---|
| 1 | 45.9% |
| 2 | 86.8% |
| 3 | 98.4% |
| 5 | 100.0% |

Demonstration recovery (supervised fit, held-out test):
- Rank 1: R²=0.34, CosSim=0.43
- Rank 3: R²=0.70, CosSim=0.81
- Saturates at rank 5 (R²=0.81)

### Outcome: R2 (weak/noisy recovery)

**SDRE ActionPCA** at ranks 1-3 produces CE within ~8K of BBG (~10% of BBG CE).
Gate B passes (sd_post ≈ 6K < s_max). Gate A fails (P(ROPE) = 0.54, not 0.95).

The systematic offset of -8K represents the local quadratic approximation error.
The SDRE's diagonal revenue model and rank-1 vega penalty do not perfectly
reconstruct the BBG Hamiltonian optimization.

**Demonstration recovery** (supervised BBG action fit) fails catastrophically
at the CE level (mean wealth 180K vs 410K) because naive linear imitation of
BBG perturbations doesn't preserve fill-revenue dynamics.

**What this means:**
- The BBG action surface IS genuinely rank-3 (98.4% of variance in 3 directions)
- The SDRE machinery gets within ~10% of BBG but doesn't formally recover it
- The gap is systematic (model approximation), not statistical (noise)
- The demonstration approach shows the rank structure is real but a linear map
  from state features to action perturbations is not enough for CE recovery

## Kernelized state-conditioned recovery (corrected result)

**Initial artifact**: `finance/experiments/results/bbg_kernelized_recovery_2026-04-11.txt`
**Corrected rerun**: `finance/experiments/results/bbg_kernelized_recovery_2026-04-12_rerun.txt`

### Hypothesis

The ~8K systematic gap from non-kernelized SDRE recovery is caused by a
global-coefficient map that ignores state dependence. A KRR (ARD Matérn 3/2)
mapping state → reduced-action coordinates on a fixed ActionPCA/bilinear basis
should close it.

### Corrected results on the pre-registered test split (seeds 2000-3999)

The 2026-04-11 artifact above should **not** be used for formal gate
interpretation. That script still had a duplicate helper definition, so the
reported gate there was silently using the old bespoke bootstrap. The corrected
2026-04-12 rerun uses the metrics-layer paired CE posterior throughout.

Best kernelized controller: `kern_action_pca_r3_rich`

| Variant | CE | mean gap | sd_post | P(ROPE) | GA | GB |
|---|---|---|---|---|---|---|
| sdre_action_pca_r2 (non-kern) | 74,664 | -7,003 | 1,320 | 0.603 | F | P |
| kern_action_pca_r3_compact | 81,626 | -40 | 1,406 | 1.000 | P | P |
| kern_action_pca_r3_rich | 81,628 | -38 | 1,409 | 1.000 | P | P |
| BBG numerical | 81,666 | — | — | — | — | — |

Kernelization turns `ActionPCA r3` from a clear miss (`74,174` CE in the same
rerun) into formal recovery. The gain from non-kernelized to kernelized
`ActionPCA r3` is about `+7.45K` CE. Only the rank-3 ActionPCA basis recovers
BBG; ranks 1-2 remain around `74.4K`, and all kernelized bilinear-2stage
variants are poor.

### Interpretation

The corrected rerun establishes **formal BBG recovery on the pre-registered
split** for the kernelized `ActionPCA r3` controller. Two further points are
important:

1. `compact` and `rich` state summaries are numerically indistinguishable
   here, so the gain is coming from state-conditioned reduced coordinates,
   not from richer hand-built features.
2. The script crashed only in the late anti-triviality summary block due to an
   uninitialized local (`kern_labels`). That bug has been patched; the main
   evaluation and gate table above are valid.

## State summaries used here, and how to generalize them

The kernelized controller did **not** use the raw option-book state. It used
two low-dimensional summary maps from the full simulator state to a compact
kernel input:

### Summary actually used in this BBG env

**Compact summary (3D)**

\[
\phi_{\mathrm{compact}}(x)
= (\tau_{\mathrm{frac}}, \nu_{\mathrm{norm}}, V^\pi_{\mathrm{norm}})
\]

- `tau_frac`: horizon progress / time-to-go
- `nu_norm`: latent volatility-regime state, normalized by `nu0`
- `V^pi_norm`: aggregate signed portfolio vega, normalized by the vega limit

**Rich summary (7D)**

\[
\phi_{\mathrm{rich}}(x)
= (\tau_{\mathrm{frac}}, \nu_{\mathrm{norm}}, V^\pi_{\mathrm{norm}},
V^\pi_{\mathrm{short}}, V^\pi_{\mathrm{long}},
\mathrm{vega\_conc}, \mathrm{dist\_to\_limit})
\]

with:
- `V^pi_short`, `V^pi_long`: grouped exposure summaries by maturity bucket
- `vega_conc`: concentration of exposure in the largest single position
- `dist_to_limit`: slack to the active vega constraint

The important point is that these are **control-relevant summaries**, not
features designed to encode a quote rule directly.

### General framing

The right general object is not "use vega features." The right object is:

\[
\phi(x)
= (\text{time-to-go}, \text{latent regime}, \text{aggregate exposure},
\text{grouped exposure}, \text{concentration}, \text{constraint slack})
\]

This BBG application instantiates that template as:
- latent regime = instantaneous variance
- aggregate exposure = portfolio vega
- grouped exposure = short/long-maturity vega
- concentration = largest single-position vega share
- constraint slack = distance to the vega limit

So the generalizable claim is **not** "short-vs-long vega is the right feature
everywhere." The generalizable claim is:

> the state fed to the kernel should be a low-dimensional summary of the
> variables that determine local control geometry: horizon position, latent
> regime, aggregate exposure, grouped exposure, concentration, and distance to
> active constraints.

### A proposed menu for choosing summaries in other envs

This should be treated as a small family of specs to try, not a single fixed
recipe.

**Spec A — Minimal invariant summary**

Use the smallest summary that preserves the main control geometry:

\[
\phi_A(x)
= (\text{time-to-go}, \text{latent regime}, \text{dominant aggregate exposure})
\]

This is the cleanest first pass and is the closest analogue of the BBG reduced
state itself.

**Spec B — Grouped-exposure summary**

Add grouped exposure channels and constraint information:

\[
\phi_B(x)
= (\phi_A(x), \text{grouped exposures}, \text{concentration},
\text{constraint slack})
\]

Grouping should come from **control-channel metadata**, not ad hoc intuition:
- maturity / moneyness in option books
- spatial buckets in routing / control problems
- asset classes or factor buckets in portfolio problems
- queue tiers / venue classes in execution problems

This is the current "rich" BBG summary.

**Spec C — Learned grouped summary**

Instead of hand-chosen groups, derive grouped summaries from the learned control
geometry itself. For example:
- cluster control channels by action-basis loadings
- cluster channels by exposure vectors or response similarity
- compress the channel-exposure matrix with PCA / SVD and use the leading
  exposure factors as kernel inputs

This is the most defensible path if the goal is stronger cross-env
generalizability.

### Recommended usage across envs

The practical workflow should be:

1. start with **Spec A**
2. add **Spec B** if the controller is systematically biased
3. move to **Spec C** if grouped summaries appear necessary and hand-built
   groups are too application-specific

That gives a reusable progression:
- minimal summary first,
- structured grouped summary second,
- learned grouped summary third.

In this BBG env, the main empirical result is that moving from Spec A
(`compact`) to Spec B (`rich`) did **not** materially change recovery. That is
evidence that the remaining failure is not simply "missing more hand-built
state summaries." It shifts attention back to the local quadratic / Hamiltonian
approximation rather than feature scarcity alone.

## Nyström kernel tuning sweep (Outcome: tuning does not close the gap)

**Date**: 2026-04-12
**Script**: `finance/experiments/bbg_kernel_tuning.py`
**Results**: `finance/experiments/results/bbg_kernel_tuning_2026-04-12.txt`

### Why Nyström was added

Exact KRR on 3000 points takes ~15s per fit, making a 126-configuration
hyperparameter search impractical. Nyström KRR (M landmarks, subset-of-
regressors formulation) reduces fit time to ~0.5s, enabling a sweep over
`krr_alpha × ls_multiplier × n_landmarks` on a dev split while reserving
exact KRR for final confirmation of the top finalists.

### Sweep setup

- Basis: ActionPCA r3, compact state (3D)
- Grid: alpha ∈ {1e-4..1e-1} (7) × ls_mult ∈ {0.5..3.0} (6)
        × n_landmarks ∈ {256, 512, 1024} (3) = 126 configs
- Dev split: 200 train / 400 eval (seeds 1000-1399)
- Formal split: 500 train / 2000 test (seeds 2000-3999)
- 751s sweep (6.0s/config; fit 0.5s, eval 5.4s)

### Key findings

1. **M=256 is degenerate**: all 42 configs at M=256 produced CE≈103K
   (risk-neutral-like). 256 landmarks cannot approximate the kernel
   on 3000 training points with 3D features.

2. **All 4 exact-refit finalists converge to CE≈81625-81628:**

   | alpha | ls | dev gap | exact CE | gap | P(ROPE) |
   |---|---|---|---|---|---|
   | 3e-3 | 2.0 | -488 | 81625 | +588 | 0.747 |
   | 1e-2 | 1.5 | -227 | 81626 | +617 | 0.747 |
   | 1e-2 | 1.0 | -295 | 81626 | +661 | 0.743 |
   | 1e-1 | 0.75 | -290 | 81628 | +789 | 0.742 |

3. **Tuning moved the mean gap from +661 to +588 (11%)** — negligible.
   The KRR model has saturated: different hyperparameters all map to
   the same function.

4. The gate statistics printed in the original tuning artifact were generated
   before the duplicate-helper bug in `bbg_kernelized_recovery.py` was fixed.
   They should be treated as **stale for inference**. The useful conclusion
   from the tuning run is CE saturation, not the old bootstrap-based gate.

5. **Anti-triviality**: all finalists beat global_width_skew (CE=50974)
   by +12K, but that baseline is weak. The meaningful comparison is
   BBG (CE=81666) where the tuned exact gap is only +588 in mean CE.

6. **Nyström ranking partially preserved**: top Nyström pick → rank 2
   in exact refit. But all finalists are so close it doesn't matter.

### Conclusion

Hyperparameter tuning is not the bottleneck. The kernel model has
saturated at this basis and state representation. The corrected rerun
and precision evaluation show that the controller already recovers BBG
on the standard test split; tuning does not materially change that. The
remaining problem is **generalization**, not in-split CE fit.

### Is Nyström worth keeping?

As a **search tool**: yes. 751s for 126 configs vs an estimated ~1900s
with exact KRR (15s fit × 126). The ranking was directionally correct.

As a **production backend**: no. M=256 fails completely, and the exact
refit is only 23s per finalist — cheap enough for final confirmation.

Kept in `sdre_recovery.py` as `approx="nystrom"` with default `"exact"`.

## Inference switch: bootstrap → paired CE posterior (2026-04-12)

**Date**: 2026-04-12
**Script**: `finance/experiments/bbg_precision_evaluation.py`
**Results**: `finance/experiments/results/bbg_precision_evaluation_2026-04-12.txt`

### What changed

The recovery scripts were patched to use the metrics-layer
`paired_ce_posterior()` (delta method) instead of the bespoke paired
bootstrap. A follow-up audit found that `bbg_kernelized_recovery.py` still had
an old duplicate helper definition, so the initial 2026-04-11 kernelized
artifact remained contaminated. The corrected 2026-04-12 rerun removed that
override and is the authoritative kernelized-recovery result.

The delta method computes the asymptotic variance of the paired CE contrast via
the gradient of the CE functional and the paired utility-mean covariance — the
correct paired posterior approximation for this problem.

CRN pairing was already correct: same seed → same Heston path and
Poisson fill sequence, action-independent draw order.

### The old bootstrap was massively overestimating posterior width

| Method | mean gap | sd_post | P(ROPE) | Gate A | Gate B |
|---|---|---|---|---|---|
| Old bespoke bootstrap (N=2000) | +661 | **8,700** | 0.743 | FAIL | FAIL |
| Delta method (N=2000) | **-41** | **1,405** | **1.000** | **PASS** | **PASS** |
| Metrics-layer bootstrap (N=2000) | -80 | 1,719 | 1.000 | PASS | PASS |

The old bootstrap gave sd_post=8700; the delta method gives 1405 — a
**6× tightening**. The metrics-layer Bayesian bootstrap (Dirichlet
weights) also gives 1719, confirming the delta method is in the right
range. The old implementation was inflating variance because CARA CE
with γ=0.001 and wealth in the range 80K-440K maps to utility values
spanning hundreds of orders of magnitude; the standard resample-then-
compute-CE bootstrap is numerically unstable in this regime.

### Recovery gate passes at N=2000 (seeds 2000-3999)

With the delta method: mean=-41, sd=1405, P(ROPE)=1.0.
Both Gate A (P(ROPE)≥0.95) and Gate B (sd_post≤7348) pass.

The corrected kernelized rerun gives the same conclusion:
- `kern_action_pca_r3_compact`: mean `-40`, sd `1406`, `P(ROPE)=1.000`
- `kern_action_pca_r3_rich`: mean `-38`, sd `1409`, `P(ROPE)=1.000`

So the controller is within **40 CE** of BBG — essentially indistinguishable
on the pre-registered split.

### But: expanding the test seed range reveals fragility

| N_test | Seeds | BBG CE | Kern CE | mean gap |
|---|---|---|---|---|
| 2000 | 2000-3999 | 81,666 | 81,625 | -41 |
| 3000 | 2000-4999 | 70,858 | 54,256 | -16,602 |
| 4000 | 2000-5999 | 71,146 | 54,544 | -16,602 |

Seeds 4000+ produce harder episodes where:
- BBG drops by ~10K (harder market conditions)
- The kernelized controller drops by **27K** (catastrophic)

The controller recovers BBG on the "standard" seed range but is not
robust to distributional shift in the episode generator. This is not
a statistical precision issue — it is a **generalization failure**.

### Classification

**Outcome E1 (conditional)**: recovery passes on the pre-registered
test split (seeds 2000-3999) once the inference is corrected. The old
"no gate pass" classification was a **false negative** caused by an
unstable bootstrap implementation.

However, the expanded-seed result is a genuine finding: the kernelized
controller is fragile beyond its training distribution. This limits
the strength of the recovery claim.

### Delta vs bootstrap agreement

Delta-bootstrap relative sd_post difference is 22% at N=2000. The delta
method is tighter, as expected for a smooth functional. Both agree on
the gate outcome (PASS). The MC method (`paired_ce_posterior(...,
method="mc")`) is numerically fragile here because the CARA utility values
are extremely close to zero. That is a property of the CARA scale, not
evidence against the delta-method result.

## What the recovery results collectively say

The recovery line is now clear:

| Experiment | Best CE | mean gap | sd_post | P(ROPE) | Outcome |
|---|---|---|---|---|---|
| Non-kernelized SDRE (r=1-3) | 74,664 | -8,069 | 6,289 | 0.54 | R3 |
| Kernelized SDRE rerun (ActionPCA r3, 2000-3999) | 81,628 | **-38** | **1,409** | **1.00** | **E1** |
| Precision eval, expanded seeds 2000-4999 | 54,256 | -16,602 | — | — | fail |
| Precision eval, expanded seeds 2000-5999 | 54,544 | -16,602 | — | — | fail |

The core facts are:

1. **Formal recovery is real on the pre-registered split.**
   The corrected kernelized `ActionPCA r3` controller matches BBG within the
   ROPE and passes the precision gate comfortably.

2. **The old “no gate pass” was an inference artifact.**
   The bespoke bootstrap was the wrong tool for CARA CE at this scale.

3. **Generalization is not solved.**
   Expanding the seed range reveals a genuine failure mode on harder episodes.

4. **Kernelization, not richer hand-built summaries, is the key gain.**
   `compact` and `rich` are effectively identical; the decisive improvement is
   state-conditioned reduced coordinates on the `ActionPCA r3` basis.

5. **Hyperparameter tuning is not the next bottleneck.**
   The Nyström sweep showed saturation around the same CE. The next work should
   target robustness beyond the standard split.

## Fast prototype loop for the next generalization fixes

The next iteration should optimize for **out-of-split robustness**, not
in-split CE. The cheapest defensible loop is:

1. **Freeze the current in-split winner**
   - `kern_action_pca_r3_compact`
   - use it as the reference controller for every prototype

2. **Use a two-slice dev harness**
   - in-split slice: seeds `2000-2399`
   - hard slice: seeds `4000-4399`
   - report both BBG gap and CE drop on each slice

3. **Prototype on the smallest meaningful run**
   - keep training at 500 episodes
   - evaluate on 400 + 400 episodes first
   - only promote a fix to the full `2000-3999` / `2000-5999` runs if it
     improves the hard slice without breaking the in-split slice

4. **Cache what does not change**
   - BBG demonstrations used to fit the kernelized controller
   - the `ActionPCA r3` basis `U_r`
   - benchmark wealth arrays for the standard evaluation slices

5. **Treat the next fixes as ablations**
   - better regularization / smoother coefficient map
   - more robust state representation
   - local objective refinements
   - each change should be tested against the same two-slice harness before any
     full rerun

This keeps the expensive full evaluation only for candidates that show an
actual robustness gain on the hard slice.

## Uncertainty levels for the controller stack

The uncertainty story should be separated into three levels.

### Level 1 — posterior over held-out performance

This is the uncertainty layer we already have.

Object of inference:

\[
\Delta = CE_{\mathrm{candidate}} - CE_{\mathrm{BBG}}
\]

computed on paired held-out episodes, with posterior summaries from
`paired_ce_posterior(...)`.

What it gives:
- uncertainty about whether the controller matches or beats the benchmark
- recovery gates (ROPE + precision)
- a clean Bayesian/frequentist evaluation layer independent of the policy form

What we have already done:
- corrected the bespoke bootstrap mistake
- switched to the metrics-layer paired CE posterior
- established formal recovery on the pre-registered split
- identified out-of-split fragility on harder seeds

What it does **not** give:
- uncertainty inside the controller itself
- uncertainty about whether the representation is extrapolating

### Level 2 — uncertainty in the policy output, conditional on features

This is the next layer to pursue.

Form:

\[
z = \phi(x), \qquad a(x) \mid z \sim \text{Bayesian head}
\]

where:
- `\phi(x)` is a deterministic learned representation
- the head predicting reduced action coordinates `a(x)` (or local quadratic
  coefficients) is Bayesian

Recommended implementation:
- deterministic encoder for the option book / state
- Bayesian GP/KRR head on top of the learned latent `z`
- keep the reduced action basis fixed (`ActionPCA r3` first)
- keep the final evaluation posterior at Level 1

Why this fits the thesis:
- preserves the reduced-action / SDRE structure
- introduces uncertainty where it is most useful operationally: in the mapping
  from state representation to reduced control coordinates
- avoids full Bayesian deep learning overhead too early

What it can help with:
- detecting coefficient-map uncertainty on harder states
- identifying extrapolation beyond the training support
- regularizing the controller toward smoother out-of-split behavior

### Level 3 — uncertainty in the representation itself

This is the expensive escalation path.

Form:

\[
z \mid x \sim q_\theta(z \mid x), \qquad a(x) \mid z \sim \text{Bayesian head}
\]

Examples:
- deep ensembles over the encoder
- Bayesian neural net encoder
- Laplace / variational approximation on the encoder
- ensemble encoder + GP/KRR head

What it gives:
- uncertainty from ambiguity in the learned representation itself
- not just uncertainty in the head conditional on a fixed latent

What it costs:
- substantially more tuning and engineering
- harder separation between representation uncertainty and head uncertainty
- more difficult debugging when out-of-split performance fails

### Why we should pursue Level 2 next

Level 2 is the best next step because it is the smallest change that can expose
whether the hard-seed failure is really a coefficient-map extrapolation problem.

It preserves:
- the benchmark
- the reduced-action basis
- the SDRE / local-quadratic control structure

and changes only:
- the state representation
- the uncertainty model on the reduced-coordinate map

### Clues that would justify escalating from Level 2 to Level 3

We should move to Level 3 only if one or more of the following remain true
after a serious Level-2 pass:

1. **The Bayesian head is highly uncertain on hard seeds, but that uncertainty
   does not track the failure.**
   - the controller is wrong even when head posterior variance is modest
   - suggests the latent representation itself is collapsing important geometry

2. **Different deterministic encoders give materially different hard-slice
   behavior with similar in-split fit.**
   - suggests representation ambiguity, not just head uncertainty

3. **Posterior variance concentrates on a few states/channels that are clearly
   out-of-distribution in the latent space.**
   - if the issue is latent extrapolation rather than coefficient uncertainty,
     a fixed representation is likely the bottleneck

4. **A richer deterministic encoder plus Bayesian head still fails, while the
   latent diagnostics show poor separation of easy vs hard slices.**
   - again points to representation uncertainty rather than head uncertainty

5. **The main robustness gap is driven by representation shift.**
   - e.g. the hard slice occupies latent regions not covered by the train slice

If those clues appear, then Level 3 becomes justified.

### Practical interpretation

For this project:
- **Level 1** is already in place and is the final evaluation standard
- **Level 2** is the next controller-design step
- **Level 3** is a contingency plan, not the immediate next move

## Where a MEMM / entropy-selected measure would fit

If we decide to bring in the entropy-minimizing martingale measure (MEMM) or a
related measure selector from the parent `koopman-pricing` work, it should be
introduced as a **pricing layer**, not as a replacement for the market-making
objective.

### The correct separation

There are two different objects:

1. **Pricing measure / no-arbitrage layer**
   - defines a dynamically consistent shadow mid surface
   - this is where MEMM or another entropy-selected martingale measure belongs

2. **Market-making control objective**
   - chooses spreads and hedge around that mid surface
   - this remains a utility / certainty-equivalent optimization problem

Formally, we would write:

\[
b_i(t) = m_i^{Q}(t) - \delta_i^{bid}(t), \qquad
a_i(t) = m_i^{Q}(t) + \delta_i^{ask}(t)
\]

where:
- `m_i^Q(t)` is the arbitrage-free shadow mid under a selected measure `Q`
- `\delta^{bid}, \delta^{ask}` are chosen by the controller

The controller still solves a utility-maximizing market-making problem:

\[
\max_{\delta,\,h}\; \mathbb E[U(W_T)]
\]

or the equivalent certainty-equivalent objective. So:
- **MEMM determines the center of the quote surface**
- **utility maximization determines the spreads and hedge**

### Why this is the right architecture

This keeps the two roles clean:
- dynamic no-arbitrage / model consistency is enforced at the pricing layer
- inventory/risk preferences remain in the control layer

It would be incorrect to replace the market-making problem with pure
risk-neutral optimization under MEMM. That would collapse the control problem
into pricing and throw away the inventory-risk tradeoff that defines market
making.

### How this could help the current project

In the current OMM recovery line, a MEMM or entropy-selected shadow measure
would be most useful as:

1. a **shadow mid-surface prior**
2. a **prior mean for the Bayesian head**
3. a **residual target**, where the controller learns deviations around the
   dynamically consistent center

That is the most conservative way to use the pricing-repo ideas:
- preserve the current utility-max / reduced-action architecture
- add a stronger dynamically consistent prior
- improve robustness out of distribution without turning the controller into a
  pure pricing engine

## Files

| File | Role |
|------|------|
| sdre_recovery.py | Controller implementation (BilinearSVD + ActionPCA + 2stage + kernelized + Nyström) |
| heuristic_action_dictionary.py | Post-hoc interpretability directions |
| bbg_sdre_recovery.py | Initial recovery experiment |
| bbg_sdre_rank_sweep.py | Rank sweep (r=1,2,3,5,8) |
| bbg_action_subspace_validation.py | Heuristic alignment diagnostics |
| bbg_recovery_formal.py | Formal evaluation (Outcome C) |
| bbg_rank_interpretability.py | Direction heatmaps and BBG reconstruction plots |
| bbg_recovery_equivalence.py | Paired ROPE recovery gate (Outcome R3) |
| bbg_action_recovery.py | Action-level reconstruction analysis |
| bbg_kernelized_recovery.py | Kernelized state-conditioned recovery |
| bbg_kernel_tuning.py | Nyström tuning sweep |
| bbg_precision_evaluation.py | Precision evaluation + inference switch (E1) |
| bbg_level2_uncertainty.py | Level-2 encoder + Bayesian KRR head experiment |
| state_encoder.py | DeepSets permutation-invariant option-book encoder |

## Level-2 uncertainty experiment (2026-04-13)

**Date**: 2026-04-13
**Script**: `finance/experiments/bbg_level2_uncertainty.py`
**Results**: `finance/experiments/results/bbg_level2_uncertainty_2026-04-13.txt`

### Setup

Three encoder families with Bayesian KRR head on a fixed ActionPCA r3 basis:

| Encoder | Input dim | Description |
|---|---|---|
| compact | 3 | (tau_frac, nu_norm, vpi_norm) — same as kernelized winner |
| rich | 7 | + short/long vega, concentration, constraint slack |
| deepsets | 8 (learned) | Per-option MLP → mean pool → concat global → MLP → z |

Two-slice evaluation:
- In-split: seeds 2000-2399 (400 eps)
- Hard: seeds 4000-4399 (400 eps)

DeepSets encoder: train R²=0.979, final MSE loss=0.000593.

### Critical confound: Basis mismatch

The experiment used `get_action_pca_r3_basis` (random noise exploration via
`collect_exploration_data`) instead of `build_sdre_from_bbg` (BBG-regime
exploration). This produces a **different ActionPCA r3 basis** from the one
used in the successful kernelized recovery experiment.

As a result, all three controllers converge to near-RN behavior (+17K to +45K
above BBG), not recovery. The CE comparison is invalid for gate purposes, but
the diagnostic findings (prediction accuracy, variance, coverage) remain
informative.

**Fix for next round**: use `build_sdre_from_bbg` to get the basis, matching
the kernelized recovery experiment exactly.

### CE results (confounded by basis mismatch)

| Controller | in-split CE | gap vs BBG | hard CE | gap vs BBG |
|---|---|---|---|---|
| BBG | 88,962 | — | 68,844 | — |
| RN | 122,106 | — | 109,639 | — |
| compact | 106,095 | +17,134 | 113,774 | +44,931 |
| rich | 106,479 | +17,517 | 114,612 | +45,768 |
| deepsets | 106,475 | +17,513 | 113,966 | +45,122 |

All controllers behave near-RN, confirming the basis is wrong (not aligned
with BBG action structure).

### Prediction error and posterior variance (valid findings)

| Encoder | Slice | RMSE (mean over coords) | Mean GP var | Error-var corr |
|---|---|---|---|---|
| compact | in-split | 0.040 | 0.019 | 0.30 |
| compact | hard | 0.041 | 0.018 | 0.34 |
| rich | in-split | 0.044 | 0.181 | 0.31 |
| rich | hard | 0.046 | 0.175 | 0.36 |
| deepsets | in-split | 0.006 | 0.001 | 0.28 |
| deepsets | hard | 0.006 | 0.001 | 0.21 |

**Key finding**: DeepSets achieves **~7x lower RMSE** on reduced-coordinate
prediction than compact/rich encoders. Its GP posterior variance is 15-18x
lower (high confidence, low error). But this excellent fit doesn't translate
to BBG recovery because the basis is wrong.

**GP variance does NOT distinguish in-split from hard** — variance is
essentially the same on both slices for all encoders. This means the hard-seed
failure (from the expanded-seed precision evaluation) is not caused by
coefficient-map extrapolation.

**Error-variance correlation is moderate** (~0.3). The GP variance is mildly
informative but far from a reliable failure detector.

### Latent coverage (valid findings)

All feature shifts between train and eval are <0.05σ for all encoders,
on both in-split and hard slices. The train, in-split, and hard episodes
are **indistinguishable** in feature space.

This rules out latent extrapolation as the cause of hard-seed failure.

### Level 3 escalation assessment

| # | Indicator | Present? | Notes |
|---|-----------|----------|-------|
| 1 | Head var modest on failing hard seeds | YES | GP var same across slices |
| 2 | Different encoders → different hard behavior | NO | All three similar CE |
| 3 | Train vs hard latent badly shifted | NO | Shifts <0.05σ |
| 4 | Richer encoder + Bayesian head still fails | YES | DeepSets fails same pattern |
| 5 | Encoder doesn't separate easy vs hard | YES | Distributions identical |

Indicators 1, 4, 5 are present. However, the basis confound means we cannot
draw Level 3 conclusions from this run. **The next step is to re-run with the
correct basis** (from `build_sdre_from_bbg`) before deciding on Level 3.

### What this means for next steps

1. **Fix the basis**: Use `build_sdre_from_bbg` instead of
   `get_action_pca_r3_basis` to produce the same ActionPCA r3 basis as the
   successful kernelized recovery.

2. **Re-run Level-2 with correct basis**: The diagnostic infrastructure
   (Bayesian KRR variance, per-step error tracking, latent coverage) is all
   in place. Only the basis source needs to change.

3. **Confirmed**: DeepSets encoder is a viable learned representation —
   10x better reduced-coordinate prediction, near-zero GP variance, and
   fast training (80s). Worth retaining for the corrected re-run.

4. **Confirmed**: No latent distribution shift between train/in-split/hard.
   The generalization failure is not a feature-space extrapolation problem.
