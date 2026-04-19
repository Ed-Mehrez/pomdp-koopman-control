# Gated State Compression for Partially Observed Control

**A research note, not a theorem paper.**
Guiding thesis principles are labeled as propositions; proofs are sketches only. The propositions codify a design discipline, not a universal claim. Companion documents: `docs/slides/thesis_pitch_gated_compression.md` (pitch), `docs/benchmark_ladder_gated_compression.md` (validation program).

---

## 1. Thesis

Partially observed stochastic control forces a representation decision: *what state summary does the policy condition on?* The temptation is to default to a rich, universal feature map (signatures, deep representations, full particle beliefs). The claim of this note is that the representation decision should be **gated**:

> Default to the simplest summary that passes calibration and closes most of the control-relevant gap.
> Escalate to signature-based path compression only when the simple summary is unavailable, misspecified, or leaves path-dependent residual structure.

This is a design rule, not a theorem. Its force comes from two observations:

1. Richer representations pay a statistical price (estimation noise) and a computational price (memory, compute) even when they are *asymptotically* no worse than simpler ones.
2. In structured models like Heston, a low-dimensional sufficient statistic exists *in principle*; a scalar smoother (EWMA on realized variance) is close to that statistic *in practice* at daily frequency.
3. In jump regimes like Bates, the relevant design object is the **summary-target pair**: the same signature state can recover much of its performance once the supervised target is switched from raw `r^2/dt` to a jump-robust local characteristic.

When both hold, richer path features cannot improve a Bayes-optimal controller's achievable value; they can only *hurt* via finite-sample variance. The signature fallback earns its keep when either fails.

---

## 2. Completed evidence: daily Heston is an easy-case gate

**Setup.** ρ=−0.7, γ=3, dt=1/252, CIR (κ, θ, ξ) = (2, 0.04, 0.3). 80 seeds × 200 steps, filter-only (u=0), warm-up 20 steps, post-update V̂ vs spot V_t. Filters consume dr_S. Corr pooled across seeds. Source: `finance/experiments/study_heston_multiscale_signature_filters.py`.

| Lane | Class | corr(spot V) |
|---|---|---|
| ewma (halflife 21d) | scalar smoother | **+0.8229** |
| ms_cum_stride (1/5/20d) | multires signature | +0.8149 |
| bayesian_sig | signature BLF | +0.7710 |
| ms_forget_spectral (13/53/210d) | multires signature | +0.7667 |
| blr_kf_leadlag (γ=0.99) | single-scale signature | +0.7571 |
| ms_forget_fixed (1/5/20d) | multires signature | +0.6829 |

Two orthogonal readings:

- **Between classes.** The scalar smoother is the top performer. Signatures do not beat EWMA on this benchmark.
- **Within signatures.** The architectural gradient is sharp and reproducible:
  - cumulative-stride + Chen-level-2 window recovery > naive forgetting-factor fusion: **+0.132 corr** (ms_cum_stride vs ms_forget_fixed).
  - spectral ladder (τ̂ ≈ 53d from EWMA(r²/dt) ACF) > fixed 1/5/20d: **+0.084 corr**.
  - ms_cum_stride > blr_kf_leadlag: **+0.058 corr**.

The second reading is what the thesis direction *keeps*: a principled, model-free multiresolution construction. The first reading is what it *honestly concedes*: at a well-specified low-dimensional benchmark, a handcrafted scalar summary is hard to beat.

### 2.1 5-minute Heston: the gate becomes nontrivial, but does not flip

**Setup.** 8 seeds × 60 trading days × 78 bars/day. Primary targets: spot latent `V_t` and one-day forward-averaged latent variance. Warm-start ablation over within-path windows. Source: `finance/experiments/study_heston_5min_signature_filters.py`.

| Lane | corr(spot V) | RMSE | corr(fwd V, h=1d) |
|---|---|---:|---:|
| ms_cum_stride | **+0.9705** | **0.0050** | **+0.9671** |
| ewma_5min_1d | +0.9668 | 0.0053 | +0.9631 |
| ewma_5min_5d | +0.9597 | 0.0058 | +0.9575 |
| ms_forget_spec | +0.9252 | 0.0080 | +0.9257 |
| blr_kf_leadlag | +0.9088 | 0.0088 | +0.9098 |

The pre-registered flip bar was `+0.03 corr`; the observed gap is `+0.0037`. So the signature lane becomes **competitive**, but the gate does **not** flip. The warm-start ablation shows the residual gap is essentially flat across windows, so the non-flip is architectural at this config rather than a cold-start artifact.

### 2.2 Bates: most of the failure was target design

**Setup.** 40 seeds × 400 daily steps, compound-Poisson price jumps (`lambda_j = 30/yr`). Initial study: raw `r_t^2/dt` target. Follow-up: same cumulative-stride signature state, but with a BV-style target and a two-channel diagnostic head. Sources: `finance/experiments/study_bates_signature_filters.py`, `finance/experiments/study_bates_signature_proxy_channels.py`.

| Lane | corr(all) | corr(jump) | corr(calm) | RMSE(all) |
|---|---|---:|---:|---:|
| winsor_ewma | **+0.8260** | **+0.8366** | +0.8098 | **0.0216** |
| bv_ewma | +0.7838 | +0.7962 | +0.7687 | 0.0362 |
| ms_cum_stride_bv_target | +0.6997 | +0.7181 | +0.6669 | 0.0373 |
| rv_ewma | +0.6534 | +0.6630 | +0.6651 | 0.0617 |
| ms_cum_stride_raw | +0.5226 | +0.5151 | +0.6996 | 0.0589 |

The dominant finding is **target recovery**:

- Jump-adjacent gain of `ms_cum_stride_bv_target` over `ms_cum_stride_raw`: `+0.2030`.
- Overall raw→winsor gap: `+0.3034`; BV-target signature lane closes `+0.1771`, or about **58%** of that gap.
- The continuous-channel diagnostic from the two-channel lane tracks smoothed BV with corr `+0.819`, showing that the signature state already contains the robust continuous-variation information; the original raw `r^2/dt` supervision was the main bottleneck.

The residual gap to `winsor_ewma` is real. The jump channel remains weak, which is consistent with the jump-aware theory notes: level-2 lead-lag geometry captures continuous variation, while jump-sensitive information naturally lives in higher-order / Marcus-style structure. So Bates is a **positive result for the gated-compression thesis**, but not yet a win for the current level-2 signature lane as a standalone champion.

---

## 3. The gate, precisely

**Structural admissibility.** A summary `s: path → ℝᵈ` is *structurally admissible* for a control problem if `s_t` can be interpreted as an approximate Bayesian filter state, or a sufficient statistic for the continuation value `V(t, history) ≈ Ṽ(t, s_t)` under the chosen objective.

**Empirical sufficiency (control-relevant gap).** Let `Ṽ_s` be the best continuation value achievable with conditioning on `s` and `Ṽ_full` the best achievable with full history. The *control-relevant gap* is `Ṽ_full − Ṽ_s`. The summary is *empirically sufficient* if this gap is small relative to the noise floor on the evaluation metric.

**Gate statement.** Prefer the simplest **summary-target pair** `(s, y)` that is
1. structurally admissible,
2. empirically sufficient on a held-out evaluation,
3. posterior-calibrated (filter diagnostics, coverage, z-scores within acceptable bands).

Escalate to signature-based compression when any of (1)–(3) fails, and re-check whether the target `y` itself is matched to the latent characteristic of interest. Bates shows that a rich representation can still fail under the wrong target.

**Decision rule (one line).**
*Default to the simplest summary that passes calibration and closes most of the control-relevant gap.*

---

## 4. Guiding propositions

**Proposition 1 (Sufficiency dominates richness).**
*If a summary `s_t` is a sufficient statistic for the continuation value under the chosen objective, then enlarging the conditioning set to `(s_t, φ_t)` for any additional path functional `φ_t` cannot improve the Bayes-optimal controller's value.*

**Sketch.** Standard dynamic programming. If `s_t` is sufficient, the Bellman operator factors through it: `V_t(history) = Ṽ_t(s_t)`. The optimal policy `π*_t(history) = π̃*_t(s_t)` is measurable with respect to `s_t`. Conditioning on `(s_t, φ_t)` gives a strictly larger σ-algebra, but `E[V_{t+1} | s_t, φ_t] = E[V_{t+1} | s_t]` almost surely by sufficiency, so the Bellman update is unchanged. Finite-sample policies using `φ_t` inherit the variance of its estimation without an asymptotic value improvement.

**Consequence.** Under structural admissibility *and* empirical sufficiency, signatures (or any richer path functional) add only estimation noise. The daily-Heston benchmark is consistent with this: the structural sufficiency of a smoothed realized-variance summary is close enough that EWMA ≥ signatures in finite samples.

---

**Proposition 2 (Multiresolution signatures as a universal fallback).**
*Truncated multiresolution lead-lag log-signatures form a model-free approximation class for continuous path functionals: under mild regularity, any bounded continuous path functional can be approximated uniformly on compact sets of bounded-variation-plus-Brownian paths by a linear functional of a sufficiently deep signature feature map.*

**Sketch.** Restatement of the universal approximation theorem for signatures (Lyons–Levin–Hambly; Chevyrev–Kormilitzin; Fermanian). The multiresolution part is a stability / scale-selection refinement: by constructing features at calendar strides `m_k` and performing Chen-level-2 window recovery from one cumulative log-signature, one obtains a library of scale-indexed features whose linear span covers the approximating family without re-running the signature at every scale. No structural assumption on the drift, diffusion, or observation law is required.

**Consequence.** Whenever a handcrafted sufficient statistic is absent, misspecified, or unreliable, signatures provide a principled fallback with strong approximation guarantees. The architectural finding — cumulative + Chen-level-2 + spectral ladder — gives a ready-made implementation of that fallback.

---

## 5. Validation ladder (summary)

Full spec lives in `docs/benchmark_ladder_gated_compression.md`. The ladder is designed to move from easy cases with known sufficient summaries to hard cases where those summaries become unavailable or inadequate.

| # | Benchmark | Status | Gating hypothesis |
|---|---|---|---|
| 1 | Daily Heston, daily bars | completed | Gate chooses EWMA. **Confirmed.** |
| 2 | Heston, 5-min bars | completed | Signature lane becomes competitive but does not flip the gate. Warm-start says gap is intrinsic at this config. |
| 3 | Bates / jump-vol | completed | Raw target flips gate to a robust scalar. BV-style target recovers most of the signature loss; residual gap points to jump-aware representation. |
| 4 | Nonlinear-observation latent | next | No linear/Gaussian sufficient statistic; signatures prefer. |
| 5 | Irregular / missing samples | planned | Fixed-clock smoother awkward; signatures prefer. |
| 6 | Multifactor / two-timescale | later | One scalar cannot summarize multiple factors. |

**Honest framing.** The thesis is not that signatures dominate on every benchmark; it is that the gate identifies *when* simple summaries are enough and *when* richer model-free compression is warranted. A benchmark where signatures win is as informative as one where they lose: both calibrate the gate.

---

## 6. What this note deliberately does NOT claim

- It does **not** claim signatures beat EWMA on this benchmark. They do not.
- It does **not** claim hetero_kalman is the default winner; the filter-target study flagged it as an observation-discounting baseline needing audit.
- It does **not** claim the multiresolution work is a new theoretical contribution. The architectural finding (cumulative-stride + Chen-level-2 + spectral ladder) is an *engineering* contribution within a well-known approximation class.
- It does **not** pivot to EM-KKF, full Bayesian generative modeling, or any new architecture. The thesis is about *when* to escalate, and the fallback class we already have.
- It does **not** claim the current signature lane is already the best jump-regime filter. Bates improved sharply under the right target, but `winsor_ewma` still wins at the current jump intensity.
- It does **not** present Rows 4–6 of the validation ladder as already-run. Rows 1–3 are completed; the rest are the remaining program.

The deck accompanying this note (`docs/slides/thesis_pitch_gated_compression.md`) is the 4-slide public version of these claims. The benchmark spec (`docs/benchmark_ladder_gated_compression.md`) is the program that tests them.
