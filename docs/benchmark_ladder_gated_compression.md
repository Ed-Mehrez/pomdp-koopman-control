# Benchmark Ladder for the Gated-Compression Thesis

A per-benchmark specification for the validation program introduced in `docs/gated_compression_note.md` and `docs/slides/thesis_pitch_gated_compression.md`.

**Design principle (explicit).**
*The benchmark ladder is designed to move from easy cases with known sufficient summaries to hard cases where those summaries become unavailable or inadequate.*

Each row below is a *study*, not a *method*. For each benchmark we specify:

- **Question** being tested.
- **Baseline** (simple / handcrafted summary).
- **Model-free lane** (signature-based, from `src/sskf/multiscale_leadlag_filters.py`).
- **Gating hypothesis** (which way the gate should flip).
- **Primary metric**.
- **What would count as a meaningful flip of the gate** — the statistical bar, set in advance.

Rows 1–3 are now completed. Rows 4–6 remain the forward program. The whole point of the ladder is to let the gate tell us, one benchmark at a time, where the simple summary stops being enough.

---

## Metric language (used consistently below)

- **Filter correlation / RMSE against latent target.** Post-update V̂ (or the analogous latent) vs the true latent, pooled across seeds, with a fixed warm-up drop. This is the instrument used in Row 1 today.
- **Control-relevant held-out metric.** Paired CRRA-style score `mean(Δ logW) − 0.5·(γ−1)·Var(Δ logW)` on held-out seeds, matched against a reference policy (myopic Merton at u=0). Reports point estimate + bootstrap 5/95% interval.
- **Calibration / posterior diagnostics.** Coverage of posterior credible intervals, standardized innovation z-score statistics, observation-noise EWMA stability. Applicable where the filter exposes a posterior, not just a point estimate.

Where a benchmark is a *filter* task, filter metric is primary. Where it is a *control* task, control metric is primary and filter metric is supporting.

---

## Row 1 — Daily Heston at daily bars

**Status.** completed.

- **Question.** At a well-specified daily-bar Heston, does a scalar realized-variance smoother (EWMA) already act as an empirically sufficient summary, making signature-based path compression redundant?
- **Baseline.** `EWMAVEstimator` at halflife 21d on `dr_S² / dt`. Single exponential memory, one hyperparameter.
- **Model-free lane.** `CumulativeStrideLeadLagBLRKFilter` (strides 1/5/20d) and companion `MultiScaleLeadLagBLRKFilter` variants. See `src/sskf/multiscale_leadlag_filters.py`.
- **Gating hypothesis.** Gate should choose EWMA; signature lane should not improve filter correlation materially.
- **Primary metric.** Pooled `corr(V̂_post, V_true_pre)` across 80 seeds × 200 steps, warm-up 20. Also reported: corr against one-step and forward-averaged latent V, and RMSE.
- **Flip criterion.** A signature lane would have *flipped* the gate if it had beaten EWMA by ≥ +0.03 corr on spot V with a reproducible margin across at least two independent seed banks.
- **Outcome.** EWMA: +0.8229. Best signature lane (ms_cum_stride): +0.8149. Gate held. The architectural gradient *within* signatures is nevertheless strong and is reported separately.

---

## Row 2 — Heston at 5-minute bars

**Status.** completed.

- **Question.** As observation frequency increases, do multiresolution signatures meaningfully beat fixed-halflife EWMA on filter quality, control-relevant score, or both?
- **Baseline.** Realized-variance EWMA computed on 5-minute dr_S² / dt aggregated into an exponentially weighted variance series; alternately, a bias-corrected RV with microstructure noise filtering.
- **Model-free lane.** `ms_cum_stride` (calendar strides chosen to straddle the intraday seasonality window); optionally `ms_forget_spectral` with a pilot τ̂ from intraday ACF. Cross-scale precision fusion as in the daily study.
- **Gating hypothesis.** Gate flips toward signatures. Intraday patterns and the stronger separation between micro and macro volatility timescales should make a single EWMA halflife visibly suboptimal.
- **Primary metric.** Filter corr and RMSE vs spot V (when the simulator is ours), plus posterior calibration. For a realistic variant where V is latent, use the pricing / variance-swap metric from the sibling scripts below.
- **Flip criterion.** Signature lane beats the EWMA baseline on filter corr by ≥ +0.04 with a 90% bootstrap interval that excludes zero, **and** posterior diagnostics remain acceptable.
- **Source references (for setup, not for results).** `../koopman-pricing/experiments/realistic_dt_emkkf_test.py`, `../koopman-pricing/experiments/hf_panel_transfer_benchmark.py`.
- **Outcome.** The best signature lane (`ms_cum_stride`) reached `corr(spot V) = +0.9705` versus the best scalar EWMA at `+0.9668`; the gap (`+0.0037`) stayed well below the pre-registered flip bar. Warm-start ablation showed both lanes improved in parallel, so the residual gap is intrinsic at this config rather than a cold-start artifact. Interpretation: the intraday benchmark makes the gate nontrivial, but it still does not flip.

---

## Row 3 — Bates / jump-volatility benchmark

**Status.** completed.

- **Question.** Under Bates dynamics (Heston + compound-Poisson jumps in log-price and/or in variance), does jump contamination break the one-dimensional r²/dt variance proxy enough for signatures to gain over EWMA?
- **Baseline.** EWMA on r²/dt, plus robust scalar variants (BV-style EWMA, winsorized EWMA).
- **Model-free lane.** `ms_cum_stride`. The Chen-level-2 window recovery exposes antisymmetric level-2 pairs beyond the Lévy-area/QV term; jumps perturb those pairs in a structured way that a scalar summary cannot see.
- **Gating hypothesis.** Signature lane should gain because path geometry matters more: at and around jump events, the scalar proxy is dominated by the jump atom and gives a misleading V estimate.
- **Primary metric.** Filter corr and RMSE vs latent V; additionally, a jump-adjacent window metric (corr / RMSE restricted to windows within ±k steps of a jump event) to isolate the regime where the gate should flip.
- **Flip criterion.** On the jump-adjacent subset, signature lane beats the baseline by ≥ +0.10 corr; on the full sample, ≥ +0.04 corr. Both with bootstrap intervals excluding zero.
- **Source reference.** Starting point: `archive/05_bates_vol/` (empty placeholder; Bates simulator to be rebuilt). The existing `HestonMertonEnv` in `finance/experiments/merton_kronic_bilinear.py` can be extended with compound-Poisson jump terms without changing the filter API.
- **Outcome.** The initial raw-target study did **not** flip the gate toward signatures: `winsor_ewma` won decisively, and `ms_cum_stride_raw` underperformed near jumps. A follow-up using the **same cumulative-stride signature state** but a **BV-style target** lifted corr from `+0.5226` to `+0.6997`, improving jump-adjacent corr by `+0.2030` and closing about **58%** of the raw→winsor gap. This clears the pre-registered recovery bar and shows that the Bates failure was primarily a **target problem**. The residual gap to `winsor_ewma` is consistent with a remaining representation issue in jump regimes.

---

## Row 4 — Nonlinear-observation latent-state benchmark

**Status.** next benchmark.

- **Question.** When the observation model is a nonlinear functional of the latent state (e.g. option-implied variance, a square-root or log observation of V, or an order-book-derived V proxy), does a linear/Gaussian filter baseline become structurally insufficient?
- **Baseline.** Kalman / heteroskedastic Kalman on the closest linear approximation of the observation map; or handcrafted filter built around a specific assumed observation law.
- **Model-free lane.** `ms_cum_stride` with an observation channel appropriate to the benchmark (possibly augmented to accept two feature streams, price and the nonlinear observation).
- **Gating hypothesis.** Gate should prefer signatures. The signature lane is indifferent to the observation law by construction; the Kalman baseline encodes a specific map that is wrong by design.
- **Primary metric.** Filter corr and RMSE vs latent V; additionally, filter calibration (posterior interval coverage) — under misspecification the Kalman baseline typically fails coverage first.
- **Flip criterion.** Signature lane matches or beats baseline on corr (≥ 0) **and** has materially better calibration (coverage gap < 5% of nominal vs baseline > 20%). Meaningful flip = both legs pass.

---

## Row 5 — Irregular / missing-sample benchmark

**Status.** planned ablation.

- **Question.** When observations arrive at irregular times or with random missingness, do event-time summaries (signature on the irregular stream) dominate fixed-clock smoothers (EWMA with imputed or interpolated samples)?
- **Baseline.** Two variants: (a) interpolate to regular grid and apply EWMA; (b) event-time EWMA with per-arrival decay.
- **Model-free lane.** `ms_cum_stride` driven on `(dt_k, r_k)` increments directly; no interpolation needed — signatures are well-defined on irregular paths.
- **Gating hypothesis.** Signatures more attractive. The harder the irregularity (shorter gaps between samples vs longer ones, burstiness), the larger the expected margin.
- **Primary metric.** Filter corr and RMSE vs latent V under a family of irregularity regimes parameterized by the coefficient of variation of inter-arrival times.
- **Flip criterion.** At moderate irregularity (CV ≥ 0.5), signature lane beats the best-of-two EWMA baselines by ≥ +0.03 corr; at high irregularity (CV ≥ 1.0), ≥ +0.08 corr.

---

## Row 6 — Multifactor / two-timescale latent state

**Status.** later-stage benchmark.

- **Question.** Under a multifactor stochastic-volatility model (e.g. two-factor Heston with a fast and a slow factor), does a single scalar smoother fail because it cannot summarize two latent factors simultaneously?
- **Baseline.** Scalar EWMA. Secondary baseline: a hand-designed two-EWMA pair with factor-specific halflives.
- **Model-free lane.** `MultiScaleLeadLagBLRKFilter` with spectral ladder over the joint ACF, and `ms_cum_stride` with strides straddling the two timescales.
- **Gating hypothesis.** Signature lane more likely to win, especially on the continuation-value-relevant metric: the multifactor structure propagates into the pricing kernel and the controller's Hamiltonian in ways a single scalar cannot track.
- **Primary metric.** Control-relevant CRRA score under a multifactor-Heston Merton problem, paired against a reference controller; filter corr as a supporting metric (vs each factor individually and vs the total V).
- **Flip criterion.** Signature lane's CRRA point estimate exceeds baseline's by a margin whose 90% bootstrap interval excludes zero; *and* signature lane tracks at least one of the two factors with corr ≥ 0.5 while the scalar EWMA cannot.

---

## How a benchmark gets promoted

A row moves from `next` to `completed` only when:
1. A held-out evaluation with at least two independent seed banks is run.
2. The primary metric is reported with a bootstrap interval or a power-adjusted test.
3. Posterior diagnostics are reported where the lane exposes them.
4. The flip criterion, as stated, is either met or is not — the outcome is written into `docs/gated_compression_note.md` §5 with a one-line hook in the deck's Slide 4 status column.

No retroactive flip-criterion adjustments. A failed flip is data; it does not invalidate the thesis and is not to be silently rerun at higher N. A target-swap follow-up is allowed only when the original row explicitly leaves open the possibility that the summary was supervised against the wrong latent characteristic, as in the completed Bates row.

---

## Not in scope of this ladder

- EM-KKF eigenfunction pricing (a separate research line; referenced for 5-min setup only).
- Full rollout of multi-asset transfer benchmarks (that is the pricing-repo program).
- New signature architectures. The ladder uses the existing `src/sskf/multiscale_leadlag_filters.py` lanes as the model-free fallback. If a benchmark makes the gate flip but the existing signature lane still loses, that is a separate research signal to revisit architecture, not an excuse to retune this ladder.
