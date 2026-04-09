# fSDE Sibling Repo Audit Task — Scoping & Intended Usage

**Status**: Task spec for codex (or for Ed). Created 2026-04-08.
**Target repo**: `/home/ed/SynologyDrive/Documents/Research/PE_Research/fsde-identifiability/`
**Scope**: This is a **scouting / audit task**, not an integration commitment. The output is an audit report that informs Stage 5+ planning. **Do NOT import any code from `fsde-identifiability` into `pomdp-koopman-control` as a result of this audit** without an explicit re-spec from Ed.
**Time budget**: ~1 day of careful reading and test execution.
**Output location**: `docs/refs/fsde_identifiability_audit_report.md` (to be created by codex when the audit runs).
**Companion docs**:
- `docs/derivation_omm_sdre_v2.md` — the Stage 4 v2 derivation note (Markov-only scope)
- `docs/refs/fractional_methods_references.md` — the forward-looking fractional methods reference report
- `docs/plan_omm_research.md` — the OMM research plan

---

## 0. Why this audit exists

The `fsde-identifiability` sibling repo contains substantial machinery (Koopman generator estimation, signature features, Hurst estimation, fGN whitening) that could plausibly be imported into `pomdp-koopman-control` for Stage 5+ work on non-Markov / rough volatility dynamics.

**However, the sibling repo is WIP** — it's a JMLR-target paper-in-progress, and its API, tests, and methodological choices are not yet stable. Importing WIP infrastructure means inheriting whatever bugs / unproven assumptions / API churn the sibling has, which is exactly the kind of dependency risk we work hard to avoid in `pomdp-koopman-control` (see `feedback_no_framework_up_front.md`, `feedback_no_silent_n_changes.md`).

The discipline: **scout before committing, audit before importing, document before deciding.** This task does the scouting. The actual import decision (and any code changes) waits until Ed has read the audit report and explicitly authorized them.

This is the same audit pattern codex used for the Stage 4 v1 SDRE controller (the Q1+Q2 audit that revealed the v1 implementation was missing spread-capture revenue). The lesson there was: **read the math before translating to code, and report concerns rather than papering over them.** Same lesson applies here, just to someone else's code (Ed's other repo) rather than codex's own.

---

## 1. Background: what the `fsde-identifiability` repo does

(Summarized from `fsde-identifiability/README.md` and `fsde-identifiability/papers/jmlr_fsde_identifiability_outline.md`. Read those files first for the full picture.)

The repo provides nonparametric estimation of fractional stochastic differential equation parameters from trajectory data:
```
dX_t = μ(X_t) dt + σ(X_t) dB^H_t
```
where the goal is to estimate the *functions* `μ(x)`, `σ(x)` and the *scalar* `H` (Hurst exponent) without parametric assumptions.

**Reported error rates** (from the README, as of the Feb 2026 paper outline):
- `H` (Hurst exponent) via aggregated variance: **2-3% error**
- `σ(x)` (diffusion) via local quadratic variation: **<3% error**
- `μ(x)` (drift) via fGN-whitened regression with eigenvalue constraint: **25-40% MRE** (much harder due to low SNR)

**Key methodological insight (verbatim from the README)**: At typical `dt` and `H < 0.5`, **noise dominates signal** — drift contribution is `O(dt)` while noise contribution is `O(dt^H) >> O(dt)`. **Without fGN whitening, regression learns the noise, not the drift.** The whitening procedure:
1. Estimate `H` from the trajectory
2. Build the fGN correlation matrix
3. Cholesky factorize: `Σ = L L^T`
4. Whiten: `dx_w = L^{-1} dx`
5. Regress whitened features on whitened `dx`

**Identifiability theorem** (from the JMLR paper outline): For an fSDE `dX = μ(X) dt + σ(X) dB^H` with `H ∈ (0, 1)`, `μ` continuous, `σ > 0` continuous, the triple `(μ, σ, H)` is uniquely determined by the law of the path. This extends Stroock-Varadhan to the non-Markov setting.

**Project structure** (relevant subset):
```
fsde-identifiability/
├── src/
│   ├── generator_estimator.py    # Core estimator (Profile REML) — likely WIP, drift estimator has 25-40% error
│   ├── rough_paths_generator.py  # fBM simulation — should be stable (it's just simulation)
│   ├── hurst_estimators.py       # H estimation methods — multiple methods, varying maturity
│   ├── spectral_hurst.py         # Spectral H estimation
│   └── sskf/
│       ├── nystrom_koopman.py    # Nyström eigenfunction extraction
│       └── tensor_features.py    # Signature features
├── examples/
│   ├── test_generator_estimator.py
│   ├── test_nystrom_koopman_signatures.py
│   └── test_signature_consistency_training.py
├── papers/
│   └── jmlr_fsde_identifiability_outline.md
└── docs/
    ├── generator_estimator_walkthrough.md
    └── spectral_roughness_theory.md   # Paper 6a — Koopman spectral roughness theory
```

---

## 2. Intended usage: what we'd want to use this for in `pomdp-koopman-control`

This section captures the "how we plan to use the fSDE infrastructure" context that was originally in the v2 derivation note. It is preserved here so codex understands what to *look for* during the audit, not just whether the code runs.

### 2.1 Use case #1: Koopman Carré-du-Champ inventory variance estimator on signature features

**Where this would slot into the OMM SDRE controller**: as a `inventory_variance_estimator` callable (see `docs/derivation_omm_sdre_v2.md` Section 7.5 for the interface).

**The mathematical idea**: For a Koopman generator `L̂` learned from observed (state, action, transition) data via KGEDMD on signature features, the local variance of any observable `O(s_t)` is given by the Carré-du-Champ identity:

$$
\sigma^2(s_t) = (\hat{L} O^2)(s_t) - 2 O(s_t) \cdot (\hat{L} O)(s_t)
$$

For OMM, `O = C_Q` (the option mid), so:

$$
\sigma^2_{\text{inv,Koopman}}(s_t) = (\hat{L} C_Q^2)(s_t) - 2 C_Q(s_t) \cdot (\hat{L} C_Q)(s_t)
$$

**Why this matters**: this estimator is:
- **Model-free**: doesn't require knowing the underlying SDE (Heston vs rough Heston vs anything else)
- **Path-dependent compatible**: works on signature features that lift non-Markov dynamics to a Markov representation
- **Principled**: follows from the Carré-du-Champ identity, which is a standard tool in the existing repo (see `docs/gedmd_ito_correction.md` and `docs/theory_signature_stationarity_transforms.md`)
- **Connects to Paper 1**: ties the OMM application directly to the methodology paper, since both use the same Koopman/CdC machinery

**Stage where this would be used**: Stage 5+ at the earliest, after we've validated the simpler estimators (Heston-specific, empirical sliding-window) on Stage 4 v2.

**Audit relevance**: codex should specifically check whether `fsde-identifiability/src/sskf/nystrom_koopman.py` and `src/sskf/tensor_features.py` are stable enough that we could call them from the OMM controller. Specifically:
- Is the API stable enough to depend on?
- Do the existing tests cover the relevant functionality (Koopman generator on signature features)?
- Is there a clean way to compute `(L̂ O)(s)` and `(L̂ O²)(s)` for an arbitrary observable `O` given a learned generator?
- Are the signature features computed in a way compatible with our env's price observations?

### 2.2 Use case #2: Dynamics regime detection via spectral roughness

**Where this would slot in**: as a *regime detector* that runs before the SDRE controller is configured. For real Alpaca data (Stage 7+), the agent needs to know whether the underlying is closer to classical Markov (Heston-like) or rough (rBergomi-like) so it can pick the right `inventory_variance_estimator`.

**The mathematical idea** (from `fsde-identifiability/docs/spectral_roughness_theory.md`): For an fBm with Hurst parameter `H ∈ (0, 1)`, the eigenvalues `{λ_k}` of the Koopman generator satisfy
```
|λ_k| ~ k^{-(2H+1)}    as k → ∞
```
This enables direct extraction of `H` from the observed eigenvalue decay, **independent of the Cont-Das (2022) microstructure noise contamination**, because microstructure noise affects only the high-frequency tail of the spectrum (high `k`) while `H` is determined by the asymptotic decay rate at the low/middle eigenvalues.

**Why this matters**: gives us a **principled way to estimate the dynamics class** of any observed price series before deploying the OMM controller. The procedure:
1. Observe a price path
2. Compute the Koopman generator via KGEDMD on signature features
3. Extract the eigenvalue decay rate
4. Estimate `H` from `|λ_k| ~ k^{-(2H+1)}`
5. Pick the estimator:
   - `H ≈ 0.5`: classical Markov dynamics → use Heston-specific or empirical estimator
   - `H < 0.5`: rough volatility → use Koopman-CdC on signatures (model-free, robust to long memory)
   - `H > 0.5`: persistent / smoothed → use a smoothed estimator with explicit memory handling

**Stage where this would be used**: Stage 7+ at the earliest, when we deploy on real Alpaca data and need to characterize the empirical dynamics class.

**Audit relevance**: codex should check whether `spectral_hurst.py` and `hurst_estimators.py` implement the spectral approach described in the Paper 6a doc (`docs/spectral_roughness_theory.md`), and whether the implementation matches the paper's claims (eigenvalue decay rate `k^{-(2H+1)}`). Note that Paper 6a is itself in-progress, so the implementation may not yet reflect the final version of the theory.

### 2.3 Use case #3: Signature features as a lifted Markov state representation

**Where this would slot in**: as a replacement for EWMA-style filtering when the underlying is non-Markov. Instead of filtering a hidden variance via Kalman or BPF, the controller computes signature features of the observed price path and uses them as the "lifted state" on which the Koopman generator and SDRE solver operate.

**The mathematical idea**: For continuous paths, the signature `Sig(path)_t = (1, ∫dX, ∫∫dX⊗dX, ...)` is a universal nonlinear feature representation. Truncated signatures provide finite-dimensional summaries that capture path-dependent information. For non-Markov dynamics (rough vol), no finite-dimensional Markov state suffices in the raw `(S, V)` space, but signature features lift the dynamics to a Markov representation in signature space (Boedihardjo-Geng-Lyons 2016; Hambly-Lyons uniqueness theorem).

**Why this matters for OMM**: the agent's state representation becomes
```
s_t = (q_t, h_t, Sig(price path)_t, τ_t, W_t)
```
instead of the v2 Heston-specific
```
s_t = (q_t, h_t, V̂_t, τ_t, W_t)
```
where `V̂_t` is computed by an EWMA filter on log-returns of the spot.

**The same SDRE controller code** consumes either state representation. Only the inventory variance estimator changes.

**Stage where this would be used**: Stage 5+ at the earliest, when we want to deploy on rough vol envs (rBergomi, rough Heston, fBm log-spot).

**Audit relevance**: codex should check whether `src/sskf/tensor_features.py` provides signature features in a form that's compatible with our env. Specifically:
- What signature library does it use? (`iisignature`? `signatory`? Custom implementation?)
- Are the signature features computed efficiently enough for online use during a paired-bootstrap evaluation?
- Are there pre-computed signature levels (truncation orders) that we can configure?
- Does the implementation handle the lead-lag transformation needed to capture quadratic variation? (Per Pinned Gotcha #10 in `MEMORY.md`: "Lévy area ≠ QV for (time, price) path: Must use LEAD-LAG transform.")

### 2.4 Use case #4: Hurst estimation for empirical dynamics characterization

**Where this would slot in**: as a diagnostic for understanding whether real data (or our rough vol simulation envs) actually exhibit `H < 0.5`. This is relevant for the Cont-Das vs Gatheral-Jaisson-Rosenbaum debate about whether volatility is "really" rough.

**Why this matters**: if we deploy our methodology on a regime where `H ≈ 0.5` empirically, the rough vol machinery is overkill — classical Markov filtering is sufficient. If `H < 0.5` empirically, the rough vol machinery is justified. **The Hurst estimator is the regime detector** that tells us which path we're on.

**Stage where this would be used**: Stage 5+ for testing on simulation envs with known `H`; Stage 7+ for real data.

**Audit relevance**: codex should check whether `hurst_estimators.py` and `spectral_hurst.py` are stable enough to use as black-box diagnostics. The README claims 2-3% error on `H`, which is excellent, but we should verify the test coverage and the methodology.

---

## 3. The audit checklist

Codex should work through this checklist systematically. For each item, the output is a brief written assessment (1-3 sentences) that goes into the final audit report.

### 3.1 Repo-level audit

- [ ] **README.md**: Read the full README. Identify claimed error rates, identifiability claims, and any disclaimers. Note the date of the most recent update.
- [ ] **AGENTS.md**: Read this file. Identify any conventions, gotchas, or "do not do" rules from the user's previous work in this repo.
- [ ] **`papers/jmlr_fsde_identifiability_outline.md`**: Read the full paper outline. This is the math spec the implementation should match. Identify the main theorems, the key estimation procedures, and any open issues / TODOs the paper outline flags.
- [ ] **`docs/spectral_roughness_theory.md`** (Paper 6a): Read this. This is the theoretical basis for the Koopman spectral roughness estimator that we'd want to use for regime detection. Note any unverified claims or open questions.
- [ ] **`docs/generator_estimator_walkthrough.md`**: Read this for implementation details on the generator estimator.
- [ ] **`environment.yml`**: Check the dependencies. Are they compatible with `pomdp-koopman-control`'s `rkhs-kronic-gpu` environment, or would we need a separate env?
- [ ] **Recent git log**: `cd fsde-identifiability && git log --oneline -20`. Identify how active development is, what's been changing recently, and whether the API appears stable or churning.

### 3.2 Module-level audit

For each of the following modules, codex should: (a) read the source code end-to-end, (b) cross-check the implementation against the math in the paper outline, (c) run the existing tests, (d) write a brief assessment.

- [ ] **`src/sskf/nystrom_koopman.py`** — Nyström eigenfunction extraction for Koopman generator estimation. **Critical for use cases 1 and 2.** Assess: API stability, test coverage, whether the API supports computing `(L̂ f)(s)` for arbitrary `f`, and whether the eigenvalue spectrum is exposed for the regime detection use case.

- [ ] **`src/sskf/tensor_features.py`** — Signature features. **Critical for use cases 1, 2, and 3.** Assess: which signature library it uses, whether lead-lag transformations are supported (per `MEMORY.md` gotcha #10), efficiency for online use, and whether the truncation order is configurable.

- [ ] **`src/generator_estimator.py`** — Core generator estimator using Profile REML. **WIP indicator**: the README reports 25-40% MRE on drift estimation, suggesting this module is not yet stable. Assess: how mature is the API, what's the test coverage, what are the known limitations, and how much work would it be to wrap as an `inventory_variance_estimator` callable.

- [ ] **`src/rough_paths_generator.py`** — fBm simulation. **Probably stable** (simulation is easier than estimation). Assess: is it well-tested, can we use it to generate rough vol envs (rBergomi, rough Heston) for our Stage 5+ testing?

- [ ] **`src/hurst_estimators.py`** — Hurst estimation methods. **Critical for use case 4 and indirectly for use cases 1-3** (since the whitening discipline depends on a good `H` estimate). Assess: which methods are implemented, what's the test coverage, what error rates do they achieve.

- [ ] **`src/spectral_hurst.py`** — Spectral Hurst estimation. **Critical for use case 2** (regime detection). Assess: does it match the Paper 6a theoretical claim `|λ_k| ~ k^{-(2H+1)}`, what's the test coverage, is it numerically stable.

### 3.3 Test execution

- [ ] **`conda activate fsde && python -m pytest -q tests/`** (or whatever the equivalent command is — check `environment.yml` and any test runner configuration). Report: number of tests, number passing, number failing, time to run.
- [ ] **`python examples/test_generator_estimator.py`** — does the example run end-to-end without errors?
- [ ] **`python examples/test_nystrom_koopman_signatures.py`** — does this example work?
- [ ] **`python examples/test_signature_consistency_training.py`** — does this work?
- [ ] **Note any tests that hang, fail, or produce warnings.** These are red flags for stability.

### 3.4 Methodological audit

Apply the same discipline rules from `pomdp-koopman-control`'s feedback memories to assess methodological maturity:

- [ ] **Train/test split**: Do the estimation procedures use proper held-out evaluation, or are reported error rates in-sample? **Per `pomdp-koopman-control`'s March 2026 audit findings**, in-sample R² is one of the most common methodology bugs. Check whether the `fsde-identifiability` reported error rates (2-3% on `H`, etc.) are in-sample or held-out.
- [ ] **Multi-seed evaluation**: Are the reported error rates from a single seed or averaged over multiple seeds? Are error bars reported?
- [ ] **Paired evaluation**: When comparing methods (e.g., signature features vs raw features), are the comparisons paired across seeds, or marginal? Paired comparisons are required for the disciplined evaluation in `pomdp-koopman-control`.
- [ ] **Pre-registration**: Are the estimation choices (signature truncation, Nyström rank, kernel bandwidth, etc.) pre-registered, or selected post-hoc to optimize the reported metrics?
- [ ] **Bayesian inference**: Does the repo use Bayesian inference (consistent with `pomdp-koopman-control`'s `feedback_no_frequentist_bayesian_mixing.md` rule) or frequentist? If frequentist, is there a Bayesian variant available?

These items are NOT meant as criticisms of the sibling repo — it's a paper-in-progress and may legitimately be at an earlier stage of methodological hardening. The point is to identify which modules are mature enough to import without inheriting methodology debt.

### 3.5 Stability and import readiness

For each module marked as "potentially importable" in section 3.2, assess:

- [ ] **API stability**: Has the module's public interface changed in the last 30 days? In the last 90 days?
- [ ] **Documentation completeness**: Is there a clear way to call the module's main functions without reading the source?
- [ ] **Dependencies**: What does this module depend on (within the repo and externally)? Are there any heavy dependencies (PyTorch, JAX, specific BLAS versions) that would conflict with `pomdp-koopman-control`?
- [ ] **Testability**: Can we test the module in isolation, or does it require fixtures from elsewhere in the repo?
- [ ] **Import vs reimplement decision**: Based on the above, would it be cleaner to import from `fsde-identifiability` or to re-implement the small subset we need *inside* `pomdp-koopman-control`?

The default answer should be **re-implement the small subset we need**, unless the module is genuinely large and stable. Importing creates a cross-repo dependency that's hard to manage.

---

## 4. Audit report format

The output of the audit is a single markdown file at `docs/refs/fsde_identifiability_audit_report.md` with the following sections:

1. **Header**: date of audit, time spent, codex version
2. **Executive summary**: 1-paragraph verdict on whether the sibling repo is mature enough for any kind of import, with a clear "yes / no / partial" recommendation
3. **Module-by-module assessment**: for each of the modules in section 3.2, a brief (3-5 sentence) assessment of stability, test coverage, and import readiness
4. **Methodological audit findings**: results of the section 3.4 checks
5. **Test execution results**: from section 3.3
6. **Recommended action**: one of:
   - **(a) Import as-is** (specific modules that are stable enough to depend on)
   - **(b) Import with audit-level changes** (modules that need small fixes before import)
   - **(c) Re-implement the small subset we need** (modules where re-implementation is cleaner than import)
   - **(d) Wait** (modules that are too WIP to use; revisit after the JMLR paper is complete)
7. **Next steps**: what would have to happen for us to actually use the `fsde-identifiability` infrastructure in Stage 5+
8. **Audit checklist completion record**: each item from section 3 with its status (✓ / ✗ / N/A)

---

## 5. Decision criteria — when do we import?

After the audit, Ed will use the report to decide. The decision criteria:

**Import as-is** if all of:
- The module's API has been stable for 30+ days
- All existing tests pass
- The methodological audit (section 3.4) reveals no major concerns
- The reported error rates are held-out, not in-sample
- Dependencies are compatible with `pomdp-koopman-control`'s env
- Re-implementation would take more than ~2-3 days

**Import with audit-level changes** if all of the above EXCEPT:
- Some tests fail or some methodological items are weak, but the issues are fixable in <1 day of work
- Ed agrees the issues are fixable

**Re-implement the small subset we need** if any of:
- The module is small (< 500 lines) and re-implementation is faster than auditing
- We only need a tiny fraction of what the module provides
- The cross-repo dependency creates ongoing maintenance burden
- The module's API is not stable

**Wait** if any of:
- Major methodology concerns (in-sample fitting, single-seed reporting, pre-registration violations)
- Active churn in the public API
- The math in the implementation doesn't match the math in the JMLR paper outline
- The relevant tests fail and the failures aren't easily fixable

---

## 6. Discipline rules for this audit

These apply to codex performing the audit:

1. **Read the math first.** Read the JMLR paper outline (`papers/jmlr_fsde_identifiability_outline.md`) and Paper 6a (`docs/spectral_roughness_theory.md`) BEFORE reading the code. The paper is the spec; the code is the implementation. Cross-check the implementation against the spec.
2. **Don't modify any code in `fsde-identifiability`.** This is a read-only audit. If you find issues, document them in the audit report — don't fix them.
3. **Don't commit anything to `fsde-identifiability`.** Same reason.
4. **Don't import any code from `fsde-identifiability` into `pomdp-koopman-control`.** Even if you're tempted to write a quick proof-of-concept import, don't. The audit is informational, not transformational.
5. **Report concerns honestly.** This is the same discipline as the Stage 4 v1 SDRE audit. If a module looks WIP, say so. If a test fails, report it. If the math in the implementation doesn't match the math in the paper, flag the discrepancy.
6. **Be skeptical of reported error rates.** "2-3% error on H" sounds great, but is it in-sample or held-out? Single-seed or multi-seed? On what kind of trajectories? The README's headline numbers are claims, not facts — verify them in the test code.
7. **Estimate your confidence.** For each module assessment, indicate your confidence (low / medium / high). If you only had time for a superficial read, say so. Don't pretend to a confidence level you don't have.
8. **Time-box the audit.** ~1 day total. If you're going significantly over, stop and report what you have rather than rushing the rest. A partial audit with a clear "next step is to read modules X and Y" is more useful than a full audit with shallow assessments.

---

## 7. What happens after the audit

Once the audit report is written:

1. **Ed reads the report.** ~30-60 minutes.
2. **Ed decides** which modules (if any) to import, re-implement, or defer.
3. **For any module that gets imported**, codex does an additional pass to verify the import works in the `pomdp-koopman-control` environment, and writes tests for the integration.
4. **For modules deferred**, the audit report becomes a living document — re-audit when the JMLR paper is finalized or when the module's API stabilizes.
5. **The Stage 5+ planning in `docs/plan_omm_research.md`** is updated to reflect what's importable and what isn't.

---

## 8. Relationship to Stage 4 v2

**This audit is independent of Stage 4 v2.** It can run in parallel to v2 implementation, or it can run after v2 lands. Stage 4 v2 does NOT require any imports from `fsde-identifiability` — it's Heston-only with Markov filtering, no signatures or Koopman CdC.

**The audit informs Stage 5+**, which is when we'd potentially need rough vol / signature / Koopman CdC machinery. If the audit clears, Stage 5+ has a green light to import. If the audit doesn't clear, Stage 5+ either re-implements the relevant pieces from scratch or defers the rough vol experiments until the sibling repo is more mature.

**The audit should NOT be conflated with Stage 4 v2 work.** The user's previous instruction (this conversation) was explicit: stay POMDP-only for v2, defer fSDEs, audit the sibling repo as a separate task.

---

## 9. Why this matters (the bigger picture)

The OMM SDRE work in `pomdp-koopman-control` and the fSDE identifiability work in `fsde-identifiability` are intellectually connected — both use signatures, Koopman generators, and the user's nonparametric approach to learning dynamics from data. **There's a real opportunity for the two projects to reinforce each other**: the OMM application could become a deployment demonstration for the fSDE identifiability methodology, and the fSDE identifiability work could provide the model-free dynamics learner for OMM under non-Markov conditions.

But this synergy only works if the dependency direction is clean. The OMM thesis chapter and the JMLR fSDE identifiability paper should each stand on their own, with the OMM chapter *citing* the JMLR paper as related work and *optionally* using its infrastructure (with audit), rather than *depending* on it.

The audit task here is what makes this clean separation possible. It tells us which parts of the JMLR work are stable enough to depend on, which parts we'd re-implement, and which parts are still research-in-progress that we shouldn't touch.

---

## 10. Changelog

| Date | Change |
|---|---|
| 2026-04-08 | Initial spec (Claude). Created when scope-narrowing the OMM v2 derivation note moved the "fSDE / signature / Koopman CdC" material out of the v2 critical path. This task is the new home for the "intended usage" context. |
