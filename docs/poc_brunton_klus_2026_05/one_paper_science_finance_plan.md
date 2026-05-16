# One-Paper Plan: Signature/RKHS Koopman Generators for Non-Markovian Dynamics

Date: 2026-05-16

## Purpose

This note is the handoff plan for a new chat instance. The current RKHS-SDRE
control line should be paused as a flagship paper. The strongest one-paper
direction is a science-first operator-learning paper that keeps the finance
application as a final stress test.

## Working Title

**Signature/RKHS Koopman Generators for Partially Observed Non-Markovian
Dynamics**

Alternative shorter title:

**Operator Learning for Non-Markovian Scientific Dynamics via Signature RKHS
Lifts**

## One-Sentence Claim

Delay/signature RKHS lifts can turn partially observed, non-Markovian stochastic
dynamics into a tractable Koopman-generator learning problem, and residual /
spectral diagnostics identify when the learned generator is reliable.

## Why This Pivot

The recent CartPole/RKHS-SDRE work gave a useful negative result but not a
strong paper by itself:

- RKHS/Nystrom structured control-affine SDRE is a good near-upright terminal
  controller.
- It does not solve CartPole swing-up, because swing-up is primarily a planning
  and reachability problem.
- Oracle LQR is too strong near equilibrium, so "better than LQR" is not the
  right claim.

The stronger contribution is not another local controller. It is a general
operator-learning method for systems where the observed state is not Markovian:
coarse-grained scientific systems, hidden-state stochastic systems, rough paths,
and latent-volatility finance.

## Target Audience

Primary:

- Koopman/operator-learning researchers
- scientific machine learning researchers
- data-driven stochastic dynamics researchers

Relevant to Brunton/Klus because it emphasizes:

- generators rather than black-box predictors,
- residual/spectral diagnostics,
- scientific dynamical systems before finance,
- model reduction and coarse-graining,
- partial observation and memory.

Finance should be included, but not as the lead motivation.

## Core Method

Given observations `y_t` from a system whose observed coordinate is not Markov,
construct a lifted state:

```text
z_t = Phi(history_t)
```

where `Phi` is one of:

- delay coordinates,
- truncated signatures / log-signatures,
- recurrent signature features,
- RKHS/Nystrom features on path summaries.

Then learn a generator or controlled generator:

```text
L f(z) ~= drift/diffusion/generator action on observables
```

or, for discrete data:

```text
K Phi(history_t) ~= Phi(history_{t+dt})
L ~= (K - I) / dt.
```

The important distinction from the failed CartPole line:

- use RKHS/signatures to repair non-Markovian observation geometry,
- do not claim this alone solves global planning,
- evaluate generator quality and downstream scientific quantities.

## Diagnostics to Emphasize

Every benchmark should report at least:

- one-step and multi-step prediction error,
- generator residual / ResDMD-style residual if available,
- implied timescale / eigenvalue stability across ranks,
- held-out likelihood or transition error,
- committor / transition-rate error when applicable,
- filtering RMSE/correlation for hidden states,
- downstream task metric, e.g. hedging variance reduction.

The paper should avoid relying on prediction RMSE alone.

## Application Ladder

### 1. Two-Scale Langevin / Generalized Langevin Double-Well

Science-first benchmark. Observe only the slow coordinate while hidden fast
variables induce memory.

Goal:

- raw-state generator fails,
- delay/signature RKHS generator recovers drift/diffusion or transition
  statistics,
- diagnostics predict success/failure.

Metrics:

- drift/diffusion error if ground truth is known,
- committor error,
- transition-rate / mean-first-passage-time error,
- generator residual.

Source repos to inspect:

- `../fsde-identifiability`
- `../rkhs_kronic`
- `../rkhs-koopman-control`

### 2. Two-Scale Lorenz-96 or HAVOK-Style Partial Observation

Brunton-friendly deterministic or stochastic science benchmark.

Goal:

- demonstrate that path/signature features recover effective Markovian state
  information from partial observations.

Metrics:

- forecast error,
- spectral stability,
- implied timescales,
- residual diagnostics.

Possible variants:

- Lorenz-96 slow variables with unresolved fast variables,
- fluid wake / cylinder wake delay-coordinate example if data/tooling exists,
- Duffing/HAVOK as a lighter initial test.

### 3. Metastable Stochastic Dynamics / Transfer-Operator Quantity

Klus-friendly benchmark.

Goal:

- show that the learned generator gives useful metastability quantities, not
  just forecasts.

Metrics:

- committor error,
- dominant eigenfunction/eigenvalue stability,
- transition rates,
- Chapman-Kolmogorov or implied-timescale consistency.

### 4. Finance Stress Test: Latent Volatility Filtering / Hedging

Keep finance in the paper as the final real-world non-Markovian stochastic
application.

Candidate claim:

- streaming signature/RKHS Koopman belief features estimate latent volatility
  competitively with particle/Kalman baselines,
  then improve hedging or variance-control metrics.

Metrics:

- latent `V_t` filtering RMSE/correlation on Heston/Bates/rough-vol synthetic
  data,
- hedging error / variance reduction,
- runtime per step,
- comparison to EWMA, Kalman/EKF, BPF, GARCH-style baselines where applicable.

Source repo:

- `../pomdp-koopman-control`

Finance should not dominate the paper's introduction. It should demonstrate
cross-domain scope after the science benchmarks establish the method.

## What Not To Claim

Do not claim:

- this solves general POMDP control,
- RKHS-SDRE beats oracle LQR,
- signatures automatically solve planning,
- finance is the central motivation for Brunton/Klus.

Do claim, if validated:

- signatures/delay RKHS features repair non-Markovian observation geometry,
- generator diagnostics distinguish trustworthy from untrustworthy learned
  operators,
- the same pipeline works on scientific and financial partially observed
  stochastic systems.

## Immediate Next Steps for New Chat

Start in the source repo:

```bash
cd /home/ed/SynologyDrive/Documents/Research/PE_Research/pomdp-koopman-control
conda activate rkhs-kronic-gpu
```

Use this internal split:

- `docs/poc_brunton_klus_2026_05/` for the Brunton/Klus narrative.
- `experiments/science_poc/` for runnable science-first examples.
- `finance/` only for the final stress test, not the lead motivation.

Then work in this order:

1. Current science POC seed
   - Rerun `experiments/science_poc/latent_ou_representation_demo.py`.
   - Rerun `experiments/science_poc/latent_ou_representation_demo_bayesian.py`.
   - Use this as the warm-up figure, not the whole POC.

2. First generator benchmark
   - Implement two-scale Langevin / generalized Langevin double-well under
     `experiments/science_poc/`.
   - Report raw-state vs delay/signature/RKHS lift.
   - Include generator residual, spectral/timescale stability, and a
     downstream quantity such as committor or transition rate.

3. `../fsde-identifiability`
   - Find runnable examples/tests.
   - Verify Hurst, diffusion, and drift estimation on held-out seeds.
   - Check whether reported drift error is in-sample or held-out.

4. `../rkhs_kronic`
   - Reuse only the parts that support the new paper:
     RKHS/Nystrom generator machinery, residual diagnostics, and the negative
     CartPole result as a cautionary appendix if useful.

5. Finance stress test
   - Rerun the strongest filtering/hedging claims after the science figure is
     coherent.
   - Confirm whether `55%` variance reduction and `0.91x` BPF MSE survive
     clean seed sweeps before using those numbers.

## Go / No-Go Criteria

This is a good one-paper direction only if at least two science benchmarks and
one finance benchmark satisfy:

- signature/RKHS/delay lift beats raw-state generator on held-out metrics,
- diagnostics correlate with downstream success,
- baselines are fair and not artificially weak,
- commands and outputs are reproducible.

Minimum publishable evidence package:

- one synthetic science benchmark with known ground truth,
- one higher-dimensional or partial-observation science benchmark,
- one finance benchmark,
- one diagnostic figure showing why the lift works or fails,
- one ablation table: raw state vs delay vs signature vs RKHS/Nystrom.

## Relationship to Current RKHS-SDRE Results

The current CartPole and double-well control results should be used only as
supporting evidence:

- double-well: controlled generator geometry matters;
- CartPole: local terminal control works, but global planning is separate;
- conclusion: operator learning should be paired with the right downstream
  objective, and diagnostics matter.

This keeps the useful lessons without forcing them to be the paper's central
contribution.
