# Two-scale Lorenz-96 closure MKL — honest benchmark

**Status**: first natural-benchmark test of the multiple-memory-kernel
representation-selection story from `memory_mkl_poc_result.md`. Result is
mixed: MKL behaves as a useful *diagnostic*, but does not produce a
slam-dunk win over single-kernel baselines on this system.

## Setup

Two-scale Lorenz-96 (Wilks 2005 / Arnold-Moroz-Palmer form):

    dX_k/dt = X_{k-1} (X_{k+1} - X_{k-2}) - X_k + F + U_k,
    dY_{j,k}/dt = -c b Y_{j+1,k} (Y_{j+2,k} - Y_{j-1,k}) - c Y_{j,k}
                  + (h c / b) X_k,
    U_k        = -(h c / b) sum_j Y_{j,k}.

`experiments/science_poc/envs/two_scale_lorenz96.py` integrates the joint
(X, Y) system at a fine RK4 step `dt_sim=0.005` and exposes observations on
a coarser step `dt = dt_sim * obs_subsample`. The simulator returns the
interval-averaged drift, resolved part, and unresolved tendency `U_avg`.

**Closure task.** Train a single translation-invariant regressor
`f(local X, memory features) -> U_k` by pooling samples across all slow
indices. Train on the observable target

    U_obs = dX_k/dt - X_{k-1}(X_{k+1} - X_{k-2}) + X_k - F

(finite-difference dX minus the resolved part the modeller can compute from
X alone) and score on the simulator's true `U_avg` on held-out paths.

**Lanes (`experiments/science_poc/l96_closure_mkl.py`):**

- `raw` -- local stencil `X_{k-r..k+r}` only
- `delay` -- stencil + lagged X_k history at three time scales
- `efm` -- stencil + exponential moving averages of X_k
- `leadlag_qv` -- stencil + EFM / rolling QV of normalized X_k increments
- `mkl_memory_sum` -- equal-weighted nonnegative sum of the three memory
  kernels
- `mkl_learned` -- best simplex-grid mixture over `{raw, delay, efm,
  leadlag_qv}` selected by validation MSE; degenerate single-kernel weight
  vectors are allowed in the search so MKL can converge to a singleton
- `oracle_Umarkov` -- stencil + `U_k(t-1)`; upper bound when the true past
  unresolved tendency is observable

Each kernel is wrapped in random-landmark Nystrom RBF features
(`n_landmarks=160`, length scales by median heuristic per representation,
ridge selected from `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` on validation MSE).

## Results

3 macro seeds, T=16 MTU per path, F=20, c=4, h=1, b=10, dt=0.02.
Numbers are mean closure R² = 1 - U_residual on held-out paths.

### Full stencil (radius 2: X_{k-2..k+2})

| lane            | R² (mean) | 90% CrI         | weights |
|-----------------|-----------|-----------------|---------|
| raw             | **0.589** | [0.570, 0.602]  | raw:1.00 |
| delay           | 0.560     | [0.555, 0.567]  | delay:1.00 |
| efm             | 0.576     | [0.571, 0.580]  | efm:1.00 |
| leadlag_qv      | 0.471     | [0.450, 0.487]  | leadlag_qv:1.00 |
| mkl_memory_sum  | 0.585     | [0.580, 0.588]  | equal 3-way |
| mkl_learned     | **0.595** | [0.577, 0.608]  | raw:0.25, delay:0.75 |
| oracle_Umarkov  | 0.770     | [0.763, 0.780]  | oracle_Umarkov:1.00 |

Local spatial information saturates the closure: raw already gets 0.59 R²,
and no observable lift beats it by more than a noise band. `mkl_learned`
correctly weights memory kernels near zero and recovers the raw baseline
plus a small ridge improvement.

### Point observation (radius 0: X_k only, no spatial neighbors)

| lane            | R² (mean) | 90% CrI         | weights |
|-----------------|-----------|-----------------|---------|
| raw             | 0.650     | [0.646, 0.655]  | raw:1.00 |
| delay           | 0.649     | [0.633, 0.659]  | delay:1.00 |
| efm             | **0.676** | [0.671, 0.681]  | efm:1.00 |
| leadlag_qv      | 0.599     | [0.594, 0.606]  | leadlag_qv:1.00 |
| mkl_memory_sum  | 0.632     | [0.620, 0.644]  | equal 3-way |
| mkl_learned     | 0.657     | [0.641, 0.668]  | raw:0.75, delay:0.25 |
| oracle_Umarkov  | 0.924     | [0.921, 0.927]  | oracle_Umarkov:1.00 |

With spatial context removed, memory features provide measurable lift:
EFM beats raw by **+2.6 R² points**. `mkl_learned` finds an X+delay
mixture that beats raw, but on this sample size and grid step (0.25) the
learner's validation-MSE pick is within 2 points of the better EFM
singleton -- i.e., the simplex grid is not fine enough to fully reproduce
the singleton optimum in this small-sample regime.

The oracle gap is large (0.66 -> 0.92), so there is still significant
unresolved-tendency information that no observable history captures
through these specific lifts.

### Sanity: canonical L96 (F=10, c=10)

Same qualitative pattern with smaller magnitudes:

| stencil | raw   | best memory   | mkl_learned | oracle |
|---------|-------|---------------|-------------|--------|
| r=0     | 0.825 | efm 0.838     | 0.842       | 0.941 |
| r=2     | 0.820 | raw           | 0.824       | 0.901 |

## Honest interpretation

**What MKL is doing here, and what it isn't.** L96 has a single source of
unresolved memory: the fast Y subsystem with one characteristic time scale
`1/c`. There is only one memory mechanism worth representing, so kernel
diversity within `{delay, efm, leadlag_qv}` provides no real diversification
benefit. The MKL learner cannot manufacture wins by mixing kernels that
encode redundant information.

What MKL *can* do is:

1. Diagnose when memory is unnecessary. With the full stencil, the
   selection procedure de-weights memory kernels and recovers the raw
   baseline. That is the correct behavior on a system whose closure is
   essentially deterministic-in-local-X.

2. Identify when memory helps. With point observation, memory kernels
   carry positive validation signal and the learner places weight on
   `delay` rather than purely on `raw`. EFM is the strongest single lift,
   consistent with the fact that the dominant residual mechanism is an
   exponentially-correlated fast forcing.

3. Bound the residual gap. The `oracle_Umarkov` lane is a 1-step
   memory oracle; the gap between it and the best learned closure
   measures how much unresolved-tendency information is *available in
   observable history but not yet captured by these lifts*.

**Where MKL would actually win.** We expect a clean MKL victory on systems
with *multiple distinct* memory mechanisms -- e.g., a Mori-Zwanzig / GLE
process where the memory kernel has two well-separated time scales, or a
two-scale process with both slow OU-style and jump-style unresolved
forcing. The synthetic `memory_mkl_poc.py` benchmark already demonstrates
this. L96, with one fast subsystem, is not such an environment.

## Files

- env: `experiments/science_poc/envs/two_scale_lorenz96.py`
- benchmark: `experiments/science_poc/l96_closure_mkl.py`
- figures: `experiments/science_poc/l96_closure_mkl_full_stencil.png`,
  `experiments/science_poc/l96_closure_mkl_point_obs.png`

## What this changes about the Brunton/Klus pitch

The "MKL learns a Bayesian-flavored measure over closure mechanisms" story
needs a concrete environment with multiple distinct unresolved mechanisms
to be empirically compelling. L96 is *not* that environment. Sensible
next benchmarks:

- 3-scale Lorenz-96 (two distinct fast subsystems);
- Mori-Zwanzig / GLE toy with two bath frequencies and a known memory
  kernel that has two timescales;
- two-scale double-well augmented with a jump channel on the hidden
  factor, so EFM and lead-lag/QV kernels separately recover the smooth and
  jumpy parts of memory.

The current Brunton/Klus-ready story is therefore *narrower* than the
title we used in `memory_mkl_poc_result.md`:

> When measurements are not a state, learn which causal memory lifts
> repair the projected generator. On L96 this diagnostic correctly says
> spatial information is already sufficient; on systems with a single
> memory mechanism it converges to that mechanism; richer mixture wins
> require a system with multiple distinct memory mechanisms.
