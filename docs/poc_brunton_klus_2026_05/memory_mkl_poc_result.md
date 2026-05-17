# Multiple-Memory-Kernel POC Result

Date: 2026-05-16

## Command

```bash
conda activate rkhs-kronic-gpu
python experiments/science_poc/memory_mkl_poc.py
```

Equivalent non-interactive check:

```bash
conda run -n rkhs-kronic-gpu python experiments/science_poc/memory_mkl_poc.py
```

## Purpose

This is an exploratory stronger-story POC, not the primary Brunton/Klus
benchmark. The environment is intentionally constructed so the generator drift
depends additively on two different observable memory functionals:

- `M_t`: an exponential filter of past observed increments.
- `Q_t`: an exponential filter of past squared increments, equivalent to a
  semantic lead-lag/QV signature channel.

The true generator action is used only for held-out scoring. Training and
validation use observed `dX / dt` targets only.

## Latest Output

```text
dt=0.02 T=10.0 sigma_X=0.22 tau_m=0.85 tau_q=0.25
Training/tuning target: observed dX/dt only. True drift is held-out scoring only.
Model: random-landmark RBF blocks with nonnegative kernel-sum weights.

model           gen_resid mean [90% CrI]     drift_corr mean [90% CrI]   inc_mse   selected weights
raw_x             0.946 [ 0.875,  1.012]         +0.144 [+0.084, +0.216]     3.025   raw_x:1.00
delay_coords      0.493 [ 0.471,  0.524]         +0.705 [+0.677, +0.721]     2.750   delay_coords:1.00
efm_memory        0.496 [ 0.483,  0.516]         +0.722 [+0.715, +0.732]     2.748   efm_memory:1.00
leadlag_qv        0.379 [ 0.336,  0.413]         +0.806 [+0.795, +0.822]     2.684   leadlag_qv:1.00
mkl_memory_sum    0.288 [ 0.235,  0.335]         +0.886 [+0.875, +0.899]     2.619   efm_memory:0.50,leadlag_qv:0.50
mkl_learned       0.235 [ 0.182,  0.275]         +0.899 [+0.880, +0.916]     2.579   raw_x:0.25,delay_coords:0.50,leadlag_qv:0.25
oracle_mq         0.165 [ 0.130,  0.203]         +0.933 [+0.920, +0.947]     2.564   oracle_mq:1.00

best single / learned MKL generator-residual ratio: 1.61x
```

Figure:

```text
experiments/science_poc/memory_mkl_poc.png
```

## Reading

This gives the desired proof-of-concept pattern: no single observable memory
kernel recovers the generator well, but a nonnegative kernel sum gets
substantially closer to the hidden-memory oracle.

The caveat is important: this environment was designed to have additive hidden
memory components. Its value is as a controlled demonstration of the
operator-aware MKL mechanism, not yet as evidence that the approach wins on
natural science systems.
