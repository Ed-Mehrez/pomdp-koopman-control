# Two-Scale Generator POC Result

Date: 2026-05-16

## Command

```bash
conda activate rkhs-kronic-gpu
python experiments/science_poc/two_scale_generator_poc.py
```

Equivalent non-interactive check:

```bash
conda run -n rkhs-kronic-gpu python experiments/science_poc/two_scale_generator_poc.py
```

## Benchmark

Partially observed two-scale double-well:

```text
dX_t = (X_t - X_t^3 + beta H_t) dt + sigma_X dW_X
dH_t = -kappa_H H_t dt + sigma_H dW_H
```

Only `X_t` is observed. The hidden OU factor `H_t` makes the observed
coordinate non-Markovian. The goal is to recover the generator action on the
coordinate observable, `L x = X - X^3 + beta H`, from different state
representations.

The script reuses:

- `src.control.kernel_residual_control.GPResidualModel` as the RBF GP/KRR head.
- `src.control.state_transform.EFMLevel1` for the fading-memory lift.

## Latest Output

```text
dt=0.01 T=8.0 beta=1.4 sigma_X=0.15 kappa_H=1.5 hidden_timescale=0.667
Target: recover the known generator action on f(x,h)=x from each representation.
Secondary column `noisy_corr` fits the same GP/KRR head on dX/dt targets.

rep          dim  drift_corr  drift_nrmse  gen_resid  noisy_corr  hidden_corr  hidden_rmse  koopman_times
raw_x          1      +0.061        1.002      1.004      +0.028       +0.653        0.450  [26.91]
delay_x        3      +0.638        0.789      0.623      +0.456       +0.794        0.361  [16.52, 0.37, 0.37]
efm_dx         4      +0.705        0.713      0.507      +0.501       +0.828        0.333  [8.07, 8.07, 1.27]
oracle_xh      2      +0.998        0.060      0.004      +0.909       +0.999        0.024  [4.24, 0.67]

generator residual improvement raw_x / delay_x: 1.61x
generator residual improvement raw_x / efm_dx: 1.98x
generator residual improvement raw_x / oracle_xh: 277.71x
```

Figure:

```text
experiments/science_poc/two_scale_generator_poc.png
```

## Reading

Raw `X_t` is not enough to recover the generator because the drift depends on
the hidden `H_t`. Delay and fading-memory path lifts recover substantially more
of the hidden driver and cut the normalized generator residual. The hidden-state
oracle verifies that the GP/KRR head itself is not the bottleneck.

The `noisy_corr` column is deliberately harder: it trains on single-step
`dX/dt` targets rather than the known generator action. The same ranking holds,
but the correlations are lower because the target includes diffusion noise.
