# Fair Two-Scale Benchmark Result

Date: 2026-05-16

## Command

```bash
conda activate rkhs-kronic-gpu
python experiments/science_poc/two_scale_fair_benchmark.py
```

Equivalent non-interactive check:

```bash
conda run -n rkhs-kronic-gpu python experiments/science_poc/two_scale_fair_benchmark.py
```

## Fairness Rules

- Training target is observed `dX / dt` only.
- Hyperparameters are selected on independent validation trajectories using
  validation increment MSE.
- Final scores are evaluated on independent held-out trajectories.
- True drift is used only for scoring because the synthetic benchmark has known
  generator action.
- `oracle_xh` sees hidden `H_t` and is an upper bound, not an observable
  baseline.
- Model is Nyström RBF kernel ridge regression with random training landmarks.
- Signature lanes:
  - Signature paths are constructed from observed increments and explicit
    endpoint anchors. This is equivalent to using level paths up to translation,
    but it avoids hiding absolute-state information inside the signature.
  - `leadlag_summary`: endpoint anchors, displacement-rate, and QV-rate per
    lead-lag window.
  - `leadlag_summary_l3`: `leadlag_summary` plus cubic variation,
    time-ordered QV slope, and displacement-weighted QV. This is a semantic
    third-order summary, not a full level-3 log-signature.
  - `leadlag_sig`: full rolling lead-lag level-2 log-signature windows with
    endpoint anchors.
  - `leadlag_logsig_l3`: true rolling lead-lag level-3 log-signature windows
    from `signatory` in bracket coordinates, with endpoint anchors.
  - `cum_leadlag_sig`: cumulative lead-lag level-2 log-signature with `(X_t,
    X_0)` endpoint anchors.
  - `cum_leadlag_logsig_l3`: true cumulative lead-lag level-3 log-signature
    from `signatory` in bracket coordinates, with `(X_t, X_0)` anchors.
- Landmark RNGs are fixed per `(seed, representation)` so adding a benchmark
  lane does not change other lanes' landmark draws.

## Latest Output

```text
Training/tuning target: observed dX/dt only. True drift is used only for held-out scoring.
Model: Nyström RBF kernel ridge regression with random training landmarks.
dt=0.01 T=8.0 beta=1.4 sigma_X=0.15 kappa_H=1.5 hidden_timescale=0.667 seeds=[20260516, 20260517, 20260518, 20260519, 20260520]

rep           dim  m    gen_resid mean [90% CrI]     drift_corr mean [90% CrI]   test_inc_mse   trans_rmse
raw_x           1  96    1.078 [ 1.047,  1.112]         -0.049 [-0.099, -0.008]          2.676      0.0164
delay_coords    4  96    0.715 [ 0.636,  0.797]         +0.568 [+0.512, +0.620]          2.563      0.0160
efm_dx          4  96    0.708 [ 0.674,  0.750]         +0.576 [+0.549, +0.598]          2.539      0.0159
leadlag_summary  10  96    0.901 [ 0.824,  0.998]         +0.443 [+0.397, +0.488]          2.604      0.0161
leadlag_summary_l3  19  96    1.064 [ 1.048,  1.078]         +0.125 [+0.065, +0.215]          2.722      0.0165
leadlag_logsig_l3  94  96    1.011 [ 1.003,  1.024]         +0.029 [+0.007, +0.052]          2.646      0.0163
leadlag_sig    34  96    0.986 [ 0.945,  1.028]         +0.253 [+0.179, +0.311]          2.659      0.0163
cum_leadlag_sig  12  96    1.213 [ 1.106,  1.320]         +0.169 [+0.145, +0.194]          2.796      0.0167
cum_leadlag_logsig_l3  32  96    1.066 [ 1.030,  1.120]         +0.146 [+0.104, +0.176]          2.722      0.0165
oracle_xh       2  96    0.332 [ 0.286,  0.379]         +0.839 [+0.804, +0.868]          2.387      0.0154

raw_x / delay_coords generator-residual ratio: 1.55 [1.40, 1.70], P(delay_coords improves over raw_x)=1.000
raw_x / efm_dx generator-residual ratio: 1.53 [1.44, 1.61], P(efm_dx improves over raw_x)=1.000
raw_x / leadlag_summary generator-residual ratio: 1.22 [1.10, 1.30], P(leadlag_summary improves over raw_x)=0.998
raw_x / leadlag_summary_l3 generator-residual ratio: 1.01 [0.98, 1.04], P(leadlag_summary_l3 improves over raw_x)=0.759
raw_x / leadlag_logsig_l3 generator-residual ratio: 1.07 [1.04, 1.09], P(leadlag_logsig_l3 improves over raw_x)=1.000
raw_x / leadlag_sig generator-residual ratio: 1.10 [1.04, 1.14], P(leadlag_sig improves over raw_x)=0.999
raw_x / cum_leadlag_sig generator-residual ratio: 0.90 [0.83, 0.99], P(cum_leadlag_sig improves over raw_x)=0.013
raw_x / cum_leadlag_logsig_l3 generator-residual ratio: 1.02 [0.96, 1.08], P(cum_leadlag_logsig_l3 improves over raw_x)=0.626
raw_x / oracle_xh generator-residual ratio: 3.42 [2.86, 4.02], P(oracle_xh improves over raw_x)=1.000
```

Figure:

```text
experiments/science_poc/two_scale_fair_benchmark.png
```

## Reading

The fair benchmark still supports the core claim: observed raw state is not a
sufficient generator state, and memory/path lifts recover predictive generator
information from trajectories alone.

The strongest observable baselines here are EFM and standard delay coordinates.
The compact semantic level-2 signature lane improves over raw state, and full
rolling lead-lag signatures are also mildly positive after order-invariant
landmark seeding. The true rolling level-3 log-signature lane is positive but
weak, while the cumulative true level-3 log-signature is roughly neutral. The
semantic level-3 add-on does not help, and the cumulative level-2 lead-lag lane
is negative in this small-data Nyström setting.

This should be framed honestly: signature geometry can recover useful path
information, but this environment is currently an EFM/delay-coordinate win, not
a signature win.
