# Brunton/Klus Science POC

This folder is the science-first outreach package for the May 2026
Brunton/Klus proof of concept.

## Positioning

Lead with:

- Koopman generators rather than black-box prediction.
- Partial observation, memory, and coarse-graining.
- Delay/signature/RKHS lifts as the representation repair.
- Residual, spectral, and downstream diagnostics.

Do not lead with:

- Finance.
- Global POMDP-control claims.
- CartPole swing-up or broad SDRE claims.

Finance remains valuable as a final stress test, but the first viewport for
Brunton/Klus should be a scientific dynamical system.

## Repo Split

The repo is split internally rather than forked:

| Lane | Location | Role |
|---|---|---|
| Science POC | `experiments/science_poc/` | Brunton/Klus runnable examples and figures |
| Science POC docs | `docs/poc_brunton_klus_2026_05/` | outreach narrative, evidence checklist, paper plan |
| Finance stress tests | `finance/` | Heston/Bates/OMM application evidence |
| Shared filtering/control code | `src/sskf/`, `src/control/` | reusable pieces only when both lanes actually share them |
| Retrospective slides | `docs/slides/` | audience-specific decks, including Jose finance retrospective |

## Near-Term Evidence Package

For the next few days, keep the package narrow:

1. A clean partially observed science example where raw observations are the
   wrong state and a fading-memory/path lift is the right state.
2. A generator-learning benchmark on a science system, preferably two-scale or
   double-well, with raw-state vs delay/signature/RKHS comparison.
3. Diagnostics beyond RMSE: generator residual, eigenvalue/timescale stability,
   and a downstream quantity such as committor or transition rate.
4. One finance stress-test slide only after the science evidence is coherent.

## Current Runnable Seed

```bash
conda activate rkhs-kronic-gpu
python experiments/science_poc/latent_ou_representation_demo.py
python experiments/science_poc/latent_ou_representation_demo_bayesian.py
```

The latent-OU demo is a fast warm-up figure: hidden ergodic factor, non-ergodic
observed level, exact Kalman reference, and no finance framing. It should not
be the full POC by itself; use it to motivate the two-scale/double-well
generator benchmark.
