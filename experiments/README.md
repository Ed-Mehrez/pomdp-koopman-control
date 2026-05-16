# Experiments

This directory is split by research lane.

## `science_poc/`

Science-first proof-of-concept work for the Brunton/Klus outreach package.
This lane should lead with operator learning, partial observation, memory, and
diagnostics. Finance vocabulary should stay out of the core example unless it
is explicitly being used as a final stress test.

Current runnable checks:

```bash
conda activate rkhs-kronic-gpu
python experiments/science_poc/latent_ou_representation_demo.py
python experiments/science_poc/latent_ou_representation_demo_bayesian.py
```

## Finance Experiments

Finance experiments remain under `finance/experiments/`. They are not the lead
POC lane for Brunton/Klus; use them as stress tests or appendix evidence after
the science benchmark is clear.
