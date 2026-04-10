# Note: The sigma_sq_inv estimation channel is negative under tested Heston regimes

**Date**: 2026-04-10
**Status**: Frozen finding. Do not expand this line.

## Summary

The inventory-skew estimation channel (`sigma_sq_inv -> skew`) adds no value
under the Heston regimes tested in Stages 1-4. Oracle screening confirms
that even with *perfect* knowledge of `sigma_sq_inv`, the inventory skew
contribution is small (~7.8% of `1/k = 0.20` at default parameters) and
does not translate into CE gains over the risk-neutral baseline.

## Evidence

1. **Stage 3 filter ablation** (locked 2026-04-07): Oracle/BPF/RecSig/EWMA
   filters are within 0.138 CE of each other vs a 26.97 CE controller gap.
   Filter quality accounts for <0.5% of the total advantage. Improving
   variance estimation does not improve control.

2. **Oracle screening grid** (`oracle_screening_grid.py`, 2026-04-10):
   Sweeping gamma from 0 to 0.001 under oracle variance shows the
   risk-neutral optimal (`1/k`) is essentially saturated. The skew
   contribution from `gamma_local * sigma_sq_inv * tau * q` is real but
   negligible at daily frequency Heston.

3. **Magnitude check** (derivation_omm_sdre_v2.md, Section 6): At Heston
   defaults, CRRA(gamma=2) at W=100k gives `gamma_local = 2e-5`,
   `inventory_skew ~ 0.0156 $/share` per unit q. This is 7.8% of the
   base half-spread `1/k = 0.20`.

## Implication

Signatures, kernels, and other sophisticated estimators should **not** be
spent on improving `sigma_sq_inv` estimation for the inventory-skew formula
in this regime. The signal is too small.

Instead, the model-free / data-driven methodology should be directed at
learning the local reward landscape *directly* from belief/path features
(Track B of the salvage plan).

## What this does NOT say

- This does not mean inventory control is unimportant. It means the
  *analytic BG/AS formula's inventory skew term* is small in Heston daily.
- This does not mean the estimation channel is negative for *all* regimes.
  Under rough volatility, higher gamma, shorter dt, or more volatile
  inventory dynamics, the channel may become significant. That is a Stage 5+
  question.
- The result is specific to the Heston parameters in the Stage 1-4 grid
  (kappa=2, theta=0.04, xi=0.5, rho=-0.7, gamma=2).
