# BBG SDRE Recovery Plan

## Theory separation

The BBG benchmark and the SDRE candidate controller are strictly separate:

**Benchmark** (src/applications/option_mm_bbg/solver.py):
- Solved reduced HJB on (t, nu, V^pi)
- Uses exact BBG intensity formulas, grid solver, Hamiltonian optimization
- Produces optimal quote tables given full model knowledge
- This is the **ground truth** we compare against

**Candidate** (src/applications/option_mm_bbg/sdre_recovery.py):
- Learns local controlled dynamics from simulated transitions
- Forms a local quadratic Hamiltonian from estimated quantities
- Solves for quotes via local SDRE
- **No BBG formulas inside the policy at action time**
- BBG appears only as an external benchmark

## What the candidate controller will do

1. **Collect transition data**: run episodes with generic exploration, recording
   (book_state, quote_action, next_book_state, wealth_change, vega_change)

2. **Estimate local controlled dynamics**: fit a control-quadratic model for
   how (V^pi, wealth) evolve as a function of (state, quotes). The key
   difference from the stylized env: the BBG book has 20 options contributing
   to portfolio vega, so the action space is 2N-dimensional (bid/ask per option).

3. **Form local quadratic Hamiltonian**: assemble stage cost (spread revenue -
   gamma-weighted vega risk) plus continuation-value estimate into a
   quadratic form in the quote actions.

4. **Extract optimal quotes**: solve the local quadratic for each option's
   bid/ask distances. This is a 2N-dimensional SDRE solve, but the structure
   may allow per-option decomposition if the cross-option coupling is weak.

## What would count as success

- **Recovery**: the SDRE candidate produces quotes close to BBG numerical
  in the standard regime. "Close" means |CE_sdre - CE_bbg| / sd_post < 2.

- **Robustness**: the candidate performs comparably under parameter
  perturbations where the BBG solver was not retrained.

- **Interpretability**: the local quadratic structure reveals which state
  features drive the quote correction, connecting back to the repo's
  Koopman/signature story.

## What would be scientifically interesting

- **In-regime recovery**: confirms the SDRE methodology can approximate
  a known HJB solution from data alone.

- **Out-of-regime robustness**: when Heston parameters change, the
  SDRE candidate (trained on diverse exploration) may adapt better
  than a BBG solver locked to the wrong model.

- **Multi-option structure**: the BBG benchmark's key feature is portfolio
  vega coupling across options. If the SDRE controller captures this
  coupling from data, that is a genuine methodological contribution.

## Not yet decided

- Exact state-feature representation for the 20-option book
- Whether to learn in the full 40D action space or reduce to (V^pi, skew) first
- Training data collection strategy (uniform exploration vs policy-guided)
- Whether the value-gradient backward recursion from the stylized env
  carries over, or a different approach is needed for the richer book

These will be decided in the implementation spec.
