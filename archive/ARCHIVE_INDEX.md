# Archive Index

This folder contains exploratory experiments organized by approach.
The validated work is in `kronic_pomdp/experiments/`.

## Archive Categories

### 01_eigenfunction_hjb/
Eigenfunction-based HJB approaches - learning value function eigenfunctions.
- eigenfunction_hjb.py, eigenfunction_value_kronic.py
- cev_eigenfunction_hedging.py, cev_eigenfunction_hedging_low_beta.py
- general_utility_eigenfunction.py, eigenfunction_sdre_general.py
- eigenfunction_bandwidth_optimization.py, eigenfunction_conditional_expectation.py
- control_affine_hjb.py, kernel_hjb_control.py, general_koopman_hjb.py
- iterative_eigenfunction_refinement.py, iterative_eigenfunction_refinement_v2.py

### 02_bilinear_kkt/
Bilinear/KKT formulation for Merton problem.
**WARNING**: Contains formula bug (hedging demand 625x inflated due to /theta error).
See MEMORY.md for details.
- merton_kronic_bilinear.py (BUGGY - do not use formulas)
- bilinear_eigenfunction_lp.py, bilinear_methods_comparison.py
- kkt_value_portrait.py, kkt_stress_test.py, backtest_kkt_outcomes.py
- merton_kkt_benchmark_master.py

### 03_cdc_generator/
Carré du Champ generator extraction - proven to work for ergodic diffusions.
Level 2 of graduated_sanity_checks.py validates this approach.
- carre_du_champ_extraction.py
- general_sde_eigenfunctions.py
- kronic_hybrid_cdc_eigen.py
- cev_logsig_eigenfunctions.py

### 04_signature_kronic/
Signature-based KRONIC control approaches.
- merton_kronic_signatures.py, merton_kronic_signatures_bounded.py
- merton_kronic_simulation.py, merton_kronic_online.py
- signature_kernel_edmd.py, nystrom_signature_edmd.py
- signature_ablation.py, signature_ablation_v2.py, signature_ablation_final.py
- signature_chen_normalized.py, signature_debug.py
- online_kronic_pipeline.py, kronic_option_a_eigenfunction.py

### 05_bates_vol/
Bates model volatility estimation with jumps.
- bates_volatility.py

### 06_market_microstructure/
Kyle model, bluffing, spoofing detection.
- kyle_signature_mm.py (if exists)
- bluffing_regime_change.py
- money_pump.py

### 07_misc_experiments/
Various other experiments.
- cev_bubble_detection.py, cev_myopic_vs_full_hjb.py
- cev_sigkkf_comparison.py, cev_sigkkf_debug.py
- high_dim_factor_portfolio.py, cvar_constrained_portfolio.py
- hedging_demand_approaches.py, two_for_one_eigenfunction.py
- fsde_inspired_estimator.py, edmd_eigenvalue_validation.py
- merton_kgedmd_utility.py, merton_koopman_lqr.py
- local_utility_sdre.py, sdre_regret_bounds.py
- three_prior_levels.py, two_phase_kronic.py, two_phase_multi_utility.py
- unified_portfolio_validation.py, observable_state_validation.py
- merton_validation.py, verify_heston_baseline.py

## Validated Work (NOT archived)

The following are the validated experiments with proper methodology:

- `kronic_pomdp/experiments/graduated_sanity_checks.py` - All 5 levels pass
- `kronic_pomdp/experiments/honest_benchmark.py` - Train/test, error bars
- `kronic_pomdp/experiments/level5_hedging_demand.py` - Hedging demand estimation
- `kronic_pomdp/experiments/fair_comparison.py` - Train/test seed pattern
- `kronic_pomdp/experiments/stage1_documented_approach.py` - RecSig-RLS filter

## Common Issues in Archived Experiments

See MEMORY.md for details:
1. No train/test split - all R^2 metrics are in-sample
2. Hardcoded true mu (risk premium) - the hardest parameter
3. Answer-in-the-basis for utility features
4. Single seed, no error bars
5. CdC extraction fails for non-ergodic processes (GBM)
