"""SDRE recovery experiment scaffold against the BBG benchmark.

When the SDRE recovery controller is implemented, this script will:
1. Load the BBG benchmark env with paper-default config
2. Run benchmark controllers (risk-neutral, BBG numerical)
3. Run the SDRE recovery candidate
4. Report paired Bayesian posteriors

NOT YET RUNNABLE. The candidate controller slot raises NotImplementedError.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from applications.option_mm_bbg.spec import BBGBenchmarkConfig
from applications.option_mm_bbg.env import OptionBookMarketMakingEnv
from applications.option_mm_bbg.solver import (
    solve_bbg_value_function,
    make_bbg_numerical_controller,
    make_bbg_risk_neutral_controller,
)
from applications.option_mm_bbg.sdre_recovery import (
    SDRERecoveryConfig,
    make_sdre_recovery_controller,
)


def main() -> int:
    config = BBGBenchmarkConfig.paper_default()

    print("=" * 60)
    print("  BBG SDRE Recovery Experiment (scaffold)")
    print("=" * 60)
    print(f"  Options: {config.book.n_options}")
    print(f"  Gamma: {config.control.gamma}")
    print()

    # Benchmark controllers
    print("  Benchmark controllers: available")
    rn_ctrl = make_bbg_risk_neutral_controller(config)
    print("    risk_neutral: OK")

    # Solve BBG HJB for numerical controller
    print("  Solving BBG HJB...")
    t_grid, nu_grid, vpi_grid, values = solve_bbg_value_function(
        config, n_nu=15, n_vpi=30, n_time=60,
    )
    bbg_ctrl = make_bbg_numerical_controller(config, values, t_grid, nu_grid, vpi_grid)
    print("    bbg_numerical: OK")

    # Candidate controller (not yet implemented)
    print()
    print("  Candidate controller: NOT YET IMPLEMENTED")
    print("  The SDRE recovery controller will be added in the next phase.")
    print()
    print("  When implemented, this experiment will compare:")
    print("    1. risk_neutral_optimal")
    print("    2. bbg_numerical")
    print("    3. sdre_recovery (the candidate)")
    print()
    print("  Primary contrast: sdre_recovery - bbg_numerical")
    print("  Success criterion: gap reduction, not necessarily beating BBG")

    return 0


if __name__ == "__main__":
    sys.exit(main())
