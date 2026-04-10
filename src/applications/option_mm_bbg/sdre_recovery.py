"""SDRE recovery controller scaffold for the BBG benchmark.

This is the candidate controller track — separate from the BBG benchmark.
The benchmark (solved HJB) lives in solver.py. This module will contain
the data-driven local-quadratic / SDRE-style controller that attempts to
recover the benchmark without embedding it.

NOT YET IMPLEMENTED. This file contains only the interface scaffold.

Theory separation:
    - BBG benchmark = solved reduced HJB (solver.py)
    - SDRE candidate = local quadratic / value-gradient recovery from
      simulated transitions (this file)
    - No BBG formulas inside the candidate policy at action time
    - BBG appears only as an external benchmark for comparison
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import OptionBookMMAction, OptionBookMMState, OptionBookMarketMakingEnv
from .spec import BBGBenchmarkConfig


CandidateController = Callable[[OptionBookMMState, Any], OptionBookMMAction]


@dataclass(frozen=True)
class SDRERecoveryConfig:
    """Configuration for the SDRE recovery controller.

    Parameters will be filled in during implementation.
    """
    # State features
    # Action parameterization
    # Local model structure
    # Training parameters
    pass


def make_sdre_recovery_controller(
    env: OptionBookMarketMakingEnv,
    config: SDRERecoveryConfig,
    initial_state: OptionBookMMState,
) -> CandidateController:
    """Factory for the SDRE recovery controller.

    NOT YET IMPLEMENTED.

    The candidate will:
    1. Extract local state features from the book state.
    2. Estimate local controlled dynamics or value geometry.
    3. Form a local quadratic Hamiltonian.
    4. Solve for optimal quotes via local SDRE.

    No BBG solver or quote tables used at action time.
    """
    raise NotImplementedError(
        "SDRE recovery controller is not yet implemented. "
        "This is a scaffold for the next implementation phase."
    )
