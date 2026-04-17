"""Shared analytical helpers for Heston-Merton benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StationaryHestonCRRATheory:
    """Stationary CRRA benchmark at a fixed variance state.

    Under the stationary ansatz h(v) = v^p for the reduced value factor
    J(w, v) = w^(1-gamma) h(v) / (1-gamma), the HJB reduces to a quadratic
    equation for p:

      0.5 * xi^2 * p * (p - 1)
      + kappa * theta * p
      + ((1 - gamma) / (2 * gamma)) * (a + rho * xi * p)^2 = 0

    where a = mu - r. The optimal allocation at state v is then

      pi*(v) = a / (gamma v) + rho * xi * p / (gamma v).

    We select the finite branch with the smallest non-negative real exponent.
    This avoids the blow-up branch h(v) = v^p as v -> 0 and preserves
    continuity to the myopic limit.
    """

    exponent_p: float
    myopic_pi: float
    hedging_demand: float
    optimal_pi: float
    discriminant: float
    raw_roots: Tuple[complex, complex]


def canonical_state_history(
    v_eval: float,
    pi_prev: float,
    *,
    log_w: float = 0.0,
    length: int = 3,
) -> List[Tuple[float, float, float]]:
    """Canonical benchmark history shared by momentum/non-momentum controllers."""
    if length < 1:
        raise ValueError("length must be at least 1")
    row = (float(log_w), float(v_eval), float(pi_prev))
    return [row] * length


def stationary_heston_crra_theory(env, v_eval: float) -> StationaryHestonCRRATheory:
    v_safe = max(float(v_eval), 1e-8)
    gamma = float(env.gamma)
    if abs(gamma - 1.0) < 1e-12:
        myopic_pi = float(env.merton_optimal(v_safe))
        return StationaryHestonCRRATheory(
            exponent_p=0.0,
            myopic_pi=myopic_pi,
            hedging_demand=0.0,
            optimal_pi=myopic_pi,
            discriminant=np.nan,
            raw_roots=(0.0 + 0.0j, 0.0 + 0.0j),
        )

    a = float(env.mu - env.r)
    xi = float(env.xi)
    rho = float(env.rho)
    kappa = float(env.kappa)
    theta = float(env.theta)

    quad_a = 0.5 * xi**2 * (1.0 + ((1.0 - gamma) / gamma) * rho**2)
    quad_b = kappa * theta - 0.5 * xi**2 + ((1.0 - gamma) / gamma) * a * rho * xi
    quad_c = ((1.0 - gamma) / (2.0 * gamma)) * a**2

    discriminant = quad_b**2 - 4.0 * quad_a * quad_c
    roots = np.roots([quad_a, quad_b, quad_c])

    real_roots = [float(np.real(root)) for root in roots if abs(np.imag(root)) < 1e-10]
    if not real_roots:
        raise ValueError(
            f"No real stationary Heston/CRRA exponent for rho={rho:+.3f}, "
            f"gamma={gamma:.3f}, discriminant={discriminant:.6e}"
        )

    nonnegative_roots = [root for root in real_roots if root >= -1e-12]
    if nonnegative_roots:
        exponent_p = min(nonnegative_roots, key=abs)
    else:
        exponent_p = max(real_roots)

    myopic_pi = float(env.merton_optimal(v_safe))
    hedging_demand = rho * xi * exponent_p / (gamma * v_safe)
    optimal_pi = myopic_pi + hedging_demand

    return StationaryHestonCRRATheory(
        exponent_p=float(exponent_p),
        myopic_pi=myopic_pi,
        hedging_demand=float(hedging_demand),
        optimal_pi=float(optimal_pi),
        discriminant=float(discriminant),
        raw_roots=(complex(roots[0]), complex(roots[1])),
    )
