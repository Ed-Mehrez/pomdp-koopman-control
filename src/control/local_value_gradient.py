r"""
Generic finite-horizon local value-gradient controller with SCALAR local
control `u` and a compact control-quadratic lifted dynamics model.

Mathematical form (see docs/signature_based_filtering_control.md Prop 3.3,
Prop 4.1, Prop 4.3, Prop 5.1):

    psi_t        :=  psi(z_t)     -- lifted state, dim m
    E[psi_{t+1} | psi_t, u]  ~  (A_0^{(t)} + u * A_1^{(t)} + u**2 * A_2^{(t)}) psi_t
    V_T(psi)     ~  beta_T^T psi
    V_t(psi)     ~  beta_t^T psi
    Q_t(psi, u)  =  ell_t(psi, u)  +  beta_{t+1}^T (A_0 + u A_1 + u**2 A_2) psi
                 =  alpha_0(psi) + alpha_1(psi) u + alpha_2(psi) u**2
    u*(psi_t)    =  -alpha_1 / (2 alpha_2)  (if concave)

Backward value recursion:

    beta_T  :=  fit(psi_T, terminal_target(W_T))
    beta_t  :=  fit(psi_t,  ell_t(observed u_t) + beta_{t+1}^T psi_{t+1})

Key design choices
------------------
1. **Scalar u only.**  The multiplicative overlay coordinate keeps u in a
   compact trust region; Prop 5.1 of the extrapolation doc guarantees that
   the RBF / polynomial extrapolation breakdown of Section 3 is avoided.

2. **Transfer form** on per-step dt, not generator form.  For short dt the
   signal-to-noise of a single generator step is weak; transfer-form over
   the observed dt aggregates cleanly and is directly identifiable.

3. **Proper action-dependent continuation.**  The Hamiltonian includes the
   u-dependence of psi_{t+1} through the fitted (A_0, A_1, A_2).  This is
   where `src/applications/option_mm/local_value_bilinear.py` short-cuts;
   we do NOT short-cut here.

4. **Response-agnostic.**  The generic module knows nothing about finance;
   the domain adapter supplies psi(.), terminal_target(.), and ell(.).

Currently used by:
    - finance/experiments/merton_value_gradient.py   (Heston)

Expect refactor when a second non-adapter caller appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


# ==========================================================================
# Types and protocols
# ==========================================================================


class LiftedStateMap(Protocol):
    r"""psi: state -> (m,) lifted features.  Must be continuous in state."""
    dim: int
    name: str

    def __call__(self, state) -> np.ndarray: ...


class TerminalTarget(Protocol):
    r"""terminal_target(final_state, psi_final) -> scalar target value.

    Examples the adapter can supply:
        log W_T  (Kelly residual terminal target; numerically stable)
        W_T^(1-gamma)/(1-gamma)  (exact CRRA; numerically delicate)
    """
    name: str

    def __call__(self, state_final, psi_final: np.ndarray) -> float: ...


class StageCost(Protocol):
    r"""ell_t(psi_t, u, state_t, transition_observation) -> scalar.

    The transition observation is supplied so stage cost can include
    realized dW or similar if the adapter wants to.  Most "Kelly residual"
    targets have `ell_t == 0` (only terminal matters).
    """

    def __call__(
        self,
        psi_t: np.ndarray,
        u: float,
        state_t,
        transition_observation,
    ) -> float: ...


# ==========================================================================
# Training data container (generic)
# ==========================================================================


@dataclass
class EpisodeRecord:
    r"""One episode's paired-noise trajectory under an exploration action u_t.

    psis:         (T+1, m)  lifted states at each time
    us:           (T,)      exploration actions (scalar)
    states:       length T+1 list of adapter-defined state objects (opaque)
    transitions:  length T  list of per-step observations for stage cost
    terminal_Wt:  float (adapter-supplied final-time value for terminal_target)

    ``transitions[t]`` is whatever scalar/dict the adapter wants the stage
    cost to see for that step.
    """
    psis: np.ndarray
    us: np.ndarray
    states: List
    transitions: List
    terminal_state: object
    psi_final: np.ndarray


# ==========================================================================
# Local control-quadratic transition model (time-indexed, per step t)
# ==========================================================================


@dataclass
class CQTransitionFit:
    r"""Least-squares fit of
        psi_{t+1}  ~  (A_0 + u A_1 + u**2 A_2) psi_t  +  noise.

    Training data: stacked (psi_t, u_t, psi_{t+1}) at time step t.

    The design matrix has columns [psi_t, u*psi_t, u^2*psi_t] (shape m x 3)
    per row.  Ridge-regularized solve produces stacked coefficients
    [A_0 | A_1 | A_2] of shape (m, 3m).  Storage: (3m, m) after reshape.
    """
    A0: np.ndarray  # (m, m)
    A1: np.ndarray  # (m, m)
    A2: np.ndarray  # (m, m)
    n_samples: int
    r2: float


def fit_cq_transition_step(
    psi_t: np.ndarray,    # (N, m)
    u_t: np.ndarray,      # (N,)
    psi_tp1: np.ndarray,  # (N, m)
    ridge: float = 1e-4,
) -> CQTransitionFit:
    r"""Fit time-step transition with stacked ridge regression.

    For each row i we regress psi_{t+1,i,:} on the design
        x_i  =  [ psi_t,i,: ,  u_i * psi_t,i,: ,  u_i**2 * psi_t,i,: ]  in R^(3m).
    Solution W in R^(3m x m).  Split into A_0, A_1, A_2 each (m, m).
    """
    N, m = psi_t.shape
    if u_t.shape != (N,):
        raise ValueError("u_t shape mismatch")
    if psi_tp1.shape != (N, m):
        raise ValueError("psi_tp1 shape mismatch")
    X = np.hstack([psi_t, u_t[:, None] * psi_t, (u_t[:, None] ** 2) * psi_t])  # (N, 3m)
    # Solve W in R^(3m x m): X @ W ~ Y
    Y = psi_tp1
    G = X.T @ X + ridge * np.eye(3 * m)
    W = np.linalg.solve(G, X.T @ Y)  # (3m, m)
    A0 = W[:m].T        # (m, m)
    A1 = W[m:2*m].T     # (m, m)
    A2 = W[2*m:].T      # (m, m)
    # R^2 of the joint regression (pooled across m outputs)
    Y_pred = X @ W
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-18)
    return CQTransitionFit(A0=A0, A1=A1, A2=A2, n_samples=int(N), r2=float(r2))


# ==========================================================================
# Terminal and intermediate value-function fits
# ==========================================================================


def fit_value_function_linear(
    psi: np.ndarray,     # (N, m)
    target: np.ndarray,  # (N,)
    ridge: float = 1e-4,
) -> np.ndarray:
    r"""Fit V(psi) = beta^T psi by ridge regression."""
    N, m = psi.shape
    G = psi.T @ psi + ridge * np.eye(m)
    beta = np.linalg.solve(G, psi.T @ target)
    return beta


# ==========================================================================
# Full backward value iteration
# ==========================================================================


@dataclass
class ValueGradientModel:
    r"""Finite-horizon local value-gradient model.

    Components:
      * betas[t]:   (m,) value-function coefficients at time t, V_t(psi) = beta^T psi
      * transitions[t]:  CQTransitionFit for step t
      * terminal_target_info: provenance string for debugging
      * stage_cost_info:     provenance string for debugging
    """
    betas: List[np.ndarray]
    transitions: List[CQTransitionFit]
    terminal_target_name: str
    stage_cost_name: str
    lift_dim: int
    horizon: int


def backward_value_iteration(
    episodes: List[EpisodeRecord],
    terminal_target: TerminalTarget,
    stage_cost: StageCost,
    ridge_transition: float = 1e-4,
    ridge_value: float = 1e-4,
) -> ValueGradientModel:
    r"""Run backward recursion over a batch of paired-noise episodes.

    1. Terminal:  beta_T fitted on (psi_final, terminal_target(state_final, psi_final)).
    2. For t = T-1 ... 0:
        (a) fit CQ transition at step t from (psi_t, u_t, psi_{t+1}) across episodes.
        (b) form training targets for V_t using the OBSERVED transition:
               V_t_target_i  =  ell_t(psi_{t,i}, u_i, state_t,i, trans_i)
                              + beta_{t+1}^T psi_{t+1,i}
        (c) fit beta_t on (psi_{t,i}, V_t_target_i).

    Notes on correctness
    --------------------
    * Step (b) uses the OBSERVED u_i and OBSERVED psi_{t+1,i}.  This is
      standard fitted-value-iteration under a stationary exploration policy.
      The model "knows about" u-dependence of continuation through the
      CQ transition fit (used at decision time, NOT at beta-fitting time).
    * The action-dependent continuation IS assembled at decision time by
      `hamiltonian_at(psi_t, t)` below, using the fitted (A_0, A_1, A_2).
    """
    T = int(episodes[0].us.size)
    for ep in episodes:
        if ep.us.size != T:
            raise ValueError("episodes must share horizon length")
    m = episodes[0].psis.shape[1]

    # ---- Terminal: beta_T from terminal_target ----
    psi_final = np.array([ep.psi_final for ep in episodes])  # (E, m)
    tterm = np.array(
        [terminal_target(ep.terminal_state, ep.psi_final) for ep in episodes]
    )  # (E,)
    beta_T = fit_value_function_linear(psi_final, tterm, ridge=ridge_value)

    betas: List[Optional[np.ndarray]] = [None] * (T + 1)
    betas[T] = beta_T
    transitions: List[Optional[CQTransitionFit]] = [None] * T

    # ---- Backward: t = T-1, ..., 0 ----
    for t in range(T - 1, -1, -1):
        psi_t = np.array([ep.psis[t] for ep in episodes])        # (E, m)
        psi_tp1 = np.array([ep.psis[t + 1] for ep in episodes])  # (E, m)
        u_t = np.array([ep.us[t] for ep in episodes])            # (E,)
        transitions[t] = fit_cq_transition_step(
            psi_t, u_t, psi_tp1, ridge=ridge_transition,
        )
        # Stage cost per episode
        ell = np.array([
            stage_cost(ep.psis[t], float(ep.us[t]), ep.states[t], ep.transitions[t])
            for ep in episodes
        ])  # (E,)
        continuation = psi_tp1 @ betas[t + 1]
        V_t_target = ell + continuation
        betas[t] = fit_value_function_linear(psi_t, V_t_target, ridge=ridge_value)

    return ValueGradientModel(
        betas=[b for b in betas if b is not None],
        transitions=[tr for tr in transitions if tr is not None],
        terminal_target_name=terminal_target.name,
        stage_cost_name=getattr(stage_cost, "name", "stage_cost"),
        lift_dim=m,
        horizon=T,
    )


# ==========================================================================
# Hamiltonian assembly and action extraction (THE key routine)
# ==========================================================================


def hamiltonian_coefficients(
    model: ValueGradientModel,
    psi_t: np.ndarray,
    t: int,
    stage_cost_quadratic: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    r"""Coefficients (alpha0, alpha1, alpha2) of Q_t(u) = alpha0 + alpha1 u + alpha2 u^2.

    Q_t(psi_t, u)
        =  stage_cost(psi_t, u)  +  beta_{t+1}^T (A_0 + u A_1 + u**2 A_2) psi_t
        =  [ell_0(psi) + beta'A_0 psi]   +   [ell_1(psi) + beta'A_1 psi] u
           + [ell_2(psi) + beta'A_2 psi] u**2.

    stage_cost_quadratic, if provided, is (ell_0, ell_1, ell_2) at this
    state; otherwise stage cost is treated as zero.  Most Kelly-residual
    adapters use terminal-only (ell = 0 everywhere except T).
    """
    if t < 0 or t >= model.horizon:
        raise IndexError(f"t={t} out of [0, {model.horizon})")
    beta_next = model.betas[t + 1]
    tr = model.transitions[t]
    base0 = float(beta_next @ (tr.A0 @ psi_t))
    base1 = float(beta_next @ (tr.A1 @ psi_t))
    base2 = float(beta_next @ (tr.A2 @ psi_t))
    if stage_cost_quadratic is not None:
        ell0, ell1, ell2 = stage_cost_quadratic
        return base0 + ell0, base1 + ell1, base2 + ell2
    return base0, base1, base2


@dataclass(frozen=True)
class ActionRecommendation:
    u_star: float
    alpha0: float
    alpha1: float
    alpha2: float
    concave: bool
    clipped: bool
    rationale: str


def extract_action(
    model: ValueGradientModel,
    psi_t: np.ndarray,
    t: int,
    u_max: float,
    stage_cost_quadratic: Optional[Tuple[float, float, float]] = None,
    concavity_threshold: float = 1e-12,
) -> ActionRecommendation:
    r"""Closed-form quadratic optimizer on [-u_max, u_max].

    If alpha_2 < -concavity_threshold (concave max): u_star = -alpha_1/(2 alpha_2),
    clipped to trust region.  If not concave, return u_star = 0 with
    rationale 'not concave: abstain'.  (The adapter may override with a
    grid search if desired.)
    """
    a0, a1, a2 = hamiltonian_coefficients(model, psi_t, t, stage_cost_quadratic)
    if a2 < -concavity_threshold:
        u_unclipped = -a1 / (2.0 * a2)
        clipped = abs(u_unclipped) > u_max
        u_star = float(np.clip(u_unclipped, -u_max, u_max))
        return ActionRecommendation(
            u_star=u_star,
            alpha0=a0, alpha1=a1, alpha2=a2,
            concave=True, clipped=clipped,
            rationale=(
                f"concave: a2={a2:+.3e} < 0; u_unclipped={u_unclipped:+.4f}, "
                f"clipped={clipped}"
            ),
        )
    return ActionRecommendation(
        u_star=0.0,
        alpha0=a0, alpha1=a1, alpha2=a2,
        concave=False, clipped=False,
        rationale=f"not concave (a2={a2:+.3e} >= 0): abstain at u=0",
    )


# ==========================================================================
# Convenience: sanity-check the backward fit by evaluating on training data
# ==========================================================================


def training_value_r2(
    model: ValueGradientModel,
    episodes: List[EpisodeRecord],
    terminal_target: TerminalTarget,
    stage_cost: StageCost,
) -> Dict[int, float]:
    r"""Per-time-step R^2 of beta_t^T psi_t against the observed V_t target.

    Useful as a basic diagnostic: if R^2 collapses at some t, the lifted
    state is too impoverished OR the transition fit is bad there.
    """
    T = model.horizon
    out: Dict[int, float] = {}
    # Terminal
    psi_final = np.array([ep.psi_final for ep in episodes])
    tterm = np.array(
        [terminal_target(ep.terminal_state, ep.psi_final) for ep in episodes]
    )
    pred_T = psi_final @ model.betas[T]
    ss_res = float(np.sum((tterm - pred_T) ** 2))
    ss_tot = float(np.sum((tterm - tterm.mean()) ** 2))
    out[T] = 1.0 - ss_res / max(ss_tot, 1e-18)
    # Intermediate steps
    for t in range(T - 1, -1, -1):
        psi_t = np.array([ep.psis[t] for ep in episodes])
        psi_tp1 = np.array([ep.psis[t + 1] for ep in episodes])
        ell = np.array([
            stage_cost(ep.psis[t], float(ep.us[t]), ep.states[t], ep.transitions[t])
            for ep in episodes
        ])
        V_target = ell + psi_tp1 @ model.betas[t + 1]
        pred = psi_t @ model.betas[t]
        ss_res = float(np.sum((V_target - pred) ** 2))
        ss_tot = float(np.sum((V_target - V_target.mean()) ** 2))
        out[t] = 1.0 - ss_res / max(ss_tot, 1e-18)
    return out
