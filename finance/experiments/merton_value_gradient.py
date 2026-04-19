r"""
Heston myopic-reference local value-gradient controller with SCALAR
multiplicative overlay and proper action-dependent continuation.

This revision applies a PROTOCOL-AUDIT correction to the filter
observation pipeline.  Prior to the fix:

  * the Heston env step produces the CONTROLLED wealth return
        d_logW  =  (r + pi*(mu-r) - 0.5*pi^2*V) dt  +  pi*sqrt(V)*dB,
  * filter lanes were consuming d_logW and treating (d_logW)^2/dt as a
    proxy for V.  Under the multiplicative overlay pi = (1+u)*pi_ref,
    (d_logW)^2/dt  ~=  pi^2 * V , not V, so V_hat was scaled by pi^2
    (action-dependent bias).

Fix: `_step_state(...)` also returns the UNDERLYING asset log-return
`dr_S = (mu - 0.5*V)*dt + sqrt(V)*dB` (reconstructed from the same z1
and pre-step V that env.step_explicit used).  Filter lanes now consume
`dr_S`.  Wealth return is still the controller's stage cost.

Alignment fix: per-step V_hat histories now come in two variants,
`V_hat_pre` (filter posterior BEFORE assimilating r_t -- the object
the controller sees at decision time) and `V_hat_post` (after).  The
filter-quality audit plot uses the POST-update pair and a
median-range test seed for clarity.

This file in the current commit ships with Oracle + EWMA lanes only;
the signature / Kalman lane infrastructure is added in a follow-up
`feat(sskf)` commit.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))

from merton_kronic_bilinear import HestonMertonEnv

from src.control.local_value_gradient import (
    ActionRecommendation,
    EpisodeRecord,
    ValueGradientModel,
    backward_value_iteration,
    extract_action,
    training_value_r2,
)


# ==========================================================================
# Risk-sensitive evaluation target (adapter-level)
# ==========================================================================


RiskTargetName = Literal["mean_only", "mean_variance", "entropic"]


@dataclass(frozen=True)
class RiskSensitiveTarget:
    name: RiskTargetName
    gamma: float = 1.0

    def evaluate(self, y_samples: np.ndarray) -> float:
        y = np.asarray(y_samples, dtype=float).flatten()
        if self.name == "mean_only":
            return float(np.mean(y))
        if self.name == "mean_variance":
            var = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
            return float(np.mean(y) - 0.5 * (self.gamma - 1.0) * var)
        if self.name == "entropic":
            if self.gamma <= 0.0:
                raise ValueError("entropic requires gamma > 0")
            z = -self.gamma * y
            m = float(np.max(z))
            return float(-(1.0 / self.gamma) * (m + np.log(np.mean(np.exp(z - m)))))
        raise ValueError(f"unknown risk target: {self.name}")


# ==========================================================================
# Config and state
# ==========================================================================


@dataclass(frozen=True)
class VGConfig:
    rho: float = -0.7
    gamma: float = 3.0
    V_floor_for_pi_ref: float = 0.005
    T_steps: int = 60
    dt: float = 1.0 / 252.0
    u_max: float = 0.3
    n_train: int = 250
    n_test: int = 120
    V0_low: float = 0.02
    V0_high: float = 0.08
    ewma_halflife_days: float = 21.0
    ridge_transition: float = 1e-4
    ridge_value: float = 1e-4

    @property
    def ewma_lam(self) -> float:
        return float(np.log(2.0) / max(self.ewma_halflife_days, 1e-3))


@dataclass
class HVGState:
    logW: float
    V: float
    t: int
    logW0: float
    ewma_r: float
    ewma_r2: float

    def clone(self) -> "HVGState":
        return HVGState(
            logW=self.logW, V=self.V, t=self.t, logW0=self.logW0,
            ewma_r=self.ewma_r, ewma_r2=self.ewma_r2,
        )


# ==========================================================================
# Lifted state maps (controller side)
# ==========================================================================


class PsiA:
    r"""Minimal lift [1, log(V/theta), 1/V, tau].  Accepts a V_override so
    lanes with different V estimators can plug in their V_hat without
    changing env state itself."""
    name = "A_minimal"

    def __init__(self, env: HestonMertonEnv, T_steps: int):
        self.env = env
        self.T_steps = int(T_steps)
        self.theta = float(env.theta)

    @property
    def dim(self) -> int:
        return 4

    def __call__(self, state: HVGState, V_override: Optional[float] = None) -> np.ndarray:
        V_use = state.V if V_override is None else V_override
        V = max(V_use, 1e-8)
        tau_frac = (self.T_steps - state.t) / self.T_steps
        return np.array([
            1.0,
            float(np.log(V / self.theta)),
            1.0 / V,
            tau_frac,
        ])


class PsiB:
    name = "B_path_aware"

    def __init__(self, env: HestonMertonEnv, T_steps: int):
        self.A = PsiA(env, T_steps)

    @property
    def dim(self) -> int:
        return 6

    def __call__(self, state: HVGState, V_override: Optional[float] = None) -> np.ndarray:
        base = self.A(state, V_override=V_override)
        return np.concatenate([base, [state.ewma_r, state.ewma_r2]])


# ==========================================================================
# Targets
# ==========================================================================


class TerminalZero:
    name = "zero_terminal"

    def __call__(self, state_final: HVGState, psi_final: np.ndarray) -> float:
        return 0.0


class ObservedLogReturnStageCost:
    name = "observed_log_return"

    def __call__(self, psi_t, u, state_t, transition_observation) -> float:
        return float(transition_observation["r_t"])


def heston_stage_cost_u_quadratic(
    env: HestonMertonEnv, state: HVGState, cfg: VGConfig,
) -> Tuple[float, float, float]:
    r"""Analytical u-quadratic of E[r_t | state, u] at decision time."""
    V = max(state.V, cfg.V_floor_for_pi_ref)
    a = float(env.mu - env.r)
    gamma = float(env.gamma)
    pi_ref = a / (gamma * V)
    pi_ref_sq = pi_ref * pi_ref
    e_0 = (env.r + pi_ref * a - 0.5 * pi_ref_sq * V) * cfg.dt
    e_1 = (pi_ref * a - pi_ref_sq * V) * cfg.dt
    e_2 = -0.5 * pi_ref_sq * V * cfg.dt
    return float(e_0), float(e_1), float(e_2)


# ==========================================================================
# V estimator lanes (Oracle + EWMA only in this commit)
# ==========================================================================


class OracleVEstimator:
    name = "oracle"

    def __init__(self):
        self._V = 0.04

    def reset(self, V0: float):
        self._V = float(V0)

    def set_true_V(self, V_true: float):
        self._V = float(V_true)

    def observe(self, r_t: float, dt: float):
        pass

    def V_hat(self) -> float:
        return float(self._V)

    def V_interval(self) -> Tuple[float, float]:
        return (float("nan"), float("nan"))


class EWMAVEstimator:
    name = "ewma"

    def __init__(self, halflife_days: float = 21.0, dt: float = 1.0 / 252.0):
        self.halflife_days = float(halflife_days)
        self.dt = float(dt)
        self._alpha = 1.0 - float(np.exp(-np.log(2.0) / max(halflife_days, 1e-3)))
        self._V = 0.04

    def reset(self, V0: float):
        self._V = float(V0)

    def observe(self, r_t: float, dt: float):
        y = float(r_t) ** 2 / float(dt)
        self._V = (1.0 - self._alpha) * self._V + self._alpha * y
        self._V = max(self._V, 1e-8)

    def V_hat(self) -> float:
        return float(self._V)

    def V_interval(self) -> Tuple[float, float]:
        return (float("nan"), float("nan"))


# ==========================================================================
# Rollout helpers  (observation-pipeline FIX lives in _step_state)
# ==========================================================================


def _paired_noise(rho: float, T: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    zA = rng.standard_normal(T)
    zB = rng.standard_normal(T)
    z1 = zA
    z2 = rho * zA + np.sqrt(max(1.0 - rho * rho, 0.0)) * zB
    return np.column_stack([z1, z2])


def _pi_ref(env: HestonMertonEnv, V: float, V_floor: float) -> float:
    return float(env.merton_optimal(max(V, V_floor)))


def _step_state(
    env: HestonMertonEnv,
    cfg: VGConfig,
    state: HVGState,
    u: float,
    z1: float,
    z2: float,
) -> Tuple[HVGState, Dict[str, float]]:
    r"""Advance one step under multiplicative overlay; return (new_state, obs).

    Returned observation dict
    -------------------------
    r_t    : d_logW = (r + pi*(mu-r) - 0.5*pi^2*V)*dt + pi*sqrt(V)*dB
             Controlled wealth log-return.  Brownian term scales with
             pi*sqrt(V), so (r_t)^2/dt ~ pi^2 * V.  APPROPRIATE as the
             controller's stage cost; INAPPROPRIATE as a V estimator
             input when pi != 1.

    dr_S   : d_logS = (mu - 0.5*V)*dt + sqrt(V)*dB   (underlying asset
             log-return; pi-free).  Brownian term has E[(dr_S)^2/dt|V] = V.
             Use this for variance-filter observations.

    dr_S is reconstructed from the same z1 and pre-step V that
    env.step_explicit used, so it is realization-consistent with the
    rollout's wealth return.
    """
    pi_ref = _pi_ref(env, state.V, cfg.V_floor_for_pi_ref)
    pi_t = (1.0 + float(u)) * pi_ref
    logW_new, V_new = env.step_explicit(
        state.logW, state.V, pi_t, z1, z2, cfg.dt,
    )
    r_t_wealth = logW_new - state.logW
    V_pre = max(state.V, 1e-8)
    sqrt_V = float(np.sqrt(V_pre))
    dr_S = (
        (env.mu - 0.5 * V_pre) * cfg.dt
        + sqrt_V * float(np.sqrt(cfg.dt)) * float(z1)
    )
    rho_e = float(np.exp(-cfg.ewma_lam * cfg.dt))
    new_state = HVGState(
        logW=logW_new,
        V=V_new,
        t=state.t + 1,
        logW0=state.logW0,
        ewma_r=rho_e * state.ewma_r + (1.0 - rho_e) * r_t_wealth,
        ewma_r2=rho_e * state.ewma_r2 + (1.0 - rho_e) * (r_t_wealth * r_t_wealth),
    )
    return new_state, {
        "r_t": r_t_wealth,
        "dr_S": dr_S,
        "pi_t": pi_t,
        "pi_ref": pi_ref,
    }


# ==========================================================================
# Training data collection
# ==========================================================================


def collect_training_episodes(
    env: HestonMertonEnv,
    cfg: VGConfig,
    psi_map: object,
    n_episodes: int,
    base_seed: int,
    u_sampler: Optional[Callable[[np.random.RandomState], float]] = None,
    v_estimator_factory: Optional[Callable[[], object]] = None,
) -> List[EpisodeRecord]:
    if u_sampler is None:
        def u_sampler(rng):
            return float(rng.uniform(-cfg.u_max, cfg.u_max))
    episodes: List[EpisodeRecord] = []
    v0_rng = np.random.RandomState(base_seed)
    for k in range(n_episodes):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        noise = _paired_noise(cfg.rho, cfg.T_steps, base_seed + 1 + k)
        u_rng = np.random.RandomState(base_seed + 10_000 + k)

        state = HVGState(
            logW=0.0, V=V0, t=0, logW0=0.0,
            ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
        )
        v_est = v_estimator_factory() if v_estimator_factory is not None else None
        if v_est is not None:
            v_est.reset(V0)
        psis = np.zeros((cfg.T_steps + 1, psi_map.dim))
        us = np.zeros(cfg.T_steps)
        states_list: List[HVGState] = [state.clone()]
        trans_list: List[Dict[str, float]] = []
        if v_est is None:
            psis[0] = psi_map(state)
        else:
            if isinstance(v_est, OracleVEstimator):
                v_est.set_true_V(state.V)
            psis[0] = psi_map(state, V_override=v_est.V_hat())
        for t in range(cfg.T_steps):
            u_t = u_sampler(u_rng)
            us[t] = u_t
            state, obs = _step_state(env, cfg, state, u_t, float(noise[t, 0]), float(noise[t, 1]))
            # Filter observation: UNDERLYING asset return (fix for pi^2*V bias).
            if v_est is not None:
                if isinstance(v_est, OracleVEstimator):
                    v_est.set_true_V(state.V)
                else:
                    v_est.observe(obs["dr_S"], cfg.dt)
            if v_est is None:
                psis[t + 1] = psi_map(state)
            else:
                psis[t + 1] = psi_map(state, V_override=v_est.V_hat())
            states_list.append(state.clone())
            trans_list.append(obs)
        episodes.append(EpisodeRecord(
            psis=psis,
            us=us,
            states=states_list[:-1],
            transitions=trans_list,
            terminal_state=states_list[-1],
            psi_final=psis[-1].copy(),
        ))
    return episodes


# ==========================================================================
# Held-out evaluation  (pre/post V_hat histories are both recorded)
# ==========================================================================


@dataclass
class HeldOutResult:
    name: str
    delta_logW: np.ndarray
    u_history: np.ndarray
    abstention_flags: np.ndarray
    per_seed_V_at_decision: np.ndarray
    model_name: str
    # Pre-update: controller-seen forecast; post-update: matching-epoch audit pair.
    V_hat_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V_true_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V_hat_post_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V_true_post_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def _myopic_rollout(env: HestonMertonEnv, cfg: VGConfig, V0: float, noise: np.ndarray) -> float:
    state = HVGState(
        logW=0.0, V=V0, t=0, logW0=0.0,
        ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
    )
    for t in range(cfg.T_steps):
        state, _ = _step_state(env, cfg, state, 0.0, float(noise[t, 0]), float(noise[t, 1]))
    return float(state.logW)


def _value_gradient_rollout(
    env: HestonMertonEnv, cfg: VGConfig, V0: float, noise: np.ndarray,
    psi_map: object, model: ValueGradientModel,
    v_estimator: Optional[object] = None,
) -> Tuple[float, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Rollout with pre/post V_hat histories.

    Per-step sequence:
      1. V_hat_pre_t = v_estimator.V_hat()  (posterior from r_0..r_{t-1})
         Used by the controller at decision time.
      2. Extract u_t* from local Hamiltonian using V_hat_pre.
      3. Step env -> obs has r_t (wealth return) AND dr_S (asset return).
      4. V_true_post_t = state.V.
      5. v_estimator.observe(dr_S, dt)  (fix: use underlying asset return).
      6. V_hat_post_t = v_estimator.V_hat().
    """
    state = HVGState(
        logW=0.0, V=V0, t=0, logW0=0.0,
        ewma_r=0.0, ewma_r2=cfg.V_floor_for_pi_ref * cfg.dt,
    )
    if v_estimator is not None:
        v_estimator.reset(V0)
        if isinstance(v_estimator, OracleVEstimator):
            v_estimator.set_true_V(state.V)
    u_history = np.zeros(cfg.T_steps)
    abstained = np.zeros(cfg.T_steps, dtype=bool)
    V_hat_pre_history = np.zeros(cfg.T_steps)
    V_hat_post_history = np.zeros(cfg.T_steps)
    V_true_pre_history = np.zeros(cfg.T_steps)
    V_true_post_history = np.zeros(cfg.T_steps)
    for t in range(cfg.T_steps):
        V_hat_pre = state.V if v_estimator is None else v_estimator.V_hat()
        V_hat_pre_history[t] = V_hat_pre
        V_true_pre_history[t] = state.V
        psi_t = (
            psi_map(state, V_override=V_hat_pre)
            if v_estimator is not None else psi_map(state)
        )
        stage_state = state.clone()
        stage_state.V = V_hat_pre
        stage_uq = heston_stage_cost_u_quadratic(env, stage_state, cfg)
        rec = extract_action(
            model, psi_t, t, u_max=cfg.u_max,
            stage_cost_quadratic=stage_uq,
        )
        u = rec.u_star
        u_history[t] = u
        abstained[t] = (not rec.concave)
        state, obs = _step_state(env, cfg, state, u, float(noise[t, 0]), float(noise[t, 1]))
        V_true_post_history[t] = state.V
        if v_estimator is not None:
            if isinstance(v_estimator, OracleVEstimator):
                v_estimator.set_true_V(state.V)
            else:
                v_estimator.observe(obs["dr_S"], cfg.dt)
            V_hat_post_history[t] = v_estimator.V_hat()
        else:
            V_hat_post_history[t] = state.V
    return (
        float(state.logW), u_history, abstained,
        V_hat_pre_history, V_hat_post_history,
        V_true_pre_history, V_true_post_history,
    )


def evaluate_controller_paired(
    env: HestonMertonEnv,
    cfg: VGConfig,
    psi_map: object,
    model: Optional[ValueGradientModel],
    base_seed: int,
    name: str,
    v_estimator_factory: Optional[Callable[[], object]] = None,
) -> HeldOutResult:
    v0_rng = np.random.RandomState(base_seed)
    delta_logW = np.zeros(cfg.n_test)
    u_history = np.zeros((cfg.n_test, cfg.T_steps))
    abst_rate = np.zeros(cfg.n_test)
    V_decision = np.zeros(cfg.n_test)
    V_hat_hist = np.zeros((cfg.n_test, cfg.T_steps))
    V_true_hist = np.zeros((cfg.n_test, cfg.T_steps))
    V_hat_post_hist = np.zeros((cfg.n_test, cfg.T_steps))
    V_true_post_hist = np.zeros((cfg.n_test, cfg.T_steps))
    for k in range(cfg.n_test):
        V0 = float(v0_rng.uniform(cfg.V0_low, cfg.V0_high))
        V_decision[k] = V0
        noise = _paired_noise(cfg.rho, cfg.T_steps, base_seed + 1 + k)
        logW_my = _myopic_rollout(env, cfg, V0, noise)
        if model is None:
            logW_policy = logW_my
            abst_rate[k] = 1.0
        else:
            v_est = v_estimator_factory() if v_estimator_factory is not None else None
            (logW_policy, u_h, abst,
             Vh_pre, Vh_post, Vt_pre, Vt_post) = _value_gradient_rollout(
                env, cfg, V0, noise, psi_map, model, v_estimator=v_est,
            )
            u_history[k] = u_h
            abst_rate[k] = float(np.mean(abst))
            V_hat_hist[k] = Vh_pre
            V_true_hist[k] = Vt_pre
            V_hat_post_hist[k] = Vh_post
            V_true_post_hist[k] = Vt_post
        delta_logW[k] = logW_policy - logW_my
    return HeldOutResult(
        name=name,
        delta_logW=delta_logW,
        u_history=u_history,
        abstention_flags=abst_rate,
        per_seed_V_at_decision=V_decision,
        model_name=getattr(psi_map, "name", "?"),
        V_hat_history=V_hat_hist,
        V_true_history=V_true_hist,
        V_hat_post_history=V_hat_post_hist,
        V_true_post_history=V_true_post_hist,
    )


# ==========================================================================
# Reporting helpers
# ==========================================================================


def _crra_score_bootstrap(
    y: np.ndarray, gamma: float, n_boot: int = 2000,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, float]:
    if rng is None:
        rng = np.random.RandomState(0)
    y = np.asarray(y, dtype=float)
    lam = 0.5 * (gamma - 1.0)
    point = float(np.mean(y) - lam * np.var(y, ddof=1)) if y.size > 1 else float(np.mean(y))
    boot = np.zeros(n_boot)
    n = y.size
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y[idx]
        boot[b] = float(np.mean(yb) - lam * np.var(yb, ddof=1)) if n > 1 else float(np.mean(yb))
    return {
        "point": point,
        "se": float(np.std(boot, ddof=1)),
        "q05": float(np.quantile(boot, 0.05)),
        "q95": float(np.quantile(boot, 0.95)),
    }


# ==========================================================================
# Minimal runner: Oracle + EWMA lanes (two-lane baseline).
# ==========================================================================


def _run_oracle_ewma_audit():
    cfg = VGConfig()
    env = HestonMertonEnv(rho=cfg.rho, gamma=cfg.gamma)
    psi = PsiA(env, cfg.T_steps)
    terminal_target = TerminalZero()
    stage_cost = ObservedLogReturnStageCost()

    lanes: List[Tuple[str, Callable[[], object]]] = [
        ("oracle", lambda: OracleVEstimator()),
        ("ewma",   lambda: EWMAVEstimator(halflife_days=21.0, dt=cfg.dt)),
    ]

    print("=" * 96)
    print("HESTON VALUE-GRADIENT CONTROLLER  --  Oracle + EWMA protocol-audit baseline")
    print(f"  rho={cfg.rho} gamma={cfg.gamma} T={cfg.T_steps} steps  n_test={cfg.n_test}")
    print(f"  Filter lanes consume dr_S (underlying asset return), not d_logW.")
    print(f"  Post-update V_hat is used for filter-quality audit below.")
    print("=" * 96)

    results: List[Tuple[str, HeldOutResult]] = []
    for name, factory in lanes:
        print(f"-- Lane: {name}")
        eps = collect_training_episodes(
            env, cfg, psi, n_episodes=cfg.n_train, base_seed=1_000,
            v_estimator_factory=factory,
        )
        model = backward_value_iteration(
            eps, terminal_target=terminal_target, stage_cost=stage_cost,
            ridge_transition=cfg.ridge_transition, ridge_value=cfg.ridge_value,
        )
        ev = evaluate_controller_paired(
            env, cfg, psi, model=model,
            base_seed=500_000, name=f"value_grad_{name}",
            v_estimator_factory=factory,
        )
        results.append((name, ev))

    print()
    print(f"{'lane':10s} | RMSE V_hat | corr V_hat | corr V_lead")
    print("-" * 60)
    warmup = min(10, cfg.T_steps // 6)
    for name, ev in results:
        Vh = ev.V_hat_post_history if ev.V_hat_post_history.size else ev.V_hat_history
        Vt = ev.V_true_post_history if ev.V_true_post_history.size else ev.V_true_history
        vh = Vh[:, warmup:].flatten()
        vt = Vt[:, warmup:].flatten()
        rmse = float(np.sqrt(np.mean((vh - vt) ** 2)))
        corr = float(np.corrcoef(vh, vt)[0, 1])
        vh_t = Vh[:, warmup:-1].flatten()
        vt_tp1 = Vt[:, warmup + 1:].flatten()
        corr_lead = float(np.corrcoef(vh_t, vt_tp1)[0, 1])
        print(f"{name:10s} | {rmse:10.4f} | {corr:+10.4f} | {corr_lead:+10.4f}")


def main():
    _run_oracle_ewma_audit()


if __name__ == "__main__":
    main()
