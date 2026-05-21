r"""
Closure-MKL benchmark on the two-scale Lorenz-96 system.

Observed: slow variables X_k (k = 0..K-1). The unresolved tendency
    U_k(t) = -(h c / b) sum_j Y_{j,k}(t)
must be inferred from the observable history of X. Because L96 is locally
translation-invariant, we pool samples over all k and learn a single
*stencil* closure f(local X, memory features) -> dX_k/dt.

Each candidate kernel is a different memory geometry:

    raw           : local stencil X_{k-2..k+2} only
    delay         : stencil + lagged X_k history at three time scales
    efm           : stencil + exponential moving averages of X_k
    leadlag_qv    : stencil + EFM/rolling QV of normalized X_k increments

Each is wrapped in random-landmark Nystrom RBF features. A nonnegative
kernel mixture sum_m w_m K_m is fit by grid-searched simplex weights with
validation MSE on observed dX_k/dt. Held-out scoring uses the true full
drift dX_k/dt (deterministic in L96), the unresolved tendency U_k, and an
oracle "Y-sum" kernel as upper bound.

Train and tune on observed dX/dt only. True drift / U are reserved for
scoring on a held-out path/k bank.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.science_poc.envs.two_scale_lorenz96 import (
    TwoScaleL96Config,
    simulate_two_scale_l96,
)
from experiments.science_poc.two_scale_generator_poc import FeatureScaler, _safe_corr
from src.control.kernel_residual_control import median_heuristic_length_scale


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureConfig:
    stencil_radius: int = 2          # use X_{k-2..k+2}
    delay_lags: Tuple[float, ...] = (0.05, 0.20, 0.60)  # in MTU
    efm_tau: Tuple[float, ...] = (0.05, 0.20, 0.80)
    qv_tau: Tuple[float, ...] = (0.10, 0.40)
    qv_window: Tuple[float, ...] = (0.20, 0.60)
    n_paths_train: int = 6
    n_paths_valid: int = 4
    n_paths_test: int = 6
    n_train_samples: int = 4000
    n_valid_samples: int = 2000
    n_test_samples: int = 6000
    n_landmarks: int = 160


@dataclass
class Representation:
    name: str
    Z: np.ndarray             # (N, d) flattened over (path, t, k)
    U_obs: np.ndarray         # observable U = dX/dt - resolved(X)  (training target)
    U_true: np.ndarray        # true unresolved tendency (held-out scoring)
    drift_true: np.ndarray    # full slow drift (for context corr)


@dataclass
class FeatureBlock:
    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    dim: int


@dataclass(frozen=True)
class ModelResult:
    name: str
    weights: Dict[str, float]
    ridge: float
    valid_U_mse: float
    U_corr: float                  # corr(pred_U, U_true) on held-out paths
    U_residual: float              # MSE(pred_U, U_true) / Var(U_true)  (closure R^c = 1 - U_residual)
    drift_corr: float              # corr(resolved + pred_U, drift_true)


# ---------------------------------------------------------------------------
# Per-path memory features
# ---------------------------------------------------------------------------


def _per_k_stencil(X: np.ndarray, radius: int) -> np.ndarray:
    """Translation-invariant local stencil: returns (n_paths, T, K, 2r+1)."""

    shifts = [np.roll(X, shift, axis=2) for shift in range(-radius, radius + 1)]
    return np.stack(shifts, axis=-1)


def _per_k_delays(X: np.ndarray, lags_steps: Sequence[int]) -> np.ndarray:
    """Per-k lagged values of X_k: returns (n_paths, T, K, len(lags))."""

    n_paths, T, K = X.shape
    out = np.zeros((n_paths, T, K, len(lags_steps)), dtype=float)
    for i, lag in enumerate(lags_steps):
        if lag == 0:
            out[:, :, :, i] = X
        else:
            out[:, lag:, :, i] = X[:, :-lag, :]
            out[:, :lag, :, i] = X[:, :1, :]
    return out


def _per_k_efm(X: np.ndarray, dt: float, taus: Sequence[float]) -> np.ndarray:
    """Causal exponential moving average per (path, k) at multiple time scales."""

    n_paths, T, K = X.shape
    out = np.zeros((n_paths, T, K, len(taus)), dtype=float)
    states = np.zeros((n_paths, K, len(taus)), dtype=float)
    rhos = np.array([np.exp(-dt / float(t)) for t in taus])
    for t in range(T):
        out[:, t, :, :] = states
        for i, rho in enumerate(rhos):
            states[:, :, i] = rho * states[:, :, i] + (1.0 - rho) * X[:, t, :]
    return out


def _per_k_qv(
    X: np.ndarray,
    dt: float,
    taus: Sequence[float],
    windows_mtu: Sequence[float],
) -> np.ndarray:
    """Causal QV-rate features per (path, k) on normalized X_k increments.

    Returns concatenation of exponential-filter QV (one per tau) and rolling
    mean QV (one per window). Increments are normalized by their pooled std,
    so the QV channels are dimensionless.
    """

    n_paths, T, K = X.shape
    dX = np.zeros_like(X)
    dX[:, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
    dx_std = float(np.std(dX[:, :-1, :])) or 1.0
    inc = dX / dx_std
    sq = inc ** 2

    # Exponential filters.
    exp_out = np.zeros((n_paths, T, K, len(taus)), dtype=float)
    exp_states = np.zeros((n_paths, K, len(taus)), dtype=float)
    rhos = np.array([np.exp(-dt / float(t)) for t in taus])
    for t in range(T):
        exp_out[:, t, :, :] = exp_states
        for i, rho in enumerate(rhos):
            exp_states[:, :, i] = rho * exp_states[:, :, i] + (1.0 - rho) * sq[:, t, :]

    # Rolling means.
    csum = np.cumsum(sq, axis=1)
    roll_out = np.zeros((n_paths, T, K, len(windows_mtu)), dtype=float)
    for i, w_mtu in enumerate(windows_mtu):
        w = max(2, int(round(float(w_mtu) / dt)))
        for t in range(T):
            lo = max(0, t - w)
            if t <= lo:
                continue
            total = csum[:, t - 1, :] - (csum[:, lo - 1, :] if lo > 0 else 0.0)
            roll_out[:, t, :, i] = total / float(t - lo)

    return np.concatenate([exp_out, roll_out], axis=-1)


# ---------------------------------------------------------------------------
# Build representations
# ---------------------------------------------------------------------------


def build_representations(
    sim: Dict[str, np.ndarray],
    closure: ClosureConfig,
    cfg: TwoScaleL96Config,
) -> Dict[str, Representation]:
    X = sim["X"]
    drift = sim["drift_X"]           # interval-averaged true drift
    U = sim["U_avg"]                  # interval-averaged unresolved tendency
    resolved = sim["resolved"]        # interval-averaged resolved part on X stencil
    dt = cfg.dt
    # Observable U at the observation step is the closure target: take a finite
    # difference of X and subtract the resolved part the modeller can compute
    # from the X stencil alone. By construction U_obs ~ U_avg up to higher-order
    # corrections; we use the observable version for training, the simulator's
    # U_avg as held-out scoring truth.
    inc = sim["dX"] / dt
    inc[:, -1] = drift[:, -1]
    U_obs = inc - resolved

    stencil = _per_k_stencil(X, closure.stencil_radius)
    lags_steps = [max(1, int(round(t / dt))) for t in closure.delay_lags]
    delays = _per_k_delays(X, lags_steps)
    efm_x = _per_k_efm(X, dt, closure.efm_tau)
    qv = _per_k_qv(X, dt, closure.qv_tau, closure.qv_window)

    raw_feat = stencil
    delay_feat = np.concatenate([stencil, delays], axis=-1)
    efm_feat = np.concatenate([stencil, efm_x], axis=-1)
    qv_feat = np.concatenate([stencil, qv], axis=-1)
    # Oracle "U-Markov" lane: stencil + lag-1 unresolved tendency U_k(t-1).
    # Target is U_k(t); since U is highly autocorrelated this is the upper
    # bound a representation could reach if it perfectly recovered the recent
    # unresolved state. Lag-1 (not lag-0) avoids the trivial identity fit.
    U_lag1 = np.zeros_like(U)
    U_lag1[:, 1:] = U[:, :-1]
    U_lag1[:, 0] = U[:, 0]
    oracle_feat = np.concatenate([stencil, U_lag1[..., None]], axis=-1)

    reps_4d = {
        "raw": raw_feat,
        "delay": delay_feat,
        "efm": efm_feat,
        "leadlag_qv": qv_feat,
        "oracle_Umarkov": oracle_feat,
    }

    n_paths, T, K = X.shape
    flat_U_obs = U_obs.reshape(-1)
    flat_U = U.reshape(-1)
    flat_drift = drift.reshape(-1)

    out: Dict[str, Representation] = {}
    for name, feat in reps_4d.items():
        Z = feat.reshape(-1, feat.shape[-1])
        out[name] = Representation(
            name=name,
            Z=Z,
            U_obs=flat_U_obs,
            U_true=flat_U,
            drift_true=flat_drift,
        )
    return out


# ---------------------------------------------------------------------------
# Nystrom RBF + ridge (reused pattern from memory_mkl_poc)
# ---------------------------------------------------------------------------


def _sample(n: int, size: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.choice(np.arange(n), size=min(int(size), int(n)), replace=False)


def _rbf_features(
    X: np.ndarray,
    landmarks: np.ndarray,
    length_scale: np.ndarray,
) -> np.ndarray:
    Xs = X / length_scale
    Ls = landmarks / length_scale
    a = np.sum(Xs ** 2, axis=1, keepdims=True)
    b = np.sum(Ls ** 2, axis=1, keepdims=True)
    sqdist = np.maximum(a + b.T - 2.0 * (Xs @ Ls.T), 0.0)
    return np.exp(-0.5 * sqdist)


def make_feature_blocks(
    train_reps: Dict[str, Representation],
    valid_reps: Dict[str, Representation],
    test_reps: Dict[str, Representation],
    rep_names: Sequence[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    rng: np.random.RandomState,
    n_landmarks: int,
) -> Dict[str, FeatureBlock]:
    blocks: Dict[str, FeatureBlock] = {}
    for name in rep_names:
        scaler = FeatureScaler.fit(train_reps[name].Z[train_idx])
        X_train = scaler.transform(train_reps[name].Z[train_idx])
        X_valid = scaler.transform(valid_reps[name].Z[valid_idx])
        X_test = scaler.transform(test_reps[name].Z[test_idx])
        length_scale = np.maximum(median_heuristic_length_scale(X_train), 1e-6)
        # Fix landmark RNG per representation so kernel mixtures are reproducible.
        landmark_rng = np.random.RandomState(rng.randint(1, 2**31 - 1))
        landmark_idx = _sample(X_train.shape[0], n_landmarks, landmark_rng)
        landmarks = X_train[landmark_idx]
        blocks[name] = FeatureBlock(
            name=name,
            train=_rbf_features(X_train, landmarks, length_scale),
            valid=_rbf_features(X_valid, landmarks, length_scale),
            test=_rbf_features(X_test, landmarks, length_scale),
            dim=int(train_reps[name].Z.shape[1]),
        )
    return blocks


def _combine_blocks(blocks: Dict[str, FeatureBlock], weights: Dict[str, float], split: str) -> np.ndarray:
    parts = []
    for name, weight in weights.items():
        if weight <= 0.0:
            continue
        parts.append(np.sqrt(weight) * getattr(blocks[name], split))
    if not parts:
        raise ValueError("at least one kernel weight must be positive")
    return np.hstack(parts)


def _fit_ridge(Phi: np.ndarray, y: np.ndarray, ridge: float) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=float).flatten()
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
    if y_scale <= 1e-12:
        y_scale = 1.0
    ys = (y - y_mean) / y_scale
    gram = Phi.T @ Phi + float(ridge) * np.eye(Phi.shape[1])
    coef = np.linalg.solve(gram, Phi.T @ ys)
    return coef, y_mean, y_scale


def _predict_ridge(Phi: np.ndarray, coef: np.ndarray, y_mean: float, y_scale: float) -> np.ndarray:
    return y_mean + y_scale * (Phi @ coef)


def _simplex_grid(names: Sequence[str], step: float = 0.25) -> List[Dict[str, float]]:
    levels = np.arange(0.0, 1.0 + 1e-12, step)
    out: List[Dict[str, float]] = []
    for vals in product(levels, repeat=len(names)):
        total = float(np.sum(vals))
        if abs(total - 1.0) <= 1e-12:
            out.append({name: float(val) for name, val in zip(names, vals)})
    return out


def evaluate_weight_grid(
    blocks: Dict[str, FeatureBlock],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test_U: np.ndarray,
    y_test_drift: np.ndarray,
    candidate_weights: Sequence[Tuple[str, Dict[str, float]]],
    ridges: Sequence[float],
) -> List[ModelResult]:
    """Fit, validate, and score each kernel mixture.

    `y_train` / `y_valid` are observable U = dX/dt - resolved(X).
    `y_test_U` is the held-out true unresolved tendency. `y_test_drift` is the
    full slow drift; we report drift_corr by adding the resolved part back to
    pred_U at test time (resolved = y_test_drift - y_test_U).
    """

    resolved_test = y_test_drift - y_test_U
    U_var = float(np.var(y_test_U, ddof=1))
    results: List[ModelResult] = []
    for model_name, weights in candidate_weights:
        best = None
        for ridge in ridges:
            Phi_train = _combine_blocks(blocks, weights, "train")
            Phi_valid = _combine_blocks(blocks, weights, "valid")
            coef, y_mean, y_scale = _fit_ridge(Phi_train, y_train, ridge)
            pred_valid = _predict_ridge(Phi_valid, coef, y_mean, y_scale)
            valid_mse = float(np.mean((pred_valid - y_valid) ** 2))
            if best is None or valid_mse < best[0]:
                best = (valid_mse, ridge, coef, y_mean, y_scale)
        assert best is not None
        valid_mse, ridge, coef, y_mean, y_scale = best
        Phi_test = _combine_blocks(blocks, weights, "test")
        pred_U = _predict_ridge(Phi_test, coef, y_mean, y_scale)

        u_err = pred_U - y_test_U
        u_resid = float(np.mean(u_err ** 2) / max(U_var, 1e-12))
        u_corr = _safe_corr(pred_U, y_test_U)
        d_corr = _safe_corr(pred_U + resolved_test, y_test_drift)

        results.append(
            ModelResult(
                name=model_name,
                weights={k: v for k, v in weights.items() if v > 0.0},
                ridge=float(ridge),
                valid_U_mse=float(valid_mse),
                U_corr=u_corr,
                U_residual=u_resid,
                drift_corr=d_corr,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Per-seed driver
# ---------------------------------------------------------------------------


def run_seed(
    seed: int,
    sim_cfg: TwoScaleL96Config,
    closure: ClosureConfig,
) -> List[ModelResult]:
    train_sim = simulate_two_scale_l96(sim_cfg, n_paths=closure.n_paths_train, seed=seed * 10 + 1)
    valid_sim = simulate_two_scale_l96(sim_cfg, n_paths=closure.n_paths_valid, seed=seed * 10 + 2)
    test_sim = simulate_two_scale_l96(sim_cfg, n_paths=closure.n_paths_test, seed=seed * 10 + 3)
    train_reps = build_representations(train_sim, closure, sim_cfg)
    valid_reps = build_representations(valid_sim, closure, sim_cfg)
    test_reps = build_representations(test_sim, closure, sim_cfg)

    rng = np.random.RandomState(seed)
    n_train = train_reps["raw"].Z.shape[0]
    n_valid = valid_reps["raw"].Z.shape[0]
    n_test = test_reps["raw"].Z.shape[0]
    train_idx = _sample(n_train, closure.n_train_samples, rng)
    valid_idx = _sample(n_valid, closure.n_valid_samples, rng)
    test_idx = _sample(n_test, closure.n_test_samples, rng)

    kernel_names = ["raw", "delay", "efm", "leadlag_qv"]
    blocks = make_feature_blocks(
        train_reps, valid_reps, test_reps, kernel_names,
        train_idx, valid_idx, test_idx, rng, n_landmarks=closure.n_landmarks,
    )
    y_train = train_reps["raw"].U_obs[train_idx]
    y_valid = valid_reps["raw"].U_obs[valid_idx]
    y_test_U = test_reps["raw"].U_true[test_idx]
    y_test_drift = test_reps["raw"].drift_true[test_idx]

    candidates: List[Tuple[str, Dict[str, float]]] = [
        (name, {name: 1.0}) for name in kernel_names
    ]
    candidates.append(("mkl_memory_sum",
                       {"efm": 1.0 / 3.0, "leadlag_qv": 1.0 / 3.0, "delay": 1.0 / 3.0}))
    # mkl_grid scans the full nonnegative simplex including singletons -- this
    # lets the learner converge to a single kernel when mixing doesn't help.
    candidates.extend(
        ("mkl_grid", weights)
        for weights in _simplex_grid(kernel_names, step=0.25)
        if any(v > 0.0 for v in weights.values())
    )

    # Oracle "U-Markov" lane: stencil + lag-1 true U as additional feature.
    oracle_blocks = make_feature_blocks(
        train_reps, valid_reps, test_reps, ["oracle_Umarkov"],
        train_idx, valid_idx, test_idx, rng, n_landmarks=closure.n_landmarks,
    )
    blocks.update(oracle_blocks)
    candidates.append(("oracle_Umarkov", {"oracle_Umarkov": 1.0}))

    results = evaluate_weight_grid(
        blocks, y_train, y_valid, y_test_U, y_test_drift,
        candidates,
        ridges=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    )

    best_grid = min(
        (r for r in results if r.name == "mkl_grid"),
        key=lambda r: r.valid_U_mse,
    )
    keep_names = set(kernel_names) | {"mkl_memory_sum", "oracle_Umarkov"}
    compact = [r for r in results if r.name in keep_names]
    compact.append(ModelResult(name="mkl_learned", **{k: getattr(best_grid, k) for k in (
        "weights", "ridge", "valid_U_mse", "U_corr", "U_residual", "drift_corr",
    )}))
    return compact


def _mean_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95))


def print_summary(results: Sequence[ModelResult], sim_cfg: TwoScaleL96Config, closure: ClosureConfig) -> None:
    order = ["raw", "delay", "efm", "leadlag_qv", "mkl_memory_sum", "mkl_learned", "oracle_Umarkov"]
    by_name: Dict[str, List[ModelResult]] = {name: [] for name in order}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    print("=" * 122)
    print("TWO-SCALE LORENZ-96 CLOSURE BENCHMARK")
    print("=" * 122)
    print(
        f"K={sim_cfg.K} J={sim_cfg.J} F={sim_cfg.F} h={sim_cfg.h} c={sim_cfg.c} b={sim_cfg.b} "
        f"dt={sim_cfg.dt} T={sim_cfg.T}"
    )
    print(
        f"stencil_radius={closure.stencil_radius} delays={closure.delay_lags} "
        f"efm_tau={closure.efm_tau} qv_tau={closure.qv_tau} qv_window={closure.qv_window}"
    )
    print("Train/tune on observed U = dX/dt - resolved(X); score against true U_k.")
    print("Closure R^2 = 1 - U_residual.")
    print("-" * 122)
    header = (
        "model              U_resid mean [90% CrI]   closure_R2   U_corr   drift_corr   weights"
    )
    print(header)
    print("-" * len(header))
    for name in order:
        rows = by_name.get(name, [])
        if not rows:
            continue
        ur = _mean_interval([r.U_residual for r in rows])
        cr2 = 1.0 - ur[0]
        uc = float(np.mean([r.U_corr for r in rows]))
        dc = float(np.mean([r.drift_corr for r in rows]))
        if name.startswith("mkl"):
            weights = rows[-1].weights
        else:
            weights = rows[0].weights
        weight_str = ",".join(f"{k}:{v:.2f}" for k, v in weights.items())
        print(
            f"{name:18s} {ur[0]:7.4f} [{ur[1]:6.4f}, {ur[2]:6.4f}]"
            f"   {cr2:+6.3f}     {uc:+6.3f}   {dc:+6.3f}      {weight_str}"
        )
    print("-" * 122)
    single_U = {
        name: float(np.mean([r.U_residual for r in by_name.get(name, [])]))
        for name in ("raw", "delay", "efm", "leadlag_qv")
    }
    learned = [r for r in results if r.name == "mkl_learned"]
    if learned and single_U:
        ratio = min(single_U.values()) / float(np.mean([r.U_residual for r in learned]))
        print(f"best-single / learned-MKL U-residual ratio: {ratio:.2f}x")


def plot_summary(results: Sequence[ModelResult], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["raw", "delay", "efm", "leadlag_qv", "mkl_learned", "oracle_Umarkov"]
    U_mean, U_lo, U_hi = [], [], []
    R2_mean = []
    for name in order:
        vals_u = [r.U_residual for r in results if r.name == name]
        mu, lu, hu = _mean_interval(vals_u)
        U_mean.append(mu); U_lo.append(mu - lu); U_hi.append(hu - mu)
        R2_mean.append(1.0 - mu)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    colors = ["#666666", "#4C78A8", "#54A24B", "#F58518", "#B279A2", "#2F4B7C"]
    axes[0].bar(order, U_mean, yerr=[U_lo, U_hi], color=colors)
    axes[0].set_ylabel("held-out U residual  (MSE / Var(U))")
    axes[0].set_title("Two-scale Lorenz-96: unresolved-tendency closure")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(order, R2_mean, color=colors)
    axes[1].set_ylabel("held-out closure R^2  = 1 - U_residual")
    axes[1].set_title("Variance of U_k explained")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=0.8)

    fig.suptitle("Two-scale Lorenz-96: kernel mixtures for partial-observation closure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    # Closure regime where the deterministic part of U is non-trivial:
    # F=20, c=4 leaves about 30 percent of Var(U) unexplained by a polynomial
    # in the local X stencil.
    sim_cfg = TwoScaleL96Config(F=20.0, c=4.0, obs_subsample=4)
    seeds = [20260516, 20260517, 20260518]

    regimes = [
        ("full_stencil",  ClosureConfig(stencil_radius=2)),
        ("point_obs",     ClosureConfig(stencil_radius=0)),
    ]

    for label, closure in regimes:
        print()
        print(f"##### regime: {label} (stencil_radius={closure.stencil_radius}) #####")
        all_results: List[ModelResult] = []
        for seed in seeds:
            print(f"running seed {seed}...", flush=True)
            all_results.extend(run_seed(seed, sim_cfg, closure))
        print_summary(all_results, sim_cfg, closure)
        out_path = os.path.join(HERE, f"l96_closure_mkl_{label}.png")
        plot_summary(all_results, out_path)
        print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
