r"""
Multiple-memory-kernel POC for partially observed generator learning.

This is a deliberately targeted benchmark for the MKL story:

    observed history -> several causal memory lifts -> additive kernel sum

The synthetic observed process has generator drift

    b_t = -0.20 X_t
          + beta_m tanh(M_t / scale_m)
          + beta_q tanh(Q_t / scale_q),

where M_t is an exponential filter of past observed increments and Q_t is an
exponential filter of past squared increments.  Raw observations are therefore
not a sufficient state.  An EFM kernel can see M_t but not Q_t; a lead-lag/QV
kernel can see Q_t but not M_t.  A nonnegative sum of kernels can represent the
additive generator action.

The true drift is used only for held-out scoring.  Training and validation use
only observed finite-difference targets dX/dt.
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

from experiments.science_poc.two_scale_generator_poc import FeatureScaler, _safe_corr
from src.control.kernel_residual_control import median_heuristic_length_scale


@dataclass(frozen=True)
class MemoryMKLConfig:
    dt: float = 0.02
    T: float = 10.0
    sigma_X: float = 0.22
    tau_m: float = 0.85
    tau_q: float = 0.25
    beta_m: float = 0.85
    beta_q: float = 0.85
    scale_m: float = 0.55
    scale_q: float = 0.16
    x_reversion: float = 0.20

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def rho_m(self) -> float:
        return float(np.exp(-self.dt / self.tau_m))

    @property
    def rho_q(self) -> float:
        return float(np.exp(-self.dt / self.tau_q))


@dataclass
class Representation:
    name: str
    Z: np.ndarray
    Z_next: np.ndarray
    drift_true: np.ndarray
    dX_over_dt: np.ndarray


@dataclass
class FeatureBlock:
    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    dim: int
    length_scale: np.ndarray


@dataclass(frozen=True)
class ModelResult:
    name: str
    weights: Dict[str, float]
    ridge: float
    valid_increment_mse: float
    test_increment_mse: float
    drift_corr: float
    generator_residual: float
    drift_nrmse: float


def simulate_memory_mkl_system(
    config: MemoryMKLConfig,
    n_paths: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Simulate the partially observed additive-memory process."""

    rng = np.random.RandomState(seed)
    n_steps = config.n_steps
    dt = float(config.dt)
    sqrt_dt = float(np.sqrt(dt))

    x = 0.35 * rng.standard_normal(n_paths)
    m = np.zeros(n_paths, dtype=float)
    q = np.zeros(n_paths, dtype=float)

    X = np.zeros((n_paths, n_steps), dtype=float)
    M = np.zeros((n_paths, n_steps), dtype=float)
    Q = np.zeros((n_paths, n_steps), dtype=float)
    drift = np.zeros((n_paths, n_steps), dtype=float)
    dX = np.zeros((n_paths, n_steps), dtype=float)

    for t in range(n_steps):
        b = (
            -config.x_reversion * x
            + config.beta_m * np.tanh(m / config.scale_m)
            + config.beta_q * np.tanh(q / config.scale_q)
        )
        dx = b * dt + config.sigma_X * sqrt_dt * rng.standard_normal(n_paths)

        X[:, t] = x
        M[:, t] = m
        Q[:, t] = q
        drift[:, t] = b
        dX[:, t] = dx

        innovation = dx / max(config.sigma_X * sqrt_dt, 1e-12)
        m = config.rho_m * m + (1.0 - config.rho_m) * innovation
        q = config.rho_q * q + (1.0 - config.rho_q) * (innovation ** 2 - 1.0)
        x = x + dx

    return {"X": X, "M": M, "Q": Q, "drift_X": drift, "dX": dX}


def _exp_filter(values: np.ndarray, rho: float) -> np.ndarray:
    """Causal exponential filter; output at t uses values before t."""

    n_paths, n_steps = values.shape
    out = np.zeros((n_paths, n_steps), dtype=float)
    state = np.zeros(n_paths, dtype=float)
    for t in range(n_steps):
        out[:, t] = state
        state = rho * state + (1.0 - rho) * values[:, t]
    return out


def _rolling_qv_rate(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean of squared normalized increments."""

    n_paths, n_steps = values.shape
    out = np.zeros((n_paths, n_steps), dtype=float)
    sq = values ** 2 - 1.0
    csum = np.cumsum(sq, axis=1)
    for t in range(n_steps):
        lo = max(0, t - int(window))
        if t <= lo:
            out[:, t] = 0.0
        else:
            total = csum[:, t - 1] - (csum[:, lo - 1] if lo > 0 else 0.0)
            out[:, t] = total / float(t - lo)
    return out


def build_representations(
    sim: Dict[str, np.ndarray],
    config: MemoryMKLConfig,
) -> Dict[str, Representation]:
    """Build observable causal memory lifts."""

    X = sim["X"]
    dX = sim["dX"]
    drift = sim["drift_X"]
    n_paths, n_steps = X.shape

    normalized_inc = dX / max(config.sigma_X * np.sqrt(config.dt), 1e-12)
    efm_slow = _exp_filter(normalized_inc, config.rho_m)
    efm_fast = _exp_filter(normalized_inc, np.exp(-config.dt / 0.20))
    qv_fast = _exp_filter(normalized_inc ** 2 - 1.0, config.rho_q)
    qv_roll_short = _rolling_qv_rate(normalized_inc, window=max(2, int(round(0.20 / config.dt))))
    qv_roll_mid = _rolling_qv_rate(normalized_inc, window=max(2, int(round(0.50 / config.dt))))

    lag_1 = max(1, int(round(0.20 / config.dt)))
    lag_2 = max(1, int(round(0.70 / config.dt)))
    delay = np.zeros((n_paths, n_steps, 3), dtype=float)
    for t in range(n_steps):
        delay[:, t, 0] = X[:, t]
        delay[:, t, 1] = X[:, max(0, t - lag_1)]
        delay[:, t, 2] = X[:, max(0, t - lag_2)]

    reps_3d = {
        "raw_x": X[:, :, None],
        "delay_coords": delay,
        "efm_memory": np.stack([X, efm_slow, efm_fast], axis=-1),
        # QV channels are the semantic level-2 lead-lag signature readouts:
        # 2 * Area(lead, lag) / elapsed equals average squared increment.
        "leadlag_qv": np.stack([X, qv_fast, qv_roll_short, qv_roll_mid], axis=-1),
        "oracle_mq": np.stack([X, sim["M"], sim["Q"]], axis=-1),
    }

    aligned_drift = drift[:, :-1].reshape(-1)
    aligned_dXdt = (dX[:, :-1] / config.dt).reshape(-1)
    out: Dict[str, Representation] = {}
    for name, Z in reps_3d.items():
        out[name] = Representation(
            name=name,
            Z=Z[:, :-1, :].reshape(-1, Z.shape[-1]),
            Z_next=Z[:, 1:, :].reshape(-1, Z.shape[-1]),
            drift_true=aligned_drift,
            dX_over_dt=aligned_dXdt,
        )
    return out


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
        landmark_idx = _sample(X_train.shape[0], n_landmarks, rng)
        landmarks = X_train[landmark_idx]
        blocks[name] = FeatureBlock(
            name=name,
            train=_rbf_features(X_train, landmarks, length_scale),
            valid=_rbf_features(X_valid, landmarks, length_scale),
            test=_rbf_features(X_test, landmarks, length_scale),
            dim=int(train_reps[name].Z.shape[1]),
            length_scale=length_scale,
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
    y_test_inc: np.ndarray,
    y_test_drift: np.ndarray,
    candidate_weights: Sequence[Tuple[str, Dict[str, float]]],
    ridges: Sequence[float],
) -> List[ModelResult]:
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
        pred_test = _predict_ridge(Phi_test, coef, y_mean, y_scale)
        drift_err = pred_test - y_test_drift
        results.append(
            ModelResult(
                name=model_name,
                weights={k: v for k, v in weights.items() if v > 0.0},
                ridge=float(ridge),
                valid_increment_mse=float(valid_mse),
                test_increment_mse=float(np.mean((pred_test - y_test_inc) ** 2)),
                drift_corr=_safe_corr(pred_test, y_test_drift),
                generator_residual=float(
                    np.mean(drift_err ** 2) / max(np.mean(y_test_drift ** 2), 1e-12)
                ),
                drift_nrmse=float(
                    np.sqrt(np.mean(drift_err ** 2))
                    / max(np.std(y_test_drift, ddof=1), 1e-12)
                ),
            )
        )
    return results


def run_seed(seed: int, config: MemoryMKLConfig) -> List[ModelResult]:
    train_sim = simulate_memory_mkl_system(config, n_paths=18, seed=seed * 10 + 1)
    valid_sim = simulate_memory_mkl_system(config, n_paths=10, seed=seed * 10 + 2)
    test_sim = simulate_memory_mkl_system(config, n_paths=14, seed=seed * 10 + 3)
    train_reps = build_representations(train_sim, config)
    valid_reps = build_representations(valid_sim, config)
    test_reps = build_representations(test_sim, config)

    rng = np.random.RandomState(seed)
    train_idx = _sample(train_reps["raw_x"].Z.shape[0], 900, rng)
    valid_idx = _sample(valid_reps["raw_x"].Z.shape[0], 900, rng)
    test_idx = _sample(test_reps["raw_x"].Z.shape[0], 1400, rng)

    kernel_names = ["raw_x", "delay_coords", "efm_memory", "leadlag_qv"]
    blocks = make_feature_blocks(
        train_reps, valid_reps, test_reps, kernel_names,
        train_idx, valid_idx, test_idx, rng, n_landmarks=120,
    )
    y_train = train_reps["raw_x"].dX_over_dt[train_idx]
    y_valid = valid_reps["raw_x"].dX_over_dt[valid_idx]
    y_test_inc = test_reps["raw_x"].dX_over_dt[test_idx]
    y_test_drift = test_reps["raw_x"].drift_true[test_idx]

    candidates: List[Tuple[str, Dict[str, float]]] = [
        (name, {name: 1.0}) for name in kernel_names
    ]
    candidates.append(("mkl_memory_sum", {"efm_memory": 0.5, "leadlag_qv": 0.5}))
    candidates.extend(
        ("mkl_grid", weights)
        for weights in _simplex_grid(kernel_names, step=0.25)
        if sum(v > 0.0 for v in weights.values()) >= 2
    )
    candidates.append(("oracle_mq", {"oracle_mq": 1.0}))

    # Add oracle block after observable kernels so it cannot influence landmarks.
    oracle_blocks = make_feature_blocks(
        train_reps, valid_reps, test_reps, ["oracle_mq"],
        train_idx, valid_idx, test_idx, rng, n_landmarks=120,
    )
    blocks.update(oracle_blocks)

    results = evaluate_weight_grid(
        blocks,
        y_train,
        y_valid,
        y_test_inc,
        y_test_drift,
        candidates,
        ridges=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    )

    best_grid = min((r for r in results if r.name == "mkl_grid"), key=lambda r: r.valid_increment_mse)
    keep_names = set(kernel_names) | {"mkl_memory_sum", "oracle_mq"}
    compact = [r for r in results if r.name in keep_names]
    compact.append(ModelResult(name="mkl_learned", **{k: getattr(best_grid, k) for k in (
        "weights", "ridge", "valid_increment_mse", "test_increment_mse",
        "drift_corr", "generator_residual", "drift_nrmse",
    )}))
    return compact


def _mean_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95))


def print_summary(results: Sequence[ModelResult], config: MemoryMKLConfig) -> None:
    order = [
        "raw_x",
        "delay_coords",
        "efm_memory",
        "leadlag_qv",
        "mkl_memory_sum",
        "mkl_learned",
        "oracle_mq",
    ]
    by_name: Dict[str, List[ModelResult]] = {name: [] for name in order}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    print("=" * 112)
    print("MULTIPLE-MEMORY-KERNEL GENERATOR POC")
    print("=" * 112)
    print(
        f"dt={config.dt} T={config.T} sigma_X={config.sigma_X} "
        f"tau_m={config.tau_m} tau_q={config.tau_q}"
    )
    print("Training/tuning target: observed dX/dt only. True drift is held-out scoring only.")
    print("Model: random-landmark RBF blocks with nonnegative kernel-sum weights.")
    print("-" * 112)
    header = "model           gen_resid mean [90% CrI]     drift_corr mean [90% CrI]   inc_mse   selected weights"
    print(header)
    print("-" * len(header))
    for name in order:
        rows = by_name.get(name, [])
        if not rows:
            continue
        resid = _mean_interval([r.generator_residual for r in rows])
        corr = _mean_interval([r.drift_corr for r in rows])
        inc = float(np.mean([r.test_increment_mse for r in rows]))
        if name.startswith("mkl"):
            weights = rows[-1].weights
        else:
            weights = rows[0].weights
        weight_str = ",".join(f"{k}:{v:.2f}" for k, v in weights.items())
        print(
            f"{name:15s} {resid[0]:7.3f} [{resid[1]:6.3f}, {resid[2]:6.3f}]"
            f"        {corr[0]:+7.3f} [{corr[1]:+6.3f}, {corr[2]:+6.3f}]"
            f"   {inc:7.3f}   {weight_str}"
        )
    print("-" * 112)
    single_means = {
        name: np.mean([r.generator_residual for r in by_name.get(name, [])])
        for name in ("raw_x", "delay_coords", "efm_memory", "leadlag_qv")
    }
    learned = [r for r in results if r.name == "mkl_learned"]
    if learned:
        ratio = min(single_means.values()) / np.mean([r.generator_residual for r in learned])
        print(f"best single / learned MKL generator-residual ratio: {ratio:.2f}x")


def plot_summary(results: Sequence[ModelResult], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["raw_x", "delay_coords", "efm_memory", "leadlag_qv", "mkl_learned", "oracle_mq"]
    means = []
    lows = []
    highs = []
    for name in order:
        vals = [r.generator_residual for r in results if r.name == name]
        mean, lo, hi = _mean_interval(vals)
        means.append(mean)
        lows.append(mean - lo)
        highs.append(hi - mean)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(order, means, yerr=[lows, highs], color=["#666666", "#4C78A8", "#54A24B", "#F58518", "#B279A2", "#2F4B7C"])
    ax.set_ylabel("held-out generator residual")
    ax.set_title("Multiple memory kernels: additive hidden path functionals")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    config = MemoryMKLConfig()
    seeds = [20260516, 20260517, 20260518]
    all_results: List[ModelResult] = []
    for seed in seeds:
        print(f"running seed {seed}...", flush=True)
        all_results.extend(run_seed(seed, config))
    print_summary(all_results, config)
    out_path = os.path.join(HERE, "memory_mkl_poc.png")
    plot_summary(all_results, out_path)
    print()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
