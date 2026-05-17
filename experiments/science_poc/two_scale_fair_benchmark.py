r"""
Fair trajectory-only benchmark for the two-scale double-well POC.

This script is stricter than `two_scale_generator_poc.py`:

1. Models train only on observed finite-difference targets dX / dt.
2. Hyperparameters are selected on independent validation trajectories using
   only observed finite-difference error.
3. Final scores are reported on independent held-out trajectories.
4. The true drift is used only for evaluation, because this is a synthetic
   benchmark with known generator action.

The hidden-state representation `oracle_xh` is included only as an upper bound.
It is not a fair observable baseline.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.science_poc.envs.two_scale_double_well import (
    TwoScaleDoubleWellConfig,
    simulate_two_scale_double_well,
)
from experiments.science_poc.two_scale_generator_poc import (
    FeatureScaler,
    Representation,
    _koopman_timescales,
    _safe_corr,
    build_representations,
    leadlag_qv_sanity_check,
)
from src.control.kernel_residual_control import median_heuristic_length_scale


@dataclass(frozen=True)
class HyperParams:
    alpha: float
    length_multiplier: float


@dataclass
class NystromRBFKRR:
    """Nyström RBF ridge regressor with fixed landmarks.

    The feature map is k(x, landmark_j).  Ridge is solved in the finite
    landmark feature space.  This is the scalable approximation we want for
    the science POC; exact GP/KRR remains available in the older diagnostic.
    """

    landmarks: np.ndarray
    length_scale: np.ndarray
    ridge: float
    y_mean: float = 0.0
    y_scale: float = 1.0
    weights: Optional[np.ndarray] = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Zx = X / self.length_scale
        Zl = self.landmarks / self.length_scale
        a = np.sum(Zx ** 2, axis=1, keepdims=True)
        b = np.sum(Zl ** 2, axis=1, keepdims=True)
        sqdist = np.maximum(a + b.T - 2.0 * (Zx @ Zl.T), 0.0)
        return np.exp(-0.5 * sqdist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NystromRBFKRR":
        y = np.asarray(y, dtype=float).flatten()
        Phi = self._features(X)
        self.y_mean = float(np.mean(y))
        self.y_scale = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
        if self.y_scale <= 1e-12:
            self.y_scale = 1.0
        ys = (y - self.y_mean) / self.y_scale
        gram = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        self.weights = np.linalg.solve(gram, Phi.T @ ys)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("call fit() before predict()")
        return self.y_mean + self.y_scale * (self._features(X) @ self.weights)


@dataclass(frozen=True)
class RepFairResult:
    rep: str
    dim: int
    seed: int
    alpha: float
    length_multiplier: float
    n_landmarks: int
    valid_increment_mse: float
    test_increment_mse: float
    transition_rmse: float
    drift_corr: float
    drift_nrmse: float
    generator_residual: float
    top_timescale: float


def _sample_indices(n: int, size: int, rng: np.random.RandomState) -> np.ndarray:
    size = min(int(size), int(n))
    return rng.choice(np.arange(n), size=size, replace=False)


def _choose_landmarks(
    X_scaled: np.ndarray,
    n_landmarks: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    idx = _sample_indices(X_scaled.shape[0], n_landmarks, rng)
    return X_scaled[idx]


def _fit_nystrom_fixed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hp: HyperParams,
    rng: np.random.RandomState,
    n_landmarks: int,
) -> Tuple[FeatureScaler, NystromRBFKRR]:
    scaler = FeatureScaler.fit(X_train)
    Xs = scaler.transform(X_train)
    ls = median_heuristic_length_scale(Xs) * float(hp.length_multiplier)
    ls = np.maximum(ls, 1e-6)
    landmarks = _choose_landmarks(Xs, n_landmarks=n_landmarks, rng=rng)
    model = NystromRBFKRR(
        landmarks=landmarks,
        length_scale=ls,
        ridge=float(hp.alpha),
    ).fit(Xs, y_train)
    return scaler, model


def _predict(
    scaler: FeatureScaler,
    model: NystromRBFKRR,
    X: np.ndarray,
) -> np.ndarray:
    return model.predict(scaler.transform(X))


def _concat_representations(name: str, reps: Sequence[Representation]) -> Representation:
    return Representation(
        name=name,
        Z=np.vstack([r.Z for r in reps]),
        Z_next=np.vstack([r.Z_next for r in reps]),
        drift_true=np.concatenate([r.drift_true for r in reps]),
        dX_over_dt=np.concatenate([r.dX_over_dt for r in reps]),
        H=np.concatenate([r.H for r in reps]),
    )


def _select_hyperparams(
    train: Representation,
    valid: Representation,
    rng: np.random.RandomState,
    train_size: int,
    valid_size: int,
    grid: Sequence[HyperParams],
    n_landmarks: int,
) -> Tuple[HyperParams, float]:
    train_idx = _sample_indices(train.Z.shape[0], train_size, rng)
    valid_idx = _sample_indices(valid.Z.shape[0], valid_size, rng)
    X_train = train.Z[train_idx]
    y_train = train.dX_over_dt[train_idx]
    X_valid = valid.Z[valid_idx]
    y_valid = valid.dX_over_dt[valid_idx]

    best_hp = grid[0]
    best_mse = float("inf")
    for hp in grid:
        scaler, model = _fit_nystrom_fixed(X_train, y_train, hp, rng, n_landmarks)
        pred = _predict(scaler, model, X_valid)
        mse = float(np.mean((pred - y_valid) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_hp = hp
    return best_hp, best_mse


def _evaluate_representation(
    rep_name: str,
    train: Representation,
    valid: Representation,
    test: Representation,
    config: TwoScaleDoubleWellConfig,
    rng: np.random.RandomState,
    seed: int,
    grid: Sequence[HyperParams],
    n_landmarks: int = 96,
    tune_train_size: int = 240,
    tune_valid_size: int = 650,
    final_train_size: int = 450,
    test_size: int = 1500,
) -> RepFairResult:
    hp, valid_mse = _select_hyperparams(
        train, valid, rng, tune_train_size, tune_valid_size, grid, n_landmarks,
    )
    train_valid = _concat_representations(rep_name, [train, valid])
    final_idx = _sample_indices(train_valid.Z.shape[0], final_train_size, rng)
    scaler, model = _fit_nystrom_fixed(
        train_valid.Z[final_idx],
        train_valid.dX_over_dt[final_idx],
        hp,
        rng,
        n_landmarks,
    )

    test_idx = _sample_indices(test.Z.shape[0], test_size, rng)
    pred = _predict(scaler, model, test.Z[test_idx])
    y_inc = test.dX_over_dt[test_idx]
    true_drift = test.drift_true[test_idx]
    x_now = test.Z[test_idx, 0]
    x_next = test.Z_next[test_idx, 0]

    drift_err = pred - true_drift
    transition_err = x_now + config.dt * pred - x_next
    spec_idx = _sample_indices(test.Z.shape[0], min(5000, test.Z.shape[0]), rng)
    times = _koopman_timescales(
        test.Z[spec_idx], test.Z_next[spec_idx], dt=config.dt, max_out=1,
    )
    return RepFairResult(
        rep=rep_name,
        dim=int(test.Z.shape[1]),
        seed=int(seed),
        alpha=float(hp.alpha),
        length_multiplier=float(hp.length_multiplier),
        n_landmarks=int(min(n_landmarks, final_idx.size)),
        valid_increment_mse=float(valid_mse),
        test_increment_mse=float(np.mean((pred - y_inc) ** 2)),
        transition_rmse=float(np.sqrt(np.mean(transition_err ** 2))),
        drift_corr=_safe_corr(pred, true_drift),
        drift_nrmse=float(
            np.sqrt(np.mean(drift_err ** 2)) / max(np.std(true_drift, ddof=1), 1e-12)
        ),
        generator_residual=float(
            np.mean(drift_err ** 2) / max(np.mean(true_drift ** 2), 1e-12)
        ),
        top_timescale=float(times[0]) if times else float("nan"),
    )


def run_seed(
    seed: int,
    config: TwoScaleDoubleWellConfig,
    rep_order: Sequence[str],
    grid: Sequence[HyperParams],
) -> List[RepFairResult]:
    train_sim = simulate_two_scale_double_well(config, n_paths=22, seed=seed * 10 + 1)
    valid_sim = simulate_two_scale_double_well(config, n_paths=12, seed=seed * 10 + 2)
    test_sim = simulate_two_scale_double_well(config, n_paths=18, seed=seed * 10 + 3)
    train_reps = build_representations(train_sim, config)
    valid_reps = build_representations(valid_sim, config)
    test_reps = build_representations(test_sim, config)
    results: List[RepFairResult] = []
    for j, name in enumerate(rep_order):
        # Per-representation RNG makes landmark draws invariant to table order
        # and to adding/removing unrelated benchmark lanes.
        rng = np.random.RandomState(seed * 100 + j)
        results.append(_evaluate_representation(
            name,
            train_reps[name],
            valid_reps[name],
            test_reps[name],
            config,
            rng,
            seed,
            grid,
        ))
    return results


def _bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.RandomState,
    n_draws: int = 12000,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float).flatten()
    weights = rng.dirichlet(np.ones(values.size), size=n_draws)
    draws = weights @ values
    return (
        float(np.mean(draws)),
        float(np.quantile(draws, 0.05)),
        float(np.quantile(draws, 0.95)),
    )


def _rep_values(results: Sequence[RepFairResult], rep: str, metric: str) -> np.ndarray:
    return np.array([getattr(r, metric) for r in results if r.rep == rep], dtype=float)


def _paired_metric(
    results: Sequence[RepFairResult],
    rep_a: str,
    rep_b: str,
    metric: str,
) -> np.ndarray:
    by_key = {(r.rep, r.seed): getattr(r, metric) for r in results}
    seeds = sorted({r.seed for r in results if r.rep == rep_a} & {r.seed for r in results if r.rep == rep_b})
    return np.array([by_key[(rep_a, s)] - by_key[(rep_b, s)] for s in seeds], dtype=float)


def print_summary(
    results: Sequence[RepFairResult],
    rep_order: Sequence[str],
    config: TwoScaleDoubleWellConfig,
) -> None:
    rng = np.random.RandomState(20260518)
    seeds = sorted({r.seed for r in results})
    print("=" * 118)
    print("FAIR TWO-SCALE DOUBLE-WELL BENCHMARK")
    print("=" * 118)
    print(
        "Training/tuning target: observed dX/dt only. True drift is used only for held-out scoring."
    )
    print("Model: Nyström RBF kernel ridge regression with random training landmarks.")
    print(
        f"dt={config.dt} T={config.T} beta={config.beta} sigma_X={config.sigma_X} "
        f"kappa_H={config.kappa_H} hidden_timescale={config.hidden_timescale:.3f} seeds={seeds}"
    )
    print("-" * 118)
    header = (
        "rep           dim  m    gen_resid mean [90% CrI]     drift_corr mean [90% CrI]   "
        "test_inc_mse   trans_rmse"
    )
    print(header)
    print("-" * len(header))
    for rep in rep_order:
        rows = [r for r in results if r.rep == rep]
        dim = rows[0].dim
        m_landmarks = rows[0].n_landmarks
        resid = _rep_values(results, rep, "generator_residual")
        corr = _rep_values(results, rep, "drift_corr")
        inc = _rep_values(results, rep, "test_increment_mse")
        trans = _rep_values(results, rep, "transition_rmse")
        resid_m, resid_lo, resid_hi = _bootstrap_mean_interval(resid, rng)
        corr_m, corr_lo, corr_hi = _bootstrap_mean_interval(corr, rng)
        inc_m, _, _ = _bootstrap_mean_interval(inc, rng)
        trans_m, _, _ = _bootstrap_mean_interval(trans, rng)
        print(
            f"{rep:13s} {dim:3d} {m_landmarks:3d}  {resid_m:7.3f} [{resid_lo:6.3f}, {resid_hi:6.3f}]"
            f"        {corr_m:+7.3f} [{corr_lo:+6.3f}, {corr_hi:+6.3f}]"
            f"      {inc_m:9.3f}   {trans_m:9.4f}"
        )
    print("-" * 118)
    raw_resid = _rep_values(results, "raw_x", "generator_residual")
    for rep in rep_order:
        if rep == "raw_x":
            continue
        rep_resid = _rep_values(results, rep, "generator_residual")
        ratio = raw_resid / np.maximum(rep_resid, 1e-12)
        ratio_m, ratio_lo, ratio_hi = _bootstrap_mean_interval(ratio, rng)
        diff = raw_resid - rep_resid
        weights = rng.dirichlet(np.ones(diff.size), size=12000)
        p_better = float(np.mean((weights @ diff) > 0.0))
        print(
            f"raw_x / {rep} generator-residual ratio: "
            f"{ratio_m:.2f} [{ratio_lo:.2f}, {ratio_hi:.2f}], "
            f"P({rep} improves over raw_x)={p_better:.3f}"
        )
    print("-" * 118)
    print("Selected hyperparameters by seed:")
    for rep in rep_order:
        selected = [
            f"s{r.seed}:a={r.alpha:g},l={r.length_multiplier:g},m={r.n_landmarks}"
            for r in results
            if r.rep == rep
        ]
        print(f"  {rep:13s} " + "; ".join(selected))


def plot_summary(
    results: Sequence[RepFairResult],
    rep_order: Sequence[str],
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(20260519)
    means = []
    lo = []
    hi = []
    corr_means = []
    corr_lo = []
    corr_hi = []
    for rep in rep_order:
        m, q05, q95 = _bootstrap_mean_interval(
            _rep_values(results, rep, "generator_residual"), rng,
        )
        means.append(m)
        lo.append(q05)
        hi.append(q95)
        cm, cq05, cq95 = _bootstrap_mean_interval(
            _rep_values(results, rep, "drift_corr"), rng,
        )
        corr_means.append(cm)
        corr_lo.append(cq05)
        corr_hi.append(cq95)

    x = np.arange(len(rep_order))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = [
        "#4C78A8", "#F58518", "#54A24B", "#72B7B2",
        "#E45756", "#B279A2", "#FF9DA6", "#9D755D",
    ]

    ax = axes[0]
    yerr = [np.array(means) - np.array(lo), np.array(hi) - np.array(means)]
    ax.bar(x, means, yerr=yerr, capsize=4, color=colors[: len(rep_order)])
    ax.set_xticks(x)
    ax.set_xticklabels(rep_order, rotation=20)
    ax.set_ylabel("generator residual")
    ax.set_title("Trajectory-trained GP/KRR: lower is better")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    yerr = [
        np.array(corr_means) - np.array(corr_lo),
        np.array(corr_hi) - np.array(corr_means),
    ]
    ax.bar(x, corr_means, yerr=yerr, capsize=4, color=colors[: len(rep_order)])
    ax.set_xticks(x)
    ax.set_xticklabels(rep_order, rotation=20)
    ax.set_ylabel("drift correlation")
    ax.set_title("Held-out true generator drift score")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Fair two-scale double-well benchmark: train/tune on trajectories only")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    config = TwoScaleDoubleWellConfig()
    qv_err = leadlag_qv_sanity_check()
    if qv_err > 1e-12:
        raise RuntimeError(f"lead-lag QV sanity check failed: error={qv_err:g}")
    rep_order = [
        "raw_x",
        "delay_coords",
        "efm_dx",
        "leadlag_summary",
        "leadlag_summary_l3",
        "leadlag_logsig_l3",
        "leadlag_sig",
        "cum_leadlag_sig",
        "cum_leadlag_logsig_l3",
        "oracle_xh",
    ]
    grid = [
        HyperParams(alpha=a, length_multiplier=l)
        for a in (0.10, 0.50, 1.00)
        for l in (1.00, 2.00)
    ]
    seeds = [20260516, 20260517, 20260518, 20260519, 20260520]
    all_results: List[RepFairResult] = []
    for seed in seeds:
        print(f"running seed {seed}...", flush=True)
        all_results.extend(run_seed(seed, config, rep_order, grid))
    print_summary(all_results, rep_order, config)
    out_path = os.path.join(HERE, "two_scale_fair_benchmark.png")
    plot_summary(all_results, rep_order, out_path)
    print()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
