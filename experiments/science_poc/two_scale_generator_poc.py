r"""
Science-focused POC: RKHS generator learning under partial observation.

Benchmark
---------
We observe only X_t from a two-scale double-well system:

    dX_t = (X_t - X_t^3 + beta H_t) dt + sigma_X dW_X,
    dH_t = -kappa_H H_t dt + sigma_H dW_H.

The hidden OU factor H_t makes the observed coordinate non-Markovian.  The
POC compares generator drift learning on several states:

    raw_x      : X_t only
    delay_x    : X_t plus delayed finite-difference summaries
    efm_dx     : X_t plus fading-memory transforms of dX_t
    oracle_xh  : X_t and hidden H_t

The GP/KRR head is intentionally reused from the finance residual-control
tooling (`src.control.kernel_residual_control`).  The fading-memory lift is
reused from `src.control.state_transform`.

Outputs
-------
Printed diagnostics:
    - held-out generator residual against the true drift,
    - held-out drift correlation,
    - drift recovery when the same head is trained on noisy increments,
    - hidden-factor recoverability from the representation,
    - linear Koopman implied timescales of each representation.

Saved figure:
    experiments/science_poc/two_scale_generator_poc.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import iisignature

try:
    import signatory
    import torch
except ImportError:  # pragma: no cover - exercised only outside project env
    signatory = None
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.science_poc.envs.two_scale_double_well import (
    TwoScaleDoubleWellConfig,
    simulate_two_scale_double_well,
)
from examples.proof_of_concept.signature_features import RecurrentLeadLagLogSigMap
from src.control.kernel_residual_control import (
    GPResidualModel,
    RBFKernel,
    median_heuristic_length_scale,
)
from src.control.state_transform import EFMLevel1


@dataclass(frozen=True)
class FeatureScaler:
    """Standardize feature columns using training statistics."""

    mean: np.ndarray
    scale: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "FeatureScaler":
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0, ddof=1)
        scale = np.where(scale > 1e-10, scale, 1.0)
        return FeatureScaler(mean=mean, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.scale


@dataclass
class Representation:
    """Flattened representation and aligned targets."""

    name: str
    Z: np.ndarray
    Z_next: np.ndarray
    drift_true: np.ndarray
    dX_over_dt: np.ndarray
    H: np.ndarray


@dataclass
class FitResult:
    name: str
    dim: int
    drift_rmse: float
    drift_nrmse: float
    drift_corr: float
    generator_residual: float
    noisy_fit_corr: float
    noisy_fit_residual: float
    hidden_corr: float
    hidden_rmse: float
    top_timescales: List[float]
    y_true: np.ndarray
    y_pred: np.ndarray


def _flatten_time_major(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def _build_delay_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
    lag_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature [X_t, trailing slope over lag, previous trailing slope]."""

    n_paths, n_steps = X.shape
    Z = np.zeros((n_paths, n_steps, 3), dtype=float)
    csum = np.cumsum(dX, axis=1)
    for t in range(n_steps):
        t0 = max(0, t - lag_steps)
        t1 = max(0, t - 2 * lag_steps)
        recent = csum[:, t - 1] - (csum[:, t0 - 1] if t0 > 0 else 0.0) if t > 0 else 0.0
        previous = (
            csum[:, t0 - 1] - (csum[:, t1 - 1] if t1 > 0 else 0.0)
            if t0 > 0
            else 0.0
        )
        recent_denom = max((t - t0) * config.dt, config.dt)
        previous_denom = max((t0 - t1) * config.dt, config.dt)
        Z[:, t, 0] = X[:, t]
        Z[:, t, 1] = recent / recent_denom
        Z[:, t, 2] = previous / previous_denom
    return Z[:, :-1, :].reshape(-1, 3), Z[:, 1:, :].reshape(-1, 3)


def _build_delay_coordinate_features(
    X: np.ndarray,
    config: TwoScaleDoubleWellConfig,
    lag_steps: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard delay-coordinate baseline [X_t, X_{t-lag_1}, ...]."""

    n_paths, n_steps = X.shape
    dim = 1 + len(lag_steps)
    Z = np.zeros((n_paths, n_steps, dim), dtype=float)
    for t in range(n_steps):
        Z[:, t, 0] = X[:, t]
        for j, lag in enumerate(lag_steps, start=1):
            Z[:, t, j] = X[:, max(0, t - int(lag))]
    return Z[:, :-1, :].reshape(-1, dim), Z[:, 1:, :].reshape(-1, dim)


def _level2_bracket(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Antisymmetric level-2 bracket for vectors in lead-lag space."""

    d = a.size
    out = np.zeros(d * (d - 1) // 2, dtype=float)
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            out[idx] = a[i] * b[j] - a[j] * b[i]
            idx += 1
    return out


def _leadlag_window_from_snapshots(
    l1_past: np.ndarray,
    l2_past: np.ndarray,
    l1_now: np.ndarray,
    l2_now: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Recover a level-2 log-signature window by Chen/BCH inversion."""

    l1_win = l1_now - l1_past
    l2_win = l2_now - l2_past - 0.5 * _level2_bracket(l1_past, l1_win)
    return l1_win, l2_win


def _build_leadlag_signature_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
    window_steps: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Rolling lead-lag level-2 log-signatures of observed increments.

    Input stream is `(dt, dX_t)`.  Feature at time `t` uses only increments
    before `dX_t`, so it can fairly predict the next observed increment.
    """

    n_paths, n_steps = X.shape
    sig = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
    dim_l1 = sig.dim_l1
    dim_l2 = sig.dim_l2
    per_window_dim = dim_l1 + dim_l2
    # First column is always current X_t because the fair benchmark uses it
    # for transition diagnostics.  Each window then adds its initial endpoint
    # X_{t-m} plus the lead-lag log-signature over that window.
    feat_dim = 1 + len(window_steps) * (1 + per_window_dim)
    Z = np.zeros((n_paths, n_steps, feat_dim), dtype=float)

    # Pair ordering in 4D lead-lag space for input (time, x):
    # (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).  The observed-increment QV
    # channel is (x_lead, x_lag) = (1,3), index 4.
    qv_idx = 4

    for p in range(n_paths):
        state = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
        l1_snap = np.zeros((n_steps + 1, dim_l1), dtype=float)
        l2_snap = np.zeros((n_steps + 1, dim_l2), dtype=float)
        for t in range(n_steps):
            state.update(np.array([config.dt, float(dX[p, t])]))
            l1_snap[t + 1] = state.l1
            l2_snap[t + 1] = state.l2

        for t in range(n_steps):
            Z[p, t, 0] = X[p, t]
            offset = 1
            for w in window_steps:
                past = max(0, t - int(w))
                l1_win, l2_win = _leadlag_window_from_snapshots(
                    l1_snap[past], l2_snap[past], l1_snap[t], l2_snap[t],
                )
                scale = max(float(t - past) * config.dt, config.dt)
                l1_feat = l1_win / scale
                l2_feat = l2_win / scale
                # Put QV on variance-rate scale: 2*A / elapsed time.
                l2_feat = l2_feat.copy()
                l2_feat[qv_idx] = 2.0 * l2_win[qv_idx] / scale
                block = np.concatenate([l1_feat, l2_feat])
                Z[p, t, offset] = X[p, past]
                Z[p, t, offset + 1: offset + 1 + per_window_dim] = block
                offset += 1 + per_window_dim

    return Z[:, :-1, :].reshape(-1, feat_dim), Z[:, 1:, :].reshape(-1, feat_dim)


def _build_leadlag_signature_summary_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
    window_steps: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Low-dimensional semantic lead-lag signature features.

    This mirrors the finance filters' successful pattern: use the signature to
    expose stable semantic channels instead of throwing the full coordinate set
    at a small-data regression.  Per window we keep:

    - initial endpoint `X_{t-m}`;
    - displacement rate from level-1 ret lead;
    - QV rate from `2 * A(x_lead, x_lag) / elapsed`.
    """

    n_paths, n_steps = X.shape
    sig = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
    dim_l1 = sig.dim_l1
    dim_l2 = sig.dim_l2
    feat_dim = 1 + len(window_steps) * 3
    qv_idx = 4
    ret_lead_idx = 1
    Z = np.zeros((n_paths, n_steps, feat_dim), dtype=float)

    for p in range(n_paths):
        state = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
        l1_snap = np.zeros((n_steps + 1, dim_l1), dtype=float)
        l2_snap = np.zeros((n_steps + 1, dim_l2), dtype=float)
        for t in range(n_steps):
            state.update(np.array([config.dt, float(dX[p, t])]))
            l1_snap[t + 1] = state.l1
            l2_snap[t + 1] = state.l2

        for t in range(n_steps):
            Z[p, t, 0] = X[p, t]
            offset = 1
            for w in window_steps:
                past = max(0, t - int(w))
                l1_win, l2_win = _leadlag_window_from_snapshots(
                    l1_snap[past], l2_snap[past], l1_snap[t], l2_snap[t],
                )
                scale = max(float(t - past) * config.dt, config.dt)
                Z[p, t, offset] = X[p, past]
                Z[p, t, offset + 1] = l1_win[ret_lead_idx] / scale
                Z[p, t, offset + 2] = 2.0 * l2_win[qv_idx] / scale
                offset += 3

    return Z[:, :-1, :].reshape(-1, feat_dim), Z[:, 1:, :].reshape(-1, feat_dim)


def _build_leadlag_signature_summary_l3_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
    window_steps: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Anchored semantic level-3 path-summary features.

    This is deliberately not a full level-3 log-signature dump.  It extends
    the compact lead-lag summary with stable third-order channels:

    - cubic variation rate, `sum dX^3 / elapsed`;
    - time-ordered QV slope, showing whether roughness is early or late;
    - displacement-weighted QV, an `integral path_position d[X]`-type
      third-order interaction.
    """

    n_paths, n_steps = X.shape
    per_window_dim = 6
    feat_dim = 1 + len(window_steps) * per_window_dim
    Z = np.zeros((n_paths, n_steps, feat_dim), dtype=float)

    for p in range(n_paths):
        for t in range(n_steps):
            Z[p, t, 0] = X[p, t]
            offset = 1
            for w in window_steps:
                past = max(0, t - int(w))
                dx_win = dX[p, past:t]
                elapsed = max(float(t - past) * config.dt, config.dt)
                if dx_win.size == 0:
                    disp_rate = qv_rate = cubic_rate = qv_slope = disp_qv = 0.0
                else:
                    disp = np.cumsum(dx_win)
                    disp_prev = np.concatenate([[0.0], disp[:-1]])
                    qv = dx_win * dx_win
                    centered_time = (
                        np.linspace(-0.5, 0.5, dx_win.size)
                        if dx_win.size > 1
                        else np.array([0.0])
                    )
                    disp_rate = float(np.sum(dx_win) / elapsed)
                    qv_rate = float(np.sum(qv) / elapsed)
                    cubic_rate = float(np.sum(dx_win ** 3) / elapsed)
                    qv_slope = float(np.sum(centered_time * qv) / elapsed)
                    disp_qv = float(np.sum(disp_prev * qv) / elapsed)

                Z[p, t, offset] = X[p, past]
                Z[p, t, offset + 1] = disp_rate
                Z[p, t, offset + 2] = qv_rate
                Z[p, t, offset + 3] = cubic_rate
                Z[p, t, offset + 4] = qv_slope
                Z[p, t, offset + 5] = disp_qv
                offset += per_window_dim

    return Z[:, :-1, :].reshape(-1, feat_dim), Z[:, 1:, :].reshape(-1, feat_dim)


def _leadlag_paths_from_increment_arrays(increments: np.ndarray) -> np.ndarray:
    """Build lead-then-lag paths from batched `(dt, observed increment)` arrays."""

    increments = np.asarray(increments, dtype=float)
    if increments.ndim != 3 or increments.shape[2] != 2:
        raise ValueError("increments must have shape (batch, steps, 2)")

    path = np.concatenate(
        [
            np.zeros((increments.shape[0], 1, 2), dtype=float),
            np.cumsum(increments, axis=1),
        ],
        axis=1,
    )
    repeated = np.repeat(path, 2, axis=1)
    lead = repeated[:, 1:, :]
    lag = repeated[:, :-1, :]
    return np.concatenate([lead, lag], axis=2)


def _leadlag_path_from_increment_window(dx_win: np.ndarray, dt: float) -> np.ndarray:
    """Build the lead-then-lag path for a `(time, observed increment)` window."""

    dx_win = np.asarray(dx_win, dtype=float)
    increments = np.zeros((1, dx_win.size, 2), dtype=float)
    increments[0, :, 0] = float(dt)
    increments[0, :, 1] = dx_win
    return _leadlag_paths_from_increment_arrays(increments)[0]


def _logsignature_batch(paths: np.ndarray, depth: int) -> np.ndarray:
    """Compute log-signatures with signatory when available, else iisignature."""

    paths = np.asarray(paths, dtype=float)
    if paths.shape[0] == 0:
        dim = iisignature.logsiglength(paths.shape[2], depth)
        return np.zeros((0, dim), dtype=float)

    if signatory is not None and torch is not None:
        path_tensor = torch.as_tensor(paths, dtype=torch.float64)
        with torch.no_grad():
            logsig = signatory.logsignature(path_tensor, depth, mode="brackets")
        return logsig.detach().cpu().numpy()

    prep = iisignature.prepare(paths.shape[2], depth)
    return np.vstack([iisignature.logsig(path, prep) for path in paths])


def _logsignature_stream(paths: np.ndarray, depth: int) -> np.ndarray:
    """Compute prefix log-signatures for every non-initial path point."""

    paths = np.asarray(paths, dtype=float)
    if signatory is not None and torch is not None:
        path_tensor = torch.as_tensor(paths, dtype=torch.float64)
        with torch.no_grad():
            logsig = signatory.logsignature(
                path_tensor, depth, stream=True, mode="brackets",
            )
        return logsig.detach().cpu().numpy()

    prep = iisignature.prepare(paths.shape[2], depth)
    out = np.zeros(
        (paths.shape[0], paths.shape[1] - 1, iisignature.logsiglength(paths.shape[2], depth)),
        dtype=float,
    )
    for p, path in enumerate(paths):
        for t in range(1, path.shape[0]):
            out[p, t - 1] = iisignature.logsig(path[: t + 1], prep)
    return out


def _build_rolling_leadlag_logsig_l3_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
    window_steps: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""True rolling level-3 lead-lag log-signatures via `signatory`.

    Feature at time `t` uses only increments before `dX_t`.  We include
    observable endpoint anchors because signatures of increments are
    translation-invariant by construction.
    """

    n_paths, n_steps = X.shape
    sig_dim = iisignature.logsiglength(4, 3)
    feat_dim = 1 + len(window_steps) * (1 + sig_dim)
    Z = np.zeros((n_paths * n_steps, feat_dim), dtype=float)
    Z[:, 0] = X.reshape(-1)

    offset = 1
    for w in window_steps:
        increments = np.zeros((n_paths * n_steps, int(w), 2), dtype=float)
        anchors = np.zeros(n_paths * n_steps, dtype=float)
        row = 0
        for p in range(n_paths):
            for t in range(n_steps):
                k = min(int(w), t)
                past = t - k
                anchors[row] = X[p, past]
                if k > 0:
                    increments[row, int(w) - k :, 0] = config.dt
                    increments[row, int(w) - k :, 1] = dX[p, past:t]
                row += 1
        logsigs = _logsignature_batch(_leadlag_paths_from_increment_arrays(increments), depth=3)
        Z[:, offset] = anchors
        Z[:, offset + 1: offset + 1 + sig_dim] = logsigs
        offset += 1 + sig_dim

    Z = Z.reshape(n_paths, n_steps, feat_dim)
    return Z[:, :-1, :].reshape(-1, feat_dim), Z[:, 1:, :].reshape(-1, feat_dim)


def _build_cumulative_leadlag_logsig_l3_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""True cumulative level-3 lead-lag log-signatures via `signatory`."""

    n_paths, n_steps = X.shape
    sig_dim = iisignature.logsiglength(4, 3)
    feat_dim = 2 + sig_dim
    Z = np.zeros((n_paths, n_steps, feat_dim), dtype=float)
    Z[:, :, 0] = X
    Z[:, :, 1] = X[:, [0]]

    increments = np.zeros((n_paths, n_steps, 2), dtype=float)
    increments[:, :, 0] = config.dt
    increments[:, :, 1] = dX
    stream = _logsignature_stream(_leadlag_paths_from_increment_arrays(increments), depth=3)

    for t in range(1, n_steps):
        # signatory stream output omits the initial path point.  Prefix ending
        # at lead-lag point 2*t has stream index 2*t - 1.
        Z[:, t, 2:] = stream[:, 2 * t - 1, :]

    return Z[:, :-1, :].reshape(-1, feat_dim), Z[:, 1:, :].reshape(-1, feat_dim)


def _build_cumulative_leadlag_signature_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Cumulative lead-lag log-signature with observable endpoint anchors.

    Signature channels are increments and therefore do not carry absolute
    location by themselves.  The feature explicitly includes `(X_t, X_0)`
    before the cumulative lead-lag log-signature of `(dt, dX)`.
    """

    n_paths, n_steps = X.shape
    sig = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
    sig_dim = sig.feature_dim
    feat_dim = 2 + sig_dim
    Z = np.zeros((n_paths, n_steps, feat_dim), dtype=float)

    for p in range(n_paths):
        state = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
        for t in range(n_steps):
            elapsed = max(float(t) * config.dt, config.dt)
            sig_feat = state.get_features().copy()
            # Normalize cumulative signature coordinates to rates so the
            # feature scale does not grow trivially with path length.
            sig_feat[: state.dim_l1] = sig_feat[: state.dim_l1] / elapsed
            sig_feat[state.dim_l1:] = sig_feat[state.dim_l1:] / elapsed
            # QV channel in l2 index 4 gets variance-rate normalization.
            sig_feat[state.dim_l1 + 4] = 2.0 * state.l2[4] / elapsed
            # Current endpoint first; benchmark transition diagnostics assume
            # column 0 is the observed current coordinate.
            Z[p, t, 0] = X[p, t]
            Z[p, t, 1] = X[p, 0]
            Z[p, t, 2:] = sig_feat
            state.update(np.array([config.dt, float(dX[p, t])]))

    return Z[:, :-1, :].reshape(-1, feat_dim), Z[:, 1:, :].reshape(-1, feat_dim)


def leadlag_qv_sanity_check() -> float:
    """Return absolute error in `2 * A(x_lead, x_lag) = sum dx^2`."""

    increments = np.array([0.10, -0.20, 0.05, 0.30], dtype=float)
    state = RecurrentLeadLagLogSigMap(state_dim=1, level=2, forgetting_factor=1.0)
    for dx in increments:
        state.update(np.array([dx]))
    qv_from_sig = 2.0 * state.l2[0]
    qv_direct = float(np.sum(increments ** 2))
    return float(abs(qv_from_sig - qv_direct))


def _build_efm_features(
    X: np.ndarray,
    dX: np.ndarray,
    config: TwoScaleDoubleWellConfig,
    lambdas: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature [X_t, lambda_1 Z_1, ..., lambda_K Z_K] from EFM(dX)."""

    n_paths, n_steps = X.shape
    dim = 1 + len(lambdas)
    Z = np.zeros((n_paths, n_steps, dim), dtype=float)
    for p in range(n_paths):
        filters = [EFMLevel1(dim=1, lam=lam, name=f"efm_{lam:g}") for lam in lambdas]
        prev_dx = 0.0
        for t in range(n_steps):
            Z[p, t, 0] = X[p, t]
            for j, (lam, filt) in enumerate(zip(lambdas, filters), start=1):
                z = filt.update(np.array([prev_dx]), config.dt)[0]
                Z[p, t, j] = lam * z
            prev_dx = float(dX[p, t])
    return Z[:, :-1, :].reshape(-1, dim), Z[:, 1:, :].reshape(-1, dim)


def build_representations(
    sim: Dict[str, np.ndarray],
    config: TwoScaleDoubleWellConfig,
) -> Dict[str, Representation]:
    X = sim["X"]
    H = sim["H"]
    dX = sim["dX"]
    drift = sim["drift_X"]

    aligned_drift = drift[:, :-1].reshape(-1)
    aligned_dXdt = (dX[:, :-1] / config.dt).reshape(-1)
    aligned_H = H[:, :-1].reshape(-1)

    raw = X[:, :-1, None].reshape(-1, 1)
    raw_next = X[:, 1:, None].reshape(-1, 1)

    lag_steps = max(2, int(round(0.25 / config.dt)))
    delay, delay_next = _build_delay_features(X, dX, config, lag_steps=lag_steps)

    delay_coord_steps = [
        max(1, int(round(tau / config.dt)))
        for tau in (0.10, 0.25, 0.50)
    ]
    delay_coords, delay_coords_next = _build_delay_coordinate_features(
        X, config, delay_coord_steps,
    )

    efm_lambdas = [0.75, config.kappa_H, 4.0]
    efm, efm_next = _build_efm_features(X, dX, config, lambdas=efm_lambdas)

    sig_windows = [10, 25, 50]
    leadlag_sig, leadlag_sig_next = _build_leadlag_signature_features(
        X, dX, config, window_steps=sig_windows,
    )
    leadlag_summary, leadlag_summary_next = _build_leadlag_signature_summary_features(
        X, dX, config, window_steps=sig_windows,
    )
    leadlag_summary_l3, leadlag_summary_l3_next = _build_leadlag_signature_summary_l3_features(
        X, dX, config, window_steps=sig_windows,
    )
    leadlag_logsig_l3, leadlag_logsig_l3_next = _build_rolling_leadlag_logsig_l3_features(
        X, dX, config, window_steps=sig_windows,
    )
    cum_leadlag_logsig_l3, cum_leadlag_logsig_l3_next = _build_cumulative_leadlag_logsig_l3_features(
        X, dX, config,
    )
    cum_leadlag_sig, cum_leadlag_sig_next = _build_cumulative_leadlag_signature_features(
        X, dX, config,
    )

    oracle = np.stack([X[:, :-1], H[:, :-1]], axis=-1).reshape(-1, 2)
    oracle_next = np.stack([X[:, 1:], H[:, 1:]], axis=-1).reshape(-1, 2)

    reps = {
        "raw_x": (raw, raw_next),
        "delay_x": (delay, delay_next),
        "delay_coords": (delay_coords, delay_coords_next),
        "efm_dx": (efm, efm_next),
        "leadlag_sig": (leadlag_sig, leadlag_sig_next),
        "leadlag_summary": (leadlag_summary, leadlag_summary_next),
        "leadlag_summary_l3": (leadlag_summary_l3, leadlag_summary_l3_next),
        "leadlag_logsig_l3": (leadlag_logsig_l3, leadlag_logsig_l3_next),
        "cum_leadlag_sig": (cum_leadlag_sig, cum_leadlag_sig_next),
        "cum_leadlag_logsig_l3": (cum_leadlag_logsig_l3, cum_leadlag_logsig_l3_next),
        "oracle_xh": (oracle, oracle_next),
    }
    return {
        name: Representation(
            name=name,
            Z=Z,
            Z_next=Z_next,
            drift_true=aligned_drift,
            dX_over_dt=aligned_dXdt,
            H=aligned_H,
        )
        for name, (Z, Z_next) in reps.items()
    }


def _subsample_indices(n: int, size: int, rng: np.random.RandomState) -> np.ndarray:
    size = min(int(size), int(n))
    return rng.choice(np.arange(n), size=size, replace=False)


def _fit_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> Tuple[FeatureScaler, GPResidualModel]:
    scaler = FeatureScaler.fit(X_train)
    Xs = scaler.transform(X_train)
    ls = median_heuristic_length_scale(Xs)
    kernel = RBFKernel(length_scale=ls, amplitude_sq=1.0)
    model = GPResidualModel(kernel=kernel, alpha=alpha).fit(Xs, y_train, standardize=True)
    return scaler, model


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _koopman_timescales(
    Z: np.ndarray,
    Z_next: np.ndarray,
    dt: float,
    max_out: int = 3,
    ridge: float = 1e-6,
) -> List[float]:
    """Ridge linear Koopman map Z_next ~= A Z and implied timescales."""

    scaler = FeatureScaler.fit(Z)
    X = scaler.transform(Z)
    Y = scaler.transform(Z_next)
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    A_t = np.linalg.solve(XtX, X.T @ Y)
    eigvals = np.linalg.eigvals(A_t.T)
    times = []
    for eig in eigvals:
        mag = float(np.abs(eig))
        if 1e-6 < mag < 0.999999:
            tau = -dt / np.log(mag)
            if np.isfinite(tau) and tau > 0:
                times.append(float(tau))
    times.sort(reverse=True)
    return times[:max_out]


def fit_representation(
    rep: Representation,
    rng: np.random.RandomState,
    dt: float,
    train_size: int = 900,
    test_size: int = 1200,
) -> FitResult:
    n = rep.Z.shape[0]
    perm = rng.permutation(n)
    train_idx = perm[: min(train_size, n)]
    test_idx = perm[min(train_size, n): min(train_size + test_size, n)]
    if test_idx.size == 0:
        test_idx = _subsample_indices(n, test_size, rng)

    X_train = rep.Z[train_idx]
    X_test = rep.Z[test_idx]

    # Main generator diagnostic: this science benchmark has known generator
    # action on the coordinate observable f(x, h)=x.  Fitting that known action
    # isolates whether the representation contains the missing Markov state.
    scaler_drift, drift_model = _fit_gp(X_train, rep.drift_true[train_idx], alpha=0.02)
    drift_pred, _ = drift_model.posterior(scaler_drift.transform(X_test), include_noise=False)

    true_drift = rep.drift_true[test_idx]
    drift_err = drift_pred - true_drift
    drift_rmse = float(np.sqrt(np.mean(drift_err ** 2)))
    drift_scale = float(np.std(true_drift, ddof=1))
    drift_nrmse = float(drift_rmse / max(drift_scale, 1e-12))
    generator_residual = float(np.mean(drift_err ** 2) / max(np.mean(true_drift ** 2), 1e-12))

    # Secondary harder diagnostic: fit the same GP/KRR head on noisy one-step
    # increments, then score against the known drift.  This is closer to the
    # pure data-driven setting and is intentionally noisier.
    noisy_scaler, noisy_model = _fit_gp(X_train, rep.dX_over_dt[train_idx], alpha=0.20)
    noisy_pred, _ = noisy_model.posterior(noisy_scaler.transform(X_test), include_noise=False)
    noisy_err = noisy_pred - true_drift
    noisy_fit_residual = float(np.mean(noisy_err ** 2) / max(np.mean(true_drift ** 2), 1e-12))

    hidden_scaler, hidden_model = _fit_gp(X_train, rep.H[train_idx], alpha=0.05)
    hidden_pred, _ = hidden_model.posterior(hidden_scaler.transform(X_test), include_noise=False)
    hidden_true = rep.H[test_idx]
    hidden_rmse = float(np.sqrt(np.mean((hidden_pred - hidden_true) ** 2)))

    # Use a modest subsample for the linear spectral diagnostic.
    spec_idx = _subsample_indices(n, size=4000, rng=rng)
    top_times = _koopman_timescales(
        rep.Z[spec_idx], rep.Z_next[spec_idx], dt=dt, max_out=3,
    )

    return FitResult(
        name=rep.name,
        dim=rep.Z.shape[1],
        drift_rmse=drift_rmse,
        drift_nrmse=drift_nrmse,
        drift_corr=_safe_corr(drift_pred, true_drift),
        generator_residual=generator_residual,
        noisy_fit_corr=_safe_corr(noisy_pred, true_drift),
        noisy_fit_residual=noisy_fit_residual,
        hidden_corr=_safe_corr(hidden_pred, hidden_true),
        hidden_rmse=hidden_rmse,
        top_timescales=top_times,
        y_true=true_drift,
        y_pred=drift_pred,
    )


def _format_times(times: Sequence[float]) -> str:
    if not times:
        return "[]"
    return "[" + ", ".join(f"{t:.2f}" for t in times) + "]"


def print_results(results: Sequence[FitResult], config: TwoScaleDoubleWellConfig) -> None:
    print("=" * 108)
    print("TWO-SCALE DOUBLE-WELL RKHS GENERATOR POC")
    print("=" * 108)
    print(
        f"dt={config.dt} T={config.T} beta={config.beta} sigma_X={config.sigma_X} "
        f"kappa_H={config.kappa_H} hidden_timescale={config.hidden_timescale:.3f}"
    )
    print(
        "Target: recover the known generator action on f(x,h)=x from each representation."
    )
    print("Secondary column `noisy_corr` fits the same GP/KRR head on dX/dt targets.")
    print("-" * 108)
    header = (
        "rep          dim  drift_corr  drift_nrmse  gen_resid  noisy_corr  "
        "hidden_corr  hidden_rmse  koopman_times"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:12s} {r.dim:3d}  {r.drift_corr:+10.3f}  {r.drift_nrmse:11.3f}  "
            f"{r.generator_residual:9.3f}  {r.noisy_fit_corr:+10.3f}  "
            f"{r.hidden_corr:+11.3f}  {r.hidden_rmse:11.3f}  "
            f"{_format_times(r.top_timescales)}"
        )
    print("-" * 108)
    raw = next(r for r in results if r.name == "raw_x")
    for r in results:
        if r.name == "raw_x":
            continue
        gain = raw.generator_residual / max(r.generator_residual, 1e-12)
        print(f"generator residual improvement raw_x / {r.name}: {gain:.2f}x")


def plot_results(results: Sequence[FitResult], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax = axes[0]
    names = [r.name for r in results]
    vals = [r.generator_residual for r in results]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    ax.bar(names, vals, color=colors[: len(names)])
    ax.set_ylabel("normalized generator residual")
    ax.set_title("Generator residual: lower is better")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    for r, color in zip(results, colors):
        idx = np.linspace(0, r.y_true.size - 1, min(350, r.y_true.size)).astype(int)
        ax.scatter(
            r.y_true[idx],
            r.y_pred[idx],
            s=10,
            alpha=0.45,
            label=f"{r.name} corr={r.drift_corr:.2f}",
            color=color,
        )
    lo = min(float(np.min(r.y_true)) for r in results)
    hi = max(float(np.max(r.y_true)) for r in results)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax.set_xlabel("true generator drift")
    ax.set_ylabel("learned GP/KRR drift")
    ax.set_title("Held-out drift recovery")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Partially observed two-scale double-well: path lifts repair generator state")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    config = TwoScaleDoubleWellConfig()
    sim = simulate_two_scale_double_well(config, n_paths=28, seed=20260516)
    reps = build_representations(sim, config)
    rng = np.random.RandomState(20260517)
    order = ["raw_x", "delay_x", "efm_dx", "oracle_xh"]
    results = [fit_representation(reps[name], rng=rng, dt=config.dt) for name in order]
    print_results(results, config)
    out_path = os.path.join(HERE, "two_scale_generator_poc.png")
    plot_results(results, out_path)
    print()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
