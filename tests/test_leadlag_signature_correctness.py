import sys
from pathlib import Path

import numpy as np
import iisignature

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.proof_of_concept.signature_features import (
    RecurrentLeadLagLogSigMap,
    compute_lead_lag_path,
    compute_lead_lag_signature,
    compute_log_signature,
    compute_path_signature,
    compute_signature_level_3,
)
from experiments.science_poc.two_scale_generator_poc import (
    _leadlag_path_from_increment_window,
    _leadlag_paths_from_increment_arrays,
    _leadlag_window_from_snapshots,
    _logsignature_batch,
    _logsignature_stream,
)
from src.sskf.online_path_features import RandomProjectionNystrom


def test_batch_lead_lag_logsig_has_positive_qv_orientation():
    path = np.array([0.0, 0.10, -0.10, -0.05, 0.25])
    qv_direct = float(np.sum(np.diff(path) ** 2))

    lead_lag_path = compute_lead_lag_path(path)
    assert np.allclose(lead_lag_path[1], [path[1], path[0]])

    logsig = compute_lead_lag_signature(path, level=2)
    assert np.isclose(2.0 * logsig[2], qv_direct)


def test_full_signature_helpers_match_iisignature():
    path = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.5, -0.2],
            [0.7, 0.4],
        ],
        dtype=float,
    )

    assert np.allclose(compute_path_signature(path, level=2), iisignature.sig(path, 2))

    prep = iisignature.prepare(2, 3)
    assert np.allclose(compute_log_signature(path, level=3), iisignature.logsig(path, prep))
    assert np.allclose(compute_signature_level_3(path), iisignature.sig(path, 3))


def test_recurrent_lead_lag_logsig_has_positive_qv_orientation():
    increments = np.array([0.10, -0.20, 0.05, 0.30])
    state = RecurrentLeadLagLogSigMap(state_dim=1, level=2, forgetting_factor=1.0)

    for dx in increments:
        state.update(np.array([dx]))

    assert np.isclose(2.0 * state.l2[0], np.sum(increments ** 2))


def test_time_return_qv_channel_index_is_pair_ret_lead_ret_lag():
    increments = np.array([0.10, -0.20, 0.05, 0.30])
    state = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)

    for dx in increments:
        state.update(np.array([0.01, dx]))

    # Pair ordering in 4D is (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
    # The return lead/lag pair is therefore level-2 index 4, or full index 8.
    assert np.isclose(2.0 * state.l2[4], np.sum(increments ** 2))
    assert np.isclose(2.0 * state.get_features()[8], np.sum(increments ** 2))


def test_true_level3_lead_lag_path_preserves_qv_channel():
    increments = np.array([0.10, -0.20, 0.05, 0.30])
    lead_lag_path = _leadlag_path_from_increment_window(increments, dt=0.01)

    prep = iisignature.prepare(4, 3)
    logsig = iisignature.logsig(lead_lag_path, prep)
    batch_logsig = _logsignature_batch(lead_lag_path[None, :, :], depth=3)[0]

    assert logsig.shape == (iisignature.logsiglength(4, 3),)
    assert np.allclose(batch_logsig, logsig)
    assert np.isclose(2.0 * logsig[8], np.sum(increments ** 2))


def test_cumulative_lead_lag_stream_prefixes_preserve_qv_channel():
    increments = np.array([0.10, -0.20, 0.05, 0.30])
    increment_array = np.zeros((1, increments.size, 2), dtype=float)
    increment_array[0, :, 0] = 0.01
    increment_array[0, :, 1] = increments

    lead_lag_path = _leadlag_paths_from_increment_arrays(increment_array)
    stream = _logsignature_stream(lead_lag_path, depth=3)

    for t in range(1, increments.size + 1):
        assert np.isclose(2.0 * stream[0, 2 * t - 1, 8], np.sum(increments[:t] ** 2))


def test_chen_level2_window_recovery_preserves_qv():
    increments = np.array([0.10, -0.20, 0.05, 0.30])
    state = RecurrentLeadLagLogSigMap(state_dim=2, level=2, forgetting_factor=1.0)
    l1_snap = [state.l1.copy()]
    l2_snap = [state.l2.copy()]

    for dx in increments:
        state.update(np.array([0.01, dx]))
        l1_snap.append(state.l1.copy())
        l2_snap.append(state.l2.copy())

    l1_win, l2_win = _leadlag_window_from_snapshots(
        l1_snap[1],
        l2_snap[1],
        l1_snap[4],
        l2_snap[4],
    )
    assert np.isclose(l1_win[1], np.sum(increments[1:4]))
    assert np.isclose(2.0 * l2_win[4], np.sum(increments[1:4] ** 2))


def test_online_path_features_lead_lag_transform_has_positive_qv_orientation():
    path = np.column_stack(
        [
            np.arange(5, dtype=float),
            np.array([0.0, 0.10, -0.10, -0.05, 0.25]),
        ]
    )
    extractor = RandomProjectionNystrom(dim=2, depth=2, use_leadlag=True)
    lead_lag_path = extractor._lead_lag_transform(path)

    assert np.allclose(lead_lag_path[1, :2], path[1])
    assert np.allclose(lead_lag_path[1, 2:], path[0])

    prep = iisignature.prepare(4, 2)
    logsig = iisignature.logsig(lead_lag_path, prep)
    qv_direct = float(np.sum(np.diff(path[:, 1]) ** 2))
    assert np.isclose(2.0 * logsig[8], qv_direct)
