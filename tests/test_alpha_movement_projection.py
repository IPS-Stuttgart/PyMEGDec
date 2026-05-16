import numpy as np

import pymegdec  # noqa: F401  # Import package so projection patches are applied.
from pymegdec import alpha_movement


def _cell(values):
    out = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        out[0, index] = value
    return out


def _projection_data(positions, labels):
    trial = np.zeros((len(labels), 2))
    time = np.array([[0.0, 0.01]])
    return {
        "label": np.asarray(labels, dtype=object)[:, None],
        "trial": _cell([trial]),
        "time": _cell([time]),
        "trialinfo": np.array([[1]]),
        "grad": {"chanpos": np.asarray(positions, dtype=float)},
    }


def test_alpha_movement_uses_common_projection_reference_frame_for_subsets():
    positions = np.array(
        [
            [-20.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [100.0, 20.0, 0.0],
            [1000.0, 1000.0, 0.0],
        ]
    )
    data = _projection_data(
        positions,
        ["MLO11", "MRO11", "MZO01", "MLF11", "EEG001"],
    )
    selected = np.array([0, 1, 2])

    geometry = alpha_movement._selected_geometry(
        data,
        data["trial"][0, 0],
        selected,
        "mm",
    )

    expected = positions[selected, :2] - np.mean(positions[:4, :2], axis=0)
    np.testing.assert_allclose(geometry.projected_positions, expected, rtol=1e-12, atol=1e-12)
