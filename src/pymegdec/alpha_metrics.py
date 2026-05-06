"""Exploratory alpha-band metrics for MEG trials."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal
from pymegdec.alpha_signal import get_data_field, get_time_vector, get_trial_signal
from pymegdec.data_config import resolve_data_folder
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

DEFAULT_OCCIPITAL_PATTERN = r"^M[LRZ]O"
DEFAULT_TIME_WINDOW = (-0.4, -0.05)
DEFAULT_FREQUENCY_RANGE = (8.0, 12.0)


@dataclass(frozen=True)
class AlphaMetricConfig:
    """Parameters controlling alpha metric extraction."""

    location_pattern: str = DEFAULT_OCCIPITAL_PATTERN
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW
    frequency_range: tuple[float, float] = DEFAULT_FREQUENCY_RANGE
    filter_order: int = 5


def _unwrap_singleton(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def _get_struct_field(value, field_name):
    if isinstance(value, dict):
        return value[field_name]
    if isinstance(value, np.void):
        return value[field_name]
    if isinstance(value, np.ndarray) and value.dtype.names:
        return value[field_name]
    raise TypeError(f"Cannot read field {field_name!r} from {type(value).__name__}.")


def _label_to_string(label):
    value = _unwrap_singleton(label)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_channel_names(data, n_channels=None):
    """Return channel names from a FieldTrip-like MATLAB structure."""

    labels = np.asarray(get_data_field(data, "label"), dtype=object).ravel()
    if n_channels is not None:
        labels = labels[:n_channels].ravel()
    return [_label_to_string(label) for label in labels]


def get_channel_positions(data, n_channels=None):
    """Return channel positions from ``data.grad.chanpos``."""

    grad = get_data_field(data, "grad")
    chanpos = _unwrap_singleton(_get_struct_field(grad, "chanpos"))
    positions: np.ndarray = np.asarray(chanpos, dtype=float)
    if n_channels is None:
        return positions
    return positions[:n_channels]


def select_channels(data, location_pattern=DEFAULT_OCCIPITAL_PATTERN):
    """Select channels whose labels match ``location_pattern``."""

    n_channels = get_trial_signal(data, 0).shape[0]
    pattern = re.compile(location_pattern)
    channel_names = get_channel_names(data, n_channels)
    return [index for index, channel_name in enumerate(channel_names) if pattern.search(channel_name)]


def project_sensor_positions(positions):
    """Project sensor positions to a 2D plane with PCA/SVD."""

    centered = np.asarray(positions, dtype=float) - np.mean(positions, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _delaunay_edges(coords2d):
    if len(coords2d) < 3:
        raise ValueError("At least three sensor positions are required.")

    triangulation = Delaunay(coords2d)
    edges = set()
    for simplex in triangulation.simplices:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edges.add(tuple(sorted((int(simplex[first]), int(simplex[second])))))

    edge_indices = np.array(sorted(edges), dtype=int)
    edge_vectors = coords2d[edge_indices[:, 1]] - coords2d[edge_indices[:, 0]]
    return edge_indices, edge_vectors, np.linalg.pinv(edge_vectors)


def _trial_label(data, trial_idx):
    if isinstance(data, dict) and "trialinfo" not in data:
        return np.nan
    if not isinstance(data, dict) and "trialinfo" not in data.dtype.names:
        return np.nan
    trialinfo = np.asarray(get_data_field(data, "trialinfo")).ravel()
    return trialinfo[trial_idx].item()


def count_trials(data):
    """Return the number of trials in a FieldTrip-like data structure."""

    trial_field = np.asarray(get_data_field(data, "trial"), dtype=object)
    if trial_field.ndim == 2 and trial_field.shape[0] == 1:
        return trial_field.shape[1]
    return len(trial_field.ravel())


def _time_mask(time_vector, time_window):
    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")
    mask = (time_vector >= start) & (time_vector <= stop)
    if not np.any(mask):
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    return mask


def _phase_gradient_metrics(phase, edge_indices, edge_vectors, edge_pinv, center_frequency):
    phase_delta = np.angle(np.exp(1j * (phase[edge_indices[:, 1], :] - phase[edge_indices[:, 0], :])))
    gradients = edge_pinv @ phase_delta
    predicted_delta = edge_vectors @ gradients
    residual = np.angle(np.exp(1j * (phase_delta - predicted_delta)))
    fit = np.abs(np.mean(np.exp(1j * residual), axis=0))
    gradient_norm = np.linalg.norm(gradients, axis=0)
    weights = fit + 1e-12
    mean_gradient = np.average(gradients, axis=1, weights=weights)
    valid = (fit > 0.5) & (gradient_norm > 1e-4)

    speed_m_per_s = np.nan
    if np.any(valid):
        # alpha phase velocity = angular frequency / spatial angular frequency.
        speed_m_per_s = np.nanmedian((2 * np.pi * center_frequency / gradient_norm[valid]) / 1000.0)

    return {
        "phase_plane_fit": float(np.mean(fit)),
        "spatial_freq_rad_per_mm": float(np.average(gradient_norm, weights=weights)),
        "speed_m_per_s": float(speed_m_per_s),
        "gradient_x": float(mean_gradient[0]),
        "gradient_y": float(mean_gradient[1]),
        "direction_rad": float(np.arctan2(mean_gradient[1], mean_gradient[0])),
    }


def _resolve_channel_indices(data, channel_indices, config):
    if channel_indices is None:
        channel_indices = select_channels(data, config.location_pattern)
    channel_indices = np.asarray(channel_indices, dtype=int)
    if channel_indices.size == 0:
        raise ValueError(f"No channels matched pattern: {config.location_pattern}")
    return channel_indices


def compute_alpha_analytic_window(signal, time_vector, config):
    """Return alpha-band analytic signal samples in ``config.time_window``."""

    sampling_rate = 1 / np.diff(time_vector[:2])[0]
    time_indices = np.flatnonzero(_time_mask(time_vector, config.time_window))
    low_freq, high_freq = config.frequency_range

    sos = scipy.signal.butter(
        config.filter_order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    alpha_signal = scipy.signal.sosfiltfilt(sos, signal, axis=-1)
    analytic_signal = scipy.signal.hilbert(alpha_signal, axis=-1)
    alpha_window = np.take(analytic_signal, time_indices, axis=-1)
    return alpha_window, time_indices


def _alpha_window_and_phase(signal, time_vector, config):
    alpha_window, _ = compute_alpha_analytic_window(signal, time_vector, config)
    return alpha_window, np.angle(alpha_window)


def _phase_geometry(data, channel_indices):
    positions = np.take(
        get_channel_positions(data, get_trial_signal(data, 0).shape[0]),
        channel_indices,
        axis=0,
    )
    coords2d = project_sensor_positions(positions)
    return _delaunay_edges(coords2d)


def compute_alpha_trial_metrics(
    data,
    trial_idx,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Compute exploratory prestimulus alpha metrics for one trial."""

    config = config or AlphaMetricConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config)
    time_vector = get_time_vector(data, trial_idx)
    signal = np.take(get_trial_signal(data, trial_idx), channel_indices, axis=0)
    alpha_window, phase = _alpha_window_and_phase(signal, time_vector, config)
    edge_indices, edge_vectors, edge_pinv = _phase_geometry(data, channel_indices)

    row = {
        "participant": participant_id if participant_id is not None else "",
        "dataset": dataset,
        "trial": trial_idx,
        "trial_label": _trial_label(data, trial_idx),
        "time_window_start": config.time_window[0],
        "time_window_stop": config.time_window[1],
        "low_freq": config.frequency_range[0],
        "high_freq": config.frequency_range[1],
        "n_channels": int(len(channel_indices)),
        "alpha_power": float(np.mean(np.abs(alpha_window) ** 2)),
        "log_alpha_power": float(np.mean(np.log(np.abs(alpha_window) ** 2 + 1e-12))),
        "phase_concentration": float(np.abs(np.mean(np.exp(1j * phase)))),
    }
    row.update(
        _phase_gradient_metrics(
            phase,
            edge_indices,
            edge_vectors,
            edge_pinv,
            center_frequency=sum(config.frequency_range) / 2,
        )
    )
    return row


def compute_alpha_metrics(
    data,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Compute alpha metrics for every trial in ``data``."""

    config = config or AlphaMetricConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config)

    n_trials = count_trials(data)
    return [
        compute_alpha_trial_metrics(
            data,
            trial_idx,
            participant_id=participant_id,
            dataset=dataset,
            channel_indices=channel_indices,
            config=config,
        )
        for trial_idx in range(n_trials)
    ]


def write_alpha_metrics_csv(rows, output_path):
    """Write alpha metric rows to ``output_path``."""

    if not rows:
        raise ValueError("At least one row is required.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_participant_data(data_folder, participant_id, *, cue=False):
    """Load a participant's main or cue MATLAB data file."""

    data_folder = resolve_data_folder(data_folder)
    suffix = "CueData" if cue else "Data"
    data_path = Path(data_folder) / f"Part{participant_id}{suffix}.mat"
    return sio.loadmat(data_path)["data"][0]


def export_participant_alpha_metrics(
    data_folder,
    participant_id,
    output_path,
    *,
    cue=False,
    config=None,
):
    """Load participant data, compute alpha metrics, and write a CSV."""

    config = config or AlphaMetricConfig()
    data = load_participant_data(data_folder, participant_id, cue=cue)
    dataset = "cue" if cue else "main"
    rows = compute_alpha_metrics(
        data,
        participant_id=participant_id,
        dataset=dataset,
        config=config,
    )
    write_alpha_metrics_csv(rows, output_path)
    return rows
