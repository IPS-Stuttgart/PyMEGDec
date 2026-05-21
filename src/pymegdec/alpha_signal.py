"""Compatibility helpers for alpha-band signal extraction.

Generic filtering, Hilbert-phase extraction, sampling-rate validation, and
phase averaging are delegated to NeuRepTrace.  This module keeps only the
FieldTrip/MATLAB cell-array accessors and PyMEGDec's historical
``extract_time_basis`` convenience wrapper.
"""

from __future__ import annotations

import numpy as np
from neureptrace.signal.band import (
    average_phases,
    extract_phase,
    sampling_rate_from_time_vector,
)


def get_data_field(data, field_name):
    if isinstance(data, dict):
        return data[field_name]

    field = data[field_name]
    if isinstance(field, np.ndarray) and field.size == 1:
        return field.item()
    return field


def _unwrap_outer_cell_array(cell_array):
    values = np.asarray(cell_array, dtype=object)
    while values.dtype == object and values.size == 1:
        item = values.item()
        item_array = np.asarray(item)
        if not isinstance(item, np.ndarray) or item_array.dtype != object:
            break
        values = np.asarray(item, dtype=object)
    return values


def _cell_item(cell_array, index):
    values = _unwrap_outer_cell_array(cell_array)
    if values.ndim == 0:
        return values.item()
    if values.ndim == 2 and values.shape[0] == 1:
        return values[0, index]
    if values.ndim == 2 and values.shape[1] == 1:
        return values[index, 0]
    return values[index]


def get_time_vector(data, trial_idx=0):
    time_vector = _cell_item(get_data_field(data, "time"), trial_idx)
    return np.asarray(time_vector, dtype=float).ravel()


def get_trial_signal(data, trial_idx=0):
    trial_signal = _cell_item(get_data_field(data, "trial"), trial_idx)
    return np.asarray(trial_signal, dtype=float)


def _validated_trial_signal(data, trial_idx, time_vector):
    signal = get_trial_signal(data, trial_idx)
    if signal.ndim != 2:
        raise ValueError(f"Trial {trial_idx} must be a 2D channels-by-time array.")
    if signal.shape[1] != time_vector.size:
        raise ValueError(
            f"Trial {trial_idx} has {signal.shape[1]} samples but its time vector has "
            f"{time_vector.size} entries."
        )
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Trial {trial_idx} signal must contain only finite values.")
    return signal


def _parse_channel_index(value):
    if not isinstance(value, (int, np.integer)):
        raise ValueError("channel_range must contain integer channel indices.")
    return int(value)


def _channel_indices_from_range(channel_range, n_channels):
    try:
        start, stop = channel_range
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_range must contain exactly two integer indices.") from exc

    start = _parse_channel_index(start)
    stop = _parse_channel_index(stop)
    if start > stop:
        raise ValueError("channel_range start must be less than or equal to stop.")
    if start < 0 or stop >= int(n_channels):
        raise ValueError(
            "channel_range is outside the available channels: "
            f"got ({start}, {stop}) for {n_channels} channels."
        )
    return range(start, stop + 1)


def extract_time_basis(data, trial_idx=0, channel_range=(187, 198)):
    """
    Extract a robust alpha-phase time basis across multiple channels.

    Generic alpha filtering and Hilbert phase extraction are implemented in
    :mod:`neureptrace.signal.band`; this wrapper keeps PyMEGDec's historical
    FieldTrip/MATLAB trial and channel-range conventions.
    """

    time_vector = get_time_vector(data, trial_idx)
    sampling_rate = sampling_rate_from_time_vector(time_vector)
    signal = _validated_trial_signal(data, trial_idx, time_vector)
    channel_indices = _channel_indices_from_range(channel_range, signal.shape[0])

    phases = []
    for channel_idx in channel_indices:
        signal_curr_chan = signal[channel_idx, :]
        phase = extract_phase(signal_curr_chan, sampling_rate)
        phases.append(phase)

    mean_phase = average_phases(phases)
    return mean_phase


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.io as sio

    demo_data_folder = r"."
    demo_part = 2
    demo_data = sio.loadmat(f"{demo_data_folder}/Part{demo_part}Data.mat")["data"][0]

    demo_time_basis = extract_time_basis(demo_data, trial_idx=0, channel_range=(187, 198))

    print("Robust time basis (average phase):", demo_time_basis)

    demo_time_vector = get_time_vector(demo_data)
    plt.plot(demo_time_vector, demo_time_basis, label="Average Phase")
    plt.title("Average Alpha Phase Across Channels 187-198")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.show()
