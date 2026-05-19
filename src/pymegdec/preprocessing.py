"""Feature preprocessing helpers for MEG decoding data."""

import copy

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi


def preprocess_features(
    data,
    frequency_range,
    new_framerate,
    window_size,
    train_window_center,
    null_window_center,
    *,
    filter_phase="zero_phase",
):
    data = _copy_preprocessing_data(data)
    _require_antialias_filter_for_downsampling(data, frequency_range[1], new_framerate)
    data = _filter_features_inplace(data, frequency_range[0], frequency_range[1], filter_phase=filter_phase)
    if new_framerate != float("inf"):
        data = _downsample_data_inplace(data, new_framerate)

    train_window = (
        train_window_center - window_size / 2,
        train_window_center + window_size / 2,
    )
    null_time_window = (null_window_center - window_size / 2, null_window_center + window_size / 2) if not np.isnan(null_window_center) else (np.nan, np.nan)
    _require_null_window_before_train_window(train_window, null_time_window)

    stimuli_features_cell, null_features_cell = extract_windows(data, train_window, null_time_window)
    return stimuli_features_cell, null_features_cell


def filter_features(data, low_freq, high_freq, *, filter_phase="zero_phase"):
    return _filter_features_inplace(_copy_preprocessing_data(data), low_freq, high_freq, filter_phase=filter_phase)


def downsample_data(data, new_framerate):
    return _downsample_data_inplace(_copy_preprocessing_data(data), new_framerate)


def _copy_preprocessing_data(data):
    """Return ``data`` with independent trial/time cell arrays.

    SciPy-loaded MATLAB structs keep the trial and time cell arrays inside object
    arrays. A shallow copy of the struct/dict alone would keep those leaf arrays
    shared, so subsequent preprocessing could still mutate the caller's data.
    """

    try:
        copied = data.copy()
    except AttributeError:
        copied = copy.copy(data)

    for key in ("trial", "time"):
        copied[key] = _copy_nested_array(data[key])
    return copied


def _copy_nested_array(value):
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value.copy()
        copied = np.empty(value.shape, dtype=object)
        for index, element in np.ndenumerate(value):
            copied[index] = _copy_nested_array(element)
        return copied
    return copy.deepcopy(value)


def _filter_features_inplace(data, low_freq, high_freq, *, filter_phase="zero_phase"):
    filter_phase = _normalize_filter_phase(filter_phase)
    sample_interval = _common_uniform_sample_interval(data)
    sample_rate = float(1 / sample_interval)

    if low_freq < 0:
        raise ValueError("Low frequency must be greater than or equal to 0")
    if high_freq < 0:
        raise ValueError("High frequency must be greater than or equal to 0")
    if high_freq < low_freq:
        raise ValueError("High frequency must be greater than or equal to low frequency")

    if low_freq == 0 and high_freq == float("inf"):
        return data
    if low_freq == 0:
        b, a = butter(4, high_freq / (sample_rate / 2), "low")
    elif high_freq != float("inf"):
        cutoff = [low_freq / (sample_rate / 2), high_freq / (sample_rate / 2)]
        b, a = butter(4, cutoff, "bandpass")
    else:
        raise ValueError("Highpass filter not supported.")

    for i in range(_trial_count(data)):
        trial, _time = _trial_and_time(data, i)
        data["trial"][0][0][i] = _apply_iir_filter_to_trial(b, a, trial, filter_phase)
    return data


def _normalize_filter_phase(filter_phase):
    if filter_phase in {"zero_phase", "zero-phase", "filtfilt"}:
        return "zero_phase"
    if filter_phase in {"causal", "forward", "lfilter"}:
        return "causal"
    raise ValueError(
        "filter_phase must be 'zero_phase' for offline zero-phase filtering "
        "or 'causal' to avoid future-sample leakage in timing analyses."
    )


def _apply_iir_filter_to_trial(b, a, trial, filter_phase):
    samples_by_channel = trial.T
    if filter_phase == "zero_phase":
        return filtfilt(b, a, samples_by_channel, axis=0).T

    zi = lfilter_zi(b, a)[:, None] * samples_by_channel[0:1, :]
    filtered, _ = lfilter(b, a, samples_by_channel, axis=0, zi=zi)
    return filtered.T


def _require_antialias_filter_for_downsampling(data, high_freq, new_framerate):
    """Reject preprocessing configurations that can alias during downsampling."""

    if new_framerate == float("inf"):
        return
    if new_framerate <= 0:
        return

    sample_interval = _common_uniform_sample_interval(data)
    raw_framerate = float(1 / sample_interval)
    if new_framerate >= raw_framerate or np.isclose(new_framerate, raw_framerate, rtol=1e-6, atol=1e-12):
        return

    target_nyquist = new_framerate / 2
    if high_freq == float("inf") or high_freq > target_nyquist:
        raise ValueError(
            "Downsampling requires a low-pass frequency_range with high frequency "
            f"<= the new Nyquist frequency ({target_nyquist:g} Hz); got high frequency {high_freq:g} Hz for new_framerate={new_framerate:g} Hz."
        )


def _downsample_data_inplace(data, new_framerate):
    if new_framerate <= 0:
        raise ValueError("New framerate must be positive.")

    sample_interval = _common_uniform_sample_interval(data)
    raw_fs = round(float(1 / sample_interval))
    if new_framerate != raw_fs:
        step = 1 / new_framerate
        for i in range(_trial_count(data)):
            trial, time = _trial_and_time(data, i)
            new_t = _regular_time_grid_within_support(time, step)
            interpolator = interp1d(
                time,
                trial,
                axis=1,
                bounds_error=True,
            )
            data["trial"][0][0][i] = interpolator(new_t)
            data["time"][0][0][i] = new_t[None]
    return data


def extract_windows(data, train_window, null_time_window):
    if train_window[1] < train_window[0]:
        raise ValueError("Train window stop must be after train window start")

    null_requested = not np.isnan(null_time_window).all()
    if null_requested and null_time_window[1] > 0:
        raise ValueError("Null window should not contain positive time points")
    if null_requested and null_time_window[1] - null_time_window[0] < 0:
        raise ValueError("Invalid null window")
    if null_requested:
        _require_null_window_before_train_window(train_window, null_time_window)

    stimuli_features_cell = []
    null_features_cell = []
    n_train_values = None
    n_null_values = None
    for trial_idx in range(_trial_count(data)):
        trial, time = _trial_and_time(data, trial_idx)
        train_slice = _nearest_window_slice(time, train_window, trial_idx, "train")
        train_feature = trial[:, train_slice].reshape(-1, 1, order="F")
        n_train_values = _require_consistent_feature_size(
            train_feature,
            n_train_values,
            trial_idx,
            "train",
        )
        stimuli_features_cell.append(train_feature)

        if null_requested:
            _require_rounded_null_window_before_train_slice(
                time,
                train_window,
                train_slice,
                null_time_window,
                trial_idx,
            )
            null_slice = _matching_sample_window_slice(
                time,
                null_time_window[0],
                train_slice.stop - train_slice.start,
                trial_idx,
                "null",
            )
            _require_disjoint_window_slices(train_slice, null_slice, trial_idx)
            null_feature = trial[:, null_slice].reshape(-1, 1, order="F")
            n_null_values = _require_consistent_feature_size(
                null_feature,
                n_null_values,
                trial_idx,
                "null",
            )
            null_features_cell.append(null_feature)

    return stimuli_features_cell, null_features_cell


def _require_null_window_before_train_window(train_window, null_time_window):
    if np.isnan(null_time_window).all():
        return
    if null_time_window[1] > train_window[0]:
        raise ValueError("Null window must not extend past train window start")


def _require_disjoint_window_slices(train_slice, null_slice, trial_idx):
    if null_slice.start < train_slice.stop and train_slice.start < null_slice.stop:
        raise ValueError(f"Null window selects samples that overlap the train window for trial {trial_idx}.")


def _require_rounded_null_window_before_train_slice(time, train_window, train_slice, null_time_window, trial_idx):
    if np.isnan(null_time_window).all() or null_time_window[1] >= train_window[0]:
        return
    rounded_stop = int(np.argmin(np.abs(time - null_time_window[1])))
    if rounded_stop >= train_slice.start:
        raise ValueError(f"Null window selects samples that overlap the train window for trial {trial_idx}.")


def _trial_count(data):
    trials = data["trial"][0][0]
    times = data["time"][0][0]
    if len(trials) != len(times):
        raise ValueError("Number of trial and time entries must match.")
    return len(trials)


def _trial_and_time(data, trial_idx):
    trial = np.asarray(data["trial"][0][0][trial_idx], dtype=float)
    time = _time_vector(data, trial_idx)
    if trial.ndim != 2:
        raise ValueError(f"Trial {trial_idx} must be a 2D channels-by-time array.")
    if trial.shape[1] != time.size:
        raise ValueError(
            f"Trial {trial_idx} has {trial.shape[1]} samples but its time vector has "
            f"{time.size} entries."
        )
    return trial, time


def _time_vector(data, trial_idx):
    time = np.asarray(data["time"][0][0][trial_idx], dtype=float).ravel()
    if time.size == 0:
        raise ValueError(f"Time vector for trial {trial_idx} is empty or not provided correctly.")
    if time.size > 1 and np.any(np.diff(time) <= 0):
        raise ValueError(f"Time vector for trial {trial_idx} must be strictly increasing.")
    return time


def _common_uniform_sample_interval(data):
    reference_interval = None
    for trial_idx in range(_trial_count(data)):
        _trial, time = _trial_and_time(data, trial_idx)
        interval = _uniform_sample_interval(time, trial_idx)
        if reference_interval is None:
            reference_interval = interval
        elif not np.isclose(interval, reference_interval, rtol=1e-6, atol=1e-12):
            raise ValueError("All trials must have the same sampling interval before filtering or downsampling.")
    return reference_interval


def _uniform_sample_interval(time, trial_idx):
    if time.size < 2:
        raise ValueError(f"Time vector for trial {trial_idx} must contain at least two samples.")
    diffs = np.diff(time)
    interval = float(np.median(diffs))
    if not np.allclose(diffs, interval, rtol=1e-6, atol=1e-12):
        raise ValueError(f"Time vector for trial {trial_idx} must be uniformly sampled.")
    return interval


def _regular_time_grid_within_support(time, step):
    n_samples = int(np.floor((time[-1] - time[0]) / step + 1e-9)) + 1
    if n_samples <= 0:
        raise ValueError("Cannot downsample an empty time range.")
    new_t = time[0] + np.arange(n_samples, dtype=float) * step
    return new_t


def _nearest_window_slice(time, time_window, trial_idx, window_name):
    start, stop = time_window
    _require_window_supported(time, start, stop, trial_idx, window_name)
    begin_index = int(np.argmin(np.abs(time - start)))
    end_index = int(np.argmin(np.abs(time - stop)))
    if end_index == begin_index and start == stop:
        return slice(begin_index, begin_index + 1)
    if end_index <= begin_index:
        raise ValueError(f"{window_name.capitalize()} window is empty for trial {trial_idx}.")
    # Windows are half-open [start, stop): a stop value that lands on a
    # sampled time is a duration boundary, not another selected sample.
    return slice(begin_index, end_index)


def _matching_sample_window_slice(time, start, sample_count, trial_idx, window_name):
    if sample_count <= 0:
        raise ValueError(f"{window_name.capitalize()} window sample count must be positive.")
    _require_time_supported(time, start, trial_idx, window_name)
    begin_index = int(np.argmin(np.abs(time - start)))
    end_index = begin_index + int(sample_count)
    if end_index > time.size:
        raise ValueError(f"{window_name.capitalize()} window extends beyond trial {trial_idx}'s time support.")
    return slice(begin_index, end_index)


def _require_window_supported(time, start, stop, trial_idx, window_name):
    _require_time_supported(time, start, trial_idx, window_name)
    _require_time_supported(time, stop, trial_idx, window_name)


def _require_time_supported(time, value, trial_idx, window_name):
    tolerance = _time_support_tolerance(time)
    if value < time[0] - tolerance or value > time[-1] + tolerance:
        raise ValueError(f"{window_name.capitalize()} window is outside trial {trial_idx}'s time support.")


def _time_support_tolerance(time):
    if time.size < 2:
        return 1e-12
    return 0.5 * float(np.median(np.diff(time))) + 1e-12


def _require_consistent_feature_size(feature, expected_size, trial_idx, window_name):
    feature_size = int(feature.shape[0])
    if expected_size is None:
        return feature_size
    if feature_size != expected_size:
        raise ValueError(
            f"{window_name.capitalize()} window for trial {trial_idx} produced {feature_size} values; "
            f"expected {expected_size}. Check per-trial time vectors."
        )
    return expected_size
