"""Sensor-level alpha movement trajectories for MEG trials."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pymegdec.alpha_metrics import (
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
    DEFAULT_PROJECTION_REFERENCE_PATTERN,
    DEFAULT_SENSOR_POSITION_UNIT,
    compute_alpha_analytic_window,
    count_trials,
    get_channel_names,
    load_participant_data,
    project_channel_positions,
    select_channels,
    uniform_sample_interval,
    write_alpha_metrics_csv,
)
from pymegdec.alpha_signal import get_data_field, get_time_vector, get_trial_signal

DEFAULT_SENSOR_PATTERN = r"^M"
DEFAULT_MOVEMENT_TIME_WINDOW = (-0.4, 0.8)
DEFAULT_TRAJECTORY_STEP_S = 0.02
POWER_EPSILON = 1e-12
DEFAULT_MIN_TOTAL_ALPHA_POWER = 0.0


@dataclass(frozen=True)
class AlphaMovementConfig:
    """Parameters for sensor-level alpha movement trajectory extraction."""

    location_pattern: str = DEFAULT_SENSOR_PATTERN
    time_window: tuple[float, float] = DEFAULT_MOVEMENT_TIME_WINDOW
    frequency_range: tuple[float, float] = DEFAULT_FREQUENCY_RANGE
    trajectory_step_s: float | None = DEFAULT_TRAJECTORY_STEP_S
    filter_order: int = 5
    sensor_position_unit: str = DEFAULT_SENSOR_POSITION_UNIT
    projection_reference_pattern: str | None = DEFAULT_PROJECTION_REFERENCE_PATTERN
    min_reference_axis_projection: float = DEFAULT_MIN_REFERENCE_AXIS_PROJECTION
    min_total_alpha_power: float = DEFAULT_MIN_TOTAL_ALPHA_POWER


@dataclass(frozen=True)
class _MovementContext:
    data: object
    trial_idx: int
    participant_id: object
    dataset: str
    config: AlphaMovementConfig


@dataclass(frozen=True)
class _MovementGeometry:
    channel_indices: np.ndarray
    channel_names: np.ndarray
    positions: np.ndarray
    projected_positions: np.ndarray


@dataclass
class _MovementState:
    first: dict | None = None
    previous: dict | None = None
    previous_time: float = np.nan


def _trial_label(data, trial_idx):
    try:
        trialinfo = np.asarray(get_data_field(data, "trialinfo")).ravel()
    except (KeyError, ValueError):
        return np.nan
    return trialinfo[trial_idx].item()


def _resolve_channel_indices(data, channel_indices, location_pattern):
    if channel_indices is None:
        channel_indices = select_channels(data, location_pattern)

    resolved = np.asarray(channel_indices, dtype=int)
    if resolved.size == 0:
        raise ValueError(f"No channels matched pattern: {location_pattern}")
    return resolved


def _sampling_rate(time_vector):
    return float(1 / uniform_sample_interval(time_vector))


def sample_time_indices(time_vector, time_window, trajectory_step_s):
    """Return time indices inside ``time_window`` sampled at ``trajectory_step_s``."""

    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")

    time_vector = np.asarray(time_vector, dtype=float).ravel()
    sample_interval = uniform_sample_interval(time_vector)
    tolerance = max(abs(sample_interval) * 1e-6, 1e-12)
    window_indices = np.flatnonzero((time_vector >= start - tolerance) & (time_vector <= stop + tolerance))
    if window_indices.size == 0:
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    if trajectory_step_s is None:
        return window_indices
    if trajectory_step_s <= 0:
        raise ValueError("trajectory_step_s must be positive.")

    first_time = max(start, float(time_vector[window_indices[0]]))
    last_time = min(stop, float(time_vector[window_indices[-1]]))
    targets = np.arange(first_time, last_time + trajectory_step_s / 2, trajectory_step_s)
    sampled = [int(window_indices[np.argmin(np.abs(time_vector[window_indices] - target))]) for target in targets]
    return np.unique(sampled)


def _alpha_power(signal, time_vector, sample_indices, config):
    _sampling_rate(time_vector)
    alpha_window, window_indices = compute_alpha_analytic_window(signal, time_vector, config)
    relative_indices = np.array(
        [int(np.argmin(np.abs(window_indices - sample_index))) for sample_index in sample_indices],
        dtype=int,
    )
    return np.abs(np.take(alpha_window, relative_indices, axis=-1)) ** 2


def _validated_min_total_alpha_power(min_total_alpha_power):
    value = float(min_total_alpha_power)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("min_total_alpha_power must be finite and non-negative.")
    return value


def _has_reliable_power(weights, min_total_alpha_power):
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size == 0:
        raise ValueError("alpha power weights must contain at least one channel.")
    if np.any(weights < 0.0):
        raise ValueError("alpha power weights must be non-negative.")
    if not np.all(np.isfinite(weights)):
        return False
    return float(np.sum(weights)) > min_total_alpha_power


def _spatial_concentration(weights):
    weights = np.asarray(weights, dtype=float).ravel()
    total_weight = float(np.sum(weights)) if weights.size else np.nan
    if weights.size == 0 or not np.isfinite(total_weight) or total_weight <= 0.0 or not np.all(np.isfinite(weights)):
        return np.nan
    probabilities = weights / total_weight
    entropy = -float(np.sum(probabilities * np.log(probabilities + POWER_EPSILON)))
    max_entropy = np.log(weights.size)
    if max_entropy <= 0:
        return 1.0
    return float(1 - entropy / max_entropy)


def _movement_values(centroid, projected, first, previous, previous_time, time_s):
    if previous is None:
        return {
            "displacement_mm": 0.0,
            "projected_displacement_mm": 0.0,
            "speed_mm_per_s": np.nan,
            "projected_speed_mm_per_s": np.nan,
            "projected_direction_rad": np.nan,
        }

    dt = time_s - previous_time
    if dt <= 0:
        speed = np.nan
        projected_speed = np.nan
    else:
        speed = float(np.linalg.norm(centroid - previous["centroid"]) / dt)
        projected_speed = float(np.linalg.norm(projected - previous["projected"]) / dt)

    projected_step = projected - previous["projected"]
    if float(np.linalg.norm(projected_step)) == 0.0:
        projected_direction = np.nan
    else:
        projected_direction = float(np.arctan2(projected_step[1], projected_step[0]))
    return {
        "displacement_mm": float(np.linalg.norm(centroid - first["centroid"])),
        "projected_displacement_mm": float(np.linalg.norm(projected - first["projected"])),
        "speed_mm_per_s": speed,
        "projected_speed_mm_per_s": projected_speed,
        "projected_direction_rad": projected_direction,
    }


def _selected_geometry(
    data,
    trial_signal,
    channel_indices,
    sensor_position_unit=DEFAULT_SENSOR_POSITION_UNIT,
    projection_reference_pattern=DEFAULT_PROJECTION_REFERENCE_PATTERN,
    min_reference_axis_projection=DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
):
    positions, projected_positions = project_channel_positions(
        data,
        channel_indices,
        sensor_position_unit=sensor_position_unit,
        projection_reference_pattern=projection_reference_pattern,
        min_reference_axis_projection=min_reference_axis_projection,
    )
    channel_names = np.asarray(get_channel_names(data, trial_signal.shape[0]), dtype=object)[channel_indices]
    return _MovementGeometry(
        channel_indices=channel_indices,
        channel_names=channel_names,
        positions=positions,
        projected_positions=projected_positions,
    )


def _empty_movement_values():
    return {
        "displacement_mm": np.nan,
        "projected_displacement_mm": np.nan,
        "speed_mm_per_s": np.nan,
        "projected_speed_mm_per_s": np.nan,
        "projected_direction_rad": np.nan,
    }


def _trajectory_row(context, geometry, weights, time_s, state, min_total_alpha_power=DEFAULT_MIN_TOTAL_ALPHA_POWER):
    weights = np.asarray(weights, dtype=float).ravel()
    total_alpha_power = float(np.sum(weights)) if weights.size else np.nan
    mean_alpha_power = float(np.mean(weights)) if weights.size else np.nan
    finite_weights = weights[np.isfinite(weights)]
    peak_alpha_power = float(np.max(finite_weights)) if finite_weights.size else np.nan
    reliable_power = _has_reliable_power(weights, min_total_alpha_power)

    row = {
        "participant": (context.participant_id if context.participant_id is not None else ""),
        "dataset": context.dataset,
        "trial": context.trial_idx,
        "trial_label": _trial_label(context.data, context.trial_idx),
        "time_s": time_s,
        "low_freq": context.config.frequency_range[0],
        "high_freq": context.config.frequency_range[1],
        "n_channels": int(geometry.channel_indices.size),
        "mean_alpha_power": mean_alpha_power,
        "total_alpha_power": total_alpha_power,
        "peak_alpha_power": peak_alpha_power,
        "peak_channel": -1,
        "peak_channel_name": "",
        "spatial_concentration": _spatial_concentration(weights),
        "centroid_x_mm": np.nan,
        "centroid_y_mm": np.nan,
        "centroid_z_mm": np.nan,
        "projected_x_mm": np.nan,
        "projected_y_mm": np.nan,
    }

    if not reliable_power:
        row.update(_empty_movement_values())
        return row

    centroid = np.average(geometry.positions, axis=0, weights=weights)
    projected = np.average(geometry.projected_positions, axis=0, weights=weights)
    peak_local_index = int(np.argmax(weights))
    current = {"centroid": centroid, "projected": projected}
    if state.first is None:
        state.first = current

    row.update(
        {
            "peak_channel": int(geometry.channel_indices[peak_local_index]),
            "peak_channel_name": str(geometry.channel_names[peak_local_index]),
            "centroid_x_mm": float(centroid[0]),
            "centroid_y_mm": float(centroid[1]),
            "centroid_z_mm": float(centroid[2]),
            "projected_x_mm": float(projected[0]),
            "projected_y_mm": float(projected[1]),
        }
    )
    row.update(
        _movement_values(
            centroid,
            projected,
            state.first,
            state.previous,
            state.previous_time,
            time_s,
        )
    )
    state.previous = current
    state.previous_time = time_s
    return row


def compute_alpha_movement_trajectory(
    data,
    trial_idx,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Track the alpha-power centroid across the MEG sensor array for one trial."""

    config = config or AlphaMovementConfig()
    min_total_alpha_power = _validated_min_total_alpha_power(config.min_total_alpha_power)
    trial_signal = get_trial_signal(data, trial_idx)
    channel_indices = _resolve_channel_indices(data, channel_indices, config.location_pattern)
    time_vector = get_time_vector(data, trial_idx)
    sample_indices = sample_time_indices(time_vector, config.time_window, config.trajectory_step_s)
    powers = _alpha_power(
        np.take(trial_signal, channel_indices, axis=0),
        time_vector,
        sample_indices,
        config,
    )
    geometry = _selected_geometry(
        data,
        trial_signal,
        channel_indices,
        sensor_position_unit=config.sensor_position_unit,
        projection_reference_pattern=config.projection_reference_pattern,
        min_reference_axis_projection=config.min_reference_axis_projection,
    )
    context = _MovementContext(data, trial_idx, participant_id, dataset, config)
    state = _MovementState()
    return [
        _trajectory_row(
            context,
            geometry,
            powers[:, column],
            float(time_vector[time_index]),
            state,
            min_total_alpha_power,
        )
        for column, time_index in enumerate(sample_indices)
    ]


def compute_alpha_movement(
    data,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Track alpha-power centroids for every trial in ``data``."""

    config = config or AlphaMovementConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config.location_pattern)
    rows = []
    for trial_idx in range(count_trials(data)):
        rows.extend(
            compute_alpha_movement_trajectory(
                data,
                trial_idx,
                participant_id=participant_id,
                dataset=dataset,
                channel_indices=channel_indices,
                config=config,
            )
        )
    return rows


def _finite_mean(values):
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.mean(array))


def _summary_vector(row, fields):
    values = np.asarray([row[field] for field in fields], dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return values


def _vector_norm_delta(current, reference):
    if current is None or reference is None:
        return np.nan
    return float(np.linalg.norm(current - reference))


def _speed_from_delta(current, previous, time_s, previous_time_s):
    if current is None or previous is None:
        return np.nan
    dt = time_s - previous_time_s
    if dt <= 0:
        return np.nan
    return float(np.linalg.norm(current - previous) / dt)


def _projected_direction(current, previous):
    if current is None or previous is None:
        return np.nan
    step = current - previous
    if float(np.linalg.norm(step)) == 0.0:
        return np.nan
    return float(np.arctan2(step[1], step[0]))


def _add_mean_trajectory_movement(summary_rows):
    first_centroid = None
    first_projected = None
    previous_centroid = None
    previous_projected = None
    previous_centroid_time_s = np.nan
    previous_projected_time_s = np.nan

    for row in sorted(summary_rows, key=lambda item: item["time_s"]):
        time_s = float(row["time_s"])
        centroid = _summary_vector(row, ("centroid_x_mm", "centroid_y_mm", "centroid_z_mm"))
        projected = _summary_vector(row, ("projected_x_mm", "projected_y_mm"))

        if first_centroid is None and centroid is not None:
            first_centroid = centroid
        if first_projected is None and projected is not None:
            first_projected = projected

        mean_trajectory_displacement = _vector_norm_delta(centroid, first_centroid)
        mean_trajectory_projected_displacement = _vector_norm_delta(projected, first_projected)
        mean_trajectory_speed = _speed_from_delta(
            centroid,
            previous_centroid,
            time_s,
            previous_centroid_time_s,
        )
        mean_trajectory_projected_speed = _speed_from_delta(
            projected,
            previous_projected,
            time_s,
            previous_projected_time_s,
        )
        mean_trajectory_projected_direction = _projected_direction(projected, previous_projected)

        row["mean_trajectory_displacement_mm"] = mean_trajectory_displacement
        row["mean_trajectory_projected_displacement_mm"] = mean_trajectory_projected_displacement
        row["mean_trajectory_speed_mm_per_s"] = mean_trajectory_speed
        row["mean_trajectory_projected_speed_mm_per_s"] = mean_trajectory_projected_speed
        row["mean_trajectory_projected_direction_rad"] = mean_trajectory_projected_direction

        # Backwards-compatible summary column names now carry the corrected
        # mean-trajectory semantics. The old trial-average values are exposed
        # separately under ``mean_trial_*`` columns.
        row["displacement_mm"] = mean_trajectory_displacement
        row["projected_displacement_mm"] = mean_trajectory_projected_displacement
        row["speed_mm_per_s"] = mean_trajectory_speed
        row["projected_speed_mm_per_s"] = mean_trajectory_projected_speed
        row["projected_direction_rad"] = mean_trajectory_projected_direction

        if centroid is not None:
            previous_centroid = centroid
            previous_centroid_time_s = time_s
        if projected is not None:
            previous_projected = projected
            previous_projected_time_s = time_s


def summarize_alpha_movement(rows):
    """Average alpha centroids and derive movement from the averaged trajectory.

    The summary row is a trajectory of the mean centroid for each participant,
    dataset, condition, and time. Therefore ``mean_trajectory_*`` movement
    columns, and their backwards-compatible aliases ``displacement_mm`` and
    ``speed_mm_per_s``, are computed from that mean trajectory rather than by
    averaging each trial's scalar displacement or speed. Trial-averaged scalar
    movement is retained under explicit ``mean_trial_*`` columns.
    """

    grouped = defaultdict(list)
    for row in rows:
        key = (
            str(row["participant"]),
            str(row["dataset"]),
            str(row["trial_label"]),
            round(float(row["time_s"]), 9),
        )
        grouped[key].append(row)

    rows_by_trajectory = defaultdict(list)
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        participant, dataset, trial_label, time_s = key
        trials = {int(row["trial"]) for row in group_rows}
        rows_by_trajectory[(participant, dataset, trial_label)].append(
            {
                "participant": participant,
                "dataset": dataset,
                "trial_label": trial_label,
                "time_s": time_s,
                "n_trials": len(trials),
                "mean_alpha_power": _finite_mean(row["mean_alpha_power"] for row in group_rows),
                "spatial_concentration": _finite_mean(row["spatial_concentration"] for row in group_rows),
                "centroid_x_mm": _finite_mean(row["centroid_x_mm"] for row in group_rows),
                "centroid_y_mm": _finite_mean(row["centroid_y_mm"] for row in group_rows),
                "centroid_z_mm": _finite_mean(row["centroid_z_mm"] for row in group_rows),
                "projected_x_mm": _finite_mean(row["projected_x_mm"] for row in group_rows),
                "projected_y_mm": _finite_mean(row["projected_y_mm"] for row in group_rows),
                "mean_trajectory_displacement_mm": np.nan,
                "mean_trajectory_projected_displacement_mm": np.nan,
                "mean_trajectory_speed_mm_per_s": np.nan,
                "mean_trajectory_projected_speed_mm_per_s": np.nan,
                "mean_trajectory_projected_direction_rad": np.nan,
                "displacement_mm": np.nan,
                "projected_displacement_mm": np.nan,
                "speed_mm_per_s": np.nan,
                "projected_speed_mm_per_s": np.nan,
                "projected_direction_rad": np.nan,
                "mean_trial_displacement_mm": _finite_mean(
                    row["displacement_mm"] for row in group_rows
                ),
                "mean_trial_projected_displacement_mm": _finite_mean(
                    row["projected_displacement_mm"] for row in group_rows
                ),
                "mean_trial_speed_mm_per_s": _finite_mean(
                    row["speed_mm_per_s"] for row in group_rows
                ),
                "mean_trial_projected_speed_mm_per_s": _finite_mean(
                    row["projected_speed_mm_per_s"] for row in group_rows
                ),
            }
        )

    summary_rows = []
    for trajectory_key in sorted(rows_by_trajectory):
        trajectory_rows = sorted(rows_by_trajectory[trajectory_key], key=lambda item: item["time_s"])
        _add_mean_trajectory_movement(trajectory_rows)
        summary_rows.extend(trajectory_rows)
    return summary_rows


def write_alpha_movement_csv(rows, output_path):
    """Write alpha movement rows to ``output_path``."""

    write_alpha_metrics_csv(rows, output_path)


def export_alpha_movement(
    data_folder,
    participants,
    trajectory_output_path,
    *,
    summary_output_path=None,
    cue=False,
    config=None,
):
    """Export sensor-level alpha movement trajectories for participants."""

    config = config or AlphaMovementConfig()
    dataset = "cue" if cue else "main"
    rows = []
    for participant_id in participants:
        data = load_participant_data(data_folder, participant_id, cue=cue)
        rows.extend(
            compute_alpha_movement(
                data,
                participant_id=participant_id,
                dataset=dataset,
                config=config,
            )
        )

    write_alpha_movement_csv(rows, trajectory_output_path)
    summary_rows = summarize_alpha_movement(rows)
    if summary_output_path:
        write_alpha_movement_csv(summary_rows, summary_output_path)
    return rows, summary_rows
