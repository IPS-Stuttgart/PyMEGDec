"""Public time-resolved stimulus decoding API."""

from __future__ import annotations

import pymegdec._stimulus_decoding_core as _core
from pymegdec._stimulus_decoding_core import *  # noqa: F403
from pymegdec._stimulus_summary import (
    summarize_stimulus_decoding,
    summarize_stimulus_temporal_generalization,
)


def export_time_resolved_stimulus_decoding(
    data_folder,
    participants,
    output_path,
    *,
    summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    participant_peaks_output_path=None,
    diagnostic_window_centers=None,
    plots_dir=None,
    config=None,
    progress=None,
):
    """Run time-resolved stimulus decoding and write CSV/plot artifacts."""

    config = config or StimulusDecodingConfig()  # noqa: F405
    data_folder = _core.resolve_data_folder(data_folder)
    rows = []
    prediction_rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        participant_rows, participant_prediction_rows = _core._evaluate_participant_time_resolved_stimulus_transfer(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=diagnostic_window_centers,
        )
        rows.extend(participant_rows)
        prediction_rows.extend(participant_prediction_rows)
        if progress is not None:
            progress(f"DONE participant={participant}")
    _core.write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_decoding(rows)
    if summary_output_path:
        _core.write_alpha_metrics_csv(summary_rows, summary_output_path)
    if participant_peaks_output_path:
        _core.write_alpha_metrics_csv(summarize_stimulus_decoding_peaks(rows), participant_peaks_output_path)  # noqa: F405
    if predictions_output_path and prediction_rows:
        _core.write_alpha_metrics_csv(prediction_rows, predictions_output_path)
    if (confusion_output_path or per_stimulus_output_path) and prediction_rows:
        confusion_rows, per_stimulus_rows = summarize_stimulus_prediction_diagnostics(prediction_rows)  # noqa: F405
        if confusion_output_path:
            _core.write_alpha_metrics_csv(confusion_rows, confusion_output_path)
        if per_stimulus_output_path:
            _core.write_alpha_metrics_csv(per_stimulus_rows, per_stimulus_output_path)
    if plots_dir:
        write_stimulus_decoding_plots(summary_rows, plots_dir)  # noqa: F405
    return rows, summary_rows


def export_stimulus_temporal_generalization(
    data_folder,
    participants,
    output_path,
    *,
    summary_output_path=None,
    config=None,
    progress=None,
):
    """Run stimulus temporal generalization and write CSV artifacts."""

    config = config or StimulusDecodingConfig()  # noqa: F405
    data_folder = _core.resolve_data_folder(data_folder)
    rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        rows.extend(_core.evaluate_participant_stimulus_temporal_generalization(data_folder, participant, config=config))
        if progress is not None:
            progress(f"DONE participant={participant}")
    _core.write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_temporal_generalization(rows)
    if summary_output_path:
        _core.write_alpha_metrics_csv(summary_rows, summary_output_path)
    return rows, summary_rows


def __getattr__(name):
    """Delegate private legacy helpers for compatibility with existing tests/scripts."""

    return getattr(_core, name)


__all__ = [name for name in globals() if not name.startswith("_")]
