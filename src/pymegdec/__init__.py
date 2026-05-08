"""Utilities for MEG decoding experiments."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pymegdec.alpha_metrics import (
    AlphaMetricConfig,
    compute_alpha_metrics,
    export_participant_alpha_metrics,
)
from pymegdec.alpha_movement import (
    AlphaMovementConfig,
    compute_alpha_movement,
    export_alpha_movement,
)
from pymegdec.alpha_movement_analysis import (
    AlphaMovementAnalysisConfig,
    analyze_alpha_movement_windows,
    export_alpha_movement_analysis,
    summarize_alpha_movement_effects,
)
from pymegdec.alpha_signal import extract_phase, extract_time_basis
from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.data_config import DATA_DIR_ENV_VAR, resolve_data_folder
from pymegdec.model_transfer import (
    evaluate_model_transfer,
    get_original_feature_importance,
)
from pymegdec.reaction_time_analysis import (
    AlphaReactionTimeExportConfig,
    ReactionTimeCsvConfig,
    ReactionTimeUnavailableError,
    analyze_alpha_reaction_times,
    join_alpha_reaction_times,
)
from pymegdec.stimulus_decoding import (
    TRANSFER_DIRECTIONS,
    StimulusDecodingConfig,
    evaluate_participant_stimulus_decoding_diagnostics,
    evaluate_participant_stimulus_onset_scan,
    evaluate_participant_stimulus_temporal_generalization,
    evaluate_time_resolved_stimulus_transfer,
    export_stimulus_onset_scan,
    export_stimulus_temporal_generalization,
    export_time_resolved_stimulus_decoding,
    summarize_stimulus_decoding,
    summarize_stimulus_decoding_peaks,
    summarize_stimulus_onset_events,
    summarize_stimulus_onset_scan,
    summarize_stimulus_prediction_diagnostics,
    summarize_stimulus_temporal_generalization,
)

__version__ = "0.1.0"

_VALUE_OPTIONS_THAT_CAN_START_WITH_DASH = {
    "--window-centers",
    "--time-window",
    "--scan-time-window",
    "--threshold-window",
}


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    """Normalize selected option-value pairs for negative comma-separated ranges."""

    if argv is None:
        argv = sys.argv[1:]
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _VALUE_OPTIONS_THAT_CAN_START_WITH_DASH and index + 1 < len(argv):
            normalized.append(f"{token}={argv[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


# Keep the historical pymegdec.cli helper surface available without replacing
# the legacy CLI module wholesale. The grouped stimulus commands and old script
# wrappers share this helper for arguments such as "--time-window -0.4,0.8".
try:  # pragma: no cover - exercised through command-line entry points
    from pymegdec import cli as _legacy_cli

    if not hasattr(_legacy_cli, "_normalize_argv"):
        _legacy_cli._normalize_argv = _normalize_argv
finally:
    if "_legacy_cli" in locals():
        del _legacy_cli


__all__ = [
    "__version__",
    "DATA_DIR_ENV_VAR",
    "AlphaMetricConfig",
    "AlphaMovementConfig",
    "AlphaMovementAnalysisConfig",
    "AlphaReactionTimeExportConfig",
    "ReactionTimeCsvConfig",
    "ReactionTimeUnavailableError",
    "StimulusDecodingConfig",
    "TRANSFER_DIRECTIONS",
    "analyze_alpha_reaction_times",
    "analyze_alpha_movement_windows",
    "compute_alpha_movement",
    "compute_alpha_metrics",
    "cross_validate_single_dataset",
    "evaluate_participant_stimulus_decoding_diagnostics",
    "evaluate_participant_stimulus_onset_scan",
    "evaluate_participant_stimulus_temporal_generalization",
    "evaluate_model_transfer",
    "evaluate_time_resolved_stimulus_transfer",
    "export_alpha_movement",
    "export_alpha_movement_analysis",
    "export_participant_alpha_metrics",
    "export_stimulus_onset_scan",
    "export_stimulus_temporal_generalization",
    "export_time_resolved_stimulus_decoding",
    "extract_phase",
    "extract_time_basis",
    "get_original_feature_importance",
    "join_alpha_reaction_times",
    "resolve_data_folder",
    "summarize_stimulus_decoding",
    "summarize_stimulus_decoding_peaks",
    "summarize_stimulus_onset_events",
    "summarize_stimulus_onset_scan",
    "summarize_stimulus_prediction_diagnostics",
    "summarize_stimulus_temporal_generalization",
    "summarize_alpha_movement_effects",
]
