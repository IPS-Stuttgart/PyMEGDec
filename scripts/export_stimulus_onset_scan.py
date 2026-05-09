"""Backward-compatible wrapper for the grouped stimulus onset-scan command."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec import stimulus_decoding as _stimulus_decoding  # noqa: E402


def _stable_group_value(value):
    """Canonicalize missing scalar values before grouping onset summaries."""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return value
    return value


def _canonicalize_group_fields(rows, group_fields):
    stable_rows = []
    for row in rows:
        stable_row = dict(row)
        for field in group_fields:
            stable_row[field] = _stable_group_value(stable_row.get(field, ""))
        stable_rows.append(stable_row)
    return stable_rows


_SUMMARY_GROUP_FIELDS = (
    "participant",
    "variant",
    "transfer_direction",
    "train_window_center_s",
    "threshold_method",
    "min_consecutive",
    "min_duration_s",
    "require_stable_prediction",
    "classifier",
    "components_pca",
    "frequency_low_hz",
    "frequency_high_hz",
)
_SCAN_GROUP_FIELDS = (*_SUMMARY_GROUP_FIELDS[:8], "scan_window_center_s", *_SUMMARY_GROUP_FIELDS[8:])

_ORIGINAL_SUMMARIZE_STIMULUS_ONSET_EVENTS = _stimulus_decoding.summarize_stimulus_onset_events
_ORIGINAL_SUMMARIZE_STIMULUS_ONSET_SCAN = _stimulus_decoding.summarize_stimulus_onset_scan


def _summarize_stimulus_onset_events_stable(rows):
    return _ORIGINAL_SUMMARIZE_STIMULUS_ONSET_EVENTS(_canonicalize_group_fields(rows, _SUMMARY_GROUP_FIELDS))


def _summarize_stimulus_onset_scan_stable(rows):
    return _ORIGINAL_SUMMARIZE_STIMULUS_ONSET_SCAN(_canonicalize_group_fields(rows, _SCAN_GROUP_FIELDS))


_stimulus_decoding.summarize_stimulus_onset_events = _summarize_stimulus_onset_events_stable
_stimulus_decoding.summarize_stimulus_onset_scan = _summarize_stimulus_onset_scan_stable

from pymegdec.stimulus_cli import stimulus_onset_scan  # noqa: E402


def main() -> int:
    return stimulus_onset_scan()


if __name__ == "__main__":
    raise SystemExit(main())
