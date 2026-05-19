"""Participant-aware stimulus decoding summaries."""

from __future__ import annotations

import pymegdec._stimulus_decoding_core as _core
from neureptrace.results.tables import (  # pylint: disable=no-name-in-module
    DEFAULT_CHANCE_CLASS_COLUMNS,
    DEFAULT_CHANCE_SUMMARY_COLUMNS,
    append_temporal_diagonal_flag,
    metric_summary_columns,
    summarize_decoding_metric_rows,
    summary_records,
)

_CHANCE_CLASS_COLUMNS = DEFAULT_CHANCE_CLASS_COLUMNS
_CHANCE_SUMMARY_COLUMNS = DEFAULT_CHANCE_SUMMARY_COLUMNS


# pylint: disable=protected-access

def summarize_stimulus_decoding(rows):
    """Summarize decoding rows while counting unique participants."""

    if not rows:
        return []
    summary, group_fields = _stimulus_summary_frame(
        rows,
        _core.SUMMARY_GROUP_FIELDS,
        permutation_p_column="permutation_p_value",
    )
    return summary_records(summary, _summary_columns(group_fields, include_permutation=True))


def summarize_stimulus_temporal_generalization(rows):
    """Summarize temporal-generalization rows while counting unique participants."""

    if not rows:
        return []
    summary, group_fields = _stimulus_summary_frame(rows, _core.TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS)
    summary = append_temporal_diagonal_flag(
        summary,
        train_column="train_window_center_s",
        test_column="test_window_center_s",
    )
    return summary_records(summary, _summary_columns(group_fields, include_diagonal=True))


def _stimulus_summary_frame(rows, group_field_candidates, *, permutation_p_column=None):
    summary, group_fields = summarize_decoding_metric_rows(
        rows,
        "accuracy",
        group_column_candidates=group_field_candidates,
        participant_column=_participant_summary_column(rows),
        chance_column="chance_accuracy",
        percent_scale=100.0,
        chance_percent_column="chance_percent",
        chance_class_columns=_CHANCE_CLASS_COLUMNS,
        permutation_p_column=permutation_p_column,
    )
    return summary, group_fields


def _summary_columns(group_fields, *, include_permutation=False, include_diagonal=False):
    return metric_summary_columns(
        group_fields,
        include_permutation=include_permutation,
        include_diagonal=include_diagonal,
        chance_summary_columns=_CHANCE_SUMMARY_COLUMNS,
    )


def _participant_summary_column(rows):
    if rows and all("participant" in row for row in rows):
        return "participant"
    return None
