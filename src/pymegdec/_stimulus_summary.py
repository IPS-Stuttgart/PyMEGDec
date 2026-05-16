"""Participant-aware stimulus decoding summaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import pymegdec._stimulus_decoding_core as _core


# jscpd:ignore-start
# pylint: disable=protected-access,too-many-locals

def summarize_stimulus_decoding(rows):
    """Summarize decoding rows while counting unique participants."""

    if not rows:
        return []

    group_fields = _core._present_group_fields(rows, _core.SUMMARY_GROUP_FIELDS)
    frame = _core._rows_frame(rows)
    participant_column = _participant_summary_column(rows)
    metric_summary = _core.summarize_metric_table(
        frame,
        "accuracy",
        group_fields,
        participant_column=participant_column,
        chance_column="chance_accuracy",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for base_summary in metric_summary.to_dict("records"):
        key = tuple(base_summary.get(field, "") for field in group_fields)
        group_rows = grouped[key]
        accuracies = [_core._to_float(row["accuracy"]) for row in group_rows]
        std = _core._legacy_std(base_summary["accuracy_std"], accuracies)
        sem = _core._legacy_sem(base_summary["accuracy_sem"], accuracies)
        permutation_p = [_core._to_float(row.get("permutation_p_value")) for row in group_rows]
        n_with_permutation = sum(np.isfinite(permutation_p))
        significant_05 = sum(value < 0.05 for value in permutation_p if np.isfinite(value))
        significant_01 = sum(value < 0.01 for value in permutation_p if np.isfinite(value))
        chance_accuracy = _core._to_float(group_rows[0]["chance_accuracy"])
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": int(base_summary.get("n_participants", len(group_rows))),
                "accuracy_mean": base_summary["accuracy_mean"],
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * base_summary["accuracy_mean"],
                "percent_median": 100.0 * base_summary["accuracy_median"],
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "chance_accuracy": chance_accuracy,
                "chance_percent": 100.0 * chance_accuracy,
                "above_chance_count": int(base_summary["accuracy_above_chance_count"]),
                "n_with_permutation": int(n_with_permutation),
                "n_significant_p_0.05": int(significant_05),
                "n_significant_p_0.01": int(significant_01),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


def summarize_stimulus_temporal_generalization(rows):
    """Summarize temporal-generalization rows while counting unique participants."""

    if not rows:
        return []

    group_fields = _core._present_group_fields(rows, _core.TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS)
    frame = _core._rows_frame(rows)
    participant_column = _participant_summary_column(rows)
    metric_summary = _core.summarize_metric_table(
        frame,
        "accuracy",
        group_fields,
        participant_column=participant_column,
        chance_column="chance_accuracy",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for base_summary in metric_summary.to_dict("records"):
        key = tuple(base_summary.get(field, "") for field in group_fields)
        group_rows = grouped[key]
        accuracies = [_core._to_float(row["accuracy"]) for row in group_rows]
        std = _core._legacy_std(base_summary["accuracy_std"], accuracies)
        sem = _core._legacy_sem(base_summary["accuracy_sem"], accuracies)
        chance_accuracy = _core._to_float(group_rows[0]["chance_accuracy"])
        diagonal_values = {
            _core._window_center_key(row["train_window_center_s"])
            == _core._window_center_key(row["test_window_center_s"])
            for row in group_rows
        }
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": int(base_summary.get("n_participants", len(group_rows))),
                "accuracy_mean": base_summary["accuracy_mean"],
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * base_summary["accuracy_mean"],
                "percent_median": 100.0 * base_summary["accuracy_median"],
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "chance_accuracy": chance_accuracy,
                "chance_percent": 100.0 * chance_accuracy,
                "above_chance_count": int(base_summary["accuracy_above_chance_count"]),
                "is_diagonal": bool(diagonal_values == {True}),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


def _participant_summary_column(rows):
    if rows and all("participant" in row for row in rows):
        return "participant"
    return None


# jscpd:ignore-end
