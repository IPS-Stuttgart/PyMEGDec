"""Chance-level inference fixes for cross-subject stimulus decoding."""

from __future__ import annotations

from collections import Counter

import numpy as np

from pymegdec import _stimulus_cross_subject_core as _core

_impl = _core._impl

_ORIGINAL_SCORE_OUTER_FOLD_MODEL = _impl._score_outer_fold_model
_ORIGINAL_SUMMARIZE_CROSS_SUBJECT_STIMULUS_SMOKE = _impl.summarize_cross_subject_stimulus_smoke
_ORIGINAL_SUMMARIZE_NESTED_CROSS_SUBJECT_STIMULUS = _impl.summarize_nested_cross_subject_stimulus


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    """Score an outer fold and report chance from the held-out labels."""

    outer_row, prediction_rows = _ORIGINAL_SCORE_OUTER_FOLD_MODEL(
        fitted_model,
        test_set,
        config,
        include_predictions=include_predictions,
    )
    _patch_row_chance_fields(outer_row, _inferred_test_chance_classes(test_set))
    return outer_row, prediction_rows


def summarize_cross_subject_stimulus_smoke(outer_rows, *, config=None):
    """Summarize cross-subject scores using per-fold chance levels."""

    summary_rows = _ORIGINAL_SUMMARIZE_CROSS_SUBJECT_STIMULUS_SMOKE(
        outer_rows,
        config=config,
    )
    config = _impl._normalized_config(config or _impl.CrossSubjectStimulusConfig())
    return _patch_summary_chance_fields(
        summary_rows,
        outer_rows,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def summarize_nested_cross_subject_stimulus(
    outer_rows,
    *,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    """Summarize nested cross-subject scores using per-fold chance levels."""

    summary_rows = _ORIGINAL_SUMMARIZE_NESTED_CROSS_SUBJECT_STIMULUS(
        outer_rows,
        signflip_permutations=signflip_permutations,
        signflip_seed=signflip_seed,
    )
    return _patch_summary_chance_fields(
        summary_rows,
        outer_rows,
        signflip_permutations=signflip_permutations,
        signflip_seed=signflip_seed,
    )


def _inferred_test_chance_classes(test_set):
    labels = np.asarray(test_set.labels).ravel()
    if labels.size == 0:
        return None
    return int(np.unique(labels).size)


def _patch_row_chance_fields(row, chance_classes):
    chance_classes = _positive_int(chance_classes)
    if chance_classes is None:
        return row

    chance_accuracy = 1.0 / chance_classes
    row["chance_classes"] = int(chance_classes)
    row["chance_accuracy"] = chance_accuracy
    row["chance_percent"] = 100.0 * chance_accuracy
    row["top2_chance_accuracy"] = min(2.0 * chance_accuracy, 1.0)
    row["top2_chance_percent"] = min(200.0 * chance_accuracy, 100.0)
    row["top3_chance_accuracy"] = min(3.0 * chance_accuracy, 1.0)
    row["top3_chance_percent"] = min(300.0 * chance_accuracy, 100.0)
    row["chance_mean_rank"] = 0.5 * (chance_classes + 1)

    score = _to_float(row.get("balanced_accuracy", row.get("accuracy")))
    row["above_chance"] = bool(np.isfinite(score) and score > chance_accuracy)
    return row


def _patch_summary_chance_fields(
    summary_rows,
    outer_rows,
    *,
    signflip_permutations,
    signflip_seed,
):
    if not summary_rows or not outer_rows:
        return summary_rows

    balanced = _row_values(outer_rows, "balanced_accuracy")
    chance = _summary_chance_values(outer_rows)
    finite = np.isfinite(balanced) & np.isfinite(chance)
    differences = np.full(balanced.shape, np.nan, dtype=float)
    differences[finite] = balanced[finite] - chance[finite]

    chance_classes = _summary_chance_classes(outer_rows, chance)
    top2_chance = _summary_chance_metric_values(
        outer_rows,
        "top2_chance_accuracy",
        multiplier=2.0,
    )
    top3_chance = _summary_chance_metric_values(
        outer_rows,
        "top3_chance_accuracy",
        multiplier=3.0,
    )
    chance_mean_rank = _summary_chance_rank_values(outer_rows, chance_classes)

    for summary_row in summary_rows:
        mean_chance = _nanmean(chance)
        summary_row["chance_accuracy"] = mean_chance
        summary_row["chance_percent"] = _percent(mean_chance)
        summary_row["chance_accuracy_min"] = _nanmin(chance)
        summary_row["chance_accuracy_max"] = _nanmax(chance)
        summary_row["chance_classes_mean"] = _nanmean(chance_classes)
        summary_row["chance_classes_counts"] = _chance_classes_counts(
            outer_rows,
            chance_classes,
        )
        summary_row["top2_chance_accuracy"] = _nanmean(top2_chance)
        summary_row["top2_chance_percent"] = _percent(_nanmean(top2_chance))
        summary_row["top3_chance_accuracy"] = _nanmean(top3_chance)
        summary_row["top3_chance_percent"] = _percent(_nanmean(top3_chance))
        summary_row["chance_mean_rank"] = _nanmean(chance_mean_rank)
        summary_row["mean_above_chance"] = _nanmean(differences)
        summary_row["percent_above_chance"] = _percent(_nanmean(differences))
        summary_row["participants_above_chance"] = _impl._participants_above_chance(
            differences,
        )
        summary_row["participants_total"] = _impl._participants_total(differences)
        summary_row["participants_at_or_below_chance"] = _participants_at_or_below_chance(
            differences,
        )
        summary_row["one_sided_exact_sign_p_value"] = _impl._one_sided_exact_sign_p_value(
            differences,
        )
        summary_row["one_sided_signflip_p_value"] = _impl._one_sided_signflip_p_value(
            differences,
            n_permutations=signflip_permutations,
            seed=signflip_seed,
        )
    return summary_rows


def _row_values(rows, key):
    return np.asarray([_to_float(row.get(key)) for row in rows], dtype=float)


def _summary_chance_values(rows):
    values = []
    for row in rows:
        chance = _row_chance_accuracy(row)
        values.append(np.nan if chance is None else chance)
    return np.asarray(values, dtype=float)


def _summary_chance_classes(rows, chance_values):
    classes = []
    for row, chance in zip(rows, chance_values, strict=True):
        class_count = _row_chance_classes(row)
        if class_count is None and np.isfinite(chance) and chance > 0.0:
            class_count = int(round(1.0 / chance))
        classes.append(np.nan if class_count is None else float(class_count))
    return np.asarray(classes, dtype=float)


def _summary_chance_metric_values(rows, key, *, multiplier):
    values = []
    for row in rows:
        value = _positive_float(row.get(key))
        if value is None:
            chance = _row_chance_accuracy(row)
            value = None if chance is None else min(multiplier * chance, 1.0)
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=float)


def _summary_chance_rank_values(rows, chance_classes):
    values = []
    for row, class_count in zip(rows, chance_classes, strict=True):
        value = _positive_float(row.get("chance_mean_rank"))
        if value is None and np.isfinite(class_count) and class_count > 0.0:
            value = 0.5 * (class_count + 1.0)
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=float)


def _row_chance_accuracy(row):
    chance = _positive_float(row.get("chance_accuracy"))
    if chance is not None:
        return chance
    class_count = _row_chance_classes(row)
    if class_count is None:
        return None
    return 1.0 / class_count


def _row_chance_classes(row):
    for key in ("chance_classes", "n_chance_classes", "n_test_classes"):
        class_count = _positive_int(row.get(key))
        if class_count is not None:
            return class_count
    return None


def _chance_classes_counts(rows, chance_classes):
    counter: Counter[int] = Counter()
    for row, class_count in zip(rows, chance_classes, strict=True):
        parsed = _row_chance_classes(row)
        if parsed is None and np.isfinite(class_count) and class_count > 0.0:
            parsed = int(round(float(class_count)))
        if parsed is not None:
            counter[int(parsed)] += 1
    return _impl._format_counter(counter) if counter else ""


def _participants_at_or_below_chance(differences):
    differences = np.asarray(differences, dtype=float)
    finite = differences[np.isfinite(differences)]
    return int(np.sum(finite <= 0.0))


def _positive_int(value):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value):
    parsed = _to_float(value)
    return float(parsed) if np.isfinite(parsed) and parsed > 0.0 else None


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _nanmin(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def _nanmax(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.nan


def _percent(value):
    return float(100.0 * value) if np.isfinite(value) else np.nan


def _install_module_fixes():
    _impl._score_outer_fold_model = _score_outer_fold_model
    _core._score_outer_fold_model = _score_outer_fold_model
    _impl.summarize_cross_subject_stimulus_smoke = summarize_cross_subject_stimulus_smoke
    _core.summarize_cross_subject_stimulus_smoke = summarize_cross_subject_stimulus_smoke
    _impl.summarize_nested_cross_subject_stimulus = summarize_nested_cross_subject_stimulus
    setattr(_core, "summarize_nested_cross_subject_stimulus", summarize_nested_cross_subject_stimulus)


_install_module_fixes()
