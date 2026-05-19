"""Composed cross-subject stimulus decoding implementation.

The bulk implementation lives in ``_stimulus_cross_subject_impl``. This module
adds the current result-changing scoring and train-only target-alignment
behavior at import time, so the public API no longer depends on package
``__init__`` side effects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
import os

import numpy as np

from pymegdec._reptrace_score_overrides import install_cross_subject

_IMPL_IMPORT_GUARD = "PYMEGDEC_ALLOW_STIMULUS_CROSS_SUBJECT_IMPL_IMPORT"
_previous_impl_import_guard = os.environ.get(_IMPL_IMPORT_GUARD)
os.environ[_IMPL_IMPORT_GUARD] = "1"
try:
    from pymegdec import _stimulus_cross_subject_impl as _impl
finally:
    if _previous_impl_import_guard is None:
        os.environ.pop(_IMPL_IMPORT_GUARD, None)
    else:
        os.environ[_IMPL_IMPORT_GUARD] = _previous_impl_import_guard

DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"
DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0
DEFAULT_CROSS_SUBJECT_TRAINING_SAMPLE_WEIGHTING = "none"
BASELINE_WHITENING_SHRINKAGE = _impl.BASELINE_WHITENING_SHRINKAGE
DEFAULT_CROSS_SUBJECT_NESTED_ALIGNMENTS = (
    _impl.DEFAULT_CROSS_SUBJECT_ALIGNMENT,
    "train_class_procrustes",
)
DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS = (0.0,)
WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER = "window_jitter_score_ensemble"
NESTED_TOPK_WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER = "nested_topk_window_jitter_score_ensemble"
DEFAULT_CROSS_SUBJECT_TEMPORAL_ENSEMBLE_MODE = "probability_mean"
TEMPORAL_ENSEMBLE_MODES = ("probability_mean",)
TEMPORAL_WINDOW_SCORE_ENSEMBLE_CLASSIFIER = "temporal_window_score_ensemble"
TRAINING_SAMPLE_WEIGHTING_MODES = (
    "none",
    "subject_balanced",
    "subject_class_balanced",
)
TRIAL_SELECTION_MODES = ("random", "first")
AUTO_CLASSIFIER_PARAM_GRID_TOKEN = "auto-grid"
TARGET_COVARIANCE_RECOLOR_ALIGNMENT = "target_covariance_recolor"
TARGET_CORAL_ALIGNMENT = "target_coral_unsupervised"
COVARIANCE_ALIGNMENT_SHRINKAGE = 0.1
CORAL_ALIGNMENT_SHRINKAGE = 0.1
_BASE_ALIGNMENT_MODES = tuple(_impl.ALIGNMENT_MODES)
ALIGNMENT_MODES = tuple(
    dict.fromkeys(
        (*_BASE_ALIGNMENT_MODES, TARGET_COVARIANCE_RECOLOR_ALIGNMENT, TARGET_CORAL_ALIGNMENT)
    )
)
AUTO_COMPONENTS_PCA_GRID_TOKEN = "auto-grid"
COMPONENTS_PCA_AUTO_GRID = (32, 64, 128)
CLASSIFIER_AUTO_PARAM_GRIDS = {
    "gaussian-naive-bayes": (1e-12, 1e-9, 1e-6),
    "multiclass-svm": (0.1, 1.0, 10.0),
    "multiclass-svm-weighted": (0.1, 1.0, 10.0),
    "multinomial-logistic": (0.1, 1.0, 10.0),
    "multinomial-logistic-weighted": (0.1, 1.0, 10.0),
    "regularized-qda": (0.25, 0.5, 0.75),
    "shrinkage-lda": ("auto", 0.1, 0.5, 0.9),
    "shrinkage-prototype": (0.0, 0.25, 0.5, 0.75),
}
FEATURE_MODE_PRESETS = {
    "compact_time": ("sensor_mean", "sensor_mean_slope", "sensor_mean_slope_std", "sensor_mean_slope_std_halves"),
    "rich_time": ("sensor_mean", "sensor_mean_slope", "sensor_mean_slope_std", "sensor_mean_slope_std_halves", "sensor_flat"),
}

_BASE_CROSS_SUBJECT_CONFIG = _impl.CrossSubjectStimulusConfig
_BASE_PARTICIPANT_FEATURE_SET = _impl.ParticipantFeatureSet
_ORIGINAL_SCORE_OUTER_FOLD_MODEL = _impl._score_outer_fold_model
_ORIGINAL_SUMMARIZE_CROSS_SUBJECT_STIMULUS_SMOKE = _impl.summarize_cross_subject_stimulus_smoke
_ORIGINAL_SUMMARIZE_NESTED_CROSS_SUBJECT_STIMULUS = _impl.summarize_nested_cross_subject_stimulus
_ORIGINAL_EVALUATE_NESTED_CROSS_SUBJECT_STIMULUS = _impl.evaluate_nested_cross_subject_stimulus
_ORIGINAL_EXPORT_NESTED_CROSS_SUBJECT_STIMULUS = _impl.export_nested_cross_subject_stimulus
_ORIGINAL_CACHED_NESTED_PAIR_ROWS = _impl._cached_nested_pair_rows
_ORIGINAL_SELECT_NESTED_CANDIDATE_ENSEMBLE = _impl._select_nested_candidate_ensemble


@dataclass(frozen=True)
class CrossSubjectStimulusConfig(_BASE_CROSS_SUBJECT_CONFIG):
    """Cross-subject stimulus config with reproducible trial-cap sampling.

    ``train_class_procrustes`` applies only train-derived alignment parameters
    to held-out subjects; scored target trials are not used for centering.

    ``target_coral_unsupervised`` is an explicitly transductive control: it
    uses the held-out participant's unlabeled scored feature distribution to
    match covariance, but never uses held-out labels.
    """

    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED
    training_sample_weighting: str = DEFAULT_CROSS_SUBJECT_TRAINING_SAMPLE_WEIGHTING
    window_jitter_offsets: tuple[float, ...] = DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS
    baseline_whitening_shrinkage: float = BASELINE_WHITENING_SHRINKAGE


@dataclass(frozen=True)
class ParticipantFeatureSet(_BASE_PARTICIPANT_FEATURE_SET):
    """Windowed features with original trial-index bookkeeping."""

    trial_indices: np.ndarray | None = None
    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED


def _ranked_label_metrics(true_labels, class_scores, score_classes):
    """Return rank metrics without dropping unscoreable true-label trials."""

    true_label_ranks = _impl._true_label_ranks(true_labels, class_scores, score_classes)
    finite_ranks = true_label_ranks[np.isfinite(true_label_ranks)]
    if true_label_ranks.size == 0 or class_scores.ndim != 2 or class_scores.shape[1] == 0:
        return {
            "true_label_ranks": true_label_ranks,
            "top2_accuracy": np.nan,
            "top3_accuracy": np.nan,
            "mean_true_label_rank": np.nan,
            "median_true_label_rank": np.nan,
        }
    return {
        "true_label_ranks": true_label_ranks,
        "top2_accuracy": float(np.mean(true_label_ranks <= 2)),
        "top3_accuracy": float(np.mean(true_label_ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
        "median_true_label_rank": float(np.median(finite_ranks)) if finite_ranks.size else np.nan,
    }


def _alignment_model(alignment, *, common_classes, aligned_participants, transforms=(), target_transform=None):
    return {
        "metadata": _impl._alignment_metadata(
            alignment,
            common_classes=common_classes,
            aligned_participants=aligned_participants,
        ),
        "transforms": tuple(transforms),
        "target_transform": target_transform,
    }


def _group_average_channel_procrustes_transform(transforms):
    transforms = tuple(transforms)
    if not transforms:
        return None

    rotations = np.stack([np.asarray(transform["rotation"], dtype=float) for transform in transforms], axis=0)
    mean_rotation = np.mean(rotations, axis=0)
    left, _singular_values, right_t = np.linalg.svd(mean_rotation, full_matrices=False)
    rotation = left @ right_t
    return {
        "source_center": np.mean(
            np.stack([np.asarray(transform["source_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "target_center": np.mean(
            np.stack([np.asarray(transform["target_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "rotation": rotation,
    }


def _fitted_alignment_model(fitted_model):
    alignment_metadata = fitted_model.get("alignment_metadata", {})
    if isinstance(alignment_metadata, dict) and "metadata" in alignment_metadata:
        return alignment_metadata
    return {
        "metadata": alignment_metadata,
        "transforms": tuple(),
        "target_transform": None,
    }


def _train_only_channel_procrustes_transform(target_transform):
    return {
        "source_center": np.asarray(target_transform["source_center"], dtype=float),
        "target_center": np.asarray(target_transform["target_center"], dtype=float),
        "rotation": np.asarray(target_transform["rotation"], dtype=float),
    }


def _test_alignment_metadata(test_transform, target_centering):
    return {"test_transform": test_transform, "target_centering": target_centering}


def _fit_target_covariance_recolor_alignment(features_by_subject):
    features_by_subject = tuple(np.asarray(features, dtype=float) for features in features_by_subject)
    if not features_by_subject:
        return tuple(), None, tuple()

    _validate_same_feature_width(features_by_subject)
    subject_centers = tuple(np.mean(features, axis=0) for features in features_by_subject)
    subject_covariances = tuple(_feature_covariance(features) for features in features_by_subject)
    target_center = np.mean(np.stack(subject_centers, axis=0), axis=0)
    target_covariance = _regularized_covariance(np.mean(np.stack(subject_covariances, axis=0), axis=0))
    transforms = tuple(
        _feature_covariance_recolor_transform(
            features,
            target_center=target_center,
            target_covariance=target_covariance,
        )
        for features in features_by_subject
    )
    aligned_features = tuple(
        _apply_feature_space_affine_transform(features, transform)
        for features, transform in zip(features_by_subject, transforms, strict=True)
    )
    return aligned_features, _target_covariance_template(target_center, target_covariance), transforms


def _validate_same_feature_width(features_by_subject):
    widths = []
    for features in features_by_subject:
        if features.ndim != 2:
            raise ValueError("Covariance alignment requires two-dimensional feature matrices.")
        widths.append(int(features.shape[1]))
    if len(set(widths)) > 1:
        raise ValueError("Covariance alignment requires all participants to have the same feature width.")


def _target_covariance_template(target_center, target_covariance):
    return {
        "target_center": np.asarray(target_center, dtype=float),
        "target_covariance": np.asarray(target_covariance, dtype=float),
    }


def _feature_covariance_recolor_transform(features, *, target_center, target_covariance):
    features = np.asarray(features, dtype=float)
    source_center = np.mean(features, axis=0)
    source_covariance = _feature_covariance(features)
    transform_matrix = _impl._whitening_matrix(source_covariance) @ _covariance_square_root(target_covariance)
    return {
        "source_center": source_center,
        "target_center": np.asarray(target_center, dtype=float),
        "matrix": transform_matrix,
        "source_covariance": source_covariance,
        "target_covariance": np.asarray(target_covariance, dtype=float),
    }


def _feature_covariance(features):
    return _regularized_covariance(_impl._covariance_matrix(features))


def _regularized_covariance(covariance):
    covariance = np.asarray(covariance, dtype=float)
    covariance = _impl._shrink_covariance(covariance, shrinkage=COVARIANCE_ALIGNMENT_SHRINKAGE)
    return 0.5 * (covariance + covariance.T)


def _covariance_square_root(covariance):
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigen_floor = max(float(np.max(eigenvalues)) * _impl.BASELINE_WHITENING_EIGENVALUE_FLOOR, 1e-12)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, eigen_floor))
    square_root = (eigenvectors * sqrt_eigenvalues) @ eigenvectors.T
    return 0.5 * (square_root + square_root.T)


def _apply_feature_space_affine_transform(features, transform):
    return (np.asarray(features, dtype=float) - transform["source_center"]) @ transform["matrix"] + transform["target_center"]


def _align_test_features_by_subject(test_features, test_set, config, alignment_model):
    if config.alignment == "none":
        return test_features, _test_alignment_metadata("none", "none")
    if config.alignment == TARGET_CORAL_ALIGNMENT:
        return test_features, _test_alignment_metadata("none", "target_unsupervised")
    if config.alignment == TARGET_COVARIANCE_RECOLOR_ALIGNMENT:
        target_template = alignment_model.get("target_transform")
        if target_template is None:
            return test_features, _test_alignment_metadata("none", "none")
        test_transform = _feature_covariance_recolor_transform(
            test_features,
            target_center=target_template["target_center"],
            target_covariance=target_template["target_covariance"],
        )
        return (
            _apply_feature_space_affine_transform(test_features, test_transform),
            _test_alignment_metadata(TARGET_COVARIANCE_RECOLOR_ALIGNMENT, "unlabeled_target_features"),
        )
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    target_transform = alignment_model.get("target_transform")
    if target_transform is None:
        return test_features, _test_alignment_metadata("none", "none")

    # Use only train-derived parameters. Re-centering with ``test_features``
    # would make the held-out subject's scored feature distribution influence
    # evaluation, which is a transductive target-alignment step rather than a
    # strict LOSO test.
    test_transform = _train_only_channel_procrustes_transform(target_transform)
    return (
        _impl._apply_channel_procrustes_transform(test_features, test_set, test_transform),
        _test_alignment_metadata("group_average_train_transform", "train_only_group_average"),
    )


def _prediction_group_columns_with_alignment():
    columns = tuple(_impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS)
    additions = ("alignment_test_transform", "alignment_target_centering")
    if all(column in columns for column in additions):
        return columns
    output = []
    for column in columns:
        output.append(column)
        if column == "alignment":
            output.extend(addition for addition in additions if addition not in output)
    return tuple(output)


def _prediction_group_columns_with_trial_selection(columns):
    output = list(columns)
    for column in ("trial_selection", "trial_selection_seed"):
        if column not in output:
            output.append(column)
    return tuple(output)


def _prediction_group_columns_with_window_jitter(columns):
    output = list(columns)
    for column in ("window_jitter_offsets_s", "window_jitter_window_centers_s", "window_jitter_n_offsets"):
        if column not in output:
            output.append(column)
    return tuple(output)


def make_cross_subject_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=_impl.DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
    window_size=_impl.DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    baseline_window=_impl.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    feature_modes=(_impl.DEFAULT_CROSS_SUBJECT_FEATURE_MODE,),
    normalizations=(_impl.DEFAULT_CROSS_SUBJECT_NORMALIZATION,),
    alignments=(_impl.DEFAULT_CROSS_SUBJECT_ALIGNMENT,),
    classifiers=(_impl.DEFAULT_CROSS_SUBJECT_CLASSIFIER,),
    classifier_params=(float("nan"),),
    components_pca_values=(_impl.DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,),
    max_trials_per_class_per_participant=None,
    trial_selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    trial_selection_seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    training_sample_weighting=DEFAULT_CROSS_SUBJECT_TRAINING_SAMPLE_WEIGHTING,
    window_jitter_offsets=DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS,
    baseline_whitening_shrinkage_values=(BASELINE_WHITENING_SHRINKAGE,),
    chance_classes=_impl.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    random_state=0,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    """Build a candidate grid for nested cross-subject model selection."""

    feature_modes = _expand_feature_modes(feature_modes)
    window_jitter_offsets = _normalize_window_jitter_offsets(window_jitter_offsets)

    return tuple(
        CrossSubjectStimulusConfig(
            window_center=window_center,
            window_size=window_size,
            baseline_window=baseline_window,
            feature_mode=feature_mode,
            normalization=normalization,
            alignment=alignment,
            classifier=classifier,
            classifier_param=classifier_param,
            components_pca=components_pca,
            max_trials_per_class_per_participant=max_trials_per_class_per_participant,
            trial_selection=trial_selection,
            trial_selection_seed=trial_selection_seed,
            training_sample_weighting=training_sample_weighting,
            window_jitter_offsets=window_jitter_offsets,
            baseline_whitening_shrinkage=baseline_whitening_shrinkage,
            chance_classes=chance_classes,
            random_state=random_state,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        for window_center, feature_mode, normalization, alignment, classifier, components_pca in _impl.product(
            window_centers,
            feature_modes,
            normalizations,
            alignments,
            classifiers,
            _components_pca_values_for_grid(components_pca_values),
        )
        for baseline_whitening_shrinkage in _baseline_whitening_shrinkage_values_for_normalization(
            normalization,
            baseline_whitening_shrinkage_values,
        )
        for classifier_param in _classifier_params_for_classifier(classifier, classifier_params)
    )


def evaluate_nested_cross_subject_stimulus(
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_participants=None,
    selection_ensemble_size=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
    selection_ensemble_weighting=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    selection_ensemble_temperature=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    selection_ensemble_score_normalization=_impl.DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    selection_ensemble_diversity=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    temporal_ensemble_window_centers=(),
    temporal_ensemble_mode=DEFAULT_CROSS_SUBJECT_TEMPORAL_ENSEMBLE_MODE,
    progress=None,
    existing_artifacts=None,
    after_outer_fold=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested LOSO selection, optionally averaging explicit temporal windows."""

    temporal_ensemble_window_centers = _normalize_temporal_ensemble_window_centers(temporal_ensemble_window_centers)
    temporal_ensemble_mode = _normalize_temporal_ensemble_mode(temporal_ensemble_mode)
    if not temporal_ensemble_window_centers:
        return _ORIGINAL_EVALUATE_NESTED_CROSS_SUBJECT_STIMULUS(
            data_folder,
            participants,
            candidate_configs=candidate_configs,
            outer_participants=outer_participants,
            selection_ensemble_size=selection_ensemble_size,
            selection_ensemble_weighting=selection_ensemble_weighting,
            selection_ensemble_temperature=selection_ensemble_temperature,
            selection_ensemble_score_normalization=selection_ensemble_score_normalization,
            selection_ensemble_diversity=selection_ensemble_diversity,
            progress=progress,
            existing_artifacts=existing_artifacts,
            after_outer_fold=after_outer_fold,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
        )

    candidate_configs = _impl._normalized_candidate_configs(candidate_configs)
    data_folder = _impl.resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for nested cross-subject decoding.")
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required.")
    outer_participants = _impl._normalize_outer_participants(participants, outer_participants)
    selection_ensemble_size = _impl._normalize_selection_ensemble_size(selection_ensemble_size)
    selection_ensemble_weighting = _impl._normalize_selection_ensemble_weighting(selection_ensemble_weighting)
    selection_ensemble_temperature = _impl._normalize_selection_ensemble_temperature(selection_ensemble_temperature)
    selection_ensemble_score_normalization = _impl._normalize_ensemble_score_normalization(selection_ensemble_score_normalization)
    selection_ensemble_diversity = _impl._normalize_selection_ensemble_diversity(selection_ensemble_diversity)

    resumed = _impl._existing_nested_artifact_rows(existing_artifacts)
    inner_rows = resumed["inner_validation"]
    outer_rows = resumed["outer"]
    selected_rows = resumed["selected"]
    prediction_rows = resumed["predictions"]
    completed_outer_folds = {int(row["test_participant"]) for row in outer_rows}
    missing_participants = tuple(participant for participant in outer_participants if participant not in completed_outer_folds)
    feature_cache_configs = _configs_with_temporal_ensemble(candidate_configs, temporal_ensemble_window_centers)
    feature_cache = _load_feature_cache(data_folder, participants, feature_cache_configs, progress=progress) if missing_participants else {}
    inner_pair_cache = {}
    for test_participant in outer_participants:
        if int(test_participant) in completed_outer_folds:
            if progress is not None:
                progress(f"SKIP outer_test_participant={test_participant} resume=complete")
            continue
        outer_row, outer_inner_rows, selected_row, participant_predictions = _evaluate_nested_outer_fold(
            test_participant,
            participants,
            candidate_configs,
            feature_cache,
            inner_pair_cache,
            selection_ensemble_size=selection_ensemble_size,
            selection_ensemble_weighting=selection_ensemble_weighting,
            selection_ensemble_temperature=selection_ensemble_temperature,
            selection_ensemble_score_normalization=selection_ensemble_score_normalization,
            selection_ensemble_diversity=selection_ensemble_diversity,
            temporal_ensemble_window_centers=temporal_ensemble_window_centers,
            temporal_ensemble_mode=temporal_ensemble_mode,
            progress=progress,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
        )
        inner_rows.extend(outer_inner_rows)
        outer_rows.append(outer_row)
        selected_rows.append(selected_row)
        prediction_rows.extend(participant_predictions)
        if after_outer_fold is not None:
            after_outer_fold(_impl._assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs))

    return _impl._assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs)


def export_nested_cross_subject_stimulus(  # pylint: disable=too-many-arguments
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_output_path,
    group_summary_output_path=None,
    inner_validation_output_path=None,
    selected_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    resume=False,
    write_incremental=False,
    outer_participants=None,
    selection_ensemble_size=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
    selection_ensemble_weighting=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    selection_ensemble_temperature=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    selection_ensemble_score_normalization=_impl.DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    selection_ensemble_diversity=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    temporal_ensemble_window_centers=(),
    temporal_ensemble_mode=DEFAULT_CROSS_SUBJECT_TEMPORAL_ENSEMBLE_MODE,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested LOSO cross-subject decoding and write compact CSV artifacts."""

    existing_artifacts = (
        _impl._read_nested_output_rows(
            outer_output_path=outer_output_path,
            inner_validation_output_path=inner_validation_output_path,
            selected_output_path=selected_output_path,
            predictions_output_path=predictions_output_path,
        )
        if resume
        else None
    )

    def write_outputs(current_artifacts):
        _impl._write_nested_output_rows(
            current_artifacts,
            outer_output_path=outer_output_path,
            group_summary_output_path=group_summary_output_path,
            inner_validation_output_path=inner_validation_output_path,
            selected_output_path=selected_output_path,
            predictions_output_path=predictions_output_path,
            confusion_output_path=confusion_output_path,
            per_stimulus_output_path=per_stimulus_output_path,
            confusion_pairs_output_path=confusion_pairs_output_path,
        )

    artifacts = evaluate_nested_cross_subject_stimulus(
        data_folder,
        participants,
        candidate_configs=candidate_configs,
        outer_participants=outer_participants,
        progress=progress,
        existing_artifacts=existing_artifacts,
        after_outer_fold=write_outputs if write_incremental else None,
        selection_ensemble_size=selection_ensemble_size,
        selection_ensemble_weighting=selection_ensemble_weighting,
        selection_ensemble_temperature=selection_ensemble_temperature,
        selection_ensemble_score_normalization=selection_ensemble_score_normalization,
        selection_ensemble_diversity=selection_ensemble_diversity,
        temporal_ensemble_window_centers=temporal_ensemble_window_centers,
        temporal_ensemble_mode=temporal_ensemble_mode,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    write_outputs(artifacts)
    return artifacts


def summarize_nested_cross_subject_stimulus(outer_rows, *, signflip_permutations=10_000, signflip_seed=0):
    """Summarize nested outer scores and preserve temporal-ensemble metadata."""

    rows = _ORIGINAL_SUMMARIZE_NESTED_CROSS_SUBJECT_STIMULUS(
        outer_rows,
        signflip_permutations=signflip_permutations,
        signflip_seed=signflip_seed,
    )
    if not rows or not any(row.get("temporal_ensemble_mode") not in (None, "") for row in outer_rows):
        return rows
    temporal_fields = {
        "temporal_ensemble_mode": _impl._single_row_value(outer_rows, "temporal_ensemble_mode", default=""),
        "temporal_ensemble_size": _impl._single_row_value(outer_rows, "temporal_ensemble_size", default=""),
        "temporal_ensemble_window_centers_s": _impl._single_row_value(outer_rows, "temporal_ensemble_window_centers_s", default=""),
        "temporal_ensemble_base_candidate_counts": _impl._format_counter(
            _impl._row_value_counts(outer_rows, "temporal_ensemble_base_candidate_index", transform=int)
        ),
    }
    for row in rows:
        row.update(temporal_fields)
    return rows


def _expand_feature_modes(feature_modes):
    """Expand nested-grid feature-mode presets into concrete feature modes."""

    expanded: list[str] = []
    for feature_mode in feature_modes:
        normalized = str(feature_mode).strip().lower().replace("-", "_")
        if normalized in FEATURE_MODE_PRESETS:
            expanded.extend(FEATURE_MODE_PRESETS[normalized])
        else:
            expanded.append(_impl._normalize_feature_mode(normalized))
    return tuple(_dedupe_feature_modes(expanded))


def _dedupe_feature_modes(feature_modes):
    seen: set[str] = set()
    for feature_mode in feature_modes:
        if feature_mode not in seen:
            seen.add(feature_mode)
            yield feature_mode


def _components_pca_values_for_grid(components_pca_values):
    values: list[object] = []
    for components_pca in components_pca_values:
        if _is_auto_components_pca_grid(components_pca):
            values.extend(COMPONENTS_PCA_AUTO_GRID)
        else:
            values.append(components_pca)
    return tuple(_dedupe_classifier_params(values))


def _is_auto_components_pca_grid(value):
    return isinstance(value, str) and value.strip().lower().replace("_", "-") == AUTO_COMPONENTS_PCA_GRID_TOKEN


def _baseline_whitening_shrinkage_values_for_grid(values):
    return tuple(
        _normalize_baseline_whitening_shrinkage(value)
        for value in _dedupe_classifier_params(values)
    )


def _baseline_whitening_shrinkage_values_for_normalization(normalization, values):
    if _impl._normalize_normalization(normalization) == "subject_baseline_whiten":
        return _baseline_whitening_shrinkage_values_for_grid(values)
    return (BASELINE_WHITENING_SHRINKAGE,)


def _classifier_params_for_classifier(classifier, classifier_params):
    """Expand classifier-specific parameter grids while preserving explicit values."""

    params: list[object] = []
    for classifier_param in classifier_params:
        if _is_auto_classifier_param_grid(classifier_param):
            params.extend(CLASSIFIER_AUTO_PARAM_GRIDS.get(str(classifier), (float("nan"),)))
        else:
            params.append(classifier_param)
    return tuple(_dedupe_classifier_params(params))


def _is_auto_classifier_param_grid(value):
    return isinstance(value, str) and value.strip().lower().replace("_", "-") == AUTO_CLASSIFIER_PARAM_GRID_TOKEN


def _dedupe_classifier_params(params):
    seen = set()
    for param in params:
        key = _classifier_param_dedupe_key(param)
        if key in seen:
            continue
        seen.add(key)
        yield param


def _classifier_param_dedupe_key(param):
    if isinstance(param, float) and np.isnan(param):
        return ("nan",)
    if isinstance(param, np.generic):
        param = param.item()
    try:
        hash(param)
    except TypeError:
        return ("repr", repr(param))
    return (type(param).__name__, param)


def load_participant_stimulus_features(data_folder, participant, *, config=None):
    """Load one participant's main ``Part*Data.mat`` file and extract fixed-window features."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_path = _impl.Path(_impl.resolve_data_folder(data_folder)) / f"Part{int(participant)}Data.mat"
    data = _impl.sio.loadmat(data_path)["data"][0]
    all_labels = _impl._trialinfo_labels(data)
    trial_indices = _selected_trial_indices(
        all_labels,
        config.max_trials_per_class_per_participant,
        selection=config.trial_selection,
        seed=config.trial_selection_seed,
        participant=participant,
    )
    labels = all_labels[trial_indices]
    features, n_window_samples = _impl._extract_window_features(
        data,
        _impl._centered_window(config.window_center, config.window_size),
        feature_mode=config.feature_mode,
        trial_indices=trial_indices,
    )
    baseline_feature_mean = None
    baseline_feature_std = None
    baseline_whitening_matrix = None
    n_baseline_samples = 0
    if config.normalization in ("subject_baseline_z", "subject_baseline_whiten"):
        baseline_feature_mean, baseline_feature_std, n_baseline_samples = _impl._baseline_feature_statistics(
            data,
            config,
            n_window_samples,
            trial_indices,
        )
    if config.normalization == "subject_baseline_whiten":
        baseline_whitening_matrix, n_baseline_samples = _impl._baseline_channel_whitening_matrix(
            data,
            config.baseline_window,
            trial_indices,
            shrinkage=config.baseline_whitening_shrinkage,
        )
    normalized_features = _impl._normalize_features(
        features,
        config,
        baseline_feature_mean,
        baseline_feature_std,
        baseline_whitening_matrix,
    )
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} labels but {features.shape[0]} feature rows.")
    return ParticipantFeatureSet(
        participant=int(participant),
        labels=labels,
        features=normalized_features,
        normalization=config.normalization,
        baseline_features=None,
        baseline_feature_mean=baseline_feature_mean,
        baseline_feature_std=baseline_feature_std,
        baseline_whitening_matrix=baseline_whitening_matrix,
        n_channels=int(_impl._trial_signal(data, 0).shape[0]),
        n_window_samples=int(n_window_samples),
        n_baseline_samples=int(n_baseline_samples),
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        trial_indices=np.asarray(trial_indices, dtype=int),
        trial_selection=config.trial_selection,
        trial_selection_seed=config.trial_selection_seed,
    )


def summarize_cross_subject_stimulus_smoke(outer_rows, *, config=None):
    """Summarize held-out participant scores and include trial-selection metadata."""

    rows = _ORIGINAL_SUMMARIZE_CROSS_SUBJECT_STIMULUS_SMOKE(outer_rows, config=config)
    config = _normalized_config(config or CrossSubjectStimulusConfig())
    for row in rows:
        row["trial_selection"] = config.trial_selection
        row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
        row["training_sample_weighting"] = config.training_sample_weighting
        row["baseline_whitening_shrinkage"] = config.baseline_whitening_shrinkage
    return rows


def _align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config):
    if config.alignment == "none":
        return features_by_subject, _alignment_model(
            config.alignment,
            common_classes=(),
            aligned_participants=(),
        )
    if config.alignment == TARGET_CORAL_ALIGNMENT:
        return features_by_subject, _alignment_model(
            config.alignment,
            common_classes=(),
            aligned_participants=(),
        )
    if config.alignment == TARGET_COVARIANCE_RECOLOR_ALIGNMENT:
        aligned_features, target_template, transforms = _fit_target_covariance_recolor_alignment(features_by_subject)
        return list(aligned_features), _alignment_model(
            config.alignment,
            common_classes=(),
            aligned_participants=(feature_set.participant for feature_set in feature_sets),
            transforms=transforms,
            target_transform=target_template,
        )
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    common_classes = _impl._common_label_values(labels_by_subject)
    if len(common_classes) < 2:
        return features_by_subject, _alignment_model(
            config.alignment,
            common_classes=common_classes,
            aligned_participants=(),
        )

    class_patterns = [
        _impl._participant_class_channel_patterns(features, labels, feature_set, common_classes)
        for feature_set, features, labels in zip(feature_sets, features_by_subject, labels_by_subject, strict=True)
    ]
    transforms = _impl._fit_channel_procrustes_transforms(class_patterns)
    aligned_features = [
        _impl._apply_channel_procrustes_transform(features, feature_set, transform)
        for feature_set, features, transform in zip(feature_sets, features_by_subject, transforms, strict=True)
    ]
    return aligned_features, _alignment_model(
        config.alignment,
        common_classes=common_classes,
        aligned_participants=(feature_set.participant for feature_set in feature_sets),
        transforms=transforms,
        target_transform=_group_average_channel_procrustes_transform(transforms),
    )


def _training_sample_weights_by_subject(train_sets, train_label_arrays, mode):
    """Return row weights that prevent source participants with many trials from dominating."""

    mode = _normalize_training_sample_weighting(mode)
    if mode == "none":
        return None
    if len(train_sets) != len(train_label_arrays):
        raise ValueError("train_sets and train_label_arrays must have the same length.")

    weights_by_subject = []
    for feature_set, labels in zip(train_sets, train_label_arrays, strict=True):
        labels = np.asarray(labels, dtype=int).ravel()
        if labels.shape[0] != np.asarray(feature_set.features).shape[0]:
            raise ValueError(f"Participant {feature_set.participant} has inconsistent feature and label counts.")
        if labels.size == 0:
            raise ValueError("Cannot build subject-balanced weights for an empty training participant.")
        if mode == "subject_balanced":
            weights = np.full(labels.shape[0], 1.0 / labels.shape[0], dtype=float)
        else:
            counts = Counter(int(label) for label in labels.tolist())
            n_classes = max(len(counts), 1)
            weights = np.asarray([1.0 / (n_classes * counts[int(label)]) for label in labels], dtype=float)
        weights_by_subject.append(weights)

    weights = np.concatenate(weights_by_subject)
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0 or not np.isfinite(mean_weight):
        raise ValueError("Computed invalid training sample weights.")
    return weights / mean_weight


def _sample_weight_summary(sample_weight):
    if sample_weight is None:
        return {
            "train_sample_weight_min": np.nan,
            "train_sample_weight_max": np.nan,
            "train_sample_weight_mean": np.nan,
        }
    sample_weight = np.asarray(sample_weight, dtype=float)
    return {
        "train_sample_weight_min": float(np.min(sample_weight)),
        "train_sample_weight_max": float(np.max(sample_weight)),
        "train_sample_weight_mean": float(np.mean(sample_weight)),
    }


def _fit_outer_fold_model(train_sets, config, classifier_param, *, label_shuffle_seed=None, label_shuffle_context=()):
    train_features_by_subject = [_impl._normalized_subject_features(feature_set, config) for feature_set in train_sets]
    train_label_arrays = [
        _impl._training_labels(
            feature_set,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=label_shuffle_context,
        )
        for feature_set in train_sets
    ]
    train_sample_weight = _training_sample_weights_by_subject(train_sets, train_label_arrays, config.training_sample_weighting)
    train_features_by_subject, alignment_metadata = _align_training_features_by_subject(
        train_sets,
        train_features_by_subject,
        train_label_arrays,
        config,
    )
    train_features = np.vstack(train_features_by_subject)
    train_labels_one_based = np.concatenate(train_label_arrays)
    train_labels = train_labels_one_based - 1

    train_window = _impl._centered_window(config.window_center, config.window_size)
    model_bundle = _impl.fit_reptrace_window_model(
        train_features,
        train_labels,
        fit_model=lambda features, labels: _impl.train_multiclass_classifier(
            features,
            labels,
            config.classifier,
            classifier_param,
            random_state=config.random_state,
            sample_weight=train_sample_weight,
        ),
        components_pca=config.components_pca,
        train_window=train_window,
    )
    return {
        "classifier_param": classifier_param,
        "model_bundle": model_bundle,
        "n_train_participants": len(train_sets),
        "train_class_counts": Counter(train_labels_one_based.tolist()),
        "train_labels": train_labels,
        "train_participants": tuple(feature_set.participant for feature_set in train_sets),
        "train_window": train_window,
        "label_shuffle_control": label_shuffle_seed is not None,
        "label_shuffle_seed": "" if label_shuffle_seed is None else int(label_shuffle_seed),
        "alignment_metadata": alignment_metadata,
        "training_sample_weighting": config.training_sample_weighting,
        "target_coral_model": _fit_target_coral_model(train_features, config),
        **_sample_weight_summary(train_sample_weight),
    }


def _fit_target_coral_model(train_features, config):
    if config.alignment != TARGET_CORAL_ALIGNMENT:
        return None
    source_features = np.asarray(train_features, dtype=float)
    source_covariance = _coral_covariance(source_features)
    return {
        "source_mean": np.mean(source_features, axis=0, keepdims=True),
        "source_coloring_matrix": _covariance_square_root(source_covariance),
    }


def _apply_target_coral_model(test_features, config, fitted_model):
    if config.alignment != TARGET_CORAL_ALIGNMENT:
        return test_features, None
    coral_model = fitted_model.get("target_coral_model")
    if coral_model is None:
        raise ValueError("target_coral_unsupervised requires a fitted target_coral_model.")

    test_features = np.asarray(test_features, dtype=float)
    target_mean = np.mean(test_features, axis=0, keepdims=True)
    target_whitening_matrix = _impl._whitening_matrix(_coral_covariance(test_features))
    source_coloring_matrix = np.asarray(coral_model["source_coloring_matrix"], dtype=float)
    source_mean = np.asarray(coral_model["source_mean"], dtype=float)
    aligned_features = (
        (test_features - target_mean)
        @ target_whitening_matrix.T
        @ source_coloring_matrix.T
        + source_mean
    )
    return aligned_features, _test_alignment_metadata(
        "target_coral_to_source", "target_unsupervised"
    )


def _coral_covariance(features):
    covariance = _impl._covariance_matrix(features)
    return _impl._shrink_covariance(covariance, shrinkage=CORAL_ALIGNMENT_SHRINKAGE)


def _symmetric_matrix_square_root(covariance):
    return _covariance_square_root(covariance)


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    alignment_model = _fitted_alignment_model(fitted_model)
    test_features = _impl._normalized_subject_features(test_set, config)
    test_features, test_alignment_metadata = _align_test_features_by_subject(
        test_features,
        test_set,
        config,
        alignment_model,
    )
    test_features, target_coral_metadata = _apply_target_coral_model(test_features, config, fitted_model)
    if target_coral_metadata is not None:
        test_alignment_metadata = target_coral_metadata
    scoring_set = replace(test_set, features=test_features, normalization=config.normalization)
    scoring_model = dict(fitted_model)
    scoring_model["alignment_metadata"] = alignment_model["metadata"]
    outer_row, prediction_rows = _ORIGINAL_SCORE_OUTER_FOLD_MODEL(
        scoring_model,
        scoring_set,
        config,
        include_predictions=include_predictions,
    )
    outer_row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
    outer_row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
    outer_row["training_sample_weighting"] = config.training_sample_weighting
    outer_row["baseline_whitening_shrinkage"] = config.baseline_whitening_shrinkage
    outer_row["train_sample_weight_min"] = fitted_model.get("train_sample_weight_min", np.nan)
    outer_row["train_sample_weight_max"] = fitted_model.get("train_sample_weight_max", np.nan)
    outer_row["train_sample_weight_mean"] = fitted_model.get("train_sample_weight_mean", np.nan)
    outer_row["trial_selection"] = config.trial_selection
    outer_row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
    for row in prediction_rows:
        row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
        row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
        row["training_sample_weighting"] = config.training_sample_weighting
        row["baseline_whitening_shrinkage"] = config.baseline_whitening_shrinkage
        row["trial_selection"] = config.trial_selection
        row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
    return outer_row, prediction_rows


def _candidate_model_scores(fitted_model, test_set, config):
    alignment_model = _fitted_alignment_model(fitted_model)
    test_features = _impl._normalized_subject_features(test_set, config)
    test_features, _test_alignment_metadata = _align_test_features_by_subject(
        test_features,
        test_set,
        config,
        alignment_model,
    )
    test_features, _target_coral_metadata = _apply_target_coral_model(test_features, config, fitted_model)
    return _impl._model_class_scores(fitted_model["model_bundle"], test_features)


def _load_feature_cache(data_folder, participants, candidate_configs, *, progress=None):
    """Load all feature windows required by nominal and jittered candidates."""

    representative_configs = {}
    for candidate_config in candidate_configs:
        for feature_config in _feature_cache_configs(candidate_config):
            representative_configs.setdefault(_feature_cache_key(feature_config), feature_config)

    feature_cache = {}
    for key, feature_config in representative_configs.items():
        if progress is not None:
            progress(
                "LOAD feature_set "
                f"window_center={feature_config.window_center} "
                f"feature_mode={feature_config.feature_mode} "
                f"normalization={feature_config.normalization} "
                f"alignment={feature_config.alignment}"
            )
        feature_cache[key] = {
            participant: _impl.load_participant_stimulus_features(data_folder, participant, config=feature_config)
            for participant in participants
        }
    return feature_cache


def _evaluate_nested_inner_rows(
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    inner_pair_cache,
    *,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    inner_rows = []
    completed = 0
    total = len(candidate_configs) * len(outer_train_participants)
    for candidate_index, candidate_config in enumerate(candidate_configs, start=1):
        feature_sets = feature_cache[_feature_cache_key(candidate_config)]
        for validation_participant in outer_train_participants:
            excluded_pair = tuple(sorted((int(test_participant), int(validation_participant))))
            if _has_window_jitter(candidate_config):
                pair_rows = _cached_window_jitter_nested_pair_rows(
                    candidate_index,
                    candidate_config,
                    excluded_pair,
                    feature_cache,
                    inner_pair_cache,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                )
            else:
                pair_rows = _ORIGINAL_CACHED_NESTED_PAIR_ROWS(
                    candidate_index,
                    candidate_config,
                    excluded_pair,
                    feature_sets,
                    inner_pair_cache,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                )
            inner_rows.append(pair_rows[(int(test_participant), int(validation_participant))])
            completed += 1
            if progress is not None:
                progress(
                    "DONE inner_validation "
                    f"outer_test_participant={test_participant} "
                    f"candidate={candidate_index}/{len(candidate_configs)} "
                    f"validation_participant={validation_participant} "
                    f"progress={completed}/{total}"
                )
    return inner_rows


def _cached_window_jitter_nested_pair_rows(
    candidate_index,
    candidate_config,
    excluded_pair,
    feature_cache,
    inner_pair_cache,
    *,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    jitter_offsets = _window_jitter_offsets(candidate_config)
    cache_key = (
        "window_jitter",
        int(candidate_index),
        tuple(int(participant) for participant in excluded_pair),
        jitter_offsets,
        bool(label_shuffle_control),
        int(label_shuffle_seed),
    )
    if cache_key not in inner_pair_cache:
        member_configs = _jittered_window_configs(candidate_config)
        fitted_models = []
        for offset_rank, member_config in enumerate(member_configs):
            member_feature_sets = feature_cache[_feature_cache_key(member_config)]
            train_sets = [feature_set for participant, feature_set in member_feature_sets.items() if int(participant) not in excluded_pair]
            fitted_models.append(
                _impl._fit_outer_fold_model(
                    train_sets,
                    member_config,
                    _impl._resolved_classifier_param(member_config),
                    label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
                    label_shuffle_context=(int(candidate_index), *tuple(int(participant) for participant in excluded_pair), int(offset_rank)),
                )
            )

        first_participant, second_participant = excluded_pair
        pair_rows = {}
        for outer_test_participant, validation_participant in (
            (first_participant, second_participant),
            (second_participant, first_participant),
        ):
            test_sets = [feature_cache[_feature_cache_key(member_config)][validation_participant] for member_config in member_configs]
            inner_row, _predictions = _score_window_model_collection(
                fitted_models,
                test_sets,
                member_configs,
                _repeated_selected_rows({"selected_candidate_index": int(candidate_index)}, len(member_configs)),
                nominal_configs=(candidate_config,),
                ensemble_weights=None,
                ensemble_weighting=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
                ensemble_temperature=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
                ensemble_score_normalization=_impl.DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
                include_predictions=False,
            )
            pair_rows[(outer_test_participant, validation_participant)] = _impl._nested_inner_row(
                inner_row,
                outer_test_participant,
                validation_participant,
                candidate_index,
            )
        inner_pair_cache[cache_key] = pair_rows
    return inner_pair_cache[cache_key]


def _evaluate_nested_outer_fold(
    test_participant,
    participants,
    candidate_configs,
    feature_cache,
    inner_pair_cache,
    *,
    selection_ensemble_size=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
    selection_ensemble_weighting=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    selection_ensemble_temperature=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    selection_ensemble_score_normalization=_impl.DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    selection_ensemble_diversity=_impl.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    temporal_ensemble_window_centers=(),
    temporal_ensemble_mode=DEFAULT_CROSS_SUBJECT_TEMPORAL_ENSEMBLE_MODE,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    temporal_ensemble_window_centers = _normalize_temporal_ensemble_window_centers(temporal_ensemble_window_centers)
    temporal_ensemble_mode = _normalize_temporal_ensemble_mode(temporal_ensemble_mode)
    if progress is not None:
        progress(f"START outer_test_participant={test_participant}")
    outer_train_participants = tuple(participant for participant in participants if participant != test_participant)
    outer_inner_rows = _evaluate_nested_inner_rows(
        test_participant,
        outer_train_participants,
        candidate_configs,
        feature_cache,
        inner_pair_cache,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    selected_row, selected_candidate_rows = _select_nested_candidate_ensemble(
        outer_inner_rows,
        selection_ensemble_size=selection_ensemble_size,
        selection_ensemble_weighting=selection_ensemble_weighting,
        selection_ensemble_temperature=selection_ensemble_temperature,
        selection_ensemble_score_normalization=selection_ensemble_score_normalization,
        selection_ensemble_diversity=selection_ensemble_diversity,
        candidate_configs=candidate_configs,
    )
    base_weights = _impl._nested_ensemble_weights(
        selected_candidate_rows,
        weighting=selected_row["selection_ensemble_weighting"],
        temperature=selected_row["selection_ensemble_temperature"],
    )
    if temporal_ensemble_window_centers:
        fitted_models, test_sets, member_configs, model_selected_rows, model_weights = _fit_temporal_window_model_collection(
            int(test_participant),
            outer_train_participants,
            candidate_configs,
            feature_cache,
            selected_candidate_rows,
            base_weights,
            window_centers=temporal_ensemble_window_centers,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
        )
        outer_row, participant_predictions = _score_window_model_collection(
            fitted_models,
            test_sets,
            member_configs,
            model_selected_rows,
            nominal_configs=member_configs,
            ensemble_weights=model_weights,
            ensemble_weighting=selected_row["selection_ensemble_weighting"],
            ensemble_temperature=selected_row["selection_ensemble_temperature"],
            ensemble_score_normalization=selected_row["selection_ensemble_score_normalization"],
            include_predictions=True,
        )
        _add_temporal_ensemble_selected_fields(
            selected_row,
            window_centers=temporal_ensemble_window_centers,
            mode=temporal_ensemble_mode,
        )
        _add_temporal_ensemble_output_fields(
            outer_row,
            temporal_configs=member_configs,
            window_centers=temporal_ensemble_window_centers,
            mode=temporal_ensemble_mode,
            base_candidate_index=int(selected_row["selected_candidate_index"]),
            weights=model_weights,
        )
        for prediction_row in participant_predictions:
            _add_temporal_ensemble_output_fields(
                prediction_row,
                temporal_configs=member_configs,
                window_centers=temporal_ensemble_window_centers,
                mode=temporal_ensemble_mode,
                base_candidate_index=int(selected_row["selected_candidate_index"]),
                weights=model_weights,
            )
    else:
        fitted_models, test_sets, member_configs, model_selected_rows, model_weights, nominal_configs = _fit_selected_window_model_collection(
            int(test_participant),
            outer_train_participants,
            candidate_configs,
            feature_cache,
            selected_candidate_rows,
            base_weights,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
        )
        outer_row, participant_predictions = _score_window_model_collection(
            fitted_models,
            test_sets,
            member_configs,
            model_selected_rows,
            nominal_configs=nominal_configs,
            ensemble_weights=model_weights,
            ensemble_weighting=selected_row["selection_ensemble_weighting"],
            ensemble_temperature=selected_row["selection_ensemble_temperature"],
            ensemble_score_normalization=selected_row["selection_ensemble_score_normalization"],
            include_predictions=True,
        )
    _impl._add_selected_candidate_fields(outer_row, selected_row)
    for prediction_row in participant_predictions:
        _impl._add_selected_candidate_fields(prediction_row, selected_row)
    if progress is not None:
        progress(
            "DONE outer_test_participant="
            f"{test_participant} selected_candidate={selected_row['selected_candidate_index']} "
            f"selection_ensemble_size={selected_row['selection_ensemble_size']} "
            f"selection_ensemble_diversity={selected_row['selection_ensemble_diversity']} "
            f"score_normalization={selected_row['selection_ensemble_score_normalization']} "
            f"selection_ensemble_weighting={selected_row['selection_ensemble_weighting']} "
            f"temporal_ensemble_size={len(temporal_ensemble_window_centers) if temporal_ensemble_window_centers else 0} "
            f"inner_mean={selected_row['selected_inner_balanced_accuracy_mean']:.4f} "
            f"outer_balanced_accuracy={outer_row['balanced_accuracy']:.4f}"
        )
    return outer_row, outer_inner_rows, selected_row, participant_predictions


def _fit_selected_window_model_collection(
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    selected_candidate_rows,
    base_weights,
    *,
    label_shuffle_seed=None,
):
    fitted_models = []
    test_sets = []
    member_configs = []
    model_selected_rows = []
    model_weights = []
    nominal_configs = []
    for ensemble_rank, (candidate_row, base_weight) in enumerate(zip(selected_candidate_rows, base_weights, strict=True)):
        candidate_index = int(candidate_row["selected_candidate_index"])
        nominal_config = candidate_configs[candidate_index - 1]
        nominal_configs.append(nominal_config)
        jittered_configs = _jittered_window_configs(nominal_config)
        member_weight = float(base_weight) / len(jittered_configs)
        for offset_rank, member_config in enumerate(jittered_configs):
            member_feature_sets = feature_cache[_feature_cache_key(member_config)]
            train_sets = [member_feature_sets[participant] for participant in outer_train_participants]
            fitted_models.append(
                _impl._fit_outer_fold_model(
                    train_sets,
                    member_config,
                    _impl._resolved_classifier_param(member_config),
                    label_shuffle_seed=label_shuffle_seed,
                    label_shuffle_context=(int(test_participant), candidate_index, int(ensemble_rank), int(offset_rank)),
                )
            )
            test_sets.append(member_feature_sets[test_participant])
            member_configs.append(member_config)
            model_selected_rows.append(candidate_row)
            model_weights.append(member_weight)
    return fitted_models, test_sets, member_configs, tuple(model_selected_rows), np.asarray(model_weights, dtype=float), tuple(nominal_configs)


def _fit_temporal_window_model_collection(
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    selected_candidate_rows,
    base_weights,
    *,
    window_centers,
    label_shuffle_seed=None,
):
    window_centers = _normalize_temporal_ensemble_window_centers(window_centers)
    fitted_models = []
    test_sets = []
    member_configs = []
    model_selected_rows = []
    model_weights = []
    for ensemble_rank, (candidate_row, base_weight) in enumerate(zip(selected_candidate_rows, base_weights, strict=True)):
        candidate_index = int(candidate_row["selected_candidate_index"])
        nominal_config = candidate_configs[candidate_index - 1]
        temporal_configs = _temporal_window_configs(nominal_config, window_centers)
        member_weight = float(base_weight) / len(temporal_configs)
        for temporal_rank, member_config in enumerate(temporal_configs):
            member_feature_sets = feature_cache[_feature_cache_key(member_config)]
            train_sets = [member_feature_sets[participant] for participant in outer_train_participants]
            fitted_models.append(
                _impl._fit_outer_fold_model(
                    train_sets,
                    member_config,
                    _impl._resolved_classifier_param(member_config),
                    label_shuffle_seed=label_shuffle_seed,
                    label_shuffle_context=(int(test_participant), candidate_index, int(ensemble_rank), int(temporal_rank)),
                )
            )
            test_sets.append(member_feature_sets[test_participant])
            member_configs.append(member_config)
            model_selected_rows.append(candidate_row)
            model_weights.append(member_weight)
    return fitted_models, test_sets, tuple(member_configs), tuple(model_selected_rows), np.asarray(model_weights, dtype=float)


def _score_window_model_collection(
    fitted_models,
    test_sets,
    member_configs,
    selected_rows,
    *,
    nominal_configs,
    ensemble_weights=None,
    ensemble_weighting,
    ensemble_temperature,
    ensemble_score_normalization,
    include_predictions=True,
):
    if len(fitted_models) == 1:
        outer_row, prediction_rows = _score_outer_fold_model(fitted_models[0], test_sets[0], member_configs[0], include_predictions=include_predictions)
    else:
        outer_row, prediction_rows = _impl._score_outer_fold_ensemble_models(
            fitted_models,
            test_sets,
            member_configs,
            selected_rows,
            ensemble_weights=ensemble_weights,
            ensemble_weighting=ensemble_weighting,
            ensemble_temperature=ensemble_temperature,
            ensemble_score_normalization=ensemble_score_normalization,
            include_predictions=include_predictions,
        )
    if any(_has_window_jitter(config) for config in nominal_configs):
        _add_window_jitter_output_fields(
            outer_row,
            nominal_configs,
            member_configs,
            ensemble_score_normalization=ensemble_score_normalization,
        )
        for prediction_row in prediction_rows:
            _add_window_jitter_output_fields(
                prediction_row,
                nominal_configs,
                member_configs,
                ensemble_score_normalization=ensemble_score_normalization,
            )
    return outer_row, prediction_rows


def _select_nested_candidate_ensemble(
    inner_rows,
    *,
    selection_ensemble_size,
    selection_ensemble_weighting,
    selection_ensemble_temperature,
    selection_ensemble_score_normalization,
    selection_ensemble_diversity,
    candidate_configs,
):
    selected, selected_rows = _ORIGINAL_SELECT_NESTED_CANDIDATE_ENSEMBLE(
        inner_rows,
        selection_ensemble_size=selection_ensemble_size,
        selection_ensemble_weighting=selection_ensemble_weighting,
        selection_ensemble_temperature=selection_ensemble_temperature,
        selection_ensemble_score_normalization=selection_ensemble_score_normalization,
        selection_ensemble_diversity=selection_ensemble_diversity,
        candidate_configs=candidate_configs,
    )
    selected_rows = tuple(_candidate_row_with_window_jitter_fields(row, candidate_configs) for row in selected_rows)
    _add_selected_window_jitter_fields(selected, selected_rows, candidate_configs)
    return selected, selected_rows


def _feature_cache_key(config):
    return (
        float(config.window_center),
        float(config.window_size),
        float(config.baseline_window[0]),
        float(config.baseline_window[1]),
        str(config.feature_mode),
        str(config.normalization),
        float(config.baseline_whitening_shrinkage),
        config.max_trials_per_class_per_participant,
        str(getattr(config, "trial_selection", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)),
        _seed_field(getattr(config, "trial_selection_seed", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)),
    )


def _feature_cache_configs(config):
    return _jittered_window_configs(config) if _has_window_jitter(config) else (_concrete_window_config(config),)


def _configs_with_temporal_ensemble(candidate_configs, window_centers):
    window_centers = _normalize_temporal_ensemble_window_centers(window_centers)
    if not window_centers:
        return tuple(candidate_configs)
    configs = list(candidate_configs)
    for config in candidate_configs:
        configs.extend(_temporal_window_configs(config, window_centers))
    return tuple(configs)


def _temporal_window_configs(base_config, window_centers):
    return tuple(
        _concrete_window_config(replace(base_config, window_center=float(center)))
        for center in _normalize_temporal_ensemble_window_centers(window_centers)
    )


def _jittered_window_configs(config):
    offsets = _window_jitter_offsets(config)
    return tuple(_concrete_window_config(replace(config, window_center=float(np.round(config.window_center + offset, 10)))) for offset in offsets)


def _concrete_window_config(config):
    return replace(config, window_jitter_offsets=DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS)


def _has_window_jitter(config):
    offsets = _window_jitter_offsets(config)
    return len(offsets) > 1


def _window_jitter_offsets(config):
    return _normalize_window_jitter_offsets(getattr(config, "window_jitter_offsets", DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS))


def _normalize_window_jitter_offsets(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        values = [0.0]
    elif isinstance(value, str):
        values = [float(offset.strip()) for offset in value.split(",") if offset.strip()]
    elif isinstance(value, (int, float, np.integer, np.floating)):
        values = [float(value)]
    else:
        values = [float(offset) for offset in value]
    if not values:
        values = [0.0]

    normalized = []
    seen = set()
    for offset in (0.0, *values):
        offset = float(offset)
        if not np.isfinite(offset):
            raise ValueError("window_jitter_offsets must contain only finite offsets.")
        if abs(offset) < 1e-12:
            offset = 0.0
        key = round(offset, 12)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(offset)
    return tuple(normalized)


def _normalize_temporal_ensemble_window_centers(value):
    if value is None or value == "":
        return tuple()
    if isinstance(value, str):
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        try:
            values = tuple(float(item) for item in value)
        except TypeError:
            values = (float(value),)
    if not all(np.isfinite(item) for item in values):
        raise ValueError("temporal_ensemble_window_centers must contain only finite values.")
    return values


def _normalize_temporal_ensemble_mode(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in TEMPORAL_ENSEMBLE_MODES:
        raise ValueError(f"temporal_ensemble_mode must be one of {TEMPORAL_ENSEMBLE_MODES}.")
    return normalized


def _candidate_row_with_window_jitter_fields(row, candidate_configs):
    output = dict(row)
    candidate_config = candidate_configs[int(output["selected_candidate_index"]) - 1]
    output["selected_classifier"] = candidate_config.classifier
    output["selected_baseline_whitening_shrinkage"] = candidate_config.baseline_whitening_shrinkage
    output["selected_window_jitter_offsets_s"] = _format_window_jitter_offsets(candidate_config)
    output["selected_window_jitter_n_offsets"] = len(_window_jitter_offsets(candidate_config))
    return output


def _add_selected_window_jitter_fields(selected, selected_rows, candidate_configs):
    selected_config = candidate_configs[int(selected["selected_candidate_index"]) - 1]
    selected["selected_classifier"] = selected_config.classifier
    selected["selected_baseline_whitening_shrinkage"] = selected_config.baseline_whitening_shrinkage
    selected["selected_ensemble_baseline_whitening_shrinkage_counts"] = _impl._format_counter(
        Counter(
            float(candidate_configs[int(row["selected_candidate_index"]) - 1].baseline_whitening_shrinkage)
            for row in selected_rows
        )
    )
    selected["selected_window_jitter_offsets_s"] = _format_window_jitter_offsets(selected_config)
    selected["selected_window_jitter_n_offsets"] = len(_window_jitter_offsets(selected_config))
    selected["selected_ensemble_window_jitter_offsets_s"] = _format_sequence(
        _format_window_jitter_offsets(candidate_configs[int(row["selected_candidate_index"]) - 1]) for row in selected_rows
    )
    selected["selected_ensemble_window_jitter_n_offsets"] = _format_sequence(
        len(_window_jitter_offsets(candidate_configs[int(row["selected_candidate_index"]) - 1])) for row in selected_rows
    )


def _repeated_selected_rows(row, n_rows):
    return tuple(dict(row) for _ in range(int(n_rows)))


def _add_window_jitter_output_fields(row, nominal_configs, member_configs, *, ensemble_score_normalization):
    nominal_configs = tuple(nominal_configs)
    member_configs = tuple(member_configs)
    selected_candidate_count = len(nominal_configs)
    jitter_offsets = tuple(_window_jitter_offsets(config) for config in nominal_configs)
    jitter_n_models = int(sum(len(offsets) for offsets in jitter_offsets))
    row["outer_evaluation_mode"] = (
        "window_jitter_score_ensemble" if selected_candidate_count == 1 else "nested_topk_window_jitter_score_ensemble"
    )
    row["classifier"] = (
        WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER
        if selected_candidate_count == 1
        else NESTED_TOPK_WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER
    )
    row["window_center_s"] = nominal_configs[0].window_center
    window_start, window_stop = _impl._centered_window(nominal_configs[0].window_center, nominal_configs[0].window_size)
    row["window_start_s"] = window_start
    row["window_stop_s"] = window_stop
    row["window_jitter_offsets_s"] = _format_window_jitter_offset_groups(nominal_configs)
    row["window_jitter_window_centers_s"] = _format_sequence(float(config.window_center) for config in member_configs)
    row["window_jitter_n_offsets"] = jitter_n_models
    row["window_jitter_score_normalization"] = _impl._normalize_ensemble_score_normalization(ensemble_score_normalization)
    row["ensemble_baseline_whitening_shrinkages"] = _format_sequence(
        config.baseline_whitening_shrinkage for config in nominal_configs
    )


def _add_temporal_ensemble_selected_fields(selected_row, *, window_centers, mode):
    window_centers = _normalize_temporal_ensemble_window_centers(window_centers)
    selected_row["temporal_ensemble_mode"] = _normalize_temporal_ensemble_mode(mode)
    selected_row["temporal_ensemble_size"] = int(len(window_centers))
    selected_row["temporal_ensemble_window_centers_s"] = _format_sequence(window_centers)


def _add_temporal_ensemble_output_fields(
    row,
    *,
    temporal_configs,
    window_centers,
    mode,
    base_candidate_index,
    weights,
):
    window_centers = _normalize_temporal_ensemble_window_centers(window_centers)
    weights = np.asarray(weights, dtype=float).ravel()
    row["outer_evaluation_mode"] = "temporal_window_score_ensemble"
    row["classifier"] = TEMPORAL_WINDOW_SCORE_ENSEMBLE_CLASSIFIER
    row["classifier_param"] = ""
    row["window_center_s"] = ""
    row["window_start_s"] = ""
    row["window_stop_s"] = ""
    row["temporal_ensemble_mode"] = _normalize_temporal_ensemble_mode(mode)
    row["temporal_ensemble_size"] = int(len(window_centers))
    row["temporal_ensemble_base_candidate_index"] = int(base_candidate_index)
    row["temporal_ensemble_window_centers_s"] = _format_sequence(window_centers)
    row["temporal_ensemble_model_window_centers_s"] = _format_sequence(config.window_center for config in temporal_configs)
    row["temporal_ensemble_weights"] = _impl._format_float_mapping(enumerate(weights, start=1))


def _format_window_jitter_offsets(config):
    return _format_sequence(float(offset) for offset in _window_jitter_offsets(config))


def _format_window_jitter_offset_groups(configs):
    groups = []
    for config in configs:
        group = _format_window_jitter_offsets(config)
        if group not in groups:
            groups.append(group)
    return groups[0] if len(groups) == 1 else "|".join(groups)


def _format_sequence(values):
    return ";".join(str(value) for value in values)


def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):
    train_window = _impl._centered_window(config.window_center, config.window_size)
    trial_indices = _feature_set_trial_indices(test_set)
    rows = []
    for trial_idx, true_label, predicted_label, true_label_rank in zip(trial_indices, test_labels, predictions, true_label_ranks, strict=True):
        true_stimulus = int(true_label) + 1
        predicted_stimulus = int(predicted_label) + 1
        rows.append(
            {
                "outer_fold": int(test_set.participant),
                "test_participant": int(test_set.participant),
                "window_center_s": config.window_center,
                "window_start_s": train_window[0],
                "window_stop_s": train_window[1],
                "feature_mode": config.feature_mode,
                "normalization": config.normalization,
                "baseline_whitening_shrinkage": config.baseline_whitening_shrinkage,
                "alignment": config.alignment,
                "classifier": config.classifier,
                "components_pca": config.components_pca,
                "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
                "trial_selection": config.trial_selection,
                "training_sample_weighting": config.training_sample_weighting,
                "trial_selection_seed": _seed_field(config.trial_selection_seed),
                "actual_components_pca": actual_components_pca,
                "trial": int(trial_idx),
                "test_trial_index": int(trial_idx),
                "test_trial_number": int(trial_idx + 1),
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "true_stimulus": true_stimulus,
                "predicted_stimulus": predicted_stimulus,
                "correct": bool(predicted_label == true_label),
                "true_label_rank": float(true_label_rank) if np.isfinite(true_label_rank) else np.nan,
                "top2_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 2),
                "top3_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 3),
            }
        )
    return rows


def _selected_trial_indices(
    labels,
    max_trials_per_class,
    *,
    selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    participant=None,
):
    labels = np.asarray(labels).ravel()
    if max_trials_per_class is None:
        return np.arange(labels.shape[0], dtype=int)
    max_trials_per_class = int(max_trials_per_class)
    if max_trials_per_class <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")
    selection = _normalize_trial_selection(selection)

    if selection == "first":
        selected = []
        counts: Counter[int] = Counter()
        for index, label in enumerate(labels):
            if counts[int(label)] < max_trials_per_class:
                selected.append(index)
                counts[int(label)] += 1
        return np.asarray(selected, dtype=int)

    rng = _trial_selection_rng(seed, participant)
    selected = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size > max_trials_per_class:
            class_indices = rng.choice(class_indices, size=max_trials_per_class, replace=False)
        selected.extend(int(index) for index in class_indices)
    return np.asarray(sorted(selected), dtype=int)


def _trial_selection_rng(seed, participant):
    if seed is None:
        return np.random.default_rng()
    seed_values = [int(seed)]
    if participant is not None:
        seed_values.append(int(participant))
    return np.random.default_rng(np.random.SeedSequence(seed_values))


def _feature_set_trial_indices(feature_set):
    trial_indices = getattr(feature_set, "trial_indices", None)
    if trial_indices is None:
        return np.arange(np.asarray(feature_set.labels).shape[0], dtype=int)
    return np.asarray(trial_indices, dtype=int).ravel()


def _seed_field(seed):
    return "" if seed is None else int(seed)


def _normalized_config(config):
    return CrossSubjectStimulusConfig(
        window_center=config.window_center,
        window_size=config.window_size,
        baseline_window=config.baseline_window,
        feature_mode=_impl._normalize_feature_mode(config.feature_mode),
        normalization=_impl._normalize_normalization(config.normalization),
        alignment=_normalize_alignment(config.alignment),
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=_impl._normalize_trial_cap(config.max_trials_per_class_per_participant),
        trial_selection=_normalize_trial_selection(getattr(config, "trial_selection", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)),
        trial_selection_seed=_normalize_trial_selection_seed(getattr(config, "trial_selection_seed", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)),
        training_sample_weighting=_normalize_training_sample_weighting(getattr(config, "training_sample_weighting", DEFAULT_CROSS_SUBJECT_TRAINING_SAMPLE_WEIGHTING)),
        window_jitter_offsets=_normalize_window_jitter_offsets(getattr(config, "window_jitter_offsets", DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS)),
        baseline_whitening_shrinkage=_normalize_baseline_whitening_shrinkage(
            getattr(config, "baseline_whitening_shrinkage", BASELINE_WHITENING_SHRINKAGE)
        ),
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _normalize_alignment(value):
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "coral": TARGET_COVARIANCE_RECOLOR_ALIGNMENT,
        "covariance_recolor": TARGET_COVARIANCE_RECOLOR_ALIGNMENT,
        "covariance_recoloring": TARGET_COVARIANCE_RECOLOR_ALIGNMENT,
        "target_covariance": TARGET_COVARIANCE_RECOLOR_ALIGNMENT,
        "unsupervised_covariance": TARGET_COVARIANCE_RECOLOR_ALIGNMENT,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALIGNMENT_MODES:
        raise ValueError(f"alignment must be one of {ALIGNMENT_MODES}.")
    return normalized


def _normalize_trial_selection(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in TRIAL_SELECTION_MODES:
        raise ValueError(f"trial_selection must be one of {TRIAL_SELECTION_MODES}.")
    return normalized


def _normalize_training_sample_weighting(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in TRAINING_SAMPLE_WEIGHTING_MODES:
        raise ValueError(f"training_sample_weighting must be one of {TRAINING_SAMPLE_WEIGHTING_MODES}.")
    return normalized


def _normalize_trial_selection_seed(value):
    if value is None or value == "":
        return None
    value = int(value)
    if value < 0:
        raise ValueError("trial_selection_seed must be non-negative or None.")
    return value


def _normalize_baseline_whitening_shrinkage(value):
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            "baseline_whitening_shrinkage must be a finite value between 0 and 1."
        )
    return value


def _prediction_group_columns_with_training_sample_weighting(columns):
    output = list(columns)
    if "training_sample_weighting" not in output:
        try:
            output.insert(output.index("label_shuffle_control"), "training_sample_weighting")
        except ValueError:
            output.append("training_sample_weighting")
    return tuple(output)


def _prediction_group_columns_with_baseline_whitening_shrinkage(columns):
    output = list(columns)
    if "baseline_whitening_shrinkage" not in output:
        try:
            output.insert(output.index("alignment"), "baseline_whitening_shrinkage")
        except ValueError:
            output.append("baseline_whitening_shrinkage")
    return tuple(output)


def _prediction_group_columns_with_temporal_ensemble(columns):
    output = list(columns)
    for column in (
        "outer_evaluation_mode",
        "temporal_ensemble_mode",
        "temporal_ensemble_window_centers_s",
        "temporal_ensemble_size",
    ):
        if column not in output:
            output.append(column)
    return tuple(output)


def _install_module_fixes():
    _impl.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION  # type: ignore[attr-defined]
    _impl.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED  # type: ignore[attr-defined]
    _impl.DEFAULT_CROSS_SUBJECT_NESTED_ALIGNMENTS = DEFAULT_CROSS_SUBJECT_NESTED_ALIGNMENTS  # type: ignore[attr-defined]
    _impl.BASELINE_WHITENING_SHRINKAGE = BASELINE_WHITENING_SHRINKAGE  # type: ignore[attr-defined]
    _impl.TRIAL_SELECTION_MODES = TRIAL_SELECTION_MODES  # type: ignore[attr-defined]
    _impl.DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS = DEFAULT_CROSS_SUBJECT_WINDOW_JITTER_OFFSETS  # type: ignore[attr-defined]
    _impl.WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER = WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER  # type: ignore[attr-defined]
    _impl.NESTED_TOPK_WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER = NESTED_TOPK_WINDOW_JITTER_SCORE_ENSEMBLE_CLASSIFIER  # type: ignore[attr-defined]
    _impl.DEFAULT_CROSS_SUBJECT_TEMPORAL_ENSEMBLE_MODE = DEFAULT_CROSS_SUBJECT_TEMPORAL_ENSEMBLE_MODE  # type: ignore[attr-defined]
    _impl.TEMPORAL_ENSEMBLE_MODES = TEMPORAL_ENSEMBLE_MODES  # type: ignore[attr-defined]
    _impl.TEMPORAL_WINDOW_SCORE_ENSEMBLE_CLASSIFIER = TEMPORAL_WINDOW_SCORE_ENSEMBLE_CLASSIFIER  # type: ignore[attr-defined]
    _impl.TARGET_COVARIANCE_RECOLOR_ALIGNMENT = TARGET_COVARIANCE_RECOLOR_ALIGNMENT  # type: ignore[attr-defined]
    _impl.TARGET_CORAL_ALIGNMENT = TARGET_CORAL_ALIGNMENT  # type: ignore[attr-defined]
    _impl.ALIGNMENT_MODES = ALIGNMENT_MODES
    _impl.COVARIANCE_ALIGNMENT_SHRINKAGE = COVARIANCE_ALIGNMENT_SHRINKAGE  # type: ignore[attr-defined]
    _impl.CORAL_ALIGNMENT_SHRINKAGE = CORAL_ALIGNMENT_SHRINKAGE  # type: ignore[attr-defined]
    _impl.AUTO_CLASSIFIER_PARAM_GRID_TOKEN = AUTO_CLASSIFIER_PARAM_GRID_TOKEN  # type: ignore[attr-defined]
    _impl.AUTO_COMPONENTS_PCA_GRID_TOKEN = AUTO_COMPONENTS_PCA_GRID_TOKEN  # type: ignore[attr-defined]
    _impl.CLASSIFIER_AUTO_PARAM_GRIDS = CLASSIFIER_AUTO_PARAM_GRIDS  # type: ignore[attr-defined]
    _impl.COMPONENTS_PCA_AUTO_GRID = COMPONENTS_PCA_AUTO_GRID  # type: ignore[attr-defined]
    _impl.FEATURE_MODE_PRESETS = FEATURE_MODE_PRESETS  # type: ignore[attr-defined]
    _impl.DEFAULT_CROSS_SUBJECT_TRAINING_SAMPLE_WEIGHTING = DEFAULT_CROSS_SUBJECT_TRAINING_SAMPLE_WEIGHTING  # type: ignore[attr-defined]
    _impl.TRAINING_SAMPLE_WEIGHTING_MODES = TRAINING_SAMPLE_WEIGHTING_MODES  # type: ignore[attr-defined]
    _impl.CrossSubjectStimulusConfig = CrossSubjectStimulusConfig  # type: ignore[misc]
    _impl.ParticipantFeatureSet = ParticipantFeatureSet  # type: ignore[misc]
    _impl._fit_target_covariance_recolor_alignment = _fit_target_covariance_recolor_alignment  # type: ignore[attr-defined]
    _impl._feature_covariance_recolor_transform = _feature_covariance_recolor_transform  # type: ignore[attr-defined]
    _impl._apply_feature_space_affine_transform = _apply_feature_space_affine_transform  # type: ignore[attr-defined]
    _impl._feature_covariance = _feature_covariance  # type: ignore[attr-defined]
    _impl._covariance_square_root = _covariance_square_root  # type: ignore[attr-defined]
    _impl.make_cross_subject_candidate_configs = make_cross_subject_candidate_configs
    _impl._classifier_params_for_classifier = _classifier_params_for_classifier  # type: ignore[attr-defined]
    _impl._is_auto_classifier_param_grid = _is_auto_classifier_param_grid  # type: ignore[attr-defined]
    _impl._components_pca_values_for_grid = _components_pca_values_for_grid  # type: ignore[attr-defined]
    _impl._is_auto_components_pca_grid = _is_auto_components_pca_grid  # type: ignore[attr-defined]
    _impl._baseline_whitening_shrinkage_values_for_grid = _baseline_whitening_shrinkage_values_for_grid  # type: ignore[attr-defined]
    _impl._baseline_whitening_shrinkage_values_for_normalization = _baseline_whitening_shrinkage_values_for_normalization  # type: ignore[attr-defined]
    _impl._normalize_baseline_whitening_shrinkage = _normalize_baseline_whitening_shrinkage  # type: ignore[attr-defined]
    _impl.load_participant_stimulus_features = load_participant_stimulus_features
    _impl._expand_feature_modes = _expand_feature_modes  # type: ignore[attr-defined]
    _impl.summarize_cross_subject_stimulus_smoke = summarize_cross_subject_stimulus_smoke
    _impl.summarize_nested_cross_subject_stimulus = summarize_nested_cross_subject_stimulus
    _impl.evaluate_nested_cross_subject_stimulus = evaluate_nested_cross_subject_stimulus
    _impl.export_nested_cross_subject_stimulus = export_nested_cross_subject_stimulus
    _impl._ranked_label_metrics = _ranked_label_metrics
    _impl._align_training_features_by_subject = _align_training_features_by_subject
    _impl._load_feature_cache = _load_feature_cache
    _impl._evaluate_nested_inner_rows = _evaluate_nested_inner_rows
    _impl._evaluate_nested_outer_fold = _evaluate_nested_outer_fold
    _impl._select_nested_candidate_ensemble = _select_nested_candidate_ensemble
    _impl._align_test_features_by_subject = _align_test_features_by_subject  # type: ignore[attr-defined]
    _impl._fit_outer_fold_model = _fit_outer_fold_model
    _impl._score_outer_fold_model = _score_outer_fold_model
    _impl._candidate_model_scores = _candidate_model_scores
    _impl._fit_target_coral_model = _fit_target_coral_model  # type: ignore[attr-defined]
    _impl._apply_target_coral_model = _apply_target_coral_model  # type: ignore[attr-defined]
    _impl._coral_covariance = _coral_covariance  # type: ignore[attr-defined]
    _impl._symmetric_matrix_square_root = _symmetric_matrix_square_root  # type: ignore[attr-defined]
    _impl._feature_cache_key = _feature_cache_key
    _impl._configs_with_temporal_ensemble = _configs_with_temporal_ensemble  # type: ignore[attr-defined]
    _impl._temporal_window_configs = _temporal_window_configs  # type: ignore[attr-defined]
    _impl._jittered_window_configs = _jittered_window_configs  # type: ignore[attr-defined]
    _impl._normalize_window_jitter_offsets = _normalize_window_jitter_offsets  # type: ignore[attr-defined]
    _impl._normalize_temporal_ensemble_window_centers = _normalize_temporal_ensemble_window_centers  # type: ignore[attr-defined]
    _impl._normalize_temporal_ensemble_mode = _normalize_temporal_ensemble_mode  # type: ignore[attr-defined]
    _impl._prediction_rows = _prediction_rows
    _impl._selected_trial_indices = _selected_trial_indices
    _impl._feature_set_trial_indices = _feature_set_trial_indices  # type: ignore[attr-defined]
    _impl._seed_field = _seed_field  # type: ignore[attr-defined]
    _impl._normalized_config = _normalized_config
    _impl._normalize_alignment = _normalize_alignment
    _impl._normalize_trial_selection = _normalize_trial_selection  # type: ignore[attr-defined]
    _impl._normalize_training_sample_weighting = _normalize_training_sample_weighting  # type: ignore[attr-defined]
    _impl._training_sample_weights_by_subject = _training_sample_weights_by_subject  # type: ignore[attr-defined]
    _impl._normalize_trial_selection_seed = _normalize_trial_selection_seed  # type: ignore[attr-defined]
    install_cross_subject(_impl)
    _impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = _prediction_group_columns_with_temporal_ensemble(
        _prediction_group_columns_with_window_jitter(
            _prediction_group_columns_with_training_sample_weighting(
                _prediction_group_columns_with_trial_selection(
                    _prediction_group_columns_with_baseline_whitening_shrinkage(
                        _prediction_group_columns_with_alignment()
                    )
                )
            )
        )
    )


_install_module_fixes()

globals().update(_impl.__dict__)
