"""Cross-subject stimulus decoding smoke benchmarks."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import scipy.io as sio
from reptrace.decoding.alignment import class_pattern_procrustes_alignment
from reptrace.decoding.normalization import baseline_whitening_matrix as reptrace_baseline_whitening_matrix
from reptrace.decoding.normalization import nonzero_scale
from reptrace.decoding.normalization import normalize_features as reptrace_normalize_features
from reptrace.decoding.windowed import fit_window_model as fit_reptrace_window_model
from reptrace.decoding.windowed import predict_window_model as predict_reptrace_window_model
from reptrace.decoding.windowed import transform_window_features as transform_reptrace_window_features
from reptrace.metrics.classification import ranked_accuracy_metrics, subject_level_signflip_summary
from reptrace.metrics.confusion import category_confusion_enrichment as reptrace_category_confusion_enrichment
from reptrace.metrics.confusion import category_confusion_matrix as reptrace_category_confusion_matrix
from reptrace.metrics.confusion import confusion_counts, confusion_pair_summary, per_class_accuracy
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.alpha_signal import get_data_field
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder

DEFAULT_CROSS_SUBJECT_PARTICIPANTS = "1-4,6,8,9,10,13-27"
DEFAULT_CROSS_SUBJECT_WINDOW_CENTER = 0.175
DEFAULT_CROSS_SUBJECT_WINDOW_SIZE = 0.1
DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW = (-0.5, 0.0)
DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES = 16
DEFAULT_CROSS_SUBJECT_FEATURE_MODE = "sensor_mean"
DEFAULT_CROSS_SUBJECT_NORMALIZATION = "subject_baseline_z"
DEFAULT_CROSS_SUBJECT_ALIGNMENT = "none"
DEFAULT_CROSS_SUBJECT_CLASSIFIER = "multiclass-svm"
DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA = 64
DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS = (0.150, 0.175, 0.200)
DEFAULT_CROSS_SUBJECT_SELECTION_METRIC = "balanced_accuracy"
INNER_VALIDATION_SCHEMES = ("loso", "random-holdout")
FEATURE_MODES = ("sensor_mean", "sensor_flat")
NORMALIZATION_MODES = ("none", "subject_z", "subject_trial_z", "subject_baseline_z", "subject_baseline_whiten")
ALIGNMENT_MODES = ("none", "train_class_procrustes")
BASELINE_WHITENING_SHRINKAGE = 0.1
BASELINE_WHITENING_EIGENVALUE_FLOOR = 1e-6
CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = (
    "window_center_s",
    "feature_mode",
    "normalization",
    "alignment",
    "classifier",
    "components_pca",
    "max_trials_per_class_per_participant",
    "label_shuffle_control",
    "label_shuffle_seed",
)
@dataclass(frozen=True)
class CrossSubjectStimulusConfig:  # pylint: disable=too-many-instance-attributes
    """Parameters for the fixed-pipeline cross-subject stimulus smoke test."""

    window_center: float = DEFAULT_CROSS_SUBJECT_WINDOW_CENTER
    window_size: float = DEFAULT_CROSS_SUBJECT_WINDOW_SIZE
    baseline_window: tuple[float, float] = DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    feature_mode: str = DEFAULT_CROSS_SUBJECT_FEATURE_MODE
    normalization: str = DEFAULT_CROSS_SUBJECT_NORMALIZATION
    alignment: str = DEFAULT_CROSS_SUBJECT_ALIGNMENT
    classifier: str = DEFAULT_CROSS_SUBJECT_CLASSIFIER
    classifier_param: object = float("nan")
    components_pca: int | float = DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA
    max_trials_per_class_per_participant: int | None = None
    chance_classes: int = DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES
    random_state: int | None = 0
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0


@dataclass(frozen=True)
class ParticipantFeatureSet:
    """Windowed features for one participant."""

    participant: int
    labels: np.ndarray
    features: np.ndarray
    normalization: str
    baseline_features: np.ndarray | None
    baseline_feature_mean: np.ndarray | None
    baseline_feature_std: np.ndarray | None
    baseline_whitening_matrix: np.ndarray | None
    n_channels: int
    n_window_samples: int
    n_baseline_samples: int
    max_trials_per_class_per_participant: int | None


def evaluate_cross_subject_stimulus_smoke(data_folder, participants, *, config=None, progress=None):
    """Run fixed-pipeline leave-one-subject-out stimulus decoding on ``Part*Data.mat`` files only."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for a cross-subject smoke benchmark.")

    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    feature_sets = []
    for participant in participants:
        if progress is not None:
            progress(f"LOAD participant={participant}")
        feature_sets.append(load_participant_stimulus_features(data_folder, participant, config=config))

    outer_rows = []
    prediction_rows = []
    for test_participant in participants:
        if progress is not None:
            progress(f"START outer_test_participant={test_participant}")
        train_sets = [feature_set for feature_set in feature_sets if feature_set.participant != test_participant]
        test_set = next(feature_set for feature_set in feature_sets if feature_set.participant == test_participant)
        outer_row, participant_predictions = _evaluate_outer_fold(
            train_sets,
            test_set,
            config=config,
            classifier_param=classifier_param,
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(participant_predictions)
        if progress is not None:
            progress(f"DONE outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")

    group_summary_rows = summarize_cross_subject_stimulus_smoke(outer_rows, config=config)
    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    confusion_pair_rows = summarize_cross_subject_confusion_pairs(prediction_rows)
    return {
        "outer": outer_rows,
        "predictions": prediction_rows,
        "group_summary": group_summary_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": confusion_pair_rows,
    }


def evaluate_nested_cross_subject_stimulus(
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_participants=None,
    inner_validation_scheme="loso",
    inner_validation_seed=0,
    progress=None,
    existing_artifacts=None,
    after_outer_fold=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested model selection and evaluate each untouched outer participant once."""

    candidate_configs = _normalized_candidate_configs(candidate_configs)
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    inner_validation_scheme = _normalize_inner_validation_scheme(inner_validation_scheme)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for nested cross-subject decoding.")
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required.")
    outer_participants = _normalize_outer_participants(participants, outer_participants)

    resumed = _existing_nested_artifact_rows(existing_artifacts)
    inner_rows = resumed["inner_validation"]
    outer_rows = resumed["outer"]
    selected_rows = resumed["selected"]
    prediction_rows = resumed["predictions"]
    completed_outer_folds = {int(row["test_participant"]) for row in outer_rows}
    missing_participants = tuple(participant for participant in outer_participants if participant not in completed_outer_folds)
    feature_cache = _load_feature_cache(data_folder, participants, candidate_configs, progress=progress) if missing_participants else {}
    inner_pair_cache: dict[tuple[int, tuple[int, int]], dict] = {}
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
            inner_validation_scheme=inner_validation_scheme,
            inner_validation_seed=inner_validation_seed,
            progress=progress,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
        )
        inner_rows.extend(outer_inner_rows)
        outer_rows.append(outer_row)
        selected_rows.append(selected_row)
        prediction_rows.extend(participant_predictions)
        if after_outer_fold is not None:
            after_outer_fold(_assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs))

    return _assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs)


def make_cross_subject_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
    window_size=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    baseline_window=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    feature_modes=(DEFAULT_CROSS_SUBJECT_FEATURE_MODE,),
    normalizations=(DEFAULT_CROSS_SUBJECT_NORMALIZATION,),
    alignments=(DEFAULT_CROSS_SUBJECT_ALIGNMENT,),
    classifiers=(DEFAULT_CROSS_SUBJECT_CLASSIFIER,),
    classifier_params=(float("nan"),),
    components_pca_values=(DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,),
    max_trials_per_class_per_participant=None,
    chance_classes=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    random_state=0,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    """Build a candidate grid for nested cross-subject model selection."""

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
            chance_classes=chance_classes,
            random_state=random_state,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        for window_center, feature_mode, normalization, alignment, classifier, classifier_param, components_pca in product(
            window_centers,
            feature_modes,
            normalizations,
            alignments,
            classifiers,
            classifier_params,
            components_pca_values,
        )
    )


def load_participant_stimulus_features(data_folder, participant, *, config=None):
    """Load one participant's main ``Part*Data.mat`` file and extract fixed-window features."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_path = Path(resolve_data_folder(data_folder)) / f"Part{int(participant)}Data.mat"
    data = sio.loadmat(data_path)["data"][0]
    all_labels = _trialinfo_labels(data)
    trial_indices = _selected_trial_indices(all_labels, config.max_trials_per_class_per_participant)
    labels = all_labels[trial_indices]
    features, n_window_samples = _extract_window_features(
        data,
        _centered_window(config.window_center, config.window_size),
        feature_mode=config.feature_mode,
        trial_indices=trial_indices,
    )
    baseline_features = None
    baseline_feature_mean = None
    baseline_feature_std = None
    baseline_whitening_matrix = None
    n_baseline_samples = 0
    if config.normalization in ("subject_baseline_z", "subject_baseline_whiten"):
        baseline_feature_mean, baseline_feature_std, n_baseline_samples = _baseline_feature_statistics(data, config, n_window_samples, trial_indices)
    if config.normalization == "subject_baseline_whiten":
        baseline_whitening_matrix, n_baseline_samples = _baseline_channel_whitening_matrix(data, config.baseline_window, trial_indices)
    normalized_features = _normalize_features(features, config, baseline_feature_mean, baseline_feature_std, baseline_whitening_matrix)
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} labels but {features.shape[0]} feature rows.")
    return ParticipantFeatureSet(
        participant=int(participant),
        labels=labels,
        features=normalized_features,
        normalization=config.normalization,
        baseline_features=baseline_features,
        baseline_feature_mean=baseline_feature_mean,
        baseline_feature_std=baseline_feature_std,
        baseline_whitening_matrix=baseline_whitening_matrix,
        n_channels=int(_trial_signal(data, 0).shape[0]),
        n_window_samples=int(n_window_samples),
        n_baseline_samples=int(n_baseline_samples),
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
    )


def summarize_cross_subject_stimulus_smoke(outer_rows, *, config=None):
    """Summarize held-out participant scores with a one-sided subject-level sign-flip test."""

    if not outer_rows:
        return []

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    top2 = _finite_metric_values(outer_rows, "top2_accuracy")
    top3 = _finite_metric_values(outer_rows, "top3_accuracy")
    mean_ranks = _finite_metric_values(outer_rows, "mean_true_label_rank")
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    sign_summary = subject_level_signflip_summary(
        balanced,
        chance=chance,
        n_permutations=config.signflip_permutations,
        random_state=config.signflip_seed,
    )
    participants_above_chance = int(sign_summary["n_above_chance"])
    participants_total = int(sign_summary["n_subjects"])
    exact_sign_p_value = float(sign_summary["one_sided_exact_sign_p_value"])
    signflip_p_value = float(sign_summary["one_sided_signflip_p_value"])
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "window_center_s": config.window_center,
            "window_size_s": config.window_size,
            "window_start_s": _centered_window(config.window_center, config.window_size)[0],
            "window_stop_s": _centered_window(config.window_center, config.window_size)[1],
            "baseline_window_start_s": config.baseline_window[0],
            "baseline_window_stop_s": config.baseline_window[1],
            "feature_mode": config.feature_mode,
            "normalization": config.normalization,
            "alignment": config.alignment,
            "classifier": config.classifier,
            "components_pca": config.components_pca,
            "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(raw)),
            "accuracy_median": float(np.median(raw)),
            "accuracy_sem": _sem(raw),
            "percent_mean": float(100.0 * np.mean(raw)),
            "top2_accuracy_mean": _nanmean_or_nan(top2),
            "top2_percent_mean": _percent_nanmean_or_nan(top2),
            "top2_percent_sem": _percent_sem_or_nan(top2),
            "top2_chance_accuracy": min(2.0 / config.chance_classes, 1.0),
            "top2_chance_percent": min(200.0 / config.chance_classes, 100.0),
            "top3_accuracy_mean": _nanmean_or_nan(top3),
            "top3_percent_mean": _percent_nanmean_or_nan(top3),
            "top3_percent_sem": _percent_sem_or_nan(top3),
            "top3_chance_accuracy": min(3.0 / config.chance_classes, 1.0),
            "top3_chance_percent": min(300.0 / config.chance_classes, 100.0),
            "mean_true_label_rank_mean": _nanmean_or_nan(mean_ranks),
            "mean_true_label_rank_sem": _sem_or_nan(mean_ranks),
            "chance_mean_rank": 0.5 * (config.chance_classes + 1),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "mean_above_chance": float(np.mean(differences)),
            "percent_above_chance": float(100.0 * np.mean(differences)),
            "participants_above_chance": participants_above_chance,
            "participants_total": participants_total,
            "participants_at_or_below_chance": int(np.sum(balanced <= chance)),
            "one_sided_exact_sign_p_value": exact_sign_p_value,
            "one_sided_signflip_p_value": signflip_p_value,
        }
    ]


def summarize_nested_cross_subject_stimulus(outer_rows, *, signflip_permutations=10_000, signflip_seed=0):
    """Summarize nested cross-subject held-out scores without assuming one fixed configuration."""

    if not outer_rows:
        return []

    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    top2 = _finite_metric_values(outer_rows, "top2_accuracy")
    top3 = _finite_metric_values(outer_rows, "top3_accuracy")
    mean_ranks = _finite_metric_values(outer_rows, "mean_true_label_rank")
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    sign_summary = subject_level_signflip_summary(
        balanced,
        chance=chance,
        n_permutations=signflip_permutations,
        random_state=signflip_seed,
    )
    participants_above_chance = int(sign_summary["n_above_chance"])
    participants_total = int(sign_summary["n_subjects"])
    exact_sign_p_value = float(sign_summary["one_sided_exact_sign_p_value"])
    signflip_p_value = float(sign_summary["one_sided_signflip_p_value"])
    selected_counts = Counter(int(row["selected_candidate_index"]) for row in outer_rows)
    classifier_counts = _row_value_counts(outer_rows, "selected_classifier", fallback_key="classifier")
    window_counts = _row_value_counts(outer_rows, "selected_window_center_s", fallback_key="window_center_s", transform=float)
    feature_mode_counts = _row_value_counts(outer_rows, "selected_feature_mode", fallback_key="feature_mode")
    normalization_counts = _row_value_counts(outer_rows, "selected_normalization", fallback_key="normalization")
    alignment_counts = _row_value_counts(outer_rows, "selected_alignment", fallback_key="alignment")
    components_pca_counts = _row_value_counts(outer_rows, "selected_components_pca", fallback_key="components_pca")
    trial_cap_counts = Counter(str(row["max_trials_per_class_per_participant"]) for row in outer_rows)
    winner_margins = _finite_metric_values(outer_rows, "selected_inner_winner_margin")
    label_shuffle_control = _single_row_value(outer_rows, "label_shuffle_control", default=False)
    label_shuffle_seed = _single_row_value(outer_rows, "label_shuffle_seed", default="")
    selection_mode = _single_row_value(outer_rows, "selection_mode", default="nested_loso")
    inner_validation_scheme = _single_row_value(outer_rows, "inner_validation_scheme", default="loso")
    inner_validation_seed = _single_row_value(outer_rows, "inner_validation_seed", default="")
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "selection_mode": selection_mode,
            "inner_validation_scheme": inner_validation_scheme,
            "inner_validation_seed": inner_validation_seed,
            "selection_metric": DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
            "label_shuffle_control": label_shuffle_control,
            "label_shuffle_seed": label_shuffle_seed,
            "n_candidates": int(max(int(row["n_candidates"]) for row in outer_rows)),
            "selected_candidate_counts": _format_counter(selected_counts),
            "selected_classifier_counts": _format_counter(classifier_counts),
            "selected_window_center_counts": _format_counter(window_counts),
            "selected_feature_mode_counts": _format_counter(feature_mode_counts),
            "selected_normalization_counts": _format_counter(normalization_counts),
            "selected_alignment_counts": _format_counter(alignment_counts),
            "selected_components_pca_counts": _format_counter(components_pca_counts),
            "max_trials_per_class_per_participant_counts": _format_counter(trial_cap_counts),
            "inner_winner_margin_mean": _nanmean_or_nan(winner_margins),
            "inner_winner_margin_median": _nanmedian_or_nan(winner_margins),
            "inner_winner_margin_min": _nanmin_or_nan(winner_margins),
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(raw)),
            "accuracy_median": float(np.median(raw)),
            "accuracy_sem": _sem(raw),
            "percent_mean": float(100.0 * np.mean(raw)),
            "top2_accuracy_mean": _nanmean_or_nan(top2),
            "top2_percent_mean": _percent_nanmean_or_nan(top2),
            "top2_percent_sem": _percent_sem_or_nan(top2),
            "top2_chance_accuracy": min(2.0 * chance, 1.0),
            "top2_chance_percent": min(200.0 * chance, 100.0),
            "top3_accuracy_mean": _nanmean_or_nan(top3),
            "top3_percent_mean": _percent_nanmean_or_nan(top3),
            "top3_percent_sem": _percent_sem_or_nan(top3),
            "top3_chance_accuracy": min(3.0 * chance, 1.0),
            "top3_chance_percent": min(300.0 * chance, 100.0),
            "mean_true_label_rank_mean": _nanmean_or_nan(mean_ranks),
            "mean_true_label_rank_sem": _sem_or_nan(mean_ranks),
            "chance_mean_rank": 0.5 * ((1.0 / chance) + 1.0),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "mean_above_chance": float(np.mean(differences)),
            "percent_above_chance": float(100.0 * np.mean(differences)),
            "participants_above_chance": participants_above_chance,
            "participants_total": participants_total,
            "participants_at_or_below_chance": int(np.sum(balanced <= chance)),
            "one_sided_exact_sign_p_value": exact_sign_p_value,
            "one_sided_signflip_p_value": signflip_p_value,
        }
    ]


def summarize_cross_subject_predictions(prediction_rows):
    """Return confusion-count and per-stimulus recall summaries for cross-subject predictions."""

    if not prediction_rows:
        return [], []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    group_columns = _present_group_columns(frame)
    confusion = confusion_counts(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=group_columns,
    )
    per_stimulus = per_class_accuracy(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        participant_column="test_participant",
        group_columns=group_columns,
    )
    return confusion.to_dict(orient="records"), per_stimulus.to_dict(orient="records")


def summarize_cross_subject_confusion_pairs(prediction_rows, *, stimulus_metadata_rows=None):
    """Summarize off-diagonal errors as unordered, bidirectional stimulus pairs."""

    if not prediction_rows:
        return []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    required = {"true_stimulus", "predicted_stimulus"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {sorted(missing)}")

    group_columns = _present_group_columns(frame)
    pairs = confusion_pair_summary(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        participant_column="test_participant" if "test_participant" in frame.columns else None,
        group_columns=group_columns,
        metadata=stimulus_metadata_rows,
        category_columns=None,
    )
    return _stimulus_pair_records(pairs)


def summarize_cross_subject_confusion_category_enrichment(
    prediction_rows,
    *,
    stimulus_metadata_rows,
    category_columns=None,
    n_permutations=10_000,
    seed=0,
):
    """Test whether off-diagonal errors stay within stimulus metadata categories."""

    if not prediction_rows:
        return []
    if not _has_metadata_rows(stimulus_metadata_rows):
        return []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    required = {"true_stimulus", "predicted_stimulus"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {sorted(missing)}")

    group_columns = _present_group_columns(frame)
    enrichment = reptrace_category_confusion_enrichment(
        frame,
        metadata=stimulus_metadata_rows,
        category_columns=category_columns,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        participant_column="test_participant" if "test_participant" in frame.columns else None,
        group_columns=group_columns,
        n_permutations=n_permutations,
        random_state=seed,
    )
    return enrichment.to_dict(orient="records")


def summarize_cross_subject_confusion_category_matrix(
    prediction_rows,
    *,
    stimulus_metadata_rows,
    category_columns=None,
):
    """Summarize directional category-to-category error counts and lifts."""

    if not prediction_rows:
        return []
    if not _has_metadata_rows(stimulus_metadata_rows):
        return []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    required = {"true_stimulus", "predicted_stimulus"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {sorted(missing)}")

    group_columns = _present_group_columns(frame)
    matrix = reptrace_category_confusion_matrix(
        frame,
        metadata=stimulus_metadata_rows,
        category_columns=category_columns,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        participant_column="test_participant" if "test_participant" in frame.columns else None,
        group_columns=group_columns,
    )
    return matrix.to_dict(orient="records")


def _stimulus_pair_records(pair_frame):
    if pair_frame.empty:
        return []
    renamed = pair_frame.rename(columns={"label_a": "stimulus_a", "label_b": "stimulus_b"}).copy()
    metadata_columns = {
        column: column.replace("label_a_", "stimulus_a_").replace("label_b_", "stimulus_b_")
        for column in renamed.columns
        if column.startswith("label_a_") or column.startswith("label_b_")
    }
    if metadata_columns:
        renamed = renamed.rename(columns=metadata_columns)
    return renamed.to_dict(orient="records")


def _has_metadata_rows(stimulus_metadata_rows):
    if stimulus_metadata_rows is None:
        return False
    if hasattr(stimulus_metadata_rows, "empty"):
        return not bool(stimulus_metadata_rows.empty)
    return bool(stimulus_metadata_rows)


def _assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs):
    group_summary_rows = summarize_nested_cross_subject_stimulus(
        outer_rows,
        signflip_permutations=candidate_configs[0].signflip_permutations,
        signflip_seed=candidate_configs[0].signflip_seed,
    )
    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    confusion_pair_rows = summarize_cross_subject_confusion_pairs(prediction_rows)
    return {
        "outer": outer_rows,
        "inner_validation": inner_rows,
        "selected": selected_rows,
        "predictions": prediction_rows,
        "group_summary": group_summary_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": confusion_pair_rows,
    }


def _existing_nested_artifact_rows(existing_artifacts):
    empty_artifacts: dict[str, list] = {
        "outer": [],
        "inner_validation": [],
        "selected": [],
        "predictions": [],
    }
    if existing_artifacts is None:
        return empty_artifacts
    return {key: list(existing_artifacts.get(key, [])) for key in empty_artifacts}


def _normalize_outer_participants(participants, outer_participants):
    if outer_participants is None:
        return tuple(participants)
    outer_participants = tuple(int(participant) for participant in outer_participants)
    if not outer_participants:
        raise ValueError("At least one outer participant is required.")
    unknown = sorted(set(outer_participants) - set(participants))
    if unknown:
        raise ValueError(f"Outer participants must be part of participants: {unknown}")
    return outer_participants


def _normalize_inner_validation_scheme(scheme):
    scheme = str(scheme or "loso").strip().lower().replace("_", "-")
    aliases = {
        "leave-one-subject-out": "loso",
        "leave-one-out": "loso",
        "inner-loso": "loso",
        "random": "random-holdout",
        "random-hold-out": "random-holdout",
        "single-random-holdout": "random-holdout",
    }
    scheme = aliases.get(scheme, scheme)
    if scheme not in INNER_VALIDATION_SCHEMES:
        raise ValueError(f"inner_validation_scheme must be one of {INNER_VALIDATION_SCHEMES}.")
    return scheme


def _selection_mode_for_inner_scheme(scheme):
    scheme = _normalize_inner_validation_scheme(scheme)
    return "nested_loso" if scheme == "loso" else "nested_random_holdout"


def _inner_validation_participants(test_participant, outer_train_participants, *, scheme, seed):
    scheme = _normalize_inner_validation_scheme(scheme)
    outer_train_participants = tuple(int(participant) for participant in outer_train_participants)
    if not outer_train_participants:
        raise ValueError("At least one outer-training participant is required for inner validation.")
    if scheme == "loso":
        return outer_train_participants
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(test_participant)]))
    return (outer_train_participants[int(rng.integers(0, len(outer_train_participants)))],)


def _read_csv_rows(path):
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_nested_output_rows(
    *,
    outer_output_path,
    inner_validation_output_path=None,
    selected_output_path=None,
    predictions_output_path=None,
):
    return {
        "outer": _read_csv_rows(outer_output_path),
        "inner_validation": _read_csv_rows(inner_validation_output_path),
        "selected": _read_csv_rows(selected_output_path),
        "predictions": _read_csv_rows(predictions_output_path),
    }


def _write_nested_output_rows(
    artifacts,
    *,
    outer_output_path,
    group_summary_output_path=None,
    inner_validation_output_path=None,
    selected_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
):
    _write_rows_if_present(artifacts["outer"], outer_output_path)
    _write_rows_if_present(artifacts["group_summary"], group_summary_output_path)
    _write_rows_if_present(artifacts["inner_validation"], inner_validation_output_path)
    _write_rows_if_present(artifacts["selected"], selected_output_path)
    _write_rows_if_present(artifacts["predictions"], predictions_output_path)
    _write_rows_if_present(artifacts["confusion"], confusion_output_path)
    _write_rows_if_present(artifacts["per_stimulus"], per_stimulus_output_path)
    _write_rows_if_present(artifacts["confusion_pairs"], confusion_pairs_output_path)


def _write_rows_if_present(rows, path):
    if path and rows:
        write_alpha_metrics_csv(_rows_with_consistent_fields(rows), path)


def _rows_with_consistent_fields(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def _present_group_columns(frame):
    return tuple(column for column in CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS if column in frame.columns)


def export_cross_subject_stimulus_smoke(  # pylint: disable=too-many-arguments
    data_folder,
    participants,
    *,
    outer_output_path,
    group_summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    config=None,
    progress=None,
):
    """Run the cross-subject smoke benchmark and write compact CSV artifacts."""

    artifacts = evaluate_cross_subject_stimulus_smoke(data_folder, participants, config=config, progress=progress)
    write_alpha_metrics_csv(artifacts["outer"], outer_output_path)
    if group_summary_output_path:
        write_alpha_metrics_csv(artifacts["group_summary"], group_summary_output_path)
    if predictions_output_path:
        write_alpha_metrics_csv(artifacts["predictions"], predictions_output_path)
    if confusion_output_path:
        write_alpha_metrics_csv(artifacts["confusion"], confusion_output_path)
    if per_stimulus_output_path:
        write_alpha_metrics_csv(artifacts["per_stimulus"], per_stimulus_output_path)
    if confusion_pairs_output_path and artifacts["confusion_pairs"]:
        write_alpha_metrics_csv(artifacts["confusion_pairs"], confusion_pairs_output_path)
    return artifacts


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
    inner_validation_scheme="loso",
    inner_validation_seed=0,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested cross-subject decoding and write compact CSV artifacts."""

    existing_artifacts = (
        _read_nested_output_rows(
            outer_output_path=outer_output_path,
            inner_validation_output_path=inner_validation_output_path,
            selected_output_path=selected_output_path,
            predictions_output_path=predictions_output_path,
        )
        if resume
        else None
    )

    def write_outputs(current_artifacts):
        _write_nested_output_rows(
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
        inner_validation_scheme=inner_validation_scheme,
        inner_validation_seed=inner_validation_seed,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    write_outputs(artifacts)
    return artifacts


def _load_feature_cache(data_folder, participants, candidate_configs, *, progress=None):
    representative_configs: dict[tuple[float, float, float, float, str, str, int | None], CrossSubjectStimulusConfig] = {}
    for candidate_config in candidate_configs:
        representative_configs.setdefault(_feature_cache_key(candidate_config), candidate_config)

    feature_cache = {}
    for key, candidate_config in representative_configs.items():
        if progress is not None:
            progress(
                "LOAD feature_set "
                f"window_center={candidate_config.window_center} "
                f"feature_mode={candidate_config.feature_mode} "
                f"normalization={candidate_config.normalization} "
                f"alignment={candidate_config.alignment}"
            )
        feature_cache[key] = {participant: load_participant_stimulus_features(data_folder, participant, config=candidate_config) for participant in participants}
    return feature_cache


def _evaluate_nested_outer_fold(
    test_participant,
    participants,
    candidate_configs,
    feature_cache,
    inner_pair_cache,
    *,
    inner_validation_scheme="loso",
    inner_validation_seed=0,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    if progress is not None:
        progress(f"START outer_test_participant={test_participant}")
    outer_train_participants = tuple(participant for participant in participants if participant != test_participant)
    outer_inner_rows = _evaluate_nested_inner_rows(
        test_participant,
        outer_train_participants,
        candidate_configs,
        feature_cache,
        inner_pair_cache,
        inner_validation_scheme=inner_validation_scheme,
        inner_validation_seed=inner_validation_seed,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    selected_row = _select_nested_candidate(outer_inner_rows)
    selected_config = candidate_configs[int(selected_row["selected_candidate_index"]) - 1]
    selected_feature_sets = feature_cache[_feature_cache_key(selected_config)]
    train_sets = [selected_feature_sets[participant] for participant in outer_train_participants]
    test_set = selected_feature_sets[test_participant]
    outer_row, participant_predictions = _evaluate_outer_fold(
        train_sets,
        test_set,
        config=selected_config,
        classifier_param=_resolved_classifier_param(selected_config),
        include_predictions=True,
        label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
        label_shuffle_context=(int(test_participant), int(selected_row["selected_candidate_index"]), 0),
    )
    _add_selected_candidate_fields(outer_row, selected_row)
    for prediction_row in participant_predictions:
        _add_selected_candidate_fields(prediction_row, selected_row)
    if progress is not None:
        progress(
            "DONE outer_test_participant="
            f"{test_participant} selected_candidate={selected_row['selected_candidate_index']} "
            f"inner_mean={selected_row['selected_inner_balanced_accuracy_mean']:.4f} "
            f"outer_balanced_accuracy={outer_row['balanced_accuracy']:.4f}"
        )
    return outer_row, outer_inner_rows, selected_row, participant_predictions


def _evaluate_nested_inner_rows(
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    inner_pair_cache,
    *,
    inner_validation_scheme="loso",
    inner_validation_seed=0,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    inner_validation_scheme = _normalize_inner_validation_scheme(inner_validation_scheme)
    validation_participants = _inner_validation_participants(
        test_participant,
        outer_train_participants,
        scheme=inner_validation_scheme,
        seed=inner_validation_seed,
    )
    inner_rows = []
    completed = 0
    total = len(candidate_configs) * len(validation_participants)
    for candidate_index, candidate_config in enumerate(candidate_configs, start=1):
        feature_sets = feature_cache[_feature_cache_key(candidate_config)]
        for validation_participant in validation_participants:
            excluded_pair = tuple(sorted((int(test_participant), int(validation_participant))))
            pair_rows = _cached_nested_pair_rows(
                candidate_index,
                candidate_config,
                excluded_pair,
                feature_sets,
                inner_pair_cache,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
            )
            inner_rows.append(
                pair_rows[(int(test_participant), int(validation_participant))]
                | {
                    "selection_mode": _selection_mode_for_inner_scheme(inner_validation_scheme),
                    "inner_validation_scheme": inner_validation_scheme,
                    "inner_validation_seed": int(inner_validation_seed),
                }
            )
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


def _cached_nested_pair_rows(
    candidate_index,
    candidate_config,
    excluded_pair,
    feature_sets,
    inner_pair_cache,
    *,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    cache_key = (int(candidate_index), tuple(excluded_pair), bool(label_shuffle_control), int(label_shuffle_seed))
    if cache_key not in inner_pair_cache:
        train_sets = [feature_set for participant, feature_set in feature_sets.items() if int(participant) not in excluded_pair]
        fitted_model = _fit_outer_fold_model(
            train_sets,
            candidate_config,
            _resolved_classifier_param(candidate_config),
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(int(candidate_index), *tuple(int(participant) for participant in excluded_pair)),
        )
        first_participant, second_participant = excluded_pair
        pair_rows = {}
        for outer_test_participant, validation_participant in (
            (first_participant, second_participant),
            (second_participant, first_participant),
        ):
            inner_row, _predictions = _score_outer_fold_model(
                fitted_model,
                feature_sets[validation_participant],
                candidate_config,
                include_predictions=False,
            )
            pair_rows[(outer_test_participant, validation_participant)] = _nested_inner_row(
                inner_row,
                outer_test_participant,
                validation_participant,
                candidate_index,
            )
        inner_pair_cache[cache_key] = pair_rows
    return inner_pair_cache[cache_key]


def _training_labels(feature_set, *, label_shuffle_seed=None, label_shuffle_context=()):
    labels = np.asarray(feature_set.labels, dtype=int)
    if label_shuffle_seed is None:
        return labels
    seed_values = [int(label_shuffle_seed), *[int(value) for value in label_shuffle_context], int(feature_set.participant)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(labels)


def _align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config):
    if config.alignment == "none":
        return features_by_subject, _alignment_metadata(config.alignment, common_classes=(), aligned_participants=())
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    common_classes = _common_label_values(labels_by_subject)
    if len(common_classes) < 2:
        return features_by_subject, _alignment_metadata(config.alignment, common_classes=common_classes, aligned_participants=())

    alignment = class_pattern_procrustes_alignment(
        features_by_subject,
        labels_by_subject,
        n_channels=int(feature_sets[0].n_channels),
        common_classes=common_classes,
    )
    return list(alignment.aligned_features), _alignment_metadata(
        config.alignment,
        common_classes=alignment.common_classes,
        aligned_participants=(feature_set.participant for feature_set in feature_sets),
    )


def _alignment_metadata(alignment, *, common_classes, aligned_participants):
    return {
        "alignment": alignment,
        "common_classes": ",".join(str(int(label)) for label in common_classes),
        "aligned_participants": ",".join(str(int(participant)) for participant in aligned_participants),
    }


def _common_label_values(labels_by_subject):
    label_sets = [set(np.asarray(labels, dtype=int).tolist()) for labels in labels_by_subject]
    if not label_sets:
        return tuple()
    return tuple(sorted(set.intersection(*label_sets)))


def _feature_cache_key(config):
    return (
        float(config.window_center),
        float(config.window_size),
        float(config.baseline_window[0]),
        float(config.baseline_window[1]),
        str(config.feature_mode),
        str(config.normalization),
        config.max_trials_per_class_per_participant,
    )


def _resolved_classifier_param(config):
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)
    return classifier_param


def _nested_inner_row(row, outer_test_participant, validation_participant, candidate_index):
    inner_row = dict(row)
    inner_row.update(
        {
            "selection_mode": "nested_loso",
            "selection_metric": DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
            "outer_test_participant": int(outer_test_participant),
            "inner_fold": int(validation_participant),
            "inner_validation_participant": int(validation_participant),
            "inner_train_participants": row["train_participants"],
            "n_inner_train_participants": row["n_train_participants"],
            "candidate_index": int(candidate_index),
        }
    )
    return inner_row


def _select_nested_candidate(inner_rows):
    if not inner_rows:
        raise ValueError("At least one inner-validation row is required for nested selection.")

    summaries = []
    candidate_indices = sorted({int(row["candidate_index"]) for row in inner_rows})
    for candidate_index in candidate_indices:
        candidate_rows = [row for row in inner_rows if int(row["candidate_index"]) == candidate_index]
        balanced = np.asarray([float(row["balanced_accuracy"]) for row in candidate_rows], dtype=float)
        raw = np.asarray([float(row["accuracy"]) for row in candidate_rows], dtype=float)
        example = candidate_rows[0]
        summaries.append(
            {
                "selection_mode": example.get("selection_mode", "nested_loso"),
                "inner_validation_scheme": example.get("inner_validation_scheme", "loso"),
                "inner_validation_seed": example.get("inner_validation_seed", ""),
                "selection_metric": DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
                "outer_fold": int(example["outer_test_participant"]),
                "test_participant": int(example["outer_test_participant"]),
                "selected_candidate_index": int(candidate_index),
                "n_candidates": len(candidate_indices),
                "n_inner_folds": len(candidate_rows),
                "selected_inner_balanced_accuracy_mean": float(np.mean(balanced)),
                "selected_inner_balanced_accuracy_median": float(np.median(balanced)),
                "selected_inner_balanced_accuracy_sem": _sem(balanced),
                "selected_inner_accuracy_mean": float(np.mean(raw)),
                "selected_inner_accuracy_median": float(np.median(raw)),
                "selected_inner_accuracy_sem": _sem(raw),
                "selected_window_center_s": example["window_center_s"],
                "selected_window_size_s": example["window_size_s"],
                "selected_window_start_s": example["window_start_s"],
                "selected_window_stop_s": example["window_stop_s"],
                "selected_feature_mode": example["feature_mode"],
                "selected_normalization": example["normalization"],
                "selected_alignment": example["alignment"],
                "selected_classifier": example["classifier"],
                "selected_classifier_param": example["classifier_param"],
                "selected_components_pca": example["components_pca"],
                "selected_max_trials_per_class_per_participant": example["max_trials_per_class_per_participant"],
                "label_shuffle_control": example.get("label_shuffle_control", False),
                "label_shuffle_seed": example.get("label_shuffle_seed", ""),
            }
        )
    ranked = sorted(
        summaries,
        key=lambda row: (
            float(row["selected_inner_balanced_accuracy_mean"]),
            float(row["selected_inner_balanced_accuracy_median"]),
            -int(row["selected_candidate_index"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    selected_mean = float(selected["selected_inner_balanced_accuracy_mean"])
    if len(ranked) > 1:
        second_best_mean = float(ranked[1]["selected_inner_balanced_accuracy_mean"])
        winner_margin = selected_mean - second_best_mean
    else:
        second_best_mean = np.nan
        winner_margin = np.nan
    selected["selected_inner_second_best_balanced_accuracy_mean"] = second_best_mean
    selected["selected_inner_winner_margin"] = winner_margin
    return selected


def _add_selected_candidate_fields(row, selected_row):
    for key, value in selected_row.items():
        row[key] = value


def _evaluate_outer_fold(
    train_sets,
    test_set,
    *,
    config,
    classifier_param,
    include_predictions=True,
    label_shuffle_seed=None,
    label_shuffle_context=(),
):
    fitted_model = _fit_outer_fold_model(
        train_sets,
        config,
        classifier_param,
        label_shuffle_seed=label_shuffle_seed,
        label_shuffle_context=label_shuffle_context,
    )
    return _score_outer_fold_model(fitted_model, test_set, config, include_predictions=include_predictions)


def _fit_outer_fold_model(train_sets, config, classifier_param, *, label_shuffle_seed=None, label_shuffle_context=()):
    train_features_by_subject = [_normalized_subject_features(feature_set, config) for feature_set in train_sets]
    train_label_arrays = [
        _training_labels(
            feature_set,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=label_shuffle_context,
        )
        for feature_set in train_sets
    ]
    train_features_by_subject, alignment_metadata = _align_training_features_by_subject(
        train_sets,
        train_features_by_subject,
        train_label_arrays,
        config,
    )
    train_features = np.vstack(train_features_by_subject)
    train_labels_one_based = np.concatenate(train_label_arrays)
    train_labels = train_labels_one_based - 1

    train_window = _centered_window(config.window_center, config.window_size)
    model_bundle = fit_reptrace_window_model(
        train_features,
        train_labels,
        fit_model=lambda features, labels: train_multiclass_classifier(
            features,
            labels,
            config.classifier,
            classifier_param,
            random_state=config.random_state,
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
    }


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    model_bundle = fitted_model["model_bundle"]
    test_features = _normalized_subject_features(test_set, config)
    test_labels_one_based = test_set.labels
    test_labels = test_labels_one_based - 1
    predictions, _scores = predict_reptrace_window_model(model_bundle, test_features)
    class_scores, score_classes = _model_class_scores(model_bundle, test_features)
    rank_metrics = ranked_accuracy_metrics(test_labels, class_scores, score_classes, top_ks=(2, 3))
    accuracy = float(accuracy_score(test_labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(test_labels, predictions))
    chance_accuracy = 1.0 / config.chance_classes
    train_class_counts = fitted_model["train_class_counts"]
    test_class_counts = Counter(test_labels_one_based.tolist())
    train_participants = fitted_model["train_participants"]
    train_labels = fitted_model["train_labels"]
    train_window = fitted_model["train_window"]
    alignment_metadata = fitted_model["alignment_metadata"]

    outer_row = {
        "outer_fold": int(test_set.participant),
        "test_participant": int(test_set.participant),
        "train_participants": ",".join(str(participant) for participant in train_participants),
        "n_train_participants": fitted_model["n_train_participants"],
        "n_test_participants": 1,
        "window_center_s": config.window_center,
        "window_size_s": config.window_size,
        "window_start_s": train_window[0],
        "window_stop_s": train_window[1],
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": config.feature_mode,
        "normalization": config.normalization,
        "alignment": config.alignment,
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_percent": 100.0 * balanced_accuracy,
        "top2_accuracy": rank_metrics["top2_accuracy"],
        "top2_percent": 100.0 * rank_metrics["top2_accuracy"],
        "top3_accuracy": rank_metrics["top3_accuracy"],
        "top3_percent": 100.0 * rank_metrics["top3_accuracy"],
        "mean_true_label_rank": rank_metrics["mean_true_label_rank"],
        "median_true_label_rank": rank_metrics["median_true_label_rank"],
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "top2_chance_accuracy": min(2.0 * chance_accuracy, 1.0),
        "top2_chance_percent": min(200.0 * chance_accuracy, 100.0),
        "top3_chance_accuracy": min(3.0 * chance_accuracy, 1.0),
        "top3_chance_percent": min(300.0 * chance_accuracy, 100.0),
        "chance_mean_rank": 0.5 * (config.chance_classes + 1),
        "above_chance": bool(balanced_accuracy > chance_accuracy),
        "n_train_trials": int(train_labels.shape[0]),
        "n_test_trials": int(test_labels.shape[0]),
        "n_train_classes": int(len(train_class_counts)),
        "n_test_classes": int(len(test_class_counts)),
        "min_train_trials_per_class": int(min(train_class_counts.values())),
        "min_test_trials_per_class": int(min(test_class_counts.values())),
        "classifier": config.classifier,
        "classifier_param": fitted_model["classifier_param"],
        "components_pca": config.components_pca,
        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
        "actual_components_pca": model_bundle.actual_components_pca,
        "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        "n_channels": test_set.n_channels,
        "n_window_samples": test_set.n_window_samples,
        "n_baseline_samples": test_set.n_baseline_samples,
        "label_shuffle_control": bool(fitted_model["label_shuffle_control"]),
        "label_shuffle_seed": fitted_model["label_shuffle_seed"],
        "alignment_common_classes": alignment_metadata["common_classes"],
        "alignment_aligned_participants": alignment_metadata["aligned_participants"],
    }
    prediction_rows = []
    if include_predictions:
        prediction_rows = _prediction_rows(
            test_set,
            test_labels,
            predictions,
            rank_metrics["true_label_ranks"],
            config=config,
            actual_components_pca=model_bundle.actual_components_pca,
        )
    return outer_row, prediction_rows


def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):
    train_window = _centered_window(config.window_center, config.window_size)
    rows = []
    for trial_idx, (true_label, predicted_label, true_label_rank) in enumerate(zip(test_labels, predictions, true_label_ranks)):
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
                "alignment": config.alignment,
                "classifier": config.classifier,
                "components_pca": config.components_pca,
                "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
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


def _model_class_scores(model_bundle, features):
    transformed_features = transform_reptrace_window_features(model_bundle, features)
    model = model_bundle.model
    classes = np.asarray(getattr(model, "classes_", np.arange(len(np.unique(model_bundle.train_labels)))))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed_features), dtype=float)
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(transformed_features), dtype=float)
    else:
        return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=int)

    if scores.ndim == 1:
        if classes.size != 2:
            return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=int)
        scores = np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[1] != classes.size:
        return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=int)
    return scores, classes


def _extract_window_features(data, time_window, *, feature_mode, trial_indices=None):
    feature_mode = _normalize_feature_mode(feature_mode)
    time_vector = _time_vector(data, 0)
    mask = _time_mask(time_vector, time_window)
    features = []
    for trial_idx in _iter_trial_indices(data, trial_indices):
        signal = _trial_signal(data, trial_idx)
        window_signal = signal[:, mask]
        if feature_mode == "sensor_mean":
            feature = np.mean(window_signal, axis=1)
        elif feature_mode == "sensor_flat":
            feature = window_signal.reshape(-1, order="F")
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        features.append(feature)
    return np.vstack(features), int(np.sum(mask))


def _baseline_feature_statistics(data, config, n_window_samples, trial_indices):
    if config.feature_mode == "sensor_mean":
        baseline_features, n_baseline_samples = _extract_window_features(data, config.baseline_window, feature_mode="sensor_mean", trial_indices=trial_indices)
        mean = np.mean(baseline_features, axis=0, keepdims=True)
        std = np.std(baseline_features, axis=0, keepdims=True)
        return mean, nonzero_scale(std), n_baseline_samples

    if config.feature_mode == "sensor_flat":
        channel_mean, channel_std, n_baseline_samples = _baseline_channel_statistics(data, config.baseline_window, trial_indices)
        mean = np.tile(channel_mean, int(n_window_samples))[None, :]
        std = np.tile(channel_std, int(n_window_samples))[None, :]
        return mean, nonzero_scale(std), n_baseline_samples

    raise ValueError(f"Unsupported feature_mode: {config.feature_mode}")


def _baseline_channel_statistics(data, baseline_window, trial_indices):
    time_vector = _time_vector(data, 0)
    mask = _time_mask(time_vector, baseline_window)
    n_channels = int(_trial_signal(data, 0).shape[0])
    sum_values = np.zeros(n_channels, dtype=float)
    sum_squares = np.zeros(n_channels, dtype=float)
    n_values = 0
    for trial_idx in _iter_trial_indices(data, trial_indices):
        baseline_signal = _trial_signal(data, trial_idx)[:, mask]
        sum_values += np.sum(baseline_signal, axis=1)
        sum_squares += np.sum(np.square(baseline_signal), axis=1)
        n_values += baseline_signal.shape[1]
    mean = sum_values / n_values
    variance = np.maximum(sum_squares / n_values - np.square(mean), 0.0)
    return mean, np.sqrt(variance), int(np.sum(mask))


def _baseline_channel_whitening_matrix(data, baseline_window, trial_indices):
    baseline_features, n_baseline_samples = _extract_window_features(data, baseline_window, feature_mode="sensor_mean", trial_indices=trial_indices)
    whitening = reptrace_baseline_whitening_matrix(
        baseline_features,
        shrinkage=BASELINE_WHITENING_SHRINKAGE,
        eigenvalue_floor=BASELINE_WHITENING_EIGENVALUE_FLOOR,
    )
    return whitening, n_baseline_samples


def _selected_trial_indices(labels, max_trials_per_class):
    labels = np.asarray(labels).ravel()
    if max_trials_per_class is None:
        return np.arange(labels.shape[0], dtype=int)
    max_trials_per_class = int(max_trials_per_class)
    if max_trials_per_class <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")

    selected = []
    counts: Counter[int] = Counter()
    for index, label in enumerate(labels):
        if counts[int(label)] < max_trials_per_class:
            selected.append(index)
            counts[int(label)] += 1
    return np.asarray(selected, dtype=int)


def _iter_trial_indices(data, trial_indices):
    if trial_indices is None:
        return range(_count_trials(data))
    return (int(index) for index in np.asarray(trial_indices, dtype=int).ravel())


def _trialinfo_labels(data):
    trialinfo = _unwrap_singleton(get_data_field(data, "trialinfo"))
    return np.asarray(trialinfo, dtype=int).ravel()


def _count_trials(data):
    trial_field = _unwrap_outer_cell(get_data_field(data, "trial"))
    values = np.asarray(trial_field, dtype=object)
    if values.ndim == 2 and values.shape[0] == 1:
        return int(values.shape[1])
    if values.ndim == 2 and values.shape[1] == 1:
        return int(values.shape[0])
    return int(values.size)


def _time_vector(data, trial_idx):
    return np.asarray(_cell_item(get_data_field(data, "time"), trial_idx), dtype=float).ravel()


def _trial_signal(data, trial_idx):
    return np.asarray(_cell_item(get_data_field(data, "trial"), trial_idx), dtype=float)


def _cell_item(cell, index):
    values = np.asarray(_unwrap_outer_cell(cell), dtype=object)
    if values.ndim == 0:
        return _unwrap_singleton(values.item())
    if values.ndim == 2 and values.shape[0] == 1:
        return _unwrap_singleton(values[0, index])
    if values.ndim == 2 and values.shape[1] == 1:
        return _unwrap_singleton(values[index, 0])
    return _unwrap_singleton(values[index])


def _unwrap_outer_cell(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _unwrap_singleton(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _time_mask(time_vector, time_window):
    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")
    tolerance = 1e-12
    mask = (time_vector >= start - tolerance) & (time_vector <= stop + tolerance)
    if not np.any(mask):
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    return mask


def _normalized_subject_features(feature_set, config):
    if feature_set.normalization == config.normalization:
        return feature_set.features
    if config.normalization == "none":
        return feature_set.features
    return _normalize_features(
        feature_set.features,
        config,
        feature_set.baseline_feature_mean,
        feature_set.baseline_feature_std,
        feature_set.baseline_whitening_matrix,
    )


def _normalize_features(features, config, baseline_feature_mean, baseline_feature_std, baseline_whitening_matrix):
    n_channels = _whitening_n_channels(config, baseline_whitening_matrix)
    return reptrace_normalize_features(
        features,
        mode=config.normalization,
        baseline_mean=baseline_feature_mean,
        baseline_std=baseline_feature_std,
        whitening=baseline_whitening_matrix,
        n_channels=n_channels,
    )


def _whitening_n_channels(config, baseline_whitening_matrix):
    if config.feature_mode != "sensor_flat" or baseline_whitening_matrix is None:
        return None
    return int(np.asarray(baseline_whitening_matrix).shape[0])


def _centered_window(center, size):
    return float(np.round(center - size / 2, 10)), float(np.round(center + size / 2, 10))


def _sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _finite_metric_values(rows, key):
    values = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _nanmedian_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def _nanmax_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.max(values))


def _nanmin_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.min(values))


def _sem_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return _sem(values)


def _percent_nanmean_or_nan(values):
    value = _nanmean_or_nan(values)
    return float(100.0 * value) if np.isfinite(value) else np.nan


def _percent_sem_or_nan(values):
    value = _sem_or_nan(values)
    return float(100.0 * value) if np.isfinite(value) else np.nan


def _format_counter(counter):
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _row_value_counts(rows, key, *, fallback_key=None, transform=str):
    values = []
    for row in rows:
        value = row.get(key)
        if (value is None or value == "") and fallback_key is not None:
            value = row.get(fallback_key)
        if value is None or value == "":
            continue
        try:
            values.append(transform(value))
        except (TypeError, ValueError):
            continue
    return Counter(values)


def _single_row_value(rows, key, *, default=""):
    values = []
    for row in rows:
        value = row.get(key, default)
        if value in (None, ""):
            continue
        if value not in values:
            values.append(value)
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    return ";".join(str(value) for value in values)


def _normalized_config(config):
    return CrossSubjectStimulusConfig(
        window_center=config.window_center,
        window_size=config.window_size,
        baseline_window=config.baseline_window,
        feature_mode=_normalize_feature_mode(config.feature_mode),
        normalization=_normalize_normalization(config.normalization),
        alignment=_normalize_alignment(config.alignment),
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=_normalize_trial_cap(config.max_trials_per_class_per_participant),
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _normalized_candidate_configs(candidate_configs):
    normalized_configs = tuple(_normalized_config(config) for config in candidate_configs)
    if not normalized_configs:
        raise ValueError("At least one candidate configuration is required.")
    chance_classes = {config.chance_classes for config in normalized_configs}
    if len(chance_classes) != 1:
        raise ValueError("All nested candidate configurations must use the same chance_classes value.")
    return normalized_configs


def _normalize_trial_cap(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")
    return value


def _normalize_feature_mode(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}.")
    return normalized


def _normalize_normalization(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_MODES}.")
    return normalized


def _normalize_alignment(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in ALIGNMENT_MODES:
        raise ValueError(f"alignment must be one of {ALIGNMENT_MODES}.")
    return normalized
