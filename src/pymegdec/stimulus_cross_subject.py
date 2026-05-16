"""Public compatibility facade for cross-subject stimulus decoding.

The implementation lives in :mod:`pymegdec._stimulus_cross_subject_core`.
This module is kept as the stable public import path used by PyMEGDec's
CLI, tests, and downstream modules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import sys

import numpy as np

from pymegdec import _stimulus_cross_subject_core as _core

DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"
DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0
TRIAL_SELECTION_MODES = ("random", "first")

_ORIGINAL_SCORE_OUTER_FOLD_MODEL = _core._score_outer_fold_model
_ORIGINAL_SUMMARIZE_CROSS_SUBJECT_STIMULUS_SMOKE = _core.summarize_cross_subject_stimulus_smoke


@dataclass(frozen=True)
class CrossSubjectStimulusConfig(_core.CrossSubjectStimulusConfig):
    """Cross-subject stimulus config with reproducible trial-cap sampling."""

    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED


@dataclass(frozen=True)
class ParticipantFeatureSet(_core.ParticipantFeatureSet):
    """Windowed features with original trial-index bookkeeping."""

    trial_indices: np.ndarray | None = None
    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED


def make_cross_subject_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=_core.DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
    window_size=_core.DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    baseline_window=_core.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    feature_modes=(_core.DEFAULT_CROSS_SUBJECT_FEATURE_MODE,),
    normalizations=(_core.DEFAULT_CROSS_SUBJECT_NORMALIZATION,),
    alignments=(_core.DEFAULT_CROSS_SUBJECT_ALIGNMENT,),
    classifiers=(_core.DEFAULT_CROSS_SUBJECT_CLASSIFIER,),
    classifier_params=(float("nan"),),
    components_pca_values=(_core.DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,),
    max_trials_per_class_per_participant=None,
    trial_selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    trial_selection_seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    chance_classes=_core.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
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
            trial_selection=trial_selection,
            trial_selection_seed=trial_selection_seed,
            chance_classes=chance_classes,
            random_state=random_state,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        for window_center, feature_mode, normalization, alignment, classifier, classifier_param, components_pca in _core.product(
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
    data_path = _core.Path(_core.resolve_data_folder(data_folder)) / f"Part{int(participant)}Data.mat"
    data = _core.sio.loadmat(data_path)["data"][0]
    all_labels = _core._trialinfo_labels(data)  # pylint: disable=protected-access
    trial_indices = _selected_trial_indices(
        all_labels,
        config.max_trials_per_class_per_participant,
        selection=config.trial_selection,
        seed=config.trial_selection_seed,
        participant=participant,
    )
    labels = all_labels[trial_indices]
    features, n_window_samples = _core._extract_window_features(  # pylint: disable=protected-access
        data,
        _core._centered_window(config.window_center, config.window_size),  # pylint: disable=protected-access
        feature_mode=config.feature_mode,
        trial_indices=trial_indices,
    )
    baseline_feature_mean = None
    baseline_feature_std = None
    baseline_whitening_matrix = None
    n_baseline_samples = 0
    if config.normalization in ("subject_baseline_z", "subject_baseline_whiten"):
        baseline_feature_mean, baseline_feature_std, n_baseline_samples = _core._baseline_feature_statistics(  # pylint: disable=protected-access
            data,
            config,
            n_window_samples,
            trial_indices,
        )
    if config.normalization == "subject_baseline_whiten":
        baseline_whitening_matrix, n_baseline_samples = _core._baseline_channel_whitening_matrix(  # pylint: disable=protected-access
            data,
            config.baseline_window,
            trial_indices,
        )
    normalized_features = _core._normalize_features(  # pylint: disable=protected-access
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
        n_channels=int(_core._trial_signal(data, 0).shape[0]),  # pylint: disable=protected-access
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
    return rows


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    outer_row, prediction_rows = _ORIGINAL_SCORE_OUTER_FOLD_MODEL(
        fitted_model,
        test_set,
        config,
        include_predictions=include_predictions,
    )
    outer_row["trial_selection"] = config.trial_selection
    outer_row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
    for prediction_row in prediction_rows:
        prediction_row["trial_selection"] = config.trial_selection
        prediction_row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
    return outer_row, prediction_rows


def _feature_cache_key(config):
    return (
        float(config.window_center),
        float(config.window_size),
        float(config.baseline_window[0]),
        float(config.baseline_window[1]),
        str(config.feature_mode),
        str(config.normalization),
        config.max_trials_per_class_per_participant,
        str(getattr(config, "trial_selection", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)),
        _seed_field(getattr(config, "trial_selection_seed", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)),
    )


def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):
    train_window = _core._centered_window(config.window_center, config.window_size)  # pylint: disable=protected-access
    trial_indices = _feature_set_trial_indices(test_set)
    rows = []
    for trial_idx, true_label, predicted_label, true_label_rank in zip(trial_indices, test_labels, predictions, true_label_ranks):
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
                "trial_selection": config.trial_selection,
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
        feature_mode=_core._normalize_feature_mode(config.feature_mode),  # pylint: disable=protected-access
        normalization=_core._normalize_normalization(config.normalization),  # pylint: disable=protected-access
        alignment=_core._normalize_alignment(config.alignment),  # pylint: disable=protected-access
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=_core._normalize_trial_cap(config.max_trials_per_class_per_participant),  # pylint: disable=protected-access
        trial_selection=_normalize_trial_selection(getattr(config, "trial_selection", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)),
        trial_selection_seed=_normalize_trial_selection_seed(
            getattr(config, "trial_selection_seed", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)
        ),
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _normalize_trial_selection(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in TRIAL_SELECTION_MODES:
        raise ValueError(f"trial_selection must be one of {TRIAL_SELECTION_MODES}.")
    return normalized


def _normalize_trial_selection_seed(value):
    if value is None or value == "":
        return None
    value = int(value)
    if value < 0:
        raise ValueError("trial_selection_seed must be non-negative or None.")
    return value


_core.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
_core.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED
_core.TRIAL_SELECTION_MODES = TRIAL_SELECTION_MODES
_core.CrossSubjectStimulusConfig = CrossSubjectStimulusConfig
_core.ParticipantFeatureSet = ParticipantFeatureSet
_core.make_cross_subject_candidate_configs = make_cross_subject_candidate_configs
_core.load_participant_stimulus_features = load_participant_stimulus_features
_core.summarize_cross_subject_stimulus_smoke = summarize_cross_subject_stimulus_smoke
_core._score_outer_fold_model = _score_outer_fold_model  # pylint: disable=protected-access
_core._feature_cache_key = _feature_cache_key  # pylint: disable=protected-access
_core._prediction_rows = _prediction_rows  # pylint: disable=protected-access
_core._selected_trial_indices = _selected_trial_indices  # pylint: disable=protected-access
_core._feature_set_trial_indices = _feature_set_trial_indices  # pylint: disable=protected-access
_core._seed_field = _seed_field  # pylint: disable=protected-access
_core._normalized_config = _normalized_config  # pylint: disable=protected-access
_core._normalize_trial_selection = _normalize_trial_selection  # pylint: disable=protected-access
_core._normalize_trial_selection_seed = _normalize_trial_selection_seed  # pylint: disable=protected-access
_core.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = (
    *_core.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS,
    "trial_selection",
    "trial_selection_seed",
)

# Make imports of ``pymegdec.stimulus_cross_subject`` resolve to the core module
# object.  This keeps private helper monkey-patches and existing direct imports
# operating on the implementation module rather than on a shallow re-export copy.
sys.modules[__name__] = _core
globals().update(_core.__dict__)
