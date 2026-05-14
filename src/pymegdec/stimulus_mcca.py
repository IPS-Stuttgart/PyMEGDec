"""LOSO cross-subject stimulus decoding with RepTrace M-CCA alignment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.cli import normalize_argv, parse_classifier_param, parse_int_or_inf
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec
from pymegdec.stimulus_cross_subject import (
    DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    DEFAULT_CROSS_SUBJECT_CLASSIFIER,
    DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,
    DEFAULT_CROSS_SUBJECT_NORMALIZATION,
    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
    DEFAULT_CROSS_SUBJECT_WINDOW_CENTER,
    DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    FEATURE_MODES,
    NORMALIZATION_MODES,
    CrossSubjectStimulusConfig,
    load_participant_stimulus_features,
    summarize_cross_subject_confusion_pairs,
    summarize_cross_subject_predictions,
)
from reptrace.decoding.mcca import CLASS_ALIGNMENT_SAMPLE_MODES, fit_class_mcca
from reptrace.decoding.windowed import (
    fit_window_model,
    predict_window_model,
    transform_window_features,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score

TARGET_CENTERING_MODES = ("group_mean", "target_unsupervised")


@dataclass(frozen=True)
class CrossSubjectMCCAConfig:  # pylint: disable=too-many-instance-attributes
    window_center: float = DEFAULT_CROSS_SUBJECT_WINDOW_CENTER
    window_size: float = DEFAULT_CROSS_SUBJECT_WINDOW_SIZE
    baseline_window: tuple[float, float] = DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    feature_mode: str = "sensor_flat"
    normalization: str = DEFAULT_CROSS_SUBJECT_NORMALIZATION
    classifier: str = DEFAULT_CROSS_SUBJECT_CLASSIFIER
    classifier_param: object = float("nan")
    components_pca: int | float = DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA
    max_trials_per_class_per_participant: int | None = None
    chance_classes: int = DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES
    random_state: int | None = 0
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0
    mcca_components: int | float = 64
    mcca_regularization: float = 1e-6
    mcca_subject_pca_components: int | float | None = None
    mcca_sample_mode: str = "class_repetition"
    mcca_repetitions_per_class: int | None = None
    target_centering: str = "target_unsupervised"


def evaluate_cross_subject_mcca(data_folder, participants, *, config=None, outer_participants=None, progress=None, label_shuffle_control=False, label_shuffle_seed=0):
    config = _checked(config or CrossSubjectMCCAConfig())
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    outer_participants = tuple(participants if outer_participants is None else [int(participant) for participant in outer_participants])
    if len(participants) < 3:
        raise ValueError("At least three participants are required.")
    if set(outer_participants) - set(participants):
        raise ValueError("outer_participants must be a subset of participants.")
    feature_config = CrossSubjectStimulusConfig(
        window_center=config.window_center,
        window_size=config.window_size,
        baseline_window=config.baseline_window,
        feature_mode=config.feature_mode,
        normalization=config.normalization,
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )
    sets = []
    for participant in participants:
        if progress:
            progress(f"LOAD participant={participant}")
        sets.append(load_participant_stimulus_features(data_folder, participant, config=feature_config))
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)
    outer_rows, prediction_rows = [], []
    for test_participant in outer_participants:
        if progress:
            progress(f"START outer_test_participant={test_participant}")
        train_sets = [item for item in sets if item.participant != test_participant]
        test_set = next(item for item in sets if item.participant == test_participant)
        outer, preds = _fold(train_sets, test_set, config, classifier_param, label_shuffle_seed if label_shuffle_control else None)
        outer.update(label_shuffle_control=bool(label_shuffle_control), label_shuffle_seed=int(label_shuffle_seed) if label_shuffle_control else "")
        for row in preds:
            row.update(label_shuffle_control=bool(label_shuffle_control), label_shuffle_seed=int(label_shuffle_seed) if label_shuffle_control else "")
        outer_rows.append(outer)
        prediction_rows.extend(preds)
        if progress:
            progress(f"DONE outer_test_participant={test_participant} balanced_accuracy={outer['balanced_accuracy']:.4f}")
    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    return {
        "outer": outer_rows,
        "group_summary": summarize_cross_subject_mcca(outer_rows, config=config),
        "predictions": prediction_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": summarize_cross_subject_confusion_pairs(prediction_rows),
    }


def export_cross_subject_mcca(  # pylint: disable=too-many-arguments
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
    outer_participants=None,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    artifacts = evaluate_cross_subject_mcca(
        data_folder,
        participants,
        config=config,
        outer_participants=outer_participants,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    for rows, path in (
        (artifacts["outer"], outer_output_path),
        (artifacts["group_summary"], group_summary_output_path),
        (artifacts["predictions"], predictions_output_path),
        (artifacts["confusion"], confusion_output_path),
        (artifacts["per_stimulus"], per_stimulus_output_path),
        (artifacts["confusion_pairs"], confusion_pairs_output_path),
    ):
        if path and rows:
            write_alpha_metrics_csv(rows, path)
    return artifacts


def _fold(train_sets, test_set, config, classifier_param, label_shuffle_seed):
    labels_by_subject = {item.participant: _labels(item.labels, label_shuffle_seed, test_set.participant, item.participant) for item in train_sets}
    features_by_subject = {item.participant: item.features for item in train_sets}
    model, alignment = fit_class_mcca(
        features_by_subject,
        labels_by_subject,
        sample_mode=config.mcca_sample_mode,
        n_repetitions_per_class=config.mcca_repetitions_per_class,
        n_components=config.mcca_components,
        regularization=config.mcca_regularization,
        subject_pca_components=config.mcca_subject_pca_components,
    )
    train_x = np.vstack([model.transform(item.participant, item.features) for item in train_sets])
    train_y = np.concatenate([labels_by_subject[item.participant] for item in train_sets])
    test_mean = np.mean(test_set.features, axis=0) if config.target_centering == "target_unsupervised" else None
    test_x = model.transform_group(test_set.features, feature_mean=test_mean)
    bundle = fit_window_model(
        train_x,
        train_y,
        fit_model=lambda x, y: train_multiclass_classifier(x, y, config.classifier, classifier_param, random_state=config.random_state),
        components_pca=config.components_pca,
        train_window=(config.window_center - config.window_size / 2, config.window_center + config.window_size / 2),
    )
    y_pred, _ = predict_window_model(bundle, test_x)
    score_matrix, class_order = _score_matrix(bundle, test_x)
    top2, top3, mean_rank, rank_rows = _rank_metrics(score_matrix, class_order, test_set.labels)
    accuracy = float(accuracy_score(test_set.labels, y_pred))
    balanced = float(balanced_accuracy_score(test_set.labels, y_pred))
    meta = _meta(config)
    outer = {
        **meta,
        "test_participant": test_set.participant,
        "n_train_participants": len(train_sets),
        "n_train_trials": int(train_x.shape[0]),
        "n_test_trials": int(test_set.labels.shape[0]),
        "n_classes": int(np.unique(test_set.labels).size),
        "chance_accuracy": 1.0 / config.chance_classes,
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "balanced_accuracy": balanced,
        "balanced_percent": 100.0 * balanced,
        "top2_accuracy": top2,
        "top2_percent": 100.0 * top2 if np.isfinite(top2) else np.nan,
        "top3_accuracy": top3,
        "top3_percent": 100.0 * top3 if np.isfinite(top3) else np.nan,
        "mean_true_label_rank": mean_rank,
        "mcca_actual_components": model.n_components,
        "mcca_alignment_rows": int(next(iter(alignment.aligned_by_subject.values())).shape[0]),
        "mcca_repetitions_per_class": alignment.n_repetitions_per_class,
        "classifier_param": classifier_param,
        "actual_components_pca": bundle.actual_components_pca,
        "pca_explained_variance_percent": bundle.explained_variance_percent,
    }
    rows = []
    for i, (truth, pred) in enumerate(zip(test_set.labels, y_pred, strict=True)):
        rows.append(
            {
                **meta,
                "test_participant": test_set.participant,
                "trial_index": i,
                "true_stimulus": int(truth),
                "predicted_stimulus": int(pred),
                "correct": bool(truth == pred),
                **rank_rows[i],
            }
        )
    return outer, rows


def summarize_cross_subject_mcca(outer_rows, *, config=None):
    if not outer_rows:
        return []
    config = _checked(config or CrossSubjectMCCAConfig())
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows])
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows])
    chance = float(outer_rows[0]["chance_accuracy"])
    diff = balanced - chance
    return [
        {
            **_meta(config),
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(raw)),
            "accuracy_median": float(np.median(raw)),
            "accuracy_sem": _sem(raw),
            "percent_mean": float(100.0 * np.mean(raw)),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "top2_accuracy_mean": _nanmean([row.get("top2_accuracy", np.nan) for row in outer_rows]),
            "top3_accuracy_mean": _nanmean([row.get("top3_accuracy", np.nan) for row in outer_rows]),
            "mean_true_label_rank_mean": _nanmean([row.get("mean_true_label_rank", np.nan) for row in outer_rows]),
            "chance_mean_rank": 0.5 * ((1.0 / chance) + 1.0),
            "mean_above_chance": float(np.mean(diff)),
            "percent_above_chance": float(100.0 * np.mean(diff)),
            "participants_above_chance": int(np.sum(diff > 0)),
            "participants_total": int(len(diff)),
            "participants_at_or_below_chance": int(np.sum(diff <= 0)),
            "one_sided_exact_sign_p_value": _exact_sign_p(diff),
            "one_sided_signflip_p_value": _signflip_p(diff, config.signflip_permutations, config.signflip_seed),
            "label_shuffle_control": outer_rows[0].get("label_shuffle_control", False),
            "label_shuffle_seed": outer_rows[0].get("label_shuffle_seed", ""),
        }
    ]


def _meta(config):
    return {
        "window_center_s": config.window_center,
        "window_size_s": config.window_size,
        "window_start_s": config.window_center - config.window_size / 2,
        "window_stop_s": config.window_center + config.window_size / 2,
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": config.feature_mode,
        "normalization": config.normalization,
        "alignment": "mcca_group_projection",
        "mcca_sample_mode": config.mcca_sample_mode,
        "mcca_requested_components": config.mcca_components,
        "mcca_regularization": config.mcca_regularization,
        "mcca_subject_pca_components": config.mcca_subject_pca_components,
        "target_centering": config.target_centering,
        "classifier": config.classifier,
        "components_pca": config.components_pca,
        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
    }


def _score_matrix(bundle, features):
    x = transform_window_features(bundle, features)
    model = bundle.model
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x), dtype=float)
    else:
        return None, None
    if scores.ndim != 2:
        return None, None
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(list(model.named_steps.values())[-1], "classes_", None)
    return scores, None if classes is None else np.asarray(classes)


def _rank_metrics(scores, classes, y_true):
    empty: list[dict[str, object]] = [{} for _ in y_true]
    if scores is None or classes is None:
        return np.nan, np.nan, np.nan, empty
    order = np.argsort(scores, axis=1)[:, ::-1]
    top2, top3, ranks, rows = [], [], [], []
    for i, truth in enumerate(y_true):
        ranked = classes[order[i]]
        top2.append(truth in ranked[:2])
        top3.append(truth in ranked[:3])
        match = np.flatnonzero(ranked == truth)
        rank = int(match[0]) + 1 if match.size else np.nan
        ranks.append(rank)
        row = {"true_label_rank": rank, "true_label_score": np.nan}
        true_index = np.flatnonzero(classes == truth)
        if true_index.size:
            row["true_label_score"] = float(scores[i, true_index[0]])
        for k, idx in enumerate(order[i, :3], start=1):
            row[f"rank{k}_stimulus"] = int(classes[idx])
            row[f"rank{k}_score"] = float(scores[i, idx])
        rows.append(row)
    return float(np.mean(top2)), float(np.mean(top3)), _nanmean(ranks), rows


def _labels(labels, seed, test_participant, train_participant):
    labels = np.asarray(labels).copy()
    if seed is None:
        return labels
    rng = np.random.default_rng(abs(hash((int(seed), int(test_participant), int(train_participant)))) % (2**32))
    rng.shuffle(labels)
    return labels


def _checked(config):
    if config.feature_mode not in FEATURE_MODES or config.normalization not in NORMALIZATION_MODES:
        raise ValueError("Unsupported feature mode or normalization.")
    if config.mcca_sample_mode not in CLASS_ALIGNMENT_SAMPLE_MODES or config.target_centering not in TARGET_CENTERING_MODES:
        raise ValueError("Unsupported M-CCA mode.")
    return config


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def _sem(values):
    values = np.asarray(values, dtype=float)
    return 0.0 if values.size < 2 else float(np.std(values, ddof=1) / np.sqrt(values.size))


def _exact_sign_p(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    n = int(diff.size)
    if n == 0:
        return 1.0
    pos = int(np.sum(diff > 0))
    return float(sum(comb(n, k) for k in range(pos, n + 1)) / (2**n))


def _signflip_p(diff, n_perm, seed):
    if n_perm <= 0:
        return np.nan
    diff = np.asarray(diff, dtype=float)
    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    null = np.mean(rng.choice([-1.0, 1.0], size=(int(n_perm), diff.size)) * diff[None, :], axis=1)
    return float((np.sum(null >= obs) + 1.0) / (int(n_perm) + 1.0))


def _parse_window(value: str) -> tuple[float, float]:
    lo, hi = value.split(",", maxsplit=1)
    return float(lo), float(hi)


def _optional_int(value: str):
    return None if value.lower() in {"none", "auto", "null"} else parse_int_or_inf(value)


def _parser(prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Run LOSO stimulus decoding with RepTrace M-CCA alignment.")
    parser.add_argument("--data-dir", dest="data_folder", default=None)
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS)
    parser.add_argument("--outer-participants", default=None)
    parser.add_argument("--window-center", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_CENTER)
    parser.add_argument("--window-size", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE)
    parser.add_argument("--baseline-window", type=_parse_window, default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="sensor_flat")
    parser.add_argument("--normalization", choices=NORMALIZATION_MODES, default=DEFAULT_CROSS_SUBJECT_NORMALIZATION)
    parser.add_argument("--classifier", default=DEFAULT_CROSS_SUBJECT_CLASSIFIER)
    parser.add_argument("--classifier-param", default=None)
    parser.add_argument("--components-pca", type=parse_int_or_inf, default=DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA)
    parser.add_argument("--mcca-components", type=parse_int_or_inf, default=64)
    parser.add_argument("--mcca-regularization", type=float, default=1e-6)
    parser.add_argument("--mcca-subject-pca-components", type=_optional_int, default=None)
    parser.add_argument("--mcca-sample-mode", choices=CLASS_ALIGNMENT_SAMPLE_MODES, default="class_repetition")
    parser.add_argument("--mcca-repetitions-per-class", type=int, default=None)
    parser.add_argument("--target-centering", choices=TARGET_CENTERING_MODES, default="target_unsupervised")
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None)
    parser.add_argument("--chance-classes", type=int, default=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--label-shuffle-control", action="store_true")
    parser.add_argument("--label-shuffle-seed", type=int, default=0)
    parser.add_argument("--signflip-permutations", type=int, default=10000)
    parser.add_argument("--signflip-seed", type=int, default=0)
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_mcca_outer.csv")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_mcca_group_summary.csv")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_mcca_predictions.csv")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_mcca_confusion.csv")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_mcca_per_stimulus.csv")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_mcca_confusion_pairs.csv")
    return parser


def stimulus_cross_subject_mcca(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    args = _parser(prog).parse_args(normalize_argv(argv))
    participants = parse_participant_spec(args.participants)
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    config = CrossSubjectMCCAConfig(
        window_center=args.window_center,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_mode=args.feature_mode,
        normalization=args.normalization,
        classifier=args.classifier,
        classifier_param=parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
        mcca_components=args.mcca_components,
        mcca_regularization=args.mcca_regularization,
        mcca_subject_pca_components=args.mcca_subject_pca_components,
        mcca_sample_mode=args.mcca_sample_mode,
        mcca_repetitions_per_class=args.mcca_repetitions_per_class,
        target_centering=args.target_centering,
    )
    artifacts = export_cross_subject_mcca(
        args.data_folder,
        participants,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        config=config,
        outer_participants=outer_participants,
        progress=lambda msg: print(msg, flush=True),
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
    )
    print(f"Wrote {len(artifacts['outer'])} held-out participant rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return stimulus_cross_subject_mcca(argv, prog="pymegdec stimulus-cross-subject-mcca")


if __name__ == "__main__":
    raise SystemExit(main())
