"""PyMEGDec adapter for NeuRepTrace nested cross-person decoding."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from neureptrace.cross_person import (
    ALIGNMENT_MODES as NEUREPTRACE_ALIGNMENT_MODES,
    CrossPersonCandidate,
    SubjectFeatureMatrix,
    make_cross_person_candidate_grid,
    run_nested_cross_person_from_loader,
)

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import normalize_argv, parse_float_list, parse_int_or_inf
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec
from pymegdec.stimulus_cross_subject import (
    DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,
    DEFAULT_CROSS_SUBJECT_FEATURE_MODE,
    DEFAULT_CROSS_SUBJECT_NORMALIZATION,
    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
    DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    FEATURE_MODES,
    NORMALIZATION_MODES,
    CrossSubjectStimulusConfig,
    load_participant_stimulus_features,
)
from pymegdec.stimulus_cue_calibration import load_participant_cue_calibration_features

DEFAULT_CROSS_PERSON_WINDOW_CENTERS = (0.150, 0.175, 0.200)
DEFAULT_CROSS_PERSON_ALIGNMENTS = ("none", "train_class_procrustes", "cue_class_procrustes")
DEFAULT_CROSS_PERSON_DECODERS = ("linear_svm", "shrinkage_lda", "logistic")
DEFAULT_CROSS_PERSON_FEATURE_PREPROCESSORS = ("pca_whiten",)
DEFAULT_CROSS_PERSON_PREFIX = "stimulus_cross_person_nested"


def _parse_token_list(value: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_pca_components_list(value: str) -> tuple[int | float | str | None, ...]:
    values: list[int | float | str | None] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered in {"none", "auto", "default"}:
            values.append(None)
        else:
            parsed = parse_int_or_inf(stripped)
            if parsed == float("inf"):
                values.append(None)
            else:
                values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("At least one PCA component value is required.")
    return tuple(values)


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", maxsplit=1))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time window must have the form start,stop.")
    if parts[1] < parts[0]:
        raise argparse.ArgumentTypeError("Time-window stop must be after start.")
    return parts


def _candidate_to_pymegdec_config(candidate: CrossPersonCandidate) -> CrossSubjectStimulusConfig:
    """Build a PyMEGDec feature-extraction config for a NeuRepTrace candidate.

    PCA/classifier settings stay in NeuRepTrace.  PyMEGDec only extracts the
    candidate's task-specific feature matrix and subject-level normalization.
    """

    candidate = candidate.normalized()
    return CrossSubjectStimulusConfig(
        window_center=candidate.window_center,
        window_size=candidate.window_size,
        baseline_window=candidate.baseline_window,
        feature_mode=candidate.feature_mode,
        normalization=candidate.normalization,
        alignment="none",
        classifier="multiclass-svm",
        classifier_param=None,
        components_pca=float("inf"),
        max_trials_per_class_per_participant=candidate.max_trials_per_class_per_subject,
        chance_classes=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
        random_state=candidate.random_state,
    )


def make_pymegdec_feature_loader(data_folder):
    data_folder = resolve_data_folder(data_folder)

    def loader(subject: str, candidate: CrossPersonCandidate) -> SubjectFeatureMatrix:
        participant = int(subject)
        config = _candidate_to_pymegdec_config(candidate)
        main = load_participant_stimulus_features(data_folder, participant, config=config)
        calibration_features = None
        calibration_labels = None
        if candidate.normalized().alignment == "cue_class_procrustes":
            cue = load_participant_cue_calibration_features(data_folder, participant, config=config)
            calibration_features = cue.features
            calibration_labels = cue.labels
        return SubjectFeatureMatrix(
            subject=str(participant),
            features=main.features,
            labels=main.labels,
            trial_indices=getattr(main, "trial_indices", None),
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            metadata={
                "participant": participant,
                "n_channels": getattr(main, "n_channels", ""),
                "n_window_samples": getattr(main, "n_window_samples", ""),
                "normalization": main.normalization,
            },
        )

    return loader


def _write_artifacts(artifacts, *, out_dir: Path, prefix: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "outer": out_dir / f"{prefix}_outer.csv",
        "inner_validation": out_dir / f"{prefix}_inner_validation.csv",
        "selected": out_dir / f"{prefix}_selected.csv",
        "predictions": out_dir / f"{prefix}_predictions.csv",
    }
    artifact_dict = artifacts.as_dict()
    for key, path in paths.items():
        write_alpha_metrics_csv(artifact_dict[key], path)
    return paths


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run NeuRepTrace nested cross-person decoding on PyMEGDec Part*Data.mat files.",
    )
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and optional Part*CueData.mat files.")
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participant ids to evaluate.")
    parser.add_argument("--window-centers", type=parse_float_list, default=DEFAULT_CROSS_PERSON_WINDOW_CENTERS)
    parser.add_argument("--window-size", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE)
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW)
    parser.add_argument("--feature-modes", type=_parse_token_list, default=(DEFAULT_CROSS_SUBJECT_FEATURE_MODE,), help=f"Comma-separated values from {FEATURE_MODES}.")
    parser.add_argument("--normalizations", type=_parse_token_list, default=(DEFAULT_CROSS_SUBJECT_NORMALIZATION,), help=f"Comma-separated values from {NORMALIZATION_MODES}.")
    parser.add_argument("--alignments", type=_parse_token_list, default=("none",), help=f"Comma-separated values from {NEUREPTRACE_ALIGNMENT_MODES}.")
    parser.add_argument("--decoders", type=_parse_token_list, default=DEFAULT_CROSS_PERSON_DECODERS, help="NeuRepTrace decoder names.")
    parser.add_argument("--emission-modes", type=_parse_token_list, default=("calibrated",))
    parser.add_argument("--feature-preprocessors", type=_parse_token_list, default=DEFAULT_CROSS_PERSON_FEATURE_PREPROCESSORS)
    parser.add_argument("--pca-components-values", type=_parse_pca_components_list, default=(DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,))
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None)
    parser.add_argument("--selection-metric", choices=("accuracy", "balanced_accuracy", "log_loss"), default="balanced_accuracy")
    parser.add_argument("--selection-ensemble-size", type=int, default=1)
    parser.add_argument("--selection-ensemble-diversity", choices=("none", "window", "alignment", "decoder"), default="none")
    parser.add_argument("--label-shuffle-control", action="store_true")
    parser.add_argument("--label-shuffle-seed", type=int, default=0)
    parser.add_argument("--target-calibration-label-shuffle-control", action="store_true")
    parser.add_argument("--target-calibration-label-shuffle-seed", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--prefix", default=DEFAULT_CROSS_PERSON_PREFIX)
    return parser


def stimulus_cross_person_decode(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    participants = tuple(str(participant) for participant in parse_participant_spec(args.participants))
    if len(participants) < 3:
        parser.error("At least three participants are required for nested cross-person decoding.")
    outer_participants = None
    if args.outer_participants:
        outer_participants = tuple(str(participant) for participant in parse_participant_spec(args.outer_participants))
    candidates = make_cross_person_candidate_grid(
        window_centers=args.window_centers,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_modes=args.feature_modes,
        normalizations=args.normalizations,
        alignments=args.alignments,
        decoders=args.decoders,
        emission_modes=args.emission_modes,
        feature_preprocessors=args.feature_preprocessors,
        pca_components_values=args.pca_components_values,
        max_trials_per_class_per_subject=args.max_trials_per_class_per_participant,
        random_state=args.random_state,
    )
    artifacts = run_nested_cross_person_from_loader(
        participants,
        candidate_configs=candidates,
        feature_loader=make_pymegdec_feature_loader(args.data_folder),
        outer_subjects=outer_participants,
        selection_metric=args.selection_metric,
        selection_ensemble_size=args.selection_ensemble_size,
        selection_ensemble_diversity=args.selection_ensemble_diversity,
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
        target_calibration_label_shuffle_control=args.target_calibration_label_shuffle_control,
        target_calibration_label_shuffle_seed=args.target_calibration_label_shuffle_seed,
        progress=lambda message: print(message, flush=True),
    )
    paths = _write_artifacts(artifacts, out_dir=args.out_dir, prefix=args.prefix)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return stimulus_cross_person_decode(argv)


if __name__ == "__main__":
    raise SystemExit(main())
