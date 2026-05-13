"""Export cross-subject smoke benchmark with optional training-label controls."""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymegdec.alpha_metrics import write_alpha_metrics_csv  # noqa: E402
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig  # noqa: E402
from pymegdec.stimulus_cross_subject_controls import (  # noqa: E402
    LABEL_CONTROL_MODES,
    evaluate_cross_subject_stimulus_smoke_controlled,
    normalize_label_control,
)


def _parse_participants(value: str) -> tuple[int, ...]:
    participants: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", maxsplit=1)
            participants.extend(range(int(start_text), int(stop_text) + 1))
        else:
            participants.append(int(token))
    if not participants:
        raise argparse.ArgumentTypeError("At least one participant is required.")
    return tuple(participants)


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", maxsplit=1))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected a start,stop time window.")
    if parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("Window start must be before stop.")
    return parts


def _parse_int_or_inf(value: str):
    value = str(value).strip().lower()
    if value in {"inf", "infinity"}:
        return float("inf")
    return int(value)


def _parse_classifier_param(value: str | None):
    if value is None:
        return float("nan")
    value = value.strip()
    if value == "" or value.lower() in {"default", "nan"}:
        return float("nan")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value
    if isinstance(parsed, (int, float)) and math.isnan(float(parsed)):
        return float("nan")
    return parsed


def _write(rows, path):
    if path and rows:
        write_alpha_metrics_csv(rows, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", dest="data_folder", required=True)
    parser.add_argument("--participants", type=_parse_participants, required=True)
    parser.add_argument("--window-center", type=float, required=True)
    parser.add_argument("--window-size", type=float, required=True)
    parser.add_argument("--baseline-window", type=_parse_time_window, required=True)
    parser.add_argument("--feature-mode", default="sensor_mean")
    parser.add_argument("--normalization", default="subject_baseline_z")
    parser.add_argument("--classifier", default="multiclass-svm")
    parser.add_argument("--classifier-param", default=None)
    parser.add_argument("--components-pca", type=_parse_int_or_inf, default=64)
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None)
    parser.add_argument("--chance-classes", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--signflip-permutations", type=int, default=10000)
    parser.add_argument("--signflip-seed", type=int, default=0)
    parser.add_argument("--label-control", default="none", choices=LABEL_CONTROL_MODES)
    parser.add_argument("--label-control-seed", type=int, default=0)
    parser.add_argument("--outer-output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--confusion-output", default=None)
    parser.add_argument("--per-stimulus-output", default=None)
    parser.add_argument("--confusion-pairs-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = CrossSubjectStimulusConfig(
        window_center=args.window_center,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_mode=args.feature_mode,
        normalization=args.normalization,
        classifier=args.classifier,
        classifier_param=_parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    artifacts = evaluate_cross_subject_stimulus_smoke_controlled(
        args.data_folder,
        args.participants,
        config=config,
        label_control=normalize_label_control(args.label_control),
        label_control_seed=args.label_control_seed,
        progress=lambda message: print(message, flush=True),
    )
    _write(artifacts["outer"], args.outer_output)
    _write(artifacts["group_summary"], args.summary_output)
    _write(artifacts["predictions"], args.predictions_output)
    _write(artifacts["confusion"], args.confusion_output)
    _write(artifacts["per_stimulus"], args.per_stimulus_output)
    _write(artifacts["confusion_pairs"], args.confusion_pairs_output)
    print(f"Wrote {len(artifacts['outer'])} held-out participant rows to {args.outer_output}")
    if args.summary_output:
        print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
