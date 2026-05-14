"""Export fixed-pipeline cross-subject LOSO decoding across time windows."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pymegdec.alpha_metrics import write_alpha_metrics_csv  # noqa: E402
from pymegdec.cli import (  # noqa: E402
    parse_classifier_param,
    parse_float_list,
    parse_int_or_inf,
)
from pymegdec.data_config import resolve_data_folder  # noqa: E402
from pymegdec.reaction_time_analysis import parse_participant_spec  # noqa: E402
from pymegdec.stimulus_cross_subject import (  # noqa: E402
    ALIGNMENT_MODES,
    DEFAULT_CROSS_SUBJECT_ALIGNMENT,
    DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    DEFAULT_CROSS_SUBJECT_CLASSIFIER,
    DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,
    DEFAULT_CROSS_SUBJECT_FEATURE_MODE,
    DEFAULT_CROSS_SUBJECT_NORMALIZATION,
    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
    DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    FEATURE_MODES,
    NORMALIZATION_MODES,
    CrossSubjectStimulusConfig,
    evaluate_cross_subject_stimulus_smoke,
)

DEFAULT_WINDOW_CENTERS = (
    -0.200,
    -0.150,
    -0.100,
    -0.050,
    0.000,
    0.050,
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
    0.225,
    0.250,
    0.300,
    0.400,
    0.500,
    0.600,
)


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", maxsplit=1))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time window must have the form start,stop.")
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Time window start must be before stop.")
    return parts


def _token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _parse_centers(value: str) -> tuple[float, ...]:
    centers = tuple(parse_float_list(value))
    if not centers:
        raise argparse.ArgumentTypeError("At least one window center is required.")
    return centers


def _compact_float(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.3f}".replace("-", "m").replace(".", "p")


def _rows_with_time(rows, center: float):
    result = []
    for row in rows:
        updated = dict(row)
        updated.setdefault("window_center_s", center)
        updated["time_window_center_s"] = center
        result.append(updated)
    return result


def _readable_table(summary_rows: list[dict]) -> str:
    if not summary_rows:
        return "No summary rows produced."
    frame = pd.DataFrame(summary_rows)
    columns = [
        "window_center_s",
        "balanced_percent_mean",
        "balanced_percent_sem",
        "top2_percent_mean",
        "top3_percent_mean",
        "mean_true_label_rank_mean",
        "participants_above_chance",
        "one_sided_exact_sign_p_value",
    ]
    present = [column for column in columns if column in frame.columns]
    return frame[present].to_string(index=False, float_format=lambda value: f"{value:.4f}")


def _plot_curve(summary_rows: list[dict], output_path: str | None) -> None:
    if not output_path or not summary_rows:
        return
    frame = pd.DataFrame(summary_rows).sort_values("window_center_s")
    x = frame["window_center_s"].astype(float)
    y = frame["balanced_percent_mean"].astype(float)
    sem = frame["balanced_percent_sem"].astype(float) if "balanced_percent_sem" in frame else None
    chance = float(frame["chance_percent"].iloc[0]) if "chance_percent" in frame else 100.0 / 16.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(chance, linestyle="--", linewidth=1.0, label=f"chance ({chance:.2f}%)")
    if sem is not None:
        ax.errorbar(x, y, yerr=sem, marker="o", linewidth=1.5, capsize=3, label="balanced accuracy")
    else:
        ax.plot(x, y, marker="o", linewidth=1.5, label="balanced accuracy")
    ax.set_xlabel("Window center relative to stimulus onset (s)")
    ax.set_ylabel("Held-out participant balanced accuracy (%)")
    ax.set_title("Cross-subject 22-to-1 stimulus decoding over time")
    ax.legend(loc="best")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--window-centers", type=_parse_centers, default=DEFAULT_WINDOW_CENTERS, help="Comma-separated window centers in seconds.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE, help="Window size in seconds.")
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, help="Baseline window as start,stop in seconds.")
    parser.add_argument("--feature-mode", type=_token, choices=FEATURE_MODES, default=DEFAULT_CROSS_SUBJECT_FEATURE_MODE, help="Feature extraction mode.")
    parser.add_argument("--normalization", type=_token, choices=NORMALIZATION_MODES, default=DEFAULT_CROSS_SUBJECT_NORMALIZATION, help="Subject normalization mode.")
    parser.add_argument("--alignment", type=_token, choices=ALIGNMENT_MODES, default=DEFAULT_CROSS_SUBJECT_ALIGNMENT, help="Cross-subject training alignment mode.")
    parser.add_argument("--classifier", default=DEFAULT_CROSS_SUBJECT_CLASSIFIER, help="Classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Classifier parameter value, JSON, Python literal, numeric value, or nan.")
    parser.add_argument("--components-pca", type=parse_int_or_inf, default=DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA, help="Number of PCA components, or inf.")
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None, help="Optional deterministic per-class trial cap.")
    parser.add_argument("--chance-classes", type=int, default=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to the classifier.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for group summaries.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_time_resolved_outer.csv", help="All held-out participant score rows.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_time_resolved_group_summary.csv", help="One group summary row per time window.")
    parser.add_argument("--plot-output", default="outputs/stimulus_cross_subject_time_resolved_curve.png", help="Output PNG for the mean time curve.")
    parser.add_argument("--readme-output", default="outputs/README.md", help="Markdown summary output.")
    parser.add_argument("--write-predictions", action="store_true", help="Also write trial predictions and confusion diagnostics for every time window.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_time_resolved_predictions.csv", help="Trial prediction CSV if --write-predictions is set.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_time_resolved_confusion.csv", help="Confusion-count CSV if --write-predictions is set.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_time_resolved_per_stimulus.csv", help="Per-stimulus CSV if --write-predictions is set.")
    parser.add_argument(
        "--confusion-pairs-output", default="outputs/stimulus_cross_subject_time_resolved_confusion_pairs.csv", help="Confusion-pair CSV if --write-predictions is set."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        raise SystemExit("At least one participant is required.")

    all_outer_rows = []
    all_summary_rows = []
    all_prediction_rows = []
    all_confusion_rows = []
    all_per_stimulus_rows = []
    all_confusion_pair_rows = []

    for center in args.window_centers:
        print(f"START time_window_center={center:.3f}", flush=True)
        config = CrossSubjectStimulusConfig(
            window_center=float(center),
            window_size=args.window_size,
            baseline_window=args.baseline_window,
            feature_mode=args.feature_mode,
            normalization=args.normalization,
            alignment=args.alignment,
            classifier=args.classifier,
            classifier_param=parse_classifier_param(args.classifier_param),
            components_pca=args.components_pca,
            max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
            chance_classes=args.chance_classes,
            random_state=args.random_state,
            signflip_permutations=args.signflip_permutations,
            signflip_seed=args.signflip_seed,
        )
        artifacts = evaluate_cross_subject_stimulus_smoke(data_folder, participants, config=config, progress=lambda message: print(message, flush=True))
        all_outer_rows.extend(_rows_with_time(artifacts["outer"], float(center)))
        all_summary_rows.extend(_rows_with_time(artifacts["group_summary"], float(center)))
        if args.write_predictions:
            all_prediction_rows.extend(_rows_with_time(artifacts["predictions"], float(center)))
            all_confusion_rows.extend(_rows_with_time(artifacts["confusion"], float(center)))
            all_per_stimulus_rows.extend(_rows_with_time(artifacts["per_stimulus"], float(center)))
            all_confusion_pair_rows.extend(_rows_with_time(artifacts["confusion_pairs"], float(center)))
        print(f"DONE time_window_center={center:.3f}", flush=True)

    write_alpha_metrics_csv(all_outer_rows, args.outer_output)
    write_alpha_metrics_csv(all_summary_rows, args.summary_output)
    if args.write_predictions:
        write_alpha_metrics_csv(all_prediction_rows, args.predictions_output)
        write_alpha_metrics_csv(all_confusion_rows, args.confusion_output)
        write_alpha_metrics_csv(all_per_stimulus_rows, args.per_stimulus_output)
        if all_confusion_pair_rows:
            write_alpha_metrics_csv(all_confusion_pair_rows, args.confusion_pairs_output)
    _plot_curve(all_summary_rows, args.plot_output)

    table = _readable_table(all_summary_rows)
    readme = "\n".join(
        [
            "# Stimulus cross-subject time-resolved benchmark",
            "",
            "Fixed-pipeline leave-one-subject-out image-identity decoding using only `Part*Data.mat`.",
            "Each row is a separate 22-to-1 LOSO benchmark at one window center; no nested tuning is performed.",
            "",
            "```",
            table,
            "```",
            "",
            f"Plot: `{args.plot_output}`",
            "",
        ]
    )
    Path(args.readme_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.readme_output).write_text(readme, encoding="utf-8")
    print(readme)
    print(f"Wrote {len(all_outer_rows)} held-out participant/time rows to {args.outer_output}")
    print(f"Wrote {len(all_summary_rows)} time-summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
