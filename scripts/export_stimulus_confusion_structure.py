"""Export bidirectional stimulus-pair confusion structure from prediction rows."""

import argparse
import csv
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymegdec.alpha_metrics import write_alpha_metrics_csv  # noqa: E402
from pymegdec.stimulus_cross_subject import summarize_cross_subject_confusion_pairs  # noqa: E402


def _read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export unordered, bidirectional stimulus-pair confusion summaries.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV containing true_stimulus and predicted_stimulus columns.")
    parser.add_argument("--output", required=True, help="Output CSV with one row per confused stimulus pair.")
    parser.add_argument("--stimulus-metadata", default=None, help="Optional CSV mapping stimulus ids to names/categories for interpretation.")
    args = parser.parse_args(argv)

    prediction_rows = _read_csv_rows(args.predictions)
    metadata_rows = _read_csv_rows(args.stimulus_metadata) if args.stimulus_metadata else None
    pair_rows = summarize_cross_subject_confusion_pairs(prediction_rows, stimulus_metadata_rows=metadata_rows)
    if pair_rows:
        write_alpha_metrics_csv(pair_rows, args.output)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    print(f"Wrote {len(pair_rows)} confusion-pair rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
