"""Export bidirectional stimulus-pair confusion structure from prediction rows."""

import argparse
import csv
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymegdec.alpha_metrics import write_alpha_metrics_csv  # noqa: E402
from pymegdec.stimulus_cross_subject import (  # noqa: E402
    summarize_cross_subject_confusion_category_enrichment,
    summarize_cross_subject_confusion_category_matrix,
    summarize_cross_subject_confusion_pairs,
)


def _read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(rows, path):
    output_path = Path(path)
    if rows:
        write_alpha_metrics_csv(rows, output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")


def _parse_category_columns(value):
    if value in (None, ""):
        return None
    return tuple(column.strip() for column in value.split(",") if column.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export unordered, bidirectional stimulus-pair confusion summaries.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV containing true_stimulus and predicted_stimulus columns.")
    parser.add_argument("--output", required=True, help="Output CSV with one row per confused stimulus pair.")
    parser.add_argument("--stimulus-metadata", default=None, help="Optional CSV mapping stimulus ids to names/categories for interpretation.")
    parser.add_argument("--category-output", default=None, help="Optional output CSV testing same-category error enrichment.")
    parser.add_argument("--category-matrix-output", default=None, help="Optional output CSV with category-to-category error counts and lifts.")
    parser.add_argument("--category-columns", default=None, help="Comma-separated metadata columns to treat as categories. Defaults to inferred repeated metadata columns.")
    parser.add_argument("--category-permutations", type=int, default=10000, help="Permutation count for same-category enrichment p-values.")
    parser.add_argument("--category-seed", type=int, default=0, help="Permutation seed for same-category enrichment p-values.")
    args = parser.parse_args(argv)

    prediction_rows = _read_csv_rows(args.predictions)
    metadata_rows = _read_csv_rows(args.stimulus_metadata) if args.stimulus_metadata else None
    pair_rows = summarize_cross_subject_confusion_pairs(prediction_rows, stimulus_metadata_rows=metadata_rows)
    _write_rows(pair_rows, args.output)
    print(f"Wrote {len(pair_rows)} confusion-pair rows to {args.output}")

    category_columns = _parse_category_columns(args.category_columns)
    if args.category_output:
        category_rows = summarize_cross_subject_confusion_category_enrichment(
            prediction_rows,
            stimulus_metadata_rows=metadata_rows,
            category_columns=category_columns,
            n_permutations=args.category_permutations,
            seed=args.category_seed,
        )
        _write_rows(category_rows, args.category_output)
        print(f"Wrote {len(category_rows)} category-enrichment rows to {args.category_output}")
    if args.category_matrix_output:
        category_matrix_rows = summarize_cross_subject_confusion_category_matrix(
            prediction_rows,
            stimulus_metadata_rows=metadata_rows,
            category_columns=category_columns,
        )
        _write_rows(category_matrix_rows, args.category_matrix_output)
        print(f"Wrote {len(category_matrix_rows)} category-matrix rows to {args.category_matrix_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
