"""Backward-compatible wrapper for :mod:`pymegdec.model_transfer`."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

if TYPE_CHECKING:
    from pymegdec.classifiers import MLPClassifierTorch

from pymegdec.classifiers import (  # noqa: E402
    get_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.model_transfer import (  # noqa: E402
    evaluate_model_transfer,
    get_original_feature_importance,
)
from pymegdec.preprocessing import (  # noqa: E402
    downsample_data,
    extract_windows,
    filter_features,
    preprocess_features,
)

__all__ = [
    "MLPClassifierTorch",
    "downsample_data",
    "evaluate_model_transfer",
    "extract_windows",
    "filter_features",
    "get_default_classifier_param",
    "get_original_feature_importance",
    "preprocess_features",
    "train_multiclass_classifier",
]


def __getattr__(name):
    if name == "MLPClassifierTorch":
        # pylint: disable-next=no-name-in-module
        from pymegdec.classifiers import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model transfer for one participant.")
    parser.add_argument("--data-dir", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participant", type=int, default=2, help="Participant id to evaluate.")
    args = parser.parse_args()

    acc = evaluate_model_transfer(args.data_dir, args.participant, classifier="multiclass-svm", components_pca=100)
    print(acc)
