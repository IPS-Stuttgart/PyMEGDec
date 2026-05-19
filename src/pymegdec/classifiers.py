"""Backward-compatible classifier adapters.

Classifier implementations now live in :mod:`reptrace.decoding.classifiers`.
This module preserves historical ``pymegdec.classifiers`` imports and adds
PyMEGDec-local registry entries that are useful for the stimulus benchmarks.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from reptrace.decoding.classifiers import (
    ClassifierSpec,
    CorrelationPrototypeClassifier,
    DecodedLabelClassifier,
    _build_pytorch_data_loaders,
    encode_classifier_labels,
    positive_class_score,
    prediction_scores,
    should_use_default_classifier_param,
    train_binary_svm,
    train_for_stimulus_lasso_glm,
    train_gradient_boosting,
    train_lasso_logistic,
)
from reptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY as REPTRACE_CLASSIFIER_REGISTRY,
)
from reptrace.decoding.classifiers import (
    DEFAULT_CLASSIFIER_PARAMS as REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
)
from reptrace.decoding.classifiers import (
    get_default_classifier_param as get_reptrace_default_classifier_param,
)
from reptrace.decoding.classifiers import (
    train_classifier as train_reptrace_classifier,
)
from reptrace.decoding.classifiers import (
    train_multiclass_classifier as train_reptrace_multiclass_classifier,
)

_DecodedLabelClassifier = DecodedLabelClassifier
_encode_classifier_labels = encode_classifier_labels

PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS = {
    "multinomial-logistic-weighted": 1.0,
}
DEFAULT_CLASSIFIER_PARAMS = {
    **REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
    **PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS,
}


def _build_weighted_multinomial_logistic(_features, _labels, classifier_param, random_state):
    return LogisticRegression(
        C=float(classifier_param),
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )


CLASSIFIER_REGISTRY = {
    **REPTRACE_CLASSIFIER_REGISTRY,
    "multinomial-logistic-weighted": ClassifierSpec(_build_weighted_multinomial_logistic),
}

__all__ = [
    "CLASSIFIER_REGISTRY",
    "DEFAULT_CLASSIFIER_PARAMS",
    "ClassifierSpec",
    "CorrelationPrototypeClassifier",
    "DecodedLabelClassifier",
    "_build_pytorch_data_loaders",
    "get_default_classifier_param",
    "positive_class_score",
    "prediction_scores",
    "should_use_default_classifier_param",
    "train_binary_svm",
    "train_classifier",
    "train_for_stimulus_lasso_glm",
    "train_gradient_boosting",
    "train_lasso_logistic",
    "train_multiclass_classifier",
]


def get_default_classifier_param(classifier: str) -> Any:
    """Return a defensive copy of the PyMEGDec classifier default."""

    if classifier in DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    return get_reptrace_default_classifier_param(classifier)


def train_classifier(
    features,
    labels,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
):
    """Build and fit a classifier from the PyMEGDec-extended registry."""

    return train_reptrace_classifier(
        features,
        labels,
        classifier,
        classifier_param,
        random_state=random_state,
        registry=CLASSIFIER_REGISTRY if registry is None else registry,
    )


def train_multiclass_classifier(
    features,
    labels,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
):
    """Train a multiclass classifier from the PyMEGDec-extended registry."""

    return train_reptrace_multiclass_classifier(
        features,
        labels,
        classifier,
        classifier_param,
        random_state=random_state,
        registry=CLASSIFIER_REGISTRY if registry is None else registry,
    )


def __getattr__(name: str) -> Any:
    if name == "MLPClassifierTorch":
        from reptrace.decoding.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
