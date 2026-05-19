"""Backward-compatible classifier adapters.

Classifier implementations now live in :mod:`neureptrace.decoding.classifiers`.
This module preserves historical ``pymegdec.classifiers`` imports.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from neureptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY,
    DEFAULT_CLASSIFIER_PARAMS,
    ClassifierSpec,
    CorrelationPrototypeClassifier,
    DecodedLabelClassifier,
    ShrinkagePrototypeClassifier,
    _build_pytorch_data_loaders,
    encode_classifier_labels,
    get_default_classifier_param,
    positive_class_score,
    prediction_scores,
    should_use_default_classifier_param,
    train_binary_svm,
    train_for_stimulus_lasso_glm,
    train_gradient_boosting,
    train_lasso_logistic,
    train_classifier as train_reptrace_classifier,
    train_multiclass_classifier as train_reptrace_multiclass_classifier,
)

_DecodedLabelClassifier = DecodedLabelClassifier
_encode_classifier_labels = encode_classifier_labels

__all__ = [
    "CLASSIFIER_REGISTRY",
    "DEFAULT_CLASSIFIER_PARAMS",
    "ClassifierSpec",
    "CorrelationPrototypeClassifier",
    "DecodedLabelClassifier",
    "ShrinkagePrototypeClassifier",
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


def _optional_sample_weight(sample_weight, expected_length: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(int(expected_length), dtype=float)
    weights = np.asarray(sample_weight, dtype=float).ravel()
    if weights.shape[0] != int(expected_length):
        raise ValueError(
            "sample_weight length must match feature rows: "
            f"{weights.shape[0]} != {expected_length}."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weight values must be finite and non-negative.")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("sample_weight must contain at least one positive value.")
    return weights


def _fit_accepts_sample_weight(estimator) -> bool:
    fit = getattr(estimator, "fit", None)
    if fit is None:
        return False
    try:
        return "sample_weight" in inspect.signature(fit).parameters
    except (TypeError, ValueError):
        return False


def _sample_weight_fit_kwargs(model, sample_weight):
    if _fit_accepts_sample_weight(model):
        return {"sample_weight": sample_weight}
    steps = getattr(model, "steps", None)
    if steps:
        final_name, final_estimator = steps[-1]
        if _fit_accepts_sample_weight(final_estimator):
            return {f"{final_name}__sample_weight": sample_weight}
    return {}


def _fit_model_with_optional_sample_weight(model, features, labels, sample_weight=None):
    fit_kwargs = (
        {} if sample_weight is None else _sample_weight_fit_kwargs(model, sample_weight)
    )
    model.fit(features, labels, **fit_kwargs)
    return model


def train_classifier(
    features,
    labels,
    classifier: str,
    classifier_param=None,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
    sample_weight=None,
):
    """Build and fit a classifier from the upstream registry."""

    if sample_weight is None:
        return train_reptrace_classifier(
            features,
            labels,
            classifier,
            classifier_param,
            random_state=random_state,
            registry=CLASSIFIER_REGISTRY if registry is None else registry,
        )

    registry = CLASSIFIER_REGISTRY if registry is None else registry
    features = np.asarray(features)
    labels = np.asarray(labels).ravel()
    sample_weight = _optional_sample_weight(sample_weight, features.shape[0])
    try:
        classifier_spec = registry[classifier]
    except KeyError as exc:
        supported_classifiers = ", ".join(sorted(registry))
        raise ValueError(
            f"Unsupported classifier: {classifier}. "
            f"Supported classifiers: {supported_classifiers}"
        ) from exc
    model = classifier_spec.builder(features, labels, classifier_param, random_state)
    if classifier_spec.fits_in_builder:
        return model
    return _fit_model_with_optional_sample_weight(
        model,
        features,
        labels,
        sample_weight,
    )


def train_multiclass_classifier(
    features,
    labels,
    classifier: str,
    classifier_param=None,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
    sample_weight=None,
):
    """Train a multiclass classifier from the upstream registry."""

    if sample_weight is None:
        return train_reptrace_multiclass_classifier(
            features,
            labels,
            classifier,
            classifier_param,
            random_state=random_state,
            registry=CLASSIFIER_REGISTRY if registry is None else registry,
        )

    classes, encoded_labels = encode_classifier_labels(labels)
    model = train_classifier(
        features,
        encoded_labels,
        classifier,
        classifier_param,
        random_state=random_state,
        registry=registry,
        sample_weight=sample_weight,
    )
    return DecodedLabelClassifier(model, classes)


def __getattr__(name: str) -> Any:
    if name == "MLPClassifierTorch":
        from neureptrace.decoding.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
