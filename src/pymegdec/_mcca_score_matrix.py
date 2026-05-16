"""Robust score-matrix extraction for M-CCA stimulus decoding."""

from __future__ import annotations

import numpy as np


def apply_mcca_score_matrix():
    """Patch M-CCA decoding to expose binary margins as rankable class scores."""
    from pymegdec import stimulus_mcca

    stimulus_mcca._score_matrix = _score_matrix


def _score_matrix(bundle, features):
    from pymegdec import stimulus_mcca

    transformed = stimulus_mcca.transform_window_features(bundle, features)
    model = bundle.model
    classes = _score_classes(model, bundle)
    if classes is None or classes.size == 0:
        return None, None

    for method_name in ("decision_function", "predict_proba"):
        if not hasattr(model, method_name):
            continue
        scores = _as_class_score_matrix(
            getattr(model, method_name)(transformed),
            classes,
            n_samples=int(np.shape(transformed)[0]),
        )
        if scores is not None:
            return scores, classes
    return None, None


def _score_classes(model, bundle):
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                break
    if classes is None:
        train_labels = getattr(bundle, "train_labels", None)
        if train_labels is not None:
            classes = np.unique(np.asarray(train_labels))
    if classes is None:
        return None
    return np.asarray(classes).ravel()


def _as_class_score_matrix(raw_scores, classes, *, n_samples):
    scores = np.asarray(raw_scores, dtype=float)
    if scores.ndim == 1:
        if scores.shape[0] != n_samples or classes.size != 2:
            return None
        return np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[0] != n_samples:
        return None
    if scores.shape[1] == classes.size:
        return scores
    if scores.shape[1] == 1 and classes.size == 2:
        column = scores[:, 0]
        return np.column_stack((-column, column))
    return None
