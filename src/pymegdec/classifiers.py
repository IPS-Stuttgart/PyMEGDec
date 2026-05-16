"""Classifier compatibility wrappers used by PyMEGDec decoding routines."""

import numpy as np
from reptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY as REPTRACE_CLASSIFIER_REGISTRY,
)
from reptrace.decoding.classifiers import (
    DEFAULT_CLASSIFIER_PARAMS as REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
)
from reptrace.decoding.classifiers import (
    ClassifierSpec,
)
from reptrace.decoding.classifiers import (
    get_default_classifier_param as get_reptrace_default_classifier_param,
)
from reptrace.decoding.classifiers import (
    should_use_default_classifier_param,
    train_binary_svm,
)
from reptrace.decoding.classifiers import train_classifier as train_reptrace_classifier
from reptrace.decoding.classifiers import (
    train_gradient_boosting,
    train_lasso_logistic,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

_PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS = {
    "correlation-prototype": None,
    "multinomial-logistic": 1.0,
    "shrinkage-lda": None,
    "xgboost": 100,
    "pytorch-mlp": {
        "hidden_dim": 720,
        "max_epochs": 500,
        "learning_rate": 1e-3,
        "dropout_rate": 0.2,
        "random_seed": 0,
    },
}
_DEFAULT_CLASSIFIER_PARAMS = {
    **REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
    **_PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS,
}
__all__ = [
    "CLASSIFIER_REGISTRY",
    "ClassifierSpec",
    "get_default_classifier_param",
    "should_use_default_classifier_param",
    "train_binary_svm",
    "train_for_stimulus_lasso_glm",
    "train_gradient_boosting",
    "train_multiclass_classifier",
]


def __getattr__(name):
    if name == "MLPClassifierTorch":
        from pymegdec.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_xgboost(_features, _labels, classifier_param, random_state):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the xgboost extra to use classifier='xgboost'.") from exc

    return xgb.XGBClassifier(
        n_estimators=int(classifier_param),
        eval_metric="mlogloss",
        random_state=random_state,
    )


class CorrelationPrototypeClassifier:
    """Classify by correlation to class-average feature prototypes."""

    def __init__(self):
        self.classes_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None
        self.normalized_prototypes_: np.ndarray | None = None

    def fit(self, features, labels):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels).ravel()
        self.classes_ = np.unique(labels)
        if self.classes_.size == 0:
            raise ValueError("At least one class is required.")
        self.prototypes_ = np.vstack([np.mean(features[labels == class_label], axis=0) for class_label in self.classes_])
        self.normalized_prototypes_ = self._row_center_normalize(self.prototypes_)
        return self

    def decision_function(self, features):
        if self.normalized_prototypes_ is None:
            raise RuntimeError("CorrelationPrototypeClassifier must be fitted before scoring.")
        features = np.asarray(features, dtype=float)
        return self._row_center_normalize(features) @ self.normalized_prototypes_.T

    def predict(self, features):
        if self.classes_ is None:
            raise RuntimeError("CorrelationPrototypeClassifier must be fitted before prediction.")
        scores = self.decision_function(features)
        return self.classes_[np.argmax(scores, axis=1)]

    @staticmethod
    def _row_center_normalize(values):
        values = np.asarray(values, dtype=float)
        centered = values - np.mean(values, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return centered / norms


def _build_correlation_prototype(_features, _labels, _classifier_param, _random_state):
    return CorrelationPrototypeClassifier()


def _build_multinomial_logistic(_features, _labels, classifier_param, random_state):
    return LogisticRegression(
        C=float(classifier_param),
        max_iter=1000,
        random_state=random_state,
    )


def _build_shrinkage_lda(_features, _labels, classifier_param, _random_state):
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=_normalize_lda_shrinkage(classifier_param))


def _normalize_lda_shrinkage(classifier_param):
    if classifier_param is None:
        return "auto"
    if isinstance(classifier_param, str):
        normalized = classifier_param.strip().lower()
        if normalized == "auto":
            return "auto"
    shrinkage = float(classifier_param)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage-lda classifier_param must be 'auto' or a numeric shrinkage in [0, 1].")
    return shrinkage


def _build_pytorch_mlp_classifier(features, labels, classifier_param, random_state):
    return _train_pytorch_mlp(
        features,
        labels,
        classifier_param,
        random_state=random_state,
    )


CLASSIFIER_REGISTRY = {
    **REPTRACE_CLASSIFIER_REGISTRY,
    "correlation-prototype": ClassifierSpec(_build_correlation_prototype),
    "multinomial-logistic": ClassifierSpec(_build_multinomial_logistic),
    "shrinkage-lda": ClassifierSpec(_build_shrinkage_lda),
    "xgboost": ClassifierSpec(_build_xgboost),
    "pytorch-mlp": ClassifierSpec(_build_pytorch_mlp_classifier, fits_in_builder=True),
}


class _DecodedLabelClassifier:
    """Expose original stimulus labels while fitting the wrapped model on 0-based labels."""

    def __init__(self, model, classes):
        self.model = model
        self.classes_ = np.asarray(classes, dtype=int)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, features):
        encoded_predictions = np.asarray(self.model.predict(features), dtype=int)
        return self._decode(encoded_predictions)

    def decision_function(self, features):
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(features), dtype=float)
            if scores.ndim == 1 and self.classes_.shape[0] == 2:
                return np.column_stack((-scores, scores))
            return scores
        if hasattr(self.model, "forward"):
            return self._torch_logits(features)
        predictions = np.asarray(self.model.predict(features), dtype=int)
        scores = np.zeros((predictions.shape[0], self.classes_.shape[0]), dtype=float)
        for row_index, encoded_label in enumerate(predictions):
            if 0 <= int(encoded_label) < self.classes_.shape[0]:
                scores[row_index, int(encoded_label)] = 1.0
        return scores

    def predict_proba(self, features):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.model.__class__.__name__!r} object has no attribute 'predict_proba'")
        return self.model.predict_proba(features)

    def _decode(self, encoded_labels):
        encoded_labels = np.asarray(encoded_labels, dtype=int)
        if np.any(encoded_labels < 0) or np.any(encoded_labels >= self.classes_.shape[0]):
            raise ValueError("Classifier returned an encoded label outside the fitted class range.")
        return self.classes_[encoded_labels]

    def _torch_logits(self, features):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install PyMEGDec with the torch extra to score classifier='pytorch-mlp'.") from exc

        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32)
            logits = self.model.forward(tensor)
        return logits.detach().cpu().numpy()


def train_multiclass_classifier(
    features,
    labels,
    classifier,
    classifier_param,
    random_state=None,
):
    classes, encoded_labels = _encode_classifier_labels(labels)
    model = train_reptrace_classifier(
        features,
        encoded_labels,
        classifier,
        classifier_param,
        random_state=random_state,
        registry=CLASSIFIER_REGISTRY,
    )
    return _DecodedLabelClassifier(model, classes)


def _encode_classifier_labels(labels):
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.size == 0:
        raise ValueError("At least one class label is required.")
    classes = np.unique(labels)
    encoded = np.searchsorted(classes, labels).astype(int, copy=False)
    return classes, encoded


def _train_pytorch_mlp(features, labels, classifier_param, random_state=None):
    random_seed = _resolve_pytorch_random_seed(classifier_param, random_state)
    if random_seed is not None:
        _seed_pytorch_training(random_seed)

    model = _build_pytorch_mlp(features, labels, classifier_param)
    train_loader, val_loader = _build_pytorch_data_loaders(features, labels, random_seed=random_seed)
    trainer = _build_pytorch_trainer(classifier_param, random_seed=random_seed)
    trainer.fit(model, train_loader, val_loader)
    return model


def _resolve_pytorch_random_seed(classifier_param, random_state):
    random_seed = random_state
    if random_seed is None:
        random_seed = classifier_param.get("random_seed")
    if random_seed is None:
        return None
    return int(random_seed)


def _seed_pytorch_training(random_seed):
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    pl.seed_everything(random_seed, workers=True)


def _build_pytorch_mlp(features, labels, classifier_param):
    try:
        from pymegdec.torch_models import MLPClassifierTorch
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    return MLPClassifierTorch(
        features.shape[1],
        int(classifier_param["hidden_dim"]),
        len(np.unique(labels)),
        learning_rate=classifier_param["learning_rate"],
        dropout_rate=classifier_param["dropout_rate"],
    )


def _build_pytorch_data_loaders(features, labels, *, random_seed=None):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    train_dataset, val_dataset = _split_pytorch_dataset(torch, features, labels, random_seed)
    train_generator = _build_torch_generator(torch, random_seed)
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, generator=train_generator),
        torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False),
    )


def _split_pytorch_dataset(torch, features, labels, random_seed):
    full_dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    split_generator = _build_torch_generator(torch, random_seed)
    return torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=split_generator)


def _build_torch_generator(torch, random_seed):
    if random_seed is None:
        return None

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    return generator


def _build_pytorch_trainer(classifier_param, *, random_seed=None):
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    return pl.Trainer(
        max_epochs=int(classifier_param["max_epochs"]),
        default_root_dir=r"lightning_logs",
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
        deterministic=random_seed is not None,
    )


def train_for_stimulus_lasso_glm(
    train_features,
    train_labels,
    lambda_,
    random_state=None,
):
    return train_lasso_logistic(
        train_features,
        train_labels,
        lambda_,
        random_state=random_state,
    )


def get_default_classifier_param(classifier):
    if classifier in _PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = _PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    return get_reptrace_default_classifier_param(classifier)
