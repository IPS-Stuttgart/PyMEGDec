import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.data_config import resolve_data_folder
from tests.matlab_fixtures import cell_array


def _mat_data(labels):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = labels
    return {
        "trial": cell_array([np.zeros((1, 2)) for _ in labels]),
        "trialinfo": trialinfo,
    }


class _ConstantClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, features):
        return np.full(features.shape[0], self.label)


class TestCrossValidateSingleDataset(unittest.TestCase):
    def setUp(self) -> None:
        participant_id = 2
        try:
            data_folder = resolve_data_folder(
                required=True,
                required_files=[f"Part{participant_id}Data.mat"],
            )
        except FileNotFoundError as exc:
            if os.getenv("PYMEGDEC_REQUIRE_DATA"):
                self.fail(str(exc))
            self.skipTest(str(exc))

        self.params = {
            "data_folder": data_folder,
            "participant_id": participant_id,
            "n_folds": 10,
            "window_size": 0.1,
            "train_window_center": 0.2,
            "null_window_center": -0.2,
            "new_framerate": float("inf"),
            "classifier": "multiclass-svm",
            "classifier_param": np.nan,
            "components_pca": 200,
            "frequency_range": (0, float("inf")),
        }

    def _accuracy(self, classifier):
        return cross_validate_single_dataset(
            **{
                **self.params,
                "classifier": classifier,
            }
        )

    def test_cross_validate_single_dataset_accuracy_svm(self):
        accuracy = self._accuracy("multiclass-svm")

        self.assertGreaterEqual(accuracy, 0.25, "Accuracy should be at least 0.25")

    def test_cross_validate_single_dataset_accuracy_scikit_mlp(self):
        accuracy = self._accuracy("scikit-mlp")

        self.assertGreaterEqual(accuracy, 0.15, "Accuracy should be at least 0.15")


class TestCrossValidateSingleDatasetSynthetic(unittest.TestCase):
    def test_cross_validate_single_dataset_without_null_window(self):
        labels = np.array([1, 2, 1, 2])
        stimuli_features = [np.array([[index], [index + 1]], dtype=float) for index in range(len(labels))]

        with (
            patch(
                "pymegdec.cross_validation.sio.loadmat",
                return_value={"data": np.array([_mat_data(labels)], dtype=object)},
            ),
            patch(
                "pymegdec.cross_validation.preprocess_features",
                return_value=(stimuli_features, []),
            ),
            patch(
                "pymegdec.cross_validation.train_multiclass_classifier",
                return_value=_ConstantClassifier(1),
            ),
        ):
            accuracy = cross_validate_single_dataset(
                "unused",
                1,
                n_folds=2,
                null_window_center=np.nan,
                components_pca=float("inf"),
            )

        self.assertEqual(accuracy, 0.5)

    def test_cross_validate_single_dataset_zero_bases_labels_without_null_window(self):
        labels = np.array([1, 2, 1, 2])
        stimuli_features = [np.array([[index], [index + 1]], dtype=float) for index in range(len(labels))]
        observed = {}

        def fake_cross_validate_feature_decoding(features, labels_, **kwargs):
            observed["labels"] = np.asarray(labels_).copy()
            observed["null_features"] = kwargs["null_features"]
            return SimpleNamespace(accuracy=0.0)

        with (
            patch(
                "pymegdec.cross_validation.sio.loadmat",
                return_value={"data": np.array([_mat_data(labels)], dtype=object)},
            ),
            patch(
                "pymegdec.cross_validation.preprocess_features",
                return_value=(stimuli_features, []),
            ),
            patch(
                "pymegdec.cross_validation.cross_validate_feature_decoding",
                side_effect=fake_cross_validate_feature_decoding,
            ),
        ):
            cross_validate_single_dataset(
                "unused",
                1,
                n_folds=2,
                null_window_center=np.nan,
                components_pca=float("inf"),
            )

        np.testing.assert_array_equal(observed["labels"], np.array([0, 1, 0, 1]))
        self.assertIsNone(observed["null_features"])

    def test_cross_validate_single_dataset_preserves_labels_with_null_window(self):
        labels = np.array([1, 2, 1, 2])
        stimuli_features = [np.array([[index], [index + 1]], dtype=float) for index in range(len(labels))]
        null_features = [np.array([[index + 10], [index + 11]], dtype=float) for index in range(len(labels))]
        observed = {}

        def fake_cross_validate_feature_decoding(features, labels_, **kwargs):
            observed["labels"] = np.asarray(labels_).copy()
            observed["null_features"] = kwargs["null_features"]
            return SimpleNamespace(accuracy=0.0)

        with (
            patch(
                "pymegdec.cross_validation.sio.loadmat",
                return_value={"data": np.array([_mat_data(labels)], dtype=object)},
            ),
            patch(
                "pymegdec.cross_validation.preprocess_features",
                return_value=(stimuli_features, null_features),
            ),
            patch(
                "pymegdec.cross_validation.cross_validate_feature_decoding",
                side_effect=fake_cross_validate_feature_decoding,
            ),
        ):
            cross_validate_single_dataset(
                "unused",
                1,
                n_folds=2,
                null_window_center=-0.2,
                components_pca=float("inf"),
            )

        np.testing.assert_array_equal(observed["labels"], labels)
        self.assertIsNotNone(observed["null_features"])


if __name__ == "__main__":
    unittest.main()
