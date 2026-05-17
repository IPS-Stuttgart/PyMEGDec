import unittest

import numpy as np
from pymegdec.classifiers import get_default_classifier_param
from pymegdec.preprocessing import downsample_data, extract_windows, filter_features, preprocess_features
from tests.matlab_fixtures import cell_array


def _data(trials, times):
    return {
        "trial": cell_array(trials),
        "time": cell_array(times),
    }


def _structured_data(trials, times):
    data = np.empty((1,), dtype=[("trial", "O"), ("time", "O")])
    data["trial"][0] = cell_array(trials)[0]
    data["time"][0] = cell_array(times)[0]
    return data


class TestPreprocessing(unittest.TestCase):
    def test_extract_windows_uses_inclusive_matlab_column_order(self):
        time = np.array([[-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]])
        trial = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [11, 12, 13, 14, 15, 16, 17],
            ]
        )
        data = _data([trial], [time])

        stimuli, null = extract_windows(data, (-0.1, 0.1), (-0.4, -0.2))

        np.testing.assert_array_equal(stimuli[0].ravel(), [4, 14, 5, 15, 6, 16])
        np.testing.assert_array_equal(null[0].ravel(), [1, 11, 2, 12, 3, 13])

    def test_preprocess_features_rejects_touching_null_train_windows(self):
        time = np.array([[-0.3, -0.2, -0.1, 0.0, 0.1]])
        trial = np.array([[1, 2, 3, 4, 5]], dtype=float)
        data = _data([trial], [time])

        with self.assertRaisesRegex(ValueError, "strictly before"):
            preprocess_features(data, (0, float("inf")), float("inf"), 0.2, 0.0, -0.2)

    def test_extract_windows_rejects_rounded_null_sample_overlap(self):
        time = np.array([[-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]])
        trial = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [11, 12, 13, 14, 15, 16, 17],
            ]
        )
        data = _data([trial], [time])

        with self.assertRaisesRegex(ValueError, "overlap"):
            extract_windows(data, (-0.1, 0.1), (-0.25, -0.11))

    def test_downsample_returns_new_data_without_mutating_input(self):
        time = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        trials = [
            np.array([[0, 1, 0, -1, 0], [2, 3, 2, 1, 2]], dtype=float),
            np.array([[10, 11, 10, 9, 10], [-1, 0, -1, -2, -1]], dtype=float),
        ]
        data = _data([trial.copy() for trial in trials], [time.copy(), time.copy()])

        downsampled = downsample_data(data, 2)

        for index in range(2):
            np.testing.assert_allclose(data["time"][0][0][index], time)
            np.testing.assert_allclose(data["trial"][0][0][index], trials[index])
            np.testing.assert_allclose(downsampled["time"][0][0][index], [[0.0, 0.5, 1.0]])
            self.assertEqual(downsampled["trial"][0][0][index].shape, (2, 3))

    def test_downsample_preserves_signal_values_without_implicit_detrending(self):
        time = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        trial = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 12.0, 14.0, 16.0, 18.0],
            ]
        )
        data = _data([trial.copy()], [time.copy()])

        downsampled = downsample_data(data, 2)

        np.testing.assert_allclose(downsampled["time"][0][0][0], [[0.0, 0.5, 1.0]])
        np.testing.assert_allclose(
            downsampled["trial"][0][0][0],
            [[1.0, 3.0, 5.0], [10.0, 14.0, 18.0]],
        )

    def test_downsample_does_not_mutate_scipy_loaded_struct_arrays(self):
        time = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        trial = np.array([[0, 1, 0, -1, 0], [2, 3, 2, 1, 2]], dtype=float)
        data = _structured_data([trial.copy()], [time.copy()])

        downsampled = downsample_data(data, 2)

        np.testing.assert_allclose(data["time"][0][0][0], time)
        np.testing.assert_allclose(data["trial"][0][0][0], trial)
        np.testing.assert_allclose(downsampled["time"][0][0][0], [[0.0, 0.5, 1.0]])
        self.assertEqual(downsampled["trial"][0][0][0].shape, (2, 3))

    def test_filter_returns_new_data_without_mutating_input(self):
        time = np.arange(0.0, 1.0, 0.01)[None, :]
        trial = np.vstack(
            [
                np.sin(2 * np.pi * 5 * time.ravel()) + 0.5 * np.sin(2 * np.pi * 40 * time.ravel()),
                np.cos(2 * np.pi * 5 * time.ravel()) + 0.5 * np.cos(2 * np.pi * 40 * time.ravel()),
            ]
        )
        data = _data([trial.copy(), 2 * trial.copy()], [time.copy(), time.copy()])
        original_second_trial = data["trial"][0][0][1].copy()

        filtered = filter_features(data, 0, 10)

        np.testing.assert_allclose(data["trial"][0][0][1], original_second_trial)
        self.assertFalse(np.allclose(filtered["trial"][0][0][1], original_second_trial))
        self.assertEqual(filtered["trial"][0][0][1].shape, original_second_trial.shape)

    def test_preprocess_features_does_not_mutate_input_between_configurations(self):
        time = np.arange(-0.4, 0.6, 0.01)[None, :]
        trial = np.vstack(
            [
                np.sin(2 * np.pi * 5 * time.ravel()),
                np.cos(2 * np.pi * 5 * time.ravel()),
            ]
        )
        data = _data([trial.copy()], [time.copy()])
        original_trial = data["trial"][0][0][0].copy()
        original_time = data["time"][0][0][0].copy()

        preprocess_features(data, (0, 10), 50, 0.1, 0.2, np.nan)
        preprocess_features(data, (0, float("inf")), float("inf"), 0.1, 0.2, np.nan)

        np.testing.assert_allclose(data["trial"][0][0][0], original_trial)
        np.testing.assert_allclose(data["time"][0][0][0], original_time)

    def test_bandpass_filter_accepts_low_and_high_frequency_cutoffs(self):
        time = np.arange(0.0, 1.0, 0.01)[None, :]
        trial = np.vstack(
            [
                np.sin(2 * np.pi * 5 * time.ravel()),
                np.cos(2 * np.pi * 5 * time.ravel()),
            ]
        )
        data = _data([trial.copy()], [time.copy()])

        filtered = filter_features(data, 1, 10)

        np.testing.assert_allclose(data["trial"][0][0][0], trial)
        self.assertEqual(filtered["trial"][0][0][0].shape, trial.shape)

    def test_matlab_classifier_defaults_are_preserved(self):
        self.assertEqual(get_default_classifier_param("multiclass-svm"), 0.5)
        self.assertEqual(get_default_classifier_param("svm-binary"), 0.5)
        self.assertEqual(get_default_classifier_param("lasso"), 0.005)
        self.assertEqual(get_default_classifier_param("random-forest"), 100)


if __name__ == "__main__":
    unittest.main()
