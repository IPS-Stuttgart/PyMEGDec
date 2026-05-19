import unittest

import numpy as np

from pymegdec.alpha_signal import bandpass_filter_signal
from pymegdec.preprocessing import filter_features


def _one_trial_data(trial, time):
    trials = np.empty((1, 1), dtype=object)
    times = np.empty((1, 1), dtype=object)
    trials[0, 0] = [np.asarray(trial, dtype=float)]
    times[0, 0] = [np.asarray(time, dtype=float)[None, :]]
    return {"trial": trials, "time": times}


class TestFilterPhase(unittest.TestCase):
    def test_causal_feature_filter_does_not_respond_before_step(self):
        sampling_rate = 1000.0
        time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
        step_index = time.size // 2
        trial = np.zeros((1, time.size), dtype=float)
        trial[0, step_index:] = 1.0
        data = _one_trial_data(trial, time)

        causal = filter_features(data, 0.0, 20.0, filter_phase="causal")
        zero_phase = filter_features(data, 0.0, 20.0, filter_phase="zero_phase")

        causal_trial = causal["trial"][0][0][0]
        zero_phase_trial = zero_phase["trial"][0][0][0]
        pre_step = slice(step_index - 25, step_index)

        np.testing.assert_allclose(causal_trial[:, pre_step], 0.0, atol=1e-12)
        self.assertGreater(np.max(np.abs(zero_phase_trial[:, pre_step])), 1e-4)

    def test_causal_alpha_filter_does_not_respond_before_step(self):
        sampling_rate = 1000.0
        time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
        step_index = time.size // 2
        signal = np.zeros(time.size, dtype=float)
        signal[step_index:] = 1.0

        causal = bandpass_filter_signal(signal, sampling_rate, 1.0, 20.0, filter_phase="causal")
        zero_phase = bandpass_filter_signal(signal, sampling_rate, 1.0, 20.0, filter_phase="zero_phase")
        pre_step = slice(step_index - 25, step_index)

        np.testing.assert_allclose(causal[pre_step], 0.0, atol=1e-12)
        self.assertGreater(np.max(np.abs(zero_phase[pre_step])), 1e-4)

    def test_zero_phase_remains_default_for_backwards_compatibility(self):
        rng = np.random.default_rng(123)
        sampling_rate = 500.0
        time = np.arange(0.0, 1.0, 1.0 / sampling_rate)
        trial = rng.normal(size=(2, time.size))
        data = _one_trial_data(trial, time)

        default_features = filter_features(data, 0.0, 30.0)
        explicit_features = filter_features(data, 0.0, 30.0, filter_phase="zero_phase")
        np.testing.assert_allclose(
            default_features["trial"][0][0][0],
            explicit_features["trial"][0][0][0],
        )

        signal = rng.normal(size=time.size)
        default_alpha = bandpass_filter_signal(signal, sampling_rate, 8.0, 12.0)
        explicit_alpha = bandpass_filter_signal(signal, sampling_rate, 8.0, 12.0, filter_phase="zero_phase")
        np.testing.assert_allclose(default_alpha, explicit_alpha)

    def test_invalid_filter_phase_is_rejected(self):
        time = np.arange(0.0, 1.0, 0.01)
        trial = np.zeros((1, time.size), dtype=float)
        data = _one_trial_data(trial, time)

        with self.assertRaisesRegex(ValueError, "filter_phase"):
            filter_features(data, 0.0, 20.0, filter_phase="bidirectional")
        with self.assertRaisesRegex(ValueError, "filter_phase"):
            bandpass_filter_signal(np.zeros(time.size), 100.0, filter_phase="bidirectional")
