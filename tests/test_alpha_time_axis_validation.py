import unittest

import numpy as np

from pymegdec.alpha_metrics import AlphaMetricConfig, compute_alpha_analytic_window
from pymegdec.alpha_movement import sample_time_indices


def _signal_and_time(n_samples=300, fs=200.0):
    time = np.arange(n_samples, dtype=float) / fs - 0.5
    carrier = np.sin(2 * np.pi * 10 * time)
    signal = np.vstack([carrier, 0.5 * carrier])
    return signal, time


class TestAlphaTimeAxisValidation(unittest.TestCase):
    def test_alpha_analytic_window_accepts_row_vector_time_axis(self):
        signal, time = _signal_and_time()

        alpha_window, time_indices = compute_alpha_analytic_window(signal, time[None, :], AlphaMetricConfig())

        self.assertEqual(alpha_window.shape, (signal.shape[0], time_indices.size))

    def test_alpha_analytic_window_rejects_signal_time_length_mismatch(self):
        signal, time = _signal_and_time()

        with self.assertRaisesRegex(ValueError, "samples along its last axis"):
            compute_alpha_analytic_window(signal[:, :-1], time, AlphaMetricConfig())

    def test_alpha_analytic_window_rejects_non_uniform_time_axis(self):
        signal, time = _signal_and_time()
        time = time.copy()
        time[50] += 0.001

        with self.assertRaisesRegex(ValueError, "uniformly sampled"):
            compute_alpha_analytic_window(signal, time, AlphaMetricConfig())

    def test_alpha_analytic_window_rejects_non_increasing_time_axis(self):
        signal, time = _signal_and_time()
        time = time.copy()
        time[50] = time[49]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            compute_alpha_analytic_window(signal, time, AlphaMetricConfig())

    def test_alpha_movement_sampling_rejects_non_uniform_time_axis(self):
        _signal, time = _signal_and_time()
        time = time.copy()
        time[50] += 0.001

        with self.assertRaisesRegex(ValueError, "uniformly sampled"):
            sample_time_indices(time, (-0.3, -0.1), 0.02)
