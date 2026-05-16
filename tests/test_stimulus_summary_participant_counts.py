import importlib

import numpy as np

import pymegdec.stimulus_decoding as stimulus_decoding
from pymegdec.stimulus_decoding import summarize_stimulus_decoding, summarize_stimulus_temporal_generalization


def test_summarize_stimulus_decoding_counts_unique_participants():
    rows = [
        {
            "participant": 1,
            "variant": "without_null",
            "window_center_s": 0.0,
            "accuracy": 0.25,
            "chance_accuracy": 0.0625,
            "permutation_p_value": 0.04,
        },
        {
            "participant": 1,
            "variant": "without_null",
            "window_center_s": 0.0,
            "accuracy": 0.5,
            "chance_accuracy": 0.0625,
            "permutation_p_value": 0.02,
        },
        {
            "participant": 2,
            "variant": "without_null",
            "window_center_s": 0.0,
            "accuracy": 0.75,
            "chance_accuracy": 0.0625,
            "permutation_p_value": 0.006,
        },
        {
            "participant": 3,
            "variant": "without_null",
            "window_center_s": 0.1,
            "accuracy": 0.5,
            "chance_accuracy": 0.0625,
            "permutation_p_value": np.nan,
        },
    ]

    summary = summarize_stimulus_decoding(rows)

    zero_window = [row for row in summary if row["window_center_s"] == 0.0][0]
    fallback_window = [row for row in summary if row["window_center_s"] == 0.1][0]
    assert zero_window["n_participants"] == 2
    assert fallback_window["n_participants"] == 1


def test_summarize_stimulus_temporal_generalization_counts_unique_participants():
    rows = [
        {
            "participant": 1,
            "variant": "without_null",
            "train_window_center_s": 0.0,
            "test_window_center_s": 0.0,
            "accuracy": 0.25,
            "chance_accuracy": 0.0625,
        },
        {
            "participant": 1,
            "variant": "without_null",
            "train_window_center_s": 0.0,
            "test_window_center_s": 0.0,
            "accuracy": 0.5,
            "chance_accuracy": 0.0625,
        },
        {
            "participant": 2,
            "variant": "without_null",
            "train_window_center_s": 0.0,
            "test_window_center_s": 0.0,
            "accuracy": 0.75,
            "chance_accuracy": 0.0625,
        },
    ]

    summary = summarize_stimulus_temporal_generalization(rows)

    assert len(summary) == 1
    assert summary[0]["n_participants"] == 2


def test_direct_stimulus_decoding_module_counts_unique_participants_after_reload():
    """Guard against import-time monkey patches hiding the real implementation."""

    module = importlib.reload(stimulus_decoding)
    rows = [
        {
            "participant": 1,
            "variant": "without_null",
            "window_center_s": 0.0,
            "accuracy": 0.25,
            "chance_accuracy": 0.0625,
        },
        {
            "participant": 1,
            "variant": "without_null",
            "window_center_s": 0.0,
            "accuracy": 0.5,
            "chance_accuracy": 0.0625,
        },
        {
            "participant": 2,
            "variant": "without_null",
            "window_center_s": 0.0,
            "accuracy": 0.75,
            "chance_accuracy": 0.0625,
        },
    ]

    summary = module.summarize_stimulus_decoding(rows)

    assert len(summary) == 1
    assert summary[0]["n_participants"] == 2
