import numpy as np

from pymegdec import stimulus_cross_subject as cross_subject


def _prediction_row(true_stimulus, predicted_stimulus, true_label_rank):
    finite_rank = np.isfinite(true_label_rank)
    return {
        "test_participant": 1,
        "window_center_s": 0.175,
        "feature_mode": "sensor_mean",
        "normalization": "subject_baseline_z",
        "alignment": "none",
        "classifier": "multiclass-svm",
        "components_pca": 64,
        "max_trials_per_class_per_participant": "",
        "true_stimulus": int(true_stimulus),
        "predicted_stimulus": int(predicted_stimulus),
        "correct": bool(int(true_stimulus) == int(predicted_stimulus)),
        "true_label_rank": true_label_rank,
        "top2_correct": bool(finite_rank and true_label_rank <= 2),
        "top3_correct": bool(finite_rank and true_label_rank <= 3),
    }


def test_per_stimulus_summary_reports_topk_and_rank_metrics():
    prediction_rows = [
        _prediction_row(1, 1, 1.0),
        _prediction_row(1, 2, 2.0),
        _prediction_row(1, 3, 4.0),
        _prediction_row(2, 3, np.nan),
    ]

    _confusion_rows, per_stimulus_rows = cross_subject.summarize_cross_subject_predictions(prediction_rows)

    stimulus_1 = next(row for row in per_stimulus_rows if int(row["true_label"]) == 1)
    stimulus_2 = next(row for row in per_stimulus_rows if int(row["true_label"]) == 2)

    assert stimulus_1["n_trials"] == 3
    assert stimulus_1["n_correct"] == 1
    assert stimulus_1["accuracy"] == 1 / 3
    assert stimulus_1["n_ranked_trials"] == 3
    assert stimulus_1["top2_accuracy"] == 2 / 3
    assert stimulus_1["top2_percent"] == 100.0 * 2 / 3
    assert stimulus_1["top3_accuracy"] == 2 / 3
    assert stimulus_1["top3_percent"] == 100.0 * 2 / 3
    assert stimulus_1["mean_true_label_rank"] == (1.0 + 2.0 + 4.0) / 3
    assert stimulus_1["median_true_label_rank"] == 2.0

    assert stimulus_2["n_trials"] == 1
    assert stimulus_2["accuracy"] == 0.0
    assert stimulus_2["n_ranked_trials"] == 0
    assert stimulus_2["top2_accuracy"] == 0.0
    assert stimulus_2["top3_accuracy"] == 0.0
    assert np.isnan(stimulus_2["mean_true_label_rank"])
    assert np.isnan(stimulus_2["median_true_label_rank"])


def test_per_stimulus_summary_stays_backward_compatible_without_ranks():
    prediction_rows = [
        {key: value for key, value in _prediction_row(1, 1, 1.0).items() if key != "true_label_rank"},
        {key: value for key, value in _prediction_row(1, 2, 2.0).items() if key != "true_label_rank"},
    ]

    _confusion_rows, per_stimulus_rows = cross_subject.summarize_cross_subject_predictions(prediction_rows)

    stimulus_1 = next(row for row in per_stimulus_rows if int(row["true_label"]) == 1)
    assert stimulus_1["n_trials"] == 2
    assert stimulus_1["n_correct"] == 1
    assert "top2_accuracy" not in stimulus_1
    assert "mean_true_label_rank" not in stimulus_1
