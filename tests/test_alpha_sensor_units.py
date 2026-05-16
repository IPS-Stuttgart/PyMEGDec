import numpy as np

from pymegdec.alpha_metrics import AlphaMetricConfig, compute_alpha_metrics, get_channel_positions_mm
from pymegdec.alpha_movement import AlphaMovementConfig, compute_alpha_movement


def _cell(values):
    out = np.empty((1, len(values)), dtype=object)
    for i, value in enumerate(values):
        out[0, i] = value
    return out


def _data(scale=1.0, unit=None):
    fs = 200
    time = np.arange(-0.5, 1.0, 1 / fs)
    carrier = np.sin(2 * np.pi * 10 * time)
    trial = np.vstack([1.5 * carrier, 0.8 * carrier, 0.4 * carrier, 0.3 * carrier])
    grad = {
        "chanpos": scale
        * np.array(
            [
                [-20.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, -20.0, 0.0],
            ]
        )
    }
    if unit is not None:
        grad["unit"] = unit
    return {
        "label": np.array(["MLO11", "MRO11", "MZO01", "MLF11"], dtype=object)[:, None],
        "trial": _cell([trial]),
        "time": _cell([time[None, :]]),
        "trialinfo": np.array([[1]]),
        "grad": grad,
    }


def test_channel_positions_auto_convert_m_to_mm():
    np.testing.assert_allclose(get_channel_positions_mm(_data(scale=0.001, unit="m")), get_channel_positions_mm(_data()))


def test_alpha_metrics_use_mm_after_auto_unit_conversion():
    baseline = compute_alpha_metrics(_data())[0]
    converted = compute_alpha_metrics(_data(scale=0.001, unit="m"))[0]
    for field in ("spatial_freq_rad_per_mm", "speed_m_per_s", "gradient_x", "gradient_y"):
        np.testing.assert_allclose(converted[field], baseline[field], rtol=1e-12, atol=1e-12)


def test_alpha_metrics_allow_explicit_unit_override():
    baseline = compute_alpha_metrics(_data())[0]
    converted = compute_alpha_metrics(_data(scale=0.001), config=AlphaMetricConfig(sensor_position_unit="m"))[0]
    np.testing.assert_allclose(converted["spatial_freq_rad_per_mm"], baseline["spatial_freq_rad_per_mm"], rtol=1e-12, atol=1e-12)


def test_alpha_movement_uses_mm_after_auto_unit_conversion():
    cfg = AlphaMovementConfig(time_window=(-0.2, 0.4), trajectory_step_s=0.2)
    baseline = compute_alpha_movement(_data(), config=cfg)
    converted = compute_alpha_movement(_data(scale=0.001, unit="m"), config=cfg)
    for base_row, converted_row in zip(baseline, converted, strict=True):
        for field in ("centroid_x_mm", "displacement_mm", "speed_mm_per_s"):
            np.testing.assert_allclose(converted_row[field], base_row[field], rtol=1e-12, atol=1e-12)
