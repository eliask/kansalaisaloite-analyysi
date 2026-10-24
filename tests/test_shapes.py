import numpy as np

from src.shapes import fit_median


def test_known_logarithm_is_recovered_and_beats_the_square_root():
    horizon = 183
    offset = 16.0
    ages = np.arange(horizon + 1, dtype=float)
    median = np.log1p(ages / offset) / np.log1p(horizon / offset)
    fit = fit_median(median, horizon)
    assert fit is not None
    assert fit["rmse_log"] < 1e-3
    assert abs(fit["log_offset"] - offset) < 0.3
    assert fit["rmse_log"] < fit["rmse_sqrt"]
    assert fit["rmse_log"] < fit["rmse_exp"]
    assert fit["rmse_log"] < fit["rmse_power"]


def test_square_root_curve_stays_closer_to_a_square_root():
    horizon = 183
    median = np.sqrt(np.arange(horizon + 1) / horizon)
    median[0] = 1e-9
    fit = fit_median(median, horizon)
    assert fit is not None
    assert fit["rmse_sqrt"] < 1e-9
    assert fit["rmse_sqrt"] < fit["rmse_log"]


def test_incomplete_curve_is_left_unfitted():
    assert fit_median(np.linspace(0.1, 1.0, 183)) is None
    median = np.sqrt(np.arange(184) / 183)
    median[20] = 0.0
    assert fit_median(median) is None
