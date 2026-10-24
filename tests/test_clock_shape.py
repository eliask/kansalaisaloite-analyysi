import numpy as np

from src.analysis import cumulative_shares, path_fit


def test_share_stays_at_one_after_the_window_ends():
    counts = np.array([3.0, 7.0])
    share = cumulative_shares(counts, length=1, total=10, horizon=3)
    assert np.allclose(share, [0.3, 1.0, 1.0, 1.0])


def test_share_at_the_right_edge_keeps_signatures_past_the_horizon():
    counts = np.zeros(185)
    counts[0] = 40
    counts[184] = 60
    share = cumulative_shares(counts, length=184, total=100, horizon=183)
    assert share[0] == 0.4
    assert share[183] == 0.4
    assert share[-1] == 0.4


def test_square_root_path_is_closer_to_a_square_root_than_to_a_straight_line():
    horizon = 183
    median = np.sqrt(np.arange(horizon + 1) / horizon)
    median[0] = 1e-9
    fit = path_fit(median, horizon)
    assert abs(fit["power_alpha"] - 0.5) < 1e-6
    assert fit["rmse_sqrt"] < 1e-9
    assert fit["rmse_linear"] > 0.1
