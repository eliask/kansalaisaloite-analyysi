from types import SimpleNamespace

import pandas as pd

from src.figures import PHRASE
from src.narrate import clock_after, clock_before, holiday_name, initiatives, num, prose_share


def test_phrase_keys_match():
    assert set(PHRASE["en"]) == set(PHRASE["fi"])


def test_finnish_numbers_use_a_comma_and_a_nbsp():
    assert prose_share(0.705, "en") == "71%"
    assert prose_share(0.028, "en") == "2.8%"
    assert prose_share(0.028, "fi") == "2,8 prosenttia"
    assert num(12048033, "fi") == "12\u00a0048\u00a0033"
    assert initiatives(1, "en") == "1 initiative"
    assert initiatives(2, "fi") == "2 aloitetta"


def test_holiday_names():
    assert holiday_name("Christmas Eve", "en") == "Christmas Eve (jouluaatto)"
    assert holiday_name("Christmas Eve", "fi") == "jouluaatto"


def _shape_row():
    return {
        "group": "full_under_50k",
        "n": 10,
        "median_length": 182,
        "share_d0": 0.0,
        "share_d91": 0.758,
        "share_d91_p25": 0.639,
        "share_d91_p75": 0.845,
        "share_d150": 0.91,
        "first14": 0.217,
        "second14": 0.112,
        "last14": 0.028,
        "slow_start_d91": 0.721,
        "fast_start_d91": 0.848,
        "power_alpha": 0.515,
        "rmse_sqrt": 0.038,
        "rmse_linear": 0.216,
        "rmse_exp": 0.026,
        "sqrt_at_91": (91 / 183) ** 0.5,
        "linear_at_91": 91 / 183,
    }


def _clock_report(notes=None):
    shape = pd.DataFrame([_shape_row()])
    summary = pd.DataFrame([{
        "group": "full_under_50k",
        "piece": "Days 0–6",
        "n": 10,
        "p25": 0.02,
        "p50": 0.095,
        "p75": 0.22,
    }])
    report = SimpleNamespace(clock_shape=shape, clock_summary=summary, time_to_50k=pd.DataFrame())
    if notes is not None:
        report.path_notes = notes
    return report


def _log_notes():
    return {
        "log_offset": 16.8,
        "log_doubling": 0.28,
        "rmse_log": 0.006,
        "rmse_exp": 0.026,
        "rmse_sqrt": 0.038,
        "rmse_linear": 0.216,
        "rmse_power": 0.032,
        "power_p": 0.46,
        "exp_half_life_days": 50.2,
        "log_abs_from_day14": 0.015,
        "common_log_p50": 0.117,
        "own_log_p50": 0.065,
        "own_exp_p50": 0.068,
        "sqrt_individual_p50": 0.126,
        "median_daily_7": 0.0105,
        "median_daily_30": 0.0050,
        "median_daily_90": 0.0021,
        "median_daily_150": 0.0013,
        "median_slope_7": 0.0238,
        "median_slope_30": 0.0109,
        "median_slope_90": 0.0053,
        "median_slope_150": 0.0025,
    }


def test_clock_prose_reads_the_shape_row():
    report = _clock_report(_log_notes())
    english = clock_before(report, "en")
    finnish = clock_before(report, "fi")
    assert "76%" in english
    assert "75%" in english
    assert "71%" in english
    assert "50%" in english
    assert "2.8%" in english
    assert "ln(1 + t/16.8)" in english
    assert "0.006" in english
    assert "smallest of these gaps" in english
    assert "0.117" in english
    assert "1.05%" in english
    assert "1/(t + 16.8)" in english
    assert "closest of the three" not in english
    assert "0.52" not in english
    assert "campaign" not in english.lower()
    assert "significant" not in english.lower()
    assert "76 prosenttia" in finnish
    assert "2,8 prosenttia" in finnish
    assert "ln(1 + t/16,8)" in finnish
    assert "keskineliöpoikkeama" in finnish.lower()
    assert "0,006" in finnish
    assert "pienin" in finnish
    assert "1,05 prosenttia" in finnish
    assert "lähinnä mediaania" not in finnish
    assert "kampanja" not in finnish
    assert "merkittäv" not in finnish


def test_clock_prose_without_a_log_fit_keeps_the_three_curve_ranking():
    report = _clock_report()
    english = clock_before(report, "en")
    finnish = clock_before(report, "fi")
    assert "closest of the three" in english
    assert "0.52" in english
    assert "lähinnä mediaania" in finnish
    assert "kampanja" not in finnish
    assert "campaign" not in english.lower()


def test_blue_rise_is_the_median_line_and_the_median_addition():
    ordinary = _shape_row()
    stayed = dict(ordinary)
    stayed.update({
        "group": "full_50k",
        "n": 55,
        "share_d91": 0.453,
        "first14": 0.157,
        "last14": 0.023,
        "cross_p50": 163,
        "share_at_cross": 0.926,
    })
    early = dict(ordinary)
    early.update({
        "group": "early_50k",
        "n": 37,
        "median_length": 41,
        "share_d30": 0.98,
        "first14": 0.793,
        "share_half_own": 0.786,
        "cross_p50": 23,
        "share_at_cross": 0.923,
    })
    report = _clock_report()
    report.clock_shape = pd.DataFrame([ordinary, stayed, early])
    report.path_notes = {
        "blue_day25": 0.215,
        "blue_day100": 0.473,
        "blue_day150": 0.736,
        "blue_day160": 0.904,
        "blue_path_150_160": 0.169,
        "blue_own_150_160": 0.022,
    }
    english = clock_after(report, "en")
    finnish = clock_after(report, "fi")
    assert "90%" in english
    assert "rises by 17%" in english
    assert "adds 2.2%" in english
    assert "days 170" in english
    assert "90 prosenttia" in finnish
    assert "nousee 17 prosenttia" in finnish
    assert "lisää 2,2 prosenttia" in finnish
    assert "kampanja" not in finnish
    assert "campaign" not in english.lower()
