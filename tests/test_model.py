from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.analysis import within_week
from src.model import Factor, fit_factors


def _staggered(start_rule, n_campaigns=40, n_days=140):
    rows = []
    for campaign in range(n_campaigns):
        start = start_rule(campaign)
        for age in range(n_days):
            day = start + timedelta(days=age)
            rows.append(
                {
                    "id": campaign,
                    "age": age,
                    "age_week": age // 7,
                    "age_mod": age % 7,
                    "dow": day.weekday(),
                    "month": day.month - 1,
                    "date": day,
                }
            )
    return pd.DataFrame(rows)


def _fit(frame, y, with_age_mod=True):
    factors = [
        Factor("age_week", frame["age_week"].to_numpy(), [str(i) for i in range(int(frame.age_week.max()) + 1)], False),
        Factor("dow", frame["dow"].to_numpy(), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], True),
        Factor("month", frame["month"].to_numpy(), [str(i) for i in range(12)], True),
    ]
    if with_age_mod:
        factors.insert(1, Factor("age_mod", frame["age_mod"].to_numpy(), [str(i) for i in range(7)], False))
    fit = fit_factors(y, frame["id"].to_numpy(), factors, n_boot=0, name="test")
    dow = fit.coefficients[fit.coefficients.factor == "dow"].set_index("level")["coef"]
    month = fit.coefficients[fit.coefficients.factor == "month"].set_index("level")["coef"]
    return fit, dow, month


def _center(effects, weights):
    effects = np.asarray(effects, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return effects - np.average(effects, weights=weights)


def test_calendar_effects_are_recovered_when_starts_are_staggered():
    frame = _staggered(lambda i: date(2018, 1, 1) + timedelta(days=11 * i))
    dow_true = np.array([0.2, -0.1, 0.0, 0.05, -0.15, 0.3, -0.05])
    month_true = np.array([0.15, 0.0, -0.1, 0.05, 0.0, -0.2, -0.25, -0.05, 0.1, 0.2, 0.0, 0.05])
    age_effect = np.linspace(0.4, -0.4, int(frame.age_week.max()) + 1)
    campaign_effect = np.linspace(-0.3, 0.3, frame["id"].nunique())
    y = (
        campaign_effect[frame["id"].to_numpy()]
        + age_effect[frame["age_week"].to_numpy()]
        + dow_true[frame["dow"].to_numpy()]
        + month_true[frame["month"].to_numpy()]
    )
    fit, dow, month = _fit(frame, y)
    assert fit.rank == fit.n_columns
    dow_counts = frame.groupby("dow").size().reindex(range(7)).to_numpy()
    month_counts = frame.groupby("month").size().reindex(range(12)).to_numpy()
    got_dow = dow.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).to_numpy()
    got_month = month.reindex([str(i) for i in range(12)]).to_numpy()
    assert np.allclose(got_dow, _center(dow_true, dow_counts), atol=1e-8)
    assert np.allclose(got_month, _center(month_true, month_counts), atol=1e-8)


def test_within_week_position_is_not_read_as_a_weekday():
    def start(i):
        # Half the campaigns start on Monday, half on Thursday.
        base = date(2019, 1, 7)  # Monday
        return base if i % 2 == 0 else base + timedelta(days=3)

    frame = _staggered(start, n_campaigns=60, n_days=84)
    decay = np.array([0.30, 0.20, 0.10, 0.0, -0.10, -0.20, -0.30])
    y = decay[frame["age_mod"].to_numpy()] + (frame["id"].to_numpy() % 5) * 0.01
    controlled, dow_controlled, _month = _fit(frame, y, with_age_mod=True)
    raw, dow_raw, _month = _fit(frame, y, with_age_mod=False)
    assert controlled.rank == controlled.n_columns
    assert np.max(np.abs(dow_controlled.to_numpy())) < 1e-8
    assert np.max(np.abs(dow_raw.to_numpy())) > 0.05


def test_bootstrap_recenters_when_campaigns_have_unequal_length():
    frame = _staggered(lambda i: date(2018, 1, 1) + timedelta(days=11 * i), n_campaigns=24, n_days=80)
    frame = frame[~((frame["id"] % 2 == 0) & (frame["age"] >= 50))].copy()
    y = (frame["dow"].to_numpy() * 0.01).astype(float)
    factors = [
        Factor("dow", frame["dow"].to_numpy(), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], True),
        Factor("month", frame["month"].to_numpy(), [str(i) for i in range(12)], True),
    ]
    point = fit_factors(y, frame["id"].to_numpy(), factors, n_boot=0, name="point")
    fit = fit_factors(y, frame["id"].to_numpy(), factors, n_boot=8, seed=1, name="boot")
    assert fit.draws.shape == (8, len(fit.coefficients))
    assert np.isfinite(fit.coefficients["lo"]).all()
    assert np.isfinite(fit.coefficients["hi"]).all()
    assert np.allclose(fit.coefficients["coef"], point.coefficients["coef"])
    assert np.allclose(
        fit.coefficients["n_campaigns"],
        point.coefficients["n_campaigns"],
    )


def test_within_week_ratio_uses_each_weekday_once():
    # Start Wednesday 2020-01-01. Age 7 is also Wednesday.
    start = date(2020, 1, 1)
    assert start.weekday() == 2
    rows = []
    counts = [1, 1, 1, 1, 1, 1, 7]
    for offset, count in enumerate(counts):
        day = start + timedelta(days=7 + offset)
        rows.append({"id": 1, "age": 7 + offset, "dow": day.weekday(), "n": count, "holiday_code": 0})
    table = within_week(pd.DataFrame(rows), n_boot=0, seed=0)
    # Age 7 is Wednesday, so the 7 lands on the following Tuesday.
    expected = np.array([1, 7, 1, 1, 1, 1, 1]) * 7 / sum(counts)
    got = table.sort_values("dow")["equal_median"].to_numpy()
    assert np.allclose(got, expected)
    assert table["n_blocks"].iloc[0] == 1
