from datetime import date

import pandas as pd

from src.panel import build_panel


def _initiatives(rows):
    frame = pd.DataFrame(rows)
    for column in ("external_support_count", "total_support_count"):
        if column not in frame:
            frame[column] = frame["support_count"]
    frame["state"] = frame.get("state", "DONE")
    return frame


def test_omitted_zero_days_are_filled_when_the_sum_matches():
    initiatives = _initiatives(
        [
            {
                "id": 1,
                "state": "DONE",
                "start_date": date(2020, 1, 1),
                "end_date": date(2020, 1, 5),
                "support_count": 7,
                "external_support_count": 0,
                "total_support_count": 7,
            }
        ]
    )
    daily = pd.DataFrame({"id": [1, 1], "date": [date(2020, 1, 1), date(2020, 1, 3)], "n": [5, 2]})
    daily.attrs["fetched_ids"] = {1}
    panel, checks = build_panel(initiatives, daily, date(2020, 1, 10))
    assert checks.loc[1, "status"] == "ok"
    assert checks.loc[1, "usable"]
    assert list(panel["n"]) == [5, 0, 2, 0, 0]
    assert int(panel["n"].sum()) == 7
    assert list(panel["age"]) == [0, 1, 2, 3, 4]


def test_closed_sum_mismatch_is_not_usable():
    initiatives = _initiatives(
        [
            {
                "id": 2,
                "state": "DONE",
                "start_date": date(2020, 1, 1),
                "end_date": date(2020, 1, 3),
                "support_count": 100,
                "external_support_count": 0,
                "total_support_count": 100,
            }
        ]
    )
    daily = pd.DataFrame({"id": [2], "date": [date(2020, 1, 1)], "n": [4]})
    daily.attrs["fetched_ids"] = {2}
    panel, checks = build_panel(initiatives, daily, date(2020, 2, 1))
    assert checks.loc[2, "status"] == "sum_mismatch"
    assert not checks.loc[2, "usable"]
    assert panel.empty


def test_small_closed_gap_stays_in_the_panel():
    initiatives = _initiatives(
        [
            {
                "id": 4,
                "state": "DONE",
                "start_date": date(2020, 1, 1),
                "end_date": date(2020, 1, 3),
                "support_count": 1000,
                "external_support_count": 0,
                "total_support_count": 1000,
            }
        ]
    )
    daily = pd.DataFrame({"id": [4], "date": [date(2020, 1, 1)], "n": [990]})
    daily.attrs["fetched_ids"] = {4}
    panel, checks = build_panel(initiatives, daily, date(2020, 2, 1))
    assert checks.loc[4, "status"] == "ok_gap"
    assert checks.loc[4, "usable"]
    assert int(panel["n"].sum()) == 990


def test_open_campaign_keeps_the_unpublished_tail_out_of_the_panel():
    initiatives = _initiatives(
        [
            {
                "id": 3,
                "state": "ACCEPTED",
                "start_date": date(2020, 1, 1),
                "end_date": date(2020, 6, 1),
                "support_count": 10,
                "external_support_count": 1,
                "total_support_count": 11,
            }
        ]
    )
    daily = pd.DataFrame({"id": [3], "date": [date(2020, 1, 1)], "n": [8]})
    daily.attrs["fetched_ids"] = {3}
    panel, checks = build_panel(initiatives, daily, date(2020, 1, 3))
    assert checks.loc[3, "status"] == "open"
    assert checks.loc[3, "residual"] == 2
    assert list(panel["date"]) == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    assert int(panel["n"].sum()) == 8
