"""Zero-fill the nightly series inside each collection window.

The API omits dates with no new web signatures. When the positive days add up
to the web counter, a missing date inside the window is a zero. A closed
campaign whose series is short by at most 25 signatures, or by at most 2
percent, is still used: the unmatched signatures are not placed on a date.
A larger gap is left out of the panel.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .holidays import holiday_table


def _window(start: date | None, end: date | None, cutoff: date) -> tuple[date, date] | None:
    if start is None or end is None:
        return None
    last = end if end < cutoff else cutoff
    if last < start:
        return None
    return start, last


def build_panel(
    initiatives: pd.DataFrame, daily: pd.DataFrame, cutoff: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the zero-filled panel and one check row per initiative."""
    fetched = set(daily.attrs.get("fetched_ids", set(daily["id"].unique()) if len(daily) else set()))
    grouped: dict[int, pd.DataFrame] = {int(i): g for i, g in daily.groupby("id", sort=False)} if len(daily) else {}
    check_rows: list[dict] = []
    parts: list[pd.DataFrame] = []
    for rec in initiatives.itertuples(index=False):
        identifier = int(rec.id)
        window = _window(rec.start_date, rec.end_date, cutoff)
        series = grouped.get(identifier)
        all_sum = int(series["n"].sum()) if series is not None and len(series) else 0
        outside = 0
        in_sum = 0
        positive_days = 0
        if series is not None and len(series) and window is not None:
            start, last = window
            offsets = np.array([(day - start).days for day in series["date"]], dtype=np.int32)
            counts = series["n"].to_numpy(dtype=np.int64)
            inside = (offsets >= 0) & (offsets <= (last - start).days)
            in_sum = int(counts[inside].sum())
            outside = all_sum - in_sum
            positive_days = int(inside.sum())
        elif series is not None and len(series):
            outside = all_sum
        residual = int(rec.support_count) - all_sum
        closed = rec.end_date is not None and rec.end_date <= cutoff
        if rec.start_date is None or rec.end_date is None:
            status = "no_dates"
        elif window is None:
            status = "future" if rec.start_date > cutoff else "no_window"
        elif identifier not in fetched:
            status = "not_fetched"
        elif int(rec.support_count) > 0 and (series is None or len(series) == 0):
            status = "missing_series"
        elif closed and residual != 0:
            tolerance = max(25, int(round(0.02 * int(rec.support_count))))
            status = "ok_gap" if abs(residual) <= tolerance else "sum_mismatch"
        elif closed:
            status = "ok"
        else:
            status = "open"
        usable = status in {"ok", "ok_gap", "open"} and window is not None
        check_rows.append(
            {
                "id": identifier,
                "state": rec.state,
                "start_date": rec.start_date,
                "end_date": rec.end_date,
                "support_count": int(rec.support_count),
                "external_support_count": int(rec.external_support_count),
                "total_support_count": int(rec.total_support_count),
                "series_sum": all_sum,
                "series_sum_in_window": in_sum,
                "series_outside_window": outside,
                "residual": residual,
                "positive_days": positive_days,
                "status": status,
                "usable": usable,
                "canceled": rec.state == "CANCELED",
            }
        )
        if not usable or window is None:
            continue
        start, last = window
        n_days = (last - start).days + 1
        counts = np.zeros(n_days, dtype=np.int32)
        if series is not None and len(series):
            for day, count in zip(series["date"], series["n"], strict=True):
                offset = (day - start).days
                if 0 <= offset < n_days:
                    counts[offset] += int(count)
        dates = [start + timedelta(days=offset) for offset in range(n_days)]
        parts.append(pd.DataFrame({"id": identifier, "date": dates, "n": counts, "age": np.arange(n_days, dtype=np.int16)}))
    checks = pd.DataFrame(check_rows).set_index("id", drop=False)
    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", "date", "n", "age"])
    if len(panel):
        panel = _attach_calendar(panel)
    return panel, checks


def _attach_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(panel["date"])
    iso = dates.dt.isocalendar()
    years = range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1)
    kinds = holiday_table(years)
    public = []
    eves = []
    codes = []
    labels = []
    for day in panel["date"]:
        kind = kinds.get(day)
        if kind is None:
            public.append("")
            eves.append("")
            codes.append(0)
            labels.append("")
        else:
            public.append(kind.public_name)
            eves.append(kind.eve_name)
            codes.append(kind.code)
            labels.append(kind.label)
    out = panel.copy()
    out["dow"] = dates.dt.day_of_week.to_numpy(dtype=np.int8)
    out["month"] = dates.dt.month.to_numpy(dtype=np.int8)
    out["year"] = dates.dt.year.to_numpy(dtype=np.int16)
    out["dom"] = dates.dt.day.to_numpy(dtype=np.int8)
    out["doy"] = dates.dt.day_of_year.to_numpy(dtype=np.int16)
    out["iso_week"] = iso.week.to_numpy(dtype=np.int8)
    out["age_week"] = (out["age"] // 7).astype(np.int16)
    out["age_mod"] = (out["age"] % 7).astype(np.int8)
    out["holiday_code"] = np.array(codes, dtype=np.int8)
    out["holiday_name"] = labels
    out["public_name"] = public
    out["eve_name"] = eves
    out["summer_weekend"] = ((out["month"].isin((6, 7, 8))) & (out["dow"] >= 5)).astype(np.int8)
    return out
