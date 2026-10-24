"""Seasonality of web signatures, with campaign age held fixed.

The raw daily total mixes three different things: how many campaigns are open,
how old those campaigns are, and a handful of very large campaigns. Weekday
and season are estimated only after those are separated.

The weekday picture that does not depend on a regression is a seven-day block
starting at campaign day 7. Each block contains every weekday once. The day's
ratio is seven times its share of the block, so 1 is a fair share of that
week. Blocks that contain a statutory holiday or a named eve are left out.

The regression uses log(1 + count). It removes each campaign's level, then
controls for the week of the campaign and the day inside that week. Month,
weekday and holidays are estimated together. A month effect is available
because campaigns start on different dates: the same week of a campaign falls
in different months.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .holidays import NAMED_ORDER, label_fi
from .load import Snapshot, load_snapshot
from .model import Extra, Factor, Fit, fit_factors, residual_group_means
from .panel import build_panel

BREAK = pd.Timestamp("2017-01-01")
MIN_SIGNATURES = 100
MIN_DAYS = 28
MIN_AGE = 7
THRESHOLD = 50_000
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
HOLIDAY_LABELS = ["Ordinary day", "Public holiday", "Eve"]


def boot_count() -> int:
    return int(os.environ.get("ANALYSIS_BOOT", "60"))


@dataclass
class Report:
    snapshot: Snapshot
    panel: pd.DataFrame
    checks: pd.DataFrame
    n_boot: int
    coverage_status: pd.DataFrame
    coverage_year: pd.DataFrame
    volume_month: pd.DataFrame
    volume_year: pd.DataFrame
    volume_era: pd.DataFrame
    top_campaigns: pd.DataFrame
    clock: pd.DataFrame
    clock_summary: pd.DataFrame
    clock_shape: pd.DataFrame
    time_to_50k: pd.DataFrame
    launch: pd.DataFrame
    dow: pd.DataFrame
    dow_era: pd.DataFrame
    dow_sensitivity: pd.DataFrame
    month: pd.DataFrame
    month_era: pd.DataFrame
    week: pd.DataFrame
    dom: pd.DataFrame
    holiday_pooled: pd.DataFrame
    holiday_named: pd.DataFrame
    interaction: pd.DataFrame
    day_of_year: pd.DataFrame
    samples: pd.DataFrame
    example: pd.DataFrame
    summary: dict
    warnings: list[str] = field(default_factory=list)
    figures: dict[str, str] = field(default_factory=dict)


def _factor(name: str, codes: np.ndarray, labels: list[str], report: bool) -> Factor:
    return Factor(name, np.asarray(codes, dtype=np.int32), labels, report)


def _labels_for(codes: np.ndarray, text) -> list[str]:
    hi = int(np.max(codes)) if len(codes) else 0
    return [text(i) for i in range(hi + 1)]


def _sample(panel: pd.DataFrame, life: pd.Series, *, min_signatures: int, start, end) -> pd.DataFrame:
    sub = panel[(panel["date"] >= start) & (panel["date"] < end) & (panel["age"] >= MIN_AGE)]
    keep = life[life >= min_signatures].index
    sub = sub[sub["id"].isin(keep)]
    if sub.empty:
        return sub
    days = sub.groupby("id", sort=False).size()
    return sub[sub["id"].isin(days[days >= MIN_DAYS].index)].copy()


def within_week(sample: pd.DataFrame, *, n_boot: int, seed: int, drop_holidays: bool = True) -> pd.DataFrame:
    """Equal-campaign and signature-weighted weekday ratios inside 7-day blocks."""
    empty = pd.DataFrame(columns=["dow", "weekday", "n_blocks", "n_campaigns", "equal_mean", "equal_mean_lo",
                                  "equal_mean_hi", "equal_median", "equal_median_lo", "equal_median_hi",
                                  "signature_weighted", "signature_lo", "signature_hi"])
    if sample.empty:
        return empty
    df = sample.loc[sample["age"] >= MIN_AGE, ["id", "age", "dow", "n", "holiday_code"]].copy()
    df["block"] = (df["age"] - MIN_AGE) // 7
    sizes = df.groupby(["id", "block"], sort=False).size()
    full = sizes[sizes == 7].index
    df = df.set_index(["id", "block"])
    df = df.loc[df.index.isin(full)].reset_index()
    if drop_holidays and len(df):
        flagged = df.groupby(["id", "block"], sort=False)["holiday_code"].transform("max") > 0
        df = df.loc[~flagged]
    if df.empty:
        return empty
    block_sum = df.groupby(["id", "block"], sort=False)["n"].transform("sum")
    df = df.loc[block_sum > 0].copy()
    if df.empty:
        return empty
    df["block_sum"] = df.groupby(["id", "block"], sort=False)["n"].transform("sum")
    df["ratio"] = df["n"] / df["block_sum"] * 7
    df["weighted_piece"] = df["ratio"] * df["block_sum"]
    grouped = df.groupby(["id", "dow"], sort=False).agg(
        mean_ratio=("ratio", "mean"),
        weighted_piece=("weighted_piece", "sum"),
        weight=("block_sum", "sum"),
        signatures=("n", "sum"),
    )
    grouped["weighted_ratio"] = grouped["weighted_piece"] / grouped["weight"]
    mean_wide = grouped["mean_ratio"].unstack("dow").reindex(columns=range(7))
    weight_wide = grouped["weighted_ratio"].unstack("dow").reindex(columns=range(7))
    signatures = grouped["signatures"].groupby("id").sum()
    ids = mean_wide.index.to_numpy()
    matrix = mean_wide.to_numpy(dtype=float)
    weighted = weight_wide.to_numpy(dtype=float)
    sig = signatures.reindex(mean_wide.index).to_numpy(dtype=float)

    def summarize(chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.nanmedian(matrix[chosen], axis=0)
        avg = np.nanmean(matrix[chosen], axis=0)
        w = sig[chosen]
        sw = np.nansum(weighted[chosen] * w[:, None], axis=0) / np.nansum(w)
        return med, avg, sw

    point_med, point_avg, point_sw = summarize(np.arange(len(ids)))
    lo_m = hi_m = lo_a = hi_a = lo_s = hi_s = [np.nan] * 7
    if n_boot and len(ids):
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot, 3, 7), dtype=float)
        for draw in range(n_boot):
            chosen = rng.integers(0, len(ids), size=len(ids))
            draws[draw] = np.vstack(summarize(chosen))
        lo_m, hi_m = np.nanpercentile(draws[:, 0], [2.5, 97.5], axis=0)
        lo_a, hi_a = np.nanpercentile(draws[:, 1], [2.5, 97.5], axis=0)
        lo_s, hi_s = np.nanpercentile(draws[:, 2], [2.5, 97.5], axis=0)
    n_blocks = int(df.groupby(["id", "block"], sort=False).ngroups)
    rows = []
    for dow in range(7):
        rows.append(
            {
                "dow": dow,
                "weekday": WEEKDAYS[dow],
                "n_blocks": n_blocks,
                "n_campaigns": int(len(ids)),
                "equal_mean": point_avg[dow],
                "equal_mean_lo": lo_a[dow],
                "equal_mean_hi": hi_a[dow],
                "equal_median": point_med[dow],
                "equal_median_lo": lo_m[dow],
                "equal_median_hi": hi_m[dow],
                "signature_weighted": point_sw[dow],
                "signature_lo": lo_s[dow],
                "signature_hi": hi_s[dow],
            }
        )
    return pd.DataFrame(rows)


def crude_weekday(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame(columns=["dow", "day_share", "signature_share", "crude_ratio"])
    days = sample.groupby("dow").size().reindex(range(7), fill_value=0)
    signatures = sample.groupby("dow")["n"].sum().reindex(range(7), fill_value=0)
    day_share = (days / days.sum()).to_numpy(dtype=float)
    signature_share = (signatures / signatures.sum() if signatures.sum() else signatures).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "dow": list(range(7)),
            "day_share": day_share,
            "signature_share": signature_share,
            "crude_ratio": np.divide(signature_share, day_share, out=np.full(7, np.nan), where=day_share > 0),
        }
    )


def _calendar_factors(sample: pd.DataFrame, which: str) -> list[Factor]:
    """which is 'month', 'week' or 'dom'. Age controls are always included and not reported."""
    age_week = sample["age_week"].to_numpy()
    age_mod = sample["age_mod"].to_numpy()
    dow = sample["dow"].to_numpy()
    holiday = sample["holiday_code"].to_numpy()
    factors = [
        _factor("age_week", age_week, _labels_for(age_week, lambda i: str(i)), False),
        _factor("age_mod", age_mod, WEEKDAYS, False),
        _factor("dow", dow, WEEKDAYS, which == "month"),
        _factor("holiday", holiday, HOLIDAY_LABELS, which == "month"),
    ]
    if which == "month":
        month = sample["month"].to_numpy() - 1
        factors.append(_factor("month", month, MONTHS, True))
    elif which == "week":
        week = sample["iso_week"].to_numpy()
        factors.append(_factor("iso_week", week, _labels_for(week, lambda i: f"{i:02d}"), True))
    elif which == "dom":
        month = sample["month"].to_numpy() - 1
        dom = sample["dom"].to_numpy()
        factors.append(_factor("month", month, MONTHS, False))
        factors.append(_factor("dom", dom, _labels_for(dom, lambda i: str(i)), True))
    else:
        raise ValueError(which)
    return factors


def _fit_sample(sample: pd.DataFrame, which: str, *, n_boot: int, seed: int, name: str, extras=None, with_residual=False) -> Fit | None:
    if sample["id"].nunique() < 30 or len(sample) < 500:
        return None
    y = np.log1p(sample["n"].to_numpy(dtype=np.float64))
    return fit_factors(
        y,
        sample["id"].to_numpy(),
        _calendar_factors(sample, which),
        extras,
        n_boot=n_boot,
        seed=seed,
        name=name,
        with_residual=with_residual,
    )


def _reported(fit: Fit | None, factor: str) -> pd.DataFrame:
    if fit is None or fit.coefficients.empty:
        return pd.DataFrame()
    out = fit.coefficients
    out = out[(out["factor"] == factor) & out["report"]].copy()
    return out.reset_index(drop=True)


def _contrast_table(fit: Fit, factor: str, baseline: str) -> pd.DataFrame:
    """Multiplier of each level versus a baseline level. Intervals use the same draws."""
    frame = fit.coefficients.reset_index(drop=True)
    rows = frame.index[frame["factor"] == factor].tolist()
    by_level = {frame.loc[i, "level"]: i for i in rows}
    if baseline not in by_level:
        return pd.DataFrame()
    base = by_level[baseline]
    records = []
    for level, index in by_level.items():
        if level == baseline:
            continue
        contrast = frame.loc[index, "coef"] - frame.loc[base, "coef"]
        lo = hi = np.nan
        if fit.draws is not None:
            delta = fit.draws[:, index] - fit.draws[:, base]
            lo, hi = np.nanpercentile(delta, [2.5, 97.5])
        records.append(
            {
                "level": level,
                "coef": contrast,
                "lo": lo,
                "hi": hi,
                "multiplier": float(np.exp(contrast)),
                "multiplier_lo": float(np.exp(lo)) if np.isfinite(lo) else np.nan,
                "multiplier_hi": float(np.exp(hi)) if np.isfinite(hi) else np.nan,
                "n": int(frame.loc[index, "n"]),
                "n_campaigns": int(frame.loc[index, "n_campaigns"]),
            }
        )
    return pd.DataFrame(records)


def _order(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_ord"] = frame[column].map({label: i for i, label in enumerate(order)})
    return frame.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def _volume(panel: pd.DataFrame, initiatives: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = panel.copy()
    frame["month_start"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    names = initiatives.set_index("id")["name_fi"]
    monthly_rows = []
    for month, part in frame.groupby("month_start", sort=True):
        by_campaign = part.groupby("id")["n"].sum()
        total = int(by_campaign.sum())
        top_id = int(by_campaign.idxmax()) if total and len(by_campaign) else None
        top = int(by_campaign.max()) if total and len(by_campaign) else 0
        monthly_rows.append(
            {
                "month_start": month,
                "signatures": total,
                "campaign_days": int(len(part)),
                "campaigns": int(part["id"].nunique()),
                "n_dates": int(part["date"].nunique()),
                "per_open_day": total / len(part) if len(part) else np.nan,
                "top_id": top_id,
                "top_name": "" if top_id is None else str(names.get(top_id, "")).replace("\n", " ").strip(),
                "top_signatures": top,
                "top_share": top / total if total else np.nan,
            }
        )
    month = pd.DataFrame(monthly_rows)
    frame["year"] = frame["date"].dt.year
    yearly_rows = []
    for year, part in frame.groupby("year", sort=True):
        by_campaign = part.groupby("id")["n"].sum()
        total = int(by_campaign.sum())
        top_id = int(by_campaign.idxmax()) if total and len(by_campaign) else None
        top = int(by_campaign.max()) if total and len(by_campaign) else 0
        yearly_rows.append(
            {
                "year": int(year),
                "signatures": total,
                "campaign_days": int(len(part)),
                "campaigns": int(part["id"].nunique()),
                "per_open_day": total / len(part) if len(part) else np.nan,
                "top_id": top_id,
                "top_name": "" if top_id is None else str(names.get(top_id, "")).replace("\n", " ").strip(),
                "top_share": top / total if total else np.nan,
            }
        )
    return month, pd.DataFrame(yearly_rows)


def _era_volume(month: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("Through 2016", "2000-01-01", "2017-01-01"),
        ("2017", "2017-01-01", "2018-01-01"),
        ("2018–2025", "2018-01-01", "2026-01-01"),
    ]
    rows = []
    full = month[month["n_dates"] >= 28]
    for label, start, end in windows:
        part = full[(full["month_start"] >= pd.Timestamp(start)) & (full["month_start"] < pd.Timestamp(end))]
        rows.append(
            {
                "era": label,
                "full_months": int(len(part)),
                "signatures": int(part["signatures"].sum()) if len(part) else 0,
                "median_per_open_day": float(part["per_open_day"].median()) if len(part) else np.nan,
                "mean_per_open_day": float(part["per_open_day"].mean()) if len(part) else np.nan,
                "median_top_share": float(part["top_share"].median()) if len(part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cumulative_shares(counts: np.ndarray, length: int, total: float, horizon: int = 183) -> np.ndarray:
    """Share of `total` collected by each age.

    `counts[age]` is that day's signatures. Ages after `length` stay at 1,
    because the recorded window has ended. Ages inside the window use the
    running sum, so a window longer than `horizon` is still short of 1 at the
    right edge.
    """
    out = np.ones(horizon + 1, dtype=float)
    if total <= 0:
        return out
    running = 0.0
    last = min(int(length), horizon)
    for age in range(last + 1):
        if age < len(counts):
            running += float(counts[age])
        out[age] = running / total
    return out


def path_fit(median_by_age: np.ndarray, horizon: int = 183) -> dict[str, float]:
    """Compare a median cumulative path with a square root, a straight line, and a constant hazard."""
    ages = np.arange(7, 177)
    observed = np.asarray(median_by_age, dtype=float)[ages]
    empty = {
        "power_alpha": np.nan, "rmse_sqrt": np.nan, "rmse_linear": np.nan, "rmse_exp": np.nan,
        "sqrt_at_91": float(np.sqrt(91 / horizon)), "linear_at_91": 91 / horizon,
    }
    if np.any(~np.isfinite(observed)) or np.any(observed <= 0):
        return empty
    fraction = ages / horizon
    slope = float(np.sum(np.log(fraction) * np.log(observed)) / np.sum(np.log(fraction) ** 2))
    sqrt_curve = np.sqrt(fraction)
    linear_curve = fraction
    lambdas = np.linspace(0.2, 8.0, 200)
    hazards = np.array([(1 - np.exp(-lam * fraction)) / (1 - np.exp(-lam)) for lam in lambdas])
    rmse = lambda curve: float(np.sqrt(np.mean((observed - curve) ** 2)))
    return {
        "power_alpha": slope,
        "rmse_sqrt": rmse(sqrt_curve),
        "rmse_linear": rmse(linear_curve),
        "rmse_exp": float(min(rmse(curve) for curve in hazards)),
        "sqrt_at_91": float(np.sqrt(91 / horizon)),
        "linear_at_91": 91 / horizon,
    }


def _clock(panel: pd.DataFrame, checks: pd.DataFrame, cutoff: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Median cumulative share by age.

    Three groups. `full_under_50k` and `full_50k` ran at least 170 days.
    `early_50k` reached 50,000 and the recorded window ended before that.
    After a window ends, the share is held at 1.
    """
    meta = checks[checks["usable"] & ~checks["canceled"] & checks["end_date"].notna()].copy()
    if len(meta):
        meta = meta[meta["end_date"].map(lambda day: day is not None and not pd.isna(day) and day <= cutoff)]
    if meta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    meta["length"] = (pd.to_datetime(meta["end_date"]) - pd.to_datetime(meta["start_date"])).dt.days
    meta = meta[meta["series_sum_in_window"] >= 50]
    if meta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sub = panel[panel["id"].isin(meta.index)]
    wide = sub.pivot_table(index="id", columns="age", values="n", aggfunc="sum", fill_value=0)
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    max_age = int(max(wide.columns.max(), 183))
    wide = wide.reindex(columns=range(0, max_age + 1), fill_value=0)
    horizon = 183
    shares = []
    kept = []
    cross_age: dict[int, int] = {}
    cross_share: dict[int, float] = {}
    for initiative in meta.index:
        if initiative not in wide.index:
            continue
        counts = wide.loc[initiative].to_numpy(dtype=float)
        length = int(meta.loc[initiative, "length"])
        total = float(meta.loc[initiative, "series_sum_in_window"])
        shares.append(cumulative_shares(counts, length, total, horizon))
        kept.append(initiative)
        running = 0.0
        limit = min(length, len(counts) - 1)
        for age in range(limit + 1):
            running += float(counts[age])
            if running >= THRESHOLD:
                cross_age[initiative] = age
                cross_share[initiative] = running / total if total else np.nan
                break
    if not kept:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    share = pd.DataFrame(shares, index=kept)
    meta = meta.loc[kept]
    group_of = {
        "full_under_50k": (meta["length"] >= 170) & (meta["series_sum_in_window"] < THRESHOLD),
        "full_50k": (meta["length"] >= 170) & (meta["series_sum_in_window"] >= THRESHOLD),
        "early_50k": (meta["length"] < 170) & (meta["series_sum_in_window"] >= THRESHOLD),
    }
    curve_rows = []
    summary_rows = []
    shape_rows = []
    for name, mask in group_of.items():
        ids = meta.index[mask.to_numpy()]
        if len(ids) == 0:
            continue
        band = share.loc[ids]
        for age in range(horizon + 1):
            curve_rows.append(
                {
                    "group": name,
                    "age": age,
                    "p25": float(band[age].quantile(0.25)),
                    "p50": float(band[age].quantile(0.50)),
                    "p75": float(band[age].quantile(0.75)),
                    "n": int(len(ids)),
                }
            )
        summary_rows.append(
            {
                "group": name, "piece": "Day 0", "n": int(len(ids)),
                "p25": float(band[0].quantile(0.25)), "p50": float(band[0].quantile(0.50)),
                "p75": float(band[0].quantile(0.75)),
            }
        )
        summary_rows.append(
            {
                "group": name, "piece": "Days 0–6", "n": int(len(ids)),
                "p25": float(band[6].quantile(0.25)), "p50": float(band[6].quantile(0.50)),
                "p75": float(band[6].quantile(0.75)),
            }
        )
        half = [
            float(band.loc[initiative, min(horizon, int(round(0.5 * meta.loc[initiative, "length"])))])
            for initiative in ids
        ]
        fit = path_fit(band.median(axis=0).to_numpy(), horizon) if name == "full_under_50k" else {
            "power_alpha": np.nan, "rmse_sqrt": np.nan, "rmse_linear": np.nan, "rmse_exp": np.nan,
            "sqrt_at_91": float(np.sqrt(91 / horizon)), "linear_at_91": 91 / horizon,
        }
        first_week = band[6]
        slow_at = fast_at = np.nan
        if name == "full_under_50k" and len(ids) >= 40:
            q1, q3 = first_week.quantile(0.25), first_week.quantile(0.75)
            slow_at = float(band.loc[first_week <= q1, 91].median())
            fast_at = float(band.loc[first_week >= q3, 91].median())
        crossed = [initiative for initiative in ids if initiative in cross_age]
        shape_rows.append(
            {
                "group": name,
                "n": int(len(ids)),
                "median_length": float(meta.loc[ids, "length"].median()),
                "share_d0": float(band[0].median()),
                "share_d7": float(band[7].median()),
                "share_d14": float(band[13].median()),
                "share_d30": float(band[30].median()),
                "share_d91": float(band[91].median()),
                "share_d91_p25": float(band[91].quantile(0.25)),
                "share_d91_p75": float(band[91].quantile(0.75)),
                "share_d150": float(band[150].median()),
                "share_half_own": float(np.median(half)),
                "first14": float(band[13].median()),
                "second14": float((band[27] - band[13]).median()),
                "last14": float((band[183] - band[169]).median()),
                "slow_start_d91": slow_at,
                "fast_start_d91": fast_at,
                "cross_n": int(len(crossed)),
                "cross_p50": float(np.median([cross_age[i] for i in crossed])) if crossed else np.nan,
                "share_at_cross": float(np.median([cross_share[i] for i in crossed])) if crossed else np.nan,
                **fit,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows), pd.DataFrame(shape_rows)


def _time_to_50k(panel: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    ids = checks.index[checks["usable"] & ~checks["canceled"]]
    sub = panel[panel["id"].isin(ids)]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index="id", columns="age", values="n", aggfunc="sum", fill_value=0)
    ages = np.array(sorted(wide.columns))
    cum = wide.cumsum(axis=1)
    hit = cum.to_numpy() >= THRESHOLD
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, ages[hit.argmax(axis=1)], np.nan)
    reached = pd.Series(first, index=wide.index).dropna()
    web_ids = set(checks.index[checks["series_sum_in_window"] >= THRESHOLD])
    total_ids = set(checks.index[checks["total_support_count"] >= THRESHOLD])
    paper_only = total_ids - web_ids
    if reached.empty:
        stats = {"n_web": 0}
    else:
        values = reached.to_numpy()
        stats = {
            "n_web": int(len(values)),
            "p10_age": float(np.percentile(values, 10)),
            "p25_age": float(np.percentile(values, 25)),
            "p50_age": float(np.percentile(values, 50)),
            "p75_age": float(np.percentile(values, 75)),
            "p90_age": float(np.percentile(values, 90)),
            "share_on_day_0": float(np.mean(values <= 0)),
            "share_within_7_days": float(np.mean(values <= 6)),
            "share_within_30_days": float(np.mean(values <= 29)),
        }
    stats["n_total_50k_without_50k_web"] = int(len(paper_only))
    stats["n_total_50k"] = int(len(total_ids))
    return pd.DataFrame([stats])


def _launch(initiatives: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    frame = initiatives[initiatives["start_date"].notna()].copy()
    frame = frame[frame["start_date"].map(lambda day: day <= cutoff)]
    if frame.empty:
        return frame
    frame["dow"] = pd.to_datetime(frame["start_date"]).dt.day_of_week
    rows = []
    for dow, part in frame.groupby("dow"):
        rows.append(
            {
                "dow": int(dow),
                "weekday": WEEKDAYS[int(dow)],
                "n_starts": int(len(part)),
                "share": len(part) / len(frame),
                "median_web": float(part["support_count"].median()),
                "median_total": float(part["total_support_count"].median()),
            }
        )
    return _order(pd.DataFrame(rows), "weekday", WEEKDAYS)


def _gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0 or v.sum() <= 0:
        return float("nan")
    index = np.arange(1, len(v) + 1)
    return float((2 * np.sum(index * v) / (len(v) * v.sum())) - (len(v) + 1) / len(v))


def _day_of_year(sample: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Descriptive smooth. Month is left in the residual on purpose. No bootstrap."""
    del n_boot
    if sample["id"].nunique() < 30:
        return pd.DataFrame()
    y = np.log1p(sample["n"].to_numpy(dtype=np.float64))
    factors = _calendar_factors(sample, "month")
    factors = [factor for factor in factors if factor.name != "month"]
    # Keep holiday in, so Christmas is not the entire December shape. Drop month.
    fit = fit_factors(y, sample["id"].to_numpy(), factors, n_boot=0, with_residual=True, name="doy")
    if fit.residual is None:
        return pd.DataFrame()
    frame = pd.DataFrame({"doy": sample["doy"].to_numpy(), "residual": fit.residual})
    means = frame.groupby("doy")["residual"].mean()
    counts = frame.groupby("doy").size()
    full = means.reindex(range(1, 367))
    if np.isnan(full.loc[366]):
        full.loc[366] = np.nanmean([full.loc[365], full.loc[1]])
    values = full.to_numpy(dtype=float)
    half = 7
    tiled = np.concatenate([values[-half:], values, values[:half]])
    smooth = np.convolve(tiled, np.ones(15) / 15, mode="valid")
    return pd.DataFrame(
        {
            "doy": np.arange(1, 367),
            "residual": values,
            "multiplier": np.exp(values),
            "smooth": smooth,
            "smooth_multiplier": np.exp(smooth),
            "n": counts.reindex(range(1, 367)).fillna(0).to_numpy(),
        }
    )


def _top(initiatives: pd.DataFrame) -> pd.DataFrame:
    frame = initiatives.sort_values("total_support_count", ascending=False).head(12).copy()
    total_web = initiatives["support_count"].sum()
    frame["web_share"] = frame["support_count"] / total_web if total_web else np.nan
    frame["name"] = frame["name_fi"].map(lambda value: str(value).replace("\n", " ").strip())
    return frame[
        ["id", "name", "start_date", "end_date", "state", "support_count", "external_support_count",
         "total_support_count", "web_share"]
    ].reset_index(drop=True)


def _coverage(checks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = (
        checks.groupby("status", dropna=False)
        .agg(campaigns=("id", "size"), web_counter=("support_count", "sum"), series_sum=("series_sum", "sum"),
             residual=("residual", "sum"))
        .reset_index()
    )
    years = checks[checks["start_date"].notna()].copy()
    years["start_year"] = pd.to_datetime(years["start_date"]).dt.year
    rows = []
    for year, part in years.groupby("start_year"):
        rows.append(
            {
                "start_year": int(year),
                "campaigns": int(len(part)),
                "usable": int(part["usable"].sum()),
                "missing_series": int((part["status"] == "missing_series").sum()),
                "sum_mismatch": int((part["status"] == "sum_mismatch").sum()),
                "canceled": int(part["canceled"].sum()),
                "web_counter": int(part["support_count"].sum()),
                "series_sum": int(part["series_sum"].sum()),
                "residual": int(part["residual"].sum()),
            }
        )
    return status, pd.DataFrame(rows)


def _remember(fit: Fit | None, bag: list[str]) -> None:
    if fit is None:
        return
    for warning in fit.warnings:
        bag.append(f"{fit.name}: {warning}")


def build_report(root, stamp: str | None = None, n_boot: int | None = None) -> Report:
    snapshot = load_snapshot(root, stamp)
    n_boot = boot_count() if n_boot is None else n_boot
    panel, checks = build_panel(snapshot.initiatives, snapshot.daily, snapshot.cutoff)
    panel["date"] = pd.to_datetime(panel["date"])
    panel.attrs["cutoff"] = snapshot.cutoff
    analytical = panel[panel["id"].isin(checks.index[checks["usable"] & ~checks["canceled"]])].copy()
    warnings: list[str] = []
    coverage_status, coverage_year = _coverage(checks)
    volume_month, volume_year = _volume(analytical, snapshot.initiatives)
    volume_era = _era_volume(volume_month)
    life = analytical.groupby("id")["n"].sum()
    post = _sample(analytical, life, min_signatures=MIN_SIGNATURES, start=BREAK, end=pd.Timestamp("2100-01-01"))
    pre = _sample(analytical, life, min_signatures=MIN_SIGNATURES, start=pd.Timestamp("2000-01-01"), end=BREAK)
    post_30 = _sample(analytical, life, min_signatures=30, start=BREAK, end=pd.Timestamp("2100-01-01"))
    post_1000 = _sample(analytical, life, min_signatures=1000, start=BREAK, end=pd.Timestamp("2100-01-01"))

    dow_post = within_week(post, n_boot=n_boot, seed=10)
    dow_pre = within_week(pre, n_boot=n_boot, seed=11)
    crude = crude_weekday(post)
    headline = _fit_sample(post, "month", n_boot=n_boot, seed=20, name="post-2017 month")
    pre_fit = _fit_sample(pre, "month", n_boot=max(n_boot // 2, 1) if n_boot else 0, seed=21, name="pre-2017 month")
    week_fit = _fit_sample(post, "week", n_boot=n_boot, seed=22, name="post-2017 week")
    dom_fit = _fit_sample(post, "dom", n_boot=n_boot, seed=23, name="post-2017 day-of-month")
    interaction_fit = None
    if not post.empty and post["id"].nunique() >= 30:
        interaction_fit = _fit_sample(
            post, "month", n_boot=n_boot, seed=24, name="summer-weekend",
            extras=[Extra("summer_weekend", post["summer_weekend"].to_numpy())],
        )
    named = None
    if not post.empty and post["id"].nunique() >= 30:
        # Holiday names stay in the residual. Weekday and month are controlled.
        factors = [factor for factor in _calendar_factors(post, "month") if factor.name != "holiday"]
        group_labels = {}
        codes = np.full(len(post), -1, dtype=np.int32)
        for code, label in enumerate(NAMED_ORDER, start=1):
            group_labels[code] = label
            codes[post["holiday_name"].to_numpy() == label] = code
        named = residual_group_means(
            np.log1p(post["n"].to_numpy(dtype=float)),
            post["id"].to_numpy(),
            factors,
            codes,
            group_labels,
            n_boot=n_boot,
            seed=25,
            name="named-days",
        )
    for fit in (headline, pre_fit, week_fit, dom_fit, interaction_fit, named):
        _remember(fit, warnings)
    if headline is None:
        warnings.append("post-2017 calendar model was not fit; the sample is too small")

    dow = dow_post.merge(crude, on="dow", how="left")
    fe = _reported(headline, "dow").rename(columns={
        "level": "weekday",
        "multiplier": "model_multiplier",
        "multiplier_lo": "model_lo",
        "multiplier_hi": "model_hi",
    })
    if not fe.empty:
        dow = dow.merge(fe[["weekday", "model_multiplier", "model_lo", "model_hi"]], on="weekday", how="left")
    dow = _order(dow, "weekday", WEEKDAYS)

    def era_rows(frame: pd.DataFrame, era: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame[["weekday", "equal_median", "equal_median_lo", "equal_median_hi", "n_campaigns", "n_blocks"]].copy()
        out.insert(0, "era", era)
        return out

    dow_era = pd.concat([era_rows(dow_post, "2017 onward"), era_rows(dow_pre, "Through 2016")], ignore_index=True)

    sensitivity = []
    for label, frame, drop in (
        ("30 web signatures, holidays out", post_30, True),
        ("100 web signatures, holidays kept", post, False),
        ("1,000 web signatures, holidays out", post_1000, True),
    ):
        table = within_week(frame, n_boot=0, seed=0, drop_holidays=drop)
        if table.empty:
            continue
        table = table[["weekday", "equal_median", "signature_weighted", "n_campaigns"]].copy()
        table.insert(0, "sample", label)
        sensitivity.append(table)
    dow_sensitivity = pd.concat(sensitivity, ignore_index=True) if sensitivity else pd.DataFrame()

    month = _order(_reported(headline, "month"), "level", MONTHS)
    month_pre = _order(_reported(pre_fit, "month"), "level", MONTHS)
    if not month.empty:
        month = month.assign(era="2017 onward")
    if not month_pre.empty:
        month_pre = month_pre.assign(era="Through 2016")
    month_era = pd.concat([month, month_pre], ignore_index=True)
    week = _reported(week_fit, "iso_week")
    if not week.empty:
        week = week.sort_values("level").reset_index(drop=True)
    dom = _reported(dom_fit, "dom")
    if not dom.empty:
        dom = dom[dom["level"] != "0"].copy()
        dom["_n"] = dom["level"].astype(int)
        dom = dom.sort_values("_n").drop(columns="_n").reset_index(drop=True)
    holiday_pooled = _contrast_table(headline, "holiday", "Ordinary day") if headline is not None else pd.DataFrame()
    holiday_named = named.coefficients.copy() if named is not None else pd.DataFrame()
    if not holiday_named.empty:
        holiday_named["label"] = holiday_named["level"].map(label_fi)
        holiday_named = _order(holiday_named, "level", list(NAMED_ORDER))
    interaction = _reported(interaction_fit, "summer_weekend") if interaction_fit is not None else pd.DataFrame()
    day_of_year = _day_of_year(post, n_boot)
    clock, clock_summary, clock_shape = _clock(analytical, checks, snapshot.cutoff)
    samples = pd.DataFrame(
        [
            {"sample": "2017 onward, age ≥ 7, ≥ 100 web signatures", "campaigns": int(post["id"].nunique()) if len(post) else 0,
             "campaign_days": int(len(post))},
            {"sample": "Through 2016, age ≥ 7, ≥ 100 web signatures", "campaigns": int(pre["id"].nunique()) if len(pre) else 0,
             "campaign_days": int(len(pre))},
            {"sample": "Open campaign-days in the zero-filled panel", "campaigns": int(analytical["id"].nunique()) if len(analytical) else 0,
             "campaign_days": int(len(analytical))},
        ]
    )
    example_id = 17951 if 17951 in set(checks["id"]) else None
    if example_id is None and len(snapshot.daily):
        example_id = int(snapshot.daily["id"].iloc[-1])
    example = snapshot.daily[snapshot.daily["id"] == example_id].copy() if example_id is not None else pd.DataFrame()
    usable_web = int(checks.loc[checks["usable"], "series_sum"].sum())
    started = checks["start_date"].map(lambda day: day is not None and not pd.isna(day) and day <= snapshot.cutoff)
    started_web = checks.loc[started, "support_count"]
    summary = {
        "stamp": snapshot.stamp,
        "fetched_at": snapshot.fetched_at.isoformat(),
        "cutoff": snapshot.cutoff.isoformat(),
        "campaigns": int(len(checks)),
        "usable_campaigns": int(checks["usable"].sum()),
        "web_counter": int(checks["support_count"].sum()),
        "series_sum": int(checks["series_sum"].sum()),
        "external": int(checks["external_support_count"].sum()),
        "usable_series_sum": usable_web,
        "mismatch_campaigns": int((checks["status"] == "sum_mismatch").sum()),
        "mismatch_residual": int(checks.loc[checks["status"] == "sum_mismatch", "residual"].sum()),
        "mismatch_web": int(checks.loc[checks["status"] == "sum_mismatch", "support_count"].sum()),
        "gap_campaigns": int((checks["status"] == "ok_gap").sum()),
        "gap_residual": int(checks.loc[checks["status"] == "ok_gap", "residual"].sum()),
        "missing_series_campaigns": int((checks["status"] == "missing_series").sum()),
        "missing_series_web": int(checks.loc[checks["status"] == "missing_series", "support_count"].sum()),
        "canceled": int(checks["canceled"].sum()),
        "gini_web": _gini(started_web.to_numpy()),
        "started_campaigns": int(len(started_web)),
        "share_campaigns_under_100": float((started_web < 100).mean()) if len(started_web) else float("nan"),
        "top10_web_share": float(started_web.sort_values(ascending=False).head(10).sum() / started_web.sum()) if started_web.sum() else float("nan"),
        "top1pct_web_share": float(
            started_web.sort_values(ascending=False).head(max(1, round(0.01 * len(started_web)))).sum() / started_web.sum()
        ) if started_web.sum() else float("nan"),
        "series_outside": int(checks["series_outside_window"].sum()),
        "example_id": example_id,
        "example_support": None if example_id is None else int(checks.loc[checks["id"] == example_id, "support_count"].iloc[0]),
        "example_series": None if example_id is None else int(checks.loc[checks["id"] == example_id, "series_sum"].iloc[0]),
        "example_residual": None if example_id is None else int(checks.loc[checks["id"] == example_id, "residual"].iloc[0]),
        "n_boot": n_boot,
        "min_signatures": MIN_SIGNATURES,
        "min_age": MIN_AGE,
    }
    if len(volume_month):
        peak = volume_month.loc[volume_month["signatures"].idxmax()]
        summary["peak_month"] = str(pd.Timestamp(peak["month_start"]).date())
        summary["peak_signatures"] = int(peak["signatures"])
        summary["peak_top_name"] = peak["top_name"]
        summary["peak_top_share"] = float(peak["top_share"])
    return Report(
        snapshot=snapshot,
        panel=analytical,
        checks=checks,
        n_boot=n_boot,
        coverage_status=coverage_status,
        coverage_year=coverage_year,
        volume_month=volume_month,
        volume_year=volume_year,
        volume_era=volume_era,
        top_campaigns=_top(snapshot.initiatives),
        clock=clock,
        clock_summary=clock_summary,
        clock_shape=clock_shape,
        time_to_50k=_time_to_50k(analytical, checks),
        launch=_launch(snapshot.initiatives, snapshot.cutoff),
        dow=dow,
        dow_era=dow_era,
        dow_sensitivity=dow_sensitivity,
        month=month,
        month_era=month_era,
        week=week,
        dom=dom,
        holiday_pooled=holiday_pooled,
        holiday_named=holiday_named,
        interaction=interaction,
        day_of_year=day_of_year,
        samples=samples,
        example=example,
        summary=summary,
        warnings=warnings,
    )
