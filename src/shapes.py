"""Reference curves for the median cumulative path.

The fits are descriptive. They are computed from the report's clock, and,
for the individual gaps, from the same finished windows the clock uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import THRESHOLD, cumulative_shares

HORIZON = 183


def _rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - fitted) ** 2)))


def log_series(offset: float, horizon: int = HORIZON) -> np.ndarray:
    ages = np.arange(horizon + 1, dtype=float)
    return np.log1p(ages / offset) / np.log1p(horizon / offset)


def fit_median(median_by_age: np.ndarray, horizon: int = HORIZON) -> dict[str, float] | None:
    """Absolute-error fits on ages 7 through 176, each curve pinned at 0 and at the horizon."""
    median_by_age = np.asarray(median_by_age, dtype=float)
    if median_by_age.shape[0] <= horizon:
        return None
    ages = np.arange(7, 177)
    observed = median_by_age[ages]
    if np.any(~np.isfinite(observed)) or np.any(observed <= 0):
        return None
    fraction = ages / horizon
    sqrt_curve = np.sqrt(fraction)
    linear_curve = fraction

    powers = np.linspace(0.25, 1.2, 381)
    power_gaps = [(_rmse(observed, fraction ** power), float(power)) for power in powers]
    power_gap, power_p = min(power_gaps)

    lambdas = np.linspace(0.2, 8.0, 400)
    exp_gaps = []
    for lam in lambdas:
        fitted = (1 - np.exp(-lam * fraction)) / (1 - np.exp(-lam))
        exp_gaps.append((_rmse(observed, fitted), float(lam)))
    exp_gap, lam = min(exp_gaps)
    tau = horizon / lam

    offsets = np.geomspace(0.5, 120.0, 500)
    log_gaps = []
    for offset in offsets:
        fitted = np.log1p(ages / offset) / np.log1p(horizon / offset)
        log_gaps.append((_rmse(observed, fitted), float(offset)))
    log_gap, offset = min(log_gaps)
    full = log_series(offset, horizon)
    from_day14 = np.max(np.abs(median_by_age[14:177] - full[14:177]))
    return {
        "log_offset": offset,
        "log_doubling": float(np.log(2) / np.log1p(horizon / offset)),
        "rmse_log": log_gap,
        "rmse_exp": exp_gap,
        "rmse_sqrt": _rmse(observed, sqrt_curve),
        "rmse_linear": _rmse(observed, linear_curve),
        "rmse_power": power_gap,
        "power_p": power_p,
        "exp_half_life_days": float(tau * np.log(2)),
        "log_abs_from_day14": float(from_day14),
    }


def _windows(report) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    panel = getattr(report, "panel", None)
    checks = getattr(report, "checks", None)
    if panel is None or checks is None or len(panel) == 0 or len(checks) == 0:
        return None
    if "series_sum_in_window" not in checks.columns or "end_date" not in checks.columns:
        return None
    meta = checks[checks["usable"] & ~checks["canceled"] & checks["end_date"].notna()].copy()
    cutoff = report.snapshot.cutoff
    meta = meta[meta["end_date"].map(lambda day: day is not None and not pd.isna(day) and day <= cutoff)]
    if meta.empty:
        return None
    meta["length"] = (pd.to_datetime(meta["end_date"]) - pd.to_datetime(meta["start_date"])).dt.days
    meta = meta[meta["series_sum_in_window"] >= 50]
    sub = panel[panel["id"].isin(meta.index)]
    if sub.empty:
        return None
    wide = sub.pivot_table(index="id", columns="age", values="n", aggfunc="sum", fill_value=0)
    wide = wide.reindex(columns=range(0, int(max(wide.columns.max(), HORIZON)) + 1), fill_value=0)
    rows = []
    kept = []
    for initiative in meta.index:
        if initiative not in wide.index:
            continue
        counts = wide.loc[initiative].to_numpy(dtype=float)
        length = int(meta.loc[initiative, "length"])
        total = float(meta.loc[initiative, "series_sum_in_window"])
        rows.append(cumulative_shares(counts, length, total, HORIZON))
        kept.append(initiative)
    if not kept:
        return None
    return pd.DataFrame(rows, index=kept), meta.loc[kept]


def _best_gaps(paths: np.ndarray, ages: np.ndarray, family: str) -> np.ndarray:
    best = np.full(paths.shape[0], np.inf)
    if family == "log":
        grid = np.geomspace(1.0, 80.0, 80)
        for offset in grid:
            fitted = np.log1p(ages / offset) / np.log1p(HORIZON / offset)
            best = np.minimum(best, np.sqrt(np.mean((paths - fitted) ** 2, axis=1)))
    else:
        fraction = ages / HORIZON
        for lam in np.linspace(0.4, 8.0, 40):
            fitted = (1 - np.exp(-lam * fraction)) / (1 - np.exp(-lam))
            best = np.minimum(best, np.sqrt(np.mean((paths - fitted) ** 2, axis=1)))
    return best


def describe(report) -> dict[str, float]:
    """Numbers the clock prose needs. Missing pieces are simply absent."""
    notes: dict[str, float] = {}
    clock = getattr(report, "clock", None)
    if clock is not None and len(clock):
        ordinary = clock[clock["group"] == "full_under_50k"].sort_values("age")
        if len(ordinary):
            idx = ordinary["age"].to_numpy(dtype=int)
            if set(range(HORIZON + 1)).issubset(set(idx.tolist())):
                median = np.full(int(idx.max()) + 1, np.nan)
                median[idx] = ordinary["p50"].to_numpy(dtype=float)
                fitted = fit_median(median, HORIZON)
                if fitted:
                    notes.update(fitted)
            level = ordinary.drop_duplicates("age").set_index("age")["p50"]
            for age in (7, 30, 90, 150):
                if age in level.index and (age - 1) in level.index:
                    notes[f"median_slope_{age}"] = float(level.loc[age] - level.loc[age - 1])
        blue = clock[clock["group"] == "full_50k"]
        for age, key in ((25, "blue_day25"), (100, "blue_day100"), (150, "blue_day150"), (160, "blue_day160")):
            hit = blue[blue["age"] == age]
            if len(hit):
                notes[key] = float(hit["p50"].iloc[0])
        if "blue_day150" in notes and "blue_day160" in notes:
            notes["blue_path_150_160"] = notes["blue_day160"] - notes["blue_day150"]

    built = _windows(report)
    if built is None or "log_offset" not in notes:
        return notes
    share, meta = built
    ordinary_ids = meta.index[(meta["length"] >= 170) & (meta["series_sum_in_window"] < THRESHOLD)]
    ordinary_ids = ordinary_ids.intersection(share.index)
    ages = np.arange(7, 177)
    paths = share.loc[ordinary_ids, ages].to_numpy(dtype=float)
    if len(paths):
        offset = notes["log_offset"]
        common = np.log1p(ages / offset) / np.log1p(HORIZON / offset)
        sqrt_curve = np.sqrt(ages / HORIZON)
        notes["common_log_p50"] = float(np.median(np.sqrt(np.mean((paths - common) ** 2, axis=1))))
        notes["sqrt_individual_p50"] = float(np.median(np.sqrt(np.mean((paths - sqrt_curve) ** 2, axis=1))))
        notes["own_log_p50"] = float(np.median(_best_gaps(paths, ages, "log")))
        notes["own_exp_p50"] = float(np.median(_best_gaps(paths, ages, "exp")))
        daily = share.loc[ordinary_ids].diff(axis=1)
        for age in (7, 30, 90, 150):
            notes[f"median_daily_{age}"] = float(daily[age].median())
    late_ids = meta.index[(meta["length"] >= 170) & (meta["series_sum_in_window"] >= THRESHOLD)]
    late_ids = late_ids.intersection(share.index)
    if len(late_ids):
        notes["blue_own_150_160"] = float((share.loc[late_ids, 160] - share.loc[late_ids, 150]).median())
    return notes
