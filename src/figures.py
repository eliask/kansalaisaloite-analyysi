"""Figures for the seasonality report. One claim per file."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import MONTHS
from .holidays import NAMES_FI
from .shapes import fit_median, log_series

INK = "#1f4e79"
CLAY = "#b35c1e"
GREEN = "#2c6e49"
BAND = "#9eb6cf"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_FI = {
    "Monday": "ma", "Tuesday": "ti", "Wednesday": "ke", "Thursday": "to",
    "Friday": "pe", "Saturday": "la", "Sunday": "su",
}
MONTH_FI = {
    "January": "tammi", "February": "helmi", "March": "maalis", "April": "huhti",
    "May": "touko", "June": "kesä", "July": "heinä", "August": "elo",
    "September": "syys", "October": "loka", "November": "marras", "December": "joulu",
}
MONTH_ABBR_FI = ["tammi", "helmi", "maalis", "huhti", "touko", "kesä", "heinä", "elo", "syys", "loka", "marras", "joulu"]
ERA_FI = {"Through 2016": "vuoteen 2016", "2017 onward": "vuodesta 2017"}

PHRASE = {
    "en": {
        "vol_sum": "All open initiatives, summed",
        "vol_y": "Web signatures",
        "vol_per": "Per open initiative-day",
        "vol_y2": "Signatures per open initiative-day",
        "month_x": "Month",
        "clock_left": "Under 50,000, open for six months",
        "clock_right": "Reached 50,000",
        "clock_x": "Days after the start date",
        "clock_y": "Share of the initiative's web signatures",
        "sqrt": "Square root of time",
        "log": "Logarithm",
        "linear": "Constant daily rate",
        "median": "Median",
        "stayed": "Stayed open six months",
        "closed": "Closed before day 170",
        "dow_y": "Times a fair share of that week",
        "dow_title": "Weekday after day 7, holiday weeks left out",
        "median_week": "Median initiative-week",
        "weighted": "Signature-weighted",
        "era_y": "Median times a fair share",
        "era_title": "The same weekday contrast in both eras",
        "month_era_y": "Times the average open day, on 1 + count",
        "month_era_title": "Month in both eras, initiative and age held fixed",
        "holiday_x": "Times the model's ordinary day of that weekday and month",
        "holiday_title": "Named days, 2017 onward",
        "doy_y": "Times the average open day",
        "doy_title": "Day of year, 15-day smooth, initiative and weekday removed",
        "month_title": "Month, 2017 onward, initiative and age held fixed",
        "week_title": "ISO week, 2017 onward",
        "dom_title": "Day of month, holding the month fixed",
        "mult_y": "Times the average open day, on 1 + count",
    },
    "fi": {
        "vol_sum": "Kaikki avoimet aloitteet yhteensä",
        "vol_y": "Verkkoallekirjoituksia",
        "vol_per": "Avointa aloitepäivää kohti",
        "vol_y2": "Allekirjoituksia avointa aloitepäivää kohti",
        "month_x": "Kuukausi",
        "clock_left": "Alle 50 000, auki puoli vuotta",
        "clock_right": "Ylitti 50 000",
        "clock_x": "Päiviä aloituksesta",
        "clock_y": "Osuus aloitteen verkkoallekirjoituksista",
        "sqrt": "Ajan neliöjuuri",
        "log": "Logaritmi",
        "linear": "Tasainen päivävauhti",
        "median": "Mediaani",
        "stayed": "Auki koko puoli vuotta",
        "closed": "Sulkeutui ennen päivää 170",
        "dow_y": "Kerroin oman viikon tasajakoon",
        "dow_title": "Viikonpäivä päivästä 7, pyhäviikot pois",
        "median_week": "Mediaaniviikko",
        "weighted": "Allekirjoituksilla painotettu",
        "era_y": "Mediaani, oman viikon tasajako",
        "era_title": "Sama viikonpäiväero molemmissa jaksoissa",
        "month_era_y": "Kerroin tavalliseen avoimeen päivään, 1 + lukumäärä",
        "month_era_title": "Kuukausi molemmissa jaksoissa, aloite ja ikä vakioitu",
        "holiday_x": "Kerroin siihen päivään, jonka viikonpäivä ja kuukausi jo ennustavat",
        "holiday_title": "Nimetyt päivät, vuodesta 2017",
        "doy_y": "Kerroin tavalliseen avoimeen päivään",
        "doy_title": "Vuodenpäivä, 15 päivän tasoitus, aloite ja viikonpäivä poistettu",
        "month_title": "Kuukausi vuodesta 2017, aloite ja ikä vakioitu",
        "week_title": "ISO-viikko vuodesta 2017",
        "dom_title": "Kuukaudenpäivä, kuukausi vakioitu",
        "mult_y": "Kerroin tavalliseen avoimeen päivään, 1 + lukumäärä",
    },
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 12,
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _hline(ax):
    ax.axhline(1, color="#888888", linewidth=0.8, zorder=0)


def _t(lang: str, key: str) -> str:
    return PHRASE.get(lang, PHRASE["en"])[key]


def _ticks(labels, lang: str):
    if lang != "fi":
        return list(labels)
    return [MONTH_FI.get(label, WEEKDAY_FI.get(label, label)) for label in labels]


def volume(month: pd.DataFrame, path: Path, lang: str = "en") -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    x = pd.to_datetime(month["month_start"])
    axes[0].plot(x, month["signatures"], color=INK, linewidth=1.2)
    axes[0].set_ylabel(_t(lang, "vol_y"))
    axes[0].set_title(_t(lang, "vol_sum"))
    axes[1].plot(x, month["per_open_day"], color=CLAY, linewidth=1.2)
    axes[1].set_ylabel(_t(lang, "vol_y2"))
    axes[1].set_title(_t(lang, "vol_per"))
    for ax in axes:
        ax.axvline(pd.Timestamp("2017-01-01"), color="#444444", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel(_t(lang, "month_x"))
    _save(fig, path)


def clock(curve: pd.DataFrame, path: Path, lang: str = "en") -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    ages = np.arange(0, 184)
    left, right = axes
    ordinary = curve[curve["group"] == "full_under_50k"].sort_values("age")
    if len(ordinary):
        left.fill_between(ordinary["age"], ordinary["p25"], ordinary["p75"], color=CLAY, alpha=0.18, linewidth=0)
        left.plot(ordinary["age"], ordinary["p50"], color=CLAY, linewidth=1.8, label=_t(lang, "median"))
        idx = ordinary["age"].to_numpy(dtype=int)
        inside = (idx >= 0) & (idx <= 183)
        if inside.any():
            median = np.full(184, np.nan)
            median[idx[inside]] = ordinary["p50"].to_numpy(dtype=float)[inside]
            fitted = fit_median(median)
            if fitted is not None:
                left.plot(
                    ages, log_series(fitted["log_offset"]),
                    color="#222222", linewidth=1.15, linestyle="-.", label=_t(lang, "log"),
                )
    left.plot(ages, np.sqrt(ages / 183), color="#666666", linewidth=1.0, linestyle="--", label=_t(lang, "sqrt"))
    left.plot(ages, ages / 183, color="#666666", linewidth=1.0, linestyle=":", label=_t(lang, "linear"))
    stayed = curve[curve["group"] == "full_50k"].sort_values("age")
    closed = curve[curve["group"] == "early_50k"].sort_values("age")
    if len(stayed):
        right.fill_between(stayed["age"], stayed["p25"], stayed["p75"], color=INK, alpha=0.15, linewidth=0)
        right.plot(stayed["age"], stayed["p50"], color=INK, linewidth=1.8, label=_t(lang, "stayed"))
    if len(closed):
        right.plot(closed["age"], closed["p50"], color=GREEN, linewidth=1.8, label=_t(lang, "closed"))
    for ax, title in ((left, "clock_left"), (right, "clock_right")):
        ax.set_xlim(0, 183)
        ax.set_ylim(0, 1)
        ax.set_xlabel(_t(lang, "clock_x"))
        ax.set_title(_t(lang, title))
    left.legend(frameon=False, fontsize=8, loc="upper left")
    right.legend(frameon=False, fontsize=8, loc="lower right")
    left.set_ylabel(_t(lang, "clock_y"))
    _save(fig, path)


def _whisker(estimate, low, high):
    estimate = np.asarray(estimate, dtype=float)
    low = np.minimum(np.asarray(low, dtype=float), estimate)
    high = np.maximum(np.asarray(high, dtype=float), estimate)
    return [estimate - low, high - estimate]


def dow(table: pd.DataFrame, path: Path, lang: str = "en") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(table))
    ax.errorbar(
        x - 0.08,
        table["equal_median"],
        yerr=_whisker(table["equal_median"], table["equal_median_lo"], table["equal_median_hi"]),
        fmt="o",
        color=INK,
        capsize=3,
        label=_t(lang, "median_week"),
    )
    ax.scatter(x + 0.08, table["signature_weighted"], marker="D", color=CLAY, label=_t(lang, "weighted"), zorder=3)
    _hline(ax)
    ax.set_xticks(x, _ticks(table["weekday"], lang))
    ax.set_ylabel(_t(lang, "dow_y"))
    ax.set_title(_t(lang, "dow_title"))
    ax.legend(frameon=False)
    _save(fig, path)


def dow_era(table: pd.DataFrame, path: Path, lang: str = "en") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for era, color, shift in (("2017 onward", INK, -0.08), ("Through 2016", CLAY, 0.08)):
        part = table[table["era"] == era]
        if part.empty:
            continue
        part = part.set_index("weekday").reindex(WEEKDAYS)
        x = np.arange(len(WEEKDAYS)) + shift
        label = ERA_FI.get(era, era) if lang == "fi" else era
        ax.errorbar(
            x,
            part["equal_median"],
            yerr=_whisker(part["equal_median"], part["equal_median_lo"], part["equal_median_hi"]),
            fmt="o",
            color=color,
            capsize=3,
            label=label,
        )
    _hline(ax)
    ax.set_xticks(range(7), _ticks(WEEKDAYS, lang))
    ax.set_ylabel(_t(lang, "era_y"))
    ax.set_title(_t(lang, "era_title"))
    ax.legend(frameon=False)
    _save(fig, path)


def month_era(table: pd.DataFrame, path: Path, lang: str = "en") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = np.arange(len(MONTHS))
    for era, color in (("2017 onward", INK), ("Through 2016", CLAY)):
        part = table[table["era"] == era].set_index("level").reindex(MONTHS)
        if part["multiplier"].isna().all():
            continue
        if part["multiplier_lo"].notna().any():
            ax.fill_between(
                x, part["multiplier_lo"], part["multiplier_hi"], color=color, alpha=0.15, linewidth=0,
            )
        label = ERA_FI.get(era, era) if lang == "fi" else era
        ax.plot(x, part["multiplier"], color=color, marker="o", linewidth=1.3, label=label)
    _hline(ax)
    ticks = [MONTH_FI[name] if lang == "fi" else name[:3] for name in MONTHS]
    ax.set_xticks(x, ticks)
    ax.set_ylabel(_t(lang, "month_era_y"))
    ax.set_title(_t(lang, "month_era_title"))
    ax.legend(frameon=False)
    _save(fig, path)


def multipliers(
    table: pd.DataFrame,
    path: Path,
    *,
    title: str,
    ylabel: str,
    level: str = "level",
    wide: bool = False,
    lang: str = "en",
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(12.4, 4.8) if wide else (8.6, 4.4))
    x = np.arange(len(table))
    if table["multiplier_lo"].notna().any():
        ax.fill_between(x, table["multiplier_lo"], table["multiplier_hi"], color=BAND, linewidth=0)
    ax.plot(x, table["multiplier"], color=INK, marker="o", linewidth=1.3)
    _hline(ax)
    labels = _ticks(table[level], lang)
    rotation = 90 if len(table) > 20 else (45 if len(table) > 12 else 0)
    ax.set_xticks(
        x, labels, rotation=rotation,
        ha="center" if rotation == 90 else ("right" if rotation else "center"),
        fontsize=8 if len(table) > 20 else 10,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)


def holidays(table: pd.DataFrame, path: Path, lang: str = "en") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    y = np.arange(len(table))
    if table["multiplier_lo"].notna().any():
        ax.hlines(y, table["multiplier_lo"], table["multiplier_hi"], color=BAND, linewidth=2)
    ax.plot(table["multiplier"], y, "o", color=INK)
    ax.axvline(1, color="#888888", linewidth=0.8)
    names = table["level"] if "level" in table else table["label"]
    if lang == "fi":
        labels = [NAMES_FI.get(name, name) for name in names]
    else:
        labels = [f"{name} ({NAMES_FI[name]})" if name in NAMES_FI else name for name in names]
    ax.set_yticks(y, labels)
    ax.set_xlabel(_t(lang, "holiday_x"))
    ax.set_title(_t(lang, "holiday_title"))
    ax.invert_yaxis()
    _save(fig, path)


def day_of_year(table: pd.DataFrame, path: Path, lang: str = "en") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(table["doy"], table["multiplier"], color="#d0d7e2", linewidth=0.7)
    ax.plot(table["doy"], table["smooth_multiplier"], color=INK, linewidth=1.6)
    _hline(ax)
    months = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    names = MONTH_ABBR_FI if lang == "fi" else ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(months, names)
    ax.set_ylabel(_t(lang, "doy_y"))
    ax.set_title(_t(lang, "doy_title"))
    _save(fig, path)


def draw(report, directory: Path, lang: str = "en") -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "volume": directory / "volume_month.png",
        "clock": directory / "campaign_clock.png",
        "dow": directory / "dow.png",
        "dow_era": directory / "dow_era.png",
        "month": directory / "month.png",
        "month_era": directory / "month_era.png",
        "week": directory / "iso_week.png",
        "dom": directory / "day_of_month.png",
        "holidays": directory / "holidays.png",
        "day_of_year": directory / "day_of_year.png",
    }
    if len(report.volume_month):
        volume(report.volume_month, paths["volume"], lang)
    if len(report.clock):
        clock(report.clock, paths["clock"], lang)
    if len(report.dow) and report.dow["equal_median_lo"].notna().any():
        dow(report.dow, paths["dow"], lang)
    elif len(report.dow):
        table = report.dow.copy()
        table["equal_median_lo"] = table["equal_median"]
        table["equal_median_hi"] = table["equal_median"]
        dow(table, paths["dow"], lang)
    if len(report.dow_era) and report.dow_era["equal_median_lo"].notna().any():
        dow_era(report.dow_era, paths["dow_era"], lang)
    if len(report.month):
        multipliers(
            report.month, paths["month"], level="level", lang=lang,
            title=_t(lang, "month_title"), ylabel=_t(lang, "mult_y"),
        )
    if len(report.month_era):
        month_era(report.month_era, paths["month_era"], lang)
    if len(report.week):
        multipliers(
            report.week, paths["week"], level="level", wide=True, lang=lang,
            title=_t(lang, "week_title"), ylabel=_t(lang, "mult_y"),
        )
    if len(report.dom):
        multipliers(
            report.dom, paths["dom"], level="level", wide=True, lang=lang,
            title=_t(lang, "dom_title"), ylabel=_t(lang, "mult_y"),
        )
    if len(report.holiday_named):
        holidays(report.holiday_named, paths["holidays"], lang)
    if len(report.day_of_year):
        day_of_year(report.day_of_year, paths["day_of_year"], lang)
    return {key: str(path) for key, path in paths.items() if path.exists()}
