"""Notebook display. The essay is generated from the report."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Image, Markdown, display

from .analysis import MONTHS, Report, boot_count, build_report
from .figures import draw
from .shapes import describe
from .narrate import (
    checks_intro,
    checks_tail,
    checks_warnings,
    clock_after,
    clock_before,
    coverage_gaps,
    coverage_head,
    dec,
    dom,
    dow_era,
    dow_main,
    dow_sensitivity_line,
    era_name,
    holiday_main,
    holiday_name,
    holiday_pool,
    holiday_summer,
    iv,
    launch,
    lead,
    month_era,
    month_main,
    month_name,
    num,
    pct,
    pooled_name,
    sample_name,
    status_name,
    structure,
    volume_finding,
    volume_intro,
    volume_year,
    week_note,
    weekday_name,
)

TABLES = (
    "coverage_status", "coverage_year", "volume_month", "volume_year", "volume_era",
    "top_campaigns", "clock", "clock_summary", "clock_shape", "time_to_50k", "launch",
    "dow", "dow_era", "dow_sensitivity", "month", "month_era", "week", "dom",
    "holiday_pooled", "holiday_named", "interaction", "day_of_year", "samples",
)
SOURCE_FILES = ("analysis.py", "model.py", "panel.py", "holidays.py", "load.py")
CAPTION = {
    "en": {
        "volume": "Monthly web signatures",
        "clock": "Share of an initiative's web signatures by age",
        "dow": "Weekday after day 7",
        "dow_era": "Weekday median in both eras",
        "month": "Month from 2017 onward",
        "month_era": "Month in both eras",
        "week": "ISO week from 2017 onward",
        "day_of_year": "Day of year",
        "holidays": "Named days from 2017 onward",
        "dom": "Day of month",
    },
    "fi": {
        "volume": "Verkkoallekirjoitukset kuukausittain",
        "clock": "Osuus aloitteen verkkoallekirjoituksista iän mukaan",
        "dow": "Viikonpäivä seitsemännen päivän jälkeen",
        "dow_era": "Viikonpäivän mediaani molemmissa jaksoissa",
        "month": "Kuukausi vuodesta 2017",
        "month_era": "Kuukausi molemmissa jaksoissa",
        "week": "ISO-viikko vuodesta 2017",
        "day_of_year": "Vuodenpäivä",
        "holidays": "Nimetyt päivät vuodesta 2017",
        "dom": "Kuukaudenpäivä",
    },
}
GROUP = {
    "en": {
        "full_under_50k": "Under 50,000, open six months",
        "full_50k": "Reached 50,000, stayed open",
        "early_50k": "Reached 50,000, closed before day 170",
    },
    "fi": {
        "full_under_50k": "Alle 50 000, auki puoli vuotta",
        "full_50k": "Yli 50 000, auki puoli vuotta",
        "early_50k": "Yli 50 000, sulkeutui ennen päivää 170",
    },
}
STATUS_ORDER = (
    "ok", "ok_gap", "open", "sum_mismatch", "missing_series",
    "no_window", "no_dates", "future", "not_fetched", "canceled",
)


def _token(root: Path, n_boot: int) -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update((root / "src" / name).read_bytes())
        digest.update(b"\0")
    digest.update((root / "data" / "live" / "LATEST").read_bytes())
    digest.update(b"\0")
    digest.update(str(n_boot).encode())
    return digest.hexdigest()


def _load_or_build(root: Path, n_boot: int) -> tuple[Report, bool]:
    out = root / "reports"
    token_path = out / "report.token"
    pkl_path = out / "report.pkl"
    token = _token(root, n_boot)
    if token_path.is_file() and pkl_path.is_file() and token_path.read_text(encoding="utf-8").strip() == token:
        with pkl_path.open("rb") as handle:
            report = pickle.load(handle)
        report.figures = {}
        return report, False
    report = build_report(root, n_boot=n_boot)
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "report.pkl.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(report, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(pkl_path)
    token_path.write_text(token + "\n", encoding="utf-8")
    return report, True


def prepare(root: Path | None = None, n_boot: int | None = None, lang: str = "en") -> Report:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    n_boot = boot_count() if n_boot is None else int(n_boot)
    lang = "fi" if lang == "fi" else "en"
    report, fresh = _load_or_build(root, n_boot)
    report.path_notes = describe(report)
    out = root / "reports"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        frame = getattr(report, name)
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(tables / f"{name}.csv", index=False)
    report.figures = draw(report, out / "figures" / lang, lang)
    manifest_path = out / "run_manifest.json"
    previous: dict = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except json.JSONDecodeError:
            previous = {}
    figures = previous.get("figures") if isinstance(previous.get("figures"), dict) else {}
    if fresh or not figures or any(not isinstance(value, dict) for value in figures.values()):
        figures = {}
    figures[lang] = report.figures
    payload = {
        "stamp": report.summary["stamp"],
        "cutoff": report.summary["cutoff"],
        "n_boot": report.n_boot,
        "summary": report.summary,
        "warnings": report.warnings,
        "samples": report.samples.to_dict(orient="records"),
        "figures": figures,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def _md(text: str) -> None:
    if text:
        display(Markdown(text))


def _table(frame: pd.DataFrame) -> None:
    if len(frame):
        display(HTML(frame.to_html(index=False, border=0)))


def _caption(lang: str, key: str) -> str:
    return CAPTION.get(lang, CAPTION["en"])[key]


def _pct_cell(value, lang: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    scaled = abs(float(value) * 100)
    digits = 0 if scaled == 0 or scaled >= 10 else 1
    return pct(value, lang, digits)


def show_figure(report: Report, key: str, lang: str = "en") -> None:
    path = report.figures.get(key)
    caption = _caption(lang, key)
    if not path:
        missing = "Kuvaa ei piirretty." if lang == "fi" else "The figure was not drawn."
        _md(f"*{caption}. {missing}*")
        return
    _md(f"**{caption}**")
    display(Image(filename=path))


def show_lead(report: Report, lang: str = "en") -> None:
    _md(lead(report, lang))


def show_coverage(report: Report, lang: str = "en") -> None:
    _md(coverage_head(report, lang))
    status = report.coverage_status.copy()
    status["_ord"] = status["status"].map({code: i for i, code in enumerate(STATUS_ORDER)}).fillna(99)
    status = status.sort_values(["_ord", "status"])
    show = pd.DataFrame({
        ("Tila" if lang == "fi" else "Status"): status["status"].map(lambda code: status_name(code, lang)),
        ("Aloitteita" if lang == "fi" else "Initiatives"): status["campaigns"].map(lambda v: num(v, lang)),
        ("Verkkolaskuri" if lang == "fi" else "Web counter"): status["web_counter"].map(lambda v: num(v, lang)),
        ("Päiväsarja" if lang == "fi" else "Daily series"): status["series_sum"].map(lambda v: num(v, lang)),
        ("Laskuri miinus sarja" if lang == "fi" else "Counter minus series"): status["residual"].map(lambda v: num(v, lang)),
    })
    _table(show)
    _md(coverage_gaps(report, lang))
    year = report.coverage_year.copy()
    columns = {
        "start_year": "Aloitusvuosi" if lang == "fi" else "Start year",
        "campaigns": "Aloitteita" if lang == "fi" else "Initiatives",
        "usable": "Käyttökelpoisia" if lang == "fi" else "Usable",
        "missing_series": "Ei päiväsarjaa" if lang == "fi" else "No daily series",
        "sum_mismatch": "Iso aukko" if lang == "fi" else "Large gap",
        "web_counter": "Verkkolaskuri" if lang == "fi" else "Web counter",
        "series_sum": "Päiväsarja" if lang == "fi" else "Daily series",
        "residual": "Ilman päivää" if lang == "fi" else "Unmatched",
    }
    keep = [column for column in columns if column in year]
    out = year[keep].copy()
    for column in keep:
        if column != "start_year":
            out[column] = out[column].map(lambda v: num(v, lang))
    _table(out.rename(columns=columns))


def show_volume(report: Report, lang: str = "en") -> None:
    _md(volume_intro(report, lang))
    show_figure(report, "volume", lang)
    _md(volume_finding(report, lang))
    era = report.volume_era.copy()
    show = pd.DataFrame({
        ("Jakso" if lang == "fi" else "Era"): era["era"].map(lambda v: era_name(v, lang)),
        ("Täysiä kuukausia" if lang == "fi" else "Full months"): era["full_months"].map(lambda v: num(v, lang)),
        ("Allekirjoituksia" if lang == "fi" else "Signatures"): era["signatures"].map(lambda v: num(v, lang)),
        ("Mediaani avointa päivää kohti" if lang == "fi" else "Median per open day"): era["median_per_open_day"].map(lambda v: dec(v, lang, 1)),
        ("Keskiarvo avointa päivää kohti" if lang == "fi" else "Mean per open day"): era["mean_per_open_day"].map(lambda v: dec(v, lang, 1)),
        ("Suurimman aloitteen mediaaniosuus" if lang == "fi" else "Median share of the largest initiative"): era["median_top_share"].map(lambda v: _pct_cell(v, lang)),
    })
    _table(show)
    _md(volume_year(report, lang))
    year = report.volume_year
    table = pd.DataFrame({
        ("Vuosi" if lang == "fi" else "Year"): year["year"],
        ("Allekirjoituksia" if lang == "fi" else "Signatures"): year["signatures"].map(lambda v: num(v, lang)),
        ("Avointa päivää kohti" if lang == "fi" else "Per open day"): year["per_open_day"].map(lambda v: dec(v, lang, 1)),
        ("Aloitteita" if lang == "fi" else "Initiatives"): year["campaigns"].map(lambda v: num(v, lang)),
        ("Suurin aloite" if lang == "fi" else "Largest initiative"): year["top_name"],
        ("Sen osuus" if lang == "fi" else "Its share"): year["top_share"].map(lambda v: _pct_cell(v, lang)),
    })
    _table(table)


def _shape_table(report: Report, lang: str) -> pd.DataFrame:
    frame = report.clock_shape.copy()
    if frame.empty:
        return frame
    order = {name: i for i, name in enumerate(GROUP["en"])}
    frame["_ord"] = frame["group"].map(order).fillna(99)
    frame = frame.sort_values("_ord")
    labels = GROUP.get(lang, GROUP["en"])
    return pd.DataFrame({
        ("Ryhmä" if lang == "fi" else "Group"): frame["group"].map(lambda name: labels.get(name, name)),
        ("Aloitteita" if lang == "fi" else "Initiatives"): frame["n"].map(lambda v: num(v, lang)),
        ("Ikkunan mediaani, päivää" if lang == "fi" else "Median window, days"): frame["median_length"].map(lambda v: num(v, lang)),
        ("Osuus päivänä 91" if lang == "fi" else "Share by day 91"): frame["share_d91"].map(lambda v: _pct_cell(v, lang)),
        ("Keskimmäinen puolikas päivänä 91" if lang == "fi" else "Middle half at day 91"): [
            f"{_pct_cell(lo, lang)}–{_pct_cell(hi, lang)}" for lo, hi in zip(frame["share_d91_p25"], frame["share_d91_p75"])
        ],
        ("Osuus oman ikkunan puolivälissä" if lang == "fi" else "Share at halfway of own window"): frame["share_half_own"].map(lambda v: _pct_cell(v, lang)),
        ("Ensimmäiset 14 päivää" if lang == "fi" else "First fortnight"): frame["first14"].map(lambda v: _pct_cell(v, lang)),
        ("Viimeiset 14 päivää" if lang == "fi" else "Last fortnight"): frame["last14"].map(lambda v: _pct_cell(v, lang)),
    })


def show_clock(report: Report, lang: str = "en") -> None:
    _md(clock_before(report, lang))
    show_figure(report, "clock", lang)
    _table(_shape_table(report, lang))
    _md(clock_after(report, lang))


def _dow_table(report: Report, lang: str) -> pd.DataFrame:
    frame = report.dow
    model = frame["model_multiplier"].map(lambda v: dec(v, lang)) if "model_multiplier" in frame else "—"
    model_iv = (
        [iv(a, b, lang) for a, b in zip(frame["model_lo"], frame["model_hi"])]
        if "model_lo" in frame else ""
    )
    return pd.DataFrame({
        ("Viikonpäivä" if lang == "fi" else "Weekday"): frame["weekday"].map(lambda v: weekday_name(v, lang)),
        ("Osuus päivistä" if lang == "fi" else "Share of days"): frame["day_share"].map(lambda v: pct(v, lang, 1)),
        ("Osuus allekirjoituksista" if lang == "fi" else "Share of signatures"): frame["signature_share"].map(lambda v: pct(v, lang, 1)),
        ("Allekirjoitukset / päivät" if lang == "fi" else "Signatures / days"): frame["crude_ratio"].map(lambda v: dec(v, lang)),
        ("Mediaaniviikko" if lang == "fi" else "Median week"): frame["equal_median"].map(lambda v: dec(v, lang)),
        ("Mediaanin väli" if lang == "fi" else "Median interval"): [iv(a, b, lang) for a, b in zip(frame["equal_median_lo"], frame["equal_median_hi"])],
        ("Allekirjoituksilla painotettu" if lang == "fi" else "Signature-weighted"): frame["signature_weighted"].map(lambda v: dec(v, lang)),
        ("Painotettu väli" if lang == "fi" else "Weighted interval"): [iv(a, b, lang) for a, b in zip(frame["signature_lo"], frame["signature_hi"])],
        ("Malli, 1 + lukumäärä" if lang == "fi" else "Model on 1 + count"): model,
        ("Mallin väli" if lang == "fi" else "Model interval"): model_iv,
    })


def show_dow(report: Report, lang: str = "en") -> None:
    _md(dow_main(report, lang))
    show_figure(report, "dow", lang)
    if len(report.dow):
        _table(_dow_table(report, lang))
    if len(report.dow_sensitivity):
        _md(dow_sensitivity_line(report, lang))
        sens = report.dow_sensitivity.copy()
        show = pd.DataFrame({
            ("Otos" if lang == "fi" else "Sample"): sens["sample"].map(lambda v: sample_name(v, lang)),
            ("Viikonpäivä" if lang == "fi" else "Weekday"): sens["weekday"].map(lambda v: weekday_name(v, lang)),
            ("Mediaaniviikko" if lang == "fi" else "Median week"): sens["equal_median"].map(lambda v: dec(v, lang)),
            ("Allekirjoituksilla painotettu" if lang == "fi" else "Signature-weighted"): sens["signature_weighted"].map(lambda v: dec(v, lang)),
            ("Aloitteita" if lang == "fi" else "Initiatives"): sens["n_campaigns"].map(lambda v: num(v, lang)),
        })
        _table(show)
    if len(report.dow_era):
        _md(dow_era(report, lang))
        show_figure(report, "dow_era", lang)
        era = report.dow_era.copy()
        table = pd.DataFrame({
            ("Jakso" if lang == "fi" else "Era"): era["era"].map(lambda v: era_name(v, lang)),
            ("Viikonpäivä" if lang == "fi" else "Weekday"): era["weekday"].map(lambda v: weekday_name(v, lang)),
            ("Mediaani" if lang == "fi" else "Median"): era["equal_median"].map(lambda v: dec(v, lang)),
            ("Alaraja" if lang == "fi" else "Low"): era["equal_median_lo"].map(lambda v: dec(v, lang)),
            ("Yläraja" if lang == "fi" else "High"): era["equal_median_hi"].map(lambda v: dec(v, lang)),
            ("Aloitteita" if lang == "fi" else "Initiatives"): era["n_campaigns"].map(lambda v: num(v, lang)),
            ("Lohkoja" if lang == "fi" else "Blocks"): era["n_blocks"].map(lambda v: num(v, lang)),
        })
        _table(table)


def _multiplier_table(frame: pd.DataFrame, level_name: str, lang: str, translate) -> pd.DataFrame:
    return pd.DataFrame({
        level_name: frame["level"].map(lambda v: translate(v, lang)),
        ("Kerroin" if lang == "fi" else "Multiplier"): frame["multiplier"].map(lambda v: dec(v, lang)),
        ("Väli" if lang == "fi" else "Interval"): [iv(a, b, lang) for a, b in zip(frame["multiplier_lo"], frame["multiplier_hi"])],
        ("Aloitepäiviä" if lang == "fi" else "Initiative-days"): frame["n"].map(lambda v: num(v, lang)),
    })


def show_month(report: Report, lang: str = "en") -> None:
    _md(month_main(report, lang))
    show_figure(report, "month", lang)
    if len(report.month):
        ordered = report.month.copy()
        ordered["_ord"] = ordered["level"].map({name: i for i, name in enumerate(MONTHS)})
        ordered = ordered.sort_values("_ord")
        _table(_multiplier_table(ordered, "Kuukausi" if lang == "fi" else "Month", lang, month_name))
    if len(report.month_era):
        _md(month_era(report, lang))
        show_figure(report, "month_era", lang)
        both = report.month_era.copy()
        both["_ord"] = both["level"].map({name: i for i, name in enumerate(MONTHS)})
        both["_era"] = both["era"].map({"Through 2016": 0, "2017 onward": 1})
        both = both.sort_values(["_era", "_ord"])
        table = pd.DataFrame({
            ("Jakso" if lang == "fi" else "Era"): both["era"].map(lambda v: era_name(v, lang)),
            ("Kuukausi" if lang == "fi" else "Month"): both["level"].map(lambda v: month_name(v, lang)),
            ("Kerroin" if lang == "fi" else "Multiplier"): both["multiplier"].map(lambda v: dec(v, lang)),
            ("Alaraja" if lang == "fi" else "Low"): both["multiplier_lo"].map(lambda v: dec(v, lang)),
            ("Yläraja" if lang == "fi" else "High"): both["multiplier_hi"].map(lambda v: dec(v, lang)),
        })
        _table(table)
    _md(week_note(report, lang))
    show_figure(report, "week", lang)
    show_figure(report, "day_of_year", lang)


def show_holidays(report: Report, lang: str = "en") -> None:
    _md(holiday_main(report, lang))
    show_figure(report, "holidays", lang)
    if len(report.holiday_named):
        frame = report.holiday_named
        table = pd.DataFrame({
            ("Päivä" if lang == "fi" else "Day"): frame["level"].map(lambda v: holiday_name(v, lang)),
            ("Kerroin" if lang == "fi" else "Multiplier"): frame["multiplier"].map(lambda v: dec(v, lang)),
            ("Väli" if lang == "fi" else "Interval"): [iv(a, b, lang) for a, b in zip(frame["multiplier_lo"], frame["multiplier_hi"])],
            ("Aloitepäiviä" if lang == "fi" else "Initiative-days"): frame["n"].map(lambda v: num(v, lang)),
            ("Aloitteita" if lang == "fi" else "Initiatives"): frame["n_campaigns"].map(lambda v: num(v, lang)),
        })
        _table(table)
    if len(report.holiday_pooled):
        _md(holiday_pool(report, lang))
        pooled = report.holiday_pooled
        table = pd.DataFrame({
            ("Luokka" if lang == "fi" else "Class"): pooled["level"].map(lambda v: pooled_name(v, lang)),
            ("Tavalliseen päivään verrattuna" if lang == "fi" else "Versus an ordinary day"): pooled["multiplier"].map(lambda v: dec(v, lang)),
            ("Alaraja" if lang == "fi" else "Low"): pooled["multiplier_lo"].map(lambda v: dec(v, lang)),
            ("Yläraja" if lang == "fi" else "High"): pooled["multiplier_hi"].map(lambda v: dec(v, lang)),
            ("Aloitepäiviä" if lang == "fi" else "Initiative-days"): pooled["n"].map(lambda v: num(v, lang)),
        })
        _table(table)
    _md(holiday_summer(report, lang))
    _md(dom(report, lang))
    show_figure(report, "dom", lang)


def show_structure(report: Report, lang: str = "en") -> None:
    _md(structure(report, lang))
    top = report.top_campaigns
    table = pd.DataFrame({
        ("Tunnus" if lang == "fi" else "Id"): top["id"],
        ("Nimi" if lang == "fi" else "Name"): top["name"],
        ("Alku" if lang == "fi" else "Start"): top["start_date"],
        ("Loppu" if lang == "fi" else "End"): top["end_date"],
        ("Tila" if lang == "fi" else "State"): top["state"],
        ("Verkko" if lang == "fi" else "Web"): top["support_count"].map(lambda v: num(v, lang)),
        ("Muut kanavat" if lang == "fi" else "Other channels"): top["external_support_count"].map(lambda v: num(v, lang)),
        ("Julkinen yhteensä" if lang == "fi" else "Public total"): top["total_support_count"].map(lambda v: num(v, lang)),
        ("Osuus verkosta" if lang == "fi" else "Share of all web"): top["web_share"].map(lambda v: _pct_cell(v, lang)),
    })
    _table(table)
    _md(launch(report, lang))
    frame = report.launch
    if len(frame):
        show = pd.DataFrame({
            ("Aloituspäivä" if lang == "fi" else "Start weekday"): frame["weekday"].map(lambda v: weekday_name(v, lang)),
            ("Aloitteita" if lang == "fi" else "Initiatives"): frame["n_starts"].map(lambda v: num(v, lang)),
            ("Osuus" if lang == "fi" else "Share"): frame["share"].map(lambda v: _pct_cell(v, lang)),
            ("Mediaani verkossa" if lang == "fi" else "Median web"): frame["median_web"].map(lambda v: num(v, lang)),
            ("Mediaani, julkinen yhteensä" if lang == "fi" else "Median public total"): frame["median_total"].map(lambda v: num(v, lang)),
        })
        _table(show)


def show_checks(report: Report, lang: str = "en") -> None:
    _md(checks_intro(report, lang))
    samples = report.samples.copy()
    table = pd.DataFrame({
        ("Otos" if lang == "fi" else "Sample"): samples["sample"].map(lambda v: sample_name(v, lang)),
        ("Aloitteita" if lang == "fi" else "Initiatives"): samples["campaigns"].map(lambda v: num(v, lang)),
        ("Aloitepäiviä" if lang == "fi" else "Initiative-days"): samples["campaign_days"].map(lambda v: num(v, lang)),
    })
    _table(table)
    _md(checks_warnings(report, lang))
    _md(checks_tail(report, lang))
