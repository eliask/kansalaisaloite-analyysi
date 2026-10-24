"""Prose for the English and Finnish pages.

Each language has its own paragraphs. Numbers are read from the report.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from .analysis import BREAK
from .holidays import NAMES_FI

NBSP = "\u00a0"
FIRST_WEEK = "Days 0–6"

WEEKDAY_FI = {
    "Monday": "maanantai",
    "Tuesday": "tiistai",
    "Wednesday": "keskiviikko",
    "Thursday": "torstai",
    "Friday": "perjantai",
    "Saturday": "lauantai",
    "Sunday": "sunnuntai",
}
WEEKDAY_FI_GEN = {
    "Monday": "maanantain",
    "Tuesday": "tiistain",
    "Wednesday": "keskiviikon",
    "Thursday": "torstain",
    "Friday": "perjantain",
    "Saturday": "lauantain",
    "Sunday": "sunnuntain",
}
MONTH_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_FI = {
    "January": "tammikuu",
    "February": "helmikuu",
    "March": "maaliskuu",
    "April": "huhtikuu",
    "May": "toukokuu",
    "June": "kesäkuu",
    "July": "heinäkuu",
    "August": "elokuu",
    "September": "syyskuu",
    "October": "lokakuu",
    "November": "marraskuu",
    "December": "joulukuu",
}
ERA_FI = {
    "Through 2016": "vuoteen 2016",
    "2017": "2017",
    "2018–2025": "2018–2025",
    "2017 onward": "vuodesta 2017",
}
SAMPLE_EN = {
    "Open campaign-days in the zero-filled panel": "Open initiative-days in the zero-filled panel",
}
SAMPLE_FI = {
    "2017 onward, age ≥ 7, ≥ 100 web signatures": "Vuodesta 2017, ikä ≥ 7, vähintään 100 verkkoallekirjoitusta",
    "Through 2016, age ≥ 7, ≥ 100 web signatures": "Vuoteen 2016, ikä ≥ 7, vähintään 100 verkkoallekirjoitusta",
    "Open campaign-days in the zero-filled panel": "Avoimet aloitepäivät nollilla täydennetyssä paneelissa",
    "30 web signatures, holidays out": "30 verkkoallekirjoitusta, pyhät pois",
    "100 web signatures, holidays kept": "100 verkkoallekirjoitusta, pyhät mukana",
    "1,000 web signatures, holidays out": "1 000 verkkoallekirjoitusta, pyhät pois",
}
STATUS_EN = {
    "ok": "Matches",
    "ok_gap": "Small gap",
    "open": "Still open",
    "sum_mismatch": "Large gap",
    "missing_series": "No daily series",
    "no_window": "No window",
    "no_dates": "No dates",
    "future": "Not started",
    "not_fetched": "Not fetched",
    "canceled": "Canceled",
}
STATUS_FI = {
    "ok": "Täsmää",
    "ok_gap": "Pieni aukko",
    "open": "Yhä auki",
    "sum_mismatch": "Iso aukko",
    "missing_series": "Ei päiväsarjaa",
    "no_window": "Ei ikkunaa",
    "no_dates": "Ei päivämääriä",
    "future": "Ei alkanut",
    "not_fetched": "Ei haettu",
    "canceled": "Peruttu",
}
POOLED_EN = {"Public holiday": "Public holiday", "Eve": "Eve"}
POOLED_FI = {"Public holiday": "Julkinen pyhä", "Eve": "Aatto"}


def _fi(lang: str) -> bool:
    return lang == "fi"


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value)


def _quantize(value: float, digits: int) -> str:
    quant = Decimal("1") if digits == 0 else Decimal("0." + "0" * (digits - 1) + "1")
    rounded = Decimal(f"{float(value):.10f}").quantize(quant, rounding=ROUND_HALF_UP)
    return f"{rounded:.{digits}f}"


def num(value, lang: str = "en") -> str:
    if _missing(value):
        return "—"
    text = f"{int(round(float(value))):,}"
    return text.replace(",", NBSP) if _fi(lang) else text


def dec(value, lang: str = "en", digits: int = 2) -> str:
    if _missing(value):
        return "—"
    text = _quantize(float(value), digits)
    return text.replace(".", ",") if _fi(lang) else text


def pct(value, lang: str = "en", digits: int = 0) -> str:
    """Share for a table cell."""
    if _missing(value):
        return "—"
    text = _quantize(float(value) * 100, digits)
    if _fi(lang):
        return text.replace(".", ",") + NBSP + "%"
    return text + "%"


def prose_share(value, lang: str = "en") -> str:
    """Share inside a sentence. Small shares keep one decimal."""
    if _missing(value):
        return "—"
    scaled = abs(float(value) * 100)
    digits = 0 if scaled == 0 or scaled >= 10 else 1
    text = _quantize(float(value) * 100, digits)
    if _fi(lang):
        return text.replace(".", ",") + " prosenttia"
    return text + "%"


def iv(lo, hi, lang: str = "en", digits: int = 2) -> str:
    if _missing(lo) or _missing(hi):
        return ""
    return f"{dec(lo, lang, digits)}–{dec(hi, lang, digits)}"


def piv(lo, hi, lang: str = "en", digits: int = 2) -> str:
    text = iv(lo, hi, lang, digits)
    return f" ({text})" if text else ""


def pct_span(lo, hi, lang: str = "en") -> str:
    if _missing(lo) or _missing(hi):
        return "—"
    return f"{pct(lo, lang)}–{pct(hi, lang)}"


def prose_range(lo, hi, lang: str = "en") -> str:
    if _missing(lo) or _missing(hi):
        return "—"
    left = prose_share(lo, "en").removesuffix("%")
    right = prose_share(hi, "en").removesuffix("%")
    if _fi(lang):
        return f"{left.replace('.', ',')}–{right.replace('.', ',')} prosenttia"
    return f"{prose_share(lo, 'en')} to {prose_share(hi, 'en')}"


def weekday_name(name: str, lang: str) -> str:
    return WEEKDAY_FI.get(str(name), str(name)) if _fi(lang) else str(name)


def weekday_genitive(name: str, lang: str) -> str:
    if _fi(lang):
        return WEEKDAY_FI_GEN.get(str(name), str(name))
    return str(name) + "'s"


def month_name(name: str, lang: str) -> str:
    return MONTH_FI.get(str(name), str(name)) if _fi(lang) else str(name)


def era_name(name: str, lang: str) -> str:
    return ERA_FI.get(str(name), str(name)) if _fi(lang) else str(name)


def sample_name(name: str, lang: str) -> str:
    text = str(name)
    table = SAMPLE_FI if _fi(lang) else SAMPLE_EN
    return table.get(text, text)


def holiday_name(name: str, lang: str) -> str:
    fi = NAMES_FI.get(str(name), "")
    if _fi(lang):
        return fi or str(name)
    return f"{name} ({fi})" if fi else str(name)


def status_name(name: str, lang: str) -> str:
    table = STATUS_FI if _fi(lang) else STATUS_EN
    return table.get(str(name), str(name))


def pooled_name(name: str, lang: str) -> str:
    table = POOLED_FI if _fi(lang) else POOLED_EN
    return table.get(str(name), str(name))


def initiatives(n, lang: str = "en") -> str:
    count = int(n)
    if _fi(lang):
        word = "aloite" if count == 1 else "aloitetta"
    else:
        word = "initiative" if count == 1 else "initiatives"
    return f"{num(count, lang)} {word}"


def _join(items: list[str], lang: str) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    conj = " ja " if _fi(lang) else " and "
    return ", ".join(items[:-1]) + conj + items[-1]


def _row(frame: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    if frame is None or len(frame) == 0 or column not in frame:
        return None
    hit = frame[frame[column] == value]
    if hit.empty:
        return None
    return hit.iloc[0]


def _shape(report, group: str) -> pd.Series | None:
    return _row(report.clock_shape, "group", group)


def _piece(report, group: str, piece: str) -> pd.Series | None:
    frame = report.clock_summary
    if frame is None or len(frame) == 0:
        return None
    hit = frame[(frame["group"] == group) & (frame["piece"] == piece)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _includes_one(lo, hi) -> bool:
    if _missing(lo) or _missing(hi):
        return False
    return float(lo) <= 1 <= float(hi)


def _below_one(row: pd.Series) -> bool:
    hi = row["multiplier_hi"]
    return not _missing(hi) and float(hi) < 1


def _above_one(row: pd.Series) -> bool:
    lo = row["multiplier_lo"]
    return not _missing(lo) and float(lo) > 1


def _month_stamp(value, lang: str) -> str:
    ts = pd.Timestamp(value)
    if _fi(lang):
        return f"{MONTH_FI[MONTH_EN[ts.month - 1]]} {ts.year}"
    return f"{MONTH_EN[ts.month - 1]} {ts.year}"


def _fetched_day(report) -> str:
    return str(report.summary["fetched_at"])[:10]


def missing_years(report, lang: str) -> str:
    years = report.coverage_year
    hit = years.loc[years["missing_series"] > 0, "start_year"] if len(years) else years
    if hit.empty:
        return "ei käynnistysvuotta tässä poiminnassa" if _fi(lang) else "no start year in this snapshot"
    first, last = int(hit.min()), int(hit.max())
    if first == last:
        return f"vuonna {first}" if _fi(lang) else str(first)
    if last == first + 1:
        return f"vuonna {first} tai {last}" if _fi(lang) else f"{first} or {last}"
    return f"vuosina {first}–{last}" if _fi(lang) else f"{first}–{last}"


def _mismatches(report) -> list[tuple[int, str, int, int]]:
    hit = report.checks[report.checks["status"] == "sum_mismatch"]
    if hit.empty:
        return []
    names = report.snapshot.initiatives.set_index("id")["name_fi"]
    rows = []
    for ident, row in hit.iterrows():
        name = str(names.get(int(ident), "")).replace("\n", " ").strip()
        rows.append((int(ident), name, int(row["support_count"]), int(row["residual"])))
    return rows


def _status(report, code: str) -> pd.Series | None:
    return _row(report.coverage_status, "status", code)


def lead(report, lang: str) -> str:
    summary = report.summary
    if _fi(lang):
        return (
            f"Katkaisuun {summary['cutoff']} mennessä päiväsarja kirjaa {num(summary['series_sum'], lang)} "
            f"verkkoallekirjoitusta {num(summary['campaigns'], lang)} aloitteessa. "
            f"Kukin päivä on yksi lukumäärä, ja tiedosto päättyy yön katkaisuun. "
            f"Aloite {summary['example_id']} on sarjassa {num(summary['example_series'], lang)} "
            f"ja laskurissa {num(summary['example_support'], lang)}. "
            f"Paperi ja muut kanavat ovat yhteensä {num(summary['external'], lang)} ilman päivämääriä. "
            "Avoin rajapinta ei kerro vuorokaudenaikaa eikä viikon tuntia."
        )
    return (
        f"Through {summary['cutoff']}, the daily series records {num(summary['series_sum'], lang)} "
        f"web signatures on {initiatives(summary['campaigns'], lang)}. "
        "Each date is one count, and the file ends at the nightly cutoff. "
        f"Initiative {summary['example_id']} is {num(summary['example_series'], lang)} in the series "
        f"and {num(summary['example_support'], lang)} on the counter. "
        f"Paper and other channels are a total of {num(summary['external'], lang)} with no dates. "
        "The open API has no hour of day and no hour of week."
    )


def coverage_head(report, lang: str) -> str:
    summary = report.summary
    if _fi(lang):
        return (
            f"Poiminta `{summary['stamp']}`, haettu {_fetched_day(report)}. "
            f"Paneeliin tulee {initiatives(summary['usable_campaigns'], lang)} "
            f"{num(summary['campaigns'], lang)} aloitteesta. "
            f"Verkkolaskuri on {num(summary['web_counter'], lang)} ja päiväsarja {num(summary['series_sum'], lang)}."
        )
    return (
        f"Snapshot `{summary['stamp']}`, fetched {_fetched_day(report)}. "
        f"{initiatives(summary['usable_campaigns'], lang)} of {num(summary['campaigns'], lang)} enter the panel. "
        f"The web counter sums to {num(summary['web_counter'], lang)}. "
        f"The daily series sums to {num(summary['series_sum'], lang)}."
    )


def coverage_gaps(report, lang: str) -> str:
    summary = report.summary
    gaps = _mismatches(report)
    open_row = _status(report, "open")
    if _fi(lang):
        parts = [
            "Ikkunan sisältä puuttuva päivä täytetään nollalla, kun kirjatut päivät selittävät laskurin "
            "25 allekirjoituksen tai 2 prosentin tarkkuudella. "
            f"{initiatives(summary['gap_campaigns'], lang)} jää tällä säännöllä mukaan, "
            f"ja {num(summary['gap_residual'], lang)} allekirjoituksella ei ole päivää."
        ]
        if len(gaps) == 1:
            ident, name, web, residual = gaps[0]
            parts.append(
                f"Yksi suurempi aukko jää pois: {name} (tunnus {ident}), laskuri {num(web, lang)}, "
                f"josta {num(residual, lang)} ilman päivää."
            )
        elif len(gaps) > 1:
            parts.append(
                f"{initiatives(summary['mismatch_campaigns'], lang)} jää pois suurempana aukkona, "
                f"laskuri yhteensä {num(summary['mismatch_web'], lang)}."
            )
        else:
            parts.append("Jokainen päättynyt sarja mahtuu tuohon toleranssiin.")
        parts.append(
            f"{num(summary['missing_series_campaigns'], lang)} aloitteella on laskuri eikä päiväsarjaa "
            f"({num(summary['missing_series_web'], lang)} allekirjoitusta). "
            f"Ne on aloitettu {missing_years(report, lang)}."
        )
        if open_row is not None and int(open_row["campaigns"]) > 0:
            parts.append(
                f"{initiatives(open_row['campaigns'], lang)} on yhä auki. "
                f"Niiden sarja jää laskurista {num(open_row['residual'], lang)} allekirjoitusta, "
                "julkaisematon kuluva päivä."
            )
        return " ".join(parts)
    parts = [
        "A date missing inside the collection window is filled with zero when the positive days account for the counter, "
        "to within 25 signatures or 2 percent. "
        f"{initiatives(summary['gap_campaigns'], lang)} stay in on that rule, "
        f"and {num(summary['gap_residual'], lang)} signatures have no date."
    ]
    if len(gaps) == 1:
        ident, name, web, residual = gaps[0]
        parts.append(
            f"One larger gap is left out: {name} (id {ident}), web counter {num(web, lang)}, "
            f"of which {num(residual, lang)} has no date."
        )
    elif len(gaps) > 1:
        parts.append(
            f"{initiatives(summary['mismatch_campaigns'], lang)} are left out on a larger gap, "
            f"web counter {num(summary['mismatch_web'], lang)} altogether."
        )
    else:
        parts.append("Every closed series sits inside that tolerance.")
    parts.append(
        f"{initiatives(summary['missing_series_campaigns'], lang)} have a counter and no daily series "
        f"({num(summary['missing_series_web'], lang)} signatures). "
        f"They started in {missing_years(report, lang)}."
    )
    if open_row is not None and int(open_row["campaigns"]) > 0:
        parts.append(
            f"{initiatives(open_row['campaigns'], lang)} are still open. "
            f"Their series is short of the counter by {num(open_row['residual'], lang)}, "
            "the unpublished current day."
        )
    return " ".join(parts)


def volume_intro(report, lang: str) -> str:
    if _fi(lang):
        return (
            "Ylempi kuvio summaa käyttökelpoiset aloitteet. Se nousee, kun suuri aloite on nuori ja kun aloitteita on auki paljon. "
            "Alempi jakaa avoimilla aloitepäivillä. Nuori aloite tai yksi suuri kuukausi liikuttaa yhä tuota tahtia. "
            "Katkoviiva on 1. tammikuuta 2017."
        )
    return (
        "The upper panel adds every usable initiative. It rises when a large initiative is young and when many are open. "
        "The lower panel divides by the number of open initiative-days. A young initiative, or one large month, still moves that rate. "
        "The dashed line is 1 January 2017."
    )


def volume_finding(report, lang: str) -> str:
    era = report.volume_era.set_index("era")
    early, year, late = era.loc["Through 2016"], era.loc["2017"], era.loc["2018–2025"]
    if _fi(lang):
        return (
            f"Mediaanikuukaudessa on {dec(early['median_per_open_day'], lang, 1)} allekirjoitusta avointa aloitepäivää kohti "
            f"vuoteen 2016 ({num(early['full_months'], lang)} täyttä kuukautta), "
            f"{dec(year['median_per_open_day'], lang, 1)} vuonna 2017 ({num(year['full_months'], lang)} kuukautta) "
            f"ja {dec(late['median_per_open_day'], lang, 1)} vuosina 2018–2025 ({num(late['full_months'], lang)} kuukautta). "
            "Vuoden 2017 mediaani istuu aiempien vuosien joukkoon. Korkeampi taso alkaa vuodesta 2018. "
            f"Mediaanikuukauden suurin aloite on {prose_share(early['median_top_share'], lang)} kuukaudesta vuoteen 2016, "
            f"{prose_share(year['median_top_share'], lang)} vuonna 2017 "
            f"ja {prose_share(late['median_top_share'], lang)} vuosina 2018–2025."
        )
    return (
        f"The median month has {dec(early['median_per_open_day'], lang, 1)} signatures per open initiative-day "
        f"through 2016 ({num(early['full_months'], lang)} full months), "
        f"{dec(year['median_per_open_day'], lang, 1)} in 2017 ({num(year['full_months'], lang)} months) "
        f"and {dec(late['median_per_open_day'], lang, 1)} from 2018 through 2025 ({num(late['full_months'], lang)} months). "
        "The 2017 median sits with the earlier years. The higher level is 2018 onward. "
        f"In the median month the largest initiative is {prose_share(early['median_top_share'], lang)} of the month through 2016, "
        f"{prose_share(year['median_top_share'], lang)} in 2017 "
        f"and {prose_share(late['median_top_share'], lang)} from 2018 through 2025."
    )


def volume_year(report, lang: str) -> str:
    summary = report.summary
    year = report.volume_year
    top = year.loc[year["per_open_day"].idxmax()]
    row_2017 = year[year["year"] == 2017]
    extra = ""
    if len(row_2017):
        row = row_2017.iloc[0]
        if _fi(lang):
            extra = (
                f" Vuonna 2017 suurin aloite on {row['top_name']}, "
                f"{prose_share(row['top_share'], lang)} sen vuoden allekirjoituksista."
            )
        else:
            extra = (
                f" In 2017 the largest initiative is {row['top_name']}, "
                f"{prose_share(row['top_share'], lang)} of that year's signatures."
            )
    if "peak_month" not in summary:
        return extra.strip()
    if _fi(lang):
        return (
            f"Raskain kuukausi on {_month_stamp(summary['peak_month'], lang)}, "
            f"{num(summary['peak_signatures'], lang)} allekirjoitusta. "
            f"Sen suurin aloite, {summary['peak_top_name']}, on {prose_share(summary['peak_top_share'], lang)} kuukaudesta. "
            f"Eniten allekirjoituksia avointa aloitepäivää kohti on vuonna {int(top['year'])}, "
            f"ja sitä johtaa {top['top_name']} ({prose_share(top['top_share'], lang)} vuodesta)."
            f"{extra} "
            "Yksi aloite liikuttaa kuukautta. Alla olevat kalenterivaikutukset estimoidaan aloitteen sisällä."
        )
    return (
        f"The heaviest month is {_month_stamp(summary['peak_month'], lang)}, "
        f"with {num(summary['peak_signatures'], lang)} signatures. "
        f"Its largest initiative, {summary['peak_top_name']}, is {prose_share(summary['peak_top_share'], lang)} of that month. "
        f"The year with the most signatures per open initiative-day is {int(top['year'])}, "
        f"led by {top['top_name']} ({prose_share(top['top_share'], lang)} of that year)."
        f"{extra} "
        "One initiative moves a month. The calendar effects below are estimated inside initiatives."
    )


def _days(value, lang: str) -> str:
    count = int(round(float(value)))
    if _fi(lang):
        return f"{count} päivää" if count != 1 else "1 päivä"
    return f"{count} day" if count == 1 else f"{count} days"


def _path_notes(report) -> dict:
    notes = getattr(report, "path_notes", None)
    return notes if isinstance(notes, dict) else {}


def _has(notes: dict, *keys: str) -> bool:
    return all(key in notes and not _missing(notes[key]) for key in keys)


def _log_at(offset: float, age: float, horizon: int = 183) -> float:
    return math.log1p(age / float(offset)) / math.log1p(horizon / float(offset))


def _fine_share(value, lang: str) -> str:
    """Two decimals, for daily shares that sit near one percent."""
    text = dec(float(value) * 100, lang, 2)
    return text + (" prosenttia" if _fi(lang) else "%")


def _fine_list(values, lang: str) -> str:
    items = [_fine_share(value, lang) for value in values]
    if len(items) < 2:
        return items[0] if items else ""
    conjunction = " ja " if _fi(lang) else " and "
    return ", ".join(items[:-1]) + conjunction + items[-1]


def _gap_sentence(fit: pd.Series, lang: str) -> str:
    if _missing(fit["rmse_exp"]) or _missing(fit["rmse_sqrt"]) or _missing(fit["rmse_linear"]):
        return ""
    exp, root, line = float(fit["rmse_exp"]), float(fit["rmse_sqrt"]), float(fit["rmse_linear"])
    if _fi(lang):
        text = (
            f"Jos vielä tulevat allekirjoitukset jakautuvat jäljellä oleville päiville suunnilleen samalla todennäköisyydellä, "
            f"poikkeama mediaanista päivinä 7–176 on {dec(exp, lang, 3)}. "
            f"Ajan neliöjuurella poikkeama on {dec(root, lang, 3)}. "
            f"Tasaisella päivävauhdilla se on {dec(line, lang, 3)}."
        )
        if exp < root < line:
            text += " Näistä kolmesta hiipuva virta on lähinnä mediaania."
        return text
    text = (
        f"Measured on days 7–176, a fading inflow, each signature still to come assigned to the remaining days "
        f"with roughly equal chance, has gap {dec(exp, lang, 3)}. "
        f"A square root of elapsed time has gap {dec(root, lang, 3)}. "
        f"A constant daily rate has gap {dec(line, lang, 3)}."
    )
    if exp < root < line:
        text += " The fading inflow is the closest of the three."
    return text


def _day91(ordinary: pd.Series, notes: dict, lang: str) -> str:
    median = prose_share(ordinary["share_d91"], lang)
    band = prose_range(ordinary["share_d91_p25"], ordinary["share_d91_p75"], lang)
    root = prose_share(ordinary["sqrt_at_91"], lang)
    line = prose_share(ordinary["linear_at_91"], lang)
    log_sentence = ""
    fortnight = ""
    if _has(notes, "log_offset"):
        log91 = prose_share(_log_at(notes["log_offset"], 91), lang)
        if _fi(lang):
            log_sentence = f"Logaritmi on tuona päivänä {log91}. "
        else:
            log_sentence = f"The logarithm is at {log91} that day. "
        if not _missing(ordinary.get("first14")):
            held = prose_share(ordinary["first14"], lang)
            log14 = prose_share(_log_at(notes["log_offset"], 13), lang)
            root14 = prose_share(math.sqrt(13 / 183), lang)
            if _fi(lang):
                fortnight = (
                    f"Ensimmäisten 14 päivän lopussa mediaani on {held}, logaritmi {log14} ja neliöjuuri {root14}. "
                )
            else:
                fortnight = (
                    f"At the end of the first fortnight the median has {held}, the logarithm {log14} and the square root {root14}. "
                )
    if _fi(lang):
        return (
            f"Päivänä 91 mediaani on {median}. "
            f"Keskimmäinen puolikas on {band}. "
            f"{log_sentence}"
            f"Kuluneen ajan neliöjuuri 183 päivän ikkunassa olisi {root}. "
            f"Tasainen päivävauhti olisi {line}. "
            f"{fortnight}"
        ).strip()
    return (
        f"By day 91 the median has {median}. "
        f"The middle half runs from {band}. "
        f"{log_sentence}"
        f"A square root of elapsed time, on a 183-day window, is at {root}. "
        f"A constant daily rate is at {line}. "
        f"{fortnight}"
    ).strip()


def _fit_paragraphs(ordinary: pd.Series, notes: dict, lang: str) -> list[str]:
    keys = (
        "log_offset", "log_doubling", "rmse_log", "rmse_exp", "rmse_sqrt",
        "rmse_linear", "rmse_power", "power_p", "exp_half_life_days",
    )
    if not _has(notes, *keys):
        alpha = ""
        if not _missing(ordinary.get("power_alpha")):
            power = dec(ordinary["power_alpha"], lang, 2)
            if _fi(lang):
                alpha = f"Logaritmisella asteikolla mediaani kasvaa ajan potenssilla {power}. "
            else:
                alpha = f"On a log scale the median rises as time to the power {power}. "
        fallback = (alpha + _gap_sentence(ordinary, lang)).strip()
        return [fallback] if fallback else []
    c = dec(notes["log_offset"], lang, 1)
    half = num(notes["exp_half_life_days"], lang)
    gaps = (
        dec(notes["rmse_log"], lang, 3),
        dec(notes["rmse_exp"], lang, 3),
        dec(notes["rmse_power"], lang, 3),
        dec(notes["power_p"], lang, 2),
        dec(notes["rmse_sqrt"], lang, 3),
        dec(notes["rmse_linear"], lang, 3),
    )
    doubling = prose_share(notes["log_doubling"], lang)
    others = [float(notes[key]) for key in ("rmse_exp", "rmse_sqrt", "rmse_linear", "rmse_power")]
    smallest = float(notes["rmse_log"]) < min(others)
    within = ""
    if _has(notes, "log_abs_from_day14"):
        points = dec(float(notes["log_abs_from_day14"]) * 100, lang, 1)
        if _fi(lang):
            within = f"Päivästä 14 eteenpäin logaritmi pysyy enintään {points} prosenttiyksikön päässä mediaanista. "
        else:
            within = f"From day 14 on, the logarithm stays within {points} percentage points of the median. "
    if _fi(lang):
        lead = "Logaritmin poikkeama on näistä pienin. " if smallest else ""
        paragraphs = [(
            f"Mediaaniin sovitettu logaritmi on ln(1 + t/{c}) / ln(1 + 183/{c}). "
            "Jokainen käyrä alkaa nollasta ja päättyy ykköseen päivänä 183. "
            f"Vakio {c} päivää on sovitettu päiville 7–176. "
            f"Keskineliöpoikkeama näiltä päiviltä on logaritmilla {gaps[0]} ja hiipuvalla virralla {gaps[1]}. "
            f"Kun kulunut aika korotetaan potenssiin {gaps[3]}, poikkeama on {gaps[2]}. "
            f"Neliöjuurella se on {gaps[4]} ja tasaisella päivävauhdilla {gaps[5]}. "
            f"{lead}"
            "Hiipuvassa virrassa vielä tulevat allekirjoitukset jakautuvat jäljellä oleville päiville "
            "suunnilleen samalla todennäköisyydellä, "
            f"ja päivävauhti puolittuu {half} päivän välein. "
            f"Logaritmissa päivävauhti vähenee kuten 1/(t + {c}). "
            "Vauhti puolittuu, kun ikä ja vakio yhdessä kaksinkertaistuvat, "
            f"ja jokainen kaksinkertaistuminen tuo {doubling} aloitteen verkkoallekirjoituksista. "
            f"{within}"
        ).strip()]
    else:
        lead = "The logarithm has the smallest of these gaps. " if smallest else ""
        paragraphs = [(
            f"The logarithm fitted to the median is ln(1 + t/{c}) / ln(1 + 183/{c}). "
            "Each curve starts at 0 and ends at 1 on day 183. "
            f"The offset {c} days is fit on ages 7–176. "
            f"The root-mean-square gap on those days is {gaps[0]} for the logarithm, {gaps[1]} for a fading inflow, "
            f"{gaps[2]} for elapsed time to the power {gaps[3]}, {gaps[4]} for a square root "
            f"and {gaps[5]} for a constant daily rate. "
            f"{lead}"
            "In the fading inflow each signature still to come is assigned to the remaining days with roughly equal chance, "
            f"and the daily inflow halves every {half} days. "
            f"Under the logarithm the daily inflow falls as 1/(t + {c}). "
            "The inflow halves when age and the offset together double, "
            f"and each doubling adds {doubling} of the initiative's web signatures. "
            f"{within}"
        ).strip()]
    if _has(notes, "common_log_p50", "own_log_p50", "own_exp_p50", "sqrt_individual_p50"):
        common = dec(notes["common_log_p50"], lang, 3)
        own_log = dec(notes["own_log_p50"], lang, 3)
        own_exp = dec(notes["own_exp_p50"], lang, 3)
        own_root = dec(notes["sqrt_individual_p50"], lang, 3)
        if _fi(lang):
            paragraphs.append(
                "Yllä olevat poikkeamat koskevat kumulatiivisten osuuksien pisteittäistä mediaania. "
                f"Sama logaritmi jokaiselle aloitteelle jättää mediaanipoikkeamaksi {common}. "
                f"Kun kullakin aloitteella on oma vakionsa, mediaanipoikkeama on {own_log}. "
                f"Oman hiipuvan virran mediaanipoikkeama on {own_exp}. "
                f"Neliöjuuren mediaanipoikkeama on {own_root}."
            )
        else:
            paragraphs.append(
                "The gaps above are for the pointwise median of the cumulative shares. "
                f"One logarithm for every initiative leaves a median gap of {common}. "
                f"Each initiative's own offset brings the median gap to {own_log}. "
                f"Each initiative's own fading inflow leaves a median gap of {own_exp}. "
                f"A square root leaves a median gap of {own_root}."
            )
    daily_keys = tuple(f"median_daily_{age}" for age in (7, 30, 90, 150))
    slope_keys = tuple(f"median_slope_{age}" for age in (7, 30, 90, 150))
    if _has(notes, *daily_keys, *slope_keys):
        days = _fine_list([notes[key] for key in daily_keys], lang)
        rises = _fine_list([notes[key] for key in slope_keys], lang)
        if _fi(lang):
            paragraphs.append(
                f"Mediaanialoitteen osuus omista verkkoallekirjoituksistaan päivinä 7, 30, 90 ja 150 on {days}. "
                f"Osuudet vähenevät kuten 1/(t + {c}). "
                f"Samoina päivinä mediaanikäyrä nousee {rises}. "
                "Kukin nousu on noin kaksinkertainen saman päivän osuuteen verrattuna."
            )
        else:
            paragraphs.append(
                f"The median initiative's own share of its total on days 7, 30, 90 and 150 is {days}. "
                f"Those shares fall as 1/(t + {c}). "
                f"On the same days the median line rises by {rises}. "
                "Each of those rises is about twice the share on that day."
            )
    if _fi(lang):
        paragraphs.append(
            "Yksi kuvaus on löytymisaikojen laaja kirjo, jossa ihmisiä on saman verran "
            "jokaisella odotuksen kaksinkertaistumisella. "
            "Toinen on näkyvyys, joka ohenee aloitteen vanhetessa. "
            f"Kumpikin antaa päivävauhdin 1/(t + {c}), ja mediaanikäyrällä on tämä muoto."
        )
    else:
        paragraphs.append(
            "One account is a wide spread of discovery times, with similar numbers of people at each doubling of the wait. "
            "Another is visibility that thins as the initiative ages. "
            f"Either account yields an inflow proportional to 1/(t + {c}), and the median curve has that shape."
        )
    return paragraphs


def _blue_rise(report, lang: str) -> str:
    notes = _path_notes(report)
    keys = ("blue_day25", "blue_day100", "blue_day150", "blue_day160", "blue_path_150_160", "blue_own_150_160")
    if not _has(notes, *keys):
        return ""
    d25 = prose_share(notes["blue_day25"], lang)
    d100 = prose_share(notes["blue_day100"], lang)
    d150 = prose_share(notes["blue_day150"], lang)
    d160 = prose_share(notes["blue_day160"], lang)
    rise = prose_share(notes["blue_path_150_160"], lang)
    own = prose_share(notes["blue_own_150_160"], lang)
    if _fi(lang):
        return (
            f"Päivänä 25 mediaaniosuus on {d25}, päivänä 100 se on {d100}, "
            f"päivänä 150 se on {d150} ja päivänä 160 se on {d160}. "
            f"Päivästä 150 päivään 160 mediaaniosuus nousee {rise}. "
            f"Noina kymmenenä päivänä mediaanialoite lisää {own}."
        )
    return (
        f"The median share is {d25} by day 25, {d100} by day 100, "
        f"{d150} by day 150 and {d160} by day 160. "
        f"From day 150 to day 160 the median share rises by {rise}. "
        f"Over those ten days the median initiative adds {own}."
    )


def clock_before(report, lang: str) -> str:
    ordinary = _shape(report, "full_under_50k")
    if ordinary is None:
        return "Puolen vuoden otos on tyhjä." if _fi(lang) else "The six-month sample is empty."
    week = _piece(report, "full_under_50k", FIRST_WEEK)
    week_share = prose_share(week["p50"], lang) if week is not None else "—"
    notes = _path_notes(report)
    slow = ""
    if not _missing(ordinary.get("slow_start_d91")) and not _missing(ordinary.get("fast_start_d91")):
        if _fi(lang):
            slow = (
                f"Ensimmäisen viikon hitain neljännes on päivänä 91 kohdassa {prose_share(ordinary['slow_start_d91'], lang)}. "
                f"Nopein neljännes on kohdassa {prose_share(ordinary['fast_start_d91'], lang)}."
            )
        else:
            slow = (
                f"Initiatives in the slowest quarter of the first week are at {prose_share(ordinary['slow_start_d91'], lang)} by day 91. "
                f"Initiatives in the fastest quarter are at {prose_share(ordinary['fast_start_d91'], lang)}."
            )
    if _fi(lang):
        opening = (
            f"Vasemmalla ovat päättyneet aloitteet, jotka jäivät alle 50 000 verkkoallekirjoituksen "
            f"ja olivat auki vähintään 170 päivää ({initiatives(ordinary['n'], lang)}, "
            f"mediaani-ikkuna {_days(ordinary['median_length'], lang)}). "
            "Käyrä on mediaani osuus aloitteen omista verkkoallekirjoituksista. Varjostus on keskimmäinen puolikas."
        )
        census = (
            f"Aloituspäivän mediaaniosuus on {prose_share(ordinary['share_d0'], lang)}. "
            f"Ensimmäinen viikko kerryttää {week_share}. "
            f"Ensimmäiset 14 päivää kerryttävät {prose_share(ordinary['first14'], lang)}, "
            f"seuraavat 14 päivää {prose_share(ordinary['second14'], lang)} "
            f"ja viimeiset 14 päivää {prose_share(ordinary['last14'], lang)}. "
            f"Päivänä 150 mediaani on {prose_share(ordinary['share_d150'], lang)}."
        )
    else:
        opening = (
            f"The left panel is every finished initiative that stayed under 50,000 web signatures "
            f"and was open at least 170 days ({initiatives(ordinary['n'], lang)}, "
            f"median window {_days(ordinary['median_length'], lang)}). "
            "The line is the median share of that initiative's own web signatures. The band is the middle half."
        )
        census = (
            f"The median share on the start date is {prose_share(ordinary['share_d0'], lang)}. "
            f"The first week holds {week_share}. "
            f"The first fortnight holds {prose_share(ordinary['first14'], lang)}, "
            f"the second fortnight {prose_share(ordinary['second14'], lang)}, "
            f"and the last fortnight {prose_share(ordinary['last14'], lang)}. "
            f"By day 150 the median is {prose_share(ordinary['share_d150'], lang)}."
        )
    parts = [opening, _day91(ordinary, notes, lang), *_fit_paragraphs(ordinary, notes, lang), census, slow]
    return "\n\n".join(part for part in parts if part)


def clock_after(report, lang: str) -> str:
    ordinary = _shape(report, "full_under_50k")
    early = _shape(report, "early_50k")
    stayed = _shape(report, "full_50k")
    parts: list[str] = []
    if early is not None:
        if _fi(lang):
            parts.append(
                f"Oikealla ovat verkossa 50 000 ylittäneet ja jo päättyneet aloitteet. "
                f"{initiatives(early['n'], lang)} sulkeutui ennen päivää 170. "
                f"Niiden mediaani-ikkuna on {_days(early['median_length'], lang)}. "
                "Kirjatun ikkunan päätyttyä osuus pysyy ykkösenä, koska keräys on päättynyt. "
                f"Päivänä 30 niiden mediaani on {prose_share(early['share_d30'], lang)}. "
                f"Ensimmäiset 14 päivää kerryttävät {prose_share(early['first14'], lang)}, "
                f"ja oman ikkunan puolivälissä niillä on {prose_share(early['share_half_own'], lang)}."
            )
        else:
            parts.append(
                f"The right panel is the initiatives that reached 50,000 on the web and have finished. "
                f"{initiatives(early['n'], lang)} closed before day 170. "
                f"Their median window is {_days(early['median_length'], lang)}. "
                "After the recorded end the share stays at 1, because collection has stopped. "
                f"Their median is {prose_share(early['share_d30'], lang)} by day 30. "
                f"The first fortnight holds {prose_share(early['first14'], lang)}, "
                f"and halfway through their own window they have {prose_share(early['share_half_own'], lang)}."
            )
        if not _missing(early["cross_p50"]):
            if _fi(lang):
                parts.append(
                    f"Niiden mediaani ylittää 50 000 päivänä {int(round(float(early['cross_p50'])))}, "
                    f"ja tuona päivänä niillä on {prose_share(early['share_at_cross'], lang)} "
                    "siitä, minkä ne lopulta kirjaavat."
                )
            else:
                parts.append(
                    f"Their median crosses 50,000 on day {int(round(float(early['cross_p50'])))}, "
                    f"holding {prose_share(early['share_at_cross'], lang)} of the signatures they eventually record."
                )
    if stayed is not None:
        band = ""
        if not _missing(stayed["share_d91_p25"]):
            span = prose_range(stayed["share_d91_p25"], stayed["share_d91_p75"], lang)
            band = f" (keskimmäinen puolikas {span})" if _fi(lang) else f" (middle half {span})"
        if _fi(lang):
            parts.append(
                f"{initiatives(stayed['n'], lang)} pysyi auki vähintään 170 päivää. "
                f"Päivänä 91 niiden mediaani on {prose_share(stayed['share_d91'], lang)}{band}. "
                f"Ensimmäiset 14 päivää kerryttävät {prose_share(stayed['first14'], lang)}. "
                f"Päivinä 170–183 mediaanialoite lisää {prose_share(stayed['last14'], lang)}."
            )
        else:
            parts.append(
                f"{initiatives(stayed['n'], lang)} stayed open at least 170 days. "
                f"At day 91 their median is {prose_share(stayed['share_d91'], lang)}{band}. "
                f"Their first fortnight holds {prose_share(stayed['first14'], lang)}. "
                f"Over days 170–183 the median initiative adds {prose_share(stayed['last14'], lang)}."
            )
        blue = _blue_rise(report, lang)
        if blue:
            parts.append(blue)
        if not _missing(stayed["cross_p50"]):
            if _fi(lang):
                parts.append(
                    f"Niiden mediaani ylittää 50 000 päivänä {int(round(float(stayed['cross_p50'])))}, "
                    f"ja tuona päivänä niillä on {prose_share(stayed['share_at_cross'], lang)} "
                    "lopullisesta verkkosummasta."
                )
            else:
                parts.append(
                    f"Their median crosses 50,000 on day {int(round(float(stayed['cross_p50'])))}, "
                    f"holding {prose_share(stayed['share_at_cross'], lang)} of the signatures they eventually record."
                )
    if ordinary is not None and early is not None:
        if _fi(lang):
            parts.append(
                f"Aikaiset sulkeutumiset ovat oman lyhyemmän ikkunansa puolivälissä keränneet "
                f"{prose_share(early['share_half_own'], lang)}. "
                f"Täyden puolen vuoden aloite, joka jää alle 50 000, on päivänä 91 kohdassa "
                f"{prose_share(ordinary['share_d91'], lang)}. "
                "Sama muoto, eri pituisessa ikkunassa."
            )
        else:
            parts.append(
                f"Halfway through their own shorter window the early closures have collected "
                f"{prose_share(early['share_half_own'], lang)}. "
                f"A full six-month initiative that stays under 50,000 has collected "
                f"{prose_share(ordinary['share_d91'], lang)} by day 91. "
                "Same shape, different length of window."
            )
    if len(report.time_to_50k):
        timing = report.time_to_50k.iloc[0]
        early_n = int(early["n"]) if early is not None else 0
        stayed_n = int(stayed["n"]) if stayed is not None else 0
        open_n = int(timing["n_web"]) - early_n - stayed_n
        if _fi(lang):
            sentence = (
                f"{initiatives(timing['n_web'], lang)} yltää verkossa 50 000:een. "
                f"Niiden ylitysiän mediaani on päivä {num(timing['p50_age'], lang)}. "
                f"{prose_share(timing['share_on_day_0'], lang)} ylittää aloituspäivänä, "
                f"{prose_share(timing['share_within_7_days'], lang)} seitsemässä päivässä "
                f"ja {prose_share(timing['share_within_30_days'], lang)} 30 päivässä."
            )
            if open_n > 0:
                sentence += (
                    f" {initiatives(open_n, lang)} on yhä ikkunansa sisällä ja jo yli 50 000 verkossa, "
                    "joten ne ovat ylitysluvussa. Kuva näyttää päättyneet ikkunat."
                )
            sentence += (
                f" {initiatives(timing['n_total_50k_without_50k_web'], lang)} "
                f"{num(timing['n_total_50k'], lang)} aloitteesta, joiden julkinen kokonaismäärä on vähintään 50 000, "
                "jää verkossa alle 50 000 allekirjoituksen. Kirjattu päättymispäivä on päivä, jona keräys loppui."
            )
        else:
            web_verb = "reaches" if int(timing["n_web"]) == 1 else "reach"
            sentence = (
                f"{initiatives(timing['n_web'], lang)} {web_verb} 50,000 on the web. "
                f"Their median crossing is day {num(timing['p50_age'], lang)}. "
                f"{prose_share(timing['share_on_day_0'], lang)} cross on the start date, "
                f"{prose_share(timing['share_within_7_days'], lang)} within seven days "
                f"and {prose_share(timing['share_within_30_days'], lang)} within 30 days."
            )
            if open_n > 0:
                sentence += (
                    f" {initiatives(open_n, lang)} are still inside their window and already at 50,000 on the web, "
                    "so they sit in the crossing count. The figure shows finished windows."
                )
            paper_n = int(timing["n_total_50k_without_50k_web"])
            stay = "stays" if paper_n == 1 else "stay"
            sentence += (
                f" {initiatives(paper_n, lang)} of the "
                f"{num(timing['n_total_50k'], lang)} with a public total of at least 50,000 "
                f"{stay} under 50,000 on the web series. The recorded end date is the day collection stopped."
            )
        parts.append(sentence)
    return "\n\n".join(parts)


def _ends(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    high = frame.loc[frame["equal_median"].idxmax()]
    low = frame.loc[frame["equal_median"].idxmin()]
    return high, low


def dow_main(report, lang: str) -> str:
    if report.dow is None or len(report.dow) == 0:
        return ""
    frame = report.dow.set_index("weekday", drop=False)
    high, low = _ends(report.dow)
    n_initiatives = int(report.dow["n_campaigns"].iloc[0]) if len(report.dow) else 0
    n_blocks = int(report.dow["n_blocks"].iloc[0]) if len(report.dow) else 0
    thu = frame.loc["Thursday"] if "Thursday" in frame.index else None
    bits = []
    if "model_multiplier" in frame.columns and not _missing(high.get("model_multiplier")) and not _missing(low.get("model_multiplier")):
        if _fi(lang):
            bits.append(
                f"Regressio, jossa vaste on 1 plus päivän lukumäärä, antaa "
                f"{weekday_genitive(high['weekday'], lang)} kertoimeksi {dec(high['model_multiplier'], lang)}"
                f"{piv(high.get('model_lo'), high.get('model_hi'), lang)} "
                f"ja {weekday_genitive(low['weekday'], lang)} kertoimeksi {dec(low['model_multiplier'], lang)}"
                f"{piv(low.get('model_lo'), low.get('model_hi'), lang)}."
            )
        else:
            bits.append(
                f"The regression on 1 plus the day's count puts {high['weekday']} at "
                f"{dec(high['model_multiplier'], lang)}{piv(high.get('model_lo'), high.get('model_hi'), lang)} "
                f"and {low['weekday']} at {dec(low['model_multiplier'], lang)}"
                f"{piv(low.get('model_lo'), low.get('model_hi'), lang)}."
            )
    if thu is not None and high["weekday"] != "Thursday" and low["weekday"] != "Thursday":
        if _fi(lang):
            thu_bit = (
                f"Torstai on mediaaniviikossa {dec(thu['equal_median'], lang)}"
                f"{piv(thu['equal_median_lo'], thu['equal_median_hi'], lang)} "
                f"ja allekirjoituksilla painotettuna {dec(thu['signature_weighted'], lang)}"
                f"{piv(thu['signature_lo'], thu['signature_hi'], lang)}."
            )
        else:
            thu_bit = (
                f"Thursday is {dec(thu['equal_median'], lang)}"
                f"{piv(thu['equal_median_lo'], thu['equal_median_hi'], lang)} in the median initiative-week "
                f"and {dec(thu['signature_weighted'], lang)}"
                f"{piv(thu['signature_lo'], thu['signature_hi'], lang)} when weeks are weighted by signatures."
            )
        if _includes_one(thu["signature_lo"], thu["signature_hi"]):
            thu_bit += (
                " Raskaat viikot eivät nosta torstaita esiin niin kuin tyypillinen viikko."
                if _fi(lang)
                else " Heavy weeks do not single out Thursday the way the typical week does."
            )
        bits.append(thu_bit)
    if _fi(lang):
        opening = (
            f"Vuodesta 2017 katkaisuun, seitsemännestä päivästä eteenpäin, "
            f"{initiatives(n_initiatives, lang)} ja {num(n_blocks, lang)} seitsemän päivän lohkoa. "
            "Lohko, jossa on pyhäpäivä tai nimetty aatto, jää pois. "
            "Mediaani antaa kullekin aloiteviikolle saman painon. "
            f"{weekday_name(high['weekday'], lang).capitalize()} on {dec(high['equal_median'], lang)}"
            f"{piv(high['equal_median_lo'], high['equal_median_hi'], lang)} kertaa viikon tasajako. "
            f"{weekday_name(low['weekday'], lang).capitalize()} on {dec(low['equal_median'], lang)}"
            f"{piv(low['equal_median_lo'], low['equal_median_hi'], lang)}."
        )
    else:
        opening = (
            f"From 2017 through the cutoff, after day 7, {initiatives(n_initiatives, lang)} "
            f"and {num(n_blocks, lang)} seven-day blocks. "
            "A block that contains a statutory holiday or a named eve is left out. "
            "The median gives each initiative-week the same weight. "
            f"{high['weekday']} is {dec(high['equal_median'], lang)}"
            f"{piv(high['equal_median_lo'], high['equal_median_hi'], lang)} times a fair share of that week. "
            f"{low['weekday']} is {dec(low['equal_median'], lang)}"
            f"{piv(low['equal_median_lo'], low['equal_median_hi'], lang)}."
        )
    return " ".join([opening, *bits])


def dow_sensitivity_line(report, lang: str) -> str:
    frame = report.dow_sensitivity
    stable = True
    if len(frame):
        for _, part in frame.groupby("sample"):
            if part.loc[part["equal_median"].idxmax(), "weekday"] != "Wednesday":
                stable = False
            if part.loc[part["equal_median"].idxmin(), "weekday"] != "Saturday":
                stable = False
    if _fi(lang):
        text = "Sama mediaani kolmella otosrajalla."
        if stable and len(frame):
            text += " Keskiviikko on jokaisen rivin korkein päivä ja lauantai matalin."
        return text
    text = "The same median under three sample rules."
    if stable and len(frame):
        text += " Wednesday is the high day in each row and Saturday the low day."
    return text


def dow_era(report, lang: str) -> str:
    def pick(era: str, day: str) -> pd.Series | None:
        hit = report.dow_era[(report.dow_era["era"] == era) & (report.dow_era["weekday"] == day)]
        return None if hit.empty else hit.iloc[0]

    pairs = []
    for era in ("Through 2016", "2017 onward"):
        sat, wed = pick(era, "Saturday"), pick(era, "Wednesday")
        if sat is None or wed is None:
            continue
        pairs.append((era, sat, wed))
    if not pairs:
        return ""
    if _fi(lang):
        lead_in = {"Through 2016": "Vuoteen 2016 saakka", "2017 onward": "Vuodesta 2017"}
        sentences = []
        for era, sat, wed in pairs:
            sentences.append(
                f"{lead_in.get(era, era_name(era, lang))} lauantai on {dec(sat['equal_median'], lang)}"
                f"{piv(sat['equal_median_lo'], sat['equal_median_hi'], lang)} "
                f"ja keskiviikko {dec(wed['equal_median'], lang)}"
                f"{piv(wed['equal_median_lo'], wed['equal_median_hi'], lang)}."
            )
        return " ".join(sentences)
    sentences = []
    for era, sat, wed in pairs:
        sentences.append(
            f"{era_name(era, lang)}, Saturday is {dec(sat['equal_median'], lang)}"
            f"{piv(sat['equal_median_lo'], sat['equal_median_hi'], lang)} "
            f"and Wednesday is {dec(wed['equal_median'], lang)}"
            f"{piv(wed['equal_median_lo'], wed['equal_median_hi'], lang)}."
        )
    return " ".join(sentences)


def _sample_2017(report) -> pd.Series | None:
    return _row(report.samples, "sample", "2017 onward, age ≥ 7, ≥ 100 web signatures")


def month_main(report, lang: str) -> str:
    sample = _sample_2017(report)
    high = report.month.loc[report.month["multiplier"].idxmax()] if len(report.month) else None
    low = report.month.loc[report.month["multiplier"].idxmin()] if len(report.month) else None
    if _fi(lang):
        opening = (
            "Kerroin pitää aloitteen, aloitteen viikon ja viikon sisäisen päivän vakiona, "
            "ja se luetaan luvusta 1 plus päivän lukumäärä. "
            "Ykköstä suurempi kerroin on tavallista avointa päivää vilkkaampi. "
            "Aloitteen viikko vie nuoruuden, joten kuukausi tulee aloitteista, jotka ovat samanikäisiä eri kuukausina."
        )
        if sample is not None:
            opening += (
                f" Otos on {initiatives(sample['campaigns'], lang)} ja "
                f"{num(sample['campaign_days'], lang)} avointa päivää vuodesta 2017."
            )
    else:
        opening = (
            "The multiplier holds the initiative, the week of the initiative and the day inside that week fixed, "
            "and it is read on 1 plus the day's count. "
            "A value above 1 is a day above the average open day in the sample. "
            "The week of the initiative absorbs how young it is, so the month comes from initiatives that reach the same age in different months."
        )
        if sample is not None:
            opening += (
                f" The sample is {initiatives(sample['campaigns'], lang)} and "
                f"{num(sample['campaign_days'], lang)} open days from 2017 onward."
            )
    if high is None or low is None:
        return opening
    if _fi(lang):
        return (
            opening + " "
            f"{month_name(high['level'], lang).capitalize()} on {dec(high['multiplier'], lang)}"
            f"{piv(high['multiplier_lo'], high['multiplier_hi'], lang)}. "
            f"{month_name(low['level'], lang).capitalize()} on {dec(low['multiplier'], lang)}"
            f"{piv(low['multiplier_lo'], low['multiplier_hi'], lang)}."
        )
    return (
        opening + " "
        f"{high['level']} is {dec(high['multiplier'], lang)}{piv(high['multiplier_lo'], high['multiplier_hi'], lang)}. "
        f"{low['level']} is {dec(low['multiplier'], lang)}{piv(low['multiplier_lo'], low['multiplier_hi'], lang)}."
    )


def month_era(report, lang: str) -> str:
    def pick(era: str, level: str) -> pd.Series | None:
        hit = report.month_era[(report.month_era["era"] == era) & (report.month_era["level"] == level)]
        return None if hit.empty else hit.iloc[0]

    june_pre, june_post = pick("Through 2016", "June"), pick("2017 onward", "June")
    dec_pre, dec_post = pick("Through 2016", "December"), pick("2017 onward", "December")
    if june_pre is None or june_post is None or dec_pre is None or dec_post is None:
        return ""
    if _fi(lang):
        text = (
            f"Kesäkuu on {dec(june_pre['multiplier'], lang)}{piv(june_pre['multiplier_lo'], june_pre['multiplier_hi'], lang)} "
            f"vuoteen 2016 ja {dec(june_post['multiplier'], lang)}{piv(june_post['multiplier_lo'], june_post['multiplier_hi'], lang)} "
            f"vuodesta 2017. "
            f"Joulukuu on {dec(dec_pre['multiplier'], lang)}{piv(dec_pre['multiplier_lo'], dec_pre['multiplier_hi'], lang)} "
            f"vuoteen 2016 ja {dec(dec_post['multiplier'], lang)}{piv(dec_post['multiplier_lo'], dec_post['multiplier_hi'], lang)} "
            "vuodesta 2017."
        )
    else:
        text = (
            f"June is {dec(june_pre['multiplier'], lang)}{piv(june_pre['multiplier_lo'], june_pre['multiplier_hi'], lang)} "
            f"through 2016 and {dec(june_post['multiplier'], lang)}{piv(june_post['multiplier_lo'], june_post['multiplier_hi'], lang)} "
            f"from 2017. "
            f"December is {dec(dec_pre['multiplier'], lang)}{piv(dec_pre['multiplier_lo'], dec_pre['multiplier_hi'], lang)} "
            f"through 2016 and {dec(dec_post['multiplier'], lang)}{piv(dec_post['multiplier_lo'], dec_post['multiplier_hi'], lang)} "
            "from 2017."
        )
    if _below_one(june_pre) and _below_one(june_post):
        text += " Kesäkuu on matala molemmissa jaksoissa." if _fi(lang) else " June is low in both eras."
    if _below_one(dec_pre) and _includes_one(dec_post["multiplier_lo"], dec_post["multiplier_hi"]):
        text += (
            " Joulukuu on matala vuoteen 2016. Vuodesta 2017 sen väli sisältää ykkösen."
            if _fi(lang)
            else " December is low through 2016. From 2017 its interval includes 1."
        )
    return text


def _week_row(report, number: int) -> pd.Series | None:
    if report.week is None or len(report.week) == 0:
        return None
    label = f"{number:02d}"
    hit = report.week[report.week["level"].astype(str) == label]
    return None if hit.empty else hit.iloc[0]


def _week_phrase(row: pd.Series, lang: str) -> str:
    if _fi(lang):
        return (
            f"{int(row['level'])}, kerroin {dec(row['multiplier'], lang)}"
            f"{piv(row['multiplier_lo'], row['multiplier_hi'], lang)}"
        )
    return (
        f"{int(row['level'])} at {dec(row['multiplier'], lang)}"
        f"{piv(row['multiplier_lo'], row['multiplier_hi'], lang)}"
    )


def week_note(report, lang: str) -> str:
    frame = report.week
    if frame is None or len(frame) == 0:
        return ""
    main = frame[~frame["level"].astype(str).isin(["00", "53"])].copy()
    if main.empty:
        main = frame
    low_value = float(main["multiplier"].min())
    near = main[main["multiplier"] <= low_value + 0.02].sort_values("level")
    high = main.loc[main["multiplier"].idxmax()]
    low_bits = [_week_phrase(row, lang) for _, row in near.iterrows()]
    ski = [_week_row(report, number) for number in (8, 9, 10)]
    ski = [row for row in ski if row is not None]
    autumn = _week_row(report, 42)
    week_53 = _week_row(report, 53)
    if _fi(lang):
        text = (
            f"Matalimmat ISO-viikot ovat {_join(low_bits, lang)}. "
            f"Korkea viikko on {int(high['level'])}, kerroin {dec(high['multiplier'], lang)}"
            f"{piv(high['multiplier_lo'], high['multiplier_hi'], lang)}. "
            "Hiihtoloma on kunnasta riippuen viikko 8, 9 tai 10, ja syysloma yleensä viikko 42. "
            "Ne jäävät viikoiksi."
        )
        if ski:
            joined = ", ".join(
                f"viikko {int(row['level'])} on {dec(row['multiplier'], lang)}{piv(row['multiplier_lo'], row['multiplier_hi'], lang)}"
                for row in ski
            )
            text += " " + joined[0].upper() + joined[1:] + "."
        if autumn is not None:
            text += (
                f" Viikko 42 on {dec(autumn['multiplier'], lang)}"
                f"{piv(autumn['multiplier_lo'], autumn['multiplier_hi'], lang)}."
            )
        if week_53 is not None:
            text += (
                f" Viikko 53 on {dec(week_53['multiplier'], lang)}"
                f"{piv(week_53['multiplier_lo'], week_53['multiplier_hi'], lang)}, "
                f"{num(week_53['n'], lang)} aloitepäivää."
            )
        text += (
            " Tasoitettu vuodenpäivä on jäännös sen jälkeen, kun aloite, aloitteen viikko, viikonpäivä ja pyhäluokka on poistettu. "
            "Se on 15 päivän liukuva keskiarvo, eikä sillä ole väliä."
        )
        return text
    text = (
        f"The low ISO weeks are {_join(low_bits, lang)}. "
        f"The high week is {int(high['level'])} at {dec(high['multiplier'], lang)}"
        f"{piv(high['multiplier_lo'], high['multiplier_hi'], lang)}. "
        "The ski break is week 8, 9 or 10 depending on the municipality, and the autumn break is usually week 42. "
        "They stay as weeks."
    )
    if ski:
        joined = ", ".join(
            f"week {int(row['level'])} is {dec(row['multiplier'], lang)}{piv(row['multiplier_lo'], row['multiplier_hi'], lang)}"
            for row in ski
        )
        text += " " + joined[0].upper() + joined[1:] + "."
    if autumn is not None:
        text += (
            f" Week 42 is {dec(autumn['multiplier'], lang)}"
            f"{piv(autumn['multiplier_lo'], autumn['multiplier_hi'], lang)}."
        )
    if week_53 is not None:
        text += (
            f" Week 53 is {dec(week_53['multiplier'], lang)}"
            f"{piv(week_53['multiplier_lo'], week_53['multiplier_hi'], lang)} "
            f"on {num(week_53['n'], lang)} initiative-days."
        )
    text += (
        " The smooth day-of-year line is the residual after the initiative, the week of the initiative, the weekday and the holiday class. "
        "It is a 15-day moving average, and it has no interval."
    )
    return text


def _holiday_bits(report, lang: str) -> tuple[str, str, str]:
    frame = report.holiday_named
    lows, highs, flat = [], [], []
    for _, row in frame.iterrows():
        name = holiday_name(row["level"], lang)
        bit = f"{name} {dec(row['multiplier'], lang)}{piv(row['multiplier_lo'], row['multiplier_hi'], lang)}"
        if _below_one(row):
            lows.append((float(row["multiplier"]), bit))
        elif _above_one(row):
            highs.append((float(row["multiplier"]), bit))
        else:
            flat.append(name)
    lows.sort()
    highs.sort(reverse=True)
    return _join([bit for _, bit in lows], lang), _join([bit for _, bit in highs], lang), _join(flat, lang)


def holiday_main(report, lang: str) -> str:
    if report.holiday_named is None or len(report.holiday_named) == 0:
        return ""
    lows, highs, flat = _holiday_bits(report, lang)
    if _fi(lang):
        parts = [
            "Nimetty päivä on jäännös sen jälkeen, kun aloite, aloitteen viikko, viikon sisäinen päivä, viikonpäivä ja kuukausi on vakioitu. "
            "Kerroin on luvulle 1 plus lukumäärä, siihen päivään verrattuna, jonka viikonpäivä ja kuukausi jo ennustavat."
        ]
        if lows:
            parts.append(f"Tuon päivän alle jäävät {lows}.")
        if highs:
            parts.append(f"Yläpuolelle jäävät {highs}.")
        if flat:
            parts.append(f"Väli sisältää ykkösen: {flat}.")
        return " ".join(parts)
    parts = [
        "A named day is the residual after the initiative, the week of the initiative, the day inside that week, the weekday and the month. "
        "The multiplier is on 1 plus the count, against the day the weekday and the month already imply."
    ]
    if lows:
        parts.append(f"Below that day: {lows}.")
    if highs:
        parts.append(f"Above that day: {highs}.")
    if flat:
        parts.append(f"Intervals that include 1: {flat}.")
    return " ".join(parts)


def holiday_pool(report, lang: str) -> str:
    frame = report.holiday_pooled
    if frame is None or len(frame) == 0:
        return ""
    public = _row(frame, "level", "Public holiday")
    eve = _row(frame, "level", "Eve")
    if public is None or eve is None:
        return ""
    if _fi(lang):
        return (
            f"Jos kaikki lakisääteiset pyhät pannaan yhteen kertoimeen, tulos on {dec(public['multiplier'], lang)}"
            f"{piv(public['multiplier_lo'], public['multiplier_hi'], lang)} tavalliseen päivään verrattuna, "
            f"ja neljä aattoa antavat {dec(eve['multiplier'], lang)}{piv(eve['multiplier_lo'], eve['multiplier_hi'], lang)}. "
            "Yhdistelmä sekoittaa keskenään erilaisia päiviä. Nimetty rivi on estimaatti."
        )
    return (
        f"Pooling every statutory holiday into one coefficient gives {dec(public['multiplier'], lang)}"
        f"{piv(public['multiplier_lo'], public['multiplier_hi'], lang)} against an ordinary day, "
        f"and the four eves give {dec(eve['multiplier'], lang)}{piv(eve['multiplier_lo'], eve['multiplier_hi'], lang)}. "
        "The pool mixes unlike days. The named rows are the estimate."
    )


def holiday_summer(report, lang: str) -> str:
    if report.interaction is None or len(report.interaction) == 0:
        return ""
    row = report.interaction.iloc[0]
    if _fi(lang):
        return (
            f"Kesä-, heinä- ja elokuun viikonlopuilla on lisäkerroin {dec(row['multiplier'], lang)}"
            f"{piv(row['multiplier_lo'], row['multiplier_hi'], lang)} "
            f"kyseisen kuukauden ja viikonpäivän päällä, {num(row['n'], lang)} aloitepäivää."
        )
    return (
        f"June, July and August weekends carry an extra multiplier of {dec(row['multiplier'], lang)}"
        f"{piv(row['multiplier_lo'], row['multiplier_hi'], lang)} "
        f"on top of that month and that weekday, on {num(row['n'], lang)} initiative-days."
    )


def dom(report, lang: str) -> str:
    if report.dom is None or len(report.dom) == 0:
        return ""
    low = report.dom.loc[report.dom["multiplier"].idxmin()]
    high = report.dom.loc[report.dom["multiplier"].idxmax()]
    if _fi(lang):
        return (
            "Kuukauden sisällä kuukaudenpäivän kertoimet kulkevat "
            f"{dec(low['multiplier'], lang)}{piv(low['multiplier_lo'], low['multiplier_hi'], lang)} "
            f"päivänä {low['level']} ja "
            f"{dec(high['multiplier'], lang)}{piv(high['multiplier_lo'], high['multiplier_hi'], lang)} "
            f"päivänä {high['level']}."
        )
    return (
        "Inside each month, the day-of-month multipliers run from "
        f"{dec(low['multiplier'], lang)}{piv(low['multiplier_lo'], low['multiplier_hi'], lang)} on day {low['level']} "
        f"to {dec(high['multiplier'], lang)}{piv(high['multiplier_lo'], high['multiplier_hi'], lang)} on day {high['level']}."
    )


def structure(report, lang: str) -> str:
    summary = report.summary
    if _fi(lang):
        return (
            f"Katkaisuun mennessä alkaneista {num(summary['started_campaigns'], lang)} aloitteesta "
            f"{prose_share(summary['share_campaigns_under_100'], lang)} jää alle 100 verkkoallekirjoituksen. "
            f"Kymmenen suurinta on {prose_share(summary['top10_web_share'], lang)} verkkoallekirjoituksista. "
            f"Suurin prosentti aloitteista on {prose_share(summary['top1pct_web_share'], lang)}. "
            f"Verkkolaskurin Gini-kerroin on {dec(summary['gini_web'], lang, 2)}."
        )
    return (
        f"Of {num(summary['started_campaigns'], lang)} initiatives started by the cutoff, "
        f"{prose_share(summary['share_campaigns_under_100'], lang)} have fewer than 100 web signatures. "
        f"The largest ten are {prose_share(summary['top10_web_share'], lang)} of web signatures. "
        f"The largest 1 percent of initiatives are {prose_share(summary['top1pct_web_share'], lang)}. "
        f"The Gini coefficient on the web counter is {dec(summary['gini_web'], lang, 2)}."
    )


def launch(report, lang: str) -> str:
    frame = report.launch
    if frame is None or len(frame) == 0:
        return ""
    common = frame.loc[frame["share"].idxmax()]
    small = frame.loc[frame["median_web"].idxmin()]
    if _fi(lang):
        text = (
            f"{weekday_name(common['weekday'], lang).capitalize()} on yleisin aloituspäivä, "
            f"{prose_share(common['share'], lang)} aloitteista, mediaani verkossa {num(common['median_web'], lang)}."
        )
        if small["weekday"] != common["weekday"]:
            text += (
                f" {weekday_genitive(small['weekday'], lang).capitalize()} mediaani verkossa on "
                f"{num(small['median_web'], lang)} ({prose_share(small['share'], lang)} aloituksista)."
            )
        text += " Viikonpäivävertailu alkaa aloitteen päivästä 7, joten tämä aloituspäivien jakauma on eri lasku."
        return text
    text = (
        f"{common['weekday']} is the most common start, {prose_share(common['share'], lang)} of initiatives, "
        f"with median web {num(common['median_web'], lang)}."
    )
    if small["weekday"] != common["weekday"]:
        text += (
            f" {small['weekday']}'s median web is {num(small['median_web'], lang)} "
            f"({prose_share(small['share'], lang)} of starts)."
        )
    text += " The weekday ratios start at day 7 of the initiative, so this mix of start dates is a separate count."
    return text


def checks_intro(report, lang: str) -> str:
    if _fi(lang):
        return "Estimoinneissa käytetyt otokset. Varoitus tarkoittaa, että asetelma ei ollut täyttä astetta tai oli huonosti ehdollistunut."
    return "Samples used in the fits. A warning means a design was short of full rank or poorly conditioned."


def checks_warnings(report, lang: str) -> str:
    if report.warnings:
        return "\n".join(f"- {warning}" for warning in report.warnings)
    return "Ei aste- tai ehdollisuusvaroituksia." if _fi(lang) else "No rank or conditioning warnings."


def checks_tail(report, lang: str) -> str:
    if report.n_boot == 0:
        if _fi(lang):
            return "Tässä ajossa ei ole uusintaottoja, joten välit ovat tyhjiä."
        return "This run has no bootstrap draws, so the intervals are blank."
    if _fi(lang):
        return (
            f"Välit ovat {num(report.n_boot, lang)} koko aloitteen noston 2,5. ja 97,5. prosenttipistettä. "
            f"Siemen on kiinnitetty. Jaksojen raja on {BREAK.date().isoformat()}, "
            "vuosi, jonka volyymivertailu nimeää."
        )
    return (
        f"Intervals are the 2.5 and 97.5 percentiles of {num(report.n_boot, lang)} draws of whole initiatives. "
        f"The seed is fixed. The era split is {BREAK.date().isoformat()}, the year named in the volume comparison."
    )
