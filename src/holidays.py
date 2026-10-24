"""Finnish statutory holidays and four eves, for the kansalaisaloite era.

Epiphany stays on 6 January. Ascension, Good Friday and the Easter days follow
the Gregorian computus. Midsummer Day and All Saints' Day are the statutory
Saturdays. The four eves are not statutory holidays. They are the days when
offices still shut: May Day eve, Midsummer eve, Christmas Eve, New Year's Eve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


NAMES_FI = {
    "New Year's Day": "uudenvuodenpäivä",
    "Epiphany": "loppiainen",
    "Good Friday": "pitkäperjantai",
    "Easter Sunday": "pääsiäispäivä",
    "Easter Monday": "2. pääsiäispäivä",
    "May Day": "vappu",
    "Ascension": "helatorstai",
    "Midsummer Day": "juhannuspäivä",
    "All Saints' Day": "pyhäinpäivä",
    "Independence Day": "itsenäisyyspäivä",
    "Christmas Day": "joulupäivä",
    "Boxing Day": "tapaninpäivä",
    "May Day eve": "vappuaatto",
    "Midsummer eve": "juhannusaatto",
    "Christmas Eve": "jouluaatto",
    "New Year's Eve": "uudenvuodenaatto",
}

# Calendar order used in tables. Not sorted by effect size.
NAMED_ORDER = (
    "New Year's Day",
    "Epiphany",
    "Good Friday",
    "Easter Sunday",
    "Easter Monday",
    "May Day eve",
    "May Day",
    "Ascension",
    "Midsummer eve",
    "Midsummer Day",
    "All Saints' Day",
    "Independence Day",
    "Christmas Eve",
    "Christmas Day",
    "Boxing Day",
    "New Year's Eve",
)


@dataclass(frozen=True)
class DayKind:
    public_name: str = ""
    eve_name: str = ""

    @property
    def code(self) -> int:
        """0 ordinary, 1 statutory holiday, 2 eve. A statutory day wins if both were set."""
        if self.public_name:
            return 1
        if self.eve_name:
            return 2
        return 0

    @property
    def label(self) -> str:
        return self.public_name or self.eve_name


def easter_sunday(year: int) -> date:
    """Gregorian Easter, Anonymous computus."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _saturday_on_or_after(start: date) -> date:
    return start + timedelta(days=(5 - start.weekday()) % 7)


def midsummer_day(year: int) -> date:
    """Saturday in 20–26 June."""
    return _saturday_on_or_after(date(year, 6, 20))


def all_saints(year: int) -> date:
    """Saturday in 31 October–6 November."""
    return _saturday_on_or_after(date(year, 10, 31))


def holiday_table(years: range) -> dict[date, DayKind]:
    out: dict[date, DayKind] = {}

    def put(day: date, *, public: str = "", eve: str = "") -> None:
        prev = out.get(day, DayKind())
        out[day] = DayKind(public or prev.public_name, eve or prev.eve_name)

    for year in years:
        easter = easter_sunday(year)
        midsummer = midsummer_day(year)
        put(date(year, 1, 1), public="New Year's Day")
        put(date(year, 1, 6), public="Epiphany")
        put(easter - timedelta(days=2), public="Good Friday")
        put(easter, public="Easter Sunday")
        put(easter + timedelta(days=1), public="Easter Monday")
        put(date(year, 5, 1), public="May Day")
        put(easter + timedelta(days=39), public="Ascension")
        put(midsummer, public="Midsummer Day")
        put(all_saints(year), public="All Saints' Day")
        put(date(year, 12, 6), public="Independence Day")
        put(date(year, 12, 25), public="Christmas Day")
        put(date(year, 12, 26), public="Boxing Day")
        put(date(year, 4, 30), eve="May Day eve")
        put(midsummer - timedelta(days=1), eve="Midsummer eve")
        put(date(year, 12, 24), eve="Christmas Eve")
        put(date(year, 12, 31), eve="New Year's Eve")
    return out


def label_fi(name: str) -> str:
    fi = NAMES_FI.get(name, "")
    return f"{name} ({fi})" if fi else name
