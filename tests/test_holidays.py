from datetime import date

from src.holidays import all_saints, easter_sunday, holiday_table, midsummer_day


def test_easter_and_midsummer_known_dates():
    assert easter_sunday(2017) == date(2017, 4, 16)
    assert easter_sunday(2020) == date(2020, 4, 12)
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)
    assert midsummer_day(2024) == date(2024, 6, 22)
    assert midsummer_day(2026) == date(2026, 6, 20)
    assert all_saints(2025) == date(2025, 11, 1)
    assert all_saints(2026) == date(2026, 10, 31)


def test_public_days_and_eves_do_not_collapse():
    table = holiday_table(range(2024, 2027))
    assert table[date(2024, 12, 25)].public_name == "Christmas Day"
    assert table[date(2024, 12, 25)].code == 1
    assert table[date(2024, 12, 24)].eve_name == "Christmas Eve"
    assert table[date(2024, 12, 24)].code == 2
    assert table[date(2026, 6, 19)].eve_name == "Midsummer eve"
    assert date(2024, 6, 12) not in table
    assert table[easter_sunday(2026) + __import__("datetime").timedelta(days=39)].public_name == "Ascension"
