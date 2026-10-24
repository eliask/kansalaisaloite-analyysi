import pandas as pd

from src.figures import clock


def test_clock_draws_both_languages(tmp_path):
    rows = []
    for group in ("full_under_50k", "full_50k", "early_50k"):
        for age in range(184):
            rows.append({"group": group, "age": age, "p25": 0.2, "p50": age / 183, "p75": 0.9, "n": 10})
    curve = pd.DataFrame(rows)
    for lang in ("en", "fi"):
        path = tmp_path / f"{lang}.png"
        clock(curve, path, lang)
        assert path.stat().st_size > 1000
