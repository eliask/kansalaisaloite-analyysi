# Daily signatures on kansalaisaloite.fi

The [English notebook](kansalaisaloite_en.ipynb) and the [Finnish notebook](kansalaisaloite.ipynb) are the pages. They read a cached snapshot of the public API and estimate how web signatures vary by weekday, week of the year, month, public holiday and day of the month, after each initiative is put on its own scale and initiative age is held fixed. The six-month path of a median initiative is estimated on the same snapshot.

Suomenkielinen sivu on [kansalaisaloite.ipynb](kansalaisaloite.ipynb) ja sen HTML-versio [kansalaisaloite.html](kansalaisaloite.html).

The open series is one integer per date, updated nightly. It does not contain hour of day or hour of week.

## Reproduce

Python 3.11 or newer. From this directory:

```bash
uv sync
uv run python -m src.fetch
uv run python -m src.reproduce
uv run pytest -q
```

`src.fetch` writes `data/live/<stamp>/` and points `data/live/LATEST` at it only when every initiative page and every daily series has been stored. The notebooks read that snapshot and do not call the network. `src.reproduce` executes both notebooks and writes `reports/` plus the HTML pages. The first notebook fits the models. The second loads that fit from `reports/report.pkl` and redraws the figures. Bootstrap draws default to 60 whole initiatives. `src.reproduce --boot 0` writes the point estimates without intervals.

The API documentation is [kansalaisaloite.fi/api](https://www.kansalaisaloite.fi/api). Daily counts are `GET /api/v1/supports-by-date/{id}`.
