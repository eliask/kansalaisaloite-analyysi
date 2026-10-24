"""Within-campaign calendar contrasts.

One demean by initiative removes each campaign's level. Age week, the day
inside that week, and the calendar factors then sit in one regression. The
day-inside-week term is what keeps a campaign's own decline from being read
as a weekday effect. That separation exists only because campaigns do not all
start on the same weekday.

Coefficients of a factor are shifted so the average open day in the sample is
zero. ``exp(coef)`` multiplies ``1 + count`` relative to that average day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Factor:
    name: str
    codes: np.ndarray
    labels: list[str]
    report: bool = True


@dataclass
class Extra:
    name: str
    column: np.ndarray
    report: bool = True


@dataclass
class Fit:
    name: str
    coefficients: pd.DataFrame
    n_rows: int
    n_campaigns: int
    rank: int
    n_columns: int
    condition: float
    warnings: list[str] = field(default_factory=list)
    residual: np.ndarray | None = None
    draws: np.ndarray | None = None


def _present_factors(factors: list[Factor]) -> list[tuple[Factor, np.ndarray, np.ndarray]]:
    """Drop levels that never occur. Return (factor, column codes 0..L-1, labels)."""
    encoded = []
    for factor in factors:
        values = np.unique(factor.codes)
        mapping = np.full(int(values.max()) + 1 if len(values) else 1, -1, dtype=np.int32)
        mapping[values] = np.arange(len(values), dtype=np.int32)
        labels = [factor.labels[int(value)] for value in values]
        encoded.append((factor, mapping[factor.codes], np.array(values), labels))
    return encoded


def _design(encoded, extras: list[Extra]) -> tuple[np.ndarray, list[dict]]:
    pieces: list[np.ndarray] = []
    columns: list[dict] = []
    n = len(encoded[0][1]) if encoded else len(extras[0].column)
    for factor, codes, _values, labels in encoded:
        n_levels = len(labels)
        if n_levels < 2:
            continue
        width = n_levels - 1
        dummies = np.zeros((n, width), dtype=np.float64)
        rows = np.flatnonzero(codes > 0)
        dummies[rows, codes[rows] - 1] = 1.0
        start = 0 if not pieces else sum(piece.shape[1] for piece in pieces)
        pieces.append(dummies)
        columns.append(
            {
                "kind": "factor",
                "name": factor.name,
                "report": factor.report,
                "labels": labels,
                "codes": codes,
                "start": start,
                "width": width,
            }
        )
    for extra in extras:
        start = 0 if not pieces else sum(piece.shape[1] for piece in pieces)
        column = np.asarray(extra.column, dtype=np.float64).reshape(-1, 1)
        pieces.append(column)
        columns.append(
            {
                "kind": "extra",
                "name": extra.name,
                "report": extra.report,
                "start": start,
                "width": 1,
                "n": int(np.count_nonzero(column[:, 0])),
            }
        )
    if not pieces:
        return np.zeros((n, 0), dtype=np.float64), columns
    return np.concatenate(pieces, axis=1), columns


def demean(y: np.ndarray, x: np.ndarray, cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(cluster).astype(np.float64)
    counts[counts == 0] = np.nan
    y_mean = np.bincount(cluster, weights=y) / counts
    y_dm = y - y_mean[cluster]
    if x.shape[1] == 0:
        return y_dm, x
    x_dm = np.empty_like(x)
    for column in range(x.shape[1]):
        mean = np.bincount(cluster, weights=x[:, column]) / counts
        x_dm[:, column] = x[:, column] - mean[cluster]
    return y_dm, x_dm


def _center_factor(beta: np.ndarray, codes: np.ndarray, n_levels: int) -> np.ndarray:
    full = np.zeros(n_levels, dtype=np.float64)
    if n_levels > 1:
        full[1:] = beta
    counts = np.bincount(codes, minlength=n_levels).astype(np.float64)
    present = counts > 0
    if present.any():
        full[present] -= np.average(full[present], weights=counts[present])
    full[~present] = np.nan
    return full


def _unpack(beta: np.ndarray, columns: list[dict], cluster: np.ndarray) -> list[dict]:
    rows = []
    for spec in columns:
        sl = slice(spec["start"], spec["start"] + spec["width"])
        piece = beta[sl]
        if spec["kind"] == "extra":
            rows.append(
                {
                    "factor": spec["name"],
                    "level": spec["name"],
                    "report": spec["report"],
                    "coef": float(piece[0]),
                    "n": int(spec.get("n", 0)),
                    "n_campaigns": int(len(np.unique(cluster))),
                    "centered": False,
                }
            )
            continue
        centered = _center_factor(piece, spec["codes"], len(spec["labels"]))
        counts = np.bincount(spec["codes"], minlength=len(spec["labels"]))
        for level, coef, count in zip(spec["labels"], centered, counts, strict=True):
            level_index = spec["labels"].index(level)
            rows.append(
                {
                    "factor": spec["name"],
                    "level": level,
                    "report": spec["report"],
                    "coef": float(coef) if np.isfinite(coef) else np.nan,
                    "n": int(count),
                    "n_campaigns": int(len(np.unique(cluster[spec["codes"] == level_index]))),
                    "centered": True,
                }
            )
    return rows


def _fit_once(y: np.ndarray, x: np.ndarray, cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    y_dm, x_dm = demean(y, x, cluster)
    if x_dm.shape[1] == 0:
        return np.zeros(0), y_dm, 0, np.nan
    beta, _residuals, rank, singular = np.linalg.lstsq(x_dm, y_dm, rcond=None)
    positive = singular[singular > 0]
    condition = float(positive.max() / positive.min()) if len(positive) else np.inf
    residual = y_dm - x_dm @ beta
    return beta, residual, int(rank), condition


def residual_group_means(
    y: np.ndarray,
    cluster: np.ndarray,
    factors: list[Factor],
    group: np.ndarray,
    labels: dict[int, str],
    *,
    n_boot: int = 0,
    seed: int = 0,
    name: str = "residual-means",
) -> Fit:
    """Mean within-campaign residual for each group code, with a cluster bootstrap."""
    y = np.asarray(y, dtype=np.float64)
    cluster = _remap_clusters(np.asarray(cluster))
    group = np.asarray(group, dtype=np.int32)
    encoded = _present_factors(factors)
    aligned = [Factor(factor.name, codes, labs, factor.report) for factor, codes, _values, labs in encoded]
    x, _columns = _design([(factor, factor.codes, None, factor.labels) for factor in aligned], [])
    _beta, residual, rank, condition = _fit_once(y, x, cluster)
    codes = sorted(code for code in labels if np.any(group == code))
    point = np.array([residual[group == code].mean() if np.any(group == code) else np.nan for code in codes])
    counts = np.array([int(np.count_nonzero(group == code)) for code in codes])
    campaign_counts = np.array([int(len(np.unique(cluster[group == code]))) for code in codes])
    draws = np.empty((n_boot, len(codes)), dtype=np.float64)
    if n_boot:
        rng = np.random.default_rng(seed)
        n_clusters = int(cluster.max()) + 1
        order = np.argsort(cluster, kind="mergesort")
        lengths = np.bincount(cluster)
        starts = lengths.cumsum() - lengths
        for draw in range(n_boot):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            rows = np.concatenate([order[starts[item] : starts[item] + lengths[item]] for item in chosen])
            _beta_b, resid_b, _, _ = _fit_once(y[rows], x[rows], cluster[rows])
            grp = group[rows]
            draws[draw] = [resid_b[grp == code].mean() if np.any(grp == code) else np.nan for code in codes]
    records = []
    for index, code in enumerate(codes):
        lo = hi = np.nan
        if n_boot:
            lo = float(np.nanpercentile(draws[:, index], 2.5))
            hi = float(np.nanpercentile(draws[:, index], 97.5))
        records.append(
            {
                "factor": name,
                "level": labels[code],
                "report": True,
                "coef": float(point[index]),
                "lo": lo,
                "hi": hi,
                "n": int(counts[index]),
                "n_campaigns": int(campaign_counts[index]),
                "centered": False,
                "multiplier": float(np.exp(point[index])),
                "multiplier_lo": float(np.exp(lo)) if np.isfinite(lo) else np.nan,
                "multiplier_hi": float(np.exp(hi)) if np.isfinite(hi) else np.nan,
            }
        )
    warnings: list[str] = []
    if rank < x.shape[1]:
        warnings.append(f"design rank {rank} is below {x.shape[1]} columns")
    return Fit(
        name=name,
        coefficients=pd.DataFrame(records),
        n_rows=int(len(y)),
        n_campaigns=int(len(np.unique(cluster))),
        rank=rank,
        n_columns=int(x.shape[1]),
        condition=condition,
        warnings=warnings,
        residual=residual,
        draws=draws if n_boot else None,
    )


def _remap_clusters(cluster: np.ndarray) -> np.ndarray:
    _values, inverse = np.unique(cluster, return_inverse=True)
    return inverse.astype(np.int32)


def fit_factors(
    y: np.ndarray,
    cluster: np.ndarray,
    factors: list[Factor],
    extras: list[Extra] | None = None,
    *,
    n_boot: int = 0,
    seed: int = 0,
    name: str = "calendar",
    with_residual: bool = False,
) -> Fit:
    extras = extras or []
    y = np.asarray(y, dtype=np.float64)
    cluster = _remap_clusters(np.asarray(cluster))
    encoded = _present_factors(factors)
    # Rebuild factor code arrays so bootstrap slices stay aligned to this encoding.
    aligned: list[Factor] = []
    encoded_codes = []
    for factor, codes, _values, labels in encoded:
        aligned.append(Factor(factor.name, codes, labels, factor.report))
        encoded_codes.append(codes)
    x, columns = _design([(factor, factor.codes, None, factor.labels) for factor in aligned], extras)
    beta, residual, rank, condition = _fit_once(y, x, cluster)
    point = _unpack(beta, columns, cluster)
    warnings: list[str] = []
    if rank < x.shape[1]:
        warnings.append(f"design rank {rank} is below {x.shape[1]} columns")
    if np.isfinite(condition) and condition > 1e8:
        warnings.append(f"within-campaign design condition number is {condition:.2e}")
    draws = np.empty((n_boot, len(point)), dtype=np.float64)
    if n_boot:
        rng = np.random.default_rng(seed)
        n_clusters = int(cluster.max()) + 1
        order = np.argsort(cluster, kind="mergesort")
        starts = np.bincount(cluster).cumsum() - np.bincount(cluster)
        lengths = np.bincount(cluster)
        for draw in range(n_boot):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            rows = np.concatenate([order[starts[item] : starts[item] + lengths[item]] for item in chosen])
            # Copies of one campaign keep its id, so a repeated draw doubles its weight.
            beta_b, _, _, _ = _fit_once(y[rows], x[rows], cluster[rows])
            # Coefs are recentered on the full-sample level weights in `columns`.
            # The resampled cluster is a different length, so it cannot index those codes.
            unpacked = _unpack(beta_b, columns, cluster)
            draws[draw] = [row["coef"] for row in unpacked]
    records = []
    for index, row in enumerate(point):
        lo = hi = np.nan
        if n_boot:
            lo = float(np.nanpercentile(draws[:, index], 2.5))
            hi = float(np.nanpercentile(draws[:, index], 97.5))
        records.append({**row, "lo": lo, "hi": hi})
    frame = pd.DataFrame(records)
    frame["multiplier"] = np.exp(frame["coef"])
    frame["multiplier_lo"] = np.exp(frame["lo"])
    frame["multiplier_hi"] = np.exp(frame["hi"])
    return Fit(
        name=name,
        coefficients=frame,
        n_rows=int(len(y)),
        n_campaigns=int(len(np.unique(cluster))),
        rank=rank,
        n_columns=int(x.shape[1]),
        condition=condition,
        warnings=warnings,
        residual=residual if with_residual else None,
        draws=draws if n_boot else None,
    )
