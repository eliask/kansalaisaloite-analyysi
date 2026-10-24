"""Read one immutable API snapshot. Raw JSON is the source of truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .fetch import initiative_id


@dataclass(frozen=True)
class Snapshot:
    root: Path
    stamp: str
    path: Path
    fetched_at: datetime
    cutoff: date
    initiatives: pd.DataFrame
    daily: pd.DataFrame


def snapshot_dir(root: Path, stamp: str | None = None) -> tuple[str, Path]:
    live = root / "data" / "live"
    if stamp is None:
        pointer = live / "LATEST"
        if not pointer.exists():
            raise FileNotFoundError("No snapshot. Run: uv run python -m src.fetch")
        stamp = pointer.read_text(encoding="utf-8").strip()
    path = live / stamp
    if not path.is_dir():
        raise FileNotFoundError(f"Snapshot {stamp} is not on disk")
    return stamp, path


def _as_date(value: object) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if not text or text == "None":
        return None
    return date.fromisoformat(text[:10])


def _name(value: object, language: str) -> str:
    if not isinstance(value, dict):
        return ""
    text = value.get(language)
    return "" if text is None else str(text)


def load_initiatives(path: Path) -> pd.DataFrame:
    pages = sorted((path / "raw" / "initiatives").glob("page_*.json"))
    if not pages:
        raise FileNotFoundError(f"no initiative pages in {path}")
    rows: list[dict] = []
    seen: set[int] = set()
    for page in pages:
        for raw in json.loads(page.read_text(encoding="utf-8")):
            identifier = initiative_id(raw["id"])
            if identifier in seen:
                continue
            seen.add(identifier)
            url = raw.get("url") or {}
            rows.append(
                {
                    "id": identifier,
                    "name_fi": _name(raw.get("name"), "fi"),
                    "name_sv": _name(raw.get("name"), "sv"),
                    "state": raw.get("state") or "",
                    "state_date": _as_date(raw.get("stateDate")),
                    "start_date": _as_date(raw.get("startDate")),
                    "end_date": _as_date(raw.get("endDate")),
                    "support_count": int(raw.get("supportCount") or 0),
                    "external_support_count": int(raw.get("externalSupportCount") or 0),
                    "total_support_count": int(raw.get("totalSupportCount") or 0),
                    "sent_support_count": int(raw.get("sentSupportCount") or 0),
                    "verified_support_count": int(raw.get("verifiedSupportCount") or 0),
                    "verified": _as_date(raw.get("verified")),
                    "proposal_type": raw.get("proposalType") or "",
                    "primary_language": raw.get("primaryLanguage") or "",
                    "financial_support": bool(raw.get("financialSupport")),
                    "on_paper": bool(raw.get("supportStatementsOnPaper")),
                    "in_other_web": bool(raw.get("supportStatementsInWeb")),
                    "statements_removed": raw.get("supportStatementsRemoved"),
                    "voting_in_progress": bool(raw.get("votingInProgress")),
                    "verification_passed": raw.get("verificationPassed"),
                    "url_fi": (url.get("fi") if isinstance(url, dict) else "") or "",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("initiative snapshot is empty")
    return frame.sort_values("id").reset_index(drop=True)


def load_daily(path: Path) -> pd.DataFrame:
    files = sorted((path / "raw" / "supports").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no support series in {path}")
    ids: list[int] = []
    days: list[str] = []
    counts: list[int] = []
    fetched: list[int] = []
    for file in files:
        identifier = int(file.stem)
        fetched.append(identifier)
        for row in json.loads(file.read_text(encoding="utf-8")):
            ids.append(identifier)
            days.append(row["d"])
            counts.append(int(row["n"]))
    daily = pd.DataFrame({"id": ids, "date": days, "n": counts})
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
    else:
        daily["date"] = pd.Series(dtype="object")
    daily.attrs["fetched_ids"] = set(fetched)
    return daily


def load_snapshot(root: Path, stamp: str | None = None) -> Snapshot:
    stamp, path = snapshot_dir(root, stamp)
    manifest_path = path / "manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else json.loads(
        (path / "run.json").read_text(encoding="utf-8")
    )
    if meta.get("status") != "complete":
        raise ValueError(f"Snapshot {stamp} is {meta.get('status')}, not complete")
    if "cutoff" not in meta:
        raise ValueError(f"Snapshot {stamp} has no cutoff")
    initiatives = load_initiatives(path)
    daily = load_daily(path)
    return Snapshot(
        root=root,
        stamp=stamp,
        path=path,
        fetched_at=datetime.fromisoformat(meta["created_utc"]),
        cutoff=date.fromisoformat(meta["cutoff"]),
        initiatives=initiatives,
        daily=daily,
    )
