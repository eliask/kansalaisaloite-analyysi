"""Fetch an immutable snapshot of the public kansalaisaloite.fi API.

    uv run python -m src.fetch
    uv run python -m src.fetch --resume

A partial run never becomes LATEST. Raw responses are kept under the stamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API = "https://www.kansalaisaloite.fi/api/v1"
UA = "kansalaisaloite-analysis/0.1 (public-data snapshot)"
PAGE_LIMIT = 50
HELSINKI = ZoneInfo("Europe/Helsinki")
_THREAD = threading.local()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _session() -> requests.Session:
    session = getattr(_THREAD, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": UA, "Accept": "application/json"})
        _THREAD.session = session
    return session


def get_bytes(url: str, timeout: float = 60) -> bytes:
    delay = 1.0
    last: Exception | None = None
    for _ in range(6):
        try:
            response = _session().get(url, timeout=timeout)
        except requests.RequestException as exc:
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        if response.status_code in {429, 500, 502, 503, 504, 403}:
            last = requests.HTTPError(f"{response.status_code} {url}")
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = min(max(delay, int(retry_after)), 60)
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        if response.status_code != 200:
            raise requests.HTTPError(f"{response.status_code} {url}")
        return response.content
    raise RuntimeError(f"gave up on {url}") from last


def initiative_id(uri: str) -> int:
    return int(str(uri).rstrip("/").rsplit("/", 1)[-1])


def parse_supports(body: bytes) -> list[dict]:
    data = json.loads(body)
    if not isinstance(data, list):
        raise ValueError("supports payload is not a list")
    for row in data:
        if not isinstance(row, dict) or "d" not in row or "n" not in row:
            raise ValueError("supports row is missing d or n")
        datetime.strptime(row["d"], "%Y-%m-%d")
        if isinstance(row["n"], bool) or not isinstance(row["n"], int) or row["n"] < 0:
            raise ValueError(f"bad support count {row['n']!r}")
    return data


def parse_page(body: bytes) -> list[dict]:
    data = json.loads(body)
    if not isinstance(data, list):
        raise ValueError("initiative page is not a list")
    for row in data:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError("initiative row has no id")
        initiative_id(row["id"])
    return data


def _latest_incomplete(live: Path) -> Path | None:
    if not live.is_dir():
        return None
    candidates = sorted(path for path in live.iterdir() if path.is_dir())
    for path in reversed(candidates):
        run = path / "run.json"
        if not run.exists():
            continue
        status = json.loads(run.read_text(encoding="utf-8")).get("status")
        if status != "complete":
            return path
    return None


def fetch_pages(raw_pages: Path) -> list[dict]:
    raw_pages.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0
    while True:
        url = f"{API}/initiatives?limit={PAGE_LIMIT}&offset={offset}&orderBy=createdOldest"
        body = get_bytes(url)
        page = parse_page(body)
        (raw_pages / f"page_{offset:05d}.json").write_bytes(body)
        if not page:
            break
        rows.extend(page)
        print(f"initiatives offset={offset} rows={len(page)} total={len(rows)}", flush=True)
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    return rows


def _supports_one(initiative: int, dest: Path) -> str:
    path = dest / f"{initiative}.json"
    if path.exists() and path.stat().st_size > 0:
        parse_supports(path.read_bytes())
        return "cached"
    url = f"{API}/supports-by-date/{initiative}"
    body = get_bytes(url)
    parse_supports(body)
    path.write_bytes(body)
    return "fetched"


def fetch_supports(ids: list[int], dest: Path, workers: int) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    counts = {"cached": 0, "fetched": 0}
    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_supports_one, initiative, dest): initiative for initiative in ids}
        for future in as_completed(futures):
            initiative = futures[future]
            done += 1
            try:
                counts[future.result()] += 1
            except Exception as exc:  # noqa: BLE001 — recorded, then the snapshot is refused
                failures.append(f"{initiative}: {type(exc).__name__}: {exc}")
            if done % 100 == 0 or done == len(ids):
                print(
                    f"supports {done}/{len(ids)} fetched={counts['fetched']} "
                    f"cached={counts['cached']} failed={len(failures)}",
                    flush=True,
                )
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"{len(failures)} support series failed\n{preview}")
    return counts


def choose_cutoff(support_dir: Path, fetched_at: datetime) -> str:
    max_date: datetime | None = None
    for path in support_dir.glob("*.json"):
        for row in json.loads(path.read_text(encoding="utf-8")):
            day = datetime.strptime(row["d"], "%Y-%m-%d")
            if max_date is None or day > max_date:
                max_date = day
    helsinki_today = fetched_at.astimezone(HELSINKI).date()
    if max_date is None:
        cutoff = helsinki_today - timedelta(days=1)
    else:
        latest = max_date.date()
        cutoff = helsinki_today - timedelta(days=1) if latest >= helsinki_today else latest
    return cutoff.isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Continue the newest incomplete snapshot.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    root = args.root.resolve()
    live = root / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    if args.resume:
        base = _latest_incomplete(live)
        if base is None:
            parser.error("no incomplete snapshot to resume")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base = live / stamp
        base.mkdir(parents=True)
    fetched_at = datetime.now(timezone.utc)
    status: dict = {
        "schema_version": 1,
        "status": "started",
        "api": API,
        "order_by": "createdOldest",
        "page_limit": PAGE_LIMIT,
        "created_utc": fetched_at.isoformat(),
        "python": platform.python_version(),
        "errors": [],
    }
    write_json(base / "run.json", status)
    try:
        print(f"Snapshot: {base}", flush=True)
        rows = fetch_pages(base / "raw" / "initiatives")
        ids = [initiative_id(row["id"]) for row in rows]
        unique_ids = list(dict.fromkeys(ids))
        status["initiatives_rows"] = len(rows)
        status["initiatives_unique"] = len(unique_ids)
        if len(unique_ids) != len(ids):
            status["duplicate_ids"] = len(ids) - len(unique_ids)
        counts = fetch_supports(unique_ids, base / "raw" / "supports", args.workers)
        status["supports"] = counts
        status["cutoff"] = choose_cutoff(base / "raw" / "supports", fetched_at)
        status["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 — snapshot must stay off LATEST
        status["status"] = "failed"
        status["errors"].append(f"{type(exc).__name__}: {exc}")
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(base / "run.json", status)
        print(f"FAILED: {base}\n{exc}", flush=True)
        raise SystemExit(1) from exc
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(base / "run.json", status)
    files = {str(path.relative_to(base)): digest(path) for path in sorted(base.rglob("*")) if path.is_file()}
    status["files"] = files
    write_json(base / "manifest.json", status)
    pointer = live / "LATEST"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(base.name + "\n", encoding="utf-8")
    temporary.replace(pointer)
    print(f"LATEST {base.name} cutoff={status['cutoff']} initiatives={status['initiatives_unique']}", flush=True)


if __name__ == "__main__":
    main()
