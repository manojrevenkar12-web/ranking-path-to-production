"""The data lake the pipelines read: daily partitions of impression logs, item snapshots and user profiles, each
with a _SUCCESS marker that says when it landed. On GCP these are BigQuery exports in GCS; here a directory with
the same layout. The simulator writes it through this module, so the format is one definition."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TABLES = ("impressions", "items", "users")
DAY = 86400
_hour_cache: dict[tuple[str, float], pd.DataFrame] = {}


def day_of(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(ts, dt.UTC).date()


def day_start(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC).timestamp())


@dataclass(frozen=True)
class Partition:
    table: str
    date: dt.date
    landed_at: int
    rows: int
    sha256: str


class Lake:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _dir(self, table: str, date: dt.date) -> Path:
        return self.root / table / f"date={date.isoformat()}"

    def write(self, table: str, date: dt.date, frame: pd.DataFrame, landed_at: int) -> Partition:
        d = self._dir(table, date)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "part-0.parquet"
        frame.to_parquet(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        marker = {"landed_at": int(landed_at), "rows": len(frame), "sha256": digest}
        (d / "_SUCCESS").write_text(json.dumps(marker))
        return Partition(table, date, int(landed_at), len(frame), digest)

    def partitions(self, table: str, landed_by: int | None = None) -> list[Partition]:
        """Complete partitions of a table (those with a marker), oldest first; only those landed by `landed_by`."""
        base = self.root / table
        if not base.exists():
            return []
        out = []
        for d in sorted(base.iterdir()):
            marker = d / "_SUCCESS"
            if not marker.exists() or not d.name.startswith("date="):
                continue
            m = json.loads(marker.read_text())
            if landed_by is not None and m["landed_at"] > landed_by:
                continue
            out.append(Partition(table, dt.date.fromisoformat(d.name[5:]), m["landed_at"], m["rows"], m["sha256"]))
        return out

    def read(self, table: str, dates: list[dt.date]) -> pd.DataFrame:
        frames = [pd.read_parquet(self._dir(table, d) / "part-0.parquet") for d in dates]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ---- the hourly session stream (near-real-time, what canary checks read) -------------------------------------

    def write_hour(self, table: str, hour_start: int, frame: pd.DataFrame) -> None:
        d = self.root / table / f"date={day_of(hour_start).isoformat()}"
        d.mkdir(parents=True, exist_ok=True)
        hour = (hour_start % DAY) // 3600
        frame.to_parquet(d / f"hour={hour:02d}.parquet", index=False)

    def read_since(self, table: str, since: int, until: int) -> pd.DataFrame:
        frames = []
        day = day_of(since)
        while day_start(day) <= until:
            d = self.root / table / f"date={day.isoformat()}"
            if d.exists():
                for f in sorted(d.glob("hour=*.parquet")):
                    hour = int(f.stem.split("=")[1])
                    start = day_start(day) + hour * 3600
                    if start + 3600 > since and start <= until:
                        key = (str(f), f.stat().st_mtime)
                        if key not in _hour_cache:  # hourly files never change once written
                            if len(_hour_cache) > 512:
                                _hour_cache.clear()
                            _hour_cache[key] = pd.read_parquet(f)
                        frames.append(_hour_cache[key])
            day = day + dt.timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out[(out["ts"] >= since) & (out["ts"] <= until)]

    @staticmethod
    def snapshot_id(parts: list[Partition]) -> str:
        """A content hash of exactly the partitions a run used: same id, same bytes."""
        h = hashlib.sha256()
        for p in sorted(parts, key=lambda p: (p.table, p.date)):
            h.update(f"{p.table}/{p.date.isoformat()}/{p.sha256}".encode())
        return h.hexdigest()[:16]
