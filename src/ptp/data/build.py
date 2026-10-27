"""Builds a ranker training set from the impression logs.

Features come from the logged match-features (what Vespa computed when it served the impression). A feature that
was not logged can be backfilled from an item snapshot; the column `__asof_<feature>` records when that snapshot
was taken, so the provenance gate can check that no value was observed after the impression it describes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..contract import Contract
from ..lake import DAY, Lake, Partition, day_of, day_start


class NoData(RuntimeError):
    pass


@dataclass
class Window:
    newest: dt.date
    train: list[dt.date]
    holdout: list[dt.date]
    partitions: list[Partition]

    @property
    def dates(self) -> list[dt.date]:
        return self.train + self.holdout


def window(lake: Lake, as_of: int, window_days: int, holdout_days: int, table: str = "impressions") -> Window:
    """The newest `window_days` complete days that landed by `as_of` and end before it."""
    today = day_of(as_of)
    parts = [p for p in lake.partitions(table, landed_by=as_of) if p.date < today]
    if not parts:
        raise NoData(f"no complete {table} partitions landed by {as_of}")
    newest = parts[-1].date
    dates = [newest - dt.timedelta(days=i) for i in range(window_days - 1, -1, -1)]
    have = {p.date: p for p in parts}
    dates = [d for d in dates if d in have]
    holdout = dates[-holdout_days:]
    train = dates[:-holdout_days]
    if not train:
        raise NoData("the window has no training days")
    return Window(newest=newest, train=train, holdout=holdout, partitions=[have[d] for d in dates])


@dataclass
class TrainingSet:
    frame: pd.DataFrame  # one row per impression after deduplication; split column 'split'
    features: list[str]
    provenance: dict[str, str]  # feature -> logged | backfill:<snapshot>
    snapshot: str
    raw_rows: int
    duplicates: int
    extra_partitions: list[Partition] = field(default_factory=list)

    def split(self, name: str) -> pd.DataFrame:
        return self.frame[self.frame["split"] == name]


def grade(frame: pd.DataFrame, labels: dict[str, int]) -> np.ndarray:
    g = np.zeros(len(frame), dtype=int)
    for col, value in sorted(labels.items(), key=lambda kv: kv[1]):
        flag = {"purchase": "purchased", "favourite": "favourited", "click": "clicked"}[col]
        g = np.where(frame[flag].to_numpy(), np.maximum(g, value), g)
    return g


def read_raw(
    lake: Lake, win: Window, surface: str = "search", quarantine: list[dict[str, Any]] | None = None
) -> pd.DataFrame:
    raw = lake.read("impressions", win.dates)
    if raw.empty:
        raise NoData("the window is empty")
    raw = raw[raw["surface"] == surface].copy()
    raw["day"] = pd.to_datetime(raw["ts"], unit="s", utc=True).dt.date
    raw = drop_quarantined(raw, quarantine or [])
    raw["split"] = np.where(raw["day"].isin(win.holdout), "holdout", "train")
    return raw


def build(lake: Lake, win: Window, raw: pd.DataFrame, contract: Contract, training: dict[str, Any]) -> TrainingSet:
    features: list[str] = list(training["features"])
    backfill: dict[str, dict[str, str]] = training.get("backfill") or {}
    frame = raw.drop_duplicates(subset=["impression_id"]).copy()
    duplicates = len(raw) - len(frame)
    provenance: dict[str, str] = {}
    extra: list[Partition] = []
    for name in features:
        feat = contract.feature(name)
        col = f"f_{name}"
        if name in backfill:
            spec = backfill[name]
            frame[col], frame[f"__asof_{name}"], used = _backfill(lake, frame, feat.expression, spec)
            extra += used
            provenance[name] = f"backfill:{spec.get('snapshot', 'previous_day')}"
        else:
            if col not in frame.columns:
                frame[col] = np.nan
            provenance[name] = "logged"
    frame["label"] = grade(frame, training["labels"])
    frame = frame.sort_values(["session_id", "position"], kind="stable").reset_index(drop=True)
    snapshot = Lake.snapshot_id(win.partitions + extra)
    return TrainingSet(frame, features, provenance, snapshot, len(raw), duplicates, extra)


def _backfill(
    lake: Lake, frame: pd.DataFrame, expression: str, spec: dict[str, str]
) -> tuple[pd.Series, pd.Series, list[Partition]]:
    if not (expression.startswith("attribute(") and expression.endswith(")")):
        raise ValueError(f"only single-attribute features can be backfilled from snapshots, not {expression!r}")
    attr = expression[len("attribute(") : -1]
    offset = {"previous_day": 1, "same_day": 0}[spec.get("snapshot", "previous_day")]
    days = sorted(set(frame["day"]))
    snap_days = sorted({d - dt.timedelta(days=offset) for d in days})
    parts = {p.date: p for p in lake.partitions("items")}
    have = [d for d in snap_days if d in parts]
    snaps = lake.read("items", have)
    if snaps.empty:
        return pd.Series(np.nan, index=frame.index), pd.Series(np.nan, index=frame.index), []
    snaps["day"] = pd.to_datetime(snaps["as_of"] - 1, unit="s", utc=True).dt.date + dt.timedelta(days=offset)
    joined = frame[["item_id", "day"]].merge(
        snaps[["item_id", "day", attr, "as_of"]], on=["item_id", "day"], how="left"
    )
    joined.index = frame.index
    return joined[attr].astype(float), joined["as_of"].astype(float), [parts[d] for d in have]


def drop_quarantined(raw: pd.DataFrame, quarantine: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows an operator declared bad are left out; raw.attrs['quarantined'] says how many."""
    if not quarantine or raw.empty:
        raw.attrs["quarantined"] = 0
        return raw
    bad = np.zeros(len(raw), dtype=bool)
    for q in quarantine:
        m = raw["day"] == q["day"]
        if q.get("market"):
            m &= raw["market"] == q["market"]
        bad |= m.to_numpy()
    out = raw[~bad].copy()
    out.attrs["quarantined"] = int(bad.sum())
    out.attrs["quarantined_pairs"] = sorted({(str(q["day"]), q.get("market")) for q in quarantine}, key=str)
    return out


def matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[[f"f_{f}" for f in features]].to_numpy(dtype=np.float32)


def groups(frame: pd.DataFrame) -> np.ndarray:
    """Group sizes for a frame sorted by session (one search per session)."""
    return frame.groupby("session_id", sort=False).size().to_numpy()


def end_of(d: dt.date) -> int:
    return day_start(d) + DAY
