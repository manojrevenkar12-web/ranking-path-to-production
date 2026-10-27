"""Daily monitoring of production models: the features production served yesterday against the ones the model was
trained on (PSI per feature and market), and contract ranges on what was served. A drifted feature does not roll
anything back by itself; it pages the owning team, and the next training run's data gates see it too."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from .. import artifacts, config, contract
from ..gates.serving import psi
from ..lake import Lake
from ..registry.store import Registry


def run() -> list[dict[str, Any]]:
    reg, store = Registry(), artifacts.Store()
    c = contract.load()
    lake = Lake(config.resolve_path(config.load("pipeline")["lake"]))
    cfg = config.load("gates")
    parts = lake.partitions("impressions")
    out: list[dict[str, Any]] = []
    if not parts:
        return out
    served = lake.read("impressions", [parts[-1].date])
    route = reg.routing("search")
    if route and route["production"].startswith("ranker_"):
        version = route["production"].removeprefix("ranker_")
        v = reg.version(version)
        if v:
            a = v["artifacts"]["train_sample"]
            train = pd.read_parquet(io.BytesIO(store.get(a["key"], a["sha256"])))
            s = served[(served["surface"] == "search") & (served["profile"] == route["production"])]
            problems: dict[str, Any] = {}
            per: dict[str, Any] = {}
            for m in cfg["segments"]["market"]:
                tm, sm = train[train["market"] == m], s[s["market"] == m]
                for f in v["features"]:
                    col = f"f_{f}"
                    if col not in sm or col not in tm or len(sm) == 0:
                        continue
                    vals = sm[col].to_numpy(dtype=float)
                    lo, hi = c.feature(f).range
                    outside = float(np.mean((vals < lo) | (vals > hi))) if len(vals) else 0.0
                    p = psi(tm[col].to_numpy(dtype=float), vals)
                    per[f"{f}@{m}"] = {"psi": round(p, 4), "outside_range": round(outside, 4)}
                    if p > cfg["serving"]["skew_psi_max"] * 2 or outside > cfg["data"]["range_violation_max"]:
                        problems[f"{f}@{m}"] = per[f"{f}@{m}"]
            result = {"date": parts[-1].date.isoformat(), "problems": problems, "features": per}
            reg.record_monitor("search", version, not problems, result)
            out.append({"surface": "search", "version": version, "healthy": not problems, "problems": problems})
    home = reg.routing("home")
    if home and home["production"].startswith("t"):
        h = served[(served["surface"] == "home") & (served["profile"] == home["production"])]
        sessions = h["session_id"].nunique()
        ctr = float(h["clicked"].mean()) if len(h) else 0.0
        home_result: dict[str, Any] = {"date": parts[-1].date.isoformat(), "sessions": int(sessions), "ctr": ctr}
        reg.record_monitor("home", home["production"], sessions > 0, home_result)
        out.append({"surface": "home", "version": home["production"], "healthy": sessions > 0, **home_result})
    return out
