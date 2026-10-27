"""Position bias, measured from exploration traffic.

A share of search sessions shows the production results in random order. In those, an item's position does not
depend on its relevance, so click rate by position measures how much attention a position gets. Offline metrics
weight logged labels by the inverse of that propensity; without it, the model that produced the logs always looks
best offline, because clicks land where it put things.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def estimate(raw: pd.DataFrame, positions: int = 20, min_sessions: int = 300) -> dict[str, Any]:
    ex = raw[(raw.get("arm") == "explore")] if "arm" in raw else raw.iloc[0:0]
    sessions = int(ex["session_id"].nunique()) if len(ex) else 0
    if sessions < min_sessions:
        return {"source": "none", "sessions": sessions, "eta": None, "per_position": None}
    by_pos = ex.groupby("position")["clicked"].agg(["sum", "count"]).reindex(range(positions), fill_value=0)
    ctr = (by_pos["sum"] + 0.5) / (by_pos["count"] + 1.0)
    k = np.arange(1, positions + 1, dtype=float)
    w = by_pos["count"].to_numpy(dtype=float)
    y = np.log(ctr.to_numpy() / ctr.iloc[0])
    x = np.log(k)
    eta = float(max(0.0, -np.sum(w * x * y) / max(np.sum(w * x * x), 1e-9)))
    return {"source": "explore", "sessions": sessions, "eta": round(eta, 4), "per_position": (k ** -eta).round(5).tolist()}


def weights(positions: np.ndarray, propensity: dict[str, Any] | None, max_weight: float) -> np.ndarray:
    if not propensity or not propensity.get("per_position"):
        return np.ones(len(positions))
    p = np.asarray(propensity["per_position"], dtype=float)
    pos = np.clip(positions.astype(int), 0, len(p) - 1)
    return np.minimum(1.0 / p[pos], max_weight)
