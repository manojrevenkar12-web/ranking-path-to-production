"""Data gates: run on the raw training window before anything is trained. They compare the window with itself
(volume), with the feature contract (ranges) and with the profile of the data the production model was trained on
(null rates, click rates), and check freshness and provenance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..contract import Contract
from ..data.build import TrainingSet, Window, end_of
from .base import GateResult


def profile(raw: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """What the data looked like, per market: kept with every model so the next run has a baseline."""
    out: dict[str, Any] = {}
    for market, g in raw.groupby("market"):
        per_day = g.groupby("day").size()
        out[str(market)] = {
            "rows": len(g),
            "rows_per_day": {str(k): int(v) for k, v in per_day.items()},
            "click_rate": float(g["clicked"].mean()) if len(g) else 0.0,
            "null_rate": {f: float(g[f"f_{f}"].isna().mean()) if f"f_{f}" in g else 1.0 for f in features},
        }
    return out


def run(
    raw: pd.DataFrame, ts: TrainingSet, win: Window, as_of: int, contract: Contract, baseline: dict[str, Any] | None
) -> tuple[list[GateResult], dict[str, Any]]:
    cfg = config.load("gates")["data"]
    markets = config.load("gates")["segments"]["market"]
    results: list[GateResult] = []
    prof = profile(raw, ts.features)
    prof["_quarantined"] = int(raw.attrs.get("quarantined", 0))

    # freshness
    age_h = (as_of - end_of(win.newest)) / 3600
    results.append(
        GateResult(
            "data.freshness",
            age_h <= cfg["freshness_max_hours"],
            f"newest complete partition is {win.newest.isoformat()}, {age_h:.0f} h old at pipeline start",
            {"newest": win.newest.isoformat(), "age_hours": round(age_h, 1), "max_hours": cfg["freshness_max_hours"]},
        )
    )

    # volume: newest day per market against the median of the window's days (days an operator quarantined for a
    # market are not missing data, they were left out on purpose)
    quarantined = {(d, mk) for d, mk in raw.attrs.get("quarantined_pairs", [])}

    def kept(m: str) -> list[Any]:
        return [
            d for d in win.dates if (d.isoformat(), m) not in quarantined and (d.isoformat(), None) not in quarantined
        ]

    vol_bad = {}
    for m in markets:
        days = prof.get(m, {}).get("rows_per_day", {})
        counts = [days.get(d.isoformat(), 0) for d in kept(m)]
        med = float(np.median(counts[:-1])) if len(counts) > 1 else float(counts[-1] if counts else 0)
        last = counts[-1] if counts else 0
        if (med > 0 and abs(last - med) / med > cfg["volume_band"]) or med == 0:
            vol_bad[m] = {"newest_day": last, "median": med}
    results.append(
        GateResult(
            "data.volume",
            not vol_bad,
            "daily volume per market is within band" if not vol_bad else f"volume out of band in {', '.join(vol_bad)}",
            {"out_of_band": vol_bad, "band": cfg["volume_band"]},
        )
    )

    # null rates against the production model's training data
    null_bad: dict[str, dict[str, float]] = {}
    if baseline:
        for m in markets:
            for f in ts.features:
                if ts.provenance.get(f) != "logged":
                    continue
                now_rate = prof.get(m, {}).get("null_rate", {}).get(f, 1.0)
                was = baseline.get(m, {}).get("null_rate", {}).get(f)
                if was is not None and now_rate - was > cfg["null_rate_max_increase"]:
                    null_bad[f"{f}@{m}"] = {"now": round(now_rate, 4), "baseline": round(was, 4)}
    results.append(
        GateResult(
            "data.null_rate",
            not null_bad,
            "no feature lost values" if not null_bad else f"null rate up for {', '.join(sorted(null_bad))}",
            {"increases": null_bad, "max_increase": cfg["null_rate_max_increase"], "baseline": bool(baseline)},
        )
    )

    # ranges from the contract
    range_bad = {}
    for m in markets:
        g = raw[raw["market"] == m]
        if g.empty:
            continue
        for f in ts.features:
            col = f"f_{f}"
            if col not in g or ts.provenance.get(f) != "logged":
                continue
            lo, hi = contract.feature(f).range
            vals = g[col].dropna().to_numpy()
            if len(vals) == 0:
                continue
            share = float(np.mean((vals < lo) | (vals > hi)))
            if share > cfg["range_violation_max"]:
                range_bad[f"{f}@{m}"] = {
                    "outside": round(share, 4),
                    "range": [lo, hi],
                    "median": float(np.median(vals)),
                }
    results.append(
        GateResult(
            "data.range",
            not range_bad,
            "every feature within its contract range"
            if not range_bad
            else f"out of range: {', '.join(sorted(range_bad))}",
            {"violations": range_bad, "max_share": cfg["range_violation_max"]},
        )
    )

    # duplicates, per day and market (a logging retry shows up on one day in one market, diluted over a window)
    dup_bad = {}
    dup = raw.duplicated(subset=["impression_id"])
    for (day, m), g in raw.assign(_dup=dup).groupby(["day", "market"]):
        rate = float(g["_dup"].mean())
        if rate > cfg["duplicate_rate_max"]:
            dup_bad[f"{m}@{day}"] = round(rate, 4)
    results.append(
        GateResult(
            "data.duplicates",
            not dup_bad,
            f"{ts.duplicates} duplicate impressions dropped"
            + (f"; logging duplicated events in {', '.join(sorted(dup_bad))}" if dup_bad else ""),
            {
                "duplicates": ts.duplicates,
                "rows": ts.raw_rows,
                "by_day_market": dup_bad,
                "max": cfg["duplicate_rate_max"],
            },
        )
    )

    # label rate: the newest day against the day before, per market (a logging loss shows as a step)
    label_bad = {}
    for m in markets:
        dates = kept(m)
        if len(dates) < 2:
            continue
        newest, before = dates[-1], dates[-2]
        g = raw[raw["market"] == m]
        a = g[g["day"] == before]["clicked"]
        b = g[g["day"] == newest]["clicked"]
        if len(a) < cfg["label_rate_min_rows"] or len(b) < cfg["label_rate_min_rows"]:
            continue
        ra, rb = float(a.mean()), float(b.mean())
        if ra > 0 and abs(rb - ra) / ra > cfg["label_rate_band"]:
            label_bad[m] = {"newest": round(rb, 4), "day_before": round(ra, 4)}
    results.append(
        GateResult(
            "data.label_rate",
            not label_bad,
            "click rates per market are in band" if not label_bad else f"click rate moved in {', '.join(label_bad)}",
            {"moved": label_bad, "band": cfg["label_rate_band"]},
        )
    )

    # provenance: backfilled values must have been observable before the impression
    prov_bad = {}
    frame = ts.frame
    for f, how in ts.provenance.items():
        if how == "logged":
            missing = float(frame[f"f_{f}"].isna().mean())
            if missing > cfg["provenance_missing_max"]:
                prov_bad[f] = {"problem": "not in the serving logs", "missing": round(missing, 3)}
            continue
        asof = frame[f"__asof_{f}"]
        late = float((asof > frame["ts"]).mean())
        absent = float(asof.isna().mean())
        if late > 0:
            prov_bad[f] = {"problem": "observed after the impression", "share_late": round(late, 4)}
        elif absent > cfg["provenance_missing_max"]:
            prov_bad[f] = {"problem": "snapshot missing", "share_missing": round(absent, 3)}
    results.append(
        GateResult(
            "data.provenance",
            not prov_bad,
            "every feature was observable at serving time"
            if not prov_bad
            else f"provenance fails for {', '.join(prov_bad)}",
            {"provenance": ts.provenance, "problems": prov_bad},
        )
    )
    return results, prof
