"""Serving gates, run by the deploy workflow against staging after the candidate profile is deployed there.

Requests are replayed from the newest impression logs (the search service logs each request's inputs), so staging
sees the production query mix against the live catalogue.

- score parity: Vespa's relevance for each hit must equal the model run offline on the match-features Vespa returned
  for that hit. It catches a wrong input order, a dtype or shape mistake, or a runtime that evaluates differently.
- skew: a logged feature cannot differ between training and serving (training read what serving computed), but a
  backfilled one can. Each backfilled feature's distribution in staging is compared with its training distribution.
  Every feature is also checked against its contract range on the live catalogue.
- latency: p95 of the replayed queries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..contract import Contract
from ..export.onnx_export import run_onnx
from ..serving.vespa.client import Vespa, VespaError
from .base import GateResult


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("inf")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # nearly constant in training: compare the share at the constant
        c = edges[0]
        e, a = float(np.mean(np.isclose(expected, c))), float(np.mean(np.isclose(actual, c)))
        e, a = min(max(e, 1e-4), 1 - 1e-4), min(max(a, 1e-4), 1 - 1e-4)
        return float((a - e) * np.log(a / e) + ((1 - a) - (1 - e)) * np.log((1 - a) / (1 - e)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.maximum(e, 1e-4), np.maximum(a, 1e-4)
    return float(np.sum((a - e) * np.log(a / e)))


def replay_requests(impressions: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    s = impressions[(impressions["surface"] == "search") & (impressions["position"] == 0)]
    s = s.dropna(subset=["q_user_log_price"])
    if len(s) > n:
        s = s.sample(n, random_state=seed)
    return s[["session_id", "market", "query", "q_user_brand", "q_user_log_price"]]


def run(
    vespa: Vespa,
    profile: str,
    onnx: bytes,
    features: list[str],
    provenance: dict[str, str],
    train_sample: pd.DataFrame,
    requests: pd.DataFrame,
    now: int,
    contract: Contract,
) -> tuple[list[GateResult], dict[str, Any]]:
    cfg = config.load("gates")["serving"]
    rows: list[dict[str, float]] = []
    rels: list[float] = []
    latencies: list[float] = []
    errors: list[str] = []
    for r in requests.itertuples():
        try:
            res = vespa.search(
                [f'market contains "{r.market}"', f'category contains "{r.query}"', "sold = false"],
                ranking=profile,
                hits=10,
                inputs={"now": float(now), "user_brand": r.q_user_brand, "user_log_price": float(r.q_user_log_price)},
            )
        except VespaError as e:
            errors.append(str(e)[:200])
            continue
        latencies.append(res.elapsed_ms)
        for h in res.hits:
            if all(f in h.features for f in features):
                rows.append({f: h.features[f] for f in features})
                rels.append(h.relevance)
    results: list[GateResult] = []
    evidence: dict[str, Any] = {"requests": len(requests), "errors": errors[:5], "hits": len(rows)}
    if errors or not rows:
        results.append(
            GateResult(
                "serving.score_parity",
                False,
                f"{len(errors)} of {len(requests)} replayed queries failed on staging"
                if errors
                else "staging returned no hits with features",
                {"errors": errors[:5]},
            )
        )
        return results, evidence
    x = np.array([[row[f] for f in features] for row in rows], dtype=np.float32)
    offline = run_onnx(onnx, x)[:, 0]
    served = np.array(rels, dtype=float)
    diff = np.abs(offline - served)
    worst = float(np.nanmax(diff)) if np.all(np.isfinite(served)) else float("inf")
    results.append(
        GateResult(
            "serving.score_parity",
            worst <= cfg["score_parity_max_abs"],
            f"max |Vespa relevance - offline model| = {worst:.2e} over {len(rows)} hits",
            {"max_abs": worst, "limit": cfg["score_parity_max_abs"], "hits": len(rows)},
        )
    )
    skew: dict[str, Any] = {}
    bad: dict[str, Any] = {}
    for i, f in enumerate(features):
        lo, hi = contract.feature(f).range
        vals = x[:, i]
        outside = float(np.mean((vals < lo) | (vals > hi)))
        entry: dict[str, Any] = {"outside_range": round(outside, 4), "provenance": provenance.get(f, "logged")}
        if provenance.get(f, "logged") != "logged" and f"f_{f}" in train_sample:
            entry["psi"] = round(psi(train_sample[f"f_{f}"].to_numpy(dtype=float), vals.astype(float)), 4)
            if entry["psi"] > cfg["skew_psi_max"]:
                bad[f] = entry
        if outside > config.load("gates")["data"]["range_violation_max"]:
            bad[f] = entry
        skew[f] = entry
    results.append(
        GateResult(
            "serving.skew",
            not bad,
            "serving computes every feature as training saw it"
            if not bad
            else "skew in "
            + ", ".join(
                f"{f} (psi {e.get('psi', '-')}, {e['outside_range']:.0%} out of range)" for f, e in bad.items()
            ),
            {"features": skew, "psi_max": cfg["skew_psi_max"]},
        )
    )
    p95 = float(np.percentile(latencies, 95)) if latencies else float("inf")
    results.append(
        GateResult(
            "serving.latency",
            p95 <= cfg["latency_p95_ms"],
            f"p95 {p95:.1f} ms over {len(latencies)} queries",
            {"p95_ms": p95, "limit_ms": cfg["latency_p95_ms"], "queries": len(latencies)},
        )
    )
    evidence["latency_p95_ms"] = p95
    return results, evidence
