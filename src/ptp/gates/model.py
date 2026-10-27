"""Model gates: the candidate against the production model on the same time-split holdout, query by query, with a
paired bootstrap so the decision is about the interval, not the point estimate. Continuous training ships refreshes
that are not worse (non-inferiority), overall and in every market."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..data.build import matrix
from ..data.propensity import weights as propensity_weights
from .base import GateResult


def dcg_per_query(
    frame: pd.DataFrame, scores: np.ndarray, k: int = 10, propensity: dict[str, Any] | None = None
) -> pd.DataFrame:
    """IPS-weighted DCG@k per query: re-rank each logged query's impressions by `scores` and sum the gains of the
    top k, each label's gain divided by the propensity of the position it was shown at.

    Deliberately not normalised per query. With one click in a query, NDCG is just the discount at the rank the
    scorer gives that item, and the propensity weight cancels out of the ratio: the metric then rewards whichever
    model put the clicked item first, which is the model that produced the logs. The unnormalised sum is the
    unbiased IPS estimator (docs/RESEARCH.md shows the measurement on a season).
    """
    max_w = config.load("gates")["model"]["ips_max_weight"]
    df = frame[["session_id", "market", "label", "position"]].copy()
    df["score"] = scores
    df["w"] = propensity_weights(df["position"].to_numpy(), propensity, max_w)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    for (sid, market), g in df.groupby(["session_id", "market"], sort=False):
        gain = (2.0 ** g["label"].to_numpy() - 1) * g["w"].to_numpy()
        order = np.argsort(-g["score"].to_numpy(), kind="stable")[:k]
        out.append((sid, market, float(np.sum(gain[order] * discounts[: len(order)]))))
    return pd.DataFrame(out, columns=["session_id", "market", "value"])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, samples: int, confidence: float, seed: int = 0) -> dict[str, float]:
    """Relative difference (b - a) / a of the means of paired per-query metrics, with a percentile interval."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    idx = rng.integers(0, n, (samples, n))
    ma, mb = a[idx].mean(axis=1), b[idx].mean(axis=1)
    rel = (mb - ma) / np.maximum(ma, 1e-9)
    alpha = (1 - confidence) / 2
    return {
        "delta": float((b.mean() - a.mean()) / max(a.mean(), 1e-9)),
        "low": float(np.quantile(rel, alpha)),
        "high": float(np.quantile(rel, 1 - alpha)),
        "n": int(n),
        "champion": float(a.mean()),
        "candidate": float(b.mean()),
    }


Scorer = Callable[[pd.DataFrame], np.ndarray]


def compare(
    holdout: pd.DataFrame, candidate: Scorer, champion: Scorer, k: int = 10, propensity: dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg = config.load("gates")["model"]
    markets = config.load("gates")["segments"]["market"]
    a = dcg_per_query(holdout, champion(holdout), k, propensity)
    b = dcg_per_query(holdout, candidate(holdout), k, propensity)
    merged = a.merge(b, on=["session_id", "market"], suffixes=("_champion", "_candidate"))
    overall = paired_bootstrap(
        merged["value_champion"].to_numpy(),
        merged["value_candidate"].to_numpy(),
        cfg["bootstrap_samples"],
        cfg["confidence"],
    )
    per_market = {}
    for m in markets:
        g = merged[merged["market"] == m]
        per_market[m] = paired_bootstrap(
            g["value_champion"].to_numpy(),
            g["value_candidate"].to_numpy(),
            cfg["bootstrap_samples"],
            cfg["confidence"],
            seed=1,
        )
    weighting = f"ips (eta {propensity['eta']})" if propensity and propensity.get("eta") is not None else "none"
    return {"metric": f"ips-dcg@{k}", "weighting": weighting, "overall": overall, "markets": per_market}


def run(comparison: dict[str, Any], importance: dict[str, float], kind: str = "model") -> list[GateResult]:
    cfg = config.load("gates")["model" if kind == "model" else "tower"]
    min_q = config.load("gates")["data"]["min_queries_per_market"]
    overall = comparison["overall"]
    metric = comparison["metric"]
    results = [
        GateResult(
            "model.non_inferiority",
            overall["low"] > -cfg["non_inferiority_margin"],
            f"{metric} {overall['delta']:+.2%} vs production (95% interval {overall['low']:+.2%} to {overall['high']:+.2%})",
            {**overall, "margin": cfg["non_inferiority_margin"]},
        )
    ]
    worse = {}
    for m, r in comparison["markets"].items():
        if r["n"] < min_q:
            continue
        if r["low"] <= -cfg["segment_non_inferiority_margin"]:
            worse[m] = r
    results.append(
        GateResult(
            "model.segment_non_inferiority",
            not worse,
            "no market is worse"
            if not worse
            else "worse in " + ", ".join(f"{m} ({r['delta']:+.1%})" for m, r in worse.items()),
            {"markets": comparison["markets"], "margin": cfg["segment_non_inferiority_margin"], "min_queries": min_q},
        )
    )
    results.append(
        GateResult(
            "model.too_good_to_be_true",
            overall["delta"] <= cfg["too_good_to_be_true"],
            f"gain {overall['delta']:+.2%} is plausible"
            if overall["delta"] <= cfg["too_good_to_be_true"]
            else f"gain {overall['delta']:+.2%} is above {cfg['too_good_to_be_true']:.0%}: held for review (leakage, a label bug)",
            {"delta": overall["delta"], "limit": cfg["too_good_to_be_true"]},
        )
    )
    if importance:
        top = max(importance, key=lambda f: importance[f])
        share = importance[top]
        limit = config.load("gates")["model"]["importance_concentration_max"]
        results.append(
            GateResult(
                "model.importance_concentration",
                share <= limit,
                f"largest feature is {top} with {share:.0%} of gain",
                {"importance": importance, "top": top, "share": share, "limit": limit},
            )
        )
    return results


def frame_scorer(features: list[str], predict: Callable[[np.ndarray], np.ndarray]) -> Scorer:
    return lambda frame: predict(matrix(frame, features))
