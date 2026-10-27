"""The search ranker: LightGBM LambdaRank on graded labels, one query group per search session.

This is the data scientists' training code moved out of a notebook (ad_hoc/train_ranker.py) without changing what it
computes; tests/test_migration.py checks that both produce the same model on the same data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data.build import groups, matrix


@dataclass
class Trained:
    booster: lgb.Booster
    features: list[str]
    importance: dict[str, float]  # share of total gain

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(x), dtype=float)


def _weighted(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Market weighting by repeating whole query groups (LambdaRank weights are per group)."""
    if not weights:
        return frame
    parts = []
    for market, g in frame.groupby("market", sort=False):
        times = max(1, round(weights.get(str(market), 1.0)))
        for k in range(times):
            dup = g.copy()
            if k:
                dup["session_id"] = dup["session_id"].astype(str) + f"#{k}"
            parts.append(dup)
    return pd.concat(parts).sort_values(["session_id", "position"], kind="stable").reset_index(drop=True)


def train(
    train_frame: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    market_weights: dict[str, float] | None = None,
    position_debias: bool = False,
) -> Trained:
    """position_debias: LightGBM's unbiased LambdaMART, which learns a bias factor per logged position alongside the
    trees, so the model is not rewarded for agreeing with the order the logs were shown in."""
    frame = _weighted(train_frame, market_weights or {})
    x = matrix(frame, features)
    y = frame["label"].to_numpy()
    g = groups(frame)
    p = dict(params)
    p.setdefault("verbose", -1)
    if position_debias:
        rounds = int(p.pop("n_estimators", 100))
        ds = lgb.Dataset(x, y, group=g, position=frame["position"].to_numpy(), free_raw_data=False)
        booster = lgb.train(p, ds, num_boost_round=rounds)
    else:
        model = lgb.LGBMRanker(**p)
        model.fit(x, y, group=g)
        booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    total = float(gain.sum()) or 1.0
    importance = {f: float(v) / total for f, v in zip(features, gain, strict=True)}
    return Trained(booster, features, importance)
