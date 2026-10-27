"""Input vectors of the homepage two-tower model. One implementation, used by training, by the item-embedding batch
scoring and by the homepage service that calls Triton, so the towers see the same features everywhere."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .. import config

_raw = config.load("features")
MARKETS: list[str] = list(_raw["taxonomy"]["markets"])
CATEGORIES: list[str] = list(_raw["taxonomy"]["categories"])
BRAND_BUCKETS: int = int(_raw["tower"]["brand_buckets"])


def brand_bucket(brand: str) -> int:
    return int(hashlib.md5(brand.encode()).hexdigest(), 16) % BRAND_BUCKETS


def user_dim() -> int:
    return len(MARKETS) + len(CATEGORIES) + 1 + BRAND_BUCKETS


def item_dim() -> int:
    return len(CATEGORIES) + BRAND_BUCKETS + 3


def user_vector(profile: dict[str, Any]) -> np.ndarray:
    """profile: market, cat_interest (list per CATEGORIES), band, brand_weights (dict brand -> weight)."""
    v = np.zeros(user_dim(), dtype=np.float32)
    if profile.get("market") in MARKETS:
        v[MARKETS.index(profile["market"])] = 1.0
    raw = profile.get("cat_interest")
    ci = np.asarray(
        raw if raw is not None and len(raw) == len(CATEGORIES) else [0.0] * len(CATEGORIES), dtype=np.float32
    )
    s = float(ci.sum())
    if s > 0:
        v[len(MARKETS) : len(MARKETS) + len(CATEGORIES)] = ci / s
    v[len(MARKETS) + len(CATEGORIES)] = float(profile.get("band", 0.0))
    bw = profile.get("brand_weights") or {}
    if isinstance(bw, str):
        bw = json.loads(bw)
    base = len(MARKETS) + len(CATEGORIES) + 1
    for brand, w in bw.items():
        v[base + brand_bucket(brand)] = max(v[base + brand_bucket(brand)], float(w))
    return v


def item_matrix(items: Any) -> np.ndarray:
    """items: a frame with category, brand, price, category_median_price, condition, photo_quality."""
    n = len(items)
    m = np.zeros((n, item_dim()), dtype=np.float32)
    cats = items["category"].tolist()
    brands = items["brand"].tolist()
    for i in range(n):
        if cats[i] in CATEGORIES:
            m[i, CATEGORIES.index(cats[i])] = 1.0
        m[i, len(CATEGORIES) + brand_bucket(brands[i])] = 1.0
    base = len(CATEGORIES) + BRAND_BUCKETS
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log(
            np.maximum(items["price"].to_numpy(float), 0.01)
            / np.maximum(items["category_median_price"].to_numpy(float), 0.01)
        )
    m[:, base] = np.clip(rel, -3, 3)
    m[:, base + 1] = items["condition"].to_numpy(float) / 3.0
    m[:, base + 2] = items["photo_quality"].to_numpy(float)
    return m
