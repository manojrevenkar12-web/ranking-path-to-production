"""The simulated marketplace: members with latent tastes, single-item second-hand listings, and the true utility a
member gets from seeing an item. Only the simulator and src/ptp/evaluate.py (through truth.json) know these.

What the platform sees is what a real one sees: the documents the listing service feeds to Vespa, the features
Vespa computes, the impressions, clicks, favourites and purchases the search service logs, and the user profiles
the profile service builds from those logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DAY = 86400
CATEGORIES = ["dresses", "jeans", "sneakers", "jackets", "bags", "tops", "kids", "home", "books", "coats"]
BASE_PRICE = {
    "dresses": 18,
    "jeans": 16,
    "sneakers": 32,
    "jackets": 30,
    "bags": 26,
    "tops": 9,
    "kids": 8,
    "home": 14,
    "books": 6,
    "coats": 38,
}


@dataclass
class MarketParams:
    share: float
    price_weight: float  # how much a member here cares about price fit
    brand_weight: float  # how much brand taste matters
    quality_weight: float
    price_level: float  # multiplies base prices


@dataclass
class WorldConfig:
    seed: int = 1
    t0: int = 1_767_225_600  # 2026-01-01T00:00:00Z
    users: int = 5000
    brands: int = 40
    initial_items: int = 6000
    new_items_per_day: int = 700
    listing_days: int = 30
    search_sessions_per_day: int = 1800
    homepage_sessions_per_day: int = 600
    hits: int = 20
    explore_share: float = 0.1  # search sessions whose top results are shown in random order (propensity data)
    markets: dict[str, MarketParams] = field(
        default_factory=lambda: {
            "FR": MarketParams(0.40, 1.0, 1.4, 1.0, 1.00),
            "DE": MarketParams(0.30, 1.2, 1.0, 1.1, 1.10),
            "PL": MarketParams(0.20, 2.0, 0.6, 0.8, 0.55),
            "LT": MarketParams(0.10, 1.6, 0.8, 0.9, 0.60),
        }
    )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorldConfig:
        raw = dict(raw)
        markets = raw.pop("markets", None)
        cfg = cls(**raw)
        if markets:
            cfg.markets = {k: MarketParams(**v) for k, v in markets.items()}
        return cfg


def position_bias(n: int) -> np.ndarray:
    return 1.0 / np.power(np.arange(1, n + 1), 0.75)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class World:
    """Latent state. Item arrays grow; item i has id f"it{i:06d}"."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        rng = self.rng
        self.markets = list(cfg.markets)
        shares = np.array([cfg.markets[m].share for m in self.markets])
        self.market_p = shares / shares.sum()
        self.brands = [f"b{i:02d}" for i in range(cfg.brands)]
        self.brand_premium = np.exp(rng.normal(0, 0.35, cfg.brands))
        self.brand_quality = rng.normal(0, 0.5, cfg.brands)
        # which categories a brand sells
        self.brand_cat = rng.dirichlet(np.full(len(CATEGORIES), 0.4), cfg.brands)

        n = cfg.users
        self.user_market = rng.choice(len(self.markets), n, p=self.market_p)
        self.user_band = rng.normal(0, 0.35, n)  # preferred price relative to the market median
        self.user_cat = rng.dirichlet(np.full(len(CATEGORIES), 0.5), n)
        self.user_activity = rng.lognormal(0, 0.8, n)
        pref = np.zeros((n, cfg.brands), dtype=np.float32)
        popular = rng.dirichlet(np.full(cfg.brands, 0.6))
        for u in range(n):
            k = int(rng.integers(3, 7))
            liked = rng.choice(cfg.brands, k, replace=False, p=popular)
            w = rng.dirichlet(np.ones(k))
            pref[u, liked] = w / w.max()
        self.user_brand = pref

        cap = cfg.initial_items + cfg.new_items_per_day * 400
        self.n_items = 0
        self.item_market = np.zeros(cap, dtype=np.int16)
        self.item_cat = np.zeros(cap, dtype=np.int16)
        self.item_brand = np.zeros(cap, dtype=np.int16)
        self.item_price = np.zeros(cap)  # euros, true
        self.item_cond = np.zeros(cap, dtype=np.int8)
        self.item_photo = np.zeros(cap)
        self.item_seller = np.zeros(cap)
        self.item_quality = np.zeros(cap)
        self.item_listed = np.zeros(cap, dtype=np.int64)
        self.item_sold = np.zeros(cap, dtype=bool)
        self.item_expired = np.zeros(cap, dtype=bool)

    # ---- catalogue --------------------------------------------------------------------------------------------

    def median_price(self, market: int, cat: int) -> float:
        return BASE_PRICE[CATEGORIES[cat]] * self.cfg.markets[self.markets[market]].price_level

    def list_items(self, count: int, now: int) -> np.ndarray:
        rng = self.rng
        start = self.n_items
        idx = np.arange(start, start + count)
        self.n_items += count
        m = rng.choice(len(self.markets), count, p=self.market_p)
        brand = rng.integers(0, self.cfg.brands, count)
        cat = np.array([rng.choice(len(CATEGORIES), p=self.brand_cat[b]) for b in brand])
        cond = rng.choice(4, count, p=[0.15, 0.35, 0.35, 0.15])
        quality = rng.normal(0, 1, count) + self.brand_quality[brand]
        med = np.array([self.median_price(int(mm), int(c)) for mm, c in zip(m, cat, strict=True)])
        price = med * self.brand_premium[brand] * (0.7 + 0.15 * cond) * np.exp(rng.normal(0, 0.45, count))
        self.item_market[idx] = m
        self.item_cat[idx] = cat
        self.item_brand[idx] = brand
        self.item_price[idx] = np.round(np.maximum(price, 1.0), 2)
        self.item_cond[idx] = cond
        self.item_photo[idx] = np.clip(0.55 + 0.18 * quality + rng.normal(0, 0.15, count), 0, 1)
        self.item_seller[idx] = np.clip(4.3 + 0.25 * quality + rng.normal(0, 0.3, count), 1, 5)
        self.item_quality[idx] = quality
        self.item_listed[idx] = now + rng.integers(0, 3600, count)  # listed during the first hour of the day
        return idx

    def live(self, now: int) -> np.ndarray:
        n = self.n_items
        return np.nonzero(~self.item_sold[:n] & ~self.item_expired[:n] & (self.item_listed[:n] <= now))[0]

    def expire(self, now: int) -> np.ndarray:
        n = self.n_items
        old = (now - self.item_listed[:n] > self.cfg.listing_days * DAY) & ~self.item_sold[:n] & ~self.item_expired[:n]
        idx = np.nonzero(old)[0]
        self.item_expired[idx] = True
        return idx

    # ---- truth --------------------------------------------------------------------------------------------------

    def utility(self, user: int, items: np.ndarray, now: int, surface: str = "search") -> np.ndarray:
        mp = self.cfg.markets[self.markets[int(self.user_market[user])]]
        cat = self.item_cat[items]
        med = np.array([self.median_price(int(self.user_market[user]), int(c)) for c in cat])
        price_fit = np.abs(np.log(self.item_price[items]) - (np.log(med) + self.user_band[user]))
        age = np.maximum(now - self.item_listed[items], 0) / DAY
        u = (
            mp.brand_weight * self.user_brand[user, self.item_brand[items]]
            - mp.price_weight * price_fit
            + mp.quality_weight * 0.45 * self.item_quality[items]
            + 0.35 * self.item_cond[items] / 3.0
            - 0.25 * np.log1p(age)
            + 0.35 * _noise(user, items)
        )
        if surface == "home":
            u = u + 2.2 * self.user_cat[user, cat] - 0.4
        return u

    @staticmethod
    def p_click(u: np.ndarray) -> np.ndarray:
        return sigmoid(1.6 * u + 1.0)

    @staticmethod
    def p_purchase(u: np.ndarray) -> np.ndarray:
        return sigmoid(1.3 * u - 0.8)

    @staticmethod
    def p_favourite(u: np.ndarray) -> np.ndarray:
        return sigmoid(1.0 * u - 1.0)

    def expected_purchases(self, user: int, ranked: np.ndarray, now: int, surface: str = "search") -> float:
        if len(ranked) == 0:
            return 0.0
        u = self.utility(user, ranked, now, surface)
        return float(np.sum(position_bias(len(ranked)) * self.p_click(u) * self.p_purchase(u)))


def _noise(user: int, items: np.ndarray) -> np.ndarray:
    """A fixed idiosyncratic taste term per (member, item): the same every time it is asked for."""
    x = np.sin(user * 12.9898 + items.astype(float) * 78.233) * 43758.5453
    frac = x - np.floor(x)
    # map uniform to approximately standard normal (logistic quantile, scaled)
    frac = np.clip(frac, 1e-6, 1 - 1e-6)
    return np.log(frac / (1 - frac)) * (math.sqrt(3) / math.pi)
