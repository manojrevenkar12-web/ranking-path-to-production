"""The services around the models, as the platform sees them: the listing feed into Vespa, the counter service, the
profile service, the search and homepage services that query Vespa (and Triton) and log every impression with the
match-features Vespa returned, and the daily exports to the lake. Incidents from the scenario are applied here, where
they would happen in production."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ptp import config
from ptp.experiment import interleaving as IL
from ptp.lake import DAY, Lake, day_of
from ptp.serving.vespa.client import dense_tensor, mapped_tensor
from ptp.training import tower_features as TF

from .emulators.vespa import VespaEmulator
from .world import CATEGORIES, World, position_bias

Routing = Callable[[str], dict[str, Any] | None]
SESSION_COLUMNS = [
    "session_id", "ts", "user_id", "market", "surface", "profile", "arm", "hits", "purchases",
    "canary", "credit_candidate", "credit_production", "candidate_empty",
]  # fmt: skip


@dataclass
class Incidents:
    """World-side incidents, keyed by season day."""

    raw: list[dict[str, Any]] = field(default_factory=list)

    def active(self, kind: str, day: int) -> list[dict[str, Any]]:
        out = []
        for inc in self.raw:
            if inc["type"] != kind:
                continue
            start = int(inc.get("from_day", inc.get("day", 0)))
            end = int(inc.get("to_day", inc.get("day", 10_000) if "day" in inc else 10_000))
            if start <= day <= end:
                out.append(inc)
        return out


@dataclass
class SessionLog:
    rows: list[dict[str, Any]] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)


class Marketplace:
    def __init__(
        self,
        world: World,
        prod: VespaEmulator,
        staging: VespaEmulator,
        lake: Lake,
        incidents: Incidents,
        routing: Routing,
        triton: Any = None,
    ):
        self.w = world
        self.prod = prod
        self.staging = staging
        self.lake = lake
        self.incidents = incidents
        self.routing = routing
        self.triton = triton
        cap = len(world.item_price)
        self.imp = np.zeros((7, cap), dtype=np.int32)
        self.clk = np.zeros((7, cap), dtype=np.int32)
        self.fav = np.zeros(cap, dtype=np.int32)
        self.fav_today = np.zeros(cap, dtype=np.int32)
        self.fav_flushed = np.zeros(cap, dtype=np.int32)
        n = world.cfg.users
        self.p_brand = np.zeros((n, world.cfg.brands), dtype=np.float32)
        self.p_band_sum = np.zeros(n)
        self.p_band_n = np.zeros(n)
        self.p_cat = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
        self.day = 0
        self.log = SessionLog()
        self.served: list[
            dict[str, Any]
        ] = []  # per-session record for evaluation: surface, served profile, user, ranked items
        self.cents_markets: set[int] = set()
        self.rng = np.random.default_rng(world.cfg.seed + 101)
        self.session_seq = 0
        self._flushed = 0

    # ---- time -----------------------------------------------------------------------------------------------------

    def t(self, day: int, seconds: float = 0) -> int:
        return self.w.cfg.t0 + day * DAY + int(seconds)

    # ---- feed -----------------------------------------------------------------------------------------------------

    def doc(self, i: int) -> dict[str, Any]:
        w = self.w
        m = int(w.item_market[i])
        price = float(w.item_price[i])
        if m in self.cents_markets:
            price = round(price * 100)
        brand = w.brands[int(w.item_brand[i])]
        return {
            "item_id": f"it{i:06d}",
            "market": w.markets[m],
            "category": CATEGORIES[int(w.item_cat[i])],
            "brand": brand,
            "brand_t": {"cells": [{"address": {"brand": brand}, "value": 1.0}]},
            "price": price,
            "category_median_price": round(w.median_price(m, int(w.item_cat[i])), 2),
            "condition": int(w.item_cond[i]),
            "photo_quality": round(float(w.item_photo[i]), 3),
            "seller_rating": round(float(w.item_seller[i]), 2),
            "listed_at": int(w.item_listed[i]),
            "clicks_7d": int(self.clk[:, i].sum()),
            "impressions_7d": int(self.imp[:, i].sum()),
            "favourites": int(self.fav[i]),
            "favourites_today": int(self.fav_today[i]),
            "sold": bool(w.item_sold[i]),
            "emb_versions": [],
        }

    def feed(self, idx: np.ndarray) -> None:
        for i in idx:
            d = self.doc(int(i))
            for env in (self.prod, self.staging):
                status, body = env.put(d["item_id"], d)
                if status != 200:
                    raise RuntimeError(f"feed failed: {body}")

    def update(self, i: int, ops: dict[str, Any]) -> None:
        for env in (self.prod, self.staging):
            env.update(f"it{i:06d}", {k: {"assign": v} for k, v in ops.items()}, create=False)

    def remove(self, idx: np.ndarray) -> None:
        for i in idx:
            for env in (self.prod, self.staging):
                env.remove(f"it{int(i):06d}")

    # ---- profile service --------------------------------------------------------------------------------------------

    def user_inputs(self, u: int, cat: int | None) -> dict[str, Any]:
        bw = self.p_brand[u]
        top = np.argsort(-bw)[:8]
        mx = float(bw[top[0]]) if len(top) else 0.0
        brands = {self.w.brands[b]: float(bw[b] / mx) for b in top if mx > 0 and bw[b] > 0}
        band = float(self.p_band_sum[u] / (self.p_band_n[u] + 3.0))
        out: dict[str, Any] = {"brands": brands, "band": band}
        if cat is not None:
            out["log_price"] = math.log(self.w.median_price(int(self.w.user_market[u]), cat)) + band
        return out

    def profile_row(self, u: int) -> dict[str, Any]:
        ins = self.user_inputs(u, None)
        return {
            "user_id": f"u{u:05d}",
            "market": self.w.markets[int(self.w.user_market[u])],
            "band": ins["band"],
            "brand_weights": json.dumps(ins["brands"]),
            "cat_interest": self.p_cat[u].tolist(),
        }

    def learn(self, u: int, i: int, weight: float, now: int) -> None:
        w = self.w
        self.p_brand[u, int(w.item_brand[i])] += weight
        m = int(w.item_market[i])
        self.p_band_sum[u] += weight * math.log(w.item_price[i] / w.median_price(m, int(w.item_cat[i])))
        self.p_band_n[u] += weight
        self.p_cat[u, int(w.item_cat[i])] += weight

    # ---- sessions ---------------------------------------------------------------------------------------------------

    def _route_production(self, surface: str) -> str:
        r = self.routing(surface)
        return r["production"] if r else "popularity"

    def _pick_profile(self, surface: str) -> tuple[str, str]:
        r = self.routing(surface)
        if not r:
            return ("popularity", "production")
        if r.get("canary") and self.rng.random() < float(r.get("share") or 0):
            return (r["canary"], "canary")
        return (r["production"], "production")

    def _search(
        self, profile: str, u: int, cat: int, ins: dict[str, Any], now: int
    ) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
        w = self.w
        market = w.markets[int(w.user_market[u])]
        body = {
            "yql": f'select * from item where market contains "{market}" and category contains "{CATEGORIES[cat]}" and sold = false',
            "ranking": profile,
            "hits": w.cfg.hits,
            "input.query(now)": float(now),
            "input.query(user_brand)": mapped_tensor("brand", ins["brands"]),
            "input.query(user_log_price)": float(ins["log_price"]),
        }
        status, res = self.prod.search(body)
        children = res.get("root", {}).get("children", []) if status == 200 else []
        return status == 200, children, body

    def _interleave(
        self, prod_children: list[dict[str, Any]], cand_children: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        by_id = {c["fields"]["item_id"]: c for c in prod_children}
        by_id.update({c["fields"]["item_id"]: c for c in cand_children if c["fields"]["item_id"] not in by_id})
        merged, teams = IL.team_draft(
            [c["fields"]["item_id"] for c in prod_children],
            [c["fields"]["item_id"] for c in cand_children],
            self.w.cfg.hits,
            self.rng,
        )
        return [by_id[i] for i in merged], teams

    def search_session(self, u: int, now: int) -> None:
        w = self.w
        cat = int(self.rng.choice(len(CATEGORIES), p=w.user_cat[u]))
        profile, arm = self._pick_profile("search")
        ins = self.user_inputs(u, cat)
        teams: list[str] | None = None
        canary = None
        cand_empty = False
        if arm == "canary":
            route = self.routing("search") or {}
            canary = profile
            ok_p, prod_children, body = self._search(route.get("production", "popularity"), u, cat, ins, now)
            ok_c, cand_children, _ = self._search(profile, u, cat, ins, now)
            cand_empty = (not ok_c or not cand_children) and bool(prod_children)
            children, teams = self._interleave(prod_children, cand_children if ok_c else [])
            ok = ok_p
            arm = "interleave"
        else:
            ok, children, body = self._search(profile, u, cat, ins, now)
        self.session_seq += 1
        sid = f"s{self.session_seq:08d}"
        if children and arm == "production" and self.rng.random() < w.cfg.explore_share:
            # exploration: the same results in random order, so click rates by position measure position bias alone
            children = [children[i] for i in self.rng.permutation(len(children))]
            arm = "explore"
        items = np.array([int(c["fields"]["item_id"][2:]) for c in children], dtype=int)
        self.served.append(
            {
                "surface": "search",
                "profile": canary or profile,
                "arm": arm,
                "user": u,
                "now": now,
                "items": items,
                "ok": ok,
                "cat": cat,
            }
        )
        request = {
            "q_user_brand": body["input.query(user_brand)"],
            "q_user_log_price": body["input.query(user_log_price)"],
        }
        self._behave(
            sid,
            u,
            now,
            "search",
            CATEGORIES[cat],
            profile if not teams else "interleave",
            children,
            items,
            arm,
            request,
            teams,
            canary,
            cand_empty,
        )

    def _home(self, version: str, u: int, now: int) -> tuple[bool, list[dict[str, Any]]]:
        w = self.w
        market = w.markets[int(w.user_market[u])]
        if version != "popularity" and self.triton is not None:
            emb = self.triton.embed(version, TF.user_vector(self.profile_row(u)))
            if emb is None:
                return False, []
            body = {
                "yql": f'select * from item where ({{targetHits:100}}nearestNeighbor(emb_{version}, user_emb)) and market contains "{market}" and sold = false',
                "ranking": f"homepage_{version}",
                "hits": w.cfg.hits,
                "input.query(user_emb)": dense_tensor(emb.tolist()),
                "input.query(now)": float(now),
            }
        else:
            body = {
                "yql": f'select * from item where market contains "{market}" and sold = false',
                "ranking": "popularity",
                "hits": w.cfg.hits,
                "input.query(now)": float(now),
            }
        status, res = self.prod.search(body)
        return status == 200, res.get("root", {}).get("children", []) if status == 200 else []

    def home_session(self, u: int, now: int) -> None:
        version, arm = self._pick_profile("home")
        teams: list[str] | None = None
        canary = None
        cand_empty = False
        if arm == "canary":
            route = self.routing("home") or {}
            canary = version
            ok, prod_children = self._home(route.get("production", "popularity"), u, now)
            ok_c, cand_children = self._home(version, u, now)
            cand_empty = (not ok_c or not cand_children) and bool(prod_children)
            children, teams = self._interleave(prod_children, cand_children if ok_c else [])
            arm = "interleave"
        else:
            ok, children = self._home(version, u, now)
        self.session_seq += 1
        sid = f"s{self.session_seq:08d}"
        items = np.array([int(c["fields"]["item_id"][2:]) for c in children], dtype=int)
        self.served.append(
            {
                "surface": "home",
                "profile": canary or version,
                "arm": arm,
                "user": u,
                "now": now,
                "items": items,
                "ok": ok,
                "cat": None,
            }
        )
        self._behave(
            sid,
            u,
            now,
            "home",
            "",
            version if not teams else "interleave",
            children,
            items,
            arm,
            None,
            teams,
            canary,
            cand_empty,
        )

    def _behave(
        self,
        sid: str,
        u: int,
        now: int,
        surface: str,
        query: str,
        profile: str,
        children: list[dict[str, Any]],
        items: np.ndarray,
        arm: str,
        request: dict[str, Any] | None = None,
        teams: list[str] | None = None,
        canary: str | None = None,
        cand_empty: bool = False,
    ) -> None:
        w = self.w
        market = w.markets[int(w.user_market[u])]
        base = {
            "session_id": sid,
            "ts": now,
            "user_id": f"u{u:05d}",
            "market": market,
            "surface": surface,
            "profile": profile,
            "arm": arm,
            "canary": canary,
            "candidate_empty": cand_empty,
        }
        if len(items) == 0:
            self.log.sessions.append(
                {**base, "hits": 0, "purchases": 0, "credit_candidate": 0.0, "credit_production": 0.0}
            )
            return
        util = w.utility(u, items, now, surface)
        bias = position_bias(len(items))
        pc = bias * w.p_click(util)
        clicked = self.rng.random(len(items)) < pc
        fav = clicked & (self.rng.random(len(items)) < w.p_favourite(util))
        bought = clicked & (self.rng.random(len(items)) < w.p_purchase(util))
        day = self.day
        slot = day % 7
        purchases = 0
        for pos, i in enumerate(items):
            c = children[pos]
            row = {
                "impression_id": f"{sid}-{pos:02d}",
                "ts": now,
                "session_id": sid,
                "user_id": f"u{u:05d}",
                "market": w.markets[int(w.user_market[u])],
                "surface": surface,
                "query": query,
                "item_id": f"it{int(i):06d}",
                "position": pos,
                "profile": (canary if teams and teams[pos] == IL.CANDIDATE else self._route_production(surface))
                if teams
                else profile,
                "arm": arm,
                "relevance": float(c.get("relevance", 0.0))
                if not isinstance(c.get("relevance"), str)
                else float("nan"),
                "clicked": bool(clicked[pos]),
                "favourited": bool(fav[pos]),
                "purchased": bool(bought[pos]) and not w.item_sold[i],
            }
            if request:
                row.update(request)
            for k, v in (c.get("fields", {}).get("matchfeatures") or {}).items():
                row[f"f_{k}"] = float(v) if not isinstance(v, str) else float("nan")
            self.log.rows.append(row)
            self.imp[slot, i] += 1
            if clicked[pos]:
                self.clk[slot, i] += 1
                self.learn(u, int(i), 1.0, now)
            if fav[pos]:
                self.fav[i] += 1
                self.fav_today[i] += 1
                self.learn(u, int(i), 1.0, now)
            if row["purchased"]:
                purchases += 1
                w.item_sold[i] = True
                self.update(int(i), {"sold": True})
                self.learn(u, int(i), 2.0, now)
        cc = cp = 0.0
        if teams:
            weights = config.load("gates")["canary"]["credit"]
            bought_logged = [bool(b) for b in bought]
            cc, cp = IL.credit(teams, clicked.tolist(), fav.tolist(), bought_logged, weights)
        self.log.sessions.append(
            {**base, "hits": len(items), "purchases": purchases, "credit_candidate": cc, "credit_production": cp}
        )

    # ---- a day ------------------------------------------------------------------------------------------------------

    def start_day(self, day: int) -> None:
        self.day = day
        w = self.w
        now = self.t(day)
        active = {w.markets.index(inc["market"]) for inc in self.incidents.active("price_cents", day)}
        for m in active ^ self.cents_markets:
            # the listing service republishes the market: in cents when the bug ships, in euros when it is fixed
            self.cents_markets = active | (self.cents_markets - {m}) if m in active else self.cents_markets - {m}
            live = w.live(now)
            self.feed(live[w.item_market[live] == m])
        self.cents_markets = active
        slot = day % 7
        self.imp[slot] = 0
        self.clk[slot] = 0
        self.fav_today[:] = 0
        self.fav_flushed[:] = 0
        live = w.live(now)
        for i in live:
            self.update(
                int(i),
                {
                    "favourites_today": 0,
                    "clicks_7d": int(self.clk[:, i].sum()),
                    "impressions_7d": int(self.imp[:, i].sum()),
                    "favourites": int(self.fav[i]),
                },
            )
        expired = w.expire(now)
        self.remove(expired)
        new = w.list_items(w.cfg.new_items_per_day, now)
        self.feed(new)

    def flush_counters(self) -> None:
        changed = np.nonzero(self.fav_today != self.fav_flushed)[0]
        for i in changed:
            self.update(int(i), {"favourites_today": int(self.fav_today[i]), "favourites": int(self.fav[i])})
        self.fav_flushed[changed] = self.fav_today[changed]

    def run_day(self, day: int, on_hour: Callable[[int, int], None] | None = None) -> None:
        w = self.w
        self.start_day(day)
        n_s, n_h = w.cfg.search_sessions_per_day, w.cfg.homepage_sessions_per_day
        act = w.user_activity / w.user_activity.sum()
        users_s = self.rng.choice(w.cfg.users, n_s, p=act)
        users_h = self.rng.choice(w.cfg.users, n_h, p=act)
        times_s = np.sort(self.rng.uniform(0, DAY, n_s))
        times_h = np.sort(self.rng.uniform(0, DAY, n_h))
        events = sorted(
            [(t, "s", u) for t, u in zip(times_s, users_s, strict=True)]
            + [(t, "h", u) for t, u in zip(times_h, users_h, strict=True)]
        )
        hour = -1
        for t, kind, u in events:
            h = int(t // 3600)
            while hour < h:
                if hour >= 0:
                    self.flush_hour(day, hour)
                hour += 1
                self.flush_counters()
                if on_hour:
                    on_hour(day, hour)
            now = self.t(day, t)
            if kind == "s":
                self.search_session(int(u), now)
            else:
                self.home_session(int(u), now)
        while hour < 23:
            if hour >= 0:
                self.flush_hour(day, hour)
            hour += 1
            if on_hour:
                on_hour(day, hour)
        self.flush_hour(day, 23)
        self.flush_counters()
        self.end_day(day)

    def flush_hour(self, day: int, hour: int) -> None:
        start = self.t(day, hour * 3600)
        rows = [r for r in self.log.sessions[self._flushed :] if r["ts"] < start + 3600]
        self._flushed += len(rows)
        self.lake.write_hour("sessions", start, pd.DataFrame(rows, columns=SESSION_COLUMNS))

    def end_day(self, day: int) -> None:
        w = self.w
        date = day_of(self.t(day))
        end = self.t(day + 1)
        frame = pd.DataFrame(self.log.rows)
        if not frame.empty:
            for inc in self.incidents.active("null_feature", day):
                mask = frame["market"] == inc["market"]
                frame.loc[mask, f"f_{inc['feature']}"] = np.nan
            for inc in self.incidents.active("click_drop", day):
                mask = (frame["market"] == inc["market"]) & frame["clicked"]
                drop = mask & (self.rng.random(len(frame)) < float(inc["rate"]))
                frame.loc[drop, "clicked"] = False
            for inc in self.incidents.active("dup_events", day):
                mask = frame["market"] == inc["market"]
                dup = frame[mask].sample(frac=float(inc["rate"]), random_state=day)
                frame = pd.concat([frame, dup], ignore_index=True)
        landed = end + 1800
        for inc in self.incidents.active("late_partition", day):
            landed = end + int(float(inc.get("hours", 30)) * 3600)
        self.lake.write("impressions", date, frame, landed)
        live = w.live(end)
        snap = (
            pd.DataFrame([self.doc(int(i)) for i in live]).drop(columns=["brand_t", "emb_versions"])
            if len(live)
            else pd.DataFrame()
        )
        if not snap.empty:
            snap["as_of"] = end
            for inc in self.incidents.active("snapshot_missing_market", day):
                snap = snap[snap["market"] != inc["market"]]
        self.lake.write("items", date, snap, end + 1800)
        users = pd.DataFrame([self.profile_row(u) for u in range(w.cfg.users)])
        self.lake.write("users", date, users, end + 1800)
        # profiles decay a little each day
        self.p_brand *= 0.97
        self.p_cat *= 0.97
        self.log = SessionLog()
        self._flushed = 0

    def initial_listings(self) -> None:
        now = self.t(0)
        idx = self.w.list_items(self.w.cfg.initial_items, now)
        self.w.item_listed[idx] = now - self.rng.integers(0, 20 * DAY, len(idx))
        self.feed(idx)

    def as_date(self, day: int) -> dt.date:
        return day_of(self.t(day))
