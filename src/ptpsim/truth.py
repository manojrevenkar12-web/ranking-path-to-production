"""The true effect of a release: expected purchases per session against the model it would replace, on the same
fresh sessions, served exactly as production would serve it (the world's Vespa and Triton runtimes, the live
catalogue with whatever the feed currently gets wrong). Written to truth.json, which only src/ptp/evaluate.py reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ptp import artifacts, contract
from ptp.registry.store import Registry
from ptp.serving.vespa import package as P
from ptp.serving.vespa.client import mapped_tensor
from ptp.training import tower_features as TF

from .emulators.triton import TritonRuntime
from .emulators.vespa import Runtime, VespaEmulator
from .marketplace import Marketplace
from .world import CATEGORIES


@dataclass
class Judge:
    mk: Marketplace
    runtime: Runtime
    triton_runtime: TritonRuntime
    harmful_overall: float
    harmful_market: float
    beneficial: float
    sessions: int = 1200

    def _eval_users(self, day: int, surface: str) -> list[tuple[int, int]]:
        rng = np.random.default_rng(10_000 + day * 7 + (1 if surface == "home" else 0))
        w = self.mk.w
        act = w.user_activity / w.user_activity.sum()
        users = rng.choice(w.cfg.users, self.sessions, p=act)
        return [(int(u), int(rng.choice(len(CATEGORIES), p=w.user_cat[u]))) for u in users]

    def _scratch(self, spec: P.PackageSpec) -> tuple[VespaEmulator | None, str]:
        emu = VespaEmulator(runtime=self.runtime)
        status, body = emu.deploy(P.to_zip(P.build(contract.load(), spec)))
        if status != 200:
            return None, str(body.get("message", body))
        emu.docs = dict(self.mk.prod.docs)
        return emu, ""

    def _search_value(
        self, emu: VespaEmulator, profile: str, now: int, day: int
    ) -> tuple[dict[str, list[float]], bool]:
        w, mk = self.mk.w, self.mk
        per: dict[str, list[float]] = {m: [] for m in w.markets}
        broken = False
        for u, cat in self._eval_users(day, "search"):
            ins = mk.user_inputs(u, cat)
            market = w.markets[int(w.user_market[u])]
            body = {
                "yql": f'select * from item where market contains "{market}" and category contains "{CATEGORIES[cat]}" and sold = false',
                "ranking": profile,
                "hits": w.cfg.hits,
                "input.query(now)": float(now),
                "input.query(user_brand)": mapped_tensor("brand", ins["brands"]),
                "input.query(user_log_price)": float(ins["log_price"]),
            }
            status, res = emu.search(body)
            if status != 200:
                broken = True
                per[market].append(0.0)
                continue
            kids = res["root"]["children"]
            if any(isinstance(k.get("relevance"), str) for k in kids):
                broken = True
            items = np.array([int(k["fields"]["item_id"][2:]) for k in kids], dtype=int)
            per[market].append(w.expected_purchases(u, items, now))
        return per, broken

    def _classify(
        self, cand: dict[str, list[float]], base: dict[str, list[float]], broken: bool, note: str = ""
    ) -> dict[str, Any]:
        markets = {}
        for m in cand:
            c, b = float(np.mean(cand[m])) if cand[m] else 0.0, float(np.mean(base[m])) if base[m] else 0.0
            markets[m] = {"candidate": c, "baseline": b, "delta": (c - b) / b if b else 0.0}
        c_all = float(np.mean([x for v in cand.values() for x in v]))
        b_all = float(np.mean([x for v in base.values() for x in v]))
        overall = (c_all - b_all) / b_all if b_all else 0.0
        if broken:
            klass = "broken"
        elif overall < -self.harmful_overall or any(r["delta"] < -self.harmful_market for r in markets.values()):
            klass = "harmful"
        elif overall > self.beneficial:
            klass = "beneficial"
        else:
            klass = "neutral"
        return {
            "class": klass,
            "overall_delta": overall,
            "markets": markets,
            "candidate": c_all,
            "baseline": b_all,
            "note": note,
        }

    def ranker(self, reg: Registry, version: str, now: int, day: int) -> dict[str, Any]:
        store = artifacts.Store()
        v = reg.version(version)
        assert v
        route = reg.routing("search") or {"production": "popularity"}
        base_profile = route["production"]
        rankers = []
        if base_profile.startswith("ranker_"):
            b = reg.version(base_profile.removeprefix("ranker_"))
            assert b
            rankers.append(P.RankerProfile(b["version"], store.get(b["artifacts"]["onnx"]["key"]), list(b["features"])))
        cand = P.RankerProfile(version, store.get(v["artifacts"]["onnx"]["key"]), list(v["features"]))
        base_emu, _ = self._scratch(P.PackageSpec(rankers=rankers))
        assert base_emu is not None
        base, _ = self._search_value(base_emu, base_profile, now, day)
        emu, err = self._scratch(P.PackageSpec(rankers=[*rankers, cand]))
        if emu is None:
            zero = {m: [0.0] for m in base}
            return {
                **self._classify(zero, base, True, f"production Vespa refuses the package: {err}"),
                "version": version,
                "against": base_profile,
                "day": day,
            }
        cand_v, broken = self._search_value(emu, cand.name, now, day)
        return {**self._classify(cand_v, base, broken), "version": version, "against": base_profile, "day": day}

    # ---- homepage --------------------------------------------------------------------------------------------

    def _home_value(
        self, version: str | None, now: int, day: int, reg: Registry
    ) -> tuple[dict[str, list[float]], bool, str]:
        w, mk = self.mk.w, self.mk
        per: dict[str, list[float]] = {m: [] for m in w.markets}
        live = w.live(now)
        if version is None or version == "popularity":
            emu = VespaEmulator(runtime=self.runtime)
            emu.deploy(P.to_zip(P.build(contract.load(), P.PackageSpec())))
            emu.docs = dict(mk.prod.docs)
            for u, _ in self._eval_users(day, "home"):
                market = w.markets[int(w.user_market[u])]
                _status, res = emu.search(
                    {
                        "yql": f'select * from item where market contains "{market}" and sold = false',
                        "ranking": "popularity",
                        "hits": w.cfg.hits,
                        "input.query(now)": float(now),
                    }
                )
                items = np.array([int(k["fields"]["item_id"][2:]) for k in res["root"]["children"]], dtype=int)
                per[market].append(w.expected_purchases(u, items, now, "home"))
            return per, False, ""
        store = artifacts.Store()
        v = reg.version(version)
        assert v
        import onnx
        import onnxruntime as ort

        u_bytes = store.get(v["artifacts"]["user_onnx"]["key"])
        i_bytes = store.get(v["artifacts"]["item_onnx"]["key"])
        m = onnx.load_from_string(u_bytes)
        if m.ir_version > self.triton_runtime.max_ir_version or any(
            o.version > self.triton_runtime.max_opset for o in m.opset_import if o.domain in ("", "ai.onnx")
        ):
            return {k: [0.0] for k in per}, True, "Triton cannot load the user tower"
        us = ort.InferenceSession(u_bytes, providers=["CPUExecutionProvider"])
        isess = ort.InferenceSession(i_bytes, providers=["CPUExecutionProvider"])
        import pandas as pd

        docs = pd.DataFrame([mk.doc(int(i)) for i in live])
        emb = isess.run(None, {"item": TF.item_matrix(docs)})[0]
        markets = docs["market"].to_numpy()
        for u, _ in self._eval_users(day, "home"):
            market = w.markets[int(w.user_market[u])]
            q = us.run(None, {"user": TF.user_vector(mk.profile_row(u))[None, :]})[0][0]
            mask = markets == market
            idx = np.nonzero(mask)[0]
            top = idx[np.argsort(-(emb[idx] @ q))[: w.cfg.hits]]
            items = live[top]
            per[market].append(w.expected_purchases(u, items, now, "home"))
        return per, False, ""

    def tower(self, reg: Registry, version: str, now: int, day: int) -> dict[str, Any]:
        route = reg.routing("home") or {"production": "popularity"}
        base, _, _ = self._home_value(route["production"], now, day, reg)
        cand, broken, note = self._home_value(version, now, day, reg)
        return {
            **self._classify(cand, base, broken, note),
            "version": version,
            "against": route["production"],
            "day": day,
        }


def session_value(mk: Marketplace) -> None:
    """Expected purchases of every served session, from the truth, for exposure accounting."""
    for s in mk.served:
        if "value" in s:
            continue
        s["value"] = mk.w.expected_purchases(s["user"], s["items"], s["now"], s["surface"]) if len(s["items"]) else 0.0
