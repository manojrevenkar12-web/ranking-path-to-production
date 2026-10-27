"""Steps of the homepage tower pipelines.

ct-tower: data (pairs from interactions, joined to the member profile and item snapshot of the day *before*, so no
feature saw the interaction), train, evaluate against the production tower on recall@20, export both towers, gate,
register.

item-embeddings: batch-score the live catalogue with a tower version's item tower and feed the embeddings into
Vespa, in that version's own field. A tower is switched on only once its field covers the catalogue.
"""

from __future__ import annotations

import datetime as dt
import io
from typing import Any

import numpy as np
import pandas as pd

from .. import artifacts, clock, config
from ..data.build import NoData, end_of, window
from ..export import onnx_export
from ..gates import base as gates_base
from ..gates import model as model_gates
from ..gates.base import GateResult
from ..lake import Lake
from ..registry.store import Registry
from ..training import tower as tower_model
from ..training import tower_features as TF
from .steps import _ws, code_commit, training_config


def _prev(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=1)


def _pairs(lake: Lake, dates: list[dt.date]) -> pd.DataFrame:
    imp = lake.read("impressions", dates)
    pos = imp[imp["clicked"] | imp["favourited"] | imp["purchased"]].copy()
    pos["day"] = pd.to_datetime(pos["ts"], unit="s", utc=True).dt.date
    return pos[["user_id", "item_id", "market", "day", "surface"]]


def _join(lake: Lake, pos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    users_parts = {p.date for p in lake.partitions("users")}
    items_parts = {p.date for p in lake.partitions("items")}
    out_u, out_i, rows = [], [], []
    for day, g in pos.groupby("day"):
        prev = _prev(day)
        if prev not in users_parts or prev not in items_parts:
            continue
        users = lake.read("users", [prev]).set_index("user_id")
        items = lake.read("items", [prev]).set_index("item_id")
        g = g[g["user_id"].isin(users.index) & g["item_id"].isin(items.index)]
        if g.empty:
            continue
        uv = np.stack(
            [TF.user_vector(users.loc[u].to_dict() | {"market": users.loc[u, "market"]}) for u in g["user_id"]]
        )
        iv = TF.item_matrix(items.loc[g["item_id"]].reset_index())
        out_u.append(uv)
        out_i.append(iv)
        rows.append(g)
    if not rows:
        return np.zeros((0, TF.user_dim()), np.float32), np.zeros((0, TF.item_dim()), np.float32), pos.iloc[0:0]
    return np.concatenate(out_u), np.concatenate(out_i), pd.concat(rows, ignore_index=True)


def data(run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any]) -> str:
    reg, store = Registry(), artifacts.Store()
    gcfg = config.load("gates")
    lake = Lake(lake_root)
    try:
        win = window(lake, as_of, gcfg["data"]["window_days"], gcfg["data"]["holdout_days"])
    except NoData as e:
        reg.record_gates(None, run_id, "data", [GateResult("data.freshness", False, str(e), {})])
        reg.finish_run(run_id, "SUCCEEDED", "blocked")
        return "block"
    results = []
    age_h = (as_of - end_of(win.newest)) / 3600
    results.append(
        GateResult(
            "data.freshness",
            age_h <= gcfg["data"]["freshness_max_hours"],
            f"newest partition {win.newest.isoformat()}, {age_h:.0f} h old",
            {"age_hours": age_h},
        )
    )
    # the item snapshot the scoring and the holdout use must cover every market as the day before did
    snaps = [p.date for p in lake.partitions("items", landed_by=as_of)]
    vol_bad = {}
    if len(snaps) >= 2:
        a, b = lake.read("items", [snaps[-2]]), lake.read("items", [snaps[-1]])
        for m in gcfg["segments"]["market"]:
            na, nb = int((a["market"] == m).sum()), int((b["market"] == m).sum())
            if na and abs(nb - na) / na > gcfg["data"]["volume_band"]:
                vol_bad[m] = {"previous": na, "newest": nb}
    results.append(
        GateResult(
            "data.volume",
            not vol_bad,
            "item snapshot covers every market" if not vol_bad else f"item snapshot short in {', '.join(vol_bad)}",
            {"snapshot": vol_bad},
        )
    )
    pos = _pairs(lake, win.dates)
    u, i, rows = _join(lake, pos)
    coverage = len(rows) / max(len(pos), 1)
    results.append(
        GateResult(
            "data.provenance",
            coverage >= gcfg["tower"]["join_coverage_min"],
            f"{coverage:.0%} of interactions joined to the previous day's profile and snapshot",
            {"coverage": coverage, "pairs": len(rows)},
        )
    )
    reg.record_gates(None, run_id, "data", results)
    ws = _ws(run_id)
    rows = rows.assign(split=np.where(rows["day"].isin(win.holdout), "holdout", "train"))
    buf = io.BytesIO()
    np.savez_compressed(buf, users=u, items=i)
    store.put(f"{ws}/pairs.npz", buf.getvalue())
    buf = io.BytesIO()
    rows.astype({"day": str}).to_parquet(buf, index=False)
    store.put(f"{ws}/pairs.parquet", buf.getvalue())
    store.put_json(
        f"{ws}/data.json",
        {
            "window": [d.isoformat() for d in win.dates],
            "holdout": [d.isoformat() for d in win.holdout],
            "snapshot": Lake.snapshot_id(win.partitions),
            "training": training_config("tower", overrides),
            "pairs": len(rows),
        },
    )
    passed = gates_base.decision(results) == "pass"
    if not passed:
        reg.finish_run(run_id, "SUCCEEDED", "blocked")
    return "pass" if passed else "block"


def _champion_embedders(reg: Registry, store: artifacts.Store) -> tuple[str, Any]:
    champ = reg.production("tower")
    if champ is None:
        return "popularity", None
    a = champ["artifacts"]
    return champ["version"], (
        store.get(a["user_onnx"]["key"], a["user_onnx"]["sha256"]),
        store.get(a["item_onnx"]["key"], a["item_onnx"]["sha256"]),
    )


def recall_per_user(
    lake: Lake, holdout_rows: pd.DataFrame, embed_user: Any, embed_items: Any, k: int = 50
) -> pd.DataFrame:
    out = []
    for day, g in holdout_rows.groupby("day"):
        prev = _prev(dt.date.fromisoformat(str(day)))
        users = lake.read("users", [prev]).set_index("user_id")
        items = lake.read("items", [prev])
        items = items[~items["sold"]].reset_index(drop=True)
        by_market = {m: items[items["market"] == m] for m in items["market"].unique()}
        cache: dict[str, Any] = {}
        for uid, ug in g.groupby("user_id"):
            if uid not in users.index:
                continue
            market = users.loc[uid, "market"]
            cand = by_market.get(market)
            if cand is None or cand.empty:
                continue
            if market not in cache:
                cache[market] = embed_items(cand)
            scores = (
                cache[market]
                if embed_user is None
                else cache[market] @ embed_user(TF.user_vector(users.loc[uid].to_dict()))
            )
            top = set(cand["item_id"].to_numpy()[np.argsort(-scores)[:k]])
            truth = set(ug["item_id"])
            out.append((uid, market, len(top & truth) / len(truth)))
    return pd.DataFrame(out, columns=["session_id", "market", "value"])


def train(run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any]) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    ws = _ws(run_id)
    meta = store.get_json(f"{ws}/data.json")
    cfg = meta["training"]
    npz = np.load(io.BytesIO(store.get(f"{ws}/pairs.npz")))
    rows = pd.read_parquet(io.BytesIO(store.get(f"{ws}/pairs.parquet")))
    tr = (rows["split"] == "train").to_numpy()
    model = tower_model.train(npz["users"][tr], npz["items"][tr], cfg)
    lake = Lake(lake_root)
    # Recall on what members sought out in search: homepage interactions were shaped by the homepage model in
    # production, and would favour it.
    hold = rows[(rows["split"] == "holdout") & (rows["surface"] == "search")]
    cand = recall_per_user(
        lake, hold, lambda v: model.user.embed(v[None, :])[0], lambda f: model.item.embed(TF.item_matrix(f))
    )
    champ_version, champ = _champion_embedders(reg, store)
    if champ is None:
        base = recall_per_user(
            lake, hold, None, lambda f: ((f["clicks_7d"] + 1) / (f["impressions_7d"] + 20)).to_numpy(float)
        )
    else:
        u_onnx, i_onnx = champ
        base = recall_per_user(
            lake,
            hold,
            lambda v: onnx_export.run_onnx(u_onnx, v[None, :], "user")[0],
            lambda f: onnx_export.run_onnx(i_onnx, TF.item_matrix(f), "item"),
        )
    gcfg = config.load("gates")
    merged = base.merge(cand, on=["session_id", "market"], suffixes=("_champion", "_candidate"))
    overall = model_gates.paired_bootstrap(
        merged["value_champion"].to_numpy(),
        merged["value_candidate"].to_numpy(),
        gcfg["model"]["bootstrap_samples"],
        gcfg["model"]["confidence"],
    )
    markets = {
        m: model_gates.paired_bootstrap(
            g["value_champion"].to_numpy(),
            g["value_candidate"].to_numpy(),
            gcfg["model"]["bootstrap_samples"],
            gcfg["model"]["confidence"],
            seed=1,
        )
        for m, g in merged.groupby("market")
    }
    comparison = {
        "metric": "recall@50",
        "overall": overall,
        "markets": markets,
        "champion": champ_version,
        "loss": model.history,
    }
    if champ is None:
        # nothing comparable in production: record the popularity comparison, gate against random retrieval
        items = lake.read("items", [dt.date.fromisoformat(max(meta["holdout"])) - dt.timedelta(days=1)])
        per_market = items.groupby("market").size()
        random_recall = (
            float(np.mean([50 / max(per_market.get(m, 50), 50) for m in merged["market"]])) if len(merged) else 0.0
        )
        floor = gcfg["tower"]["random_recall_multiple_min"] * random_recall
        got = float(merged["value_candidate"].mean()) if len(merged) else 0.0
        results = [
            GateResult(
                "model.non_inferiority",
                got >= floor,
                f"recall@50 {got:.3f} is {got / max(random_recall, 1e-9):.1f}x random retrieval ({random_recall:.3f}); no tower in production to compare with",
                {**overall, "random_recall": random_recall, "floor": floor, "against": "random"},
            ),
            GateResult(
                "model.segment_non_inferiority",
                True,
                "no tower in production: the canary compares with the homepage as it is",
                {"markets": markets, "against": "popularity, informative"},
            ),
        ]
    else:
        results = model_gates.run(comparison, {}, kind="tower")
    exp = cfg.get("export", {})
    user_onnx = tower_model.to_onnx(
        model.user, "user", "embedding", int(exp.get("target_opset", 17)), exp.get("ir_version")
    )
    item_onnx = tower_model.to_onnx(
        model.item, "item", "embedding", int(exp.get("target_opset", 17)), exp.get("ir_version")
    )
    sample_u = npz["users"][~tr][:2000] if (~tr).any() else npz["users"][:2000]
    native = model.user.embed(sample_u)
    try:
        exported = onnx_export.run_onnx(user_onnx, sample_u, "user")
        worst = float(np.max(np.abs(exported - native)))
        par = GateResult(
            "export.onnx_parity",
            worst <= 1e-4,
            f"max |numpy - onnx| = {worst:.2e} over {len(sample_u)} members",
            {"max_abs": worst},
        )
    except Exception as e:
        par = GateResult("export.onnx_parity", False, f"onnxruntime could not run the export: {e}", {"error": str(e)})
    compat = onnx_export.compat(user_onnx, "triton", {"user": TF.user_dim()}, ["embedding"])
    results += [par, *compat]
    reg.record_gates(None, run_id, "model", [r for r in results if r.gate.startswith("model.")])
    reg.record_gates(None, run_id, "export", [r for r in results if r.gate.startswith("export.")])
    store.put(f"{ws}/user.onnx", user_onnx)
    store.put(f"{ws}/item.onnx", item_onnx)
    store.put_json(f"{ws}/metrics.json", comparison)
    return {"delta": overall["delta"]}


def register(run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any]) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    ws = _ws(run_id)
    meta = store.get_json(f"{ws}/data.json")
    blocked = [g["gate"] for g in reg.gates(run_id=run_id) if g["blocking"]]
    version = reg.next_version("tower")
    key = f"models/tower/{version}"
    arts = {}
    for name, src in (("user_onnx", "user.onnx"), ("item_onnx", "item.onnx")):
        _, sha = store.put(f"{key}/{src}", store.get(f"{ws}/{src}"))
        arts[name] = {"key": f"{key}/{src}", "sha256": sha}
    arts["onnx"] = arts["user_onnx"]
    metrics = store.get_json(f"{ws}/metrics.json")
    owners = config.load("standards")["owners"]["tower"]
    cfg = meta["training"]
    card = {
        "intended_use": "Homepage candidate retrieval: the user tower embeds a member on Triton; Vespa finds nearest items.",
        "training_window": meta["window"],
        "data_snapshot": meta["snapshot"],
        "metrics": metrics,
        "limits": [
            "Recall is measured on interactions with items the previous model showed.",
            "New listings are retrievable only after the next batch scoring.",
        ],
        "owner": owners,
        "provenance": {"user": "profile of the previous day", "item": "snapshot of the previous day"},
    }
    status = "rejected" if blocked else "candidate"
    reg.register(
        {
            "version": version,
            "kind": "tower",
            "created_at": clock.now(),
            "pipeline_run": run_id,
            "code_commit": code_commit(),
            "data_snapshot": meta["snapshot"],
            "training": cfg,
            "features": {"dim": int(cfg["dim"]), "user_dim": TF.user_dim(), "item_dim": TF.item_dim()},
            "artifacts": arts,
            "metrics": metrics,
            "dataset_profile": {"pairs": meta["pairs"]},
            "model_card": card,
            "owner": owners["team"],
            "status": status,
        }
    )
    reg.finish_run(run_id, "SUCCEEDED", "candidate" if status == "candidate" else "blocked", version)
    return {"version": version, "status": status, "blocked": blocked}


def score(run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Batch scoring for one tower version: embeddings for every item in the newest snapshot, fed to both Vespas."""
    from ..deploy import envs
    from ..serving.vespa.client import VespaError

    reg, store = Registry(), artifacts.Store()
    version = str(overrides["version"])
    v = reg.version(version)
    if v is None:
        raise RuntimeError(f"unknown tower {version}")
    a = v["artifacts"]["item_onnx"]
    item_onnx = store.get(a["key"], a["sha256"])
    lake = Lake(lake_root)
    parts = lake.partitions("items", landed_by=as_of)
    if not parts:
        raise NoData("no item snapshot")
    snap = lake.read("items", [parts[-1].date])
    snap = snap[~snap["sold"]].reset_index(drop=True)
    emb = onnx_export.run_onnx(item_onnx, TF.item_matrix(snap), "item")
    field = f"emb_{version}"
    fed = {"prod": 0, "staging": 0}
    errors: list[str] = []
    for env in ("staging", "prod"):
        client = envs.vespa(env)
        for item_id, vec in zip(snap["item_id"], emb, strict=True):
            try:
                client.update(
                    item_id,
                    {field: {"assign": {"values": [float(x) for x in vec]}}, "emb_versions": {"add": [version]}},
                )
                fed[env] += 1
            except VespaError as e:
                if e.status == 404:
                    continue  # sold or expired since the snapshot
                errors.append(str(e)[:200])
                if len(errors) > 20:
                    break
    ok = not errors
    per_market = snap.groupby("market").size().to_dict()
    result = {
        "version": version,
        "snapshot": parts[-1].date.isoformat(),
        "snapshot_as_of": int(snap["as_of"].max()) if len(snap) else 0,
        "fed": fed,
        "items": per_market,
        "errors": errors[:5],
    }
    store.put_json(f"{_ws(run_id)}/scoring.json", result)
    reg.finish_run(
        run_id,
        "SUCCEEDED" if ok else "FAILED",
        "embeddings_ready" if ok else "embeddings_failed",
        version,
        None if ok else errors[0],
    )
    return result


STAGES = {"data": data, "train": train, "register": register, "score": score}


def run_direct(run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = Registry()
    overrides = overrides or {}
    reg.start_run(run_id, "tower", "direct", {"as_of": as_of, "overrides": overrides})
    try:
        if data(run_id, as_of, lake_root, overrides) != "pass":
            return {"run_id": run_id, "state": "SUCCEEDED", "outcome": "blocked", "stage": "data"}
        train(run_id, as_of, lake_root, overrides)
        r = register(run_id, as_of, lake_root, overrides)
        return {
            "run_id": run_id,
            "state": "SUCCEEDED",
            "outcome": "candidate" if r["status"] == "candidate" else "blocked",
            **r,
        }
    except Exception as e:
        reg.finish_run(run_id, "FAILED", "error", error=f"{type(e).__name__}: {e}")
        return {"run_id": run_id, "state": "FAILED", "outcome": "error", "error": f"{type(e).__name__}: {e}"}


def score_items_direct(
    run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    reg = Registry()
    overrides = overrides or {}
    reg.start_run(run_id, "tower", "direct", {"as_of": as_of, "overrides": overrides})
    try:
        out = score(run_id, as_of, lake_root, overrides)
        row = reg.run(run_id) or {}
        return {"run_id": run_id, "state": row.get("state"), "outcome": row.get("outcome"), **out}
    except Exception as e:
        reg.finish_run(
            run_id, "FAILED", "embeddings_failed", str(overrides.get("version")), error=f"{type(e).__name__}: {e}"
        )
        return {
            "run_id": run_id,
            "state": "FAILED",
            "outcome": "embeddings_failed",
            "error": f"{type(e).__name__}: {e}",
        }
