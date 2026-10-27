"""The continuous-training steps for the search ranker. Each step is a plain function over a run workspace in the
artifact store (`runs/<run_id>/`); the KFP components in ct.py are thin wrappers that call them, so Vertex AI, the
KFP local runner and the in-process engine run exactly the same code."""

from __future__ import annotations

import copy
import datetime as dt
import io
import json
import os
import subprocess
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import artifacts, clock, config, contract
from ..data import build, propensity
from ..export import onnx_export
from ..gates import base as gates_base
from ..gates import data as data_gates
from ..gates import model as model_gates
from ..lake import Lake
from ..registry.store import Registry


def code_commit() -> str:
    env = os.environ.get("PTP_CODE_COMMIT")
    if env:
        return env
    try:
        return subprocess.run(
            ["git", "-C", str(config.root()), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def training_config(kind: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(config.load(f"training/{kind}"))
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def _ws(run_id: str) -> str:
    return f"runs/{run_id}"


def _put_frame(store: artifacts.Store, key: str, frame: pd.DataFrame) -> None:
    buf = io.BytesIO()
    frame = frame.copy()
    frame.attrs = {}
    frame.to_parquet(buf, index=False)
    store.put(key, buf.getvalue())


def _get_frame(store: artifacts.Store, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.get(key)))


def champion(reg: Registry, kind: str = "ranker") -> dict[str, Any] | None:
    return reg.production(kind)


def champion_trained_until(champ: dict[str, Any] | None) -> dt.date | None:
    """Last day of data the production model may have trained on: its holdout must not overlap the candidate's."""
    if champ is None:
        return None
    window = (champ.get("model_card") or {}).get("training_window") or {}
    days = [*window.get("train", []), *window.get("holdout", [])] if isinstance(window, dict) else list(window)
    if days:
        return max(dt.date.fromisoformat(d) for d in days)
    return dt.datetime.fromtimestamp(int(champ["created_at"]), dt.UTC).date() - dt.timedelta(days=1)


def champion_scorer(reg: Registry, store: artifacts.Store) -> tuple[str, list[str], model_gates.Scorer]:
    champ = champion(reg)
    if champ is None:
        return "popularity", ["item_ctr"], lambda f: f["f_item_ctr"].to_numpy(dtype=float)
    art = champ["artifacts"]["onnx"]
    data = store.get(art["key"], art["sha256"])
    feats = list(champ["features"])
    return champ["version"], feats, model_gates.frame_scorer(feats, lambda x: onnx_export.run_onnx(data, x)[:, 0])


# ---- step 1: data -----------------------------------------------------------------------------------------------


def validate_data(run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any]) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    c = contract.load()
    training = training_config("ranker", overrides)
    gcfg = config.load("gates")["data"]
    lake = Lake(lake_root)
    try:
        win = build.window(lake, as_of, gcfg["window_days"], gcfg["holdout_days"])
    except build.NoData as e:
        res = gates_base.GateResult("data.freshness", False, str(e), {})
        reg.record_gates(None, run_id, "data", [res])
        return {"passed": False, "reason": str(e)}
    champ_version, champ_features, _ = champion_scorer(reg, store)
    wanted = dict(training)
    wanted["features"] = list(dict.fromkeys([*training["features"], *champ_features]))
    raw = build.read_raw(lake, win, quarantine=reg.quarantined("impressions"))
    ts = build.build(lake, win, raw, c, wanted)
    ts.features = list(training["features"])
    ts.provenance = {f: ts.provenance[f] for f in ts.features}
    champ = champion(reg)
    baseline = champ["dataset_profile"] if champ else None
    results, profile = data_gates.run(raw, ts, win, as_of, c, baseline)
    mcfg = config.load("gates")["model"]
    prop = propensity.estimate(raw, min_sessions=mcfg["propensity_min_sessions"])
    if prop["source"] == "none" and champ and champ["dataset_profile"].get("_propensity", {}).get("per_position"):
        prop = {**champ["dataset_profile"]["_propensity"], "source": f"champion {champ['version']}"}
    results.append(
        gates_base.GateResult(
            "data.propensity",
            prop["source"] == "explore",
            f"position bias eta {prop['eta']} from {prop['sessions']} exploration sessions"
            if prop["source"] == "explore"
            else f"only {prop['sessions']} exploration sessions; using {prop['source']}",
            prop,
        )
    )
    profile["_propensity"] = prop
    reg.record_gates(None, run_id, "data", results)
    ws = _ws(run_id)
    _put_frame(store, f"{ws}/training_set.parquet", ts.frame)
    store.put_json(
        f"{ws}/data.json",
        {
            "window": {"train": [d.isoformat() for d in win.train], "holdout": [d.isoformat() for d in win.holdout]},
            "snapshot": ts.snapshot,
            "provenance": ts.provenance,
            "features": ts.features,
            "profile": profile,
            "propensity": prop,
            "champion": champ_version,
            "training": training,
            "rows": len(ts.frame),
            "duplicates": ts.duplicates,
            "quarantined": int(raw.attrs.get("quarantined", 0)),
        },
    )
    return {"passed": gates_base.decision(results) == "pass", "snapshot": ts.snapshot, "rows": len(ts.frame)}


def clock_day(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(ts, dt.UTC).date()


# ---- step 2: train ----------------------------------------------------------------------------------------------


def train(run_id: str) -> dict[str, Any]:
    from ..training import ranker

    store = artifacts.Store()
    ws = _ws(run_id)
    meta = store.get_json(f"{ws}/data.json")
    frame = _get_frame(store, f"{ws}/training_set.parquet")
    training = meta["training"]
    trained = ranker.train(
        frame[frame["split"] == "train"],
        meta["features"],
        training["params"],
        training.get("market_weights"),
        bool(training.get("position_debias", False)),
    )
    store.put(f"{ws}/model.txt", trained.booster.model_to_string().encode())
    store.put_json(f"{ws}/importance.json", trained.importance)
    return {"features": meta["features"], "importance": trained.importance}


def _booster(store: artifacts.Store, ws: str) -> lgb.Booster:
    return lgb.Booster(model_str=store.get(f"{ws}/model.txt").decode())


# ---- step 3: evaluate -------------------------------------------------------------------------------------------


def evaluate(run_id: str) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    ws = _ws(run_id)
    meta = store.get_json(f"{ws}/data.json")
    frame = _get_frame(store, f"{ws}/training_set.parquet")
    holdout = frame[frame["split"] == "holdout"]
    booster = _booster(store, ws)
    feats = meta["features"]
    cand = model_gates.frame_scorer(feats, lambda x: np.asarray(booster.predict(x), dtype=float))
    champ_version, _, champ = champion_scorer(reg, store)
    # A production model looks better than it is on days it trained on: compare only on days after its data.
    until = champion_trained_until(champion(reg))
    days = pd.to_datetime(holdout["ts"], unit="s", utc=True).dt.date
    clean = holdout[days > until] if until else holdout
    used = sorted({d.isoformat() for d in pd.to_datetime(clean["ts"], unit="s", utc=True).dt.date})
    if len(clean) == 0:
        clean, used = holdout, sorted({d.isoformat() for d in days})
    comparison = model_gates.compare(clean, cand, champ, propensity=meta.get("propensity"))
    comparison["champion"] = champ_version
    comparison["holdout_days"] = used
    comparison["holdout_overlaps_champion"] = (
        len(clean) == len(holdout) and until is not None and any(d <= until for d in days)
    )
    importance = store.get_json(f"{ws}/importance.json")
    results = model_gates.run(comparison, importance)
    reg.record_gates(None, run_id, "model", results)
    store.put_json(f"{ws}/metrics.json", comparison)
    return {"passed": gates_base.decision(results) == "pass", "delta": comparison["overall"]["delta"]}


# ---- step 4: export ---------------------------------------------------------------------------------------------


def export(run_id: str) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    ws = _ws(run_id)
    meta = store.get_json(f"{ws}/data.json")
    frame = _get_frame(store, f"{ws}/training_set.parquet")
    holdout = frame[frame["split"] == "holdout"]
    booster = _booster(store, ws)
    feats = meta["features"]
    data = onnx_export.ranker_to_onnx(booster, len(feats), meta["training"].get("export", {}))
    x = build.matrix(holdout, feats)
    results = [onnx_export.parity(data, x, np.asarray(booster.predict(x), dtype=float))]
    results += onnx_export.compat(data, "vespa", {"features": len(feats)}, ["variable"])
    reg.record_gates(None, run_id, "export", results)
    store.put(f"{ws}/model.onnx", data)
    return {"passed": gates_base.decision(results) == "pass", "bytes": len(data)}


# ---- step 5: register -------------------------------------------------------------------------------------------


def register(run_id: str) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    ws = _ws(run_id)
    meta = store.get_json(f"{ws}/data.json")
    pipeline_gates = reg.gates(run_id=run_id)
    blocked = [g["gate"] for g in pipeline_gates if g["blocking"]]
    version = reg.next_version("ranker")
    key = f"models/ranker/{version}"
    arts: dict[str, dict[str, str]] = {}
    for name, src in (("onnx", "model.onnx"), ("native", "model.txt")):
        if store.exists(f"{ws}/{src}"):
            data = store.get(f"{ws}/{src}")
            _, sha = store.put(f"{key}/{src}", data)
            arts[name] = {"key": f"{key}/{src}", "sha256": sha}
    frame = _get_frame(store, f"{ws}/training_set.parquet")
    sample = frame[frame["split"] == "train"]
    sample = sample.sample(min(len(sample), 5000), random_state=0)[[f"f_{f}" for f in meta["features"]] + ["market"]]
    buf = io.BytesIO()
    sample.to_parquet(buf, index=False)
    _, sha = store.put(f"{key}/train_sample.parquet", buf.getvalue())
    arts["train_sample"] = {"key": f"{key}/train_sample.parquet", "sha256": sha}
    metrics = store.get_json(f"{ws}/metrics.json") if store.exists(f"{ws}/metrics.json") else {}
    owners = config.load("standards")["owners"]["ranker"]
    status = "rejected" if blocked else "candidate"
    card = {
        "intended_use": "Second-phase ranking of search results for a category query, all markets.",
        "training_window": meta["window"],
        "data_snapshot": meta["snapshot"],
        "features": meta["features"],
        "provenance": meta["provenance"],
        "champion_at_training": meta["champion"],
        "metrics": metrics,
        "limits": [
            "Evaluated offline on logged impressions only: it re-ranks what the previous model showed.",
            "Labels are position-biased clicks, favourites and purchases.",
        ],
        "owner": owners,
    }
    _, card_sha = store.put_json(f"{key}/model_card.json", card)
    arts["model_card"] = {"key": f"{key}/model_card.json", "sha256": card_sha}
    reg.register(
        {
            "version": version,
            "kind": "ranker",
            "created_at": clock.now(),
            "pipeline_run": run_id,
            "code_commit": code_commit(),
            "data_snapshot": meta["snapshot"],
            "training": meta["training"],
            "features": meta["features"],
            "artifacts": arts,
            "metrics": metrics,
            "dataset_profile": meta["profile"],
            "model_card": card,
            "owner": owners["team"],
            "status": status,
        }
    )
    return {"version": version, "status": status, "blocked": blocked, "artifact": arts.get("onnx")}


def run_direct(run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The pipeline in-process: same steps, same order, same conditions as the KFP definition."""
    reg = Registry()
    reg.start_run(run_id, "ranker", "direct", {"as_of": as_of, "overrides": overrides or {}})
    out: dict[str, Any] = {"run_id": run_id, "kind": "ranker"}
    try:
        d = validate_data(run_id, as_of, lake_root, overrides or {})
        if not d["passed"]:
            reg.finish_run(run_id, "SUCCEEDED", "blocked")
            out.update(state="SUCCEEDED", outcome="blocked", stage="data")
            return out
        train(run_id)
        evaluate(run_id)
        export(run_id)
        r = register(run_id)
        outcome = "candidate" if r["status"] == "candidate" else "blocked"
        reg.finish_run(run_id, "SUCCEEDED", outcome, r["version"])
        out.update(
            state="SUCCEEDED", outcome=outcome, version=r["version"], artifact=r["artifact"], blocked=r["blocked"]
        )
        return out
    except Exception as e:
        reg.finish_run(run_id, "FAILED", "error", error=f"{type(e).__name__}: {e}")
        out.update(state="FAILED", outcome="error", error=f"{type(e).__name__}: {e}")
        return out


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
