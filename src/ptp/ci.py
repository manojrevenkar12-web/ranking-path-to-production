"""Helpers for the deploy loop that runs for real on GitHub (ci.yml job `loop`, deploy-model.yml job `deploy-ci`).

The loop job runs a short simulated season, registers one more candidate and bundles the registry, the artifacts,
the newest logs and the catalogue. The relay dispatches to GitHub; the deploy workflow restores the bundle next
to two real Vespa containers (staging and production), feeds them the catalogue, deploys the production package,
and then runs the same `ptp deploy run` it runs in production.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from typing import Any

import httpx
from psycopg.types.json import Jsonb

from . import artifacts, config
from .deploy import envs
from .registry.store import TABLES, Registry
from .serving.vespa.client import VespaError

JSON_COLUMNS = {
    "pipeline_runs": ["params"],
    "model_versions": ["training", "features", "artifacts", "metrics", "dataset_profile", "model_card"],
    "gate_results": ["evidence"],
    "deployments": ["steps"],
    "packages": ["profiles"],
    "relay_messages": ["event"],
    "monitor_runs": ["result"],
}


def bundle(out: Path, docs: Path | None) -> dict[str, Any]:
    reg = Registry()
    out.mkdir(parents=True, exist_ok=True)
    dump: dict[str, list[dict[str, Any]]] = {}
    for table in reversed(TABLES):
        rows = reg.q(f"SELECT * FROM {table}")
        for r in rows:
            for k, v in list(r.items()):
                if isinstance(v, (bytes, memoryview)):
                    r[k] = {"__bytes__": base64.b64encode(bytes(v)).decode()}
        dump[table] = rows
    (out / "registry.json").write_text(json.dumps(dump, default=str))
    store = artifacts.Store()
    if (out / "artifacts").exists():
        shutil.rmtree(out / "artifacts")
    shutil.copytree(store.root / "models", out / "artifacts" / "models")
    lake = config.resolve_path(config.load("pipeline")["lake"])
    for table in ("impressions", "users", "items"):
        parts = sorted((lake / table).glob("date=*"))[-2:]
        for p in parts:
            shutil.copytree(p, out / "lake" / table / p.name, dirs_exist_ok=True)
    if docs:
        shutil.copy(docs, out / "vespa_docs.jsonl")
    from . import clock

    (out / "meta.json").write_text(json.dumps({"now": clock.now()}))
    return {"tables": {k: len(v) for k, v in dump.items()}, "out": str(out)}


def restore(src: Path) -> dict[str, Any]:
    reg = Registry()
    reg.reset()
    dump = json.loads((src / "registry.json").read_text())
    for table in reversed(TABLES):
        rows = dump.get(table, [])
        if not rows:
            continue
        cols = list(rows[0].keys())
        for r in rows:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, dict) and "__bytes__" in v:
                    v = base64.b64decode(v["__bytes__"])
                elif c in JSON_COLUMNS.get(table, []):
                    v = Jsonb(v)
                vals.append(v)
            reg.conn().execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})", vals
            )
        if "id" in cols:
            reg.conn().execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), (SELECT MAX(id) FROM {table}))")
    store = artifacts.Store()
    shutil.copytree(src / "artifacts", store.root, dirs_exist_ok=True)
    lake = config.resolve_path(config.load("pipeline")["lake"])
    shutil.copytree(src / "lake", lake, dirs_exist_ok=True)
    return {"tables": {k: len(v) for k, v in dump.items()}}


def _feed_fields(doc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in doc.items():
        if k == "brand_t" and isinstance(v, dict) and "cells" not in v:
            out[k] = {"cells": [{"address": {"brand": b}, "value": float(x)} for b, x in v.items()]}
        elif k.startswith("emb_") and k != "emb_versions":
            out[k] = {"values": list(v)}
        else:
            out[k] = v
    return out


def seed_vespa(docs_path: Path) -> dict[str, Any]:
    """Deploy each environment's active package from the registry and feed it the catalogue."""
    reg = Registry()
    fed: dict[str, int] = {}
    for env in ("staging", "prod"):
        pkg = reg.active_package(env)
        if pkg is None:
            raise RuntimeError(f"no {env} package in the registry")
        v = envs.vespa(env)
        v.wait_ready_config()
        v.deploy(bytes(pkg["zip"]))
        v.wait_converged()
        v.wait_ready()
        n = 0
        with open(docs_path) as f:
            for line in f:
                doc = json.loads(line)
                try:
                    v.put(doc["item_id"], _feed_fields(doc))
                    n += 1
                except (VespaError, httpx.HTTPError) as e:
                    raise RuntimeError(f"feeding {doc['item_id']} to {env}: {e}") from e
        fed[env] = n
    return {"fed": fed}
