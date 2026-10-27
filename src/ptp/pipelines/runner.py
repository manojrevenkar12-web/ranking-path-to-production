"""Runs a pipeline with an engine, and publishes its final state. The KFP definition publishes from its exit
handler; the in-process engine calls the same function when it finishes."""

from __future__ import annotations

import json
from typing import Any

from .. import config, events
from ..registry.store import Registry


def outcome_event(run_id: str, kind: str, pipeline: str, failed: bool = False) -> events.PipelineEvent:
    reg = Registry()
    run = reg.run(run_id) or {}
    version = run.get("version")
    artifact = None
    state = "FAILED" if failed or run.get("state") == "FAILED" else "SUCCEEDED"
    outcome = run.get("outcome") or ("error" if state == "FAILED" else "blocked")
    if version:
        v = reg.version(version)
        if v and v["kind"] == kind:
            artifact = v["artifacts"].get("onnx")
    blocked = [g["gate"] for g in reg.gates(run_id=run_id) if g["blocking"]]
    return events.PipelineEvent(
        run_id=run_id,
        pipeline=pipeline,
        kind=kind,
        state=state,
        outcome=outcome,
        version=version,
        artifact=artifact,
        detail={"blocked": blocked, "error": run.get("error")},
    )


def finish_and_publish(run_id: str, kind: str, pipeline: str, failed: bool, publish: bool) -> dict[str, Any]:
    ev = outcome_event(run_id, kind, pipeline, failed)
    out = ev.model_dump()
    if publish:
        out["message_id"] = events.publish(ev)
    return out


PIPELINES = {"ranker": "ct-ranker", "tower": "ct-tower", "item-embeddings": "item-embeddings"}


def run(
    kind: str, engine: str, as_of: int, run_id: str | None, overrides: dict[str, Any], publish: bool = True
) -> dict[str, Any]:
    run_id = run_id or f"{PIPELINES[kind]}-{as_of}"
    lake_root = str(config.resolve_path(config.load("pipeline")["lake"]))
    if engine == "kfp":
        from . import ct

        ct.run_local(kind, run_id, as_of, lake_root, overrides)
        reg = Registry()
        row = reg.run(run_id) or {}
        return {
            "run_id": run_id,
            "state": row.get("state"),
            "outcome": row.get("outcome"),
            "version": row.get("version"),
            "engine": "kfp",
        }
    if kind == "ranker":
        from . import steps

        out = steps.run_direct(run_id, as_of, lake_root, overrides)
    elif kind == "tower":
        from . import tower_steps

        out = tower_steps.run_direct(run_id, as_of, lake_root, overrides)
    else:
        from . import tower_steps

        out = tower_steps.score_items_direct(run_id, as_of, lake_root, overrides)
    ev = finish_and_publish(
        run_id, "ranker" if kind == "ranker" else "tower", PIPELINES[kind], out.get("state") == "FAILED", publish
    )
    return {**out, "event": ev}


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
