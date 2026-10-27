"""Submitting and scheduling the compiled pipelines on Vertex AI Pipelines (google-cloud-aiplatform).

Nothing here runs without a GCP project; tests/test_vertex.py checks the PipelineJob and schedule this module
builds against a stand-in for the SDK."""

from __future__ import annotations

import json
from typing import Any

from .. import clock, config
from .ct import DEFINITIONS, compiled_dir
from .runner import PIPELINES


def _sdk() -> Any:
    from google.cloud import aiplatform  # imported lazily: the gcp extra

    return aiplatform


def job_spec(kind: str, overrides: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    v = config.load("pipeline")["vertex"]
    as_of = clock.now()
    run_id = run_id or f"{PIPELINES[kind]}-{as_of}"
    lake = config.load("pipeline")["lake"]
    return {
        "display_name": run_id,
        "template_path": str(compiled_dir() / f"{DEFINITIONS[kind].name}.yaml"),
        "pipeline_root": v["pipeline_root"],
        "job_id": run_id.replace("_", "-").lower(),
        "parameter_values": {
            "run_id": run_id,
            "as_of": as_of,
            "lake_root": lake,
            "overrides": json.dumps(overrides, sort_keys=True),
        },
        "enable_caching": bool(v["enable_caching"]),
        "labels": {**v["labels"], "pipeline": PIPELINES[kind]},
        "project": v["project"],
        "location": v["location"],
    }


def submit(kind: str, overrides: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    aip = _sdk()
    v = config.load("pipeline")["vertex"]
    spec = job_spec(kind, overrides, run_id)
    aip.init(project=spec.pop("project"), location=spec.pop("location"))
    job = aip.PipelineJob(**spec)
    job.submit(service_account=v["service_account"])
    return {"job": getattr(job, "resource_name", spec["job_id"]), "run_id": spec["parameter_values"]["run_id"]}


def schedule(kind: str) -> dict[str, Any]:
    aip = _sdk()
    v = config.load("pipeline")["vertex"]
    cron = config.load("pipeline")["schedules"][kind]
    spec = job_spec(kind, {}, run_id=f"{PIPELINES[kind]}-scheduled")
    aip.init(project=spec.pop("project"), location=spec.pop("location"))
    job = aip.PipelineJob(**spec)
    sched = job.create_schedule(
        display_name=f"{PIPELINES[kind]}-schedule",
        cron=cron,
        max_concurrent_run_count=1,
        service_account=v["service_account"],
    )
    return {"schedule": getattr(sched, "resource_name", "created"), "cron": cron}
