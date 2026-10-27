"""Adopting a model that was deployed by hand.

The ad-hoc process left a model in production that the registry knows nothing about. Adoption registers it (with
lineage marked unknown), profiles the data it most plausibly saw so the next pipeline run has a baseline, and
replaces the hand-built Vespa package with a platform-built one carrying the same model. From then on the
pipeline's candidates are compared with it and can replace it. The standards check keeps flagging it until then.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import artifacts, clock, config
from .data import build, propensity
from .gates import data as data_gates
from .lake import Lake
from .registry.store import Registry
from .serving.vespa import package as P


def adopt(onnx_path: str, features: list[str], note: str, window_days: int) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    now = clock.now()
    data = Path(onnx_path).read_bytes()
    version = reg.next_version("ranker")
    run_id = f"adopt-{version}"
    reg.start_run(run_id, "ranker", "adopted", {"note": note})
    reg.finish_run(run_id, "SUCCEEDED", "adopted", version)
    key = f"models/ranker/{version}"
    _, sha = store.put(f"{key}/model.onnx", data)
    lake = Lake(config.resolve_path(config.load("pipeline")["lake"]))
    win = build.window(lake, now, window_days, 1)
    raw = build.read_raw(lake, win)
    profile = data_gates.profile(raw, features)
    profile["_propensity"] = propensity.estimate(
        raw, min_sessions=config.load("gates")["model"]["propensity_min_sessions"]
    )
    sample = raw[[f"f_{f}" for f in features] + ["market"]].sample(min(len(raw), 5000), random_state=0)
    import io

    buf = io.BytesIO()
    sample.to_parquet(buf, index=False)
    _, s_sha = store.put(f"{key}/train_sample.parquet", buf.getvalue())
    owners = config.load("standards")["owners"]["ranker"]
    reg.register(
        {
            "version": version,
            "kind": "ranker",
            "created_at": now,
            "pipeline_run": run_id,
            "code_commit": "unknown (ad hoc notebook)",
            "data_snapshot": "unknown",
            "training": {"note": note},
            "features": features,
            "artifacts": {
                "onnx": {"key": f"{key}/model.onnx", "sha256": sha},
                "train_sample": {"key": f"{key}/train_sample.parquet", "sha256": s_sha},
            },
            "metrics": {},
            "dataset_profile": profile,
            "model_card": {"note": note, "provenance": {f: "logged" for f in features}},
            "owner": owners["team"],
            "status": "production",
        }
    )
    from .deploy.ranker import deploy_package

    spec = P.PackageSpec(rankers=[P.RankerProfile(version, data, features)])
    for env in ("staging", "prod"):
        deploy_package(reg, env, spec)
    reg.set_routing("search", f"ranker_{version}", None, 0.0, f"adopted hand-deployed model as {version}")
    return {"version": version, "sha256": sha, "routing": f"ranker_{version}"}
