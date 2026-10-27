"""Deploying a homepage tower, in two phases, because the item side and the user side of a two-tower model must
change together and live in different systems.

Phase 1 (`ptp deploy run`, on the training pipeline's candidate event): deploy a Vespa package that adds the
version's embedding field and homepage profile next to the production one, then start the item-embeddings batch
scoring for that version. The deployment waits.

Phase 2 (`ptp deploy embeddings-ready`, on the scoring pipeline's event): check that the field covers the
catalogue the scoring saw; write the user tower into the Triton repository and load it on staging; check Triton
against the exported model and that staging retrieval returns results; then the same on production, and start the
homepage canary. Until this point the homepage never queried the new field.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from .. import artifacts, clock, config
from ..export.onnx_export import run_onnx
from ..gates.base import GateResult
from ..lake import Lake
from ..registry.store import Registry
from ..serving.triton.client import TritonError, model_name, version_number
from ..serving.vespa import package as P
from ..serving.vespa.client import VespaError, dense_tensor
from ..training import tower_features as TF
from . import envs
from .ranker import DeployError, current_spec, deploy_package, preflight


def _dim(v: dict[str, Any]) -> int:
    return int(v["features"]["dim"])


def package_with_embeddings(reg: Registry, env: str, candidate: dict[str, Any] | None) -> P.PackageSpec:
    keep = int(config.load("targets")["vespa"]["keep_profiles"])
    spec = current_spec(reg, env)
    route = reg.routing("home") or {}
    wanted: list[str] = []
    for version in (
        route.get("production"),
        candidate and candidate["version"],
        route.get("previous"),
        route.get("canary"),
    ):
        if version and version.startswith("t") and version not in wanted:
            wanted.append(version)
    have = {e.version: e for e in spec.embeddings}
    embeddings = []
    for version in wanted[:keep]:
        if version in have:
            embeddings.append(have[version])
        else:
            v = reg.version(version)
            if v:
                embeddings.append(P.EmbeddingField(version, _dim(v)))
    return P.PackageSpec(rankers=spec.rankers, embeddings=embeddings)


def run(version: str, trigger: str) -> dict[str, Any]:
    reg = Registry()
    dep_id, created = reg.open_deployment(version, trigger)
    if not created:
        d = reg.deployment(dep_id)
        return {"deployment": dep_id, "state": d["state"] if d else "unknown", "note": "already in progress"}
    today = dt.datetime.fromtimestamp(clock.now(), dt.UTC).date()
    route = reg.routing("home") or {}
    if route.get("canary") and route["canary"] != version:
        reg.step(dep_id, "deferred", True, f"{route['canary']} is in canary")
        reg.set_deployment(dep_id, "failed", error=f"deferred: {route['canary']} is in canary")
        return {"deployment": dep_id, "state": "deferred", "canary": route["canary"]}
    problems = preflight(reg, version, today)
    if problems:
        reg.step(dep_id, "preflight", False, "; ".join(problems))
        reg.set_deployment(dep_id, "failed", error=problems[0])
        return {"deployment": dep_id, "state": "failed", "error": problems[0]}
    v = reg.version(version)
    assert v
    reg.step(dep_id, "preflight", True, "candidate tower")
    reg.transition(version, "staging", "deploy", f"deployment {dep_id}")
    try:
        for env in ("staging", "prod"):
            deploy_package(reg, env, package_with_embeddings(reg, env, v), dep_id)
    except (VespaError, DeployError) as e:
        return _fail(reg, dep_id, version, "deploy_schema", str(e))
    reg.set_deployment(dep_id, "waiting_embeddings")
    reg.step(
        dep_id, "schema", True, f"field emb_{version} and profile homepage_{version} deployed; batch scoring requested"
    )
    # The scoring job is its own pipeline; its event brings this deployment to phase 2. On GCP it is submitted to
    # Vertex AI; without a project (here, CI, the seasons) it runs in-process with the same steps.
    now = clock.now()
    run_id = f"item-embeddings-{version}-{now}"
    if config.load("pipeline").get("scoring_engine", "direct") == "vertex":
        from ..pipelines import vertex

        out = vertex.submit("item-embeddings", {"version": version}, run_id=run_id)
        reg.step(dep_id, "scoring_requested", True, f"Vertex AI pipeline job {out.get('job')}")
    else:
        from ..pipelines import runner

        out = runner.run("item-embeddings", "direct", now, run_id, {"version": version})
        reg.step(
            dep_id,
            "scoring_requested",
            out.get("state") == "SUCCEEDED",
            f"item-embeddings run {run_id}: {out.get('outcome')}",
        )
    return {"deployment": dep_id, "state": "waiting_embeddings", "scoring_run": run_id}


def coverage(version: str, snapshot_as_of: int, env: str) -> GateResult:
    cfg = config.load("gates")["serving"]
    vespa = envs.vespa(env)
    base = ["sold = false", f"listed_at < {snapshot_as_of}"]
    total = vespa.count(base)
    have = vespa.count([*base, f'emb_versions contains "{version}"'])
    share = have / total if total else 0.0
    return GateResult(
        "serving.embedding_coverage",
        share >= cfg["embedding_coverage_min"],
        f"{share:.1%} of {total} live items listed before the snapshot carry emb_{version} in {env}",
        {"env": env, "share": share, "items": total, "with_embedding": have, "min": cfg["embedding_coverage_min"]},
    )


def _load_triton(reg: Registry, env: str, v: dict[str, Any], keep: list[str]) -> str:
    store = artifacts.Store()
    t = config.load("targets")["triton"]
    a = v["artifacts"]["user_onnx"]
    data = store.get(a["key"], a["sha256"])
    name = model_name(t["model_name"], _dim(v))
    client = envs.triton(env)
    client.write_version(name, version_number(v["version"]), data)
    versions = [
        version_number(x)
        for x in keep
        if x and x.startswith("t") and (reg.version(x) or {}).get("features", {}).get("dim") == _dim(v)
    ]
    client.write_config(
        name, TF.user_dim(), _dim(v), sorted(set([*versions, version_number(v["version"])])), int(t["max_batch_size"])
    )
    client.load(name)
    if not client.version_ready(name, version_number(v["version"])):
        raise TritonError(f"{name} version {version_number(v['version'])} did not become ready in {env}")
    return name


def embeddings_ready(version: str, trigger: str) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    deps = [d for d in reg.deployments(version) if d["state"] == "waiting_embeddings"]
    if not deps:
        return {"state": "ignored", "note": f"no deployment of {version} is waiting for embeddings"}
    dep_id = int(deps[-1]["id"])
    v = reg.version(version)
    assert v
    runs = reg.q(
        "SELECT run_id FROM pipeline_runs WHERE version=%s AND outcome='embeddings_ready' ORDER BY finished_at DESC LIMIT 1",
        version,
    )
    scoring = store.get_json(f"runs/{runs[0]['run_id']}/scoring.json") if runs else None
    if not scoring:
        return _fail(reg, dep_id, version, "coverage", "no successful scoring run")
    results = [coverage(version, int(scoring["snapshot_as_of"]), env) for env in ("staging", "prod")]
    route = reg.routing("home") or {"production": "popularity", "previous": None}
    keep = [x for x in (route.get("production"), route.get("previous")) if x]
    try:
        name = _load_triton(reg, "staging", v, keep)
        # Triton against the exported model, on member vectors from the newest profiles
        lake = Lake(config.resolve_path(config.load("pipeline")["lake"]))
        parts = lake.partitions("users")
        users = lake.read("users", [parts[-1].date]).head(64)
        x = np.stack([TF.user_vector(r) for r in users.to_dict("records")]).astype(np.float32)
        served = envs.triton("staging").infer(name, version_number(version), x)
        offline = run_onnx(store.get(v["artifacts"]["user_onnx"]["key"]), x, "user")
        worst = float(np.max(np.abs(served - offline)))
        results.append(
            GateResult(
                "serving.score_parity",
                worst <= config.load("gates")["serving"]["score_parity_max_abs"],
                f"max |Triton - offline| = {worst:.2e} over {len(x)} members",
                {"max_abs": worst},
            )
        )
        # staging retrieval with the served embeddings returns items
        empty = 0
        for row, emb in zip(users.head(20).to_dict("records"), served[:20], strict=True):
            res = envs.vespa("staging").search(
                [f'market contains "{row["market"]}"', "sold = false"],
                ranking=f"homepage_{version}",
                hits=20,
                inputs={"user_emb": dense_tensor(emb.tolist())},
                nearest=(f"emb_{version}", "user_emb", 100),
            )
            empty += int(len(res.hits) == 0)
        results.append(
            GateResult(
                "serving.latency", empty == 0, f"{20 - empty} of 20 staging retrievals returned items", {"empty": empty}
            )
        )
    except (TritonError, VespaError) as e:
        results.append(GateResult("serving.score_parity", False, f"staging: {e}", {"error": str(e)}))
    reg.record_gates(version, None, "serving", results)
    blocking = [r.gate for r in results if r.blocking]
    reg.step(dep_id, "serving_gates", not blocking, "; ".join(r.summary for r in results))
    if blocking:
        return _fail(
            reg, dep_id, version, "serving_gates", f"blocked by {', '.join(sorted(set(blocking)))}", recorded=True
        )
    try:
        _load_triton(reg, "prod", v, keep)
    except TritonError as e:
        return _fail(reg, dep_id, version, "triton_prod", str(e))
    share = float(config.load("gates")["canary"]["traffic_share"])
    reg.set_routing("home", route["production"], version, share, f"canary {version} (deployment {dep_id})")
    reg.transition(version, "canary", "deploy", f"{share:.0%} of homepage traffic")
    reg.step(dep_id, "canary", True, f"{share:.0%} of homepage traffic on {version}")
    reg.set_deployment(dep_id, "canary")
    return {"deployment": dep_id, "state": "canary", "version": version}


def _fail(reg: Registry, dep_id: int, version: str, step: str, error: str, recorded: bool = False) -> dict[str, Any]:
    if not recorded:
        reg.step(dep_id, step, False, error)
    reg.set_deployment(dep_id, "failed", error=error)
    reg.transition(version, "failed", "deploy", f"{step}: {error}"[:500])
    return {"deployment": dep_id, "state": "failed", "error": error}
