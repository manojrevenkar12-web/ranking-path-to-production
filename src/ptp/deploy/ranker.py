"""Deploying a ranker: what the GitHub Action runs when a training pipeline reports a candidate.

1. Preflight: the version is a candidate (or every blocking gate it failed has an unexpired waiver), and its
   artifact still has the bytes the pipeline registered.
2. Build the Vespa package: the production profile, the candidate and one previous profile for rollback.
3. Deploy to staging and wait for convergence; replay logged requests; score parity, skew and latency gates.
4. Deploy the same package to production. Routing still sends every query to the production profile.
5. Start the canary: a share of search traffic goes to the candidate profile. The canary check (canary.py, on a
   schedule) promotes or rolls back.

Every step is recorded on the deployment; a repeated dispatch for the same version finds the open deployment and
does nothing. A failure leaves production untouched: the package keeps the production profile, and routing has not
moved.
"""

from __future__ import annotations

import datetime as dt
import io
from typing import Any

import pandas as pd

from .. import artifacts, clock, config, contract
from ..gates import base as gates_base
from ..gates import serving as serving_gates
from ..lake import Lake
from ..registry.store import Registry
from ..serving.vespa import package as P
from ..serving.vespa.client import VespaError
from . import envs


class DeployError(RuntimeError):
    pass


def _ranker_profile(reg: Registry, store: artifacts.Store, version: str) -> P.RankerProfile:
    v = reg.version(version)
    if v is None:
        raise DeployError(f"unknown version {version}")
    art = v["artifacts"]["onnx"]
    return P.RankerProfile(version=version, onnx=store.get(art["key"], art["sha256"]), features=list(v["features"]))


def current_spec(reg: Registry, env: str) -> P.PackageSpec:
    active = reg.active_package(env)
    if active is None:
        return P.PackageSpec()
    return P.PackageSpec.from_files(P.from_zip(bytes(active["zip"])))


def package_for(reg: Registry, store: artifacts.Store, env: str, candidate: str | None) -> P.PackageSpec:
    """Keep production, canary candidate and the previous production profile; drop the rest."""
    keep = int(config.load("targets")["vespa"]["keep_profiles"])
    spec = current_spec(reg, env)
    route = reg.routing("search") or {}
    wanted: list[str] = []
    for profile in (route.get("production"), candidate and f"ranker_{candidate}", route.get("previous")):
        if profile and profile.startswith("ranker_") and profile not in wanted:
            wanted.append(profile)
    wanted = wanted[:keep]
    have = {r.name: r for r in spec.rankers}
    rankers = []
    for name in wanted:
        version = name.removeprefix("ranker_")
        rankers.append(have.get(name) or _ranker_profile(reg, store, version))
    return P.PackageSpec(rankers=rankers, embeddings=spec.embeddings)


def deploy_package(reg: Registry, env: str, spec: P.PackageSpec, dep_id: int | None = None) -> dict[str, Any]:
    files = P.build(contract.load(), spec)
    z = P.to_zip(files)
    v = envs.vespa(env)
    res = v.deploy(z)
    if res.get("needs_action"):
        raise DeployError(f"{env}: the package needs {res['configChangeActions']}; refusing an automatic deploy")
    v.wait_converged()
    generation = int(res.get("session-id", 0))
    reg.save_package(env, generation, P.digest(files), spec.profiles(), z)
    if dep_id is not None:
        reg.step(
            dep_id,
            f"deploy_{env}",
            True,
            f"{env} generation {generation}: {', '.join(spec.profiles())}",
            {"generation": generation},
        )
    return {"generation": generation, "sha": P.digest(files)}


def preflight(reg: Registry, version: str, today: dt.date) -> list[str]:
    v = reg.version(version)
    if v is None:
        return [f"unknown version {version}"]
    if v["status"] == "candidate":
        return []
    if v["status"] != "rejected":
        return [f"{version} is {v['status']}, not a candidate"]
    rows = reg.gates(version)
    results = [
        gates_base.GateResult(r["gate"], r["passed"], r["summary"], r["evidence"], r["severity"])
        for r in rows
        if r["blocking"]
    ]
    gates_base.apply_waivers(version, results, today)
    open_gates = [r.gate for r in results if r.blocking]
    if open_gates:
        return [f"{version} was rejected by {', '.join(sorted(set(open_gates)))} and has no waiver for it"]
    reg.transition(version, "candidate", "waiver", "every blocking gate is waived")
    return []


def run(version: str, trigger: str) -> dict[str, Any]:
    reg, store = Registry(), artifacts.Store()
    dep_id, created = reg.open_deployment(version, trigger)
    if not created:
        d = reg.deployment(dep_id)
        return {"deployment": dep_id, "state": d["state"] if d else "unknown", "note": "already in progress"}
    now = clock.now()
    today = dt.datetime.fromtimestamp(now, dt.UTC).date()
    route = reg.routing("search") or {}
    if route.get("canary") and route["canary"] != f"ranker_{version}":
        # One canary at a time: the running one finishes; tomorrow's pipeline brings a fresher candidate than this.
        reg.step(dep_id, "deferred", True, f"{route['canary']} is in canary")
        reg.set_deployment(dep_id, "failed", error=f"deferred: {route['canary']} is in canary")
        return {"deployment": dep_id, "state": "deferred", "canary": route["canary"]}
    problems = preflight(reg, version, today)
    if problems:
        reg.step(dep_id, "preflight", False, "; ".join(problems))
        reg.set_deployment(dep_id, "failed", error=problems[0])
        return {"deployment": dep_id, "state": "failed", "error": problems[0]}
    try:
        cand = _ranker_profile(reg, store, version)
    except artifacts.IntegrityError as e:
        reg.step(dep_id, "preflight", False, str(e))
        reg.set_deployment(dep_id, "failed", error=str(e))
        return {"deployment": dep_id, "state": "failed", "error": str(e)}
    reg.step(dep_id, "preflight", True, "candidate with verified artifact")
    reg.transition(version, "staging", "deploy", f"deployment {dep_id}")

    try:
        spec = package_for(reg, store, "staging", version)
        deploy_package(reg, "staging", spec, dep_id)
    except (VespaError, DeployError) as e:
        return _fail(reg, dep_id, version, "deploy_staging", str(e))

    # serving gates on staging
    v = reg.version(version)
    assert v is not None
    sample = pd.read_parquet(
        io.BytesIO(store.get(v["artifacts"]["train_sample"]["key"], v["artifacts"]["train_sample"]["sha256"]))
    )
    lake = Lake(config.resolve_path(config.load("pipeline")["lake"]))
    parts = lake.partitions("impressions", landed_by=now)
    if not parts:
        return _fail(reg, dep_id, version, "serving_gates", "no impression logs to replay")
    impressions = lake.read("impressions", [parts[-1].date])
    requests = serving_gates.replay_requests(impressions, int(config.load("gates")["serving"]["sample_queries"]))
    provenance = (v["model_card"] or {}).get("provenance", {})
    results, evidence = serving_gates.run(
        envs.vespa("staging"), cand.name, cand.onnx, cand.features, provenance, sample, requests, now, contract.load()
    )
    reg.record_gates(version, None, "serving", results)
    blocking = [r.gate for r in results if r.blocking]
    reg.step(dep_id, "serving_gates", not blocking, "; ".join(r.summary for r in results), evidence)
    if blocking:
        return _fail(reg, dep_id, version, "serving_gates", f"blocked by {', '.join(blocking)}", recorded=True)

    try:
        prod = deploy_package(reg, "prod", package_for(reg, store, "prod", version), dep_id)
    except (VespaError, DeployError) as e:
        return _fail(reg, dep_id, version, "deploy_prod", str(e))

    route = reg.routing("search") or {"production": "popularity", "previous": None}
    share = float(config.load("gates")["canary"]["traffic_share"])
    reg.set_routing("search", route["production"], cand.name, share, f"canary {version} (deployment {dep_id})")
    reg.transition(version, "canary", "deploy", f"{share:.0%} of search traffic")
    reg.step(dep_id, "canary", True, f"{share:.0%} of search traffic on {cand.name}")
    reg.set_deployment(dep_id, "canary", package_sha=prod["sha"])
    return {"deployment": dep_id, "state": "canary", "version": version}


def _fail(reg: Registry, dep_id: int, version: str, step: str, error: str, recorded: bool = False) -> dict[str, Any]:
    if not recorded:
        reg.step(dep_id, step, False, error)
    reg.set_deployment(dep_id, "failed", error=error)
    reg.transition(version, "failed", "deploy", f"{step}: {error}"[:500])
    return {"deployment": dep_id, "state": "failed", "error": error}
