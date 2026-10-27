"""The console's API: the registry, the pipeline runs, deployments, routing, standards and the season reports, read
from PostgreSQL. One write: the on-call rollback, behind a token."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, standards
from .registry.store import Registry

SUMMARY_FIELDS = ("version", "kind", "status", "created_at", "pipeline_run", "owner", "code_commit", "data_snapshot")


STATIONS = ("data", "model", "export", "staging", "canary", "production")


def path_of(reg: Registry, v: dict[str, Any]) -> list[dict[str, Any]]:
    """Where a version got to on the path, station by station: passed, stopped, active, waiting, pending, skipped."""
    gates = reg.gates(v["version"])
    statuses = [t["to_status"] for t in reg.transitions(v["version"])]
    deps = reg.deployments(v["version"])
    out: list[dict[str, Any]] = []
    stopped = False
    for st in STATIONS[:3]:
        mine = [g for g in gates if g["stage"] == st]
        blocking = sorted({g["gate"] for g in mine if g["blocking"]})
        warn = sorted({g["gate"] for g in mine if not g["passed"] and not g["blocking"]})
        if blocking and "candidate" not in statuses:
            out.append({"station": st, "state": "stopped", "gates": blocking})
            stopped = True
        elif mine:
            out.append({"station": st, "state": "passed", "gates": warn, "waived": bool(blocking)})
        else:
            out.append({"station": st, "state": "skipped" if stopped else "pending", "gates": []})
    serving = [g for g in gates if g["stage"] == "serving"]
    failed = [d for d in deps if d["state"] == "failed"]
    if stopped:
        out.append({"station": "staging", "state": "skipped", "gates": []})
    elif any(d.get("error", "") and str(d["error"]).startswith("deferred") for d in failed) and not serving:
        out.append({"station": "staging", "state": "waiting", "gates": [], "note": "another version was in canary"})
    elif any(g["blocking"] for g in serving) or (
        failed and "canary" not in statuses and not str(failed[-1].get("error", "")).startswith("deferred")
    ):
        out.append(
            {
                "station": "staging",
                "state": "stopped",
                "gates": sorted({g["gate"] for g in serving if g["blocking"]}),
                "note": failed[-1]["error"] if failed else None,
            }
        )
        stopped = True
    elif serving or "staging" in statuses:
        state = (
            "passed"
            if "canary" in statuses
            else ("waiting" if any(d["state"] == "waiting_embeddings" for d in deps) else "active")
        )
        out.append({"station": "staging", "state": state, "gates": []})
    else:
        out.append({"station": "staging", "state": "pending", "gates": []})
    if stopped:
        out.append({"station": "canary", "state": "skipped", "gates": []})
    elif "canary" in statuses:
        if "production" in statuses:
            out.append({"station": "canary", "state": "passed", "gates": []})
        elif "rolled_back" in statuses:
            out.append({"station": "canary", "state": "stopped", "gates": ["canary.guard"]})
            stopped = True
        else:
            out.append({"station": "canary", "state": "active", "gates": []})
    else:
        out.append({"station": "canary", "state": "pending", "gates": []})
    if stopped:
        out.append({"station": "production", "state": "skipped", "gates": []})
    elif "production" in statuses:
        out.append(
            {"station": "production", "state": "active" if v["status"] == "production" else "passed", "gates": []}
        )
    else:
        out.append({"station": "production", "state": "pending", "gates": []})
    return out


def _summary(v: dict[str, Any], reg: Registry | None = None) -> dict[str, Any]:
    out = {k: v[k] for k in SUMMARY_FIELDS}
    if reg is not None:
        out["path"] = path_of(reg, v)
    m = v.get("metrics") or {}
    if m.get("overall"):
        out["offline"] = {
            "metric": m.get("metric"),
            "delta": m["overall"]["delta"],
            "low": m["overall"]["low"],
            "high": m["overall"]["high"],
            "champion": m.get("champion"),
        }
    return out


def create_app(registry: Registry | None = None) -> FastAPI:
    app = FastAPI(
        title="ranking-path-to-production", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json"
    )
    reg = registry or Registry()

    def tokens() -> set[str]:
        return {t for t in os.environ.get("PTP_API_TOKENS", "").split(",") if t}

    @app.get("/api/health")
    def health() -> dict[str, str]:
        reg.one("SELECT 1 AS ok")
        return {"status": "ok"}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        versions = reg.versions()
        by_status: dict[str, int] = {}
        for v in versions:
            by_status[v["status"]] = by_status.get(v["status"], 0) + 1
        runs = reg.q(
            "SELECT run_id, kind, state, outcome, version, started_at, finished_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 12"
        )
        for r in runs:
            r["blocking"] = sorted({g["gate"] for g in reg.gates(run_id=r["run_id"]) if g["blocking"]})
        std = standards.check()
        return {
            "routing": reg.all_routing(),
            "models": by_status,
            "production": {k: _summary(p, reg) for k in ("ranker", "tower") if (p := reg.production(k))},
            "recent": [_summary(v, reg) for v in list(reversed(versions))[:10]],
            "runs": runs,
            "open_deployments": [
                d for d in reg.deployments() if d["state"] in ("running", "canary", "waiting_embeddings")
            ],
            "standards": {
                "checked": len(std),
                "production_conforming": sum(1 for s in std if s["status"] == "production" and s["conforms"]),
                "production": sum(1 for s in std if s["status"] == "production"),
            },
            "monitors": reg.q("SELECT surface, version, at, healthy FROM monitor_runs ORDER BY id DESC LIMIT 4"),
        }

    @app.get("/api/models")
    def models(kind: str | None = None) -> list[dict[str, Any]]:
        return [_summary(v, reg) for v in reversed(reg.versions(kind))]

    @app.get("/api/models/{version}")
    def model(version: str) -> dict[str, Any]:
        v = reg.version(version)
        if v is None:
            raise HTTPException(404, f"no model {version}")
        return {
            "model": v,
            "path": path_of(reg, v),
            "gates": reg.gates(version),
            "transitions": reg.transitions(version),
            "deployments": reg.deployments(version),
            "standards": next((s for s in standards.check() if s["version"] == version), None),
        }

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        out = reg.q("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 200")
        for r in out:
            r["blocking"] = sorted({g["gate"] for g in reg.gates(run_id=r["run_id"]) if g["blocking"]})
        return out

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        r = reg.run(run_id)
        if r is None:
            raise HTTPException(404, f"no run {run_id}")
        return {
            "run": r,
            "gates": reg.gates(run_id=run_id),
            "relay": [m for m in reg.relay_messages() if m["pipeline_run"] == run_id],
        }

    @app.get("/api/deployments")
    def deployments() -> list[dict[str, Any]]:
        return list(reversed(reg.deployments()))

    @app.get("/api/routing")
    def routing() -> dict[str, Any]:
        return {"current": reg.all_routing(), "log": list(reversed(reg.routing_log()))[:200]}

    @app.get("/api/standards")
    def standards_report() -> dict[str, Any]:
        return {"standards": config.load("standards")["standards"], "models": standards.check()}

    @app.get("/api/relay")
    def relay() -> list[dict[str, Any]]:
        return list(reversed(reg.relay_messages()))[:200]

    @app.get("/api/seasons")
    def seasons() -> list[dict[str, Any]]:
        out = []
        for path in sorted((config.root() / "reports").glob("**/evaluation.json")):
            data = json.loads(path.read_text())
            out.append(
                {
                    "path": str(path.relative_to(config.root())),
                    "scenario": data.get("scenario"),
                    "summary": data["summary"],
                    "releases": data["releases"],
                    "data_blocks": data.get("data_blocks", []),
                }
            )
        return out

    @app.post("/api/routing/{surface}/rollback")
    def rollback(surface: str, body: dict[str, Any], authorization: str = Header(default="")) -> dict[str, Any]:
        if authorization.removeprefix("Bearer ") not in tokens():
            raise HTTPException(401, "a valid token is needed to change routing")
        from .deploy.canary import rollback_production

        try:
            return rollback_production(reg, surface, str(body.get("reason") or "rollback from the console"))
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    web = Path(os.environ.get("PTP_WEB", str(config.root() / "web" / "dist")))
    if (web / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> Any:
            if path.startswith("api/"):
                return JSONResponse({"detail": "not found"}, status_code=404)
            f = web / path
            return FileResponse(f if path and f.is_file() else web / "index.html")

    return app
