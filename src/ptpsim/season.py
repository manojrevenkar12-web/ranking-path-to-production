"""A simulated season: days of marketplace traffic with the platform running for real around it.

Every scheduled training run goes through the platform's pipeline code; its exit event goes through the Pub/Sub
emulator to the real relay app (signed push token, deduplication in PostgreSQL), the relay's repository_dispatch goes
to the GitHub emulator, which executes the repository's actual workflow files, whose `ptp` commands deploy to the
Vespa and Triton emulators. The hourly canary-check and the daily monitor workflows run on their schedule. The
search and homepage services read the platform's routing and serve members, whose behaviour comes from the truth.

At the end, the season directory holds truth.json (what each release really did), registry.json (what the
platform decided) and served.json (who saw what); `ptp evaluate` compares them.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml

from ptp import cli as ptp_cli
from ptp import config, contract
from ptp import http as ptp_http
from ptp.lake import Lake
from ptp.registry.store import Registry
from ptp.serving.triton.client import model_name, version_number
from ptp.serving.vespa import package as P

from . import truth as truth_mod
from .emulators.github import GitHubEmulator
from .emulators.pubsub import PubSubEmulator
from .emulators.triton import TritonEmulator, TritonRuntime
from .emulators.vespa import Runtime, VespaEmulator
from .marketplace import Incidents, Marketplace
from .world import World, WorldConfig

GITHUB_TOKEN = "ghs_emulator_token"


@dataclass
class Scenario:
    name: str
    days: int
    world: dict[str, Any]
    adopt_day: int = 7
    ranker_days: list[int] = field(default_factory=list)
    tower_days: list[int] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    incidents: list[dict[str, Any]] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)
    delivery: dict[str, Any] = field(default_factory=dict)
    operations: list[dict[str, Any]] = field(default_factory=list)
    engine: str = "direct"
    description: str = ""

    @classmethod
    def load(cls, path: Path | str) -> Scenario:
        raw = yaml.safe_load(Path(path).read_text())
        sched = raw.get("schedule", {})
        ranker = sched.get("ranker", {})
        days = int(raw["days"])
        if isinstance(ranker, dict):
            start, every = int(ranker.get("from_day", raw.get("adopt_day", 7) + 2)), int(ranker.get("every", 1))
            ranker_days = list(range(start, days, every))
        else:
            ranker_days = list(ranker)
        return cls(
            name=raw["name"],
            days=days,
            world=raw.get("world", {}),
            adopt_day=int(raw.get("adopt_day", 7)),
            ranker_days=ranker_days,
            tower_days=list(sched.get("tower", [])),
            changes=list(raw.get("changes", [])),
            incidents=list(raw.get("incidents", [])),
            runtime=raw.get("runtime", {}),
            delivery=raw.get("delivery", {}),
            operations=list(raw.get("operations", [])),
            engine=raw.get("engine", "direct"),
            description=raw.get("description", ""),
        )


class Router(httpx.BaseTransport):
    def __init__(self) -> None:
        self.routes: dict[str, Any] = {}

    def add(self, hostport: str, handler: Any) -> None:
        self.routes[hostport] = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.url.host}:{request.url.port}" if request.url.port else request.url.host
        h = self.routes.get(key) or self.routes.get(request.url.host)
        if h is None:
            return httpx.Response(502, json={"message": f"no route to {key}"})
        if hasattr(h, "handle"):
            return h.handle(request)
        return h(request)


class AsgiBridge:
    """Forwards a request to an ASGI app through Starlette's synchronous test client."""

    def __init__(self, app: Any) -> None:
        from fastapi.testclient import TestClient

        self.client = TestClient(app, raise_server_exceptions=False)

    def handle(self, request: httpx.Request) -> httpx.Response:
        r = self.client.request(
            request.method, request.url.path, content=request.content, headers=dict(request.headers)
        )
        return httpx.Response(r.status_code, content=r.content, headers=r.headers)


class TritonHome:
    """The homepage service's Triton call: which model name a tower version lives under is in its metadata."""

    def __init__(self, emu: TritonEmulator, reg: Registry):
        self.emu, self.reg = emu, reg
        self._dims: dict[str, int] = {}

    def embed(self, version: str, vec: np.ndarray) -> np.ndarray | None:
        if version not in self._dims:
            v = self.reg.version(version)
            if v is None:
                return None
            self._dims[version] = int(v["features"]["dim"])
        name = model_name(config.load("targets")["triton"]["model_name"], self._dims[version])
        out = self.emu.infer_local(name, version_number(version), vec[None, :])
        return None if out is None else out[0]


def ptp(args: list[str]) -> int:
    """Runs a ptp command in-process, keeping its JSON output off the console."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return ptp_cli.main(args)


class Season:
    def __init__(self, scenario: Scenario, out: Path):
        self.sc = scenario
        self.out = out
        self.log: list[dict[str, Any]] = []
        self.truth: dict[str, Any] = {"releases": {}, "runs": {}, "scenario": scenario.name}

    # ---- setup ---------------------------------------------------------------------------------------------------

    def setup(self) -> None:
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        env = {
            "PTP_LAKE": str(self.out / "lake"),
            "PTP_ARTIFACTS": str(self.out / "artifacts"),
            "PTP_TRITON_STAGING_REPOSITORY": str(self.out / "triton" / "staging"),
            "PTP_TRITON_PROD_REPOSITORY": str(self.out / "triton" / "prod"),
            "PTP_PUBSUB_API": "http://pubsub.local",
            "PTP_RELAY_JWKS_URL": "http://pubsub.local/oauth2/v3/certs",
            "PTP_GITHUB_API": "http://github.local",
            "PTP_GITHUB_TOKEN": GITHUB_TOKEN,
            "PTP_SCORING_ENGINE": "direct",
            "PTP_CODE_COMMIT": os.environ.get("PTP_CODE_COMMIT", "season"),
        }
        os.environ.update(env)
        self.reg = Registry()
        self.reg.reset()
        rt = self.sc.runtime
        vr = Runtime(**rt.get("vespa", {}))
        tr = TritonRuntime(**rt.get("triton", {}))
        self.prod, self.staging = VespaEmulator(runtime=vr), VespaEmulator(runtime=vr)
        self.t_prod = TritonEmulator(self.out / "triton" / "prod", tr)
        self.t_staging = TritonEmulator(self.out / "triton" / "staging", tr)
        relay_cfg = config.load("relay")["auth"]
        self.pubsub = PubSubEmulator(
            push_url="http://relay.local/pubsub/push",
            audience=relay_cfg["audience"],
            service_account=relay_cfg["service_account"],
            duplicate_every=int(self.sc.delivery.get("duplicate_every", 0)),
        )
        self.github = GitHubEmulator(
            config.root(),
            config.load("targets")["github"]["repository"],
            GITHUB_TOKEN,
            ptp_cli.main,
            {"PTP_DATABASE_URL": os.environ.get("PTP_DATABASE_URL", "")},
        )
        from ptp.relay.app import create_app

        router = Router()
        for port in (18080, 19071):
            router.add(f"127.0.0.1:{port}", self.staging)
        for port in (28080, 29071):
            router.add(f"127.0.0.1:{port}", self.prod)
        router.add("127.0.0.1:18000", self.t_staging)
        router.add("127.0.0.1:28000", self.t_prod)
        router.add("pubsub.local", self.pubsub)
        router.add("relay.local", AsgiBridge(create_app()))
        router.add("github.local", self.github)
        ptp_http.install(router)

        def send(url: str, envelope: dict[str, Any], headers: dict[str, str]) -> int:
            with ptp_http.client() as c:
                return c.post(url, json=envelope, headers=headers).status_code

        self.pubsub.sender = send
        z = P.to_zip(P.build(contract.load(), P.PackageSpec()))
        for emu in (self.prod, self.staging):
            emu.deploy(z)
        self.world = World(WorldConfig.from_dict(self.sc.world))
        self.mk = Marketplace(
            self.world,
            self.prod,
            self.staging,
            Lake(self.out / "lake"),
            Incidents(self.sc.incidents),
            self._routing,
            TritonHome(self.t_prod, self.reg),
        )
        ev = config.load("evaluation")
        self.judge = truth_mod.Judge(
            self.mk,
            vr,
            tr,
            ev["harmful"]["overall_drop"],
            ev["harmful"]["market_drop"],
            ev["beneficial"]["overall_gain"],
        )
        self.mk.initial_listings()
        self._route_cache: dict[str, Any] = {}

    def _routing(self, surface: str) -> dict[str, Any] | None:
        return self._route_cache.get(surface)

    def refresh_routing(self) -> None:
        self._route_cache = {r["surface"]: r for r in self.reg.all_routing()}

    # ---- the day ---------------------------------------------------------------------------------------------------

    def overrides(self, kind: str, day: int) -> dict[str, Any]:
        """DS changes are commits: each applies from its day on, until a later change replaces the same keys."""
        out: dict[str, Any] = {}
        for ch in sorted(self.sc.changes, key=lambda c: c["day"]):
            if ch.get("kind", "ranker") != kind or ch["day"] > day:
                continue
            if ch.get("once") and ch["day"] != day:
                continue
            for k, v in ch.get("overrides", {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = {**out[k], **v}
                else:
                    out[k] = v
        return out

    def on_hour(self, day: int, hour: int) -> None:
        now = self.mk.t(day, hour * 3600 + 600)
        os.environ["PTP_NOW"] = str(now)
        if day == self.sc.adopt_day and hour == 2:
            self.adopt(day)
        for op in self.sc.operations:
            if op["day"] == day and op.get("hour", 1) == hour:
                try:
                    self.operate(op)
                except Exception as e:  # an on-call command that fails (nothing to roll back to) is logged, not fatal
                    self.log.append({"at": now, "operation": op, "error": f"{type(e).__name__}: {e}"})
        if hour == 3 and day in self.sc.ranker_days:
            self.pipeline("ranker", day, now)
        if hour == 4 and day in self.sc.tower_days:
            self.pipeline("tower", day, now)
        if hour == 5 and day > self.sc.adopt_day:
            self.github.schedule("monitor.yml")
            self.daily_scoring(day, now)
        self.github.schedule("canary-check.yml")
        self.refresh_routing()

    def deliver(self) -> None:
        self.pubsub.flush()
        for _ in range(5):
            runs = self.github.run_pending()
            self.pubsub.flush()
            if not runs and not self.github.queue:
                break
            for r in runs:
                self.log.append(
                    {
                        "at": int(os.environ["PTP_NOW"]),
                        "workflow": r.workflow,
                        "conclusion": r.conclusion,
                        "payload": r.payload,
                        "ran": [c for s in r.steps for c in s.ran],
                    }
                )

    def pipeline(self, kind: str, day: int, now: int) -> None:
        ov = self.overrides(kind, day)
        if kind == "ranker":
            for inc in self.mk.incidents.active("dispatch_outage", day):
                self.github.fail_next = int(inc.get("failures", 3))
        run_id = f"ct-{kind}-d{day:02d}"
        t0 = time.time()
        if self.sc.engine == "kfp":
            code = ptp(
                [
                    "pipeline",
                    "run",
                    "--kind",
                    kind,
                    "--engine",
                    "kfp",
                    "--as-of",
                    str(now),
                    "--run-id",
                    run_id,
                    "--overrides",
                    json.dumps(ov),
                ]
            )
        else:
            code = ptp(
                [
                    "pipeline",
                    "run",
                    "--kind",
                    kind,
                    "--as-of",
                    str(now),
                    "--run-id",
                    run_id,
                    "--overrides",
                    json.dumps(ov),
                ]
            )
        run = self.reg.run(run_id) or {}
        self.log.append(
            {
                "at": now,
                "pipeline": run_id,
                "exit": code,
                "outcome": run.get("outcome"),
                "version": run.get("version"),
                "seconds": round(time.time() - t0, 1),
            }
        )
        self.truth["runs"][run_id] = {
            "day": day,
            "kind": kind,
            "incidents": self.window_incidents(kind, day),
            "change": ov,
            "version": run.get("version"),
        }
        version = run.get("version")
        if version:
            judged = (
                self.judge.ranker(self.reg, version, now, day)
                if kind == "ranker"
                else self.judge.tower(self.reg, version, now, day)
            )
            judged["change"] = ov
            self.truth["releases"][version] = judged
        self.deliver()
        # a dispatch that failed every retry is redelivered by Pub/Sub later in the day
        if self.pubsub._undelivered:
            self.pubsub.redeliver()
            self.deliver()

    def operate(self, op: dict[str, Any]) -> None:
        """What the on-call does after an incident, through the same commands they would use."""
        if op["action"] == "quarantine":
            ptp(
                [
                    "data", "quarantine", "--table", op.get("table", "impressions"),
                    "--date", self.mk.as_date(int(op["from_day"])).isoformat(),
                    "--until", self.mk.as_date(int(op.get("to_day", op["from_day"]))).isoformat(),
                    "--market", op.get("market", ""), "--reason", op.get("reason", "incident"),
                ]
            )  # fmt: skip
        elif op["action"] == "rollback":
            ptp(
                [
                    "deploy",
                    "rollback",
                    "--surface",
                    op.get("surface", "search"),
                    "--reason",
                    op.get("reason", "on-call rollback"),
                ]
            )
        self.log.append({"at": int(os.environ["PTP_NOW"]), "operation": op})

    def window_incidents(self, kind: str, day: int) -> list[dict[str, Any]]:
        """World-side incidents whose effects are in this run's training window and were not quarantined."""
        cfg = config.load("gates")["data"]
        first, last = day - int(cfg["window_days"]), day - 1
        quarantined = [o for o in self.sc.operations if o["action"] == "quarantine" and o["day"] <= day]
        out = []
        for inc in self.sc.incidents:
            if inc["type"] in ("dispatch_outage",):
                continue
            start = int(inc.get("from_day", inc.get("day", 0)))
            end = int(inc.get("to_day", inc.get("day", 10_000)))
            if end < first or start > last:
                continue
            covered = any(
                int(q["from_day"]) <= max(start, first)
                and int(q.get("to_day", q["from_day"])) >= min(end, last)
                and q.get("market") in (None, "", inc.get("market"))
                for q in quarantined
            )
            if inc["type"] == "late_partition" and end != last:
                continue  # only the newest partition being late matters at pipeline start
            if not covered:
                out.append(inc)
        return out

    def daily_scoring(self, day: int, now: int) -> None:
        """Item embeddings for the towers in production and canary, for yesterday's new listings (scheduled job)."""
        route = self.reg.routing("home") or {}
        for version in {route.get("production"), route.get("canary")}:
            if version and version.startswith("t"):
                ptp(
                    [
                        "pipeline",
                        "run",
                        "--kind",
                        "item-embeddings",
                        "--as-of",
                        str(now),
                        "--run-id",
                        f"item-embeddings-{version}-d{day:02d}",
                        "--overrides",
                        json.dumps({"version": version}),
                        "--no-publish",
                    ]
                )

    def adopt(self, day: int) -> None:
        """The ad-hoc process, once: export a week of impressions to CSV, run the notebook, deploy by hand; then the
        platform adopts what is in production."""
        import importlib.util

        lake = Lake(self.out / "lake")
        parts = lake.partitions("impressions")[-6:]
        frame = lake.read("impressions", [p.date for p in parts])
        csv = self.out / "adhoc_export.csv"
        frame.to_csv(csv, index=False)
        spec = importlib.util.spec_from_file_location("adhoc_train", config.root() / "ad_hoc" / "train_ranker.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        booster = mod.train(mod.load(csv))
        onnx_path = self.out / "adhoc_model.onnx"
        onnx_path.write_bytes(mod.to_onnx(booster))
        csv.unlink()
        ptp(["adopt", "--onnx", str(onnx_path), "--features", ",".join(mod.FEATURES)])
        self.refresh_routing()
        self.log.append({"at": int(os.environ["PTP_NOW"]), "adopted": True})

    # ---- run -------------------------------------------------------------------------------------------------------

    def run(self, progress: bool = False) -> dict[str, Any]:
        self.setup()
        t0 = time.time()
        for day in range(self.sc.days):
            self.mk.run_day(day, self.on_hour)
            if progress:
                route = self._route_cache.get("search", {})
                print(
                    f"{self.sc.name} day {day:2d}  search={route.get('production')} canary={route.get('canary')}  {time.time() - t0:5.0f}s",
                    flush=True,
                )
        return self.finish()

    def finish(self) -> dict[str, Any]:
        truth_mod.session_value(self.mk)
        served = pd.DataFrame([{k: v for k, v in s.items() if k != "items"} for s in self.mk.served])
        served["day"] = ((served["now"] - self.world.cfg.t0) // 86400).astype(int)
        summary = (
            served.groupby(["day", "surface", "profile", "arm"])
            .agg(sessions=("user", "size"), value=("value", "mean"))
            .reset_index()
        )
        (self.out / "served.json").write_text(summary.to_json(orient="records", indent=1))
        (self.out / "truth.json").write_text(json.dumps(self.truth, indent=2, sort_keys=True, default=str))
        reg = self.reg
        snapshot = {
            "versions": [
                {k: v[k] for k in ("version", "kind", "status", "created_at", "pipeline_run")} for v in reg.versions()
            ],
            "transitions": reg.q("SELECT * FROM stage_transitions ORDER BY id"),
            "deployments": reg.deployments(),
            "gates": reg.q(
                "SELECT version, pipeline_run, stage, gate, passed, blocking, summary FROM gate_results ORDER BY id"
            ),
            "routing_log": reg.routing_log(),
            "relay": reg.relay_messages(),
            "runs": reg.q("SELECT * FROM pipeline_runs ORDER BY started_at"),
            "deliveries": [d.__dict__ for d in self.pubsub.deliveries],
            "dispatches": len(self.github.dispatches),
            "workflow_runs": [
                {"workflow": r.workflow, "conclusion": r.conclusion, "payload": r.payload}
                for r in self.github.runs
                if r.workflow != "canary-check.yml" or r.conclusion != "success"
            ],
            "log": self.log,
            "t0": self.world.cfg.t0,
        }
        (self.out / "registry.json").write_text(json.dumps(snapshot, indent=1, default=str))
        with open(self.out / "vespa_docs.jsonl", "w") as f:
            for doc in self.prod.docs.values():
                f.write(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in doc.items()}) + "\n")
        return {"out": str(self.out)}
