"""ptp: the path-to-production command line. The GitHub workflows, the Vertex AI components and the operators all
use these commands."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import clock, config


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_registry(a: argparse.Namespace) -> int:
    from .registry.store import Registry

    reg = Registry()
    if a.action == "bootstrap":
        reg.bootstrap()
    elif a.action == "reset":
        reg.reset()
    elif a.action == "show":
        if a.version:
            v = reg.version(a.version)
            _print(
                {
                    "version": v,
                    "gates": reg.gates(a.version),
                    "transitions": reg.transitions(a.version),
                    "deployments": reg.deployments(a.version),
                }
            )
        else:
            _print(
                [{k: v[k] for k in ("version", "kind", "status", "created_at", "pipeline_run")} for v in reg.versions()]
            )
    return 0


def cmd_adopt(a: argparse.Namespace) -> int:
    from .adopt import adopt

    _print(adopt(a.onnx, a.features.split(","), a.note, a.window_days))
    return 0


def cmd_data(a: argparse.Namespace) -> int:
    import datetime as dt

    from .registry.store import Registry

    reg = Registry()
    if a.action == "quarantine":
        start = dt.date.fromisoformat(a.date)
        end = dt.date.fromisoformat(a.until) if a.until else start
        d = start
        while d <= end:
            reg.quarantine(a.table, d, a.market or None, a.reason, a.by)
            d += dt.timedelta(days=1)
    _print(reg.quarantined(a.table))
    return 0


def cmd_pipeline(a: argparse.Namespace) -> int:
    from .pipelines import runner

    if a.action == "run":
        overrides = json.loads(a.overrides) if a.overrides else {}
        as_of = int(a.as_of) if a.as_of else clock.now()
        out = runner.run(a.kind, a.engine, as_of, a.run_id, overrides, publish=not a.no_publish)
        _print(out)
        return 0 if out.get("state") == "SUCCEEDED" else 1
    if a.action == "publish":
        from .pipelines import runner as r

        kind = "ranker" if a.kind == "ranker" else "tower"
        _print(r.finish_and_publish(a.run_id, kind, r.PIPELINES[a.kind], False, publish=True))
        return 0
    if a.action == "compile":
        from .pipelines import ct

        _print(ct.compile_all())
        return 0
    if a.action == "submit":
        from .pipelines import vertex

        _print(vertex.submit(a.kind, json.loads(a.overrides) if a.overrides else {}))
        return 0
    if a.action == "schedule":
        from .pipelines import vertex

        _print(vertex.schedule(a.kind))
        return 0
    return 2


def cmd_deploy(a: argparse.Namespace) -> int:
    from .registry.store import Registry

    if a.action == "run":
        from .deploy import ranker, tower

        kind = a.kind or (Registry().version(a.version) or {}).get("kind", "ranker")
        out = ranker.run(a.version, a.trigger) if kind == "ranker" else tower.run(a.version, a.trigger)
        _print(out)
        return 0 if out.get("state") not in ("failed",) else 1
    if a.action == "check-canaries":
        from .deploy import canary
        from .lake import Lake

        _print(canary.check(Lake(config.resolve_path(config.load("pipeline")["lake"]))))
        return 0
    if a.action == "rollback":
        from .deploy import canary

        _print(canary.rollback_production(Registry(), a.surface, a.reason))
        return 0
    if a.action == "embeddings-ready":
        from .deploy import tower

        out = tower.embeddings_ready(a.version, a.trigger)
        _print(out)
        return 0 if out.get("state") not in ("failed",) else 1
    return 2


def cmd_monitor(a: argparse.Namespace) -> int:
    from .monitor import skew

    out = skew.run()
    _print(out)
    return 0


def cmd_standards(a: argparse.Namespace) -> int:
    from . import standards

    out = standards.check()
    _print(out)
    return 0 if all(v["conforms"] for v in out if v["status"] == "production") or not a.strict else 1


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    if a.what == "relay":
        from .relay.app import create_app

        host, port = config.load("relay")["listen"].rsplit(":", 1)
        uvicorn.run(create_app(), host=host, port=int(port))
    else:
        from .api import create_app as api_app

        uvicorn.run(api_app(), host=a.host, port=a.port)
    return 0


def cmd_ci(a: argparse.Namespace) -> int:
    from pathlib import Path

    from . import ci

    if a.action == "bundle":
        _print(ci.bundle(Path(a.dir), Path(a.docs) if a.docs else None))
    elif a.action == "restore":
        _print(ci.restore(Path(a.dir)))
    elif a.action == "seed-vespa":
        _print(ci.seed_vespa(Path(a.docs)))
    elif a.action == "verify":
        from .registry.store import Registry

        reg = Registry()
        v = reg.version(a.version)
        deps = reg.deployments(a.version)
        gates = [g for g in reg.gates(a.version) if g["stage"] == "serving"]
        print(
            f"::notice title=deployment of {a.version}::"
            + json.dumps(
                {
                    "status": v and v["status"],
                    "deployments": [
                        {"state": d["state"], "steps": [(s["step"], s["ok"]) for s in d["steps"]]} for d in deps
                    ],
                    "serving_gates": [(g["gate"], g["passed"], g["summary"]) for g in gates],
                },
                default=str,
            )
        )
        ok = bool(
            v and v["status"] == "canary" and gates and all(g["passed"] for g in gates if g["severity"] == "block")
        )
        if not ok:
            print(
                f"::error title=deployment of {a.version} did not reach canary::"
                + json.dumps([d.get("error") for d in deps], default=str)
            )
        return 0 if ok else 1
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from . import evaluate

    out = evaluate.run(a.season)
    _print(out)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ptp")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("registry")
    r.add_argument("action", choices=["bootstrap", "reset", "show"])
    r.add_argument("--version")
    r.set_defaults(fn=cmd_registry)

    ad = sub.add_parser("adopt", help="register a model that was deployed by hand, so the platform can replace it")
    ad.add_argument("--onnx", required=True)
    ad.add_argument("--features", required=True)
    ad.add_argument("--note", default="deployed by hand before the pipeline existed")
    ad.add_argument("--window-days", type=int, default=6)
    ad.set_defaults(fn=cmd_adopt)

    dq = sub.add_parser("data", help="declare lake rows bad after an incident, so training leaves them out")
    dq.add_argument("action", choices=["quarantine", "show"])
    dq.add_argument("--table", default="impressions")
    dq.add_argument("--date")
    dq.add_argument("--until")
    dq.add_argument("--market")
    dq.add_argument("--reason", default="")
    dq.add_argument("--by", default="oncall")
    dq.set_defaults(fn=cmd_data)

    pl = sub.add_parser("pipeline")
    pl.add_argument("action", choices=["run", "publish", "compile", "submit", "schedule"])
    pl.add_argument("--kind", choices=["ranker", "tower", "item-embeddings"], default="ranker")
    pl.add_argument("--engine", choices=["direct", "kfp"], default="direct")
    pl.add_argument("--as-of")
    pl.add_argument("--run-id")
    pl.add_argument("--overrides")
    pl.add_argument("--no-publish", action="store_true")
    pl.set_defaults(fn=cmd_pipeline)

    d = sub.add_parser("deploy")
    d.add_argument("action", choices=["run", "check-canaries", "rollback", "embeddings-ready"])
    d.add_argument("--version")
    d.add_argument("--kind", choices=["ranker", "tower", ""], default="")
    d.add_argument("--trigger", default="manual")
    d.add_argument("--surface", choices=["search", "home"], default="search")
    d.add_argument("--reason", default="manual rollback")
    d.set_defaults(fn=cmd_deploy)

    m = sub.add_parser("monitor")
    m.add_argument("action", choices=["run"])
    m.set_defaults(fn=cmd_monitor)

    st = sub.add_parser("standards")
    st.add_argument("action", choices=["check"])
    st.add_argument("--strict", action="store_true")
    st.set_defaults(fn=cmd_standards)

    sv = sub.add_parser("serve")
    sv.add_argument("what", choices=["api", "relay"], nargs="?", default="api")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    c = sub.add_parser(
        "ci", help="the real deploy loop on GitHub: bundle and restore a registry, seed real Vespa containers"
    )
    c.add_argument("action", choices=["bundle", "restore", "seed-vespa", "verify"])
    c.add_argument("--dir", default="bundle")
    c.add_argument("--docs")
    c.add_argument("--version")
    c.set_defaults(fn=cmd_ci)

    ev = sub.add_parser("evaluate")
    ev.add_argument("season")
    ev.set_defaults(fn=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
