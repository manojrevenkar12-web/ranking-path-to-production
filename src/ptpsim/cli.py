"""ptpsim: run simulated seasons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ptp import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ptpsim")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a season and evaluate it")
    r.add_argument("scenario")
    r.add_argument("--out")
    r.add_argument("--quiet", action="store_true")
    h = sub.add_parser("heldout", help="run a held-out set once and record its first run (never overwritten)")
    h.add_argument("set")
    h.add_argument("--only", help="run one season of the set (no recording)")
    h.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="seasons in parallel, each in its own database (PTP_DATABASE_URL + _<season>)",
    )
    sv = sub.add_parser("serve", help="an emulator as an HTTP server (CI)")
    sv.add_argument("what", choices=["pubsub", "github-stand-in"])
    sv.add_argument("--port", type=int, required=True)
    sv.add_argument("--push-url", default="http://127.0.0.1:8090/pubsub/push")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from . import servers

        auth = config.load("relay")["auth"]
        if a.what == "pubsub":
            servers.run_pubsub(a.port, a.push_url, auth["audience"], auth["service_account"])
        else:
            servers.run_github_stand_in(a.port, auth["audience"], auth["service_account"])
        return 0
    if a.cmd == "heldout":
        return heldout(a.set, a.only, a.jobs)
    if a.cmd == "run":
        from ptp import evaluate

        from .season import Scenario, Season

        sc = Scenario.load(a.scenario)
        out = Path(a.out) if a.out else config.data_dir() / "seasons" / sc.name
        Season(sc, out).run(progress=not a.quiet)
        result = evaluate.run(out)
        print(json.dumps(result["summary"], indent=2))
        return 0
    return 2


def heldout(name: str, only: str | None, jobs: int = 1) -> int:
    """Runs every season of heldout/<set>, writes reports/heldout/<set>/<season>/evaluation.json and, once,
    first_run.json with the aggregate. A recorded first run is never rewritten."""
    import hashlib

    from ptp import evaluate

    from .season import Scenario, Season

    root = config.root()
    folder = root / "heldout" / name
    status = (folder / "STATUS").read_text().strip()
    reports = root / "reports" / "heldout" / name
    first = reports / "first_run.json"
    if first.exists() and not only:
        print(f"{first} exists: a held-out first run is recorded once and never rewritten")
        return 1
    seasons = sorted(folder.glob("*.yaml"))
    if only:
        seasons = [s for s in seasons if s.stem == only]
    results = {}
    if jobs > 1 and not only:
        import os
        import subprocess
        import sys

        base = os.environ["PTP_DATABASE_URL"]
        pending = list(seasons)
        running: list[tuple[subprocess.Popen[bytes], str]] = []
        while pending or running:
            while pending and len(running) < jobs:
                path = pending.pop(0)
                env = {**os.environ, "PTP_DATABASE_URL": f"{base}_{path.stem}"}
                running.append(
                    (
                        subprocess.Popen(
                            [sys.executable, "-m", "ptpsim.cli", "heldout", name, "--only", path.stem], env=env
                        ),
                        path.stem,
                    )
                )
            for proc, stem in list(running):
                if proc.poll() is not None:
                    running.remove((proc, stem))
                    if proc.returncode:
                        print(f"season {stem} failed ({proc.returncode})")
                        return 1
            import time

            time.sleep(5)
        for path in seasons:
            sc = Scenario.load(path)
            results[sc.name] = json.loads((reports / sc.name / "evaluation.json").read_text())["summary"]
        seasons = []
    for path in seasons:
        sc = Scenario.load(path)
        out = config.data_dir() / "seasons" / "heldout" / name / sc.name
        Season(sc, out).run(progress=False)
        ev = evaluate.run(out)
        target = reports / sc.name
        target.mkdir(parents=True, exist_ok=True)
        (target / "evaluation.json").write_text(json.dumps(ev, indent=2, default=str))
        results[sc.name] = ev["summary"]
        print(sc.name, json.dumps(ev["summary"]), flush=True)
    if only:
        return 0
    total: dict[str, object] = {
        "set": name,
        "status_at_run": status,
        "seasons": results,
        "releases": sum(int(r["releases"]) for r in results.values()),
        "harmful": sum(int(r["harmful"]) for r in results.values()),
        "harmful_promoted": sum(int(r["harmful_promoted"]) for r in results.values()),
        "beneficial": sum(int(r["beneficial"]) for r in results.values()),
        "beneficial_promoted": sum(int(r["beneficial_promoted"]) for r in results.values()),
        "data_blocks": sum(int(r["data_blocks"]) for r in results.values()),
        "data_blocks_unjustified": sum(len(r["data_blocks_unjustified"]) for r in results.values()),
        "broken_reaching_prod_deploy": sum(int(r["broken_reaching_prod_deploy"]) for r in results.values()),
    }
    first.write_text(json.dumps(total, indent=2, sort_keys=True))
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    pin = root / "heldout" / "first_run.sha256"
    lines = pin.read_text().splitlines() if pin.exists() else []
    lines.append(f"{digest}  reports/heldout/{name}/first_run.json")
    pin.write_text("\n".join(lines) + "\n")
    print(json.dumps(total, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
