"""Prints a season's releases next to their truth and the gates that decided them.

python scripts/explain_season.py data/seasons/season-a
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(d: Path) -> None:
    ev = json.loads((d / "evaluation.json").read_text())
    reg = json.loads((d / "registry.json").read_text())
    run_of = {v["version"]: v["pipeline_run"] for v in reg["versions"]}
    print(f"{ev['scenario']}: {json.dumps(ev['summary'])}\n")
    for r in ev["releases"]:
        print(
            f"{r['version']:6} truth {r['truth']:10} {r['overall_delta']:+.3f} worst {r['worst_market']} {r['worst_market_delta']:+.3f}  ->  {r['platform']} {r['failed_step'] or ''}"
        )
        for g in reg["gates"]:
            if (g["pipeline_run"] == run_of.get(r["version"]) or g["version"] == r["version"]) and (
                g["blocking"] or g["stage"] == "canary"
            ):
                print(f"         {g['stage']:7} {g['gate']:32} {g['summary'][:110]}")
    for b in ev.get("data_blocks", []):
        print(f"data block {b['run']}: {', '.join(b['gates'])} causes={b['causes']} justified={b['justified']}")
    errors = [r for r in reg["runs"] if r.get("error")]
    for r in errors:
        print(f"ERROR {r['run_id']}: {r['error'][:200]}")
    print(json.dumps(ev["delivery"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
