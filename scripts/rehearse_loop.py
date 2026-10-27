"""Rehearses the CI deploy loop's second half locally: restore a bundle into an empty registry, seed two Vespa
emulators served over real HTTP on the ports config/targets.yaml names, and run the deploy.

    PTP_DATABASE_URL=<empty db> python scripts/rehearse_loop.py bundle r0007
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ptp import cli
from ptpsim.emulators.vespa import VespaEmulator
from ptpsim.servers import serve


def main(bundle: str, version: str) -> int:
    staging, prod = VespaEmulator(), VespaEmulator()
    servers = [serve(staging.handle, p, "127.0.0.1") for p in (18080, 19071)] + [
        serve(prod.handle, p, "127.0.0.1") for p in (28080, 29071)
    ]
    os.environ["PTP_NOW"] = str(json.loads((Path(bundle) / "meta.json").read_text())["now"])
    try:
        for argv in (
            ["ci", "restore", "--dir", bundle],
            ["ci", "seed-vespa", "--docs", f"{bundle}/vespa_docs.jsonl"],
            ["deploy", "run", "--version", version, "--kind", "ranker", "--trigger", "rehearsal"],
            ["ci", "verify", "--version", version],
        ):
            code = cli.main(argv)
            print(f"ptp {' '.join(argv)} -> {code}", flush=True)
            if code:
                return code
        return 0
    finally:
        for s in servers:
            s.shutdown()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
