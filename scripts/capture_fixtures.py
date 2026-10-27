"""Captures API answers from a season's registry into web/test/fixtures, for the console's unit tests.

    PTP_DATABASE_URL=... python scripts/capture_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ptp.api import create_app

OUT = Path(__file__).resolve().parents[1] / "web" / "test" / "fixtures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = TestClient(create_app())
    fixtures: dict[str, object] = {}

    def get(path: str) -> object:
        r = c.get(f"/api/{path}")
        r.raise_for_status()
        return r.json()

    for key in ("overview", "models", "runs", "routing", "standards", "seasons", "deployments"):
        fixtures[key] = get(key)
    fixtures["models_kind-ranker"] = get("models?kind=ranker")
    fixtures["models_kind-tower"] = get("models?kind=tower")
    versions = [m["version"] for m in fixtures["models"]]  # type: ignore[index, union-attr]
    for v in versions:
        fixtures[f"models/{v}"] = get(f"models/{v}")
    for r in fixtures["runs"][:40]:  # type: ignore[index]
        fixtures[f"runs/{r['run_id']}"] = get(f"runs/{r['run_id']}")
    (OUT / "season.json").write_text(json.dumps(fixtures, indent=1, sort_keys=True))
    print(f"{len(fixtures)} answers, {len(versions)} versions")


if __name__ == "__main__":
    main()
