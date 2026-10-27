"""The platform never sees the truth: src/ptp does not import the simulator, and only evaluate.py reads truth.json."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_platform_never_imports_the_simulator() -> None:
    for p in (SRC / "ptp").rglob("*.py"):
        assert not any(m.startswith("ptpsim") for m in _imports(p)), p


def test_only_evaluate_reads_truth() -> None:
    readers = [p.relative_to(SRC).as_posix() for p in (SRC / "ptp").rglob("*.py") if "truth.json" in p.read_text()]
    assert readers == ["ptp/evaluate.py"]


def test_nothing_that_decides_imports_evaluate() -> None:
    for p in (SRC / "ptp").rglob("*.py"):
        if p.name in ("evaluate.py", "cli.py", "api.py"):
            continue
        assert not any(m.endswith("evaluate") for m in _imports(p)), p


def test_the_world_is_only_in_the_simulator() -> None:
    for p in (SRC / "ptp").rglob("*.py"):
        text = p.read_text()
        assert "user_brand_pref" not in text and "item_quality" not in text, p
