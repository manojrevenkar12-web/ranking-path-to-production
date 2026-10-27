"""Settings live in config/*.yaml. Values may reference the environment as ${NAME:-default}."""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def root() -> Path:
    env = os.environ.get("PTP_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "gates.yaml").exists():
            return parent
    return Path.cwd()


def data_dir() -> Path:
    return Path(os.environ.get("PTP_DATA", str(root() / "data")))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@cache
def _load(path: str, mtime: float) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def load(name: str) -> dict[str, Any]:
    """config/<name>.yaml with environment references expanded. `name` may contain a subdirectory."""
    path = root() / "config" / f"{name}.yaml"
    raw = _load(str(path), path.stat().st_mtime)
    return _expand(raw)


def resolve_path(value: str) -> Path:
    """A path from config: absolute as given, relative to the data directory's parent (the repo) otherwise."""
    p = Path(value)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "data":
        return data_dir().joinpath(*p.parts[1:])
    return root() / p
