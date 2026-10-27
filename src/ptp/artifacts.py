"""Model artifacts: a bucket (gs://buyer-ml-models on GCP), a directory here. Every artifact is stored with its
SHA-256, and readers verify it, so the deploy workflow installs exactly the bytes the pipeline registered."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import config


class IntegrityError(RuntimeError):
    pass


class Store:
    def __init__(self, root: str | Path | None = None):
        raw = str(root) if root is not None else config.load("pipeline")["artifacts"]["root"]
        if raw.startswith("gs://"):
            raise NotImplementedError("the GCS backend needs google-cloud-storage; this checkout runs on a directory")
        self.root = config.resolve_path(raw) if not Path(raw).is_absolute() else Path(raw)

    def uri(self, key: str) -> str:
        return f"file://{self.root / key}"

    def put(self, key: str, data: bytes) -> tuple[str, str]:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return self.uri(key), hashlib.sha256(data).hexdigest()

    def put_json(self, key: str, obj: Any) -> tuple[str, str]:
        return self.put(key, json.dumps(obj, indent=2, sort_keys=True, default=str).encode())

    def get(self, key: str, sha256: str | None = None) -> bytes:
        data = (self.root / key).read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise IntegrityError(f"{key}: content does not match its registered sha256")
        return data

    def get_json(self, key: str) -> Any:
        return json.loads(self.get(key))

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def path(self, key: str) -> Path:
        return self.root / key
