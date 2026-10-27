"""Triton: the model repository layout and the HTTP/REST protocol (KServe v2 with Triton's repository extension).

Triton runs with --model-control-mode=explicit. A tower version is written into the repository, then loaded with
the repository API; the config pins exactly the versions to keep loaded (production and canary), and clients ask
for a version explicitly, so a canary is a routing decision and a rollback needs no reload. A model's tensor shapes
are fixed by its config, so towers with a different embedding size get a different model name.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np


class TritonError(RuntimeError):
    pass


def model_name(base: str, dim: int) -> str:
    return f"{base}_d{dim}"


def version_number(version: str) -> int:
    """t0007 -> 7: Triton versions are integers."""
    return int(version.lstrip("t"))


def config_pbtxt(name: str, in_dim: int, out_dim: int, versions: list[int], max_batch: int) -> str:
    vs = ", ".join(str(v) for v in sorted(set(versions)))
    return (
        f'name: "{name}"\n'
        'platform: "onnxruntime_onnx"\n'
        f"max_batch_size: {max_batch}\n"
        f'input [ {{ name: "user" data_type: TYPE_FP32 dims: [ {in_dim} ] }} ]\n'
        f'output [ {{ name: "embedding" data_type: TYPE_FP32 dims: [ {out_dim} ] }} ]\n'
        f"version_policy: {{ specific: {{ versions: [ {vs} ] }} }}\n"
        "dynamic_batching { }\n"
        "instance_group [ { kind: KIND_CPU count: 1 } ]\n"
    )


class Triton:
    def __init__(self, http: httpx.Client, url: str, repository: Path, base_name: str):
        self.http = http
        self.url = url.rstrip("/")
        self.repository = Path(repository)
        self.base_name = base_name

    # repository -----------------------------------------------------------------------------------------------

    def write_version(self, name: str, version: int, onnx: bytes) -> Path:
        d = self.repository / name / str(version)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.onnx").write_bytes(onnx)
        return d

    def write_config(self, name: str, in_dim: int, out_dim: int, versions: list[int], max_batch: int) -> None:
        d = self.repository / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.pbtxt").write_text(config_pbtxt(name, in_dim, out_dim, versions, max_batch))

    # protocol ---------------------------------------------------------------------------------------------------

    def ready(self) -> bool:
        try:
            return self.http.get(f"{self.url}/v2/health/ready", timeout=5).status_code == 200
        except httpx.TransportError:
            return False

    def wait_ready(self, timeout_s: float = 120) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready():
                return
            time.sleep(1)
        raise TritonError("server is not ready")

    def load(self, name: str) -> None:
        r = self.http.post(f"{self.url}/v2/repository/models/{name}/load", json={}, timeout=120)
        if r.status_code != 200:
            raise TritonError(f"load {name} failed ({r.status_code}): {_error(r)}")

    def unload(self, name: str) -> None:
        r = self.http.post(f"{self.url}/v2/repository/models/{name}/unload", json={}, timeout=60)
        if r.status_code != 200:
            raise TritonError(f"unload {name} failed ({r.status_code}): {_error(r)}")

    def version_ready(self, name: str, version: int) -> bool:
        r = self.http.get(f"{self.url}/v2/models/{name}/versions/{version}/ready", timeout=10)
        return r.status_code == 200

    def index(self) -> list[dict[str, Any]]:
        r = self.http.post(f"{self.url}/v2/repository/index", json={}, timeout=10)
        return list(r.json()) if r.status_code == 200 else []

    def infer(self, name: str, version: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        body = {"inputs": [{"name": "user", "shape": list(x.shape), "datatype": "FP32", "data": x.ravel().tolist()}]}
        r = self.http.post(f"{self.url}/v2/models/{name}/versions/{version}/infer", json=body, timeout=30)
        if r.status_code != 200:
            raise TritonError(f"infer {name}/{version} failed ({r.status_code}): {_error(r)}")
        out = r.json()["outputs"][0]
        return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])


def _error(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error", r.text))[:300]
    except ValueError:
        return r.text[:300]
