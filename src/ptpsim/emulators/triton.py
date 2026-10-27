"""A Triton emulator: the HTTP/REST endpoints the platform uses (health, repository index, load, unload, version
readiness, infer) over a model repository directory, in explicit model-control mode, with onnxruntime.

Like Triton, a load reads config.pbtxt and refuses a model whose ONNX inputs or outputs do not match it, loads
exactly the versions the version policy names, and keeps serving the previously loaded versions if a load fails.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import onnx
import onnxruntime as ort


@dataclass
class TritonRuntime:
    max_ir_version: int = 10
    max_opset: int = 20


@dataclass
class Loaded:
    config: dict[str, Any]
    sessions: dict[int, ort.InferenceSession] = field(default_factory=dict)


def parse_config(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r'name:\s*"([^"]+)"', text)
    out["name"] = m.group(1) if m else ""
    m = re.search(r"max_batch_size:\s*(\d+)", text)
    out["max_batch_size"] = int(m.group(1)) if m else 0
    out["inputs"] = [
        {"name": a, "dims": [int(x) for x in b.split(",")]}
        for a, b in re.findall(r'input\s*\[\s*\{\s*name:\s*"([^"]+)"[^}]*dims:\s*\[\s*([\d,\s]+)\]', text)
    ]
    out["outputs"] = [
        {"name": a, "dims": [int(x) for x in b.split(",")]}
        for a, b in re.findall(r'output\s*\[\s*\{\s*name:\s*"([^"]+)"[^}]*dims:\s*\[\s*([\d,\s]+)\]', text)
    ]
    m = re.search(r"specific:\s*\{\s*versions:\s*\[([\d,\s]+)\]", text)
    out["versions"] = [int(x) for x in m.group(1).split(",")] if m else None
    return out


class TritonEmulator:
    def __init__(self, repository: Path, runtime: TritonRuntime | None = None):
        self.repository = Path(repository)
        self.runtime = runtime or TritonRuntime()
        self.models: dict[str, Loaded] = {}
        self.lock = threading.RLock()
        self.log: list[str] = []

    def _load(self, name: str) -> tuple[int, str]:
        d = self.repository / name
        cfg_path = d / "config.pbtxt"
        if not cfg_path.exists():
            return 400, f"failed to load '{name}': no config.pbtxt"
        cfg = parse_config(cfg_path.read_text())
        versions = cfg["versions"] or sorted(int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit())[-1:]
        loaded = Loaded(cfg)
        for v in versions:
            f = d / str(v) / "model.onnx"
            if not f.exists():
                return 400, f"failed to load '{name}' version {v}: model.onnx not found"
            data = f.read_bytes()
            model = onnx.load_from_string(data)
            if model.ir_version > self.runtime.max_ir_version:
                return (
                    400,
                    f"failed to load '{name}' version {v}: Unsupported model IR version: {model.ir_version}, max supported IR version: {self.runtime.max_ir_version}",
                )
            for imp in model.opset_import:
                if imp.domain in ("", "ai.onnx") and imp.version > self.runtime.max_opset:
                    return 400, f"failed to load '{name}' version {v}: opset {imp.version} is not supported"
            sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
            ins = {i.name: i.shape for i in sess.get_inputs()}
            for spec in cfg["inputs"]:
                shape = ins.get(spec["name"])
                if shape is None:
                    return 400, f"failed to load '{name}' version {v}: unexpected inference input '{spec['name']}'"
                if isinstance(shape[-1], int) and shape[-1] != spec["dims"][-1]:
                    return (
                        400,
                        f"failed to load '{name}' version {v}: input '{spec['name']}' dims {shape[1:]} don't match configuration dims {spec['dims']}",
                    )
            outs = {o.name: o.shape for o in sess.get_outputs()}
            for spec in cfg["outputs"]:
                shape = outs.get(spec["name"])
                if shape is None:
                    return 400, f"failed to load '{name}' version {v}: unexpected inference output '{spec['name']}'"
                if isinstance(shape[-1], int) and shape[-1] != spec["dims"][-1]:
                    return (
                        400,
                        f"failed to load '{name}' version {v}: output '{spec['name']}' dims {shape[1:]} don't match configuration dims {spec['dims']}",
                    )
            loaded.sessions[v] = sess
        self.models[name] = loaded  # only replaces what was loaded once everything loaded
        self.log.append(f"loaded {name} versions {sorted(loaded.sessions)}")
        return 200, ""

    def infer_local(self, name: str, version: int, x: np.ndarray) -> np.ndarray | None:
        m = self.models.get(name)
        if m is None or version not in m.sessions:
            return None
        sess = m.sessions[version]
        return np.asarray(sess.run(None, {m.config["inputs"][0]["name"]: x.astype(np.float32)})[0])

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        with self.lock:
            if path in ("/v2/health/ready", "/v2/health/live"):
                return httpx.Response(200)
            if path == "/v2/repository/index" and method == "POST":
                listing: list[dict[str, str]] = []
                for name, loaded in self.models.items():
                    listing += [{"name": name, "version": str(v), "state": "READY"} for v in loaded.sessions]
                return httpx.Response(200, json=listing)
            if m := re.match(r"^/v2/repository/models/([^/]+)/(load|unload)$", path):
                name, action = m.groups()
                if action == "unload":
                    self.models.pop(name, None)
                    return httpx.Response(200, json={})
                status, err = self._load(name)
                return httpx.Response(status, json={} if status == 200 else {"error": err})
            if m := re.match(r"^/v2/models/([^/]+)/versions/(\d+)/ready$", path):
                name, v = m.group(1), int(m.group(2))
                ok = name in self.models and v in self.models[name].sessions
                return httpx.Response(200 if ok else 400)
            if m := re.match(r"^/v2/models/([^/]+)/versions/(\d+)/infer$", path):
                name, v = m.group(1), int(m.group(2))
                body = json.loads(request.content)
                inp = body["inputs"][0]
                x = np.asarray(inp["data"], dtype=np.float32).reshape(inp["shape"])
                result = self.infer_local(name, v, x)
                if result is None:
                    return httpx.Response(
                        400, json={"error": f"Request for unknown model: '{name}' version {v} is not at ready state"}
                    )
                return httpx.Response(
                    200,
                    json={
                        "model_name": name,
                        "model_version": str(v),
                        "outputs": [
                            {
                                "name": self.models[name].config["outputs"][0]["name"],
                                "datatype": "FP32",
                                "shape": list(result.shape),
                                "data": result.ravel().tolist(),
                            }
                        ],
                    },
                )
        return httpx.Response(404, json={"error": "not found"})
