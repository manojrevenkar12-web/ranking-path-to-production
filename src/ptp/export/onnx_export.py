"""ONNX export and the export gates: parity with the native model, compatibility with the runtime that will load it
(an IR version or opset the serving runtime does not know fails here, in the pipeline), and size."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

from .. import config
from ..gates.base import GateResult


def ranker_to_onnx(booster: Any, n_features: int, export: dict[str, Any]) -> bytes:
    model = onnxmltools.convert_lightgbm(
        booster,
        initial_types=[("features", FloatTensorType([None, n_features]))],
        target_opset=int(export.get("target_opset", 15)),
        zipmap=False,
    )
    # The serialiser stamps its own IR version. A newer onnx package writes a newer one, which an older runtime
    # refuses to load; `ir_version` in the export settings records which one the training image produces.
    if "ir_version" in export:
        model.ir_version = int(export["ir_version"])
    model.doc_string = "ranker"
    return model.SerializeToString()


def run_onnx(data: bytes, x: np.ndarray, input_name: str = "features") -> np.ndarray:
    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    out = sess.run(None, {input_name: x.astype(np.float32)})[0]
    return np.asarray(out, dtype=float).reshape(len(x), -1)


def parity(data: bytes, x: np.ndarray, native: np.ndarray, input_name: str = "features") -> GateResult:
    cfg = config.load("gates")["export"]
    try:
        got = run_onnx(data, x, input_name)[:, 0]
    except Exception as e:  # an export onnxruntime itself cannot run
        return GateResult("export.onnx_parity", False, f"onnxruntime could not run the export: {e}", {"error": str(e)})
    diff = np.abs(got - native)
    worst = float(np.nanmax(diff)) if len(diff) else 0.0
    share = float(np.mean(diff > cfg["onnx_parity_tolerance"])) if len(diff) else 0.0
    nan_mismatch = int(np.sum(np.isnan(got) != np.isnan(native)))
    ok = worst <= cfg["onnx_parity_max_abs"] and share <= cfg["onnx_parity_mismatch_share_max"] and nan_mismatch == 0
    return GateResult(
        "export.onnx_parity",
        ok,
        f"{share:.3%} of {len(x)} holdout rows differ by more than {cfg['onnx_parity_tolerance']:g} (max {worst:.2e})",
        {
            "max_abs": worst,
            "mismatch_share": share,
            "tolerance": cfg["onnx_parity_tolerance"],
            "max_share": cfg["onnx_parity_mismatch_share_max"],
            "max_abs_limit": cfg["onnx_parity_max_abs"],
            "rows": len(x),
            "nan_mismatch": nan_mismatch,
        },
    )


def compat(data: bytes, target: str, expect_inputs: dict[str, int], expect_outputs: list[str]) -> list[GateResult]:
    """target: 'vespa' or 'triton' (config/targets.yaml)."""
    tcfg = config.load("targets")[target]
    limits = tcfg["onnx"]
    model = onnx.load_from_string(data)
    problems = []
    if model.ir_version > limits["max_ir_version"]:
        problems.append(
            f"IR version {model.ir_version} is newer than {target} {tcfg['version']} loads ({limits['max_ir_version']})"
        )
    for imp in model.opset_import:
        lim = limits["max_opset"].get(imp.domain)
        if lim is not None and imp.version > lim:
            problems.append(
                f"opset {imp.version} of '{imp.domain or 'ai.onnx'}' is newer than {target} supports ({lim})"
            )
    inputs = {i.name: i for i in model.graph.input}
    for name, width in expect_inputs.items():
        if name not in inputs:
            problems.append(f"no input named {name!r}")
            continue
        t = inputs[name].type.tensor_type
        if t.elem_type != onnx.TensorProto.FLOAT:
            problems.append(f"input {name} is not float32")
        dims = [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim]
        if len(dims) != 2 or dims[1] != width:
            problems.append(f"input {name} has shape {dims}, expected [batch, {width}]")
    outputs = {o.name for o in model.graph.output}
    for name in expect_outputs:
        if name not in outputs:
            problems.append(f"no output named {name!r}")
    size_limit = min(
        config.load("gates")["export"]["max_model_bytes"],
        int(limits.get("max_bytes", 1 << 62)) if "max_bytes" in limits else 1 << 62,
    )
    return [
        GateResult(
            "export.target_compat",
            not problems,
            f"loads in {target} {tcfg['version']}" if not problems else "; ".join(problems),
            {
                "target": target,
                "ir_version": model.ir_version,
                "opsets": {imp.domain or "ai.onnx": imp.version for imp in model.opset_import},
                "problems": problems,
            },
        ),
        GateResult(
            "export.size",
            len(data) <= size_limit,
            f"{len(data) / 1e6:.2f} MB",
            {"bytes": len(data), "limit": size_limit},
        ),
    ]
