"""The homepage two-tower model: a user tower and an item tower (one hidden ReLU layer each, L2-normalised outputs),
trained with an in-batch softmax over (member, item) interactions. Small enough to train in NumPy in seconds;
exported to ONNX as two models, the user tower for Triton and the item tower for batch scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass
class Tower:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        h_pre = x @ self.w1 + self.b1
        h = np.maximum(h_pre, 0)
        e = h @ self.w2 + self.b2
        n = np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        return e / n, {"x": x, "h": h, "e": e, "n": n}

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(x, dtype=np.float32))[0].astype(np.float32)

    def params(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]


def _init(rng: np.random.Generator, d_in: int, hidden: int, dim: int) -> Tower:
    return Tower(
        rng.normal(0, np.sqrt(2 / d_in), (d_in, hidden)).astype(np.float32),
        np.zeros(hidden, np.float32),
        rng.normal(0, np.sqrt(1 / hidden), (hidden, dim)).astype(np.float32),
        np.zeros(dim, np.float32),
    )


def _backward(t: Tower, cache: dict[str, np.ndarray], z: np.ndarray, dz: np.ndarray) -> list[np.ndarray]:
    de = (dz - z * np.sum(dz * z, axis=1, keepdims=True)) / cache["n"]
    dw2 = cache["h"].T @ de
    db2 = de.sum(axis=0)
    dh = de @ t.w2.T
    dh[cache["h"] <= 0] = 0
    dw1 = cache["x"].T @ dh
    db1 = dh.sum(axis=0)
    return [dw1, db1, dw2, db2]


@dataclass
class TwoTower:
    user: Tower
    item: Tower
    history: list[float] = field(default_factory=list)


def train(users: np.ndarray, items: np.ndarray, cfg: dict[str, Any]) -> TwoTower:
    """users[i], items[i] is a positive pair. In-batch softmax: every other item in the batch is a negative."""
    rng = np.random.default_rng(int(cfg.get("random_state", 0)))
    dim, hidden = int(cfg["dim"]), int(cfg["hidden"])
    temp, lr = float(cfg["temperature"]), float(cfg["learning_rate"])
    bs, epochs = int(cfg["batch_size"]), int(cfg["epochs"])
    ut = _init(rng, users.shape[1], hidden, dim)
    it = _init(rng, items.shape[1], hidden, dim)
    params = ut.params() + it.params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0
    model = TwoTower(ut, it)
    n = len(users)
    for _ in range(epochs):
        order = rng.permutation(n)
        losses = []
        for s in range(0, n - 1, bs):
            idx = order[s : s + bs]
            if len(idx) < 2:
                continue
            zu, cu = ut.forward(users[idx])
            zi, ci = it.forward(items[idx])
            logits = zu @ zi.T / temp
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            b = len(idx)
            losses.append(float(-np.mean(np.log(p[np.arange(b), np.arange(b)] + 1e-12))))
            ds = p.copy()
            ds[np.arange(b), np.arange(b)] -= 1
            ds /= b
            dzu = ds @ zi / temp
            dzi = ds.T @ zu / temp
            grads = _backward(ut, cu, zu, dzu) + _backward(it, ci, zi, dzi)
            step += 1
            for k, (prm, g) in enumerate(zip(params, grads, strict=True)):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                mh = m[k] / (1 - 0.9**step)
                vh = v[k] / (1 - 0.999**step)
                prm -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        model.history.append(float(np.mean(losses)) if losses else float("nan"))
    return model


def to_onnx(t: Tower, input_name: str, output_name: str, target_opset: int, ir_version: int | None = None) -> bytes:
    d_in, dim = t.w1.shape[0], t.w2.shape[1]
    inits = [
        numpy_helper.from_array(t.w1.astype(np.float32), "w1"),
        numpy_helper.from_array(t.b1.astype(np.float32), "b1"),
        numpy_helper.from_array(t.w2.astype(np.float32), "w2"),
        numpy_helper.from_array(t.b2.astype(np.float32), "b2"),
    ]
    nodes = [
        helper.make_node("MatMul", [input_name, "w1"], ["h0"]),
        helper.make_node("Add", ["h0", "b1"], ["h1"]),
        helper.make_node("Relu", ["h1"], ["h"]),
        helper.make_node("MatMul", ["h", "w2"], ["e0"]),
        helper.make_node("Add", ["e0", "b2"], ["e"]),
        helper.make_node("LpNormalization", ["e"], [output_name], axis=1, p=2),
    ]
    graph = helper.make_graph(
        nodes,
        "tower",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", d_in])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["batch", dim])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", target_opset)], producer_name="ptp")
    model.ir_version = ir_version if ir_version is not None else min(model.ir_version, 8)
    onnx.checker.check_model(model)
    return model.SerializeToString()
