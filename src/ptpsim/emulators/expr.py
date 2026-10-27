"""A small evaluator for the subset of Vespa ranking expressions the package builder writes, vectorised over the
documents of one query. It is part of the emulator, not of the platform: CI checks the same packages against a
real Vespa.

Supported: numbers, + - * /, unary minus, parentheses, attribute(f), query(f), calls to rank-profile functions,
log, exp, fabs (Vespa has no abs), sqrt, sum(mapped * mapped), dense tensor literals tensor<float>(d0[1],d1[N]):[[e1, ..., eN]],
onnx(model).output and closeness(field, f).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

TOKEN = re.compile(
    r"\s*(?:(?P<tensor>tensor(?:<\w+>)?\((?P<dims>[^)]*)\):)|(?P<num>\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/(),.\[\]]))"
)


@dataclass
class Node:
    kind: str
    value: Any = None
    args: tuple[Node, ...] = ()


class ParseError(ValueError):
    pass


def tokenize(src: str) -> list[tuple[str, str]]:
    pos, out = 0, []
    src = src.strip()
    while pos < len(src):
        m = TOKEN.match(src, pos)
        if not m or m.end() == pos:
            raise ParseError(f"cannot parse {src[pos : pos + 20]!r} in {src!r}")
        pos = m.end()
        for kind in ("tensor", "num", "ident", "op"):
            if m.group(kind) is not None:
                out.append((kind, m.group("dims") if kind == "tensor" else m.group(kind)))
                break
    return out


class Parser:
    def __init__(self, src: str):
        self.toks = tokenize(src)
        self.i = 0
        self.src = src

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or (value is not None and tok[1] != value):
            raise ParseError(f"expected {value!r} in {self.src!r}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.expr()
        if self.peek() is not None:
            raise ParseError(f"unexpected {self.peek()} in {self.src!r}")
        return node

    def expr(self) -> Node:
        node = self.term()
        while (t := self.peek()) and t[1] in "+-" and t[0] == "op":
            self.take()
            node = Node("bin", t[1], (node, self.term()))
        return node

    def term(self) -> Node:
        node = self.unary()
        while (t := self.peek()) and t[1] in "*/" and t[0] == "op":
            self.take()
            node = Node("bin", t[1], (node, self.unary()))
        return node

    def unary(self) -> Node:
        t = self.peek()
        if t and t == ("op", "-"):
            self.take()
            return Node("neg", None, (self.unary(),))
        return self.atom()

    def atom(self) -> Node:
        t = self.take()
        kind, val = t
        if kind == "num":
            return Node("num", float(val))
        if kind == "tensor":
            dims = [d.strip() for d in val.split(",")]
            size = int(re.match(r"\w+\[(\d+)\]", dims[-1]).group(1))  # type: ignore[union-attr]
            depth = 0
            while self.peek() == ("op", "["):
                self.take("[")
                depth += 1
            cells = [self.expr()]
            while self.peek() == ("op", ","):
                self.take(",")
                cells.append(self.expr())
            for _ in range(depth):
                self.take("]")
            if len(cells) != size:
                raise ParseError(f"tensor literal has {len(cells)} cells, type says {size}")
            return Node("dense", size, tuple(cells))
        if val == "(":
            node = self.expr()
            self.take(")")
            return node
        if kind != "ident":
            raise ParseError(f"unexpected {val!r} in {self.src!r}")
        if val in ("attribute", "query"):
            self.take("(")
            name = self.take()[1]
            self.take(")")
            return Node(val, name)
        if val == "onnx":
            self.take("(")
            model = self.take()[1]
            self.take(")")
            output = None
            if self.peek() == ("op", "."):
                self.take(".")
                output = self.take()[1]
            return Node("onnx", (model, output))
        if val == "closeness":
            self.take("(")
            self.take()  # 'field'
            self.take(",")
            field = self.take()[1]
            self.take(")")
            return Node("closeness", field)
        if val in ("log", "exp", "fabs", "sqrt", "sum"):
            self.take("(")
            arg = self.expr()
            self.take(")")
            return Node("fn", val, (arg,))
        # a call to a rank-profile function, with or without ()
        if self.peek() == ("op", "("):
            self.take("(")
            self.take(")")
        return Node("ref", val)


def parse(src: str) -> Node:
    return Parser(src).parse()


def references(node: Node) -> dict[str, set[str]]:
    """attribute/query/function/onnx/closeness names an expression uses."""
    out: dict[str, set[str]] = {"attribute": set(), "query": set(), "ref": set(), "onnx": set(), "closeness": set()}

    def walk(n: Node) -> None:
        if n.kind in ("attribute", "query", "ref", "closeness"):
            out[n.kind].add(n.value)
        elif n.kind == "onnx":
            out["onnx"].add(n.value[0])
        for a in n.args:
            walk(a)

    walk(node)
    return out


class _Missing:
    pass


MISSING = _Missing()


class Mapped(list):
    """Per-document mapped tensors (a list of dicts), or one dict for a query tensor."""


class Context:
    """Evaluates expressions over n documents. `columns` holds attribute arrays; `queries` the query inputs."""

    def __init__(
        self,
        n: int,
        columns: dict[str, Any],
        queries: dict[str, Any],
        functions: dict[str, Node],
        models: dict[str, Any],
    ):
        self.n = n
        self.columns = columns
        self.queries = queries
        self.functions = functions
        self.models = models
        self.memo: dict[str, Any] = {}

    def function(self, name: str) -> Any:
        if name not in self.memo:
            if name not in self.functions:
                raise KeyError(f"unknown function {name!r}")
            self.memo[name] = self.eval(self.functions[name])
        return self.memo[name]

    def eval(self, node: Node) -> Any:
        k = node.kind
        if k == "num":
            return np.full(self.n, node.value)
        if k == "attribute":
            col = self.columns.get(node.value)
            if col is None:
                raise KeyError(f"unknown attribute {node.value!r}")
            return col
        if k == "query":
            if node.value not in self.queries:
                return MISSING  # Vespa gives an unset input its default: 0, or an empty tensor
            return self.queries[node.value]
        if k == "ref":
            return self.function(node.value)
        if k == "neg":
            return -_num(self.eval(node.args[0]), self.n)
        if k == "bin":
            a, b = self.eval(node.args[0]), self.eval(node.args[1])
            if (isinstance(a, (Mapped, dict)) or isinstance(b, (Mapped, dict))) and node.value == "*":
                return _mapped_product({} if a is MISSING else a, {} if b is MISSING else b, self.n)
            if isinstance(a, (Mapped, dict)) or isinstance(b, (Mapped, dict)):
                if node.value != "*":
                    raise ValueError("only * is supported between mapped tensors")
                return _mapped_product(a, b, self.n)
            a, b = _num(a, self.n), _num(b, self.n)
            if node.value == "+":
                return a + b
            if node.value == "-":
                return a - b
            if node.value == "*":
                return a * b
            with np.errstate(divide="ignore", invalid="ignore"):
                return a / b
        if k == "fn":
            v = self.eval(node.args[0])
            if node.value == "sum":
                if v is MISSING:
                    return np.zeros(self.n)
                if isinstance(v, Mapped):
                    return np.array([sum(d.values()) for d in v], dtype=float)
                v = np.asarray(v, dtype=float)
                return v.reshape(self.n, -1).sum(axis=1) if v.ndim > 1 else v
            v = _num(v, self.n)
            with np.errstate(divide="ignore", invalid="ignore"):
                return {"log": np.log, "exp": np.exp, "fabs": np.abs, "sqrt": np.sqrt}[node.value](v)
        if k == "dense":
            cols = [_num(self.eval(c), self.n) for c in node.args]
            return np.stack(cols, axis=1).astype(np.float32)
        if k == "onnx":
            model, output = node.value
            return self.models[model](self, output)
        if k == "closeness":
            emb = self.columns.get(node.value)
            q = np.asarray(self.queries.get("user_emb"), dtype=np.float32)
            if emb is None:
                raise KeyError(f"unknown attribute {node.value!r}")
            dots = np.clip(emb @ q, -1.0, 1.0)
            return 1.0 / (1.0 + np.arccos(dots))
        raise ValueError(f"cannot evaluate {k}")


def _num(v: Any, n: int) -> np.ndarray:
    if v is MISSING:
        return np.zeros(n)
    a = np.asarray(v, dtype=float)
    if a.ndim == 0:
        return np.full(n, float(a))
    return a


def _mapped_product(a: Any, b: Any, n: int) -> Mapped:
    def per_doc(x: Any, i: int) -> dict[str, float]:
        if isinstance(x, Mapped):
            return x[i]
        if isinstance(x, dict):
            return x
        raise ValueError("cannot multiply a mapped tensor with a number")

    out = Mapped()
    for i in range(n):
        da, db = per_doc(a, i), per_doc(b, i)
        out.append({key: da[key] * db[key] for key in da.keys() & db.keys()})
    return out
