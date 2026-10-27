"""Reads the Vespa schema files the package builder writes: document fields and rank profiles (inputs, functions,
onnx-model blocks, first and second phase, match-features). A generic brace parser, tolerant of layout."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    head: str
    body: list[Any] = field(default_factory=list)  # str lines and Blocks


def _blocks(text: str) -> list[Any]:
    text = re.sub(r"#[^\n]*", "", text)
    stack: list[Block] = [Block("root")]
    buf = ""
    parens = 0  # braces inside parentheses belong to a tensor type such as tensor<float>(brand{})
    for ch in text:
        if ch == "(":
            parens += 1
        elif ch == ")":
            parens = max(0, parens - 1)
        if ch == "{" and parens == 0 and not _in_tensor_literal(buf):
            stack.append(Block(buf.strip()))
            buf = ""
        elif ch == "}" and parens == 0 and not _in_tensor_literal(buf) and len(stack) > 1:
            for line in buf.split("\n"):
                if line.strip():
                    stack[-1].body.append(line.strip())
            buf = ""
            done = stack.pop()
            stack[-1].body.append(done)
        elif ch == "\n":
            if buf.strip():
                stack[-1].body.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        stack[-1].body.append(buf.strip())
    if len(stack) != 1:
        raise ValueError("unbalanced braces in schema")
    return stack[0].body


def _in_tensor_literal(buf: str) -> bool:
    # braces inside an expression line (mapped tensor literals) are not blocks; our expressions never use them,
    # but stay safe: a line that already has 'expression:' is not opening a block
    return "expression:" in buf.split("\n")[-1]


@dataclass
class OnnxModel:
    name: str
    file: str
    inputs: dict[str, str]
    outputs: dict[str, str]


@dataclass
class Profile:
    name: str
    inherits: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    functions: dict[str, str] = field(default_factory=dict)
    onnx: dict[str, OnnxModel] = field(default_factory=dict)
    first_phase: str | None = None
    second_phase: str | None = None
    rerank_count: int = 100
    match_features: list[str] = field(default_factory=list)


@dataclass
class Schema:
    name: str
    fields: dict[str, str]
    profiles: dict[str, Profile]

    def resolved(self, name: str) -> Profile:
        """A profile with everything it inherits merged in (child wins)."""
        p = self.profiles[name]
        if not p.inherits:
            return p
        parent = self.resolved(p.inherits)
        return Profile(
            name=p.name,
            inherits=None,
            inputs={**parent.inputs, **p.inputs},
            functions={**parent.functions, **p.functions},
            onnx={**parent.onnx, **p.onnx},
            first_phase=p.first_phase or parent.first_phase,
            second_phase=p.second_phase or parent.second_phase,
            rerank_count=p.rerank_count if p.second_phase else parent.rerank_count,
            match_features=p.match_features or parent.match_features,
        )


def _kv(line: str) -> tuple[str, str]:
    k, _, v = line.partition(":")
    return k.strip(), v.strip()


def _profile(block: Block) -> Profile:
    m = re.match(r"rank-profile\s+(\w+)(?:\s+inherits\s+(\w+))?", block.head)
    if not m:
        raise ValueError(f"bad rank-profile header {block.head!r}")
    p = Profile(name=m.group(1), inherits=m.group(2))
    for item in block.body:
        if isinstance(item, str):
            continue
        head = item.head
        if head == "inputs":
            for line in item.body:
                im = re.match(r"query\((\w+)\)\s*(.*)", line)
                if im:
                    p.inputs[im.group(1)] = im.group(2).lstrip(":").strip() or "double"
        elif head.startswith("function"):
            fm = re.match(r"function\s+(\w+)\s*\(\s*\)", head)
            if not fm:
                raise ValueError(f"bad function header {head!r}")
            expr = " ".join(
                _kv(line)[1] for line in item.body if isinstance(line, str) and line.startswith("expression")
            )
            p.functions[fm.group(1)] = expr
        elif head.startswith("onnx-model"):
            name = head.split()[1]
            model = OnnxModel(name=name, file="", inputs={}, outputs={})
            for line in item.body:
                if line.startswith("file"):
                    model.file = _kv(line)[1]
                elif line.startswith("input"):
                    im = re.match(r'input\s+"?([^":]+)"?\s*:\s*(.+)', line)
                    if im:
                        model.inputs[im.group(1).strip()] = im.group(2).strip()
                elif line.startswith("output"):
                    om = re.match(r'output\s+"?([^":]+)"?\s*:\s*(.+)', line)
                    if om:
                        model.outputs[om.group(2).strip()] = om.group(1).strip()
            p.onnx[name] = model
        elif head == "first-phase":
            p.first_phase = " ".join(_kv(line)[1] for line in item.body if line.startswith("expression"))
        elif head == "second-phase":
            for line in item.body:
                k, v = _kv(line)
                if k == "expression":
                    p.second_phase = v
                elif k == "rerank-count":
                    p.rerank_count = int(v)
        elif head == "match-features":
            for line in item.body:
                p.match_features += line.split()
    return p


def parse(text: str) -> Schema:
    top = _blocks(text)
    schema_block = next(b for b in top if isinstance(b, Block) and b.head.startswith("schema"))
    name = schema_block.head.split()[1]
    fields: dict[str, str] = {}
    profiles: dict[str, Profile] = {}
    for item in schema_block.body:
        if not isinstance(item, Block):
            continue
        if item.head.startswith("document"):
            for f in item.body:
                if isinstance(f, Block) and f.head.startswith("field"):
                    fm = re.match(r"field\s+(\w+)\s+type\s+(.+)", f.head)
                    if fm:
                        fields[fm.group(1)] = fm.group(2).strip()
        elif item.head.startswith("rank-profile"):
            p = _profile(item)
            profiles[p.name] = p
    return Schema(name=name, fields=fields, profiles=profiles)
