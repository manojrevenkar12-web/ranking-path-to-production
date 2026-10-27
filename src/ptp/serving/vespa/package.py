"""Builds the Vespa application package that serves the buyer models.

A ranker is deployed as a rank profile with its ONNX model in `files/`; the package keeps the production profile,
the canary and one previous profile, so promotion and rollback are a change of the `ranking` parameter the search
service sends, not a redeploy. Every serving profile returns the contract's features as match-features, which is
how the impression logs (and so the training sets) get exactly what serving computed.

A tower version adds an embedding field and a homepage profile that searches it. The field has to exist, and be
filled by the item-embedding batch scoring, before the user tower that matches it is switched on.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any

from ...contract import Contract


@dataclass
class RankerProfile:
    version: str  # e.g. r0007
    onnx: bytes
    features: list[str]  # model input order

    @property
    def name(self) -> str:
        return f"ranker_{self.version}"


@dataclass
class EmbeddingField:
    version: str  # e.g. t0003
    dim: int

    @property
    def field(self) -> str:
        return f"emb_{self.version}"

    @property
    def profile(self) -> str:
        return f"homepage_{self.version}"


@dataclass
class PackageSpec:
    rankers: list[RankerProfile] = field(default_factory=list)
    embeddings: list[EmbeddingField] = field(default_factory=list)

    def profiles(self) -> list[str]:
        return ["popularity", *(r.name for r in self.rankers), *(e.profile for e in self.embeddings)]

    def manifest(self) -> dict[str, Any]:
        return {
            "rankers": [
                {
                    "version": r.version,
                    "profile": r.name,
                    "features": r.features,
                    "sha256": hashlib.sha256(r.onnx).hexdigest(),
                }
                for r in self.rankers
            ],
            "embeddings": [
                {"version": e.version, "field": e.field, "profile": e.profile, "dim": e.dim} for e in self.embeddings
            ],
        }

    @classmethod
    def from_files(cls, files: dict[str, bytes]) -> PackageSpec:
        """The spec of a package built by this module (reads its manifest)."""
        manifest = json.loads(files["files/ptp-manifest.json"])
        rankers = [
            RankerProfile(version=r["version"], onnx=files[f"files/{r['profile']}.onnx"], features=list(r["features"]))
            for r in manifest["rankers"]
        ]
        embeddings = [EmbeddingField(version=e["version"], dim=int(e["dim"])) for e in manifest["embeddings"]]
        return cls(rankers=rankers, embeddings=embeddings)


def _field_block(name: str, spec: dict[str, Any]) -> str:
    vespa_type = spec["type"]
    index = spec.get("index", "attribute")
    lines = [f"        field {name} type {vespa_type} {{"]
    if vespa_type.startswith("tensor"):
        lines.append("            indexing: attribute")
    else:
        lines.append("            indexing: summary | attribute")
    if index == "fast-search":
        lines.append("            attribute: fast-search")
    lines.append("        }")
    return "\n".join(lines)


def _embedding_block(e: EmbeddingField) -> str:
    return "\n".join(
        [
            f"        field {e.field} type tensor<float>(x[{e.dim}]) {{",
            "            indexing: attribute | index",
            "            attribute {",
            "                distance-metric: prenormalized-angular",
            "            }",
            "        }",
        ]
    )


def _inputs(contract: Contract) -> list[str]:
    out = []
    for name, vespa_type in contract.query_inputs.items():
        if vespa_type == "double":
            out.append(f"            query({name}): 0.0")
        else:
            out.append(f"            query({name}) {vespa_type}")
    return out


def schema(contract: Contract, spec: PackageSpec) -> str:
    lines = ["schema item {", "    document item {"]
    for name, fspec in contract.fields.items():
        lines.append(_field_block(name, fspec))
    for e in spec.embeddings:
        lines.append(_embedding_block(e))
    lines.append("    }")

    lines += ["    rank-profile base {", "        inputs {", *_inputs(contract), "        }"]
    for f in contract.features:
        lines += [f"        function {f.name}() {{", f"            expression: {f.expression}", "        }"]
    lines.append("    }")

    match = ["        match-features {", *(f"            {n}" for n in contract.logged), "        }"]

    lines += [
        "    rank-profile popularity inherits base {",
        "        first-phase {",
        f"            expression: {contract.first_phase}",
        "        }",
        *match,
        "    }",
    ]
    for r in spec.rankers:
        cells = ", ".join(r.features)
        # a model that uses a feature not logged yet starts logging it, so the next training run has it from serving
        extra = [f for f in r.features if f not in contract.logged]
        match = ["        match-features {", *(f"            {n}" for n in [*contract.logged, *extra]), "        }"]
        lines += [
            f"    rank-profile {r.name} inherits base {{",
            f"        onnx-model {r.name} {{",
            f"            file: files/{r.name}.onnx",
            f'            input "features": features_{r.version}',
            '            output "variable": score',
            "        }",
            f"        function features_{r.version}() {{",
            f"            expression: tensor<float>(d0[1],d1[{len(r.features)}]):[[{cells}]]",
            "        }",
            "        first-phase {",
            f"            expression: {contract.first_phase}",
            "        }",
            "        second-phase {",
            f"            rerank-count: {contract.rerank_count}",
            f"            expression: sum(onnx({r.name}).score)",
            "        }",
            *match,
            "    }",
        ]
    for e in spec.embeddings:
        lines += [
            f"    rank-profile {e.profile} inherits base {{",
            "        inputs {",
            f"            query(user_emb) tensor<float>(x[{e.dim}])",
            "        }",
            "        first-phase {",
            f"            expression: closeness(field, {e.field})",
            "        }",
            "    }",
        ]
    lines.append("}")
    return "\n".join(lines) + "\n"


SERVICES = """<?xml version="1.0" encoding="utf-8" ?>
<services version="1.0">
    <container id="default" version="1.0">
        <document-api/>
        <search/>
        <nodes>
            <node hostalias="node1"/>
        </nodes>
    </container>
    <content id="items" version="1.0">
        <min-redundancy>1</min-redundancy>
        <documents>
            <document type="item" mode="index"/>
        </documents>
        <nodes>
            <node hostalias="node1" distribution-key="0"/>
        </nodes>
    </content>
</services>
"""

HOSTS = """<?xml version="1.0" encoding="utf-8" ?>
<hosts>
    <host name="localhost">
        <alias>node1</alias>
    </host>
</hosts>
"""


def build(contract: Contract, spec: PackageSpec) -> dict[str, bytes]:
    files = {
        "services.xml": SERVICES.encode(),
        "hosts.xml": HOSTS.encode(),
        "schemas/item.sd": schema(contract, spec).encode(),
        "files/ptp-manifest.json": json.dumps(spec.manifest(), indent=2, sort_keys=True).encode(),
    }
    for r in spec.rankers:
        files[f"files/{r.name}.onnx"] = r.onnx
    return files


def to_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            info = zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
            z.writestr(info, files[path])
    return buf.getvalue()


def from_zip(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist() if not n.endswith("/")}


def digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.encode())
        h.update(hashlib.sha256(files[path]).digest())
    return h.hexdigest()
