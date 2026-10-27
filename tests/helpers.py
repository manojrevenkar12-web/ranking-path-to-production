"""Shared steps for tests that need a production model and a populated Vespa."""

from __future__ import annotations

import importlib.util
import os
import pickle
from pathlib import Path
from typing import Any

from ptp import cli, config
from ptp.lake import Lake

ROOT = Path(__file__).resolve().parents[1]


def adhoc_module() -> Any:
    spec = importlib.util.spec_from_file_location("adhoc_train", ROOT / "ad_hoc" / "train_ranker.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def adopt_adhoc(workdir: Path, now: int) -> str:
    """The ad-hoc notebook's model, adopted as production (deploys to the emulated staging and prod)."""
    mod = adhoc_module()
    lake = Lake(config.resolve_path(config.load("pipeline")["lake"]))
    parts = lake.partitions("impressions")[-6:]
    csv = workdir / "export.csv"
    lake.read("impressions", [p.date for p in parts]).to_csv(csv, index=False)
    onnx_path = workdir / "adhoc.onnx"
    onnx_path.write_bytes(mod.to_onnx(mod.train(mod.load(csv))))
    os.environ["PTP_NOW"] = str(now)
    assert cli.main(["adopt", "--onnx", str(onnx_path), "--features", ",".join(mod.FEATURES)]) == 0
    return "r0001"


def load_docs(small_lake: Path, *emulators: Any) -> None:
    docs = pickle.loads((small_lake / "docs.pkl").read_bytes())
    for emu in emulators:
        emu.docs = dict(docs)
        emu._cols.clear()
