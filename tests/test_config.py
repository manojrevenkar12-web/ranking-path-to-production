"""Settings live in config/, and the code and the config agree about them."""

from __future__ import annotations

import re
from pathlib import Path

from ptp import config, contract

SRC = Path(__file__).resolve().parents[1] / "src" / "ptp"


def test_markets_agree_between_taxonomy_and_segments() -> None:
    assert config.load("features")["taxonomy"]["markets"] == config.load("gates")["segments"]["market"]


def test_every_gate_the_code_raises_has_a_severity_in_config() -> None:
    raised = set()
    for p in SRC.rglob("*.py"):
        raised |= set(re.findall(r'GateResult\(\s*"([a-z_.]+)"', p.read_text()))
    configured = set(config.load("gates")["gates"])
    assert raised, "no gates found"
    assert raised <= configured, f"gates without a severity in config/gates.yaml: {sorted(raised - configured)}"
    assert set(config.load("gates")["gates"].values()) <= {"block", "warn"}


def test_contract_features_have_ranges_and_expressions() -> None:
    c = contract.load()
    names = [f.name for f in c.features]
    assert len(names) == len(set(names))
    for f in c.features:
        assert f.range[0] < f.range[1]
        assert f.expression
    assert set(config.load("training/ranker")["features"]) <= set(names)


def test_environment_references_expand(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PTP_VESPA_PROD_URL", "http://vespa.example:8080")
    assert config.load("targets")["vespa"]["environments"]["prod"]["query_url"] == "http://vespa.example:8080"
    monkeypatch.delenv("PTP_VESPA_PROD_URL")
    assert config.load("targets")["vespa"]["environments"]["prod"]["query_url"] == "http://127.0.0.1:28080"


def test_no_thresholds_hard_coded_in_gates() -> None:
    """Gate modules read their limits from config: no bare decimal literals in comparisons."""
    for name in ("data.py", "model.py", "serving.py"):
        text = (SRC / "gates" / name).read_text()
        assert not re.search(r"[<>]=?\s*0\.\d", text), name
