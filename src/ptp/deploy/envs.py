"""Clients for the serving environments named in config/targets.yaml."""

from __future__ import annotations

from .. import config, http
from ..serving.triton.client import Triton
from ..serving.vespa.client import Vespa


def vespa(env: str) -> Vespa:
    t = config.load("targets")["vespa"]
    e = t["environments"][env]
    return Vespa(
        http.client(timeout=float(t["deploy_timeout_s"])), e["query_url"], e["config_url"], float(t["deploy_timeout_s"])
    )


def triton(env: str) -> Triton:
    t = config.load("targets")["triton"]
    e = t["environments"][env]
    return Triton(http.client(timeout=60.0), e["url"], config.resolve_path(e["repository"]), t["model_name"])
