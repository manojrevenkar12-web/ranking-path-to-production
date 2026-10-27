"""The repository's workflow files, executed by the GitHub emulator: the dispatch the relay sends starts the deploy
workflow, which runs the right `ptp` command for the event."""

from __future__ import annotations

from pathlib import Path

import yaml

from ptpsim.emulators.github import GitHubEmulator, evaluate

ROOT = Path(__file__).resolve().parents[1]


def _gh() -> tuple[GitHubEmulator, list[list[str]]]:
    calls: list[list[str]] = []

    def main(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    return GitHubEmulator(ROOT, "o/r", "tok", main), calls


def test_a_candidate_dispatch_runs_deploy_run_for_that_version() -> None:
    gh, calls = _gh()
    gh.queue.append(
        (
            "repository_dispatch",
            {
                "event_type": "model-ready",
                "client_payload": {"version": "r0007", "kind": "ranker", "run_id": "ct-1", "outcome": "candidate"},
            },
        )
    )
    runs = gh.run_pending()
    assert [r.workflow for r in runs] == ["deploy-model.yml"] and runs[0].conclusion == "success"
    assert calls[0] == ["deploy", "run", "--version", "r0007", "--kind", "ranker", "--trigger", "dispatch:ct-1"]
    assert calls[1][:2] == ["registry", "show"]
    skipped = [s for st in runs[0].steps for s in st.skipped]
    assert any("uses: google-github-actions/auth" in s for s in skipped)


def test_an_embeddings_ready_dispatch_switches_the_tower_on() -> None:
    gh, calls = _gh()
    gh.queue.append(
        (
            "repository_dispatch",
            {
                "event_type": "model-ready",
                "client_payload": {
                    "version": "t0003",
                    "kind": "tower",
                    "run_id": "ie-1",
                    "outcome": "embeddings_ready",
                },
            },
        )
    )
    gh.run_pending()
    assert calls[0] == ["deploy", "embeddings-ready", "--version", "t0003", "--trigger", "dispatch:ie-1"]


def test_other_event_types_start_nothing() -> None:
    gh, calls = _gh()
    gh.queue.append(("repository_dispatch", {"event_type": "something-else", "client_payload": {}}))
    assert gh.run_pending() == [] and calls == []


def test_a_failing_command_fails_the_run_and_later_steps_only_run_if_always() -> None:
    calls: list[list[str]] = []
    gh = GitHubEmulator(ROOT, "o/r", "tok", lambda argv: (calls.append(argv), 1)[1])
    gh.queue.append(
        (
            "repository_dispatch",
            {
                "event_type": "model-ready",
                "client_payload": {"version": "r1", "kind": "ranker", "run_id": "x", "outcome": "candidate"},
            },
        )
    )
    run = gh.run_pending()[0]
    assert run.conclusion == "failure"
    assert [c[:2] for c in calls] == [
        ["deploy", "run"],
        ["registry", "show"],
    ]  # `if: always()` still shows the deployment


def test_schedules_exist_for_canary_checks_and_monitoring() -> None:
    canary = yaml.safe_load((ROOT / ".github/workflows/canary-check.yml").read_text())
    monitor = yaml.safe_load((ROOT / ".github/workflows/monitor.yml").read_text())
    assert canary[True]["schedule"][0]["cron"].endswith("* * * *")  # hourly
    assert "schedule" in monitor[True]
    gh, calls = _gh()
    gh.schedule("canary-check.yml")
    assert calls == [["deploy", "check-canaries"]]


def test_expression_subset() -> None:
    ctx = {"github": {"event": {"client_payload": {"kind": "tower", "env": None}}}, "inputs": {}}
    assert evaluate("github.event.client_payload.kind || inputs.kind", ctx) == "tower"
    assert evaluate("github.event.client_payload.env != 'ci'", ctx) is True
    assert evaluate("github.event.client_payload.kind == 'ranker' && true", ctx) is False


def test_deploy_workflow_uses_workload_identity_and_serialises_per_surface() -> None:
    wf = yaml.safe_load((ROOT / ".github/workflows/deploy-model.yml").read_text())
    assert wf["permissions"]["id-token"] == "write" and wf["permissions"]["contents"] == "read"
    assert wf["concurrency"]["cancel-in-progress"] is False
    assert "client_payload.kind" in wf["concurrency"]["group"]
