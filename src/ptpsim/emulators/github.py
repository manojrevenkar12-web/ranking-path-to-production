"""A GitHub emulator: the repository_dispatch API, and a runner that executes this repository's real workflow files.

The runner reads .github/workflows/*.yml, picks the workflows a dispatch (or a schedule) triggers, evaluates each
step's `if:` and the `${{ ... }}` expressions it uses, skips `uses:` actions (checkout, setup, auth), and runs the
`ptp ...` commands of each `run:` script in-process. Commands it does not run (pip install, echo) are recorded. So
a season exercises the commands and conditions the workflow file actually contains.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml


@dataclass
class StepRun:
    name: str
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    exit_code: int = 0
    output: str = ""


@dataclass
class WorkflowRun:
    id: int
    workflow: str
    event: str
    payload: dict[str, Any]
    conclusion: str = "pending"
    steps: list[StepRun] = field(default_factory=list)


EXPR = re.compile(r"\$\{\{\s*(.+?)\s*\}\}")


def _lookup(path: str, ctx: dict[str, Any]) -> Any:
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def evaluate(expr: str, ctx: dict[str, Any]) -> Any:
    """The subset of GitHub expressions the workflows use: context paths, string literals, ==, !=, &&, ||, !."""
    expr = expr.strip()
    for op in ("||", "&&"):
        depth, i = 0, 0
        while i < len(expr) - 1:
            ch = expr[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and expr[i : i + 2] == op:
                a, b = evaluate(expr[:i], ctx), evaluate(expr[i + 2 :], ctx)
                return (a or b) if op == "||" else (a and b)
            i += 1
    for op in ("==", "!="):
        if op in expr:
            a, b = expr.split(op, 1)
            va, vb = evaluate(a, ctx), evaluate(b, ctx)
            return (va == vb) if op == "==" else (va != vb)
    if expr.startswith("!"):
        return not evaluate(expr[1:], ctx)
    if expr.startswith("(") and expr.endswith(")"):
        return evaluate(expr[1:-1], ctx)
    if expr.startswith("'") and expr.endswith("'"):
        return expr[1:-1]
    if expr in ("true", "false"):
        return expr == "true"
    if re.fullmatch(r"-?\d+(\.\d+)?", expr):
        return float(expr)
    return _lookup(expr, ctx)


def substitute(text: str, ctx: dict[str, Any]) -> str:
    def rep(m: re.Match[str]) -> str:
        v = evaluate(m.group(1), ctx)
        return "" if v is None else str(v)

    return EXPR.sub(rep, text)


class GitHubEmulator:
    def __init__(
        self,
        repo_root: Path,
        repository: str,
        token: str,
        ptp_main: Callable[[list[str]], int],
        env: dict[str, str] | None = None,
    ):
        self.root = Path(repo_root)
        self.repository = repository
        self.token = token
        self.ptp_main = ptp_main
        self.env = env or {}
        self.queue: list[tuple[str, dict[str, Any]]] = []
        self.runs: list[WorkflowRun] = []
        self.dispatches: list[dict[str, Any]] = []
        self.fail_next: int = 0  # answer the next N dispatches with 502, as GitHub sometimes does
        self._parsed: dict[tuple[str, float], dict[str, Any]] = {}

    def workflows(self) -> dict[str, dict[str, Any]]:
        out = {}
        for p in sorted((self.root / ".github" / "workflows").glob("*.yml")):
            key = (p.name, p.stat().st_mtime)
            if key not in self._parsed:
                self._parsed[key] = yaml.safe_load(p.read_text())
            out[p.name] = self._parsed[key]
        return out

    # ---- REST API ---------------------------------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == f"/repos/{self.repository}/dispatches":
            if request.headers.get("authorization") != f"Bearer {self.token}":
                return httpx.Response(401, json={"message": "Bad credentials"})
            if self.fail_next > 0:
                self.fail_next -= 1
                return httpx.Response(502, json={"message": "Server Error"})
            body = json.loads(request.content)
            if not body.get("event_type"):
                return httpx.Response(422, json={"message": "event_type is required"})
            if len(body.get("client_payload", {})) > 10:
                return httpx.Response(422, json={"message": "client_payload can have at most 10 top-level properties"})
            self.dispatches.append(body)
            self.queue.append(("repository_dispatch", body))
            return httpx.Response(204)
        return httpx.Response(404, json={"message": "Not Found"})

    # ---- runner -----------------------------------------------------------------------------------------------

    def triggered(self, event: str, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for name, wf in self.workflows().items():
            on = wf.get("on") or dict(wf).get(True, {})  # type: ignore[call-overload]  # yaml reads a bare `on:` key as True
            if isinstance(on, dict) and event in on:
                spec = on[event] or {}
                if event == "repository_dispatch":
                    types = spec.get("types") if isinstance(spec, dict) else None
                    if types and body.get("event_type") not in types:
                        continue
                out.append((name, wf))
        return out

    def schedule(self, workflow: str) -> WorkflowRun:
        wf = self.workflows()[workflow]
        return self._run(workflow, wf, "schedule", {})

    def run_pending(self) -> list[WorkflowRun]:
        done = []
        while self.queue:
            event, body = self.queue.pop(0)
            for name, wf in self.triggered(event, body):
                done.append(self._run(name, wf, event, body))
        return done

    def _run(self, name: str, wf: dict[str, Any], event: str, body: dict[str, Any]) -> WorkflowRun:
        run = WorkflowRun(len(self.runs) + 1, name, event, body.get("client_payload", {}))
        self.runs.append(run)
        ctx: dict[str, Any] = {
            "github": {"event": body, "event_name": event, "repository": self.repository, "run_id": str(run.id)},
            "secrets": {k: v for k, v in self.env.items()},
            "vars": {},
            "env": dict(self.env),
        }
        ok = True
        for job_name, job in (wf.get("jobs") or {}).items():
            job_env = {k: substitute(str(v), ctx) for k, v in (job.get("env") or {}).items()}
            ctx["env"].update(job_env)
            if "if" in job and not evaluate(str(job["if"]).replace("${{", "").replace("}}", ""), ctx):
                continue
            for step in job.get("steps", []):
                sname = step.get("name") or step.get("uses") or (step.get("run") or "").split("\n")[0]
                sr = StepRun(f"{job_name}/{sname}")
                run.steps.append(sr)
                cond = step.get("if")
                if cond is not None:
                    cond_s = str(cond).replace("${{", "").replace("}}", "")
                    if cond_s.strip() in ("always()",):
                        pass
                    elif cond_s.strip() == "failure()":
                        if ok:
                            sr.skipped.append("if: failure()")
                            continue
                    elif not ok or not evaluate(cond_s, ctx):
                        sr.skipped.append(f"if: {cond}")
                        continue
                elif not ok:
                    sr.skipped.append("an earlier step failed")
                    continue
                if "uses" in step:
                    sr.skipped.append(f"uses: {step['uses']}")
                    continue
                script = substitute(step.get("run", ""), ctx)
                for line in _commands(script):
                    if line.startswith("ptp "):
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            try:
                                code = self.ptp_main(shlex.split(line)[1:])
                            except Exception:  # a crashing command fails its step, as it would on a runner
                                import traceback

                                buf.write(traceback.format_exc())
                                code = 1
                        sr.output += buf.getvalue()
                        sr.ran.append(line)
                        if code != 0:
                            sr.exit_code = code
                            ok = False
                            break
                    else:
                        sr.skipped.append(line)
        run.conclusion = "success" if ok else "failure"
        return run


def _commands(script: str) -> list[str]:
    lines, buf = [], ""
    for raw in script.split("\n"):
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip() and not buf.strip().startswith("#"):
            lines.append(buf.strip())
        buf = ""
    return lines
