# Working on this repository

## Fast feedback

- `pytest -q -m "not db and not season and not kfp"`: seconds, no services.
- `pytest -q -m "not season and not kfp"`: adds PostgreSQL (`PTP_TEST_DATABASE_URL`); pipelines, deploys, relay.
- `pytest -q -m kfp`: the pipeline definition run by the KFP local runner.
- `ruff check src tests && ruff format --check src tests && mypy src`.
- `ptpsim run scenarios/dev/smoke.yaml` (`PTP_DATABASE_URL` set): a short season through the whole loop.
- `cd web && npm run check`; `npx playwright test` against `ptp serve` after a season.
- The real Vespa, Triton and Kubernetes tests (`-m vespa`, `-m triton`, `-m kind`) run in CI.

## Invariants (each has a test)

- The platform (`src/ptp`) never imports the simulator (`src/ptpsim`) and never reads a season's truth; only
  `src/ptp/evaluate.py` reads `truth.json` (`tests/test_isolation.py`).
- Training reads features from the serving logs (Vespa match-features). A feature that is not logged must be
  backfilled from a snapshot taken before the impression; the provenance gate checks timestamps
  (`tests/test_pipeline.py`).
- Every artifact is read back against its registered sha256 before it is deployed (`tests/test_deploy.py`).
- A failed deploy never changes production: the package keeps the production profile and routing does not move
  until the canary starts (`tests/test_deploy.py`).
- A repeated dispatch or a duplicated Pub/Sub message never starts a second deployment (`tests/test_relay.py`,
  `tests/test_deploy.py`).
- Promotion and rollback are routing changes: production, canary and the previous model stay loaded
  (`tests/test_deploy.py`, `tests/test_tower.py`).
- A tower is switched on only after its item embeddings cover the catalogue (`tests/test_tower.py`).
- Thresholds, margins, schedules, owners and standards live in `config/`; every gate the code raises has a
  severity there (`tests/test_config.py`).
- Stage transitions, gate results, routing changes and relay messages are append-only (database triggers).
- The compiled pipelines in `pipelines/compiled/` match the definitions (`tests/test_vertex_and_migration.py`).

## Held-out seasons

Never change the platform to pass a held-out set before its first run is recorded. A set's first run
(`reports/heldout/<set>/`) is recorded once and never rewritten; `heldout/first_run.sha256` pins it and CI checks
it. If you study a set's failures to change the platform, write `regression` to `heldout/<set>/STATUS` and have
someone who has not seen the code write a new set from `heldout/WRITING_SCENARIOS.md` only.

## Where settings live

`config/features.yaml` (the feature contract and taxonomy), `config/gates.yaml` (every gate and its threshold),
`config/targets.yaml` (Vespa, Triton, GitHub), `config/pipeline.yaml` (Vertex AI, schedules, events),
`config/training/` (what data scientists change), `config/standards.yaml`, `config/relay.yaml`,
`config/evaluation.yaml` (how seasons are judged), `config/waivers/`.
Environment: `PTP_DATABASE_URL`, `PTP_LAKE`, `PTP_ARTIFACTS`, `PTP_VESPA_*`, `PTP_TRITON_*`, `PTP_GITHUB_*`,
`PTP_RELAY_*`, `PTP_API_TOKENS`.
