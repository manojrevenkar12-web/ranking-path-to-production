# Held-out seasons

Seasons the platform was never tuned on. Each set was written by someone (an agent) who read only
`WRITING_SCENARIOS.md`: not the code, not the dev scenarios, not the gates. `ptpsim heldout <set>` runs a set once
and records its first run in `reports/heldout/<set>/first_run.json`; `first_run.sha256` pins it and CI fails if it
changes.

If the platform is changed using what a set's failures show, that set becomes a regression set (`STATUS`:
`regression`) and a new one is written blind.

| Set | Status | First run |
|---|---|---|
| v1 | held-out, first run recorded at `5ee1d0e` | 42 releases; 7 of 11 harmful stopped, 13 of 19 better promoted |
