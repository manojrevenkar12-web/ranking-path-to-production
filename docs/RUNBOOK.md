# Runbook

For the MLOps on-call and the owning data-science team. Every action here is a `ptp` command (or the console),
so it is logged in the registry like the automation's own actions.

## A training run was blocked

Open the run in the console (Pipeline runs) or `ptp registry show --version <v>`; the blocking gate and its evidence
say what to do.

| Gate | Usually means | Do |
|---|---|---|
| `data.freshness` | yesterday's log export has not landed | Check the export job. The next scheduled run picks it up; nothing to roll back. |
| `data.null_rate`, `data.range` | a logging or feed change broke a feature in one market | Find the change with the owning team. Once fixed, quarantine the bad days: `ptp data quarantine --date <d> --until <d> --market <m> --reason "<what broke>"`. |
| `data.label_rate` | click events lost (an app release, a tracking change) | As above: fix, then quarantine the affected days for that market. |
| `data.duplicates` (warning) | logging retried events | The builder already dropped them; tell the logging owners. |
| `data.provenance` | a new feature was backfilled from a snapshot taken after the impression | Log the feature first (add it to a profile's match-features) or backfill from `previous_day`. Never waive. |
| `model.non_inferiority`, `model.segment_non_inferiority` | the candidate is worse offline, overall or in a market | Look at the per-market interval on the model page. A refresh that loses in a small market is usually noise on a bad day; the next run decides. A change the team made on purpose needs a look at the market that lost. |
| `export.target_compat` | the training image writes an ONNX IR or opset the serving runtime cannot load | Pin the exporter (`export.ir_version`, `export.target_opset` in `config/training/`), or upgrade the runtime first and then `config/targets.yaml`. |

A waiver is for a known, understood exception: add it to `config/waivers/` by pull request with the version, the
gate, an owner, the reason and an expiry, then `ptp deploy run --version <v>`.

## A deploy failed

`ptp registry show --version <v>` lists the deployment's steps. Production is untouched by construction: the
package always carries the production profile and routing moves only when the canary starts.

- `deploy_staging` with `INVALID_APPLICATION_PACKAGE`: Vespa's message names the profile and the reason. Usually
  an export the compatibility gate did not know about; add the limit to `config/targets.yaml`.
- `serving_gates`: score parity (the model evaluates differently in Vespa), skew (a backfilled feature is computed
  differently live), latency, or for a tower, embedding coverage (the batch scoring did not finish).
- `deferred`: another version was in canary. Nothing to do; the next run brings a fresher candidate.

## A canary was rolled back

The canary check's reason is on the model page under Gates (canary). Members preferred production overall or in
one market, or the candidate returned nothing. Production never stopped serving. Tell the owning team; their next
change goes through the same path.

## Production looks wrong

Roll back first, investigate second: the previous model is still loaded.

```sh
ptp deploy rollback --surface search --reason "purchases in PL down since 14:00"
```

(or Routing in the console, with a token). Then check the daily monitor (`ptp monitor run`): a feature drifting
in one market usually means the listing feed or the counters changed, not the model.

## The relay

Pub/Sub retries a push the relay answered with 5xx (GitHub down); duplicates are recorded and dropped. If GitHub
was down longer than Pub/Sub's retention, republish the run's event: `ptp pipeline publish --run-id <run> --kind ranker`.
