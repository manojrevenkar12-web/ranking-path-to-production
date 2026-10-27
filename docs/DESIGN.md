# Design

## The path

```
 Vertex AI Pipelines (KFP v2)                     Pub/Sub            relay (GKE)          GitHub Actions
 ct-ranker: data gates -> train -> offline  --exit-->  topic --push--> verify OIDC --->  deploy-model.yml
            gates -> export gates -> register  handler              dedupe in PG        ptp deploy run
                                                                    repository_dispatch      |
                                                                                             v
     staging Vespa: package with candidate profile -> replay logged requests -> serving gates
     prod Vespa:    same package (production + candidate + previous profiles)
     routing:       30% of search sessions interleave candidate and production
     canary-check.yml (hourly): promote or roll back from members' preference
```

The homepage tower adds a second loop: the deploy adds the version's embedding field to the Vespa schema and asks
for the item-embeddings batch scoring; the scoring pipeline's own event comes back through the relay and the
same workflow switches the tower on (Triton load, coverage, parity, canary).

## Decisions

- **Features come from serving logs.** Every rank profile returns the contract's features as match-features and the
  search service logs them with the impression. Training never recomputes a feature, so the model learns exactly
  what serving computed. A new feature is either logged first or backfilled from a snapshot taken before the
  impression, and the provenance gate checks the timestamps in the data (ADR 0001).
- **One package, several profiles.** Production, canary and the previous ranker are rank profiles in one Vespa
  package. Promotion and rollback change the `ranking` parameter the search service sends. A failed deploy cannot
  take production down: until the canary starts, routing points where it pointed before (ADR 0002).
- **Offline gates are non-inferiority on IPS-DCG.** Continuous training ships refreshes that are not worse; it does
  not wait for improvements. The metric is unnormalised IPS-weighted DCG on a holdout after the production
  model's data, compared query by query with a paired bootstrap, overall and per market (ADR 0003).
- **Canaries interleave.** A canary gets 30% of a surface's sessions as one list merged from both models (team
  draft), and the canary check decides from which model members credited: roll back early on a clear preference
  for production, overall or in a market; promote once the candidate's win share is not worse (ADR 0004). The
  homepage has about a third of search's sessions, so its canary takes 50% for up to 72 hours (search: 30%, 48 h);
  at 30% the dev seasons' homepage canaries ran out of time before they had enough decided sessions.
- **Exactly one deploy per version.** Pub/Sub delivers at least once and exit handlers can publish twice. The relay
  records message ids and dispatched runs in PostgreSQL and acknowledges only after GitHub accepted; the deploy is
  idempotent per version (an open deployment is reused), so a duplicate dispatch is a no-op.
- **Towers switch on after coverage.** A two-tower model changes the user side (Triton) and the item side (Vespa
  field) together. The new field is added and filled first; the homepage only queries it after the coverage gate
  (ADR 0005).
- **Standards are data.** `config/standards.yaml` says what every model owes; `ptp standards check` scores every
  version, and the adopted ad-hoc model stays flagged until a pipeline model replaces it.

## What is real and what is emulated

| | In this repository | In CI | Not here |
|---|---|---|---|
| Pipelines | KFP v2 definitions compiled for Vertex AI; run by the KFP local runner and in-process | KFP local runner | A GCP project |
| Vespa | Package builder, client, emulator for seasons | Real Vespa container: package, feed, ranking parity, NN, refusals; the real deploy loop | Vespa Cloud |
| Triton | Repository writer, client, emulator | Real Triton container: explicit load, pinned versions, parity, refusal | GPU serving |
| Relay | FastAPI service, PostgreSQL dedup | In a kind cluster with the base manifests | GKE ingress, IAP |
| GitHub | Workflow files, executed by an emulator in seasons | Real `repository_dispatch` starting `deploy-model.yml` against two real Vespa containers | Org-level environments |
