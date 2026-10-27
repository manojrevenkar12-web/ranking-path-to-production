# ranking-path-to-production

**The path to production for a second-hand marketplace's search ranker and homepage recommender: continuous
training on Vertex AI Pipelines, the quality gates the pipelines lacked, ONNX models deployed to Vespa and Triton
by a GitHub Action that a successful training run triggers, canaries decided by interleaving, and the lifecycle
standards that hold it together. Simulated seasons with a hidden truth measure how often the path itself makes
the wrong call.**

A data scientist's model used to reach production like this: export a week of impressions to CSV, run a notebook
(`ad_hoc/train_ranker.py`), copy `model.onnx` into the Vespa application package, deploy it by hand, and post in
the team channel. No holdout, no comparison with what was serving, no record of which data trained what. This
repository is what replaces that, built on the platforms already there (GCP and Vertex AI, Vespa, Triton,
Kubernetes, GitHub Actions) rather than a new one.

```
Vertex AI Pipelines (KFP v2)                       Pub/Sub              relay (Kubernetes)        GitHub Actions
ct-ranker: data gates > train > offline gates  ->  topic  --push-->  verify OIDC token     ->  deploy-model.yml
           > ONNX export gates > register                            exactly-once dispatch      ptp deploy run
           exit handler publishes the outcome                        (PostgreSQL)                    |
                                                                                                     v
staging Vespa: package + candidate profile, replay logged requests: score parity, skew, latency
prod Vespa:    production + candidate + previous profile in one package (rollback is a routing change)
canary:        30% of sessions interleave candidate and production; canary-check.yml promotes or rolls back
```

| | What is here |
|---|---|
| Continuous training | KFP v2 pipelines for Vertex AI: `ct-ranker` (LightGBM LambdaRank with position debiasing), `ct-tower` (two-tower homepage retrieval), `item-embeddings` (batch scoring into Vespa). Exit handlers publish every run's outcome. Compiled specs are checked in; tests run the definitions with the KFP local runner. |
| Quality gates | 20 gates in four stages, every threshold in `config/gates.yaml`: data (freshness, volume, nulls, contract ranges, duplicates, click rates, **provenance**, position bias), offline (**IPS-DCG non-inferiority** overall and per market, plausibility, feature concentration), export (**ONNX parity**, loads in the target runtime's IR and opset, size), serving on staging (**score parity**, skew, latency, embedding coverage), canary (interleaving preference). |
| Serving | A Vespa application package built from one feature contract: features are rank-profile functions, returned as match-features and logged with every impression, so training reads exactly what serving computed. ONNX rankers in the second phase. Embedding fields per tower version. Triton with explicit model control and pinned versions. |
| The deploy loop | Pub/Sub push to a relay that verifies Google's OIDC token, deduplicates in PostgreSQL and sends `repository_dispatch`; `deploy-model.yml` runs `ptp deploy run`: staging, serving gates, production, canary. Hourly `canary-check.yml`, daily `monitor.yml`. CI runs the loop for real against two Vespa containers. |
| Lifecycle | Registry with lineage (pipeline run, commit, data snapshot hash), append-only stage transitions and gate results, model cards, waivers with owner and expiry, `config/standards.yaml` checked for every version, adoption of the hand-deployed model, quarantine of bad lake partitions. |
| Kubernetes | Dockerfile; kustomize base and overlays for the relay, console and Triton (Workload Identity, restricted pod security, PDB, HPA, network policies); CI deploys the relay to kind and tests it there. |
| Console | React, TanStack Query and Framer Motion: each version's path station by station, gates with evidence, the offline interval per market, deployments, routing with the on-call rollback, standards, and the season evidence. |

## The simulated seasons

There is no Vinted here, so the marketplace is simulated (src/ptpsim): members with latent tastes in four markets,
single-item second-hand listings that sell once, position-biased attention, a listing feed, a counter service and
a profile service. The platform runs for real around it: every pipeline run goes through the platform's code, its
event through a Pub/Sub emulator to the real relay app, the relay's dispatch to a GitHub emulator that executes
this repository's workflow files, whose `ptp` commands deploy to Vespa and Triton emulators (which evaluate the
package's ranking expressions and ONNX models for real). The search and homepage services read the platform's
routing and serve the members.

The simulator knows what each release would really do to purchases; only `src/ptp/evaluate.py` reads it, and a
test checks that nothing that decides does.

MEASURED_PLACEHOLDER

## Tests

TESTS_PLACEHOLDER

## Run it

```sh
pip install -e ".[dev]"
export PTP_DATABASE_URL=postgresql://ptp:ptp@localhost:5432/ptp
ptpsim run scenarios/dev/smoke.yaml        # a short season through the whole loop, then its evaluation
ptp serve                                   # the API and console on :8000 (cd web && npm ci && npm run build first)

ptp pipeline run --kind ranker              # one continuous-training run, in-process
ptp pipeline run --kind ranker --engine kfp # the same, with the KFP local runner
ptp pipeline compile                        # pipelines/compiled/*.yaml for Vertex AI
ptp pipeline submit --kind ranker           # on GCP (pip install -e ".[gcp]")
ptp deploy run --version r0007              # what deploy-model.yml runs
ptp deploy check-canaries                   # what canary-check.yml runs
ptp deploy rollback --surface search --reason "conversion dropped"
ptp standards check
```

## Limits

- No GCP project was used. The pipelines are compiled for Vertex AI and run by the KFP local runner; submission
  and scheduling are tested against a stand-in for the SDK. The event path uses the Pub/Sub REST API, here against
  an emulator.
- Vespa and Triton are real in CI (containers), emulated in the seasons. The emulators evaluate the same packages
  and models, but they are not Vespa: no HNSW (exact nearest neighbour), no content distribution, one node.
- The marketplace and its members are simulated; the effects are the simulator's, not Vinted's. The offline metric
  only re-orders what production showed, so it understates a new model's effect; interleaving is the second check.
- Small markets rarely get enough canary sessions to be judged alone; the offline per-market gate is their main
  protection, and it is noisy.
