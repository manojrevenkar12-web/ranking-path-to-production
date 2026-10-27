# Research notes

What the role describes, what is public about the systems it names, and what this repository measured while
building the path to production. Sources are at the end.

## The role

Vinted's Buyer domain runs its data scientists' ranking, retrieval and recommendation models on GCP / Vertex AI,
Vespa and Triton. The job is to replace ad-hoc paths to production with automated ones: continuous training,
batch scoring, ONNX models deployed to Vespa and Triton, a GitHub Action that deploys to Vespa when a Vertex AI
training pipeline succeeds, the quality gates the pipelines lack, and lifecycle standards. The posting calls it a
bridge role: the expert user of existing platforms, not a builder of new ones.

## What is public about Vinted's stack

- Search moved from Elasticsearch to Vespa: one deployment, about half the servers, ranking depth up to 200,000
  candidates, changes visible in search in about 5 seconds instead of 300.
- Homepage recommendations use a two-tower model with approximate nearest-neighbour retrieval in Vespa, which
  replaced FAISS because a read-only index needed rebuilds and could not prefilter (size, price, brand).
- Dense retrieval for search: a two-tower model (multilingual CLIP query tower, item tower over categorical, visual
  and metadata features), HNSW indexes split by market bloc, rolled out after about 50 A/B tests.
- Vespa `match-features` serve as an in-engine feature store and for feature logging: the features Vespa computes
  for a hit are returned with it, and the learning-to-rank work logs them.

This repository follows those choices: ranking features are Vespa rank-profile functions, returned as
match-features and logged with every impression; training reads the logs; the homepage retrieves by nearest
neighbour over item embeddings with a market prefilter.

## Facts the design depends on

- **Vespa** evaluates ONNX models in rank profiles (`onnx-model` with input and output mappings, usually in the
  second or global phase), takes the model as part of the application package, and deploys a package through the
  config server's `prepareandactivate` endpoint on port 19071. A deploy answers with config-change actions
  (restart, refeed, reindex). Ranking models must be self-contained ONNX files below 2 GB, evaluated with batch
  size 1. Consequence used here: several rank profiles, each with its own model, can live in one package, so
  promotion and rollback are a change of the `ranking` parameter, not a redeploy.
- **Triton** in explicit model-control mode loads only what it is asked to through the repository API; a model's
  config fixes its tensor shapes, and a version policy says which versions stay loaded. Consequence: production,
  canary and the rollback target are pinned in the config; a tower with a different embedding size is a different
  model name.
- **Vertex AI Pipelines** run KFP v2 definitions; an `ExitHandler` task runs whatever the pipeline's outcome, which
  is where the run publishes its final state to Pub/Sub. A Pub/Sub push subscription with authentication sends a
  Google-signed OIDC token for its service account, and delivers at least once.
- **GitHub** `repository_dispatch` starts workflows on `repository_dispatch` events; the client payload has at most
  ten top-level properties; a dispatch made with a workflow's `GITHUB_TOKEN` still starts the target workflow.

## What this repository measured

Seasons are simulated (src/ptpsim): members with latent tastes, single-item listings, position-biased attention.
The simulator knows each release's true effect on purchases; the platform never reads it.

1. **Per-query NDCG with inverse-propensity weights still prefers the model that produced the logs.** With one click
   in a query, NDCG is the discount at the rank a scorer gives the clicked item, and the propensity weight cancels
   out of the ratio; clicks sit where the production model put things, so it wins. On season A's logs, the
   offline gate first reported candidates as 3 to 9% worse than production while the truth had them 5 to 9%
   better. Unnormalised IPS-DCG (the unbiased estimator) gave +2 to +7% on the same candidates, the right sign and
   close to the right size. `tests/test_offline_metric.py` reproduces the effect on synthetic logs.
2. **Position bias is measurable from a small exploration slice.** 10% of search sessions show production's results
   in random order; the click-rate-by-position fit recovers the simulator's exponent (0.75) within 0.01.
3. **The ad-hoc model left most of the gain on the table.** The first pipeline model, trained on the same kind of
   logs with LightGBM's unbiased LambdaMART (position debiasing), gave about 60% more expected purchases than the
   notebook's model; without debiasing the same training code gave about 15% less than with it.
4. **A production model's own training days inflate it offline.** The first comparison in the smoke season (still on
   per-query NDCG, see 1) used a holdout day the adopted model had trained on and rated the candidate 12% worse;
   the truth had it 40% better. The pipeline now compares only on days after the production model's data.
5. **Interleaving decides canaries that an A/B split at canary traffic cannot.** At 30% of search traffic for a day,
   team-draft interleaving gives a few hundred decided sessions, enough to see a 55/45 preference.

## Sources

- Vinted moves from Elasticsearch to Vespa (Vespa blog): https://blog.vespa.ai/vinted-moves-from-elasticsearch-to-vespa/
- Adopting Vespa for recommendation retrieval at Vinted (summary): https://frutik.github.io/awesome-search/Articles/Adopting-Vespa-for-Recommendation-Retrieval-at-Vinted
- Dense retrieval at Vinted (summary): https://frutik.github.io/awesome-search/Articles/Dense-Retrieval-at-Vinted
- Optimizing Vespa latency with match-features at Vinted (summary): https://frutik.github.io/awesome-search/Articles/Optimizing-Vespa-Latency-with-Match-Features-at-Vinted
- Vinted Senior Data Scientist, Buyer (team description): https://arc.dev/remote-jobs/j/vinted-senior-data-scientist-buyer-pnbxmm1y2z
- Vespa, ranking with ONNX models: https://docs.vespa.ai/en/ranking/onnx.html
- Vespa, deploy API: https://docs.vespa.ai/en/reference/api/deploy-v2.html
- Triton, model management: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2700/user-guide/docs/user_guide/model_management.html
- Vertex AI, continuous training tutorial: https://docs.cloud.google.com/vertex-ai/docs/pipelines/continuous-training-tutorial
- KFP local execution: https://kubeflow-pipelines.readthedocs.io/en/stable/source/local.html
- Chapelle, Joachims, Radlinski, Yue (2012), Large-scale validation and analysis of interleaved search evaluation, ACM TOIS.
- Joachims, Swaminathan, Schnabel (2017), Unbiased learning-to-rank with biased feedback, WSDM.
