# 0005: A tower switches on only after its item embeddings cover the catalogue

Status: accepted

## Context

A two-tower model's query side runs in Triton and its item side lives in Vespa as a document field. If the user
tower changes before every item has the matching embedding, retrieval compares vectors from two different models.

## Decision

Each tower version gets its own embedding field and homepage profile. Deploying a tower first adds the field
(Vespa schema change, no restart) and requests the item-embeddings batch scoring for that version. The scoring
pipeline's event brings the deployment to its second phase: the coverage gate counts, in both Vespas, the live
items listed before the scored snapshot that carry the version's embedding (at least 99.5%), Triton loads the
version, its output is checked against the exported model, and only then does the homepage canary start.

## Consequences

A tower rollout takes two pipeline runs and two deploy runs. New listings are retrievable by a tower only after the
next daily scoring; the homepage keeps serving the previous tower meanwhile.
