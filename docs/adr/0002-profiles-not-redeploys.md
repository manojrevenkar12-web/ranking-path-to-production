# 0002: Promotion and rollback are routing changes

Status: accepted

## Context

A Vespa model lives in the application package. Redeploying the package to roll back takes minutes and can itself
fail, at the moment production most needs to change.

## Decision

The package holds the production ranker, the canary and the previous production ranker as separate rank profiles.
The search service sends `ranking=<profile>` per request from the routing table. Promotion moves production to the
canary's profile and keeps the old one as the rollback target; rollback is the reverse, one row in PostgreSQL. The
next deploy drops profiles older than the rollback target. Triton does the same with pinned versions.

## Consequences

The package is at most three models larger. A deploy that fails anywhere leaves routing where it was. The
standards check flags a production model whose rollback target is not loaded.
