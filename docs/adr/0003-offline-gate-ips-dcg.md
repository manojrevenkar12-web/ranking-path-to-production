# 0003: Offline gates are non-inferiority on unnormalised IPS-DCG

Status: accepted

## Context

The logs come from the production model, and attention falls off with position. Per-query NDCG on such logs, even
with inverse-propensity weights, favours the logging model: with one click in a query the weight cancels out of
the ratio. On season A it rated better candidates 3 to 9% worse than production.

## Decision

Measure position bias on a 10% exploration slice (results shown in random order). The offline metric is IPS-DCG@10,
not normalised per query, on holdout days after the production model's training data. Compare candidate and
production query by query with a paired bootstrap. Gate on non-inferiority: the lower bound must be above -2%
overall and above -8% in every market with at least 300 queries. A gain above 30% is a warning (it pages the owner;
leakage is caught by provenance, real step changes are confirmed by interleaving).

## Consequences

Refreshes ship when they are not worse, without waiting for a significant improvement. The offline metric only
re-orders what production showed, so it understates new candidates' effect; interleaving is the second check.
