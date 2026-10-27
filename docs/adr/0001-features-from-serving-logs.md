# 0001: Training reads features from serving logs

Status: accepted

## Context

The ad-hoc notebook trained on a CSV export and recomputed nothing, but every new feature work item started with a
data scientist joining a table onto impressions. Joins written for training are where training-serving skew and
label leakage come from: an end-of-day aggregate joined onto a morning impression contains what happened after it.

## Decision

Every ranking feature is a Vespa rank-profile function defined once in `config/features.yaml`. Every profile
returns the logged features as match-features; the search service logs them with the impression. Training reads
them from the logs. A feature that is not logged yet can be backfilled from an item snapshot, and the training set
records for each backfilled value the time its snapshot was taken; the provenance gate fails a run where any value
was observed after its impression. A model that uses a new feature starts logging it, so later runs read it from
the logs.

## Consequences

Skew between training and serving is impossible for logged features and is checked on staging for backfilled
ones (serving.skew). A new feature takes one window of logging before it can be trained without a backfill.
