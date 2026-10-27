# 0004: Canaries use team-draft interleaving

Status: accepted

## Context

A canary at a share of traffic cannot run an A/B test with enough power for the effects that matter (a few
percent of purchases) in a day.

## Decision

The canary share of a surface's sessions sees one list merged from production and the candidate by team draft.
Clicks, favourites and purchases are credited to the model that contributed the item. The hourly canary check
rolls back when members clearly prefer production (upper bound of the candidate's win share below 0.5 at 99%),
overall or in one market, or when the candidate returns nothing; promotes when there are enough decided sessions
and the lower bound of the win share is above 0.45 overall and 0.35 in every market with enough data; rolls back
at the deadline otherwise.

## Consequences

Members in the canary share see half their results from production, which limits harm. Small markets often lack
the sessions to be judged on their own within the deadline; the offline segment gate is their main protection.
