# Writing held-out seasons

A season is a few weeks of a simulated second-hand fashion marketplace (search and a homepage feed, four markets:
FR, DE, PL, LT) with an ML platform running around it. Every day members search and browse; every morning a
continuous-training pipeline retrains the search ranker on the previous days' logs; now and then a two-tower
homepage model is retrained. Each new model version goes through checks and, if it passes, a canary, then
production.

You write seasons the platform has never seen: what the data scientists change, and what goes wrong in the data
and the infrastructure. The simulator knows what each change really does to purchases; the platform does not.
Write what you think a real team would plausibly do and suffer over three weeks. Do not try to guess what the
platform checks; vary things the way real teams and real systems vary.

## File format

One YAML file per season in `heldout/<set>/`. Keys:

```yaml
name: heldout-v1-a          # unique, used as a directory name
description: >              # a few sentences: what happens in this season and why it is plausible
  ...
days: 24                    # 18 to 24
adopt_day: 7                # keep 7: the team's hand-deployed model is adopted on day 7
world: {seed: 101, search_sessions_per_day: 2000, homepage_sessions_per_day: 700}
schedule:
  ranker: {from_day: 9, every: 1}   # from_day between 9 and 11; every 1 (daily) or 2
  tower: [12, 19]                   # zero to three days, each >= 10
delivery: {duplicate_every: 0}      # Pub/Sub delivers every Nth message twice (0: never); 0 to 5
changes: []
incidents: []
operations: []
```

`world`: `seed` (any integer, different per season), `search_sessions_per_day` (1500 to 2200),
`homepage_sessions_per_day` (500 to 800). Leave the rest at their defaults.

## Changes (what the data scientists merge)

Each change applies to the training run of its `day`. With `once: true` it applies to that day's run only
(someone fixes it the next day); without it, it stays from that day on until a later change sets the same keys.

```yaml
- {day: 13, kind: ranker, once: true, overrides: {...}, note: why someone did this}
```

Ranker overrides (`kind: ranker`):

- `params`: LightGBM parameters, merged into the current ones. Current values: `n_estimators: 150`,
  `learning_rate: 0.06`, `num_leaves: 31`, `min_child_samples: 40`, `subsample: 0.8`, `colsample_bytree: 0.9`.
- `features`: the full list of model inputs. The current list is
  `[price_rel, log_age_days, condition, photo_quality, seller_rating, item_ctr, log_favourites, brand_affinity, price_fit]`.
  You may drop any of them, reorder them, or add `favourites_today` (a live counter of today's favourites on the
  item; it is not in the impression logs, so adding it needs `backfill`).
- `backfill`: `{favourites_today: {snapshot: previous_day}}` or `{favourites_today: {snapshot: same_day}}`: fill a
  feature that is not logged from the daily item snapshot (the snapshot of the day before the impression, or the
  snapshot taken at the end of the impression's own day).
- `market_weights`: e.g. `{FR: 3}`: repeat a market's training sessions to weight it more.
- `position_debias`: `true` (current) or `false`.
- `labels`: `{purchase: 3, favourite: 2, click: 1}` (current); you may change the grades.
- `export`: `{ir_version: 11}` (the training image was upgraded and writes a newer ONNX IR version) or
  `{target_opset: N}`.

Tower overrides (`kind: tower`): `dim` (embedding size, current 32), `hidden` (64), `epochs` (12),
`learning_rate` (0.02), `temperature` (0.1), `batch_size` (256).

## Incidents (what breaks)

```yaml
- {type: dup_events, market: DE, rate: 0.05, day: 11}          # logging duplicates a share of impression events
- {type: null_feature, feature: seller_rating, market: FR, day: 10}   # a feature missing from the logs in one market
- {type: click_drop, market: DE, rate: 0.6, day: 16}           # a share of click events lost
- {type: late_partition, day: 17, hours: 9}                    # that day's log export lands this many hours late
- {type: price_cents, market: PL, from_day: 16, to_day: 17}    # the listing feed sends prices in cents (serving too)
- {type: snapshot_missing_market, market: PL, day: 20}         # the item snapshot export drops a market
- {type: dispatch_outage, failures: 4, day: 19}                # GitHub answers 5xx to the next N dispatches
```

`day` / `from_day` / `to_day` are the days the incident happens (its effect on training shows up in the runs
after it). `null_feature` features: any of the current feature names. Rates between 0.01 and 0.9.

## Operations (what the on-call does)

```yaml
- {day: 18, hour: 1, action: quarantine, table: impressions, from_day: 16, to_day: 17, market: PL, reason: ...}
- {day: 20, hour: 10, action: rollback, surface: search, reason: ...}
```

`quarantine` excludes those days' logs for that market (or all markets without `market`) from training.
`rollback` sends production back to the previous model. Use operations the way an on-call would: after noticing
an incident, a day or two later, and not always.

## What to write

Write five seasons for a set. Make them different from each other:

- at least one season with no incidents and only ordinary data-science changes;
- the others mixing two to five incidents and two to four changes, in different orders and markets;
- some changes that are probably improvements, some that are probably harmful, some that are neither;
- at least one change in some season that is plausible but subtly bad for a small market.

Keep each season to at most 24 days. Do not reuse the seeds 5 and 11. Put a `README.md` in the set's folder
listing the seasons with one line each.
