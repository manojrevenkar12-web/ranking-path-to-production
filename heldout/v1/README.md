# Held-out set v1

Written blind from `heldout/WRITING_SCENARIOS.md` only.

- `a.yaml` (heldout-v1-a): no incidents; ordinary tuning, a leak-free trending feature, a wider tower and a cosmetic feature reorder.
- `b.yaml` (heldout-v1-b): FR weighted 3x in training (subtly bad for LT), DE duplicate events, LT null seller_rating (quarantined), a late export.
- `c.yaml` (heldout-v1-c): same-day favourites backfill (leakage), ONNX IR 11 export and revert, PL prices in cents, FR click loss, dispatch outage, LT snapshot missing, rollback.
- `d.yaml` (heldout-v1-d): min_child_samples raised to 200 (subtly bad for LT), position debias off then on, PL null price_fit, LT click loss (quarantined), a very late export.
- `e.yaml` (heldout-v1-e): sound previous-day favourites feature, a one-run learning-rate typo, a shrunk tower, five infrastructure incidents, two quarantines and a rollback.
