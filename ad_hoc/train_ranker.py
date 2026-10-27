# Exported from notebooks/search_ranker_v3.ipynb (Search Relevance). Kept as the team left it: this is the "before".
#
# How it was used: export the last week of impressions from BigQuery to CSV by hand, run this, copy model.onnx into
# the Vespa application package, `vespa deploy`, and post in #search-relevance that a new model is live.
# No holdout, no comparison with what is in production, no record of which data trained what.
#
# The platform keeps what it computes (src/ptp/training/ranker.py with position_debias: false); tests/test_migration.py
# checks that both produce the same model on the same data.

import sys

import lightgbm as lgb
import numpy as np
import onnxmltools
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

FEATURES = ["price_rel", "log_age_days", "condition", "photo_quality", "seller_rating", "item_ctr", "log_favourites", "brand_affinity", "price_fit"]

PARAMS = dict(
    objective="lambdarank",
    n_estimators=150,
    learning_rate=0.06,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    lambdarank_truncation_level=20,
    random_state=7,
    verbose=-1,
)


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["surface"] == "search"]
    df = df.drop_duplicates(subset=["impression_id"])
    label = np.zeros(len(df), dtype=int)
    label = np.where(df["clicked"], np.maximum(label, 1), label)
    label = np.where(df["favourited"], np.maximum(label, 2), label)
    label = np.where(df["purchased"], np.maximum(label, 3), label)
    df["label"] = label
    df = df.sort_values(["session_id", "position"], kind="stable").reset_index(drop=True)
    return df


def train(df):
    X = df[["f_" + f for f in FEATURES]].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    group = df.groupby("session_id", sort=False).size().to_numpy()
    m = lgb.LGBMRanker(**PARAMS)
    m.fit(X, y, group=group)
    return m.booster_


def to_onnx(booster):
    onx = onnxmltools.convert_lightgbm(booster, initial_types=[("features", FloatTensorType([None, len(FEATURES)]))], target_opset=15, zipmap=False)
    return onx.SerializeToString()


if __name__ == "__main__":
    booster = train(load(sys.argv[1]))
    with open(sys.argv[2] if len(sys.argv) > 2 else "model.onnx", "wb") as f:
        f.write(to_onnx(booster))
    print("done, now deploy it")
