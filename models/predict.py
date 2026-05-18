"""Score new customer features against the trained churn model."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb

from features.feature_engineering import FEATURE_COLS


def score(model_path: str, features_path: str, out_path: str):
    df = pd.read_parquet(features_path)
    booster = xgb.Booster()
    booster.load_model(model_path)
    dmat = xgb.DMatrix(df[FEATURE_COLS].values, feature_names=FEATURE_COLS)
    df = df.copy()
    df["churn_proba"] = booster.predict(dmat)
    df["risk_decile"] = pd.qcut(df["churn_proba"], 10, labels=False, duplicates="drop")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df[["customer_id", "churn_proba", "risk_decile"]].to_parquet(out_path, index=False)
    print(f"Scored {len(df):,} customers → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    score(args.model, args.inp, args.out)


if __name__ == "__main__":
    main()
