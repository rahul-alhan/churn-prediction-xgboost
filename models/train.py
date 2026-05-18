"""Train XGBoost churn model with stratified CV and class-imbalance handling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from features.feature_engineering import FEATURE_COLS


def train(features_path: str, out_path: str, n_splits: int = 5):
    df = pd.read_parquet(features_path).dropna(subset=["churn"])
    X = df[FEATURE_COLS].values
    y = df["churn"].astype(int).values

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = n_neg / max(1, n_pos)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "verbosity": 0,
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=FEATURE_COLS)
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=FEATURE_COLS)
        booster = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=0,
        )
        proba = booster.predict(dval)
        auroc = roc_auc_score(y[val_idx], proba)
        auprc = average_precision_score(y[val_idx], proba)
        # recall at top-decile threshold
        thr = np.quantile(proba, 0.9)
        pred_high = (proba >= thr).astype(int)
        rec_top = recall_score(y[val_idx], pred_high)
        fold_metrics.append({"auroc": auroc, "auprc": auprc, "recall_top10pct": rec_top})
        print(f"fold {i+1}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}  recall@top10%={rec_top:.3f}")

    # Final model on all data
    dall = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
    final = xgb.train(params, dall, num_boost_round=300)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final.save_model(out_path)
    metrics_path = Path(out_path).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"folds": fold_metrics}, indent=2))
    print(f"\nSaved → {out_path}\nMetrics → {metrics_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    train(args.features, args.out)


if __name__ == "__main__":
    main()
