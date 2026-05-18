"""SHAP report — global importance and per-customer reasons."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import shap
import xgboost as xgb

from features.feature_engineering import FEATURE_COLS


def report(model_path: str, features_path: str, out_path: str, sample: int = 1000):
    booster = xgb.Booster()
    booster.load_model(model_path)
    df = pd.read_parquet(features_path).sample(min(sample, 999_999), random_state=1)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(df[FEATURE_COLS])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = shap.summary_plot(
        shap_values, df[FEATURE_COLS], feature_names=FEATURE_COLS, show=False
    )
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(str(Path(out_path).with_suffix(".png")), dpi=120)
    plt.close()

    # Per-customer top-drivers JSON
    drivers = []
    for i, row_vals in enumerate(shap_values):
        ranked = sorted(
            zip(FEATURE_COLS, row_vals.tolist()), key=lambda kv: -abs(kv[1])
        )[:5]
        drivers.append(
            {
                "customer_id": str(df.iloc[i].get("customer_id", i)),
                "top_drivers": [{"feature": k, "shap": float(v)} for k, v in ranked],
            }
        )
    pd.DataFrame(drivers).to_json(out_path, orient="records", indent=2)
    print(f"Per-customer drivers → {out_path}")
    print(f"Global summary plot → {Path(out_path).with_suffix('.png')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample", type=int, default=1000)
    args = p.parse_args()
    report(args.model, args.features, args.out, args.sample)


if __name__ == "__main__":
    main()
