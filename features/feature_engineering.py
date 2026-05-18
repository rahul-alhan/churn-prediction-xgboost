"""Feature engineering: behavioral ratios + bucketing + tenure-aware features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PLAN_MAP = {"bronze": 0, "silver": 1, "gold": 2}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["plan_ord"] = df["plan"].map(PLAN_MAP)
    df["charges_per_tenure"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
    df["tickets_per_claim"] = df["n_support_tickets_90d"] / df["n_claims_90d"].clip(lower=1)
    df["log_total_charges"] = np.log1p(df["total_charges"])
    df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    df["inactive_30d"] = (df["days_since_last_login"] > 30).astype(int)
    df["high_ticket_load"] = (df["n_support_tickets_90d"] >= 3).astype(int)
    return df


FEATURE_COLS = [
    "plan_ord", "tenure_months", "monthly_charges", "total_charges",
    "n_claims_90d", "n_support_tickets_90d", "days_since_last_login",
    "autopay", "charges_per_tenure", "tickets_per_claim", "log_total_charges",
    "is_new_customer", "inactive_30d", "high_ticket_load",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.inp)
    feats = build_features(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    print(f"Wrote {len(feats):,} rows ({len(FEATURE_COLS)} features) → {args.out}")


if __name__ == "__main__":
    main()
