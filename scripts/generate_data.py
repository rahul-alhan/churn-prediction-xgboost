"""Synthetic membership data with churn labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["bronze", "silver", "gold"], n, p=[0.55, 0.30, 0.15])
    tenure_months = rng.integers(1, 60, n)
    monthly_charges = np.where(
        plan == "bronze", rng.normal(35, 5, n),
        np.where(plan == "silver", rng.normal(60, 8, n), rng.normal(95, 12, n)),
    ).clip(15, None)
    total_charges = monthly_charges * tenure_months * rng.uniform(0.85, 1.05, n)
    n_claims_90d = rng.poisson(1.2, n)
    n_support_tickets_90d = rng.poisson(0.8, n)
    days_since_last_login = rng.integers(0, 120, n)
    autopay = rng.choice([0, 1], n, p=[0.35, 0.65])

    # churn risk increases with: short tenure, high charges, many tickets, no autopay, long inactivity
    risk = (
        - 0.04 * tenure_months
        + 0.01 * monthly_charges
        + 0.25 * n_support_tickets_90d
        + 0.02 * days_since_last_login
        - 0.6 * autopay
        + rng.normal(0, 0.5, n)
    )
    proba = 1 / (1 + np.exp(-risk - 1.5))
    churn = (rng.uniform(0, 1, n) < proba).astype(int)

    return pd.DataFrame({
        "customer_id": [f"c_{i:06d}" for i in range(n)],
        "plan": plan,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "n_claims_90d": n_claims_90d,
        "n_support_tickets_90d": n_support_tickets_90d,
        "days_since_last_login": days_since_last_login,
        "autopay": autopay,
        "churn": churn,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--customers", type=int, default=5000)
    p.add_argument("--out", default="data/customers.parquet")
    args = p.parse_args()

    df = generate(args.customers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"n={len(df)}  churn_rate={df['churn'].mean():.3f}  → {args.out}")


if __name__ == "__main__":
    main()
