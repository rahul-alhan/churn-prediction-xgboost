"""PSI-based feature drift detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.feature_engineering import FEATURE_COLS

PSI_WARN = 0.10
PSI_TRIGGER = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_p = np.clip(ref_hist / max(1, ref_hist.sum()), 1e-6, None)
    cur_p = np.clip(cur_hist / max(1, cur_hist.sum()), 1e-6, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def check(reference_path: str, current_path: str) -> dict:
    ref = pd.read_parquet(reference_path)
    cur = pd.read_parquet(current_path)

    results = {}
    n_trigger = 0
    for f in FEATURE_COLS:
        if f not in ref or f not in cur:
            continue
        v = psi(ref[f].dropna().to_numpy(), cur[f].dropna().to_numpy())
        status = "ok" if v < PSI_WARN else ("warn" if v < PSI_TRIGGER else "trigger")
        if status == "trigger":
            n_trigger += 1
        results[f] = {"psi": round(v, 4), "status": status}

    return {
        "n_features": len(results),
        "n_trigger": n_trigger,
        "trigger_retrain": n_trigger > 0,
        "thresholds": {"warn": PSI_WARN, "trigger": PSI_TRIGGER},
        "features": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--out", default="reports/drift.json")
    args = p.parse_args()
    rep = check(args.reference, args.current)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
