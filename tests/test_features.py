"""Smoke tests for feature engineering."""
from __future__ import annotations

import pandas as pd

from features.feature_engineering import FEATURE_COLS, build_features
from scripts.generate_data import generate


def test_features_present():
    raw = generate(50)
    feats = build_features(raw)
    for col in FEATURE_COLS:
        assert col in feats.columns, f"missing {col}"
    assert len(feats) == len(raw)


def test_no_nan_in_features():
    raw = generate(100)
    feats = build_features(raw)
    assert feats[FEATURE_COLS].isna().sum().sum() == 0
