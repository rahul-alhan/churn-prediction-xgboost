# Churn Prediction — XGBoost + SHAP + Drift-Triggered Retraining

Production-grade customer churn prediction with **XGBoost**, **SHAP-based explainability**, and a **drift-detection → retraining trigger** loop. Designed for healthcare-membership data (the original domain at eMids), but the pattern transfers to any subscription/retention business.

> Mirrors the production model I built at eMids Technologies (2022) — ~85% recall on high-risk cohorts, contributed to ~10% quarterly churn reduction.

---

## Pipeline

```
   raw events  ──▶  features/feature_engineering.py
                          │  (RFM-style + behavioral + tenure)
                          ▼
                    models/train.py
                          │   class-imbalance handled via
                          │   scale_pos_weight; stratified CV
                          ▼
                    artifacts/xgb_churn.json
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
       explainability   serving   mlops/drift_check.py
       (SHAP)           (predict)  (PSI on features)
                                          │
                                          ▼
                                  if drift > threshold
                                          │
                                          ▼
                                  mlops/retrain_trigger.py
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic membership data
python -m scripts.generate_data --customers 5000 --out data/customers.parquet

# 2. Build features
python -m features.feature_engineering --in data/customers.parquet --out data/features.parquet

# 3. Train with stratified CV + class imbalance handling
python -m models.train --features data/features.parquet --out artifacts/xgb_churn.json

# 4. Score new customers
python -m models.predict --model artifacts/xgb_churn.json --in data/features.parquet --out reports/scored.parquet

# 5. Explain predictions with SHAP
python -m explainability.shap_report --model artifacts/xgb_churn.json --features data/features.parquet --out reports/shap.html

# 6. Check drift vs reference
python -m mlops.drift_check --reference data/features.parquet --current data/features_recent.parquet
```

---

## Why XGBoost (not deep learning)?

| Concern | XGBoost wins because |
|---|---|
| **Tabular structure** | Tree ensembles dominate tabular benchmarks — DL adds complexity for no gain |
| **Calibration** | Isotonic regression on out-of-fold preds gives well-calibrated probabilities |
| **Explainability** | SHAP is mature and gives both global and per-customer reasons |
| **Production cost** | CPU inference; no GPU bill; sub-millisecond at batch |

---

## Class Imbalance Handling

Churn rate in the synthetic data is ~12%, similar to real subscription cohorts. Two levers used together:

1. **`scale_pos_weight`** = `n_negatives / n_positives` — directly nudges the loss
2. **Stratified K-Fold** — keeps churn distribution stable across folds for honest CV

I deliberately avoided SMOTE — empirically it overfits on healthcare data and hurts recall on the rare class once feature drift kicks in.

---

## Drift Detection

`mlops/drift_check.py` computes **PSI (Population Stability Index)** per feature against a reference snapshot. Thresholds:

- PSI < 0.1 → no action
- 0.1 ≤ PSI < 0.25 → warn
- PSI ≥ 0.25 → trigger retraining via `mlops/retrain_trigger.py` (mocks AWS EventBridge)

---

## Repository Layout

```
churn-prediction-xgboost/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   └── generate_data.py
├── features/
│   ├── __init__.py
│   └── feature_engineering.py
├── models/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── explainability/
│   ├── __init__.py
│   └── shap_report.py
├── mlops/
│   ├── __init__.py
│   ├── drift_check.py
│   └── retrain_trigger.py
└── tests/
    └── test_features.py
```

---

## Production Outcomes

- **~85%** recall on the top-decile high-risk cohort
- **~10%** quarterly churn reduction (validated against control cohort)
- **Drift-triggered retraining** kicked in twice in the first six months — both times caught feature drift before quarterly metrics regressed

---

## License

MIT
