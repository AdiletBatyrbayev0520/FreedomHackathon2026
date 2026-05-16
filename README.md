# FreedomProfile Analytics Pipeline v2

ML pipeline for FreedomBank SuperApp: churn prediction, Freedom Score regression, product propensity, SHAP-based segmentation, and Next Best Action.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# Diagnostics only (Task 0, fast)
python run_pipeline.py --dry-run

# Skip model training (Tasks 0–7 only)
python run_pipeline.py --skip-models
```

## Directory Layout

```
data/
├── raw/           # Original CSVs (READ ONLY — never modify)
├── processed/     # Cleaned CSVs (ground truth input)
├── interim/       # Pipeline artifacts: features_*.parquet, cutoffs.json
└── final/         # Business output: freedom_scores, segments, nba_recommendations

src/
├── data_loading.py         # Task 1: load all 5 datasets
├── diagnostics/            # Task 0: ID overlap + temporal checks
├── features/               # Tasks 2–7: feature engineering
│   ├── time_split.py       # Cutoff strategy
│   ├── rfm_temporal.py     # RFM + slope features
│   ├── mcc_mapping.py      # MCC → macro-category dict
│   ├── transaction_mix.py  # MCC shares, online/P2P
│   ├── engagement.py       # App events features
│   ├── products.py         # Product activations via process_code
│   ├── demo.py             # Age, gender, city, channel
│   ├── partners.py         # Partner purchase features
│   ├── build_matrix.py     # Assemble feature matrix
│   ├── preprocess.py       # log1p, RobustScaler, imputation
│   ├── labels.py           # Churn, Freedom Score, propensity targets
│   └── second_order.py     # Loop-back: pred + segment as features
├── models/                 # Tasks 8–10: CatBoost models
├── interpretation/         # Tasks 11–12: SHAP + UMAP+GMM segmentation
└── business/               # Tasks 14–15: NBA + Channel LTV

models/              # Trained .cbm files (gitignored)
reports/             # Metrics JSONs, CSVs, figures
run_pipeline.py      # Single-button pipeline orchestrator
```

## Key Design Decisions

### Why events and users may not overlap (Task 0)
Events `customer_id` range (~900k–1M) doesn't match users (~466k, 1.9M, 3M+). Pipeline runs Task 0 diagnostics first and stops if overlap < 50%. See `reports/id_overlap.csv` for the actual numbers.

### Freedom Score target construction (Task 7)
`0.7 × log1p(revenue_normalized) + 0.3 × activity_zscore_clipped` — NOT raw predicted LTV. This avoids the 50%-users-at-zero problem caused by narrow label windows.

### Why no ROMI (Task 15)
Previous iteration showed ROMI=5681% for organic channel. This was based on a synthetic CAC=300. We report honest metrics: median Freedom Score, retention curves, time-to-first-transaction.

### Segmentation approach (Task 12)
UMAP+GMM on top-20 SHAP features (not on raw predicted score). This produces driver-based segments ("high-activity", "product-activators") rather than arbitrary percentile buckets.

## Running Tests

```bash
pytest tests/ -v
```

All tests use synthetic mini fixtures (5 users, ~20–200 rows) — no real data needed.

## Data Notes

See `reports/data_issues.md` for the full running log of data quality issues.

Key issues:
- `customer_id` was not unique in users — deduped by latest `reg_date`
- `transaction_sum` and `purchase_amount` had systematic shift of −101,667 (compensated)
- Events customer_ids may not overlap with cleaned users (run Task 0 to verify)
