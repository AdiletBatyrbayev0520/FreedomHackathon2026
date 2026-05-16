"""Gather all pipeline metrics for business recommendations."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import polars as pl
import numpy as np

# NBA
nba = pl.read_parquet('data/final/nba_recommendations.parquet')
print("=== NBA ===")
print("Columns:", nba.columns)
print(nba.group_by('recommended_product').agg(pl.len()).sort('len', descending=True))

# Freedom scores
fs = pl.read_parquet('data/final/freedom_scores.parquet')
print("\n=== Freedom Scores ===")
print("Columns:", fs.columns, "Shape:", fs.shape)
vals = fs['freedom_score_pred'].to_numpy()
print(f"  mean={vals.mean():.4f}  median={np.median(vals):.4f}  std={vals.std():.4f}")
print(f"  min={vals.min():.4f}  max={vals.max():.4f}")
print(f"  zero_pct={(vals == 0).mean()*100:.1f}%")
print(f"  >0 pct={(vals > 0).mean()*100:.1f}%")
q = np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9])
print(f"  quantiles [10,25,50,75,90]: {q}")

# Churn
ch = pl.read_parquet('data/final/churn_predictions.parquet')
print("\n=== Churn ===")
cvals = ch['churn_prob'].to_numpy()
print(f"  mean={cvals.mean():.4f}  median={np.median(cvals):.4f}")
print(f"  high_churn (>0.7): {(cvals > 0.7).sum()} ({(cvals > 0.7).mean()*100:.1f}%)")
print(f"  low_churn (<0.3): {(cvals < 0.3).sum()} ({(cvals < 0.3).mean()*100:.1f}%)")

# Channel
cs = pl.read_csv('reports/channel_summary.csv')
print("\n=== Channel Summary ===")
print(cs)

# Labels
ld = pl.read_csv('reports/label_diagnostics.csv')
print("\n=== Label Diagnostics ===")
print(ld)

# Segments
seg = pl.read_csv('reports/segments_profile.csv')
print("\n=== Segment Profiles ===")
print(seg)

# SHAP
shap_imp = pl.read_csv('reports/shap_importance.csv')
print("\n=== Top-10 SHAP Features ===")
print(shap_imp.head(10))

# Model comparison
mc = pl.read_csv('reports/model_comparison.csv')
print("\n=== Model Comparison ===")
print(mc)

# Compute business metrics from the TZ
print("\n=== COMPUTING BUSINESS METRICS ===")

# Join freedom scores and churn for action matrix
combined = fs.join(ch, on='customer_id', how='inner')
combined = combined.join(nba.select(['customer_id', 'recommended_product', 'propensity_score']), on='customer_id', how='left')

# Action matrix: churn_prob vs freedom_score quartiles
fs_median = np.median(vals)
churn_threshold = 0.7

vip = combined.filter((pl.col('churn_prob') <= churn_threshold) & (pl.col('freedom_score_pred') >= fs_median))
persuadable = combined.filter((pl.col('churn_prob') > churn_threshold) & (pl.col('freedom_score_pred') >= fs_median))
inactive = combined.filter((pl.col('churn_prob') > churn_threshold) & (pl.col('freedom_score_pred') < fs_median))
standard = combined.filter((pl.col('churn_prob') <= churn_threshold) & (pl.col('freedom_score_pred') < fs_median))

print(f"\nAction Matrix:")
print(f"  VIP:         {vip.height:>7,} ({vip.height/combined.height*100:.1f}%)")
print(f"  Persuadable: {persuadable.height:>7,} ({persuadable.height/combined.height*100:.1f}%)")
print(f"  Inactive:    {inactive.height:>7,} ({inactive.height/combined.height*100:.1f}%)")
print(f"  Standard:    {standard.height:>7,} ({standard.height/combined.height*100:.1f}%)")

# Retention value at risk
p_vals = persuadable['freedom_score_pred'].to_numpy()
p_churn = persuadable['churn_prob'].to_numpy()
retention_value_at_risk = float((p_churn * p_vals).sum())
print(f"\n  Retention value at risk (Persuadables): {retention_value_at_risk:.2f}")
print(f"  30% retention uplift potential: {retention_value_at_risk * 0.3:.2f}")

# Conversion uplift
prop_vals = combined['propensity_score'].to_numpy()
q90 = np.quantile(prop_vals, 0.9)
top_decile = prop_vals[prop_vals >= q90]
np.random.seed(42)
random_sample = np.random.choice(prop_vals, size=len(top_decile), replace=False)
conversion_uplift = top_decile.mean() / max(random_sample.mean(), 1e-6)
print(f"\n  Top-decile propensity mean: {top_decile.mean():.4f}")
print(f"  Random sample mean: {random_sample.mean():.4f}")
print(f"  Conversion uplift: {conversion_uplift:.2f}x")

# Budget efficiency
prop_sorted = np.sort(prop_vals)[::-1]
top_20_pct = prop_sorted[:int(0.2 * len(prop_sorted))]
total_ev = prop_vals.sum()
budget_concentration = top_20_pct.sum() / max(total_ev, 1e-6)
print(f"\n  Budget concentration (top-20% by EV): {budget_concentration*100:.1f}%")

# Build business_metrics.json
biz_metrics = {
    "total_users": int(combined.height),
    "action_matrix": {
        "VIP": {"count": int(vip.height), "pct": round(vip.height/combined.height*100, 1)},
        "Persuadable": {"count": int(persuadable.height), "pct": round(persuadable.height/combined.height*100, 1)},
        "Inactive": {"count": int(inactive.height), "pct": round(inactive.height/combined.height*100, 1)},
        "Standard": {"count": int(standard.height), "pct": round(standard.height/combined.height*100, 1)},
    },
    "retention_value_at_risk": round(retention_value_at_risk, 2),
    "retention_uplift_30pct": round(retention_value_at_risk * 0.3, 2),
    "conversion_uplift_top_decile": round(float(conversion_uplift), 2),
    "budget_concentration_top20pct": round(float(budget_concentration * 100), 1),
    "freedom_score_stats": {
        "mean": round(float(vals.mean()), 4),
        "median": round(float(np.median(vals)), 4),
        "zero_pct": round(float((vals == 0).mean() * 100), 1),
    },
    "churn_stats": {
        "mean_prob": round(float(cvals.mean()), 4),
        "high_churn_pct": round(float((cvals > 0.7).mean() * 100), 1),
    },
}

with open('reports/business_metrics.json', 'w') as f:
    json.dump(biz_metrics, f, indent=2)
print(f"\nSaved reports/business_metrics.json")
print(json.dumps(biz_metrics, indent=2))
