"""
src/business/nba.py
====================
Task 14 — Next Best Action recommendation engine.

Business rules:
  - Exclude products user already has (has_{product}=True in features)
  - Users < 18 years old: no credit products (loan)
  - Users with churn_prob > 0.7: retention priority, no cross-sell
  - Top-1 recommendation = highest propensity for eligible products

Outputs:
  data/final/nba_recommendations.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Products classified as credit (restricted for minors)
CREDIT_PRODUCTS = {"loan"}

# Product display names for reporting
PRODUCT_DISPLAY_NAMES = {
    "card": "Дебетовая карта",
    "freedom_rating": "Freedom Rating",
    "frhc": "FRHC",
    "deposit": "Депозит",
    "loan": "Кредит",
    "liveness": "Биометрия",
}


def build_nba(
    feature_df: pl.DataFrame,
    propensity_scores: dict[str, pl.DataFrame],
    churn_predictions: pl.DataFrame,
    shap_explanations: pl.DataFrame | None = None,
    data_final_dir: str | Path = "data/final",
) -> pl.DataFrame:
    """
    Build Next Best Action recommendations for all users.

    Parameters
    ----------
    feature_df         : DataFrame with customer_id, has_{product} flags, customer_age
    propensity_scores  : dict {product_name: DataFrame(customer_id, propensity_{product})}
    churn_predictions  : DataFrame(customer_id, churn_prob)
    shap_explanations  : optional per-user top SHAP drivers

    Returns
    -------
    pl.DataFrame(customer_id, recommended_product, propensity_score,
                 expected_value, reason, churn_flagged)
    """
    data_final_dir = Path(data_final_dir)
    data_final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building NBA recommendations for %d users", feature_df.height)

    # Start from users
    result = feature_df.select(["customer_id", "customer_age"]).clone()

    # Join churn prob
    result = result.join(
        churn_predictions.select(["customer_id", "churn_prob"]),
        on="customer_id",
        how="left",
    ).with_columns(pl.col("churn_prob").fill_null(0.0))

    # Flag high-churn users
    result = result.with_columns(
        (pl.col("churn_prob") > 0.7).alias("churn_flagged")
    )

    # Join all propensity scores
    for product, prop_df in propensity_scores.items():
        prop_col = f"propensity_{product}"
        if prop_col not in prop_df.columns:
            continue
        result = result.join(
            prop_df.select(["customer_id", prop_col]),
            on="customer_id",
            how="left",
        ).with_columns(pl.col(prop_col).fill_null(0.0))

    # Join has_product flags from features
    product_flags = [c for c in feature_df.columns if c.startswith("has_")]
    if product_flags:
        result = result.join(
            feature_df.select(["customer_id"] + product_flags),
            on="customer_id",
            how="left",
        )

    # -----------------------------------------------------------------------
    # Apply business rules and pick top recommendation
    # -----------------------------------------------------------------------
    all_products = list(propensity_scores.keys())

    def pick_recommendation(row: dict) -> dict:
        age = row.get("customer_age", 18) or 18
        churn_flagged = row.get("churn_flagged", False)
        best_product = None
        best_score = -1.0

        for product in all_products:
            prop_col = f"propensity_{product}"
            has_col = f"has_{product}"

            score = row.get(prop_col, 0.0) or 0.0

            # Rule: skip if user already has product
            if row.get(has_col, False):
                continue

            # Rule: no credit products for minors
            if age < 18 and product in CREDIT_PRODUCTS:
                continue

            # Rule: no cross-sell for high-churn users
            if churn_flagged:
                continue

            if score > best_score:
                best_score = score
                best_product = product

        return {
            "recommended_product": best_product or "retention",
            "propensity_score": best_score if best_product else 0.0,
        }

    rows = result.to_dicts()
    recs = [pick_recommendation(r) for r in rows]

    rec_df = pl.DataFrame(recs)
    result = pl.concat(
        [result.select(["customer_id", "churn_flagged", "churn_prob"]), rec_df],
        how="horizontal",
    )

    # Add display reason
    result = result.with_columns(
        pl.when(pl.col("churn_flagged"))
        .then(pl.lit("Retention priority: high churn risk"))
        .when(pl.col("recommended_product") == "retention")
        .then(pl.lit("No eligible products found"))
        .otherwise(
            pl.lit("Propensity model: top eligible product")
        )
        .alias("reason")
    )

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    # Distribution should not be degenerate
    rec_counts = result["recommended_product"].value_counts().sort("count", descending=True)
    logger.info("NBA recommendation distribution:\n%s", rec_counts)

    top_product = rec_counts["recommended_product"][0]
    top_pct = rec_counts["count"][0] / result.height * 100
    if top_pct > 80:
        logger.warning(
            "[FLAG] %.1f%% of users get the same recommendation ('%s'). "
            "Distribution is degenerate — check propensity model scores.",
            top_pct, top_product,
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = data_final_dir / "nba_recommendations.parquet"
    result.write_parquet(out_path)
    logger.info("NBA recommendations saved to %s (%d users)", out_path, result.height)

    return result
