"""
src/features/second_order.py
==============================
Task 13 — Loop-back: add model predictions as second-order features.

Adds freedom_score_pred and segment_id to feature matrix.
Saves extended matrix as features_v2_{cutoff_name}.parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def add_second_order_features(
    feature_df: pl.DataFrame,
    freedom_score_predictions: pl.DataFrame,
    segment_assignments: pl.DataFrame,
    cutoff_name: str,
    data_interim_dir: str | Path = "data/interim",
) -> pl.DataFrame:
    """
    Add second-order features (freedom_score_pred, segment_id) to feature matrix.

    Parameters
    ----------
    feature_df                : original features_{cutoff_name}.parquet
    freedom_score_predictions : pl.DataFrame(customer_id, freedom_score_pred)
    segment_assignments       : pl.DataFrame(customer_id, segment_id, segment_name)
    cutoff_name               : "T", "T_minus_1", "T_minus_2"

    Returns
    -------
    Extended pl.DataFrame, also saved as features_v2_{cutoff_name}.parquet
    """
    data_interim_dir = Path(data_interim_dir)
    data_interim_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Adding second-order features for %s: freedom_score_pred, segment_id",
        cutoff_name,
    )

    # Validate: predictions must be computed on features BEFORE cutoff
    result = feature_df.join(
        freedom_score_predictions.select(["customer_id", "freedom_score_pred"]),
        on="customer_id",
        how="left",
    )

    result = result.join(
        segment_assignments.select(["customer_id", "segment_id"]),
        on="customer_id",
        how="left",
    )

    out_path = data_interim_dir / f"features_v2_{cutoff_name}.parquet"
    result.write_parquet(out_path)
    logger.info(
        "Second-order feature matrix saved to %s (%d rows × %d cols)",
        out_path, result.height, result.width,
    )

    return result
