"""
src/features/demo.py
=====================
Task 5 — Demographic features from users + acquisition.

Output columns per user:
  customer_age      (from users, already cleaned)
  gender            (M/F, categorical)
  lifetime_days     = (cutoff − reg_date).days  — null if reg_date > cutoff
  city              (categorical, for CatBoost cat_features)
  channel           (from acquisition.secondary_category_filled, categorical)
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

logger = logging.getLogger(__name__)


def build_demo_features(
    users: pl.DataFrame,
    acquisition: pl.DataFrame,
    cutoff,
) -> pl.DataFrame:
    """
    Build demographic features for all users.

    Parameters
    ----------
    users       : DataFrame with customer_id, customer_age, gender, reg_date, city
    acquisition : DataFrame with customer_id, secondary_category_filled (= channel)
    cutoff      : date or str

    Returns
    -------
    pl.DataFrame with one row per user from users table
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    logger.info("Building demo features for cutoff=%s", cutoff)

    # Start from users table
    result = users.select(
        ["customer_id", "customer_age", "gender", "reg_date", "city"]
    )

    # lifetime_days = cutoff − reg_date, null if reg_date > cutoff
    result = result.with_columns(
        pl.when(
            pl.col("reg_date").is_not_null()
            & (pl.col("reg_date") <= pl.lit(cutoff).cast(pl.Utf8).str.to_date(format="%Y-%m-%d"))
        )
        .then(
            (pl.lit(cutoff) - pl.col("reg_date")).dt.total_days()
        )
        .otherwise(pl.lit(None).cast(pl.Int64))
        .alias("lifetime_days")
    )

    # Sanity check
    neg_lifetime = result.filter(
        pl.col("lifetime_days").is_not_null() & (pl.col("lifetime_days") < 0)
    ).height
    if neg_lifetime > 0:
        logger.warning(
            "[demo] %d users have lifetime_days < 0 (reg_date > cutoff). "
            "Setting to null for these users.",
            neg_lifetime,
        )

    # Join channel from acquisition (deduped — one row per customer_id)
    acq = acquisition.select(
        ["customer_id", "secondary_category_filled"]
    ).rename({"secondary_category_filled": "channel"})
    # Dedup acquisition (should already be clean, but just in case)
    acq = acq.unique(subset=["customer_id"], keep="first")

    result = result.join(acq, on="customer_id", how="left")

    # Drop reg_date (already encoded as lifetime_days)
    result = result.drop("reg_date")

    logger.info(
        "Demo features built: %d users, %d columns. "
        "channel null rate: %.1f%%",
        result.height,
        result.width,
        result["channel"].null_count() / result.height * 100,
    )
    return result
