"""
src/features/partners.py
=========================
Task 5 — Partner purchases features.

Output columns per user (90d window):
  partner_purchases_count_90d
  partner_purchases_sum_90d
  cashback_earned_90d
  avg_cashback_rate
  unique_partners_90d
  partner_share_arbuz
  partner_share_ticketon
  partner_share_train_tickets
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

logger = logging.getLogger(__name__)

TOP_PARTNERS = ["arbuz", "ticketon", "train_tickets"]


def build_partner_features(
    partner_purchases: pl.DataFrame,
    cutoff,
    window_days: int = 90,
) -> pl.DataFrame:
    """
    Build partner purchase features for all users.

    Parameters
    ----------
    partner_purchases : DataFrame with customer_id, purchase_date,
                        purchase_amount_real, cashback_amount_real,
                        cashback_rate, app_name_normalized
    cutoff            : date or str
    window_days       : look-back window (default 90d)
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    window_start = cutoff - timedelta(days=window_days)
    logger.info("Building partner features for cutoff=%s (window=%dd)", cutoff, window_days)

    pp = partner_purchases.filter(
        (pl.col("purchase_date") >= pl.lit(window_start).cast(pl.Date))
        & (pl.col("purchase_date") < pl.lit(cutoff).cast(pl.Date))
    )

    user_base = partner_purchases["customer_id"].unique().to_frame()

    # -----------------------------------------------------------------------
    # Basic aggregates
    # -----------------------------------------------------------------------
    basic_agg = (
        pp.group_by("customer_id")
        .agg(
            pl.len().alias("partner_purchases_count_90d"),
            pl.col("purchase_amount_real").sum().alias("partner_purchases_sum_90d"),
            pl.col("cashback_amount_real").sum().alias("cashback_earned_90d"),
            pl.col("cashback_rate").mean().alias("avg_cashback_rate"),
            pl.col("app_name_normalized").n_unique().alias("unique_partners_90d"),
        )
    )

    result = user_base.join(basic_agg, on="customer_id", how="left").with_columns(
        pl.col("partner_purchases_count_90d").fill_null(0),
        pl.col("partner_purchases_sum_90d").fill_null(0.0),
        pl.col("cashback_earned_90d").fill_null(0.0),
        pl.col("unique_partners_90d").fill_null(0),
    )

    # -----------------------------------------------------------------------
    # Per-partner shares
    # -----------------------------------------------------------------------
    # Total purchases per user
    user_total = pp.group_by("customer_id").agg(
        pl.len().alias("pp_total")
    )

    for partner in TOP_PARTNERS:
        partner_pp = pp.filter(
            pl.col("app_name_normalized").str.to_lowercase() == partner.lower()
        ).group_by("customer_id").agg(
            pl.len().alias(f"pp_{partner}_cnt")
        )

        share_col = f"partner_share_{partner}"
        combined = partner_pp.join(user_total, on="customer_id", how="left")
        combined = combined.with_columns(
            (pl.col(f"pp_{partner}_cnt") / pl.col("pp_total")).alias(share_col)
        ).select(["customer_id", share_col])

        result = result.join(combined, on="customer_id", how="left").with_columns(
            pl.col(share_col).fill_null(0.0)
        )

    logger.info(
        "Partner features built: %d users, %d columns",
        result.height, result.width,
    )
    return result
