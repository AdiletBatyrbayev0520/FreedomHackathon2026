"""
src/features/transaction_mix.py
================================
Task 4 — Transaction channel/category mix features.

Output columns per user (90d window):
  mcc_share_{essentials,travel,leisure,transfers,financial,ecommerce,health,education,telecom,other}
  share_of_online      — ePOS / total
  share_of_p2p         — P2P Credit / total
  unique_terminals_30d — count distinct terminal_type in 30d
  imputed_share        — fraction of txs with transaction_sum_was_missing=True
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from src.features.mcc_mapping import MACRO_CATEGORIES, map_mcc

logger = logging.getLogger(__name__)


def build_transaction_mix_features(
    transactions: pl.DataFrame,
    cutoff,
    window_days: int = 90,
) -> pl.DataFrame:
    """
    Build transaction mix features for all users as of cutoff.

    Uses all transactions (not just successful) for channel shares,
    but only successful ones for MCC mix.
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    window_start = cutoff - timedelta(days=window_days)
    window_start_30d = cutoff - timedelta(days=30)

    logger.info("Building transaction mix features for cutoff=%s (window=%dd)", cutoff, window_days)

    # Filter to window
    tx = transactions.filter(
        (pl.col("transaction_date") >= pl.lit(window_start).cast(pl.Date))
        & (pl.col("transaction_date") < pl.lit(cutoff).cast(pl.Date))
    )

    # User base
    user_base = transactions["customer_id"].unique().to_frame()

    # -----------------------------------------------------------------------
    # MCC shares (90d, successful txs only)
    # -----------------------------------------------------------------------
    tx_success = tx.filter(pl.col("transaction_status_clean") == "success")
    tx_with_macro = map_mcc(tx_success)

    # Count per user per macro category
    mcc_counts = (
        tx_with_macro
        .group_by(["customer_id", "mcc_macro"])
        .agg(pl.len().alias("cnt"))
    )

    # Total per user
    user_total = (
        mcc_counts.group_by("customer_id")
        .agg(pl.col("cnt").sum().alias("total_cnt"))
    )

    # Pivot to wide format
    mcc_wide = mcc_counts.join(user_total, on="customer_id")

    # Build share columns using explicit pivoting
    mcc_shares = {}
    for cat in MACRO_CATEGORIES:
        cat_counts = mcc_wide.filter(pl.col("mcc_macro") == cat).select(
            ["customer_id", "cnt", "total_cnt"]
        ).with_columns(
            (pl.col("cnt") / pl.col("total_cnt")).alias(f"mcc_share_{cat}")
        ).select(["customer_id", f"mcc_share_{cat}"])
        mcc_shares[cat] = cat_counts

    result = user_base
    for cat in MACRO_CATEGORIES:
        result = result.join(mcc_shares[cat], on="customer_id", how="left").with_columns(
            pl.col(f"mcc_share_{cat}").fill_null(0.0)
        )

    # Normalise so shares sum to 1.0 per user (handle rounding)
    share_cols = [f"mcc_share_{c}" for c in MACRO_CATEGORIES]
    result = result.with_columns(
        pl.sum_horizontal(*[pl.col(c) for c in share_cols]).alias("_share_sum")
    )
    for cat in MACRO_CATEGORIES:
        result = result.with_columns(
            pl.when(pl.col("_share_sum") > 0)
            .then(pl.col(f"mcc_share_{cat}") / pl.col("_share_sum"))
            .otherwise(pl.lit(None).cast(pl.Float64))
            .alias(f"mcc_share_{cat}")
        )
    result = result.drop("_share_sum")

    # -----------------------------------------------------------------------
    # Online share (ePOS) — all txs in window
    # -----------------------------------------------------------------------
    terminal_agg = (
        tx.group_by("customer_id")
        .agg(
            pl.len().alias("tx_total"),
            (pl.col("terminal_type").str.to_lowercase() == "epos")
            .cast(pl.Int32)
            .sum()
            .alias("epos_cnt"),
            (pl.col("operation_type").str.to_lowercase() == "p2p credit")
            .cast(pl.Int32)
            .sum()
            .alias("p2p_cnt"),
        )
    )
    terminal_agg = terminal_agg.with_columns(
        (pl.col("epos_cnt") / pl.col("tx_total")).alias("share_of_online"),
        (pl.col("p2p_cnt") / pl.col("tx_total")).alias("share_of_p2p"),
    )
    result = result.join(
        terminal_agg.select(["customer_id", "share_of_online", "share_of_p2p"]),
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------------------------
    # Unique terminals 30d
    # -----------------------------------------------------------------------
    tx_30 = transactions.filter(
        (pl.col("transaction_date") >= pl.lit(window_start_30d).cast(pl.Date))
        & (pl.col("transaction_date") < pl.lit(cutoff).cast(pl.Date))
    )
    unique_term = (
        tx_30.group_by("customer_id")
        .agg(pl.col("terminal_type").n_unique().alias("unique_terminals_30d"))
    )
    result = result.join(unique_term, on="customer_id", how="left").with_columns(
        pl.col("unique_terminals_30d").fill_null(0)
    )

    # -----------------------------------------------------------------------
    # Imputed share
    # -----------------------------------------------------------------------
    imputed_agg = (
        tx.group_by("customer_id")
        .agg(
            pl.len().alias("tx_cnt"),
            pl.col("transaction_sum_was_missing").cast(pl.Int32).sum().alias("imputed_cnt"),
        )
        .with_columns(
            (pl.col("imputed_cnt") / pl.col("tx_cnt")).alias("imputed_share")
        )
    )
    result = result.join(
        imputed_agg.select(["customer_id", "imputed_share"]),
        on="customer_id",
        how="left",
    ).with_columns(pl.col("imputed_share").fill_null(0.0))

    logger.info(
        "Transaction mix features built: %d users, %d columns",
        result.height, result.width,
    )
    return result
