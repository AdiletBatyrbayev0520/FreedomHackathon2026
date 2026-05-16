"""
src/features/rfm_temporal.py
==============================
Task 3 — RFM features + temporal slopes and delta MoM.

Uses ONLY transaction_status_clean == 'success' for monetary/frequency.
Failed transactions tracked separately (see build_matrix.py for merge).

Output columns per user:
  recency_tx_{7,14,30}d       — days since last successful tx in window
  recency_event_{7,14,30}d    — days since last event in window
  frequency_tx_{7,14,30,90}d  — count of successful txs in window
  monetary_sum_{7,30,90}d     — sum of transaction_sum
  monetary_median_30d
  frequency_slope_4w          — slope of weekly counts (linear trend)
  monetary_delta_mom          — MoM change in sum
  active_days_{7,30}d         — distinct days with ≥1 tx
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

WINDOWS = [7, 14, 30, 90]


def build_rfm_features(
    transactions: pl.DataFrame,
    events: pl.DataFrame,
    cutoff,
) -> pl.DataFrame:
    """
    Build RFM + temporal features for all users as of cutoff date.

    Parameters
    ----------
    transactions : DataFrame with columns customer_id, transaction_date,
                   transaction_sum, transaction_status_clean
    events       : DataFrame with columns customer_id, started_at
    cutoff       : date or str — observation point (features use data < cutoff)

    Returns
    -------
    pl.DataFrame with one row per customer_id
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    logger.info("Building RFM features for cutoff=%s", cutoff)

    # -----------------------------------------------------------------------
    # Filter to successful transactions strictly before cutoff
    # -----------------------------------------------------------------------
    tx = transactions.filter(
        (pl.col("transaction_status_clean") == "success")
        & (pl.col("transaction_date") < pl.lit(cutoff).cast(pl.Date))
    )

    # -----------------------------------------------------------------------
    # Build recency / frequency / monetary per window
    # -----------------------------------------------------------------------
    user_base = transactions["customer_id"].unique().to_frame()

    feature_dfs = [user_base]

    for window in WINDOWS:
        window_start = cutoff - timedelta(days=window)
        tx_w = tx.filter(
            pl.col("transaction_date") >= pl.lit(window_start).cast(pl.Date)
        )

        agg_exprs = [
            pl.col("transaction_id").count().alias(f"frequency_tx_{window}d"),
            pl.col("transaction_sum").sum().fill_null(0).alias(f"monetary_sum_{window}d"),
            pl.col("transaction_date").n_unique().alias(f"active_days_{window}d"),
        ]
        if window == 30:
            agg_exprs.append(
                pl.col("transaction_sum").median().alias("monetary_median_30d")
            )

        agg_w = tx_w.group_by("customer_id").agg(agg_exprs)

        # Recency: days since last tx (relative to cutoff)
        last_tx = tx_w.group_by("customer_id").agg(
            pl.col("transaction_date").max().alias("last_tx_date")
        )
        recency = last_tx.with_columns(
            (pl.lit(cutoff) - pl.col("last_tx_date").cast(pl.Date))
            .dt.total_days()
            .alias(f"recency_tx_{window}d")
        ).select(["customer_id", f"recency_tx_{window}d"])

        merged = agg_w.join(recency, on="customer_id", how="left")
        feature_dfs.append(merged)

    # Combine all window features
    result = feature_dfs[0]
    for df in feature_dfs[1:]:
        result = result.join(df, on="customer_id", how="left")

    # Fill frequency/active_days nulls with 0 (no transactions = 0, not null)
    freq_cols = [c for c in result.columns if c.startswith("frequency_") or c.startswith("active_days_")]
    monetary_cols = [c for c in result.columns if c.startswith("monetary_")]
    result = result.with_columns(
        [pl.col(c).fill_null(0) for c in freq_cols + monetary_cols]
    )

    # -----------------------------------------------------------------------
    # Event recency features
    # -----------------------------------------------------------------------
    ev = events.filter(
        pl.col("started_at") < pl.lit(cutoff).cast(pl.Date).cast(pl.Datetime)
    )
    for window in [7, 14, 30]:
        ev_w = ev.filter(
            pl.col("started_at") >= pl.lit(cutoff - timedelta(days=window)).cast(pl.Date).cast(pl.Datetime)
        )
        last_ev = ev_w.group_by("customer_id").agg(
            pl.col("started_at").max().alias("last_ev")
        )
        # Convert Datetime to Date for subtraction
        recency_ev = last_ev.with_columns(
            (
                pl.lit(cutoff).cast(pl.Utf8).str.to_date(format="%Y-%m-%d")
                - pl.col("last_ev").cast(pl.Date)
            ).dt.total_days().alias(f"recency_event_{window}d")
        ).select(["customer_id", f"recency_event_{window}d"])

        result = result.join(recency_ev, on="customer_id", how="left")

    # -----------------------------------------------------------------------
    # frequency_slope_4w — linear slope of weekly tx counts over 4 weeks
    # -----------------------------------------------------------------------
    slope_series = _compute_frequency_slope_4w(tx, cutoff)
    result = result.join(slope_series, on="customer_id", how="left")

    # -----------------------------------------------------------------------
    # monetary_delta_mom — MoM change
    # -----------------------------------------------------------------------
    delta_mom = _compute_monetary_delta_mom(tx, cutoff)
    result = result.join(delta_mom, on="customer_id", how="left")

    logger.info(
        "RFM features built: %d users, %d columns", result.height, result.width
    )
    return result


def _compute_frequency_slope_4w(tx: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """
    Compute linear slope of weekly transaction counts over 4 weeks before cutoff.
    Returns pl.DataFrame(customer_id, frequency_slope_4w).
    """
    weeks = []
    for w in range(4):
        wend = cutoff - timedelta(days=w * 7)
        wstart = wend - timedelta(days=7)
        tx_w = tx.filter(
            (pl.col("transaction_date") >= pl.lit(wstart).cast(pl.Date))
            & (pl.col("transaction_date") < pl.lit(wend).cast(pl.Date))
        ).group_by("customer_id").agg(
            pl.col("transaction_id").count().alias(f"cnt_w{w}")
        )
        weeks.append((w, tx_w))

    # Merge all weeks
    all_users = tx["customer_id"].unique().to_frame()
    for w_idx, wdf in weeks:
        all_users = all_users.join(wdf, on="customer_id", how="left").with_columns(
            pl.col(f"cnt_w{w_idx}").fill_null(0)
        )

    # Compute slope: week 3 = oldest, week 0 = most recent → x = [−3, −2, −1, 0]
    def slope_fn(row_dict: dict) -> float:
        y = np.array([row_dict.get(f"cnt_w{w}", 0) or 0 for w in range(3, -1, -1)], dtype=float)
        x = np.arange(len(y), dtype=float)
        if np.std(y) < 1e-9:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    slopes = []
    for row in all_users.to_dicts():
        slopes.append({"customer_id": row["customer_id"], "frequency_slope_4w": slope_fn(row)})

    return pl.DataFrame(slopes)


def _compute_monetary_delta_mom(tx: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """
    monetary_delta_mom = (sum_30d − sum_60d_to_30d) / (sum_60d_to_30d + 1e-9)
    """
    cutoff_30 = cutoff - timedelta(days=30)
    cutoff_60 = cutoff - timedelta(days=60)

    sum_recent = (
        tx.filter(
            (pl.col("transaction_date") >= pl.lit(cutoff_30).cast(pl.Date))
            & (pl.col("transaction_date") < pl.lit(cutoff).cast(pl.Date))
        )
        .group_by("customer_id")
        .agg(pl.col("transaction_sum").sum().alias("sum_30d"))
    )

    sum_prior = (
        tx.filter(
            (pl.col("transaction_date") >= pl.lit(cutoff_60).cast(pl.Date))
            & (pl.col("transaction_date") < pl.lit(cutoff_30).cast(pl.Date))
        )
        .group_by("customer_id")
        .agg(pl.col("transaction_sum").sum().alias("sum_60d_to_30d"))
    )

    delta = sum_recent.join(sum_prior, on="customer_id", how="full", coalesce=True).with_columns(
        pl.col("sum_30d").fill_null(0),
        pl.col("sum_60d_to_30d").fill_null(0),
    ).with_columns(
        (
            (pl.col("sum_30d") - pl.col("sum_60d_to_30d"))
            / (pl.col("sum_60d_to_30d") + 1e-9)
        ).alias("monetary_delta_mom")
    ).select(["customer_id", "monetary_delta_mom"])

    return delta
