"""
src/features/labels.py
=======================
Task 7 — Build target labels for all three ML tasks.

Labels:
  churn_label         — 1 if NO tx AND NO event in [cutoff, cutoff+window]
  freedom_score_target — composite: 70% log-revenue + 30% norm_activity
  propensity_{product} — 1 if product activation in [cutoff, cutoff+window]

Critical test: if >30% of users have freedom_score_target=0, there's a signal gap.
Reports: reports/label_diagnostics.csv
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def build_churn_label(
    transactions: pl.DataFrame,
    events: pl.DataFrame,
    cutoff,
    window_days: int = 30,
) -> pl.DataFrame:
    """
    Build churn label: 1 if user has no tx AND no event in [cutoff, cutoff+window].

    Returns pl.DataFrame(customer_id, churn_label)
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    label_end = cutoff + timedelta(days=window_days)
    logger.info(
        "Building churn label: window=[%s, %s]", cutoff, label_end
    )

    # Users with ≥1 successful tx in window
    active_tx = transactions.filter(
        (pl.col("transaction_status_clean") == "success")
        & (pl.col("transaction_date") >= pl.lit(cutoff).cast(pl.Date))
        & (pl.col("transaction_date") < pl.lit(label_end).cast(pl.Date))
    )["customer_id"].unique()

    # Users with ≥1 event in window
    active_ev = events.filter(
        (pl.col("started_at") >= pl.lit(cutoff).cast(pl.Date).cast(pl.Datetime))
        & (pl.col("started_at") < pl.lit(label_end).cast(pl.Date).cast(pl.Datetime))
    )["customer_id"].unique()

    all_active = set(active_tx.to_list()) | set(active_ev.to_list())
    all_users = set(transactions["customer_id"].unique().to_list())

    rows = [
        {"customer_id": uid, "churn_label": 0 if uid in all_active else 1}
        for uid in all_users
    ]

    label_df = pl.DataFrame(rows)
    pos_rate = label_df["churn_label"].mean()
    logger.info(
        "Churn label distribution: positive=%.1f%%  (window=%dd)",
        (pos_rate or 0) * 100, window_days,
    )
    if pos_rate is not None and (pos_rate > 0.9 or pos_rate < 0.05):
        logger.warning(
            "[FLAG] Churn positive rate=%.1f%% is outside expected [5%%, 40%%]. "
            "Check cutoff and window.",
            pos_rate * 100,
        )

    return label_df


def build_freedom_score_target(
    transactions: pl.DataFrame,
    partner_purchases: pl.DataFrame,
    cutoff,
    window_days: int = 90,
) -> pl.DataFrame:
    """
    Build Freedom Score target (regression):
      70% × log1p(revenue) + 30% × clipped_z_score(activity)

    revenue = sum(successful txs) + sum(partner purchases) in [cutoff, cutoff+window]

    Returns pl.DataFrame(customer_id, freedom_score_target)
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    label_end = cutoff + timedelta(days=window_days)
    logger.info(
        "Building Freedom Score target: window=[%s, %s]", cutoff, label_end
    )

    # Revenue from transactions
    tx_rev = (
        transactions.filter(
            (pl.col("transaction_status_clean") == "success")
            & (pl.col("transaction_date") >= pl.lit(cutoff).cast(pl.Date))
            & (pl.col("transaction_date") < pl.lit(label_end).cast(pl.Date))
        )
        .group_by("customer_id")
        .agg(
            pl.col("transaction_sum").sum().alias("tx_revenue"),
            pl.len().alias("tx_count"),
        )
    )

    # Revenue from partner purchases
    pp_rev = (
        partner_purchases.filter(
            (pl.col("purchase_date") >= pl.lit(cutoff).cast(pl.Date))
            & (pl.col("purchase_date") < pl.lit(label_end).cast(pl.Date))
        )
        .group_by("customer_id")
        .agg(
            pl.col("purchase_amount_real").sum().alias("pp_revenue"),
        )
    )

    # Activity count from transactions
    all_tx_activity = (
        transactions.filter(
            (pl.col("transaction_date") >= pl.lit(cutoff).cast(pl.Date))
            & (pl.col("transaction_date") < pl.lit(label_end).cast(pl.Date))
        )
        .group_by("customer_id")
        .agg(pl.len().alias("activity_count"))
    )

    all_users = transactions["customer_id"].unique().to_frame()

    combined = (
        all_users
        .join(tx_rev.select(["customer_id", "tx_revenue", "tx_count"]), on="customer_id", how="left")
        .join(pp_rev, on="customer_id", how="left")
        .join(all_tx_activity, on="customer_id", how="left")
        .with_columns(
            pl.col("tx_revenue").fill_null(0.0),
            pl.col("pp_revenue").fill_null(0.0),
            pl.col("tx_count").fill_null(0),
            pl.col("activity_count").fill_null(0),
        )
        .with_columns(
            (pl.col("tx_revenue") + pl.col("pp_revenue")).alias("total_revenue")
        )
    )

    # Compute log1p revenue component
    revenue_vals = combined["total_revenue"].to_numpy()
    log_revenue = np.log1p(np.maximum(revenue_vals, 0))

    # Compute activity z-score, clip to [0,1]
    activity_vals = combined["activity_count"].to_numpy().astype(float)
    act_mean = np.mean(activity_vals)
    act_std = np.std(activity_vals)
    if act_std < 1e-9:
        act_z = np.zeros_like(activity_vals)
    else:
        act_z = (activity_vals - act_mean) / act_std
    act_z_clipped = np.clip(act_z, 0, 1)

    # Normalise log_revenue to [0, 1]
    rev_min = log_revenue.min()
    rev_max = log_revenue.max()
    if rev_max - rev_min < 1e-9:
        log_revenue_norm = np.zeros_like(log_revenue)
    else:
        log_revenue_norm = (log_revenue - rev_min) / (rev_max - rev_min)

    freedom_score = 0.7 * log_revenue_norm + 0.3 * act_z_clipped

    combined = combined.with_columns(
        pl.Series(name="freedom_score_target", values=freedom_score.tolist())
    )

    # Diagnostic
    zero_pct = (combined["freedom_score_target"] == 0).sum() / combined.height * 100
    logger.info(
        "Freedom Score target: mean=%.4f  zero_pct=%.1f%%",
        combined["freedom_score_target"].mean() or 0,
        zero_pct,
    )
    if zero_pct > 30:
        logger.warning(
            "[CRITICAL] %.1f%% of users have freedom_score_target=0. "
            "This is the pLTV=0 problem from the previous iteration! "
            "Check: (a) label window [%s, %s] vs data range, "
            "(b) customer_id overlap between tables.",
            zero_pct, cutoff, label_end,
        )

    return combined.select(["customer_id", "freedom_score_target"])


def build_propensity_labels(
    events: pl.DataFrame,
    cutoff,
    window_days: int = 30,
) -> pl.DataFrame:
    """
    Build propensity labels: for each product, was there a COMPLETED activation
    in [cutoff, cutoff+window]?

    Returns pl.DataFrame(customer_id, propensity_{product} for each product)
    """
    from src.features.products import PRODUCT_EVENTS, ALL_PRODUCTS

    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    label_end = cutoff + timedelta(days=window_days)
    logger.info(
        "Building propensity labels: window=[%s, %s]", cutoff, label_end
    )

    ev_window = events.filter(
        (pl.col("status").str.to_uppercase() == "COMPLETED")
        & (pl.col("started_at") >= pl.lit(cutoff).cast(pl.Date).cast(pl.Datetime))
        & (pl.col("started_at") < pl.lit(label_end).cast(pl.Date).cast(pl.Datetime))
    )

    all_users = events["customer_id"].unique().to_frame()
    result = all_users

    for product, process_codes in PRODUCT_EVENTS.items():
        activated = ev_window.filter(
            pl.col("process_code").is_in(process_codes)
        )["customer_id"].unique()

        result = result.with_columns(
            pl.col("customer_id").is_in(activated.to_list()).cast(pl.Int8).alias(f"propensity_{product}")
        )

    # Log distribution
    for product in ALL_PRODUCTS:
        col = f"propensity_{product}"
        if col in result.columns:
            pos_rate = result[col].mean()
            logger.info(
                "propensity_%s: positive=%.1f%%", product, (pos_rate or 0) * 100
            )

    return result


def save_labels(
    labels: pl.DataFrame,
    cutoff_name: str,
    data_interim_dir: str | Path = "data/interim",
) -> None:
    """Save labels parquet to data/interim/labels_{cutoff_name}.parquet."""
    data_interim_dir = Path(data_interim_dir)
    data_interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_interim_dir / f"labels_{cutoff_name}.parquet"
    labels.write_parquet(out_path)
    logger.info("Labels saved to %s", out_path)


def save_label_diagnostics(
    churn_df: pl.DataFrame,
    freedom_df: pl.DataFrame,
    propensity_df: pl.DataFrame,
    reports_dir: str | Path = "reports",
) -> None:
    """Save reports/label_diagnostics.csv."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Churn
    rows.append({
        "label": "churn_label",
        "type": "binary",
        "n_users": churn_df.height,
        "positive_count": churn_df["churn_label"].sum(),
        "positive_rate": churn_df["churn_label"].mean(),
        "zero_pct": None,
        "mean": None,
        "std": None,
    })

    # Freedom Score
    rows.append({
        "label": "freedom_score_target",
        "type": "continuous",
        "n_users": freedom_df.height,
        "positive_count": None,
        "positive_rate": None,
        "zero_pct": (freedom_df["freedom_score_target"] == 0).sum() / freedom_df.height * 100,
        "mean": freedom_df["freedom_score_target"].mean(),
        "std": freedom_df["freedom_score_target"].std(),
    })

    # Propensity
    for col in propensity_df.columns:
        if col == "customer_id":
            continue
        rows.append({
            "label": col,
            "type": "binary",
            "n_users": propensity_df.height,
            "positive_count": propensity_df[col].sum(),
            "positive_rate": propensity_df[col].mean(),
            "zero_pct": None,
            "mean": None,
            "std": None,
        })

    diag_df = pl.DataFrame(rows)
    out_path = reports_dir / "label_diagnostics.csv"
    diag_df.write_csv(out_path)
    logger.info("Label diagnostics saved to %s", out_path)
