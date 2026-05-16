"""
src/features/products.py
=========================
Task 5 — Product activation features via process_code in events.

There is NO separate products table.
Product activations are inferred from COMPLETED events with matching process_codes.

PRODUCT_EVENTS mapping is populated based on actual process_codes in the data.
Common codes found in FreedomSuperApp events:
  LivenessProcess, OpenCardProcess, FreedomRatingActivationProcess,
  FrhcActivationProcess, SignCardDocumentsProcess, DepositWithdrawalProcess, etc.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product → process_code mapping
# Adjust after reviewing unique process_codes in the data
# ---------------------------------------------------------------------------
PRODUCT_EVENTS: dict[str, list[str]] = {
    "card": [
        "OpenCardProcess",
        "SignCardDocumentsProcess",
        "CardActivationProcess",
        "CardOrderProcess",
        "CardIssueProcess",
    ],
    "freedom_rating": [
        "FreedomRatingActivationProcess",
        "FreedomRatingProcess",
        "RatingActivationProcess",
    ],
    "frhc": [
        "FrhcActivationProcess",
        "FRHCActivationProcess",
        "FrhcProcess",
    ],
    "deposit": [
        "DepositWithdrawalProcess",
        "DepositOpenProcess",
        "DepositProcess",
        "SavingsProcess",
    ],
    "loan": [
        "LoanApplicationProcess",
        "LoanProcess",
        "CreditApplicationProcess",
        "MicroLoanProcess",
    ],
    "liveness": [
        "LivenessProcess",
        "BiometricProcess",
        "IdentificationProcess",
    ],
}

ALL_PRODUCTS = list(PRODUCT_EVENTS.keys())


def build_product_features(
    events: pl.DataFrame,
    cutoff,
) -> pl.DataFrame:
    """
    Build product activation features for all users up to cutoff.

    Features:
      has_{product}                          — bool: COMPLETED activation ever before cutoff
      products_count                         — number of activated products
      days_since_first_product_activation    — days since earliest activation
      days_since_last_product_activation     — days since most recent activation

    Parameters
    ----------
    events  : DataFrame with customer_id, process_code, status, started_at
    cutoff  : date or str (exclusive upper bound)
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    logger.info("Building product features for cutoff=%s", cutoff)

    # Filter to completed events strictly before cutoff
    ev_completed = events.filter(
        (pl.col("status").str.to_uppercase() == "COMPLETED")
        & (pl.col("started_at") < pl.lit(cutoff).cast(pl.Date).cast(pl.Datetime))
    )

    user_base = events["customer_id"].unique().to_frame()
    result = user_base

    activation_dates = []  # list of (customer_id, activation_date) for all products

    for product, process_codes in PRODUCT_EVENTS.items():
        # Case-insensitive match
        product_ev = ev_completed.filter(
            pl.col("process_code").is_in(process_codes)
        )

        # has_product flag
        activated_users = product_ev["customer_id"].unique()
        has_col = pl.col("customer_id").is_in(activated_users.to_list()).alias(f"has_{product}")
        result = result.with_columns(has_col)

        # Collect activation dates for days_since calc
        if not product_ev.is_empty():
            activation_dates.append(
                product_ev.select(["customer_id", "started_at"])
            )

    # products_count
    flag_cols = [f"has_{p}" for p in ALL_PRODUCTS]
    result = result.with_columns(
        pl.sum_horizontal(*[pl.col(c).cast(pl.Int32) for c in flag_cols])
        .alias("products_count")
    )

    # days_since first/last product activation
    if activation_dates:
        all_activations = pl.concat(activation_dates)
        first_last = (
            all_activations.group_by("customer_id")
            .agg(
                pl.col("started_at").min().cast(pl.Date).alias("first_activation"),
                pl.col("started_at").max().cast(pl.Date).alias("last_activation"),
            )
            .with_columns(
                (pl.lit(cutoff) - pl.col("first_activation"))
                .dt.total_days()
                .alias("days_since_first_product_activation"),
                (pl.lit(cutoff) - pl.col("last_activation"))
                .dt.total_days()
                .alias("days_since_last_product_activation"),
            )
            .select([
                "customer_id",
                "days_since_first_product_activation",
                "days_since_last_product_activation",
            ])
        )
        result = result.join(first_last, on="customer_id", how="left")
    else:
        result = result.with_columns(
            pl.lit(None).cast(pl.Int64).alias("days_since_first_product_activation"),
            pl.lit(None).cast(pl.Int64).alias("days_since_last_product_activation"),
        )

    # Sanity: warn if all users have 0 products (mapping may be wrong)
    avg_products = result["products_count"].mean()
    if avg_products is not None and avg_products < 0.01:
        logger.warning(
            "[products] Average products_count=%.3f. "
            "All users have 0 products — PRODUCT_EVENTS mapping may not match actual process_codes. "
            "Run: events['process_code'].value_counts() to inspect.",
            avg_products,
        )
    else:
        logger.info("Average products_count=%.2f (expected 1-3)", avg_products or 0)

    logger.info(
        "Product features built: %d users, %d columns",
        result.height, result.width,
    )
    return result
