"""
src/features/build_matrix.py
=============================
Task 6 — Assemble final feature matrix by left-joining all feature stores to users.

For each cutoff (T, T_minus_1, T_minus_2) produces:
  data/interim/features_{cutoff_name}.parquet

Also produces:
  reports/feature_summary.csv — describe() stats for all numeric features
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from src.features.demo import build_demo_features
from src.features.engagement import build_engagement_features
from src.features.mcc_mapping import make_mcc_frequency_report
from src.features.partners import build_partner_features
from src.features.products import build_product_features
from src.features.rfm_temporal import build_rfm_features
from src.features.transaction_mix import build_transaction_mix_features

logger = logging.getLogger(__name__)


def build_feature_matrix(
    tables: dict[str, pl.DataFrame],
    cutoff,
    cutoff_name: str = "T",
    data_interim_dir: str | Path = "data/interim",
    reports_dir: str | Path = "reports",
) -> pl.DataFrame:
    """
    Build the complete feature matrix for a given cutoff date.

    Parameters
    ----------
    tables        : dict from load_all()
    cutoff        : date or str — observation point
    cutoff_name   : label used in output filename (T, T_minus_1, T_minus_2)
    data_interim_dir : where to save parquet
    reports_dir   : where to save feature_summary.csv

    Returns
    -------
    pl.DataFrame with one row per user from users table
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    data_interim_dir = Path(data_interim_dir)
    data_interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Building feature matrix for %s (cutoff=%s) ===", cutoff_name, cutoff)

    users = tables["users"]
    transactions = tables["transactions"]
    events = tables["events"]
    partner_purchases = tables["partner_purchases"]
    acquisition = tables["acquisition"]

    # Run MCC EDA on first call (only once)
    mcc_report_path = reports_dir / "mcc_frequency.csv"
    if not mcc_report_path.exists():
        logger.info("Running MCC frequency EDA...")
        make_mcc_frequency_report(transactions, reports_dir)

    # -----------------------------------------------------------------------
    # Build individual feature stores
    # -----------------------------------------------------------------------
    logger.info("Building RFM features...")
    rfm = build_rfm_features(transactions, events, cutoff)

    logger.info("Building transaction mix features...")
    tx_mix = build_transaction_mix_features(transactions, cutoff)

    logger.info("Building engagement features...")
    engagement = build_engagement_features(events, cutoff)

    logger.info("Building product features...")
    products = build_product_features(events, cutoff)

    logger.info("Building demo features...")
    demo = build_demo_features(users, acquisition, cutoff)

    logger.info("Building partner features...")
    partners = build_partner_features(partner_purchases, cutoff)

    # -----------------------------------------------------------------------
    # Assemble: start from users (one row per user) and left-join all stores
    # -----------------------------------------------------------------------
    logger.info("Assembling feature matrix...")
    result = demo  # demo starts from users table → guaranteed one row per user

    for store_name, store_df in [
        ("rfm", rfm),
        ("tx_mix", tx_mix),
        ("engagement", engagement),
        ("products", products),
        ("partners", partners),
    ]:
        n_before = result.height
        result = result.join(store_df, on="customer_id", how="left")
        n_after = result.height
        if n_after != n_before:
            logger.warning(
                "Row count changed after joining %s: %d → %d",
                store_name, n_before, n_after,
            )

    # -----------------------------------------------------------------------
    # Validation checks
    # -----------------------------------------------------------------------
    assert result.height == users.height, (
        f"Feature matrix has {result.height} rows but users has {users.height}"
    )

    # Check for 100% null columns
    cols_all_null = [
        col for col in result.columns
        if result[col].null_count() == result.height
    ]
    if cols_all_null:
        logger.warning(
            "Dropping %d 100%%-null columns: %s", len(cols_all_null), cols_all_null
        )
        result = result.drop(cols_all_null)

    # Save parquet
    out_path = data_interim_dir / f"features_{cutoff_name}.parquet"
    result.write_parquet(out_path)
    logger.info("Feature matrix saved to %s (%d rows × %d cols)", out_path, result.height, result.width)

    # Save feature summary
    _save_feature_summary(result, reports_dir)

    return result


def _save_feature_summary(
    df: pl.DataFrame,
    reports_dir: Path,
) -> None:
    """Save describe() stats for all columns to reports/feature_summary.csv."""
    rows = []
    for col in df.columns:
        s = df[col]
        null_pct = s.null_count() / df.height * 100
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "null_pct": round(null_pct, 2),
            "n_unique": s.n_unique(),
        }
        if s.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
            notnull = s.drop_nulls()
            if notnull.is_empty():
                row.update({"mean": None, "std": None, "min": None, "max": None, "median": None})
            else:
                row.update({
                    "mean": notnull.mean(),
                    "std": notnull.std(),
                    "min": notnull.min(),
                    "max": notnull.max(),
                    "median": notnull.median(),
                })
        else:
            row.update({"mean": None, "std": None, "min": None, "max": None, "median": None})
        rows.append(row)

    summary_df = pl.DataFrame(rows)

    # Warn on zero-std columns
    zero_std = summary_df.filter(
        pl.col("std").is_not_null() & (pl.col("std") < 1e-9)
    )
    if not zero_std.is_empty():
        logger.warning(
            "Columns with std≈0 (useless for modelling): %s",
            zero_std["column"].to_list(),
        )

    out_path = reports_dir / "feature_summary.csv"
    summary_df.write_csv(out_path)
    logger.info("Feature summary saved to %s", out_path)


def build_all_cutoff_matrices(
    tables: dict[str, pl.DataFrame],
    cutoffs: dict,
    data_interim_dir: str | Path = "data/interim",
    reports_dir: str | Path = "reports",
) -> dict[str, pl.DataFrame]:
    """
    Build feature matrices for all three cutoffs: T, T_minus_1, T_minus_2.

    Returns dict: {"T": df_T, "T_minus_1": df_T1, "T_minus_2": df_T2}
    """
    result_matrices = {}
    for cutoff_name in ["T_minus_2", "T_minus_1", "T"]:
        cutoff_date = cutoffs[cutoff_name]
        df = build_feature_matrix(
            tables=tables,
            cutoff=cutoff_date,
            cutoff_name=cutoff_name,
            data_interim_dir=data_interim_dir,
            reports_dir=reports_dir,
        )
        result_matrices[cutoff_name] = df

    return result_matrices
