"""
src/data_loading.py
===================
Task 1 — Load all 5 cleaned datasets into Polars DataFrames with explicit types.

Key design decisions:
- customer_id comes as float in CSV (e.g. 466230.0) → cast to Int64
- Rows with null customer_id after cast are dropped and logged
- transaction_status_clean derived column: "Успешные" → success, AUSR* → failure, else unknown
- Reports data_quality.csv with shape / null counts / dtypes
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths (actual filenames — have spaces, not underscores)
# ---------------------------------------------------------------------------
_PROCESSED_DIR = Path("data/processed")

_FILE_USERS = "SAPP пользователи unique.csv"
_FILE_TRANSACTIONS = "SAPP транзакции cleaned.csv"
_FILE_EVENTS = "SAPP Процессы пользователей в приложении cleaned.csv"
_FILE_PARTNER_PURCHASES = "SAPP Покупки у партнеров cleaned normalized.csv"
_FILE_ACQUISITION = "SAPP привлечения cleaned.csv"

# ---------------------------------------------------------------------------
# transaction_status normalisation map
# ---------------------------------------------------------------------------
_STATUS_MAP = {
    "Успешные": "success",
    "успешные": "success",
    "УСПЕШНЫЕ": "success",
}

# Codes starting with AUSR are failures; everything else unknown
def _normalise_status(series: pl.Series) -> pl.Series:
    """Map raw transaction_status → success / failure / unknown."""
    return (
        pl.when(series.str.to_lowercase().is_in(["успешные"]))
        .then(pl.lit("success"))
        .when(series.str.to_uppercase().str.starts_with("AUSR"))
        .then(pl.lit("failure"))
        .otherwise(pl.lit("unknown"))
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cast_customer_id(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """Cast customer_id from Float64/Utf8 → Int64, drop nulls, log count."""
    df = df.with_columns(
        pl.col("customer_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    )
    nulls = df.filter(pl.col("customer_id").is_null()).height
    if nulls:
        logger.warning("[%s] Dropped %d rows with null customer_id after cast", source, nulls)
        df = df.filter(pl.col("customer_id").is_not_null())
    return df


def _resolve(data_dir: Path, filename: str) -> Path:
    return data_dir / filename


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_users(data_dir: str | Path = _PROCESSED_DIR) -> pl.DataFrame:
    """Load users dataset with explicit types."""
    path = _resolve(Path(data_dir), _FILE_USERS)
    df = pl.read_csv(
        path,
        schema_overrides={
            "customer_id": pl.Float64,
            "customer_age": pl.Int32,
            "city": pl.Utf8,
            "gender": pl.Utf8,
            "reg_date": pl.Utf8,
        },
        infer_schema_length=10000,
    )
    df = _cast_customer_id(df, "users")
    df = df.with_columns(
        pl.col("reg_date").str.to_date(format="%Y-%m-%d", strict=False)
    )
    logger.info("[users] Loaded %d rows, %d unique customer_ids", df.height, df["customer_id"].n_unique())
    return df


def load_transactions(data_dir: str | Path = _PROCESSED_DIR) -> pl.DataFrame:
    """Load transactions dataset with explicit types."""
    path = _resolve(Path(data_dir), _FILE_TRANSACTIONS)
    df = pl.read_csv(
        path,
        schema_overrides={
            "customer_id": pl.Float64,
            "transaction_id": pl.Utf8,
            "transaction_date": pl.Utf8,
            "operation_type": pl.Utf8,
            "transaction_status": pl.Utf8,
            "mcc": pl.Utf8,
            "terminal_type": pl.Utf8,
            "transaction_sum": pl.Float64,
            "transaction_sum_was_missing": pl.Utf8,
        },
        infer_schema_length=10000,
    )
    df = _cast_customer_id(df, "transactions")
    df = df.with_columns(
        pl.col("transaction_date").str.to_date(format="%Y-%m-%d", strict=False),
        pl.col("transaction_sum_was_missing")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .is_in(["true", "1", "yes"])
        .alias("transaction_sum_was_missing"),
    )
    # Derive clean status column using native expressions (no map_elements)
    df = df.with_columns(
        pl.when(pl.col("transaction_status").str.to_lowercase().is_in(["успешные"]))
        .then(pl.lit("success"))
        .when(pl.col("transaction_status").str.to_uppercase().str.starts_with("AUSR"))
        .then(pl.lit("failure"))
        .otherwise(pl.lit("unknown"))
        .alias("transaction_status_clean")
    )
    logger.info(
        "[transactions] Loaded %d rows, %d unique customer_ids",
        df.height, df["customer_id"].n_unique(),
    )
    return df


def load_events(data_dir: str | Path = _PROCESSED_DIR) -> pl.DataFrame:
    """Load app events dataset with explicit types."""
    path = _resolve(Path(data_dir), _FILE_EVENTS)
    df = pl.read_csv(
        path,
        schema_overrides={
            "customer_id": pl.Float64,
            "process_code": pl.Utf8,
            "lang": pl.Utf8,
            "status": pl.Utf8,
            "started_at": pl.Utf8,
            "completed_at": pl.Utf8,
        },
        infer_schema_length=10000,
    )
    df = _cast_customer_id(df, "events")
    # Parse datetimes — format may include fractional seconds
    for col in ("started_at", "completed_at"):
        df = df.with_columns(
            pl.col(col)
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
            .alias(col)
        )
    logger.info(
        "[events] Loaded %d rows, %d unique customer_ids",
        df.height, df["customer_id"].n_unique(),
    )
    return df


def load_partner_purchases(data_dir: str | Path = _PROCESSED_DIR) -> pl.DataFrame:
    """Load partner purchases dataset with explicit types."""
    path = _resolve(Path(data_dir), _FILE_PARTNER_PURCHASES)
    df = pl.read_csv(
        path,
        schema_overrides={
            "customer_id": pl.Float64,
            "purchase_date": pl.Utf8,
            "remote_flag": pl.Utf8,
            "counter": pl.Int32,
            "purchase_amount_real": pl.Float64,
            "cashback_amount_real": pl.Float64,
            "cashback_rate": pl.Float64,
            "app_name_normalized": pl.Utf8,
        },
        infer_schema_length=10000,
    )
    df = _cast_customer_id(df, "partner_purchases")
    df = df.with_columns(
        pl.col("purchase_date").str.to_date(format="%Y-%m-%d", strict=False)
    )
    logger.info(
        "[partner_purchases] Loaded %d rows, %d unique customer_ids",
        df.height, df["customer_id"].n_unique(),
    )
    return df


def load_acquisition(data_dir: str | Path = _PROCESSED_DIR) -> pl.DataFrame:
    """Load acquisition dataset with explicit types."""
    path = _resolve(Path(data_dir), _FILE_ACQUISITION)
    df = pl.read_csv(
        path,
        schema_overrides={
            "customer_id": pl.Float64,
            "secondary_category_filled": pl.Utf8,
        },
        infer_schema_length=10000,
    )
    df = _cast_customer_id(df, "acquisition")
    logger.info(
        "[acquisition] Loaded %d rows, %d unique customer_ids",
        df.height, df["customer_id"].n_unique(),
    )
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_all(data_dir: str | Path = _PROCESSED_DIR) -> dict[str, pl.DataFrame]:
    """
    Load all 5 cleaned datasets and return as a dict keyed by table name.

    Returns
    -------
    dict with keys: "users", "transactions", "events", "partner_purchases", "acquisition"
    """
    data_dir = Path(data_dir)
    tables = {
        "users": load_users(data_dir),
        "transactions": load_transactions(data_dir),
        "events": load_events(data_dir),
        "partner_purchases": load_partner_purchases(data_dir),
        "acquisition": load_acquisition(data_dir),
    }

    # Log summary
    logger.info("=== Dataset Summary ===")
    for name, df in tables.items():
        logger.info("  %-20s  rows=%-10d  cols=%d", name, df.height, df.width)

    # Save data quality report
    _save_data_quality(tables)

    return tables


def _save_data_quality(tables: dict[str, pl.DataFrame], reports_dir: str | Path = "reports") -> None:
    """Write reports/data_quality.csv with shape/null/dtype info."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for table_name, df in tables.items():
        for col in df.columns:
            null_count = df[col].null_count()
            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "n_rows": df.height,
                    "null_count": null_count,
                    "null_pct": round(null_count / df.height * 100, 2) if df.height else 0,
                    "n_unique": df[col].n_unique(),
                }
            )

    quality_df = pl.DataFrame(rows)
    out_path = reports_dir / "data_quality.csv"
    quality_df.write_csv(out_path)
    logger.info("Data quality report saved to %s", out_path)
