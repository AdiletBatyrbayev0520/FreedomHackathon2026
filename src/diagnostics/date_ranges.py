"""
src/diagnostics/date_ranges.py
================================
Task 0 — Date range analysis and temporal consistency checks.

Stop conditions:
  [STOP-2] max(events.started_at) < min(users.reg_date)

Outputs:
  reports/date_ranges.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def compute_date_ranges(
    tables: dict[str, pl.DataFrame],
    reports_dir: str | Path = "reports",
    stop_on_critical: bool = True,
) -> pl.DataFrame:
    """
    Compute min/max date ranges for all date/datetime columns across tables.
    Checks temporal stop conditions.

    Returns pl.DataFrame and writes reports/date_ranges.csv.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Define which column to check per table
    date_cols = {
        "users": "reg_date",
        "transactions": "transaction_date",
        "events": "started_at",
        "partner_purchases": "purchase_date",
    }

    rows = []
    for table_name, col_name in date_cols.items():
        if table_name not in tables:
            continue
        df = tables[table_name]
        if col_name not in df.columns:
            logger.warning("[%s] Column '%s' not found", table_name, col_name)
            continue

        col = df[col_name].drop_nulls()
        if col.is_empty():
            rows.append({
                "table": table_name,
                "date_col": col_name,
                "min_date": None,
                "max_date": None,
                "n_nulls": df[col_name].null_count(),
                "n_total": df.height,
            })
            continue

        # Cast to sortable type for min/max
        try:
            if col.dtype in (pl.Date, pl.Datetime):
                min_val = col.min()
                max_val = col.max()
            else:
                col_casted = col.cast(pl.Utf8)
                min_val = col_casted.min()
                max_val = col_casted.max()
        except Exception as e:
            logger.warning("[%s.%s] Could not compute min/max: %s", table_name, col_name, e)
            min_val, max_val = None, None

        rows.append(
            {
                "table": table_name,
                "date_col": col_name,
                "min_date": str(min_val) if min_val is not None else None,
                "max_date": str(max_val) if max_val is not None else None,
                "n_nulls": df[col_name].null_count(),
                "n_total": df.height,
            }
        )
        logger.info(
            "[%s.%s] min=%s  max=%s  nulls=%d",
            table_name, col_name, min_val, max_val, df[col_name].null_count(),
        )

    date_df = pl.DataFrame(
        rows,
        schema={
            "table": pl.Utf8,
            "date_col": pl.Utf8,
            "min_date": pl.Utf8,
            "max_date": pl.Utf8,
            "n_nulls": pl.Int64,
            "n_total": pl.Int64,
        },
    )

    out_path = reports_dir / "date_ranges.csv"
    date_df.write_csv(out_path)
    logger.info("Date ranges saved to %s", out_path)

    # -----------------------------------------------------------------------
    # Stop condition checks
    # -----------------------------------------------------------------------
    stop_flags = _check_temporal_stop_conditions(date_df, stop_on_critical)

    return date_df


def _check_temporal_stop_conditions(
    date_df: pl.DataFrame,
    stop_on_critical: bool,
) -> list[tuple[str, str]]:
    """Check STOP-2: max(events.started_at) < min(users.reg_date)."""
    stop_flags = []

    def _get(table: str, field: str) -> str | None:
        row = date_df.filter(pl.col("table") == table)
        if row.is_empty():
            return None
        val = row[field][0]
        return str(val) if val is not None else None

    events_max = _get("events", "max_date")
    users_min = _get("users", "min_date")

    if events_max and users_min:
        try:
            # Compare as strings (ISO format sorts lexicographically)
            if events_max < users_min:
                msg = (
                    f"\n{'='*70}\n"
                    f"[STOP-2] CRITICAL: max(events.started_at) = {events_max}\n"
                    f"                   min(users.reg_date)    = {users_min}\n"
                    f"All events occurred BEFORE any user was registered in the cleaned dataset.\n"
                    f"This means:\n"
                    f"  (a) Events belong to users dropped during deduplication, OR\n"
                    f"  (b) reg_date column is synthetic/truncated, OR\n"
                    f"  (c) Events and users tables reference different customer populations.\n"
                    f"ACTION: Review dedup logic. Consider using first_transaction_date as\n"
                    f"        effective user start date instead of reg_date.\n"
                    f"{'='*70}"
                )
                logger.error(msg)
                stop_flags.append(("STOP-2", msg))
            else:
                logger.info(
                    "[OK] Temporal check: events.max=%s >= users.min_reg_date=%s",
                    events_max, users_min,
                )
        except Exception as e:
            logger.warning("Could not compare temporal ranges: %s", e)

    if stop_on_critical and stop_flags:
        print("\n" + "!"*70)
        for flag_id, msg in stop_flags:
            print(msg)
        print("Halting pipeline. Resolve temporal issues above before continuing.")
        print("!"*70 + "\n")
        sys.exit(1)

    return stop_flags
