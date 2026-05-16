"""
src/diagnostics/id_overlap.py
==============================
Task 0 — Customer ID overlap analysis between tables.

Stop conditions:
  [STOP-1] users ↔ events overlap < 50%
  [STOP-2] max(events.started_at) < min(users.reg_date)

Outputs:
  reports/id_overlap.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def compute_id_overlap(
    users_df: pl.DataFrame,
    other_df: pl.DataFrame,
    table_name: str,
) -> dict:
    """
    Compute ID overlap statistics between users and another table.

    Parameters
    ----------
    users_df : pl.DataFrame  — must have column 'customer_id'
    other_df : pl.DataFrame  — must have column 'customer_id'
    table_name : str         — label for reporting

    Returns
    -------
    dict with keys:
        users_unique, other_unique, intersection_size,
        orphan_rows, users_without_any_row, overlap_pct
    """
    users_ids = set(users_df["customer_id"].drop_nulls().to_list())
    other_ids_series = other_df["customer_id"].drop_nulls()
    other_ids_set = set(other_ids_series.to_list())

    intersection = users_ids & other_ids_set
    orphan_ids = other_ids_set - users_ids

    # Orphan rows: rows in other_df whose customer_id is NOT in users
    orphan_rows = other_df.filter(
        ~pl.col("customer_id").is_in(list(users_ids))
    ).height

    # Users without any row in other table
    users_without = len(users_ids - other_ids_set)

    overlap_pct = len(intersection) / len(users_ids) * 100 if users_ids else 0.0
    orphan_row_pct = orphan_rows / other_df.height * 100 if other_df.height else 0.0

    result = {
        "table": table_name,
        "users_unique_ids": len(users_ids),
        "other_unique_ids": len(other_ids_set),
        "intersection_ids": len(intersection),
        "overlap_pct": round(overlap_pct, 2),
        "orphan_ids_in_other": len(orphan_ids),
        "orphan_rows_in_other": orphan_rows,
        "orphan_rows_pct": round(orphan_row_pct, 2),
        "users_without_any_row": users_without,
    }

    return result


def run_all_overlaps(
    tables: dict[str, pl.DataFrame],
    reports_dir: str | Path = "reports",
    stop_on_critical: bool = True,
) -> pl.DataFrame:
    """
    Compute ID overlap for all (users, X) pairs and write reports/id_overlap.csv.

    Parameters
    ----------
    tables          : dict from load_all()
    reports_dir     : where to write id_overlap.csv
    stop_on_critical: if True, sys.exit(1) on stop conditions
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    users_df = tables["users"]
    target_tables = {
        k: v for k, v in tables.items() if k != "users"
    }

    rows = []
    stop_flags = []

    for name, df in target_tables.items():
        result = compute_id_overlap(users_df, df, name)
        rows.append(result)

        logger.info(
            "Overlap users <-> %-20s | intersection=%d (%.1f%%) | orphan_rows=%d (%.1f%%) | users_without=%d",
            name,
            result["intersection_ids"],
            result["overlap_pct"],
            result["orphan_rows_in_other"],
            result["orphan_rows_pct"],
            result["users_without_any_row"],
        )

        # Stop condition 1: events overlap < 50%
        if name == "events" and result["overlap_pct"] < 50.0:
            msg = (
                f"\n{'='*70}\n"
                f"[STOP-1] CRITICAL: users ↔ events overlap is only {result['overlap_pct']:.1f}% "
                f"(threshold: 50%).\n"
                f"Events customer_ids do not match users. This will silently break all "
                f"event-based features and produce models trained on near-empty data.\n"
                f"ACTION: Investigate the deuplication step — events may reference deleted "
                f"customer_ids. Check data/raw before proceeding.\n"
                f"{'='*70}"
            )
            logger.error(msg)
            stop_flags.append(("STOP-1", msg))

        # Warn if transactions orphan rate > 30%
        if name == "transactions" and result["orphan_rows_pct"] > 30.0:
            logger.warning(
                "[FLAG] transactions orphan rows = %.1f%% (>30%%). "
                "Features will only be built for matching users.",
                result["orphan_rows_pct"],
            )

    overlap_df = pl.DataFrame(rows)
    out_path = reports_dir / "id_overlap.csv"
    overlap_df.write_csv(out_path)
    logger.info("ID overlap report saved to %s", out_path)

    if stop_on_critical and stop_flags:
        print("\n" + "!"*70)
        for flag_id, msg in stop_flags:
            print(msg)
        print("Halting pipeline. Fix the data issues above before continuing.")
        print("!"*70 + "\n")
        sys.exit(1)

    return overlap_df
