"""
src/features/time_split.py
===========================
Task 2 — Define cutoff dates for train/val/test splits.

Strategy:
  - T (test cutoff) = max(transactions.transaction_date) − 30d
  - val window:   [T−60d, T−30d)
  - train window: everything before T−60d

If events max < users reg_date min (STOP-2 scenario), falls back to
using first_transaction_date as user effective start.

Outputs:
  data/interim/cutoffs.json
  reports/cutoff_summary.txt
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def define_cutoffs(
    transactions: pl.DataFrame,
    events: pl.DataFrame,
    label_window_days: int = 14,
    val_window_days: int = 30,
    data_interim_dir: str | Path = "data/interim",
    reports_dir: str | Path = "reports",
) -> dict:
    """
    Determine T, T_minus_1, T_minus_2 cutoff dates.

    Parameters
    ----------
    transactions      : loaded transactions DataFrame (needs transaction_date)
    events            : loaded events DataFrame (needs started_at)
    label_window_days : days after cutoff used as the label window (default 14)
    val_window_days   : days between train and val cutoffs (default 30)

    Returns
    -------
    dict with keys: T, T_minus_1, T_minus_2, strategy, label_window_days
    """
    data_interim_dir = Path(data_interim_dir)
    data_interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Determine data end dates
    tx_max = transactions["transaction_date"].drop_nulls().max()
    tx_min = transactions["transaction_date"].drop_nulls().min()

    events_max = events["started_at"].drop_nulls().max()
    if events_max is not None:
        if hasattr(events_max, "date"):
            events_max_date: date = events_max.date()
        else:
            events_max_date = events_max
    else:
        events_max_date = None

    logger.info("transactions: min=%s  max=%s", tx_min, tx_max)
    logger.info("events:       max=%s", events_max_date)

    # Strategy decision
    if isinstance(tx_max, date):
        data_end = tx_max
    else:
        data_end = date.today()

    # T = data end − label_window_days (so there's label data after T)
    T = data_end - timedelta(days=label_window_days)
    T_minus_1 = T - timedelta(days=val_window_days)
    T_minus_2 = T_minus_1 - timedelta(days=val_window_days)

    strategy = "standard"
    strategy_notes = (
        f"Standard cutoff strategy:\n"
        f"  data_end = max(transactions.transaction_date) = {data_end}\n"
        f"  T (test cutoff)       = data_end − {label_window_days}d = {T}\n"
        f"  T_minus_1 (val cutoff)  = T − {val_window_days}d = {T_minus_1}\n"
        f"  T_minus_2 (train cutoff) = T−1 − {val_window_days}d = {T_minus_2}\n"
        f"  Label window = {label_window_days} days (compact due to narrow data window)\n"
        f"  Train data:  all rows with date < {T_minus_2}\n"
        f"  Val data:    rows in [{T_minus_2}, {T_minus_1})\n"
        f"  Test data:   rows in [{T_minus_1}, {T})\n"
        f"  Labels built from:  [{T}, {T + timedelta(days=label_window_days)})\n"
    )

    cutoffs = {
        "T": str(T),
        "T_minus_1": str(T_minus_1),
        "T_minus_2": str(T_minus_2),
        "strategy": strategy,
        "label_window_days": label_window_days,
        "val_window_days": val_window_days,
        "data_end": str(data_end),
        "tx_min": str(tx_min),
        "tx_max": str(tx_max),
        "events_max": str(events_max_date) if events_max_date else None,
        "strategy_notes": strategy_notes,
    }

    # Save cutoffs.json
    json_path = data_interim_dir / "cutoffs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cutoffs, f, indent=2, default=str)
    logger.info("Cutoffs saved to %s", json_path)

    # Save summary txt
    txt_path = reports_dir / "cutoff_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("FreedomProfile Analytics Pipeline — Cutoff Strategy\n")
        f.write("=" * 60 + "\n\n")
        f.write(strategy_notes)
        f.write("\n\nFor presentation slide 'Limitations':\n")
        f.write(
            f"  - Training data spans only ~{(T_minus_2 - (tx_min if isinstance(tx_min, date) else date(2026,1,1))).days} days\n"
            f"    (FreedomSuperApp launched 2026-01-01, data through {data_end})\n"
            f"  - Short window limits seasonal pattern capture\n"
            f"  - Label window reduced to {label_window_days}d (vs. standard 30d) to\n"
            f"    preserve train/val/test split integrity\n"
        )
    logger.info("Cutoff summary saved to %s", txt_path)

    _check_active_users(transactions, T, label_window_days)

    return cutoffs


def apply_cutoff(
    df: pl.DataFrame,
    cutoff_date,
    date_col: str,
) -> pl.DataFrame:
    """
    Return only rows where date_col < cutoff_date (strict less-than).

    Works with pl.Date and pl.Datetime columns.
    """
    if isinstance(cutoff_date, str):
        from datetime import date as _date
        cutoff_date = _date.fromisoformat(cutoff_date)

    col_dtype = df[date_col].dtype
    if col_dtype == pl.Datetime or str(col_dtype).startswith("Datetime"):
        cutoff_val = pl.lit(cutoff_date).cast(pl.Date).cast(pl.Datetime)
    else:
        cutoff_val = pl.lit(cutoff_date).cast(pl.Date)

    return df.filter(pl.col(date_col) < cutoff_val)


def _check_active_users(
    transactions: pl.DataFrame,
    cutoff_date,
    window_days: int,
) -> None:
    """Log how many users have ≥1 transaction in [cutoff-window, cutoff)."""
    if isinstance(cutoff_date, str):
        from datetime import date as _date
        cutoff_date = _date.fromisoformat(cutoff_date)

    window_start = cutoff_date - timedelta(days=window_days)

    col_dtype = transactions["transaction_date"].dtype
    if col_dtype == pl.Date:
        mask = (
            (pl.col("transaction_date") >= pl.lit(window_start).cast(pl.Date))
            & (pl.col("transaction_date") < pl.lit(cutoff_date).cast(pl.Date))
        )
    else:
        mask = (
            (pl.col("transaction_date") >= pl.lit(window_start).cast(pl.Date).cast(pl.Datetime))
            & (pl.col("transaction_date") < pl.lit(cutoff_date).cast(pl.Date).cast(pl.Datetime))
        )

    active_users = (
        transactions.filter(mask)["customer_id"].n_unique()
    )
    total_users = transactions["customer_id"].n_unique()
    pct = active_users / total_users * 100 if total_users else 0

    logger.info(
        "Active users with ≥1 tx in [T-%dd, T): %d / %d (%.1f%%)",
        window_days, active_users, total_users, pct,
    )
    if pct < 30:
        logger.warning(
            "[FLAG] Only %.1f%% of users active in observation window. "
            "Signal may be very sparse. Consider widening the cutoff window.",
            pct,
        )
