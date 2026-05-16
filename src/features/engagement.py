"""
src/features/engagement.py
===========================
Task 5 — App engagement features from events table.

Output columns per user:
  events_per_day_{7,30}d
  unique_processes_30d
  error_rate_30d
  declined_rate_30d
  completion_rate_30d
  avg_process_duration_sec
  weekend_share_30d
  lang_ru_share, lang_kz_share
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

logger = logging.getLogger(__name__)


def build_engagement_features(
    events: pl.DataFrame,
    cutoff,
    window_days: int = 30,
) -> pl.DataFrame:
    """
    Build engagement features from app events.

    Parameters
    ----------
    events    : DataFrame with customer_id, process_code, lang, status,
                started_at, completed_at
    cutoff    : date or str
    window_days: primary aggregation window (default 30d)
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    cutoff_dt = pl.lit(cutoff).cast(pl.Utf8).str.to_datetime(format="%Y-%m-%d", strict=False)
    window_start = cutoff - timedelta(days=window_days)
    window_start_7d = cutoff - timedelta(days=7)

    logger.info("Building engagement features for cutoff=%s", cutoff)

    user_base = events["customer_id"].unique().to_frame()

    # Filter to 30d window
    ev_30 = events.filter(
        (pl.col("started_at") >= pl.lit(window_start).cast(pl.Date).cast(pl.Datetime))
        & (pl.col("started_at") < pl.lit(cutoff).cast(pl.Date).cast(pl.Datetime))
    )

    ev_7 = events.filter(
        (pl.col("started_at") >= pl.lit(window_start_7d).cast(pl.Date).cast(pl.Datetime))
        & (pl.col("started_at") < pl.lit(cutoff).cast(pl.Date).cast(pl.Datetime))
    )

    # -----------------------------------------------------------------------
    # events_per_day_{7,30}d
    # -----------------------------------------------------------------------
    ev7_agg = (
        ev_7.group_by("customer_id")
        .agg(pl.len().alias("ev_cnt_7d"))
        .with_columns((pl.col("ev_cnt_7d") / 7.0).alias("events_per_day_7d"))
    )
    ev30_agg = (
        ev_30.group_by("customer_id")
        .agg(pl.len().alias("ev_cnt_30d"))
        .with_columns((pl.col("ev_cnt_30d") / 30.0).alias("events_per_day_30d"))
    )

    result = (
        user_base
        .join(ev7_agg.select(["customer_id", "events_per_day_7d"]), on="customer_id", how="left")
        .join(ev30_agg.select(["customer_id", "events_per_day_30d"]), on="customer_id", how="left")
    )
    result = result.with_columns(
        pl.col("events_per_day_7d").fill_null(0.0),
        pl.col("events_per_day_30d").fill_null(0.0),
    )

    # -----------------------------------------------------------------------
    # unique_processes_30d
    # -----------------------------------------------------------------------
    unique_proc = (
        ev_30.group_by("customer_id")
        .agg(pl.col("process_code").n_unique().alias("unique_processes_30d"))
    )
    result = result.join(unique_proc, on="customer_id", how="left").with_columns(
        pl.col("unique_processes_30d").fill_null(0)
    )

    # -----------------------------------------------------------------------
    # Status rates (error, declined, completion)
    # -----------------------------------------------------------------------
    status_agg = (
        ev_30.group_by("customer_id")
        .agg(
            pl.len().alias("total_events"),
            (pl.col("status").str.to_uppercase() == "ERROR").cast(pl.Int32).sum().alias("error_cnt"),
            (pl.col("status").str.to_uppercase() == "DECLINED").cast(pl.Int32).sum().alias("declined_cnt"),
            (pl.col("status").str.to_uppercase() == "COMPLETED").cast(pl.Int32).sum().alias("completed_cnt"),
        )
        .with_columns(
            (pl.col("error_cnt") / pl.col("total_events")).alias("error_rate_30d"),
            (pl.col("declined_cnt") / pl.col("total_events")).alias("declined_rate_30d"),
            (pl.col("completed_cnt") / pl.col("total_events")).alias("completion_rate_30d"),
        )
    )
    result = result.join(
        status_agg.select(["customer_id", "error_rate_30d", "declined_rate_30d", "completion_rate_30d"]),
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------------------------
    # avg_process_duration_sec (for COMPLETED events with both timestamps)
    # -----------------------------------------------------------------------
    completed_ev = ev_30.filter(
        (pl.col("status").str.to_uppercase() == "COMPLETED")
        & pl.col("started_at").is_not_null()
        & pl.col("completed_at").is_not_null()
    ).with_columns(
        (pl.col("completed_at") - pl.col("started_at")).dt.total_seconds().alias("duration_sec")
    ).filter(pl.col("duration_sec") >= 0)

    duration_agg = (
        completed_ev.group_by("customer_id")
        .agg(pl.col("duration_sec").mean().alias("avg_process_duration_sec"))
    )
    result = result.join(duration_agg, on="customer_id", how="left")

    # -----------------------------------------------------------------------
    # weekend_share_30d
    # -----------------------------------------------------------------------
    weekend_agg = (
        ev_30.with_columns(
            pl.col("started_at").dt.weekday().alias("weekday")
        )
        .group_by("customer_id")
        .agg(
            pl.len().alias("total"),
            # weekday 5=Saturday, 6=Sunday (Polars Monday=0)
            (pl.col("weekday").is_in([5, 6])).cast(pl.Int32).sum().alias("weekend_cnt"),
        )
        .with_columns(
            (pl.col("weekend_cnt") / pl.col("total")).alias("weekend_share_30d")
        )
    )
    result = result.join(
        weekend_agg.select(["customer_id", "weekend_share_30d"]),
        on="customer_id",
        how="left",
    )

    # -----------------------------------------------------------------------
    # lang shares
    # -----------------------------------------------------------------------
    lang_agg = (
        ev_30.group_by("customer_id")
        .agg(
            pl.len().alias("lang_total"),
            (pl.col("lang").str.to_uppercase() == "RU").cast(pl.Int32).sum().alias("ru_cnt"),
            (pl.col("lang").str.to_uppercase() == "KZ").cast(pl.Int32).sum().alias("kz_cnt"),
        )
        .with_columns(
            (pl.col("ru_cnt") / pl.col("lang_total")).alias("lang_ru_share"),
            (pl.col("kz_cnt") / pl.col("lang_total")).alias("lang_kz_share"),
        )
    )
    result = result.join(
        lang_agg.select(["customer_id", "lang_ru_share", "lang_kz_share"]),
        on="customer_id",
        how="left",
    )

    logger.info(
        "Engagement features built: %d users, %d columns",
        result.height, result.width,
    )
    return result
