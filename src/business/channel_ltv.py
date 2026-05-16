"""
src/business/channel_ltv.py
=============================
Task 15 — Honest channel metrics (no fake ROMI).

Instead of fabricating ROMI = 5681%, compute:
  - Median Freedom Score by channel
  - Retention survival curves by channel (7/30/60/90 days)
  - Time to first transaction by channel
  - Cohort Freedom Score quartile distribution by channel

Outputs:
  reports/channel_summary.csv
  reports/figures/channel_retention.png
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def compute_channel_ltv(
    users: pl.DataFrame,
    acquisition: pl.DataFrame,
    transactions: pl.DataFrame,
    freedom_scores: pl.DataFrame | None = None,
    cutoff=None,
    reports_dir: str | Path = "reports",
) -> pl.DataFrame:
    """
    Compute honest channel metrics and retention curves.

    Parameters
    ----------
    users         : DataFrame(customer_id, reg_date, ...)
    acquisition   : DataFrame(customer_id, secondary_category_filled)
    transactions  : DataFrame(customer_id, transaction_date, transaction_status_clean)
    freedom_scores: optional DataFrame(customer_id, freedom_score_pred)
    cutoff        : date or str — observation point for retention window
    """
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)

    logger.info("Computing channel LTV metrics...")

    # Join acquisition channel to users
    acq = acquisition.rename({"secondary_category_filled": "channel"}).select(
        ["customer_id", "channel"]
    ).unique(subset=["customer_id"])

    user_channel = users.select(["customer_id", "reg_date"]).join(acq, on="customer_id", how="left")

    # -----------------------------------------------------------------------
    # Channel user counts — flag low-sample channels
    # -----------------------------------------------------------------------
    channel_counts = (
        user_channel.group_by("channel")
        .agg(pl.len().alias("user_count"))
        .with_columns(
            (pl.col("user_count") < 100).alias("low_sample")
        )
        .sort("user_count", descending=True)
    )

    low_sample_channels = set(
        channel_counts.filter(pl.col("low_sample"))["channel"].drop_nulls().to_list()
    )
    if low_sample_channels:
        logger.warning(
            "Low-sample channels (n<100, excluded from insights): %s", low_sample_channels
        )

    # -----------------------------------------------------------------------
    # Retention curves: % of users with ≥1 successful tx in first N days
    # -----------------------------------------------------------------------
    retention_windows = [7, 30, 60, 90]
    retention_data = {}

    for window in retention_windows:
        col_name = f"retention_{window}d"
        # For each user: did they transact within 'window' days of registration?
        tx_by_user = (
            transactions.filter(pl.col("transaction_status_clean") == "success")
            .group_by("customer_id")
            .agg(pl.col("transaction_date").min().alias("first_tx_date"))
        )

        cohort = user_channel.join(tx_by_user, on="customer_id", how="left").with_columns(
            (
                (pl.col("first_tx_date") - pl.col("reg_date")).dt.total_days()
                <= window
            ).cast(pl.Int8).alias(col_name)
        )

        retention_by_channel = (
            cohort.filter(~pl.col("channel").is_in(list(low_sample_channels)))
            .filter(pl.col("channel").is_not_null())
            .group_by("channel")
            .agg(
                pl.col(col_name).mean().alias(col_name),
                pl.len().alias("n_users"),
            )
        )
        retention_data[window] = retention_by_channel

    # -----------------------------------------------------------------------
    # Time to first transaction
    # -----------------------------------------------------------------------
    tx_first = (
        transactions.filter(pl.col("transaction_status_clean") == "success")
        .group_by("customer_id")
        .agg(pl.col("transaction_date").min().alias("first_tx_date"))
    )

    ttft = user_channel.join(tx_first, on="customer_id", how="left").with_columns(
        (pl.col("first_tx_date") - pl.col("reg_date")).dt.total_days().alias("days_to_first_tx")
    )

    ttft_by_channel = (
        ttft.filter(~pl.col("channel").is_in(list(low_sample_channels)))
        .filter(pl.col("channel").is_not_null() & pl.col("days_to_first_tx").is_not_null())
        .group_by("channel")
        .agg(
            pl.col("days_to_first_tx").median().alias("median_days_to_first_tx"),
            pl.col("days_to_first_tx").mean().alias("mean_days_to_first_tx"),
        )
    )

    # -----------------------------------------------------------------------
    # Freedom Score by channel (if available)
    # -----------------------------------------------------------------------
    fs_by_channel = None
    if freedom_scores is not None:
        fs_col = "freedom_score_pred" if "freedom_score_pred" in freedom_scores.columns else "freedom_score_target"
        if fs_col in freedom_scores.columns:
            user_fs = user_channel.join(
                freedom_scores.select(["customer_id", fs_col]),
                on="customer_id",
                how="left",
            )
            fs_by_channel = (
                user_fs.filter(~pl.col("channel").is_in(list(low_sample_channels)))
                .filter(pl.col("channel").is_not_null())
                .group_by("channel")
                .agg(
                    pl.col(fs_col).median().alias("median_freedom_score"),
                    pl.col(fs_col).mean().alias("mean_freedom_score"),
                )
            )

    # -----------------------------------------------------------------------
    # Assemble summary table
    # -----------------------------------------------------------------------
    summary = channel_counts.clone()

    # Join retention data
    base_ret = retention_data[retention_windows[0]].rename({
        f"retention_{retention_windows[0]}d": f"retention_{retention_windows[0]}d",
        "n_users": "_n_drop",
    }).drop("_n_drop")

    summary = summary.join(base_ret, on="channel", how="left")
    for w in retention_windows[1:]:
        ret_df = retention_data[w].select(["channel", f"retention_{w}d"])
        summary = summary.join(ret_df, on="channel", how="left")

    summary = summary.join(ttft_by_channel, on="channel", how="left")

    if fs_by_channel is not None:
        summary = summary.join(fs_by_channel, on="channel", how="left")

    out_path = reports_dir / "channel_summary.csv"
    summary.write_csv(out_path)
    logger.info("Channel summary saved to %s", out_path)

    # -----------------------------------------------------------------------
    # Plot retention curves
    # -----------------------------------------------------------------------
    _plot_retention_curves(
        retention_data, retention_windows,
        low_sample_channels,
        figures_dir / "channel_retention.png",
    )

    return summary


def _plot_retention_curves(
    retention_data: dict,
    windows: list[int],
    low_sample_channels: set,
    out_path: Path,
) -> None:
    """Plot retention curves (line chart) per channel across time windows."""
    # Collect all channels
    all_channels = set()
    for w in windows:
        df = retention_data[w]
        all_channels.update(df.filter(~pl.col("channel").is_in(list(low_sample_channels)))["channel"].drop_nulls().to_list())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_channels)))

    for i, channel in enumerate(sorted(all_channels)):
        retention_vals = []
        for w in windows:
            df = retention_data[w]
            row = df.filter(pl.col("channel") == channel)
            val = float(row[f"retention_{w}d"][0]) if not row.is_empty() else np.nan
            retention_vals.append(val)

        ax.plot(windows, retention_vals, "o-", color=colors[i], label=channel, lw=2)

    ax.set_xlabel("Days after Registration")
    ax.set_ylabel("Retention Rate (% with ≥1 transaction)")
    ax.set_title("Channel Retention Curves (Honest — no fabricated ROMI)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Channel retention curves saved to %s", out_path)
