"""
tests/test_rfm.py
==================
Task 3 tests — RFM temporal features.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest


@pytest.fixture
def cutoff():
    return date(2026, 4, 1)


def test_rfm_output_has_one_row_per_user(transactions_mini, events_mini, cutoff):
    from src.features.rfm_temporal import build_rfm_features

    result = build_rfm_features(transactions_mini, events_mini, cutoff)
    n_users = transactions_mini["customer_id"].n_unique()
    assert result.height == n_users, (
        f"Expected {n_users} rows, got {result.height}"
    )


def test_rfm_frequency_count_non_negative(transactions_mini, events_mini, cutoff):
    """frequency_tx_* and active_days_* counts must be non-negative.
    frequency_slope_4w is intentionally excluded — it CAN be negative for declining users."""
    from src.features.rfm_temporal import build_rfm_features

    result = build_rfm_features(transactions_mini, events_mini, cutoff)
    # Only count columns should be non-negative (not slope)
    count_cols = [c for c in result.columns if c.startswith("frequency_tx_") or c.startswith("active_days_")]
    for col in count_cols:
        assert (result[col].fill_null(0) >= 0).all(), f"Column {col} has negative values"


def test_rfm_user_without_success_tx_has_zero_frequency(transactions_mini, events_mini, cutoff):
    """User 4 has only failed txs → frequency should be 0 in 30d."""
    from src.features.rfm_temporal import build_rfm_features

    result = build_rfm_features(transactions_mini, events_mini, cutoff)
    user4 = result.filter(pl.col("customer_id") == 4)

    if user4.is_empty():
        pytest.skip("User 4 not in RFM output")

    freq_30 = user4["frequency_tx_30d"][0]
    assert freq_30 == 0, f"User 4 should have 0 successful txs in 30d, got {freq_30}"


def test_rfm_user_with_high_activity_has_larger_frequency(transactions_mini, events_mini, cutoff):
    """User 1 (10 txs) should have higher frequency_tx_90d than user 2 (5 txs)."""
    from src.features.rfm_temporal import build_rfm_features

    result = build_rfm_features(transactions_mini, events_mini, cutoff)

    u1 = result.filter(pl.col("customer_id") == 1)["frequency_tx_90d"]
    u2 = result.filter(pl.col("customer_id") == 2)["frequency_tx_90d"]

    if u1.is_empty() or u2.is_empty():
        pytest.skip("Users 1 or 2 not in RFM output")

    assert u1[0] > u2[0], f"User 1 freq={u1[0]} should be > User 2 freq={u2[0]}"


def test_rfm_monetary_sums_non_negative(transactions_mini, events_mini, cutoff):
    """monetary_sum_* and monetary_median_* must be non-negative.
    monetary_delta_mom is intentionally excluded — negative = user spent less than prior month."""
    from src.features.rfm_temporal import build_rfm_features

    result = build_rfm_features(transactions_mini, events_mini, cutoff)
    sum_cols = [c for c in result.columns if c.startswith("monetary_sum_") or c.startswith("monetary_median_")]
    for col in sum_cols:
        vals = result[col].fill_null(0)
        assert (vals >= 0).all(), f"Column {col} has negative values"


def test_rfm_required_columns_present(transactions_mini, events_mini, cutoff):
    from src.features.rfm_temporal import build_rfm_features

    result = build_rfm_features(transactions_mini, events_mini, cutoff)
    required = [
        "frequency_tx_7d", "frequency_tx_30d", "frequency_tx_90d",
        "monetary_sum_30d", "monetary_sum_90d",
        "frequency_slope_4w", "monetary_delta_mom",
        "active_days_7d", "active_days_30d",
    ]
    missing = [c for c in required if c not in result.columns]
    assert not missing, f"Missing RFM columns: {missing}"
