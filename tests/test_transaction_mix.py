"""
tests/test_transaction_mix.py
==============================
Task 4 tests — MCC shares and channel mix features.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.features.mcc_mapping import MACRO_CATEGORIES


@pytest.fixture
def cutoff():
    return date(2026, 4, 1)


def test_mcc_shares_sum_to_one_or_null(transactions_mini, cutoff):
    from src.features.transaction_mix import build_transaction_mix_features

    result = build_transaction_mix_features(transactions_mini, cutoff)
    share_cols = [f"mcc_share_{c}" for c in MACRO_CATEGORIES]

    # For users with transactions, shares should sum to ~1
    for row in result.to_dicts():
        share_sum = sum(row.get(col, 0) or 0 for col in share_cols if col in row)
        # Only check users who had successful txs (share_sum > 0)
        if share_sum > 1e-6:
            assert abs(share_sum - 1.0) < 1e-4, (
                f"User {row['customer_id']}: MCC shares sum to {share_sum:.6f}, expected ~1.0"
            )


def test_shares_in_0_1_range(transactions_mini, cutoff):
    from src.features.transaction_mix import build_transaction_mix_features

    result = build_transaction_mix_features(transactions_mini, cutoff)
    share_cols = [c for c in result.columns if "share" in c]

    for col in share_cols:
        vals = result[col].drop_nulls()
        if not vals.is_empty():
            assert (vals >= 0).all(), f"Column {col} has negative shares"
            assert (vals <= 1.0 + 1e-6).all(), f"Column {col} has share > 1"


def test_p2p_user_has_high_share(transactions_mini, cutoff):
    """User 3 has only P2P Credit tx → share_of_p2p should be 1.0."""
    from src.features.transaction_mix import build_transaction_mix_features

    result = build_transaction_mix_features(transactions_mini, cutoff)
    user3 = result.filter(pl.col("customer_id") == 3)

    if user3.is_empty():
        pytest.skip("User 3 not in mix output")

    p2p_share = user3["share_of_p2p"][0]
    assert p2p_share is not None and p2p_share >= 0.99, (
        f"User 3 (P2P only) should have share_of_p2p≈1.0, got {p2p_share}"
    )


def test_required_columns_present(transactions_mini, cutoff):
    from src.features.transaction_mix import build_transaction_mix_features

    result = build_transaction_mix_features(transactions_mini, cutoff)
    required = [
        "share_of_online", "share_of_p2p",
        "unique_terminals_30d", "imputed_share",
    ] + [f"mcc_share_{c}" for c in MACRO_CATEGORIES]

    missing = [c for c in required if c not in result.columns]
    assert not missing, f"Missing tx_mix columns: {missing}"


def test_imputed_share_correct(transactions_mini, cutoff):
    """User 3 has transaction_sum_was_missing=True → imputed_share > 0."""
    from src.features.transaction_mix import build_transaction_mix_features

    result = build_transaction_mix_features(transactions_mini, cutoff)
    user3 = result.filter(pl.col("customer_id") == 3)

    if user3.is_empty():
        pytest.skip("User 3 not in output")

    imputed = user3["imputed_share"][0]
    # User 3 has 1 tx with missing=True, so imputed_share should be > 0
    assert imputed is not None and imputed > 0, (
        f"User 3 should have imputed_share > 0, got {imputed}"
    )
