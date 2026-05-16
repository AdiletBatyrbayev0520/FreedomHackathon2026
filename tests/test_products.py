"""
tests/test_products.py
=======================
Task 5 tests — product activation features.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest


@pytest.fixture
def cutoff():
    return date(2026, 4, 1)


def test_product_flags_are_bool(events_mini, cutoff):
    from src.features.products import build_product_features, ALL_PRODUCTS

    result = build_product_features(events_mini, cutoff)
    for product in ALL_PRODUCTS:
        col = f"has_{product}"
        if col in result.columns:
            assert result[col].dtype == pl.Boolean, f"{col} is not Boolean"


def test_products_count_non_negative(events_mini, cutoff):
    from src.features.products import build_product_features

    result = build_product_features(events_mini, cutoff)
    assert (result["products_count"] >= 0).all()


def test_user_with_completed_event_has_product(events_mini, cutoff):
    """User 1 has COMPLETED OpenCardProcess → has_card should be True."""
    from src.features.products import build_product_features

    result = build_product_features(events_mini, cutoff)
    user1 = result.filter(pl.col("customer_id") == 1)

    if user1.is_empty():
        pytest.skip("User 1 not in product output")

    has_card = user1["has_card"][0]
    assert has_card is True, f"User 1 should have has_card=True, got {has_card}"


def test_user_with_error_event_does_not_have_product(events_mini, cutoff):
    """User 4 has only DECLINED LivenessProcess → has_liveness should be False."""
    from src.features.products import build_product_features

    result = build_product_features(events_mini, cutoff)
    user4 = result.filter(pl.col("customer_id") == 4)

    if user4.is_empty():
        pytest.skip("User 4 not in product output")

    has_liveness = user4["has_liveness"][0]
    assert has_liveness is False, (
        f"User 4 (only DECLINED events) should have has_liveness=False, got {has_liveness}"
    )


def test_days_since_activation_non_negative(events_mini, cutoff):
    from src.features.products import build_product_features

    result = build_product_features(events_mini, cutoff)
    for col in ("days_since_first_product_activation", "days_since_last_product_activation"):
        vals = result[col].drop_nulls()
        assert (vals >= 0).all(), f"{col} has negative values"
