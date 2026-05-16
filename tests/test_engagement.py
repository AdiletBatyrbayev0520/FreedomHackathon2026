"""
tests/test_engagement.py
=========================
Task 5 tests — engagement features from app events.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest


@pytest.fixture
def cutoff():
    return date(2026, 4, 1)


def test_engagement_one_row_per_user(events_mini, cutoff):
    from src.features.engagement import build_engagement_features

    result = build_engagement_features(events_mini, cutoff)
    n_unique = events_mini["customer_id"].n_unique()
    assert result.height == n_unique


def test_error_rate_in_0_1(events_mini, cutoff):
    from src.features.engagement import build_engagement_features

    result = build_engagement_features(events_mini, cutoff)
    vals = result["error_rate_30d"].drop_nulls()
    assert (vals >= 0).all() and (vals <= 1).all()


def test_completion_rate_in_0_1(events_mini, cutoff):
    from src.features.engagement import build_engagement_features

    result = build_engagement_features(events_mini, cutoff)
    vals = result["completion_rate_30d"].drop_nulls()
    assert (vals >= 0).all() and (vals <= 1).all()


def test_lang_shares_in_0_1(events_mini, cutoff):
    from src.features.engagement import build_engagement_features

    result = build_engagement_features(events_mini, cutoff)
    for col in ("lang_ru_share", "lang_kz_share"):
        vals = result[col].drop_nulls()
        assert (vals >= 0).all() and (vals <= 1).all()


def test_events_per_day_non_negative(events_mini, cutoff):
    from src.features.engagement import build_engagement_features

    result = build_engagement_features(events_mini, cutoff)
    for col in ("events_per_day_7d", "events_per_day_30d"):
        vals = result[col].fill_null(0)
        assert (vals >= 0).all()


def test_required_engagement_columns(events_mini, cutoff):
    from src.features.engagement import build_engagement_features

    result = build_engagement_features(events_mini, cutoff)
    required = [
        "events_per_day_7d", "events_per_day_30d",
        "unique_processes_30d",
        "error_rate_30d", "declined_rate_30d", "completion_rate_30d",
        "avg_process_duration_sec",
        "weekend_share_30d",
        "lang_ru_share", "lang_kz_share",
    ]
    missing = [c for c in required if c not in result.columns]
    assert not missing, f"Missing engagement columns: {missing}"
