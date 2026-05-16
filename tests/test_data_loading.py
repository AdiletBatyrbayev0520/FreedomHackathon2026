"""
tests/test_data_loading.py
===========================
Task 1 tests — data loading validation.
"""

from __future__ import annotations

import polars as pl
import pytest


def test_load_all_returns_five_tables():
    from src.data_loading import load_all
    tables = load_all("data/processed")
    assert set(tables.keys()) == {"users", "transactions", "events", "partner_purchases", "acquisition"}


def test_customer_id_is_int64_everywhere():
    from src.data_loading import load_all
    tables = load_all("data/processed")
    for name, df in tables.items():
        assert df["customer_id"].dtype == pl.Int64, (
            f"[{name}] customer_id dtype is {df['customer_id'].dtype}, expected Int64"
        )


def test_no_null_customer_ids():
    from src.data_loading import load_all
    tables = load_all("data/processed")
    for name, df in tables.items():
        null_count = df["customer_id"].null_count()
        assert null_count == 0, f"[{name}] has {null_count} null customer_ids"


def test_transaction_status_clean_values():
    from src.data_loading import load_transactions
    df = load_transactions("data/processed")
    assert "transaction_status_clean" in df.columns
    valid_values = {"success", "failure", "unknown"}
    actual_values = set(df["transaction_status_clean"].drop_nulls().unique().to_list())
    assert actual_values.issubset(valid_values), (
        f"Unexpected values in transaction_status_clean: {actual_values - valid_values}"
    )


def test_users_row_count():
    """Users should have ~522k rows after dedup."""
    from src.data_loading import load_users
    df = load_users("data/processed")
    assert df.height > 500_000, f"Users has only {df.height} rows, expected >500k"
    assert df.height < 600_000, f"Users has {df.height} rows, expected <600k"


def test_partner_purchases_row_count():
    """Partner purchases should have ~207k rows."""
    from src.data_loading import load_partner_purchases
    df = load_partner_purchases("data/processed")
    assert df.height > 200_000, f"Partner purchases has only {df.height} rows"
    assert df.height < 220_000


def test_reg_date_is_date_type():
    from src.data_loading import load_users
    df = load_users("data/processed")
    assert df["reg_date"].dtype == pl.Date, f"reg_date dtype={df['reg_date'].dtype}"


def test_transaction_date_is_date_type():
    from src.data_loading import load_transactions
    df = load_transactions("data/processed")
    assert df["transaction_date"].dtype == pl.Date


def test_events_datetime_types():
    from src.data_loading import load_events
    df = load_events("data/processed")
    for col in ("started_at", "completed_at"):
        assert str(df[col].dtype).startswith("Datetime"), (
            f"Column {col} has dtype {df[col].dtype}, expected Datetime"
        )


def test_transaction_sum_was_missing_is_boolean():
    from src.data_loading import load_transactions
    df = load_transactions("data/processed")
    assert df["transaction_sum_was_missing"].dtype == pl.Boolean
