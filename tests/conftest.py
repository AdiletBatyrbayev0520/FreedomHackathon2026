"""
tests/conftest.py
==================
Shared pytest fixtures: synthetic mini-datasets for fast unit tests.
All fixtures use small row counts (5-20 users, 20-200 rows).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

# ─── Base users fixture ───────────────────────────────────────────────────────

@pytest.fixture
def users_mini() -> pl.DataFrame:
    return pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "city": ["Almaty", "Astana", "Almaty", "Shymkent", "Astana"],
        "gender": ["M", "F", "F", "M", "F"],
        "customer_age": [25, 32, 28, 45, 19],
        "reg_date": [
            date(2026, 1, 1),
            date(2026, 1, 15),
            date(2026, 2, 1),
            date(2026, 2, 15),
            date(2026, 3, 1),
        ],
    })


# ─── Transactions fixture ─────────────────────────────────────────────────────

@pytest.fixture
def transactions_mini() -> pl.DataFrame:
    """
    5 users, transactions spanning 90 days before cutoff (2026-04-01).
    User 1: high activity (10 txs)
    User 2: medium (5 txs)
    User 3: 1 tx only
    User 4: no successful txs (only failed)
    User 5: no txs at all
    """
    cutoff = date(2026, 4, 1)
    rows = []
    tx_id = 1

    # User 1: 10 successful txs spread over 90 days
    for d in range(0, 90, 9):
        rows.append({
            "customer_id": 1,
            "transaction_id": str(tx_id),
            "transaction_date": cutoff - timedelta(days=d),
            "operation_type": "Purchase",
            "transaction_status": "Успешные",
            "transaction_status_clean": "success",
            "mcc": "5411",
            "terminal_type": "POS",
            "transaction_sum": 10000.0 + d * 100,
            "transaction_sum_was_missing": False,
        })
        tx_id += 1

    # User 2: 5 successful txs in last 30 days
    for d in range(0, 30, 6):
        rows.append({
            "customer_id": 2,
            "transaction_id": str(tx_id),
            "transaction_date": cutoff - timedelta(days=d),
            "operation_type": "Purchase",
            "transaction_status": "Успешные",
            "transaction_status_clean": "success",
            "mcc": "5812",
            "terminal_type": "ePOS",
            "transaction_sum": 5000.0,
            "transaction_sum_was_missing": False,
        })
        tx_id += 1

    # User 3: 1 tx at 60 days ago
    rows.append({
        "customer_id": 3,
        "transaction_id": str(tx_id),
        "transaction_date": cutoff - timedelta(days=60),
        "operation_type": "P2P Credit",
        "transaction_status": "Успешные",
        "transaction_status_clean": "success",
        "mcc": "6536",
        "terminal_type": "POS",
        "transaction_sum": 2000.0,
        "transaction_sum_was_missing": True,
    })
    tx_id += 1

    # User 4: only failed txs
    rows.append({
        "customer_id": 4,
        "transaction_id": str(tx_id),
        "transaction_date": cutoff - timedelta(days=10),
        "operation_type": "Purchase",
        "transaction_status": "AUSR0401",
        "transaction_status_clean": "failure",
        "mcc": "5411",
        "terminal_type": "POS",
        "transaction_sum": 3000.0,
        "transaction_sum_was_missing": False,
    })
    tx_id += 1

    # User 5: no txs (absent from this table)

    return pl.DataFrame(rows).with_columns(
        pl.col("transaction_date").cast(pl.Date),
        pl.col("customer_id").cast(pl.Int64),
        pl.col("transaction_sum").cast(pl.Float64),
        pl.col("transaction_sum_was_missing").cast(pl.Boolean),
    )


# ─── Events fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def events_mini() -> pl.DataFrame:
    """App events for 4 users (user 5 has none)."""
    cutoff = date(2026, 4, 1)
    rows = []

    events_data = [
        (1, "OpenCardProcess", "COMPLETED", "RU", 5),
        (1, "LivenessProcess", "COMPLETED", "RU", 10),
        (1, "FreedomRatingActivationProcess", "COMPLETED", "RU", 20),
        (2, "OpenCardProcess", "COMPLETED", "KZ", 3),
        (2, "LivenessProcess", "ERROR", "KZ", 8),
        (3, "FrhcActivationProcess", "COMPLETED", "RU", 45),
        (4, "LivenessProcess", "DECLINED", "RU", 15),
    ]

    for cid, process_code, status, lang, days_ago in events_data:
        started = datetime(
            cutoff.year, cutoff.month, cutoff.day
        ) - timedelta(days=days_ago)
        completed = started + timedelta(minutes=5)
        rows.append({
            "customer_id": cid,
            "process_code": process_code,
            "lang": lang,
            "status": status,
            "started_at": started,
            "completed_at": completed,
        })

    return pl.DataFrame(rows).with_columns(
        pl.col("customer_id").cast(pl.Int64),
        pl.col("started_at").cast(pl.Datetime),
        pl.col("completed_at").cast(pl.Datetime),
    )


# ─── Partner purchases fixture ────────────────────────────────────────────────

@pytest.fixture
def partner_purchases_mini() -> pl.DataFrame:
    cutoff = date(2026, 4, 1)
    return pl.DataFrame({
        "customer_id": [1, 1, 2, 3],
        "purchase_date": [
            cutoff - timedelta(days=5),
            cutoff - timedelta(days=50),
            cutoff - timedelta(days=20),
            cutoff - timedelta(days=80),
        ],
        "remote_flag": ["Оплата в Superapp"] * 4,
        "counter": [1, 1, 1, 1],
        "purchase_amount_real": [15000.0, 8000.0, 12000.0, 5000.0],
        "cashback_amount_real": [300.0, 160.0, 240.0, 100.0],
        "cashback_rate": [0.02, 0.02, 0.02, 0.02],
        "app_name_normalized": ["arbuz", "ticketon", "arbuz", "train_tickets"],
    }).with_columns(
        pl.col("customer_id").cast(pl.Int64),
        pl.col("purchase_date").cast(pl.Date),
    )


# ─── Acquisition fixture ──────────────────────────────────────────────────────

@pytest.fixture
def acquisition_mini() -> pl.DataFrame:
    return pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "secondary_category_filled": ["organic", "bank", "Ticketon", "organic", "bank"],
    }).with_columns(pl.col("customer_id").cast(pl.Int64))


# ─── All tables fixture ───────────────────────────────────────────────────────

@pytest.fixture
def tables_mini(
    users_mini, transactions_mini, events_mini,
    partner_purchases_mini, acquisition_mini,
) -> dict:
    return {
        "users": users_mini,
        "transactions": transactions_mini,
        "events": events_mini,
        "partner_purchases": partner_purchases_mini,
        "acquisition": acquisition_mini,
    }


# ─── Standard cutoff ─────────────────────────────────────────────────────────

@pytest.fixture
def cutoff_date() -> date:
    return date(2026, 4, 1)
