"""
tests/test_time_split.py
=========================
Task 2 tests — cutoff logic and apply_cutoff filtering.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest


def test_define_cutoffs_returns_required_keys(tables_mini):
    from src.features.time_split import define_cutoffs
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmpdir:
        cutoffs = define_cutoffs(
            tables_mini["transactions"],
            tables_mini["events"],
            data_interim_dir=pathlib.Path(tmpdir) / "interim",
            reports_dir=pathlib.Path(tmpdir) / "reports",
        )

    for key in ("T", "T_minus_1", "T_minus_2", "strategy", "label_window_days"):
        assert key in cutoffs, f"Missing key: {key}"


def test_cutoffs_are_strictly_ordered(tables_mini):
    from src.features.time_split import define_cutoffs
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmpdir:
        cutoffs = define_cutoffs(
            tables_mini["transactions"],
            tables_mini["events"],
            data_interim_dir=pathlib.Path(tmpdir) / "interim",
            reports_dir=pathlib.Path(tmpdir) / "reports",
        )

    T = date.fromisoformat(cutoffs["T"])
    T1 = date.fromisoformat(cutoffs["T_minus_1"])
    T2 = date.fromisoformat(cutoffs["T_minus_2"])

    assert T2 < T1 < T, f"Cutoff ordering violated: T2={T2}  T1={T1}  T={T}"


def test_apply_cutoff_strict_less_than(transactions_mini):
    from src.features.time_split import apply_cutoff

    cutoff = date(2026, 3, 15)
    filtered = apply_cutoff(transactions_mini, cutoff, "transaction_date")

    # No row should have date >= cutoff
    assert filtered.filter(
        pl.col("transaction_date") >= pl.lit(cutoff).cast(pl.Date)
    ).is_empty(), "apply_cutoff returned rows at or after cutoff"


def test_apply_cutoff_preserves_earlier_rows(transactions_mini):
    from src.features.time_split import apply_cutoff

    cutoff = date(2026, 4, 2)  # after all txs in mini fixture
    filtered = apply_cutoff(transactions_mini, cutoff, "transaction_date")

    # Should return all rows (all before cutoff)
    assert filtered.height == transactions_mini.height


def test_apply_cutoff_removes_all_if_before_data(transactions_mini):
    from src.features.time_split import apply_cutoff

    cutoff = date(2020, 1, 1)  # before all data
    filtered = apply_cutoff(transactions_mini, cutoff, "transaction_date")

    assert filtered.height == 0, "Expected empty DataFrame when cutoff is before all data"


def test_cutoffs_json_saved(tables_mini):
    from src.features.time_split import define_cutoffs
    import tempfile, pathlib, json

    with tempfile.TemporaryDirectory() as tmpdir:
        interim = pathlib.Path(tmpdir) / "interim"
        define_cutoffs(
            tables_mini["transactions"],
            tables_mini["events"],
            data_interim_dir=interim,
            reports_dir=pathlib.Path(tmpdir) / "reports",
        )

        json_path = interim / "cutoffs.json"
        assert json_path.exists(), "cutoffs.json was not created"

        with open(json_path) as f:
            data = json.load(f)
        assert "T" in data
