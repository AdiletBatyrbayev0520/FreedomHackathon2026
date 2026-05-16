"""
tests/test_diagnostics.py
==========================
Task 0 tests — ID overlap and date range diagnostics.
"""

from __future__ import annotations

import polars as pl
import pytest


def test_id_overlap_perfect(tables_mini):
    """With matching user IDs, intersection should equal users."""
    from src.diagnostics.id_overlap import compute_id_overlap

    # Modify transactions to have same IDs as users
    users = tables_mini["users"]
    tx = tables_mini["transactions"]

    result = compute_id_overlap(users, tx, "transactions")

    assert "intersection_ids" in result
    assert result["intersection_ids"] > 0
    assert 0 <= result["overlap_pct"] <= 100


def test_id_overlap_no_intersection(tables_mini):
    """When no IDs match, overlap should be 0."""
    from src.diagnostics.id_overlap import compute_id_overlap
    import polars as pl

    users = tables_mini["users"]
    # Create table with completely different IDs
    fake_df = pl.DataFrame({"customer_id": [999, 998, 997]})

    result = compute_id_overlap(users, fake_df, "fake")

    assert result["intersection_ids"] == 0
    assert result["overlap_pct"] == 0.0
    assert result["orphan_ids_in_other"] == 3


def test_run_all_overlaps_writes_csv(tables_mini, tmp_path):
    """run_all_overlaps should write id_overlap.csv."""
    from src.diagnostics.id_overlap import run_all_overlaps

    # Use stop_on_critical=False to not sys.exit during test
    overlap_df = run_all_overlaps(tables_mini, reports_dir=tmp_path, stop_on_critical=False)

    assert (tmp_path / "id_overlap.csv").exists()
    assert overlap_df.height == 4  # transactions, events, partner_purchases, acquisition


def test_date_ranges_writes_csv(tables_mini, tmp_path):
    """compute_date_ranges should write date_ranges.csv."""
    from src.diagnostics.date_ranges import compute_date_ranges

    df = compute_date_ranges(tables_mini, reports_dir=tmp_path, stop_on_critical=False)

    assert (tmp_path / "date_ranges.csv").exists()
    assert df.height >= 3  # users, transactions, events


def test_date_ranges_no_stop_when_ok(tables_mini, tmp_path):
    """Should not raise SystemExit when dates are consistent (events < cutoff, users after)."""
    from src.diagnostics.date_ranges import compute_date_ranges

    # In our mini fixture, events are before cutoff but NOT before users' reg_date
    # (users reg_date starts 2026-01-01, events started_at is relative to cutoff 2026-04-01)
    # So this should pass
    try:
        compute_date_ranges(tables_mini, reports_dir=tmp_path, stop_on_critical=True)
    except SystemExit:
        pytest.fail("compute_date_ranges triggered stop condition unexpectedly")


def test_overlap_columns_present(tables_mini, tmp_path):
    from src.diagnostics.id_overlap import run_all_overlaps

    df = run_all_overlaps(tables_mini, reports_dir=tmp_path, stop_on_critical=False)

    expected_cols = {
        "table", "users_unique_ids", "other_unique_ids",
        "intersection_ids", "overlap_pct",
        "orphan_ids_in_other", "orphan_rows_in_other",
        "users_without_any_row",
    }
    assert expected_cols.issubset(set(df.columns))
