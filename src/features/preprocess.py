"""
src/features/preprocess.py
===========================
Task 6 — Preprocessing: log1p transformation, RobustScaler, missing imputation.

Key rules:
  - Scaler is FIT ONLY on train data (T_minus_2 cutoff)
  - Transform is applied to val and test without re-fitting
  - Missing value flags: {col}_was_missing added for imputed columns
  - Preprocessors saved to data/interim/preprocessors.pkl
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# Columns to apply log1p transform (monetary + frequency)
LOG1P_COLS_PATTERNS = [
    "monetary_sum", "monetary_median", "cashback_earned",
    "partner_purchases_sum", "frequency_tx", "events_per_day",
    "avg_process_duration",
]


def _is_log1p_candidate(col: str) -> bool:
    return any(p in col for p in LOG1P_COLS_PATTERNS)


def apply_log1p(df: pl.DataFrame, cols: list[str] | None = None) -> pl.DataFrame:
    """Apply log1p to monetary/frequency columns (in-place rename with _log suffix)."""
    if cols is None:
        cols = [c for c in df.columns if _is_log1p_candidate(c) and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    for col in cols:
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.col(col).map_elements(
                lambda v: float(np.log1p(max(v, 0))) if v is not None else None,
                return_dtype=pl.Float64,
            ).alias(col)
        )

    return df


def fit_scaler(
    df: pl.DataFrame,
    cols: list[str] | None = None,
) -> tuple[RobustScaler, list[str]]:
    """
    Fit RobustScaler on numeric columns of df (should be train set).
    Returns (fitted_scaler, list_of_scaled_cols).
    """
    if cols is None:
        cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
            and c != "customer_id"
        ]

    # Convert to numpy, fill nulls with 0 temporarily for fitting
    X = df.select(cols).fill_null(0).to_numpy().astype(np.float64)

    scaler = RobustScaler()
    scaler.fit(X)
    logger.info("RobustScaler fitted on %d rows × %d cols (train set)", X.shape[0], X.shape[1])
    return scaler, cols


def transform_scaler(
    df: pl.DataFrame,
    scaler: RobustScaler,
    cols: list[str],
) -> pl.DataFrame:
    """Apply fitted RobustScaler to df columns."""
    X = df.select(cols).fill_null(0).to_numpy().astype(np.float64)
    X_scaled = scaler.transform(X)

    for i, col in enumerate(cols):
        df = df.with_columns(
            pl.Series(name=col, values=X_scaled[:, i])
        )
    return df


def impute_missing(
    df: pl.DataFrame,
    cols: list[str] | None = None,
    medians: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """
    Impute nulls with median (computed from df if medians not provided).
    Adds {col}_was_missing binary flag for each imputed column.
    Returns (df_imputed, medians_dict).
    """
    if cols is None:
        cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            and c != "customer_id"
            and df[c].null_count() > 0
        ]

    computed_medians = {} if medians is None else medians

    for col in cols:
        if col not in df.columns:
            continue
        null_count = df[col].null_count()
        if null_count == 0:
            continue

        # Compute or use stored median
        if col not in computed_medians:
            med = df[col].drop_nulls().median()
            computed_medians[col] = float(med) if med is not None else 0.0

        med_val = computed_medians[col]

        # Add was_missing flag
        flag_col = f"{col}_was_missing"
        if flag_col not in df.columns:
            df = df.with_columns(
                pl.col(col).is_null().alias(flag_col)
            )

        # Impute
        df = df.with_columns(
            pl.col(col).fill_null(med_val)
        )

    return df, computed_medians


def check_no_inf_nan(df: pl.DataFrame) -> None:
    """Assert no inf or NaN in numeric columns after preprocessing."""
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in (pl.Float64, pl.Float32)
    ]
    for col in numeric_cols:
        s = df[col].drop_nulls()
        if s.is_nan().any() or s.is_infinite().any():
            logger.error("Column '%s' contains NaN or Inf after preprocessing!", col)
            raise ValueError(f"Column '{col}' has NaN or Inf values")


def save_preprocessors(
    scaler: RobustScaler,
    scaled_cols: list[str],
    medians: dict[str, float],
    path: str | Path = "data/interim/preprocessors.pkl",
) -> None:
    """Save scaler + medians to pkl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scaler": scaler,
        "scaled_cols": scaled_cols,
        "medians": medians,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Preprocessors saved to %s", path)


def load_preprocessors(
    path: str | Path = "data/interim/preprocessors.pkl",
) -> tuple[RobustScaler, list[str], dict[str, float]]:
    """Load preprocessors from pkl."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["scaler"], payload["scaled_cols"], payload["medians"]
