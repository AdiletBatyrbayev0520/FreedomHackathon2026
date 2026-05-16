"""
src/models/utils.py
====================
Shared utilities for model training, evaluation, save/load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

logger = logging.getLogger(__name__)


def save_model(model, path: str | Path) -> None:
    """Save CatBoost model to .cbm file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Model saved to %s", path)


def load_regressor(path: str | Path) -> CatBoostRegressor:
    """Load CatBoostRegressor from .cbm file."""
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_classifier(path: str | Path) -> CatBoostClassifier:
    """Load CatBoostClassifier from .cbm file."""
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Save metrics dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", path)


def get_numeric_and_cat_cols(
    feature_df,
    cat_cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Separate feature columns into numeric and categorical.

    Parameters
    ----------
    feature_df  : polars DataFrame
    cat_cols    : explicitly specified categorical column names
    exclude_cols: columns to exclude (e.g. customer_id, label cols)

    Returns (numeric_cols, cat_cols)
    """
    import polars as pl

    exclude = set(exclude_cols or [])
    if cat_cols is None:
        cat_cols = []

    explicit_cats = set(cat_cols)
    all_cols = set(feature_df.columns) - exclude

    numeric_cols = []
    final_cat_cols = []

    for col in sorted(all_cols):
        if col in explicit_cats:
            final_cat_cols.append(col)
        elif feature_df[col].dtype in (pl.Utf8, pl.Categorical):
            final_cat_cols.append(col)
        else:
            numeric_cols.append(col)

    return numeric_cols, final_cat_cols


def prepare_xy(
    feature_df,
    label_col: str,
    exclude_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> tuple[Any, Any, list[str], list[str]]:
    """
    Prepare X (pandas DataFrame) and y (numpy array) for CatBoost.
    Returns (X, y, feature_names, cat_feature_names).
    """
    import polars as pl
    import pandas as pd

    from src.features.validation import FEATURE_COLUMNS

    exclude = set(exclude_cols or []) | {"customer_id", label_col}

    # Intersect with whitelist to ensure NO labels get through
    valid_features = [c for c in feature_df.columns if c in FEATURE_COLUMNS and c not in exclude]

    _, cat_feature_names = get_numeric_and_cat_cols(
        feature_df, cat_cols=cat_cols, exclude_cols=list(set(feature_df.columns) - set(valid_features))
    )

    X = feature_df.select(valid_features).to_pandas()

    # CatBoost needs string type for categoricals
    for col in cat_feature_names:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("Unknown")

    y = feature_df[label_col].to_numpy()

    return X, y, valid_features, cat_feature_names
