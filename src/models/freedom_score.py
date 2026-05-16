"""
src/models/freedom_score.py
============================
Task 8 — Freedom Score regression (CatBoostRegressor).

Metrics: RMSE, MAE, Spearman ρ, Pearson r, R²
Key check: Spearman ρ > 0.4 (rank-based, robust to tail compression)

Outputs:
  models/freedom_score.cbm
  reports/metrics_freedom_score.json
  reports/figures/freedom_score_eval.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.utils import get_numeric_and_cat_cols, prepare_xy, save_metrics, save_model

logger = logging.getLogger(__name__)

CAT_FEATURES = ["city", "channel", "gender"]
EXCLUDE_COLS = ["customer_id", "freedom_score_target"]


def train_freedom_score(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    label_col: str = "freedom_score_target",
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    random_state: int = 42,
) -> CatBoostRegressor:
    """
    Train CatBoostRegressor for Freedom Score prediction.

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames with features + label_col
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing training data for Freedom Score model...")
    X_train, y_train, feature_names, cat_feature_names = prepare_xy(
        train_df, label_col=label_col, cat_cols=CAT_FEATURES
    )
    X_val, y_val, _, _ = prepare_xy(
        val_df, label_col=label_col, cat_cols=CAT_FEATURES
    )
    X_test, y_test, _, _ = prepare_xy(
        test_df, label_col=label_col, cat_cols=CAT_FEATURES
    )

    logger.info(
        "Train: %d rows | Val: %d rows | Test: %d rows | Features: %d",
        len(X_train), len(X_val), len(X_test), len(feature_names),
    )

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=150,
        random_seed=random_state,
        verbose=200,
        cat_features=cat_feature_names,
        # ── Hardware acceleration ──────────────────────────────────────
        task_type="GPU",
        devices="0",                 # RTX 5090
        thread_count=-1,             # all 32 CPU cores for preprocessing
        # GPU-optimised tree building
        grow_policy="Lossguide",
        max_leaves=64,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        border_count=254,            # full precision (GPU can afford it)
        min_data_in_leaf=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    # Evaluate
    val_metrics = _evaluate_regressor(model, X_val, y_val, "val")
    test_metrics = _evaluate_regressor(model, X_test, y_test, "test")

    # Spearman check
    spearman = test_metrics.get("spearman_rho", 0)
    if spearman > 0.95:
        logger.warning(
            "[LEAKAGE CHECK] Spearman ρ=%.4f > 0.95 on test. Possible leakage — "
            "verify that freedom_score_target was not computed from features.",
            spearman,
        )
    elif spearman < 0.4:
        logger.warning(
            "[FLAG] Spearman ρ=%.4f < 0.4 on test. Model has weak rank discrimination. "
            "Consider tail-aware training (quantile regression, VIP segment).",
            spearman,
        )
    else:
        logger.info("[OK] Spearman ρ=%.4f on test (target: >0.4)", spearman)

    all_metrics = {"val": val_metrics, "test": test_metrics}
    save_metrics(all_metrics, reports_dir / "metrics_freedom_score.json")
    save_model(model, models_dir / "freedom_score.cbm")

    # Plots
    y_pred_test = model.predict(X_test)
    _plot_freedom_score_eval(y_test, y_pred_test, figures_dir / "freedom_score_eval.png")

    return model


def _evaluate_regressor(model, X, y, split_name: str) -> dict:
    y_pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    pearson_r, _ = pearsonr(y, y_pred)
    spearman_rho, _ = spearmanr(y, y_pred)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "split": split_name,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "r2": r2,
    }
    logger.info(
        "[%s] RMSE=%.4f  MAE=%.4f  Spearman=%.4f  R²=%.4f",
        split_name, rmse, mae, spearman_rho, r2,
    )
    return metrics


def _plot_freedom_score_eval(y_true, y_pred, out_path: Path) -> None:
    """Three-subplot eval: actual vs predicted, residuals, decile calibration."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="#3B82F6")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("Actual Freedom Score")
    ax.set_ylabel("Predicted Freedom Score")
    ax.set_title("Actual vs Predicted")
    ax.legend()

    # 2. Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.3, s=5, color="#10B981")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred − actual)")
    ax.set_title("Residuals")

    # 3. Decile calibration
    ax = axes[2]
    deciles = np.percentile(y_true, np.arange(0, 110, 10))
    digitized = np.digitize(y_true, deciles[:-1]) - 1
    mean_actual = [y_true[digitized == d].mean() if (digitized == d).sum() > 0 else 0 for d in range(10)]
    mean_pred = [y_pred[digitized == d].mean() if (digitized == d).sum() > 0 else 0 for d in range(10)]
    x_pos = np.arange(10)
    ax.bar(x_pos - 0.2, mean_actual, width=0.4, label="Actual", color="#3B82F6")
    ax.bar(x_pos + 0.2, mean_pred, width=0.4, label="Predicted", color="#F59E0B")
    ax.set_xlabel("Decile of Actual Score")
    ax.set_ylabel("Mean Score")
    ax.set_title("Decile Calibration")
    ax.legend()

    plt.suptitle("Freedom Score Model Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Freedom Score eval plot saved to %s", out_path)
