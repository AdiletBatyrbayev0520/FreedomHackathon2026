"""
src/models/propensity.py
=========================
Task 10 — Propensity models (one CatBoostClassifier per product).

Metrics: AUC-ROC, AUC-PR, Precision@10% per product
Stop: if AUC-ROC < 0.65 for a product — that model is not predictive (flag, don't use)

Outputs:
  models/propensity_{product}.cbm
  reports/metrics_propensity.json
  reports/figures/propensity_per_product.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from src.features.products import ALL_PRODUCTS
from src.models.utils import prepare_xy, save_metrics, save_model

logger = logging.getLogger(__name__)

CAT_FEATURES = ["city", "channel", "gender"]


def train_propensity(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    random_state: int = 42,
) -> dict[str, CatBoostClassifier]:
    """
    Train one CatBoostClassifier per product.

    Returns dict: {product_name: fitted_model}
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    models = {}
    plot_data = {}

    for product in ALL_PRODUCTS:
        label_col = f"propensity_{product}"

        # Check if label exists in data
        if label_col not in train_df.columns:
            logger.warning("Label '%s' not found in training data, skipping.", label_col)
            continue

        # Check if there are enough positive examples
        pos_count = int(train_df[label_col].sum())
        pos_rate = float(train_df[label_col].mean() or 0)
        if pos_count < 100:
            logger.warning(
                "[propensity_%s] Only %d positive examples (%.1f%%). Skipping — insufficient data.",
                product, pos_count, pos_rate * 100,
            )
            continue

        logger.info("Training propensity model for: %s (pos_rate=%.2f%%)", product, pos_rate * 100)

        X_train, y_train, feature_names, cat_feature_names = prepare_xy(
            train_df, label_col=label_col, cat_cols=CAT_FEATURES
        )
        X_val, y_val, _, _ = prepare_xy(val_df, label_col=label_col, cat_cols=CAT_FEATURES)
        X_test, y_test, _, _ = prepare_xy(test_df, label_col=label_col, cat_cols=CAT_FEATURES)

        model = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=150,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
            cat_features=cat_feature_names,
            # ── Hardware acceleration ──────────────────────────────────────
            task_type="GPU",
            devices="0",
            thread_count=-1,
            grow_policy="Lossguide",
            max_leaves=64,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            border_count=254,
            min_data_in_leaf=20,
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )

        test_probs = model.predict_proba(X_test)[:, 1]
        metrics = _evaluate_propensity(y_test, test_probs, product)
        all_metrics[product] = metrics

        # Flag if AUC < 0.65
        if metrics["auc_roc"] < 0.65:
            logger.warning(
                "[propensity_%s] AUC-ROC=%.4f < 0.65. Model is not predictive. "
                "Not saving — insufficient signal.",
                product, metrics["auc_roc"],
            )
            continue

        save_model(model, models_dir / f"propensity_{product}.cbm")
        models[product] = model
        plot_data[product] = {"y_test": y_test, "test_probs": test_probs, "metrics": metrics}

    save_metrics(all_metrics, reports_dir / "metrics_propensity.json")
    _plot_propensity_summary(plot_data, figures_dir / "propensity_per_product.png")

    return models


def _evaluate_propensity(y_true, y_proba, product: str) -> dict:
    if y_true.sum() == 0:
        logger.warning("[propensity_%s] No positive examples in test set.", product)
        return {"product": product, "auc_roc": 0.5, "auc_pr": 0.0, "precision_at_10pct": 0.0}

    auc_roc = float(roc_auc_score(y_true, y_proba))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = float(auc(recall, precision))

    n_top = max(1, int(len(y_true) * 0.1))
    top_idx = np.argsort(y_proba)[::-1][:n_top]
    prec_at_10 = float(y_true[top_idx].mean())

    metrics = {
        "product": product,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision_at_10pct": prec_at_10,
        "baseline_positive_rate": float(y_true.mean()),
    }
    logger.info(
        "[propensity_%s] AUC-ROC=%.4f  AUC-PR=%.4f  Prec@10%%=%.4f",
        product, auc_roc, auc_pr, prec_at_10,
    )
    return metrics


def _plot_propensity_summary(
    plot_data: dict,
    out_path: Path,
) -> None:
    """Bar chart of AUC-ROC and Precision@10% per product."""
    if not plot_data:
        logger.warning("No propensity models to plot.")
        return

    products = list(plot_data.keys())
    auc_vals = [plot_data[p]["metrics"]["auc_roc"] for p in products]
    prec_vals = [plot_data[p]["metrics"]["precision_at_10pct"] for p in products]
    baseline_vals = [plot_data[p]["metrics"]["baseline_positive_rate"] for p in products]

    x = np.arange(len(products))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x, auc_vals, width, color="#3B82F6", label="AUC-ROC")
    axes[0].axhline(0.65, color="red", ls="--", label="Min threshold (0.65)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(products, rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Propensity AUC-ROC per Product")
    axes[0].legend()

    axes[1].bar(x - width / 2, prec_vals, width, color="#10B981", label="Precision@10%")
    axes[1].bar(x + width / 2, baseline_vals, width, color="#F59E0B", label="Baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(products, rotation=30, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Precision@10% vs Baseline")
    axes[1].legend()

    plt.suptitle("Propensity Models Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Propensity summary plot saved to %s", out_path)
