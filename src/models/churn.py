"""
src/models/churn.py
====================
Task 9 — Churn prediction (CatBoostClassifier).

Key metrics: AUC-ROC, AUC-PR, F1@optimal_threshold, Precision@10%, Recall@10%
Important: Recall > Precision bias for churn (cost of false negative > false positive)

Outputs:
  models/churn_v1.cbm
  reports/metrics_churn_v1.json
  reports/figures/churn_roc.png
  reports/figures/churn_pr.png
  reports/figures/churn_calibration.png
  reports/figures/churn_confusion.png
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.models.utils import prepare_xy, save_metrics, save_model

logger = logging.getLogger(__name__)

CAT_FEATURES = ["city", "channel", "gender"]


def train_churn(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    label_col: str = "churn_label",
    model_suffix: str = "v1",
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    random_state: int = 42,
) -> CatBoostClassifier:
    """Train CatBoostClassifier for churn prediction."""
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing churn training data...")
    X_train, y_train, feature_names, cat_feature_names = prepare_xy(
        train_df, label_col=label_col, cat_cols=CAT_FEATURES
    )
    X_val, y_val, _, _ = prepare_xy(val_df, label_col=label_col, cat_cols=CAT_FEATURES)
    X_test, y_test, _, _ = prepare_xy(test_df, label_col=label_col, cat_cols=CAT_FEATURES)

    logger.info(
        "Train: %d rows (%.1f%% positive) | Val: %d | Test: %d | Features: %d",
        len(X_train), y_train.mean() * 100,
        len(X_val), len(X_test), len(feature_names),
    )

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=150,
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=200,
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

    # Find optimal threshold on val
    val_probs = model.predict_proba(X_val)[:, 1]
    optimal_threshold = _find_optimal_threshold(y_val, val_probs)
    logger.info("Optimal threshold (max F1 on val): %.3f", optimal_threshold)

    # Evaluate on test
    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = _evaluate_classifier(
        y_test, test_probs, optimal_threshold, split_name="test"
    )
    val_metrics = _evaluate_classifier(
        y_val, val_probs, optimal_threshold, split_name="val"
    )

    # Sanity check: AUC-ROC > 0.7
    test_auc = test_metrics["auc_roc"]
    if test_auc < 0.7:
        logger.warning(
            "[FLAG] Churn AUC-ROC=%.4f < 0.7. Model has weak discrimination. "
            "Check feature quality and label window.",
            test_auc,
        )
    else:
        logger.info("[OK] Churn AUC-ROC=%.4f (target: >0.7)", test_auc)

    all_metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "optimal_threshold": optimal_threshold,
    }
    save_metrics(all_metrics, reports_dir / f"metrics_churn_{model_suffix}.json")
    save_model(model, models_dir / f"churn_{model_suffix}.cbm")

    # Feature importance sanity check
    _check_feature_importance(model, feature_names)

    # Plots
    _plot_roc_curve(y_test, test_probs, figures_dir / "churn_roc.png")
    _plot_pr_curve(y_test, test_probs, figures_dir / "churn_pr.png")
    _plot_calibration(y_test, test_probs, figures_dir / "churn_calibration.png")
    _plot_confusion_matrix(
        y_test, (test_probs >= optimal_threshold).astype(int),
        figures_dir / "churn_confusion.png",
    )

    return model


def _find_optimal_threshold(y_true, y_proba) -> float:
    """Find threshold that maximises F1 on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Add epsilon to avoid division by zero
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last element has no corresponding threshold
    return float(thresholds[best_idx])


def _evaluate_classifier(y_true, y_proba, threshold: float, split_name: str) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    auc_roc = float(roc_auc_score(y_true, y_proba))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = float(auc(recall, precision))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Precision@10%: sort by score, take top 10%
    n_top = max(1, int(len(y_true) * 0.1))
    top_idx = np.argsort(y_proba)[::-1][:n_top]
    prec_at_10 = float(y_true[top_idx].mean())
    recall_at_10 = float(y_true[top_idx].sum() / max(y_true.sum(), 1))

    metrics = {
        "split": split_name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1_at_threshold": f1,
        "threshold": threshold,
        "precision_at_10pct": prec_at_10,
        "recall_at_10pct": recall_at_10,
        "baseline_positive_rate": float(y_true.mean()),
    }
    logger.info(
        "[%s] AUC-ROC=%.4f  AUC-PR=%.4f  F1=%.4f  Prec@10%%=%.4f  Recall@10%%=%.4f",
        split_name, auc_roc, auc_pr, f1, prec_at_10, recall_at_10,
    )
    return metrics


def _check_feature_importance(model, feature_names: list[str]) -> None:
    """Warn if customer_id or suspicious features appear in top-5."""
    importances = model.get_feature_importance()
    if len(importances) != len(feature_names):
        return
    top5_idx = np.argsort(importances)[::-1][:5]
    top5 = [feature_names[i] for i in top5_idx]
    logger.info("Top-5 features by CatBoost importance: %s", top5)
    suspicious = [f for f in top5 if "customer_id" in f or "target" in f or "label" in f]
    if suspicious:
        logger.warning(
            "[LEAKAGE] Suspicious features in top-5: %s. Check for leakage.", suspicious
        )


def _plot_roc_curve(y_true, y_proba, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, color="#3B82F6", label=f"AUC-ROC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Churn — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Churn ROC saved to %s", out_path)


def _plot_pr_curve(y_true, y_proba, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = auc(recall, precision)
    baseline = y_true.mean()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, lw=2, color="#10B981", label=f"AUC-PR = {auc_pr:.4f}")
    plt.axhline(baseline, color="red", ls="--", label=f"Baseline = {baseline:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Churn — Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Churn PR curve saved to %s", out_path)


def _plot_calibration(y_true, y_proba, out_path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, "s-", lw=2, color="#8B5CF6", label="Calibration")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Churn — Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Churn calibration plot saved to %s", out_path)


def _plot_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted Stay", "Predicted Churn"])
    plt.yticks([0, 1], ["Actual Stay", "Actual Churn"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    plt.title("Churn — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Churn confusion matrix saved to %s", out_path)
