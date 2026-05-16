"""
src/interpretation/shap_analysis.py
=====================================
Task 11 — SHAP interpretation using TreeExplainer.

Outputs:
  reports/shap_importance.csv        — top-30 features by mean|SHAP|
  reports/figures/shap_summary.png   — beeswarm plot
  reports/figures/shap_bar.png       — bar importance
  reports/figures/shap_dependence_top5.png — dependence plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(model, X_pandas) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer.
    Returns shap_values array of shape (n_samples, n_features).
    For classifiers: returns proba-space SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_pandas)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # take positive class

    logger.info(
        "SHAP values computed: shape=%s", shap_values.shape
    )
    return shap_values, explainer


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 30,
    reports_dir: str | Path = "reports",
) -> pl.DataFrame:
    """
    Compute global feature importance as mean|SHAP|.
    Saves to reports/shap_importance.csv.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    rows = [
        {"rank": i + 1, "feature": feature_names[idx], "mean_abs_shap": float(mean_abs_shap[idx])}
        for i, idx in enumerate(sorted_idx)
    ]
    importance_df = pl.DataFrame(rows)

    out_path = reports_dir / "shap_importance.csv"
    importance_df.write_csv(out_path)
    logger.info("SHAP importance saved to %s", out_path)

    return importance_df


def per_user_explanations(
    shap_values: np.ndarray,
    feature_names: list[str],
    customer_ids: list[int],
    n_top: int = 5,
) -> pl.DataFrame:
    """
    For each user, compute top-N features driving score up/down.
    Returns pl.DataFrame(customer_id, driver_1..N, driver_value_1..N)
    """
    rows = []
    for i, cid in enumerate(customer_ids):
        sv = shap_values[i]
        # Top positive contributors
        pos_idx = np.argsort(sv)[::-1][:n_top]
        neg_idx = np.argsort(sv)[:n_top]

        row = {"customer_id": cid}
        for rank, idx in enumerate(pos_idx):
            row[f"top_driver_up_{rank+1}"] = feature_names[idx]
            row[f"top_driver_up_{rank+1}_shap"] = float(sv[idx])
        for rank, idx in enumerate(neg_idx):
            row[f"top_driver_down_{rank+1}"] = feature_names[idx]
            row[f"top_driver_down_{rank+1}_shap"] = float(sv[idx])
        rows.append(row)

    return pl.DataFrame(rows)


def plot_shap_summary(
    shap_values: np.ndarray,
    X_pandas,
    feature_names: list[str],
    figures_dir: str | Path = "reports/figures",
    max_display: int = 20,
) -> None:
    """Save SHAP beeswarm summary plot and bar importance plot."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_pandas,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_pandas,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP summary plots saved to %s", figures_dir)


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_pandas,
    importance_df: pl.DataFrame,
    n_top: int = 5,
    figures_dir: str | Path = "reports/figures",
) -> None:
    """Save SHAP dependence plots for top-N features."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    top_features = importance_df["feature"].head(n_top).to_list()

    fig, axes = plt.subplots(1, n_top, figsize=(5 * n_top, 4))
    if n_top == 1:
        axes = [axes]

    for i, feat in enumerate(top_features):
        if feat not in X_pandas.columns:
            continue
        feat_idx = list(X_pandas.columns).index(feat)
        ax = axes[i]
        ax.scatter(
            X_pandas[feat],
            shap_values[:, feat_idx],
            alpha=0.3, s=5, c=shap_values[:, feat_idx], cmap="coolwarm",
        )
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {feat}")
        ax.axhline(0, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    plt.savefig(figures_dir / "shap_dependence_top5.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP dependence plots saved to %s", figures_dir)


def verify_shap_additivity(
    shap_values: np.ndarray,
    explainer,
    predictions: np.ndarray,
    n_check: int = 10,
    tol: float = 1e-3,
) -> bool:
    """
    Verify: sum(SHAP) + base_value ≈ prediction (mathematical identity).
    Returns True if all checked rows pass within tolerance.
    """
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]  # binary classifier

    indices = np.random.choice(len(predictions), size=min(n_check, len(predictions)), replace=False)
    all_ok = True
    for idx in indices:
        shap_sum = shap_values[idx].sum() + base_value
        pred = predictions[idx]
        diff = abs(shap_sum - pred)
        if diff > tol:
            logger.error(
                "SHAP additivity FAILED at row %d: sum+base=%.6f  pred=%.6f  diff=%.6f",
                idx, shap_sum, pred, diff,
            )
            all_ok = False

    if all_ok:
        logger.info("[OK] SHAP additivity verified on %d random rows (tol=%.4f)", n_check, tol)

    return all_ok
