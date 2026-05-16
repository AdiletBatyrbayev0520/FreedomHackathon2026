"""
src/interpretation/segmentation.py
=====================================
Task 12 — UMAP + GMM segmentation in SHAP space.

Key: segments are built on SHAP contribution vectors, NOT on raw predicted scores.
This produces semantically meaningful segments (driver-based), not just percentile buckets.

Anti-pattern to avoid: "VIP / Medium / Low" by score percentile where Medium = 50% of users.

Outputs:
  data/final/segments.parquet     — customer_id → segment_id, segment_name
  reports/segments_profile.csv    — mean metrics per segment
  reports/figures/segments_umap.png
  reports/figures/segments_radar.png
  models/umap_model.pkl
  models/gmm_model.pkl
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import f_oneway

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn not available. Segmentation will use PCA fallback.")


def run_segmentation(
    shap_values: np.ndarray,
    feature_names: list[str],
    importance_df: pl.DataFrame,
    customer_ids: list[int],
    freedom_scores: np.ndarray | None = None,
    churn_probs: np.ndarray | None = None,
    n_components_range: tuple[int, int] = (4, 8),
    n_umap_components: int = 5,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    data_final_dir: str | Path = "data/final",
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Run UMAP dimensionality reduction + GMM clustering on SHAP space.

    Parameters
    ----------
    shap_values     : (n_samples, n_features) SHAP matrix
    feature_names   : list of feature names
    importance_df   : from global_feature_importance() — to select top-20 features
    customer_ids    : list of customer_id values
    freedom_scores  : optional array for naming and profiling
    churn_probs     : optional array for profiling

    Returns
    -------
    pl.DataFrame(customer_id, segment_id, segment_name)
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_final_dir = Path(data_final_dir)
    data_final_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Select top-20 features by SHAP importance
    # -----------------------------------------------------------------------
    top_features = importance_df["feature"].head(20).to_list()
    top_idx = [feature_names.index(f) for f in top_features if f in feature_names]
    if not top_idx:
        logger.warning("No top features found in feature_names. Using all SHAP columns.")
        top_idx = list(range(min(20, shap_values.shape[1])))

    X_shap = shap_values[:, top_idx]
    logger.info("SHAP matrix for segmentation: shape=%s", X_shap.shape)

    # -----------------------------------------------------------------------
    # UMAP (or PCA fallback)
    # -----------------------------------------------------------------------
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=n_umap_components,
            n_neighbors=30,
            min_dist=0.1,
            random_state=random_state,
            verbose=False,
        )
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_umap_components, random_state=random_state)
        logger.info("Using PCA as fallback for dimensionality reduction.")

    logger.info("Fitting dimensionality reducer (UMAP/PCA)...")
    X_reduced = reducer.fit_transform(X_shap)

    # Save reducer
    reducer_path = models_dir / "umap_model.pkl"
    with open(reducer_path, "wb") as f:
        pickle.dump(reducer, f)

    # -----------------------------------------------------------------------
    # GMM — select n_components by BIC
    # -----------------------------------------------------------------------
    logger.info("Selecting GMM n_components by BIC in range %s...", n_components_range)
    bic_scores = []
    gmm_range = list(range(n_components_range[0], n_components_range[1] + 1))
    for nc in gmm_range:
        gmm = GaussianMixture(n_components=nc, random_state=random_state, max_iter=200)
        gmm.fit(X_reduced)
        bic_scores.append(gmm.bic(X_reduced))
        logger.info("  n_components=%d  BIC=%.1f", nc, bic_scores[-1])

    best_nc = gmm_range[int(np.argmin(bic_scores))]
    logger.info("Best n_components by BIC: %d", best_nc)

    gmm = GaussianMixture(n_components=best_nc, random_state=random_state, max_iter=300)
    gmm.fit(X_reduced)
    labels = gmm.predict(X_reduced)

    # Save GMM
    gmm_path = models_dir / "gmm_model.pkl"
    with open(gmm_path, "wb") as f:
        pickle.dump(gmm, f)

    # -----------------------------------------------------------------------
    # Stability check (two random states)
    # -----------------------------------------------------------------------
    gmm2 = GaussianMixture(n_components=best_nc, random_state=random_state + 1, max_iter=300)
    gmm2.fit(X_reduced)
    labels2 = gmm2.predict(X_reduced)
    ari = adjusted_rand_score(labels, labels2)
    logger.info("Segmentation stability: Adjusted Rand Index=%.4f (>0.5 is stable)", ari)
    if ari < 0.5:
        logger.warning(
            "[FLAG] Segmentation ARI=%.4f < 0.5. Clusters are not stable. "
            "Consider reducing n_components or checking SHAP space quality.",
            ari,
        )

    # -----------------------------------------------------------------------
    # Check minimum cluster size
    # -----------------------------------------------------------------------
    unique, counts = np.unique(labels, return_counts=True)
    min_pct = counts.min() / len(labels) * 100
    if min_pct < 2.0:
        logger.warning(
            "[FLAG] Smallest cluster is %.1f%% of users (<2%%). "
            "Possible noise cluster. Reduce n_components.",
            min_pct,
        )

    # -----------------------------------------------------------------------
    # Name segments based on dominant SHAP drivers
    # -----------------------------------------------------------------------
    segment_names = _name_segments(
        labels, X_shap, [feature_names[i] for i in top_idx],
        freedom_scores=freedom_scores,
    )

    # -----------------------------------------------------------------------
    # Build output DataFrame
    # -----------------------------------------------------------------------
    result = pl.DataFrame({
        "customer_id": customer_ids,
        "segment_id": labels.tolist(),
        "segment_name": [segment_names[int(l)] for l in labels],
    })

    out_path = data_final_dir / "segments.parquet"
    result.write_parquet(out_path)
    logger.info("Segments saved to %s", out_path)

    # -----------------------------------------------------------------------
    # Profile segments
    # -----------------------------------------------------------------------
    _save_segment_profile(
        result, freedom_scores, churn_probs, customer_ids, reports_dir
    )

    # -----------------------------------------------------------------------
    # ANOVA check: segments differ by Freedom Score
    # -----------------------------------------------------------------------
    if freedom_scores is not None:
        groups = [freedom_scores[labels == seg] for seg in unique]
        if all(len(g) > 1 for g in groups):
            f_stat, p_val = f_oneway(*groups)
            logger.info(
                "ANOVA on Freedom Score across segments: F=%.2f  p=%.4f", f_stat, p_val
            )
            if p_val > 0.01:
                logger.warning(
                    "[FLAG] Segments do NOT significantly differ by Freedom Score (p=%.4f). "
                    "Clusters may not capture meaningful value differences.",
                    p_val,
                )

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    _plot_umap_2d(X_shap, labels, segment_names, reducer, figures_dir)
    _plot_radar_chart(result, freedom_scores, churn_probs, customer_ids, labels, figures_dir)

    return result


def _name_segments(
    labels: np.ndarray,
    X_shap: np.ndarray,
    feature_names: list[str],
    freedom_scores: np.ndarray | None,
) -> dict[int, str]:
    """Assign descriptive names to segments based on mean SHAP per cluster."""
    unique_labels = sorted(set(labels.tolist()))
    segment_names = {}

    for seg in unique_labels:
        mask = labels == seg
        mean_shap = X_shap[mask].mean(axis=0)

        # Top positive SHAP features for this segment
        top_pos_idx = np.argsort(mean_shap)[::-1][:2]
        top_neg_idx = np.argsort(mean_shap)[:2]

        top_pos = [feature_names[i] for i in top_pos_idx if mean_shap[i] > 0]
        is_all_negative = all(mean_shap[i] <= 0 for i in top_pos_idx)

        mean_fs = float(freedom_scores[mask].mean()) if freedom_scores is not None else 0

        if is_all_negative or mean_fs < 0.05:
            name = f"Seg{seg}_Inactive"
        elif "frequency_tx_30d" in top_pos or "monetary_sum_30d" in top_pos:
            name = f"Seg{seg}_HighActivity"
        elif any("has_" in f for f in top_pos):
            name = f"Seg{seg}_ProductActivators"
        elif any("partner" in f for f in top_pos):
            name = f"Seg{seg}_PartnerEngaged"
        elif any("cashback" in f for f in top_pos):
            name = f"Seg{seg}_CashbackDriven"
        else:
            name = f"Seg{seg}_Cluster"

        segment_names[seg] = name
        logger.info(
            "Segment %d → '%s' | n=%d | top_pos_features=%s | mean_fs=%.3f",
            seg, name, mask.sum(), top_pos, mean_fs,
        )

    return segment_names


def _save_segment_profile(
    segments_df: pl.DataFrame,
    freedom_scores,
    churn_probs,
    customer_ids: list[int],
    reports_dir: Path,
) -> None:
    """Save per-segment profile metrics."""
    profile_data = []
    for seg_id in segments_df["segment_id"].unique().to_list():
        seg_mask = segments_df["segment_id"] == seg_id
        seg_df = segments_df.filter(seg_mask)
        n = seg_df.height
        seg_name = seg_df["segment_name"][0]

        # Indices in customer_ids array
        seg_cids = set(seg_df["customer_id"].to_list())
        idx = [i for i, cid in enumerate(customer_ids) if cid in seg_cids]

        row = {
            "segment_id": seg_id,
            "segment_name": seg_name,
            "n_users": n,
            "pct_users": round(n / segments_df.height * 100, 2),
        }

        if freedom_scores is not None and idx:
            fs_seg = freedom_scores[idx]
            row["mean_freedom_score"] = float(fs_seg.mean())
            row["median_freedom_score"] = float(np.median(fs_seg))

        if churn_probs is not None and idx:
            cp_seg = churn_probs[idx]
            row["mean_churn_prob"] = float(cp_seg.mean())

        profile_data.append(row)

    profile_df = pl.DataFrame(profile_data)
    out_path = reports_dir / "segments_profile.csv"
    profile_df.write_csv(out_path)
    logger.info("Segment profile saved to %s", out_path)


def _plot_umap_2d(
    X_shap: np.ndarray,
    labels: np.ndarray,
    segment_names: dict[int, str],
    reducer,
    figures_dir: Path,
) -> None:
    """2D UMAP projection colored by segment."""
    if UMAP_AVAILABLE:
        reducer_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer_2d = PCA(n_components=2)

    X_2d = reducer_2d.fit_transform(X_shap)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(segment_names)))

    for i, (seg_id, seg_name) in enumerate(sorted(segment_names.items())):
        mask = labels == seg_id
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[i]], label=seg_name, alpha=0.4, s=8,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title("UMAP 2D Projection — SHAP Space Segmentation")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(figures_dir / "segments_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("UMAP plot saved to %s", figures_dir / "segments_umap.png")


def _plot_radar_chart(
    segments_df: pl.DataFrame,
    freedom_scores,
    churn_probs,
    customer_ids: list[int],
    labels: np.ndarray,
    figures_dir: Path,
) -> None:
    """Radar chart comparing key metrics across segments."""
    metrics = ["mean_freedom_score", "mean_churn_prob", "n_users_pct"]

    segment_ids = sorted(segments_df["segment_id"].unique().to_list())
    seg_names = []
    seg_values = {m: [] for m in metrics}

    n_total = len(customer_ids)

    for seg_id in segment_ids:
        seg_cids = set(segments_df.filter(pl.col("segment_id") == seg_id)["customer_id"].to_list())
        idx = [i for i, cid in enumerate(customer_ids) if cid in seg_cids]
        seg_names.append(
            segments_df.filter(pl.col("segment_id") == seg_id)["segment_name"][0]
        )

        seg_values["mean_freedom_score"].append(
            float(freedom_scores[idx].mean()) if freedom_scores is not None and idx else 0
        )
        seg_values["mean_churn_prob"].append(
            float(churn_probs[idx].mean()) if churn_probs is not None and idx else 0
        )
        seg_values["n_users_pct"].append(len(idx) / n_total)

    # Simple bar chart instead of radar for simplicity
    x = np.arange(len(segment_ids))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics):
        vals = seg_values[metric]
        ax.bar(x, vals, color=plt.cm.tab10(np.linspace(0, 1, len(vals))))
        ax.set_xticks(x)
        ax.set_xticklabels(seg_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)

    plt.suptitle("Segment Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "segments_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Segment radar chart saved to %s", figures_dir / "segments_radar.png")
