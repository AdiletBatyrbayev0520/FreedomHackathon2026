#!/usr/bin/env python3
"""
run_pipeline.py
================
Task 16 — One-button pipeline runner. Executes Tasks 0–15 in sequence.

Stop conditions emit [STOP] and sys.exit(1) immediately.
Timing logged per step.

Usage:
    python run_pipeline.py [--skip-models] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

# ─── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")

# ─── Parallelism config ───────────────────────────────────────────────────────
import os
_N_CORES = os.cpu_count() or 32
os.environ.setdefault("POLARS_MAX_THREADS", str(_N_CORES))   # Polars feature engineering
os.environ.setdefault("OMP_NUM_THREADS", str(_N_CORES))       # NumPy / scikit-learn BLAS
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_CORES))
logger.info("Parallelism: %d CPU cores | GPU: RTX 5090 | task_type=GPU for all CatBoost models", _N_CORES)



def timed_step(name: str):
    """Context manager for timing and logging pipeline steps."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        logger.info("=" * 60)
        logger.info(">> STEP: %s", name)
        t0 = time.time()
        try:
            yield
            elapsed = time.time() - t0
            logger.info("[DONE] STEP: %s  (%.1fs)", name, elapsed)
        except SystemExit:
            raise
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("[FAIL] STEP: %s  (%.1fs)  Error: %s", name, elapsed, e)
            raise

    return _ctx()


def _fill_null_labels(df, propensity_columns: list[str]):
    """
    Fill null label columns after left-joining labels to features.

    Semantics:
      freedom_score_target = 0.0  → user generated no revenue/activity
      churn_label = 1             → user was already inactive (never transacted)
      propensity_* = 0            → no product activation observed
    """
    import polars as pl

    if "freedom_score_target" in df.columns:
        df = df.with_columns(pl.col("freedom_score_target").fill_null(0.0))
    if "churn_label" in df.columns:
        df = df.with_columns(pl.col("churn_label").fill_null(1))
    for col in propensity_columns:
        if col != "customer_id" and col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0))
    return df


def main():
    parser = argparse.ArgumentParser(description="FreedomProfile Analytics Pipeline v2")
    parser.add_argument("--skip-models", action="store_true", help="Skip model training (Tasks 8-13)")
    parser.add_argument("--dry-run", action="store_true", help="Only run Tasks 0-2 (diagnostics)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    RANDOM_STATE = args.random_state
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("  FreedomProfile Analytics Pipeline v2")
    logger.info("  Started: %s", date.today())
    logger.info("  Random state: %d", RANDOM_STATE)
    logger.info("=" * 60)

    # ─── Task 1: Load data ────────────────────────────────────────────────────
    with timed_step("Task 1: Load Data"):
        from src.data_loading import load_all
        tables = load_all("data/processed")

    # ─── Task 0: Diagnostics ─────────────────────────────────────────────────
    with timed_step("Task 0: ID Overlap Diagnostics"):
        from src.diagnostics.id_overlap import run_all_overlaps
        overlap_df = run_all_overlaps(tables, reports_dir="reports", stop_on_critical=True)
        logger.info("ID overlap report written to reports/id_overlap.csv")

    with timed_step("Task 0: Date Range Diagnostics"):
        from src.diagnostics.date_ranges import compute_date_ranges
        date_df = compute_date_ranges(tables, reports_dir="reports", stop_on_critical=True)
        logger.info("Date ranges report written to reports/date_ranges.csv")

    if args.dry_run:
        logger.info("--dry-run flag set. Stopping after Task 0.")
        _print_summary(pipeline_start)
        return

    # ─── Task 2: Time Split ────────────────────────────────────────────────────
    with timed_step("Task 2: Define Cutoffs"):
        from src.features.time_split import define_cutoffs
        cutoffs = define_cutoffs(
            tables["transactions"],
            tables["events"],
            label_window_days=14,
            val_window_days=30,
        )
        logger.info("Cutoffs: T=%s  T-1=%s  T-2=%s", cutoffs["T"], cutoffs["T_minus_1"], cutoffs["T_minus_2"])

    # ─── Tasks 3–6: Feature Matrix for all cutoffs ────────────────────────────
    with timed_step("Tasks 3-6: Build Feature Matrices (T_minus_2, T_minus_1, T)"):
        from src.features.build_matrix import build_all_cutoff_matrices
        feature_matrices = build_all_cutoff_matrices(
            tables, cutoffs,
            data_interim_dir="data/interim",
            reports_dir="reports",
        )
        train_features = feature_matrices["T_minus_2"]
        val_features = feature_matrices["T_minus_1"]
        test_features = feature_matrices["T"]

    # ─── Task 6: Preprocessing ────────────────────────────────────────────────
    with timed_step("Task 6: Preprocessing (log1p, RobustScaler, imputation)"):
        from src.features.preprocess import (
            apply_log1p, fit_scaler, impute_missing,
            save_preprocessors, transform_scaler,
        )
        # Log1p transform
        train_features = apply_log1p(train_features)
        val_features = apply_log1p(val_features)
        test_features = apply_log1p(test_features)

        # Fit on train only
        scaler, scaled_cols = fit_scaler(train_features)

        # Impute train first (gets medians)
        train_features, medians = impute_missing(train_features)
        val_features, _ = impute_missing(val_features, medians=medians)
        test_features, _ = impute_missing(test_features, medians=medians)

        # Scale all splits
        train_features = transform_scaler(train_features, scaler, scaled_cols)
        val_features = transform_scaler(val_features, scaler, scaled_cols)
        test_features = transform_scaler(test_features, scaler, scaled_cols)

        save_preprocessors(scaler, scaled_cols, medians)

    # ─── Task 7: Labels ────────────────────────────────────────────────────────
    with timed_step("Task 7: Build Labels"):
        from src.features.labels import (
            build_churn_label, build_freedom_score_target,
            build_propensity_labels, save_label_diagnostics, save_labels,
        )

        churn_labels = build_churn_label(
            tables["transactions"], tables["events"], cutoffs["T"], window_days=14
        )
        freedom_labels = build_freedom_score_target(
            tables["transactions"], tables["partner_purchases"], cutoffs["T"], window_days=30
        )
        propensity_labels = build_propensity_labels(
            tables["events"], cutoffs["T"], window_days=14
        )

        save_labels(churn_labels, "T")
        save_label_diagnostics(churn_labels, freedom_labels, propensity_labels)

        import polars as pl

        # Join labels to test features
        # fill_null rationale: 322k users (61%) have NO transactions →
        #   freedom_score_target=0 (no value generated)
        #   churn_label=1 (already inactive — never transacted)
        #   propensity_*=0 (no product activation observed)
        test_features = test_features.join(churn_labels, on="customer_id", how="left")
        test_features = test_features.join(freedom_labels, on="customer_id", how="left")
        test_features = test_features.join(propensity_labels, on="customer_id", how="left")
        test_features = _fill_null_labels(test_features, propensity_labels.columns)

        # Val labels
        churn_labels_val = build_churn_label(
            tables["transactions"], tables["events"], cutoffs["T_minus_1"], window_days=14
        )
        freedom_labels_val = build_freedom_score_target(
            tables["transactions"], tables["partner_purchases"], cutoffs["T_minus_1"], window_days=30
        )
        propensity_labels_val = build_propensity_labels(
            tables["events"], cutoffs["T_minus_1"], window_days=14
        )
        val_features = val_features.join(churn_labels_val, on="customer_id", how="left")
        val_features = val_features.join(freedom_labels_val, on="customer_id", how="left")
        val_features = val_features.join(propensity_labels_val, on="customer_id", how="left")
        val_features = _fill_null_labels(val_features, propensity_labels_val.columns)

        # Train labels
        churn_labels_train = build_churn_label(
            tables["transactions"], tables["events"], cutoffs["T_minus_2"], window_days=14
        )
        freedom_labels_train = build_freedom_score_target(
            tables["transactions"], tables["partner_purchases"], cutoffs["T_minus_2"], window_days=30
        )
        propensity_labels_train = build_propensity_labels(
            tables["events"], cutoffs["T_minus_2"], window_days=14
        )
        train_features = train_features.join(churn_labels_train, on="customer_id", how="left")
        train_features = train_features.join(freedom_labels_train, on="customer_id", how="left")
        train_features = train_features.join(propensity_labels_train, on="customer_id", how="left")
        train_features = _fill_null_labels(train_features, propensity_labels_train.columns)

        # Log null counts after fill
        for split_name, split_df in [("train", train_features), ("val", val_features), ("test", test_features)]:
            fs_nulls = split_df["freedom_score_target"].null_count()
            ch_nulls = split_df["churn_label"].null_count()
            logger.info(
                "[%s] After label fill: freedom_score nulls=%d  churn nulls=%d",
                split_name, fs_nulls, ch_nulls,
            )

    if args.skip_models:
        logger.info("--skip-models flag set. Stopping before Task 8.")
        _print_summary(pipeline_start)
        return

    # ─── Task 8: Freedom Score ────────────────────────────────────────────────
    with timed_step("Task 8: Train Freedom Score Model"):
        from src.models.freedom_score import train_freedom_score
        fs_model = train_freedom_score(
            train_features, val_features, test_features,
            random_state=RANDOM_STATE,
        )

    # ─── Task 9: Churn Model ──────────────────────────────────────────────────
    with timed_step("Task 9: Train Churn Model (v1)"):
        from src.models.churn import train_churn
        churn_model = train_churn(
            train_features, val_features, test_features,
            model_suffix="v1_fixed",
            random_state=RANDOM_STATE,
        )

    # ─── Task 10: Propensity Models ───────────────────────────────────────────
    with timed_step("Task 10: Train Propensity Models"):
        from src.models.propensity import train_propensity
        propensity_models = train_propensity(
            train_features, val_features, test_features,
            random_state=RANDOM_STATE,
        )

    # ─── Task 11: SHAP Analysis ───────────────────────────────────────────────
    with timed_step("Task 11: SHAP Analysis"):
        import pandas as pd
        from src.models.utils import prepare_xy
        from src.interpretation.shap_analysis import (
            compute_shap_values, global_feature_importance,
            per_user_explanations, plot_shap_summary, plot_shap_dependence,
            verify_shap_additivity,
        )

        X_test_pd, y_test_fs, feature_names, _ = prepare_xy(
            test_features, label_col="freedom_score_target",
            cat_cols=["city", "channel", "gender"]
        )

        # SHAP on 50k subsample (full 523k takes hours)
        shap_vals, explainer, shap_sample_idx = compute_shap_values(
            fs_model, X_test_pd, max_samples=50_000,
        )
        X_shap_sample = X_test_pd.iloc[shap_sample_idx]

        importance_df = global_feature_importance(shap_vals, feature_names)
        plot_shap_summary(shap_vals, X_shap_sample, feature_names)
        plot_shap_dependence(shap_vals, X_shap_sample, importance_df)

        # Full predictions (fast — just model.predict, no SHAP)
        fs_preds = fs_model.predict(X_test_pd)

        # Additivity check on the subsampled rows
        verify_shap_additivity(shap_vals, explainer, fs_preds[shap_sample_idx])

    # ─── Task 12: Segmentation ────────────────────────────────────────────────
    with timed_step("Task 12: UMAP + GMM Segmentation"):
        from src.interpretation.segmentation import run_segmentation

        # Use full customer_ids for segment assignment
        customer_ids = test_features["customer_id"].to_list()

        # Churn probs for profiling
        X_test_churn, _, _, _ = prepare_xy(
            test_features, label_col="churn_label",
            cat_cols=["city", "channel", "gender"]
        )
        churn_preds = churn_model.predict_proba(X_test_churn)[:, 1]

        # Segmentation runs on SHAP subsample (50k) — assigns segments,
        # then we can assign remaining users to nearest segment via GMM
        shap_customer_ids = [customer_ids[i] for i in shap_sample_idx]

        segments_df = run_segmentation(
            shap_values=shap_vals,
            feature_names=feature_names,
            importance_df=importance_df,
            customer_ids=shap_customer_ids,
            freedom_scores=fs_preds[shap_sample_idx],
            churn_probs=churn_preds[shap_sample_idx],
            random_state=RANDOM_STATE,
        )

    # ─── Task 13: Loop-back Features ─────────────────────────────────────────
    with timed_step("Task 13: Second-order Features + Retrain"):
        import polars as pl
        from src.features.second_order import add_second_order_features

        fs_pred_df = pl.DataFrame({
            "customer_id": customer_ids,
            "freedom_score_pred": fs_preds.tolist(),
        })

        test_v2 = add_second_order_features(
            test_features, fs_pred_df, segments_df, "T"
        )

        # Retrain churn v2 on extended features
        from src.models.churn import train_churn

        # For val and train v2 — use their respective predictions (simplified: use T predictions)
        # In production, compute predictions for each cutoff independently
        val_fs_preds = fs_model.predict(
            prepare_xy(val_features, label_col="freedom_score_target", cat_cols=["city", "channel", "gender"])[0]
        )
        val_v2 = add_second_order_features(
            val_features,
            pl.DataFrame({"customer_id": val_features["customer_id"].to_list(), "freedom_score_pred": val_fs_preds.tolist()}),
            segments_df,
            "T_minus_1",
        )

        train_fs_preds = fs_model.predict(
            prepare_xy(train_features, label_col="freedom_score_target", cat_cols=["city", "channel", "gender"])[0]
        )
        train_v2 = add_second_order_features(
            train_features,
            pl.DataFrame({"customer_id": train_features["customer_id"].to_list(), "freedom_score_pred": train_fs_preds.tolist()}),
            segments_df,
            "T_minus_2",
        )

        churn_model_v2 = train_churn(
            train_v2, val_v2, test_v2,
            model_suffix="v2_fixed",
            random_state=RANDOM_STATE,
        )

        # Comparison
        _save_model_comparison()

    # ─── Task 14: NBA ─────────────────────────────────────────────────────────
    with timed_step("Task 14: Next Best Action"):
        import polars as pl
        from src.business.nba import build_nba

        churn_pred_df = pl.DataFrame({
            "customer_id": customer_ids,
            "churn_prob": churn_preds.tolist(),
        })

        # Build propensity score DataFrames
        prop_score_dfs = {}
        for product, model in propensity_models.items():
            prop_col = f"propensity_{product}"
            # Must use same label_col as training so features match
            X_pd, _, _, _ = prepare_xy(
                test_features, label_col=prop_col,
                cat_cols=["city", "channel", "gender"]
            )
            scores = model.predict_proba(X_pd)[:, 1]
            prop_score_dfs[product] = pl.DataFrame({
                "customer_id": customer_ids,
                prop_col: scores.tolist(),
            })

        # Add freedom_score_pred to test_features for action matrix in NBA
        test_features_with_fs = test_features.join(
            fs_pred_df, on="customer_id", how="left",
        ).with_columns(pl.col("freedom_score_pred").fill_null(0.0))

        nba_df = build_nba(
            test_features_with_fs,
            prop_score_dfs,
            churn_pred_df,
        )

    # ─── Task 15: Channel LTV ─────────────────────────────────────────────────
    with timed_step("Task 15: Channel LTV Analysis"):
        from src.business.channel_ltv import compute_channel_ltv

        channel_summary = compute_channel_ltv(
            tables["users"],
            tables["acquisition"],
            tables["transactions"],
            freedom_scores=fs_pred_df,
            cutoff=cutoffs["T"],
        )

    # ─── Save final outputs ───────────────────────────────────────────────────
    with timed_step("Task 16: Save Final Outputs"):
        import polars as pl
        from pathlib import Path

        final_dir = Path("data/final")
        final_dir.mkdir(parents=True, exist_ok=True)

        # Freedom scores
        fs_pred_df.write_parquet(final_dir / "freedom_scores.parquet")

        # Churn predictions with top drivers
        churn_pred_df.write_parquet(final_dir / "churn_predictions.parquet")

        # Propensity scores
        all_prop_scores = fs_pred_df.select("customer_id")
        for product, df in prop_score_dfs.items():
            all_prop_scores = all_prop_scores.join(df, on="customer_id", how="left")
        all_prop_scores.write_parquet(final_dir / "propensity_scores.parquet")

        # ── Business Metrics (Task 14/18) ──────────────────────────────────
        import numpy as np

        nba_data = nba_df
        if "priority_segment" in nba_data.columns:
            seg_counts = nba_data.group_by("priority_segment").agg(pl.len().alias("cnt"))
            action_matrix = {}
            for row in seg_counts.iter_rows(named=True):
                action_matrix[row["priority_segment"]] = {
                    "count": int(row["cnt"]),
                    "pct": round(row["cnt"] / nba_data.height * 100, 1),
                }

            # Conversion uplift
            prop_vals = nba_data["propensity_score"].to_numpy()
            q90 = np.quantile(prop_vals, 0.9) if len(prop_vals) > 0 else 0
            top_decile = prop_vals[prop_vals >= q90]
            np.random.seed(42)
            rand_sample = np.random.choice(prop_vals, size=len(top_decile), replace=False)
            conversion_uplift = float(top_decile.mean() / max(rand_sample.mean(), 1e-6))

            # Retention value at risk
            if "Persuadable" in action_matrix:
                persuadables = nba_data.filter(pl.col("priority_segment") == "Persuadable")
                p_churn = persuadables["churn_prob"].to_numpy()
                # Use expected_value as proxy for freedom_score * churn
                if "expected_value" in persuadables.columns:
                    retention_val = float(persuadables["expected_value"].sum())
                else:
                    retention_val = 0.0
            else:
                retention_val = 0.0

            # Budget efficiency
            if "expected_value" in nba_data.columns:
                ev_vals = nba_data["expected_value"].to_numpy()
                ev_sorted = np.sort(ev_vals)[::-1]
                top20 = ev_sorted[:int(0.2 * len(ev_sorted))]
                budget_conc = float(top20.sum() / max(ev_vals.sum(), 1e-6) * 100)
            else:
                budget_conc = 0.0

            biz_metrics = {
                "total_users": int(nba_data.height),
                "action_matrix": action_matrix,
                "conversion_uplift_top_decile": round(conversion_uplift, 2),
                "retention_value_at_risk": round(retention_val, 2),
                "retention_uplift_30pct": round(retention_val * 0.3, 2),
                "budget_concentration_top20pct": round(budget_conc, 1),
            }

            biz_path = Path("reports") / "business_metrics.json"
            with open(biz_path, "w") as f:
                json.dump(biz_metrics, f, indent=2)
            logger.info("Business metrics saved to %s", biz_path)

        logger.info("Final outputs saved to data/final/")

    _print_summary(pipeline_start)


def _save_model_comparison():
    """Compare v1 vs v2 metrics and save to reports/model_comparison.csv."""
    import json
    import polars as pl

    rows = []
    for version in ["v1", "v2"]:
        path = Path(f"reports/metrics_churn_{version}.json")
        if not path.exists():
            continue
        with open(path) as f:
            m = json.load(f)
        test_m = m.get("test", {})
        rows.append({
            "model": f"churn_{version}",
            "auc_roc": test_m.get("auc_roc"),
            "auc_pr": test_m.get("auc_pr"),
            "f1": test_m.get("f1_at_threshold"),
        })

    if rows:
        pl.DataFrame(rows).write_csv("reports/model_comparison.csv")
        logger.info("Model comparison saved to reports/model_comparison.csv")


def _print_summary(pipeline_start: float):
    total = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Total time: %.1f minutes (%.0f seconds)", total / 60, total)
    logger.info("  Artifacts:")
    logger.info("    reports/id_overlap.csv")
    logger.info("    reports/date_ranges.csv")
    logger.info("    data/interim/cutoffs.json")
    logger.info("    data/interim/features_*.parquet")
    logger.info("    models/*.cbm")
    logger.info("    reports/figures/*.png")
    logger.info("    data/final/*.parquet")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
