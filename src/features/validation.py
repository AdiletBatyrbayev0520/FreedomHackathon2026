"""
src/features/validation.py
==========================
Data leakage prevention module.
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# WHITELIST: Only these columns are allowed to be used as features.
# Any column not in this list will be dropped during prepare_xy.
FEATURE_COLUMNS = [
    # Demo
    "customer_age", "gender", "lifetime_days", "city", "channel",
    
    # RFM + Temporal
    "recency_tx_7d", "recency_tx_14d", "recency_tx_30d", "recency_tx_90d",
    "recency_event_7d", "recency_event_14d", "recency_event_30d",
    "frequency_tx_7d", "frequency_tx_14d", "frequency_tx_30d", "frequency_tx_90d",
    "monetary_sum_7d", "monetary_sum_14d", "monetary_sum_30d", "monetary_sum_90d", 
    "monetary_median_30d", "frequency_slope_4w", "monetary_delta_mom",
    "active_days_7d", "active_days_14d", "active_days_30d", "active_days_90d",
    
    # Transaction Mix
    "mcc_share_essentials", "mcc_share_travel", "mcc_share_leisure",
    "mcc_share_transfers", "mcc_share_financial", "mcc_share_other",
    "mcc_share_ecommerce", "mcc_share_health", "mcc_share_education", "mcc_share_telecom",
    "share_of_online", "share_of_p2p", "unique_terminals_30d", "imputed_share",
    
    # Engagement
    "events_per_day_7d", "events_per_day_30d", "unique_processes_30d",
    "error_rate_30d", "declined_rate_30d", "completion_rate_30d",
    "avg_process_duration_sec", "weekend_share_30d", "lang_ru_share", "lang_kz_share",
    
    # Products
    "has_card", "has_freedom_rating", "has_frhc", "has_deposit", "has_loan", "has_liveness",
    "products_count", "days_since_first_product_activation", "days_since_last_product_activation",
    
    # Partners
    "partner_purchases_count_90d", "partner_purchases_sum_90d",
    "cashback_earned_90d", "avg_cashback_rate", "unique_partners_90d",
    "partner_share_arbuz", "partner_share_ticketon", "partner_share_train_tickets",
    
    # Imputation flags
    "share_of_online_was_missing",
    
    # Loop-back features (Task 13 - allowed when explicit)
    "freedom_score_pred", "segment_id",
]

LABEL_PATTERNS = ["_label", "_target", "is_churned", "churn_", "target_", "propensity_"]

def assert_no_label_leakage(X: pd.DataFrame, model_name: str) -> None:
    """
    Fails if any column in features looks like a label.
    """
    # Exclude known safe features that match pattern (none currently match, but just in case)
    safe_features = []
    
    suspicious = [
        col for col in X.columns
        if col not in safe_features and any(pattern in col.lower() for pattern in LABEL_PATTERNS)
    ]
    
    if suspicious:
        raise ValueError(
            f"LEAKAGE in {model_name}: label-like columns found in features: {suspicious}"
        )
    
    logger.info("[OK] No label leakage detected for model: %s", model_name)
