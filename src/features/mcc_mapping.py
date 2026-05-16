"""
src/features/mcc_mapping.py
============================
Task 4 — MCC to macro-category mapping.

Run make_mcc_frequency_report() first to get the actual top-50 MCCs,
then review and fill in the MCC_TO_MACRO dict below.

Macro categories:
  essentials   — supermarkets, pharmacies, gas stations, utilities
  travel       — airlines, hotels, transport
  leisure      — restaurants, bars, entertainment
  transfers    — P2P, wallet top-ups
  financial    — financial services, insurance
  ecommerce    — online marketplaces
  health       — clinics, medical
  education    — schools, courses
  telecom      — mobile/internet providers
  other        — everything else
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCC → Macro category dictionary
# Top-50 MCCs from actual transaction data (populated after EDA)
# ---------------------------------------------------------------------------
MCC_TO_MACRO: dict[str | int, str] = {
    # Essentials
    "5411": "essentials",  # Grocery Stores, Supermarkets
    "5912": "essentials",  # Drug Stores and Pharmacies
    "5541": "essentials",  # Service Stations (Gas)
    "5542": "essentials",  # Automated Fuel Dispensers
    "5311": "essentials",  # Department Stores
    "5331": "essentials",  # Variety Stores
    "5499": "essentials",  # Misc Food Stores
    "5441": "essentials",  # Candy / Nut / Confectionery Stores
    "5451": "essentials",  # Dairy Products Stores
    "5251": "essentials",  # Hardware Stores
    "5261": "essentials",  # Lawn and Garden Stores
    "4900": "essentials",  # Utilities — Electric/Gas/Water
    "4814": "telecom",     # Telecommunication Services
    # Travel
    "4789": "travel",      # Transportation Services
    "4111": "travel",      # Transportation — Suburban/Local
    "4112": "travel",      # Passenger Railways
    "4131": "travel",      # Bus Lines
    "4511": "travel",      # Air Carriers, Airlines
    "7011": "travel",      # Lodging — Hotels/Motels
    "7512": "travel",      # Car Rental Agencies
    "4121": "travel",      # Taxicabs & Limousines
    "4722": "travel",      # Travel Agencies, Tour Operators
    # Airlines range 3000–3299
    **{str(k): "travel" for k in range(3000, 3300)},
    # Leisure
    "5812": "leisure",     # Eating Places, Restaurants
    "5813": "leisure",     # Bars, Cocktail Lounges
    "5814": "leisure",     # Fast Food Restaurants
    "7832": "leisure",     # Movie Theaters
    "7922": "leisure",     # Theatrical Producers, Ticket Agencies
    "7941": "leisure",     # Sports Clubs, Athletic Fields
    "7999": "leisure",     # Recreation Services
    "5735": "leisure",     # Musical Instruments Stores
    # Transfers / P2P
    "6536": "transfers",   # MoneySend Intracountry
    "6537": "transfers",   # MoneySend Intercountry
    "6538": "transfers",   # MoneySend Funding
    "4829": "transfers",   # Money Transfer
    "6010": "transfers",   # Manual Cash Disbursements — Bank
    "6011": "transfers",   # Automated Cash Disbursements — ATM
    # Financial
    "6012": "financial",   # Financial Institutions — Merchandise
    "6051": "financial",   # Non-Financial Institutions — Currency
    "6211": "financial",   # Security Brokers/Dealers
    "6300": "financial",   # Insurance Sales
    "7321": "financial",   # Consumer Credit Reporting Agencies
    # Ecommerce / Marketplace
    "5961": "ecommerce",   # Catalog & Mail Order Houses
    "5999": "ecommerce",   # Misc Retail Stores
    "7372": "ecommerce",   # Computer Software Stores
    "7371": "ecommerce",   # Computer Programming Services
    # Health
    "8011": "health",      # Doctors and Physicians
    "8021": "health",      # Dentists, Orthodontists
    "8049": "health",      # Optometrists
    "8062": "health",      # Hospitals
    "8099": "health",      # Health Services
    # Education
    "8211": "education",   # Elementary/Secondary Schools
    "8220": "education",   # Colleges, Universities
    "8299": "education",   # Schools and Educational Services
    # Telecom
    "4812": "telecom",     # Telephone Equipment and Supplies
    "4816": "telecom",     # Computer Networks/Info Services
}

MACRO_CATEGORIES = [
    "essentials", "travel", "leisure", "transfers",
    "financial", "ecommerce", "health", "education", "telecom", "other",
]


def map_mcc(df: pl.DataFrame, mcc_col: str = "mcc") -> pl.DataFrame:
    """
    Add 'mcc_macro' column to df based on MCC_TO_MACRO lookup.
    Unknown MCCs → "other".
    """
    mcc_str_map = {str(k): v for k, v in MCC_TO_MACRO.items()}

    df = df.with_columns(
        pl.col(mcc_col)
        .cast(pl.Utf8)
        .map_elements(
            lambda v: mcc_str_map.get(str(v), "other") if v is not None else "other",
            return_dtype=pl.Utf8,
        )
        .alias("mcc_macro")
    )
    return df


def make_mcc_frequency_report(
    transactions: pl.DataFrame,
    reports_dir: str | Path = "reports",
    top_n: int = 50,
) -> pl.DataFrame:
    """
    Count frequency and sum of each MCC in transactions.
    Saves reports/mcc_frequency.csv with top_n rows.
    Must be called BEFORE finalising MCC_TO_MACRO — review output to adjust mapping.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    freq = (
        transactions.group_by("mcc")
        .agg(
            pl.len().alias("tx_count"),
            pl.col("transaction_sum").sum().alias("tx_sum"),
            pl.col("customer_id").n_unique().alias("unique_users"),
        )
        .sort("tx_count", descending=True)
        .head(top_n)
    )

    # Add macro mapping
    mcc_str_map = {str(k): v for k, v in MCC_TO_MACRO.items()}
    freq = freq.with_columns(
        pl.col("mcc")
        .cast(pl.Utf8)
        .map_elements(
            lambda v: mcc_str_map.get(str(v), "UNMAPPED") if v is not None else "UNMAPPED",
            return_dtype=pl.Utf8,
        )
        .alias("macro_category")
    )

    out_path = reports_dir / "mcc_frequency.csv"
    freq.write_csv(out_path)
    logger.info("MCC frequency report saved to %s", out_path)

    unmapped = freq.filter(pl.col("macro_category") == "UNMAPPED")
    if not unmapped.is_empty():
        coverage_pct = (
            freq.filter(pl.col("macro_category") != "UNMAPPED")["tx_count"].sum()
            / freq["tx_count"].sum()
            * 100
        )
        logger.warning(
            "MCC mapping covers %.1f%% of top-%d MCCs. "
            "Unmapped: %s",
            coverage_pct,
            top_n,
            unmapped["mcc"].to_list(),
        )

    return freq
