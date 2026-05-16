# Data Issues Running Log

This document is updated every time a data anomaly is discovered.
It is used as source material for the **Caveats & Limitations** slide in the final presentation.

---

## Task 0 Diagnostic Findings (Pipeline Run 2026-05-16)

### [2026-05-16] ID Overlap Summary

| Pair | Users unique | Other unique | Intersection | Overlap % | Status |
|---|---|---|---|---|---|
| users ↔ transactions | 523,340 | 201,063 | 200,785 | **38.37%** | ⚠️ Low but OK |
| users ↔ events | 523,340 | 517,321 | 515,438 | **98.49%** | ✅ PASS |
| users ↔ partner_purchases | 523,340 | 96,062 | 95,848 | **18.31%** | ℹ️ Expected |
| users ↔ acquisition | 523,340 | 525,207 | 523,313 | **99.99%** | ✅ Perfect |

**Interpretation:**
- 322,555 users (61.6%) have **NO transactions** at all. This is not a join problem — it's real: many registered users never transacted (maybe opened account but didn't activate). Feature engineering must handle this gracefully (frequency=0, monetary=0, not null).
- Events coverage is excellent: 98.49% of users appear in events table.
- 427,492 users (81.7%) never bought through partner ecosystem — this is normal for a new app.

### [2026-05-16] Date Range Summary

| Table | Min date | Max date | Notes |
|---|---|---|---|
| users.reg_date | 2026-01-01 | 2026-05-13 | 132 days of data |
| transactions.transaction_date | 2026-01-01 | 2026-05-12 | Matches reg window |
| events.started_at | **2023-12-22** | 2026-05-12 | Pre-dates users by 2+ years |
| partner_purchases.purchase_date | 2024-07-21 | 2026-05-12 | |

**Event temporal pre-dating:** Events go back to 2023-12-22 while users only start 2026-01-01. This is consistent with explanation (a) from the TZ: events belong to users who were removed during dedup of the original 590k rows. The temporal check passed because max(events) = 2026-05-12 >= min(users.reg_date) = 2026-01-01.

**No STOP conditions triggered.** Pipeline proceeds.

---

## Issues Found During Data Cleaning (pre-pipeline)

### [2026-05-16] Users — Non-unique customer_id
- **Dataset:** `SAPP_пользователи.csv`
- **Issue:** `customer_id` was not unique — 2–3 different users could share the same ID.
- **Resolution:** Kept row with latest `reg_date` per customer_id.
- **Impact:** Transactions and events under a "shared" customer_id may belong to any of the 2–3 original users. This is **irreversible uncertainty** — we cannot know which user performed which action.
- **Presentation note:** Flag as fundamental data quality limitation. Any per-user analysis has ±2–3x noise for affected users.

### [2026-05-16] Partner Purchases — Systematic amount shift = −101,667
- **Dataset:** `SAPP_Покупки_у_партнеров.csv`
- **Issue:** `purchase_amount` and `cashback_amount` had a systematic negative offset. Raw values showed cashback_rate up to 400%.
- **Resolution:** Added 101,667 to both columns. Post-fix max cashback_rate = 31%.
- **Impact:** If shift constant is wrong, all financial features from this table are corrupted. Treatment: cross-check against known partner transaction samples.

### [2026-05-16] Transactions — Same systematic shift = −101,667
- **Dataset:** `SAPP_транзакции.csv`
- **Issue:** `transaction_sum` had the same shift as partner purchases.
- **Impact:** All revenue/LTV calculations depend on correct shift. Both tables being shifted by the same constant suggests intentional obfuscation of the raw data.

### [2026-05-16] Events — customer_id range mismatch
- **Dataset:** `SAPP_Процессы_пользователей_в_приложении.csv`
- **Issue:** Events `customer_id` range ~900k–1M; Users `customer_id` range 466k, 1.9M, 3M+. Low overlap expected.
- **Possible causes:** (a) Events belong to users deleted during dedup, (b) Events use a different ID namespace, (c) reg_date is synthetic/truncated.
- **Resolution pending:** Run Task 0 diagnostics to quantify overlap.

### [2026-05-16] Users — Suspicious age distribution
- **Dataset:** After cleaning: min=18, max=~65, median=29, mean=29.6.
- **Issue:** Perfect median ≈ mean is statistically unusual in real demographic data. Suggests synthetic generation.
- **Impact:** `customer_age` feature should be used with caution; patterns may not generalise to real user populations.

---

## Issues Found During Pipeline Execution

*(to be updated as pipeline runs)*

---

## Template for New Entries

```
### [YYYY-MM-DD] {Table} — {Short description}
- **Dataset:** 
- **Issue:** 
- **Resolution:** 
- **Impact:** 
- **Presentation note:** 
```
