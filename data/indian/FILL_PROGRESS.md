# Weak parts — fixed status (27 Oct 2026)

## 1. Missing core defaulter financials — FIXED (with honesty flags)
| Company | Years | Quality | Notes |
|---------|-------|---------|-------|
| **DHFL** | FY16–19 | **Medium-Strong** | Rediff BS + AR + results releases; interest/EBIT filled from PBT + known finance-cost ratios (`bs_proxy=1`) |
| **IL&FS** | FY15–18 | **Medium** | FY18 group from Firstpost/Business Standard; FY15–17 trajectory + structure proxies (`bs_proxy=1`). 4Q lead now has 2 usable pre-obs years |
| **Bhushan Steel** | FY14–16 | **Medium** | Debt/sales/NI from BusinessLine + FE; BS lines reconstructed (`bs_proxy=1`) |
| **Sintex Industries** | FY18–20 | **Proxy** | Manual reconstruction from public stress-period summaries (`bs_proxy=1`); not in bulk dump |

`bs_proxy=1` rows **must** be labelled in deck as “partial / reconstructed from secondary sources” — not full audited line items. Judges will probe this; be transparent.

## 2. Soft signals — DONE
19 rows in `soft_signals.csv` (rating_downgrade, promoter_pledge, related_party, delayed_filing, auditor_red_flag).  
Team: confirm each `date_public` ≤ final `event_date`.

## 3. Event dates
| Company | Status |
|---------|--------|
| Jet, RCap, Future, Coffee Day, RCom, Sintex | verified |
| Alok | month_verified |
| **DHFL, Bhushan** | now marked **verified** |
| **IL&FS** | month_verified (Aug–Sep 2018 window) |

## 4. Still not “hard” (do not overclaim)
- Indian recall@5% remains exploratory until `run_all.py` + `universe_backtest.py` re-run on machine with lightgbm/shap
- IL&FS / Bhushan / Sintex incomplete BS lines (CFO, inventory, COGS still sparse)
- cogs/sga/cash/shares still empty for many peers (not blocking for portable model)
- Panel now reports `ready=True` (10 defaulters + 32 peers with ≥2 clean years)

## 5. Commands
```bash
pip install -r requirements.txt
python run_all.py
python universe_backtest.py data/indian/financials.csv --model
streamlit run app.py
```
