# CreditWatch EWS — Zeckathon 2026 (Risk & Fraud / Credit Distress)

## One-line
Leakage-safe early-warning system for corporate credit distress: portable model on UCI Polish bankruptcy data, ranked quarterly watch-list under a 5% review budget, SHAP explanations in credit-officer language, Altman/Beneish/Piotroski + soft signals, and an Indian case harness.

## How to run
```bash
pip install -r requirements.txt
python run_all.py          # reproduce Polish metrics + artifacts (needs lightgbm, shap)
pytest -q
streamlit run app.py
# optional: score any company-year sheet
python score_any.py path/to/financials.csv --asof 2018-04-01
```

## Headline numbers (Polish, artefact-free, hold-out)
| Metric | Production (39 India-portable features) | Reference (64 attrs, includes Attr27 artefact) |
|--------|------------------------------------------|--------------------------------------------------|
| PR-AUC | **0.339** (baseline 0.039) | 0.78 |
| ROC-AUC | 0.82 | 0.96 |
| Recall @ 5% budget | **~48%** | ~78% |
| Cost (20·FN+1·FP) @ 5% | 604 vs flag-none 1080 | lower |

### Asymmetric cost matrix (explicit math)
Problem statement: a missed default costs **20×** a false alarm.  
Expected cost = `20 · FN + 1 · FP`.

On **calibrated** probabilities the Bayes-optimal threshold is:
```
t* = C_FP / (C_FP + C_FN) = 1 / (1 + 20) = 1/21 ≈ 0.0476
```
Code constants (`src/model.py`):
```python
COST_FN, COST_FP = 20.0, 1.0
BAYES_T = COST_FP / (COST_FP + COST_FN)   # 0.0476
```
Threshold is chosen by minimising the same cost on **out-of-fold** predictions only (hold-out never used for selection). Under the hard 5% (40-of-800) cap we evaluate the top-40 ranked names and report cost vs “flag nobody” and vs the unconstrained cost-optimal list size — see `policy.md` and `artifacts/universe_sim.json`.

**Key finding:** ~half the skill of the naive 64-attribute model is an Attr27 missing-data artefact that does **not** transfer to Indian filings. Production model excludes it.

## Deliverables vs PS
1. Distress model, leakage-safe, imbalance handled — **Done**
2. Indian panel (≥10 names, peers, 3y pre-event) — **10 core defaulters with usable data (IL&FS, DHFL, Jet, RCap, Future, Coffee Day, Alok, Bhushan, RCom, Sintex); 32 peers; proxy-flagged rows explicitly marked `bs_proxy=1`**
3. Altman Z'' / Beneish M / Piotroski F + ≥2 soft signals — **Done in code + soft_signals.csv**
4. Top-3 drivers, plain language — **SHAP templates in `src/rank.py::top_drivers` (credit-officer language)**
5. Operating policy, 40-of-800, cost of errors — **policy.md + universe_sim.json** (explicit 20× FN cost, Bayes threshold 1/21)
6. Indian recall @ 5% (4Q/8Q) — **Harness ready** (`src/indian.py`); exploratory universe_backtest (indicative only)

## Leakage controls (time-awareness)
- Duplicates removed **before** any split; 20% hold-out carved first and touched once
- No imputation (LightGBM native NaN) → no statistic learned from test rows
- Calibration + cost threshold fitted on **OOF** predictions of the dev set only
- Indian observation date = `event_date − 3×lead_quarters` months
- A fiscal year is used **only if** `public_date ≤ observation_date` (default lag = fy_end + 120 days)
- Soft signals used only if `date_public ≤ observation_date` and within prior 12 months
- Peers observed at the **same calendar date** as the matched defaulter
- Runtime assert: `assert_no_leakage` raises if any row has `public_date_used > obs_date`
- Unit test in `tests/test_pipeline.py` injects a post-observation filing and proves it is ignored
- Known residual caveat (documented in `src/screener.py`): bulk-dump figures are the restated Dec-2023 view, not the as-originally-published numbers — flagged, not hidden

## Limits (state in deck)
- Polish is cross-sectional; PR-AUC upper-bounds out-of-time performance
- Indian accounting / group structures differ; banks/NBFCs skip Z/M/F
- Survivorship in peer set; n small for Indian recovery claim
- Soft-signal dates should be re-checked against primary sources before live use

## Data included
- `data/polish/polish_{1..5}year.csv` — all UCI horizons
- `data/indian/financials.csv` — company-year panel (peers + several defaulters)
- `data/indian/soft_signals.csv` — rating / pledge / related-party / delayed filing
- `data/indian/events.csv` — verified vs unverified marked

## Team action before final submit
1. Re-verify soft-signal public dates ≤ event_date (already checked in soft_signals.csv)
2. `python run_all.py` on a machine with lightgbm+shap to refresh artefacts
3. Deck: Attr27 story, cost/5% graph, 2 case timelines (Jet + RCap or Future), explicit `bs_proxy` caveats, limits
4. 3-min video of app + one memo
5. Be ready for cross-examination on leakage (public_date ≤ obs_date asserts exist) and 20× cost matrix
