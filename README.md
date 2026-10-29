# CreditWatch EWS: corporate credit early-warning system (Zeckathon PS-02)

**Team:** Shivam Verma, Homish Tripathi

CreditWatch turns company financial statements and dated credit events into a **quarterly watch-list of 40 names out of an 800-borrower book** (the 5% review budget). Each name gets three plain-language drivers, Altman Z'' / Beneish M / Piotroski F screens, soft signals, and a review memo. Everything is **point-in-time**: a number is used only after the date it became public.

The full write-up is in [`docs/REPORT.md`](docs/REPORT.md) (submitted as a PDF). This README covers setup, usage and the headline results.

---

## Headline results (all reproducible with `python run_all.py`)

### D1. Benchmark distress model (UCI Polish Companies Bankruptcy, untouched 20% hold-out)

Each lead time gets the UCI file whose label horizon matches it: the **5year file** (bankrupt within 1 year) for the 4-quarter lead and the **4year file** (within 2 years) for the 8-quarter lead.

| Model | Hold-out PR-AUC [95% CI] | No-skill baseline | Lift | ROC-AUC | Top-5% precision / recall |
|---|---|---|---|---|---|
| **4Q lead** (39 India-portable ratios, LightGBM, median-imputed, Platt-calibrated) | **0.486** [0.376, 0.609] | 0.070 | 6.9x | 0.890 | 54% / 39% |
| **8Q lead** (same pipeline, 4year file) | **0.395** [0.310, 0.479] | 0.053 | 7.4x | 0.846 | 37% / 35% |
| Reference: all 64 attributes, native NaN (not deployable) | 0.833 | 0.070 | | | |

- **Missing-data artefact:** in the UCI data, Attr27 (operating profit / financial expenses) is missing for 390 firms, and 31.5% of them went bankrupt (5.2% otherwise). That missingness alone gives PR-AUC 0.146. Production models drop Attr27 and **median-impute every other ratio inside the training folds**, so "missing" can never mean "bankrupt". This costs almost nothing (dev-set CV PR-AUC 0.429 vs 0.432 with native NaN).
- **800-borrower book at a 2% default rate** (1,000 simulated books from the hold-out): the top 40 contain 55% of defaulters (95% range 31% to 75%). Expected cost falls from 320 (flag nobody) to 175.6 (−45%). The cost-optimal list size is 43, so the 40-name cap costs only 0.8 cost units.

### D2. Indian test: pre-registered, 10 defaulters vs 31 matched peers

Every choice (model files, overlay weights, lags, rules) was **frozen and committed before any Indian case was scored** (`config/frozen_config.json`, verified at runtime; see the commit history).

| Lead | Alert rule | Model only | Model + Z/M/F + soft signals |
|---|---|---|---|
| 4 quarters | above the peers' 95th-percentile score (= 5% budget) | **2 / 10** (Wilson 95% CI 6% to 51%) | **0 / 10** (0% to 28%) |
| 8 quarters | same | **0 / 10** | **0 / 10** |

**The pre-registered result is negative, and we report it as-is.** The benchmark-trained model does not transfer to standalone Indian statements: the threshold-free AUC of defaulters vs peers is 0.46 at both leads. The reasons we found (details in the report):

1. **Holding and infrastructure structures:** ITNL and Coffee Day carried their stress in group companies. Their standalone balance sheets look liquid because the "other assets" line includes loans to group companies.
2. **The working-capital proxy is noise:** Screener shows only "other assets / other liabilities". WC/TA has AUC 0.48 and the current ratio 0.44, and Z'' flags healthy peers (Ashoka Buildcon, Shoppers Stop) as distressed, which pushes the cutoff up.
3. **Rating actions come late:** four of the five verified pre-default rating actions arrived 2 weeks to 4 months before default. Only Future Retail had one at least 4 quarters ahead (CARE A+ → A−, watch negative, 13 May 2020).

**What does carry signal (exploratory, n = 10, not used to tune anything):** classic debt-service ratios separate Indian defaulters from peers 4 quarters ahead. Cash earnings / total liabilities has AUC 0.77, interest cover 0.75, ROA 0.74 and leverage 0.70. The recommended next step is an India-trained, monotone debt-service scorecard validated out-of-time; the pipeline already supports it.

### D3 to D5
- **D3:** Altman Z'' (emerging-market version), Beneish M (8 variables) and Piotroski F (9 tests) are implemented and checked against hand calculations in the tests. Banks and NBFCs skip Z/M/F. Soft signals (rating downgrade, promoter pledge, auditor red flag, delayed filing, related-party stress) are implemented; the panel ships five rating actions verified against agency and press sources.
- **D4:** each flagged name gets its top-3 SHAP drivers written as driver → evidence vs healthy-peer median → why it matters, plus a review memo. Missing (imputed) ratios are never shown as drivers.
- **D5:** see [`policy.md`](policy.md): p* = 1/21 ≈ 4.76% on calibrated probabilities, a hard cap of 40, overflow list, escalation path, stress trigger and recalibration.

---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py            # rebuild every number: models (if missing), simulations, Indian evaluation, post-hoc, watch-list
python -m pytest -q          # 40 unit + integration tests
```

Optional interfaces:

```bash
# FastAPI backend + React frontend
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
cd frontend && npm ci && npm run dev        # open http://127.0.0.1:3000 (proxies /api to :8000)

# Streamlit dashboard
streamlit run app.py

# Score any company-year file (CSV/XLSX, headers auto-mapped), top-5% watch-list with SHAP memos
python score_any.py data/indian/financials.csv --soft data/indian/soft_signals.csv --asof 2019-06-30
```

Retrain from scratch with `python run_all.py --retrain` (about 5 minutes on 2 CPU cores). This changes the model-file hashes, so `indian.evaluate()` will refuse to run until the change is logged in `docs/DEVIATIONS.md` and the configuration is re-frozen (`python -m src.config --freeze`).

## How leakage is prevented

| Risk | Control | Where it is checked |
|---|---|---|
| Same row in train and test | exact duplicates removed before the split; hold-out carved first, used once | `src/model.py` |
| Preprocessing on test rows | imputer, calibrator and threshold fitted on development folds only | `src/model.py` |
| "Missing = bankrupt" artefact | Attr27 dropped; median imputation | `tests/test_audit.py::test_imputer_makes_missingness_uninformative` |
| Future filings | fiscal year used only after `public_date` (FY end + 120 days); observation = event − 3L months | `src/indian.py::assert_no_leakage`, `test_real_panel_point_in_time` |
| Future soft signals | used only if dated ≤ observation date, within the prior 365 days | `test_real_soft_signals_point_in_time` |
| Peer timing | peers observed on the same calendar date as their matched defaulter | `test_real_panel_point_in_time` |
| Late event dates (inflated lead) | earliest documented credit event, each with a source URL | `test_core_events_are_sourced_and_verified` |
| Tuning on the test set | frozen config with SHA-256; evaluation refuses edited weights or models | `test_frozen_config_detects_tampering`, `test_evaluation_refuses_unfrozen_changes` |

## Data provenance

- `data/polish/`: UCI Polish Companies Bankruptcy, all five horizon files (row and bankruptcy counts match UCI).
- `data/indian/financials.csv`: 484 company-years for 41 listed companies from Screener.in standalone statements (Rs crore). Current assets/liabilities are Screener's "other assets / other liabilities" (proxy, flagged `bs_proxy=1`). The four defaulters added in this audit are transcribed in `data/indian/screener_raw/` with their source pages, and every balance sheet adds up (tested).
- `data/indian/events.csv`: 10 defaulters with the earliest documented credit event and a source URL (exchange filing, rating-agency release, trustee notice or NCLT order).
- `data/indian/peers.csv`: 31 same-sector peers, listed at the event date, with no default found through event + 8 quarters.
- `data/indian/soft_signals.csv`: 5 verified rating actions with URLs. The earlier unverified list is kept in `soft_signals_unverified.csv` and is **not used**.
- `data/indian/removed_reconstructed_rows.csv`: 14 earlier rows that were estimates rather than filings. They were removed and are never used (see `docs/DEVIATIONS.md`).

## Repository structure

```
run_all.py                 one command: train (if needed), simulate, evaluate India, post-hoc, watch-list
score_any.py               score any statements file -> top-5% watch-list with memos
app.py                     Streamlit dashboard
src/model.py               lead-matched LightGBM training, calibration, hold-out metrics, ablation, model comparison
src/features.py            39 India-portable ratios, Polish <-> Indian mapping, monotone signs (v2)
src/scores.py              Altman Z'', Beneish M, Piotroski F
src/rank.py                overlay, SHAP drivers, memos, 800-borrower budget simulation
src/indian.py              point-in-time panel, leakage guard, pre-registered evaluation (peer cutoff + Wilson CI)
src/config.py              pre-registration freeze / verify (SHA-256)
src/posthoc.py             post-hoc AUC, lead-time curve, case timelines, exploratory univariate diagnostic
src/api.py                 FastAPI backend for the React app
frontend/                  React 18 + TypeScript + Tailwind web app (7 pages)
config/                    frozen_config.json (primary), frozen_config_v2.json (post-hoc variant)
data/                      Polish benchmark + sourced Indian panel
artifacts/                 models, metrics, curves, Indian results, figures
docs/                      REPORT.md, METHODOLOGY.md, DEVIATIONS.md, CROSS_EXAM_QA.md, DATA_ENTRY.md
tests/                     40 tests: leakage, provenance, scores by hand, pre-registration, API, end-to-end
```

## Limitations (stated up front)

- The benchmark has no calendar dates or firm IDs, so the hold-out is out-of-sample in firms, not in time.
- 10 Indian defaulters cannot support strong statistical claims: a 0/10 result has a Wilson upper bound of 28%.
- Screener shows the latest restated figures, not always the numbers as first published, and its balance sheet is condensed (working-capital proxy).
- Standalone statements miss group-level stress (ITNL, Coffee Day). Consolidated data would be the first upgrade.
- Peers stand in for the performing book in place of a full 800-firm point-in-time universe.
- Banks and NBFCs (DHFL, Reliance Capital) get the model score but no Z/M/F; a lender-specific module (ALM mismatch, CP dependence) is future work.
