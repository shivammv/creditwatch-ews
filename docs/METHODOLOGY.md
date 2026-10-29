# CreditWatch EWS: methodology

## 1. Benchmark distress model (D1)

**Data.** UCI Polish Companies Bankruptcy (Tomczak 2016), all five horizon files in `data/polish/` (row and bankruptcy counts match UCI: 43,405 firm-years, 2,091 bankrupt). Per UCI, file *k* holds ratios observed (6 − *k*) years before the bankruptcy label, so the 5year file is closest to bankruptcy.

**Lead-matched models.** The Indian test measures 4- and 8-quarter lead times, so each gets the file whose label horizon matches it:

| Model | UCI file | Label horizon | Rows after dedup | Bankrupt | Hold-out (20%) |
|---|---|---|---|---|---|
| 4Q (`model_production.joblib`) | 5year | 1 year | 5,850 | 7.0% | 1,170 (82 bankrupt) |
| 8Q (`model_production_8q.joblib`) | 4year | 2 years | 9,710 | 5.3% | 1,942 (103 bankrupt) |

**Features.** 39 ratios that can be rebuilt from an Indian balance sheet and P&L (`src/features.py` maps each one to its Polish attribute and its Indian formula). No absolute amounts and no log-assets, because those are currency- and era-dependent. Two ratios are excluded from the model (Attr27 interest cover, Attr21 sales growth) because their *missingness* is a bankruptcy marker in the UCI file. Interest cover is still used as a transparent overlay rule (ICR < 1).

**Missing values.** Production models median-impute every ratio with an imputer fitted on training rows only (refitted inside every CV fold). Native-NaN gradient boosting would learn "missing = bankrupt", which is a UCI collection artefact: Attr27 is missing for 390 firms in the 5year file, 31.5% of them bankrupt vs 5.2% otherwise, and that flag alone gives PR-AUC 0.146. An Indian firm with an uncollected line item must not be scored as distressed for that reason. The cost is negligible: 5-fold dev-set PR-AUC is 0.429 imputed vs 0.432 native-NaN.

**Training protocol.** Exact duplicates are removed before splitting. A stratified 20% hold-out is carved first and used once. LightGBM is trained without class weights (the probabilities are later calibrated). Platt calibration and the empirical cost threshold are fitted on 5-fold out-of-fold predictions of the development set. Six model families were compared under identical CV (`artifacts/model_comparison.json`): logistic 0.321, random forest 0.375, XGBoost 0.438, LightGBM native-NaN 0.432, LightGBM imputed 0.429 (selected for transferability).

**Hold-out results.** 4Q: PR-AUC 0.486 [0.376, 0.609] vs baseline 0.070 (6.9x), ROC-AUC 0.890, Brier 0.047. 8Q: PR-AUC 0.395 [0.310, 0.479] vs 0.053 (7.4x), ROC-AUC 0.846. The dataset has no calendar dates or firm IDs, so these are out-of-sample in firms, not in time; treat them as an upper bound.

**Variant v2 (post-hoc, rejected).** The same pipeline with credit-logic monotone constraints, after a synthetic distressed company showed the v1 trees rating costs > sales and negative working capital as safe. On the hold-out v2 loses benchmark skill (4Q PR-AUC 0.395, 8Q 0.187) and does not improve the Indian result (defaulter-vs-peer AUC 0.37–0.43). It is reported for completeness; v1 remains the production model.

## 2. Indian case test (D2)

**Defaulters (10).** Jet Airways, DHFL, Reliance Capital, Future Retail, Coffee Day Enterprises, Alok Industries, Bhushan Steel, Reliance Communications, Sintex Industries, and IL&FS Transportation Networks (the listed IL&FS group company; the parent IL&FS Ltd was unlisted). The **event date is the earliest documented credit event**: a payment-default disclosure, a rating to D, or a CIRP admission. Each has a source URL in `data/indian/events.csv`. Using later dates (grounding, RBI supersession, CIRP) would overstate lead time.

**Peers (31).** Same sector, listed at the event date, no default found through event + 8 quarters. Devyani International was dropped because it listed after its matched event. 28 peers have enough history at the 8Q lead.

**Financials.** Screener.in standalone statements (Rs crore). Current assets and liabilities are Screener's "other assets / other liabilities" (a condensed-balance-sheet proxy, flagged `bs_proxy=1`). Receivables come from debtor days × sales / 365. COGS, SG&A, inventory, cash and shares are not available, so they stay missing (NaN) and are never zero-filled. Non-12-month fiscal periods (Alok's two 18-month years) are annualised for flow items.

**Point-in-time rules.** For lead *L* quarters, observation = event − 3*L* months. A fiscal year is usable only if FY end + 120 days ≤ observation (SEBI LODR requires audited annual results within 60 days; 120 days is conservative). Two usable years are required. Soft signals count only if dated ≤ observation and within the prior 365 days. Peers are observed on the same date as their matched defaulter. `assert_no_leakage` and four tests enforce this.

**Pre-registration.** `config/frozen_config.json` pins the model-file SHA-256 hashes, overlay weights, soft-signal rules, lags, lead-to-file mapping, cost ratio, capacity and the primary metric. It was committed before any Indian case was scored, and `indian.evaluate()` refuses to run if anything differs. One re-freeze happened before any Indian scoring (a pickling fix; see `DEVIATIONS.md`).

**Primary metric.** Peers stand in for the performing book. Defaulters are about 2% of a real book, so the book's 95th percentile is essentially the performing firms' 95th percentile. Alert = score above the 95th percentile of peer scores at the same lead (the 5% budget). Recall comes with a Wilson 95% CI, and the cutoff with a bootstrap CI over peers. Ablations cover model only, model + forensic, full, forensic only, and soft only. The secondary metric is a simulated 800-name book with a Polish hold-out background. It is biased towards flagging Indian firms (they get M/F/soft overlays the background cannot), and it gives 0 for every configuration anyway.

**Result.** 4Q: model only 2/10 [6%, 51%], full 0/10 [0%, 28%], soft only 1/10. 8Q: 0/10 for every configuration. Post-hoc threshold-free AUC is 0.46 at both leads.

## 3. Forensic and soft signals (D3)
- **Altman Z''** (emerging-market version, book equity): Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4. Zones: < 1.10 distress, 1.10–2.60 grey, > 2.60 safe.
- **Beneish M** (8 variables: DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA). M > −1.78 means manipulation zone.
- **Piotroski F** (9 binary tests). F ≤ 3 is weak.
- All three are checked against hand calculations on a worked example (`tests/test_audit.py`). Banks and NBFCs skip Z''/M/F. With Screener data, M and F are mostly unavailable (no COGS, SG&A or shares) and are reported as missing, never guessed.
- **Soft signals:** rating downgrade or watch-negative, promoter pledge > 50%, auditor red flag, delayed filing, related-party stress. They enter as capped log-odds additions (policy priors, not fitted; `src/rank.py::OVERLAY`). The shipped panel has five verified rating actions with URLs.

## 4. Explainability (D4)
`shap.TreeExplainer` runs on the imputed model input. The three largest risk-increasing contributions among ratios that were actually computed (imputed ratios are never drivers) are rendered as: ratio, the company's value, the healthy-peer median, and why it matters for credit.

## 5. Policy and cost (D5)
See `policy.md`. Cost = 20·FN + 1·FP. On calibrated probabilities, flag when 20p > 1 − p, so p* = 1/21 ≈ 0.0476. In 1,000 simulated 800-borrower books at a 2% default rate, the cost-optimal list is 43 names and the 40-name cap costs 0.8 units (0.5%). At 4.1% the optimum is 149 names and the cap costs +52%.

## 6. Limitations
Polish cross-sectional benchmark with no calendar time; 10 defaulters (wide intervals); restated and condensed Screener data; standalone statements miss group-level stress; peers instead of a full 800-firm point-in-time universe; survivorship in peer selection; lender-specific (ALM) features not built; Alok, Bhushan and Coffee Day may have earlier undocumented payment stress, which would mean their lead times are overstated.
