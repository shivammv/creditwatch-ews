# CreditWatch EWS: methodology and limitations (for the report / cross-examination)

## 1. Distress model
* Data: UCI Polish Companies Bankruptcy, file `polish_1year.csv` (7,027 rows, 82 duplicate feature-rows removed, 6,945 used; 3.9% bankrupt). UCI documents this file as first-forecast-year ratios with the bankruptcy label observed **five years later** (a long lead time). No calendar dates exist, so a time-ordered split is impossible; the 20% hold-out is company-disjoint and untouched until reporting.
* Model: LightGBM, native NaN handling (no imputation, so no statistic is learned from test rows). Platt calibration and the cost threshold are fitted on 5-fold out-of-fold predictions of the dev set only.
* Features: 39 ratios that can be recomputed from an Indian balance sheet + P&L (`src/features.py` maps each to its Polish attribute and its Indian formula). No absolute-currency features.
* **Artefact found and removed:** Attr27 is NaN for 309 firms, 38.8% of which went bankrupt (2.3% otherwise). That missing-data pattern carries about half the skill of the 64-attribute model: PR-AUC 0.766 (all 64) -> 0.368 (minus Attr27) -> 0.243 (India-portable, 5-fold CV), while a bare Attr27-missing flag added to the portable set lifts it to 0.697. We report and ship the artefact-free model.
* Hold-out result: PR-AUC **0.339** (95% bootstrap CI 0.23-0.49) vs a no-skill baseline of 0.039 (≈ 9× lift), ROC-AUC 0.82; hold-out has 54 positives, so intervals are wide.

## 2. Cost, threshold, budget
* Cost: 20·FN + 1·FP. On calibrated probabilities the Bayes-optimal cut-off is 1/21 = 0.0476 (the OOF grid optimum is reported alongside). On the hold-out it flags 24% of names (recall 70%), far above the 5% budget, so the 5% cap is evaluated explicitly: top-5% recall 48%, cost 604 vs 609 (cost-optimal) vs 1,080 (flag nobody).
* 800-borrower simulation (policy.md): recall inside 40 names ≈ 46-50% across default rates of 1-3.9%.

## 3. Forensic and soft signals
Altman Z'' (emerging-market, book equity so no market-cap look-up), Beneish M (8-variable), Piotroski F (9 tests); interest cover < 1 rule; five soft signals. Overlay weights are **policy priors**, not fitted, because 10 defaulters cannot support estimation; sensitivity is reported as "model only" vs "model + overlay" recall once the panel is filled.

## 4. Explainability
`shap.TreeExplainer` on the LightGBM model; the three largest risk-increasing contributions are rewritten from templates that name the ratio, the company's value, the healthy-peer median and the credit concern (`src/features.py`, `src/rank.py::top_drivers`).

## 5. Leakage controls (asserted in code and tests)
Duplicates removed pre-split; hold-out first; no imputation; calibration and threshold on OOF only. Indian side: observation = event − 4/8 quarters; a fiscal year is used only if its `public_date` ≤ observation date; soft signals only if published ≤ observation date; peers observed at the matched defaulter's date; `assert_no_leakage` and a unit test that a post-observation filing is ignored.

## 6. Known limitations (state these in the deck)
* Indian recall is **not reported until the panel is populated**; no Indian figures were fabricated.
* Benchmark is Polish and cross-sectional; Indian accounting (Ind AS, group structures, off-balance-sheet SPVs) differs; IL&FS-type failures hide in consolidation and related-party flows that ratios under-capture.
* Survivorship bias: peers are chosen as survivors; matched on sector and observation date only. Sector effects: banks/NBFCs lack working-capital/gross-margin concepts, so Z''/M/F are skipped and the model is indicative only.
* The 800-name background book in the Indian harness is a Polish hold-out proxy for "the rest of the book".
* Event dates in the template are unverified placeholders until checked against IBBI/NCLT/exchange records.
* PR-AUC on a random hold-out upper-bounds out-of-time performance.


## 7. Multi-horizon Polish files
All five UCI horizon files (`polish_1year` … `polish_5year`) are shipped under `data/polish/`.
Lead time mapping: file k ≈ (6−k) years of lead. Cross-file evaluation is optimistic (firms can recur). Prefer within-file hold-out for claims.
