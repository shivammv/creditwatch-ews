# Zeckathon PS-02 submission: CreditWatch EWS

**Team:** Shivam Verma, Homish Tripathi

- **Document:** `docs/REPORT.md` (submitted as `Zeckathon_PS02_CreditWatch_Report.pdf`)
- **Code:** this repository. `python run_all.py` reproduces every number; `python -m pytest -q` runs 40 tests.

| Deliverable | Where | Headline |
|---|---|---|
| D1 benchmark model | `src/model.py`, `artifacts/benchmark_report*.json` | 4Q PR-AUC 0.486 vs 0.070 baseline; 8Q 0.395 vs 0.053 (untouched hold-out) |
| D2 Indian test | `src/indian.py`, `artifacts/indian_summary.json` | pre-registered; 10 defaulters vs 31 peers; 4Q model-only 2/10, full 0/10; 8Q 0/10, reported as-is |
| D3 screens + soft signals | `src/scores.py`, `src/rank.py`, `data/indian/soft_signals.csv` | Z'', M, F checked by hand in tests; 5 verified rating actions |
| D4 explanations | `src/rank.py::top_drivers`, memos in `artifacts/watchlist_any.csv` | top-3 drivers in credit-officer language |
| D5 policy | `policy.md` | p* = 1/21; 40-name cap costs 0.5% at a 2% default rate, +52% at 4.1% |
