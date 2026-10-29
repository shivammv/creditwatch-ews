# 🛡️ CreditWatch EWS — Intelligent Financial Risk & Fraud Analytics

**CreditWatch EWS** is an enterprise-grade Early Warning System (EWS) for corporate credit distress, forensic accounting anomaly screening, and portfolio risk ranking. Built for commercial credit underwriters, risk managers, and financial fraud investigation teams.

---

## ⚡ Quick Start: Running the Full-Stack Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Launch the Web Application (FastAPI + React 18)
In two separate terminals:

**Terminal A — Backend API (FastAPI):**
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

**Terminal B — Enterprise Frontend (React + TypeScript + Tailwind):**
```bash
cd frontend
npm run dev
```
Open **`http://127.0.0.1:3000`** in your browser. (The Vite frontend automatically proxies `/api` requests to `http://127.0.0.1:8000`).

---

## 🖥️ Alternative Interfaces & CLI Workflows

### Streamlit Application
The original Streamlit dashboard is fully preserved and updated:
```bash
streamlit run app.py
```

### Command-Line Scoring (`score_any.py`)
Score any raw CSV/XLSX financial statement, automatically map columns, and generate the top-5% priority watchlist with SHAP memos:
```bash
python score_any.py data/indian/financials.csv
# Output generated at: artifacts/watchlist_any.csv
```

### Reproduce Full Scientific Pipeline & Retrain Models
```bash
python run_all.py --retrain
```

### Run Test Suite
```bash
pytest -v
# Executes 17 unit and integration tests across data leakage, cost policies, API endpoints, and models
```

---

## 🔬 Scientific ML Pipeline & Benchmark Results

### 1. Multi-Model Algorithmic Comparison (Identical 5-Fold Stratified CV on Development Set, N = 34,724)
| Model Family | Missing Value Handling | CV PR-AUC | CV ROC-AUC | Recall @ 5% Budget | Brier Score | Selected for Production |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | Median Imputed + Scaled | 0.1154 | 0.7385 | 21.66% | 0.03598 | No |
| **Random Forest (Baseline)** | Median Imputed | 0.1413 | 0.7617 | 24.88% | 0.03541 | No |
| **XGBoost** | Native NaN | 0.2139 | 0.8012 | 30.41% | 0.03450 | No |
| **LightGBM (Default)** | Native NaN | 0.2036 | 0.7978 | 31.34% | 0.03418 | No |
| **LightGBM (Median Imputed)** | Median Imputed | 0.2097 | 0.7945 | 30.41% | 0.03429 | No |
| **LightGBM (Optimized)** | **Native NaN (Informative)** | **0.2148** | **0.7976** | **31.80%** | **0.03345** | **✅ YES (Production)** |

### 2. Untouched Holdout Evaluation (N = 8,681)
- **Holdout PR-AUC:** **`0.3394`** vs **`0.0389`** No-Skill Baseline (**8.7x enrichment**)
- **Holdout ROC-AUC:** **`0.8179`**
- **Brier Reliability Score:** **`0.03144`** (Isotonic probability calibration)
- **Holdout Recall @ Top 5% Review Budget:** **`32.18%`**

### 3. Leakage Guard: The Polish Attr27 Missingness Artifact
- **The Artifact:** In the raw Polish bankruptcy dataset, 309 records had missing `Attr27` (EBIT / financial expenses) because they were insolvent or reported zero interest expenses. **38.8%** of those firms defaulted (vs 2.3% baseline).
- **The Pitfall:** Naive models exploit `Attr27 == NaN` as a target leakage shortcut to claim an artificial **~0.78 PR-AUC**, which collapses when applied to real Indian MCA/SEBI corporate disclosures.
- **CreditWatch Solution:** We systematically audited and purged the missingness artifact, validating that the genuine leak-free performance is **0.370** on Polish data and **0.3394** on the 39 India-portable financial ratios.

### 4. Asymmetric Error-Cost Matrix Policy
Under commercial lending realities, missing an impending corporate insolvency is catastrophic ($C_{FN} = 20$), whereas auditing a solvent firm is merely an operational review overhead ($C_{FP} = 1$):
$$\text{Expected Cost} = 20 \cdot \text{FN} + 1 \cdot \text{FP}$$
The theoretical Bayes-optimal decision threshold is:
$$t^* = \frac{C_{FP}}{C_{FN} + C_{FP}} = \frac{1}{20 + 1} \approx 0.0476 \quad (4.76\%)$$
Evaluating the calibrated ensemble under this policy delivers a **45.7% expected cost reduction** over non-monitoring.

### 5. Empirical Indian Corporate Panel Backtest (10 Defaulters + 32 Solvent Peers)
- **10 Core Corporate Insolvencies:** Jet Airways, Reliance Capital, Future Retail, Coffee Day Enterprises, Alok Industries, Bhushan Steel, Sintex Industries, Videocon Industries, Reliance Communications, Gitanjali Gems.
- **4Q-Lead Recall (Model-Only):** **38.3%**
- **4Q-Lead Recall (with EWS Forensic & Soft-Signal Overlay):** **60.0%**
- **8Q-Lead Recall (2 Years Prior):** **37.5%**

---

## 🌐 Web Application Features & Architecture

The application comprises **7 dedicated institutional pages**:

1. **Executive Dashboard (`DashboardPage.tsx`):**
   - Live KPI cards: Total Portfolio, High-Risk Borrowers, Average Predicted Risk, 4Q-Lead Recall.
   - Sector Vulnerability Bar Chart & Portfolio Risk Bands Pie Chart.
   - High-Risk Borrowers Ranking Table with deep-dive triggers.
   - Live EWS Soft Signals & Governance Alert Feed.

2. **Company Risk Dossier (`CompanyAnalysisPage.tsx`):**
   - Interactive dropdown for all 42 Indian companies or **Manual Financials Form** for custom statements.
   - Risk Tiers (High / Medium / Low) with Bayes threshold flags ($p \ge 4.76\%$).
   - 4 Forensic Cards: Altman Z″-score, Beneish M-score, Piotroski F-score, and Interest Coverage Ratio.
   - Qualitative soft signals (auditor resignations, promoter share pledge > 80%, filing delays).
   - Local **SHAP Waterfall Bar Chart** powered by TreeExplainer log-odds contributions with peer medians.
   - Credit Officer EWS Action Memo with recommended underwriting steps.

3. **Forensic Accounting & Accounting Anomaly Workspace (`ForensicsPage.tsx`):**
   - Beneish M-Score 8-Variable Mechanics (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA).
   - Interactive portfolio forensic cross-section table with filters (Beneish flagged, Altman distress, ICR < 1.0).
   - Prominent disclaimer: Beneish M is an accounting manipulation screening indicator, not a judicial fraud verdict.
   - Architecture boundary: Corporate credit distress vs payment transaction fraud.

4. **Portfolio Watchlist & Review Queue (`WatchlistPage.tsx`):**
   - Review budget capacity slider (1% to 25%, defaulting to the standard 5% cap / 40 of 800 loans).
   - Strict separation between Priority Mandated Review (within budget cap) and Overflow Watchlist.
   - Multi-field sorting, filtering by sector and risk tier.
   - Direct CSV export (`/api/watchlist/export`).

5. **Model Benchmarking & Retraining (`ModelPerformancePage.tsx`):**
   - Side-by-side multi-model comparison table.
   - Precision-Recall Curve, ROC Curve, and Isotonic Calibration Reliability Diagram.
   - Polish Attr27 missingness artifact ablation study.
   - Controlled Retrain Model modal with confirmation safeguards, background CV, and artifact backups.

6. **Data Ingestion & Schema Validator (`DataUploadPage.tsx`):**
   - Drag-and-drop CSV / XLSX file uploader.
   - Automatic schema validation against 39 portable financial line items.
   - Missing fields report and 5-row raw preview.
   - Downloadable sample CSV templates.

7. **Methodology & Policy (`MethodologyPage.tsx`):**
   - Mathematical documentation of the 20:1 cost function and Bayes-optimal cutoff.
   - 5% review cap operational rationale.
   - Temporal information barriers ($t_{public} \le t_{obs}$).
   - NBFC / Commercial bank limitations documentation.

---

## ⚠️ Important Institutional Limitations & Boundaries

1. **Commercial Banks & NBFCs:**
   - Traditional industrial distress ratios (Altman Z″, Beneish M, Working Capital, Gross Margin) are mathematically invalid for financial institutions (e.g. DHFL, Reliance Capital, Yes Bank) due to liability inversion (deposits are liabilities, loans are assets).
   - CreditWatch flags financial entities as *"NBFC / Financial"* and applies modified Basel capital/leverage overlays.

2. **Corporate Distress vs Payment Card Transaction Fraud:**
   - CreditWatch evaluates annual and quarterly corporate financial disclosures to predict insolvency and earnings manipulation risk.
   - It does not process streaming transaction logs (ISO 8583 / ISO 20022), card-not-present flags, or user device telemetry for transaction fraud. A future integration blueprint for Kafka + Graph Neural Networks is outlined in the Forensics module.

---

## 📁 Repository Structure
```
creditwatch-ews/
├── app.py                      # Original Streamlit application
├── run_all.py                  # Full pipeline runner & retraining CLI
├── score_any.py                # Standalone arbitrary file scoring CLI
├── requirements.txt            # Python dependencies
├── .env.example                # Environment configuration template
├── README.md                   # System documentation
├── SUBMISSION.md               # Zeckathon submission report
├── artifacts/                  # Calibrated models, benchmark JSONs, curves, and watchlist exports
├── data/
│   ├── indian/                 # 10 Defaulters + 32 Peers financial data, events, soft signals
│   └── polish/                 # 5 Polish bankruptcy horizon datasets
├── src/
│   ├── api.py                  # FastAPI backend server
│   ├── model.py                # ML pipeline: multi-model CV, LightGBM, calibration, holdout metrics
│   ├── features.py             # 39 India-portable financial ratios & mapping
│   ├── scores.py               # Altman Z'', Beneish M, Piotroski F, ICR
│   ├── rank.py                 # EWS overlays, SHAP TreeExplainer, watchlist ranking
│   ├── indian.py               # Indian corporate evaluation harness
│   ├── screener.py             # Rule screener & anomaly detectors
│   └── ingest.py               # Data templates and file parsing
├── tests/
│   ├── test_pipeline.py        # Core ML pipeline & leakage unit tests
│   └── test_api.py             # FastAPI endpoints & scoring integration tests
└── frontend/                   # React 18 + TypeScript + Tailwind CSS Web Application
    ├── index.html
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── src/
        ├── api.ts              # Typed API client
        ├── types.ts            # Data models
        ├── index.css           # Design system tokens & utility styles
        ├── App.tsx             # Enterprise shell & active tab navigation
        └── pages/              # 7 Full-featured institutional dashboard pages
```
