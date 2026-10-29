import React from 'react';
import {
  AlertTriangle,
  Award,
  BookOpen,
  Calendar,
  CheckCircle,
  Database,
  ExternalLink,
  FileText,
  HelpCircle,
  Info,
  Layers,
  Lock,
  Scale,
  Shield,
  ShieldAlert,
  Sliders,
  TrendingUp,
  UserCheck
} from 'lucide-react';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 pb-5">
        <div className="flex items-center gap-2">
          <span className="badge badge-primary">Model Governance & Risk Policy</span>
          <span className="badge badge-neutral">Institutional Underwriting Standard</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
          CreditWatch EWS Methodology, Policy & Theoretical Framework
        </h1>
        <p className="text-sm text-slate-600 mt-1">
          Complete mathematical documentation of the asymmetric cost matrix, Bayes-optimal threshold,
          target-leakage controls, 39 India-portable features, and corporate vs financial institution boundaries.
        </p>
      </div>

      {/* 1. Asymmetric Cost Matrix & Bayes-Optimal Cutoff */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Scale className="w-5 h-5 text-brand-700" />
          <h2 className="text-base font-bold text-slate-900">
            1. Asymmetric Cost Matrix & The Bayes-Optimal Decision Threshold
          </h2>
        </div>

        <p className="text-xs text-slate-700 leading-relaxed">
          Standard machine learning classifiers default to a symmetric 50% probability threshold ($t = 0.50$).
          In corporate credit risk and commercial lending, this assumption is catastrophic:
          the failure to detect an impending corporate insolvency (False Negative) results in massive loan write-offs
          and principal capital destruction, whereas an unnecessary audit of a solvent company (False Positive)
          merely incurs modest credit analyst review overhead.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 bg-red-50 rounded-lg border border-red-200 space-y-2">
            <div className="font-bold text-red-950 text-sm flex items-center justify-between">
              <span>{"False Negative Cost (C_FN = 20)"}</span>
              <span className="badge badge-danger">20x Weight</span>
            </div>
            <p className="text-red-900 leading-relaxed">
              Defaulted exposure resulting in NCLT restructuring, haircut, or liquidation.
              Loss Given Default (LGD) typically ranges between 60% and 85% of outstanding principal.
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
            <div className="font-bold text-slate-950 text-sm flex items-center justify-between">
              <span>{"False Positive Cost (C_FP = 1)"}</span>
              <span className="badge badge-neutral">1x Weight</span>
            </div>
            <p className="text-slate-700 leading-relaxed">
              An analyst spends 2–4 hours reviewing financial filings and covenant compliance for a firm that ultimately survives.
            </p>
          </div>
        </div>

        {/* Mathematical Formulation Box */}
        <div className="p-4 bg-slate-900 text-slate-100 rounded-lg font-mono text-xs space-y-2">
          <div className="text-brand-300 font-bold">Bayes-Optimal Decision Rule:</div>
          <div className="text-slate-300">
            {"Expected Loss(y, ŷ) = C_FN · P(y=1) · I(ŷ=0) + C_FP · P(y=0) · I(ŷ=1)"}
          </div>
          <div className="text-slate-300">
            {"Minimizing expected cost yields the optimal decision cutoff t*:"}
          </div>
          <div className="text-emerald-400 font-bold text-sm">
            {"t* = C_FP / (C_FN + C_FP) = 1 / (20 + 1) = 1/21 ≈ 0.0476 (4.76%)"}
          </div>
          <p className="text-slate-400 text-[11px] font-sans">
            Any borrower with a calibrated distress probability $p \ge 4.76\%$ is flagged for early warning oversight.
            On the benchmark hold-out (simulated 800-name book, 2% default rate) reviewing the top 40 cuts expected cost by <strong>45%</strong> versus no monitoring (175.6 vs 320 cost units per quarter).
          </p>
        </div>
      </div>

      {/* 2. Operational Capacity Constrained Review Budget (5%) */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Sliders className="w-5 h-5 text-amber-600" />
          <h2 className="text-base font-bold text-slate-900">
            2. The Operational Review Budget Cap (Top 5% Portfolio Scenario)
          </h2>
        </div>

        <p className="text-xs text-slate-700 leading-relaxed">
          While the theoretical Bayes cutoff ($t^* = 4.76\%$) identifies all economically vulnerable borrowers,
          commercial banks operate under strict headcount and review capacity constraints.
          For instance, a risk department monitoring <strong>800 commercial loans</strong> may have capacity to deeply
          audit only <strong>40 companies per quarter</strong> (a 5% operational budget).
        </p>

        <div className="p-4 bg-amber-50 rounded-lg border border-amber-200 text-xs text-amber-900 space-y-2">
          <div className="font-bold text-sm text-amber-950">
            Separation of Mandated Priority Review vs Overflow Watchlist
          </div>
          <p className="leading-relaxed">
            CreditWatch introduces a dual-tier priority queue:
          </p>
          <ul className="list-disc list-inside space-y-1 text-amber-900">
            <li>
              <strong>Priority Review Queue:</strong> The top 5% of highest-risk borrowers strictly ranked by calibrated risk
              and forensic triggers, guaranteed immediate analyst deep-dive.
            </li>
            <li>
              <strong>Overflow Watchlist:</strong> Borrowers exceeding the Bayes threshold ($p \ge 4.76\%$) but outside the
              top 5% review cap are placed on secondary monitoring with automated covenant triggers.
            </li>
          </ul>
        </div>
      </div>

      {/* 3. Leakage Prevention & Data Freshness */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <ShieldAlert className="w-5 h-5 text-red-600" />
          <h2 className="text-base font-bold text-slate-900">
            3. Strict Data Leakage Controls & Temporal Information Barriers
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="space-y-2">
            <h4 className="font-bold text-slate-900 flex items-center gap-1.5">
              <CheckCircle className="w-4 h-4 text-emerald-600" />
              Temporal Barrier: {"t_public ≤ t_obs"}
            </h4>
            <p className="text-slate-600 leading-relaxed">
              Every quarterly/annual evaluation strictly enforces that the financial statement publication date
              must precede the observation date. No post-default earnings statements, subsequent NCLT filings,
              or future rating actions are ever passed into the feature pipeline.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-900 flex items-center gap-1.5">
              <CheckCircle className="w-4 h-4 text-emerald-600" />
              Purging the Polish Attr27 Missingness Artifact
            </h4>
            <p className="text-slate-600 leading-relaxed">
              A 64-attribute model reaches ~0.83 PR-AUC on the Polish data partly by exploiting Attr27
              (EBIT / finance expenses) missingness. We established that this missingness pattern was an artifact
              of data collection that does not exist in Indian MCA/SEBI filings. Purging it yields an honest,
              reproducible <strong>0.486 hold-out PR-AUC</strong> (4Q model, baseline 0.070), with every remaining missing ratio median-imputed so missingness carries no signal.
            </p>
          </div>
        </div>
      </div>

      {/* 4. The 39 India-Portable Financial Ratios */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Layers className="w-5 h-5 text-indigo-600" />
          <h2 className="text-base font-bold text-slate-900">
            4. The 39 India-Portable Financial Ratios Architecture
          </h2>
        </div>

        <p className="text-xs text-slate-700 leading-relaxed">
          The original Polish dataset contained 64 financial ratios. Rather than relying on all 64 attributes
          (many of which require specialized European accounting line items), CreditWatch maps raw Indian corporate
          disclosures (Schedule III / Ind AS) into <strong>39 mathematically portable financial indicators</strong>:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="font-bold text-slate-900 block mb-1">Liquidity & Solvency</span>
            <ul className="text-slate-600 space-y-0.5 text-[11px]">
              <li>• Current Ratio (CA / CL)</li>
              <li>• Quick Ratio</li>
              <li>• Cash / Total Assets</li>
              <li>• Working Capital / TA</li>
            </ul>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="font-bold text-slate-900 block mb-1">Debt & Leverage</span>
            <ul className="text-slate-600 space-y-0.5 text-[11px]">
              <li>• Total Liabilities / TA</li>
              <li>• Debt to Equity</li>
              <li>• Equity / Total Assets</li>
              <li>• Long-term Debt Burden</li>
            </ul>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="font-bold text-slate-900 block mb-1">Profitability & Margin</span>
            <ul className="text-slate-600 space-y-0.5 text-[11px]">
              <li>• Return on Assets (EBIT/TA)</li>
              <li>• Net Profit Margin</li>
              <li>• Gross Margin</li>
              <li>• Retained Earnings / TA</li>
            </ul>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="font-bold text-slate-900 block mb-1">Cash Flow & Efficiency</span>
            <ul className="text-slate-600 space-y-0.5 text-[11px]">
              <li>• Operating CFO / Liabilities</li>
              <li>• Asset Turnover (Sales/TA)</li>
              <li>• Interest Coverage (EBIT/FinExp)</li>
              <li>• Receivable Velocity</li>
            </ul>
          </div>
        </div>
      </div>

      {/* 5. Limitations: Commercial Banks & NBFCs */}
      <div className="card border-l-4 border-l-amber-500 space-y-3">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-600" />
          <h2 className="text-base font-bold text-slate-900">
            5. Institutional Limitations: Banks, NBFCs & Housing Finance
          </h2>
        </div>

        <p className="text-xs text-slate-700 leading-relaxed">
          Standard industrial distress models (including Altman Z″, Beneish M-score, and working capital ratios)
          are mathematically inapplicable to commercial banks and Non-Banking Financial Companies (NBFCs)
          such as DHFL, Reliance Capital, or Yes Bank:
        </p>

        <ul className="list-disc list-inside text-xs text-slate-600 space-y-1">
          <li>
            <strong>Liability Inversion:</strong> For banks, customer deposits are liabilities, while loans extended are assets.
            A traditional &quot;Debt / Equity&quot; ratio exceeding 10x is standard for a bank but indicates insolvency for a manufacturer.
          </li>
          <li>
            <strong>No Inventory / COGS:</strong> Financial institutions have zero cost of goods sold or physical inventory,
            rendering gross margin and asset turnover calculations meaningless.
          </li>
          <li>
            <strong>CreditWatch Treatment:</strong> When evaluating financial firms, CreditWatch flags them as <em>&quot;NBFC / Financial&quot;</em>,
            suppresses invalid industrial scores (Altman/Beneish), and highlights regulatory capital adequacy, NPA ratios, and asset-liability maturity mismatches.
          </li>
        </ul>
      </div>

      {/* 6. Indian Corporate Panel Evaluation Summary */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Award className="w-5 h-5 text-emerald-600" />
          <h2 className="text-base font-bold text-slate-900">
            6. Empirical Indian Corporate Panel Performance
          </h2>
        </div>

        <p className="text-xs text-slate-700 leading-relaxed">
          Pre-registered test on <strong>10 Indian defaulters</strong> (Jet Airways, DHFL, Reliance Capital, Future Retail, Coffee Day,
          Alok, Bhushan Steel, Reliance Communications, Sintex, IL&amp;FS Transportation Networks; earliest documented default date, each sourced)
          against <strong>31 matched non-defaulting peers</strong>. Alert = above the peers' 95th-percentile score (the 5% budget).
          Honest result: the Polish-trained model does not transfer to standalone Indian statements (defaulter-vs-peer AUC 0.46);
          simple debt-service ratios do carry signal (cash earnings / liabilities AUC 0.77 at 4Q, exploratory).
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-center">
            <span className="text-slate-500 font-semibold block">4Q Lead Recall (Model Only)</span>
            <span className="text-2xl font-bold text-slate-900 mt-1 block">2 / 10</span>
            <span className="text-[11px] text-slate-400">Pure financial ratio model</span>
          </div>

          <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200 text-center">
            <span className="text-emerald-900 font-semibold block">4Q Lead Recall (Full EWS Overlay)</span>
            <span className="text-2xl font-bold text-emerald-700 mt-1 block">0 / 10</span>
            <span className="text-[11px] text-emerald-800">+ Z/M/F overlay (Screener working-capital proxy is noisy) + verified rating signals</span>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-center">
            <span className="text-slate-500 font-semibold block">8Q Lead Recall (2 Years Prior)</span>
            <span className="text-2xl font-bold text-slate-900 mt-1 block">0 / 10</span>
            <span className="text-[11px] text-slate-400">Early warning lead detection</span>
          </div>
        </div>
      </div>
    </div>
  );
};
