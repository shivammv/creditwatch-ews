import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle,
  ChevronDown,
  Download,
  FileSpreadsheet,
  FileText,
  HelpCircle,
  Info,
  Layers,
  PieChart,
  RefreshCw,
  Search,
  ShieldAlert,
  Sliders,
  TrendingDown,
  TrendingUp,
  UserCheck
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { fetchCompanies, fetchCompanyDetails, scoreSingleCompany, ScoreInputParams } from '../api';
import { ScoredCompanyResult, ShapDriver } from '../types';

interface CompanyAnalysisPageProps {
  initialCompany?: string | null;
}

export const CompanyAnalysisPage: React.FC<CompanyAnalysisPageProps> = ({ initialCompany }) => {
  const [companies, setCompanies] = useState<Array<{ company: string; sector: string; status: string }>>([]);
  const [selectedCompany, setSelectedCompany] = useState<string>(initialCompany || 'Jet Airways (India) Ltd');
  const [analysisResult, setAnalysisResult] = useState<ScoredCompanyResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'select' | 'custom'>('select');

  // Custom form state
  const [customForm, setCustomForm] = useState<ScoreInputParams>({
    company_name: 'Custom Corporate Borrower',
    sector: 'Manufacturing',
    financial_year: 'FY24',
    total_assets: 1500.0,
    gross_profit: 220.0,
    ebit: 60.0,
    sales: 1200.0,
    net_income: -45.0,
    total_liabilities: 1100.0,
    current_assets: 550.0,
    current_liabilities: 620.0,
    cash: 40.0,
    equity: 400.0,
    retained_earnings: 80.0,
    financial_expenses: 95.0,
    cogs: 980.0,
    receivables: 280.0,
    depreciation: 40.0,
    cfo: 15.0,
    soft_signals: ['promoter_pledging'],
  });

  // Load companies dropdown
  useEffect(() => {
    const loadComp = async () => {
      try {
        const list = await fetchCompanies();
        setCompanies(list);
        if (initialCompany) {
          setSelectedCompany(initialCompany);
        } else if (list.length > 0) {
          setSelectedCompany(list[0].company);
        }
      } catch (err: any) {
        console.error('Failed to load companies:', err);
      }
    };
    loadComp();
  }, [initialCompany]);

  // Load details when selectedCompany changes in 'select' mode
  const loadSelectedCompany = async (name: string) => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetchCompanyDetails(name);
      setAnalysisResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to score selected company');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (mode === 'select' && selectedCompany) {
      loadSelectedCompany(selectedCompany);
    }
  }, [selectedCompany, mode]);

  const handleCustomSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError(null);
      const res = await scoreSingleCompany(customForm);
      setAnalysisResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to score custom financial statement');
    } finally {
      setLoading(false);
    }
  };

  const toggleSoftSignal = (signal: string) => {
    setCustomForm((prev) => {
      const active = prev.soft_signals || [];
      if (active.includes(signal)) {
        return { ...prev, soft_signals: active.filter((s) => s !== signal) };
      } else {
        return { ...prev, soft_signals: [...active, signal] };
      }
    });
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Header & Mode Toggle */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="badge badge-primary">Credit Risk & Forensics</span>
            <span className="badge badge-neutral">Single Borrower Assessment</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
            Comprehensive Corporate Risk & Forensics Dossier
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Machine Learning distress probability, Altman Z″ solvency, Beneish M-score accounting manipulation,
            Piotroski F-score, and local SHAP factor attribution.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg">
          <button
            onClick={() => setMode('select')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
              mode === 'select' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Indian Corporate Panel ({companies.length})
          </button>
          <button
            onClick={() => setMode('custom')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
              mode === 'custom' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Custom Financials Input
          </button>
        </div>
      </div>

      {/* Select Mode Selector */}
      {mode === 'select' && (
        <div className="card bg-slate-50 border-slate-200">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex-1">
              <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                Select Company from Validated Indian Corporate Panel
              </label>
              <div className="relative">
                <select
                  value={selectedCompany}
                  onChange={(e) => setSelectedCompany(e.target.value)}
                  className="w-full bg-white border border-slate-300 rounded-lg py-2.5 px-3 pr-8 text-sm font-semibold text-slate-900 focus:outline-none focus:ring-2 focus:ring-brand-500 shadow-sm"
                >
                  <optgroup label="Core Default / Distress Cases (BSE/NSE/NCLT)">
                    {companies
                      .filter((c) => c.status === 'Defaulter')
                      .map((c) => (
                        <option key={c.company} value={c.company}>
                          ⚠️ {c.company} — {c.sector} (Distressed)
                        </option>
                      ))}
                  </optgroup>
                  <optgroup label="Solvent Industry Peers / Benchmark Corporates">
                    {companies
                      .filter((c) => c.status !== 'Defaulter')
                      .map((c) => (
                        <option key={c.company} value={c.company}>
                          ✓ {c.company} — {c.sector} (Solvent Peer)
                        </option>
                      ))}
                  </optgroup>
                </select>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0 pt-4 md:pt-0">
              <span className="text-xs text-slate-500 font-medium">Quick Picks:</span>
              {['Jet Airways (India) Ltd', 'Reliance Capital Ltd', 'Tata Motors Ltd', 'Infosys Ltd'].map((name) => (
                <button
                  key={name}
                  onClick={() => setSelectedCompany(name)}
                  className={`px-2 py-1 rounded text-xs font-medium border transition ${
                    selectedCompany === name
                      ? 'bg-brand-700 text-white border-brand-700'
                      : 'bg-white text-slate-700 border-slate-200 hover:border-brand-400'
                  }`}
                >
                  {name.split(' ')[0]}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Custom Input Mode Form */}
      {mode === 'custom' && (
        <form onSubmit={handleCustomSubmit} className="card bg-slate-50 border-slate-200 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3">
            <div>
              <h2 className="text-base font-bold text-slate-900">Custom Corporate Financial Statement Scoring</h2>
              <p className="text-xs text-slate-600">Enter raw balance sheet & P&L line items (in INR Crores or consistent units)</p>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary flex items-center gap-2 text-xs"
            >
              {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sliders className="w-3.5 h-3.5" />}
              Compute EWS Distress Risk
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Company Name</label>
              <input
                type="text"
                value={customForm.company_name}
                onChange={(e) => setCustomForm({ ...customForm, company_name: e.target.value })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Industry Sector</label>
              <select
                value={customForm.sector}
                onChange={(e) => setCustomForm({ ...customForm, sector: e.target.value })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              >
                <option value="Aviation">Aviation</option>
                <option value="Manufacturing">Manufacturing</option>
                <option value="Retail">Retail</option>
                <option value="Steel & Metals">Steel & Metals</option>
                <option value="Infrastructure">Infrastructure</option>
                <option value="Power & Utilities">Power & Utilities</option>
                <option value="IT & Tech">IT & Tech</option>
                <option value="Automotive">Automotive</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Total Assets (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.total_assets}
                onChange={(e) => setCustomForm({ ...customForm, total_assets: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Total Liabilities (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.total_liabilities}
                onChange={(e) => setCustomForm({ ...customForm, total_liabilities: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Annual Sales / Revenue (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.sales}
                onChange={(e) => setCustomForm({ ...customForm, sales: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">EBIT / Operating Profit (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.ebit}
                onChange={(e) => setCustomForm({ ...customForm, ebit: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Net Income / PAT (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.net_income}
                onChange={(e) => setCustomForm({ ...customForm, net_income: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Current Assets (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.current_assets}
                onChange={(e) => setCustomForm({ ...customForm, current_assets: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Current Liabilities (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.current_liabilities}
                onChange={(e) => setCustomForm({ ...customForm, current_liabilities: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Shareholders' Equity (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.equity}
                onChange={(e) => setCustomForm({ ...customForm, equity: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Financial / Interest Expense (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.financial_expenses}
                onChange={(e) => setCustomForm({ ...customForm, financial_expenses: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Operating Cash Flow (CFO) (₹ Cr)</label>
              <input
                type="number"
                step="any"
                value={customForm.cfo}
                onChange={(e) => setCustomForm({ ...customForm, cfo: parseFloat(e.target.value) || 0 })}
                className="w-full bg-white border border-slate-300 rounded px-2.5 py-1.5 text-xs font-medium"
              />
            </div>
          </div>

          {/* Qualitative Soft Signals checkboxes */}
          <div className="pt-2 border-t border-slate-200">
            <span className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Forensic Soft Signals & Qualitative Warning Triggers
            </span>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {[
                { id: 'auditor_resignation', label: 'Auditor Resignation' },
                { id: 'delay_in_filing', label: 'SEBI Filing Delay (>60d)' },
                { id: 'promoter_pledging', label: 'Promoter Pledge > 80%' },
                { id: 'rating_downgrade', label: 'Rating Downgrade (to D/C)' },
              ].map((sig) => (
                <label
                  key={sig.id}
                  className={`flex items-center gap-2 p-2 rounded border cursor-pointer text-xs font-medium transition ${
                    (customForm.soft_signals || []).includes(sig.id)
                      ? 'bg-red-50 border-red-300 text-red-800'
                      : 'bg-white border-slate-200 text-slate-600'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={(customForm.soft_signals || []).includes(sig.id)}
                    onChange={() => toggleSoftSignal(sig.id)}
                    className="rounded text-red-600 focus:ring-red-500"
                  />
                  {sig.label}
                </label>
              ))}
            </div>
          </div>
        </form>
      )}

      {/* Main Analysis Display */}
      {loading && !analysisResult && (
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <RefreshCw className="w-10 h-10 text-brand-600 animate-spin mb-4" />
          <p className="text-slate-600 font-medium">Computing 39 financial ratios & SHAP explanation...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-5 text-red-800">
          <div className="flex items-center gap-2 font-bold">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <span>Analysis Error</span>
          </div>
          <p className="text-sm mt-1">{error}</p>
        </div>
      )}

      {analysisResult && (
        <div className="space-y-6">
          {/* Header Card: Result & Risk Tier */}
          <div className={`card border-l-8 ${
            analysisResult.badge_color === 'red'
              ? 'border-l-red-600'
              : analysisResult.badge_color === 'amber'
              ? 'border-l-amber-500'
              : 'border-l-emerald-600'
          }`}>
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                    {analysisResult.sector}
                  </span>
                  {analysisResult.is_financial && (
                    <span className="badge badge-warning text-[10px]">
                      NBFC / Financial (Modified Basel Evaluation)
                    </span>
                  )}
                  {analysisResult.in_priority_review ? (
                    <span className="badge badge-danger text-[10px] font-bold">
                      Priority Review Cap (Top 5%)
                    </span>
                  ) : (
                    <span className="badge badge-neutral text-[10px]">
                      Standard Portfolio Track
                    </span>
                  )}
                </div>
                <h2 className="text-2xl font-bold text-slate-900 mt-1">
                  {analysisResult.company}
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Distress screening based on 39 India-portable financial ratios & non-linear gradient boosting
                </p>
              </div>

              {/* Probability & Status Gauges */}
              <div className="flex items-center gap-6 shrink-0 border-t lg:border-t-0 lg:border-l border-slate-200 pt-4 lg:pt-0 lg:pl-6">
                <div className="text-center">
                  <div className="text-xs font-semibold text-slate-500">Calibrated Distress Risk</div>
                  <div className={`text-3xl font-extrabold ${
                    analysisResult.badge_color === 'red'
                      ? 'text-red-600'
                      : analysisResult.badge_color === 'amber'
                      ? 'text-amber-600'
                      : 'text-emerald-600'
                  }`}>
                    {analysisResult.combined_risk_pct.toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Model: {(analysisResult.calibrated_model_probability * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-xs font-semibold text-slate-500">Risk Category</div>
                  <div className={`mt-1 px-3 py-1 rounded-full text-xs font-bold inline-block ${
                    analysisResult.badge_color === 'red'
                      ? 'bg-red-100 text-red-800'
                      : analysisResult.badge_color === 'amber'
                      ? 'bg-amber-100 text-amber-800'
                      : 'bg-emerald-100 text-emerald-800'
                  }`}>
                    {analysisResult.risk_category}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">
                    Bayes Cutoff: 4.76%
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-xs font-semibold text-slate-500">Action Status</div>
                  <div className="mt-1">
                    {analysisResult.above_bayes_threshold ? (
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-red-600">
                        <AlertTriangle className="w-3.5 h-3.5" /> Flagged
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-600">
                        <CheckCircle className="w-3.5 h-3.5" /> Normal
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1">
                    Cost 20:1 FN:FP
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Forensic Pillars: Altman, Beneish, Piotroski, ICR */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Altman Z'' */}
            <div className="card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500 uppercase">Altman Z″-Score</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  analysisResult.forensics.altman_zone === 'Distress'
                    ? 'bg-red-100 text-red-800'
                    : analysisResult.forensics.altman_zone === 'Gray'
                    ? 'bg-amber-100 text-amber-800'
                    : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {analysisResult.forensics.altman_zone}
                </span>
              </div>
              <div className="mt-2 text-2xl font-bold text-slate-900">
                {analysisResult.forensics.altman_z !== null
                  ? analysisResult.forensics.altman_z.toFixed(2)
                  : 'N/A (NBFC)'}
              </div>
              <p className="mt-1 text-[11px] text-slate-500">
                Emerging markets model: &lt; 1.10 = Distress, 1.10–2.60 = Gray Zone, &gt; 2.60 = Safe
              </p>
            </div>

            {/* Beneish M-Score */}
            <div className="card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500 uppercase">Beneish M-Score</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  analysisResult.forensics.beneish_flag.includes('Manipulator')
                    ? 'bg-amber-100 text-amber-800'
                    : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {analysisResult.forensics.beneish_flag}
                </span>
              </div>
              <div className="mt-2 text-2xl font-bold text-slate-900">
                {analysisResult.forensics.beneish_m !== null
                  ? analysisResult.forensics.beneish_m.toFixed(2)
                  : 'N/A (NBFC)'}
              </div>
              <p className="mt-1 text-[11px] text-slate-500">
                Accounting screening: &gt; -1.78 suggests earnings manipulation risk (not fraud verdict)
              </p>
            </div>

            {/* Piotroski F-Score */}
            <div className="card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500 uppercase">Piotroski F-Score</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  analysisResult.forensics.piotroski_band === 'Weak'
                    ? 'bg-red-100 text-red-800'
                    : analysisResult.forensics.piotroski_band === 'Moderate'
                    ? 'bg-amber-100 text-amber-800'
                    : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {analysisResult.forensics.piotroski_band}
                </span>
              </div>
              <div className="mt-2 text-2xl font-bold text-slate-900">
                {analysisResult.forensics.piotroski_f !== null
                  ? `${analysisResult.forensics.piotroski_f} / 9`
                  : 'N/A (NBFC)'}
              </div>
              <p className="mt-1 text-[11px] text-slate-500">
                9 fundamental criteria: 0–3 Weak, 4–6 Moderate, 7–9 Resilient balance sheet
              </p>
            </div>

            {/* Interest Coverage Ratio */}
            <div className="card">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500 uppercase">Interest Coverage (ICR)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  analysisResult.forensics.icr_below_one
                    ? 'bg-red-100 text-red-800'
                    : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {analysisResult.forensics.icr_below_one ? 'ICR < 1.0 (Critical)' : 'Adequate (> 1.0)'}
                </span>
              </div>
              <div className="mt-2 text-2xl font-bold text-slate-900">
                {analysisResult.forensics.interest_coverage_ratio !== null
                  ? `${analysisResult.forensics.interest_coverage_ratio.toFixed(2)}x`
                  : 'N/A'}
              </div>
              <p className="mt-1 text-[11px] text-slate-500">
                EBIT / Finance Expenses: &lt; 1.0 implies earnings fail to cover debt interest obligation
              </p>
            </div>
          </div>

          {/* Qualitative Soft Signals Banner */}
          {analysisResult.soft_signals.active_count > 0 && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
              <div className="flex items-center gap-2 font-bold text-amber-900 text-sm">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                <span>Active Forensic Red Flags Detected ({analysisResult.soft_signals.active_count})</span>
              </div>
              <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-amber-800">
                {analysisResult.soft_signals.signal_descriptions.map((desc, idx) => (
                  <div key={idx} className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-600" />
                    <span>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Local SHAP Waterfall / Feature Drivers Card */}
          <div className="card">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4 pb-3 border-b border-slate-100">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-bold text-slate-900">SHAP Explainability Waterfall</h3>
                  <span className="badge badge-primary text-[10px]">TreeExplainer v0.54</span>
                </div>
                <p className="text-xs text-slate-500">
                  Exact log-odds attribution showing which financial variables push risk upward vs downward
                </p>
              </div>
              <span className="text-xs text-slate-400">
                Reference Base Value: ~3.9% default prior
              </span>
            </div>

            {/* SHAP Drivers Chart */}
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={analysisResult.shap_drivers}
                  margin={{ top: 5, right: 30, left: 160, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    label={{ value: 'SHAP Contribution to Log-Odds', position: 'insideBottom', offset: -5, fontSize: 11 }}
                  />
                  <YAxis
                    type="category"
                    dataKey="label"
                    tick={{ fontSize: 11, fill: '#334155' }}
                    width={150}
                  />
                  <Tooltip
                    formatter={(val: any, name: any, item: any) => [
                      `${Number(val) > 0 ? '+' : ''}${Number(val).toFixed(3)} log-odds (${item.payload.text})`,
                      'SHAP Value'
                    ]}
                    contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '12px' }}
                  />
                  <ReferenceLine x={0} stroke="#94a3b8" />
                  <Bar dataKey="shap">
                    {analysisResult.shap_drivers.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.shap >= 0 ? '#dc2626' : '#16a34a'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Factor Details Table */}
            <div className="mt-4 pt-3 border-t border-slate-100 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-600 font-semibold">
                  <tr>
                    <th className="py-2 px-3">Risk Driver / Financial Metric</th>
                    <th className="py-2 px-3 text-right">Firm Value</th>
                    <th className="py-2 px-3 text-right">Peer Median</th>
                    <th className="py-2 px-3 text-right">SHAP Impact</th>
                    <th className="py-2 px-3">Plain-English Credit Interpretation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-medium">
                  {analysisResult.shap_drivers.map((drv, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/60">
                      <td className="py-2.5 px-3 font-semibold text-slate-900">{drv.label}</td>
                      <td className="py-2.5 px-3 text-right font-mono">
                        {drv.value !== null ? drv.value.toFixed(3) : 'Missing'}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-500">
                        {drv.peer_median.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono font-bold">
                        <span className={drv.shap > 0 ? 'text-red-600' : 'text-emerald-700'}>
                          {drv.shap > 0 ? `+${drv.shap.toFixed(3)}` : drv.shap.toFixed(3)}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-600">{drv.text}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-3 text-[11px] text-slate-400 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-slate-400" />
              <span>
                <strong>Methodology Note:</strong> SHAP values measure statistical attribution in the tree ensemble, not causal proof of insolvency or fraud.
              </span>
            </div>
          </div>

          {/* Credit Officer Action Memo & Recommendation */}
          <div className="card bg-slate-900 text-white">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <UserCheck className="w-5 h-5 text-brand-400" />
                <h3 className="text-base font-bold text-white">Credit Officer EWS Action Memo</h3>
              </div>
              <span className="text-xs text-slate-400 font-mono">CONFIDENTIAL // INTERNAL RISK</span>
            </div>

            <p className="text-sm text-slate-300 leading-relaxed font-mono whitespace-pre-line bg-slate-950/60 p-4 rounded-lg border border-slate-800">
              {analysisResult.watch_list_memo}
            </p>

            <div className="mt-4 pt-3 border-t border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="text-slate-400">
                <strong>Recommended Next Steps:</strong> {analysisResult.review_recommendation}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={() => alert(`Exporting Credit Memo for ${analysisResult.company}`)}
                  className="px-3 py-1.5 bg-brand-600 hover:bg-brand-500 text-white rounded text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  <Download className="w-3.5 h-3.5" /> Export Credit Memo PDF
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
