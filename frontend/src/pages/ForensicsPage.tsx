import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  ChevronRight,
  Database,
  ExternalLink,
  FileCheck,
  FileText,
  Filter,
  Info,
  Layers,
  Lock,
  RefreshCw,
  Search,
  Shield,
  ShieldAlert,
  Zap
} from 'lucide-react';
import { fetchForensicsSummary } from '../api';

interface ForensicsPageProps {
  onSelectCompany: (companyName: string) => void;
}

export const ForensicsPage: React.FC<ForensicsPageProps> = ({ onSelectCompany }) => {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'beneish' | 'altman' | 'icr'>('all');
  const [search, setSearch] = useState<string>('');

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetchForensicsSummary();
        setData(res);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch forensic analytics');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading && !data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px]">
        <RefreshCw className="w-10 h-10 text-brand-600 animate-spin mb-4" />
        <p className="text-slate-600 font-medium">Scanning forensic indicators across portfolio...</p>
      </div>
    );
  }

  const companiesList = data?.companies || [];

  const filteredCompanies = companiesList.filter((c: any) => {
    const matchesSearch = c.company.toLowerCase().includes(search.toLowerCase()) ||
                          c.sector.toLowerCase().includes(search.toLowerCase());
    if (!matchesSearch) return false;
    if (filter === 'beneish') return c.beneish_flag?.includes('Manipulator');
    if (filter === 'altman') return c.altman_zone === 'Distress';
    if (filter === 'icr') return c.icr_flag;
    return true;
  });

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div className="border-b border-slate-200 pb-5">
        <div className="flex items-center gap-2">
          <span className="badge badge-warning">Forensic Accounting & Anomaly Engine</span>
          <span className="badge badge-neutral">Regulatory Audit Grade</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
          Financial Forensic Analytics & Accounting Anomaly Workspace
        </h1>
        <p className="text-sm text-slate-600 mt-1 max-w-3xl">
          Multi-layer forensic screening evaluating earnings quality, aggressive accruals, balance sheet erosion,
          statutory auditor departures, and promoter leverage.
        </p>
      </div>

      {/* Critical Methodological Disclaimer Banner */}
      <div className="bg-amber-50 border-l-4 border-l-amber-500 border-r border-t border-b border-amber-200 rounded-xl p-5 text-amber-900">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h3 className="font-bold text-sm text-amber-950">
              Crucial Analytical Distinction: Screening Indicator vs Fraud Verdict
            </h3>
            <p className="text-xs text-amber-900 leading-relaxed">
              The <strong>Beneish M-Score</strong> is a statistical earnings manipulation screener based on 8 financial ratios.
              A score above <strong>-1.78</strong> flags high probability of accounting manipulation or aggressive revenue recognition;
              it does <strong>not</strong> constitute judicial proof of financial fraud, embezzlement, or criminal conduct.
              Similarly, corporate distress is driven by operational insolvency and macro debt burdens, which must not be conflated with intentional malfeasance.
            </p>
          </div>
        </div>
      </div>

      {/* 4 Forensic Screening Pillars KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="metric-card border-l-4 border-l-amber-500">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">Beneish M-Score Flagged</span>
            <span className="p-2 bg-amber-50 text-amber-600 rounded-lg">
              <FileCheck className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-amber-600">{data?.beneish_manipulator_count || 0}</span>
            <span className="text-xs text-slate-500">of {data?.total_analyzed || 0} firms</span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            M &gt; -1.78 indicates high accrual & revenue distortion probability
          </p>
        </div>

        <div className="metric-card border-l-4 border-l-red-500">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">Altman Z″ Distress Zone</span>
            <span className="p-2 bg-red-50 text-red-600 rounded-lg">
              <ShieldAlert className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-red-600">{data?.altman_distress_count || 0}</span>
            <span className="text-xs text-slate-500">Corporates</span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            Z″ &lt; 1.10 emerging market insolvency threshold
          </p>
        </div>

        <div className="metric-card border-l-4 border-l-rose-500">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">ICR &lt; 1.0 (Debt Stress)</span>
            <span className="p-2 bg-rose-50 text-rose-600 rounded-lg">
              <AlertCircle className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-rose-600">{data?.icr_below_one_count || 0}</span>
            <span className="text-xs text-slate-500">Firms</span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            Operating profit insufficient to service debt coupon obligations
          </p>
        </div>

        <div className="metric-card border-l-4 border-l-slate-600">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">Piotroski Weak (0–3)</span>
            <span className="p-2 bg-slate-50 text-slate-700 rounded-lg">
              <Layers className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-slate-800">{data?.piotroski_weak_count || 0}</span>
            <span className="text-xs text-slate-500">Corporates</span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            Deteriorating operational efficiency, margins & cash flow
          </p>
        </div>
      </div>

      {/* Forensic Breakdown Architecture: The 8 Beneish Sub-Indices */}
      <div className="card">
        <div className="border-b border-slate-100 pb-3 mb-4">
          <h2 className="text-base font-bold text-slate-900">Beneish M-Score 8-Variable Forensic Mechanics</h2>
          <p className="text-xs text-slate-500">
            How CreditWatch decomposes audited balance sheets and P&L statements into forensic forensic indices
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">1. DSRI (Days Sales in Receivables)</div>
            <div className="text-slate-600 text-[11px] font-mono">(Rec_t / Sales_t) / (Rec_t-1 / Sales_t-1)</div>
            <p className="text-slate-500 text-[11px]">
              &gt; 1.0 indicates receivables outgrowing sales, suggesting early revenue recognition or channel stuffing.
            </p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">2. GMI (Gross Margin Index)</div>
            <div className="text-slate-600 text-[11px] font-mono">Margin_t-1 / Margin_t</div>
            <p className="text-slate-500 text-[11px]">
              &gt; 1.0 indicates deteriorating profit margins, creating pressure on management to engage in earnings manipulation.
            </p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">3. AQI (Asset Quality Index)</div>
            <div className="text-slate-600 text-[11px] font-mono">[1 - (CA + PPE) / TA]_t / [...]_t-1</div>
            <p className="text-slate-500 text-[11px]">
              &gt; 1.0 indicates capitalization of operating expenses into intangible or non-current asset buckets.
            </p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">4. SGI (Sales Growth Index)</div>
            <div className="text-slate-600 text-[11px] font-mono">Sales_t / Sales_t-1</div>
            <p className="text-slate-500 text-[11px]">
              High growth firms experiencing decelerations are statistically more prone to financial statement manipulation.
            </p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">5. DEPI (Depreciation Index)</div>
            <div className="text-slate-600 text-[11px] font-mono">DeprRate_t-1 / DeprRate_t</div>
            <p className="text-slate-500 text-[11px]">
              &gt; 1.0 indicates firm slowed depreciation rate to artificially inflate current period earnings.
            </p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">6. SGAI (SGA Expense Index)</div>
            <div className="text-slate-600 text-[11px] font-mono">(SGA_t / Sales_t) / (SGA_t-1 / Sales_t-1)</div>
            <p className="text-slate-500 text-[11px]">
              &gt; 1.0 indicates declining sales administrative efficiency, reducing operating leverage.
            </p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">7. LVGI (Leverage Index)</div>
            <div className="text-slate-600 text-[11px] font-mono">(Debt_t / TA_t) / (Debt_t-1 / TA_t-1)</div>
            <p className="text-slate-500 text-[11px]">
              &gt; 1.0 indicates rising total leverage, increasing covenant violation risk.
            </p>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
            <div className="font-bold text-slate-900">8. TATA (Total Accruals to TA)</div>
            <div className="text-slate-600 text-[11px] font-mono">(NetIncome - CFO) / TotalAssets</div>
            <p className="text-slate-500 text-[11px]">
              High positive accruals show earnings unsupported by operating cash generation (cash flow divergence).
            </p>
          </div>
        </div>
      </div>

      {/* Portfolio Forensic Cross-Section Table */}
      <div className="card">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4 pb-3 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900">Portfolio Forensic Matrix</h2>
            <p className="text-xs text-slate-500">Corporate balance sheet forensic scores, Altman solvency, and debt coverage</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-400" />
              <input
                type="text"
                placeholder="Search firm or sector..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-8 pr-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-xs font-medium focus:outline-none focus:ring-1 focus:ring-brand-500"
              />
            </div>

            <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg text-xs">
              <button
                onClick={() => setFilter('all')}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  filter === 'all' ? 'bg-white text-slate-900 shadow-sm font-bold' : 'text-slate-600'
                }`}
              >
                All ({companiesList.length})
              </button>
              <button
                onClick={() => setFilter('beneish')}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  filter === 'beneish' ? 'bg-white text-amber-800 shadow-sm font-bold' : 'text-slate-600'
                }`}
              >
                Beneish Flagged ({data?.beneish_manipulator_count || 0})
              </button>
              <button
                onClick={() => setFilter('altman')}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  filter === 'altman' ? 'bg-white text-red-700 shadow-sm font-bold' : 'text-slate-600'
                }`}
              >
                Altman Distress ({data?.altman_distress_count || 0})
              </button>
              <button
                onClick={() => setFilter('icr')}
                className={`px-2.5 py-1 rounded font-medium transition ${
                  filter === 'icr' ? 'bg-white text-rose-700 shadow-sm font-bold' : 'text-slate-600'
                }`}
              >
                ICR &lt; 1.0 ({data?.icr_below_one_count || 0})
              </button>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3">Company</th>
                <th className="py-2.5 px-3">Sector</th>
                <th className="py-2.5 px-3 text-center">Beneish M-Score</th>
                <th className="py-2.5 px-3 text-center">Accounting Screening</th>
                <th className="py-2.5 px-3 text-center">Altman Z″-Score</th>
                <th className="py-2.5 px-3 text-center">Solvency Zone</th>
                <th className="py-2.5 px-3 text-center">Piotroski F</th>
                <th className="py-2.5 px-3 text-center">ICR</th>
                <th className="py-2.5 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-medium">
              {filteredCompanies.map((c: any) => (
                <tr key={c.company} className="hover:bg-slate-50/70 transition">
                  <td className="py-3 px-3">
                    <button
                      onClick={() => onSelectCompany(c.company)}
                      className="font-bold text-brand-700 hover:underline flex items-center gap-1"
                    >
                      {c.company}
                      <ExternalLink className="w-3 h-3 text-slate-400" />
                    </button>
                  </td>
                  <td className="py-3 px-3 text-slate-600">{c.sector}</td>
                  <td className="py-3 px-3 text-center font-mono">
                    {c.beneish_m !== null ? c.beneish_m.toFixed(2) : 'N/A*'}
                  </td>
                  <td className="py-3 px-3 text-center">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      c.beneish_flag?.includes('Manipulator')
                        ? 'bg-amber-100 text-amber-800'
                        : 'bg-emerald-50 text-emerald-700'
                    }`}>
                      {c.beneish_flag || 'N/A'}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center font-mono">
                    {c.altman_z !== null ? c.altman_z.toFixed(2) : 'N/A*'}
                  </td>
                  <td className="py-3 px-3 text-center">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      c.altman_zone === 'Distress'
                        ? 'bg-red-100 text-red-800'
                        : c.altman_zone === 'Gray'
                        ? 'bg-amber-100 text-amber-800'
                        : 'bg-emerald-100 text-emerald-800'
                    }`}>
                      {c.altman_zone || 'N/A'}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center font-mono">
                    {c.piotroski_f !== null ? `${c.piotroski_f}/9` : 'N/A*'}
                  </td>
                  <td className="py-3 px-3 text-center font-mono">
                    <span className={c.icr_flag ? 'text-red-600 font-bold' : 'text-slate-700'}>
                      {c.interest_coverage_ratio !== null ? `${c.interest_coverage_ratio.toFixed(2)}x` : 'N/A'}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center">
                    <button
                      onClick={() => onSelectCompany(c.company)}
                      className="px-2 py-1 bg-slate-100 hover:bg-brand-50 text-brand-800 rounded text-[11px] font-semibold transition"
                    >
                      Dossier
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Honest Boundary: Corporate Distress vs Payment Transaction Fraud */}
      <div className="card bg-slate-900 text-white border-slate-800 space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
          <Database className="w-5 h-5 text-brand-400" />
          <h2 className="text-base font-bold text-white">
            Architecture Distinction: Corporate Credit Distress vs Transaction-Level Fraud
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300">
          <div className="space-y-2 bg-slate-950/60 p-4 rounded-lg border border-slate-800">
            <h4 className="font-bold text-brand-300 text-sm flex items-center gap-1.5">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              What CreditWatch Currently Detects (Production Ready)
            </h4>
            <ul className="list-disc list-inside space-y-1.5 text-slate-300">
              <li>Corporate credit distress, insolvency, and default likelihood 1–3 years ahead.</li>
              <li>Earnings manipulation risk via Beneish M-score (accruals vs cash flow divergence).</li>
              <li>Balance sheet degradation, hidden debt leverage, and severe interest service deficits.</li>
              <li>Qualitative governance shocks (auditor resignations, promoter share pledges, NCLT petitions).</li>
            </ul>
          </div>

          <div className="space-y-2 bg-slate-950/60 p-4 rounded-lg border border-slate-800">
            <h4 className="font-bold text-amber-300 text-sm flex items-center gap-1.5">
              <Lock className="w-4 h-4 text-amber-400" />
              What Requires Transaction-Level Integration (Out of Scope for Financials)
            </h4>
            <p className="leading-relaxed">
              The present datasets comprise audited annual corporate filings and regulatory disclosures.
              They do <strong>not</strong> contain:
            </p>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li>Card payment transaction logs (merchant category, geo-velocity, card-not-present flags).</li>
              <li>Wire transfer / RTGS transaction streams for money laundering (AML) detection.</li>
              <li>User device telemetry, IP velocity, or identity theft patterns.</li>
            </ul>
          </div>
        </div>

        {/* Future Integration Blueprint */}
        <div className="pt-2 border-t border-slate-800">
          <h4 className="font-bold text-white text-xs mb-2">Future Real-Time Transaction Fraud Architecture Blueprint</h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="p-3 bg-slate-800/60 rounded border border-slate-700">
              <span className="font-semibold text-brand-300">1. Streaming Pipeline:</span>
              <p className="text-[11px] text-slate-400 mt-1">
                Apache Kafka / Flink ingest for ISO 8583 / ISO 20022 message streams with sub-50ms latency.
              </p>
            </div>
            <div className="p-3 bg-slate-800/60 rounded border border-slate-700">
              <span className="font-semibold text-brand-300">2. Graph Neural Networks (GNN):</span>
              <p className="text-[11px] text-slate-400 mt-1">
                Heterogeneous transaction graph modeling circular fund routing and mule account networks.
              </p>
            </div>
            <div className="p-3 bg-slate-800/60 rounded border border-slate-700">
              <span className="font-semibold text-brand-300">3. Anomaly Autoencoders:</span>
              <p className="text-[11px] text-slate-400 mt-1">
                Semi-supervised deep reconstruction loss for zero-day fraud pattern isolation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
