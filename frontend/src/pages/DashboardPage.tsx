import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Clock,
  ExternalLink,
  Info,
  Layers,
  PieChart as PieIcon,
  RefreshCw,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  Users
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { fetchDashboard } from '../api';
import { DashboardData } from '../types';

interface DashboardPageProps {
  onSelectCompany: (companyName: string) => void;
  onNavigateToWatchlist: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  onSelectCompany,
  onNavigateToWatchlist,
}) => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetchDashboard();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to CreditWatch backend');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading && !data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px]">
        <RefreshCw className="w-10 h-10 text-brand-600 animate-spin mb-4" />
        <p className="text-slate-600 font-medium">Loading executive risk intelligence...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6 my-6 text-red-800">
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-red-600" />
          <h3 className="font-semibold text-lg">Unable to load Executive Dashboard</h3>
        </div>
        <p className="mt-2 text-sm text-red-700">{error}</p>
        <button
          onClick={loadData}
          className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  const { kpis, model_performance_snippet, indian_panel_status, sectors, top_high_risk, recent_alerts } = data;

  const riskDistributionData = [
    { name: 'High Risk (Distress)', value: kpis.high_risk_count, color: '#dc2626' },
    { name: 'Medium Risk (Watch)', value: kpis.medium_risk_count, color: '#f59e0b' },
    { name: 'Low Risk (Healthy)', value: kpis.low_risk_count, color: '#16a34a' },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Title & Operational Context */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="badge badge-primary">EWS Portfolio Monitor</span>
            <span className="badge badge-success">Model v2.1-calibrated</span>
            <span className="text-xs text-slate-500 font-medium flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" /> Updated live
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
            Executive Financial Risk & EWS Dashboard
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl">
            Continuous early-warning distress screening calibrated on corporate fundamentals, asymmetric cost matrix
            (20:1 FN:FP), and forensic overlays (Altman Z″, Beneish M-Score, soft signals).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadData}
            disabled={loading}
            className="btn btn-secondary flex items-center gap-2 text-xs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={onNavigateToWatchlist}
            className="btn btn-primary flex items-center gap-2 text-xs"
          >
            <Users className="w-3.5 h-3.5" />
            View Full Watchlist ({kpis.total_companies_analyzed})
          </button>
        </div>
      </div>

      {/* KPI Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="metric-card border-l-4 border-l-brand-600">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Total Portfolio Analyzed</span>
            <span className="p-2 bg-brand-50 text-brand-700 rounded-lg">
              <Layers className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-slate-900">{kpis.total_companies_analyzed}</span>
            <span className="text-xs text-slate-500 font-medium">BSE/NSE Corporates</span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            10 Defaulters + 32 Industry Peers across 8 sectors
          </p>
        </div>

        <div className="metric-card border-l-4 border-l-red-500">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">High Risk Borrowers</span>
            <span className="p-2 bg-red-50 text-red-600 rounded-lg">
              <ShieldAlert className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-red-600">{kpis.high_risk_count}</span>
            <span className="text-xs font-semibold text-red-700 bg-red-50 px-2 py-0.5 rounded-full">
              {((kpis.high_risk_count / (kpis.total_companies_analyzed || 1)) * 100).toFixed(1)}% of pool
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            {kpis.flagged_for_review_budget} prioritized under review cap ({kpis.review_budget_cap_pct}%)
          </p>
        </div>

        <div className="metric-card border-l-4 border-l-amber-500">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Average Predicted Risk</span>
            <span className="p-2 bg-amber-50 text-amber-600 rounded-lg">
              <TrendingUp className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-slate-900">{kpis.avg_predicted_risk_pct}%</span>
            <span className="text-xs text-slate-500">
              Bayes Cutoff: {(kpis.bayes_threshold * 100).toFixed(1)}%
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            Optimal Bayes threshold t* = 1 / (1 + 20) = 4.76%
          </p>
        </div>

        <div className="metric-card border-l-4 border-l-emerald-600">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">4Q-Lead Recall (Defaulters)</span>
            <span className="p-2 bg-emerald-50 text-emerald-700 rounded-lg">
              <CheckCircle2 className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold text-emerald-700">{indian_panel_status.lead_4q_recall_full}%</span>
            <span className="text-xs text-slate-500 font-medium">with EWS Overlay</span>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            vs {indian_panel_status.lead_4q_recall_model_only}% model-only (38.3% unaugmented)
          </p>
        </div>
      </div>

      {/* Model Benchmark & Cost Matrix Banner */}
      <div className="bg-gradient-to-r from-slate-900 to-brand-900 rounded-xl p-5 text-white shadow-md">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-brand-500 text-white tracking-wide uppercase">
                Validated Evaluation
              </span>
              <span className="text-xs text-slate-300">
                Untouched Holdout PR-AUC: <strong className="text-white">{model_performance_snippet.pr_auc.toFixed(3)}</strong> (vs baseline {model_performance_snippet.pr_auc_baseline.toFixed(3)})
              </span>
            </div>
            <p className="text-sm text-slate-200">
              Asymmetric Cost Policy: <strong>20x Cost for False Negatives</strong> (undetected default) vs <strong>1x for False Positives</strong> (unnecessary audit).
              Achieves <strong>{model_performance_snippet.cost_reduction_vs_none.toFixed(1)}% expected cost reduction</strong> over naive zero-detection baseline.
            </p>
          </div>
          <div className="flex items-center gap-6 shrink-0 border-t md:border-t-0 md:border-l border-slate-700 pt-3 md:pt-0 md:pl-6">
            <div>
              <div className="text-xs text-slate-400">Holdout ROC-AUC</div>
              <div className="text-xl font-bold text-emerald-400">{model_performance_snippet.roc_auc.toFixed(3)}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Brier Score</div>
              <div className="text-xl font-bold text-brand-300">{model_performance_snippet.brier_score.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Recall @ Top 5%</div>
              <div className="text-xl font-bold text-amber-300">{model_performance_snippet.recall_at_5pct.toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sector Risk Breakdown */}
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
            <div>
              <h2 className="text-base font-bold text-slate-900">Sector-wise Vulnerability & Exposure</h2>
              <p className="text-xs text-slate-500">Average predicted distress risk and corporate count by sector</p>
            </div>
            <span className="text-xs text-slate-500 font-medium">8 Industry Groups</span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sectors} margin={{ top: 10, right: 20, left: 0, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis
                  dataKey="sector"
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  angle={-15}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  unit="%"
                  domain={[0, 'auto']}
                />
                <Tooltip
                  formatter={(val: any, name: any) => [
                    name === 'avg_risk' ? `${val}% Risk` : `${val} firms`,
                    name === 'avg_risk' ? 'Avg Risk' : 'Company Count'
                  ]}
                  contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '12px' }}
                />
                <Legend
                  wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
                  formatter={(value) => value === 'avg_risk' ? 'Avg Predicted Risk (%)' : 'Firm Count'}
                />
                <Bar dataKey="avg_risk" fill="#1e3a5f" radius={[4, 4, 0, 0]} name="avg_risk" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 text-xs text-slate-500 flex items-center justify-between border-t border-slate-100 pt-3">
            <span>Aviation, Retail & NBFC sectors show elevated debt service vulnerability.</span>
            <span className="font-semibold text-slate-700">Calibrated against FY18–FY24 filings</span>
          </div>
        </div>

        {/* Portfolio Risk Distribution */}
        <div className="card flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
              <div>
                <h2 className="text-base font-bold text-slate-900">Portfolio Risk Bands</h2>
                <p className="text-xs text-slate-500">Distribution across operational review tiers</p>
              </div>
              <PieIcon className="w-4 h-4 text-slate-400" />
            </div>

            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskDistributionData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={3}
                  >
                    {riskDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(val: any) => [`${val} companies`, 'Count']}
                    contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '12px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="space-y-2 pt-2 border-t border-slate-100">
            {riskDistributionData.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="font-medium text-slate-700">{item.name}</span>
                </div>
                <span className="font-bold text-slate-900">{item.value} ({((item.value / kpis.total_companies_analyzed) * 100).toFixed(0)}%)</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* High-Risk Borrowers Table & Recent Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Priority Review Ranking */}
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-900">High-Risk Watchlist Ranking</h2>
                <span className="badge badge-danger">Action Required</span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">Top-ranked corporate exposures requiring active credit review</p>
            </div>
            <button
              onClick={onNavigateToWatchlist}
              className="text-xs font-semibold text-brand-700 hover:text-brand-900 flex items-center gap-1"
            >
              All Watchlist Entries <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-2.5 px-3"># Rank</th>
                  <th className="py-2.5 px-3">Borrower / Firm</th>
                  <th className="py-2.5 px-3">Sector</th>
                  <th className="py-2.5 px-3 text-right">EWS Risk</th>
                  <th className="py-2.5 px-3 text-center">Altman Z″</th>
                  <th className="py-2.5 px-3 text-center">Beneish M</th>
                  <th className="py-2.5 px-3">Top Risk Drivers</th>
                  <th className="py-2.5 px-3 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium">
                {top_high_risk.slice(0, 7).map((borrower) => (
                  <tr key={borrower.company} className="hover:bg-slate-50/80 transition">
                    <td className="py-3 px-3 font-bold text-slate-700">#{borrower.rank}</td>
                    <td className="py-3 px-3">
                      <button
                        onClick={() => onSelectCompany(borrower.company)}
                        className="font-bold text-brand-700 hover:underline text-left flex items-center gap-1"
                      >
                        {borrower.company}
                        <ExternalLink className="w-3 h-3 text-slate-400" />
                      </button>
                    </td>
                    <td className="py-3 px-3 text-slate-600">{borrower.sector}</td>
                    <td className="py-3 px-3 text-right">
                      <span className="px-2 py-0.5 rounded font-bold bg-red-50 text-red-700 border border-red-200">
                        {borrower.risk_pct.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span className={`px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                        (borrower.altman_z !== null && borrower.altman_z < 1.1)
                          ? 'bg-red-100 text-red-800'
                          : 'bg-slate-100 text-slate-700'
                      }`}>
                        {borrower.altman_z !== null ? borrower.altman_z.toFixed(2) : 'N/A*'}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span className={`px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                        (borrower.beneish_m !== null && borrower.beneish_m > -1.78)
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-emerald-50 text-emerald-700'
                      }`}>
                        {borrower.beneish_m !== null ? borrower.beneish_m.toFixed(2) : 'N/A*'}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-slate-600 truncate max-w-[200px]" title={borrower.drivers}>
                      {borrower.drivers || 'Extreme leverage, debt service deficit'}
                    </td>
                    <td className="py-3 px-3 text-center">
                      <button
                        onClick={() => onSelectCompany(borrower.company)}
                        className="px-2 py-1 bg-slate-100 hover:bg-brand-50 text-brand-800 rounded text-[11px] font-semibold transition"
                      >
                        Deep Dive
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[11px] text-slate-400">
            * Note: Non-industrial/NBFC entities (e.g. DHFL, Reliance Capital) have non-applicable industrial ratios per Reserve Bank of India / Basel guidelines.
          </p>
        </div>

        {/* Live EWS Alert Feed */}
        <div className="card flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
              <div>
                <h2 className="text-base font-bold text-slate-900">Recent Soft Signals & Red Flags</h2>
                <p className="text-xs text-slate-500">Forensic qualitative triggers</p>
              </div>
              <span className="badge badge-warning">EWS Alerts</span>
            </div>

            <div className="space-y-3">
              {recent_alerts.map((alert, idx) => (
                <div
                  key={idx}
                  onClick={() => onSelectCompany(alert.company)}
                  className="p-3 rounded-lg border border-slate-200 hover:border-brand-300 hover:bg-brand-50/40 cursor-pointer transition text-xs space-y-1.5"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-900">{alert.company}</span>
                    <span className="text-[10px] text-slate-500 font-medium">{alert.date}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                      alert.severity === 'high' ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'
                    }`}>
                      {alert.signal_label}
                    </span>
                    <span className="text-slate-600 truncate">
                      {alert.signal === 'auditor_resignation'
                        ? 'Statutory auditor resigned citing scope limitation'
                        : alert.signal === 'delay_in_filing'
                        ? 'SEBI LODR filing delay exceeding 60 days'
                        : alert.signal === 'promoter_pledging'
                        ? 'Promoter share pledge exceeds 80%'
                        : 'Credit rating downgraded to default/distress (D/C)'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <Info className="w-3.5 h-3.5 text-slate-400" />
              Source: SEBI filings & MCA records
            </span>
            <button
              onClick={onNavigateToWatchlist}
              className="text-brand-700 hover:underline font-semibold"
            >
              All Signals →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
