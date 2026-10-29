import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowUpDown,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Download,
  ExternalLink,
  Eye,
  Filter,
  Info,
  RefreshCw,
  Search,
  ShieldAlert,
  Sliders
} from 'lucide-react';
import { fetchWatchlist } from '../api';
import { WatchlistItem, WatchlistResponse } from '../types';

interface WatchlistPageProps {
  onSelectCompany: (companyName: string) => void;
}

export const WatchlistPage: React.FC<WatchlistPageProps> = ({ onSelectCompany }) => {
  const [data, setData] = useState<WatchlistResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters & State
  const [budgetPct, setBudgetPct] = useState<number>(5.0);
  const [sector, setSector] = useState<string>('');
  const [riskCategory, setRiskCategory] = useState<string>('');
  const [search, setSearch] = useState<string>('');
  const [page, setPage] = useState<number>(1);
  const [limit] = useState<number>(15);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetchWatchlist({
        budget_pct: budgetPct,
        sector: sector || undefined,
        risk_category: riskCategory || undefined,
        search: search || undefined,
        page,
        limit,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch watchlist');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [budgetPct, sector, riskCategory, search, page]);

  const handleExportCSV = () => {
    const params = new URLSearchParams();
    params.set('budget_pct', budgetPct.toString());
    if (sector) params.set('sector', sector);
    if (riskCategory) params.set('risk_category', riskCategory);
    window.open(`/api/watchlist/export?${params.toString()}`, '_blank');
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="badge badge-primary">Portfolio Review Deck</span>
            <span className="badge badge-danger">Capacity Constrained</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
            Institutional Portfolio Watchlist & Review Queue
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl">
            Ranked risk queue enforcing realistic credit audit capacity (e.g., top 5% review budget / 40 of 800 loans).
            Separates high-priority mandated reviews from overflow monitoring.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleExportCSV}
            className="btn btn-secondary flex items-center gap-2 text-xs"
          >
            <Download className="w-3.5 h-3.5" />
            Export Filtered CSV
          </button>
        </div>
      </div>

      {/* Review Budget Capacity Slider Card */}
      <div className="card bg-slate-50 border-slate-200">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-1 max-w-xl">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-brand-700" />
              <h3 className="text-sm font-bold text-slate-900">
                Operational Credit Audit Review Budget Cap
              </h3>
              <span className="badge badge-primary text-[10px]">{budgetPct.toFixed(1)}% Capacity</span>
            </div>
            <p className="text-xs text-slate-600">
              In a realistic credit team with finite analytical bandwidth (e.g. 40 analysts reviewing 800 loan exposures),
              only top-ranked exposures can be audited each quarter.
            </p>
          </div>

          <div className="flex items-center gap-4 bg-white p-3 rounded-lg border border-slate-200 shrink-0">
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-semibold text-slate-700">
                <span>Budget Cap:</span>
                <span className="text-brand-700 font-bold">{budgetPct}%</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="25.0"
                step="0.5"
                value={budgetPct}
                onChange={(e) => {
                  setBudgetPct(parseFloat(e.target.value));
                  setPage(1);
                }}
                className="w-48 accent-brand-700 cursor-pointer"
              />
            </div>
            <div className="text-right pl-4 border-l border-slate-100">
              <div className="text-[10px] text-slate-400 font-medium uppercase">Priority Slots</div>
              <div className="text-xl font-extrabold text-slate-900">
                {data?.budget_cap_count || Math.round((42 * budgetPct) / 100)} / {data?.total_portfolio_size || 42}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filter Controls Bar */}
      <div className="card py-3 px-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center gap-3 flex-1 flex-wrap">
          <div className="relative min-w-[200px] flex-1 max-w-xs">
            <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search company or sector..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="w-full pl-8 pr-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-xs font-medium focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>

          <select
            value={sector}
            onChange={(e) => {
              setSector(e.target.value);
              setPage(1);
            }}
            className="bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:outline-none"
          >
            <option value="">All Sectors</option>
            <option value="Aviation">Aviation</option>
            <option value="Manufacturing">Manufacturing</option>
            <option value="Retail">Retail</option>
            <option value="Steel & Metals">Steel & Metals</option>
            <option value="Financial Services">Financial Services</option>
            <option value="Infrastructure">Infrastructure</option>
            <option value="Power & Utilities">Power & Utilities</option>
            <option value="IT & Tech">IT & Tech</option>
          </select>

          <select
            value={riskCategory}
            onChange={(e) => {
              setRiskCategory(e.target.value);
              setPage(1);
            }}
            className="bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:outline-none"
          >
            <option value="">All Risk Tiers</option>
            <option value="High Risk">High Risk</option>
            <option value="Medium Risk">Medium Risk</option>
            <option value="Low Risk">Low Risk</option>
          </select>
        </div>

        <div className="text-xs text-slate-500 flex items-center gap-2">
          <span>Showing <strong>{data?.items.length || 0}</strong> of <strong>{data?.total || 0}</strong> records</span>
        </div>
      </div>

      {/* Main Watchlist Table */}
      <div className="card p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
              <tr>
                <th className="py-3 px-3"># Priority Rank</th>
                <th className="py-3 px-3">Company / Borrower</th>
                <th className="py-3 px-3">Sector</th>
                <th className="py-3 px-3 text-right">EWS Risk</th>
                <th className="py-3 px-3 text-center">Operational Tier</th>
                <th className="py-3 px-3 text-center">Altman Z″</th>
                <th className="py-3 px-3 text-center">Beneish M</th>
                <th className="py-3 px-3 text-center">Piotroski F</th>
                <th className="py-3 px-3 text-center">Soft Signals</th>
                <th className="py-3 px-3">Primary Risk Drivers</th>
                <th className="py-3 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-medium">
              {loading && !data ? (
                <tr>
                  <td colSpan={11} className="py-8 text-center text-slate-500">
                    <RefreshCw className="w-6 h-6 text-brand-600 animate-spin mx-auto mb-2" />
                    Filtering institutional watchlist...
                  </td>
                </tr>
              ) : data?.items.length === 0 ? (
                <tr>
                  <td colSpan={11} className="py-8 text-center text-slate-500">
                    No corporate borrowers match the specified filter criteria.
                  </td>
                </tr>
              ) : (
                data?.items.map((item: WatchlistItem) => (
                  <tr
                    key={item.company}
                    className={`transition hover:bg-slate-50/80 ${
                      item.in_budget ? 'bg-red-50/30' : ''
                    }`}
                  >
                    <td className="py-3 px-3 font-bold text-slate-800">
                      <div className="flex items-center gap-1.5">
                        <span className={`w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold ${
                          item.in_budget
                            ? 'bg-red-600 text-white shadow-sm'
                            : 'bg-slate-100 text-slate-700'
                        }`}>
                          {item.rank}
                        </span>
                        {item.in_budget && (
                          <span className="text-[10px] font-bold text-red-600 uppercase">Cap</span>
                        )}
                      </div>
                    </td>

                    <td className="py-3 px-3">
                      <button
                        onClick={() => onSelectCompany(item.company)}
                        className="font-bold text-brand-700 hover:underline flex items-center gap-1 text-left"
                      >
                        {item.company}
                        <ExternalLink className="w-3 h-3 text-slate-400" />
                      </button>
                    </td>

                    <td className="py-3 px-3 text-slate-600">{item.sector}</td>

                    <td className="py-3 px-3 text-right">
                      <span className={`px-2 py-0.5 rounded font-mono font-bold text-xs ${
                        item.risk_pct >= 25
                          ? 'bg-red-100 text-red-800'
                          : item.risk_pct >= 10
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-emerald-50 text-emerald-800'
                      }`}>
                        {item.risk_pct.toFixed(1)}%
                      </span>
                    </td>

                    <td className="py-3 px-3 text-center">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        item.risk_category === 'High Risk'
                          ? 'bg-red-100 text-red-800'
                          : item.risk_category === 'Medium Risk'
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {item.risk_category}
                      </span>
                    </td>

                    <td className="py-3 px-3 text-center font-mono">
                      {item.altman_z !== null ? (
                        <span className={`font-semibold ${
                          item.altman_z < 1.1 ? 'text-red-700' : 'text-slate-700'
                        }`}>
                          {item.altman_z.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-slate-400">N/A*</span>
                      )}
                    </td>

                    <td className="py-3 px-3 text-center font-mono">
                      {item.beneish_m !== null ? (
                        <span className={`font-semibold ${
                          item.beneish_m > -1.78 ? 'text-amber-700' : 'text-emerald-700'
                        }`}>
                          {item.beneish_m.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-slate-400">N/A*</span>
                      )}
                    </td>

                    <td className="py-3 px-3 text-center font-mono">
                      {item.piotroski_f !== null ? `${item.piotroski_f}/9` : 'N/A*'}
                    </td>

                    <td className="py-3 px-3 text-center">
                      {item.soft_signals_count > 0 ? (
                        <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-bold text-[10px]">
                          {item.soft_signals_count} flags
                        </span>
                      ) : (
                        <span className="text-slate-400 text-[11px]">—</span>
                      )}
                    </td>

                    <td className="py-3 px-3 text-slate-600 truncate max-w-[220px]" title={item.drivers}>
                      {item.drivers}
                    </td>

                    <td className="py-3 px-3 text-center">
                      <button
                        onClick={() => onSelectCompany(item.company)}
                        className="px-2.5 py-1 bg-slate-100 hover:bg-brand-50 text-brand-800 rounded font-semibold text-[11px] transition flex items-center gap-1 mx-auto"
                      >
                        <Eye className="w-3 h-3" /> Dossier
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="py-3 px-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
          <div>
            Page <strong>{data?.page || 1}</strong> of <strong>{Math.ceil((data?.total || 1) / limit)}</strong>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(p - 1, 1))}
              disabled={page <= 1}
              className="p-1.5 rounded border border-slate-200 bg-white hover:bg-slate-100 disabled:opacity-40 transition"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={!data || page >= Math.ceil(data.total / limit)}
              className="p-1.5 rounded border border-slate-200 bg-white hover:bg-slate-100 disabled:opacity-40 transition"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
