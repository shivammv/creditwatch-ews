import {
  DashboardData,
  ModelMetricsData,
  ScoredCompanyResult,
  WatchlistResponse,
} from './types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<{ status: string; model_loaded: boolean; version: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDashboard(): Promise<DashboardData> {
  const res = await fetch(`${API_BASE}/dashboard`);
  if (!res.ok) throw new Error(`Failed to fetch dashboard data: ${res.statusText}`);
  return res.json();
}

export async function fetchModelMetrics(): Promise<ModelMetricsData> {
  const res = await fetch(`${API_BASE}/model/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch model metrics: ${res.statusText}`);
  return res.json();
}

export async function triggerRetrain(options: { optimize?: boolean; evaluate_all?: boolean } = {}): Promise<{
  success: boolean;
  message: string;
  report: any;
}> {
  const res = await fetch(`${API_BASE}/model/retrain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Retraining failed');
  }
  return res.json();
}

export async function fetchCompanies(): Promise<Array<{
  company: string;
  sector: string;
  latest_fy?: string;
  status: string;
}>> {
  const res = await fetch(`${API_BASE}/companies`);
  if (!res.ok) throw new Error(`Failed to fetch companies: ${res.statusText}`);
  return res.json();
}

export async function fetchCompanyDetails(companyName: string): Promise<ScoredCompanyResult> {
  const res = await fetch(`${API_BASE}/companies/${encodeURIComponent(companyName)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to fetch company details');
  }
  return res.json();
}

export interface ScoreInputParams {
  company_name: string;
  sector?: string;
  financial_year?: string;
  total_assets: number;
  gross_profit?: number;
  ebit: number;
  sales: number;
  net_income: number;
  total_liabilities: number;
  current_assets: number;
  current_liabilities: number;
  cash?: number;
  equity: number;
  retained_earnings?: number;
  financial_expenses?: number;
  cogs?: number;
  receivables?: number;
  depreciation?: number;
  cfo?: number;
  soft_signals?: string[];
}

export async function scoreSingleCompany(data: ScoreInputParams): Promise<ScoredCompanyResult> {
  const res = await fetch(`${API_BASE}/score/single`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Scoring failed');
  }
  return res.json();
}

export interface WatchlistQueryParams {
  budget_pct?: number;
  sector?: string;
  risk_category?: string;
  search?: string;
  page?: number;
  limit?: number;
}

export async function fetchWatchlist(params: WatchlistQueryParams = {}): Promise<WatchlistResponse> {
  const query = new URLSearchParams();
  if (params.budget_pct !== undefined) query.set('budget_pct', params.budget_pct.toString());
  if (params.sector) query.set('sector', params.sector);
  if (params.risk_category) query.set('risk_category', params.risk_category);
  if (params.search) query.set('search', params.search);
  if (params.page !== undefined) query.set('page', params.page.toString());
  if (params.limit !== undefined) query.set('limit', params.limit.toString());

  const res = await fetch(`${API_BASE}/watchlist?${query.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch watchlist: ${res.statusText}`);
  return res.json();
}

export async function fetchForensicsSummary(): Promise<{
  total_analyzed: number;
  altman_distress_count: number;
  altman_gray_count: number;
  altman_safe_count: number;
  beneish_manipulator_count: number;
  beneish_non_manipulator_count: number;
  piotroski_weak_count: number;
  icr_below_one_count: number;
  companies: Array<{
    company: string;
    sector: string;
    altman_z: number | null;
    altman_zone: string;
    beneish_m: number | null;
    beneish_flag: string;
    piotroski_f: number | null;
    piotroski_band: string;
    interest_coverage_ratio: number | null;
    icr_flag: boolean;
  }>;
}> {
  const res = await fetch(`${API_BASE}/forensics/summary`);
  if (!res.ok) throw new Error(`Failed to fetch forensics summary: ${res.statusText}`);
  return res.json();
}

export async function uploadAndValidateFile(file: File): Promise<{
  success: boolean;
  filename: string;
  rows_count: number;
  columns_found: string[];
  missing_required: string[];
  mapped_features: Record<string, string>;
  preview: Array<Record<string, any>>;
  validation_errors: string[];
  recommendation: string;
}> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/upload/validate`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Validation failed');
  }
  return res.json();
}
