export interface DashboardKPIs {
  total_companies_analyzed: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  flagged_for_review_budget: number;
  avg_predicted_risk_pct: number;
  review_budget_cap_pct: number;
  bayes_threshold: number;
  cost_fn_ratio: number;
}

export interface ModelPerformanceSnippet {
  pr_auc: number;
  pr_auc_baseline: number;
  roc_auc: number;
  brier_score: number;
  recall_at_5pct: number;
  cost_reduction_vs_none: number;
}

export interface IndianPanelStatus {
  ready: boolean;
  lead_4q_recall_full: number;
  lead_4q_recall_model_only: number;
  lead_8q_recall_full: number;
  defaulters_count: number;
  peers_count: number;
}

export interface SectorSummary {
  sector: string;
  count: number;
  avg_risk: number;
  high_risk_count: number;
}

export interface HighRiskBorrower {
  rank: number;
  company: string;
  sector: string;
  risk_pct: number;
  in_budget: boolean;
  altman_z: number | null;
  beneish_m: number | null;
  piotroski_f: number | null;
  drivers: string;
}

export interface AlertFeedItem {
  company: string;
  signal: string;
  signal_label: string;
  date: string;
  value: number;
  severity: 'high' | 'amber';
}

export interface DashboardData {
  kpis: DashboardKPIs;
  model_performance_snippet: ModelPerformanceSnippet;
  indian_panel_status: IndianPanelStatus;
  sectors: SectorSummary[];
  top_high_risk: HighRiskBorrower[];
  recent_alerts: AlertFeedItem[];
}

export interface ShapDriver {
  feature: string;
  label: string;
  shap: number;
  value: number | null;
  peer_median: number;
  text: string;
}

export interface ScoredCompanyResult {
  company: string;
  sector: string;
  is_financial: boolean;
  calibrated_model_probability: number;
  combined_risk_score: number;
  combined_risk_pct: number;
  risk_category: 'High Risk' | 'Medium Risk' | 'Low Risk';
  badge_color: 'red' | 'amber' | 'green';
  above_bayes_threshold: boolean;
  in_priority_review: boolean;
  forensics: {
    altman_z: number | null;
    altman_zone: string;
    beneish_m: number | null;
    beneish_flag: string;
    piotroski_f: number | null;
    piotroski_band: string;
    interest_coverage_ratio: number | null;
    icr_below_one: boolean;
  };
  soft_signals: {
    active_count: number;
    signals: string[];
    signal_descriptions: string[];
  };
  key_ratios: Record<string, {
    value: number | null;
    label: string;
    unit: string;
    peer_median: number;
  }>;
  shap_drivers: ShapDriver[];
  watch_list_memo: string;
  review_recommendation: string;
  history?: Array<{
    fy_end: string;
    total_assets: number | null;
    sales: number | null;
    ebit: number | null;
    net_income: number | null;
    equity: number | null;
    total_liabilities: number | null;
  }>;
}

export interface ModelComparisonItem {
  model: string;
  pr_auc: number;
  roc_auc: number;
  brier_score: number;
  recall_at_5pct: number;
  cv_training_time_sec: number;
  supports_native_nan: boolean;
  selected_as_production: boolean;
}

export interface ModelMetricsData {
  meta: any;
  production: any;
  reference: any;
  ablation: any;
  model_comparison: ModelComparisonItem[];
  pr_curve: Array<{ recall: number; precision: number }>;
  roc_curve: Array<{ fpr: number; tpr: number }>;
  calibration_curve: Array<{ prob_pred: number; prob_true: number }>;
  bayes_threshold: number;
}

export interface WatchlistItem {
  rank: number;
  company: string;
  sector: string;
  risk_pct: number;
  risk_category: string;
  in_budget: boolean;
  altman_z: number | null;
  beneish_m: number | null;
  piotroski_f: number | null;
  soft_signals_count: number;
  drivers: string;
  memo: string;
}

export interface WatchlistResponse {
  items: WatchlistItem[];
  total: number;
  page: number;
  limit: number;
  total_portfolio_size: number;
  budget_cap_count: number;
  budget_pct: number;
}
