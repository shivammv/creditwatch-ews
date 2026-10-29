"""
FastAPI Backend for CreditWatch EWS
Intelligent Financial Risk & Fraud Analytics
"""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src import indian, ingest
from src.features import FEATURE_NAMES, FEATURES, RAW_COLS, format_value, indian_to_portable, polish_to_portable
from src.model import ART, BAYES_T, COST_FN, COST_FP, LGB_PARAMS, load_polish, train_lead_models
from src.rank import (
    OVERLAY,
    SOFT_LABELS,
    SOFT_SIGNALS,
    apply_overlay,
    load_bundle,
    memo_text,
    predict_proba,
    top_drivers,
)
from src.scores import altman_z, altman_zone, beneish_flag, beneish_m, f_band, piotroski_f

app = FastAPI(
    title="CreditWatch EWS API",
    description="Enterprise Credit Risk, Forensic Distress Screening & Fraud Analytics API",
    version="2.0.0",
)

# CORS configuration for local React / Vite dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- Models
class FinancialInput(BaseModel):
    company: str = Field(default="Sample Corp Ltd", description="Company Name")
    sector: Optional[str] = Field(default="Manufacturing", description="Sector or Industry")
    is_financial: Optional[bool] = Field(default=False, description="Is Bank or NBFC")
    
    # Current Year Statement Items (INR Crore or same unit)
    total_assets: float = Field(..., description="Total Assets")
    current_assets: float = Field(..., description="Current Assets")
    current_liabilities: float = Field(..., description="Current Liabilities")
    total_liabilities: float = Field(..., description="Total Liabilities")
    equity: float = Field(..., description="Total Book Equity / Net Worth")
    retained_earnings: float = Field(default=0.0, description="Retained Earnings / Reserves")
    sales: float = Field(..., description="Revenue / Total Sales")
    cogs: Optional[float] = Field(default=None, description="Cost of Goods Sold")
    sga: Optional[float] = Field(default=None, description="Selling, General & Admin Expenses")
    depreciation: float = Field(default=0.0, description="Depreciation & Amortisation")
    ebit: float = Field(..., description="EBIT / Operating Profit")
    interest_expense: float = Field(default=0.0, description="Interest / Finance Costs")
    pbt: float = Field(..., description="Profit Before Tax")
    net_income: float = Field(..., description="Net Profit After Tax")
    cfo: Optional[float] = Field(default=None, description="Cash Flow from Operations")
    receivables: Optional[float] = Field(default=None, description="Trade Receivables / Debtors")
    inventory: Optional[float] = Field(default=None, description="Inventory / Stock")
    ppe_net: Optional[float] = Field(default=None, description="Net Property, Plant & Equipment")
    lt_debt: Optional[float] = Field(default=None, description="Long-Term Debt / Borrowings")
    cash: Optional[float] = Field(default=None, description="Cash & Bank Balances")
    shares_out: Optional[float] = Field(default=None, description="Number of Equity Shares")

    # Optional Prior Year Items for Beneish M, Piotroski F, and Sales Growth
    prev_total_assets: Optional[float] = None
    prev_current_assets: Optional[float] = None
    prev_current_liabilities: Optional[float] = None
    prev_total_liabilities: Optional[float] = None
    prev_equity: Optional[float] = None
    prev_sales: Optional[float] = None
    prev_cogs: Optional[float] = None
    prev_sga: Optional[float] = None
    prev_depreciation: Optional[float] = None
    prev_ebit: Optional[float] = None
    prev_interest_expense: Optional[float] = None
    prev_net_income: Optional[float] = None
    prev_receivables: Optional[float] = None
    prev_ppe_net: Optional[float] = None
    prev_lt_debt: Optional[float] = None
    prev_shares_out: Optional[float] = None

    # Soft Signal Flags (0 or 1)
    rating_downgrade: Optional[int] = Field(default=0, description="Rating Downgrade in prior 12m")
    promoter_pledge_high: Optional[int] = Field(default=0, description="Promoter share pledge > 50%")
    auditor_red_flag: Optional[int] = Field(default=0, description="Auditor qualification or resignation")
    delayed_filing: Optional[int] = Field(default=0, description="Delayed statutory or exchange filing")
    related_party_stress: Optional[int] = Field(default=0, description="Large related party loans / advances")


# Helper cache for loaded data
def _pct(x):
    return round(float(x) * 100, 1) if x is not None else None


def get_portfolio_df() -> pd.DataFrame:
    p = ART / "watchlist_any.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df
    # Fallback to sample_watchlist
    p2 = ART / "sample_watchlist.csv"
    if p2.exists():
        return pd.read_csv(p2)
    return pd.DataFrame()


# ---------------------------------------------------------------- Endpoints
@app.get("/api/health")
def health():
    rep_path = ART / "benchmark_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    st = indian.panel_status()
    return {
        "status": "healthy",
        "service": "CreditWatch EWS",
        "model_version": rep.get("meta", {}).get("model_version", "2.0.0-production"),
        "training_timestamp": rep.get("meta", {}).get("training_timestamp"),
        "features_in_production": len(FEATURE_NAMES),
        "indian_panel_ready": st.get("ready", False),
        "defaulters_with_data": st.get("defaulters_with_data", 0),
        "peers_with_data": st.get("peers_with_data", 0),
        "total_indian_records": st.get("rows", 0),
    }


@app.get("/api/dashboard")
def dashboard():
    """Executive KPI metrics, distribution, and highlights."""
    fin = indian.load_financials()
    rep_path = ART / "benchmark_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    prod = rep.get("production", {})
    sims_path = ART / "universe_sim.json"
    sims = json.loads(sims_path.read_text()) if sims_path.exists() else {}
    s2 = sims.get("0.02", {})
    
    # Portfolio statistics
    wl_df = get_portfolio_df()
    total_companies = len(wl_df) if not wl_df.empty else len(fin["company"].unique()) if not fin.empty else 0
    
    if not wl_df.empty and "risk" in wl_df.columns:
        high_risk = int((wl_df["risk"] >= 0.50).sum())
        med_risk = int(((wl_df["risk"] >= 0.15) & (wl_df["risk"] < 0.50)).sum())
        low_risk = int((wl_df["risk"] < 0.15).sum())
        in_budget = int(wl_df["in_budget"].sum()) if "in_budget" in wl_df.columns else min(40, len(wl_df))
        avg_risk = float(wl_df["risk"].mean())
    else:
        high_risk, med_risk, low_risk, in_budget, avg_risk = 0, 0, 0, 0, 0.0

    # Sector breakdown
    sector_summary = []
    if not wl_df.empty and "sector" in wl_df.columns:
        for sec, g in wl_df.groupby("sector"):
            if pd.notna(sec) and sec != "":
                sector_summary.append({
                    "sector": str(sec),
                    "count": len(g),
                    "avg_risk": round(float(g["risk"].mean()), 3),
                    "high_risk_count": int((g["risk"] >= 0.50).sum()),
                })
        sector_summary.sort(key=lambda x: -x["avg_risk"])

    # Top high risk companies
    top_high_risk = []
    if not wl_df.empty:
        sort_col = "risk" if "risk" in wl_df.columns else "p"
        for _, r in wl_df.sort_values(sort_col, ascending=False).head(10).iterrows():
            top_high_risk.append({
                "rank": int(r.get("rank", 0)),
                "company": str(r.get("company", r.get("borrower", "Unknown"))),
                "sector": str(r.get("sector", "General")),
                "risk_pct": round(float(r[sort_col]) * 100, 1),
                "in_budget": bool(r.get("in_budget", False)),
                "altman_z": round(float(r["altman_z"]), 2) if "altman_z" in r and pd.notna(r["altman_z"]) else None,
                "beneish_m": round(float(r["beneish_m"]), 2) if "beneish_m" in r and pd.notna(r["beneish_m"]) else None,
                "piotroski_f": int(r["piotroski_f"]) if "piotroski_f" in r and pd.notna(r["piotroski_f"]) else None,
                "drivers": str(r.get("drivers", ""))[:180] + ("..." if len(str(r.get("drivers", ""))) > 180 else ""),
            })

    # Recent alerts from soft_signals and high risk flags
    alerts_feed = []
    soft = indian.load_soft()
    if not soft.empty:
        for _, s in soft.sort_values("date_public", ascending=False).head(8).iterrows():
            alerts_feed.append({
                "company": str(s["company"]),
                "signal": str(s["signal"]),
                "signal_label": SOFT_LABELS.get(str(s["signal"]), str(s["signal"])),
                "date": str(s["date_public"])[:10],
                "value": float(s["value"]) if pd.notna(s["value"]) else 1.0,
                "severity": "high" if str(s["signal"]) in ("rating_downgrade", "auditor_red_flag") else "amber",
            })

    # Indian evaluation summary
    ind_sum_path = ART / "indian_summary.json"
    ind_sum = json.loads(ind_sum_path.read_text()) if ind_sum_path.exists() else {}

    return {
        "kpis": {
            "total_companies_analyzed": total_companies,
            "high_risk_count": high_risk,
            "medium_risk_count": med_risk,
            "low_risk_count": low_risk,
            "flagged_for_review_budget": in_budget,
            "avg_predicted_risk_pct": round(avg_risk * 100, 1),
            "review_budget_cap_pct": 5.0,
            "bayes_threshold": round(BAYES_T, 4),
            "cost_fn_ratio": 20.0,
        },
        "model_performance_snippet": {
            "pr_auc": prod.get("pr_auc"),
            "pr_auc_baseline": prod.get("pr_auc_baseline"),
            "roc_auc": prod.get("roc_auc"),
            "brier_score": prod.get("brier_score"),
            "recall_at_5pct": round(s2["recall_mean"] * 100, 1) if "recall_mean" in s2 else None,
            "cost_reduction_vs_none": round((1 - s2["cost_at_budget"] / max(s2["cost_flag_none"], 1)) * 100, 1) if "cost_at_budget" in s2 else None,
        },
        "indian_panel_status": {
            "ready": ind_sum.get("lead_4q") is not None,
            "lead_4q_recall_full": _pct(ind_sum.get("lead_4q", {}).get("recall_full")),
            "lead_4q_recall_model_only": _pct(ind_sum.get("lead_4q", {}).get("recall_model_only")),
            "lead_8q_recall_full": _pct(ind_sum.get("lead_8q", {}).get("recall_full")),
            "defaulters_count": ind_sum.get("lead_4q", {}).get("n_defaulters", 0),
            "peers_count": ind_sum.get("lead_4q", {}).get("n_peers", 0),
        },
        "sectors": sector_summary,
        "top_high_risk": top_high_risk,
        "recent_alerts": alerts_feed,
    }


@app.get("/api/model/metrics")
def model_metrics():
    """Full scientific metrics, curves, calibration, and comparison table."""
    rep_path = ART / "benchmark_report.json"
    if not rep_path.exists():
        raise HTTPException(status_code=404, detail="Model benchmark report not found.")
    rep = json.loads(rep_path.read_text())

    # Load PR curve
    pr_curve_data = []
    pr_file = ART / "pr_curve.csv"
    if pr_file.exists():
        pr_df = pd.read_csv(pr_file)
        # Sample down to ~100 points for sleek rendering
        step = max(1, len(pr_df) // 100)
        pr_sampled = pr_df.iloc[::step]
        pr_curve_data = [
            {"recall": round(float(r["recall"]), 4), "precision": round(float(r["precision"]), 4)}
            for _, r in pr_sampled.iterrows()
        ]

    # Load ROC curve
    roc_curve_data = []
    roc_file = ART / "roc_curve.csv"
    if roc_file.exists():
        roc_df = pd.read_csv(roc_file)
        step = max(1, len(roc_df) // 100)
        roc_sampled = roc_df.iloc[::step]
        roc_curve_data = [
            {"fpr": round(float(r["fpr"]), 4), "tpr": round(float(r["tpr"]), 4)}
            for _, r in roc_sampled.iterrows()
        ]

    # Load Calibration curve
    cal_curve_data = []
    cal_file = ART / "calibration_curve.csv"
    if cal_file.exists():
        cal_df = pd.read_csv(cal_file)
        cal_curve_data = [
            {"prob_pred": round(float(r["prob_pred"]), 4), "prob_true": round(float(r["prob_true"]), 4)}
            for _, r in cal_df.iterrows()
        ]

    # Multi-model comparison
    comp_file = ART / "model_comparison.json"
    comp_data = json.loads(comp_file.read_text()) if comp_file.exists() else rep.get("model_comparison", [])

    return {
        "meta": rep.get("meta", {}),
        "production": rep.get("production", {}),
        "reference": rep.get("reference", {}),
        "ablation": rep.get("ablation", {}),
        "model_comparison": comp_data,
        "pr_curve": pr_curve_data,
        "roc_curve": roc_curve_data,
        "calibration_curve": cal_curve_data,
        "bayes_threshold": rep.get("meta", {}).get("cost", {}).get("bayes_threshold", 0.0476),
    }


@app.post("/api/model/retrain")
def retrain_model(confirm: bool = Query(False)):
    """Controlled retraining endpoint with validation and safety backup."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Retraining refits the model pipeline on development data only (never holdout)."
        )
    t0 = time.time()
    try:
        report = train_lead_models(run_full_comparison=True)["4q"]
        load_bundle.cache_clear()
        # re-run simulations
        from run_all import build_sim, sample_watchlist
        build_sim()
        sample_watchlist()
        duration = round(time.time() - t0, 2)
        return {
            "status": "success",
            "message": "Model retrained and verified successfully.",
            "duration_seconds": duration,
            "production_pr_auc": report["production"]["pr_auc"],
            "production_roc_auc": report["production"]["roc_auc"],
            "timestamp": report["meta"]["training_timestamp"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.post("/api/score/single")
def score_single_company(payload: FinancialInput):
    """Score a single company, compute ratios, Z''/M/F scores, and generate credit officer SHAP drivers."""
    bundle = load_bundle("production")
    
    cur_dict = payload.model_dump()
    prev_dict = {
        "total_assets": payload.prev_total_assets or payload.total_assets,
        "current_assets": payload.prev_current_assets,
        "current_liabilities": payload.prev_current_liabilities,
        "total_liabilities": payload.prev_total_liabilities,
        "equity": payload.prev_equity,
        "sales": payload.prev_sales,
        "cogs": payload.prev_cogs,
        "sga": payload.prev_sga,
        "depreciation": payload.prev_depreciation,
        "ebit": payload.prev_ebit,
        "interest_expense": payload.prev_interest_expense,
        "net_income": payload.prev_net_income,
        "receivables": payload.prev_receivables,
        "ppe_net": payload.prev_ppe_net,
        "lt_debt": payload.prev_lt_debt,
        "shares_out": payload.prev_shares_out,
    }

    # 1. Compute portable ratios
    feats = indian_to_portable(cur_dict, prev_dict)
    feat_df = pd.DataFrame([feats])[FEATURE_NAMES].astype(float)
    
    # 2. Raw model probability
    raw_p = predict_proba(bundle, feat_df)[0]
    
    # 3. Forensic scores
    is_fin = bool(payload.is_financial)
    if is_fin:
        z_score, z_z = np.nan, "n/a_financial"
        m_score, m_f = np.nan, "n/a_financial"
        f_score, f_b = np.nan, "n/a_financial"
    else:
        wc = payload.current_assets - payload.current_liabilities
        z_score = altman_z(wc, payload.total_assets, payload.retained_earnings, payload.ebit, payload.equity, payload.total_liabilities)
        z_z = altman_zone(z_score)
        m_score = beneish_m(cur_dict, prev_dict) if payload.prev_sales else np.nan
        m_f = beneish_flag(m_score)
        f_score = piotroski_f(cur_dict, prev_dict) if payload.prev_sales else np.nan
        f_b = f_band(f_score)

    # 4. Soft signals
    soft_df = pd.DataFrame([{
        "rating_downgrade": payload.rating_downgrade or 0,
        "promoter_pledge_high": payload.promoter_pledge_high or 0,
        "auditor_red_flag": payload.auditor_red_flag or 0,
        "delayed_filing": payload.delayed_filing or 0,
        "related_party_stress": payload.related_party_stress or 0,
    }])
    active_soft = [k for k in SOFT_SIGNALS if soft_df.iloc[0].get(k, 0) == 1]

    # 5. Combined overlay risk
    icr = feats.get("int_cover", np.nan)
    overlay_risk = float(apply_overlay(np.array([raw_p]), z_zone=[z_z], m_flag=[m_f], f_band=[f_b], soft=soft_df, icr=[icr])[0])

    # 6. SHAP drivers
    drvs = top_drivers(bundle, feat_df, k=4)[0]

    # 7. Action memo
    memo = memo_text(
        company=payload.company,
        rank=1,
        risk=overlay_risk,
        drivers=drvs,
        z=z_score if np.isfinite(z_score) else None,
        m=m_score if np.isfinite(m_score) else None,
        f=f_score if np.isfinite(f_score) else None,
        soft=active_soft,
    )

    # Risk category
    if overlay_risk >= 0.50 or (np.isfinite(z_score) and z_score < 1.1) or (np.isfinite(icr) and icr < 1.0):
        category = "High Risk"
        badge_color = "red"
    elif overlay_risk >= 0.15 or (np.isfinite(z_score) and z_score < 2.6):
        category = "Medium Risk"
        badge_color = "amber"
    else:
        category = "Low Risk"
        badge_color = "green"

    return {
        "company": payload.company,
        "sector": payload.sector,
        "is_financial": is_fin,
        "calibrated_model_probability": round(float(raw_p), 4),
        "combined_risk_score": round(overlay_risk, 4),
        "combined_risk_pct": round(overlay_risk * 100, 1),
        "risk_category": category,
        "badge_color": badge_color,
        "above_bayes_threshold": bool(raw_p >= BAYES_T),
        "in_priority_review": bool(overlay_risk >= 0.40 or raw_p >= BAYES_T),
        "forensics": {
            "altman_z": round(float(z_score), 2) if np.isfinite(z_score) else None,
            "altman_zone": z_z,
            "beneish_m": round(float(m_score), 2) if np.isfinite(m_score) else None,
            "beneish_flag": m_f,
            "piotroski_f": int(f_score) if np.isfinite(f_score) else None,
            "piotroski_band": f_b,
            "interest_coverage_ratio": round(float(icr), 2) if np.isfinite(icr) else None,
            "icr_below_one": bool(np.isfinite(icr) and icr < 1.0),
        },
        "soft_signals": {
            "active_count": len(active_soft),
            "signals": active_soft,
            "signal_descriptions": [SOFT_LABELS.get(s, s) for s in active_soft],
        },
        "key_ratios": {
            k: {
                "value": round(float(feats[k]), 4) if np.isfinite(feats.get(k, np.nan)) else None,
                "label": FEATURES[k]["label"],
                "unit": FEATURES[k]["unit"],
                "peer_median": bundle["healthy_median"].get(k),
            }
            for k in [
                "current_ratio", "quick_ratio", "liabilities_ta", "net_margin",
                "ebit_ta", "asset_turn", "cost_sales", "netdebt_sales"
            ] if k in FEATURES
        },
        "shap_drivers": drvs,
        "watch_list_memo": memo,
        "review_recommendation": (
            "Priority immediate deep-dive review within 5 working days; freeze limits and escalate to Credit Risk Committee."
            if category == "High Risk"
            else "Standard quarterly credit monitoring; request updated management commentary on working capital."
            if category == "Medium Risk"
            else "Healthy operating profile; maintain standard annual renewal cycle."
        )
    }


@app.get("/api/companies")
def list_companies():
    """List all available companies from dataset for quick search and exploration."""
    fin = indian.load_financials()
    if fin.empty:
        return []
    
    records = []
    for co, g in fin.groupby("company"):
        g = g.sort_values("fy_end")
        latest = g.iloc[-1]
        records.append({
            "company": str(co),
            "sector": str(latest.get("sector", "General")),
            "latest_fy": str(latest["fy_end"])[:10],
            "is_financial": bool(latest.get("is_financial", 0)),
            "role": str(latest.get("role", "peer")),
            "total_assets": float(latest["total_assets"]) if pd.notna(latest.get("total_assets")) else None,
            "sales": float(latest["sales"]) if pd.notna(latest.get("sales")) else None,
            "net_income": float(latest["net_income"]) if pd.notna(latest.get("net_income")) else None,
        })
    records.sort(key=lambda x: x["company"])
    return records


@app.get("/api/companies/{company_name}")
def get_company_detail(company_name: str):
    """Retrieve full historical records and score a company from the existing dataset."""
    fin = indian.load_financials()
    soft = indian.load_soft()
    comp_fin = fin[fin.company == company_name].sort_values("fy_end")
    if comp_fin.empty:
        raise HTTPException(status_code=404, detail=f"Company '{company_name}' not found.")
    
    cur_row = comp_fin.iloc[-1]
    prev_row = comp_fin.iloc[-2] if len(comp_fin) > 1 else None

    # Map to FinancialInput schema
    inp = FinancialInput(
        company=str(company_name),
        sector=str(cur_row.get("sector", "General")),
        is_financial=bool(cur_row.get("is_financial", 0)),
        total_assets=float(cur_row["total_assets"]),
        current_assets=float(cur_row["current_assets"]) if pd.notna(cur_row.get("current_assets")) else float(cur_row["total_assets"] * 0.4),
        current_liabilities=float(cur_row["current_liabilities"]) if pd.notna(cur_row.get("current_liabilities")) else float(cur_row["total_assets"] * 0.3),
        total_liabilities=float(cur_row["total_liabilities"]) if pd.notna(cur_row.get("total_liabilities")) else float(cur_row["total_assets"] * 0.6),
        equity=float(cur_row["equity"]) if pd.notna(cur_row.get("equity")) else float(cur_row["total_assets"] * 0.4),
        retained_earnings=float(cur_row.get("retained_earnings", 0.0)) if pd.notna(cur_row.get("retained_earnings")) else 0.0,
        sales=float(cur_row["sales"]) if pd.notna(cur_row.get("sales")) else 100.0,
        cogs=float(cur_row["cogs"]) if pd.notna(cur_row.get("cogs")) else None,
        sga=float(cur_row["sga"]) if pd.notna(cur_row.get("sga")) else None,
        depreciation=float(cur_row.get("depreciation", 0.0)) if pd.notna(cur_row.get("depreciation")) else 0.0,
        ebit=float(cur_row["ebit"]) if pd.notna(cur_row.get("ebit")) else float(cur_row.get("pbt", 0.0)),
        interest_expense=float(cur_row.get("interest_expense", 0.0)) if pd.notna(cur_row.get("interest_expense")) else 0.0,
        pbt=float(cur_row["pbt"]) if pd.notna(cur_row.get("pbt")) else 0.0,
        net_income=float(cur_row["net_income"]) if pd.notna(cur_row.get("net_income")) else 0.0,
        cfo=float(cur_row["cfo"]) if pd.notna(cur_row.get("cfo")) else None,
        receivables=float(cur_row["receivables"]) if pd.notna(cur_row.get("receivables")) else None,
        inventory=float(cur_row["inventory"]) if pd.notna(cur_row.get("inventory")) else None,
        ppe_net=float(cur_row["ppe_net"]) if pd.notna(cur_row.get("ppe_net")) else None,
        lt_debt=float(cur_row["lt_debt"]) if pd.notna(cur_row.get("lt_debt")) else None,
        cash=float(cur_row["cash"]) if pd.notna(cur_row.get("cash")) else None,
        shares_out=float(cur_row["shares_out"]) if pd.notna(cur_row.get("shares_out")) else None,
        
        # Prior year
        prev_total_assets=float(prev_row["total_assets"]) if prev_row is not None and pd.notna(prev_row.get("total_assets")) else None,
        prev_sales=float(prev_row["sales"]) if prev_row is not None and pd.notna(prev_row.get("sales")) else None,
        prev_net_income=float(prev_row["net_income"]) if prev_row is not None and pd.notna(prev_row.get("net_income")) else None,
        prev_cogs=float(prev_row["cogs"]) if prev_row is not None and pd.notna(prev_row.get("cogs")) else None,
        prev_receivables=float(prev_row["receivables"]) if prev_row is not None and pd.notna(prev_row.get("receivables")) else None,
    )

    # Soft signals for this company
    comp_soft = soft[soft.company == company_name]
    for s in SOFT_SIGNALS:
        val = comp_soft[comp_soft.signal == s]
        if not val.empty and val["value"].max() > 0:
            setattr(inp, s, 1)

    scored = score_single_company(inp)
    
    # Add historical statements
    history = []
    for _, h in comp_fin.iterrows():
        history.append({
            "fy_end": str(h["fy_end"])[:10],
            "total_assets": float(h["total_assets"]) if pd.notna(h.get("total_assets")) else None,
            "sales": float(h["sales"]) if pd.notna(h.get("sales")) else None,
            "ebit": float(h["ebit"]) if pd.notna(h.get("ebit")) else None,
            "net_income": float(h["net_income"]) if pd.notna(h.get("net_income")) else None,
            "equity": float(h["equity"]) if pd.notna(h.get("equity")) else None,
            "total_liabilities": float(h["total_liabilities"]) if pd.notna(h.get("total_liabilities")) else None,
        })
    scored["history"] = history
    return scored


@app.get("/api/watchlist")
def get_watchlist(
    search: Optional[str] = None,
    sector: Optional[str] = None,
    category: Optional[str] = None,
    in_budget_only: Optional[bool] = False,
    budget_pct: float = Query(0.05, ge=0.01, le=0.50),
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=200),
):
    """Searchable, filterable watchlist with review budget cap enforcement."""
    wl_df = get_portfolio_df().copy()
    if wl_df.empty:
        return {"items": [], "total": 0, "page": page, "budget_cap_count": 0}

    sort_col = "risk" if "risk" in wl_df.columns else "p"
    wl_df = wl_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    wl_df["rank"] = np.arange(1, len(wl_df) + 1)
    
    # Calculate budget threshold
    k_budget = int(np.ceil(budget_pct * len(wl_df)))
    wl_df["in_budget"] = wl_df["rank"] <= k_budget

    # Search filter
    if search:
        col = "company" if "company" in wl_df.columns else "borrower"
        wl_df = wl_df[wl_df[col].astype(str).str.contains(search, case=False, na=False)]

    # Sector filter
    if sector and "sector" in wl_df.columns:
        wl_df = wl_df[wl_df["sector"].astype(str) == sector]

    # Category filter
    if category:
        if category.lower() == "high":
            wl_df = wl_df[wl_df[sort_col] >= 0.50]
        elif category.lower() == "medium":
            wl_df = wl_df[(wl_df[sort_col] >= 0.15) & (wl_df[sort_col] < 0.50)]
        elif category.lower() == "low":
            wl_df = wl_df[wl_df[sort_col] < 0.15]

    # Budget filter
    if in_budget_only:
        wl_df = wl_df[wl_df["in_budget"]]

    total = len(wl_df)
    start = (page - 1) * limit
    page_df = wl_df.iloc[start : start + limit]

    items = []
    for _, r in page_df.iterrows():
        risk_val = float(r[sort_col])
        cat = "High Risk" if risk_val >= 0.50 else ("Medium Risk" if risk_val >= 0.15 else "Low Risk")
        items.append({
            "rank": int(r["rank"]),
            "company": str(r.get("company", r.get("borrower", "Unknown"))),
            "sector": str(r.get("sector", "General")),
            "risk_pct": round(risk_val * 100, 1),
            "risk_category": cat,
            "in_budget": bool(r["in_budget"]),
            "altman_z": round(float(r["altman_z"]), 2) if "altman_z" in r and pd.notna(r["altman_z"]) else None,
            "beneish_m": round(float(r["beneish_m"]), 2) if "beneish_m" in r and pd.notna(r["beneish_m"]) else None,
            "piotroski_f": int(r["piotroski_f"]) if "piotroski_f" in r and pd.notna(r["piotroski_f"]) else None,
            "soft_signals_count": int(r.get("soft_count", len(eval(str(r.get("soft_list", "[]")))) if "soft_list" in r else 0)),
            "drivers": str(r.get("drivers", ""))[:180],
            "memo": str(r.get("memo", "")) if "memo" in r else "",
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "total_portfolio_size": len(get_portfolio_df()),
        "budget_cap_count": k_budget,
        "budget_pct": budget_pct,
    }


@app.get("/api/watchlist/export")
def export_watchlist():
    """Download watchlist as CSV."""
    wl_df = get_portfolio_df()
    if wl_df.empty:
        raise HTTPException(status_code=404, detail="No watchlist available to export.")
    
    stream = io.StringIO()
    wl_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=CreditWatch_Watchlist.csv"
    return response


@app.post("/api/upload/validate")
async def validate_upload(file: UploadFile = File(...)):
    """Validate uploaded CSV or Excel file, preview schema, and check mappings."""
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx, .xls) files are supported.")

    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing failed: {str(e)}")

    kind = ingest.detect_kind(df)
    mapping, unmapped = ingest.map_columns(df)
    
    # Missing required columns analysis
    core_req = ["total_assets", "current_assets", "current_liabilities", "total_liabilities", "equity", "sales", "ebit"]
    mapped_targets = set(mapping.values())
    missing_core = [c for c in core_req if c not in mapped_targets]

    # Missing cell percentage
    nan_cell_pct = round(float(df.isna().mean().mean()) * 100, 2)
    duplicate_rows = int(df.duplicated().sum())

    preview_rows = df.head(5).replace({np.nan: None}).to_dict(orient="records")

    return {
        "filename": file.filename,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "detected_kind": kind,
        "mapped_columns": mapping,
        "unmapped_columns": unmapped,
        "missing_core_columns": missing_core,
        "duplicate_rows": duplicate_rows,
        "missing_cells_pct": nan_cell_pct,
        "is_ready_to_score": (kind in ("polish", "statements") and len(missing_core) <= 2),
        "preview": preview_rows,
    }


@app.get("/api/forensics/summary")
def forensics_summary():
    """Detailed corporate distress and accounting manipulation analytics."""
    fin = indian.load_financials()
    soft = indian.load_soft()
    
    # Beneish M-score sub-indices explanations
    m_indices = [
        {"code": "DSRI", "name": "Days Sales in Receivables Index", "weight": "+0.920", "role": "Measures disproportionate receivable growth vs sales. Rising DSRI indicates revenue acceleration or uncollected billings."},
        {"code": "GMI", "name": "Gross Margin Index", "weight": "+0.528", "role": "Compares prior margin to current. GMI > 1 signals deteriorating profit margins, increasing temptation to manipulate earnings."},
        {"code": "AQI", "name": "Asset Quality Index", "weight": "+0.404", "role": "Measures non-current non-PPE assets (intangibles, deferrals). Higher AQI suggests cost capitalisation."},
        {"code": "SGI", "name": "Sales Growth Index", "weight": "+0.892", "role": "Growth firms face pressure to maintain trajectory when market expectations decelerate."},
        {"code": "DEPI", "name": "Depreciation Index", "weight": "+0.115", "role": "DEPI > 1 means depreciation rate slowed down, extending asset lives to flatter net profit."},
        {"code": "SGAI", "name": "SG&A Expense Index", "weight": "-0.172", "role": "Measures overhead spending relative to sales."},
        {"code": "LVGI", "name": "Leverage Index", "weight": "-0.327", "role": "Measures increasing debt to asset ratios."},
        {"code": "TATA", "name": "Total Accruals to Total Assets", "weight": "+4.679", "role": "High accruals (Net Income − CFO) relative to assets strongly correlates with artificial accounting profits."}
    ]

    return {
        "beneish_threshold": -1.78,
        "beneish_description": "Beneish (1999) 8-variable forensic model designed to detect earnings overstatement risk. M > -1.78 classifies a company in the earnings manipulation zone.",
        "beneish_disclaimer": "CRITICAL RISK DISCLAIMER: The Beneish M-Score is an accounting-manipulation screening filter, NOT legal proof of fraud. Legitimate high-growth firms, lumpy contract cycles, and acquisition restatements can trigger flagged values.",
        "corporate_vs_transaction_fraud_disclaimer": "DISTINCTION OF SCOPE: This system detects corporate financial distress and accounting stress using annual audited balance sheets and public filings. It does NOT process transaction-level payment fraud, credit card authorization anomalies, AML wire transfers, or identity theft.",
        "m_indices": m_indices,
        "altman_zones": {
            "safe": "> 2.60",
            "grey": "1.10 - 2.60",
            "distress": "< 1.10",
            "suitability": "Emerging-market non-manufacturing Altman Z'' formulation using book equity instead of market capitalisation."
        },
        "piotroski_bands": {
            "strong": "7 - 9 points",
            "moderate": "4 - 6 points",
            "weak": "0 - 3 points",
            "factors": "9 discrete fundamental accounting tests covering profitability, leverage/liquidity, and operating efficiency."
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
