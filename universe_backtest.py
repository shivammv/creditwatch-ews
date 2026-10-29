"""
Real-universe backtest: rank each verified defaulter against EVERY listed company with filings public at the observation date.

    python universe_backtest.py <india_company_year_financials.csv> [--model]

For each defaulter (data/indian/events.csv, date_status verified/month_verified) and lead L in (4, 8) quarters:
  obs = event - 3L months;  fiscal year Y usable only if 31-Mar-Y + 120 days <= obs  (FY assumed to end 31 March: panel stores only the year)
  universe = all companies with a row for year Y;  report the defaulter's percentile from the riskiest end.
  Recall at 5% = share of defaulters with percentile <= 5 (that is 5% of the universe = the review budget).

Scorers: heuristic baselines (always) and, with --model, the LightGBM production model (needs lightgbm/shap; not run in the authoring sandbox).
CAVEATS: tiny n, survivor-heavy universe (delisted failures are absent), proxy balance-sheet items, FY-end assumption.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

NAME_MAP = {"Jet Airways": "Jet Airways (India) Ltd", "Reliance Capital": "Reliance Capital Ltd", "Future Retail": "Future Retail Ltd",
            "Coffee Day Enterprises": "Coffee Day Enterprises Ltd", "Alok Industries": "Alok Industries Ltd", "Reliance Communications": "Reliance Communications Ltd"}


def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["company_name", "report_year"]).reset_index(drop=True)
    TA = d.total_assets.replace(0, np.nan)
    d["equity"] = d.equity_capital.fillna(0) + d.reserves.fillna(0)
    d["tl"] = d.total_assets - d.equity                       # NB: the panel's own 'total_liabilities' equals total assets (L+E), do not use it
    d["ebit"] = d.profit_before_tax + d.interest_expense
    d["lev"] = d.tl / TA
    d["debt_eq"] = np.where(d.equity > 0, d.borrowings / d.equity.replace(0, np.nan), 99.0)    # negative equity = worst
    d["debt_ta"] = d.borrowings / TA
    d["z_proxy"] = 6.56 * (d.other_assets - d.other_liabilities) / TA + 3.26 * d.reserves / TA + 6.72 * d.ebit / TA + 1.05 * d.equity / d.tl.replace(0, np.nan)
    d["icr"] = d.ebit / d.interest_expense.replace(0, np.nan)
    d["roa"] = d.net_profit / TA
    return d


# (column, higher_is_riskier)
SCORERS = {"rule_score_baseline": ("rule_risk_score", True), "anomaly_baseline": ("anomaly_score", True),
           "leverage (liab/assets)": ("lev", True), "debt/equity": ("debt_eq", True), "altman_z_proxy": ("z_proxy", False),
           "interest_cover": ("icr", False), "return_on_assets": ("roa", False)}


def pct_from_risky(s: pd.Series, v: float, hi: bool) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if pd.isna(v) or not np.isfinite(v) or s.empty:
        return np.nan
    return 100 * (((s > v).sum() + 0.5 * (s == v).sum()) if hi else ((s < v).sum() + 0.5 * (s == v).sum())) / len(s)


def model_scores(X: pd.DataFrame) -> pd.Series:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.features import FEATURE_NAMES, indian_to_portable
    from src.rank import load_bundle, predict_proba
    raw = pd.DataFrame(dict(total_assets=X.total_assets, current_assets=X.other_assets, current_liabilities=X.other_liabilities, total_liabilities=X.tl,
                            equity=X.equity, retained_earnings=X.reserves, sales=X.sales, cogs=np.nan, sga=np.nan, depreciation=X.depreciation, ebit=X.ebit,
                            interest_expense=X.interest_expense, pbt=X.profit_before_tax, net_income=X.net_profit, cfo=X.cash_from_operating,
                            receivables=np.nan, inventory=np.nan, ppe_net=X.fixed_assets, lt_debt=X.borrowings, cash=np.nan, shares_out=np.nan))
    F = pd.DataFrame([indian_to_portable(r, None) for r in raw.to_dict("records")], index=X.index)[FEATURE_NAMES].astype(float)
    return pd.Series(predict_proba(load_bundle("production"), F), index=X.index)


def run(panel_csv: str, use_model: bool = False) -> pd.DataFrame:
    d = prep(pd.read_csv(panel_csv))
    ev = pd.read_csv(Path(__file__).resolve().parent / "data/indian/events.csv", parse_dates=["event_date"])
    ev = ev[ev.date_status.isin(["verified", "month_verified"]) & ev.company.isin(NAME_MAP)]
    scorers = dict(SCORERS)
    rows = []
    for _, e in ev.iterrows():
        co = NAME_MAP[e.company]
        for lead in (4, 8):
            obs = e.event_date - pd.DateOffset(months=3 * lead)
            g = d[d.company_name == co]
            ok = g[g.report_year.apply(lambda y: pd.Timestamp(year=int(y), month=3, day=31) + pd.Timedelta(days=120) <= obs)]
            if ok.empty:
                continue
            Y = int(ok.report_year.max()); X = d[d.report_year == Y]
            me = X[X.company_name == co]
            if me.empty:
                continue
            r = dict(company=e.company, lead_q=lead, obs_date=obs.date(), fy_used=Y, universe=len(X))
            for name, (col, hi) in scorers.items():
                r[name] = round(pct_from_risky(X[col], me.iloc[0][col], hi), 1)
            if use_model:
                s = model_scores(X); r["lightgbm_model"] = round(pct_from_risky(s, s[me.index[0]], True), 1)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("panel"); ap.add_argument("--model", action="store_true"); a = ap.parse_args()
    r = run(a.panel, a.model)
    pd.set_option("display.width", 250); print(r.to_string(index=False))
    sc = [c for c in r.columns if c not in ("company", "lead_q", "obs_date", "fy_used", "universe")]
    for L in (4, 8):
        t = r[r.lead_q == L]
        print(f"\nlead {L}Q  n={len(t)}  recall inside top 5% / top 20% (percentile from the riskiest end):")
        print(pd.DataFrame({"top5%": (t[sc] <= 5).mean().round(2), "top20%": (t[sc] <= 20).mean().round(2)}).T.to_string())
    r.to_csv(Path(__file__).resolve().parent / "artifacts/universe_backtest.csv", index=False)
