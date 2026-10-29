"""
India-portable feature set.

The UCI Polish dataset exposes 64 anonymous ratios (Attr1..Attr64). Only ratios that
can be rebuilt from an Indian annual report (balance sheet + P&L, optionally the
previous year's sales) are used by the production model. That way the SAME model
scores Polish hold-out firms and Indian companies, with no unit-dependent features
(no absolute rupee/zloty amounts, no log-assets).

Polish definitions (UCI): "gross profit" ~ profit before tax, "profit on operating
activities" ~ EBIT, "financial expenses" ~ interest expense.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# name -> (polish attr, label, higher_is_riskier, unit, concern text template)
# concern templates may use {v} (firm value) and {m} (healthy-peer median).
FEATURES: dict[str, dict] = {
    "net_profit_ta":  dict(attr="Attr1",  label="Net profit / total assets", unit="pct",
        concern="Returns are weak ({v}) vs healthy peers ({m}); thin earnings leave little room to service debt."),
    "liabilities_ta": dict(attr="Attr2",  label="Total liabilities / total assets", unit="pct",
        concern="Leverage is high: liabilities fund {v} of assets (healthy peers {m}); creditors carry most of the risk."),
    "wc_ta":          dict(attr="Attr3",  label="Working capital / total assets", unit="pct",
        concern="Working capital is {v} of assets (healthy peers {m}); short-term obligations are crowding out liquidity."),
    "current_ratio":  dict(attr="Attr4",  label="Current ratio", unit="x",
        concern="Current ratio of {v} (healthy peers {m}) signals difficulty meeting obligations due within a year."),
    "retained_ta":    dict(attr="Attr6",  label="Retained earnings / total assets", unit="pct",
        concern="Retained earnings are {v} of assets (healthy peers {m}); accumulated losses have eroded the equity cushion."),
    "ebit_ta":        dict(attr="Attr7",  label="EBIT / total assets", unit="pct",
        concern="Operating return on assets is {v} (healthy peers {m}); the asset base is not generating enough to carry its debt."),
    "equity_liab":    dict(attr="Attr8",  label="Book equity / total liabilities", unit="x",
        concern="Equity covers only {v} of liabilities (healthy peers {m}); loss-absorbing capital is thin."),
    "asset_turn":     dict(attr="Attr9",  label="Sales / total assets", unit="x",
        concern="Asset turnover is {v} (healthy peers {m}); assets are not converting into revenue efficiently."),
    "equity_ta":      dict(attr="Attr10", label="Equity / total assets", unit="pct",
        concern="Equity funds only {v} of assets (healthy peers {m}); the balance sheet is debt-dependent."),
    "gp_stl":         dict(attr="Attr12", label="Pre-tax profit / short-term liabilities", unit="x",
        concern="Pre-tax profit covers only {v}x of short-term liabilities (healthy peers {m}x); refinancing risk is elevated."),
    "ebitda_proxy_sales": dict(attr="Attr13", label="(Pre-tax profit + depreciation) / sales", unit="pct",
        concern="Cash-style margin is {v} (healthy peers {m}); operations generate little cash per rupee of sales."),
    "int_adj_ta":     dict(attr="Attr14", label="(Pre-tax profit + interest) / total assets", unit="pct",
        concern="Profit before financing costs is {v} of assets (healthy peers {m}); debt service capacity is stretched."),
    "cf_liab":        dict(attr="Attr16", label="(Pre-tax profit + depreciation) / total liabilities", unit="x",
        concern="Annual cash earnings repay only {v} of total liabilities (healthy peers {m}); deleveraging would take very long."),
    "assets_liab":    dict(attr="Attr17", label="Total assets / total liabilities", unit="x",
        concern="Assets cover liabilities only {v}x (healthy peers {m}x); little asset buffer for lenders."),
    "gp_ta":          dict(attr="Attr18", label="Pre-tax profit / total assets", unit="pct",
        concern="Pre-tax return on assets is {v} (healthy peers {m}); profitability is insufficient."),
    "gp_sales":       dict(attr="Attr19", label="Pre-tax profit / sales", unit="pct",
        concern="Pre-tax margin is {v} (healthy peers {m}); pricing or cost pressure is squeezing profits."),
    "sales_growth":   dict(attr="Attr21", label="Sales growth (t vs t-1)", unit="x",
        concern="Sales moved to {v}x of last year (healthy peers {m}x); a sharp change in revenue is a stress marker."),
    "op_profit_ta":   dict(attr="Attr22", label="Operating profit / total assets", unit="pct",
        concern="Operating profit is {v} of assets (healthy peers {m}); core business earnings are weak."),
    "net_margin":     dict(attr="Attr23", label="Net profit / sales", unit="pct",
        concern="Net margin is {v} (healthy peers {m}); the business is barely profitable after all costs."),
    "int_cover":      dict(attr="Attr27", label="Operating profit / financial expenses", unit="x",
        concern="Interest cover is {v}x (healthy peers {m}x); operating profit barely covers, or fails to cover, interest."),
    "const_cap_ta":   dict(attr="Attr38", label="Constant capital / total assets", unit="pct",
        concern="Long-term funding is {v} of assets (healthy peers {m}); long assets may be funded with short-term money (ALM risk)."),
    "op_margin":      dict(attr="Attr42", label="Operating profit / sales", unit="pct",
        concern="Operating margin is {v} (healthy peers {m}); core operations are under pressure."),
    "quick_ratio":    dict(attr="Attr46", label="Quick ratio", unit="x",
        concern="Quick ratio of {v} (healthy peers {m}) means liquid assets do not cover near-term liabilities."),
    "ebitda_ta":      dict(attr="Attr48", label="EBITDA / total assets", unit="pct",
        concern="EBITDA is {v} of assets (healthy peers {m}); cash operating returns are low."),
    "ebitda_margin":  dict(attr="Attr49", label="EBITDA / sales", unit="pct",
        concern="EBITDA margin is {v} (healthy peers {m}); limited operating cash margin."),
    "ca_liab":        dict(attr="Attr50", label="Current assets / total liabilities", unit="x",
        concern="Current assets cover only {v}x of total liabilities (healthy peers {m}x)."),
    "stl_ta":         dict(attr="Attr51", label="Short-term liabilities / total assets", unit="pct",
        concern="Short-term liabilities are {v} of assets (healthy peers {m}); heavy near-term repayment burden."),
    "cost_sales":     dict(attr="Attr58", label="Total costs / total sales", unit="x",
        concern="Costs absorb {v} of sales (healthy peers {m}); there is no operating headroom."),
    "ltl_equity":     dict(attr="Attr59", label="Long-term liabilities / equity", unit="x",
        concern="Long-term debt is {v}x equity (healthy peers {m}x); structurally high gearing."),
    "sales_stl":      dict(attr="Attr63", label="Sales / short-term liabilities", unit="x",
        concern="Sales cover short-term liabilities only {v}x (healthy peers {m}x); revenue is small relative to near-term claims."),
    "cf_liab2":       dict(attr="Attr26", label="(Net profit + depreciation) / total liabilities", unit="x",
        concern="Net cash earnings repay only {v}x of total liabilities per year (healthy peers {m}x); deleveraging capacity is low."),
    "netdebt_sales":  dict(attr="Attr30", label="(Total liabilities - cash) / sales", unit="x",
        concern="Net obligations equal {v}x annual sales (healthy peers {m}x); the revenue base is small for the debt it carries."),
    "int_adj_sales":  dict(attr="Attr31", label="(Pre-tax profit + interest) / sales", unit="pct",
        concern="Profit before financing costs is {v} of sales (healthy peers {m}); little margin left to pay lenders."),
    "cash_like_ratio":dict(attr="Attr40", label="(Current assets - inventory - receivables) / current liabilities", unit="x",
        concern="Cash-like assets cover only {v}x of current liabilities (healthy peers {m}x); immediate liquidity is thin."),
    "gross_margin":   dict(attr="Attr56", label="Gross margin", unit="pct",
        concern="Gross margin is {v} (healthy peers {m}); pricing power or input-cost control is weak."),
    "sales_recv":     dict(attr="Attr61", label="Sales / receivables", unit="x",
        concern="Receivables turn only {v}x a year (healthy peers {m}x); collections are slowing, a classic early stress or revenue-quality signal."),
    "stl_days_sales": dict(attr="Attr62", label="Short-term liabilities (days of sales)", unit="x",
        concern="Short-term liabilities equal {v} days of sales (healthy peers {m}); the firm leans heavily on short-term funding."),
    "payback_days":   dict(attr="Attr15", label="Liabilities / (pre-tax profit + depreciation), in days", unit="x",
        concern="At current cash earnings it would take {v} days to repay total liabilities (healthy peers {m}); repayment capacity is stretched."),
    "inv_days_sales": dict(attr="Attr20", label="Inventory days (on sales)", unit="x",
        concern="Inventory equals {v} days of sales (healthy peers {m}); stock is building up, tying cash and risking write-downs."),
    "recv_days":      dict(attr="Attr44", label="Receivable days", unit="x",
        concern="Receivables equal {v} days of sales (healthy peers {m}); customers are paying slowly or revenue quality is doubtful."),
    "cl_days_cogs":   dict(attr="Attr32", label="Current liabilities (days of cost of goods)", unit="x",
        concern="Current liabilities equal {v} days of cost of goods (healthy peers {m}); the firm is stretching suppliers and short-term creditors."),
}

# Excluded from the MODEL (still computed and shown): their missingness in the UCI file is an artefact that would not
# transfer to Indian filings. e.g. Attr27 is NaN for 309 firms of which 38.8% went bankrupt (vs 2.3% otherwise), so an
# Indian company with zero interest cost (debt-free) would be mis-scored as distressed. Interest cover is instead used as a
# transparent rule in the overlay (ICR < 1).
NOT_IN_MODEL = {"int_cover": "missingness artefact (38.8% bankrupt when NaN)", "sales_growth": "23% missing in UCI; signal driven by missingness"}
FEATURE_NAMES = [k for k in FEATURES if k not in NOT_IN_MODEL]
ATTR_OF = {k: v["attr"] for k, v in FEATURES.items()}


def polish_to_portable(df: pd.DataFrame) -> pd.DataFrame:
    """Select + rename the portable Attr columns; inf -> NaN (LightGBM handles NaN natively)."""
    out = pd.DataFrame({k: df[a] for k, a in ATTR_OF.items()}, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def _div(a, b):
    if a is None or b is None:
        return np.nan
    try:
        a_f, b_f = float(a), float(b)
        if not np.isfinite(b_f) or b_f == 0 or not np.isfinite(a_f):
            return np.nan
        return a_f / b_f
    except (TypeError, ValueError, ZeroDivisionError):
        return np.nan


# Raw items expected for an Indian company-year (all in the same currency unit, e.g. INR crore).
RAW_COLS = [
    "total_assets", "current_assets", "current_liabilities", "total_liabilities", "equity",
    "retained_earnings", "sales", "cogs", "sga", "depreciation", "ebit", "interest_expense",
    "pbt", "net_income", "cfo", "receivables", "inventory", "ppe_net", "lt_debt", "cash",
    "shares_out",
]


def indian_to_portable(cur: dict, prev: dict | None = None) -> dict:
    """Rebuild the portable ratios from raw statement items of year t (and t-1 for growth)."""
    def g(k, default=np.nan):
        val = cur.get(k)
        if val is None:
            return default
        try:
            val_f = float(val)
            return val_f if np.isfinite(val_f) else default
        except (ValueError, TypeError):
            return default

    TA, CA, CL, TL, EQ = g("total_assets"), g("current_assets"), g("current_liabilities"), g("total_liabilities"), g("equity")
    S, EBIT, INT, PBT, NI, DEP = g("sales"), g("ebit"), g("interest_expense"), g("pbt"), g("net_income"), g("depreciation", 0.0)
    INV = g("inventory", 0.0)
    REC = g("receivables", 0.0)
    CASH = g("cash", 0.0)
    COGS = g("cogs")
    SGA = g("sga", 0.0)
    
    lt_liab = (TL - CL) if (np.isfinite(TL) and np.isfinite(CL)) else np.nan
    wc = (CA - CL) if (np.isfinite(CA) and np.isfinite(CL)) else np.nan
    
    # prev sales
    prev_s = np.nan
    if prev:
        p_val = prev.get("sales")
        if p_val is not None:
            try:
                prev_s = float(p_val)
            except (ValueError, TypeError):
                prev_s = np.nan

    f = {
        "net_profit_ta": _div(NI, TA), "liabilities_ta": _div(TL, TA), "wc_ta": _div(wc, TA),
        "current_ratio": _div(CA, CL), "retained_ta": _div(g("retained_earnings"), TA),
        "ebit_ta": _div(EBIT, TA), "equity_liab": _div(EQ, TL), "asset_turn": _div(S, TA),
        "equity_ta": _div(EQ, TA), "gp_stl": _div(PBT, CL), "ebitda_proxy_sales": _div(PBT + DEP, S),
        "int_adj_ta": _div(PBT + INT, TA), "cf_liab": _div(PBT + DEP, TL), "assets_liab": _div(TA, TL),
        "gp_ta": _div(PBT, TA), "gp_sales": _div(PBT, S),
        "sales_growth": _div(S, prev_s),
        "op_profit_ta": _div(EBIT, TA), "net_margin": _div(NI, S), "int_cover": _div(EBIT, INT),
        "const_cap_ta": _div(EQ + (lt_liab if np.isfinite(lt_liab) else 0.0), TA), "op_margin": _div(EBIT, S),
        "quick_ratio": _div(CA - INV, CL), "ebitda_ta": _div(EBIT + DEP, TA), "ebitda_margin": _div(EBIT + DEP, S),
        "ca_liab": _div(CA, TL), "stl_ta": _div(CL, TA), "cost_sales": _div((COGS if np.isfinite(COGS) else 0.0) + SGA + DEP + (INT if np.isfinite(INT) else 0.0), S),
        "ltl_equity": _div(lt_liab, EQ), "sales_stl": _div(S, CL),
        "cf_liab2": _div(NI + DEP, TL), "netdebt_sales": _div(TL - CASH, S), "int_adj_sales": _div(PBT + INT, S),
        "cash_like_ratio": _div(CA - INV - REC, CL),
        "gross_margin": _div(S - (COGS if np.isfinite(COGS) else 0.0), S), "sales_recv": _div(S, g("receivables")), "stl_days_sales": _div(CL * 365, S),
        "payback_days": _div(TL * 365, PBT + DEP), "inv_days_sales": _div(INV * 365, S),
        "recv_days": _div(g("receivables") * 365, S), "cl_days_cogs": _div(CL * 365, COGS),
    }
    return f


def format_value(v: float, unit: str) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v*100:.1f}%" if unit == "pct" else f"{v:.2f}"
