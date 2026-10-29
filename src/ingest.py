"""
Universal ingestion: take ANY tabular file (csv / xlsx) and make it scoreable.

  * detect_kind(df)      -> "polish" | "taiwan" | "statements" | "soft" | "prices" | "unknown"
  * map_columns(df)      -> fuzzy-map messy headers ("Total Assets (Rs Cr)", "TotAssets", "PAT") to the standard names
  * to_statements(df)    -> standard company-year frame (adds derived items when a source column is missing)
  * export_template(xlsx)-> splits the filled CreditWatch workbook into data/indian/*.csv for the Indian case test

Pure pandas/numpy: no model dependencies, so it can be tested on its own.
"""
from __future__ import annotations
import re
from difflib import get_close_matches
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_COLS = ["total_assets", "current_assets", "current_liabilities", "total_liabilities", "equity", "retained_earnings",
            "sales", "cogs", "sga", "depreciation", "ebit", "interest_expense", "pbt", "net_income", "cfo",
            "receivables", "inventory", "ppe_net", "lt_debt", "cash", "shares_out"]
ID_COLS = ["company", "fy_end", "public_date", "sector", "is_financial"]

# normalised (lowercase, alphanumeric only) aliases -> standard name
ALIASES = {
    "company": ["company", "companyname", "name", "borrower", "borrowername", "issuer", "entity", "firm", "ticker", "symbol", "scripname", "cin", "id"],
    "fy_end": ["fyend", "fiscalyearend", "yearend", "periodend", "period", "date", "asof", "balancesheetdate", "financialyear", "fy", "year", "reportdate"],
    "public_date": ["publicdate", "filingdate", "resultdate", "resultsdate", "announcementdate", "publisheddate", "disclosuredate", "datepublic"],
    "sector": ["sector", "industry", "segment"],
    "is_financial": ["isfinancial", "financial", "isbankorfinance", "isnbfc"],
    "total_assets": ["totalassets", "assetstotal", "totassets", "ta", "totalasset", "sumofassets"],
    "current_assets": ["currentassets", "totalcurrentassets", "ca", "currentasset"],
    "current_liabilities": ["currentliabilities", "totalcurrentliabilities", "cl", "shorttermliabilities", "currentliabilitiesprovisions"],
    "total_liabilities": ["totalliabilities", "liabilitiestotal", "totliabilities", "tl", "totaldebtandliabilities"],
    "equity": ["equity", "networth", "shareholdersequity", "totalequity", "shareholderfunds", "bookequity", "totalshareholdersfunds"],
    "retained_earnings": ["retainedearnings", "reservesandsurplus", "reserves", "reservessurplus", "retainedprofit", "re"],
    "sales": ["sales", "revenue", "netsales", "totalrevenue", "revenuefromoperations", "totalincome", "turnover", "operatingrevenue", "grossrevenue"],
    "cogs": ["cogs", "costofgoodssold", "costofsales", "costofmaterialsconsumed", "costofrevenue", "rawmaterialcost", "directcosts"],
    "sga": ["sga", "sellinggeneraladministrativeexpenses", "sgaexpenses", "otherexpenses", "administrativeexpenses", "sellingandadmin", "overheads"],
    "depreciation": ["depreciation", "depreciationandamortisation", "depreciationamortization", "dna", "da", "depreciationamortisation"],
    "ebit": ["ebit", "operatingprofit", "operatingincome", "profitbeforeinterestandtax", "pbit"],
    "interest_expense": ["interestexpense", "financecost", "financecosts", "financialexpenses", "interest", "interestpaid", "financecharges"],
    "pbt": ["pbt", "profitbeforetax", "profitbeforetaxation", "earningsbeforetax", "ebt", "grossprofit"],
    "net_income": ["netincome", "pat", "profitaftertax", "netprofit", "profitfortheyear", "profitfortheperiod", "earningsaftertax"],
    "cfo": ["cfo", "cashfromoperations", "operatingcashflow", "netcashfromoperatingactivities", "cashflowfromoperations", "cashfromoperatingactivities"],
    "receivables": ["receivables", "tradereceivables", "accountsreceivable", "debtors", "sundrydebtors", "tradereceivable"],
    "inventory": ["inventory", "inventories", "stock", "stockintrade"],
    "ppe_net": ["ppenet", "netfixedassets", "fixedassets", "propertyplantequipment", "netblock", "tangibleassets", "ppe", "netppe"],
    "lt_debt": ["ltdebt", "longtermdebt", "longtermborrowings", "noncurrentborrowings", "termloans", "borrowingslongterm", "totaldebt", "borrowings"],
    "cash": ["cash", "cashandbankbalances", "cashandcashequivalents", "cashbankbalance", "cashequivalents", "cashandbank"],
    "shares_out": ["sharesout", "sharesoutstanding", "numberofshares", "equitysharesoutstanding", "noofshares", "shares"],
}
SOFT_NAMES = ["rating_downgrade", "promoter_pledge_high", "auditor_red_flag", "delayed_filing", "related_party_stress"]


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", str(s).lower()))


def load_any(path: str | Path) -> dict[str, pd.DataFrame]:
    """csv/tsv -> {'file': df};  xlsx/xls -> {sheet: df}. Empty sheets are dropped."""
    p = Path(path)
    if p.suffix.lower() in (".csv", ".txt"):
        return {p.stem: pd.read_csv(p, sep=None, engine="python")}
    if p.suffix.lower() == ".tsv":
        return {p.stem: pd.read_csv(p, sep="\t")}
    sheets = pd.read_excel(p, sheet_name=None)
    return {k: v.dropna(how="all") for k, v in sheets.items() if len(v.dropna(how="all"))}


def map_columns(df: pd.DataFrame, extra: dict | None = None) -> tuple[dict, list]:
    """Return ({source_col: standard_name}, unmapped_source_cols). Exact alias first, then fuzzy (cutoff .85)."""
    alias_of = {a: std for std, al in ALIASES.items() for a in al}
    mapping, used = {}, set()
    for c in df.columns:
        n = _norm(c)
        std = alias_of.get(n)
        if std is None and (extra or {}).get(c):
            std = extra[c]
        if std is None and n:
            m = get_close_matches(n, alias_of.keys(), n=1, cutoff=0.85)
            std = alias_of[m[0]] if m else None
        if std and std not in used:
            mapping[c] = std
            used.add(std)
    return mapping, [c for c in df.columns if c not in mapping]


def detect_kind(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    low = [c.lower().strip() for c in cols]
    if sum(bool(re.fullmatch(r"attr\d+", c)) for c in low) >= 20:
        return "polish"
    if any("bankrupt" in c for c in low) and any("net income to total assets" in c for c in low):
        return "taiwan"
    n = {_norm(c) for c in cols}
    if {"signal", "datepublic"} <= n or ({"signal"} <= n and "date" in n):
        return "soft"
    mapping, _ = map_columns(df)
    std = set(mapping.values())
    if len(std & set(RAW_COLS)) >= 5:
        return "statements"
    if {"close", "date"} <= {c for c in low} or ({"adjclose"} <= set(low)):
        return "prices"
    if "close" in n and ("date" in n or "datetime" in n):
        return "prices"
    return "unknown"


def to_statements(df: pd.DataFrame, lag_days: int = 120, default_unit_note: bool = True) -> tuple[pd.DataFrame, dict]:
    """Standardise a messy company-year financials table. Returns (frame, report)."""
    mp, unmapped = map_columns(df)
    out = df.rename(columns=mp)[list(mp.values())].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    rep = {"mapped": {k: v for k, v in mp.items()}, "unmapped": unmapped, "derived": [], "missing": []}
    if "company" not in out:
        out["company"] = [f"Row-{i+1}" for i in range(len(out))]
    if "fy_end" not in out:
        out["fy_end"] = pd.Timestamp.today().normalize()
        rep["derived"].append("fy_end absent -> treated every row as the latest year")
    fy = out["fy_end"]
    if pd.api.types.is_numeric_dtype(fy) and fy.dropna().between(1990, 2100).all():      # plain years like 2019 -> 31 March
        out["fy_end"] = pd.to_datetime(fy.astype("Int64").astype(str) + "-03-31", errors="coerce")
    else:
        out["fy_end"] = pd.to_datetime(fy, errors="coerce", format="mixed")
    out["public_date"] = pd.to_datetime(out["public_date"], errors="coerce", format="mixed") if "public_date" in out else pd.NaT
    out["public_date"] = out["public_date"].fillna(out["fy_end"] + pd.Timedelta(days=lag_days))
    for c in RAW_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out else np.nan
    # derive what we can when a source column is missing
    def fill(col, expr, note):
        m = out[col].isna() & expr.notna()
        if m.any():
            out.loc[m, col] = expr[m]
            rep["derived"].append(f"{col} derived for {int(m.sum())} rows: {note}")
    fill("total_liabilities", out.total_assets - out.equity, "total_assets - equity")
    fill("equity", out.total_assets - out.total_liabilities, "total_assets - total_liabilities")
    fill("ebit", out.pbt + out.interest_expense, "pbt + interest_expense")
    fill("pbt", out.ebit - out.interest_expense, "ebit - interest_expense")
    fill("net_income", out.pbt * 0.75, "pbt x 0.75 (approx. tax); verify")
    rep["missing"] = [c for c in RAW_COLS if out[c].isna().all()]
    if "is_financial" in out:
        out["is_financial"] = out["is_financial"].astype(str).str.lower().isin(["1", "true", "yes", "y", "1.0"])
    else:
        sec = out["sector"].astype(str).str.lower() if "sector" in out else pd.Series("", index=out.index)
        out["is_financial"] = sec.str.contains("bank|nbfc|financ|insur|housing finance")
    if "sector" not in out:
        out["sector"] = ""
    out = out[~out["company"].astype(str).str.upper().str.startswith("EXAMPLE")]      # skip the template's worked example
    return out.sort_values(["company", "fy_end"]).reset_index(drop=True), rep


def to_soft(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [_norm(c) for c in d.columns]
    d = d.rename(columns={"datepublic": "date_public", "date": "date_public", "companyname": "company", "name": "company"})
    d["date_public"] = pd.to_datetime(d["date_public"], errors="coerce", dayfirst=True)
    if "value" not in d:
        d["value"] = 1
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(1)
    d["signal"] = d["signal"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    return d[["company", "signal", "date_public", "value"]].dropna(subset=["company", "date_public"])


def price_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Market-only fallback for price files (date, close[, ticker]): drawdown, volatility, 3-day move, relative weakness."""
    d = df.copy()
    d.columns = [_norm(c) for c in d.columns]
    tick = next((c for c in ("ticker", "symbol", "company", "name", "scripname") if c in d), None)
    dcol = next(c for c in ("date", "datetime", "timestamp") if c in d)
    ccol = "close" if "close" in d else "adjclose"
    d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
    d = d.sort_values(dcol)
    rows = []
    for t, g in (d.groupby(tick) if tick else [("ASSET", d)]):
        px = pd.to_numeric(g[ccol], errors="coerce").dropna()
        if len(px) < 3:
            continue
        r = px.pct_change().dropna()
        rows.append(dict(company=t, last=px.iloc[-1], drawdown=px.iloc[-1] / px.cummax().iloc[-1] - 1,
                         ret_window=px.iloc[-1] / px.iloc[0] - 1, vol=r.std(), worst_day=r.min()))
    o = pd.DataFrame(rows)
    if o.empty:
        return o
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    o["stress_score"] = z(-o.drawdown) + z(-o.ret_window) + z(o.vol) + z(-o.worst_day)
    return o.sort_values("stress_score", ascending=False).reset_index(drop=True)


def export_template(xlsx: str | Path, out_dir: str | Path | None = None) -> dict:
    """Split the filled CreditWatch workbook into the CSVs the Indian case harness reads. Example rows are dropped."""
    out_dir = Path(out_dir or ROOT / "data" / "indian")
    sh = pd.read_excel(xlsx, sheet_name=None)
    res = {}
    for name, fn in (("Financials", "financials.csv"), ("Events", "events.csv"), ("Peers", "peers.csv"), ("SoftSignals", "soft_signals.csv")):
        if name not in sh:
            continue
        d = sh[name].dropna(how="all")
        first = d.columns[0]
        d = d[~d[first].astype(str).str.upper().str.startswith("EXAMPLE")]
        if name == "Financials":
            d = d[d[RAW_COLS].notna().any(axis=1)]          # keep only rows where something was typed
        d.to_csv(out_dir / fn, index=False)
        res[fn] = len(d)
    return res
