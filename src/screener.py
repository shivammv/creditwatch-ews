"""
Adapter for the Screener-style bulk dump ("Detailed Financials Data Of 4456 NSE and BSE Company").
One folder per company with Yearly_Balance_Sheet / Yearly_Profit_Loss / Yearly_Cash_flow / Ratios / Basic_Info (INR crore).

Maps it to the standard company-year schema used by the pipeline (src/ingest.RAW_COLS).

HONEST MAPPING NOTES (these are approximations, flagged bs_proxy=1):
  * The balance sheet is condensed: there is no current-assets / current-liabilities split.
      current_assets      ~ 'Other Assets'      (inventory + receivables + cash + other; mostly current for non-financials)
      current_liabilities ~ 'Other Liabilities' (payables + provisions + other; EXCLUDES short-term borrowings -> flatters liquidity)
  * receivables / inventory are rebuilt from Debtor Days / Inventory Days x Sales / 365 (approximation).
  * cogs, sga, cash, shares_out are not in the dump -> NaN. Beneish M (needs COGS, SG&A) and Piotroski F (needs shares) are therefore NaN.
  * ebit = PBT + interest;  total_liabilities = total assets - (equity capital + reserves).
  * Figures are the latest restated view as of Dec-2023, not the as-originally-published filings (restatement/leakage caveat).
"""
from __future__ import annotations
import difflib, re
from pathlib import Path
import numpy as np
import pandas as pd

LAG_DAYS = 120
COLS = ["company", "fy_end", "public_date", "total_assets", "current_assets", "current_liabilities", "total_liabilities", "equity",
        "retained_earnings", "sales", "cogs", "sga", "depreciation", "ebit", "interest_expense", "pbt", "net_income", "cfo",
        "receivables", "inventory", "ppe_net", "lt_debt", "cash", "shares_out", "ebitda", "bs_proxy"]


# manual name fixes (renames / ambiguous). None = deliberately NOT matched (company absent or different entity).
ALIAS = {
    "Cholamandalam Investment": "Cholamandalam Investment _ Finance Company Ltd",
    "IRB Infrastructure": "IRB Infrastructure Developers Ltd",
    "SAIL": "Steel Authority of India Ltd",
    "Welspun India": "Welspun Living Ltd",
    "Sintex Industries": None,   # Sintex Plastics Technology is the demerged entity, not the insolvent company
    "Bhushan Steel": None, "DHFL": None, "Lanco Infratech": None, "Videocon Industries": None, "Amtek Auto": None,
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("&", "_"))


def find_root(base: str | Path) -> Path:
    b = Path(base)
    kids = [p for p in b.iterdir() if p.is_dir()]
    return kids[0] if len(kids) == 1 and not (kids[0] / "Yearly_Balance_Sheet.csv").exists() else b


class Dump:
    def __init__(self, root: str | Path):
        self.root = find_root(root)
        self.index = {_norm(p.name): p.name for p in self.root.iterdir() if p.is_dir()}

    def match(self, name: str, cutoff: float = 0.9) -> str | None:
        if name in ALIAS:
            return ALIAS[name]
        k = _norm(name)
        if k in self.index:
            return self.index[k]
        for suf in ("ltd", "limited"):
            if k + suf in self.index:
                return self.index[k + suf]
        pre = [v for kk, v in self.index.items() if kk.startswith(k) and len(k) >= 5]
        if len(pre) == 1:
            return pre[0]
        m = difflib.get_close_matches(k, list(self.index), n=1, cutoff=cutoff)
        return self.index[m[0]] if m else None

    def company(self, folder: str) -> pd.DataFrame:
        d = self.root / folder
        def rd(f):
            p = d / f
            return pd.read_csv(p, index_col=0) if p.exists() else pd.DataFrame()
        bs, pl, cf, rt = rd("Yearly_Balance_Sheet.csv"), rd("Yearly_Profit_Loss.csv"), rd("Yearly_Cash_flow.csv"), rd("Ratios.csv")
        def ym_bs(c):   # '2019-03-01'
            t = pd.to_datetime(c, errors="coerce"); return None if pd.isna(t) else (t.year, t.month)
        def ym_pl(c):   # 'Mar 2019'
            t = pd.to_datetime(c, format="%b %Y", errors="coerce"); return None if pd.isna(t) else (t.year, t.month)
        def table(df, fn):
            out = {}
            for c in df.columns:
                k = fn(c)
                if k: out[k] = df[c]
            return out
        B, P, C, R = table(bs, ym_bs), table(pl, ym_pl), table(cf, ym_bs), table(rt, ym_bs)
        rows = []
        for k in sorted(B):
            if k not in P:
                continue
            b, p = B[k], P[k]
            c, r = C.get(k, pd.Series(dtype=float)), R.get(k, pd.Series(dtype=float))
            g = lambda s, key: pd.to_numeric(s.get(key, np.nan), errors="coerce")
            ta, eqc, res = g(b, "Total Assets"), g(b, "Equity Capital"), g(b, "Reserves")
            sales = g(p, "Sales") if "Sales" in p.index else g(p, "Revenue")        # NBFC/bank layout uses "Revenue" (total income)
            pbt, intr = g(p, "Profit before tax"), g(p, "Interest")
            fy_end = (pd.Timestamp(year=k[0], month=k[1], day=1) + pd.offsets.MonthEnd(0))
            rows.append(dict(
                company=folder, fy_end=fy_end, public_date=fy_end + pd.Timedelta(days=LAG_DAYS),
                total_assets=ta, current_assets=g(b, "Other Assets"), current_liabilities=g(b, "Other Liabilities"),
                total_liabilities=ta - (eqc + res) if np.isfinite(ta) and np.isfinite(eqc) and np.isfinite(res) else np.nan,
                equity=eqc + res, retained_earnings=res, sales=sales, cogs=np.nan, sga=np.nan,
                depreciation=g(p, "Depreciation"), ebit=pbt + intr, interest_expense=intr, pbt=pbt, net_income=g(p, "Net Profit"),
                cfo=g(c, "Cash from Operating Activity"),
                receivables=g(r, "Debtor Days") * sales / 365, inventory=g(r, "Inventory Days") * sales / 365,
                ppe_net=g(b, "Fixed Assets"), lt_debt=g(b, "Borrowings"), cash=np.nan, shares_out=np.nan,
                ebitda=g(p, "Operating Profit") if "Operating Profit" in p.index else g(p, "Financing Profit"), bs_proxy=1))
        df = pd.DataFrame(rows, columns=COLS)
        # one row per fiscal year-end (drop half-year duplicates that share a year with a later March row only if sales missing)
        return df[df.sales.notna() & df.total_assets.notna()].reset_index(drop=True)

    def info(self, folder: str) -> dict:
        p = self.root / folder / f"{folder}_Basic_Info.csv"
        if not p.exists():
            return {}
        d = pd.read_csv(p, index_col=0).iloc[0]
        return dict(sector=d.get("Sector"), bse=d.get("BSE"), nse=d.get("NSE"), market_cap=d.get("Market Cap"))


def build_panel(root, events_csv, peers_csv, out_dir) -> dict:
    dump = Dump(root)
    ev, pe = pd.read_csv(events_csv), pd.read_csv(peers_csv)
    frames, report = [], []
    def add(name, role, matched_to, is_fin):
        f = dump.match(name)
        if f is None:
            report.append(dict(company=name, role=role, folder=None, years=0)); return
        df = dump.company(f)
        info = dump.info(f)
        df["company_input"] = name
        df["role"] = role; df["matched_to"] = matched_to; df["is_financial"] = int(is_fin)
        df["sector"] = info.get("sector"); df["source_url"] = f"screener dump folder: {f}"
        frames.append(df)
        report.append(dict(company=name, role=role, folder=f, years=len(df), first=str(df.fy_end.min().date()) if len(df) else None,
                           last=str(df.fy_end.max().date()) if len(df) else None))
    for _, e in ev[ev.tier == "core"].iterrows():
        add(e.company, "defaulter", e.company, e.is_financial)
    for _, p in pe.iterrows():
        add(p.peer, "peer", p.matched_to, p.is_financial)
    fin = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(fin):
        fin["company"] = fin["company_input"]          # use the names in events/peers so the harness joins cleanly
        fin = fin.drop(columns=["company_input"]).drop_duplicates(["company", "fy_end"])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fin.to_csv(Path(out_dir) / "financials.csv", index=False)
    return dict(report=pd.DataFrame(report), n_rows=len(fin))
