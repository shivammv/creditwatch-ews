"""
Score ANY file the organisers hand you:

    python score_any.py <file.csv|xlsx> [--asof 2026-10-27] [--soft soft.csv] [--budget-pct 0.05] [--sector-col sector]

Detects the format, maps columns, scores with the production model (+ Z''/M/F + soft-signal overlay),
ranks, and writes the top-5% watch-list with plain-language SHAP drivers to artifacts/watchlist_any.csv.
Formats: UCI Polish (Attr1..64) | company-year statements (any headers) | price/market file (fallback screen).
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from src import ingest

ART = Path(__file__).resolve().parent / "artifacts"


def _row_features(cur: dict, prev: dict | None, is_fin: bool):
    from src.features import indian_to_portable
    from src.scores import altman_z, altman_zone, beneish_flag, beneish_m, f_band, piotroski_f
    feats = indian_to_portable(cur, prev)
    r = dict(feats)
    if is_fin:
        r.update(altman_z=np.nan, z_zone="n/a_financial", beneish_m=np.nan, m_flag="n/a_financial", piotroski_f=np.nan, f_band="n/a_financial")
    else:
        z = altman_z(cur["current_assets"] - cur["current_liabilities"], cur["total_assets"], cur["retained_earnings"], cur["ebit"], cur["equity"], cur["total_liabilities"])
        m = beneish_m(cur, prev or {})
        f = piotroski_f(cur, prev or {})
        r.update(altman_z=z, z_zone=altman_zone(z), beneish_m=m, m_flag=beneish_flag(m), piotroski_f=f, f_band=f_band(f))
    return r


def score_statements(fin: pd.DataFrame, soft: pd.DataFrame, asof: pd.Timestamp | None):
    from src.features import FEATURE_NAMES
    from src.rank import SOFT_SIGNALS, SOFT_LABELS, apply_overlay, load_bundle, predict_proba, top_drivers
    bundle = load_bundle("production")
    rows = []
    for co, g in fin.groupby("company"):
        g = g.sort_values("fy_end")
        if asof is not None:
            g = g[g.public_date <= asof]                      # leakage guard: only what was public on the as-of date
        g = g[g.total_assets.notna()]
        if g.empty:
            continue
        cur = g.iloc[-1][ingest.RAW_COLS].to_dict()
        prev = g.iloc[-2][ingest.RAW_COLS].to_dict() if len(g) > 1 else None
        r = _row_features(cur, prev, bool(g.iloc[-1].is_financial))
        r.update(company=co, sector=g.iloc[-1].sector, fy_used=g.iloc[-1].fy_end, years_available=len(g))
        ref = asof if asof is not None else g.iloc[-1].public_date
        s = soft[(soft.company == co) & (soft.date_public <= ref) & (soft.date_public > ref - pd.Timedelta(days=365))] if len(soft) else soft
        r["soft_list"] = []
        for sig in SOFT_SIGNALS:
            v = s[s.signal == sig].value if len(s) else []
            on = int(len(v) > 0 and (max(v) >= 50 if sig == "promoter_pledge_high" else max(v) > 0))
            r[sig] = on
            if on:
                r["soft_list"].append(sig)
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No company has statements public on/before the as-of date.")
    df["p_model"] = predict_proba(bundle, df[FEATURE_NAMES].astype(float))
    df["risk"] = apply_overlay(df.p_model.values, z_zone=df.z_zone, m_flag=df.m_flag, f_band=df.f_band, soft=df, icr=df.int_cover)
    return df, bundle, FEATURE_NAMES


def score_polish(df: pd.DataFrame):
    from src.features import FEATURE_NAMES, polish_to_portable
    from src.rank import apply_overlay, load_bundle, predict_proba
    bundle = load_bundle("production")
    X = polish_to_portable(df)[FEATURE_NAMES]
    out = X.copy()
    out["company"] = df["id"].astype(str) if "id" in df else [f"Firm-{i:05d}" for i in range(len(df))]
    out["p_model"] = predict_proba(bundle, X.astype(float))
    out["risk"] = out.p_model
    return out, bundle, FEATURE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--asof", default=None, help="only use statements public on/before this date")
    ap.add_argument("--soft", default=None, help="optional soft-signals file (company, signal, date_public, value)")
    ap.add_argument("--budget-pct", type=float, default=0.05)
    ap.add_argument("--out", default=str(ART / "watchlist_any.csv"))
    a = ap.parse_args()

    sheets = ingest.load_any(a.file)
    kinds = {k: ingest.detect_kind(v) for k, v in sheets.items()}
    print("Detected:", kinds)
    soft = pd.DataFrame(columns=["company", "signal", "date_public", "value"])
    for k, v in sheets.items():
        if kinds[k] == "soft":
            soft = ingest.to_soft(v)
    if a.soft:
        soft = ingest.to_soft(next(iter(ingest.load_any(a.soft).values())))
    main_name = next((k for k, t in kinds.items() if t in ("statements", "polish", "prices", "taiwan")), None)
    if main_name is None:
        sys.exit(f"Could not recognise any sheet. Columns seen: { {k: list(v.columns)[:12] for k, v in sheets.items()} }\n"
                 "Fix: rename headers to the template names (see CreditWatch_Input_Template.xlsx) or add aliases in src/ingest.py::ALIASES.")
    kind, df = kinds[main_name], sheets[main_name]
    asof = pd.Timestamp(a.asof) if a.asof else None

    if kind == "prices":
        o = ingest.price_stress(df)
        n = len(o); k = max(1, math.ceil(a.budget_pct * n)); o["in_budget"] = o.index < k
        o.to_csv(a.out, index=False)
        print(f"PRICE-ONLY SCREEN (no statements found): {n} names, top {k} flagged -> {a.out}\n", o.head(k).to_string(index=False))
        return
    if kind == "taiwan":
        sys.exit("Taiwanese benchmark detected: this is a TRAINING file (labels present), not a book to score. "
                 "Use it to validate: add a loader in src/model.py, or score with --polish-style mapping.")
    if kind == "polish":
        res, bundle, feats = score_polish(df)
    else:
        fin, rep = ingest.to_statements(df)
        print("Mapped columns:", rep["mapped"]); print("Unmapped (ignored):", rep["unmapped"])
        print("Derived:", rep["derived"] or "none", "| entirely missing:", rep["missing"] or "none")
        res, bundle, feats = score_statements(fin, soft, asof)

    from src.rank import SOFT_LABELS, memo_text, top_drivers
    res = res.sort_values("risk", ascending=False).reset_index(drop=True)
    n = len(res); k = max(1, math.ceil(a.budget_pct * n))
    res["rank"] = np.arange(1, n + 1); res["in_budget"] = res["rank"] <= k
    drv = top_drivers(bundle, res.head(k)[feats].astype(float))
    res["drivers"] = ""; res["memo"] = ""
    for i, d in enumerate(drv):
        r = res.loc[i]
        res.loc[i, "drivers"] = " | ".join(x["text"] for x in d)
        res.loc[i, "memo"] = memo_text(r.company, i + 1, r.risk, d, z=r.get("altman_z"), m=r.get("beneish_m"), f=r.get("piotroski_f"), soft=r.get("soft_list") or None)
    res.drop(columns=["soft_list"], errors="ignore").to_csv(a.out, index=False)
    print(f"\n{n} names scored, budget = top {k} ({a.budget_pct:.0%}) -> {a.out}\n")
    for i in range(min(k, 10)):
        print(res.loc[i, "memo"], "\n")


if __name__ == "__main__":
    main()
