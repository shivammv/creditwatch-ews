"""
Indian case test: leakage-safe panel loader, as-of feature builder and recall-at-budget evaluation.

LEAKAGE RULES enforced in code
  1. For a lead of L quarters the observation date is  event_date - 3*L months.
  2. A fiscal year is usable only if its public_date (filing/results date; default fy_end + 120 days) <= observation date.
  3. A soft signal is usable only if date_public <= observation date, and counts if within the prior 12 months.
  4. Matched peers are observed at the SAME calendar date as the defaulter they are matched to.
  5. Nothing dated after the observation date is ever read; `assert_no_leakage` re-checks every feature row.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, RAW_COLS, indian_to_portable, polish_to_portable
from .model import ART
from .rank import SOFT_SIGNALS, apply_overlay, load_bundle, predict_proba, top_drivers
from .scores import altman_z, altman_zone, beneish_flag, beneish_m, f_band, piotroski_f

ROOT = Path(__file__).resolve().parents[1]
IND = ROOT / "data" / "indian"
PUBLICATION_LAG_DAYS = 120          # conservative: FY ends 31-Mar, audited results + annual report public by ~end-July
MIN_DEFAULTERS, MIN_PEERS = 10, 30
REQUIRED_RAW = ["total_assets", "current_assets", "current_liabilities", "total_liabilities", "equity", "retained_earnings",
                "sales", "ebit", "interest_expense", "pbt", "net_income"]


def load_events() -> pd.DataFrame:
    df = pd.read_csv(IND / "events.csv")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return df


def load_peers() -> pd.DataFrame:
    return pd.read_csv(IND / "peers.csv")


def load_financials() -> pd.DataFrame:
    p = IND / "financials.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["fy_end"] = pd.to_datetime(df["fy_end"], errors="coerce")
    df["public_date"] = pd.to_datetime(df["public_date"], errors="coerce")
    df["public_date"] = df["public_date"].fillna(df["fy_end"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS))
    return annualise(df)


FLOW_COLS = ["sales", "cogs", "sga", "depreciation", "ebit", "interest_expense", "pbt", "net_income", "cfo", "ebitda"]


def annualise(df: pd.DataFrame) -> pd.DataFrame:
    """Scale flow items of non-12-month fiscal periods to 12 months (e.g. Alok Industries reported two 18-month
    periods, Sep-2013 and Mar-2015). Stock items (assets, debt, equity) are left as reported. Period length is the gap
    to the company's previous fiscal year-end; the first year of each company is assumed to be 12 months."""
    if df.empty:
        return df
    df = df.sort_values(["company", "fy_end"]).copy()
    gap = df.groupby("company")["fy_end"].diff()
    months = (gap.dt.days / 30.4375).round()
    df["period_months"] = months.fillna(12.0)
    scale = np.where((df["period_months"] < 11) | (df["period_months"] > 13), 12.0 / df["period_months"], 1.0)
    for c in FLOW_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce") * scale
    df["annualised"] = scale != 1.0
    return df.reset_index(drop=True)


def load_soft() -> pd.DataFrame:
    p = IND / "soft_signals.csv"
    if not p.exists() or p.stat().st_size < 20:
        return pd.DataFrame(columns=["company", "signal", "date_public", "value"])
    df = pd.read_csv(p)
    df["date_public"] = pd.to_datetime(df["date_public"], errors="coerce")
    return df


def panel_status() -> dict:
    """How much of the required panel is actually populated (drives the app's honesty banner)."""
    fin = load_financials()
    ev = load_events()
    core = ev[(ev.tier == "core") & ev.event_date.notna()]
    peers = load_peers()
    if fin.empty:
        return dict(ready=False, defaulters_with_data=0, peers_with_data=0, core_defaulters=len(core), peers_listed=len(peers), rows=0)
    ok = fin.dropna(subset=REQUIRED_RAW).groupby("company").size()
    ok = ok[ok >= 2].index
    d = int(core.company.isin(ok).sum())
    p = int(peers.peer.isin(ok).sum())
    return dict(ready=d >= MIN_DEFAULTERS and p >= MIN_PEERS, defaulters_with_data=d, peers_with_data=p,
                core_defaulters=len(core), peers_listed=len(peers), rows=len(fin))


def obs_date(event_date: pd.Timestamp, lead_q: int) -> pd.Timestamp:
    return event_date - pd.DateOffset(months=3 * lead_q)


def build_observation(company: str, obs: pd.Timestamp, fin: pd.DataFrame, soft: pd.DataFrame, is_financial: bool) -> dict | None:
    f = fin[(fin.company == company) & (fin.public_date <= obs)].sort_values("fy_end")
    f = f.dropna(subset=REQUIRED_RAW)
    if len(f) < 2:
        return None
    cur, prev = f.iloc[-1][RAW_COLS].to_dict(), f.iloc[-2][RAW_COLS].to_dict()
    feats = indian_to_portable(cur, prev)
    row = dict(company=company, obs_date=obs, fy_used=f.iloc[-1].fy_end, public_date_used=f.iloc[-1].public_date, **feats)
    if is_financial:     # Z'' / M / F are undefined for banks & NBFCs (no working capital / gross margin concept)
        row.update(altman_z=np.nan, z_zone="n/a_financial", beneish_m=np.nan, m_flag="n/a_financial", piotroski_f=np.nan, f_band="n/a_financial")
    else:
        z = altman_z(cur["current_assets"] - cur["current_liabilities"], cur["total_assets"], cur["retained_earnings"], cur["ebit"], cur["equity"], cur["total_liabilities"])
        m = beneish_m(cur, prev)
        fs = piotroski_f(cur, prev)
        row.update(altman_z=z, z_zone=altman_zone(z), beneish_m=m, m_flag=beneish_flag(m), piotroski_f=fs, f_band=f_band(fs))
    s = soft[(soft.company == company) & (soft.date_public <= obs) & (soft.date_public > obs - pd.Timedelta(days=365))]
    for sig in SOFT_SIGNALS:
        v = s[s.signal == sig].value
        if sig == "promoter_pledge_high":
            row[sig] = int(len(v) > 0 and v.max() >= 50)
        else:
            row[sig] = int(len(v) > 0 and v.max() > 0)
    return row


def assert_no_leakage(obs_df: pd.DataFrame) -> None:
    bad = obs_df[obs_df.public_date_used > obs_df.obs_date]
    if len(bad):
        raise AssertionError(f"LEAKAGE: {len(bad)} rows use filings published after the observation date")


LEAD_BUNDLE = {4: ("production", ""), 8: ("production_8q", "_8q")}   # lead quarters -> (model bundle, artifact suffix)


def lead_bundle(lead: int, variant: str = "") -> tuple[str, str]:
    name, suf = LEAD_BUNDLE[lead]
    return name + variant, suf + variant


def _background(lead: int, variant: str = ""):
    """Secondary method: 'rest of the book' = Polish hold-out firms of the lead-matched file, scored by the SAME model.
    They get the Z'' and interest-cover overlays (computable from Polish ratios); M/F/soft signals are not available
    for them, so this method slightly favours flagging Indian firms. The primary metric (peer cutoff) has no such bias."""
    _, suf = lead_bundle(lead, variant)
    h = pd.read_csv(ART / f"holdout_predictions{suf}.csv")
    df = pd.read_pickle(ART / f"polish_dedup{suf}.pkl")
    a = df.loc[h.row]
    z = 6.56 * a.Attr3.values + 3.26 * a.Attr6.values + 6.72 * a.Attr7.values + 1.05 * a.Attr8.values
    zone = np.where(~np.isfinite(z), "unknown", np.where(z > 2.6, "safe", np.where(z >= 1.1, "grey", "distress")))
    p_adj = apply_overlay(h.p.values, z_zone=zone, icr=a.Attr27.values)
    return h.p.values, p_adj, h.y.values


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def peer_cutoff_recall(obs_df: pd.DataFrame, col: str, q: float = 0.95, n_boot: int = 1000, seed: int = 11) -> dict:
    """PRIMARY metric. Non-defaulting peers stand in for the performing book (defaulters are ~2% of a real book, so the
    book's 95th percentile is essentially the performing firms' 95th percentile). Cutoff = 95th percentile of peer scores
    at the same lead; recall = share of defaulters strictly above it; Wilson 95% CI; bootstrap CI of the cutoff."""
    peers = obs_df.loc[obs_df.is_defaulter == 0, col].astype(float).values
    d = obs_df.loc[obs_df.is_defaulter == 1, col].astype(float).values
    cut = float(np.quantile(peers, q))
    rng = np.random.default_rng(seed)
    boots = [np.quantile(rng.choice(peers, len(peers), replace=True), q) for _ in range(n_boot)]
    k, n = int((d > cut).sum()), len(d)
    return dict(cutoff=round(cut, 4), cutoff_ci95=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                caught=k, n_defaulters=n, recall=round(k / n, 3) if n else None, recall_wilson95=wilson(k, n),
                peers_flagged=int((peers > cut).sum()), n_peers=int(len(peers)))


def evaluate(lead_quarters=(4, 8), n_universe: int = 800, budget: int = 40, bg_rate: float = 0.02, draws: int = 500,
             seed: int = 11, check_frozen: bool = True, variant: str = "") -> dict:
    """variant "" = primary pre-registered system; "_v2" = post-hoc monotone variant (separately frozen)."""
    from .config import verify_frozen
    if check_frozen:
        verify_frozen(variant)   # refuses to score Indian cases if the frozen configuration was edited
    fin, soft, ev, peers = load_financials(), load_soft(), load_events(), load_peers()
    ev = ev[(ev.tier == "core") & ev.event_date.notna()]
    rng = np.random.default_rng(seed)
    out = {}
    all_rows = []
    for lead in lead_quarters:
        bundle = load_bundle(lead_bundle(lead, variant)[0])
        p_bg_model, p_bg_adj, y_bg = _background(lead, variant)
        rows = []
        for _, e in ev.iterrows():
            o = obs_date(e.event_date, lead)
            r = build_observation(e.company, o, fin, soft, bool(e.is_financial))
            if r:
                r.update(is_defaulter=1, matched_to=e.company, lead_q=lead, sector=e.sector, event_date=e.event_date)
                rows.append(r)
            for _, pr in peers[peers.matched_to == e.company].iterrows():
                r = build_observation(pr.peer, o, fin, soft, bool(pr.is_financial))
                if r:
                    r.update(is_defaulter=0, matched_to=e.company, lead_q=lead, sector=pr.sector, event_date=e.event_date)
                    rows.append(r)
        if not rows:
            out[f"lead_{lead}q"] = dict(error="no usable observations")
            continue
        obs_df = pd.DataFrame(rows)
        assert_no_leakage(obs_df)
        X = obs_df[FEATURE_NAMES].astype(float)
        obs_df["p_model"] = predict_proba(bundle, X)
        forensic = dict(z_zone=obs_df.z_zone, m_flag=obs_df.m_flag, f_band=obs_df.f_band, icr=obs_df.int_cover)
        obs_df["risk_model_forensic"] = apply_overlay(obs_df["p_model"].values, **forensic)
        obs_df["risk"] = apply_overlay(obs_df["p_model"].values, soft=obs_df, **forensic)
        base = np.full(len(obs_df), 0.02)
        obs_df["risk_soft_only"] = apply_overlay(base, soft=obs_df)
        obs_df["risk_forensic_only"] = apply_overlay(base, **forensic)

        # PRIMARY: Indian peer cutoff (apples-to-apples), with ablations
        primary = {name: peer_cutoff_recall(obs_df, col) for name, col in
                   [("model_only", "p_model"), ("model_plus_forensic", "risk_model_forensic"), ("full", "risk"),
                    ("forensic_only", "risk_forensic_only"), ("soft_only", "risk_soft_only")]}
        cut_full = primary["full"]["cutoff"]
        obs_df["above_peer_cutoff"] = obs_df["risk"] > cut_full

        # SECONDARY: simulated 800-name book with Polish hold-out background (CreditWatch v2 method)
        n_bg = n_universe - len(obs_df)
        pos, neg = np.where(y_bg == 1)[0], np.where(y_bg == 0)[0]
        n_pos = int(round(bg_rate * n_bg))
        d_idx = np.where(obs_df.is_defaulter.values == 1)[0]
        ranks_model = np.zeros((draws, len(d_idx)))
        ranks_full = np.zeros_like(ranks_model)
        for k in range(draws):
            bg = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_bg - n_pos, replace=False)])
            for col, bgv, store in (("p_model", p_bg_model, ranks_model), ("risk", p_bg_adj, ranks_full)):
                scores = np.concatenate([obs_df[col].values, bgv[bg]])
                rank = (-scores).argsort().argsort() + 1
                store[k] = rank[d_idx]
        rec_model = (ranks_model <= budget).mean(axis=1)
        rec_full = (ranks_full <= budget).mean(axis=1)
        obs_df.loc[obs_df.is_defaulter == 1, "median_rank_full"] = np.median(ranks_full, axis=0)
        obs_df.loc[obs_df.is_defaulter == 1, "median_rank_model_only"] = np.median(ranks_model, axis=0)
        obs_df.loc[obs_df.is_defaulter == 1, "p_in_top40"] = (ranks_full <= budget).mean(axis=0)
        dr = top_drivers(bundle, X)
        obs_df["drivers"] = [" | ".join(x["text"] for x in d) for d in dr]
        all_rows.append(obs_df)
        out[f"lead_{lead}q"] = dict(
            model=lead_bundle(lead, variant)[0], n_defaulters=int(obs_df.is_defaulter.sum()), n_peers=int((obs_df.is_defaulter == 0).sum()),
            defaulters_evaluated=sorted(obs_df.loc[obs_df.is_defaulter == 1, "company"].tolist()),
            defaulters_missing=sorted(set(ev.company) - set(obs_df.loc[obs_df.is_defaulter == 1, "company"])),
            primary_peer_cutoff=primary,
            recall_full=primary["full"]["recall"], recall_model_only=primary["model_only"]["recall"],
            secondary_polish_background=dict(
                universe=n_universe, budget=budget, background_default_rate=bg_rate, draws=draws,
                recall_model_only=round(float(rec_model.mean()), 3), recall_full=round(float(rec_full.mean()), 3),
                recall_full_ci95_over_draws=[round(float(np.percentile(rec_full, 2.5)), 3), round(float(np.percentile(rec_full, 97.5)), 3)],
                caveat="Background firms are Polish hold-out firms; they receive Z'' and interest-cover overlays but no M/F/soft "
                       "signals, which favours flagging Indian firms. Use the primary peer-cutoff metric for claims."))
    if all_rows:
        pd.concat(all_rows).to_csv(ART / f"indian_scored{variant}.csv", index=False)
    (ART / f"indian_summary{variant}.json").write_text(json.dumps(out, indent=2, default=float))
    return out
