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
    return df


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


def _background(draws: int, n: int, rate: float, seed: int):
    """Healthy/failed 'rest of the book' proxy: Polish hold-out firms scored by the SAME portable model + Z'' overlay."""
    h = pd.read_csv(ART / "holdout_predictions.csv")
    df = pd.read_pickle(ART / "polish_dedup.pkl")
    a = df.loc[h.row]
    z = 6.56 * a.Attr3.values + 3.26 * a.Attr6.values + 6.72 * a.Attr7.values + 1.05 * a.Attr8.values
    zone = np.where(~np.isfinite(z), "unknown", np.where(z > 2.6, "safe", np.where(z >= 1.1, "grey", "distress")))
    p_adj = apply_overlay(h.p.values, z_zone=zone)
    return h.p.values, p_adj, h.y.values


def evaluate(lead_quarters=(4, 8), n_universe: int = 800, budget: int = 40, bg_rate: float = 0.02, draws: int = 500, seed: int = 11) -> dict:
    fin, soft, ev, peers = load_financials(), load_soft(), load_events(), load_peers()
    ev = ev[(ev.tier == "core") & ev.event_date.notna()]
    bundle = load_bundle("production")
    rng = np.random.default_rng(seed)
    p_bg_model, p_bg_adj, y_bg = _background(draws, n_universe, bg_rate, seed)
    out = {}
    all_rows = []
    for lead in lead_quarters:
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
                    r.update(is_defaulter=0, matched_to=e.company, lead_q=lead, sector=pr.sector, event_date=pd.NaT)
                    rows.append(r)
        if not rows:
            out[f"lead_{lead}q"] = dict(error="no usable observations")
            continue
        obs_df = pd.DataFrame(rows)
        assert_no_leakage(obs_df)
        obs_df["p_model"] = predict_proba(bundle, obs_df[FEATURE_NAMES].astype(float))
        obs_df["risk"] = apply_overlay(obs_df["p_model"].values, z_zone=obs_df.z_zone, m_flag=obs_df.m_flag, f_band=obs_df.f_band, soft=obs_df, icr=obs_df.int_cover)
        n_bg = n_universe - len(obs_df)
        pos, neg = np.where(y_bg == 1)[0], np.where(y_bg == 0)[0]
        n_pos = int(round(bg_rate * n_bg))
        hit_model, hit_full = [], []
        ranks_model = np.zeros((draws, int(obs_df.is_defaulter.sum())))
        ranks_full = np.zeros_like(ranks_model)
        d_idx = np.where(obs_df.is_defaulter.values == 1)[0]
        for k in range(draws):
            bg = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_bg - n_pos, replace=False)])
            for tag, col, bgv, store in (("model", "p_model", p_bg_model, ranks_model), ("full", "risk", p_bg_adj, ranks_full)):
                scores = np.concatenate([obs_df[col].values, bgv[bg]])
                rank = (-scores).argsort().argsort() + 1
                store[k] = rank[d_idx]
        rec_model = (ranks_model <= budget).mean(axis=1)
        rec_full = (ranks_full <= budget).mean(axis=1)
        obs_df.loc[obs_df.is_defaulter == 1, "median_rank_full"] = np.median(ranks_full, axis=0)
        obs_df.loc[obs_df.is_defaulter == 1, "median_rank_model_only"] = np.median(ranks_model, axis=0)
        obs_df.loc[obs_df.is_defaulter == 1, "p_in_top40"] = (ranks_full <= budget).mean(axis=0)
        dr = top_drivers(bundle, obs_df[FEATURE_NAMES].astype(float))
        obs_df["drivers"] = [" | ".join(x["text"] for x in d) for d in dr]
        all_rows.append(obs_df)
        out[f"lead_{lead}q"] = dict(
            n_defaulters=int(obs_df.is_defaulter.sum()), n_peers=int((obs_df.is_defaulter == 0).sum()), universe=n_universe, budget=budget,
            recall_model_only=float(rec_model.mean()), recall_full=float(rec_full.mean()),
            recall_full_ci95=[float(np.percentile(rec_full, 2.5)), float(np.percentile(rec_full, 97.5))],
            background_default_rate=bg_rate)
    if all_rows:
        pd.concat(all_rows).to_csv(ART / "indian_scored.csv", index=False)
    (ART / "indian_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out
