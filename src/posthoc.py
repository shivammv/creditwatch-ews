"""POST-HOC descriptive analyses of the Indian panel (run after the frozen primary evaluation; no model or weight changes).

1. Threshold-free separation: ROC-AUC of defaulters vs matched peers at 4Q and 8Q lead (bootstrap 95% CI).
2. Lead-time curve: for leads of 1..12 quarters, the same frozen 4Q model + overlay is applied point-in-time, the cutoff is
   the 95th percentile of peer scores at that lead, and we record which defaulters are above it.
3. Timeline figures for the case slides.

Everything here is labelled post-hoc in the report: it describes the frozen system, it does not re-tune it.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import FEATURE_NAMES
from .indian import (ART, build_observation, load_events, load_financials, load_peers, load_soft, obs_date)
from .rank import apply_overlay, load_bundle, predict_proba

FIG = ART / "figures"


def _score_lead(lead: int, bundle, fin, soft, ev, peers) -> pd.DataFrame:
    rows = []
    for _, e in ev.iterrows():
        o = obs_date(e.event_date, lead)
        for name, is_def, is_fin in [(e.company, 1, bool(e.is_financial))] + \
                [(p.peer, 0, bool(p.is_financial)) for p in peers[peers.matched_to == e.company].itertuples()]:
            r = build_observation(name, o, fin, soft, is_fin)
            if r:
                r.update(is_defaulter=is_def, matched_to=e.company, lead_q=lead, event_date=e.event_date)
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["p_model"] = predict_proba(bundle, d[FEATURE_NAMES].astype(float))
    d["risk"] = apply_overlay(d.p_model.values, z_zone=d.z_zone, m_flag=d.m_flag, f_band=d.f_band, soft=d, icr=d.int_cover)
    return d


def _boot_auc(y, s, n=1000, seed=3):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def run(max_lead: int = 12, variant: str = "") -> dict:
    fin, soft, ev, peers = load_financials(), load_soft(), load_events(), load_peers()
    ev = ev[(ev.tier == "core") & ev.event_date.notna()]
    b4, b8 = load_bundle("production" + variant), load_bundle("production_8q" + variant)
    out = {"label": "POST-HOC (descriptive; frozen model and weights, no re-tuning)", "variant": variant or "primary"}
    # 1. AUC at the two pre-registered leads with their lead-matched models
    for lead, b in ((4, b4), (8, b8)):
        d = _score_lead(lead, b, fin, soft, ev, peers)
        y = d.is_defaulter.values
        out[f"lead_{lead}q_auc"] = {col: dict(auc=round(float(roc_auc_score(y, d[col])), 3), ci95=_boot_auc(y, d[col].values))
                                    for col in ("p_model", "risk")}
        # within-group rank: share of matched peers the defaulter outranks
        pct = []
        for g, gg in d.groupby("matched_to"):
            if gg.is_defaulter.sum() == 1 and (gg.is_defaulter == 0).any():
                dv = gg.loc[gg.is_defaulter == 1, "p_model"].iloc[0]
                pct.append(float((gg.loc[gg.is_defaulter == 0, "p_model"] < dv).mean()))
        out[f"lead_{lead}q_share_of_matched_peers_outranked_median"] = round(float(np.median(pct)), 3) if pct else None
    # 2. lead-time curve (4Q model throughout, so curves are comparable across leads)
    curve = []
    for lead in range(1, max_lead + 1):
        d = _score_lead(lead, b4, fin, soft, ev, peers)
        if d.empty:
            continue
        cut = float(np.quantile(d.loc[d.is_defaulter == 0, "risk"], 0.95))
        for _, r in d[d.is_defaulter == 1].iterrows():
            curve.append(dict(company=r.company, lead_q=lead, obs_date=str(r.obs_date)[:10], fy_used=str(r.fy_used)[:10],
                              p_model=round(float(r.p_model), 4), risk=round(float(r.risk), 4), peer_cutoff_95=round(cut, 4),
                              flagged=bool(r.risk > cut)))
    c = pd.DataFrame(curve)
    c.to_csv(ART / f"indian_lead_curve{variant}.csv", index=False)
    out["flagged_by_lead"] = {int(k): int(v) for k, v in c.groupby("lead_q").flagged.sum().items()}
    out["n_defaulters_by_lead"] = {int(k): int(v) for k, v in c.groupby("lead_q").size().items()}
    first = c[c.flagged].groupby("company").lead_q.max()
    out["earliest_flag_lead_q_per_defaulter"] = {k: int(v) for k, v in first.items()}
    out["univariate_auc_exploratory"] = univariate_auc(variant)
    (ART / f"indian_posthoc{variant}.json").write_text(json.dumps(out, indent=2))
    _figures(c, variant)
    return out


# sign: +1 if a higher value should mean higher risk
UNIVARIATE = {"cf_liab2": -1, "int_cover": -1, "net_profit_ta": -1, "ebit_ta": -1, "liabilities_ta": +1, "equity_ta": -1,
              "retained_ta": -1, "asset_turn": -1, "altman_z": -1, "wc_ta": -1, "current_ratio": -1, "op_margin": -1,
              "p_model": +1, "risk": +1}


def univariate_auc(variant: str = "") -> dict:
    """EXPLORATORY (n = 10 defaulters): how well single classic ratios separate Indian defaulters from matched peers.
    Chosen and read AFTER the frozen results; not used to build or tune any score in this submission."""
    d = pd.read_csv(ART / f"indian_scored{variant}.csv")
    res = {}
    for lead in (4, 8):
        x = d[d.lead_q == lead]
        r = {}
        for c, sgn in UNIVARIATE.items():
            v = pd.to_numeric(x[c], errors="coerce")
            m = v.notna()
            if x.loc[m, "is_defaulter"].nunique() == 2:
                r[c] = round(float(roc_auc_score(x.loc[m, "is_defaulter"], sgn * v[m])), 3)
        res[f"lead_{lead}q"] = dict(sorted(r.items(), key=lambda kv: -kv[1]))
    return res


def _figures(c: pd.DataFrame, variant: str = ""):
    FIG.mkdir(parents=True, exist_ok=True)
    # lead-time curve
    g = c.groupby("lead_q").agg(flagged=("flagged", "sum"), n=("flagged", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(g.lead_q, g.flagged, color="#3b6ea5")
    for _, r in g.iterrows():
        ax.text(r.lead_q, r.flagged + 0.1, f"{int(r.flagged)}/{int(r.n)}", ha="center", fontsize=8)
    ax.set_xlabel("Quarters before the credit event (point-in-time)")
    ax.set_ylabel("Defaulters above peer 5% cutoff")
    ax.set_title("When does the frozen system flag each defaulter?")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG / f"lead_time_curve{variant}.png", dpi=150)
    plt.close(fig)
    # per-company timelines
    for co, gg in c.groupby("company"):
        gg = gg.sort_values("lead_q")
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.plot(gg.lead_q, gg.risk, marker="o", color="#3b6ea5", label="EWS risk (point-in-time)")
        ax.plot(gg.lead_q, gg.peer_cutoff_95, ls=":", color="#7f8c8d", label="peer 95th-percentile cutoff")
        hit = gg[gg.flagged]
        ax.scatter(hit.lead_q, hit.risk, s=80, facecolors="none", edgecolors="#c0392b", lw=1.6, label="on watch-list", zorder=5)
        ax.invert_xaxis()
        ax.set_xlabel("Quarters before the credit event")
        ax.set_ylabel("Risk")
        ax.set_title(f"{co}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in co)
        fig.savefig(FIG / f"timeline_{safe}{variant}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    import sys
    print(json.dumps(run(variant=sys.argv[1] if len(sys.argv) > 1 else ""), indent=2))
