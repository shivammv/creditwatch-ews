"""CreditWatch EWS: Streamlit app. Every number shown is read from artifacts produced by run_all.py."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import indian
from src.features import FEATURE_NAMES, RAW_COLS, indian_to_portable
from src.rank import OVERLAY, SOFT_LABELS, SOFT_SIGNALS, apply_overlay, load_bundle, memo_text, predict_proba, top_drivers
from src.scores import altman_z, altman_zone, beneish_flag, beneish_m, f_band, piotroski_f

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
st.set_page_config(page_title="CreditWatch EWS", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#0f172a,#1e3a8a);color:#fff;padding:1.4rem 1.6rem;border-radius:14px;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.9rem}.hero p{margin:.4rem 0 0;color:#cbd5e1}
.card{border:1px solid #e2e8f0;border-radius:12px;padding:.9rem 1rem;background:#fff}
.warn{background:#fffbeb;border-left:4px solid #f59e0b;padding:.7rem 1rem;border-radius:6px;color:#78350f}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_all():
    rep = json.loads((ART / "benchmark_report.json").read_text())
    sims = json.loads((ART / "universe_sim.json").read_text())
    pr = pd.read_csv(ART / "pr_curve.csv")
    wl = pd.read_csv(ART / "sample_watchlist.csv")
    return rep, sims, pr, wl


rep, sims, pr, wl = load_all()
prod, ref, meta, abl = rep["production"], rep["reference"], rep["meta"], rep["ablation"]
status = indian.panel_status()

st.markdown(f"""<div class="hero"><h1>🛡️ CreditWatch EWS</h1>
<p>Ranked quarterly watch-list for a 800-borrower book · 40-name review budget (5%) · 20:1 miss-vs-false-alarm cost · SHAP explanations a credit officer can act on</p></div>""",
            unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Benchmark", "5% budget", "Watch-list", "Score a company", "Indian case test", "Policy"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    s2 = sims["0.02"]
    c = st.columns(4)
    c[0].metric("Hold-out PR-AUC", f"{prod['pr_auc']:.2f}", f"baseline {prod['pr_auc_baseline']:.3f} · CI {prod['pr_auc_ci95'][0]:.2f}-{prod['pr_auc_ci95'][1]:.2f}")
    c[1].metric("Recall @ 40 of 800 (2% default rate)", f"{s2['recall_mean']:.0%}", f"CI {s2['recall_ci95'][0]:.0%}-{s2['recall_ci95'][1]:.0%}")
    c[2].metric("Expected cost vs flag-nobody", f"{s2['cost_at_budget']:.0f} vs {s2['cost_flag_none']:.0f}", f"{1 - s2['cost_at_budget'] / s2['cost_flag_none']:.0%} lower")
    c[3].metric("Indian panel", f"{status['defaulters_with_data']}/{status['core_defaulters']} defaulters · {status['peers_with_data']}/{status['peers_listed']} peers",
                "ready" if status["ready"] else "not populated yet", delta_color="normal" if status["ready"] else "off")
    if not status["ready"]:
        st.markdown('<div class="warn"><b>Indian recall is not reported yet.</b> The pipeline, leakage guard and panel schema are built and tested, but '
                    'annual-report data for the Indian defaulters and peers has to be entered (see <code>docs/DATA_ENTRY.md</code>). '
                    'Until then this app makes <b>no</b> claim about recovering IL&amp;FS, DHFL, Jet Airways etc.</div>', unsafe_allow_html=True)
    st.markdown("#### What was built")
    st.markdown("""
1. **Distress model** (LightGBM) on the UCI Polish bankruptcy data, using only ratios that can be rebuilt from an Indian annual report; probabilities calibrated, threshold chosen on the 20:1 cost.
2. **Forensic screens**: Altman Z'' (emerging-market), Beneish M-score, Piotroski F-score, plus rule-based overlays (interest cover < 1) and five **soft signals** (rating downgrade, promoter pledge, auditor red flag, delayed filing, related-party stress).
3. **Explainability**: per-company top-3 SHAP drivers rewritten as credit-officer concerns.
4. **Operating policy** sized to 40 reviews per quarter, with the cost trade-off computed, not asserted.
5. **Leakage-safe Indian case harness**: observation date = event date minus 4 / 8 quarters, filings used only if public by then.""")
    st.markdown("#### The most important finding")
    st.info(f"The raw 64-attribute model scores PR-AUC **{abl['all_64_attributes']}**, but one attribute's *missing-data pattern* (Attr27) carries about half of that. "
            f"Remove it and it falls to **{abl['all_64_minus_Attr27']}**; the India-portable production model reaches **{abl['india_portable_39']}** in 5-fold CV. "
            "We ship the honest model because the missingness shortcut cannot exist in Indian filings.")

# ------------------------------------------------------------------ Benchmark
with tabs[1]:
    st.caption(f"{meta['dataset']} · {meta['n_rows']:,} firms after removing {meta['n_duplicates_removed']} duplicates · bankrupt rate {meta['bankrupt_rate']:.1%} · "
               f"{meta['missing_cell_rate']:.1%} of cells missing ({meta['rows_with_any_missing']:.0%} of rows have at least one)")
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=pr.recall, y=pr.precision, mode="lines", name="production model", line=dict(width=3)))
        fig.add_hline(y=prod["pr_auc_baseline"], line_dash="dot", annotation_text=f"no-skill baseline {prod['pr_auc_baseline']:.3f}")
        fig.update_layout(title=f"Precision-recall, untouched hold-out (n={prod['n_test']}, {prod['test_positives']} bankrupt) · PR-AUC {prod['pr_auc']:.3f}",
                          xaxis_title="Recall", yaxis_title="Precision", height=380, margin=dict(t=60))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = go.Figure(go.Bar(x=list(abl.keys())[:-1], y=[abl[k] for k in list(abl)[:-1]], text=[abl[k] for k in list(abl)[:-1]], textposition="outside"))
        fig.update_layout(title="Where the skill comes from (5-fold OOF PR-AUC)", height=380, margin=dict(t=60), yaxis_title="PR-AUC")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, width="stretch")
        st.caption(abl["note"])
    st.markdown("**Expected cost on the hold-out** (missed default = 20 × false alarm)")
    rows = [("Flag nobody", 0, 0, prod["baseline_flag_none"]["expected_cost"]), ("Flag everyone", None, None, prod["baseline_flag_all"]["expected_cost"])]
    t = prod["at_bayes_threshold"]; cp = prod["at_5pct_cap"]
    tab = pd.DataFrame([
        dict(policy="Flag nobody", alerts="0", recall="0%", cost=prod["baseline_flag_none"]["expected_cost"]),
        dict(policy="Flag everyone", alerts=f"{prod['n_test']}", recall="100%", cost=prod["baseline_flag_all"]["expected_cost"]),
        dict(policy=f"Cost-optimal threshold p ≥ {t['threshold']:.4f} (= 1/21)", alerts=f"{t['tp'] + t['fp']} ({t['alert_rate']:.0%})", recall=f"{t['recall']:.0%}", cost=t["expected_cost"]),
        dict(policy="Top 5% only (budget cap)", alerts=f"{cp['k']} (5%)", recall=f"{cp['recall']:.0%}", cost=cp["expected_cost"]),
    ])
    st.dataframe(tab, hide_index=True, width="stretch")
    st.caption(f"Cost = 20 × missed + 1 × false alarms. Threshold 1/(1+20) is the Bayes-optimal cut-off on calibrated probabilities; the empirical OOF optimum was {prod['empirical_threshold_oof']}. "
               "Split limits: " + meta["no_calendar_time"])
    with st.expander("Leakage controls & horizon note"):
        st.write(meta["leakage_controls"]); st.write(meta["horizon_note"])
    with st.expander("Reference model (all 64 attributes, includes the Attr27 artefact)"):
        st.write(f"PR-AUC {ref['pr_auc']:.3f} (CI {ref['pr_auc_ci95'][0]:.2f}-{ref['pr_auc_ci95'][1]:.2f}). Shown for transparency only; not used for scoring.")

# ------------------------------------------------------------------ 5% budget
with tabs[2]:
    rate = st.radio("Default rate in the 800-borrower book", ["0.01", "0.02", "0.041"], index=1, horizontal=True, format_func=lambda x: f"{float(x):.1%}")
    s = sims[rate]
    c = st.columns(4)
    c[0].metric("Defaulters in book", s["n_pos"]); c[1].metric("Recall inside 40 names", f"{s['recall_mean']:.0%}", f"CI {s['recall_ci95'][0]:.0%}-{s['recall_ci95'][1]:.0%}")
    c[2].metric("Cost-minimising list size", s["k_at_cost_min"], f"budget 40"); c[3].metric("Names above 1/21 after re-basing", f"{s['mean_alerts_above_bayes']:.0f}")
    ks = np.arange(len(s["cost_curve"]))
    fig = go.Figure(go.Scatter(x=ks, y=s["cost_curve"], mode="lines", name="expected cost"))
    fig.add_vline(x=40, line_dash="dash", annotation_text="40-name budget")
    fig.add_vline(x=s["k_at_cost_min"], line_dash="dot", line_color="green", annotation_text=f"optimum k={s['k_at_cost_min']}")
    fig.add_hline(y=s["cost_flag_none"], line_dash="dot", line_color="red", annotation_text="flag nobody")
    fig.update_layout(title="Expected cost (20×missed + false alarms) vs number of names reviewed", xaxis_title="Names reviewed per quarter", yaxis_title="Expected cost", height=380)
    st.plotly_chart(fig, width="stretch")
    h = go.Figure(go.Histogram(x=s["recalls"] if "recalls" in s else [], nbinsx=20))
    h.update_layout(title="Recall inside the 40-name list across 1,000 simulated books", xaxis_title="Recall", height=300)
    st.plotly_chart(h, width="stretch")
    st.caption("Books are drawn from the untouched hold-out at the chosen default rate; scores are ranked and the top 40 reviewed. "
               "Probabilities are re-based from the benchmark's 3.9% default rate to the book's rate before comparing with the 1/21 threshold.")

# ------------------------------------------------------------------ Watch-list
with tabs[3]:
    st.caption("One simulated quarterly book of 800 anonymised benchmark borrowers (2% default rate). 'Outcome' is hindsight from the dataset label and is never a model input.")
    top = wl[wl.in_budget].copy()
    top["risk"] = (top["risk"] * 100).round(1)
    st.dataframe(top[["rank", "borrower", "risk", "defaulted_within_horizon_hindsight", "drivers"]].rename(columns={"risk": "risk %", "defaulted_within_horizon_hindsight": "outcome"}),
                 hide_index=True, width="stretch", height=420)
    caught = int(top.defaulted_within_horizon_hindsight.sum()); tot = int(wl.defaulted_within_horizon_hindsight.sum())
    st.success(f"This draw: {caught} of {tot} eventual defaulters sit inside the 40-name list.")
    pick = st.selectbox("Open memo", top.borrower)
    st.code(top[top.borrower == pick].memo.iloc[0], language=None)

# ------------------------------------------------------------------ Score a company
with tabs[4]:
    st.caption("Enter two consecutive fiscal years of statement items (same currency unit). Defaults are **illustrative placeholders, not a real company**.")
    ex_t = dict(total_assets=1000, current_assets=400, current_liabilities=420, total_liabilities=800, equity=200, retained_earnings=-50, sales=900, cogs=720, sga=90,
                depreciation=40, ebit=30, interest_expense=60, pbt=-30, net_income=-35, cfo=10, receivables=220, inventory=110, ppe_net=450, lt_debt=380, cash=20, shares_out=100)
    ex_p = dict(total_assets=980, current_assets=430, current_liabilities=350, total_liabilities=720, equity=260, retained_earnings=40, sales=1000, cogs=760, sga=85,
                depreciation=38, ebit=90, interest_expense=50, pbt=40, net_income=28, cfo=70, receivables=170, inventory=100, ppe_net=440, lt_debt=330, cash=40, shares_out=100)
    ed = st.data_editor(pd.DataFrame({"year t": ex_t, "year t-1": ex_p}), width="stretch", height=330)
    fin = st.checkbox("Bank / NBFC (Z'', M and F are not defined; model is indicative only)")
    st.write("Soft signals observed in the last 12 months:")
    sc = st.columns(len(SOFT_SIGNALS)); soft = {k: int(sc[i].checkbox(SOFT_LABELS[k].split(" in ")[0].split(" above")[0][:28], key=k)) for i, k in enumerate(SOFT_SIGNALS)}
    if st.button("Score", type="primary"):
        cur, prev = ed["year t"].astype(float).to_dict(), ed["year t-1"].astype(float).to_dict()
        feats = indian_to_portable(cur, prev); X = pd.DataFrame([feats])[FEATURE_NAMES]
        b = load_bundle("production"); p = predict_proba(b, X)
        z = np.nan if fin else altman_z(cur["current_assets"] - cur["current_liabilities"], cur["total_assets"], cur["retained_earnings"], cur["ebit"], cur["equity"], cur["total_liabilities"])
        m = np.nan if fin else beneish_m(cur, prev); f = np.nan if fin else piotroski_f(cur, prev)
        risk = apply_overlay(p, z_zone=[altman_zone(z)], m_flag=[beneish_flag(m)], f_band=[f_band(f)], icr=[feats["int_cover"]], soft=pd.DataFrame([soft]))[0]
        drv = top_drivers(b, X)[0]
        c = st.columns(5)
        c[0].metric("Model probability", f"{p[0]:.1%}"); c[1].metric("Final risk (with overlay)", f"{risk:.1%}")
        c[2].metric("Altman Z''", "n/a" if fin else f"{z:.2f} ({altman_zone(z)})"); c[3].metric("Beneish M", "n/a" if fin else f"{m:.2f}" if np.isfinite(m) else "n/a")
        c[4].metric("Piotroski F", "n/a" if fin or not np.isfinite(f) else f"{int(f)}/9")
        st.code(memo_text("Company", 0, risk, drv, z=z, m=m, f=f, soft=[k for k, v in soft.items() if v]).replace("Rank #0 | ", ""), language=None)

# ------------------------------------------------------------------ Indian case
with tabs[5]:
    st.progress(min(1.0, (status["defaulters_with_data"] + status["peers_with_data"]) / (indian.MIN_DEFAULTERS + indian.MIN_PEERS)),
                text=f"Panel: {status['defaulters_with_data']}/{indian.MIN_DEFAULTERS} defaulters · {status['peers_with_data']}/{indian.MIN_PEERS} peers with ≥2 usable fiscal years")
    ev = indian.load_events()
    st.markdown("**Candidate defaulters** (dates are *unverified* until you confirm them against IBBI / NCLT / exchange filings)")
    st.dataframe(ev[["company", "sector", "tier", "event_date", "event_type", "date_status", "note"]], hide_index=True, width="stretch", height=300)
    summ = ART / "indian_summary.json"
    if status["ready"] and summ.exists():
        res = json.loads(summ.read_text()); sc = pd.read_csv(ART / "indian_scored.csv")
        st.markdown("#### Recall of Indian defaulters inside the 40-of-800 budget")
        st.dataframe(pd.DataFrame(res).T, width="stretch")
        lead = st.radio("Lead time", [4, 8], horizontal=True, format_func=lambda q: f"{q} quarters before event")
        d = sc[(sc.lead_q == lead) & (sc.is_defaulter == 1)]
        st.dataframe(d[["company", "obs_date", "median_rank_full", "median_rank_model_only", "p_in_top40", "altman_z", "beneish_m", "piotroski_f", "drivers"]], hide_index=True, width="stretch")
    else:
        st.markdown('<div class="warn">No Indian results yet. Fill <code>data/indian/financials.csv</code> (+ optional <code>soft_signals.csv</code>), '
                    'then run <code>python run_all.py</code>. Results appear here automatically.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Policy
with tabs[6]:
    st.markdown((ROOT / "policy.md").read_text())
