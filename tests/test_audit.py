"""Unit + integration tests added in the final audit (data provenance, leakage, pre-registration, bug regressions)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, indian, rank
from src.features import FEATURE_NAMES, indian_to_portable
from src.rank import load_bundle, predict_proba
from src.scores import altman_z, altman_zone, beneish_m, piotroski_f

ROOT = Path(__file__).resolve().parents[1]
IND = ROOT / "data" / "indian"
ART = ROOT / "artifacts"


# ----------------------------------------------------------------------------- data provenance (unit)
def test_core_events_are_sourced_and_verified():
    ev = pd.read_csv(IND / "events.csv")
    core = ev[ev.tier == "core"]
    assert len(core) >= 10
    assert core.source_url.str.startswith("http").all()
    assert (core.date_status == "verified").all()
    assert pd.to_datetime(core.event_date, errors="coerce").notna().all()
    assert "IL&FS" not in set(core.company)          # unlisted parent excluded; listed ITNL used instead


def test_financial_rows_are_sourced_and_balance():
    f = pd.read_csv(IND / "financials.csv")
    assert f.source_url.notna().all() and (f.source_url.str.len() > 10).all()
    gap = (f.equity + f.total_liabilities - f.total_assets).abs()
    assert (gap <= 1.5).all(), "equity + liabilities must equal total assets"
    assert not f.duplicated(["company", "fy_end"]).any()
    assert "IL&FS" not in set(f.company)              # reconstructed parent rows removed
    removed = pd.read_csv(IND / "removed_reconstructed_rows.csv")
    for co in removed.company.unique():               # reconstructed rows never re-enter the panel
        if co in set(f.company):
            assert not f[f.company == co].fy_end.isin(removed[removed.company == co].fy_end).all() or \
                   f[f.company == co].source_url.str.contains("screener", case=False).all()


def test_screener_transcriptions_add_up():
    for p in (IND / "screener_raw").glob("*.csv"):
        t = pd.read_csv(p).set_index("line").T.astype(float)
        lhs = t.equity_capital + t.reserves + t.borrowings + t.other_liabilities
        assert ((lhs - t.total_liabilities_and_equity).abs() <= 1.5).all(), p.name
        assert ((t.total_assets - t.total_liabilities_and_equity).abs() <= 0.5).all(), p.name
        a = t.fixed_assets + t.cwip + t.investments + t.other_assets
        assert ((a - t.total_assets).abs() <= 1.5).all(), p.name


def test_peers_listed_and_have_data():
    peers = pd.read_csv(IND / "peers.csv")
    f = pd.read_csv(IND / "financials.csv")
    assert len(peers) >= 30
    assert "Devyani International" not in set(peers.peer)   # IPO Aug 2021, after its matched event
    assert set(peers.peer) <= set(f.company)
    ev = pd.read_csv(IND / "events.csv")
    assert set(peers.matched_to) <= set(ev[ev.tier == "core"].company)


def test_soft_signals_verified_and_before_event():
    s = pd.read_csv(IND / "soft_signals.csv", parse_dates=["date_public"])
    ev = pd.read_csv(IND / "events.csv", parse_dates=["event_date"]).set_index("company")
    assert (s.verified == 1).all() and s.source_url.str.startswith("http").all()
    for _, r in s.iterrows():
        assert r.date_public < ev.loc[r.company, "event_date"], f"{r.company} signal not before its event"


# ----------------------------------------------------------------------------- bug regressions (unit)
def test_missing_items_are_not_zero_filled():
    cur = dict(total_assets=1000, current_assets=400, current_liabilities=300, total_liabilities=600, equity=400,
               retained_earnings=200, sales=1200, ebit=150, interest_expense=40, pbt=110, net_income=80, depreciation=50)
    f = indian_to_portable(cur, None)                  # no cogs, sga, inventory, receivables, cash
    for k in ("gross_margin", "quick_ratio", "inv_days_sales", "cash_like_ratio", "cost_sales", "netdebt_sales", "recv_days"):
        assert np.isnan(f[k]), f"{k} must be NaN when its input is missing, got {f[k]}"
    assert f["current_ratio"] == pytest.approx(400 / 300)


def test_eighteen_month_period_is_annualised():
    df = pd.DataFrame(dict(company=["A", "A"], fy_end=pd.to_datetime(["2012-03-31", "2013-09-30"]),
                           sales=[1000.0, 1800.0], net_income=[10.0, 30.0], total_assets=[500.0, 600.0]))
    out = indian.annualise(df)
    r = out.iloc[1]
    assert r.period_months == 18 and bool(r.annualised)
    assert r.sales == pytest.approx(1200.0) and r.net_income == pytest.approx(20.0)
    assert r.total_assets == 600.0                     # stocks are not scaled
    assert out.iloc[0].sales == 1000.0


def test_imputer_makes_missingness_uninformative():
    b = load_bundle("production")
    med = dict(zip(b["features"], b["imputer"].statistics_))
    base = pd.DataFrame([med])
    for f in FEATURE_NAMES[:10]:
        miss = base.copy()
        miss[f] = np.nan
        assert predict_proba(b, miss)[0] == pytest.approx(predict_proba(b, base)[0], abs=1e-12)


def test_lead_models_match_their_horizon():
    b4, b8 = load_bundle("production"), load_bundle("production_8q")
    assert (b4["lead"], b4["horizon_file"]) == ("4q", 5)      # 5year.arff: bankrupt within 1 year
    assert (b8["lead"], b8["horizon_file"]) == ("8q", 4)      # 4year.arff: bankrupt within 2 years
    r4 = json.loads((ART / "benchmark_report.json").read_text())
    r8 = json.loads((ART / "benchmark_report_8q.json").read_text())
    assert r4["meta"]["label_horizon_years"] == 1 and r8["meta"]["label_horizon_years"] == 2
    for r in (r4, r8):
        p = r["production"]
        assert p["pr_auc"] > 3 * p["pr_auc_baseline"]           # real skill on the untouched hold-out


# ----------------------------------------------------------------------------- hand-calculated scores (unit)
Y0 = dict(sales=1000, cogs=600, sga=200, depreciation=50, ebit=150, interest_expense=50, net_income=70, total_assets=2000,
          current_assets=800, current_liabilities=500, receivables=200, ppe_net=1000, lt_debt=600, equity=800,
          retained_earnings=600, cfo=120, shares_out=10, total_liabilities=1200)
Y1 = dict(sales=1200, cogs=780, sga=220, depreciation=55, ebit=120, interest_expense=80, net_income=30, total_assets=2400,
          current_assets=900, current_liabilities=700, receivables=330, ppe_net=1150, lt_debt=800, equity=820,
          retained_earnings=620, cfo=40, shares_out=12, total_liabilities=1580)


def test_altman_z2_by_hand():
    z = altman_z(900 - 700, 2400, 620, 120, 820, 1580)
    assert z == pytest.approx(6.56 * 200 / 2400 + 3.26 * 620 / 2400 + 6.72 * 120 / 2400 + 1.05 * 820 / 1580)
    assert z == pytest.approx(2.26977, abs=1e-4) and altman_zone(z) == "grey"


def test_beneish_by_hand():
    dsri = (330 / 1200) / (200 / 1000); gmi = 0.40 / 0.35; aqi = (1 - 2050 / 2400) / (1 - 1800 / 2000)
    sgi = 1.2; depi = (50 / 1050) / (55 / 1205); sgai = (220 / 1200) / (200 / 1000)
    lvgi = (1500 / 2400) / (1100 / 2000); tata = (30 - 40) / 2400
    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    assert beneish_m(Y1, Y0) == pytest.approx(m, abs=1e-9)
    assert beneish_m(Y1, Y0) == pytest.approx(-1.74078, abs=1e-4)


def test_piotroski_by_hand():
    # ROA>0 1; CFO>0 1; ROA up 0; CFO>NI 1; leverage down 0; current ratio up 0; no dilution 0; GM up 0; turnover up 0
    assert piotroski_f(Y1, Y0) == 3


# ----------------------------------------------------------------------------- pre-registration (unit)
def test_frozen_config_verifies():
    assert len(config.verify_frozen()) == 64


def test_frozen_config_detects_tampering(monkeypatch):
    monkeypatch.setitem(rank.OVERLAY, "z_distress", 9.9)
    with pytest.raises(RuntimeError):
        config.verify_frozen()


def test_wilson_and_peer_cutoff_math():
    assert indian.wilson(0, 10) == [0.0, 0.278]
    lo, hi = indian.wilson(5, 10)
    assert lo == pytest.approx(0.237, abs=1e-3) and hi == pytest.approx(0.763, abs=1e-3)
    d = pd.DataFrame(dict(is_defaulter=[0] * 20 + [1] * 4, s=list(np.linspace(0, 1, 20)) + [2.0, 2.0, 0.1, 0.0]))
    r = indian.peer_cutoff_recall(d, "s")
    assert r["caught"] == 2 and r["n_defaulters"] == 4 and r["peers_flagged"] == 1


# ----------------------------------------------------------------------------- integration: real Indian evaluation
@pytest.fixture(scope="module")
def scored():
    p = ART / "indian_scored.csv"
    if not p.exists():
        pytest.skip("run `python run_all.py` first")
    return pd.read_csv(p, parse_dates=["obs_date", "public_date_used", "fy_used", "event_date"])


def test_real_panel_point_in_time(scored):
    assert (scored.public_date_used <= scored.obs_date).all()
    assert (scored.fy_used + pd.Timedelta(days=indian.PUBLICATION_LAG_DAYS) <= scored.obs_date).all()
    assert (scored.obs_date < scored.event_date).all()
    # peers are observed on the same calendar date as their matched defaulter
    for (lead, grp), g in scored.groupby(["lead_q", "matched_to"]):
        assert g.obs_date.nunique() == 1


def test_real_soft_signals_point_in_time(scored):
    soft = indian.load_soft()
    for _, r in scored[scored.rating_downgrade == 1].iterrows():
        s = soft[(soft.company == r.company) & (soft.signal == "rating_downgrade")]
        assert ((s.date_public <= r.obs_date) & (s.date_public > r.obs_date - pd.Timedelta(days=365))).any()


def test_summary_matches_scored_rows(scored):
    summ = json.loads((ART / "indian_summary.json").read_text())
    for lead in (4, 8):
        d = scored[scored.lead_q == lead]
        r = indian.peer_cutoff_recall(d, "risk")
        s = summ[f"lead_{lead}q"]["primary_peer_cutoff"]["full"]
        assert (r["caught"], r["n_defaulters"], r["cutoff"]) == (s["caught"], s["n_defaulters"], s["cutoff"])
        assert len(d[d.is_defaulter == 1]) == summ[f"lead_{lead}q"]["n_defaulters"]


def test_evaluation_refuses_unfrozen_changes(monkeypatch):
    monkeypatch.setitem(rank.OVERLAY, "rating_downgrade", 5.0)
    with pytest.raises(RuntimeError):
        indian.evaluate(draws=5)


def test_watchlist_respects_budget():
    w = pd.read_csv(ART / "watchlist_any.csv")
    if "in_budget" in w:
        assert w.in_budget.sum() <= int(np.ceil(0.05 * len(w)))
