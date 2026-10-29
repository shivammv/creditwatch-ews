import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import indian
from src.features import RAW_COLS, FEATURE_NAMES, indian_to_portable
from src.scores import altman_z, beneish_m, piotroski_f
from src.model import BAYES_T, COST_FN, COST_FP
from src.rank import simulate_universes

def raw(scale=1.0, **kw):
    d = dict(total_assets=1000, current_assets=400, current_liabilities=300, total_liabilities=600, equity=400,
             retained_earnings=200, sales=1200, cogs=800, sga=100, depreciation=50, ebit=150, interest_expense=40,
             pbt=110, net_income=80, cfo=120, receivables=150, inventory=100, ppe_net=450, lt_debt=250, cash=60, shares_out=100)
    d.update(kw); return d

def test_bayes_threshold_is_one_over_twentyone():
    assert abs(BAYES_T - COST_FP / (COST_FP + COST_FN)) < 1e-12

def test_portable_ratios_by_hand():
    f = indian_to_portable(raw(), raw(sales=1000))
    assert f["liabilities_ta"] == pytest.approx(0.6) and f["current_ratio"] == pytest.approx(400 / 300)
    assert f["int_cover"] == pytest.approx(150 / 40) and f["sales_growth"] == pytest.approx(1.2)
    assert set(FEATURE_NAMES) <= set(f) and "int_cover" not in FEATURE_NAMES

def test_altman_and_beneish_reference_values():
    z = altman_z(100, 1000, 200, 150, 400, 600)
    assert z == pytest.approx(6.56 * 0.1 + 3.26 * 0.2 + 6.72 * 0.15 + 1.05 * 400 / 600)
    # identical years => all indices = 1; TATA = (80-120)/1000 = -0.04
    assert beneish_m(raw(), raw()) == pytest.approx(-2.48 + 4.679 * -0.04, abs=1e-3)
    assert np.isfinite(piotroski_f(raw(), raw(net_income=60)))

def _synthetic(tmp):
    """SYNTHETIC panel used ONLY to prove the code path and leakage guard; never shipped as results."""
    rng = np.random.default_rng(0)
    ev = indian.load_events(); ev = ev[(ev.tier == "core") & ev.event_date.notna()]
    peers = indian.load_peers(); rows = []
    for name, evd, bad in [(r.company, r.event_date, 1) for _, r in ev.iterrows()] + [(p.peer, ev[ev.company == p.matched_to].event_date.iloc[0], 0) for _, p in peers.iterrows()]:
        for k in range(4):
            fy = pd.Timestamp(evd.year - 4 + k + (1 if evd.month > 7 else 0), 3, 31)
            decay = (k / 3) * bad
            rows.append(dict(company=name, fy_end=fy, public_date=fy + pd.Timedelta(days=120),
                             **raw(ebit=150 * (1 - .9 * decay), pbt=110 * (1 - 1.2 * decay), net_income=80 * (1 - 1.3 * decay),
                                   total_liabilities=600 + 250 * decay, equity=400 - 250 * decay, retained_earnings=200 - 300 * decay,
                                   interest_expense=40 + 40 * decay, sales=1200 * (1 + rng.normal(0, .03)))))
    return pd.DataFrame(rows)

@pytest.fixture
def no_synthetic_outputs():
    """The synthetic panel must never leave results behind in artifacts/."""
    yield
    for f in ("indian_scored.csv", "indian_summary.json"):
        (indian.ART / f).unlink(missing_ok=True)

def test_end_to_end_and_leakage_guard(tmp_path, monkeypatch, no_synthetic_outputs):
    fin = _synthetic(tmp_path)
    monkeypatch.setattr(indian, "load_financials", lambda: fin)
    out = indian.evaluate(draws=30)
    assert "lead_4q" in out and out["lead_4q"]["n_defaulters"] >= 8
    obs = pd.read_csv(indian.ART / "indian_scored.csv", parse_dates=["obs_date", "public_date_used"])
    assert (obs.public_date_used <= obs.obs_date).all()          # no filing after the observation date
    # a filing published AFTER the observation date must never be picked up
    e = indian.load_events().iloc[0]
    o = indian.obs_date(e.event_date, 4)
    fut = fin.copy(); fut.loc[fut.company == e.company, "public_date"] = o + pd.Timedelta(days=1)
    assert indian.build_observation(e.company, o, fut, indian.load_soft(), True) is None

def test_alert_budget_never_exceeded():
    h = pd.read_csv(indian.ART / "holdout_predictions.csv")
    r = simulate_universes(h.p.values, h.y.values, n=800, rate=0.02, budget=40, draws=50)
    assert r["budget"] == 40 and 0 <= r["recall_mean"] <= 1
