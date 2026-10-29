"""One command to reproduce every number in the deck:  python run_all.py   (add --retrain to refit models)"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, polish_to_portable
from src.model import ART, BAYES_T, train_all
from src.rank import load_bundle, memo_text, simulate_universes, top_drivers
from src import horizons, indian


def build_sim(seed: int = 7) -> dict:
    h = pd.read_csv(ART / "holdout_predictions.csv")
    sims = {}
    for rate in (0.01, 0.02, 0.039):
        r = simulate_universes(h.p.values, h.y.values, n=800, rate=rate, budget=40, draws=1000, seed=seed, bayes_t=BAYES_T)
        sims[str(rate)] = r
    (ART / "universe_sim.json").write_text(json.dumps(sims, indent=1))
    return sims


def sample_watchlist(rate: float = 0.02, seed: int = 3) -> pd.DataFrame:
    """One illustrative 800-name quarterly book from the benchmark hold-out, ranked, top-40 with SHAP memos."""
    h = pd.read_csv(ART / "holdout_predictions.csv")
    df = pd.read_pickle(ART / "polish_dedup.pkl")
    rng = np.random.default_rng(seed)
    pos, neg = np.where(h.y.values == 1)[0], np.where(h.y.values == 0)[0]
    n_pos = int(round(rate * 800))
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, 800 - n_pos, replace=False)])
    book = h.iloc[idx].sort_values("p", ascending=False).reset_index(drop=True)
    book["rank"] = np.arange(1, len(book) + 1)
    X = polish_to_portable(df.loc[book.row])[FEATURE_NAMES]
    top = book.head(40)
    drv = top_drivers(load_bundle("production"), X.iloc[:40])
    book["borrower"] = [f"Borrower-{int(r):04d}" for r in book.row]
    book["drivers"] = ""
    book["memo"] = ""
    book["in_budget"] = book["rank"] <= 40
    for i, d in enumerate(drv):
        book.loc[i, "drivers"] = " | ".join(x["text"] for x in d)
        book.loc[i, "memo"] = memo_text(book.borrower[i], i + 1, book.p[i], d)
    book = book.rename(columns={"p": "risk", "y": "defaulted_within_horizon_hindsight"})
    book.to_csv(ART / "sample_watchlist.csv", index=False)
    return book


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    if a.retrain or not (ART / "model_production.joblib").exists():
        train_all()
    rep = json.loads((ART / "benchmark_report.json").read_text())
    print("PR-AUC production:", round(rep["production"]["pr_auc"], 3), "| reference-64:", round(rep["reference"]["pr_auc"], 3))
    sims = build_sim()
    for k, v in sims.items():
        print(f"universe 800, default rate {float(k):.1%}: recall@40 = {v['recall_mean']:.2f}  CI95 {v['recall_ci95'][0]:.2f}-{v['recall_ci95'][1]:.2f}")
    sample_watchlist()
    hc = horizons.horizon_curve()
    print('Horizon files found:', hc['files_found'], '-> artifacts/horizon_curve.json')
    st = indian.panel_status()
    print("Indian panel:", st)
    if st["ready"]:
        print(json.dumps(indian.evaluate(), indent=2, default=float))
    else:
        print("Indian panel not populated yet -> fill data/indian/financials.csv (see docs/DATA_ENTRY.md), then re-run.")
