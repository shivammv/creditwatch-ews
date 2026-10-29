"""One command to reproduce every number in the deck:  python run_all.py   (add --retrain to refit models)"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, polish_to_portable
from src.model import ART, BAYES_T, train_lead_models
from src.rank import load_bundle, memo_text, simulate_universes, top_drivers
from src import horizons, indian


# Book default rates for the 800-borrower simulation. 0.02 is the central case; 0.041 is CRISIL's all-issuer one-year
# default rate for 2017 (CRISIL Default Study 2017), an upper bound for a listed-corporate book.
BOOK_RATES = (0.01, 0.02, 0.041)


def build_sim(seed: int = 7) -> dict:
    out = {}
    for suf in ("", "_8q"):
        h = pd.read_csv(ART / f"holdout_predictions{suf}.csv")
        sims = {}
        for rate in BOOK_RATES:
            sims[str(rate)] = simulate_universes(h.p.values, h.y.values, n=800, rate=rate, budget=40, draws=1000, seed=seed,
                                                 bayes_t=BAYES_T, train_rate=float(h.y.mean()))
        (ART / f"universe_sim{suf}.json").write_text(json.dumps(sims, indent=1))
        out[suf or "_4q"] = sims
    return out["_4q"]


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
    if a.retrain or not (ART / "model_production.joblib").exists() or not (ART / "model_production_8q.joblib").exists():
        train_lead_models()
    rep = json.loads((ART / "benchmark_report.json").read_text())
    rep8 = json.loads((ART / "benchmark_report_8q.json").read_text())
    print("PR-AUC 4Q production:", round(rep["production"]["pr_auc"], 3), "(baseline", rep["production"]["pr_auc_baseline"], ")",
          "| 8Q:", round(rep8["production"]["pr_auc"], 3), "(baseline", rep8["production"]["pr_auc_baseline"], ")",
          "| reference-64 native-NaN:", round(rep["reference"]["pr_auc"], 3))
    sims = build_sim()
    for k, v in sims.items():
        print(f"universe 800, default rate {float(k):.1%}: recall@40 = {v['recall_mean']:.2f}  CI95 {v['recall_ci95'][0]:.2f}-{v['recall_ci95'][1]:.2f}")
    sample_watchlist()
    hc = horizons.horizon_curve()
    print('Horizon files found:', hc['files_found'], '-> artifacts/horizon_curve.json')
    st = indian.panel_status()
    print("Indian panel:", st)
    if st["ready"]:
        res = indian.evaluate()          # verifies config/frozen_config.json first
        for lead, r in res.items():
            pc = r["primary_peer_cutoff"]
            print(f"{lead}: recall above 95th-pct peer cutoff -> full {pc['full']['caught']}/{pc['full']['n_defaulters']} "
                  f"{pc['full']['recall_wilson95']}, model-only {pc['model_only']['caught']}/{pc['model_only']['n_defaulters']}, "
                  f"forensic-only {pc['forensic_only']['caught']}, soft-only {pc['soft_only']['caught']}")
        import subprocess
        subprocess.run([sys.executable, "score_any.py", "data/indian/financials.csv", "--soft", "data/indian/soft_signals.csv"], check=False)
    else:
        print("Indian panel not populated yet -> fill data/indian/financials.csv (see docs/DATA_ENTRY.md), then re-run.")
