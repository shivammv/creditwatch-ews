"""
Lead-time curve across the five UCI Polish horizon files.

UCI naming: file k = ratios from forecasting year k, label = bankrupt within (6-k) years of that data  (1year.arff -> 5y lead, 5year.arff -> 1y lead).
Drop `polish_{k}year.csv` (columns Attr1..Attr64 + class) into data/polish/, or a combined `polish_all.csv` with a `horizon_years` column
(= file index 1..5) into data/ or data/polish/; this module splits it automatically.

Two evaluations per file, both on the India-portable features:
  * within-file: the leakage-safe 80/20 hold-out from model._fit_eval (PR-AUC, recall at the 5% cap)
  * cross-file (train on file a, test on file b): reported with a WARNING, because the five files are drawn from the same
    pool of companies in different years, so firms can recur and cross-file is NOT company-disjoint without firm IDs.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from .features import FEATURE_NAMES, polish_to_portable
from .model import ART, DATA, LGB_PARAMS, _fit_eval, load_polish

LEAD_YEARS = {k: 6 - k for k in range(1, 6)}


def split_polish_all() -> list[int]:
    for p in (DATA / "polish_all.csv", DATA.parent / "polish_all.csv"):
        if p.exists():
            raw = pd.read_csv(p)
            col = "horizon_years"
            for k in sorted(raw[col].unique()):
                out = DATA / f"polish_{int(k)}year.csv"
                if not out.exists():
                    raw[raw[col] == k].drop(columns=[col]).to_csv(out, index=False)
    return available()


def available() -> list[int]:
    return [k for k in range(1, 6) if (DATA / f"polish_{k}year.csv").exists()]


def _recall_at_budget(p, y, frac=0.05):
    k = int(np.ceil(frac * len(y)))
    return float(y[np.argsort(-p)[:k]].sum() / max(y.sum(), 1))


def horizon_curve() -> dict:
    ks = split_polish_all()
    within, data = {}, {}
    for k in ks:
        df = load_polish(horizon=k)
        y = df["bankrupt"].astype(int).values
        X = polish_to_portable(df)[FEATURE_NAMES]
        data[k] = (X, y)
        rep, _ = _fit_eval(X, y, f"horizon_{k}")
        within[k] = dict(file=f"polish_{k}year.csv", lead_years=LEAD_YEARS[k], n=len(y), bankrupt_rate=round(float(y.mean()), 4),
                         pr_auc=rep["pr_auc"], pr_auc_ci95=rep["pr_auc_ci95"], pr_auc_baseline=rep["pr_auc_baseline"],
                         recall_at_5pct=rep["at_5pct_cap"]["recall"])
    cross = []
    for a in ks:
        for b in ks:
            if a == b:
                continue
            m = LGBMClassifier(**LGB_PARAMS).fit(*data[a])
            p = m.predict_proba(data[b][0])[:, 1]
            cross.append(dict(train_file=a, test_file=b, train_lead_years=LEAD_YEARS[a], test_lead_years=LEAD_YEARS[b],
                              pr_auc=float(average_precision_score(data[b][1], p)), baseline=float(data[b][1].mean()),
                              recall_at_5pct=_recall_at_budget(p, data[b][1])))
    out = dict(files_found=ks, within_file=within, cross_file=cross,
               warning="Cross-file rows are NOT company-disjoint (same firms recur across horizon files); treat as optimistic.")
    (ART / "horizon_curve.json").write_text(json.dumps(out, indent=1))
    return out
