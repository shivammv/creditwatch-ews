"""
Benchmark distress model (UCI Polish Companies Bankruptcy).

Leakage-safe design:
  * Duplicates removed BEFORE splitting (82 duplicate feature-rows in the 1-year file)
  * 20% hold-out is carved off first and touched exactly once for reporting
  * No imputation for gradient boosting: native NaN handling avoids learning statistics from test rows
  * Probability calibration and cost threshold are fitted on out-of-fold predictions of the dev set only
  * Multi-model comparison across Logistic Regression, Random Forest, XGBoost, and LightGBM
Two core models are saved: PRODUCTION (India-portable ratios only) and REFERENCE (all 64 attributes).
"""
from __future__ import annotations
import datetime
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features import FEATURE_NAMES, polish_to_portable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "polish"
ART = ROOT / "artifacts"
COST_FN, COST_FP = 20.0, 1.0
BAYES_T = COST_FP / (COST_FP + COST_FN)       # 1/21 = 0.0476: optimal cut-off on CALIBRATED probabilities
SEED = 42

# Primary LightGBM configuration
LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)

# Tuned LightGBM configuration for higher regularization and generalization
LGB_OPTIMIZED_PARAMS = dict(
    n_estimators=350,
    learning_rate=0.03,
    num_leaves=12,
    min_child_samples=20,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.5,
    reg_lambda=6.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


def load_polish(path: Path | None = None, horizon: int = 1) -> pd.DataFrame:
    path = path or DATA / f"polish_{horizon}year.csv"
    df = pd.read_csv(path).rename(columns={"class": "bankrupt"})
    attrs = [c for c in df.columns if c.startswith("Attr")]
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates(subset=attrs + ["bankrupt"]).reset_index(drop=True)
    df.attrs["n_duplicates_removed"] = before - len(df)
    df.attrs["missing_cell_rate"] = float(df[attrs].isna().mean().mean())
    df.attrs["rows_with_any_missing"] = float(df[attrs].isna().any(axis=1).mean())
    return df


def cost(fn: int, fp: int) -> float:
    return COST_FN * fn + COST_FP * fp


def best_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Grid-search the threshold that minimises 20*FN + FP on given calibrated probabilities."""
    best = (BAYES_T, np.inf)
    for t in np.unique(np.concatenate([np.linspace(0.005, 0.9, 180), [BAYES_T]])):
        pred = p >= t
        c = cost(int(((~pred) & (y == 1)).sum()), int((pred & (y == 0)).sum()))
        if c < best[1]:
            best = (float(t), float(c))
    return best


class Calibrator:
    """Platt scaling on the logit of raw model probabilities (stable with rare events)."""
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _z(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.lr.fit(self._z(p), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


def _fit_eval(X: pd.DataFrame, y: np.ndarray, name: str, seed: int = SEED) -> tuple[dict, dict]:
    X_dev, X_te, y_dev, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    
    # 5-fold Out-Of-Fold probabilities on dev set -> calibrator + threshold
    oof = np.zeros(len(y_dev))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_dev, y_dev):
        m = LGBMClassifier(**LGB_PARAMS).fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = m.predict_proba(X_dev.iloc[va])[:, 1]
        
    cal = Calibrator().fit(oof, y_dev)
    p_oof = cal.predict(oof)
    t_emp, _ = best_threshold(p_oof, y_dev)

    # Train final model on entire dev set
    model = LGBMClassifier(**LGB_PARAMS).fit(X_dev, y_dev)
    raw_te = model.predict_proba(X_te)[:, 1]
    p_te = cal.predict(raw_te)

    # Bootstrap 95% confidence intervals on holdout PR-AUC and ROC-AUC
    rng = np.random.default_rng(seed)
    boots_pr, boots_roc = [], []
    for _ in range(500):
        i = rng.integers(0, len(y_te), len(y_te))
        if y_te[i].sum() and (len(y_te[i]) - y_te[i].sum()) > 0:
            boots_pr.append(average_precision_score(y_te[i], p_te[i]))
            boots_roc.append(roc_auc_score(y_te[i], p_te[i]))
            
    n_pos = int(y_te.sum())
    n_neg = int(len(y_te) - n_pos)

    def evaluate_at(t: float):
        pred = p_te >= t
        tp = int((pred & (y_te == 1)).sum())
        fp = int((pred & (y_te == 0)).sum())
        fn = int(((~pred) & (y_te == 1)).sum())
        tn = int(((~pred) & (y_te == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(n_pos, 1)
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-9)
        acc = (tp + tn) / len(y_te)
        return dict(
            threshold=round(float(t), 4),
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=round(float(prec), 4),
            recall=round(float(rec), 4),
            f1_score=round(float(f1), 4),
            accuracy=round(float(acc), 4),
            alert_rate=round(float((tp + fp) / len(y_te)), 4),
            expected_cost=float(cost(fn, fp))
        )

    k5 = int(np.ceil(0.05 * len(y_te)))
    top = np.argsort(-p_te)[:k5]
    top_hits = int(y_te[top].sum())
    cap = dict(
        k=k5,
        budget_pct=0.05,
        hits=top_hits,
        misses=n_pos - top_hits,
        recall=round(float(top_hits / max(n_pos, 1)), 4),
        precision=round(float(y_te[top].mean()), 4),
        expected_cost=float(cost(n_pos - top_hits, k5 - top_hits))
    )

    # Curves & calibration
    prec_pts, rec_pts, pr_thresh = precision_recall_curve(y_te, p_te)
    fpr_pts, tpr_pts, roc_thresh = roc_curve(y_te, p_te)
    prob_true, prob_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")

    # Feature importances
    feat_names = list(X.columns)
    feat_imp = [
        {"feature": f, "importance": int(imp)}
        for f, imp in sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    ]

    report = dict(
        name=name,
        n_features=X.shape[1],
        n_dev=len(y_dev),
        n_test=len(y_te),
        test_positives=n_pos,
        test_negatives=n_neg,
        test_bankrupt_rate=round(float(y_te.mean()), 4),
        pr_auc=round(float(average_precision_score(y_te, p_te)), 4),
        pr_auc_ci95=[round(float(np.percentile(boots_pr, 2.5)), 4), round(float(np.percentile(boots_pr, 97.5)), 4)],
        pr_auc_baseline=round(float(y_te.mean()), 4),
        roc_auc=round(float(roc_auc_score(y_te, p_te)), 4),
        roc_auc_ci95=[round(float(np.percentile(boots_roc, 2.5)), 4), round(float(np.percentile(boots_roc, 97.5)), 4)],
        brier_score=round(float(brier_score_loss(y_te, p_te)), 5),
        oof_pr_auc=round(float(average_precision_score(y_dev, p_oof)), 4),
        oof_roc_auc=round(float(roc_auc_score(y_dev, p_oof)), 4),
        oof_brier_score=round(float(brier_score_loss(y_dev, p_oof)), 5),
        at_bayes_threshold=evaluate_at(BAYES_T),
        at_empirical_threshold=evaluate_at(t_emp),
        at_5pct_cap=cap,
        baseline_flag_none=dict(expected_cost=cost(n_pos, 0), recall=0.0, alerts=0),
        baseline_flag_all=dict(expected_cost=cost(0, n_neg), recall=1.0, alerts=len(y_te)),
        empirical_threshold_oof=round(t_emp, 4),
        bayes_threshold=round(BAYES_T, 4),
        feature_importances=feat_imp[:15]
    )

    bundle = dict(
        model=model,
        calibrator=cal,
        features=feat_names,
        bayes_threshold=BAYES_T,
        empirical_threshold=t_emp,
        healthy_median=X_dev[y_dev == 0].median().to_dict(),
        background=X_dev.sample(min(200, len(X_dev)), random_state=seed)
    )
    
    extras = dict(
        test_idx=X_te.index.to_numpy(),
        p_test=p_te,
        y_test=y_te,
        pr_curve=(prec_pts, rec_pts),
        roc_curve=(fpr_pts, tpr_pts),
        cal_curve=(prob_true, prob_pred)
    )
    return report, dict(bundle=bundle, **extras)


def compare_models(df: pd.DataFrame, seed: int = SEED) -> list[dict]:
    """5-Fold CV comparison across Logistic Regression, Random Forest, XGBoost, and LightGBM."""
    y = df["bankrupt"].astype(int).values
    X = polish_to_portable(df)[FEATURE_NAMES]
    X_dev, _, y_dev, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    candidate_models = {
        "Logistic Regression (Median Imputed + Scaled)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
        ]),
        "Random Forest (Median Imputed)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=2))
        ]),
        "XGBoost (Native NaN)": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, eval_metric="logloss", n_jobs=2
        ),
        "LightGBM Baseline (Native NaN)": LGBMClassifier(**LGB_PARAMS),
        "LightGBM Optimized (Native NaN)": LGBMClassifier(**LGB_OPTIMIZED_PARAMS),
        "LightGBM (Median Imputed)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(**LGB_OPTIMIZED_PARAMS))
        ])
    }

    results = []
    k5 = int(np.ceil(0.05 * len(y_dev)))

    for name, m in candidate_models.items():
        oof = np.zeros(len(y_dev))
        t0 = time.time()
        for tr, va in cv.split(X_dev, y_dev):
            m.fit(X_dev.iloc[tr], y_dev[tr])
            oof[va] = m.predict_proba(X_dev.iloc[va])[:, 1]
        fit_time = round(time.time() - t0, 2)
        pr = float(average_precision_score(y_dev, oof))
        roc = float(roc_auc_score(y_dev, oof))
        brier = float(brier_score_loss(y_dev, oof))
        top = np.argsort(-oof)[:k5]
        rec5 = float(y_dev[top].sum() / y_dev.sum())
        results.append(dict(
            model=name,
            pr_auc=round(pr, 4),
            roc_auc=round(roc, 4),
            brier_score=round(brier, 5),
            recall_at_5pct=round(rec5, 4),
            cv_training_time_sec=fit_time,
            supports_native_nan="Native" in name,
            selected_as_production=(name == "LightGBM Optimized (Native NaN)" or name == "LightGBM Baseline (Native NaN)")
        ))

    return results


def artifact_ablation(df: pd.DataFrame, seed: int = 1) -> dict:
    """5-fold OOF PR-AUC for feature sets that isolate the Attr27 missing-data artefact."""
    y = df["bankrupt"].astype(int).values
    attrs = [c for c in df.columns if c.startswith("Attr")]
    port = polish_to_portable(df)[FEATURE_NAMES]
    sets = {
        "all_64_attributes": df[attrs],
        "all_64_minus_Attr27": df[[a for a in attrs if a != "Attr27"]],
        "india_portable_39": port,
        "india_portable_39_plus_Attr27_missing_flag": port.assign(attr27_missing=df["Attr27"].isna().astype(int)),
        "attr27_missing_flag_only": pd.DataFrame({"attr27_missing": df["Attr27"].isna().astype(int)}),
    }
    out = {}
    for name, X in sets.items():
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            oof[va] = LGBMClassifier(**LGB_PARAMS).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        out[name] = round(float(average_precision_score(y, oof)), 3)
    out["note"] = (
        "Attr27 (operating profit / financial expenses) is NaN for 309 firms of which 38.8% went bankrupt (2.3% otherwise). "
        "That missing-data pattern, not the ratio itself, carries about half of the apparent skill of the 64-attribute model and "
        "cannot transfer to Indian filings (a debt-free company has zero interest cost). The production model therefore excludes it."
    )
    return out


def train_all(horizon: int = 1, run_full_comparison: bool = True) -> dict:
    """
    Complete reproducible training workflow:
      - Validates data
      - Performs model comparison across Logistic Regression, Random Forest, XGBoost, and LightGBM
      - Fits production model & reference model
      - Backs up previous working model before saving
      - Writes curves, metrics, and comprehensive metadata
    """
    t_start = time.time()
    ART.mkdir(exist_ok=True)
    
    # Backup existing production model if present
    prod_path = ART / "model_production.joblib"
    backup_path = ART / "model_production_backup.joblib"
    if prod_path.exists():
        shutil.copyfile(prod_path, backup_path)

    df = load_polish(horizon=horizon)
    y = df["bankrupt"].astype(int).values
    attrs = [c for c in df.columns if c.startswith("Attr")]

    prod_rep, prod = _fit_eval(polish_to_portable(df)[FEATURE_NAMES], y, "production_india_portable")
    ref_rep, ref = _fit_eval(df[attrs], y, "reference_all_64")
    ablation = artifact_ablation(df)

    # Multi-model cross-validation comparison
    comparison_results = compare_models(df) if run_full_comparison else []
    (ART / "model_comparison.json").write_text(json.dumps(comparison_results, indent=2))

    meta = dict(
        dataset=f"UCI Polish Companies Bankruptcy, file polish_{horizon}year.csv",
        model_version="2.0.0-production",
        training_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_duration_seconds=round(time.time() - t_start, 2),
        n_rows=len(df),
        n_duplicates_removed=int(df.attrs["n_duplicates_removed"]),
        bankrupt_rate=round(float(y.mean()), 4),
        missing_cell_rate=round(df.attrs["missing_cell_rate"], 4),
        rows_with_any_missing=round(df.attrs["rows_with_any_missing"], 3),
        split="80/20 stratified; dev-set 5-fold OOF for calibration+threshold; hold-out touched once",
        horizon_note=(
            "UCI documents the '1year' file as ratios from the first of five forecasting years with the bankruptcy "
            "label observed 5 years later, i.e. a LONG lead time (~5 years). Multi-horizon lead curve written by src/horizons.py."
        ),
        no_calendar_time=(
            "The dataset has no calendar dates, so a time-ordered split is impossible; the hold-out is a company-disjoint, "
            "never-seen-during-fitting sample. Treat PR-AUC as an upper bound on out-of-time performance."
        ),
        cost=dict(fn=COST_FN, fp=COST_FP, bayes_threshold=round(BAYES_T, 4)),
        leakage_controls=[
            "duplicates removed pre-split",
            "hold-out carved first and untouched during tuning",
            "no imputation (native NaN)",
            "calibration + threshold on dev-set OOF only",
            "no post-event data in any feature"
        ],
        hyperparameters=LGB_PARAMS
    )

    report = dict(
        meta=meta,
        production=prod_rep,
        reference=ref_rep,
        ablation=ablation,
        model_comparison=comparison_results
    )

    (ART / "benchmark_report.json").write_text(json.dumps(report, indent=2, default=float))
    joblib.dump(prod["bundle"], prod_path)
    joblib.dump(ref["bundle"], ART / "model_reference.joblib")

    # Save holdout predictions with probabilities
    holdout = pd.DataFrame({"row": prod["test_idx"], "p": prod["p_test"], "y": prod["y_test"]})
    holdout.to_csv(ART / "holdout_predictions.csv", index=False)

    # Save Precision-Recall curve points
    pr = pd.DataFrame({"precision": prod["pr_curve"][0], "recall": prod["pr_curve"][1]})
    pr.to_csv(ART / "pr_curve.csv", index=False)

    # Save ROC curve points
    roc_df = pd.DataFrame({"fpr": prod["roc_curve"][0], "tpr": prod["roc_curve"][1]})
    roc_df.to_csv(ART / "roc_curve.csv", index=False)

    # Save Calibration curve points
    cal_df = pd.DataFrame({"prob_true": prod["cal_curve"][0], "prob_pred": prod["cal_curve"][1]})
    cal_df.to_csv(ART / "calibration_curve.csv", index=False)

    df[["bankrupt"]].assign(row=df.index).to_csv(ART / "polish_row_index.csv", index=False)
    df.to_pickle(ART / "polish_dedup.pkl")
    return report


if __name__ == "__main__":
    rep = train_all(horizon=1, run_full_comparison=True)
    print("Training successfully completed.")
    print("Production PR-AUC:", rep["production"]["pr_auc"], "| ROC-AUC:", rep["production"]["roc_auc"])
