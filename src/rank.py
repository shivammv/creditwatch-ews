"""Ranking, forensic/soft-signal overlay, SHAP explanations and the 800-borrower budget simulation."""
from __future__ import annotations
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore", message="LightGBM binary classifier")

from .features import FEATURES, MONOTONE, format_value
from .model import ART, COST_FN, COST_FP

# ---- overlay: policy priors added to the model's log-odds (NOT fitted: 10 defaulters cannot support fitting) ----
OVERLAY = dict(
    z_distress=0.8, z_grey=0.3,          # Altman Z'' zone
    icr_below_1=0.5,                     # interest cover (EBIT / interest) < 1: cannot service debt from operations
    m_zone=0.5,                          # Beneish M > -1.78
    f_weak=0.4, f_mid=0.0,               # Piotroski F <= 3
    rating_downgrade=0.6, promoter_pledge_high=0.4, auditor_red_flag=0.5,
    delayed_filing=0.5, related_party_stress=0.4,
)
SOFT_SIGNALS = ["rating_downgrade", "promoter_pledge_high", "auditor_red_flag", "delayed_filing", "related_party_stress"]
SOFT_CAP = 1.5          # max log-odds uplift from soft signals combined
SOFT_LABELS = {
    "rating_downgrade": "credit-rating downgrade / negative watch in the prior 12 months",
    "promoter_pledge_high": "promoter share pledge above 50%",
    "auditor_red_flag": "auditor resignation, qualification or emphasis-of-matter",
    "delayed_filing": "delayed statutory/exchange filing",
    "related_party_stress": "large related-party loans/advances or group exposure",
}


@lru_cache(maxsize=4)
def load_bundle(name: str = "production") -> dict:
    """'production' = 4-quarter-lead model (UCI 5year file); 'production_8q' = 8-quarter-lead model (4year file)."""
    return joblib.load(ART / f"model_{name}.joblib")


def model_input(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Columns in model order, median-imputed with the imputer learned on benchmark training rows (if the bundle has one)."""
    Xf = X[bundle["features"]].astype(float)
    imp = bundle.get("imputer")
    if imp is None:
        return Xf
    out = pd.DataFrame(imp.transform(Xf), columns=Xf.columns, index=Xf.index)
    if bundle.get("consistency_clamp"):
        # v2: an imputed quick / cash-like ratio can never exceed the firm's own current ratio (quick <= current by definition)
        cr = Xf["current_ratio"]
        for c in ("quick_ratio", "cash_like_ratio"):
            m = Xf[c].isna() & cr.notna()
            out.loc[m, c] = np.minimum(out.loc[m, c], cr[m])
    return out


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    raw = bundle["model"].predict_proba(model_input(bundle, X))[:, 1]
    return bundle["calibrator"].predict(raw)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_overlay(p: np.ndarray, z_zone=None, m_flag=None, f_band=None, soft: pd.DataFrame | None = None, icr=None) -> np.ndarray:
    adj = np.zeros(len(p))
    if z_zone is not None:
        adj += np.where(np.asarray(z_zone) == "distress", OVERLAY["z_distress"], np.where(np.asarray(z_zone) == "grey", OVERLAY["z_grey"], 0.0))
    if m_flag is not None:
        adj += np.where(np.asarray(m_flag) == "manipulation_zone", OVERLAY["m_zone"], 0.0)
    if f_band is not None:
        adj += np.where(np.asarray(f_band) == "weak", OVERLAY["f_weak"], 0.0)
    if icr is not None:
        icr = np.asarray(icr, dtype=float)
        adj += np.where(np.isfinite(icr) & (icr < 1.0), OVERLAY["icr_below_1"], 0.0)
    if soft is not None:
        s = sum(soft[c].fillna(0).astype(float).to_numpy() * OVERLAY[c] for c in SOFT_SIGNALS if c in soft)
        adj += np.minimum(s, SOFT_CAP)
    return 1 / (1 + np.exp(-(_logit(p) + adj)))


def prior_shift(p: np.ndarray, pi_train: float, pi_bank: float) -> np.ndarray:
    """Re-base calibrated probabilities from the benchmark default rate to the bank's own book default rate."""
    shift = np.log(pi_bank / (1 - pi_bank)) - np.log(pi_train / (1 - pi_train))
    return 1 / (1 + np.exp(-(_logit(p) + shift)))


# ---------------------------------------------------------------- explanations
def shap_matrix(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ex = shap.TreeExplainer(bundle["model"])
        sv = ex.shap_values(model_input(bundle, X))
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv, dtype=float)
    return sv[..., 1] if sv.ndim == 3 else sv


def top_drivers(bundle: dict, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """Top-k RISK-INCREASING drivers per row, in credit-officer language."""
    sv = shap_matrix(bundle, X)
    feats = bundle["features"]
    med = bundle["healthy_median"]
    out = []
    for i in range(len(X)):
        rows = []
        # A driver is shown only if (a) the ratio was actually computed (imputed = "no information") and (b) the firm's value
        # is on the risky side of the healthy-peer median under credit logic (MONOTONE sign). Tree models can attach risk to
        # a healthy-looking value (e.g. high EBITDA/assets) through non-monotone patterns in the benchmark data; such
        # contributions are model artefacts, not reasons a credit officer can act on, so they are never written up.
        def credible(j):
            f = feats[j]
            v, m, sgn = X.iloc[i][f], med.get(f), MONOTONE.get(f, 0)
            if pd.isna(v) or m is None or sgn == 0:
                return False
            return (v > m) if sgn > 0 else (v < m)
        order = [j for j in np.argsort(-sv[i]) if credible(j)][:k]
        for j in order:
            if sv[i, j] <= 0:
                continue
            f = feats[j]
            meta = FEATURES[f]
            v = X.iloc[i][f]
            text = meta["concern"].format(v=format_value(v, meta["unit"]), m=format_value(med[f], meta["unit"]))
            rows.append(dict(feature=f, label=meta["label"], shap=float(sv[i, j]), value=None if pd.isna(v) else float(v),
                             peer_median=float(med[f]), text=text))
        out.append(rows)
    return out


def memo_text(company: str, rank: int, risk: float, drivers: list[dict], z=None, m=None, f=None, soft: list[str] | None = None,
              action: str | None = None) -> str:
    lines = [f"WATCH-LIST MEMO | {company} | Rank #{rank} | Risk {risk*100:.1f}% (calibrated distress probability)"]
    for n, d in enumerate(drivers, 1):
        lines.append(f"  Driver {n}: {d['text']}")
    fx = []
    if z is not None and np.isfinite(z):
        fx.append(f"Altman Z''={z:.2f}")
    if m is not None and np.isfinite(m):
        fx.append(f"Beneish M={m:.2f}")
    if f is not None and np.isfinite(f):
        fx.append(f"Piotroski F={int(f)}/9")
    if fx:
        lines.append("  Forensic screens: " + ", ".join(fx))
    if soft:
        lines.append("  Soft signals: " + "; ".join(SOFT_LABELS.get(s, s) for s in soft))
    lines.append("  Action: " + (action or "analyst deep-review within 15 working days; escalate to sector head if two or more drivers are confirmed."))
    return "\n".join(lines)


# ---------------------------------------------------------------- budget simulation
def simulate_universes(p: np.ndarray, y: np.ndarray, n: int = 800, rate: float = 0.039, budget: int = 40,
                       draws: int = 1000, seed: int = 7, bayes_t: float | None = None, train_rate: float = 0.039) -> dict:
    """Repeatedly build an n-borrower book from hold-out firms at a given default rate and rank it by score."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = int(round(rate * n))
    recalls, above, caps = [], [], np.zeros(n + 1)
    cost_curve = np.zeros(n + 1)
    for _ in range(draws):
        i = np.concatenate([rng.choice(pos, n_pos, replace=n_pos > len(pos)), rng.choice(neg, n - n_pos, replace=False)])
        order = np.argsort(-p[i])
        hit = np.cumsum(y[i][order])
        recalls.append(hit[budget - 1] / n_pos)
        cost_curve += COST_FN * (n_pos - np.concatenate([[0], hit])) + COST_FP * (np.arange(n + 1) - np.concatenate([[0], hit]))
        if bayes_t is not None:
            above.append(int((prior_shift(p[i], train_rate, rate) >= bayes_t).sum()))
    cost_curve /= draws
    ks = np.arange(n + 1)
    return dict(rate=rate, n=n, budget=budget, n_pos=n_pos, recall_mean=float(np.mean(recalls)),
                recall_ci95=[float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))],
                mean_alerts_above_bayes=float(np.mean(above)) if above else None,
                cost_at_budget=float(cost_curve[budget]), cost_flag_none=float(COST_FN * n_pos),
                cost_min=float(cost_curve.min()), k_at_cost_min=int(ks[cost_curve.argmin()]),
                cost_curve=cost_curve[:201].tolist(), recalls=recalls)
