"""Figures for docs/REPORT.md, built only from saved artifacts:  python -m src.report_figures"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import ART

FIG = ART / "figures"
BLUE, RED, GREY, GREEN = "#2f5d8a", "#b03a2e", "#7f8c8d", "#2e7d4f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def benchmark():
    rep = json.loads((ART / "benchmark_report.json").read_text())["production"]
    pr = pd.read_csv(ART / "pr_curve.csv")
    cal = pd.read_csv(ART / "calibration_curve.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(pr.recall, pr.precision, color=BLUE, lw=1.6, label=f"4Q model (PR-AUC {rep['pr_auc']:.3f})")
    ax[0].axhline(rep["pr_auc_baseline"], color=GREY, ls="--", label=f"no-skill baseline ({rep['pr_auc_baseline']:.3f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, untouched hold-out", ylim=(0, 1.02))
    ax[0].legend(fontsize=8)
    ax[1].plot(cal.prob_pred, cal.prob_true, marker="o", color=BLUE, label="4Q model")
    lim = max(cal.prob_pred.max(), cal.prob_true.max()) * 1.05
    ax[1].plot([0, lim], [0, lim], color=GREY, ls="--", label="perfect calibration")
    ax[1].set(xlabel="Predicted probability (Platt-calibrated)", ylabel="Observed bankruptcy rate", title="Reliability (10 quantile bins)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "report_benchmark.png", dpi=170)
    plt.close(fig)


def ablation():
    a = json.loads((ART / "benchmark_report.json").read_text())["ablation"]
    keys = [("all_64_attributes", "All 64 attributes (native NaN)"), ("all_64_minus_Attr27", "64 minus Attr27"),
            ("india_portable_39_plus_Attr27_missing_flag", "39 portable + Attr27-missing flag"),
            ("india_portable_39", "39 portable ratios"), ("attr27_missing_flag_only", "Attr27-missing flag ONLY")]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    vals = [a[k] for k, _ in keys]
    ax.barh([l for _, l in keys][::-1], vals[::-1], color=[RED if "flag" in l else BLUE for _, l in keys][::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set(xlabel="5-fold PR-AUC (5year file)", xlim=(0, 1), title="How much skill comes from a missing-data flag?")
    fig.tight_layout()
    fig.savefig(FIG / "report_ablation.png", dpi=170)
    plt.close(fig)


def cost():
    sims = json.loads((ART / "universe_sim.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (rate, s), c in zip(sims.items(), [GREEN, BLUE, RED]):
        y = np.array(s["cost_curve"])
        ax.plot(np.arange(len(y)), y, color=c, lw=1.6, label=f"default rate {float(rate):.1%} (optimum k={s['k_at_cost_min']})")
    ax.axvline(40, color="black", ls=":", lw=1.2, label="review capacity: 40 of 800")
    ax.set(xlabel="Names reviewed per quarter (top-k by risk)", ylabel="Expected cost (1 = one false alarm)",
           title="Expected cost vs list size, 800-borrower book (20 x FN + FP)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "report_cost.png", dpi=170)
    plt.close(fig)


def univariate():
    u = json.loads((ART / "indian_posthoc.json").read_text())["univariate_auc_exploratory"]["lead_4q"]
    names = {"cf_liab2": "Cash earnings / liabilities", "int_cover": "Interest cover", "net_profit_ta": "Net profit / assets",
             "ebit_ta": "EBIT / assets", "liabilities_ta": "Liabilities / assets", "equity_ta": "Equity / assets",
             "retained_ta": "Retained earnings / assets", "asset_turn": "Asset turnover", "altman_z": "Altman Z''",
             "wc_ta": "Working capital / assets (proxy)", "op_margin": "Operating margin", "p_model": "Benchmark model",
             "risk": "Model + overlay", "current_ratio": "Current ratio (proxy)"}
    items = sorted(u.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cols = [RED if k in ("p_model", "risk") else (GREY if k in ("wc_ta", "current_ratio") else BLUE) for k, _ in items]
    ax.barh([names.get(k, k) for k, _ in items], [v for _, v in items], color=cols)
    ax.axvline(0.5, color="black", ls=":", lw=1)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set(xlim=(0, 1), xlabel="AUC: Indian defaulters vs matched peers, 4 quarters before default",
           title="EXPLORATORY (n = 10 defaulters): which signals separate?")
    fig.tight_layout()
    fig.savefig(FIG / "report_univariate.png", dpi=170)
    plt.close(fig)


def rating_lead():
    s = pd.read_csv("data/indian/soft_signals.csv", parse_dates=["date_public"])
    ev = pd.read_csv("data/indian/events.csv", parse_dates=["event_date"]).set_index("company")
    s["days"] = [(ev.loc[c, "event_date"] - d).days for c, d in zip(s.company, s.date_public)]
    s = s.sort_values("days")
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.barh(s.company, s.days / 30.44, color=[GREEN if d >= 365 else BLUE for d in s.days])
    ax.axvline(12, color="black", ls=":", lw=1, label="4-quarter lead")
    for i, d in enumerate(s.days):
        ax.text(d / 30.44 + 0.2, i, f"{d / 30.44:.1f} mo", va="center", fontsize=7.5)
    ax.set(xlabel="Months from verified rating action to first default", title="Rating actions arrive late")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "report_rating_lead.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    benchmark(); ablation(); cost(); univariate(); rating_lead()
    print("figures written to", FIG)
