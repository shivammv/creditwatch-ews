"""Pre-registration: freeze every choice that affects the Indian evaluation BEFORE any Indian case is scored.

    python -m src.config --freeze    # writes config/frozen_config.json (with SHA-256), then commit it
    python -m src.config --verify    # what indian.evaluate() runs first; raises if anything changed

The frozen file pins the overlay weights, soft-signal rules, publication lag, lead-to-file mapping, LightGBM
hyper-parameters, cost ratio, review capacity, the primary metric definition and the SHA-256 of both model files.
Any later change makes verify_frozen() fail until the change is logged in docs/DEVIATIONS.md and the config is
re-frozen deliberately.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "config" / "frozen_config.json"


def frozen_path(variant: str = "") -> Path:
    return ROOT / "config" / f"frozen_config{variant}.json"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_config(variant: str = "") -> dict:
    from .model import ART, COST_FN, COST_FP, LEAD_FILES, LGB_PARAMS
    from .features import MONOTONE
    from .rank import OVERLAY, SOFT_CAP, SOFT_SIGNALS
    from .indian import PUBLICATION_LAG_DAYS
    files = (f"model_production{variant}.joblib", f"model_production_8q{variant}.joblib")
    cfg = dict(
        model=dict(family="LightGBM, median-imputed (imputer fitted on training rows only), Platt-calibrated on OOF",
                   features="39 India-portable ratios (src/features.py FEATURE_NAMES)",
                   lead_to_uci_file=LEAD_FILES, hyperparameters=LGB_PARAMS,
                   artifact_sha256={f: _sha256_file(ART / f) for f in files}),
        overlay_log_odds=OVERLAY, soft_signal_cap=SOFT_CAP, soft_signals=SOFT_SIGNALS,
        soft_signal_window_days=365, soft_signal_rule="date_public <= observation date and within prior 365 days",
        publication_lag_days=PUBLICATION_LAG_DAYS,
        observation_rule="obs = event_date - 3*L months; fiscal year usable only if public_date <= obs; 2 usable years required",
        event_rule="earliest documented credit event (payment default disclosure, rating to D, or CIRP admission) with source URL",
        cost=dict(missed_default=COST_FN, false_alarm=COST_FP, bayes_threshold="1/21"),
        capacity=dict(universe=800, review_slots=40, alert_rate=0.05),
        primary_metric=("recall of Indian defaulters above the 95th percentile of matched non-defaulting peer scores at "
                        "4Q and 8Q lead (Wilson 95% CI); secondary: simulated 800-name book with Polish hold-out background"),
    )
    if variant == "_v2":
        cfg["model"]["monotone_constraints"] = MONOTONE
        cfg["model"]["consistency_clamp"] = "imputed quick / cash-like ratio capped at the firm's own current ratio"
        cfg["status"] = "POST-HOC variant (frozen after v1 Indian results were seen; signs from credit logic, not fitted)"
    return cfg


def _digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def freeze(variant: str = "") -> dict:
    cfg = current_config(variant)
    payload = dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), config=cfg, sha256=_digest(cfg))
    frozen_path(variant).parent.mkdir(exist_ok=True)
    frozen_path(variant).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def verify_frozen(variant: str = "") -> str:
    fp = frozen_path(variant)
    if not fp.exists():
        raise RuntimeError(f"{fp.name} missing: freeze the configuration before scoring Indian cases")
    payload = json.loads(fp.read_text())
    if _digest(payload["config"]) != payload["sha256"]:
        raise RuntimeError("frozen_config.json was edited after freezing (hash mismatch)")
    cur = json.loads(json.dumps(current_config(variant), sort_keys=True, default=str))
    if cur != payload["config"]:
        diff = [k for k in cur if cur.get(k) != payload["config"].get(k)]
        raise RuntimeError(f"code/model differs from the pre-registered configuration in: {diff}. "
                           "Log the change in docs/DEVIATIONS.md and re-freeze deliberately.")
    return payload["sha256"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--variant", default="", help='"" (primary, pre-registered) or "_v2" (post-hoc monotone)')
    a = ap.parse_args()
    if a.freeze:
        p = freeze(a.variant)
        print("frozen", a.variant or "primary", p["sha256"])
    else:
        print("verified", a.variant or "primary", verify_frozen(a.variant))
