"""Forensic scores: Altman Z'' (emerging-market / non-manufacturing), Beneish M-8, Piotroski F."""
from __future__ import annotations
import numpy as np


def _d(a, b):
    if a is None or b is None:
        return np.nan
    try:
        a_f, b_f = float(a), float(b)
        return a_f / b_f if b_f != 0 and np.isfinite(a_f) and np.isfinite(b_f) else np.nan
    except (ValueError, TypeError, ZeroDivisionError):
        return np.nan


def altman_z(wc, ta, re, ebit, book_equity, tl):
    """Altman Z'' (no constant; book equity so no market-cap look-up is needed).
    Zones: > 2.60 safe | 1.10-2.60 grey | < 1.10 distress. Suited to non-manufacturers and
    emerging markets; use for non-financial firms only."""
    if not (np.isfinite(ta) and ta > 0):
        return np.nan
    return 6.56 * wc / ta + 3.26 * re / ta + 6.72 * ebit / ta + 1.05 * _d(book_equity, tl)


def altman_zone(z: float) -> str:
    if not np.isfinite(z):
        return "unknown"
    return "safe" if z > 2.6 else ("grey" if z >= 1.1 else "distress")


def beneish_m(cur: dict, prev: dict) -> float:
    """Beneish (1999) 8-variable M-score from raw items of year t and t-1.
    M > -1.78 -> manipulation zone (the conventional cut-off)."""
    try:
        if not cur or not prev:
            return np.nan
        def _sub(a, b):
            if a is None or b is None: return np.nan
            return float(a) - float(b)
        def _add(a, b):
            if a is None or b is None: return np.nan
            return float(a) + float(b)

        dsri = _d(_d(cur.get("receivables"), cur.get("sales")), _d(prev.get("receivables"), prev.get("sales")))
        gm = lambda r: _d(_sub(r.get("sales"), r.get("cogs")), r.get("sales"))
        gmi = _d(gm(prev), gm(cur))
        aq = lambda r: 1 - _d(_add(r.get("current_assets"), r.get("ppe_net")), r.get("total_assets"))
        aqi = _d(aq(cur), aq(prev))
        sgi = _d(cur.get("sales"), prev.get("sales"))
        dep = lambda r: _d(r.get("depreciation"), _add(r.get("depreciation"), r.get("ppe_net")))
        depi = _d(dep(prev), dep(cur))
        sga = lambda r: _d(r.get("sga"), r.get("sales"))
        sgai = _d(sga(cur), sga(prev))
        lev = lambda r: _d(_add(r.get("current_liabilities"), r.get("lt_debt")), r.get("total_assets"))
        lvgi = _d(lev(cur), lev(prev))
        tata = _d(_sub(cur.get("net_income"), cur.get("cfo")), cur.get("total_assets"))
        parts = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
        if not all(np.isfinite(p) for p in parts):
            return np.nan
        return (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi
                - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    except (KeyError, TypeError, ValueError):
        return np.nan


def beneish_flag(m: float) -> str:
    if not np.isfinite(m):
        return "unknown"
    return "manipulation_zone" if m > -1.78 else "unlikely"


def piotroski_f(cur: dict, prev: dict) -> float:
    """Piotroski (2000) F-score (0-9) from raw items of years t and t-1. NaN if any input is missing."""
    try:
        if not cur or not prev:
            return np.nan
        def _sub(a, b):
            if a is None or b is None: return np.nan
            return float(a) - float(b)

        ta, tap = cur.get("total_assets"), prev.get("total_assets")
        gm = lambda r: _d(_sub(r.get("sales"), r.get("cogs")), r.get("sales"))
        cr = lambda r: _d(r.get("current_assets"), r.get("current_liabilities"))
        at = lambda r: _d(r.get("sales"), r.get("total_assets"))
        pairs = [
            (_d(cur.get("net_income"), ta), 0.0, ">"), (cur.get("cfo"), 0.0, ">"),
            (_d(cur.get("net_income"), ta), _d(prev.get("net_income"), tap), ">"),
            (cur.get("cfo"), cur.get("net_income"), ">"),
            (_d(prev.get("lt_debt"), tap), _d(cur.get("lt_debt"), ta), ">"),      # leverage down
            (cr(cur), cr(prev), ">"), (prev.get("shares_out"), cur.get("shares_out"), ">="),  # no dilution
            (gm(cur), gm(prev), ">"), (at(cur), at(prev), ">"),
        ]
        if not all(a is not None and b is not None and np.isfinite(a) and np.isfinite(b) for a, b, _ in pairs):
            return np.nan
        return float(sum((a > b) if op == ">" else (a >= b) for a, b, op in pairs))
    except (KeyError, TypeError, ValueError):
        return np.nan


def f_band(f: float) -> str:
    if not np.isfinite(f):
        return "unknown"
    return "strong" if f >= 7 else ("mid" if f >= 4 else "weak")
