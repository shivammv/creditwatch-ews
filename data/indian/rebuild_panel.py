"""Rebuild the Indian panel inputs (financials, events, peers, soft signals) from sourced data only.

Run once: python data/indian/rebuild_panel.py

What it does:
1. Removes the four "reconstructed" defaulter rows (IL&FS parent, DHFL, Bhushan Steel, Sintex Industries). Those rows
   were estimates from news summaries, not filings. They are kept for audit in removed_reconstructed_rows.csv and
   never used.
2. Adds real standalone statements for IL&FS Transportation Networks (listed IL&FS group company), DHFL, Bhushan Steel
   and Sintex Industries from screener_raw/*.csv, mapped exactly like the rest of the Screener-based panel:
     equity = equity capital + reserves; total liabilities = total assets - equity; retained earnings = reserves;
     current assets ~ Screener "other assets"; current liabilities ~ Screener "other liabilities" (bs_proxy=1);
     EBIT = PBT + interest; EBITDA = operating profit; receivables = debtor days x sales / 365; lt_debt = borrowings.
3. Writes events.csv with the EARLIEST documented credit event per defaulter, each with a source URL.
4. Writes peers.csv (IL&FS peers re-matched to ITNL; Devyani dropped because it listed in Aug 2021, after its event).
5. Writes soft_signals.csv with rating actions verified against agency/press sources (with URLs). The earlier
   unverified signal list is kept in soft_signals_unverified.csv and is NOT used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "screener_raw"
LAG_DAYS = 120  # same publication lag as src/indian.py

FAKE = ["IL&FS", "DHFL", "Bhushan Steel", "Sintex Industries"]

SCREENER = {
    "itnl.csv": dict(company="IL&FS Transportation Networks", sector="Infrastructure (roads)", is_financial=0,
                     url="https://screener.in/company/IL&FSTRANS"),
    "dhfl.csv": dict(company="DHFL", sector="Housing finance", is_financial=1, url="https://screener.in/company/DHFL"),
    "bhushan_steel.csv": dict(company="Bhushan Steel", sector="Metals", is_financial=0,
                              url="https://screener.in/company/id/475"),
    "sintex_industries.csv": dict(company="Sintex Industries", sector="Industrials", is_financial=0,
                                  url="https://www.screener.in/company/SINTEX/"),
}


def screener_rows(fname: str, meta: dict) -> pd.DataFrame:
    t = pd.read_csv(RAW / fname).set_index("line").T
    t.index = pd.to_datetime(t.index, format="%b %Y") + pd.offsets.MonthEnd(0)
    t = t.astype(float)
    eq = t.equity_capital + t.reserves
    out = pd.DataFrame({
        "company": meta["company"], "fy_end": t.index, "public_date": t.index + pd.Timedelta(days=LAG_DAYS),
        "total_assets": t.total_assets, "current_assets": t.other_assets, "current_liabilities": t.other_liabilities,
        "total_liabilities": t.total_assets - eq, "equity": eq, "retained_earnings": t.reserves, "sales": t.sales,
        "cogs": np.nan, "sga": np.nan, "depreciation": t.depreciation, "ebit": t.profit_before_tax + t.interest,
        "interest_expense": t.interest, "pbt": t.profit_before_tax, "net_income": t.net_profit,
        "cfo": t.cash_from_operations,
        "receivables": (t.debtor_days * t.sales / 365) if "debtor_days" in t else np.nan,
        "inventory": np.nan, "ppe_net": t.fixed_assets, "lt_debt": t.borrowings, "cash": np.nan, "shares_out": np.nan,
        "ebitda": t.operating_profit, "bs_proxy": 1, "role": "defaulter", "matched_to": meta["company"],
        "is_financial": meta["is_financial"], "sector": meta["sector"], "source_url": meta["url"]})
    return out.reset_index(drop=True)


EVENTS = [
    # company, sector, is_financial, event_date, event_type, source_url, note
    ("Jet Airways", "Aviation", 0, "2018-12-31", "default_disclosure",
     "https://www.business-standard.com/amp/article/markets/jet-airways-slips-over-3-after-it-defaults-on-debt-payments-to-banks-119010200272_1.html",
     "Delayed interest+principal due 31 Dec 2018 to SBI-led consortium (BSE filing reported 2 Jan 2019). Later: grounding 17 Apr 2019, CIRP 20 Jun 2019."),
    ("DHFL", "Housing finance", 1, "2019-06-04", "default_disclosure",
     "https://www.business-standard.com/amp/article/pti-stories/dhfl-commercial-paper-downgraded-to-default-by-icra-crisil-119060500979_1.html",
     "Delayed NCD obligations due 4 Jun 2019; ICRA/CRISIL cut CP to D on 5 Jun 2019. Later: RBI supersession 20 Nov 2019."),
    ("Reliance Capital", "Financials", 1, "2019-09-09", "rating_default",
     "https://www.outlookbusiness.com/finance/care-rating-downgrades-reliance-capital-from-bb-to-d-3610",
     "NCD coupon due 9 Sep 2019 paid one working day late; CARE recognised this as default and cut to D on 23 Sep 2019. Later: RBI supersession 29 Nov 2021."),
    ("Future Retail", "Retail", 0, "2021-12-31", "default_disclosure",
     "https://www.deccanherald.com/business/care-ratings-downgrades-future-retail-over-non-payment-of-dues-1067191.html",
     "Missed Rs 3,494.56 cr principal due 31 Dec 2021 (filing 1 Jan 2022; CARE B -> D). Later: CIRP 20 Jul 2022."),
    ("Coffee Day Enterprises", "Consumer", 0, "2019-09-11", "rating_default",
     "https://business-standard.com/amp/article/news-ani/icra-cuts-rating-of-coffee-day-s-long-term-loans-to-d-119091101025_1.html",
     "ICRA cut CDEL term loans BB+ -> D citing group debt-servicing delays in Aug 2019. Founder's death (31 Jul 2019) is not a credit event."),
    ("Alok Industries", "Textiles", 0, "2017-07-18", "cirp_admission",
     "https://business-standard.com/amp/article/finance/sbi-gets-nclt-nod-for-insolvency-process-against-alok-industries-117071801705_1.html",
     "NCLT Ahmedabad admitted SBI petition. Earlier payment defaults are likely but were not verified in this sprint, so lead time may be overstated."),
    ("Bhushan Steel", "Metals", 0, "2015-04-01", "default_disclosure",
     "https://axistrustee.in/pdf/Press-Release-dated-October15-To-Dec15.pdf",
     "Debenture trustee notice: interest due 01/04/2015 defaulted (further defaults Aug-Nov 2015). Later: CIRP Jul 2017. Earlier stress (2013-14) reported but not verified here."),
    ("Reliance Communications", "Telecom", 0, "2017-05-30", "rating_default",
     "https://www.bloombergquint.com/business/2017/05/30/reliance-communications-downgraded-to-default-by-care-ratings",
     "CARE and ICRA cut to D on 30 May 2017 citing delayed debt servicing. Later: NCLT admission 15 May 2018."),
    ("Sintex Industries", "Industrials", 0, "2019-05-31", "rating_default",
     "https://www.careratings.com/upload/CompanyFiles/PR/Sintex Industries Limited-06-12-2019.pdf",
     "CARE (12 Jun 2019): ongoing delays on bank facilities due 31 May 2019; NCD interest+principal due 11 Jun 2019 unpaid; NCD rating C -> D. Later: CIRP 6 Apr 2021."),
    ("IL&FS Transportation Networks", "Infrastructure (roads)", 0, "2018-09-28", "rating_default",
     "https://www.careratings.com/upload/CompanyFiles/PR/IL&FS Transportation Networks Limited-09-28-2018.pdf",
     "CARE BB -> D for delays in debt servicing. Listed IL&FS group company used because the parent IL&FS Ltd was unlisted. Six SPVs had already defaulted for Jun 2018 (CARE 26 Sep 2018)."),
]

SOFT = [
    # company, signal, date_public, value, source_url, detail
    ("Jet Airways", "rating_downgrade", "2018-09-04", 1,
     "https://www.business-standard.com/amp/article/pti-stories/icra-cuts-ratings-on-jet-airways-borrowing-programmes-118090401349_1.html",
     "ICRA long-term BB+ -> BB, negative outlook (sub-investment grade)"),
    ("IL&FS Transportation Networks", "rating_downgrade", "2018-07-18", 1,
     "https://www.careratings.com/upload/CompanyFiles/PR/IL&FS Transportation Networks Limited-07-18-2018.pdf",
     "CARE A- -> BB (Negative); short-term A2+ -> A4"),
    ("Future Retail", "rating_downgrade", "2020-05-13", 1,
     "https://www.careratings.com/upload/CompanyFiles/PR/Future Retail Limited-05-13-2020.pdf",
     "CARE A+ -> A- (two notches), on credit watch with negative implications"),
    ("Reliance Capital", "rating_downgrade", "2019-05-19", 1,
     "https://www.business-standard.com/amp/article/companies/care-downgrades-reliance-capital-bonds-rating-from-a-to-bbb-119051900907_1.html",
     "CARE A -> BBB (three notches), credit watch with developing implications"),
    ("DHFL", "rating_downgrade", "2019-05-22", 1,
     "https://www.bloombergquint.com/amp/business/care-downgrades-dhfl-ratings-for-want-of-a-strategic-investor",
     "CARE A/A- -> BBB-, negative outlook (date: reported Wednesday 22 May 2019)"),
]


def main():
    fin = pd.read_csv(HERE / "financials.csv")
    removed = fin[fin.company.isin(FAKE)]
    removed.to_csv(HERE / "removed_reconstructed_rows.csv", index=False)
    fin = fin[~fin.company.isin(FAKE)]
    fin = fin[fin.company != "Devyani International"]
    new = pd.concat([screener_rows(f, m) for f, m in SCREENER.items()], ignore_index=True)
    new["fy_end"] = new.fy_end.dt.strftime("%Y-%m-%d")
    new["public_date"] = new.public_date.dt.strftime("%Y-%m-%d")
    fin = pd.concat([fin, new[fin.columns]], ignore_index=True)
    peers = pd.read_csv(HERE / "peers.csv")
    peers = peers[peers.peer != "Devyani International"].copy()
    peers.loc[peers.matched_to == "IL&FS", ["sector", "is_financial"]] = ["Infrastructure (roads)", 0]
    peers.loc[peers.matched_to == "IL&FS", "matched_to"] = "IL&FS Transportation Networks"
    peers["status_check"] = "listed at event date; no default/IBC/rating-D found through event + 8 quarters (desk check)"
    fin.loc[fin.company.isin(peers.peer), "matched_to"] = fin.loc[fin.company.isin(peers.peer), "company"].map(
        dict(zip(peers.peer, peers.matched_to)))
    fin.loc[fin.company.isin(peers.peer), "sector"] = fin.loc[fin.company.isin(peers.peer), "company"].map(
        dict(zip(peers.peer, peers.sector)))
    fin.loc[fin.company.isin(peers.peer), "is_financial"] = fin.loc[fin.company.isin(peers.peer), "company"].map(
        dict(zip(peers.peer, peers.is_financial)))
    fin.to_csv(HERE / "financials.csv", index=False)
    peers.to_csv(HERE / "peers.csv", index=False)

    ev = pd.DataFrame(EVENTS, columns=["company", "sector", "is_financial", "event_date", "event_type", "source_url", "note"])
    ev["tier"] = "core"
    ev["date_status"] = "verified"
    ev = ev[["company", "sector", "is_financial", "tier", "event_date", "event_type", "date_status", "source_url", "note"]]
    old = pd.read_csv(HERE / "events.csv")
    extra = old[~old.company.isin(ev.company) & (old.tier != "core")][["company", "sector", "is_financial", "tier", "event_date",
                                                                       "event_type", "date_status", "note"]]
    excluded = pd.DataFrame([dict(company="IL&FS", sector="Infrastructure / NBFC", is_financial=1, tier="excluded",
                                  event_date="2018-09-01", event_type="payment default (group)", date_status="month_verified",
                                  note="Parent IL&FS Ltd was unlisted, so no exchange filings; replaced by listed IL&FS Transportation Networks.")])
    pd.concat([ev, excluded, extra], ignore_index=True).to_csv(HERE / "events.csv", index=False)

    old_soft = pd.read_csv(HERE / "soft_signals.csv")
    if "verified" not in old_soft:
        old_soft.to_csv(HERE / "soft_signals_unverified.csv", index=False)
    soft = pd.DataFrame(SOFT, columns=["company", "signal", "date_public", "value", "source_url", "detail"])
    soft["verified"] = 1
    soft.to_csv(HERE / "soft_signals.csv", index=False)
    print(f"financials: {len(fin)} rows, {fin.company.nunique()} companies; removed {len(removed)} reconstructed rows")
    print(f"events core: {len(ev)}; peers: {len(peers)}; verified soft signals: {len(soft)}")


if __name__ == "__main__":
    main()
