# Adding or updating Indian data

The shipped panel (`data/indian/`) holds sourced data only. To extend it:

1. **Events** (`events.csv`): one row per defaulter with `tier=core`, the **earliest documented credit event** (payment-default disclosure, rating to D, or CIRP admission), `event_type`, `date_status=verified`, and a `source_url` (exchange filing, rating-agency release, trustee notice or NCLT/IBBI order).
2. **Peers** (`peers.csv`): same sector, listed at the event date, no default/IBC/rating-D through event + 8 quarters.
3. **Financials** (`financials.csv`): one row per company and fiscal year, in one unit (Rs crore). Columns:
   `total_assets, current_assets, current_liabilities, total_liabilities, equity, retained_earnings, sales, cogs, sga, depreciation, ebit, interest_expense, pbt, net_income, cfo, receivables, inventory, ppe_net, lt_debt, cash, shares_out`, plus `public_date` (blank = FY end + 120 days) and `source_url`.
   Leave unknown items **blank** (never 0): blanks are treated as "no information".
   For Screener pages, transcribe the standalone tables into `screener_raw/<company>.csv` and map them with `rebuild_panel.py` (the mapping is documented there; the provenance tests in `tests/test_audit.py` check that every balance sheet adds up).
4. **Soft signals** (`soft_signals.csv`): `company, signal, date_public, value, source_url, detail, verified`, where `signal` is one of `rating_downgrade, promoter_pledge_high (value = pledge %), auditor_red_flag, delayed_filing, related_party_stress`. Only verified rows with a URL are used.
5. Run `python -m pytest -q` (provenance and point-in-time tests), then `python run_all.py`.

Changing the data does not change the frozen configuration. Changing models, overlay weights or rules does, and `indian.evaluate()` will then refuse to run until the change is logged in `docs/DEVIATIONS.md` and re-frozen with `python -m src.config --freeze`.

Banks/NBFCs (`is_financial=1`): use total income as `sales`; Z''/M/F are skipped for them by design.
