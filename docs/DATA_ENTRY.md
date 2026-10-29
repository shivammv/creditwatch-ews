# Filling the Indian case panel (the one thing left to do by hand)

The code, leakage guard and templates are ready; **real filings are not shipped, and no figures were invented.**

1. `cp data/indian/financials_TEMPLATE.csv data/indian/financials.csv`
2. For each company-row enter the audited statement items for that fiscal year (same unit for the whole file, e.g. INR crore). Columns:
   `total_assets, current_assets, current_liabilities, total_liabilities, equity, retained_earnings, sales, cogs, sga, depreciation, ebit, interest_expense, pbt, net_income, cfo, receivables, inventory, ppe_net, lt_debt, cash, shares_out`.
   Put the **date the results/annual report became public** in `public_date` (if blank it defaults to `fy_end + 120 days`) and the source link in `source_url`.
3. Each name has 4 fiscal-year rows: three "years before the event" plus one earlier year for the Beneish/Piotroski year-on-year items.
4. **Confirm event dates** in `data/indian/events.csv` (all are `unverified`; some template dates are flagged as doubtful) from IBBI / NCLT orders / exchange announcements, then set `date_status=verified`.
5. Soft signals (optional but required by the brief: use at least two types): copy `soft_signals_TEMPLATE.csv` to `soft_signals.csv`; one row per observation: `company, signal, date_public, value, source_url` where `signal` is one of `rating_downgrade, promoter_pledge_high (value = pledge %), auditor_red_flag, delayed_filing, related_party_stress`.
6. Peers: keep the 32 matched peers in `peers.csv` only if they did not default / enter IBC / get rated D before the matched event date; swap any that fail.
7. `python run_all.py` computes 4-quarter and 8-quarter recall inside the 40-of-800 budget (with Indian names embedded in a 760-name background book) and unlocks the Indian tab in the app.

Banks/NBFCs (`is_financial=1`): enter the same items using total income as `sales`; Z''/M/F are skipped for them by design.
Deliberately exclude: Vodafone Idea (not a default) and Yes Bank (reconstruction) from the headline recall; report them separately.

## Shortcut: build the panel from the bulk dump (no typing)
`python build_indian_panel.py <folder with company folders>` fills `data/indian/financials.csv` for every defaulter/peer found in the
"Detailed Financials Data of 4456 NSE & BSE Companies" dump. Caveats (flagged `bs_proxy=1`): condensed balance sheet (current assets/
liabilities proxied by Other Assets/Other Liabilities), COGS/SG&A/cash/shares missing (Beneish M and Piotroski F = NaN), receivables and
inventory rebuilt from days ratios, values are the latest restated view (as of Dec-2023), and delisted failures (DHFL, Bhushan Steel,
Videocon, Lanco, Amtek, Jaypee Infratech, Era Infra, Ruchi Soya) are absent (survivorship bias). `data/indian/candidates_in_dump.csv` lists
extra names that exist in the dump and could fill the 10-defaulter quota once their event dates are verified.
