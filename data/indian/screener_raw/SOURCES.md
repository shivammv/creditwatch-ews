# Screener.in standalone statements transcribed on 29 Oct 2026 (Rs crore)

| File | Company | Source page | Notes |
|---|---|---|---|
| itnl.csv | IL&FS Transportation Networks Ltd (listed IL&FS group company; the parent IL&FS Ltd was unlisted) | https://screener.in/company/IL&FSTRANS | Standalone |
| bhushan_steel.csv | Bhushan Steel Ltd (renamed Tata Steel BSL Ltd in 2018, merged into Tata Steel 2021) | https://screener.in/company/id/475 | Standalone; FY2018+ is post-acquisition |
| sintex_industries.csv | Sintex Industries Ltd | https://www.screener.in/company/SINTEX/ | Standalone; the Mar 2016 column is restated for the 2017 demerger of the plastics business (sales 915 vs 4,006 a year earlier), so FY2016 flows are not comparable with FY2015 |
| dhfl.csv | Dewan Housing Finance Corporation Ltd (page now shows Piramal Capital & Housing Finance after the 2021 merger) | https://screener.in/company/DHFL | Standalone; only Mar 2011 to Mar 2020 kept (pre-merger DHFL years). For lenders, "sales" is total revenue and "operating profit" is financing profit before interest |

Every balance sheet column was checked: equity capital + reserves + borrowings + other liabilities = total, within Rs 1 crore of rounding (`tests/test_data_provenance.py`).
Screener shows the latest restated figures, not necessarily the numbers as first published. This is a disclosed limitation, shared with the rest of the Screener-based panel.
