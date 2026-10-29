"""Build data/indian/financials.csv straight from the extracted bulk dump (no manual typing).
   python build_indian_panel.py <folder containing the company folders>
Also writes artifacts/indian_panel_coverage.csv (who matched, how many fiscal years)."""
import sys
from pathlib import Path
from src.screener import build_panel

ROOT = Path(__file__).resolve().parent
if len(sys.argv) < 2:
    sys.exit(__doc__)
r = build_panel(sys.argv[1], ROOT / "data/indian/events.csv", ROOT / "data/indian/peers.csv", ROOT / "data/indian")
r["report"].to_csv(ROOT / "artifacts/indian_panel_coverage.csv", index=False)
rep = r["report"]
print(rep.to_string(index=False))
print(f"\n{r['n_rows']} company-year rows -> data/indian/financials.csv")
print("Defaulters with data:", int(((rep.role == 'defaulter') & (rep.years > 0)).sum()), "| peers with data:", int(((rep.role == 'peer') & (rep.years > 0)).sum()))
print("Not found (need manual entry or a different source):", rep[rep.years == 0].company.tolist())
