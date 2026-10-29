import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["features_in_production"] == 39


def test_api_dashboard():
    res = client.get("/api/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "kpis" in data
    assert data["kpis"]["total_companies_analyzed"] > 0
    assert "sectors" in data
    assert "top_high_risk" in data


def test_api_model_metrics():
    res = client.get("/api/model/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "production" in data
    assert data["production"]["pr_auc"] > 0.20
    assert "model_comparison" in data
    assert len(data["model_comparison"]) >= 4
    assert "pr_curve" in data
    assert "roc_curve" in data
    assert "calibration_curve" in data


def test_api_score_single_healthy_company():
    payload = {
        "company": "Solvent Corp",
        "sector": "Technology",
        "is_financial": False,
        "total_assets": 10000.0,
        "current_assets": 6000.0,
        "current_liabilities": 2000.0,
        "total_liabilities": 3000.0,
        "equity": 7000.0,
        "retained_earnings": 5000.0,
        "sales": 15000.0,
        "cogs": 8000.0,
        "sga": 2000.0,
        "depreciation": 500.0,
        "ebit": 4500.0,
        "interest_expense": 150.0,
        "pbt": 4350.0,
        "net_income": 3200.0,
        "cfo": 3800.0,
        "receivables": 1200.0,
        "inventory": 800.0,
        "cash": 3000.0,
    }
    res = client.post("/api/score/single", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_category"] == "Low Risk"
    assert data["combined_risk_score"] < 0.20
    assert data["forensics"]["altman_zone"] == "safe"
    assert data["forensics"]["interest_coverage_ratio"] > 10.0


def test_api_score_single_distressed_company():
    payload = {
        "company": "Distressed Aviation",
        "sector": "Aviation",
        "is_financial": False,
        "total_assets": 12000.0,
        "current_assets": 3000.0,
        "current_liabilities": 9000.0,
        "total_liabilities": 18000.0,
        "equity": -6000.0,
        "retained_earnings": -7000.0,
        "sales": 14000.0,
        "cogs": 12000.0,
        "sga": 2500.0,
        "depreciation": 800.0,
        "ebit": -1300.0,
        "interest_expense": 1200.0,
        "pbt": -2500.0,
        "net_income": -2500.0,
        "rating_downgrade": 1,
        "delayed_filing": 1,
    }
    res = client.post("/api/score/single", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_category"] == "High Risk"
    # the absolute level depends on the model's base rate (1-year horizon, 7% in the benchmark); what must hold is that
    # forensic + soft-signal overlays push this clearly distressed firm well above its raw model probability
    assert data["combined_risk_score"] > 0.20
    assert data["combined_risk_score"] > 3 * data["calibrated_model_probability"]
    assert data["forensics"]["altman_zone"] == "distress"
    assert data["forensics"]["icr_below_one"] is True
    assert len(data["shap_drivers"]) > 0


def test_api_companies_list():
    res = client.get("/api/companies")
    assert res.status_code == 200
    companies = res.json()
    assert len(companies) >= 10
    names = [c["company"] for c in companies]
    assert "Jet Airways" in names or any("Jet" in n for n in names)


def test_api_company_detail():
    res = client.get("/api/companies/Jet%20Airways")
    assert res.status_code == 200
    data = res.json()
    assert data["company"] == "Jet Airways"
    assert "combined_risk_pct" in data
    assert "history" in data
    assert len(data["history"]) >= 2


def test_api_watchlist():
    res = client.get("/api/watchlist?page=1&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert len(data["items"]) <= 10
    assert data["total"] > 0
    assert "budget_cap_count" in data


def test_api_watchlist_budget_filtering():
    res = client.get("/api/watchlist?in_budget_only=true&budget_pct=0.05")
    assert res.status_code == 200
    data = res.json()
    for item in data["items"]:
        assert item["in_budget"] is True


def test_api_forensics_summary():
    res = client.get("/api/forensics/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["beneish_threshold"] == -1.78
    assert "CRITICAL RISK DISCLAIMER" in data["beneish_disclaimer"]
    assert "DISTINCTION OF SCOPE" in data["corporate_vs_transaction_fraud_disclaimer"]
    assert len(data["m_indices"]) == 8


def test_api_upload_validation_csv():
    csv_content = b"company,fy_end,total_assets,current_assets,current_liabilities,total_liabilities,equity,sales,ebit\nAcme Corp,2023-03-31,1000,400,300,600,400,1200,150\n"
    res = client.post(
        "/api/upload/validate",
        files={"file": ("test.csv", csv_content, "text/csv")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows"] == 1
    assert data["is_ready_to_score"] is True


def test_api_upload_invalid_filetype():
    res = client.post(
        "/api/upload/validate",
        files={"file": ("test.txt", b"plain text", "text/plain")}
    )
    assert res.status_code == 400
