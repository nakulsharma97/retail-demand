import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")
from api.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_forecast():
    r = client.post("/forecast", json={
        "store_id": "S001", "sku_id": "SKU0001", "horizon_weeks": 4
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["forecasts"]) == 4
    assert "predicted_qty" in data["forecasts"][0]
