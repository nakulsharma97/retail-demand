import pytest
import pandas as pd
import sys
sys.path.insert(0, ".")
from src.data.preprocess import aggregate_weekly

def test_aggregate_weekly_shape():
    df = pd.DataFrame({
        "store_id": ["S001"]*10, "sku_id": ["SKU0001"]*10,
        "sale_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "quantity": range(10, 20), "revenue": range(100, 200, 10),
        "promotion_flag": [0]*10, "category": ["Skincare"]*10,
        "region": ["East Midlands"]*10, "store_type": ["full"]*10,
    })
    result = aggregate_weekly(df)
    assert "total_qty" in result.columns
    assert "week_start" in result.columns
    assert len(result) > 0
