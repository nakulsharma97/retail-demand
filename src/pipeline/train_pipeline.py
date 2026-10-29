"""
End-to-end training pipeline.
Run this script to: generate features -> train models -> log to MLflow -> save artifacts.
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from loguru import logger

from src.data.preprocess import load_raw_data, build_master_df, aggregate_weekly, train_test_split_temporal
from src.features.feature_engineering import build_feature_matrix
from src.models.xgboost_model import train_xgboost, evaluate_xgboost, get_feature_cols
from src.evaluation.metrics import evaluate_all


def run():
    mlflow.set_experiment("boots_demand_forecasting")
    logger.info("=== Boots Demand Forecasting Training Pipeline ===")

    # 1. Load & preprocess
    stores, products, sales = load_raw_data()
    master = build_master_df(sales, stores, products)
    weekly = aggregate_weekly(master)
    train, test = train_test_split_temporal(weekly)

    # 2. Feature engineering
    combined  = pd.concat([train, test]).sort_values(["store_id","sku_id","week_start"])
    featured  = build_feature_matrix(combined)
    cutoff    = train["week_start"].max()
    feat_train = featured[featured["week_start"] <= cutoff]
    feat_test  = featured[featured["week_start"] >  cutoff]

    # 3. Train XGBoost
    logger.info("Training XGBoost model...")
    xgb_model, feature_cols = train_xgboost(feat_train)

    # 4. Evaluate
    results = evaluate_xgboost(xgb_model, feat_test, feature_cols)
    metrics = evaluate_all(feat_test["total_qty"], results["predictions"], "XGBoost")
    logger.info(f"Final metrics: {metrics}")

    # 5. Save predictions
    Path("data/processed").mkdir(exist_ok=True)
    feat_test = feat_test.copy()
    feat_test["predicted_qty"] = results["predictions"]
    feat_test[["store_id","sku_id","week_start","total_qty","predicted_qty"]].to_csv(
        "data/processed/predictions.csv", index=False
    )

    logger.success("Pipeline complete. Run: mlflow ui  to view experiments.")


if __name__ == "__main__":
    run()
