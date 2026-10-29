"""
Inference pipeline for generating fresh demand forecasts.
Loads the latest registered MLflow model and outputs predictions.
"""

import pandas as pd
import numpy as np
import mlflow.xgboost
from loguru import logger


def load_latest_model(run_id: str = None):
    if run_id:
        model = mlflow.xgboost.load_model(f"runs:/{run_id}/xgboost_model")
    else:
        model = mlflow.xgboost.load_model("models:/boots_xgb_demand/Production")
    return model


def predict(feature_df: pd.DataFrame, model) -> pd.DataFrame:
    exclude = ["store_id","sku_id","week_start","year_week","total_qty",
               "total_revenue","sale_date"]
    feature_cols = [c for c in feature_df.columns
                    if c not in exclude and feature_df[c].dtype != "object"]
    X = feature_df[feature_cols].fillna(0)
    preds = np.maximum(model.predict(X), 0)
    result = feature_df[["store_id","sku_id","week_start"]].copy()
    result["predicted_qty"] = preds.round(0).astype(int)
    return result
