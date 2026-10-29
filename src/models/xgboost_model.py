"""
XGBoost Demand Forecasting Model
Handles multi-store, multi-SKU forecasting with rich feature engineering.
Includes MLflow experiment tracking.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger

FEATURE_COLS = None  # Set dynamically based on feature matrix


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = ["store_id","sku_id","week_start","year_week",
               "total_qty","total_revenue","sale_date"]
    return [c for c in df.columns if c not in exclude and df[c].dtype != "object"]


def train_xgboost(train_df: pd.DataFrame, params: dict = None) -> XGBRegressor:
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }

    feature_cols = get_feature_cols(train_df)
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["total_qty"]

    with mlflow.start_run(run_name="xgboost_demand_forecast"):
        mlflow.log_params(params)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train)],
                  verbose=False)

        train_preds = model.predict(X_train)
        mae  = mean_absolute_error(y_train, train_preds)
        rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        mape = np.mean(np.abs((y_train - train_preds) / (y_train + 1))) * 100

        mlflow.log_metrics({"train_mae": mae, "train_rmse": rmse, "train_mape": mape})
        mlflow.xgboost.log_model(model, "xgboost_model")

        logger.info(f"XGBoost Train | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.1f}%")

    return model, feature_cols


def evaluate_xgboost(model, test_df: pd.DataFrame, feature_cols: list) -> dict:
    X_test  = test_df[feature_cols].fillna(0)
    y_test  = test_df["total_qty"]
    preds   = model.predict(X_test)
    preds   = np.maximum(preds, 0)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100

    logger.info(f"XGBoost Test  | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape, "predictions": preds}
