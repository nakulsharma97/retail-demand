"""
MLflow experiment tracking utilities.
Centralises all logging patterns for the project.
"""

import mlflow
from pathlib import Path


def setup_mlflow(experiment_name: str = "boots_demand_forecasting",
                 tracking_uri: str = "mlruns"):
    Path(tracking_uri).mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_feature_importance(model, feature_names: list):
    importance = dict(zip(feature_names, model.feature_importances_))
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, score in top10:
        mlflow.log_metric(f"importance_{feat}", score)
