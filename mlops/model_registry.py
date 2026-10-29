"""
Model registration and promotion between staging and production.
"""

import mlflow
from mlflow.tracking import MlflowClient


def register_model(run_id: str, model_name: str = "boots_xgb_demand"):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/xgboost_model"
    mv = mlflow.register_model(model_uri, model_name)
    print(f"Registered: {model_name} v{mv.version}")
    return mv


def promote_to_production(model_name: str, version: int):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )
    print(f"Model {model_name} v{version} promoted to Production")
