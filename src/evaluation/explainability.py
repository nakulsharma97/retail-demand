"""
Model explainability using SHAP values.
Communicates 'why' a model made a prediction - critical for business stakeholders.
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def explain_xgboost(model, X: pd.DataFrame, output_dir: str = "docs"):
    Path(output_dir).mkdir(exist_ok=True)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    # Summary plot (feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_feature_importance.png", dpi=150)
    plt.close()

    # Beeswarm plot (direction of effect)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150)
    plt.close()

    # Return top 10 features by mean |SHAP|
    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False).head(10)

    return importance
