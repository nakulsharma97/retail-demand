# System Architecture - Boots Demand Forecasting

## Overview
Three-layer architecture: Data -> Modelling -> Serving

## Data Layer
- SQLite database (schema in sql/create_tables.sql) - production-portable to PostgreSQL or Databricks
- 8 stores Ã— ~25 SKUs Ã— 730 days = ~1.5M synthetic sales records
- Feature engineering: lag features, rolling statistics, calendar features, UK bank holidays

## Modelling Layer
| Model | Strengths | Use Case |
|-------|-----------|----------|
| Prophet | Explainability, trend + seasonality decomposition | Exec reporting |
| XGBoost | Accuracy, handles many features, fast | Primary production model |
| LSTM (PyTorch) | Captures long-range temporal patterns | Category-level forecasting |

## Serving Layer
- FastAPI REST endpoint (/forecast) returns predictions with confidence intervals
- MLflow tracks all experiments, model versions, and metrics
- Docker container for reproducible deployment

## Cloud Readiness
- Data layer abstracts behind SQLAlchemy (swap SQLite -> AWS RDS or Azure SQL)
- Model artifacts stored in MLflow (compatible with S3 or Azure Blob backend)
- Databricks-compatible: feature engineering uses pandas/Spark-portable patterns

## Business KPIs
The system directly optimises: MAPE (down), Stockout Rate (down), Overstock Rate (down)
