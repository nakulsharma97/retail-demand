# Results Summary - Business Impact

## Baseline vs XGBoost Model
| Metric | Naive Baseline | XGBoost | Improvement |
|--------|---------------|---------|-------------|
| MAPE   | 28.5%         | 11.3%   | -60%        |
| RMSE   | 24.1          | 12.1    | -50%        |
| Stockout Rate | 13.0% | 7.2%  | -45%        |

## Business Value
- A 1% reduction in stockout rate across 500 Boots stores translates to an estimated
  Â£500K-Â£2M in recovered revenue annually (based on average basket assumptions).
- Overstock reduction reduces waste disposal costs and markdown requirements.
- Automated forecasting frees buying team from manual spreadsheet-based planning.

## Key Insights
1. **Seasonality** is the dominant driver - Pharmacy demand spikes in winter/summer
2. **Promotions** deliver 30-100% uplift depending on category
3. **Store type** matters - Full stores outperform pharmacy-only stores for beauty SKUs
4. **Lag features** (especially 4-week and 52-week) are the strongest predictors in XGBoost
