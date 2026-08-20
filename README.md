# 🚲 Bike-Sharing Demand Prediction

Predicting hourly bike rental demand for urban bike-sharing systems using weather, seasonal, and temporal features — helping operators optimize bike availability, reduce customer wait time, and plan operations proactively.

## Problem Statement
Bike-sharing companies need a stable, predictable supply of rental bikes across urban locations. This project builds a regression model that forecasts hourly rental demand (`cnt`) based on external factors like weather, season, holidays, and time of day — enabling better fleet distribution and operational planning.

## Dataset
17,379 hourly records (2011–2012) with weather, calendar, and usage features. The raw data required significant cleaning:
- Missing values encoded as `'?'` across nearly every column
- Categorical typos (e.g. `"springer"` → `spring`)
- Numeric fields stored as text
- Outliers in weather sensor readings

## Approach
1. **EDA & Cleaning** — handled hidden missing values, fixed encodings, capped outliers via IQR, analyzed hourly/weekly/seasonal demand patterns
2. **Visualization** — time series trends, workday vs non-workday hourly demand, casual vs registered rider behavior, correlation heatmap
3. **Feature Engineering** — cyclical (sin/cos) encoding for hour & month, rush-hour flag, weekend flag, temperature-humidity interaction, one-hot encoded categoricals
4. **Modeling** — Decision Tree, Random Forest, Gradient Boosting Regressor, each with baseline + RandomizedSearchCV tuning
5. **Evaluation** — compared via MAE, RMSE, R²

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting (tuned, + lag features)** | **29.24** | **47.82** | **0.9530** |
| XGBoost (tuned, + lag features) | 29.64 | 48.33 | 0.9520 |
| Gradient Boosting (tuned, no lag features) | 25.15 | 39.88 | 0.9505 |
| Random Forest (baseline) | 26.89 | 43.92 | 0.9400 |
| Random Forest (tuned) | 29.65 | 45.18 | 0.9365 |
| Decision Tree (tuned) | 32.55 | 54.50 | 0.9076 |
| Gradient Boosting (baseline) | 41.60 | 59.39 | 0.8902 |
| Decision Tree (baseline) | 37.14 | 61.77 | 0.8812 |

**Best model: Gradient Boosting Regressor**, trained with time-series lag features (previous hour, same-hour-yesterday, 3-hour rolling average) — explains ~95% of variance in hourly rental demand. A 90% prediction interval (quantile regression) achieves 92.5% empirical coverage on held-out data.

## Advanced Analysis
- **Time-series lag features** — previous-hour and same-hour-yesterday demand, rolling averages
- **XGBoost benchmark** — compared against Gradient Boosting for a stronger baseline
- **SHAP explainability** — global feature importance and per-prediction waterfall explanations
- **Residual analysis** — diagnoses where/when the model over- or under-predicts
- **Prediction intervals** — 90% confidence range via quantile regression, not just a point estimate

## Deployment
Interactive Streamlit app (`app_v6.py`) — editorial, non-dashboard design: an oversized headline, narrative copy woven with real stats, and a dual wave visualization (teal = registered riders, purple = casual riders) built directly from actual hourly averages in the dataset. No sidebar, no card grid — one continuous visual idea per page. The Predict page runs the same real autoregressive 24-hour rollout with SHAP explanations, restyled to match.

```bash
pip install -r requirements.txt
streamlit run app_v6.py
```

Earlier versions (`app.py` through `app_v5.py`) are kept for reference/comparison.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Plotly, Streamlit

## Author
Balamurugan Renganathan
