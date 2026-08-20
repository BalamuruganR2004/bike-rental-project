import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
import pickle

df = pd.read_csv('cleaned_data.csv', parse_dates=['dteday'])
df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)

# ---- Base features (same as before) ----
df['hr_sin'] = np.sin(2*np.pi*df['hr']/24)
df['hr_cos'] = np.cos(2*np.pi*df['hr']/24)
df['mnth_sin'] = np.sin(2*np.pi*df['mnth']/12)
df['mnth_cos'] = np.cos(2*np.pi*df['mnth']/12)
df['is_rush_hour'] = df['hr'].isin([7,8,9,17,18,19]).astype(int)
df['is_weekend'] = df['weekday'].isin([0,6]).astype(int)
df['temp_hum_interaction'] = df['temp'] * df['hum']

# ---- NEW: Time-series lag & rolling features ----
# Previous hour's demand (autocorrelation is usually the single strongest signal in hourly series)
df['cnt_lag1'] = df['cnt'].shift(1)
df['cnt_lag24'] = df['cnt'].shift(24)   # same hour, previous day
# Rolling mean of last 3 hours (smoothed recent trend)
df['cnt_roll3'] = df['cnt'].shift(1).rolling(window=3, min_periods=1).mean()
# Fill the first day's NaNs (no prior history) with the column median
for c in ['cnt_lag1', 'cnt_lag24', 'cnt_roll3']:
    df[c] = df[c].fillna(df[c].median())

df = pd.get_dummies(df, columns=['season', 'weathersit'], prefix=['season', 'weather'])

feature_cols = [c for c in df.columns if c not in
                 ['instant', 'dteday', 'casual', 'registered', 'cnt', 'hr', 'mnth', 'weekday']]
X = df[feature_cols]
y = df['cnt']

# Time-respecting split: train on earlier data, test on later (avoids leakage from lag features)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

def evaluate(name, preds, y_test):
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

results = []

# ---- XGBoost with tuning ----
print("Tuning XGBoost...")
xgb_params = {'n_estimators':[200,300,400], 'max_depth':[4,6,8], 'learning_rate':[0.03,0.05,0.1], 'subsample':[0.8,1.0]}
xgb_search = RandomizedSearchCV(xgb.XGBRegressor(random_state=42, n_jobs=-1), xgb_params, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
preds_xgb = xgb_best.predict(X_test)
results.append(evaluate('XGBoost (tuned, with lag features)', preds_xgb, y_test))
print("Best params:", xgb_search.best_params_)

# ---- Gradient Boosting with lag features (re-run for fair comparison) ----
gbr = GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, subsample=0.8, random_state=42)
gbr.fit(X_train, y_train)
preds_gbr = gbr.predict(X_test)
results.append(evaluate('Gradient Boosting (with lag features)', preds_gbr, y_test))

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
print("\n=== Comparison with lag features ===")
print(results_df.to_string(index=False))
results_df.to_csv('model_results_advanced.csv', index=False)

# Pick best of the two
best_model = xgb_best if results_df.iloc[0]['model'].startswith('XGBoost') else gbr
best_preds = preds_xgb if best_model is xgb_best else preds_gbr

with open('best_model_advanced.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'features': feature_cols}, f)
print(f"\nSaved best_model_advanced.pkl ({results_df.iloc[0]['model']})")

# ---- Residual Analysis ----
residuals = y_test.values - best_preds
plt.figure(figsize=(9,5))
plt.scatter(best_preds, residuals, alpha=0.3, s=10, color='#2b6cb0')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted cnt'); plt.ylabel('Residual (actual - predicted)')
plt.title('Residual Plot - Best Model')
plt.tight_layout()
plt.savefig('plot_10_residuals.png')
plt.close()

plt.figure(figsize=(9,5))
plt.hist(residuals, bins=50, color='#2b6cb0', edgecolor='white')
plt.title('Residual Distribution')
plt.xlabel('Residual'); plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plot_11_residual_dist.png')
plt.close()
print("Saved residual plots")

# ---- SHAP Explainability ----
print("Computing SHAP values (sample of 500 rows)...")
sample = X_test.sample(min(500, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False, max_display=12)
plt.tight_layout()
plt.savefig('plot_12_shap_summary.png', dpi=110, bbox_inches='tight')
plt.close()
print("Saved SHAP summary plot")

# ---- Prediction Intervals (Quantile Regression via GBR) ----
print("Training quantile models for 90% prediction interval...")
gbr_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gbr_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gbr_lower.fit(X_train, y_train)
gbr_upper.fit(X_train, y_train)

with open('quantile_models.pkl', 'wb') as f:
    pickle.dump({'lower': gbr_lower, 'upper': gbr_upper, 'features': feature_cols}, f)

lower_preds = gbr_lower.predict(X_test.iloc[:200])
upper_preds = gbr_upper.predict(X_test.iloc[:200])
actual_sample = y_test.iloc[:200].values
coverage = np.mean((actual_sample >= lower_preds) & (actual_sample <= upper_preds))
print(f"90% prediction interval empirical coverage: {coverage:.1%}")

plt.figure(figsize=(12,5))
idx = np.arange(200)
plt.fill_between(idx, lower_preds, upper_preds, alpha=0.3, color='#2b6cb0', label='90% Prediction Interval')
plt.plot(idx, actual_sample, color='black', linewidth=1, label='Actual')
plt.title(f'Prediction Intervals vs Actual (sample) — {coverage:.0%} empirical coverage')
plt.xlabel('Test sample index'); plt.ylabel('cnt')
plt.legend()
plt.tight_layout()
plt.savefig('plot_13_prediction_intervals.png')
plt.close()
print("Saved prediction interval plot")

print("\nAll advanced analysis complete.")
