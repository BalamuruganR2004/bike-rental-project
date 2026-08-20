import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

df = pd.read_csv('cleaned_data.csv', parse_dates=['dteday'])

# ---- Feature Engineering ----
# Cyclical encoding for hour and month (captures periodicity, e.g. hr=23 close to hr=0)
df['hr_sin'] = np.sin(2*np.pi*df['hr']/24)
df['hr_cos'] = np.cos(2*np.pi*df['hr']/24)
df['mnth_sin'] = np.sin(2*np.pi*df['mnth']/12)
df['mnth_cos'] = np.cos(2*np.pi*df['mnth']/12)

# Rush hour flag (commute peaks)
df['is_rush_hour'] = df['hr'].isin([7,8,9,17,18,19]).astype(int)

# Weekend flag
df['is_weekend'] = df['weekday'].isin([0,6]).astype(int)

# Temperature-humidity interaction (feels-hotter effect)
df['temp_hum_interaction'] = df['temp'] * df['hum']

# One-hot encode season and weathersit (nominal categories)
df = pd.get_dummies(df, columns=['season', 'weathersit'], prefix=['season', 'weather'])

# Drop leakage columns (casual+registered = cnt) and non-feature columns
feature_cols = [c for c in df.columns if c not in
                 ['instant', 'dteday', 'casual', 'registered', 'cnt', 'hr', 'mnth', 'weekday']]
X = df[feature_cols]
y = df['cnt']

print(f"Feature count: {X.shape[1]}")
print("Features:", list(X.columns))

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

results = []

# ---- Baseline models ----
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
results.append(evaluate('Decision Tree (baseline)', dt, X_test, y_test))

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate('Random Forest (baseline)', rf, X_test, y_test))

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
results.append(evaluate('Gradient Boosting (baseline)', gbr, X_test, y_test))

# ---- Hyperparameter tuning ----
print("\nTuning Decision Tree...")
dt_params = {'max_depth': [5,10,15,20,None], 'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,4]}
dt_search = RandomizedSearchCV(DecisionTreeRegressor(random_state=42), dt_params, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
dt_search.fit(X_train, y_train)
results.append(evaluate('Decision Tree (tuned)', dt_search.best_estimator_, X_test, y_test))
print("Best params:", dt_search.best_params_)

print("\nTuning Random Forest...")
rf_params = {'n_estimators': [100,200,300], 'max_depth': [10,20,30,None], 'min_samples_split': [2,5,10], 'max_features': ['sqrt','log2']}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_params, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
results.append(evaluate('Random Forest (tuned)', rf_search.best_estimator_, X_test, y_test))
print("Best params:", rf_search.best_params_)

print("\nTuning Gradient Boosting...")
gbr_params = {'n_estimators': [100,200,300], 'max_depth': [3,5,7], 'learning_rate': [0.01,0.05,0.1], 'subsample': [0.8,1.0]}
gbr_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), gbr_params, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
gbr_search.fit(X_train, y_train)
results.append(evaluate('Gradient Boosting (tuned)', gbr_search.best_estimator_, X_test, y_test))
print("Best params:", gbr_search.best_params_)

# ---- Results comparison ----
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
print("\n=== Final Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv('model_results.csv', index=False)

# Save best model info
best = results_df.iloc[0]
best_models = {'Decision Tree (tuned)': dt_search.best_estimator_,
                'Random Forest (tuned)': rf_search.best_estimator_,
                'Gradient Boosting (tuned)': gbr_search.best_estimator_,
                'Decision Tree (baseline)': dt,
                'Random Forest (baseline)': rf,
                'Gradient Boosting (baseline)': gbr}
best_model = best_models[best['model']]

import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'features': feature_cols}, f)
print(f"\nBest model: {best['model']} saved as best_model.pkl")

# ---- Feature importance plot (best model) ----
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
    plt.figure(figsize=(9,6))
    imp.plot(kind='barh', color='#2b6cb0')
    plt.gca().invert_yaxis()
    plt.title(f'Top Feature Importances - {best["model"]}')
    plt.tight_layout()
    plt.savefig('plot_08_feature_importance.png')
    plt.close()
    print("Saved feature importance plot")

# ---- Model comparison bar chart ----
plt.figure(figsize=(9,5))
plt.bar(results_df['model'], results_df['R2'], color='#2b6cb0')
plt.xticks(rotation=45, ha='right')
plt.ylabel('R2 Score')
plt.title('Model Comparison - R2 Score')
plt.tight_layout()
plt.savefig('plot_09_model_comparison.png')
plt.close()
print("Saved model comparison plot")
