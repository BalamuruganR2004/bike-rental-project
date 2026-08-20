import pandas as pd
import numpy as np

df = pd.read_csv('Dataset.csv')

# Replace '?' placeholders with proper NaN across all columns
df = df.replace('?', np.nan)

# Fix season typo
df['season'] = df['season'].replace({'springer': 'spring'})

# Convert numeric columns stored as text
numeric_cols = ['yr', 'mnth', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Parse date
df['dteday'] = pd.to_datetime(df['dteday'], format='%d-%m-%Y', errors='coerce')

# Standardize categorical text
df['holiday'] = df['holiday'].map({'No': 0, 'Yes': 1})
df['workingday'] = df['workingday'].map({'No work': 0, 'Working Day': 1})

season_map = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4}
df['season'] = df['season'].map(season_map)

weather_map = {'Clear': 1, 'Mist': 2, 'Light Snow': 3, 'Heavy Rain': 4}
df['weathersit'] = df['weathersit'].map(weather_map)

print("Missing values before imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Impute dteday from yr/mnth pattern is unreliable with few missing -> drop rows with missing dteday/yr/mnth/season/holiday/workingday/weathersit (categorical, low count)
cat_cols_to_dropna = ['dteday', 'yr', 'mnth', 'season', 'holiday', 'workingday', 'weathersit']
before = len(df)
df = df.dropna(subset=cat_cols_to_dropna)
print(f"\nDropped {before - len(df)} rows with missing categorical/date fields")

# Impute numeric weather columns with median (time-series data, low missing %)
for c in ['temp', 'atemp', 'hum', 'windspeed']:
    df[c] = df[c].fillna(df[c].median())

# Impute casual/registered with median, then reconcile cnt if needed
for c in ['casual', 'registered']:
    df[c] = df[c].fillna(df[c].median())

# Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# Cap outliers in windspeed and hum using IQR (weather sensor errors are common)
for c in ['windspeed', 'hum', 'temp', 'atemp']:
    Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df[c] = df[c].clip(lower, upper)

df['season'] = df['season'].astype(int)
df['weathersit'] = df['weathersit'].astype(int)
df['holiday'] = df['holiday'].astype(int)
df['workingday'] = df['workingday'].astype(int)
df['yr'] = df['yr'].astype(int)
df['mnth'] = df['mnth'].astype(int)
df['casual'] = df['casual'].astype(int)
df['registered'] = df['registered'].astype(int)

print(f"\nFinal shape: {df.shape}")
print(df.isnull().sum().sum(), "total missing values remaining")

df.to_csv('cleaned_data.csv', index=False)
print("\nSaved cleaned_data.csv")
