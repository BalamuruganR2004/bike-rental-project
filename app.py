import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']

model, feature_cols = load_model()

st.title("🚲 Bike Rental Demand Predictor")
st.caption("Predicts hourly bike rental demand using weather, season, and time features. Model: Gradient Boosting Regressor (R² = 0.95)")

col1, col2 = st.columns(2)

with col1:
    yr = st.selectbox("Year", [2011, 2012], format_func=lambda x: str(x))
    mnth = st.slider("Month", 1, 12, 6)
    hr = st.slider("Hour of Day", 0, 23, 8)
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    weekday = st.selectbox("Weekday", ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"])

with col2:
    holiday = st.checkbox("Holiday")
    workingday = st.checkbox("Working Day", value=True)
    weathersit = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
    temp = st.slider("Temperature (normalized 0-1)", 0.0, 1.0, 0.5)
    atemp = st.slider("Feels-like Temp (normalized 0-1)", 0.0, 1.0, 0.48)
    hum = st.slider("Humidity (normalized 0-1)", 0.0, 1.0, 0.6)
    windspeed = st.slider("Windspeed (normalized 0-1)", 0.0, 1.0, 0.2)

if st.button("Predict Demand", type="primary"):
    season_map = {"Spring":1, "Summer":2, "Fall":3, "Winter":4}
    weather_map = {"Clear":1, "Mist":2, "Light Snow":3, "Heavy Rain":4}
    weekday_map = {"Sun":0,"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5,"Sat":6}

    row = {
        'yr': 1 if yr == 2012 else 0,
        'holiday': int(holiday),
        'workingday': int(workingday),
        'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed,
        'hr_sin': np.sin(2*np.pi*hr/24), 'hr_cos': np.cos(2*np.pi*hr/24),
        'mnth_sin': np.sin(2*np.pi*mnth/12), 'mnth_cos': np.cos(2*np.pi*mnth/12),
        'is_rush_hour': int(hr in [7,8,9,17,18,19]),
        'is_weekend': int(weekday_map[weekday] in [0,6]),
        'temp_hum_interaction': temp * hum,
    }
    for s in [1,2,3,4]:
        row[f'season_{s}'] = int(season_map[season] == s)
    for w in [1,2,3,4]:
        row[f'weather_{w}'] = int(weather_map[weathersit] == w)

    X = pd.DataFrame([row])[feature_cols]
    pred = model.predict(X)[0]
    st.success(f"### Predicted Rentals: **{int(max(0,pred))}** bikes")
    st.caption("Prediction is for this specific hour, based on the trained model.")
