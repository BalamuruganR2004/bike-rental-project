import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Design tokens: "transit control room" — deep navy canvas, teal live-data
# accent, amber for peak/alert states. Space Grotesk for headers (technical,
# not cold), JetBrains Mono for numbers since this is a metrics-heavy tool.
# ---------------------------------------------------------------------------
NAVY      = "#0B1C2C"
PANEL     = "#122A40"
PANEL_2   = "#0F2438"
TEAL      = "#2DD4BF"
AMBER     = "#F5A623"
TEXT      = "#E6EDF3"
MUTED     = "#7C93A8"
BORDER    = "#1E3A52"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {{
    background: {NAVY};
    color: {TEXT};
}}
html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}}
.mono {{
    font-family: 'JetBrains Mono', monospace;
}}

/* Hero */
.hero {{
    padding: 28px 32px;
    border-radius: 14px;
    background: linear-gradient(135deg, {PANEL} 0%, {PANEL_2} 100%);
    border: 1px solid {BORDER};
    margin-bottom: 24px;
}}
.hero-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    color: {TEAL};
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.hero-title {{
    font-size: 30px;
    font-weight: 700;
    color: {TEXT};
    margin: 0 0 6px 0;
}}
.hero-sub {{
    color: {MUTED};
    font-size: 14.5px;
    max-width: 640px;
}}

/* Panels */
.panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
}}
.panel-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {TEAL};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
}}

/* Metric cards */
.metric-card {{
    background: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}}
.metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: {TEAL};
}}
.metric-label {{
    font-size: 11px;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}}

/* Result banner */
.result-banner {{
    background: linear-gradient(135deg, rgba(45,212,191,0.12), rgba(45,212,191,0.02));
    border: 1px solid {TEAL};
    border-radius: 12px;
    padding: 22px 26px;
    text-align: center;
}}
.result-number {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 44px;
    font-weight: 700;
    color: {TEAL};
    line-height: 1.1;
}}
.result-caption {{
    color: {MUTED};
    font-size: 13px;
    margin-top: 6px;
}}

/* Streamlit widget overrides */
[data-testid="stSidebar"] {{
    background: {PANEL_2};
}}
div[data-baseweb="select"] > div, .stNumberInput input {{
    background: {PANEL_2} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
}}
.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {{
    color: {MUTED} !important;
    font-size: 13px !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: {PANEL};
    border-radius: 8px 8px 0 0;
    color: {MUTED};
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}}
.stTabs [aria-selected="true"] {{
    color: {TEAL} !important;
    border-bottom: 2px solid {TEAL} !important;
}}
button[kind="primary"] {{
    background: {TEAL} !important;
    color: {NAVY} !important;
    font-weight: 700 !important;
    border: none !important;
}}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, 'best_model_advanced.pkl'), 'rb') as f:
        main = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'quantile_models.pkl'), 'rb') as f:
        q = pickle.load(f)
    return main['model'], main['features'], q['lower'], q['upper']

model, feature_cols, q_lower, q_upper = load_models()

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# ---------------------------------------------------------------------------
# Hero
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">Bike-Share Operations · Demand Forecast</div>
  <div class="hero-title">🚲 Hourly Rental Demand Predictor</div>
  <div class="hero-sub">Gradient Boosting model trained on 2 years of hourly rental data with
  time-series lag features. R² = 0.953 on held-out data, with calibrated 90% prediction intervals
  for operational planning.</div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_trends, tab_model = st.tabs(["⚡ PREDICT", "📈 TRENDS", "🧠 MODEL"])

# ===========================================================================
with tab_predict:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Conditions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            yr = st.selectbox("Year", [2011, 2012])
            mnth = st.slider("Month", 1, 12, 6)
            hr = st.slider("Hour", 0, 23, 8)
            season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        with c2:
            weekday = st.selectbox("Weekday", ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"])
            weathersit = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
            holiday = st.checkbox("Holiday")
            workingday = st.checkbox("Working day", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        temp = st.slider("Temperature (norm.)", 0.0, 1.0, 0.5)
        atemp = st.slider("Feels-like (norm.)", 0.0, 1.0, 0.48)
        hum = st.slider("Humidity (norm.)", 0.0, 1.0, 0.6)
        windspeed = st.slider("Windspeed (norm.)", 0.0, 1.0, 0.2)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Recent Demand (lag context)</div>', unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        with l1:
            cnt_lag1 = st.number_input("Last hour", min_value=0, value=150)
        with l2:
            cnt_lag24 = st.number_input("Same hour yesterday", min_value=0, value=150)
        with l3:
            cnt_roll3 = st.number_input("Avg last 3h", min_value=0, value=150)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    with right:
        if predict_clicked:
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
                'cnt_lag1': cnt_lag1, 'cnt_lag24': cnt_lag24, 'cnt_roll3': cnt_roll3,
            }
            for s in [1,2,3,4]:
                row[f'season_{s}'] = int(season_map[season] == s)
            for w in [1,2,3,4]:
                row[f'weather_{w}'] = int(weather_map[weathersit] == w)

            X = pd.DataFrame([row])[feature_cols]
            pred = max(0, model.predict(X)[0])
            lower = max(0, q_lower.predict(X)[0])
            upper = max(0, q_upper.predict(X)[0])

            st.markdown(f"""
            <div class="result-banner">
              <div class="result-number">{int(pred)}</div>
              <div class="result-caption">predicted rentals this hour · 90% range {int(lower)}–{int(upper)}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={'font': {'color': TEAL, 'family': 'JetBrains Mono'}, 'suffix': ' bikes'},
                gauge={
                    'axis': {'range': [0, 1000], 'tickcolor': MUTED},
                    'bar': {'color': TEAL},
                    'bgcolor': PANEL_2,
                    'borderwidth': 1,
                    'bordercolor': BORDER,
                    'steps': [
                        {'range': [0, 300], 'color': PANEL_2},
                        {'range': [300, 650], 'color': '#1A3A50'},
                        {'range': [650, 1000], 'color': '#234A63'},
                    ],
                    'threshold': {'line': {'color': AMBER, 'width': 3}, 'thickness': 0.8, 'value': upper},
                },
                domain={'x': [0, 1], 'y': [0, 1]},
            ))
            fig.update_layout(
                height=260, margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', font={'color': TEXT}
            )
            st.plotly_chart(fig, use_container_width=True)

            # SHAP waterfall for this exact prediction
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-label">Why this prediction — top contributing factors</div>', unsafe_allow_html=True)
            shap_values = explainer(X)
            fig2, ax = plt.subplots(figsize=(8, 4.5))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            fig2.patch.set_facecolor(PANEL)
            ax.set_facecolor(PANEL)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="panel" style="text-align:center; padding:60px 20px;">
              <div style="color:{MUTED}; font-size:14px;">Set conditions on the left and run a prediction
              to see the forecast, confidence range, and driving factors.</div>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
with tab_trends:
    st.markdown('<div class="panel-label" style="margin-bottom:14px;">Demand Patterns</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.image(os.path.join(BASE_DIR, 'plot_02_hourly_workday.png'), use_container_width=True)
    with r1c2:
        st.image(os.path.join(BASE_DIR, 'plot_04_season.png'), use_container_width=True)
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.image(os.path.join(BASE_DIR, 'plot_05_weather.png'), use_container_width=True)
    with r2c2:
        st.image(os.path.join(BASE_DIR, 'plot_03_casual_vs_registered.png'), use_container_width=True)
    st.image(os.path.join(BASE_DIR, 'plot_01_timeseries.png'), use_container_width=True)

# ===========================================================================
with tab_model:
    results = pd.read_csv(os.path.join(BASE_DIR, 'model_results_advanced.csv'))
    best = results.iloc[results['R2'].idxmax()]

    m1, m2, m3 = st.columns(3)
    for col, label, val in zip([m1,m2,m3], ["R² Score", "RMSE", "MAE"],
                                 [f"{best['R2']:.3f}", f"{best['RMSE']:.1f}", f"{best['MAE']:.1f}"]):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel-label">Model Comparison</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_09_model_comparison.png'), use_container_width=True)
    with c2:
        st.markdown('<div class="panel-label">Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_12_shap_summary.png'), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="panel-label">Residuals</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_10_residuals.png'), use_container_width=True)
    with c4:
        st.markdown('<div class="panel-label">90% Prediction Interval Coverage</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_13_prediction_intervals.png'), use_container_width=True)
