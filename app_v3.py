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
from streamlit_option_menu import option_menu

st.set_page_config(page_title="PedalIQ — Demand Forecasting", page_icon="🚲", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Design tokens — modern SaaS product (Stripe/Linear-adjacent): light neutral
# canvas, white cards with soft shadow, a crisp "signal blue" primary and an
# emerald success accent. Manrope for headers, Inter for body/UI text.
# ---------------------------------------------------------------------------
BG        = "#F6F7FB"
SURFACE   = "#FFFFFF"
BORDER    = "#E7E9F0"
TEXT      = "#111827"
MUTED     = "#6B7280"
PRIMARY   = "#2F5EFF"
PRIMARY_L = "#EEF2FF"
SUCCESS   = "#12B76A"
SUCCESS_L = "#ECFDF3"
AMBER     = "#F79009"
SHADOW    = "0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

.stApp {{ background: {BG}; color: {TEXT}; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Manrope', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.02em; color: {TEXT}; }}

/* Top bar */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 2px 20px 2px;
}}
.brand {{ display: flex; align-items: center; gap: 10px; }}
.brand-mark {{
    width: 34px; height: 34px; border-radius: 9px;
    background: {PRIMARY}; display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}}
.brand-name {{ font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 18px; color: {TEXT}; }}
.brand-tag {{ font-size: 12px; color: {MUTED}; }}
.badge-live {{
    background: {SUCCESS_L}; color: {SUCCESS}; font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;
}}
.badge-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {SUCCESS}; display: inline-block; }}

/* Page heading */
.page-eyebrow {{ font-size: 12px; font-weight: 600; color: {PRIMARY}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
.page-title {{ font-size: 26px; margin-bottom: 2px; }}
.page-sub {{ color: {MUTED}; font-size: 14px; margin-bottom: 22px; }}

/* Cards */
.card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 20px 22px; box-shadow: {SHADOW}; margin-bottom: 18px;
}}
.card-title {{ font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 14px; color: {TEXT}; margin-bottom: 14px; }}

/* KPI stat cards */
.stat-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 16px 18px; box-shadow: {SHADOW};
}}
.stat-icon {{
    width: 32px; height: 32px; border-radius: 8px; background: {PRIMARY_L};
    display: flex; align-items: center; justify-content: center; font-size: 15px; margin-bottom: 10px;
}}
.stat-value {{ font-family: 'Manrope', sans-serif; font-size: 24px; font-weight: 800; color: {TEXT}; }}
.stat-label {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}

/* Result panel */
.result-card {{
    background: linear-gradient(135deg, {PRIMARY_L} 0%, {SURFACE} 70%);
    border: 1px solid {BORDER}; border-radius: 14px; padding: 26px; text-align: center;
    box-shadow: {SHADOW};
}}
.result-value {{ font-family: 'Manrope', sans-serif; font-size: 42px; font-weight: 800; color: {PRIMARY}; }}
.result-caption {{ color: {MUTED}; font-size: 13px; margin-top: 4px; }}
.empty-state {{ text-align:center; padding: 70px 20px; color: {MUTED}; font-size: 14px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 18px; }}

/* Streamlit widget resets */
div[data-baseweb="select"] > div, .stNumberInput input {{
    background: {SURFACE} !important; border-color: {BORDER} !important; color: {TEXT} !important; border-radius: 8px !important;
}}
.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {{
    color: {MUTED} !important; font-size: 13px !important; font-weight: 500 !important;
}}
button[kind="primary"] {{
    background: {PRIMARY} !important; color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 9px !important; box-shadow: {SHADOW};
}}
hr {{ border-color: {BORDER}; }}
[data-testid="stMetricValue"] {{ color: {TEXT}; }}
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

results = pd.read_csv(os.path.join(BASE_DIR, 'model_results_advanced.csv'))
best = results.iloc[results['R2'].idxmax()]

# ---------------------------------------------------------------------------
# Sidebar nav
with st.sidebar:
    st.markdown(f"""
    <div class="brand" style="margin-bottom:24px;">
        <div class="brand-mark">🚲</div>
        <div>
            <div class="brand-name">PedalIQ</div>
            <div class="brand-tag">Demand Forecasting</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Predict", "Analytics", "Model"],
        icons=["lightning-charge", "graph-up", "cpu"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": SURFACE},
            "icon": {"color": MUTED, "font-size": "15px"},
            "nav-link": {
                "font-family": "Inter", "font-size": "14px", "font-weight": "500",
                "color": MUTED, "border-radius": "8px", "margin": "2px 0", "padding": "10px 12px",
            },
            "nav-link-selected": {"background-color": PRIMARY_L, "color": PRIMARY, "font-weight": "600"},
        },
    )

    st.markdown(f"""
    <div style="margin-top:28px; padding-top:16px; border-top:1px solid {BORDER};">
        <div class="badge-live"><span class="badge-dot"></span>Model v2 · Live</div>
        <div style="color:{MUTED}; font-size:12px; margin-top:10px;">
            Gradient Boosting Regressor<br>trained on 17K+ hourly records
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top bar (KPIs shown on every page for context)
kc1, kc2, kc3, kc4 = st.columns(4)
kpis = [
    ("📊", f"{best['R2']:.3f}", "R² Score"),
    ("🎯", f"{best['MAE']:.1f}", "Mean Abs. Error"),
    ("📐", f"{best['RMSE']:.1f}", "RMSE"),
    ("✅", "92.5%", "Interval Coverage"),
]
for col, (icon, val, lbl) in zip([kc1,kc2,kc3,kc4], kpis):
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">{icon}</div>
        <div class="stat-value">{val}</div>
        <div class="stat-label">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===========================================================================
if page == "Predict":
    st.markdown('<div class="page-eyebrow">Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Hourly demand prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Set conditions to forecast rental demand for a specific hour, with a 90% confidence range.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Conditions</div>', unsafe_allow_html=True)
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
        temp = st.slider("Temperature (norm.)", 0.0, 1.0, 0.5)
        atemp = st.slider("Feels-like (norm.)", 0.0, 1.0, 0.48)
        hum = st.slider("Humidity (norm.)", 0.0, 1.0, 0.6)
        windspeed = st.slider("Windspeed (norm.)", 0.0, 1.0, 0.2)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Recent demand (lag context)</div>', unsafe_allow_html=True)
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
                'yr': 1 if yr == 2012 else 0, 'holiday': int(holiday), 'workingday': int(workingday),
                'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed,
                'hr_sin': np.sin(2*np.pi*hr/24), 'hr_cos': np.cos(2*np.pi*hr/24),
                'mnth_sin': np.sin(2*np.pi*mnth/12), 'mnth_cos': np.cos(2*np.pi*mnth/12),
                'is_rush_hour': int(hr in [7,8,9,17,18,19]),
                'is_weekend': int(weekday_map[weekday] in [0,6]),
                'temp_hum_interaction': temp * hum,
                'cnt_lag1': cnt_lag1, 'cnt_lag24': cnt_lag24, 'cnt_roll3': cnt_roll3,
            }
            for s in [1,2,3,4]: row[f'season_{s}'] = int(season_map[season] == s)
            for w in [1,2,3,4]: row[f'weather_{w}'] = int(weather_map[weathersit] == w)

            X = pd.DataFrame([row])[feature_cols]
            pred = max(0, model.predict(X)[0])
            lower = max(0, q_lower.predict(X)[0])
            upper = max(0, q_upper.predict(X)[0])

            st.markdown(f"""
            <div class="result-card">
              <div class="result-value">{int(pred)} bikes</div>
              <div class="result-caption">predicted demand this hour · 90% range {int(lower)}–{int(upper)}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pred,
                number={'font': {'color': PRIMARY, 'family': 'Manrope'}, 'suffix': ''},
                gauge={
                    'axis': {'range': [0, 1000], 'tickcolor': MUTED},
                    'bar': {'color': PRIMARY},
                    'bgcolor': SURFACE, 'borderwidth': 1, 'bordercolor': BORDER,
                    'steps': [
                        {'range': [0, 300], 'color': "#F3F4F8"},
                        {'range': [300, 650], 'color': "#E7EBFB"},
                        {'range': [650, 1000], 'color': "#D8E0FA"},
                    ],
                    'threshold': {'line': {'color': AMBER, 'width': 3}, 'thickness': 0.8, 'value': upper},
                },
            ))
            fig.update_layout(height=240, margin=dict(l=20,r=20,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': TEXT})
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Why this prediction — top contributing factors</div>', unsafe_allow_html=True)
            shap_values = explainer(X)
            fig2, ax = plt.subplots(figsize=(8, 4.3))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            fig2.patch.set_facecolor(SURFACE)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card empty-state">Set conditions on the left and run a prediction to see the forecast, confidence range, and driving factors.</div>', unsafe_allow_html=True)

# ===========================================================================
elif page == "Analytics":
    st.markdown('<div class="page-eyebrow">Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Demand patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">How rentals vary by time, season, and weather across the dataset.</div>', unsafe_allow_html=True)

    imgs = [
        ("plot_02_hourly_workday.png", "Hourly demand: working day vs non-working day"),
        ("plot_04_season.png", "Total rentals by season"),
        ("plot_05_weather.png", "Rental distribution by weather"),
        ("plot_03_casual_vs_registered.png", "Casual vs registered riders by weekday"),
    ]
    for i in range(0, len(imgs), 2):
        c1, c2 = st.columns(2)
        for col, (fname, caption) in zip([c1, c2], imgs[i:i+2]):
            with col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">{caption}</div>', unsafe_allow_html=True)
                st.image(os.path.join(BASE_DIR, fname), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Daily rentals over time (2011–2012)</div>', unsafe_allow_html=True)
    st.image(os.path.join(BASE_DIR, 'plot_01_timeseries.png'), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
elif page == "Model":
    st.markdown('<div class="page-eyebrow">Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Model evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Comparison across candidate models, explainability, and residual diagnostics.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Model comparison (R²)</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_09_model_comparison.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Global feature importance (SHAP)</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_12_shap_summary.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Residuals</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_10_residuals.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">90% prediction interval coverage</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_13_prediction_intervals.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Full results table</div>', unsafe_allow_html=True)
    st.dataframe(results.style.format({'MAE':'{:.2f}','RMSE':'{:.2f}','R2':'{:.4f}'}), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
