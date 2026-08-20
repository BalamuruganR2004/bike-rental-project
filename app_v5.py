import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from streamlit_option_menu import option_menu

st.set_page_config(page_title="PedalIQ — Bike Rental Dashboard", page_icon="🚲", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NAVY      = "#14152B"
NAVY_2    = "#1C1D3A"
BG        = "#F7F6FC"
SURFACE   = "#FFFFFF"
BORDER    = "#ECEAF6"
TEXT      = "#1A1A2E"
MUTED     = "#8A8AA3"
SUB       = "#6B6B85"
TEAL      = "#14B8A6"
TEAL_L    = "#CCFBF1"
PURPLE    = "#7B68EE"
PURPLE_L  = "#EDEBFE"
GREEN     = "#34D399"
GREEN_L   = "#DFF9EE"
ORANGE    = "#F5A623"
ORANGE_L  = "#FFF1DB"
PINK      = "#F26D96"
PINK_L    = "#FDE6ED"
BLUE      = "#5B8DEF"
BLUE_L    = "#E4EDFF"
SHADOW    = "0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06)"

with open(os.path.join(BASE_DIR, 'dashboard_stats.json')) as f:
    STATS = json.load(f)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
.stApp {{ background: {BG}; color: {TEXT}; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Poppins', sans-serif !important; font-weight: 600 !important; color: {TEXT}; }}
.page-title {{ font-size: 24px; font-weight: 600; margin: 0; color: {TEXT}; font-family:'Poppins',sans-serif; }}
.page-sub {{ color: {SUB}; font-size: 13.5px; margin: 2px 0 0; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 20px; box-shadow: {SHADOW}; margin-bottom: 16px; }}
.card-title {{ font-family:'Poppins',sans-serif; font-weight: 600; font-size: 14px; color: {TEXT}; margin-bottom: 12px; }}
.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px; box-shadow: {SHADOW}; }}
.kpi-icon {{ width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 16px; margin-bottom: 10px; }}
.kpi-val {{ font-family:'Poppins',sans-serif; font-size: 21px; font-weight: 600; color: {TEXT}; }}
.kpi-lbl {{ font-size: 11.5px; color: {MUTED}; margin-top: 1px; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #fff; }}
div[data-baseweb="select"] > div, .stNumberInput input {{ background: {SURFACE} !important; border-color: {BORDER} !important; color: {TEXT} !important; border-radius: 8px !important; }}
.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {{ color: {SUB} !important; font-size: 13px !important; font-weight: 500 !important; }}
button[kind="primary"] {{ background: {TEAL} !important; color: white !important; font-weight: 600 !important; border: none !important; border-radius: 9px !important; box-shadow: {SHADOW}; }}
hr {{ border-color: {BORDER}; }}
.sidebar-brand {{ display:flex; align-items:center; gap:10px; margin-bottom:22px; }}
.sidebar-mark {{ width:32px; height:32px; border-radius:9px; background:{TEAL}; display:flex; align-items:center; justify-content:center; font-size:16px; }}
.sidebar-name {{ font-family:'Poppins',sans-serif; font-weight:700; font-size:16px; color:#fff; margin:0; }}
.sidebar-tag {{ font-size:10.5px; color:#8A8AA3; margin:0; }}
.sidebar-callout {{ background: linear-gradient(160deg, {NAVY_2}, {NAVY}); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px; margin-top:18px; }}
</style>
""", unsafe_allow_html=True)

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

SEASON_MAP = {"Spring":1, "Summer":2, "Fall":3, "Winter":4}
WEATHER_MAP = {"Clear":1, "Mist":2, "Light Snow":3, "Heavy Rain":4}
WEEKDAY_MAP = {"Sun":0,"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5,"Sat":6}

def build_row(hr, cnt_lag1, cnt_lag24, cnt_roll3, cond):
    row = {
        'yr': 1 if cond['yr'] == 2012 else 0,
        'holiday': int(cond['holiday']), 'workingday': int(cond['workingday']),
        'temp': cond['temp'], 'atemp': cond['atemp'], 'hum': cond['hum'], 'windspeed': cond['windspeed'],
        'hr_sin': np.sin(2*np.pi*hr/24), 'hr_cos': np.cos(2*np.pi*hr/24),
        'mnth_sin': np.sin(2*np.pi*cond['mnth']/12), 'mnth_cos': np.cos(2*np.pi*cond['mnth']/12),
        'is_rush_hour': int(hr in [7,8,9,17,18,19]),
        'is_weekend': int(WEEKDAY_MAP[cond['weekday']] in [0,6]),
        'temp_hum_interaction': cond['temp'] * cond['hum'],
        'cnt_lag1': cnt_lag1, 'cnt_lag24': cnt_lag24, 'cnt_roll3': cnt_roll3,
    }
    for s in [1,2,3,4]: row[f'season_{s}'] = int(SEASON_MAP[cond['season']] == s)
    for w in [1,2,3,4]: row[f'weather_{w}'] = int(WEATHER_MAP[cond['weathersit']] == w)
    return pd.DataFrame([row])[feature_cols]

def simulate_day(cond, seed_lag):
    lag1 = seed_lag; lag24 = seed_lag; roll_hist = [seed_lag]*3
    preds = []
    for hr in range(24):
        X = build_row(hr, lag1, lag24, np.mean(roll_hist[-3:]), cond)
        p = max(0, float(model.predict(X)[0]))
        preds.append(p); lag1 = p; roll_hist.append(p)
    return preds

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="sidebar-mark">🚲</div>
        <div><p class="sidebar-name">PedalIQ</p><p class="sidebar-tag">Rental Dashboard</p></div>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        menu_title=None, options=["Overview", "Predict", "Model"],
        icons=["grid-1x2", "lightning-charge", "cpu"], default_index=0,
        styles={
            "container": {"padding": "0", "background-color": NAVY},
            "icon": {"color": "#8A8AA3", "font-size": "15px"},
            "nav-link": {"font-family": "Inter", "font-size": "14px", "font-weight": "500", "color": "#B4B4C9", "border-radius": "8px", "margin": "2px 0", "padding": "10px 12px", "background-color": NAVY},
            "nav-link-selected": {"background-color": TEAL, "color": "#fff", "font-weight": "600"},
        },
    )

    st.markdown(f"""
    <div class="sidebar-callout">
        <p style="font-size:12.5px;font-weight:600;color:#fff;margin:0 0 4px;">Model trained on real data</p>
        <p style="font-size:11px;color:#9E9EB5;margin:0 0 10px;">17,361 hourly records across 2 full years, cleaned and feature-engineered.</p>
        <p style="font-size:11px;color:{TEAL};margin:0;font-weight:600;">● Live · Gradient Boosting v4</p>
    </div>
    """, unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<p class="page-title">Good day, Bala 👋</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Here\'s what the training data says about the bike-share system.</p><br>', unsafe_allow_html=True)

    kpis = [
        ("ti-bike", TEAL, TEAL_L, f"{STATS['total_cnt']:,}", "Total rentals (2011-12)"),
        ("ti-users", PURPLE, PURPLE_L, f"{STATS['registered']:,}", f"Registered ({STATS['reg_pct']}%)"),
        ("ti-user", GREEN, GREEN_L, f"{STATS['casual']:,}", f"Casual ({STATS['cas_pct']}%)"),
        ("ti-calendar-check", ORANGE, ORANGE_L, f"{STATS['working_days']}", "Working days"),
        ("ti-temperature", PINK, PINK_L, f"{STATS['avg_temp_c']}°C", "Avg. temperature"),
    ]
    cols = st.columns(5)
    for col, (icon, c, cl, val, lbl) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi">
          <div class="kpi-icon" style="background:{cl}"><i class="ti {icon}" style="color:{c}"></i></div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Total rentals over time</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'dash_trend.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Rentals by hour × weekday</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'dash_heatmap.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Rentals by user type</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'dash_donut_users.png'), use_container_width=True)
        st.markdown(f'<p style="font-size:11px;color:{MUTED};text-align:center;margin-top:-8px;"><span style="color:{PURPLE}">●</span> Registered {STATS["reg_pct"]}% &nbsp; <span style="color:{TEAL}">●</span> Casual {STATS["cas_pct"]}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Rentals by season</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'dash_donut_season.png'), use_container_width=True)
        sp = STATS['season_pct']
        st.markdown(f'<p style="font-size:10.5px;color:{MUTED};text-align:center;margin-top:-8px;">Spring {sp["1"]}% · Summer {sp["2"]}% · Fall {sp["3"]}% · Winter {sp["4"]}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top weather situations</div>', unsafe_allow_html=True)
        wp = STATS['weather_pct']
        weather_names = [("Clear", wp["1"], TEAL), ("Mist", wp["2"], PURPLE), ("Light Snow", wp["3"], ORANGE), ("Heavy Rain", wp["4"], PINK)]
        for name, pct, col in weather_names:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:9px;">
              <span style="font-size:11.5px;color:{SUB};width:75px;">{name}</span>
              <div style="flex:1;height:8px;background:{BORDER};border-radius:4px;">
                <div style="width:{pct}%;height:100%;background:{col};border-radius:4px;"></div>
              </div>
              <span style="font-size:11.5px;color:{TEXT};width:40px;text-align:right;">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{TEAL_L},{PURPLE_L});border-radius:16px;padding:16px 24px;display:flex;justify-content:space-around;text-align:center;">
      <div><p style="font-size:18px;font-weight:600;margin:0;color:{TEXT}">{STATS['avg_temp_c']}°C</p><p style="font-size:11px;color:{SUB};margin:0;">Avg. temperature</p></div>
      <div><p style="font-size:18px;font-weight:600;margin:0;color:{TEXT}">{STATS['avg_hum_pct']}%</p><p style="font-size:11px;color:{SUB};margin:0;">Avg. humidity</p></div>
      <div><p style="font-size:18px;font-weight:600;margin:0;color:{TEXT}">{STATS['avg_windspeed']} km/h</p><p style="font-size:11px;color:{SUB};margin:0;">Avg. windspeed</p></div>
      <div><p style="font-size:18px;font-weight:600;margin:0;color:{TEXT}">{STATS['holidays']}</p><p style="font-size:11px;color:{SUB};margin:0;">Holidays</p></div>
      <div><p style="font-size:18px;font-weight:600;margin:0;color:{TEXT}">{STATS['working_days']}</p><p style="font-size:11px;color:{SUB};margin:0;">Working days</p></div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Predict":
    st.markdown('<p class="page-title">Forecast a day</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Autoregressive 24-hour rollout — each hour\'s prediction feeds the next.</p><br>', unsafe_allow_html=True)

    with st.expander("Day conditions", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            yr = st.selectbox("Year", [2011, 2012])
            season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], index=1)
        with c2:
            mnth = st.slider("Month", 1, 12, 7)
            weekday = st.selectbox("Weekday", ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"], index=1)
        with c3:
            weathersit = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
            holiday = st.checkbox("Holiday")
        with c4:
            workingday = st.checkbox("Working day", value=True)
            focus_hour = st.slider("Focus hour", 0, 23, 8)
        temp = st.slider("Temperature (norm.)", 0.0, 1.0, 0.55)
        atemp = st.slider("Feels-like (norm.)", 0.0, 1.0, 0.52)
        hum = st.slider("Humidity (norm.)", 0.0, 1.0, 0.55)
        windspeed = st.slider("Windspeed (norm.)", 0.0, 1.0, 0.18)
        seed_lag = st.number_input("Seed value (rentals at midnight)", min_value=0, value=40)

    cond = {'yr': yr, 'season': season, 'mnth': mnth, 'weekday': weekday, 'weathersit': weathersit,
            'holiday': holiday, 'workingday': workingday, 'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed}
    preds = simulate_day(cond, seed_lag)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Predicted demand — full day rollout</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(range(24), preds, color=TEAL, linewidth=2)
    ax.fill_between(range(24), preds, color=TEAL, alpha=0.15)
    ax.axvline(focus_hour, color=PURPLE, linestyle='--', linewidth=1.2)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#E5E7EB')
    ax.set_xticks(range(0,24,3))
    ax.tick_params(colors='#6B7280', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">Why {focus_hour}:00 predicts {int(preds[focus_hour])} rentals</div>', unsafe_allow_html=True)
    lag1 = preds[focus_hour-1] if focus_hour > 0 else seed_lag
    roll3 = np.mean(preds[max(0,focus_hour-3):focus_hour]) if focus_hour > 0 else seed_lag
    X_focus = build_row(focus_hour, lag1, seed_lag, roll3, cond)
    lower = max(0, q_lower.predict(X_focus)[0])
    upper = max(0, q_upper.predict(X_focus)[0])
    st.markdown(f'<p style="color:{MUTED};font-size:13px;margin:-6px 0 12px;">90% confidence range: <strong style="color:{TEXT}">{int(lower)}–{int(upper)}</strong> rentals</p>', unsafe_allow_html=True)
    shap_values = explainer(X_focus)
    fig2, ax2 = plt.subplots(figsize=(8, 4.3))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    fig2.patch.set_facecolor(SURFACE)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Model":
    st.markdown('<p class="page-title">Model evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Comparison, explainability, and residual diagnostics.</p><br>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, (label, val) in zip([m1,m2,m3], [("R² Score", f"{best['R2']:.3f}"), ("RMSE", f"{best['RMSE']:.1f}"), ("MAE", f"{best['MAE']:.1f}")]):
        col.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Model comparison (R²)</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_09_model_comparison.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Global feature importance (SHAP)</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_12_shap_summary.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><div class="card-title">Residuals</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_10_residuals.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><div class="card-title">90% prediction interval coverage</div>', unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, 'plot_13_prediction_intervals.png'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Full results table</div>', unsafe_allow_html=True)
    st.dataframe(results.style.format({'MAE':'{:.2f}','RMSE':'{:.2f}','R2':'{:.4f}'}), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
