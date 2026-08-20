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

st.set_page_config(page_title="PedalIQ — Demand Forecasting", page_icon="🚲", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
.page-eyebrow {{ font-size: 12px; font-weight: 600; color: {PRIMARY}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
.page-title {{ font-size: 26px; margin-bottom: 2px; }}
.page-sub {{ color: {MUTED}; font-size: 14px; margin-bottom: 18px; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 20px 22px; box-shadow: {SHADOW}; margin-bottom: 18px; }}
.card-title {{ font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 14px; color: {TEXT}; margin-bottom: 14px; }}
.stat-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; box-shadow: {SHADOW}; }}
.stat-icon {{ width: 32px; height: 32px; border-radius: 8px; background: {PRIMARY_L}; display: flex; align-items: center; justify-content: center; font-size: 15px; margin-bottom: 10px; }}
.stat-value {{ font-family: 'Manrope', sans-serif; font-size: 24px; font-weight: 800; color: {TEXT}; }}
.stat-label {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}
.brand {{ display: flex; align-items: center; gap: 10px; }}
.brand-mark {{ width: 34px; height: 34px; border-radius: 9px; background: {PRIMARY}; display: flex; align-items: center; justify-content: center; font-size: 18px; }}
.brand-name {{ font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 18px; color: {TEXT}; }}
.brand-tag {{ font-size: 12px; color: {MUTED}; }}
.badge-live {{ background: {SUCCESS_L}; color: {SUCCESS}; font-size: 12px; font-weight: 600; padding: 5px 12px; border-radius: 20px; display: inline-flex; align-items: center; gap: 6px; }}
.badge-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {SUCCESS}; display: inline-block; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
div[data-baseweb="select"] > div, .stNumberInput input {{ background: {SURFACE} !important; border-color: {BORDER} !important; color: {TEXT} !important; border-radius: 8px !important; }}
.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {{ color: {MUTED} !important; font-size: 13px !important; font-weight: 500 !important; }}
button[kind="primary"] {{ background: {PRIMARY} !important; color: white !important; font-weight: 600 !important; border: none !important; border-radius: 9px !important; box-shadow: {SHADOW}; }}
hr {{ border-color: {BORDER}; }}
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
    """Autoregressive rollout: each hour's prediction becomes the next hour's
    lag feature, so the full 24h curve is generated by the trained model
    itself from a single seed value, not hardcoded."""
    lag1 = seed_lag
    lag24 = seed_lag
    roll_hist = [seed_lag, seed_lag, seed_lag]
    preds = []
    for hr in range(24):
        X = build_row(hr, lag1, lag24, np.mean(roll_hist[-3:]), cond)
        p = max(0, float(model.predict(X)[0]))
        preds.append(p)
        lag1 = p
        roll_hist.append(p)
    return preds

def flow_component(preds, start_hour, height=320):
    data_json = json.dumps([round(p) for p in preds])
    peak = max(preds) if max(preds) > 0 else 1
    html = f"""
    <div style="font-family:'Inter',sans-serif;">
    <div style="position:relative;border:1px solid {BORDER};border-radius:14px;overflow:hidden;background:{SURFACE};box-shadow:{SHADOW}">
      <canvas id="flow" width="900" height="260" style="width:100%;height:260px;display:block"></canvas>
      <div style="position:absolute;top:16px;left:20px;">
        <p id="hud-val" style="font-family:'Manrope',sans-serif;font-size:32px;font-weight:800;margin:0;color:{TEXT}">--</p>
        <p id="hud-lbl" style="font-size:11px;color:{MUTED};margin:0">predicted rentals · autoregressive rollout</p>
      </div>
      <div style="position:absolute;top:16px;right:20px;text-align:right;">
        <p id="hud-hour" style="font-size:16px;margin:0;color:{MUTED};font-family:monospace">08:00</p>
      </div>
    </div>
    <div style="max-width:100%;margin:14px auto 0;">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:{MUTED};margin-bottom:2px">
        <span>12am</span><span>6am</span><span>12pm</span><span>6pm</span><span>11pm</span>
      </div>
      <input type="range" min="0" max="23" value="{start_hour}" step="1" id="hourSlider" style="width:100%">
    </div>
    </div>
    <script>
    const demand = {data_json};
    const peak = {peak};
    const canvas = document.getElementById('flow');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    function lerp(a,b,t){{ return a+(b-a)*t; }}
    function hexToRgb(h){{ const n=parseInt(h.slice(1),16); return [n>>16&255, n>>8&255, n&255]; }}
    function lerpColor(c1,c2,t){{
      const a=hexToRgb(c1), b=hexToRgb(c2);
      return `rgb(${{Math.round(lerp(a[0],b[0],t))}},${{Math.round(lerp(a[1],b[1],t))}},${{Math.round(lerp(a[2],b[2],t))}})`;
    }}
    const COOL = '#2F5EFF', WARM_MID = '#F79009', HOT = '#E24B4A';
    function colorFor(t){{
      return t < 0.5 ? lerpColor(COOL, WARM_MID, t*2) : lerpColor(WARM_MID, HOT, (t-0.5)*2);
    }}
    const lanes = [
      {{y:40, amp:14, freq:0.012, phase:0}},
      {{y:85, amp:10, freq:0.016, phase:1.5}},
      {{y:130, amp:16, freq:0.010, phase:3.0}},
      {{y:175, amp:12, freq:0.014, phase:4.5}},
      {{y:220, amp:9,  freq:0.018, phase:2.2}},
    ];
    let particles = [];
    let hour = {start_hour};
    let intensity = demand[hour]/peak;
    function laneY(lane, x){{ return lane.y + Math.sin(x*lane.freq + lane.phase)*lane.amp; }}
    function spawn(){{
      const targetCount = Math.round(15 + intensity*110);
      while(particles.length < targetCount){{
        const lane = lanes[Math.floor(Math.random()*lanes.length)];
        particles.push({{ lane, x: Math.random()*W, speed: 0.5 + intensity*2.0 + Math.random()*0.4, size: 1.3 + Math.random()*1.6 }});
      }}
      if(particles.length > targetCount) particles.length = targetCount;
    }}
    function draw(){{
      ctx.clearRect(0,0,W,H);
      ctx.strokeStyle = '{BORDER}';
      ctx.lineWidth = 1;
      lanes.forEach(lane=>{{
        ctx.beginPath();
        for(let x=0;x<=W;x+=8){{
          const y = laneY(lane,x);
          x===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }}
        ctx.stroke();
      }});
      const col = colorFor(intensity);
      particles.forEach(p=>{{
        p.x += p.speed;
        if(p.x > W+10) p.x = -10;
        const y = laneY(p.lane, p.x);
        ctx.beginPath();
        ctx.fillStyle = col;
        ctx.arc(p.x, y, p.size, 0, Math.PI*2);
        ctx.fill();
      }});
      requestAnimationFrame(draw);
    }}
    function updateHud(){{
      document.getElementById('hud-val').textContent = demand[hour];
      document.getElementById('hud-hour').textContent = String(hour).padStart(2,'0')+':00';
      intensity = demand[hour]/peak;
      spawn();
    }}
    document.getElementById('hourSlider').addEventListener('input', (e)=>{{
      hour = parseInt(e.target.value);
      updateHud();
    }});
    updateHud();
    spawn();
    draw();
    </script>
    """
    components.html(html, height=height, scrolling=False)

# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div class="brand" style="margin-bottom:24px;">
        <div class="brand-mark">🚲</div>
        <div><div class="brand-name">PedalIQ</div><div class="brand-tag">Demand Forecasting</div></div>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        menu_title=None, options=["Predict", "Analytics", "Model"],
        icons=["lightning-charge", "graph-up", "cpu"], default_index=0,
        styles={
            "container": {"padding": "0", "background-color": SURFACE},
            "icon": {"color": MUTED, "font-size": "15px"},
            "nav-link": {"font-family": "Inter", "font-size": "14px", "font-weight": "500", "color": MUTED, "border-radius": "8px", "margin": "2px 0", "padding": "10px 12px"},
            "nav-link-selected": {"background-color": PRIMARY_L, "color": PRIMARY, "font-weight": "600"},
        },
    )
    st.markdown(f"""
    <div style="margin-top:28px; padding-top:16px; border-top:1px solid {BORDER};">
        <div class="badge-live"><span class="badge-dot"></span>Model v2 · Live</div>
        <div style="color:{MUTED}; font-size:12px; margin-top:10px;">Gradient Boosting Regressor<br>trained on 17K+ hourly records</div>
    </div>
    """, unsafe_allow_html=True)

kc1, kc2, kc3, kc4 = st.columns(4)
kpis = [("📊", f"{best['R2']:.3f}", "R² Score"), ("🎯", f"{best['MAE']:.1f}", "Mean Abs. Error"),
        ("📐", f"{best['RMSE']:.1f}", "RMSE"), ("✅", "92.5%", "Interval Coverage")]
for col, (icon, val, lbl) in zip([kc1,kc2,kc3,kc4], kpis):
    col.markdown(f'<div class="stat-card"><div class="stat-icon">{icon}</div><div class="stat-value">{val}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ===========================================================================
if page == "Predict":
    st.markdown('<div class="page-eyebrow">Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">The city, in motion</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">A full day\'s demand, simulated hour-by-hour by the trained model — flow density and color are the forecast, driven by real predictions.</div>', unsafe_allow_html=True)

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
            focus_hour = st.slider("Focus hour (for explanation below)", 0, 23, 8)
        temp = st.slider("Temperature (norm.)", 0.0, 1.0, 0.55)
        atemp = st.slider("Feels-like (norm.)", 0.0, 1.0, 0.52)
        hum = st.slider("Humidity (norm.)", 0.0, 1.0, 0.55)
        windspeed = st.slider("Windspeed (norm.)", 0.0, 1.0, 0.18)
        seed_lag = st.number_input("Seed value (rentals at midnight, to start the rollout)", min_value=0, value=40)

    cond = {'yr': yr, 'season': season, 'mnth': mnth, 'weekday': weekday, 'weathersit': weathersit,
            'holiday': holiday, 'workingday': workingday, 'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed}

    preds = simulate_day(cond, seed_lag)
    flow_component(preds, focus_hour)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">Why {focus_hour}:00 predicts {int(preds[focus_hour])} rentals</div>', unsafe_allow_html=True)
    lag1 = preds[focus_hour-1] if focus_hour > 0 else seed_lag
    lag24 = seed_lag
    roll3 = np.mean(preds[max(0,focus_hour-3):focus_hour]) if focus_hour > 0 else seed_lag
    X_focus = build_row(focus_hour, lag1, lag24, roll3, cond)
    lower = max(0, q_lower.predict(X_focus)[0])
    upper = max(0, q_upper.predict(X_focus)[0])
    st.markdown(f'<p style="color:{MUTED};font-size:13px;margin:-6px 0 12px;">90% confidence range: <strong style="color:{TEXT}">{int(lower)}–{int(upper)}</strong> rentals</p>', unsafe_allow_html=True)
    shap_values = explainer(X_focus)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    fig.patch.set_facecolor(SURFACE)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
elif page == "Analytics":
    st.markdown('<div class="page-eyebrow">Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Demand patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">How rentals vary by time, season, and weather across the dataset.</div>', unsafe_allow_html=True)
    imgs = [("plot_02_hourly_workday.png", "Hourly demand: working day vs non-working day"),
            ("plot_04_season.png", "Total rentals by season"),
            ("plot_05_weather.png", "Rental distribution by weather"),
            ("plot_03_casual_vs_registered.png", "Casual vs registered riders by weekday")]
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
