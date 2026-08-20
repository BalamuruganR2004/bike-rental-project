import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import GradientBoostingRegressor
import shap
from streamlit_option_menu import option_menu

st.set_page_config(page_title="PedalIQ", page_icon="🚲", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INK    = "#14152B"
SUB    = "#4B4B63"
MUTED  = "#8A8AA3"
TEAL   = "#14B8A6"
PURPLE = "#7B68EE"
LINE   = "#ECEAF6"
SURFACE= "#FFFFFF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&family=Lora:ital@1&display=swap');
.stApp {{ background: {SURFACE}; color: {INK}; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ max-width: 760px; padding-top: 4.5rem; }}
[data-testid="stHeader"] {{ background: {SURFACE}; }}
i.bi {{ display: none !important; }}
h1, h2, h3 {{ font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; color: {INK}; }}
.eyebrow {{ font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: {PURPLE}; font-weight: 600; margin: 0 0 6px; }}
.headline {{ font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 44px; line-height: 1.05; margin: 0; color: {INK}; letter-spacing: -0.02em; }}
.headline .accent {{ color: {TEAL}; }}
.lede {{ font-family: 'Lora', serif; font-style: italic; font-size: 16px; line-height: 1.65; color: {SUB}; margin: 14px 0 0; max-width: 560px; }}
.lede b {{ color: {INK}; font-weight: 600; font-style: normal; }}
.stat-strip {{ display: flex; border-top: 1px solid {LINE}; padding-top: 16px; margin-top: 8px; }}
.stat-cell {{ flex: 1; padding: 0 16px; }}
.stat-cell:first-child {{ padding-left: 0; }}
.stat-cell:not(:first-child) {{ border-left: 1px solid {LINE}; }}
.stat-val {{ font-family: 'Poppins', sans-serif; font-size: 21px; font-weight: 700; margin: 0; }}
.stat-lbl {{ font-size: 10.5px; color: {MUTED}; margin: 0; }}
div[data-baseweb="select"] > div, .stNumberInput input {{ background: {SURFACE} !important; border-color: {LINE} !important; color: {INK} !important; border-radius: 8px !important; }}
.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {{ color: {SUB} !important; font-size: 13px !important; font-weight: 500 !important; }}
button[kind="primary"] {{ background: {INK} !important; color: white !important; font-weight: 500 !important; border: none !important; border-radius: 20px !important; }}
hr {{ border-color: {LINE}; }}
</style>
""", unsafe_allow_html=True)

def train_fallback_models():
    df = pd.read_csv(os.path.join(BASE_DIR, 'cleaned_data.csv'), parse_dates=['dteday'])
    df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)

    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
    df['is_rush_hour'] = df['hr'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
    df['temp_hum_interaction'] = df['temp'] * df['hum']
    df['cnt_lag1'] = df['cnt'].shift(1)
    df['cnt_lag24'] = df['cnt'].shift(24)
    df['cnt_roll3'] = df['cnt'].shift(1).rolling(window=3, min_periods=1).mean()
    for c in ['cnt_lag1', 'cnt_lag24', 'cnt_roll3']:
        df[c] = df[c].fillna(df[c].median())

    df = pd.get_dummies(df, columns=['season', 'weathersit'], prefix=['season', 'weather'])

    feature_cols = [c for c in df.columns if c not in ['instant', 'dteday', 'casual', 'registered', 'cnt', 'hr', 'mnth', 'weekday']]
    X = df[feature_cols]
    y = df['cnt']

    split_idx = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    model = GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)

    q_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    q_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    q_lower.fit(X_train, y_train)
    q_upper.fit(X_train, y_train)

    return model, feature_cols, q_lower, q_upper

@st.cache_resource
def load_models():
    try:
        with open(os.path.join(BASE_DIR, 'best_model_advanced.pkl'), 'rb') as f:
            main = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'quantile_models.pkl'), 'rb') as f:
            q = pickle.load(f)
        return main['model'], main['features'], q['lower'], q['upper']
    except Exception:
        return train_fallback_models()

model, feature_cols, q_lower, q_upper = load_models()

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)
results = pd.read_csv(os.path.join(BASE_DIR, 'model_results_advanced.csv'))
best = results.iloc[results['R2'].idxmax()]
with open(os.path.join(BASE_DIR, 'dashboard_stats.json')) as f:
    STATS = json.load(f)

@st.cache_data
def hourly_split():
    df = pd.read_csv(os.path.join(BASE_DIR, 'cleaned_data.csv'))
    return df.groupby('hr')['registered'].mean(), df.groupby('hr')['casual'].mean()

reg_hr, cas_hr = hourly_split()

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

def simulate_week(cond, seed_lag, start_weekday_idx):
    """Extends the autoregressive rollout across 7 days: each day's last hours
    feed the next day's lag features, and weekday/weekend flips automatically
    based on calendar day, so weekend flattening emerges from the model."""
    weekday_names = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
    lag1 = seed_lag; lag24 = seed_lag; roll_hist = [seed_lag]*3
    week = []
    for d in range(7):
        wd_idx = (start_weekday_idx + d) % 7
        day_cond = dict(cond)
        day_cond['weekday'] = weekday_names[wd_idx]
        day_cond['workingday'] = wd_idx not in [0, 6]
        day_preds = []
        for hr in range(24):
            X = build_row(hr, lag1, lag24, np.mean(roll_hist[-3:]), day_cond)
            p = max(0, float(model.predict(X)[0]))
            day_preds.append(p); lag1 = p; roll_hist.append(p)
        lag24 = day_preds[0]
        week.append({'weekday': weekday_names[wd_idx], 'preds': day_preds, 'total': sum(day_preds), 'peak': max(day_preds)})
    return week

def smooth_wave(y, points=300):
    x = np.arange(len(y))
    xs = np.linspace(0, len(y)-1, points)
    spline = make_interp_spline(x, y, k=3)
    return xs, spline(xs)

def wave_chart(reg, cas, height=3.0):
    fig, ax = plt.subplots(figsize=(9, height))
    xr, yr = smooth_wave(reg.values)
    xc, yc = smooth_wave(cas.values)
    yr = np.clip(yr, 0, None); yc = np.clip(yc, 0, None)
    ax.plot(xr, yr, color=TEAL, linewidth=2.2)
    ax.fill_between(xr, yr, color=TEAL, alpha=0.16)
    ax.plot(xc, yc, color=PURPLE, linewidth=1.8)
    ax.fill_between(xc, yc, color=PURPLE, alpha=0.14)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

page = option_menu(
    menu_title=None, options=["Overview", "Predict", "Week", "Model"],
    default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0", "background-color": SURFACE, "border-bottom": f"1px solid {LINE}"},
        "icon": {"display": "none"},
        "nav-link": {"font-family": "Inter", "font-size": "13px", "font-weight": "500", "color": MUTED, "text-align": "left", "margin": "0 18px 0 0", "padding": "0 0 10px 0", "background-color": SURFACE, "border-bottom": "2px solid transparent"},
        "nav-link-selected": {"background-color": SURFACE, "color": INK, "font-weight": "600", "border-bottom": f"2px solid {TEAL}"},
    },
)
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<p class="eyebrow">PedalIQ · demand forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="headline">Two peaks, <span class="accent">one city.</span></p>', unsafe_allow_html=True)
    st.markdown(f"""
    <p class="lede">Registered riders commute in two sharp waves. Casual riders drift through the afternoon.
    The model learned both rhythms from <b>{STATS['total_cnt']:,} rides</b> across <b>17,361 hours</b> of real data.</p>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
    fig = wave_chart(reg_hr, cas_hr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown(f"""
    <p style="font-size:11px;color:{SUB};margin-top:-10px;">
      <i class="ti ti-circle-filled" style="color:{TEAL}"></i> registered &nbsp;&nbsp;
      <i class="ti ti-circle-filled" style="color:{PURPLE}"></i> casual &nbsp;&nbsp;
      · peaks at 8:00 and 17:00, casual peaks near 14:00
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-cell"><p class="stat-val" style="color:{TEAL}">{best['R2']:.3f}</p><p class="stat-lbl">R2, held-out</p></div>
      <div class="stat-cell"><p class="stat-val" style="color:{PURPLE}">92.5%</p><p class="stat-lbl">interval coverage</p></div>
      <div class="stat-cell"><p class="stat-val">{STATS['reg_pct']:.0f} / {STATS['cas_pct']:.0f}</p><p class="stat-lbl">registered / casual split</p></div>
      <div class="stat-cell"><p class="stat-val">{STATS['avg_temp_c']}°C</p><p class="stat-lbl">avg. temperature</p></div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Predict":
    st.markdown('<p class="eyebrow">Forecast</p>', unsafe_allow_html=True)
    st.markdown('<p class="headline" style="font-size:34px;">What tomorrow looks like</p>', unsafe_allow_html=True)
    st.markdown('<p class="lede">An autoregressive rollout — each predicted hour feeds the next.</p>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

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

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    xs, ys = smooth_wave(np.array(preds))
    ys = np.clip(ys, 0, None)
    ax.plot(xs, ys, color=TEAL, linewidth=2.2)
    ax.fill_between(xs, ys, color=TEAL, alpha=0.15)
    ax.axvline(focus_hour/23*300, color=PURPLE, linestyle='--', linewidth=1.2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-cell"><p class="stat-val" style="color:{TEAL}">{int(preds[focus_hour])}</p><p class="stat-lbl">predicted at {focus_hour:02d}:00</p></div>
      <div class="stat-cell"><p class="stat-val">{int(max(preds))}</p><p class="stat-lbl">peak hour ({int(np.argmax(preds)):02d}:00)</p></div>
      <div class="stat-cell"><p class="stat-val">{int(sum(preds)):,}</p><p class="stat-lbl">total for the day</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:16px;margin:0 0 8px;">Why {focus_hour}:00 predicts {int(preds[focus_hour])} rentals</p>', unsafe_allow_html=True)
    lag1 = preds[focus_hour-1] if focus_hour > 0 else seed_lag
    roll3 = np.mean(preds[max(0,focus_hour-3):focus_hour]) if focus_hour > 0 else seed_lag
    X_focus = build_row(focus_hour, lag1, seed_lag, roll3, cond)
    lower = max(0, q_lower.predict(X_focus)[0])
    upper = max(0, q_upper.predict(X_focus)[0])
    st.markdown(f'<p style="color:{MUTED};font-size:13px;margin:0 0 10px;">90% confidence range: <strong style="color:{INK}">{int(lower)}–{int(upper)}</strong> rentals</p>', unsafe_allow_html=True)
    shap_values = explainer(X_focus)
    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    fig2.patch.set_facecolor(SURFACE)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ---- Weather comparison ----
    st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:16px;margin:0 0 10px;">Same hour, different weather</p>', unsafe_allow_html=True)
    weather_options = ["Clear", "Mist", "Light Snow", "Heavy Rain"]
    weather_preds = {}
    for w in weather_options:
        w_cond = dict(cond); w_cond['weathersit'] = w
        Xw = build_row(focus_hour, lag1, seed_lag, roll3, w_cond)
        weather_preds[w] = max(0, float(model.predict(Xw)[0]))
    wmax = max(weather_preds.values()) or 1
    for w in weather_options:
        val = weather_preds[w]
        is_selected = (w == weathersit)
        bar_color = TEAL if is_selected else LINE
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
          <span style="font-size:12px;color:{INK if is_selected else MUTED};width:80px;font-weight:{'600' if is_selected else '400'};">{w}</span>
          <div style="flex:1;height:9px;background:{LINE};border-radius:5px;">
            <div style="width:{val/wmax*100:.0f}%;height:100%;background:{bar_color if is_selected else PURPLE};opacity:{1 if is_selected else 0.4};border-radius:5px;"></div>
          </div>
          <span style="font-size:12px;color:{INK};width:44px;text-align:right;">{int(val)}</span>
        </div>
        """, unsafe_allow_html=True)

    # ---- What-if temperature slider ----
    st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:16px;margin:0 0 4px;">What if it were warmer or colder?</p>', unsafe_allow_html=True)
    delta_c = st.slider("Temperature change (°C)", -10, 10, 0, key="whatif_temp")
    delta_norm = delta_c / 41.0
    whatif_cond = dict(cond)
    whatif_cond['temp'] = float(np.clip(cond['temp'] + delta_norm, 0, 1))
    whatif_cond['atemp'] = float(np.clip(cond['atemp'] + delta_norm, 0, 1))
    X_whatif = build_row(focus_hour, lag1, seed_lag, roll3, whatif_cond)
    whatif_pred = max(0, float(model.predict(X_whatif)[0]))
    diff = whatif_pred - preds[focus_hour]
    diff_color = TEAL if diff >= 0 else "#E24B4A"
    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-cell"><p class="stat-val">{int(preds[focus_hour])}</p><p class="stat-lbl">baseline at {focus_hour:02d}:00</p></div>
      <div class="stat-cell"><p class="stat-val" style="color:{diff_color}">{'+' if diff>=0 else ''}{int(diff)}</p><p class="stat-lbl">change with {'+' if delta_c>=0 else ''}{delta_c}°C</p></div>
      <div class="stat-cell"><p class="stat-val">{int(whatif_pred)}</p><p class="stat-lbl">adjusted prediction</p></div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Downloads ----
    st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:16px;margin:0 0 10px;">Export this forecast</p>', unsafe_allow_html=True)
    csv_df = pd.DataFrame({'hour': range(24), 'predicted_rentals': [round(p) for p in preds]})
    csv_bytes = csv_df.to_csv(index=False).encode('utf-8')

    from matplotlib.backends.backend_pdf import PdfPages
    import io
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        fig_r, ax_r = plt.subplots(figsize=(8.5, 5))
        ax_r.plot(range(24), preds, color=TEAL, linewidth=2)
        ax_r.fill_between(range(24), preds, color=TEAL, alpha=0.15)
        ax_r.set_xlabel('Hour'); ax_r.set_ylabel('Predicted rentals')
        ax_r.set_title(f'PedalIQ forecast — {season}, {weathersit}, {weekday}')
        ax_r.spines[['top','right']].set_visible(False)
        pdf.savefig(fig_r); plt.close(fig_r)
        fig_t, ax_t = plt.subplots(figsize=(8.5, 5))
        ax_t.axis('off')
        summary_txt = (
            f"PedalIQ demand forecast report\n\n"
            f"Conditions: {season}, {weathersit}, {weekday}, month {mnth}, year {yr}\n"
            f"Working day: {workingday}   Holiday: {holiday}\n\n"
            f"Total predicted rentals: {int(sum(preds)):,}\n"
            f"Peak hour: {int(np.argmax(preds)):02d}:00 ({int(max(preds))} rentals)\n"
            f"Focus hour {focus_hour:02d}:00: {int(preds[focus_hour])} rentals "
            f"(90% range {int(lower)}-{int(upper)})\n\n"
            f"Model: Gradient Boosting Regressor, R2={best['R2']:.3f}, "
            f"RMSE={best['RMSE']:.1f}, MAE={best['MAE']:.1f}"
        )
        ax_t.text(0.05, 0.95, summary_txt, va='top', fontsize=11, family='monospace')
        pdf.savefig(fig_t); plt.close(fig_t)
    pdf_bytes = pdf_buf.getvalue()

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Download CSV", data=csv_bytes, file_name="pedaliq_forecast.csv", mime="text/csv", use_container_width=True)
    with dl2:
        st.download_button("Download PDF report", data=pdf_bytes, file_name="pedaliq_forecast_report.pdf", mime="application/pdf", use_container_width=True)

elif page == "Week":
    st.markdown('<p class="eyebrow">7-day forecast</p>', unsafe_allow_html=True)
    st.markdown('<p class="headline" style="font-size:34px;">The week ahead</p>', unsafe_allow_html=True)
    st.markdown('<p class="lede">One continuous rollout — each day\'s last hours feed the next day\'s lag features. Weekday and weekend rhythms emerge on their own.</p>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wk_yr = st.selectbox("Year", [2011, 2012], key="wk_yr")
        wk_season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], index=1, key="wk_season")
    with c2:
        wk_mnth = st.slider("Month", 1, 12, 7, key="wk_mnth")
        wk_start = st.selectbox("Starting weekday", ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"], index=1, key="wk_start")
    with c3:
        wk_weather = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"], key="wk_weather")
        wk_holiday = st.checkbox("Holiday on day 1", key="wk_holiday")
    with c4:
        wk_temp = st.slider("Temperature (norm.)", 0.0, 1.0, 0.55, key="wk_temp")
        wk_seed = st.number_input("Seed value", min_value=0, value=40, key="wk_seed")

    wk_cond = {'yr': wk_yr, 'season': wk_season, 'mnth': wk_mnth, 'weathersit': wk_weather,
               'holiday': wk_holiday, 'temp': wk_temp, 'atemp': wk_temp*0.94, 'hum': 0.55, 'windspeed': 0.18}
    wd_order = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
    week = simulate_week(wk_cond, wk_seed, wd_order.index(wk_start))

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    all_preds = np.concatenate([d['preds'] for d in week])
    fig, ax = plt.subplots(figsize=(9, 3))
    xs, ys = smooth_wave(all_preds, points=500)
    ys = np.clip(ys, 0, None)
    ax.plot(xs, ys, color=TEAL, linewidth=2)
    ax.fill_between(xs, ys, color=TEAL, alpha=0.15)
    for i in range(1, 7):
        ax.axvline(i*24/168*500, color=LINE, linewidth=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    cols = st.columns(7)
    for col, d in zip(cols, week):
        is_weekend = d['weekday'] in ['Sun','Sat']
        col.markdown(f"""
        <div style="text-align:center;">
          <p style="font-size:11px;color:{MUTED};margin:0 0 4px;">{d['weekday']}</p>
          <p style="font-family:'Poppins',sans-serif;font-size:16px;font-weight:700;margin:0;color:{PURPLE if is_weekend else TEAL};">{int(d['total']):,}</p>
          <p style="font-size:9.5px;color:{MUTED};margin:0;">peak {int(d['peak'])}</p>
        </div>
        """, unsafe_allow_html=True)

    week_total = sum(d['total'] for d in week)
    weekday_avg = np.mean([d['total'] for d in week if d['weekday'] not in ['Sun','Sat']])
    weekend_avg = np.mean([d['total'] for d in week if d['weekday'] in ['Sun','Sat']])
    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-cell"><p class="stat-val" style="color:{TEAL}">{int(week_total):,}</p><p class="stat-lbl">total for the week</p></div>
      <div class="stat-cell"><p class="stat-val">{int(weekday_avg):,}</p><p class="stat-lbl">avg. weekday total</p></div>
      <div class="stat-cell"><p class="stat-val" style="color:{PURPLE}">{int(weekend_avg):,}</p><p class="stat-lbl">avg. weekend total</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    week_rows = []
    for d in week:
        for hr, p in enumerate(d['preds']):
            week_rows.append({'weekday': d['weekday'], 'hour': hr, 'predicted_rentals': round(p)})
    week_csv = pd.DataFrame(week_rows).to_csv(index=False).encode('utf-8')
    st.download_button("Download week CSV", data=week_csv, file_name="pedaliq_week_forecast.csv", mime="text/csv")

elif page == "Model":
    st.markdown('<p class="eyebrow">Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="headline" style="font-size:34px;">How well it holds up</p>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-cell"><p class="stat-val" style="color:{TEAL}">{best['R2']:.3f}</p><p class="stat-lbl">R2 score</p></div>
      <div class="stat-cell"><p class="stat-val" style="color:{PURPLE}">{best['RMSE']:.1f}</p><p class="stat-lbl">RMSE</p></div>
      <div class="stat-cell"><p class="stat-val">{best['MAE']:.1f}</p><p class="stat-lbl">MAE</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:15px;">Model comparison</p>', unsafe_allow_html=True)
    st.image(os.path.join(BASE_DIR, 'plot_09_model_comparison.png'), use_container_width=True)

    st.markdown('<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:15px;margin-top:1.25rem;">Global feature importance (SHAP)</p>', unsafe_allow_html=True)
    st.image(os.path.join(BASE_DIR, 'plot_12_shap_summary.png'), use_container_width=True)

    st.markdown('<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:15px;margin-top:1.25rem;">Residuals</p>', unsafe_allow_html=True)
    st.image(os.path.join(BASE_DIR, 'plot_10_residuals.png'), use_container_width=True)

    st.markdown('<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:15px;margin-top:1.25rem;">90% prediction interval coverage</p>', unsafe_allow_html=True)
    st.image(os.path.join(BASE_DIR, 'plot_13_prediction_intervals.png'), use_container_width=True)

    st.markdown('<p style="font-family:\'Poppins\',sans-serif;font-weight:600;font-size:15px;margin-top:1.25rem;">Full results table</p>', unsafe_allow_html=True)
    st.dataframe(results.style.format({'MAE':'{:.2f}','RMSE':'{:.2f}','R2':'{:.4f}'}), use_container_width=True, hide_index=True)
