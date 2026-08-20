import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('cleaned_data.csv', parse_dates=['dteday'])

TEAL = "#14B8A6"
TEAL_L = "#CCFBF1"
PURPLE = "#7B68EE"
PURPLE_L = "#EDEBFE"

# ---- Total rentals over time (teal area chart) ----
daily = df.groupby('dteday')['cnt'].sum().reset_index()
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(daily['dteday'], daily['cnt'], color=TEAL, linewidth=1.6)
ax.fill_between(daily['dteday'], daily['cnt'], color=TEAL, alpha=0.15)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#E5E7EB')
ax.tick_params(colors='#6B7280', labelsize=9)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig('dash_trend.png', dpi=140, facecolor='white')
plt.close()

# ---- Hour x weekday heatmap (purple) ----
hour_bands = pd.cut(df['hr'], bins=[-1,3,7,11,15,19,23], labels=['00-03','04-07','08-11','12-15','16-19','20-23'])
df['hour_band'] = hour_bands
weekday_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df['weekday_lbl'] = df['weekday'].map({1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',0:'Sun'})
pivot = df.pivot_table(index='hour_band', columns='weekday_lbl', values='cnt', aggfunc='mean', observed=False)
pivot = pivot[weekday_labels]
pivot = pivot.loc[['00-03','04-07','08-11','12-15','16-19','20-23']]

fig, ax = plt.subplots(figsize=(6.5, 4))
im = ax.imshow(pivot.values, cmap='Purples', aspect='auto')
ax.set_xticks(range(7)); ax.set_xticklabels(weekday_labels, fontsize=9, color='#6B7280')
ax.set_yticks(range(6)); ax.set_yticklabels(pivot.index, fontsize=9, color='#6B7280')
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig('dash_heatmap.png', dpi=140, facecolor='white')
plt.close()

# ---- Rentals by user type donut (teal/purple) ----
reg, cas = df['registered'].sum(), df['casual'].sum()
fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.pie([reg, cas], colors=[PURPLE, TEAL], startangle=90, wedgeprops=dict(width=0.38, edgecolor='white'))
ax.text(0, 0.08, f"{reg+cas:,.0f}", ha='center', va='center', fontsize=15, fontweight='bold', color='#1A1A2E')
ax.text(0, -0.15, "Total", ha='center', va='center', fontsize=9, color='#6B7280')
plt.tight_layout()
plt.savefig('dash_donut_users.png', dpi=140, facecolor='white', transparent=False)
plt.close()

# ---- Rentals by season donut ----
season_labels = {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}
season_sum = df.groupby('season')['cnt'].sum()
colors_season = ['#34D399','#F5A623','#F26D96','#5B8DEF']
fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.pie(season_sum.values, colors=colors_season, startangle=90, wedgeprops=dict(width=0.38, edgecolor='white'))
ax.text(0, 0.08, f"{season_sum.sum():,.0f}", ha='center', va='center', fontsize=15, fontweight='bold', color='#1A1A2E')
ax.text(0, -0.15, "Total", ha='center', va='center', fontsize=9, color='#6B7280')
plt.tight_layout()
plt.savefig('dash_donut_season.png', dpi=140, facecolor='white')
plt.close()

# ---- Print real stats for the app to hardcode ----
stats = {
    'total_cnt': int(df['cnt'].sum()),
    'registered': int(df['registered'].sum()),
    'casual': int(df['casual'].sum()),
    'working_days': int(df[df['workingday']==1]['dteday'].nunique()),
    'holidays': int(df[df['holiday']==1]['dteday'].nunique()),
    'avg_temp_c': round((df['temp']*41).mean(), 1),
    'avg_hum_pct': round((df['hum']*100).mean(), 1),
    'avg_windspeed': round((df['windspeed']*67).mean(), 1),
    'season_pct': (df.groupby('season')['cnt'].sum() / df['cnt'].sum() * 100).round(1).to_dict(),
    'weather_pct': (df.groupby('weathersit')['cnt'].sum() / df['cnt'].sum() * 100).round(1).to_dict(),
    'reg_pct': round(reg/(reg+cas)*100, 1),
    'cas_pct': round(cas/(reg+cas)*100, 1),
}
import json
with open('dashboard_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
print("\nSaved dash_trend.png, dash_heatmap.png, dash_donut_users.png, dash_donut_season.png, dashboard_stats.json")
