import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_data.csv', parse_dates=['dteday'])
plt.rcParams['figure.dpi'] = 110

# 1. Time series trend of daily total rentals
daily = df.groupby('dteday')['cnt'].sum().reset_index()
plt.figure(figsize=(12, 4))
plt.plot(daily['dteday'], daily['cnt'], color='#2b6cb0')
plt.title('Daily Total Bike Rentals Over Time (2011-2012)')
plt.xlabel('Date'); plt.ylabel('Total Rentals')
plt.tight_layout()
plt.savefig('plot_01_timeseries.png')
plt.close()

# 2. Hourly demand pattern: workday vs non-workday
plt.figure(figsize=(9, 5))
sns.lineplot(data=df, x='hr', y='cnt', hue='workingday', estimator='mean', errorbar=None, palette=['#e53e3e','#2b6cb0'])
plt.title('Average Hourly Demand: Working Day vs Non-Working Day')
plt.xlabel('Hour of Day'); plt.ylabel('Avg Rentals')
plt.legend(title='Working Day', labels=['No','Yes'])
plt.tight_layout()
plt.savefig('plot_02_hourly_workday.png')
plt.close()

# 3. Casual vs registered by weekday
weekday_avg = df.groupby('weekday')[['casual','registered']].mean().reset_index()
weekday_avg.plot(x='weekday', y=['casual','registered'], kind='bar', figsize=(8,5), color=['#dd6b20','#2b6cb0'])
plt.title('Avg Casual vs Registered Riders by Weekday (0=Sun)')
plt.ylabel('Avg Riders')
plt.tight_layout()
plt.savefig('plot_03_casual_vs_registered.png')
plt.close()

# 4. Season-wise total demand
plt.figure(figsize=(7,5))
season_labels = {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}
sns.barplot(x=df['season'].map(season_labels), y=df['cnt'], estimator=sum, errorbar=None, palette='viridis')
plt.title('Total Rentals by Season')
plt.ylabel('Total Rentals')
plt.tight_layout()
plt.savefig('plot_04_season.png')
plt.close()

# 5. Weather impact
plt.figure(figsize=(7,5))
weather_labels = {1:'Clear',2:'Mist',3:'Light Snow',4:'Heavy Rain'}
sns.boxplot(x=df['weathersit'].map(weather_labels), y=df['cnt'], palette='coolwarm')
plt.title('Rental Distribution by Weather Situation')
plt.tight_layout()
plt.savefig('plot_05_weather.png')
plt.close()

# 6. Correlation heatmap
plt.figure(figsize=(10,8))
num_cols = ['temp','atemp','hum','windspeed','casual','registered','cnt','hr','mnth']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot_06_correlation.png')
plt.close()

# 7. Outlier check: cnt boxplot
plt.figure(figsize=(6,5))
sns.boxplot(y=df['cnt'], color='#2b6cb0')
plt.title('Boxplot of Total Rentals (cnt)')
plt.tight_layout()
plt.savefig('plot_07_cnt_outliers.png')
plt.close()

print("Saved 7 plots")
print("\nKey stats:")
print(df[['temp','atemp','hum','windspeed','cnt']].describe())
print("\nCorrelation of features with cnt:")
print(corr['cnt'].sort_values(ascending=False))
