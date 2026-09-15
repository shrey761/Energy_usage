import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
print("Total raw rows:", len(df))
print("Columns:", list(df.columns))

# Check TIME_ERROR
time_errors = (df['timestamp'] == 'TIME_ERROR').sum()
print("TIME_ERROR count:", time_errors)

# Check invalid datetimes
ts = pd.to_datetime(df['timestamp'], errors='coerce')
null_ts = ts.isna().sum()
print("Null or invalid datetime count:", null_ts)

df_clean = df[ts.notna()].copy()
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
print("Time range:", df_clean['timestamp'].min(), "to", df_clean['timestamp'].max())
print("Time differences between consecutive readings (median, min, max):")
time_diffs = df_clean['timestamp'].diff()
print(time_diffs.describe())

print("\nPower1 distribution:")
print(df_clean['power1'].describe())
print("\nPower2 distribution:")
print(df_clean['power2'].describe())

print("\nNumber of rows with power1 > 1200:", (df_clean['power1'] > 1200).sum())
print("Number of rows with power1 > 1500:", (df_clean['power1'] > 1500).sum())
print("Number of rows with power1 == 0:", (df_clean['power1'] == 0).sum())

print("Number of rows with power2 > 1200:", (df_clean['power2'] > 1200).sum())
print("Number of rows with power2 > 1500:", (df_clean['power2'] > 1500).sum())
print("Number of rows with power2 == 0:", (df_clean['power2'] == 0).sum())
