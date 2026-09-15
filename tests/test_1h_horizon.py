import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 1. Look at 1-step ahead (~10s) vs 1-hour ahead (3600s) target
df_query = pd.DataFrame({
    'orig_idx': df.index,
    'timestamp': df['timestamp'],
    'query_1h': df['timestamp'] + pd.Timedelta(hours=1)
})

# Merge for ~1 hour ahead (within 15-30 min window)
merged_1h = pd.merge_asof(
    df_query.sort_values('query_1h'),
    df[['timestamp', 'power1', 'power2']].rename(columns={'timestamp': 'match_time_1h', 'power1': 'power1_1h_ahead', 'power2': 'power2_1h_ahead'}),
    left_on='query_1h',
    right_on='match_time_1h',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=20)
).sort_values('orig_idx').reset_index(drop=True)

print("Rows with valid 1-hour ahead actual reading:", merged_1h['power1_1h_ahead'].notna().sum())

# For rows where future session is continuous or next sequence step
# Let's check the difference between persistence over 1-hour vs RF
df['power1_1h'] = merged_1h['power1_1h_ahead']
df['power2_1h'] = merged_1h['power2_1h_ahead']

# Where 1-hour match exists, evaluate persistence (current power as prediction for 1 hour ahead)
valid_1h = df.dropna(subset=['power1_1h', 'power2_1h'])
print("Valid 1h count:", len(valid_1h))

split = int(len(valid_1h) * 0.8)
tr = valid_1h.iloc[:split]
te = valid_1h.iloc[split:]

print("\n--- 1-HOUR AHEAD PERSISTENCE ERROR ---")
print("Room 1 Persistence MAE (1h ahead):", mean_absolute_error(te['power1_1h'], te['power1']))
print("Room 2 Persistence MAE (1h ahead):", mean_absolute_error(te['power2_1h'], te['power2']))
