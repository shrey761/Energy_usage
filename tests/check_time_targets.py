import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Test forward 1-hour merge_asof
df_next_hour = df[['timestamp', 'power1', 'power2']].copy()
df_next_hour['query_time_1h'] = df_next_hour['timestamp'] + pd.Timedelta(hours=1)
df_next_hour['query_time_24h'] = df_next_hour['timestamp'] + pd.Timedelta(hours=24)

# Merge for 1 hour ahead
m_1h = pd.merge_asof(
    df_next_hour.sort_values('query_time_1h'),
    df[['timestamp', 'power1', 'power2']].rename(columns={'timestamp': 'target_time_1h', 'power1': 'power1_1h_ahead', 'power2': 'power2_1h_ahead'}),
    left_on='query_time_1h',
    right_on='target_time_1h',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=15)
).sort_values('timestamp').reset_index(drop=True)

print("1-hour ahead matches with 15-min tolerance:", m_1h['power1_1h_ahead'].notna().sum(), "out of", len(m_1h))

# Merge for 24 hours ahead (tomorrow same hour)
m_24h = pd.merge_asof(
    df_next_hour.sort_values('query_time_24h'),
    df[['timestamp', 'power1', 'power2']].rename(columns={'timestamp': 'target_time_24h', 'power1': 'power1_24h_ahead', 'power2': 'power2_24h_ahead'}),
    left_on='query_time_24h',
    right_on='target_time_24h',
    direction='nearest',
    tolerance=pd.Timedelta(hours=1)
).sort_values('timestamp').reset_index(drop=True)

print("24-hour ahead matches with 1-hour tolerance:", m_24h['power1_24h_ahead'].notna().sum(), "out of", len(m_24h))
