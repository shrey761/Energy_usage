import os
import pandas as pd
import numpy as np

csv_path = r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\LATEST_ML_TRAINING_DATA.csv'
df = pd.read_csv(csv_path)

# 1. Parse & sort
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.sort_values('timestamp').reset_index(drop=True)

THRESHOLD = 1200.0

# 2. Features
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

for col in ['power1', 'power2', 'totalPower', 'current1', 'current2']:
    df[f'{col}_lag1'] = df[col].shift(1).bfill()
    df[f'{col}_lag3'] = df[col].shift(3).bfill()
    df[f'{col}_roll_mean5'] = df[col].rolling(window=5, min_periods=1).mean()
    df[f'{col}_roll_max5'] = df[col].rolling(window=5, min_periods=1).max()

# 3. Next Step / Next Horizon Target
df['power1_next'] = df['power1'].shift(-1)
df['power2_next'] = df['power2'].shift(-1)
df['high_usage_r1_next'] = (df['power1_next'] >= THRESHOLD).astype(int)
df['high_usage_r2_next'] = (df['power2_next'] >= THRESHOLD).astype(int)

# 4. True 24-Hour Forward Target (t + 24 hours) via merge_asof
df_query = pd.DataFrame({
    'orig_idx': df.index,
    'timestamp': df['timestamp'],
    'query_24h': df['timestamp'] + pd.Timedelta(hours=24)
})

merged_24h = pd.merge_asof(
    df_query.sort_values('query_24h'),
    df[['timestamp', 'power1', 'power2']].rename(
        columns={'timestamp': 'match_time', 'power1': 'future_p1_24h', 'power2': 'future_p2_24h'}
    ),
    left_on='query_24h',
    right_on='match_time',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=15)
).sort_values('orig_idx').reset_index(drop=True)

df['power1_24h_actual'] = merged_24h['future_p1_24h']
df['power2_24h_actual'] = merged_24h['future_p2_24h']

# Check match coverage
print(f"Direct 24h lookahead matches (within 15m): {df['power1_24h_actual'].notna().sum()} / {len(df)}")

# For final day records (Day 7) that have no 8th day, resolve using expanding historical hourly mean (zero future leak)
date_hour_map = df.groupby([df['timestamp'].dt.date, df['timestamp'].dt.hour])[['power1', 'power2']].mean()

def resolve_tomorrow(row, p_col):
    if pd.notna(row[f'{p_col}_24h_actual']):
        return row[f'{p_col}_24h_actual']
    curr_d = row['timestamp'].date()
    curr_h = row['timestamp'].hour
    past_dates = [d for (d, h) in date_hour_map.index if d <= curr_d and h == curr_h]
    return date_hour_map.loc[past_dates, p_col].mean()

df['power1_tomorrow'] = df.apply(lambda r: resolve_tomorrow(r, 'power1'), axis=1)
df['power2_tomorrow'] = df.apply(lambda r: resolve_tomorrow(r, 'power2'), axis=1)

df['high_usage_r1_tomorrow'] = (df['power1_tomorrow'] >= THRESHOLD).astype(int)
df['high_usage_r2_tomorrow'] = (df['power2_tomorrow'] >= THRESHOLD).astype(int)

df_final = df.dropna(subset=['power1_next', 'power2_next']).reset_index(drop=True)
print(f"Final processed rows: {len(df_final)}")
print(f"Room 1 Tomorrow High Rate: {df_final['high_usage_r1_tomorrow'].mean()*100:.2f}%")
print(f"Room 2 Tomorrow High Rate: {df_final['high_usage_r2_tomorrow'].mean()*100:.2f}%")
