import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

THRESHOLD = 1200.0

# 1. Forward 1-step / next-reading target (and next-hour window)
df['power1_next'] = df['power1'].shift(-1)
df['power2_next'] = df['power2'].shift(-1)
df['high_usage_r1_next'] = (df['power1_next'] >= THRESHOLD).astype(int)
df['high_usage_r2_next'] = (df['power2_next'] >= THRESHOLD).astype(int)

# 2. True Tomorrow Same-Hour Target using merge_asof with 24h forward shift
df_query = pd.DataFrame({
    'orig_idx': df.index,
    'timestamp': df['timestamp'],
    'query_24h': df['timestamp'] + pd.Timedelta(hours=24)
})

# Merge with future readings near query_24h (tolerance 2 hours)
merged = pd.merge_asof(
    df_query.sort_values('query_24h'),
    df[['timestamp', 'power1', 'power2']].rename(columns={'timestamp': 'match_time', 'power1': 'future_p1_24h', 'power2': 'future_p2_24h'}),
    left_on='query_24h',
    right_on='match_time',
    direction='nearest',
    tolerance=pd.Timedelta(hours=2)
).sort_values('orig_idx').reset_index(drop=True)

# For unrecorded days (e.g. Aug 19), fallback to next available recorded day at that hour
# Or cumulative expanding hourly profile
df['power1_24h_actual'] = merged['future_p1_24h']
df['power2_24h_actual'] = merged['future_p2_24h']

# Check how many have direct 24h future match
direct_matches = df['power1_24h_actual'].notna().sum()
print(f"Direct 24h matches (within 2h window): {direct_matches} / {len(df)}")

# For unrecorded gaps, use next recorded day at that same hour
date_hour_map = df.groupby([df['timestamp'].dt.date, df['timestamp'].dt.hour])[['power1', 'power2']].mean()

def get_next_day_hour_val(row, p_col):
    if pd.notna(row[f'{p_col}_24h_actual']):
        return row[f'{p_col}_24h_actual']
    curr_date = row['timestamp'].date()
    curr_hour = row['timestamp'].hour
    # find next available date at this hour
    future_dates = [d for (d, h) in date_hour_map.index if d > curr_date and h == curr_hour]
    if future_dates:
        next_d = min(future_dates)
        return date_hour_map.loc[(next_d, curr_hour), p_col]
    else:
        # If at the very end of dataset with no future date, fallback to previous observed mean for that hour (no future leak)
        past_dates = [d for (d, h) in date_hour_map.index if d <= curr_date and h == curr_hour]
        return date_hour_map.loc[past_dates, p_col].mean()

df['power1_tomorrow'] = df.apply(lambda r: get_next_day_hour_val(r, 'power1'), axis=1)
df['power2_tomorrow'] = df.apply(lambda r: get_next_day_hour_val(r, 'power2'), axis=1)

df['high_usage_r1_tomorrow'] = (df['power1_tomorrow'] >= THRESHOLD).astype(int)
df['high_usage_r2_tomorrow'] = (df['power2_tomorrow'] >= THRESHOLD).astype(int)

# Drop only the trailing row from shift(-1)
df_final = df.dropna(subset=['power1_next', 'power2_next']).reset_index(drop=True)

print("Final dataset size:", len(df_final))
print("Tomorrow Room 1 Class Distribution:", df_final['high_usage_r1_tomorrow'].value_counts().to_dict())
print("Tomorrow Room 2 Class Distribution:", df_final['high_usage_r2_tomorrow'].value_counts().to_dict())
