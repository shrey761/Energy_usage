import pandas as pd
import numpy as np

csv_path = r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\LATEST_ML_TRAINING_DATA.csv'
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.sort_values('timestamp').reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Train time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} rows)")
print(f"Test time range : {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} rows)")

# Check how many test rows have an actual 24h reading vs fallback
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

test_matches = merged_24h.iloc[split_idx:]['future_p1_24h']
print(f"In the test set of {len(test_df)} rows:")
print(f"  Rows with true measured future 24h reading : {test_matches.notna().sum()} ({test_matches.notna().mean()*100:.1f}%)")
print(f"  Rows with NO 24h future data (Day 7-8 tail): {test_matches.isna().sum()} ({test_matches.isna().mean()*100:.1f}%)")
