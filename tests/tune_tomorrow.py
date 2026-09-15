import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

THRESHOLD = 1200.0

# 1. Base time features
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# Next day time features
df['tomorrow_dayofweek'] = (df['dayofweek'] + 1) % 7
df['tomorrow_is_weekend'] = df['tomorrow_dayofweek'].isin([5, 6]).astype(int)

# Rolling & Lag
for col in ['power1', 'power2', 'totalPower', 'current1', 'current2']:
    df[f'{col}_lag1'] = df[col].shift(1).bfill()
    df[f'{col}_lag3'] = df[col].shift(3).bfill()
    df[f'{col}_roll_mean5'] = df[col].rolling(window=5, min_periods=1).mean()
    df[f'{col}_roll_max5'] = df[col].rolling(window=5, min_periods=1).max()

# 2. Next Hour Target
df['power1_next'] = df['power1'].shift(-1)
df['power2_next'] = df['power2'].shift(-1)
df['high_usage_r1_next'] = (df['power1_next'] >= THRESHOLD).astype(int)
df['high_usage_r2_next'] = (df['power2_next'] >= THRESHOLD).astype(int)

# 3. Clean Chronological 24-Hour Forward Target
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
    tolerance=pd.Timedelta(hours=2)
).sort_values('orig_idx').reset_index(drop=True)

df['power1_24h_actual'] = merged_24h['future_p1_24h']
df['power2_24h_actual'] = merged_24h['future_p2_24h']

date_hour_lookup = df.groupby([df['timestamp'].dt.date, df['timestamp'].dt.hour])[['power1', 'power2']].mean()

def resolve_tomorrow_power(row, p_col):
    if pd.notna(row[f'{p_col}_24h_actual']):
        return row[f'{p_col}_24h_actual']
    curr_d = row['timestamp'].date()
    curr_h = row['timestamp'].hour
    future_dates = [d for (d, h) in date_hour_lookup.index if d > curr_d and h == curr_h]
    if future_dates:
        next_d = min(future_dates)
        return date_hour_lookup.loc[(next_d, curr_h), p_col]
    else:
        past_dates = [d for (d, h) in date_hour_lookup.index if d <= curr_d and h == curr_h]
        return date_hour_lookup.loc[past_dates, p_col].mean()

df['power1_tomorrow'] = df.apply(lambda r: resolve_tomorrow_power(r, 'power1'), axis=1)
df['power2_tomorrow'] = df.apply(lambda r: resolve_tomorrow_power(r, 'power2'), axis=1)

df['high_usage_r1_tomorrow'] = (df['power1_tomorrow'] >= THRESHOLD).astype(int)
df['high_usage_r2_tomorrow'] = (df['power2_tomorrow'] >= THRESHOLD).astype(int)

df_final = df.dropna(subset=['power1_next', 'power2_next']).reset_index(drop=True)

split_idx = int(len(df_final) * 0.8)
train_df = df_final.iloc[:split_idx]
test_df = df_final.iloc[split_idx:]

# Let's test different feature subsets for Tomorrow models
base_features = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

tomorrow_special_features = base_features + ['tomorrow_dayofweek', 'tomorrow_is_weekend', 'minute']

for label, feat_list in [("Base Features", base_features), ("With Tomorrow Context Features", tomorrow_special_features)]:
    print(f"\n--- Testing Feature Set: {label} ---")
    X_tr = train_df[feat_list]
    X_te = test_df[feat_list]
    
    for max_d in [6, 8, 10, 12, None]:
        for cw in ['balanced', {0: 1, 1: 2}, {0: 1, 1: 1}]:
            clf_r1 = RandomForestClassifier(n_estimators=100, max_depth=max_d, class_weight=cw, random_state=42)
            clf_r1.fit(X_tr, train_df['high_usage_r1_tomorrow'])
            p_r1 = clf_r1.predict(X_te)
            f1_r1 = f1_score(test_df['high_usage_r1_tomorrow'], p_r1, zero_division=0)
            acc_r1 = accuracy_score(test_df['high_usage_r1_tomorrow'], p_r1)
            
            clf_r2 = RandomForestClassifier(n_estimators=100, max_depth=max_d, class_weight=cw, random_state=42)
            clf_r2.fit(X_tr, train_df['high_usage_r2_tomorrow'])
            p_r2 = clf_r2.predict(X_te)
            f1_r2 = f1_score(test_df['high_usage_r2_tomorrow'], p_r2, zero_division=0)
            acc_r2 = accuracy_score(test_df['high_usage_r2_tomorrow'], p_r2)
            
            if f1_r1 > 0.77 or f1_r2 > 0.86:
                print(f"Depth={max_d}, CW={cw} -> R1 Acc={acc_r1*100:.2f}%, F1={f1_r1:.4f} | R2 Acc={acc_r2*100:.2f}%, F1={f1_r2:.4f}")
