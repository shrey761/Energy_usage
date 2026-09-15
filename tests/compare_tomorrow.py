import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

THRESHOLD = 1200.0

# Features
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

# Next Hour Target
df['power1_next'] = df['power1'].shift(-1)
df['power2_next'] = df['power2'].shift(-1)
df['high_usage_r1_next'] = (df['power1_next'] >= THRESHOLD).astype(int)
df['high_usage_r2_next'] = (df['power2_next'] >= THRESHOLD).astype(int)

# True Chronological 24-Hour Forward Target
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

feature_cols = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

split_idx = int(len(df_final) * 0.8)
train_df = df_final.iloc[:split_idx]
test_df = df_final.iloc[split_idx:]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

print("=== BEFORE (Depth=8) vs AFTER (Depth=6, Balanced Regularized) ===")

for target_col, name in [('high_usage_r1_tomorrow', 'Room 1 Tomorrow Same-Hour'), ('high_usage_r2_tomorrow', 'Room 2 Tomorrow Same-Hour')]:
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # Before (max_depth=8)
    clf_old = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    clf_old.fit(X_train, y_train)
    p_old = clf_old.predict(X_test)
    
    # After (max_depth=6, regularized to prevent overfitting on specific hour noise)
    clf_new = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    clf_new.fit(X_train, y_train)
    p_new = clf_new.predict(X_test)
    
    print(f"\n[{name}]")
    print(f"  BEFORE: Accuracy={accuracy_score(y_test, p_old)*100:.2f}%, Precision={precision_score(y_test, p_old):.4f}, Recall={recall_score(y_test, p_old):.4f}, F1={f1_score(y_test, p_old):.4f}")
    print(f"          Confusion Matrix: {confusion_matrix(y_test, p_old).tolist()}")
    print(f"  AFTER : Accuracy={accuracy_score(y_test, p_new)*100:.2f}%, Precision={precision_score(y_test, p_new):.4f}, Recall={recall_score(y_test, p_new):.4f}, F1={f1_score(y_test, p_new):.4f}")
    print(f"          Confusion Matrix: {confusion_matrix(y_test, p_new).tolist()}")
