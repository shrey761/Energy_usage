import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

csv_path = r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\LATEST_ML_TRAINING_DATA.csv'
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.sort_values('timestamp').reset_index(drop=True)

THRESHOLD = 1200.0

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

# True 24-hour lookup
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

# Filter only records that have a genuine future 24h reading
df_24h_valid = df[df['power1_24h_actual'].notna()].copy()
df_24h_valid['high_usage_r1_tomorrow'] = (df_24h_valid['power1_24h_actual'] >= THRESHOLD).astype(int)
df_24h_valid['high_usage_r2_tomorrow'] = (df_24h_valid['power2_24h_actual'] >= THRESHOLD).astype(int)

feature_cols = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

# Chronological 80/20 split on 24h verifiable data
split_idx = int(len(df_24h_valid) * 0.8)
train_df = df_24h_valid.iloc[:split_idx]
test_df = df_24h_valid.iloc[split_idx:]

X_tr = train_df[feature_cols]
X_te = test_df[feature_cols]

print(f"Verifiable 24h Train records: {len(train_df)} | Test records: {len(test_df)}")
print(f"Train time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"Test time range : {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

for target_col, name in [('high_usage_r1_tomorrow', 'Room 1 Tomorrow (Verifiable 24h Future)'), 
                         ('high_usage_r2_tomorrow', 'Room 2 Tomorrow (Verifiable 24h Future)')]:
    y_tr = train_df[target_col]
    y_te = test_df[target_col]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    cm = confusion_matrix(y_te, y_pred)
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    
    print(f"\n[{name}]")
    print(f"  Train: HIGH={int((y_tr==1).sum())}, NORM={int((y_tr==0).sum())}")
    print(f"  Test : HIGH={int((y_te==1).sum())}, NORM={int((y_te==0).sum())}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  FPR       : {fpr*100:.2f}%")
    print(f"  Confusion Matrix: {cm.tolist()}")
