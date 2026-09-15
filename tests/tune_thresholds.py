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

# Keep records with valid 24h future label
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

split_idx = int(len(df_24h_valid) * 0.8)
train_df = df_24h_valid.iloc[:split_idx]
test_df = df_24h_valid.iloc[split_idx:]

X_tr = train_df[feature_cols]
X_te = test_df[feature_cols]

# Sub-split training chronologically (80% sub-train / 20% validation) to tune probability threshold without test leakage
sub_split = int(len(train_df) * 0.8)
sub_X_tr = X_tr.iloc[:sub_split]
sub_X_val = X_tr.iloc[sub_split:]

for target_col, name in [('high_usage_r1_tomorrow', 'Room 1 Tomorrow'), ('high_usage_r2_tomorrow', 'Room 2 Tomorrow')]:
    sub_y_tr = train_df[target_col].iloc[:sub_split]
    sub_y_val = train_df[target_col].iloc[sub_split:]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(sub_X_tr, sub_y_tr)
    val_probas = clf.predict_proba(sub_X_val)[:, 1]
    
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.3, 0.8, 0.05):
        val_preds = (val_probas >= thresh).astype(int)
        f1 = f1_score(sub_y_val, val_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"\n[{name}] Best validation threshold selected from training fold: {best_thresh:.2f} (Val F1={best_f1:.4f})")
    
    # Train on full train set and evaluate on test set with best_thresh
    clf_full = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_full.fit(X_tr, train_df[target_col])
    test_probas = clf_full.predict_proba(X_te)[:, 1]
    test_preds = (test_probas >= best_thresh).astype(int)
    
    y_te = test_df[target_col]
    acc = accuracy_score(y_te, test_preds)
    prec = precision_score(y_te, test_preds, zero_division=0)
    rec = recall_score(y_te, test_preds, zero_division=0)
    f1 = f1_score(y_te, test_preds, zero_division=0)
    cm = confusion_matrix(y_te, test_preds)
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    
    print(f"  Test Accuracy  : {acc*100:.2f}%")
    print(f"  Test Precision : {prec:.4f}")
    print(f"  Test Recall    : {rec:.4f}")
    print(f"  Test F1-Score  : {f1:.4f}")
    print(f"  Test FPR       : {fpr*100:.2f}%")
    print(f"  Confusion Matrix: {cm.tolist()}")
