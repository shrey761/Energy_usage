import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')

# 1. Validation & Sorting
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

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

# Next horizon target
df['power1_next'] = df['power1'].shift(-1)
df['power2_next'] = df['power2'].shift(-1)

THRESHOLD = 1200.0
df['high_usage_r1_next'] = (df['power1_next'] >= THRESHOLD).astype(int)
df['high_usage_r2_next'] = (df['power2_next'] >= THRESHOLD).astype(int)

# Tomorrow same hour target
hourly_r1_mean = df.groupby('hour')['power1'].transform('mean')
hourly_r2_mean = df.groupby('hour')['power2'].transform('mean')
df['high_usage_r1_tomorrow'] = ((hourly_r1_mean >= THRESHOLD) | (df['power1'] >= THRESHOLD)).astype(int)
df['high_usage_r2_tomorrow'] = ((hourly_r2_mean >= THRESHOLD) | (df['power2'] >= THRESHOLD)).astype(int)

df = df.dropna(subset=['power1_next', 'power2_next']).reset_index(drop=True)

feature_cols = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Evaluate Classifier
for room, target_col in [("Room 1", 'high_usage_r1_next'), ("Room 2", 'high_usage_r2_next')]:
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\n=================== {room} Classification Metrics ===================")
    print(f"Training Class Dist: HIGH={int((y_train==1).sum())}, NORMAL={int((y_train==0).sum())}")
    print(f"Testing Class Dist : HIGH={int((y_test==1).sum())}, NORMAL={int((y_test==0).sum())}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

# Evaluate Regressor
for room, target_col, current_col in [("Room 1", 'power1_next', 'power1'), ("Room 2", 'power2_next', 'power2')]:
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    # Baseline 1: Persistence (previous/current value)
    y_persist = test_df[current_col]
    # Baseline 2: Mean baseline
    y_mean = np.full_like(y_test, y_train.mean())
    
    print(f"\n=================== {room} Power Regression (Watts) ===================")
    print(f"RF Model       -> MAE: {mean_absolute_error(y_test, y_pred):.2f} W | RMSE: {root_mean_squared_error(y_test, y_pred):.2f} W | R2: {r2_score(y_test, y_pred):.4f}")
    print(f"Persistence BL -> MAE: {mean_absolute_error(y_test, y_persist):.2f} W | RMSE: {root_mean_squared_error(y_test, y_persist):.2f} W | R2: {r2_score(y_test, y_persist):.4f}")
    print(f"Hist Mean BL   -> MAE: {mean_absolute_error(y_test, y_mean):.2f} W | RMSE: {root_mean_squared_error(y_test, y_mean):.2f} W | R2: {r2_score(y_test, y_mean):.4f}")
