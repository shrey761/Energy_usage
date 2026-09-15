import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

csv_path = r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\LATEST_ML_TRAINING_DATA.csv'
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.sort_values('timestamp').reset_index(drop=True)

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

df['power1_next'] = df['power1'].shift(-1)
df['power2_next'] = df['power2'].shift(-1)

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

X_tr = train_df[feature_cols]
X_te = test_df[feature_cols]

for r_name, p_next, p_curr in [("Room 1", "power1_next", "power1"), ("Room 2", "power2_next", "power2")]:
    y_tr = train_df[p_next]
    y_te = test_df[p_next]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    
    y_persist = test_df[p_curr]
    y_mean = np.full_like(y_te, y_tr.mean())
    
    print(f"\n[{r_name} Regressor on 55k rows]")
    print(f"  RF Model    -> MAE: {mean_absolute_error(y_te, y_pred):.2f} W | RMSE: {root_mean_squared_error(y_te, y_pred):.2f} W | R2: {r2_score(y_te, y_pred):.4f}")
    print(f"  Persistence -> MAE: {mean_absolute_error(y_te, y_persist):.2f} W | RMSE: {root_mean_squared_error(y_te, y_persist):.2f} W | R2: {r2_score(y_te, y_persist):.4f}")
    print(f"  Hist Mean   -> MAE: {mean_absolute_error(y_te, y_mean):.2f} W | RMSE: {root_mean_squared_error(y_te, y_mean):.2f} W | R2: {r2_score(y_te, y_mean):.4f}")
