import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

df = pd.read_csv(r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\energy_data.csv')
print("Raw rows:", len(df))

# 1. Validation & Sorting
df = df[df['timestamp'] != 'TIME_ERROR'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
print("Rows after timestamp validation & sorting:", len(df))

# 2. Feature Engineering
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
print("Rows after feature engineering (dropped 1 trailing row for shift(-1)):", len(df))

# Check Chronological Split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print("\n--- Room 1 Next Usage Class Distribution ---")
print("Train HIGH:", (train_df['high_usage_r1_next'] == 1).sum(), "NORMAL:", (train_df['high_usage_r1_next'] == 0).sum())
print("Test HIGH:", (test_df['high_usage_r1_next'] == 1).sum(), "NORMAL:", (test_df['high_usage_r1_next'] == 0).sum())

print("\n--- Room 2 Next Usage Class Distribution ---")
print("Train HIGH:", (train_df['high_usage_r2_next'] == 1).sum(), "NORMAL:", (train_df['high_usage_r2_next'] == 0).sum())
print("Test HIGH:", (test_df['high_usage_r2_next'] == 1).sum(), "NORMAL:", (test_df['high_usage_r2_next'] == 0).sum())
