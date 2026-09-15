import os
import joblib
import pandas as pd
import numpy as np

from energy_advisor import load_and_clean_data, prepare_features_and_targets, CSV_PATH

df_raw, _ = load_and_clean_data(CSV_PATH)
df_features = prepare_features_and_targets(df_raw)

MODEL_DIR = r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\saved_models'
m_r1_next = joblib.load(os.path.join(MODEL_DIR, "r1_next_hour_rf.pkl"))
m_r2_next = joblib.load(os.path.join(MODEL_DIR, "r2_next_hour_rf.pkl"))
m_r1_tmrw = joblib.load(os.path.join(MODEL_DIR, "r1_tomorrow_rf.pkl"))
m_r2_tmrw = joblib.load(os.path.join(MODEL_DIR, "r2_tomorrow_rf.pkl"))

feature_cols = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

sample = df_features[df_features['power1'] > 1700].iloc[0]
sample_df = pd.DataFrame([sample])[feature_cols]

p1_pred = m_r1_next.predict(sample_df)[0]
p1_proba = m_r1_next.predict_proba(sample_df)[0][p1_pred] * 100

p1_tmrw_pred = m_r1_tmrw.predict(sample_df)[0]
p1_tmrw_proba = m_r1_tmrw.predict_proba(sample_df)[0][p1_tmrw_pred] * 100

print(f"Room 1 Power: {sample['power1']} W")
print(f"Next Hour: {'HIGH USAGE' if p1_pred == 1 else 'NORMAL'} (Confidence: {p1_proba:.1f}%)")
print(f"Tomorrow: {'HIGH USAGE' if p1_tmrw_pred == 1 else 'NORMAL'} (Confidence: {p1_tmrw_proba:.1f}%)")
