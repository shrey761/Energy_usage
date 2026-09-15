import os
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = r'C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf\saved_models'
m = joblib.load(os.path.join(MODEL_DIR, "r1_next_hour_rf.pkl"))
print("Model classes_:", m.classes_)
feature_cols = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

sample_df = pd.DataFrame([{col: 0.0 for col in feature_cols}])
sample_df['power1'] = 1750.0
sample_df['hour'] = 18

pred = m.predict(sample_df)[0]
proba = m.predict_proba(sample_df)[0]
confidence = proba[pred] * 100
print(f"Predicted class: {pred}, Proba array: {proba}, Confidence: {confidence:.1f}%")
