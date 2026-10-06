#!/usr/bin/env python3
"""
Test inference using Enhanced Random Forest models in realistic_dataset_experiment.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf"
EXP_DIR = os.path.join(BASE_DIR, "realistic_dataset_experiment")
MODELS_DIR = os.path.join(EXP_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.append(BASE_DIR)
from realistic_dataset_experiment.run_realistic_experiment import engineer_features, validate_dataset

def main():
    print("[*] Loading dataset & generating features...")
    df_raw = pd.read_csv(os.path.join(BASE_DIR, "VOLTIX_realistic_30day_synthetic_dataset.csv"))
    df, _ = validate_dataset(df_raw)
    df, ENHANCED_FEATURES = engineer_features(df)

    df['power1_next'] = df['power1'].shift(-360)
    df['power2_next'] = df['power2'].shift(-360)
    df['high_usage_r1_next'] = (df['power1_next'] >= 40.0).astype(int)
    df['high_usage_r2_next'] = (df['power2_next'] >= 10.0).astype(int)

    df_1h = df.iloc[:-360].copy().reset_index(drop=True)
    train_size = int(len(df_1h) * 0.80)
    train_1h = df_1h.iloc[:train_size]
    test_1h = df_1h.iloc[train_size:]

    # Train and serialize Enhanced RF models to models/ directory
    r1_model_path = os.path.join(MODELS_DIR, "r1_enhanced_rf.pkl")
    r2_model_path = os.path.join(MODELS_DIR, "r2_enhanced_rf.pkl")

    if not os.path.exists(r1_model_path):
        print("[*] Training Enhanced RF Room 1 (500 trees, depth 15)...")
        from sklearn.ensemble import RandomForestClassifier
        rf_r1 = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=1,
                                       max_features='sqrt', random_state=42, n_jobs=-1)
        rf_r1.fit(train_1h[ENHANCED_FEATURES], train_1h['high_usage_r1_next'])
        joblib.dump(rf_r1, r1_model_path)
    else:
        print(f"[*] Loading existing {r1_model_path}...")
        rf_r1 = joblib.load(r1_model_path)

    if not os.path.exists(r2_model_path):
        print("[*] Training Enhanced RF Room 2 (300 trees, depth 25)...")
        from sklearn.ensemble import RandomForestClassifier
        rf_r2 = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_leaf=2,
                                       max_features='sqrt', random_state=42, n_jobs=-1)
        rf_r2.fit(train_1h[ENHANCED_FEATURES], train_1h['high_usage_r2_next'])
        joblib.dump(rf_r2, r2_model_path)
    else:
        print(f"[*] Loading existing {r2_model_path}...")
        rf_r2 = joblib.load(r2_model_path)

    # Save feature metadata
    with open(os.path.join(MODELS_DIR, "enhanced_features.json"), "w") as f:
        json.dump({"feature_count": len(ENHANCED_FEATURES), "features": ENHANCED_FEATURES}, f, indent=2)

    # Select test sample
    sample_idx = 1000
    sample_row = test_1h.iloc[sample_idx:sample_idx+1]
    X_sample = sample_row[ENHANCED_FEATURES]

    r1_pred = rf_r1.predict(X_sample)[0]
    r1_prob = rf_r1.predict_proba(X_sample)[0]
    r2_pred = rf_r2.predict(X_sample)[0]
    r2_prob = rf_r2.predict_proba(X_sample)[0]

    ts_str = str(sample_row['timestamp'].values[0])
    act_r1_cls = "HIGH USAGE" if sample_row['high_usage_r1_next'].values[0] == 1 else "NORMAL"
    act_r2_cls = "HIGH USAGE" if sample_row['high_usage_r2_next'].values[0] == 1 else "NORMAL"
    act_p1 = float(sample_row['power1_next'].values[0])
    act_p2 = float(sample_row['power2_next'].values[0])

    print("\n" + "=" * 80)
    print(" TEST INFERENCE RESULTS (Enhanced Random Forest)")
    print("=" * 80)
    print(f"[*] Test Sample Timestamp : {ts_str}")
    print(f"[*] Total Features Fed    : {X_sample.shape[1]}")
    print(f"[*] Room 1 Prediction     : {'HIGH USAGE' if r1_pred == 1 else 'NORMAL'} (Probability: Normal={r1_prob[0]*100:.2f}%, High={r1_prob[1]*100:.2f}%)")
    print(f"[*] Room 1 Actual (t+1h)  : {act_r1_cls} ({act_p1:.2f} W)")
    print(f"[*] Room 2 Prediction     : {'HIGH USAGE' if r2_pred == 1 else 'NORMAL'} (Probability: Normal={r2_prob[0]*100:.2f}%, High={r2_prob[1]*100:.2f}%)")
    print(f"[*] Room 2 Actual (t+1h)  : {act_r2_cls} ({act_p2:.2f} W)")
    print("=" * 80)

if __name__ == "__main__":
    main()
