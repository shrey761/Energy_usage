#!/usr/bin/env python3
"""
VOLTIX — Isolated Enhanced Random Forest Live Predictor
Script: enhanced_deployment/predict_live.py
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "realistic_dataset_experiment", "models")
SCHEMA_PATH = os.path.join(MODELS_DIR, "enhanced_features.json")

THRESHOLD_R1 = 40.0
THRESHOLD_R2 = 10.0
MIN_HISTORY_REQUIRED = 360  # 360 samples at 10s = 1 hour

def load_expected_feature_schema():
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Feature schema not found at {SCHEMA_PATH}")
    with open(SCHEMA_PATH, 'r') as f:
        data = json.load(f)
    return data['features']

EXPECTED_FEATURES = load_expected_feature_schema()
assert len(EXPECTED_FEATURES) == 126, f"Expected 126 features, found {len(EXPECTED_FEATURES)}"

def load_enhanced_models(models_dir=MODELS_DIR):
    models = {}
    r1_path = os.path.join(models_dir, "r1_enhanced_rf.pkl")
    r2_path = os.path.join(models_dir, "r2_enhanced_rf.pkl")
    
    if not os.path.exists(r1_path):
        raise FileNotFoundError(f"Enhanced Room 1 model not found: {r1_path}")
    if not os.path.exists(r2_path):
        raise FileNotFoundError(f"Enhanced Room 2 model not found: {r2_path}")
        
    models['r1_next_hour'] = joblib.load(r1_path)
    models['r2_next_hour'] = joblib.load(r2_path)
    return models

def build_126_feature_vector(history_df):
    """
    Constructs the exact 126 causal feature vector from historical telemetry.
    Requires at least MIN_HISTORY_REQUIRED (360) sorted continuous readings ending at time t.
    """
    if history_df is None or len(history_df) < MIN_HISTORY_REQUIRED:
        avail = len(history_df) if history_df is not None else 0
        raise ValueError(f"Insufficient historical data for Enhanced RF prediction. Required: {MIN_HISTORY_REQUIRED}, Available: {avail}")
        
    df = history_df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        raise ValueError("Missing 'timestamp' column in history DataFrame.")
        
    required_cols = ['current1', 'current2', 'energy1', 'energy2', 'power1', 'power2',
                     'totalCurrent', 'totalEnergy', 'totalPower', 'voltage1', 'voltage2']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required telemetry field: '{c}'")
            
    # Base datetime features
    latest_ts = df['timestamp'].iloc[-1]
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['timestamp'].dt.month
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    # Derived electrical features
    df['power_ratio'] = df['power1'] / (df['totalPower'] + 1e-6)
    df['apparent_power_1'] = df['voltage1'] * df['current1']
    df['apparent_power_2'] = df['voltage2'] * df['current2']
    
    new_cols = {}
    
    # Extended Lags (10s, 30s, 1m, 5m, 10m, 30m, 1h)
    lags = [1, 3, 6, 30, 60, 180, 360]
    for col in ['power1', 'power2', 'totalPower']:
        for lag in lags:
            new_cols[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    # Rolling Statistics over causal windows (1m=6, 5m=30, 15m=90, 30m=180, 1h=360)
    windows = {'1m': 6, '5m': 30, '15m': 90, '30m': 180, '1h': 360}
    for col in ['power1', 'power2', 'totalPower']:
        for w_name, w_size in windows.items():
            roll = df[col].rolling(window=w_size, min_periods=1)
            new_cols[f'{col}_roll_mean{w_name}'] = roll.mean()
            new_cols[f'{col}_roll_min{w_name}'] = roll.min()
            new_cols[f'{col}_roll_max{w_name}'] = roll.max()
            new_cols[f'{col}_roll_std{w_name}'] = roll.std().fillna(0.0)
            
    # Trend differences
    for col in ['power1', 'power2', 'totalPower']:
        new_cols[f'{col}_diff_1m'] = df[col] - df[col].shift(6)
        new_cols[f'{col}_diff_5m'] = df[col] - df[col].shift(30)
        new_cols[f'{col}_diff_15m'] = df[col] - df[col].shift(90)
        new_cols[f'{col}_diff_1h'] = df[col] - df[col].shift(360)
        
    # State durations
    def compute_durations(power_series, threshold):
        is_on = (power_series >= threshold).astype(int).values
        n = len(is_on)
        consec_on = np.zeros(n)
        consec_off = np.zeros(n)
        since_trans = np.zeros(n)
        
        cur_on = 0
        cur_off = 0
        cur_trans = 0
        for i in range(n):
            if is_on[i] == 1:
                cur_on += 1
                cur_off = 0
            else:
                cur_off += 1
                cur_on = 0
            
            if i > 0 and is_on[i] != is_on[i-1]:
                cur_trans = 0
            else:
                cur_trans += 1
                
            consec_on[i] = cur_on * (10.0 / 60.0)
            consec_off[i] = cur_off * (10.0 / 60.0)
            since_trans[i] = cur_trans * (10.0 / 60.0)
            
        return is_on, consec_on, consec_off, since_trans

    r1_on_state, r1_on, r1_off, r1_trans = compute_durations(df['power1'], THRESHOLD_R1)
    new_cols['r1_is_on'] = r1_on_state
    new_cols['r1_consec_on_min'] = r1_on
    new_cols['r1_consec_off_min'] = r1_off
    new_cols['r1_min_since_trans'] = r1_trans
    
    r2_on_state, r2_on, r2_off, r2_trans = compute_durations(df['power2'], THRESHOLD_R2)
    new_cols['r2_is_on'] = r2_on_state
    new_cols['r2_consec_on_min'] = r2_on
    new_cols['r2_consec_off_min'] = r2_off
    new_cols['r2_min_since_trans'] = r2_trans
    
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    
    # Cumulative daily minutes (only for the current calendar date)
    df['date'] = df['timestamp'].dt.date
    today_date = latest_ts.date()
    today_mask = (df['date'] == today_date)
    
    df['r1_daily_cum_on_min'] = 0.0
    df.loc[today_mask, 'r1_daily_cum_on_min'] = ((df.loc[today_mask, 'power1'] >= THRESHOLD_R1).astype(float) * (10.0 / 60.0)).cumsum()
    
    df['r2_daily_cum_on_min'] = 0.0
    df.loc[today_mask, 'r2_daily_cum_on_min'] = ((df.loc[today_mask, 'power2'] >= THRESHOLD_R2).astype(float) * (10.0 / 60.0)).cumsum()
    
    # Extract the final time t observation
    latest_row = df.iloc[-1:]
    
    # Automated assertion of exact feature list and ordering
    generated_features = [c for c in EXPECTED_FEATURES if c in latest_row.columns]
    if len(generated_features) != 126 or generated_features != EXPECTED_FEATURES:
        raise ValueError(f"Feature schema mismatch! Expected 126 features matching schema. Found {len(generated_features)}")
        
    X_latest = latest_row[EXPECTED_FEATURES]
    return X_latest, latest_ts, float(latest_row['power1'].values[0]), float(latest_row['power2'].values[0])

def predict_enhanced(models, history_df):
    """
    Executes Enhanced Random Forest inference on the latest reading using historical context.
    """
    X, ts, p1, p2 = build_126_feature_vector(history_df)
    
    # Room 1 Prediction
    r1_clf = models['r1_next_hour']
    r1_pred = int(r1_clf.predict(X)[0])
    r1_probs = r1_clf.predict_proba(X)[0]
    r1_state = "HIGH USAGE" if r1_pred == 1 else "NORMAL"
    
    # Room 2 Prediction
    r2_clf = models['r2_next_hour']
    r2_pred = int(r2_clf.predict(X)[0])
    r2_probs = r2_clf.predict_proba(X)[0]
    r2_state = "HIGH USAGE" if r2_pred == 1 else "NORMAL"
    
    results = {
        'timestamp': str(ts),
        'power1_current': p1,
        'power2_current': p2,
        'room1': {
            'prediction': r1_state,
            'prediction_code': r1_pred,
            'probability_normal': float(r1_probs[0]),
            'probability_high': float(r1_probs[1]),
            'confidence': float(r1_probs[r1_pred]),
            'threshold_w': THRESHOLD_R1
        },
        'room2': {
            'prediction': r2_state,
            'prediction_code': r2_pred,
            'probability_normal': float(r2_probs[0]),
            'probability_high': float(r2_probs[1]),
            'confidence': float(r2_probs[r2_pred]),
            'threshold_w': THRESHOLD_R2
        },
        'feature_count': X.shape[1],
        'schema_match': True
    }
    return results

if __name__ == '__main__':
    print("[*] Enhanced Predict Live module loaded successfully.")
    print(f"[*] Schema Verified: {len(EXPECTED_FEATURES)} features loaded from {SCHEMA_PATH}")
