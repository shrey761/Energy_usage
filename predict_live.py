#!/usr/bin/env python3
"""
VOLTIX — Live Energy Usage Predictor (Enhanced Random Forest Production Integration)
Script: predict_live.py
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
SCHEMA_PATH = os.path.join(MODEL_DIR, "enhanced_features.json")

# Baseline 23 Features (for tomorrow and regression models)
BASELINE_FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

FEATURE_COLS = BASELINE_FEATURE_COLS  # For backwards compatibility

def load_enhanced_feature_schema():
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, 'r') as f:
            return json.load(f)['features']
    return []

ENHANCED_FEATURE_COLS = load_enhanced_feature_schema()
MIN_HISTORY_REQUIRED = 360

THRESHOLD_R1 = 40.0
THRESHOLD_R2 = 10.0

def load_trained_models(model_dir=MODEL_DIR):
    models = {}
    model_names = ['r1_next_hour', 'r2_next_hour', 'r1_reg_next', 'r2_reg_next', 'r1_tomorrow', 'r2_tomorrow']
    for name in model_names:
        path = os.path.join(model_dir, f"{name}_rf.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found.")
        models[name] = joblib.load(path)
    return models

def build_enhanced_feature_vector(history_df):
    """
    Constructs the exact 126-feature vector from historical telemetry.
    Requires at least MIN_HISTORY_REQUIRED (360) sorted continuous readings.
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
        
    if 'voltage1' not in df.columns: df['voltage1'] = 230.0
    if 'voltage2' not in df.columns: df['voltage2'] = 230.0
    if 'power1' not in df.columns: df['power1'] = 0.0
    if 'power2' not in df.columns: df['power2'] = 0.0
    if 'totalPower' not in df.columns: df['totalPower'] = df['power1'] + df['power2']
    if 'current1' not in df.columns: df['current1'] = df['power1'] / np.maximum(df['voltage1'], 1.0)
    if 'current2' not in df.columns: df['current2'] = df['power2'] / np.maximum(df['voltage2'], 1.0)
    if 'totalCurrent' not in df.columns: df['totalCurrent'] = df['current1'] + df['current2']
    if 'energy1' not in df.columns: df['energy1'] = 0.0
    if 'energy2' not in df.columns: df['energy2'] = 0.0
    if 'totalEnergy' not in df.columns: df['totalEnergy'] = 0.0
            
    latest_ts = df['timestamp'].iloc[-1]
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['timestamp'].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    df['power_ratio'] = df['power1'] / (df['totalPower'] + 1e-6)
    df['apparent_power_1'] = df['voltage1'] * df['current1']
    df['apparent_power_2'] = df['voltage2'] * df['current2']
    
    new_cols = {}
    lags = [1, 3, 6, 30, 60, 180, 360]
    for col in ['power1', 'power2', 'totalPower']:
        for lag in lags:
            new_cols[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    windows = {'1m': 6, '5m': 30, '15m': 90, '30m': 180, '1h': 360}
    for col in ['power1', 'power2', 'totalPower']:
        for w_name, w_size in windows.items():
            roll = df[col].rolling(window=w_size, min_periods=1)
            new_cols[f'{col}_roll_mean{w_name}'] = roll.mean()
            new_cols[f'{col}_roll_min{w_name}'] = roll.min()
            new_cols[f'{col}_roll_max{w_name}'] = roll.max()
            new_cols[f'{col}_roll_std{w_name}'] = roll.std().fillna(0.0)
            
    for col in ['power1', 'power2', 'totalPower']:
        new_cols[f'{col}_diff_1m'] = df[col] - df[col].shift(6)
        new_cols[f'{col}_diff_5m'] = df[col] - df[col].shift(30)
        new_cols[f'{col}_diff_15m'] = df[col] - df[col].shift(90)
        new_cols[f'{col}_diff_1h'] = df[col] - df[col].shift(360)
        
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
    
    df['date'] = df['timestamp'].dt.date
    today_date = latest_ts.date()
    today_mask = (df['date'] == today_date)
    
    df['r1_daily_cum_on_min'] = 0.0
    df.loc[today_mask, 'r1_daily_cum_on_min'] = ((df.loc[today_mask, 'power1'] >= THRESHOLD_R1).astype(float) * (10.0 / 60.0)).cumsum()
    
    df['r2_daily_cum_on_min'] = 0.0
    df.loc[today_mask, 'r2_daily_cum_on_min'] = ((df.loc[today_mask, 'power2'] >= THRESHOLD_R2).astype(float) * (10.0 / 60.0)).cumsum()
    
    latest_row = df.iloc[-1:]
    
    # Assert exact 126 features and order
    if len(ENHANCED_FEATURE_COLS) > 0:
        generated = [c for c in ENHANCED_FEATURE_COLS if c in latest_row.columns]
        if len(generated) != 126 or generated != ENHANCED_FEATURE_COLS:
            raise ValueError(f"Feature schema mismatch! Expected 126 features. Found {len(generated)}")
        X_enh = latest_row[ENHANCED_FEATURE_COLS]
    else:
        X_enh = latest_row
        
    return X_enh, latest_ts, float(latest_row['power1'].values[0]), float(latest_row['power2'].values[0])

def build_baseline_feature_vector(latest_reading, history_df=None):
    """
    Constructs the 23-feature vector for tomorrow and regression models.
    """
    ts = latest_reading.get('timestamp', None)
    if isinstance(ts, str):
        try: dt = pd.to_datetime(ts)
        except Exception: dt = datetime.now()
    elif isinstance(ts, (datetime, pd.Timestamp)): dt = ts
    else: dt = datetime.now()

    hour = int(latest_reading.get('hour', dt.hour))
    dayofweek = int(latest_reading.get('dayofweek', dt.weekday()))
    is_weekend = 1 if dayofweek in [5, 6] else 0

    p1 = float(latest_reading.get('power1', 0.0))
    p2 = float(latest_reading.get('power2', 0.0))
    v1 = float(latest_reading.get('voltage1', 230.0))
    v2 = float(latest_reading.get('voltage2', 230.0))
    c1 = float(latest_reading.get('current1', p1 / max(v1, 1.0)))
    c2 = float(latest_reading.get('current2', p2 / max(v2, 1.0)))
    tp = float(latest_reading.get('totalPower', p1 + p2))
    tc = float(latest_reading.get('totalCurrent', c1 + c2))

    if history_df is not None and len(history_df) >= 1:
        hist = history_df.sort_values('timestamp').reset_index(drop=True)
        p1_lag1 = float(hist['power1'].iloc[-2] if len(hist) >= 2 else p1)
        p1_lag3 = float(hist['power1'].iloc[-4] if len(hist) >= 4 else p1)
        p1_mean5 = float(hist['power1'].iloc[-5:].mean())
        p1_max5 = float(hist['power1'].iloc[-5:].max())

        p2_lag1 = float(hist['power2'].iloc[-2] if len(hist) >= 2 else p2)
        p2_lag3 = float(hist['power2'].iloc[-4] if len(hist) >= 4 else p2)
        p2_mean5 = float(hist['power2'].iloc[-5:].mean())
        p2_max5 = float(hist['power2'].iloc[-5:].max())

        tp_lag1 = float(hist['totalPower'].iloc[-2] if len(hist) >= 2 else tp)
        tp_mean5 = float(hist['totalPower'].iloc[-5:].mean())
    else:
        p1_lag1 = float(latest_reading.get('power1_lag1', p1))
        p1_lag3 = float(latest_reading.get('power1_lag3', p1))
        p1_mean5 = float(latest_reading.get('power1_roll_mean5', p1))
        p1_max5 = float(latest_reading.get('power1_roll_max5', p1))

        p2_lag1 = float(latest_reading.get('power2_lag1', p2))
        p2_lag3 = float(latest_reading.get('power2_lag3', p2))
        p2_mean5 = float(latest_reading.get('power2_roll_mean5', p2))
        p2_max5 = float(latest_reading.get('power2_roll_max5', p2))

        tp_lag1 = float(latest_reading.get('totalPower_lag1', tp))
        tp_mean5 = float(latest_reading.get('totalPower_roll_mean5', tp))

    row_dict = {
        'hour': hour,
        'dayofweek': dayofweek,
        'is_weekend': is_weekend,
        'hour_sin': np.sin(2 * np.pi * hour / 24.0),
        'hour_cos': np.cos(2 * np.pi * hour / 24.0),
        'power1': p1,
        'power2': p2,
        'current1': c1,
        'current2': c2,
        'voltage1': v1,
        'voltage2': v2,
        'totalPower': tp,
        'totalCurrent': tc,
        'power1_lag1': p1_lag1,
        'power1_lag3': p1_lag3,
        'power1_roll_mean5': p1_mean5,
        'power1_roll_max5': p1_max5,
        'power2_lag1': p2_lag1,
        'power2_lag3': p2_lag3,
        'power2_roll_mean5': p2_mean5,
        'power2_roll_max5': p2_max5,
        'totalPower_lag1': tp_lag1,
        'totalPower_roll_mean5': tp_mean5
    }
    return pd.DataFrame([row_dict])[BASELINE_FEATURE_COLS], dt, p1, p2, tp

def predict_and_advise(models, power1, power2, voltage1=230.0, voltage2=230.0, hour=None, dayofweek=None, history_df=None):
    if hour is None: hour = datetime.now().hour
    if dayofweek is None: dayofweek = datetime.now().weekday()
    
    current1 = power1 / max(voltage1, 1.0)
    current2 = power2 / max(voltage2, 1.0)
    total_power = power1 + power2
    total_current = current1 + current2
    
    latest_reading = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'hour': hour, 'dayofweek': dayofweek,
        'power1': power1, 'power2': power2,
        'voltage1': voltage1, 'voltage2': voltage2,
        'current1': current1, 'current2': current2,
        'totalPower': total_power, 'totalCurrent': total_current
    }
    
    X_base, dt, p1, p2, tp = build_baseline_feature_vector(latest_reading, history_df)
    
    # Check if we have enhanced history buffer (360 samples)
    if history_df is not None and len(history_df) >= MIN_HISTORY_REQUIRED:
        X_enh, _, _, _ = build_enhanced_feature_vector(history_df)
    else:
        # Construct synthetic stationary history for single point invocation
        history_df = pd.DataFrame([latest_reading] * MIN_HISTORY_REQUIRED)
        history_df['timestamp'] = [datetime.now() - pd.Timedelta(seconds=(MIN_HISTORY_REQUIRED-1-i)*10) for i in range(MIN_HISTORY_REQUIRED)]
        X_enh, _, _, _ = build_enhanced_feature_vector(history_df)
        
    # Classification Outcomes (Enhanced RF for next hour)
    r1_next_class = int(models['r1_next_hour'].predict(X_enh)[0])
    r1_next_proba = models['r1_next_hour'].predict_proba(X_enh)[0][r1_next_class] * 100
    r1_next_state = "HIGH USAGE" if r1_next_class == 1 else "NORMAL"

    r2_next_class = int(models['r2_next_hour'].predict(X_enh)[0])
    r2_next_proba = models['r2_next_hour'].predict_proba(X_enh)[0][r2_next_class] * 100
    r2_next_state = "HIGH USAGE" if r2_next_class == 1 else "NORMAL"

    # Tomorrow Classification (Baseline RF)
    r1_tmrw_class = int(models['r1_tomorrow'].predict(X_base)[0])
    r1_tmrw_proba = models['r1_tomorrow'].predict_proba(X_base)[0][r1_tmrw_class] * 100
    r1_tmrw_state = "HIGH USAGE" if r1_tmrw_class == 1 else "NORMAL"

    r2_tmrw_class = int(models['r2_tomorrow'].predict(X_base)[0])
    r2_tmrw_proba = models['r2_tomorrow'].predict_proba(X_base)[0][r2_tmrw_class] * 100
    r2_tmrw_state = "HIGH USAGE" if r2_tmrw_class == 1 else "NORMAL"

    # Continuous Power Regression
    r1_pred_power = max(0.0, float(models['r1_reg_next'].predict(X_base)[0]))
    r2_pred_power = max(0.0, float(models['r2_reg_next'].predict(X_base)[0]))

    recommendations = []
    if r1_next_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 1 next hour. Reduce non-essential loads in Room 1.")
    if r2_next_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 2 next hour. Reduce non-essential loads in Room 2.")
    if r1_tmrw_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 1 tomorrow around this time. Plan to reduce usage in Room 1.")
    if r2_tmrw_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 2 tomorrow around this time. Plan to reduce usage in Room 2.")

    return {
        "room1_status": r1_next_state,
        "room1_probability": round(r1_next_proba, 1),
        "room1_tomorrow_status": r1_tmrw_state,
        "room1_tomorrow_probability": round(r1_tmrw_proba, 1),
        "room1_predicted_power": round(r1_pred_power, 1),
        "room2_status": r2_next_state,
        "room2_probability": round(r2_next_proba, 1),
        "room2_tomorrow_status": r2_tmrw_state,
        "room2_tomorrow_probability": round(r2_tmrw_proba, 1),
        "room2_predicted_power": round(r2_pred_power, 1),
        "total_current_power": round(total_power, 1),
        "total_predicted_power": round(r1_pred_power + r2_pred_power, 1),
        "recommendations": recommendations
    }

if __name__ == '__main__':
    models = load_trained_models()
    res = predict_and_advise(models, power1=80.0, power2=20.0)
    print("Standalone Production Prediction Test:")
    print(json.dumps(res, indent=2))
