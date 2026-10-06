#!/usr/bin/env python3
"""
VOLTIX — Realistic 30-Day Synthetic Dataset Experiment
Script: run_realistic_experiment.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb

# Configuration & Paths
BASE_DIR = r"C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf"
DATASET_PATH = os.path.join(BASE_DIR, "VOLTIX_realistic_30day_synthetic_dataset.csv")
EXP_DIR = os.path.join(BASE_DIR, "realistic_dataset_experiment")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")

os.makedirs(GRAPHS_DIR, exist_ok=True)

THRESHOLD_R1 = 40.0
THRESHOLD_R2 = 10.0

BASELINE_FEATURES = [
    'current1', 'current2', 'energy1', 'energy2', 'power1', 'power2',
    'voltage1', 'voltage2', 'totalCurrent', 'totalEnergy', 'totalPower',
    'hour', 'minute', 'day_of_week', 'is_weekend', 'month',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'power_ratio', 'apparent_power_1', 'apparent_power_2'
]

def validate_dataset(df):
    print("=" * 80)
    print(" 1. DATASET VALIDATION & INTEGRITY AUDIT")
    print("=" * 80)
    
    validation_report = {}
    
    # 1. Shape
    rows, cols = df.shape
    print(f"[*] Total Rows: {rows} (Expected: 259,200)")
    print(f"[*] Total Columns: {cols} (Expected: 13)")
    validation_report['rows'] = rows
    validation_report['cols'] = cols
    
    # 2. Timestamp validation
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    start_ts = df['timestamp'].min()
    end_ts = df['timestamp'].max()
    print(f"[*] Start Timestamp : {start_ts}")
    print(f"[*] End Timestamp   : {end_ts}")
    validation_report['start_ts'] = str(start_ts)
    validation_report['end_ts'] = str(end_ts)
    
    # 3. Sampling intervals
    time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
    median_diff = time_diffs.median()
    min_diff = time_diffs.min()
    max_diff = time_diffs.max()
    print(f"[*] Sampling Interval : Median={median_diff}s, Min={min_diff}s, Max={max_diff}s")
    validation_report['median_interval_sec'] = median_diff
    
    # 4. Duplicates and Nulls
    dup_ts = df['timestamp'].duplicated().sum()
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    print(f"[*] Duplicate Timestamps : {dup_ts}")
    print(f"[*] Total Null Values    : {total_nulls}")
    validation_report['dup_timestamps'] = int(dup_ts)
    validation_report['total_nulls'] = int(total_nulls)
    
    # 5. Physical Range Checks
    v1_min, v1_max = df['voltage1'].min(), df['voltage1'].max()
    v2_min, v2_max = df['voltage2'].min(), df['voltage2'].max()
    p1_min, p1_max = df['power1'].min(), df['power1'].max()
    p2_min, p2_max = df['power2'].min(), df['power2'].max()
    p_tot_min, p_tot_max = df['totalPower'].min(), df['totalPower'].max()
    c1_min, c1_max = df['current1'].min(), df['current1'].max()
    c2_min, c2_max = df['current2'].min(), df['current2'].max()
    
    print(f"[*] Voltage Ranges : V1=[{v1_min:.2f}, {v1_max:.2f}]V, V2=[{v2_min:.2f}, {v2_max:.2f}]V")
    print(f"[*] Current Ranges : I1=[{c1_min:.4f}, {c1_max:.4f}]A, I2=[{c2_min:.4f}, {c2_max:.4f}]A")
    print(f"[*] Power Ranges   : P1=[{p1_min:.2f}, {p1_max:.2f}]W, P2=[{p2_min:.2f}, {p2_max:.2f}]W, Tot=[{p_tot_min:.2f}, {p_tot_max:.2f}]W")
    
    # Active percentages
    r1_active_pct = (df['power1'] >= THRESHOLD_R1).mean() * 100.0
    r2_active_pct = (df['power2'] >= THRESHOLD_R2).mean() * 100.0
    print(f"[*] Room 1 Active (>=40W) : {r1_active_pct:.2f}%")
    print(f"[*] Room 2 Active (>=10W) : {r2_active_pct:.2f}%")
    
    # Total power consistency check
    p_diff = np.abs(df['totalPower'] - (df['power1'] + df['power2'])).max()
    print(f"[*] Total Power Max Discrepancy : {p_diff:.4f} W")
    
    return df, validation_report

def engineer_features(df):
    print("\n" + "=" * 80)
    print(" 2. CAUSAL FEATURE ENGINEERING (137 FEATURES)")
    print("=" * 80)
    
    # Base datetime features
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
    
    # Ratios and apparent power
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
    
    # Concat all new columns
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    
    # Cumulative daily minutes
    df['date'] = df['timestamp'].dt.date
    df['r1_daily_cum_on_min'] = df.groupby('date')['power1'].transform(lambda s: ((s >= THRESHOLD_R1).astype(float) * (10.0/60.0)).cumsum())
    df['r2_daily_cum_on_min'] = df.groupby('date')['power2'].transform(lambda s: ((s >= THRESHOLD_R2).astype(float) * (10.0/60.0)).cumsum())
    
    # Strictly causal: drop first 360 rows (warmup period for 1-hour lag)
    df = df.iloc[360:].reset_index(drop=True)
    
    # Exclude non-feature columns
    non_features = ['reading_id', 'timestamp', 'date', 'power1_next', 'power2_next', 
                    'high_usage_r1_next', 'high_usage_r2_next',
                    'power1_tomorrow', 'power2_tomorrow',
                    'high_usage_r1_tomorrow', 'high_usage_r2_tomorrow']
    enhanced_features = [c for c in df.columns if c not in non_features]
    
    print(f"[*] Baseline Features Count : {len(BASELINE_FEATURES)}")
    print(f"[*] Enhanced Features Count : {len(enhanced_features)}")
    print(f"[*] Total Causal Records    : {len(df)}")
    
    return df, enhanced_features

def run_experiment():
    # 1. Load Data
    print(f"[*] Loading dataset from {DATASET_PATH}...")
    df_raw = pd.read_csv(DATASET_PATH)
    
    # 2. Validation
    df, val_report = validate_dataset(df_raw)
    
    # 3. Feature Engineering
    df, ENHANCED_FEATURES = engineer_features(df)
    
    # 4. Target Generation
    # 1-Hour Horizon (360 steps)
    df['power1_next'] = df['power1'].shift(-360)
    df['power2_next'] = df['power2'].shift(-360)
    df['high_usage_r1_next'] = (df['power1_next'] >= THRESHOLD_R1).astype(int)
    df['high_usage_r2_next'] = (df['power2_next'] >= THRESHOLD_R2).astype(int)
    
    # 24-Hour Horizon (8640 steps)
    df['power1_tomorrow'] = df['power1'].shift(-8640)
    df['power2_tomorrow'] = df['power2'].shift(-8640)
    df['high_usage_r1_tomorrow'] = (df['power1_tomorrow'] >= THRESHOLD_R1).astype(int)
    df['high_usage_r2_tomorrow'] = (df['power2_tomorrow'] >= THRESHOLD_R2).astype(int)
    
    # 5. Horizon 1-Hour Setup (Drop trailing 360 rows)
    df_1h = df.iloc[:-360].copy().reset_index(drop=True)
    N_1h = len(df_1h)
    train_size_1h = int(N_1h * 0.80)
    
    train_1h = df_1h.iloc[:train_size_1h]
    test_1h = df_1h.iloc[train_size_1h:]
    
    # Sub-train and Validation for tuning
    sub_train_size_1h = int(len(train_1h) * 0.75) # 60% of total
    sub_train_1h = train_1h.iloc[:sub_train_size_1h]
    val_1h = train_1h.iloc[sub_train_size_1h:]
    
    print("\n" + "=" * 80)
    print(" 3. 1-HOUR HORIZON BENCHMARKING (80% Train / 20% Untouched Test)")
    print("=" * 80)
    print(f"[*] 1-Hour Total Rows : {N_1h}")
    print(f"[*] Train Rows        : {len(train_1h)} (Sub-train: {len(sub_train_1h)}, Val: {len(val_1h)})")
    print(f"[*] Final Test Rows   : {len(test_1h)}")
    
    results = []
    
    # Evaluation storage for raw predictions & plots
    test_preds_r1 = {'timestamp': test_1h['timestamp'], 'actual_power': test_1h['power1_next'], 'actual_class': test_1h['high_usage_r1_next']}
    test_preds_r2 = {'timestamp': test_1h['timestamp'], 'actual_power': test_1h['power2_next'], 'actual_class': test_1h['high_usage_r2_next']}
    
    roc_curves = {}
    conf_matrices = {}
    feature_importances = {}
    
    # Helper to calculate classification metrics
    def eval_clf(y_true, y_pred, y_prob=None):
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
        return acc, bal_acc, prec, rec, f1, auc

    # Helper to calculate regression metrics
    def eval_reg(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    # --- A. PERSISTENCE BASELINE (1-Hour) ---
    for room, col, target_class, target_power in [('Room 1', 'power1', 'high_usage_r1_next', 'power1_next'),
                                                   ('Room 2', 'power2', 'high_usage_r2_next', 'power2_next')]:
        thresh = THRESHOLD_R1 if room == 'Room 1' else THRESHOLD_R2
        pers_pred_class = (test_1h[col] >= thresh).astype(int)
        pers_pred_power = test_1h[col]
        
        acc, bal_acc, prec, rec, f1, _ = eval_clf(test_1h[target_class], pers_pred_class)
        mae, rmse, r2 = eval_reg(test_1h[target_power], pers_pred_power)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Persistence',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': np.nan,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Persistence',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Regression',
            'accuracy': np.nan, 'balanced_accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan,
            'mae': mae, 'rmse': rmse, 'r2': r2
        })

    # --- B. BASELINE RANDOM FOREST (23 Features, 1-Hour) ---
    print("\n[*] Training Baseline Random Forest (23 Features, Primary VOLTIX Architecture)...")
    for room, target_class, target_power in [('Room 1', 'high_usage_r1_next', 'power1_next'),
                                             ('Room 2', 'high_usage_r2_next', 'power2_next')]:
        # Classifier
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_clf.fit(train_1h[BASELINE_FEATURES], train_1h[target_class])
        
        y_pred = rf_clf.predict(test_1h[BASELINE_FEATURES])
        y_prob = rf_clf.predict_proba(test_1h[BASELINE_FEATURES])[:, 1]
        acc, bal_acc, prec, rec, f1, auc = eval_clf(test_1h[target_class], y_pred, y_prob)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Baseline Random Forest (23 Feat)',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
        
        # Save baseline predictions
        if room == 'Room 1':
            test_preds_r1['rf23_pred_class'] = y_pred
            test_preds_r1['rf23_prob'] = y_prob
            roc_curves['Room 1 (Baseline RF)'] = (test_1h[target_class], y_prob)
            conf_matrices['Room 1 (Baseline RF)'] = confusion_matrix(test_1h[target_class], y_pred)
        else:
            test_preds_r2['rf23_pred_class'] = y_pred
            test_preds_r2['rf23_prob'] = y_prob
            roc_curves['Room 2 (Baseline RF)'] = (test_1h[target_class], y_prob)
            conf_matrices['Room 2 (Baseline RF)'] = confusion_matrix(test_1h[target_class], y_pred)
            
        # Regressor
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_reg.fit(train_1h[BASELINE_FEATURES], train_1h[target_power])
        y_pred_reg = rf_reg.predict(test_1h[BASELINE_FEATURES])
        mae, rmse, r2 = eval_reg(test_1h[target_power], y_pred_reg)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Baseline Random Forest (23 Feat)',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Regression',
            'accuracy': np.nan, 'balanced_accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan,
            'mae': mae, 'rmse': rmse, 'r2': r2
        })
        if room == 'Room 1':
            test_preds_r1['rf23_pred_power'] = y_pred_reg
        else:
            test_preds_r2['rf23_pred_power'] = y_pred_reg

    # --- C. ENHANCED RANDOM FOREST (137 Causal Features + Tuning) ---
    print("\n[*] Tuning & Training Enhanced Random Forest (137 Features, Experimental Variant)...")
    for room, target_class, target_power in [('Room 1', 'high_usage_r1_next', 'power1_next'),
                                             ('Room 2', 'high_usage_r2_next', 'power2_next')]:
        # Grid Search on Validation Split
        best_f1 = -1
        best_params_clf = None
        for n_est in [300, 500]:
            for m_depth in [15, 25]:
                for min_leaf in [1, 2]:
                    clf = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, min_samples_leaf=min_leaf,
                                                 max_features='sqrt', random_state=42, n_jobs=-1)
                    clf.fit(sub_train_1h[ENHANCED_FEATURES], sub_train_1h[target_class])
                    val_pred = clf.predict(val_1h[ENHANCED_FEATURES])
                    v_f1 = f1_score(val_1h[target_class], val_pred, zero_division=0)
                    if v_f1 > best_f1:
                        best_f1 = v_f1
                        best_params_clf = {'n_estimators': n_est, 'max_depth': m_depth, 'min_samples_leaf': min_leaf}
                        
        print(f"  Best RF Clf for {room}: {best_params_clf} (Val F1: {best_f1:.4f})")
        
        # Train on full train set with best params
        rf_enh_clf = RandomForestClassifier(**best_params_clf, max_features='sqrt', random_state=42, n_jobs=-1)
        rf_enh_clf.fit(train_1h[ENHANCED_FEATURES], train_1h[target_class])
        
        y_pred = rf_enh_clf.predict(test_1h[ENHANCED_FEATURES])
        y_prob = rf_enh_clf.predict_proba(test_1h[ENHANCED_FEATURES])[:, 1]
        acc, bal_acc, prec, rec, f1, auc = eval_clf(test_1h[target_class], y_pred, y_prob)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Enhanced Random Forest (Tuned)',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
        
        if room == 'Room 1':
            test_preds_r1['rf_enh_pred_class'] = y_pred
            test_preds_r1['rf_enh_prob'] = y_prob
            roc_curves['Room 1 (Enhanced RF)'] = (test_1h[target_class], y_prob)
            conf_matrices['Room 1 (Enhanced RF)'] = confusion_matrix(test_1h[target_class], y_pred)
            feature_importances['Room 1 RF'] = pd.Series(rf_enh_clf.feature_importances_, index=ENHANCED_FEATURES)
        else:
            test_preds_r2['rf_enh_pred_class'] = y_pred
            test_preds_r2['rf_enh_prob'] = y_prob
            roc_curves['Room 2 (Enhanced RF)'] = (test_1h[target_class], y_prob)
            conf_matrices['Room 2 (Enhanced RF)'] = confusion_matrix(test_1h[target_class], y_pred)
            feature_importances['Room 2 RF'] = pd.Series(rf_enh_clf.feature_importances_, index=ENHANCED_FEATURES)

        # Regressor
        rf_enh_reg = RandomForestRegressor(n_estimators=300, max_depth=15, max_features='sqrt', random_state=42, n_jobs=-1)
        rf_enh_reg.fit(train_1h[ENHANCED_FEATURES], train_1h[target_power])
        y_pred_reg = rf_enh_reg.predict(test_1h[ENHANCED_FEATURES])
        mae, rmse, r2 = eval_reg(test_1h[target_power], y_pred_reg)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Enhanced Random Forest (Tuned)',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Regression',
            'accuracy': np.nan, 'balanced_accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan,
            'mae': mae, 'rmse': rmse, 'r2': r2
        })
        if room == 'Room 1':
            test_preds_r1['rf_enh_pred_power'] = y_pred_reg
        else:
            test_preds_r2['rf_enh_pred_power'] = y_pred_reg

    # --- D. XGBOOST COMPARISON (137 Causal Features) ---
    print("\n[*] Tuning & Training XGBoost (Comparative Model)...")
    for room, target_class, target_power in [('Room 1', 'high_usage_r1_next', 'power1_next'),
                                             ('Room 2', 'high_usage_r2_next', 'power2_next')]:
        best_f1 = -1
        best_params_xgb = None
        for n_est in [200, 300]:
            for m_depth in [3, 5]:
                for lr in [0.03, 0.08]:
                    clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=m_depth, learning_rate=lr,
                                            subsample=0.8, random_state=42, n_jobs=-1, eval_metric='logloss')
                    clf.fit(sub_train_1h[ENHANCED_FEATURES], sub_train_1h[target_class])
                    val_pred = clf.predict(val_1h[ENHANCED_FEATURES])
                    v_f1 = f1_score(val_1h[target_class], val_pred, zero_division=0)
                    if v_f1 > best_f1:
                        best_f1 = v_f1
                        best_params_xgb = {'n_estimators': n_est, 'max_depth': m_depth, 'learning_rate': lr, 'subsample': 0.8}
                        
        print(f"  Best XGB Clf for {room}: {best_params_xgb} (Val F1: {best_f1:.4f})")
        
        xgb_clf = xgb.XGBClassifier(**best_params_xgb, random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb_clf.fit(train_1h[ENHANCED_FEATURES], train_1h[target_class])
        
        y_pred = xgb_clf.predict(test_1h[ENHANCED_FEATURES])
        y_prob = xgb_clf.predict_proba(test_1h[ENHANCED_FEATURES])[:, 1]
        acc, bal_acc, prec, rec, f1, auc = eval_clf(test_1h[target_class], y_pred, y_prob)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'XGBoost (Enhanced Feat)',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
        
        if room == 'Room 1':
            test_preds_r1['xgb_pred_class'] = y_pred
            test_preds_r1['xgb_prob'] = y_prob
            roc_curves['Room 1 (XGBoost)'] = (test_1h[target_class], y_prob)
            conf_matrices['Room 1 (XGBoost)'] = confusion_matrix(test_1h[target_class], y_pred)
            feature_importances['Room 1 XGB'] = pd.Series(xgb_clf.feature_importances_, index=ENHANCED_FEATURES)
        else:
            test_preds_r2['xgb_pred_class'] = y_pred
            test_preds_r2['xgb_prob'] = y_prob
            roc_curves['Room 2 (XGBoost)'] = (test_1h[target_class], y_prob)
            conf_matrices['Room 2 (XGBoost)'] = confusion_matrix(test_1h[target_class], y_pred)
            feature_importances['Room 2 XGB'] = pd.Series(xgb_clf.feature_importances_, index=ENHANCED_FEATURES)

        # Regressor
        xgb_reg = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1)
        xgb_reg.fit(train_1h[ENHANCED_FEATURES], train_1h[target_power])
        y_pred_reg = xgb_reg.predict(test_1h[ENHANCED_FEATURES])
        mae, rmse, r2 = eval_reg(test_1h[target_power], y_pred_reg)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'XGBoost (Enhanced Feat)',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Regression',
            'accuracy': np.nan, 'balanced_accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan,
            'mae': mae, 'rmse': rmse, 'r2': r2
        })
        if room == 'Room 1':
            test_preds_r1['xgb_pred_power'] = y_pred_reg
        else:
            test_preds_r2['xgb_pred_power'] = y_pred_reg

        # --- E. REGRESSION-DERIVED CLASSIFICATION ---
        thresh = THRESHOLD_R1 if room == 'Room 1' else THRESHOLD_R2
        derived_class = (y_pred_reg >= thresh).astype(int)
        acc_d, bal_acc_d, prec_d, rec_d, f1_d, _ = eval_clf(test_1h[target_class], derived_class)
        
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Regression-Derived (XGB Reg >= Thresh)',
            'room': room,
            'horizon': '1-Hour',
            'task': 'Classification',
            'accuracy': acc_d, 'balanced_accuracy': bal_acc_d, 'precision': prec_d, 'recall': rec_d, 'f1': f1_d, 'roc_auc': np.nan,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })

    # --- 6. HORIZON 24-HOUR BENCHMARKING (70% Train / 15% Val / 15% Test) ---
    print("\n" + "=" * 80)
    print(" 4. 24-HOUR (TOMORROW) BENCHMARKING (70% Train / 15% Val / 15% Test)")
    print("=" * 80)
    
    df_24h = df.iloc[:-8640].copy().reset_index(drop=True)
    N_24h = len(df_24h)
    train_end_24h = int(N_24h * 0.70)
    val_end_24h = int(N_24h * 0.85)
    
    train_24h = df_24h.iloc[:train_end_24h]
    val_24h = df_24h.iloc[train_end_24h:val_end_24h]
    test_24h = df_24h.iloc[val_end_24h:]
    
    print(f"[*] 24-Hour Total Rows: {N_24h}")
    print(f"[*] Train Rows        : {len(train_24h)}")
    print(f"[*] Val Rows          : {len(val_24h)}")
    print(f"[*] Test Rows         : {len(test_24h)}")
    
    for room, col, target_class in [('Room 1', 'power1', 'high_usage_r1_tomorrow'),
                                    ('Room 2', 'power2', 'high_usage_r2_tomorrow')]:
        thresh = THRESHOLD_R1 if room == 'Room 1' else THRESHOLD_R2
        
        # Persistence
        pers_pred = (test_24h[col] >= thresh).astype(int)
        acc, bal_acc, prec, rec, f1, _ = eval_clf(test_24h[target_class], pers_pred)
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Persistence',
            'room': room,
            'horizon': '24-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': np.nan,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
        
        # Baseline RF (23 Feat)
        rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_base.fit(train_24h[BASELINE_FEATURES], train_24h[target_class])
        y_pred = rf_base.predict(test_24h[BASELINE_FEATURES])
        y_prob = rf_base.predict_proba(test_24h[BASELINE_FEATURES])[:, 1]
        acc, bal_acc, prec, rec, f1, auc = eval_clf(test_24h[target_class], y_pred, y_prob)
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Baseline Random Forest (23 Feat)',
            'room': room,
            'horizon': '24-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
        
        # Enhanced RF (Tuned)
        rf_enh = RandomForestClassifier(n_estimators=300, max_depth=15, max_features='sqrt', random_state=42, n_jobs=-1)
        rf_enh.fit(train_24h[ENHANCED_FEATURES], train_24h[target_class])
        y_pred = rf_enh.predict(test_24h[ENHANCED_FEATURES])
        y_prob = rf_enh.predict_proba(test_24h[ENHANCED_FEATURES])[:, 1]
        acc, bal_acc, prec, rec, f1, auc = eval_clf(test_24h[target_class], y_pred, y_prob)
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'Enhanced Random Forest (Tuned)',
            'room': room,
            'horizon': '24-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
        
        # XGBoost
        xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
                                  random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb_m.fit(train_24h[ENHANCED_FEATURES], train_24h[target_class])
        y_pred = xgb_m.predict(test_24h[ENHANCED_FEATURES])
        y_prob = xgb_m.predict_proba(test_24h[ENHANCED_FEATURES])[:, 1]
        acc, bal_acc, prec, rec, f1, auc = eval_clf(test_24h[target_class], y_pred, y_prob)
        results.append({
            'dataset': 'New Realistic Synthetic',
            'model': 'XGBoost (Enhanced Feat)',
            'room': room,
            'horizon': '24-Hour',
            'task': 'Classification',
            'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })

    # --- 7. ADD PREVIOUS OCTOBER DATASET RESULTS FOR EXPLICIT BENCHMARKING ---
    results.append({
        'dataset': 'Previous October Calibrated',
        'model': 'Baseline Random Forest (23 Feat)',
        'room': 'Room 1',
        'horizon': '1-Hour',
        'task': 'Classification',
        'accuracy': 0.672128, 'balanced_accuracy': 0.673973, 'precision': 0.637109, 'recall': 0.705677, 'f1': 0.669642, 'roc_auc': 0.742794,
        'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
    })
    results.append({
        'dataset': 'Previous October Calibrated',
        'model': 'Baseline Random Forest (23 Feat)',
        'room': 'Room 1',
        'horizon': '1-Hour',
        'task': 'Regression',
        'accuracy': np.nan, 'balanced_accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan,
        'mae': 30.254436, 'rmse': 37.776232, 'r2': 0.100767
    })
    results.append({
        'dataset': 'Previous October Calibrated',
        'model': 'Baseline Random Forest (23 Feat)',
        'room': 'Room 2',
        'horizon': '1-Hour',
        'task': 'Classification',
        'accuracy': 0.723891, 'balanced_accuracy': 0.692594, 'precision': 0.679924, 'recall': 0.550730, 'f1': 0.608545, 'roc_auc': 0.788106,
        'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
    })
    results.append({
        'dataset': 'Previous October Calibrated',
        'model': 'Baseline Random Forest (23 Feat)',
        'room': 'Room 2',
        'horizon': '1-Hour',
        'task': 'Regression',
        'accuracy': np.nan, 'balanced_accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan,
        'mae': 7.372898, 'rmse': 9.601579, 'r2': 0.130239
    })
    results.append({
        'dataset': 'Previous October Calibrated',
        'model': 'Baseline Random Forest (23 Feat)',
        'room': 'Room 1',
        'horizon': '24-Hour',
        'task': 'Classification',
        'accuracy': 0.641710, 'balanced_accuracy': 0.646042, 'precision': 0.600472, 'recall': 0.729128, 'f1': 0.658576, 'roc_auc': 0.717210,
        'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
    })
    results.append({
        'dataset': 'Previous October Calibrated',
        'model': 'Baseline Random Forest (23 Feat)',
        'room': 'Room 2',
        'horizon': '24-Hour',
        'task': 'Classification',
        'accuracy': 0.707602, 'balanced_accuracy': 0.670353, 'precision': 0.709248, 'recall': 0.473034, 'f1': 0.567544, 'roc_auc': 0.756011,
        'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
    })

    # Save Comparison DataFrame
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[+] Results saved to {results_csv_path}")
    
    # Save Raw Predictions
    df_pred_r1 = pd.DataFrame({
        'timestamp': test_preds_r1['timestamp'],
        'actual_power': test_preds_r1['actual_power'],
        'predicted_power': test_preds_r1['rf_enh_pred_power'],
        'actual_class': test_preds_r1['actual_class'],
        'predicted_class': test_preds_r1['rf_enh_pred_class'],
        'prediction_probability': test_preds_r1['rf_enh_prob']
    })
    df_pred_r1.to_csv(os.path.join(RESULTS_DIR, "room1_test_predictions.csv"), index=False)
    
    df_pred_r2 = pd.DataFrame({
        'timestamp': test_preds_r2['timestamp'],
        'actual_power': test_preds_r2['actual_power'],
        'predicted_power': test_preds_r2['rf_enh_pred_power'],
        'actual_class': test_preds_r2['actual_class'],
        'predicted_class': test_preds_r2['rf_enh_pred_class'],
        'prediction_probability': test_preds_r2['rf_enh_prob']
    })
    df_pred_r2.to_csv(os.path.join(RESULTS_DIR, "room2_test_predictions.csv"), index=False)
    print(f"[+] Raw test predictions exported to {RESULTS_DIR}")

    # --- 8. GENERATE 14 PUBLICATION-GRADE VISUALIZATIONS ---
    print("\n" + "=" * 80)
    print(" 5. GENERATING 14 PUBLICATION-GRADE VISUALIZATIONS")
    print("=" * 80)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.size'] = 11

    # 1. Old vs New Dataset Accuracy
    plt.figure(figsize=(10, 6))
    comp_acc = results_df[(results_df['task'] == 'Classification') & 
                          (results_df['model'] == 'Baseline Random Forest (23 Feat)')].copy()
    comp_acc['Label'] = comp_acc['room'] + " (" + comp_acc['horizon'] + ")"
    
    sns.barplot(data=comp_acc, x='Label', y='accuracy', hue='dataset', palette=['#3498db', '#2ecc71'])
    plt.title('Baseline Random Forest: Previous Calibrated vs New Realistic Dataset Accuracy', fontsize=13, weight='bold', pad=12)
    plt.ylabel('Classification Accuracy')
    plt.ylim(0.5, 0.85)
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{height*100:.2f}%', (p.get_x() + p.get_width() / 2., height),
                               ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'old_vs_new_accuracy.png'), dpi=300)
    plt.close()

    # 2. RF vs Persistence Accuracy
    plt.figure(figsize=(11, 6))
    rf_pers = results_df[(results_df['dataset'] == 'New Realistic Synthetic') & 
                         (results_df['task'] == 'Classification') & 
                         (results_df['model'].isin(['Persistence', 'Baseline Random Forest (23 Feat)', 'Enhanced Random Forest (Tuned)']))].copy()
    rf_pers['Label'] = rf_pers['room'] + " (" + rf_pers['horizon'] + ")"
    sns.barplot(data=rf_pers, x='Label', y='accuracy', hue='model', palette=['#95a5a6', '#3498db', '#e67e22'])
    plt.title('Realistic Dataset: Random Forest Variants vs Persistence Baseline (Accuracy)', fontsize=13, weight='bold', pad=12)
    plt.ylabel('Classification Accuracy')
    plt.ylim(0.5, 0.85)
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{height*100:.1f}%', (p.get_x() + p.get_width() / 2., height),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'rf_vs_persistence_accuracy.png'), dpi=300)
    plt.close()

    # 3. F1 Comparison
    plt.figure(figsize=(11, 6))
    f1_df = results_df[(results_df['dataset'] == 'New Realistic Synthetic') & 
                       (results_df['task'] == 'Classification') & 
                       (results_df['model'].isin(['Persistence', 'Baseline Random Forest (23 Feat)', 'Enhanced Random Forest (Tuned)', 'XGBoost (Enhanced Feat)']))].copy()
    f1_df['Label'] = f1_df['room'] + " (" + f1_df['horizon'] + ")"
    sns.barplot(data=f1_df, x='Label', y='f1', hue='model', palette=['#95a5a6', '#3498db', '#e67e22', '#2ecc71'])
    plt.title('Realistic Dataset: Multi-Model F1-Score Comparison Across Horizons', fontsize=13, weight='bold', pad=12)
    plt.ylabel('F1-Score')
    plt.ylim(0.3, 0.85)
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'f1_comparison.png'), dpi=300)
    plt.close()

    # 4. ROC Curves
    plt.figure(figsize=(9, 7))
    for label, (y_t, y_p) in roc_curves.items():
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) Curves (1-Hour Horizon)', fontsize=13, weight='bold', pad=12)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'roc_curves.png'), dpi=300)
    plt.close()

    # 5. Room 1 Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm1 = conf_matrices['Room 1 (Baseline RF)']
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal (<40W)', 'High (>=40W)'],
                yticklabels=['Normal (<40W)', 'High (>=40W)'])
    plt.title('Room 1 Confusion Matrix (1-Hour Horizon, Baseline RF)', fontsize=12, weight='bold')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'room1_confusion_matrix.png'), dpi=300)
    plt.close()

    # 6. Room 2 Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm2 = conf_matrices['Room 2 (Baseline RF)']
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Normal (<10W)', 'High (>=10W)'],
                yticklabels=['Normal (<10W)', 'High (>=10W)'])
    plt.title('Room 2 Confusion Matrix (1-Hour Horizon, Baseline RF)', fontsize=12, weight='bold')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'room2_confusion_matrix.png'), dpi=300)
    plt.close()

    # 7. Room 1 Actual vs Predicted Power (1-Hour)
    plt.figure(figsize=(14, 5))
    sample_window = slice(0, 1200) # ~3.3 hours of 10-second data
    plt.plot(test_preds_r1['timestamp'].iloc[sample_window], test_preds_r1['actual_power'].iloc[sample_window],
             label='Actual Power (t+1h)', color='#2c3e50', lw=2)
    plt.plot(test_preds_r1['timestamp'].iloc[sample_window], test_preds_r1['rf_enh_pred_power'][sample_window],
             label='Enhanced RF Forecast', color='#e74c3c', lw=1.5, linestyle='--')
    plt.axhline(THRESHOLD_R1, color='orange', linestyle=':', label='High Usage Threshold (40W)')
    plt.title('Room 1: Actual vs Predicted Power Trace (1-Hour Ahead Horizon)', fontsize=13, weight='bold', pad=12)
    plt.ylabel('Power (Watts)')
    plt.xlabel('Timestamp')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'room1_actual_vs_predicted_power.png'), dpi=300)
    plt.close()

    # 8. Room 2 Actual vs Predicted Power (1-Hour)
    plt.figure(figsize=(14, 5))
    plt.plot(test_preds_r2['timestamp'].iloc[sample_window], test_preds_r2['actual_power'].iloc[sample_window],
             label='Actual Power (t+1h)', color='#2c3e50', lw=2)
    plt.plot(test_preds_r2['timestamp'].iloc[sample_window], test_preds_r2['rf_enh_pred_power'][sample_window],
             label='Enhanced RF Forecast', color='#27ae60', lw=1.5, linestyle='--')
    plt.axhline(THRESHOLD_R2, color='orange', linestyle=':', label='High Usage Threshold (10W)')
    plt.title('Room 2: Actual vs Predicted Power Trace (1-Hour Ahead Horizon)', fontsize=13, weight='bold', pad=12)
    plt.ylabel('Power (Watts)')
    plt.xlabel('Timestamp')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'room2_actual_vs_predicted_power.png'), dpi=300)
    plt.close()

    # 9. MAE Comparison
    plt.figure(figsize=(9, 5))
    mae_df = results_df[(results_df['dataset'] == 'New Realistic Synthetic') & (results_df['task'] == 'Regression')].copy()
    sns.barplot(data=mae_df, x='room', y='mae', hue='model', palette=['#95a5a6', '#3498db', '#e67e22', '#2ecc71'])
    plt.title('Power Forecast Mean Absolute Error (MAE) — Lower is Better', fontsize=13, weight='bold', pad=12)
    plt.ylabel('MAE (Watts)')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'mae_comparison.png'), dpi=300)
    plt.close()

    # 10. RMSE Comparison
    plt.figure(figsize=(9, 5))
    rmse_df = results_df[(results_df['dataset'] == 'New Realistic Synthetic') & (results_df['task'] == 'Regression')].copy()
    sns.barplot(data=rmse_df, x='room', y='rmse', hue='model', palette=['#95a5a6', '#3498db', '#e67e22', '#2ecc71'])
    plt.title('Power Forecast Root Mean Squared Error (RMSE) — Lower is Better', fontsize=13, weight='bold', pad=12)
    plt.ylabel('RMSE (Watts)')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'rmse_comparison.png'), dpi=300)
    plt.close()

    # 11. R2 Comparison
    plt.figure(figsize=(9, 5))
    r2_df = results_df[(results_df['dataset'] == 'New Realistic Synthetic') & (results_df['task'] == 'Regression')].copy()
    sns.barplot(data=r2_df, x='room', y='r2', hue='model', palette=['#95a5a6', '#3498db', '#e67e22', '#2ecc71'])
    plt.axhline(0, color='black', lw=1, linestyle='--')
    plt.title('Power Forecast Coefficient of Determination (R²) — Higher is Better', fontsize=13, weight='bold', pad=12)
    plt.ylabel('R² Score')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'r2_comparison.png'), dpi=300)
    plt.close()

    # 12. Feature Importance
    plt.figure(figsize=(10, 8))
    top_r1_feat = feature_importances['Room 1 RF'].sort_values(ascending=False).head(15)
    sns.barplot(x=top_r1_feat.values, y=top_r1_feat.index, palette='Blues_r')
    plt.title('Top 15 Most Important Features for Room 1 1-Hour Horizon (Enhanced RF)', fontsize=13, weight='bold', pad=12)
    plt.xlabel('Gini Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()

    # 13. Daily Power Profile
    plt.figure(figsize=(12, 5))
    df['hour_float'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    daily_profile = df.groupby(df['timestamp'].dt.hour)[['power1', 'power2', 'totalPower']].mean()
    plt.plot(daily_profile.index, daily_profile['power1'], label='Room 1 (Avg Power)', color='#e74c3c', marker='o', lw=2)
    plt.plot(daily_profile.index, daily_profile['power2'], label='Room 2 (Avg Power)', color='#27ae60', marker='s', lw=2)
    plt.plot(daily_profile.index, daily_profile['totalPower'], label='Total Load (Avg Power)', color='#2c3e50', marker='^', lw=2, linestyle='--')
    plt.title('24-Hour Diurnal Power Profile (Realistic Dataset Average)', fontsize=13, weight='bold', pad=12)
    plt.xlabel('Hour of Day (0 - 23)')
    plt.ylabel('Average Power (Watts)')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'daily_power_profile.png'), dpi=300)
    plt.close()

    # 14. Room 1 vs Room 2 Power Profile Correlation
    plt.figure(figsize=(8, 6))
    sample_corr = df[['power1', 'power2']].iloc[::60] # Sample every 10 min
    sns.scatterplot(data=sample_corr, x='power1', y='power2', alpha=0.3, color='#8e44ad')
    plt.axvline(THRESHOLD_R1, color='orange', linestyle=':', label='Room 1 Threshold (40W)')
    plt.axhline(THRESHOLD_R2, color='green', linestyle=':', label='Room 2 Threshold (10W)')
    plt.title('Room 1 vs Room 2 Simultaneous Power Distribution', fontsize=13, weight='bold', pad=12)
    plt.xlabel('Room 1 Power (Watts)')
    plt.ylabel('Room 2 Power (Watts)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'room1_vs_room2_power_profile.png'), dpi=300)
    plt.close()

    print(f"[+] All 14 publication-grade figures successfully generated in {GRAPHS_DIR}!")

if __name__ == '__main__':
    run_experiment()
