import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve, mean_absolute_error,
    root_mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

# Set paths
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXP_DIR)
DATA_PATH = os.path.join(BASE_DIR, "VOLTIX_OCTOBER_2026_calibrated_dataset.csv")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLD_R1 = 40.0
THRESHOLD_R2 = 10.0

BASELINE_FEATURES = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

def calculate_state_durations(series, threshold):
    is_on = (series >= threshold).astype(int).values
    n = len(is_on)
    on_dur = np.zeros(n, dtype=np.float32)
    off_dur = np.zeros(n, dtype=np.float32)
    trans_dur = np.zeros(n, dtype=np.float32)

    curr_on = 0
    curr_off = 0
    curr_trans = 0

    for i in range(n):
        if i == 0:
            if is_on[i] == 1:
                curr_on, curr_off = 1, 0
            else:
                curr_on, curr_off = 0, 1
            curr_trans = 1
        else:
            if is_on[i] == is_on[i-1]:
                curr_trans += 1
                if is_on[i] == 1:
                    curr_on += 1
                    curr_off = 0
                else:
                    curr_off += 1
                    curr_on = 0
            else:
                curr_trans = 1
                if is_on[i] == 1:
                    curr_on, curr_off = 1, 0
                else:
                    curr_on, curr_off = 0, 1
        on_dur[i] = curr_on * (10.0 / 60.0)
        off_dur[i] = curr_off * (10.0 / 60.0)
        trans_dur[i] = curr_trans * (10.0 / 60.0)
    return on_dur, off_dur, trans_dur

def build_enhanced_dataset(csv_path=DATA_PATH):
    print("=" * 80)
    print(" 1. DATASET INGESTION & CAUSAL ENHANCED FEATURE ENGINEERING")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    num_cols = ['current1', 'current2', 'energy1', 'energy2', 'power1', 'power2', 
                'totalCurrent', 'totalEnergy', 'totalPower', 'voltage1', 'voltage2']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 1. Cyclic Temporal Features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # 2. Baseline Lags & Rolling (5 samples)
    for col in ['power1', 'power2', 'totalPower', 'current1', 'current2']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_roll_mean5'] = df[col].rolling(window=5, min_periods=5).mean()
        df[f'{col}_roll_max5'] = df[col].rolling(window=5, min_periods=5).max()

    # 3. Enhanced Extended Lags: 6 (1m), 30 (5m), 60 (10m), 90 (15m), 180 (30m), 360 (1h)
    lag_steps = [6, 30, 60, 90, 180, 360]
    for col in ['power1', 'power2', 'totalPower']:
        for step in lag_steps:
            df[f'{col}_lag{step}'] = df[col].shift(step)

    # 4. Enhanced Rolling Statistics: mean, min, max, std
    rolling_windows = {
        '6': 6,     # 1 min
        '30': 30,   # 5 min
        '90': 90,   # 15 min
        '180': 180, # 30 min
        '360': 360  # 1 hour
    }
    for col in ['power1', 'power2', 'totalPower']:
        for w_name, w_size in rolling_windows.items():
            roll = df[col].rolling(window=w_size, min_periods=w_size)
            df[f'{col}_roll_mean{w_name}'] = roll.mean()
            df[f'{col}_roll_min{w_name}'] = roll.min()
            df[f'{col}_roll_max{w_name}'] = roll.max()
            df[f'{col}_roll_std{w_name}'] = roll.std().fillna(0.0)

    # 5. Temporal Trend / Change Features
    for col in ['power1', 'power2', 'totalPower']:
        df[f'{col}_diff_1m'] = df[col] - df[f'{col}_lag6']
        df[f'{col}_diff_5m'] = df[col] - df[f'{col}_lag30']
        df[f'{col}_diff_15m'] = df[col] - df[f'{col}_lag90']
        df[f'{col}_diff_1h'] = df[col] - df[f'{col}_lag360']

    # 6. Load-State Causal Durations
    r1_on, r1_off, r1_trans = calculate_state_durations(df['power1'], THRESHOLD_R1)
    df['r1_is_on'] = (df['power1'] >= THRESHOLD_R1).astype(int)
    df['r1_consec_on_min'] = r1_on
    df['r1_consec_off_min'] = r1_off
    df['r1_min_since_trans'] = r1_trans

    r2_on, r2_off, r2_trans = calculate_state_durations(df['power2'], THRESHOLD_R2)
    df['r2_is_on'] = (df['power2'] >= THRESHOLD_R2).astype(int)
    df['r2_consec_on_min'] = r2_on
    df['r2_consec_off_min'] = r2_off
    df['r2_min_since_trans'] = r2_trans

    df['date'] = df['timestamp'].dt.date
    df['r1_daily_cum_on_min'] = df.groupby('date')['power1'].transform(lambda s: ((s >= THRESHOLD_R1).astype(float) * (10.0/60.0)).cumsum())
    df['r2_daily_cum_on_min'] = df.groupby('date')['power2'].transform(lambda s: ((s >= THRESHOLD_R2).astype(float) * (10.0/60.0)).cumsum())

    # Drop initial 360 rows (1 hour history) to eliminate uninitialized lag NaNs completely
    df_causal = df.dropna(subset=['power1_lag360', 'power1_roll_mean360']).reset_index(drop=True)

    # 7. Targets (1-Hour and 24-Hour)
    df_q1h = pd.DataFrame({
        'orig_idx': df_causal.index,
        'timestamp': df_causal['timestamp'],
        'query_1h': df_causal['timestamp'] + pd.Timedelta(hours=1)
    })
    m1h = pd.merge_asof(
        df_q1h.sort_values('query_1h'),
        df_causal[['timestamp', 'power1', 'power2']].rename(
            columns={'timestamp': 'match_time_1h', 'power1': 'p1_1h', 'power2': 'p2_1h'}
        ),
        left_on='query_1h', right_on='match_time_1h', direction='nearest', tolerance=pd.Timedelta(seconds=15)
    ).sort_values('orig_idx').reset_index(drop=True)

    df_causal['match_time_1h'] = m1h['match_time_1h']
    df_causal['power1_1h_actual'] = m1h['p1_1h']
    df_causal['power2_1h_actual'] = m1h['p2_1h']
    df_causal['high_usage_r1_1h'] = (df_causal['power1_1h_actual'] >= THRESHOLD_R1).astype(float)
    df_causal['high_usage_r2_1h'] = (df_causal['power2_1h_actual'] >= THRESHOLD_R2).astype(float)

    df_q24h = pd.DataFrame({
        'orig_idx': df_causal.index,
        'timestamp': df_causal['timestamp'],
        'query_24h': df_causal['timestamp'] + pd.Timedelta(hours=24)
    })
    m24h = pd.merge_asof(
        df_q24h.sort_values('query_24h'),
        df_causal[['timestamp', 'power1', 'power2']].rename(
            columns={'timestamp': 'match_time_24h', 'power1': 'p1_24h', 'power2': 'p2_24h'}
        ),
        left_on='query_24h', right_on='match_time_24h', direction='nearest', tolerance=pd.Timedelta(seconds=15)
    ).sort_values('orig_idx').reset_index(drop=True)

    df_causal['match_time_24h'] = m24h['match_time_24h']
    df_causal['power1_24h_actual'] = m24h['p1_24h']
    df_causal['power2_24h_actual'] = m24h['p2_24h']
    df_causal['high_usage_r1_tomorrow'] = (df_causal['power1_24h_actual'] >= THRESHOLD_R1).astype(float)
    df_causal['high_usage_r2_tomorrow'] = (df_causal['power2_24h_actual'] >= THRESHOLD_R2).astype(float)

    print(f"Total Causal Records with 1h History: {len(df_causal)}")
    return df_causal

def get_enhanced_feature_list(df):
    excluded = ['reading_id', 'timestamp', 'date', 'match_time_1h', 'match_time_24h',
                'power1_1h_actual', 'power2_1h_actual', 'high_usage_r1_1h', 'high_usage_r2_1h',
                'power1_24h_actual', 'power2_24h_actual', 'high_usage_r1_tomorrow', 'high_usage_r2_tomorrow']
    features = [col for col in df.columns if col not in excluded]
    return features

def run_all_experiments():
    df = build_enhanced_dataset()
    enhanced_features = get_enhanced_feature_list(df)
    print(f"\nBaseline Feature Count : {len(BASELINE_FEATURES)}")
    print(f"Enhanced Feature Count : {len(enhanced_features)}")

    results_table = []
    plot_data = {}

    # =========================================================================
    # SECTION A: 1-HOUR HORIZON EXPERIMENTS (80% Train / 20% Untouched Test)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 2. 1-HOUR HORIZON MODEL BENCHMARKING (80% TRAIN / 20% FINAL TEST)")
    print("=" * 80)
    df_1h = df.dropna(subset=['power1_1h_actual', 'power2_1h_actual']).reset_index(drop=True)
    split_1h = int(len(df_1h) * 0.8)
    train_1h = df_1h.iloc[:split_1h]
    test_1h = df_1h.iloc[split_1h:]

    # Sub-split for tuning (60% sub-train / 20% val of total 1h records)
    sub_split = int(len(train_1h) * 0.75)
    sub_tr_1h = train_1h.iloc[:sub_split]
    val_1h = train_1h.iloc[sub_split:]

    for room_num, target_clf, target_reg, curr_col, thresh in [
        (1, 'high_usage_r1_1h', 'power1_1h_actual', 'power1', THRESHOLD_R1),
        (2, 'high_usage_r2_1h', 'power2_1h_actual', 'power2', THRESHOLD_R2)
    ]:
        room_name = f"Room {room_num}"
        y_test_clf = test_1h[target_clf].astype(int)
        y_test_reg = test_1h[target_reg]

        # -------------------------------------------------------------
        # 1. Persistence Baseline
        # -------------------------------------------------------------
        y_persist_clf = (test_1h[curr_col] >= thresh).astype(int)
        y_persist_reg = test_1h[curr_col]

        p_acc = accuracy_score(y_test_clf, y_persist_clf)
        p_bal = balanced_accuracy_score(y_test_clf, y_persist_clf)
        p_prec = precision_score(y_test_clf, y_persist_clf, zero_division=0)
        p_rec = recall_score(y_test_clf, y_persist_clf, zero_division=0)
        p_f1 = f1_score(y_test_clf, y_persist_clf, zero_division=0)
        p_mae = mean_absolute_error(y_test_reg, y_persist_reg)
        p_rmse = root_mean_squared_error(y_test_reg, y_persist_reg)
        p_r2 = r2_score(y_test_reg, y_persist_reg)

        results_table.append({
            'Model': 'Persistence', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Classification',
            'Accuracy': p_acc, 'Balanced_Accuracy': p_bal, 'Precision': p_prec, 'Recall': p_rec,
            'F1': p_f1, 'ROC_AUC': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })
        results_table.append({
            'Model': 'Persistence', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Regression',
            'Accuracy': np.nan, 'Balanced_Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan,
            'F1': np.nan, 'ROC_AUC': np.nan, 'MAE': p_mae, 'RMSE': p_rmse, 'R2': p_r2
        })

        # -------------------------------------------------------------
        # 2. Baseline Random Forest (23 Features, n=100, max_depth=10)
        # -------------------------------------------------------------
        rf_base_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_base_clf.fit(train_1h[BASELINE_FEATURES], train_1h[target_clf].astype(int))
        pred_rf_base_clf = rf_base_clf.predict(test_1h[BASELINE_FEATURES])
        prob_rf_base_clf = rf_base_clf.predict_proba(test_1h[BASELINE_FEATURES])[:, 1]

        rf_base_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_base_reg.fit(train_1h[BASELINE_FEATURES], train_1h[target_reg])
        pred_rf_base_reg = rf_base_reg.predict(test_1h[BASELINE_FEATURES])

        results_table.append({
            'Model': 'Baseline Random Forest (23 Feat)', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, pred_rf_base_clf),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, pred_rf_base_clf),
            'Precision': precision_score(y_test_clf, pred_rf_base_clf, zero_division=0),
            'Recall': recall_score(y_test_clf, pred_rf_base_clf, zero_division=0),
            'F1': f1_score(y_test_clf, pred_rf_base_clf, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test_clf, prob_rf_base_clf),
            'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })
        results_table.append({
            'Model': 'Baseline Random Forest (23 Feat)', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Regression',
            'Accuracy': np.nan, 'Balanced_Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan,
            'F1': np.nan, 'ROC_AUC': np.nan,
            'MAE': mean_absolute_error(y_test_reg, pred_rf_base_reg),
            'RMSE': root_mean_squared_error(y_test_reg, pred_rf_base_reg),
            'R2': r2_score(y_test_reg, pred_rf_base_reg)
        })

        # -------------------------------------------------------------
        # 3. Enhanced Random Forest (Enhanced Features + Hyperparameter Tuning)
        # Tuning on sub_tr_1h / val_1h only
        # -------------------------------------------------------------
        print(f"\n[*] Tuning Enhanced Random Forest for {room_name} 1-Hour...")
        best_rf_cfg = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}
        best_val_f1 = 0
        for n_est in [300, 500]:
            for md in [15, 25]:
                clf_cand = RandomForestClassifier(n_estimators=n_est, max_depth=md, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
                clf_cand.fit(sub_tr_1h[enhanced_features], sub_tr_1h[target_clf].astype(int))
                v_f1 = f1_score(val_1h[target_clf].astype(int), clf_cand.predict(val_1h[enhanced_features]), zero_division=0)
                if v_f1 > best_val_f1:
                    best_val_f1 = v_f1
                    best_rf_cfg = {'n_estimators': n_est, 'max_depth': md, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}

        print(f"  Best RF Configuration on Val for {room_name}: {best_rf_cfg} (Val F1: {best_val_f1:.4f})")
        rf_enh_clf = RandomForestClassifier(**best_rf_cfg, random_state=42, n_jobs=-1)
        rf_enh_clf.fit(train_1h[enhanced_features], train_1h[target_clf].astype(int))
        pred_rf_enh_clf = rf_enh_clf.predict(test_1h[enhanced_features])
        prob_rf_enh_clf = rf_enh_clf.predict_proba(test_1h[enhanced_features])[:, 1]

        rf_enh_reg = RandomForestRegressor(**best_rf_cfg, random_state=42, n_jobs=-1)
        rf_enh_reg.fit(train_1h[enhanced_features], train_1h[target_reg])
        pred_rf_enh_reg = rf_enh_reg.predict(test_1h[enhanced_features])

        results_table.append({
            'Model': 'Enhanced Random Forest (Tuned)', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, pred_rf_enh_clf),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, pred_rf_enh_clf),
            'Precision': precision_score(y_test_clf, pred_rf_enh_clf, zero_division=0),
            'Recall': recall_score(y_test_clf, pred_rf_enh_clf, zero_division=0),
            'F1': f1_score(y_test_clf, pred_rf_enh_clf, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test_clf, prob_rf_enh_clf),
            'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })
        results_table.append({
            'Model': 'Enhanced Random Forest (Tuned)', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Regression',
            'Accuracy': np.nan, 'Balanced_Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan,
            'F1': np.nan, 'ROC_AUC': np.nan,
            'MAE': mean_absolute_error(y_test_reg, pred_rf_enh_reg),
            'RMSE': root_mean_squared_error(y_test_reg, pred_rf_enh_reg),
            'R2': r2_score(y_test_reg, pred_rf_enh_reg)
        })

        # -------------------------------------------------------------
        # 4. XGBoost (Enhanced Features + Validation Tuning)
        # -------------------------------------------------------------
        print(f"[*] Tuning XGBoost for {room_name} 1-Hour...")
        best_xgb_cfg = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8}
        best_val_xgb_f1 = 0
        for lr in [0.03, 0.08]:
            for md in [5, 8]:
                xgb_cand = XGBClassifier(n_estimators=300, max_depth=md, learning_rate=lr, subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)
                xgb_cand.fit(sub_tr_1h[enhanced_features], sub_tr_1h[target_clf].astype(int))
                v_f1 = f1_score(val_1h[target_clf].astype(int), xgb_cand.predict(val_1h[enhanced_features]), zero_division=0)
                if v_f1 > best_val_xgb_f1:
                    best_val_xgb_f1 = v_f1
                    best_xgb_cfg = {'n_estimators': 300, 'max_depth': md, 'learning_rate': lr, 'subsample': 0.8}

        print(f"  Best XGB Configuration on Val for {room_name}: {best_xgb_cfg} (Val F1: {best_val_xgb_f1:.4f})")
        xgb_clf = XGBClassifier(**best_xgb_cfg, eval_metric='logloss', random_state=42, n_jobs=-1)
        xgb_clf.fit(train_1h[enhanced_features], train_1h[target_clf].astype(int))
        pred_xgb_clf = xgb_clf.predict(test_1h[enhanced_features])
        prob_xgb_clf = xgb_clf.predict_proba(test_1h[enhanced_features])[:, 1]

        xgb_reg = XGBRegressor(**best_xgb_cfg, random_state=42, n_jobs=-1)
        xgb_reg.fit(train_1h[enhanced_features], train_1h[target_reg])
        pred_xgb_reg = xgb_reg.predict(test_1h[enhanced_features])

        results_table.append({
            'Model': 'XGBoost (Enhanced Feat)', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, pred_xgb_clf),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, pred_xgb_clf),
            'Precision': precision_score(y_test_clf, pred_xgb_clf, zero_division=0),
            'Recall': recall_score(y_test_clf, pred_xgb_clf, zero_division=0),
            'F1': f1_score(y_test_clf, pred_xgb_clf, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test_clf, prob_xgb_clf),
            'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })
        results_table.append({
            'Model': 'XGBoost (Enhanced Feat)', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Regression',
            'Accuracy': np.nan, 'Balanced_Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan,
            'F1': np.nan, 'ROC_AUC': np.nan,
            'MAE': mean_absolute_error(y_test_reg, pred_xgb_reg),
            'RMSE': root_mean_squared_error(y_test_reg, pred_xgb_reg),
            'R2': r2_score(y_test_reg, pred_xgb_reg)
        })

        # -------------------------------------------------------------
        # 5. Regression-Derived Classifier
        # -------------------------------------------------------------
        reg_derived_pred = (pred_xgb_reg >= thresh).astype(int)
        results_table.append({
            'Model': 'Regression-Derived (XGB Reg >= Thresh)', 'Room': room_name, 'Horizon': '1-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, reg_derived_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, reg_derived_pred),
            'Precision': precision_score(y_test_clf, reg_derived_pred, zero_division=0),
            'Recall': recall_score(y_test_clf, reg_derived_pred, zero_division=0),
            'F1': f1_score(y_test_clf, reg_derived_pred, zero_division=0),
            'ROC_AUC': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })

        # Save plot data
        plot_data[f'r{room_num}_1h'] = {
            'y_test_clf': y_test_clf,
            'y_test_reg': y_test_reg,
            'prob_rf_base': prob_rf_base_clf,
            'prob_rf_enh': prob_rf_enh_clf,
            'prob_xgb': prob_xgb_clf,
            'pred_rf_enh_clf': pred_rf_enh_clf,
            'pred_xgb_clf': pred_xgb_clf,
            'pred_xgb_reg': pred_xgb_reg,
            'rf_enh_fi': rf_enh_clf.feature_importances_,
            'xgb_fi': xgb_clf.feature_importances_,
            'features': enhanced_features
        }

    # =========================================================================
    # SECTION B: 24-HOUR HORIZON EXPERIMENTS (70% Train / 15% Val / 15% Test)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 3. 24-HOUR (TOMORROW) MODEL BENCHMARKING (70% Train / 15% Val / 15% Test)")
    print("=" * 80)
    df_24h = df.dropna(subset=['power1_24h_actual', 'power2_24h_actual']).reset_index(drop=True)
    n_tot = len(df_24h)
    n_tr = int(n_tot * 0.70)
    n_val = int(n_tot * 0.15)
    train_24h = df_24h.iloc[:n_tr]
    val_24h = df_24h.iloc[n_tr:n_tr+n_val]
    test_24h = df_24h.iloc[n_tr+n_val:]

    for room_num, target_clf, curr_col, thresh in [
        (1, 'high_usage_r1_tomorrow', 'power1', THRESHOLD_R1),
        (2, 'high_usage_r2_tomorrow', 'power2', THRESHOLD_R2)
    ]:
        room_name = f"Room {room_num}"
        y_test_clf = test_24h[target_clf].astype(int)

        # 1. Persistence Baseline
        y_persist_clf = (test_24h[curr_col] >= thresh).astype(int)
        results_table.append({
            'Model': 'Persistence', 'Room': room_name, 'Horizon': '24-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, y_persist_clf),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, y_persist_clf),
            'Precision': precision_score(y_test_clf, y_persist_clf, zero_division=0),
            'Recall': recall_score(y_test_clf, y_persist_clf, zero_division=0),
            'F1': f1_score(y_test_clf, y_persist_clf, zero_division=0),
            'ROC_AUC': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })

        # 2. Baseline Random Forest (23 Features, n=100, max_depth=10)
        rf_base_24h = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_base_24h.fit(train_24h[BASELINE_FEATURES], train_24h[target_clf].astype(int))
        pred_rf_base_24h = rf_base_24h.predict(test_24h[BASELINE_FEATURES])
        prob_rf_base_24h = rf_base_24h.predict_proba(test_24h[BASELINE_FEATURES])[:, 1]

        results_table.append({
            'Model': 'Baseline Random Forest (23 Feat)', 'Room': room_name, 'Horizon': '24-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, pred_rf_base_24h),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, pred_rf_base_24h),
            'Precision': precision_score(y_test_clf, pred_rf_base_24h, zero_division=0),
            'Recall': recall_score(y_test_clf, pred_rf_base_24h, zero_division=0),
            'F1': f1_score(y_test_clf, pred_rf_base_24h, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test_clf, prob_rf_base_24h),
            'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })

        # 3. Enhanced Random Forest (Enhanced Features + Validation Tuning)
        rf_enh_24h = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
        rf_enh_24h.fit(train_24h[enhanced_features], train_24h[target_clf].astype(int))
        pred_rf_enh_24h = rf_enh_24h.predict(test_24h[enhanced_features])
        prob_rf_enh_24h = rf_enh_24h.predict_proba(test_24h[enhanced_features])[:, 1]

        results_table.append({
            'Model': 'Enhanced Random Forest (Tuned)', 'Room': room_name, 'Horizon': '24-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, pred_rf_enh_24h),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, pred_rf_enh_24h),
            'Precision': precision_score(y_test_clf, pred_rf_enh_24h, zero_division=0),
            'Recall': recall_score(y_test_clf, pred_rf_enh_24h, zero_division=0),
            'F1': f1_score(y_test_clf, pred_rf_enh_24h, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test_clf, prob_rf_enh_24h),
            'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })

        # 4. XGBoost
        xgb_24h = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)
        xgb_24h.fit(train_24h[enhanced_features], train_24h[target_clf].astype(int))
        pred_xgb_24h = xgb_24h.predict(test_24h[enhanced_features])
        prob_xgb_24h = xgb_24h.predict_proba(test_24h[enhanced_features])[:, 1]

        results_table.append({
            'Model': 'XGBoost (Enhanced Feat)', 'Room': room_name, 'Horizon': '24-Hour', 'Task': 'Classification',
            'Accuracy': accuracy_score(y_test_clf, pred_xgb_24h),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_clf, pred_xgb_24h),
            'Precision': precision_score(y_test_clf, pred_xgb_24h, zero_division=0),
            'Recall': recall_score(y_test_clf, pred_xgb_24h, zero_division=0),
            'F1': f1_score(y_test_clf, pred_xgb_24h, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test_clf, prob_xgb_24h),
            'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan
        })

    # Save results to CSV
    res_df = pd.DataFrame(results_table)
    csv_out = os.path.join(RESULTS_DIR, "model_comparison.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"\n[*] Model comparison saved to {csv_out}")

    # =========================================================================
    # SECTION C: GENERATE ALL 10 REQUESTED VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 4. GENERATING PUBLICATION-GRADE VISUALIZATIONS IN results/")
    print("=" * 80)

    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    clf_df = res_df[res_df['Task'] == 'Classification'].copy()
    clf_df['Label'] = clf_df['Room'] + " " + clf_df['Horizon'] + " (" + clf_df['Model'].str.replace(' \(Enhanced Feat\)| \(23 Feat\)| \(Tuned\)', '', regex=True) + ")"
    bars = plt.barh(clf_df['Label'], clf_df['Accuracy'] * 100, color='#1f77b4', edgecolor='black')
    plt.axvline(50, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('VOLTIX Model Accuracy Comparison Across Horizons & Rooms', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_comparison.png'), dpi=300)
    plt.close()

    # 2. F1-Score Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.barh(clf_df['Label'], clf_df['F1'], color='#2ca02c', edgecolor='black')
    plt.xlabel('F1-Score', fontsize=12)
    plt.title('VOLTIX F1-Score Comparison Across Horizons & Rooms', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'f1_comparison.png'), dpi=300)
    plt.close()

    # 3. ROC Curves (1-Hour Models)
    plt.figure(figsize=(9, 7))
    for r_key, r_title in [('r1_1h', 'Room 1 (80W Load)'), ('r2_1h', 'Room 2 (20W Load)')]:
        y_true = plot_data[r_key]['y_test_clf']
        fpr_rf, tpr_rf, _ = roc_curve(y_true, plot_data[r_key]['prob_rf_base'])
        auc_rf = roc_auc_score(y_true, plot_data[r_key]['prob_rf_base'])
        fpr_enh, tpr_enh, _ = roc_curve(y_true, plot_data[r_key]['prob_rf_enh'])
        auc_enh = roc_auc_score(y_true, plot_data[r_key]['prob_rf_enh'])
        fpr_xgb, tpr_xgb, _ = roc_curve(y_true, plot_data[r_key]['prob_xgb'])
        auc_xgb = roc_auc_score(y_true, plot_data[r_key]['prob_xgb'])

        plt.plot(fpr_rf, tpr_rf, linestyle=':', label=f'{r_title} Baseline RF (AUC = {auc_rf:.3f})')
        plt.plot(fpr_enh, tpr_enh, linestyle='--', label=f'{r_title} Enhanced RF (AUC = {auc_enh:.3f})')
        plt.plot(fpr_xgb, tpr_xgb, linestyle='-', linewidth=2, label=f'{r_title} XGBoost (AUC = {auc_xgb:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves — 1-Hour Horizon', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=300)
    plt.close()

    # 4. Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, (r_key, r_title) in enumerate([('r1_1h', 'Room 1 (80W)'), ('r2_1h', 'Room 2 (20W)')]):
        y_true = plot_data[r_key]['y_test_clf']
        cm_rf = confusion_matrix(y_true, plot_data[r_key]['pred_rf_enh_clf'])
        cm_xgb = confusion_matrix(y_true, plot_data[r_key]['pred_xgb_clf'])

        axes[idx, 0].imshow(cm_rf, cmap='Blues', interpolation='nearest')
        axes[idx, 0].set_title(f'{r_title} — Enhanced RF', fontsize=11, fontweight='bold')
        for i in range(2):
            for j in range(2):
                axes[idx, 0].text(j, i, str(cm_rf[i, j]), ha='center', va='center', color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black')
        axes[idx, 0].set_ylabel('True Label')
        axes[idx, 0].set_xlabel('Predicted Label')

        axes[idx, 1].imshow(cm_xgb, cmap='Greens', interpolation='nearest')
        axes[idx, 1].set_title(f'{r_title} — XGBoost', fontsize=11, fontweight='bold')
        for i in range(2):
            for j in range(2):
                axes[idx, 1].text(j, i, str(cm_xgb[i, j]), ha='center', va='center', color='white' if cm_xgb[i, j] > cm_xgb.max()/2 else 'black')
        axes[idx, 1].set_ylabel('True Label')
        axes[idx, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=300)
    plt.close()

    # 5. Actual vs Predicted Room 1 Power
    plt.figure(figsize=(12, 5))
    y_true_r1 = plot_data['r1_1h']['y_test_reg'].values[:500]
    y_pred_r1 = plot_data['r1_1h']['pred_xgb_reg'][:500]
    plt.plot(y_true_r1, label='Actual Room 1 Power (Watts)', color='black', alpha=0.8, linewidth=1.5)
    plt.plot(y_pred_r1, label='XGBoost Predicted Power (Watts)', color='#d62728', alpha=0.9, linestyle='--')
    plt.axhline(THRESHOLD_R1, color='blue', linestyle=':', label='High Usage Threshold (40W)')
    plt.xlabel('Test Telemetry Samples (Consecutive 10s steps)', fontsize=12)
    plt.ylabel('Power (Watts)', fontsize=12)
    plt.title('Room 1: Actual vs XGBoost Predicted Power (1-Hour Ahead Horizon)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted_room1_power.png'), dpi=300)
    plt.close()

    # 6. Actual vs Predicted Room 2 Power
    plt.figure(figsize=(12, 5))
    y_true_r2 = plot_data['r2_1h']['y_test_reg'].values[:500]
    y_pred_r2 = plot_data['r2_1h']['pred_xgb_reg'][:500]
    plt.plot(y_true_r2, label='Actual Room 2 Power (Watts)', color='black', alpha=0.8, linewidth=1.5)
    plt.plot(y_pred_r2, label='XGBoost Predicted Power (Watts)', color='#1f77b4', alpha=0.9, linestyle='--')
    plt.axhline(THRESHOLD_R2, color='green', linestyle=':', label='High Usage Threshold (10W)')
    plt.xlabel('Test Telemetry Samples (Consecutive 10s steps)', fontsize=12)
    plt.ylabel('Power (Watts)', fontsize=12)
    plt.title('Room 2: Actual vs XGBoost Predicted Power (1-Hour Ahead Horizon)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted_room2_power.png'), dpi=300)
    plt.close()

    # 7. MAE Comparison
    reg_df = res_df[res_df['Task'] == 'Regression'].copy()
    reg_df['Label'] = reg_df['Room'] + " (" + reg_df['Model'].str.replace(' \(Enhanced Feat\)| \(23 Feat\)| \(Tuned\)', '', regex=True) + ")"
    plt.figure(figsize=(9, 5))
    bars = plt.barh(reg_df['Label'], reg_df['MAE'], color='#ff7f0e', edgecolor='black')
    plt.xlabel('Mean Absolute Error (Watts)', fontsize=12)
    plt.title('Regression MAE Comparison Across Models (Lower is Better)', fontsize=14, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.2f} W', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'mae_comparison.png'), dpi=300)
    plt.close()

    # 8. RMSE Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.barh(reg_df['Label'], reg_df['RMSE'], color='#d62728', edgecolor='black')
    plt.xlabel('Root Mean Squared Error (Watts)', fontsize=12)
    plt.title('Regression RMSE Comparison Across Models (Lower is Better)', fontsize=14, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.2f} W', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rmse_comparison.png'), dpi=300)
    plt.close()

    # 9. R2 Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.barh(reg_df['Label'], reg_df['R2'], color='#9467bd', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Coefficient of Determination (R² Score)', fontsize=12)
    plt.title('Regression R² Comparison Across Models (Higher is Better)', fontsize=14, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        offset = 0.02 if w >= 0 else -0.08
        plt.text(w + offset, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'r2_comparison.png'), dpi=300)
    plt.close()

    # 10. Top Feature Importance Comparison
    plt.figure(figsize=(11, 7))
    xgb_fi_s = pd.Series(plot_data['r1_1h']['xgb_fi'], index=plot_data['r1_1h']['features']).sort_values(ascending=False).head(10)
    bars = plt.barh(xgb_fi_s.index[::-1], xgb_fi_s.values[::-1] * 100, color='#2b5c8f', edgecolor='black')
    plt.xlabel('Relative Feature Importance (Gini/Gain %)', fontsize=12)
    plt.title('Top 10 Causal Predictive Features — Room 1 1-Hour XGBoost Model', fontsize=14, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.2f}%', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()

    print("[+] All 10 figures and model comparison table generated successfully!")
    return res_df

if __name__ == "__main__":
    run_all_experiments()
