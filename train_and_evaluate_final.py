import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, mean_absolute_error,
    root_mean_squared_error, r2_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "VOLTIX_OCTOBER_2026_calibrated_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# CALIBRATED HARDWARE THRESHOLDS
# Channel 1 (power1): 2 x 40W bulbs (0W vs 80W) -> 40.0 W classification threshold
# Channel 2 (power2): 2 x 10W bulbs (0W vs 20W) -> 10.0 W classification threshold
# -------------------------------------------------------------------------
THRESHOLD_R1 = 40.0
THRESHOLD_R2 = 10.0

FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

def load_and_preprocess_dataset(csv_path=CSV_PATH):
    print("\n" + "=" * 80)
    print(" 1. DATASET INGESTION & STRICT CAUSAL FEATURE EXTRACTION (NO BFILL LEAKAGE)")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    raw_count = len(df)
    
    # 1. Timestamp validation & chronological sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # 2. Numeric sanitization
    num_cols = ['current1', 'current2', 'energy1', 'energy2', 'power1', 'power2', 
                'totalCurrent', 'totalEnergy', 'totalPower', 'voltage1', 'voltage2']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 3. Strictly Causal Features (Zero lookahead)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # Strictly causal lags: no bfill() to prevent backward leakage
    for col in ['power1', 'power2', 'totalPower', 'current1', 'current2']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_roll_mean5'] = df[col].rolling(window=5, min_periods=5).mean()
        df[f'{col}_roll_max5'] = df[col].rolling(window=5, min_periods=5).max()

    # Drop initial rows that lack full lag/rolling history
    df_causal = df.dropna(subset=['power1_lag3', 'power1_roll_mean5']).reset_index(drop=True)
    print(f"Raw CSV rows                : {raw_count}")
    print(f"Initial rows dropped (no lag history): {raw_count - len(df_causal)}")
    print(f"Causal dataset rows         : {len(df_causal)}")
    print(f"Time span                   : {df_causal['timestamp'].min()} to {df_causal['timestamp'].max()}")

    # 4. Multi-Horizon Timestamp Lookahead Target Construction
    # 1-Hour Horizon (t + 1h, tolerance 15s)
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
    df_causal['high_usage_r1_next_hour'] = (df_causal['power1_1h_actual'] >= THRESHOLD_R1).astype(float)
    df_causal['high_usage_r2_next_hour'] = (df_causal['power2_1h_actual'] >= THRESHOLD_R2).astype(float)

    # 24-Hour Horizon (t + 24h, tolerance 15s)
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

    return df_causal

def train_and_evaluate_pipeline():
    df = load_and_preprocess_dataset()

    # =========================================================================
    # SECTION 1: 1-HOUR HORIZON (80% TRAIN / 20% FINAL TEST)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 2. 1-HOUR MODELS TRAINING & EVALUATION (80% CHRONOLOGICAL TRAIN / 20% TEST)")
    print("=" * 80)
    
    df_1h = df.dropna(subset=['power1_1h_actual', 'power2_1h_actual']).reset_index(drop=True)
    split_1h = int(len(df_1h) * 0.8)
    train_1h = df_1h.iloc[:split_1h]
    test_1h = df_1h.iloc[split_1h:]

    X_tr_1h = train_1h[FEATURE_COLS]
    X_te_1h = test_1h[FEATURE_COLS]

    delta_1h = (df_1h['match_time_1h'] - df_1h['timestamp']).dt.total_seconds()
    print(f"Total 1-Hour Valid Records : {len(df_1h)}")
    print(f"1-Hour Lookahead Range     : {delta_1h.min():.1f} s to {delta_1h.max():.1f} s (Mean: {delta_1h.mean():.2f} s)")
    print(f"80% Training Set Count     : {len(train_1h)} ({train_1h['timestamp'].min()} to {train_1h['timestamp'].max()})")
    print(f"20% Testing Set Count      : {len(test_1h)} ({test_1h['timestamp'].min()} to {test_1h['timestamp'].max()})")
    print(f"Train Channel 1 Class Distribution : {train_1h['high_usage_r1_next_hour'].value_counts(normalize=True).to_dict()}")
    print(f"Test Channel 1 Class Distribution  : {test_1h['high_usage_r1_next_hour'].value_counts(normalize=True).to_dict()}")
    print(f"Train Channel 2 Class Distribution : {train_1h['high_usage_r2_next_hour'].value_counts(normalize=True).to_dict()}")
    print(f"Test Channel 2 Class Distribution  : {test_1h['high_usage_r2_next_hour'].value_counts(normalize=True).to_dict()}")

    # 1.1 1-Hour Ahead High Usage Classifiers
    for model_name, target_col, curr_col, thresh, desc in [
        ('r1_next_hour', 'high_usage_r1_next_hour', 'power1', THRESHOLD_R1, 'Channel 1 (80W Load) 1-Hour Classifier'),
        ('r2_next_hour', 'high_usage_r2_next_hour', 'power2', THRESHOLD_R2, 'Channel 2 (20W Load) 1-Hour Classifier')
    ]:
        y_train = train_1h[target_col].astype(int)
        y_test = test_1h[target_col].astype(int)

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_tr_1h, y_train)
        y_pred = clf.predict(X_te_1h)
        y_proba = clf.predict_proba(X_te_1h)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Persistence baseline
        y_persist = (test_1h[curr_col] >= thresh).astype(int)
        p_acc = accuracy_score(y_test, y_persist)
        p_bal = balanced_accuracy_score(y_test, y_persist)
        p_prec = precision_score(y_test, y_persist, zero_division=0)
        p_rec = recall_score(y_test, y_persist, zero_division=0)
        p_f1 = f1_score(y_test, y_persist, zero_division=0)

        joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_name}_rf.pkl"))

        print(f"\n--- [{desc}] (Threshold = {thresh} W) ---")
        print(f"  Random Forest Model   : Acc: {acc*100:6.2f}% | BalAcc: {bal_acc:6.4f} | Prec: {prec:6.4f} | Rec: {rec:6.4f} | F1: {f1:6.4f} | ROC-AUC: {roc:6.4f}")
        print(f"  Persistence Baseline  : Acc: {p_acc*100:6.2f}% | BalAcc: {p_bal:6.4f} | Prec: {p_prec:6.4f} | Rec: {p_rec:6.4f} | F1: {p_f1:6.4f}")
        print(f"  Confusion Matrix      : [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")

    # 1.2 1-Hour Ahead Power Regressors
    for model_name, target_col, curr_col, desc in [
        ('r1_reg_next', 'power1_1h_actual', 'power1', 'Channel 1 (80W Load) 1-Hour Power Regressor (Watts)'),
        ('r2_reg_next', 'power2_1h_actual', 'power2', 'Channel 2 (20W Load) 1-Hour Power Regressor (Watts)')
    ]:
        y_train = train_1h[target_col]
        y_test = test_1h[target_col]

        reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        reg.fit(X_tr_1h, y_train)
        y_pred = reg.predict(X_te_1h)

        rf_mae = mean_absolute_error(y_test, y_pred)
        rf_rmse = root_mean_squared_error(y_test, y_pred)
        rf_r2 = r2_score(y_test, y_pred)

        y_persist = test_1h[curr_col]
        p_mae = mean_absolute_error(y_test, y_persist)
        p_rmse = root_mean_squared_error(y_test, y_persist)
        p_r2 = r2_score(y_test, y_persist)

        joblib.dump(reg, os.path.join(MODEL_DIR, f"{model_name}_rf.pkl"))

        print(f"\n--- [{desc}] ---")
        print(f"  Random Forest Model   : MAE = {rf_mae:7.2f} W | RMSE = {rf_rmse:7.2f} W | R2 = {rf_r2:7.4f}")
        print(f"  Persistence Baseline  : MAE = {p_mae:7.2f} W | RMSE = {p_rmse:7.2f} W | R2 = {p_r2:7.4f}")

    # =========================================================================
    # SECTION 2: 24-HOUR HORIZON (70% TRAIN / 15% VALIDATION / 15% FINAL TEST)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 3. 24-HOUR (TOMORROW) MODELS (70% TRAIN / 15% VALIDATION / 15% FINAL TEST)")
    print("=" * 80)

    df_24h = df.dropna(subset=['power1_24h_actual', 'power2_24h_actual']).reset_index(drop=True)
    n_tot = len(df_24h)
    n_tr = int(n_tot * 0.70)
    n_val = int(n_tot * 0.15)
    train_24h = df_24h.iloc[:n_tr]
    val_24h = df_24h.iloc[n_tr:n_tr+n_val]
    test_24h = df_24h.iloc[n_tr+n_val:]

    X_tr_24h = train_24h[FEATURE_COLS]
    X_val_24h = val_24h[FEATURE_COLS]
    X_te_24h = test_24h[FEATURE_COLS]

    delta_24h = (df_24h['match_time_24h'] - df_24h['timestamp']).dt.total_seconds()
    print(f"Total 24-Hour Valid Records: {len(df_24h)}")
    print(f"24-Hour Lookahead Range    : {delta_24h.min():.1f} s to {delta_24h.max():.1f} s (Mean: {delta_24h.mean():.2f} s)")
    print(f"70% Training Set Count     : {len(train_24h)} ({train_24h['timestamp'].min()} to {train_24h['timestamp'].max()})")
    print(f"15% Validation Set Count   : {len(val_24h)} ({val_24h['timestamp'].min()} to {val_24h['timestamp'].max()})")
    print(f"15% Final Test Set Count   : {len(test_24h)} ({test_24h['timestamp'].min()} to {test_24h['timestamp'].max()})")
    print(f"Train Channel 1 Class Distribution : {train_24h['high_usage_r1_tomorrow'].value_counts(normalize=True).to_dict()}")
    print(f"Val Channel 1 Class Distribution   : {val_24h['high_usage_r1_tomorrow'].value_counts(normalize=True).to_dict()}")
    print(f"Test Channel 1 Class Distribution  : {test_24h['high_usage_r1_tomorrow'].value_counts(normalize=True).to_dict()}")
    print(f"Train Channel 2 Class Distribution : {train_24h['high_usage_r2_tomorrow'].value_counts(normalize=True).to_dict()}")
    print(f"Val Channel 2 Class Distribution   : {val_24h['high_usage_r2_tomorrow'].value_counts(normalize=True).to_dict()}")
    print(f"Test Channel 2 Class Distribution  : {test_24h['high_usage_r2_tomorrow'].value_counts(normalize=True).to_dict()}")

    frozen_thresholds = {}

    for model_name, target_col, curr_col, thresh, r_key, desc in [
        ('r1_tomorrow', 'high_usage_r1_tomorrow', 'power1', THRESHOLD_R1, 'r1', 'Channel 1 (80W Load) Tomorrow Classifier'),
        ('r2_tomorrow', 'high_usage_r2_tomorrow', 'power2', THRESHOLD_R2, 'r2', 'Channel 2 (20W Load) Tomorrow Classifier')
    ]:
        y_train = train_24h[target_col].astype(int)
        y_val = val_24h[target_col].astype(int)
        y_test = test_24h[target_col].astype(int)

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_tr_24h, y_train)

        # -------------------------------------------------------------
        # STEP A: Threshold Search STRICTLY on 15% Validation Set
        # -------------------------------------------------------------
        y_val_proba = clf.predict_proba(X_val_24h)[:, 1]
        best_th = 0.50
        best_val_f1 = f1_score(y_val, (y_val_proba >= 0.50).astype(int), zero_division=0)

        print(f"\n>>> [{desc}] (Threshold Selection on 15% Validation Set)")
        for cand_th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            cand_pred = (y_val_proba >= cand_th).astype(int)
            cand_f1 = f1_score(y_val, cand_pred, zero_division=0)
            cand_acc = accuracy_score(y_val, cand_pred)
            cand_prec = precision_score(y_val, cand_pred, zero_division=0)
            cand_rec = recall_score(y_val, cand_pred, zero_division=0)
            print(f"  Val th={cand_th:.2f}: Acc={cand_acc*100:6.2f}% | Prec={cand_prec:.4f} | Rec={cand_rec:.4f} | F1={cand_f1:.4f}")
            if cand_f1 > best_val_f1:
                best_val_f1 = cand_f1
                best_th = cand_th

        frozen_thresholds[r_key] = round(best_th, 2)
        print(f"[*] Selected Frozen Threshold for {r_key}: {best_th:.2f} (Validation F1 = {best_val_f1:.4f})")

        # -------------------------------------------------------------
        # STEP B: Single Final Evaluation on UNTOUCHED 15% Final Test Set
        # -------------------------------------------------------------
        y_te_proba = clf.predict_proba(X_te_24h)[:, 1]

        # 1. Default 0.50 Threshold Evaluation
        y_te_pred_50 = (y_te_proba >= 0.50).astype(int)
        acc_50 = accuracy_score(y_test, y_te_pred_50)
        bal_acc_50 = balanced_accuracy_score(y_test, y_te_pred_50)
        prec_50 = precision_score(y_test, y_te_pred_50, zero_division=0)
        rec_50 = recall_score(y_test, y_te_pred_50, zero_division=0)
        f1_50 = f1_score(y_test, y_te_pred_50, zero_division=0)
        roc_50 = roc_auc_score(y_test, y_te_proba)
        cm_50 = confusion_matrix(y_test, y_te_pred_50)

        # 2. Frozen Validation-Selected Threshold Evaluation
        y_te_pred_opt = (y_te_proba >= best_th).astype(int)
        acc_opt = accuracy_score(y_test, y_te_pred_opt)
        bal_acc_opt = balanced_accuracy_score(y_test, y_te_pred_opt)
        prec_opt = precision_score(y_test, y_te_pred_opt, zero_division=0)
        rec_opt = recall_score(y_test, y_te_pred_opt, zero_division=0)
        f1_opt = f1_score(y_test, y_te_pred_opt, zero_division=0)
        cm_opt = confusion_matrix(y_test, y_te_pred_opt)

        # 3. 24-Hour Persistence Baseline
        y_te_persist = (test_24h[curr_col] >= thresh).astype(int)
        p_acc = accuracy_score(y_test, y_te_persist)
        p_bal = balanced_accuracy_score(y_test, y_te_persist)
        p_prec = precision_score(y_test, y_te_persist, zero_division=0)
        p_rec = recall_score(y_test, y_te_persist, zero_division=0)
        p_f1 = f1_score(y_test, y_te_persist, zero_division=0)

        joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_name}_rf.pkl"))

        print(f"\n=== FINAL UNTOUCHED TEST RESULTS: [{desc}] ===")
        print(f"[A. Default 0.50 Threshold]")
        print(f"  RF Model (th=0.50)    : Acc: {acc_50*100:6.2f}% | BalAcc: {bal_acc_50:6.4f} | Prec: {prec_50:6.4f} | Rec: {rec_50:6.4f} | F1: {f1_50:6.4f} | ROC-AUC: {roc_50:6.4f}")
        print(f"  Confusion Matrix      : [[TN={cm_50[0,0]}, FP={cm_50[0,1]}], [FN={cm_50[1,0]}, TP={cm_50[1,1]}]]")
        
        print(f"\n[B. Frozen Validation-Selected ({best_th:.2f}) Threshold]")
        print(f"  RF Model (th={best_th:.2f})    : Acc: {acc_opt*100:6.2f}% | BalAcc: {bal_acc_opt:6.4f} | Prec: {prec_opt:6.4f} | Rec: {rec_opt:6.4f} | F1: {f1_opt:6.4f}")
        print(f"  Confusion Matrix      : [[TN={cm_opt[0,0]}, FP={cm_opt[0,1]}], [FN={cm_opt[1,0]}, TP={cm_opt[1,1]}]]")

        print(f"\n[C. 24-Hour Persistence Baseline]")
        print(f"  Persistence Baseline  : Acc: {p_acc*100:6.2f}% | BalAcc: {p_bal:6.4f} | Prec: {p_prec:6.4f} | Rec: {p_rec:6.4f} | F1: {p_f1:6.4f}")

    # Save frozen threshold JSON
    thresh_file = os.path.join(MODEL_DIR, "tomorrow_thresholds.json")
    with open(thresh_file, "w") as f:
        json.dump(frozen_thresholds, f, indent=2)
    print(f"\n[*] Saved frozen advance warning thresholds to {thresh_file}: {frozen_thresholds}")

    # =========================================================================
    # SECTION 3: MODEL-DERIVED FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 4. MODEL-DERIVED FEATURE IMPORTANCES (TOP 5 PREDICTORS)")
    print("=" * 80)
    for model_name in ['r1_next_hour', 'r2_next_hour', 'r1_reg_next', 'r2_reg_next', 'r1_tomorrow', 'r2_tomorrow']:
        m = joblib.load(os.path.join(MODEL_DIR, f"{model_name}_rf.pkl"))
        fi = pd.Series(m.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print(f"\n[{model_name}] Top 5 Predictive Features:")
        for rank, (feat, score) in enumerate(fi.head(5).items(), 1):
            print(f"  {rank}. {feat:22s}: {score*100:5.2f}%")

    print("\n[+] Full pipeline training, validation, threshold freezing, and testing completed successfully!")

if __name__ == "__main__":
    train_and_evaluate_pipeline()
