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
DEFAULT_THRESHOLD_R1 = 40.0
DEFAULT_THRESHOLD_R2 = 10.0

FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

def load_and_clean_data(csv_path=CSV_PATH):
    """
    Loads full raw dataset, validates timestamps, sorts chronologically, and cleans numeric fields.
    """
    df = pd.read_csv(csv_path)
    raw_count = len(df)
    
    # Filter out invalid / unparseable timestamps
    df_valid = df[df['timestamp'] != 'TIME_ERROR'].copy()
    df_valid['timestamp'] = pd.to_datetime(df_valid['timestamp'], errors='coerce')
    df_valid = df_valid.dropna(subset=['timestamp'])
    
    # Sort strictly chronologically
    df_sorted = df_valid.sort_values('timestamp').reset_index(drop=True)
    
    # Numeric sanitization
    num_cols = ['current1', 'current2', 'energy1', 'energy2', 'power1', 'power2', 
                'totalCurrent', 'totalEnergy', 'totalPower', 'voltage1', 'voltage2']
    for col in num_cols:
        if col in df_sorted.columns:
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').fillna(0.0)

    counts = {
        "raw_csv_rows": raw_count,
        "rows_after_timestamp_validation": len(df_valid),
        "rows_after_sorting": len(df_sorted),
        "start_timestamp": str(df_sorted['timestamp'].min()),
        "end_timestamp": str(df_sorted['timestamp'].max())
    }
    return df_sorted, counts

def prepare_features_and_targets(df, threshold_r1=DEFAULT_THRESHOLD_R1, threshold_r2=DEFAULT_THRESHOLD_R2):
    """
    Constructs strictly causal historical features at time t and timestamp lookahead targets.
    NO bfill: eliminates all backward leakage. Drops initial rows lacking lag history.
    """
    df = df.copy()

    # Time & cyclic features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # Strictly causal lags: no bfill()
    for col in ['power1', 'power2', 'totalPower', 'current1', 'current2']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_roll_mean5'] = df[col].rolling(window=5, min_periods=5).mean()
        df[f'{col}_roll_max5'] = df[col].rolling(window=5, min_periods=5).max()

    # Drop initial rows lacking lag history
    df_causal = df.dropna(subset=['power1_lag3', 'power1_roll_mean5']).reset_index(drop=True)

    # 1-Hour Forward Target (t + 1 hour, tolerance 15s)
    df_q1h = pd.DataFrame({
        'orig_idx': df_causal.index,
        'timestamp': df_causal['timestamp'],
        'query_1h': df_causal['timestamp'] + pd.Timedelta(hours=1)
    })
    m1h = pd.merge_asof(
        df_q1h.sort_values('query_1h'),
        df_causal[['timestamp', 'power1', 'power2']].rename(
            columns={'timestamp': 'match_time_1h', 'power1': 'power1_1h_actual', 'power2': 'power2_1h_actual'}
        ),
        left_on='query_1h', right_on='match_time_1h', direction='nearest', tolerance=pd.Timedelta(seconds=15)
    ).sort_values('orig_idx').reset_index(drop=True)

    df_causal['match_time_1h'] = m1h['match_time_1h']
    df_causal['power1_1h_actual'] = m1h['power1_1h_actual']
    df_causal['power2_1h_actual'] = m1h['power2_1h_actual']
    df_causal['high_usage_r1_next_hour'] = (df_causal['power1_1h_actual'] >= threshold_r1).astype(float)
    df_causal['high_usage_r2_next_hour'] = (df_causal['power2_1h_actual'] >= threshold_r2).astype(float)

    # 24-Hour Forward Target (t + 24 hours, tolerance 15s)
    df_q24h = pd.DataFrame({
        'orig_idx': df_causal.index,
        'timestamp': df_causal['timestamp'],
        'query_24h': df_causal['timestamp'] + pd.Timedelta(hours=24)
    })
    m24h = pd.merge_asof(
        df_q24h.sort_values('query_24h'),
        df_causal[['timestamp', 'power1', 'power2']].rename(
            columns={'timestamp': 'match_time_24h', 'power1': 'power1_24h_actual', 'power2': 'power2_24h_actual'}
        ),
        left_on='query_24h', right_on='match_time_24h', direction='nearest', tolerance=pd.Timedelta(seconds=15)
    ).sort_values('orig_idx').reset_index(drop=True)

    df_causal['match_time_24h'] = m24h['match_time_24h']
    df_causal['power1_24h_actual'] = m24h['power1_24h_actual']
    df_causal['power2_24h_actual'] = m24h['power2_24h_actual']
    df_causal['high_usage_r1_tomorrow'] = (df_causal['power1_24h_actual'] >= threshold_r1).astype(float)
    df_causal['high_usage_r2_tomorrow'] = (df_causal['power2_24h_actual'] >= threshold_r2).astype(float)

    return df_causal

class EnergyPredictionAdvisor:
    def __init__(self, threshold_r1=DEFAULT_THRESHOLD_R1, threshold_r2=DEFAULT_THRESHOLD_R2):
        self.threshold_r1 = threshold_r1
        self.threshold_r2 = threshold_r2
        self.feature_cols = FEATURE_COLS
        self.models = {}
        self.tomorrow_thresholds = {'r1': 0.40, 'r2': 0.40}

    def train_all_models(self, df):
        # 1-Hour Horizon (80% Train / 20% Test)
        df_1h = df.dropna(subset=['power1_1h_actual', 'power2_1h_actual']).reset_index(drop=True)
        split_1h = int(len(df_1h) * 0.8)
        train_1h = df_1h.iloc[:split_1h]
        test_1h = df_1h.iloc[split_1h:]

        X_tr_1h = train_1h[self.feature_cols]
        X_te_1h = test_1h[self.feature_cols]

        print("\n" + "=" * 80)
        print("    1-HOUR HORIZON CHRONOLOGICAL SPLIT (80% Train / 20% Test)")
        print("=" * 80)
        print(f"Total 1h Records: {len(df_1h)} | Train: {len(train_1h)} | Test: {len(test_1h)}")

        for model_key, target_col, curr_col, thresh, desc in [
            ('r1_next_hour', 'high_usage_r1_next_hour', 'power1', self.threshold_r1, 'Channel 1 1-Hour Classifier'),
            ('r2_next_hour', 'high_usage_r2_next_hour', 'power2', self.threshold_r2, 'Channel 2 1-Hour Classifier')
        ]:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_tr_1h, train_1h[target_col].astype(int))
            y_pred = clf.predict(X_te_1h)
            y_proba = clf.predict_proba(X_te_1h)[:, 1]
            y_test = test_1h[target_col].astype(int)

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_proba)
            cm = confusion_matrix(y_test, y_pred)

            y_persist = (test_1h[curr_col] >= thresh).astype(int)
            p_acc = accuracy_score(y_test, y_persist)
            p_f1 = f1_score(y_test, y_persist, zero_division=0)

            self.models[model_key] = clf
            joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_key}_rf.pkl"))
            print(f"\n[{desc}] (Threshold = {thresh} W)")
            print(f"  RF Model   : Acc: {acc*100:6.2f}% | BalAcc: {bal_acc:6.4f} | Prec: {prec:6.4f} | Rec: {rec:6.4f} | F1: {f1:6.4f} | ROC-AUC: {roc:6.4f}")
            print(f"  Persistence: Acc: {p_acc*100:6.2f}% | F1: {p_f1:6.4f}")
            print(f"  Confusion Matrix: [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")

        for model_key, target_col, curr_col, desc in [
            ('r1_reg_next', 'power1_1h_actual', 'power1', 'Channel 1 1-Hour Regressor (Watts)'),
            ('r2_reg_next', 'power2_1h_actual', 'power2', 'Channel 2 1-Hour Regressor (Watts)')
        ]:
            reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            reg.fit(X_tr_1h, train_1h[target_col])
            y_pred = reg.predict(X_te_1h)
            y_test = test_1h[target_col]

            rf_mae = mean_absolute_error(y_test, y_pred)
            rf_rmse = root_mean_squared_error(y_test, y_pred)
            rf_r2 = r2_score(y_test, y_pred)

            y_persist = test_1h[curr_col]
            p_mae = mean_absolute_error(y_test, y_persist)
            p_rmse = root_mean_squared_error(y_test, y_persist)
            p_r2 = r2_score(y_test, y_persist)

            self.models[model_key] = reg
            joblib.dump(reg, os.path.join(MODEL_DIR, f"{model_key}_rf.pkl"))
            print(f"\n[{desc}]")
            print(f"  RF Model   : MAE = {rf_mae:7.2f} W | RMSE = {rf_rmse:7.2f} W | R2 = {rf_r2:7.4f}")
            print(f"  Persistence: MAE = {p_mae:7.2f} W | RMSE = {p_rmse:7.2f} W | R2 = {p_r2:7.4f}")

        # 24-Hour Horizon (70% Train / 15% Val / 15% Test)
        df_24h = df.dropna(subset=['power1_24h_actual', 'power2_24h_actual']).reset_index(drop=True)
        n_tot = len(df_24h)
        n_tr = int(n_tot * 0.70)
        n_val = int(n_tot * 0.15)
        train_24h = df_24h.iloc[:n_tr]
        val_24h = df_24h.iloc[n_tr:n_tr+n_val]
        test_24h = df_24h.iloc[n_tr+n_val:]

        X_tr_24h = train_24h[self.feature_cols]
        X_val_24h = val_24h[self.feature_cols]
        X_te_24h = test_24h[self.feature_cols]

        print("\n" + "=" * 80)
        print("    24-HOUR HORIZON CHRONOLOGICAL SPLIT (70% Train / 15% Val / 15% Test)")
        print("=" * 80)
        print(f"Total 24h Records: {len(df_24h)} | Train: {len(train_24h)} | Val: {len(val_24h)} | Test: {len(test_24h)}")

        frozen_thresholds = {}
        for model_key, target_col, curr_col, thresh, r_key, desc in [
            ('r1_tomorrow', 'high_usage_r1_tomorrow', 'power1', self.threshold_r1, 'r1', 'Channel 1 Tomorrow Classifier'),
            ('r2_tomorrow', 'high_usage_r2_tomorrow', 'power2', self.threshold_r2, 'r2', 'Channel 2 Tomorrow Classifier')
        ]:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_tr_24h, train_24h[target_col].astype(int))

            # Threshold selection ONLY on validation set
            y_val_proba = clf.predict_proba(X_val_24h)[:, 1]
            best_th = 0.50
            best_val_f1 = f1_score(val_24h[target_col].astype(int), (y_val_proba >= 0.50).astype(int), zero_division=0)
            for cand_th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                cand_f1 = f1_score(val_24h[target_col].astype(int), (y_val_proba >= cand_th).astype(int), zero_division=0)
                if cand_f1 > best_val_f1:
                    best_val_f1 = cand_f1
                    best_th = cand_th

            frozen_thresholds[r_key] = round(best_th, 2)
            y_test = test_24h[target_col].astype(int)
            y_te_proba = clf.predict_proba(X_te_24h)[:, 1]

            # Final evaluation on untouched test set
            y_pred_50 = (y_te_proba >= 0.50).astype(int)
            acc_50 = accuracy_score(y_test, y_pred_50)
            f1_50 = f1_score(y_test, y_pred_50, zero_division=0)
            roc_50 = roc_auc_score(y_test, y_te_proba)
            cm_50 = confusion_matrix(y_test, y_pred_50)

            y_pred_opt = (y_te_proba >= best_th).astype(int)
            acc_opt = accuracy_score(y_test, y_pred_opt)
            f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)

            y_persist = (test_24h[curr_col] >= thresh).astype(int)
            p_acc = accuracy_score(y_test, y_persist)
            p_f1 = f1_score(y_test, y_persist, zero_division=0)

            self.models[model_key] = clf
            joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_key}_rf.pkl"))

            print(f"\n[{desc}] (Frozen Val-Selected Threshold = {best_th:.2f})")
            print(f"  RF Model (th=0.50)    : Acc: {acc_50*100:6.2f}% | F1: {f1_50:6.4f} | ROC-AUC: {roc_50:6.4f}")
            print(f"  RF Model (th={best_th:.2f})    : Acc: {acc_opt*100:6.2f}% | F1: {f1_opt:6.4f}")
            print(f"  Persistence Baseline  : Acc: {p_acc*100:6.2f}% | F1: {p_f1:6.4f}")
            print(f"  Confusion Matrix (th=0.50): [[TN={cm_50[0,0]}, FP={cm_50[0,1]}], [FN={cm_50[1,0]}, TP={cm_50[1,1]}]]")

        with open(os.path.join(MODEL_DIR, "tomorrow_thresholds.json"), "w") as f:
            json.dump(frozen_thresholds, f, indent=2)

def main():
    df_sorted, counts = load_and_clean_data(CSV_PATH)
    df_prepared = prepare_features_and_targets(df_sorted)
    advisor = EnergyPredictionAdvisor()
    advisor.train_all_models(df_prepared)

if __name__ == "__main__":
    main()
