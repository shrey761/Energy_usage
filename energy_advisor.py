import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "LATEST_ML_TRAINING_DATA.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(DATA_DIR, "LATEST_ML_TRAINING_DATA.xlsx")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(DATA_DIR, "energy_data.csv")

MODEL_DIR = os.path.join(DATA_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 1. DATA LOADING & VALIDATION PIPELINE
# -------------------------------------------------------------------------
def load_and_clean_data(csv_path=CSV_PATH):
    """
    Loads full raw dataset, validates timestamps, and sorts strictly chronologically.
    """
    if csv_path.endswith('.xlsx'):
        df_raw = pd.read_excel(csv_path)
    else:
        df_raw = pd.read_csv(csv_path)
    raw_count = len(df_raw)

    # 1. Filter out invalid/TIME_ERROR timestamps
    df_valid = df_raw[df_raw['timestamp'] != 'TIME_ERROR'].copy()
    df_valid['timestamp'] = pd.to_datetime(df_valid['timestamp'], errors='coerce')
    df_valid = df_valid.dropna(subset=['timestamp'])
    valid_count = len(df_valid)

    # 2. Sort strictly chronologically
    df_sorted = df_valid.sort_values('timestamp').reset_index(drop=True)
    sorted_count = len(df_sorted)

    # 3. Numeric column sanitization
    num_cols = ['current1', 'current2', 'energy1', 'energy2', 'power1', 'power2', 
                'totalCurrent', 'totalEnergy', 'totalPower', 'voltage1', 'voltage2']
    for col in num_cols:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').fillna(0.0)

    counts = {
        "raw_csv_rows": raw_count,
        "rows_after_timestamp_validation": valid_count,
        "rows_after_sorting": sorted_count,
        "start_timestamp": str(df_sorted['timestamp'].min()),
        "end_timestamp": str(df_sorted['timestamp'].max())
    }
    return df_sorted, counts

# -------------------------------------------------------------------------
# 2. FEATURE ENGINEERING & MULTI-HORIZON TARGET CREATION
# -------------------------------------------------------------------------
def prepare_features_and_targets(df, high_power_threshold=1200.0):
    """
    Constructs strictly historical/current features at time t and timestamp-based forward targets:
    1. 1-Hour Forward Target: t + 1 hour (tolerance: 15 mins)
    2. 24-Hour Forward Target: t + 24 hours (tolerance: 15 mins)
    Zero future leakage: features at time t use only observations at or before t.
    """
    df = df.copy()

    # Time-based & Cyclic features (strictly historical/current at time t)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # Backward-looking lags & rolling statistics
    for col in ['power1', 'power2', 'totalPower', 'current1', 'current2']:
        df[f'{col}_lag1'] = df[col].shift(1).bfill()
        df[f'{col}_lag3'] = df[col].shift(3).bfill()
        df[f'{col}_roll_mean5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_roll_max5'] = df[col].rolling(window=5, min_periods=1).max()

    # ---------------------------------------------------------------------
    # Target Horizon 1: True 1-Hour Forward Lookahead Target (t + 1 Hour Lookahead)
    # ---------------------------------------------------------------------
    df_query_1h = pd.DataFrame({
        'orig_idx': df.index,
        'timestamp': df['timestamp'],
        'query_1h': df['timestamp'] + pd.Timedelta(hours=1)
    })

    merged_1h = pd.merge_asof(
        df_query_1h.sort_values('query_1h'),
        df[['timestamp', 'power1', 'power2']].rename(
            columns={'timestamp': 'match_time_1h', 'power1': 'power1_1h_actual', 'power2': 'power2_1h_actual'}
        ),
        left_on='query_1h',
        right_on='match_time_1h',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=15)
    ).sort_values('orig_idx').reset_index(drop=True)

    df['match_time_1h'] = merged_1h['match_time_1h']
    df['power1_1h_actual'] = merged_1h['power1_1h_actual']
    df['power2_1h_actual'] = merged_1h['power2_1h_actual']
    df['high_usage_r1_next_hour'] = (df['power1_1h_actual'] >= high_power_threshold).astype(float)
    df['high_usage_r2_next_hour'] = (df['power2_1h_actual'] >= high_power_threshold).astype(float)

    # ---------------------------------------------------------------------
    # Target Horizon 2: True 24-Hour Forward Lookahead Target (t + 24 Hours Lookahead)
    # ---------------------------------------------------------------------
    df_query_24h = pd.DataFrame({
        'orig_idx': df.index,
        'timestamp': df['timestamp'],
        'query_24h': df['timestamp'] + pd.Timedelta(hours=24)
    })

    merged_24h = pd.merge_asof(
        df_query_24h.sort_values('query_24h'),
        df[['timestamp', 'power1', 'power2']].rename(
            columns={'timestamp': 'match_time_24h', 'power1': 'future_p1_24h', 'power2': 'future_p2_24h'}
        ),
        left_on='query_24h',
        right_on='match_time_24h',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=15)
    ).sort_values('orig_idx').reset_index(drop=True)

    df['power1_24h_actual'] = merged_24h['future_p1_24h']
    df['power2_24h_actual'] = merged_24h['future_p2_24h']
    df['high_usage_r1_tomorrow'] = (df['power1_24h_actual'] >= high_power_threshold).astype(float)
    df['high_usage_r2_tomorrow'] = (df['power2_24h_actual'] >= high_power_threshold).astype(float)

    # Drop trailing rows where 1-hour target has no future reading
    df_final = df.dropna(subset=['power1_1h_actual', 'power2_1h_actual']).reset_index(drop=True)
    return df_final

# -------------------------------------------------------------------------
# 3. ENERGY ADVISOR MODEL & CLASSIFIER-FIRST ADVISORY SYSTEM
# -------------------------------------------------------------------------
class EnergyPredictionAdvisor:
    def __init__(self, threshold_watts=1200.0):
        self.threshold_watts = threshold_watts
        self.feature_cols = [
            'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
            'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
            'totalPower', 'totalCurrent',
            'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
            'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
            'totalPower_lag1', 'totalPower_roll_mean5'
        ]
        self.models = {}
        self.tomorrow_thresholds = {'r1': 0.70, 'r2': 0.65}
        self._load_existing_tomorrow_models()

    def _load_existing_tomorrow_models(self):
        """Loads existing verified tomorrow models and thresholds without retraining them."""
        for m in ['r1_tomorrow', 'r2_tomorrow']:
            p = os.path.join(MODEL_DIR, f"{m}_rf.pkl")
            if os.path.exists(p):
                self.models[m] = joblib.load(p)
        
        t_path = os.path.join(MODEL_DIR, "tomorrow_thresholds.json")
        if os.path.exists(t_path):
            try:
                with open(t_path, "r") as f:
                    self.tomorrow_thresholds = json.load(f)
            except Exception:
                pass

    def retrain_one_hour_models(self, df):
        """
        Retrains ONLY the 4 authorized 1-hour ahead models:
        1. r1_next_hour_rf.pkl
        2. r2_next_hour_rf.pkl
        3. r1_reg_next_rf.pkl
        4. r2_reg_next_rf.pkl
        Using true 1-hour timestamp-based forward lookahead targets.
        """
        # Chronological 80/20 train/test split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[self.feature_cols]
        X_test = test_df[self.feature_cols]

        print("\n" + "=" * 78)
        print("    CHRONOLOGICAL DATASET SPLIT (80% Train / 20% Test) — 1-HOUR HORIZON")
        print("=" * 78)
        print(f"Total Processed Records : {len(df)}")
        print(f"Training Set Count      : {len(train_df)} ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
        print(f"Testing Set Count       : {len(test_df)} ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

        # -------------------------------------------------------------
        # 1. RETRAIN 1-HOUR AHEAD CLASSIFIERS
        # -------------------------------------------------------------
        print("\n" + "=" * 78)
        print("       1-HOUR AHEAD HIGH-USAGE CLASSIFIERS (TRUE t + 1 HOUR TARGET)")
        print("=" * 78)

        for model_key, target_col, desc in [
            ('r1_next_hour', 'high_usage_r1_next_hour', 'Room 1 1-Hour Ahead High Usage Classifier'),
            ('r2_next_hour', 'high_usage_r2_next_hour', 'Room 2 1-Hour Ahead High Usage Classifier')
        ]:
            y_train = train_df[target_col].astype(int)
            y_test = test_df[target_col].astype(int)

            clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_proba)
            cm = confusion_matrix(y_test, y_pred)

            self.models[model_key] = clf
            joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_key}_rf.pkl"))

            print(f"\n[{desc}]")
            print(f"  Classes   : {clf.classes_} (0=NORMAL, 1=HIGH USAGE)")
            print(f"  Accuracy  : {acc * 100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
            print(f"  Confusion Matrix [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")

        # -------------------------------------------------------------
        # 2. RETRAIN 1-HOUR AHEAD POWER REGRESSORS
        # -------------------------------------------------------------
        print("\n" + "=" * 78)
        print("    1-HOUR AHEAD POWER REGRESSORS (TRUE t + 1 HOUR TARGET IN WATTS)")
        print("=" * 78)

        for model_key, target_col, current_col, desc in [
            ('r1_reg_next', 'power1_1h_actual', 'power1', 'Room 1 1-Hour Ahead Power Regressor (Watts)'),
            ('r2_reg_next', 'power2_1h_actual', 'power2', 'Room 2 1-Hour Ahead Power Regressor (Watts)')
        ]:
            y_train = train_df[target_col]
            y_test = test_df[target_col]

            reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            rf_mae = mean_absolute_error(y_test, y_pred)
            rf_rmse = root_mean_squared_error(y_test, y_pred)
            rf_r2 = r2_score(y_test, y_pred)

            y_persist = test_df[current_col]
            p_mae = mean_absolute_error(y_test, y_persist)
            p_rmse = root_mean_squared_error(y_test, y_persist)
            p_r2 = r2_score(y_test, y_persist)

            self.models[model_key] = reg
            joblib.dump(reg, os.path.join(MODEL_DIR, f"{model_key}_rf.pkl"))

            print(f"\n[{desc}]")
            print(f"  Random Forest Model   : MAE = {rf_mae:7.2f} W | RMSE = {rf_rmse:7.2f} W | R2 = {rf_r2:7.4f}")
            print(f"  Persistence Baseline  : MAE = {p_mae:7.2f} W | RMSE = {p_rmse:7.2f} W | R2 = {p_r2:7.4f}")

        print("\n[*] Only the 4 authorized 1-hour ahead models were retrained and saved in saved_models/")
        print("[*] Tomorrow models (r1_tomorrow_rf.pkl, r2_tomorrow_rf.pkl, tomorrow_thresholds.json) remained UNTOUCHED.")

def main():
    print("[*] Loading and sanitizing dataset...")
    df_sorted, counts = load_and_clean_data()
    print(f"[*] Raw rows: {counts['raw_csv_rows']} | Sorted rows: {counts['rows_after_sorting']}")
    print(f"[*] Time span: {counts['start_timestamp']} to {counts['end_timestamp']}")

    print("[*] Constructing timestamp-based 1-hour and 24-hour targets...")
    df_prepared = prepare_features_and_targets(df_sorted)
    print(f"[*] Prepared rows with valid targets: {len(df_prepared)}")

    advisor = EnergyPredictionAdvisor()
    advisor.retrain_one_hour_models(df_prepared)

if __name__ == "__main__":
    main()
