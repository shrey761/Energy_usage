#!/usr/bin/env python3
"""
VOLTIX — Production Firebase ML Bridge (Enhanced Random Forest Integration)
Script: firebase_ml_bridge.py
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
SCHEMA_PATH = os.path.join(MODEL_DIR, "enhanced_features.json")

FIREBASE_DB_URL = "https://voltix-1bac0-default-rtdb.asia-southeast1.firebasedatabase.app"
READINGS_PATH = "/sensor_logs/esp32_01/readings"
PREDICTIONS_PATH = "/sensor_logs/esp32_01/predictions/latest"
ANALYTICS_PATH = "/sensor_logs/esp32_01/analytics/latest"
OPTIMIZATION_PATH = "/sensor_logs/esp32_01/optimization/latest"
NOTIFICATIONS_PATH = "/sensor_logs/esp32_01/notifications/latest"
ALERTS_PATH = "/sensor_logs/esp32_01/alerts/latest"

# 23 Baseline Features (for tomorrow and regression models)
BASELINE_FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

FEATURE_COLS = BASELINE_FEATURE_COLS  # Backwards compatibility

def load_enhanced_feature_schema():
    if os.path.exists(SCHEMA_PATH):
        try:
            with open(SCHEMA_PATH, 'r') as f:
                return json.load(f)['features']
        except Exception:
            pass
    return []

ENHANCED_FEATURE_COLS = load_enhanced_feature_schema()
MIN_HISTORY_REQUIRED = 360  # 360 readings @ 10s = 1 hour

THRESHOLD_R1 = 40.0
THRESHOLD_R2 = 10.0

# -------------------------------------------------------------------------
# 1. FIREBASE SECURE CLIENT
# -------------------------------------------------------------------------
class FirebaseClient:
    def __init__(self, db_url=FIREBASE_DB_URL):
        self.db_url = db_url.rstrip('/')
        self.admin_app = None
        self._init_admin_if_available()

    def _init_admin_if_available(self):
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if cred_path and os.path.exists(cred_path):
            try:
                import firebase_admin
                from firebase_admin import credentials, db
                cred = credentials.Certificate(cred_path)
                self.admin_app = firebase_admin.initialize_app(cred, {'databaseURL': self.db_url})
                print("[*] Firebase Admin SDK authenticated successfully via service account.")
            except Exception as e:
                print(f"[!] Note: Firebase Admin SDK initialization skipped ({e}), using HTTPS REST client.")

    def _rest_request(self, path, method='GET', payload=None, retries=3):
        if '?' in path:
            base_p, q_p = path.split('?', 1)
            url = f"{self.db_url}{base_p}.json?{q_p}"
        else:
            url = f"{self.db_url}{path}.json"
        
        for attempt in range(retries):
            try:
                data_bytes = json.dumps(payload).encode('utf-8') if payload is not None else None
                req = urllib.request.Request(url, data=data_bytes, method=method, headers={'Content-Type': 'application/json'})
                
                with urllib.request.urlopen(req, timeout=15) as response:
                    res_body = response.read().decode('utf-8')
                    return json.loads(res_body) if res_body and res_body != 'null' else None
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8') if e.fp else ""
                if attempt == retries - 1:
                    print(f"[!] Firebase {method} HTTPError on {path}: {e.code} {e.reason} - {err_body}")
                time.sleep(1.0)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[!] Firebase {method} error on {path}: {e}")
                time.sleep(1.0)
        return None

    def get(self, path, query_params=""):
        full_path = f"{path}?{query_params}" if query_params else path
        return self._rest_request(full_path, method='GET')

    def put(self, path, payload):
        return self._rest_request(path, method='PUT', payload=payload)

    def patch(self, path, payload):
        return self._rest_request(path, method='PATCH', payload=payload)

# -------------------------------------------------------------------------
# 2. AI ML INFERENCE & ADVISORY ENGINE
# -------------------------------------------------------------------------
class EnergyMLInferenceEngine:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = self._load_models()
        self.thresholds = self._load_thresholds()

    def _load_models(self):
        models = {}
        model_names = [
            'r1_next_hour', 'r2_next_hour', 
            'r1_tomorrow', 'r2_tomorrow', 
            'r1_reg_next', 'r2_reg_next'
        ]
        for name in model_names:
            path = os.path.join(self.model_dir, f"{name}_rf.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Trained model '{path}' not found.")
            models[name] = joblib.load(path)
        return models

    def _load_thresholds(self):
        path = os.path.join(self.model_dir, "tomorrow_thresholds.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'r1': 0.40, 'r2': 0.40}

    def prepare_baseline_features(self, latest_reading, history_df=None):
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

    def prepare_enhanced_features(self, history_df):
        """
        Constructs the exact 126-feature vector from historical telemetry.
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
                if c in ['energy1', 'energy2', 'totalEnergy']:
                    df[c] = 0.0
                else:
                    raise ValueError(f"Missing required telemetry field: '{c}'")
                
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
        
        if len(ENHANCED_FEATURE_COLS) > 0:
            generated = [c for c in ENHANCED_FEATURE_COLS if c in latest_row.columns]
            if len(generated) != 126 or generated != ENHANCED_FEATURE_COLS:
                raise ValueError(f"Feature schema mismatch! Expected 126 features. Found {len(generated)}")
            X_enh = latest_row[ENHANCED_FEATURE_COLS]
        else:
            X_enh = latest_row
            
        return X_enh, latest_ts, float(latest_row['power1'].values[0]), float(latest_row['power2'].values[0])

    def predict(self, latest_reading, history_df=None):
        X_base, dt, p1, p2, tp = self.prepare_baseline_features(latest_reading, history_df)
        
        # Determine Enhanced features vector
        if history_df is not None and len(history_df) >= MIN_HISTORY_REQUIRED:
            X_enh, _, _, _ = self.prepare_enhanced_features(history_df)
        else:
            # Steady-state baseline fallback for isolated single reading invocation
            fallback_hist = pd.DataFrame([latest_reading] * MIN_HISTORY_REQUIRED)
            fallback_hist['timestamp'] = [datetime.now() - pd.Timedelta(seconds=(MIN_HISTORY_REQUIRED-1-i)*10) for i in range(MIN_HISTORY_REQUIRED)]
            X_enh, _, _, _ = self.prepare_enhanced_features(fallback_hist)

        # 1. Room 1 Predictions (Enhanced RF for 1-Hour)
        r1_next_probas = self.models['r1_next_hour'].predict_proba(X_enh)[0]
        r1_next_state = "HIGH USAGE" if r1_next_probas[1] >= 0.5 else "NORMAL"
        r1_next_conf = float(r1_next_probas[1 if r1_next_state == "HIGH USAGE" else 0])

        # 1. Room 1 Tomorrow (Baseline RF for 24-Hour)
        th_r1 = self.thresholds.get('r1', 0.40)
        r1_tmrw_probas = self.models['r1_tomorrow'].predict_proba(X_base)[0]
        r1_tmrw_state = "HIGH USAGE" if r1_tmrw_probas[1] >= th_r1 else "NORMAL"
        r1_tmrw_conf = float(r1_tmrw_probas[1 if r1_tmrw_state == "HIGH USAGE" else 0])

        r1_pred_power = max(0.0, float(self.models['r1_reg_next'].predict(X_base)[0]))

        # 2. Room 2 Predictions (Enhanced RF for 1-Hour)
        r2_next_probas = self.models['r2_next_hour'].predict_proba(X_enh)[0]
        r2_next_state = "HIGH USAGE" if r2_next_probas[1] >= 0.5 else "NORMAL"
        r2_next_conf = float(r2_next_probas[1 if r2_next_state == "HIGH USAGE" else 0])

        # 2. Room 2 Tomorrow (Baseline RF for 24-Hour)
        th_r2 = self.thresholds.get('r2', 0.40)
        r2_tmrw_probas = self.models['r2_tomorrow'].predict_proba(X_base)[0]
        r2_tmrw_state = "HIGH USAGE" if r2_tmrw_probas[1] >= th_r2 else "NORMAL"
        r2_tmrw_conf = float(r2_tmrw_probas[1 if r2_tmrw_state == "HIGH USAGE" else 0])

        r2_pred_power = max(0.0, float(self.models['r2_reg_next'].predict(X_base)[0]))

        # Recommendations
        r1_recs = []
        if r1_next_state == "HIGH USAGE":
            r1_recs.append("Reduce non-essential loads in Room 1 during the next hour.")
        if r1_tmrw_state == "HIGH USAGE":
            r1_recs.append("Consider shifting flexible loads in Room 1 away from this period tomorrow.")
        if not r1_recs:
            r1_recs.append("Room 1 usage is normal. No load reduction required.")

        r2_recs = []
        if r2_next_state == "HIGH USAGE":
            r2_recs.append("Reduce non-essential loads in Room 2 during the next hour.")
        if r2_tmrw_state == "HIGH USAGE":
            r2_recs.append("Consider shifting flexible loads in Room 2 away from this period tomorrow.")
        if not r2_recs:
            r2_recs.append("Room 2 usage is normal. No load reduction required.")

        return {
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "room1": {
                "currentPower": round(p1, 1),
                "predictedPower": round(r1_pred_power, 1),
                "nextHour": {
                    "state": r1_next_state,
                    "confidence": round(r1_next_conf, 4),
                    "confidencePercent": round(r1_next_conf * 100, 1)
                },
                "tomorrow": {
                    "state": r1_tmrw_state,
                    "confidence": round(r1_tmrw_conf, 4),
                    "confidencePercent": round(r1_tmrw_conf * 100, 1)
                },
                "recommendations": r1_recs
            },
            "room2": {
                "currentPower": round(p2, 1),
                "predictedPower": round(r2_pred_power, 1),
                "nextHour": {
                    "state": r2_next_state,
                    "confidence": round(r2_next_conf, 4),
                    "confidencePercent": round(r2_next_conf * 100, 1)
                },
                "tomorrow": {
                    "state": r2_tmrw_state,
                    "confidence": round(r2_tmrw_conf, 4),
                    "confidencePercent": round(r2_tmrw_conf * 100, 1)
                },
                "recommendations": r2_recs
            },
            "summary": {
                "totalCurrentPower": round(tp, 1),
                "totalPredictedPower": round(r1_pred_power + r2_pred_power, 1),
                "overallNextHourState": "HIGH USAGE" if (r1_next_state == "HIGH USAGE" or r2_next_state == "HIGH USAGE") else "NORMAL"
            }
        }

# -------------------------------------------------------------------------
# 3. ENERGY ANALYTICS & OPTIMIZATION MODULE
# -------------------------------------------------------------------------
class EnergyAnalyticsOptimizer:
    def __init__(self, cost_per_kwh=8.0):
        self.cost_per_kwh = cost_per_kwh

    @staticmethod
    def _compute_delta_reset_safe(df, col):
        if col not in df.columns or len(df) == 0:
            return 0.0
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
        if len(vals) < 2:
            return 0.0
        diffs = np.diff(vals)
        pos_diffs = np.where(diffs >= 0, diffs, vals[1:])
        return float(np.sum(pos_diffs))

    def compute_historical_analytics(self, readings_df):
        if readings_df is None or len(readings_df) == 0:
            return None

        df = readings_df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            now = df['timestamp'].iloc[-1]
        else:
            now = datetime.now()

        # Day, week, month windows
        d_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        w_start = d_start - timedelta(days=now.weekday())
        m_start = d_start.replace(day=1)

        df_day = df[df['timestamp'] >= d_start] if 'timestamp' in df.columns else df
        df_week = df[df['timestamp'] >= w_start] if 'timestamp' in df.columns else df
        df_month = df[df['timestamp'] >= m_start] if 'timestamp' in df.columns else df

        e_day = self._compute_delta_reset_safe(df_day, 'totalEnergy')
        e_week = self._compute_delta_reset_safe(df_week, 'totalEnergy')
        e_month = self._compute_delta_reset_safe(df_month, 'totalEnergy')

        if e_day == 0.0 and 'totalPower' in df_day.columns and len(df_day) > 1:
            e_day = float(df_day['totalPower'].mean() * (len(df_day) * 10.0 / 3600.0) / 1000.0)
        if e_week == 0.0 and 'totalPower' in df_week.columns and len(df_week) > 1:
            e_week = float(df_week['totalPower'].mean() * (len(df_week) * 10.0 / 3600.0) / 1000.0)
        if e_month == 0.0 and 'totalPower' in df_month.columns and len(df_month) > 1:
            e_month = float(df_month['totalPower'].mean() * (len(df_month) * 10.0 / 3600.0) / 1000.0)

        # Baseline projection
        elapsed_days = max(1, now.day)
        days_in_month = 30
        daily_avg = e_month / elapsed_days if e_month > 0 else (e_day if e_day > 0 else 0.5)
        projected_monthly_kwh = round(daily_avg * days_in_month, 2)
        projected_cost = round(projected_monthly_kwh * self.cost_per_kwh, 2)

        return {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "dailyKwh": round(e_day, 3),
            "dailyCost": round(e_day * self.cost_per_kwh, 2),
            "weeklyKwh": round(e_week, 3),
            "weeklyCost": round(e_week * self.cost_per_kwh, 2),
            "monthlyKwh": round(e_month, 3),
            "monthlyCost": round(e_month * self.cost_per_kwh, 2),
            "projectedMonthlyKwh": projected_monthly_kwh,
            "projectedMonthlyCost": projected_cost
        }

    def generate_optimization_advisories(self, predictions_dict, analytics_dict):
        if not predictions_dict:
            return None

        r1 = predictions_dict.get('room1', {})
        r2 = predictions_dict.get('room2', {})
        summary = predictions_dict.get('summary', {})

        active_loads = []
        if r1.get('nextHour', {}).get('state') == 'HIGH USAGE':
            active_loads.append("Room 1 Lighting/Appliances (40W+)")
        if r2.get('nextHour', {}).get('state') == 'HIGH USAGE':
            active_loads.append("Room 2 Lighting/Appliances (10W+)")

        optimizations = []
        if active_loads:
            optimizations.append(f"Shift flexible loads away from next hour: {', '.join(active_loads)}")
        else:
            optimizations.append("Current load profile is optimal. No action needed.")

        target_monthly_kwh = 25.0
        current_proj = analytics_dict.get('projectedMonthlyKwh', 0.0) if analytics_dict else 0.0
        savings_kwh = max(0.0, current_proj - target_monthly_kwh)
        savings_cost = round(savings_kwh * self.cost_per_kwh, 2)

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "targetMonthlyKwh": target_monthly_kwh,
            "potentialSavingsKwh": round(savings_kwh, 2),
            "potentialSavingsCost": savings_cost,
            "recommendedActions": optimizations,
            "peakAlert": summary.get('overallNextHourState') == 'HIGH USAGE'
        }

    def generate_notifications(self, predictions_dict, analytics_dict):
        notifications = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        r1_st = predictions_dict.get('room1', {}).get('nextHour', {}).get('state')
        r2_st = predictions_dict.get('room2', {}).get('nextHour', {}).get('state')

        if r1_st == "HIGH USAGE":
            notifications.append({
                "id": f"notif_{int(time.time())}_r1",
                "title": "Room 1 Peak Load Warning",
                "message": "Room 1 load is predicted to exceed 40W in the next hour.",
                "type": "WARNING",
                "timestamp": now_str
            })
        if r2_st == "HIGH USAGE":
            notifications.append({
                "id": f"notif_{int(time.time())}_r2",
                "title": "Room 2 Peak Load Warning",
                "message": "Room 2 load is predicted to exceed 10W in the next hour.",
                "type": "WARNING",
                "timestamp": now_str
            })
        if not notifications:
            notifications.append({
                "id": f"notif_{int(time.time())}_ok",
                "title": "Energy Usage Optimal",
                "message": "All rooms are operating within normal baseline limits.",
                "type": "INFO",
                "timestamp": now_str
            })
        return {
            "timestamp": now_str,
            "items": notifications
        }

# -------------------------------------------------------------------------
# 4. BRIDGE RUNNER
# -------------------------------------------------------------------------
class VoltixProductionBridge:
    def __init__(self, mock_data_path=None):
        self.fb = FirebaseClient(FIREBASE_DB_URL)
        self.engine = EnergyMLInferenceEngine(MODEL_DIR)
        self.analytics = EnergyAnalyticsOptimizer()
        self.mock_data_path = mock_data_path

    def fetch_readings(self, limit=400):
        if self.mock_data_path:
            if os.path.exists(self.mock_data_path):
                df = pd.read_csv(self.mock_data_path)
                return df.tail(limit).reset_index(drop=True)
            return None
        
        # Primary query: orderBy="$key" matching Firebase RTDB indexed key order and Flutter orderByKey()
        data = self.fb.get(READINGS_PATH, f'orderBy="$key"&limitToLast={limit}')
        
        # Fallback query: orderBy="timestamp" if custom index is configured
        if not data:
            data = self.fb.get(READINGS_PATH, f'orderBy="timestamp"&limitToLast={limit}')

        if not data:
            return None
        if isinstance(data, dict):
            records = list(data.values())
        elif isinstance(data, list):
            records = [r for r in data if r is not None]
        else:
            return None
        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        return df

    def sync_once(self):
        print("=" * 80)
        print(" VOLTIX PRODUCTION FIREBASE ML BRIDGE (ENHANCED RF)")
        print("=" * 80)
        
        df = self.fetch_readings(limit=400)
        if df is None or len(df) == 0:
            print("[!] No readings retrieved from Firebase.")
            return False
            
        print(f"[*] Retrieved {len(df)} readings from telemetry stream.")
        
        if len(df) < MIN_HISTORY_REQUIRED:
            print(f"[!] Insufficient historical data for Enhanced RF prediction. Found: {len(df)}, Required: {MIN_HISTORY_REQUIRED}")
            return False
            
        latest_reading = df.iloc[-1].to_dict()
        
        # ML Inference
        preds = self.engine.predict(latest_reading, df)
        analytics_res = self.analytics.compute_historical_analytics(df)
        opt_res = self.analytics.generate_optimization_advisories(preds, analytics_res)
        notif_res = self.analytics.generate_notifications(preds, analytics_res)
        
        print(f"[*] Predictions Generated for Timestamp: {preds['timestamp']}")
        print(f"  Room 1 Next Hour: {preds['room1']['nextHour']['state']} ({preds['room1']['nextHour']['confidencePercent']}%)")
        print(f"  Room 2 Next Hour: {preds['room2']['nextHour']['state']} ({preds['room2']['nextHour']['confidencePercent']}%)")
        
        # Publish to Firebase
        if not self.mock_data_path:
            self.fb.put(PREDICTIONS_PATH, preds)
            if analytics_res: self.fb.put(ANALYTICS_PATH, analytics_res)
            if opt_res: self.fb.put(OPTIMIZATION_PATH, opt_res)
            if notif_res: self.fb.put(NOTIFICATIONS_PATH, notif_res)
            print("[+] Successfully synced predictions, analytics, optimization & notifications to Firebase!")
            
        return True

def main():
    parser = argparse.ArgumentParser(description="VOLTIX Production ML Bridge")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--mock", action="store_true", help="Use local realistic dataset as mock stream")
    args = parser.parse_args()
    
    mock_file = os.path.join(BASE_DIR, "VOLTIX_realistic_30day_synthetic_dataset.csv") if args.mock else None
    bridge = VoltixProductionBridge(mock_data_path=mock_file)
    bridge.sync_once()

if __name__ == '__main__':
    main()
