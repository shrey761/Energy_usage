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

# 23 Baseline / Production Features matching trained Random Forest models
FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

BASELINE_FEATURE_COLS = FEATURE_COLS
MIN_HISTORY_REQUIRED = 360  # Minimum historical telemetry records

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
        self._validate_startup_feature_schema()

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

    def _validate_startup_feature_schema(self):
        """
        Validates at startup that all loaded Random Forest models match FEATURE_COLS exactly.
        """
        print("=" * 80)
        print(" VOLTIX ML FEATURE SCHEMA STARTUP VALIDATION")
        print("=" * 80)
        print(f"Target Feature Schema ({len(FEATURE_COLS)} features):")
        print(f"  {FEATURE_COLS}\n")

        all_valid = True
        for model_name, model in self.models.items():
            expected = getattr(model, 'feature_names_in_', None)
            if expected is None:
                print(f"  [?] Model '{model_name}': No feature_names_in_ metadata.")
                continue

            expected_list = list(expected)
            if expected_list == FEATURE_COLS:
                print(f"  [+] Model '{model_name}': 100% SCHEMA MATCH ({len(expected_list)}/{len(FEATURE_COLS)} features identical)")
            else:
                all_valid = False
                missing = [f for f in expected_list if f not in FEATURE_COLS]
                extra = [f for f in FEATURE_COLS if f not in expected_list]
                print(f"  [!] Model '{model_name}': SCHEMA MISMATCH!")
                if missing: print(f"      Missing from inference schema: {missing}")
                if extra: print(f"      Extra in inference schema: {extra}")

        if not all_valid:
            raise ValueError("Startup Feature Schema Validation Failed: Model schema mismatch!")
        print("=" * 80 + "\n")

    def prepare_features(self, latest_reading, history_df=None):
        ts = latest_reading.get('timestamp', None)
        if isinstance(ts, str):
            ts_str = ts.strip()
            if ts_str in ['TIME_ERROR', 'null', 'None', '', 'undefined']:
                dt = datetime.now()
            else:
                try:
                    parsed_ts = pd.to_datetime(ts_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    if pd.isna(parsed_ts):
                        dt = datetime.now()
                    else:
                        dt = parsed_ts.to_pydatetime()
                except Exception:
                    dt = datetime.now()
        elif isinstance(ts, (datetime, pd.Timestamp)):
            dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        else:
            dt = datetime.now()

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

        # Calculate lag and rolling features from historical telemetry if available
        if history_df is not None and len(history_df) >= 1:
            hist = history_df.copy()
            if 'timestamp' in hist.columns:
                if not pd.api.types.is_datetime64_any_dtype(hist['timestamp']):
                    hist = hist[hist['timestamp'].notna()]
                    hist = hist[~hist['timestamp'].astype(str).str.strip().isin(['TIME_ERROR', 'null', 'None', '', 'undefined'])]
                    hist['timestamp'] = pd.to_datetime(hist['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                hist = hist.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

            if len(hist) >= 1:
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
        return pd.DataFrame([row_dict])[FEATURE_COLS], dt, p1, p2, tp

    def prepare_baseline_features(self, latest_reading, history_df=None):
        return self.prepare_features(latest_reading, history_df)

    def predict(self, latest_reading, history_df=None):
        X_feats, dt, p1, p2, tp = self.prepare_features(latest_reading, history_df)
        
        # Verify schema match before predicting
        if list(X_feats.columns) != FEATURE_COLS:
            raise ValueError(f"Feature schema mismatch! Expected {FEATURE_COLS}, got {list(X_feats.columns)}")

        # 1. Room 1 Predictions (1-Hour Ahead Classifier)
        r1_next_probas = self.models['r1_next_hour'].predict_proba(X_feats)[0]
        r1_next_state = "HIGH USAGE" if r1_next_probas[1] >= 0.50 else "NORMAL"
        r1_next_conf = float(r1_next_probas[1 if r1_next_state == "HIGH USAGE" else 0])

        # 2. Room 1 Tomorrow Predictions (24-Hour Ahead Classifier with calibrated threshold)
        th_r1 = self.thresholds.get('r1', 0.40)
        r1_tmrw_probas = self.models['r1_tomorrow'].predict_proba(X_feats)[0]
        r1_tmrw_state = "HIGH USAGE" if r1_tmrw_probas[1] >= th_r1 else "NORMAL"
        r1_tmrw_conf = float(r1_tmrw_probas[1 if r1_tmrw_state == "HIGH USAGE" else 0])

        # 3. Room 1 1-Hour Ahead Continuous Power Regressor
        r1_pred_power = max(0.0, float(self.models['r1_reg_next'].predict(X_feats)[0]))

        # 4. Room 2 Predictions (1-Hour Ahead Classifier)
        r2_next_probas = self.models['r2_next_hour'].predict_proba(X_feats)[0]
        r2_next_state = "HIGH USAGE" if r2_next_probas[1] >= 0.50 else "NORMAL"
        r2_next_conf = float(r2_next_probas[1 if r2_next_state == "HIGH USAGE" else 0])

        # 5. Room 2 Tomorrow Predictions (24-Hour Ahead Classifier with calibrated threshold)
        th_r2 = self.thresholds.get('r2', 0.40)
        r2_tmrw_probas = self.models['r2_tomorrow'].predict_proba(X_feats)[0]
        r2_tmrw_state = "HIGH USAGE" if r2_tmrw_probas[1] >= th_r2 else "NORMAL"
        r2_tmrw_conf = float(r2_tmrw_probas[1 if r2_tmrw_state == "HIGH USAGE" else 0])

        # 6. Room 2 1-Hour Ahead Continuous Power Regressor
        r2_pred_power = max(0.0, float(self.models['r2_reg_next'].predict(X_feats)[0]))

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
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df[df['timestamp'].notna()]
                df = df[~df['timestamp'].astype(str).str.strip().isin(['TIME_ERROR', 'null', 'None', '', 'undefined'])]
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            if len(df) > 0:
                now = df['timestamp'].iloc[-1]
            else:
                now = datetime.now()
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

        # Hourly load curves for Room 1 and Room 2 (24 hours)
        hourly_r1 = [0.0] * 24
        hourly_r2 = [0.0] * 24
        if 'timestamp' in df.columns:
            df['hour_idx'] = df['timestamp'].dt.hour
            for h in range(24):
                h_df = df[df['hour_idx'] == h]
                if len(h_df) > 0:
                    hourly_r1[h] = round(float(h_df['power1'].mean()), 1)
                    hourly_r2[h] = round(float(h_df['power2'].mean()), 1)

        # 7-day daily bars for current week
        day_names = ['M', 'T', 'W', 'T', 'F', 'S', 'S']
        daily_bars = []
        for i in range(7):
            b_date = (w_start + timedelta(days=i)).date()
            b_df = df[df['timestamp'].dt.date == b_date] if 'timestamp' in df.columns else pd.DataFrame()
            b_units = self._compute_delta_reset_safe(b_df, 'totalEnergy')
            if b_units == 0.0 and 'totalPower' in b_df.columns and len(b_df) > 1:
                b_units = float(b_df['totalPower'].mean() * (len(b_df) * 10.0 / 3600.0) / 1000.0)
            daily_bars.append({
                "dayIndex": i,
                "dayName": day_names[i],
                "date": b_date.strftime("%Y-%m-%d"),
                "units": round(b_units, 3)
            })

        # Room-wise monthly energy split
        r1_month_e = self._compute_delta_reset_safe(df_month, 'energy1')
        if r1_month_e == 0.0 and 'power1' in df_month.columns and len(df_month) > 1:
            r1_month_e = float(df_month['power1'].mean() * (len(df_month) * 10.0 / 3600.0) / 1000.0)
        r2_month_e = self._compute_delta_reset_safe(df_month, 'energy2')
        if r2_month_e == 0.0 and 'power2' in df_month.columns and len(df_month) > 1:
            r2_month_e = float(df_month['power2'].mean() * (len(df_month) * 10.0 / 3600.0) / 1000.0)

        # Baseline projection
        elapsed_days = max(1, now.day)
        days_in_month = 30
        daily_avg = e_month / elapsed_days if e_month > 0 else (e_day if e_day > 0 else 0.5)
        projected_monthly_kwh = round(daily_avg * days_in_month, 2)
        projected_cost = round(projected_monthly_kwh * self.cost_per_kwh, 2)

        return {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "dailyKwh": round(e_day, 3),
            "todayUnits": round(e_day, 3),
            "dailyCost": round(e_day * self.cost_per_kwh, 2),
            "weeklyKwh": round(e_week, 3),
            "weekUnits": round(e_week, 3),
            "weeklyTotalUnits": round(e_week, 3),
            "weeklyCost": round(e_week * self.cost_per_kwh, 2),
            "monthlyKwh": round(e_month, 3),
            "monthToDateUnits": round(e_month, 3),
            "monthlyCost": round(e_month * self.cost_per_kwh, 2),
            "projectedMonthlyKwh": projected_monthly_kwh,
            "projectedMonthEndUnits": projected_monthly_kwh,
            "projectedMonthlyCost": projected_cost,
            "averageDailyUnits": round(daily_avg, 3),
            "allocatedUnits": 96.0,
            "remainingUnits": round(max(0.0, 96.0 - e_month), 3),
            "projectedExcessUnits": round(max(0.0, projected_monthly_kwh - 96.0), 3),
            "usagePercentage": round(e_month / 96.0, 4),
            "daysElapsed": elapsed_days,
            "daysInMonth": days_in_month,
            "dailyBars": daily_bars,
            "hourlyLoadRoom1": hourly_r1,
            "hourlyLoadRoom2": hourly_r2,
            "room1MonthUnits": round(r1_month_e, 3),
            "room2MonthUnits": round(r2_month_e, 3)
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

        r1_units = analytics_dict.get('room1MonthUnits', 0.0) if analytics_dict else 0.0
        r2_units = analytics_dict.get('room2MonthUnits', 0.0) if analytics_dict else 0.0
        tot_units = r1_units + r2_units
        r1_pct = round((r1_units / tot_units) * 100.0, 1) if tot_units > 0 else 50.0
        r2_pct = round((r2_units / tot_units) * 100.0, 1) if tot_units > 0 else 50.0

        suggestions = []
        if r1_units >= r2_units:
            suggestions.append({
                "title": "Room 1 is the primary energy contributor",
                "subtitle": "Inspect Room 1 incandescent lighting/appliances (40W+) during active hours.",
                "icon": "warning"
            })
        else:
            suggestions.append({
                "title": "Room 2 is the primary energy contributor",
                "subtitle": "Inspect Room 2 lighting/appliances (10W+) during active hours.",
                "icon": "warning"
            })
            
        if summary.get('overallNextHourState') == 'HIGH USAGE':
            suggestions.append({
                "title": "Shift flexible loads away from next hour",
                "subtitle": "High usage predicted next hour. Defer non-essential equipment.",
                "icon": "access_time"
            })
        else:
            suggestions.append({
                "title": "Optimal load profile active",
                "subtitle": "All rooms operating within normal limits.",
                "icon": "bolt"
            })
            
        suggestions.append({
            "title": "Stay within Gruha Jyothi quota",
            "subtitle": f"Target daily average under {round(96.0 / 30.0, 1)} units/day.",
            "icon": "energy_savings_leaf"
        })

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "targetMonthlyKwh": target_monthly_kwh,
            "potentialSavingsKwh": round(savings_kwh, 2),
            "estimatedSavingsUnits": round(savings_kwh, 2),
            "potentialSavingsCost": savings_cost,
            "recommendedActions": optimizations,
            "suggestions": suggestions,
            "room1Units": r1_units,
            "room2Units": r2_units,
            "room1Percent": r1_pct,
            "room2Percent": r2_pct,
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
                "subtitle": "Room 1 load is predicted to exceed 40W in the next hour.",
                "message": "Room 1 load is predicted to exceed 40W in the next hour.",
                "type": "prediction",
                "priority": "HIGH",
                "timestamp": now_str
            })
        if r2_st == "HIGH USAGE":
            notifications.append({
                "id": f"notif_{int(time.time())}_r2",
                "title": "Room 2 Peak Load Warning",
                "subtitle": "Room 2 load is predicted to exceed 10W in the next hour.",
                "message": "Room 2 load is predicted to exceed 10W in the next hour.",
                "type": "prediction",
                "priority": "HIGH",
                "timestamp": now_str
            })
        if not notifications:
            notifications.append({
                "id": f"notif_{int(time.time())}_ok",
                "title": "Energy Usage Optimal",
                "subtitle": "All rooms are operating within normal baseline limits.",
                "message": "All rooms are operating within normal baseline limits.",
                "type": "optimal",
                "priority": "LOW",
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
                if 'timestamp' in df.columns:
                    df = df[df['timestamp'].notna()]
                    df = df[~df['timestamp'].astype(str).str.strip().isin(['TIME_ERROR', 'null', 'None', '', 'undefined'])]
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    df = df.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
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
            # 6. Remove/ignore invalid timestamp values before parsing (TIME_ERROR, empty/null, malformed)
            df = df[df['timestamp'].notna()]
            df = df[~df['timestamp'].astype(str).str.strip().isin(['TIME_ERROR', 'null', 'None', '', 'undefined'])]
            # 7. Parse valid timestamps explicitly using format='%Y-%m-%d %H:%M:%S'
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            # 8. Drop rows where timestamp became NaT
            # 9. Sort valid records chronologically by timestamp
            df = df.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        return df

    def sync_once(self, force=True):
        df = self.fetch_readings(limit=400)
        now_time = datetime.now().strftime('%H:%M:%S')
        if df is None or len(df) == 0:
            print(f"[{now_time}] [!] No readings retrieved from Firebase.")
            return False, None
            
        if len(df) < MIN_HISTORY_REQUIRED:
            print(f"[{now_time}] [!] Insufficient historical data for Enhanced RF prediction. Found: {len(df)}, Required: {MIN_HISTORY_REQUIRED}")
            return False, None
            
        latest_reading = df.iloc[-1].to_dict()
        latest_ts = str(latest_reading.get('timestamp', ''))

        # Avoid duplicate prediction & write cycles if telemetry is unchanged
        if not force and hasattr(self, '_last_synced_ts') and self._last_synced_ts == latest_ts:
            return False, latest_ts

        # ML Inference
        preds = self.engine.predict(latest_reading, df)
        analytics_res = self.analytics.compute_historical_analytics(df)
        opt_res = self.analytics.generate_optimization_advisories(preds, analytics_res)
        notif_res = self.analytics.generate_notifications(preds, analytics_res)
        
        print(f"\n[{now_time}] [*] Telemetry Updated (Timestamp: {preds['timestamp']}, Window: {len(df)} readings)")
        print(f"  Room 1 (Measured: {preds['room1']['currentPower']}W -> Predicted: {preds['room1']['predictedPower']}W) | Next Hour: {preds['room1']['nextHour']['state']} ({preds['room1']['nextHour']['confidencePercent']}%) | Tomorrow: {preds['room1']['tomorrow']['state']}")
        print(f"  Room 2 (Measured: {preds['room2']['currentPower']}W -> Predicted: {preds['room2']['predictedPower']}W) | Next Hour: {preds['room2']['nextHour']['state']} ({preds['room2']['nextHour']['confidencePercent']}%) | Tomorrow: {preds['room2']['tomorrow']['state']}")
        print(f"  Total Load: {preds['summary']['totalCurrentPower']}W -> Predicted: {preds['summary']['totalPredictedPower']}W | Overall: {preds['summary']['overallNextHourState']}")
        
        # Publish to Firebase
        if not self.mock_data_path:
            self.fb.put(PREDICTIONS_PATH, preds)
            if analytics_res: self.fb.put(ANALYTICS_PATH, analytics_res)
            if opt_res: self.fb.put(OPTIMIZATION_PATH, opt_res)
            if notif_res: self.fb.put(NOTIFICATIONS_PATH, notif_res)
            print("  [+] Successfully synced predictions, analytics, optimization & notifications to Firebase!\n")
            
        self._last_synced_ts = latest_ts
        return True, latest_ts

    def run_loop(self, poll_interval=10, max_cycles=None):
        """
        Continuous live inference loop.
        Monitors telemetry every `poll_interval` seconds, generates 126 causal features,
        and pushes predictions to Firebase in real time.
        """
        print("=" * 80)
        print(" VOLTIX PRODUCTION FIREBASE ML BRIDGE - CONTINUOUS INFERENCE LOOP")
        print(f" Telemetry Stream: {READINGS_PATH}")
        print(f" Poll Interval: {poll_interval}s | Minimum History: {MIN_HISTORY_REQUIRED} readings")
        print(" Press Ctrl+C to stop.")
        print("=" * 80)

        cycles = 0
        try:
            while True:
                try:
                    synced, latest_ts = self.sync_once(force=(cycles == 0))
                    if not synced and latest_ts is not None:
                        now_time = datetime.now().strftime('%H:%M:%S')
                        print(f"[{now_time}] [*] Telemetry unchanged ({latest_ts}). Waiting for next ESP32 update...")
                except Exception as e:
                    now_time = datetime.now().strftime('%H:%M:%S')
                    print(f"[{now_time}] [!] Error during inference cycle: {e}. Retrying in {poll_interval}s...")

                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    print(f"\n[*] Completed requested {max_cycles} cycle(s).")
                    break

                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[!] VOLTIX ML Bridge loop terminated gracefully by user (Ctrl+C).")

def main():
    parser = argparse.ArgumentParser(description="VOLTIX Production ML Bridge")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds (default: 10)")
    parser.add_argument("--max-cycles", type=int, default=None, help="Maximum cycles to run before exiting (for testing)")
    parser.add_argument("--mock", action="store_true", help="Use local realistic dataset as mock stream")
    args = parser.parse_args()
    
    mock_file = os.path.join(BASE_DIR, "VOLTIX_realistic_30day_synthetic_dataset.csv") if args.mock else None
    bridge = VoltixProductionBridge(mock_data_path=mock_file)
    
    if args.once:
        print("=" * 80)
        print(" VOLTIX PRODUCTION FIREBASE ML BRIDGE (SINGLE CYCLE)")
        print("=" * 80)
        bridge.sync_once(force=True)
    else:
        bridge.run_loop(poll_interval=args.interval, max_cycles=args.max_cycles)

if __name__ == '__main__':
    main()
