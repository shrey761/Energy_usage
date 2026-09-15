import os
import sys
import timecd 
import json
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

FIREBASE_DB_URL = "https://voltix-1bac0-default-rtdb.asia-southeast1.firebasedatabase.app"
READINGS_PATH = "/sensor_logs/esp32_01/readings"
PREDICTIONS_PATH = "/sensor_logs/esp32_01/predictions/latest"
ANALYTICS_PATH = "/sensor_logs/esp32_01/analytics/latest"
OPTIMIZATION_PATH = "/sensor_logs/esp32_01/optimization/latest"
NOTIFICATIONS_PATH = "/sensor_logs/esp32_01/notifications/latest"
ALERTS_PATH = "/sensor_logs/esp32_01/alerts/latest"

# Exact 23 features in exact training order from energy_advisor.py
FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

# -------------------------------------------------------------------------
# 1. FIREBASE SECURE CLIENT (REST + Optional Admin SDK support)
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
                req = urllib.request.Request(url, data=data_bytes, method=method)
                if payload is not None:
                    req.add_header('Content-Type', 'application/json')
                
                with urllib.request.urlopen(req, timeout=15) as response:
                    res_body = response.read().decode('utf-8')
                    return json.loads(res_body) if res_body and res_body != 'null' else None
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
        return {'r1': 0.70, 'r2': 0.65}

    def prepare_features(self, latest_reading, history_df=None):
        """
        Constructs the exact 23-feature vector matching the training pipeline in energy_advisor.py.
        Zero future leakage: only past and current values at time t are used.
        """
        ts = latest_reading.get('timestamp', None)
        if isinstance(ts, str):
            try:
                dt = pd.to_datetime(ts)
            except Exception:
                dt = datetime.now()
        elif isinstance(ts, (datetime, pd.Timestamp)):
            dt = ts
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

        # Lags & rolling windows:
        # In training (energy_advisor.py lines 82-85):
        # - lag1 was shift(1) -> value at t-1 (iloc[-2] in chronological sequence ending with latest_reading)
        # - lag3 was shift(3) -> value at t-3 (iloc[-4] in chronological sequence)
        # - roll_mean5 was rolling(5, min_periods=1).mean() -> mean of [t-4, t-3, t-2, t-1, t] (iloc[-5:])
        # - roll_max5 was rolling(5, min_periods=1).max() -> max of [t-4, t-3, t-2, t-1, t] (iloc[-5:])
        if history_df is not None and len(history_df) >= 1:
            # Ensure history_df ends with latest_reading
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
        return pd.DataFrame([row_dict])[FEATURE_COLS], dt, p1, p2, tp

    def predict(self, latest_reading, history_df=None):
        X, dt, p1, p2, tp = self.prepare_features(latest_reading, history_df)

        # 1. Room 1 Predictions
        r1_next_probas = self.models['r1_next_hour'].predict_proba(X)[0]
        r1_next_state = "HIGH USAGE" if r1_next_probas[1] >= 0.5 else "NORMAL"
        r1_next_conf = float(r1_next_probas[1 if r1_next_state == "HIGH USAGE" else 0])

        th_r1 = self.thresholds.get('r1', 0.70)
        r1_tmrw_probas = self.models['r1_tomorrow'].predict_proba(X)[0]
        r1_tmrw_state = "HIGH USAGE" if r1_tmrw_probas[1] >= th_r1 else "NORMAL"
        r1_tmrw_conf = float(r1_tmrw_probas[1 if r1_tmrw_state == "HIGH USAGE" else 0])

        r1_pred_power = max(0.0, float(self.models['r1_reg_next'].predict(X)[0]))

        # 2. Room 2 Predictions
        r2_next_probas = self.models['r2_next_hour'].predict_proba(X)[0]
        r2_next_state = "HIGH USAGE" if r2_next_probas[1] >= 0.5 else "NORMAL"
        r2_next_conf = float(r2_next_probas[1 if r2_next_state == "HIGH USAGE" else 0])

        th_r2 = self.thresholds.get('r2', 0.65)
        r2_tmrw_probas = self.models['r2_tomorrow'].predict_proba(X)[0]
        r2_tmrw_state = "HIGH USAGE" if r2_tmrw_probas[1] >= th_r2 else "NORMAL"
        r2_tmrw_conf = float(r2_tmrw_probas[1 if r2_tmrw_state == "HIGH USAGE" else 0])

        r2_pred_power = max(0.0, float(self.models['r2_reg_next'].predict(X)[0]))

        # Explainable Contextual Recommendations
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
# 3. ENERGY AGGREGATOR, STATISTICAL MONTHLY FORECAST & OPTIMIZATION
# -------------------------------------------------------------------------
class EnergyAnalyticsOptimizer:
    GRUHA_JYOTHI_TARGET = 96.0

    @staticmethod
    def _compute_delta_reset_safe(df_subset, col_name, current_val=None):
        """
        Safely computes cumulative energy consumption across time windows,
        detecting meter rollover/resets so consumption is never negative.
        Formula: Sum of non-negative consecutive diffs.
        """
        if df_subset.empty:
            return max(0.0, float(current_val)) if current_val is not None else 0.0
        
        vals = df_subset[col_name].values
        total_delta = 0.0
        for i in range(1, len(vals)):
            diff = vals[i] - vals[i-1]
            if diff >= 0:
                total_delta += diff
            else:
                # Meter reset/rollover detected: accumulate reading after reset
                total_delta += vals[i]
        
        if current_val is not None and len(vals) > 0:
            diff_last = current_val - vals[-1]
            if diff_last >= 0:
                total_delta += diff_last
            else:
                total_delta += current_val

        return max(0.0, total_delta)

    @staticmethod
    def process_readings_history(df_hist, latest_reading):
        """
        Processes chronological historical dataframe and computes accurate deltas,
        statistical month-end projections, and optimization recommendations.
        """
        ref_time = latest_reading.get('timestamp')
        if not isinstance(ref_time, datetime):
            try:
                ref_time = pd.to_datetime(ref_time)
            except Exception:
                ref_time = datetime.now()

        current_total_energy = float(latest_reading.get('totalEnergy', 0.0))
        current_e1 = float(latest_reading.get('energy1', 0.0))
        current_e2 = float(latest_reading.get('energy2', 0.0))

        # 1. Today's consumption (reset-safe delta)
        start_of_today = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_df = df_hist[df_hist['timestamp'] >= start_of_today] if not df_hist.empty else pd.DataFrame()
        today_units = EnergyAnalyticsOptimizer._compute_delta_reset_safe(today_df, 'totalEnergy', current_total_energy)
        today_r1 = EnergyAnalyticsOptimizer._compute_delta_reset_safe(today_df, 'energy1', current_e1)
        today_r2 = EnergyAnalyticsOptimizer._compute_delta_reset_safe(today_df, 'energy2', current_e2)

        # 2. This week's consumption (reset-safe delta starting Monday)
        start_of_week = (ref_time - timedelta(days=ref_time.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        week_df = df_hist[df_hist['timestamp'] >= start_of_week] if not df_hist.empty else pd.DataFrame()
        week_units = EnergyAnalyticsOptimizer._compute_delta_reset_safe(week_df, 'totalEnergy', current_total_energy)

        # 3. This month's consumption (reset-safe delta starting 1st of month)
        start_of_month = ref_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_df = df_hist[df_hist['timestamp'] >= start_of_month] if not df_hist.empty else pd.DataFrame()
        month_to_date_units = EnergyAnalyticsOptimizer._compute_delta_reset_safe(month_df, 'totalEnergy', current_total_energy)
        month_r1 = EnergyAnalyticsOptimizer._compute_delta_reset_safe(month_df, 'energy1', current_e1)
        month_r2 = EnergyAnalyticsOptimizer._compute_delta_reset_safe(month_df, 'energy2', current_e2)

        # If data history is strictly in-session, fall back to max available delta
        if month_to_date_units == 0.0 and not df_hist.empty:
            month_to_date_units = EnergyAnalyticsOptimizer._compute_delta_reset_safe(df_hist, 'totalEnergy', current_total_energy)
            month_r1 = EnergyAnalyticsOptimizer._compute_delta_reset_safe(df_hist, 'energy1', current_e1)
            month_r2 = EnergyAnalyticsOptimizer._compute_delta_reset_safe(df_hist, 'energy2', current_e2)

        # 4. Statistical Month-End Projection (Extrapolation: Average Daily * Days in Month)
        elapsed_days = max(1, ref_time.day)
        if ref_time.month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif ref_time.month in [4, 6, 9, 11]:
            days_in_month = 30
        else:
            days_in_month = 29 if (ref_time.year % 4 == 0 and (ref_time.year % 100 != 0 or ref_time.year % 400 == 0)) else 28

        avg_daily = month_to_date_units / elapsed_days
        projected_month_end = avg_daily * days_in_month
        remaining_units = max(0.0, EnergyAnalyticsOptimizer.GRUHA_JYOTHI_TARGET - month_to_date_units)
        projected_excess = max(0.0, projected_month_end - EnergyAnalyticsOptimizer.GRUHA_JYOTHI_TARGET)

        # 5. Weekly 7-Day Bar Data (Reset-safe calculation per day)
        daily_bars = []
        for i in range(7):
            d = (start_of_week + timedelta(days=i)).date()
            d_df = df_hist[df_hist['timestamp'].dt.date == d] if not df_hist.empty else pd.DataFrame()
            d_val = EnergyAnalyticsOptimizer._compute_delta_reset_safe(d_df, 'totalEnergy') if not d_df.empty else 0.0
            daily_bars.append({
                "dayIndex": i,
                "dayName": ["M", "T", "W", "T", "F", "S", "S"][i],
                "date": str(d),
                "units": round(d_val, 2)
            })

        # 6. Hourly Historical Load Profiles (0-23 hours measured averages in Watts)
        hourly_r1 = [0.0] * 24
        hourly_r2 = [0.0] * 24
        if not df_hist.empty:
            df_hist_copy = df_hist.copy()
            df_hist_copy['hour'] = df_hist_copy['timestamp'].dt.hour
            h_grp = df_hist_copy.groupby('hour')[['power1', 'power2']].mean()
            for h in range(24):
                if h in h_grp.index:
                    hourly_r1[h] = round(float(h_grp.loc[h, 'power1']), 1)
                    hourly_r2[h] = round(float(h_grp.loc[h, 'power2']), 1)

        # 7. Optimization Breakdown & Explainable Actions
        total_room_units = max(0.001, month_r1 + month_r2)
        r1_pct = (month_r1 / total_room_units) * 100.0
        r2_pct = (month_r2 / total_room_units) * 100.0

        suggestions = []
        if month_r1 > month_r2:
            suggestions.append({
                "title": "Room 1 is the primary energy contributor",
                "subtitle": f"Room 1 accounts for {r1_pct:.0f}% of measured room consumption. Inspect high-power appliances.",
                "icon": "warning"
            })
        elif month_r2 > month_r1:
            suggestions.append({
                "title": "Room 2 is the primary energy contributor",
                "subtitle": f"Room 2 accounts for {r2_pct:.0f}% of measured room consumption. Inspect high-power appliances.",
                "icon": "warning"
            })
        else:
            suggestions.append({
                "title": "Balanced Room Usage",
                "subtitle": "Room 1 and Room 2 show balanced energy consumption.",
                "icon": "bolt"
            })

        if projected_excess > 0:
            suggestions.append({
                "title": "Gruha Jyothi Limit Alert",
                "subtitle": f"Projected monthly consumption ({projected_month_end:.1f} Units) exceeds the 96-unit allocation by {projected_excess:.1f} Units.",
                "icon": "energy_savings_leaf"
            })
        else:
            suggestions.append({
                "title": "Within Free Electricity Quota",
                "subtitle": f"Current consumption trend is projected at {projected_month_end:.1f} Units, well within the 96-unit quota.",
                "icon": "bolt"
            })

        suggestions.append({
            "title": "Shift flexible loads to off-peak hours",
            "subtitle": "Running high-wattage equipment during non-peak hours reduces grid stress and optimizes efficiency.",
            "icon": "access_time"
        })

        return {
            "forecast": {
                "monthToDateUnits": round(month_to_date_units, 2),
                "todayUnits": round(today_units, 2),
                "weekUnits": round(week_units, 2),
                "averageDailyUnits": round(avg_daily, 2),
                "projectedMonthEndUnits": round(projected_month_end, 2),
                "allocatedUnits": EnergyAnalyticsOptimizer.GRUHA_JYOTHI_TARGET,
                "remainingUnits": round(remaining_units, 2),
                "projectedExcessUnits": round(projected_excess, 2),
                "usagePercentage": round(min(1.0, month_to_date_units / EnergyAnalyticsOptimizer.GRUHA_JYOTHI_TARGET), 4),
                "daysElapsed": elapsed_days,
                "daysInMonth": days_in_month
            },
            "analytics": {
                "weeklyTotalUnits": round(week_units, 2),
                "dailyBars": daily_bars,
                "hourlyLoadRoom1": hourly_r1,
                "hourlyLoadRoom2": hourly_r2,
                "room1MonthUnits": round(month_r1, 2),
                "room2MonthUnits": round(month_r2, 2)
            },
            "optimization": {
                "room1Units": round(month_r1, 2),
                "room2Units": round(month_r2, 2),
                "room1Percent": round(r1_pct, 1),
                "room2Percent": round(r2_pct, 1),
                "suggestions": suggestions
            }
        }

    @staticmethod
    def generate_notifications(prediction_result, forecast_data, latest_reading, history_df=None):
        """
        Generates dynamic rule-based notifications from actual system states and predictions.
        """
        notifications = []
        p_res = prediction_result
        fc = forecast_data["forecast"]
        opt = forecast_data["optimization"]
        total_p = float(latest_reading.get('totalPower', 0.0))

        # Alert 1: Projected monthly usage exceeds 96 units
        if fc["projectedExcessUnits"] > 0:
            notifications.append({
                "id": "notif_projected_exceed",
                "type": "warning",
                "title": "Projected Quota Exceed",
                "subtitle": f"Statistical month-end projection is {fc['projectedMonthEndUnits']:.1f} Units (exceeds 96-unit Gruha Jyothi quota by {fc['projectedExcessUnits']:.1f} Units).",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "HIGH"
            })

        # Alert 2: Approaching allocated limit (>= 85%)
        if fc["usagePercentage"] >= 0.85 and fc["remainingUnits"] > 0:
            notifications.append({
                "id": "notif_approaching_limit",
                "type": "warning",
                "title": "Approaching Allocation Limit",
                "subtitle": f"You have consumed {fc['monthToDateUnits']:.1f} of {fc['allocatedUnits']:.0f} allocated units. Only {fc['remainingUnits']:.1f} units remaining.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "HIGH"
            })

        # Alert 3: Unusually high instantaneous power (statistically derived: >2500W or >95th percentile)
        high_power_threshold = 2500.0
        if history_df is not None and len(history_df) >= 20:
            p95 = float(history_df['totalPower'].quantile(0.95))
            high_power_threshold = max(2000.0, p95)

        if total_p >= high_power_threshold:
            notifications.append({
                "id": "notif_instant_high_power",
                "type": "warning",
                "title": "High Instantaneous Load Detected",
                "subtitle": f"Current total power ({total_p:.0f} W) exceeds the {high_power_threshold:.0f} W high-load threshold.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "HIGH"
            })

        # Alert 4: Room 1 Next Hour High Usage Prediction
        if p_res["room1"]["nextHour"]["state"] == "HIGH USAGE":
            notifications.append({
                "id": "notif_r1_next_high",
                "type": "prediction",
                "title": "High Usage Expected in Room 1",
                "subtitle": f"AI model predicts high power load next hour ({p_res['room1']['nextHour']['confidencePercent']:.0f}% confidence). Reduce non-essential loads.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "MEDIUM"
            })

        # Alert 5: Room 2 Next Hour High Usage Prediction
        if p_res["room2"]["nextHour"]["state"] == "HIGH USAGE":
            notifications.append({
                "id": "notif_r2_next_high",
                "type": "prediction",
                "title": "High Usage Expected in Room 2",
                "subtitle": f"AI model predicts high power load next hour ({p_res['room2']['nextHour']['confidencePercent']:.0f}% confidence). Reduce non-essential loads.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "MEDIUM"
            })

        # Alert 6: Room 1 Tomorrow Advance Warning
        if p_res["room1"]["tomorrow"]["state"] == "HIGH USAGE":
            notifications.append({
                "id": "notif_r1_tomorrow_high",
                "type": "advance",
                "title": "Advance Alert: High Usage Tomorrow (Room 1)",
                "subtitle": f"High energy consumption predicted in Room 1 tomorrow around this time ({p_res['room1']['tomorrow']['confidencePercent']:.0f}% confidence). Plan ahead to shift heavy appliances.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "MEDIUM"
            })

        # Alert 7: Room 2 Tomorrow Advance Warning
        if p_res["room2"]["tomorrow"]["state"] == "HIGH USAGE":
            notifications.append({
                "id": "notif_r2_tomorrow_high",
                "type": "advance",
                "title": "Advance Alert: High Usage Tomorrow (Room 2)",
                "subtitle": f"High energy consumption predicted in Room 2 tomorrow around this time ({p_res['room2']['tomorrow']['confidencePercent']:.0f}% confidence). Plan ahead to shift heavy appliances.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "MEDIUM"
            })

        return notifications


# -------------------------------------------------------------------------
# 4. MAIN BRIDGE WORKER
# -------------------------------------------------------------------------
class FirebaseMLBridge:
    def __init__(self):
        self.firebase = FirebaseClient(FIREBASE_DB_URL)
        self.ml_engine = EnergyMLInferenceEngine(MODEL_DIR)
        print("[*] Initialized Firebase ML Bridge successfully.")
        print(f"[*] Firebase Target URL: {FIREBASE_DB_URL}")

    def run_sync(self):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching telemetry records from Firebase...")
        
        # Retrieve recent historical window (up to 1500 records) to support analytics and lag construction
        readings_data = self.firebase.get(READINGS_PATH, query_params='orderBy=%22$key%22&limitToLast=1500')

        if not readings_data or not isinstance(readings_data, dict):
            print("[!] No readings retrieved from Firebase. Waiting for ESP32...")
            return False

        # 1. Parse and sort all records chronologically by valid timestamp
        rows = []
        for k, v in readings_data.items():
            if isinstance(v, dict):
                ts_raw = v.get('timestamp')
                if ts_raw and ts_raw != 'TIME_ERROR':
                    try:
                        ts_dt = pd.to_datetime(ts_raw)
                        rows.append({
                            'key': k,
                            'timestamp': ts_dt,
                            'voltage1': float(v.get('voltage1', 230.0)),
                            'voltage2': float(v.get('voltage2', 230.0)),
                            'current1': float(v.get('current1', 0.0)),
                            'current2': float(v.get('current2', 0.0)),
                            'power1': float(v.get('power1', 0.0)),
                            'power2': float(v.get('power2', 0.0)),
                            'energy1': float(v.get('energy1', 0.0)),
                            'energy2': float(v.get('energy2', 0.0)),
                            'totalCurrent': float(v.get('totalCurrent', 0.0)),
                            'totalPower': float(v.get('totalPower', 0.0)),
                            'totalEnergy': float(v.get('totalEnergy', 0.0)),
                        })
                    except Exception:
                        pass

        if not rows:
            print("[!] No valid timestamped records found.")
            return False

        df_hist = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)

        # 2. Select the record with the maximum timestamp as latest_reading
        latest_row = df_hist.iloc[-1]
        latest_reading = latest_row.to_dict()
        print(f"[*] Latest Reading Key: {latest_reading['key']} | Time: {latest_reading['timestamp']} | Power: {latest_reading['totalPower']} W")

        # 3. Run ML Random Forest Inference
        prediction_result = self.ml_engine.predict(latest_reading, df_hist)

        # 4. Run Energy Aggregation & Statistical Monthly Forecast
        analytics_result = EnergyAnalyticsOptimizer.process_readings_history(df_hist, latest_reading)

        # 5. Generate Dynamic Rule-Based Notifications & Alerts
        notifications_list = EnergyAnalyticsOptimizer.generate_notifications(
            prediction_result, analytics_result, latest_reading, df_hist
        )

        # 6. Publish results to Firebase
        print("[*] Publishing ML Predictions to Firebase...")
        self.firebase.put(PREDICTIONS_PATH, prediction_result)

        print("[*] Publishing Energy Analytics & Forecast to Firebase...")
        analytics_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **analytics_result["analytics"],
            **analytics_result["forecast"]
        }
        self.firebase.put(ANALYTICS_PATH, analytics_payload)

        print("[*] Publishing Optimization Insights to Firebase...")
        opt_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **analytics_result["optimization"],
            **analytics_result["forecast"]
        }
        self.firebase.put(OPTIMIZATION_PATH, opt_payload)

        print("[*] Publishing Active Alerts to Firebase...")
        notif_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(notifications_list),
            "items": notifications_list
        }
        self.firebase.put(NOTIFICATIONS_PATH, notif_payload)
        self.firebase.put(ALERTS_PATH, notif_payload)

        print("[+] Sync cycle completed successfully!")
        return True

    def run_daemon(self, interval_seconds=10):
        print(f"[*] Starting continuous ML Bridge daemon (polling every {interval_seconds}s)...")
        while True:
            try:
                self.run_sync()
            except Exception as e:
                print(f"[!] Error in sync cycle: {e}")
            time.sleep(interval_seconds)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voltix Smart Energy Firebase ML Bridge")
    parser.add_argument("--once", action="store_true", help="Run a single sync cycle and exit")
    parser.add_argument("--interval", type=int, default=10, help="Daemon sync interval in seconds (default: 10)")
    args = parser.parse_args()

    bridge = FirebaseMLBridge()

    if args.once:
        bridge.run_sync()
    else:
        bridge.run_daemon(interval_seconds=args.interval)

if __name__ == "__main__":
    main()
