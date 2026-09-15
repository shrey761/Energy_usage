import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
CSV_PATH = os.path.join(BASE_DIR, "LATEST_ML_TRAINING_DATA.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "LATEST_ML_TRAINING_DATA.xlsx")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "energy_data.csv")

FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

class LiveRoomEnergyAdvisor:
    """
    Production-ready application layer wrapping the trained Random Forest classifiers.
    Accepts live streaming energy readings and produces classification-first advisories
    with calibrated advance warning thresholds.
    """
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models, self.thresholds = self._load_models_and_thresholds()

    def _load_models_and_thresholds(self):
        models = {}
        model_files = {
            'r1_next': 'r1_next_hour_rf.pkl',
            'r2_next': 'r2_next_hour_rf.pkl',
            'r1_tomorrow': 'r1_tomorrow_rf.pkl',
            'r2_tomorrow': 'r2_tomorrow_rf.pkl'
        }
        for key, fname in model_files.items():
            path = os.path.join(self.model_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file '{path}' not found. Ensure models are trained and saved.")
            models[key] = joblib.load(path)

        thresh_path = os.path.join(self.model_dir, "tomorrow_thresholds.json")
        thresholds = {'r1': 0.70, 'r2': 0.65}
        if os.path.exists(thresh_path):
            with open(thresh_path, 'r') as f:
                thresholds = json.load(f)

        return models, thresholds

    def prepare_input_dataframe(self, reading):
        """
        Sanitizes and prepares a single live reading or dictionary into the exact ML feature matrix.
        Supports both raw telemetry readings and pre-computed sequence rows.
        """
        if isinstance(reading, pd.Series):
            reading_dict = reading.to_dict()
        elif isinstance(reading, dict):
            reading_dict = reading.copy()
        else:
            raise TypeError("Input reading must be a dict or pandas Series.")

        # Resolve timestamp and time features
        ts = reading_dict.get('timestamp', None)
        if isinstance(ts, str):
            try:
                dt = pd.to_datetime(ts)
            except Exception:
                dt = datetime.now()
        elif isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.now()

        hour = reading_dict.get('hour', dt.hour)
        dayofweek = reading_dict.get('dayofweek', dt.weekday())
        is_weekend = 1 if dayofweek in [5, 6] else 0

        # Physical measurements
        power1 = float(reading_dict.get('power1', 0.0))
        power2 = float(reading_dict.get('power2', 0.0))
        voltage1 = float(reading_dict.get('voltage1', 230.0))
        voltage2 = float(reading_dict.get('voltage2', 230.0))
        current1 = float(reading_dict.get('current1', power1 / max(voltage1, 1.0)))
        current2 = float(reading_dict.get('current2', power2 / max(voltage2, 1.0)))
        total_power = float(reading_dict.get('totalPower', power1 + power2))
        total_current = float(reading_dict.get('totalCurrent', current1 + current2))

        # Lags & rolling metrics (defaults to current values if live single-point reading)
        p1_lag1 = float(reading_dict.get('power1_lag1', power1))
        p1_lag3 = float(reading_dict.get('power1_lag3', power1))
        p1_mean5 = float(reading_dict.get('power1_roll_mean5', power1))
        p1_max5 = float(reading_dict.get('power1_roll_max5', power1))

        p2_lag1 = float(reading_dict.get('power2_lag1', power2))
        p2_lag3 = float(reading_dict.get('power2_lag3', power2))
        p2_mean5 = float(reading_dict.get('power2_roll_mean5', power2))
        p2_max5 = float(reading_dict.get('power2_roll_max5', power2))

        tp_lag1 = float(reading_dict.get('totalPower_lag1', total_power))
        tp_mean5 = float(reading_dict.get('totalPower_roll_mean5', total_power))

        row_data = {
            'hour': hour,
            'dayofweek': dayofweek,
            'is_weekend': is_weekend,
            'hour_sin': np.sin(2 * np.pi * hour / 24.0),
            'hour_cos': np.cos(2 * np.pi * hour / 24.0),
            'power1': power1,
            'power2': power2,
            'current1': current1,
            'current2': current2,
            'voltage1': voltage1,
            'voltage2': voltage2,
            'totalPower': total_power,
            'totalCurrent': total_current,
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

        return pd.DataFrame([row_data])[FEATURE_COLS], power1, power2

    def generate_room_advisory(self, reading):
        """
        Core reusable prediction function for any live or historical reading.
        Evaluates Room 1 and Room 2 independently with actual predict_proba() confidence
        and calibrated advance warning thresholds.
        """
        input_df, power1, power2 = self.prepare_input_dataframe(reading)

        # 1. Room 1 Predictions
        r1_next_probas = self.models['r1_next'].predict_proba(input_df)[0]
        r1_next_class = 1 if r1_next_probas[1] >= 0.5 else 0
        r1_next_proba = r1_next_probas[r1_next_class] * 100
        r1_next_label = "HIGH USAGE" if r1_next_class == 1 else "NORMAL"

        th_r1 = self.thresholds.get('r1', 0.70)
        r1_tmrw_probas = self.models['r1_tomorrow'].predict_proba(input_df)[0]
        r1_tmrw_class = 1 if r1_tmrw_probas[1] >= th_r1 else 0
        r1_tmrw_proba = r1_tmrw_probas[r1_tmrw_class] * 100
        r1_tmrw_label = "HIGH USAGE" if r1_tmrw_class == 1 else "NORMAL"

        # Room 1 Recommendation
        if r1_next_label == "HIGH USAGE" and r1_tmrw_label == "HIGH USAGE":
            r1_rec = "Reduce non-essential loads in Room 1 during the next hour and plan to shift tomorrow's heavy usage."
        elif r1_next_label == "HIGH USAGE":
            r1_rec = "Reduce non-essential loads in Room 1 during the next hour."
        elif r1_tmrw_label == "HIGH USAGE":
            r1_rec = "Plan to reduce non-essential usage in Room 1 tomorrow around this time."
        else:
            r1_rec = "No load reduction required."

        # 2. Room 2 Predictions
        r2_next_probas = self.models['r2_next'].predict_proba(input_df)[0]
        r2_next_class = 1 if r2_next_probas[1] >= 0.5 else 0
        r2_next_proba = r2_next_probas[r2_next_class] * 100
        r2_next_label = "HIGH USAGE" if r2_next_class == 1 else "NORMAL"

        th_r2 = self.thresholds.get('r2', 0.65)
        r2_tmrw_probas = self.models['r2_tomorrow'].predict_proba(input_df)[0]
        r2_tmrw_class = 1 if r2_tmrw_probas[1] >= th_r2 else 0
        r2_tmrw_proba = r2_tmrw_probas[r2_tmrw_class] * 100
        r2_tmrw_label = "HIGH USAGE" if r2_tmrw_class == 1 else "NORMAL"

        # Room 2 Recommendation
        if r2_next_label == "HIGH USAGE" and r2_tmrw_label == "HIGH USAGE":
            r2_rec = "Reduce non-essential loads in Room 2 during the next hour and plan to shift tomorrow's heavy usage."
        elif r2_next_label == "HIGH USAGE":
            r2_rec = "Reduce non-essential loads in Room 2 during the next hour."
        elif r2_tmrw_label == "HIGH USAGE":
            r2_rec = "Plan to reduce non-essential usage in Room 2 tomorrow around this time."
        else:
            r2_rec = "No load reduction required."

        return {
            "room1": {
                "current_power_watts": round(power1, 1),
                "next_hour": r1_next_label,
                "next_hour_confidence": round(r1_next_proba, 1),
                "tomorrow_same_hour": r1_tmrw_label,
                "tomorrow_confidence": round(r1_tmrw_proba, 1),
                "recommendation": r1_rec
            },
            "room2": {
                "current_power_watts": round(power2, 1),
                "next_hour": r2_next_label,
                "next_hour_confidence": round(r2_next_proba, 1),
                "tomorrow_same_hour": r2_tmrw_label,
                "tomorrow_confidence": round(r2_tmrw_proba, 1),
                "recommendation": r2_rec
            }
        }

    def format_advisory_text(self, advisory_result):
        """
        Formats the advisory output dictionary into clean, human-readable text blocks.
        """
        r1 = advisory_result["room1"]
        r2 = advisory_result["room2"]

        lines = [
            "ROOM 1",
            f"Current Power: {r1['current_power_watts']:.0f} W",
            f"Next Hour: {r1['next_hour']}",
            f"Confidence: {r1['next_hour_confidence']:.0f}%",
            f"Tomorrow Same Hour: {r1['tomorrow_same_hour']}",
            f"Confidence: {r1['tomorrow_confidence']:.0f}%",
            "Recommendation:",
            f"{r1['recommendation']}",
            "",
            "ROOM 2",
            f"Current Power: {r2['current_power_watts']:.0f} W",
            f"Next Hour: {r2['next_hour']}",
            f"Confidence: {r2['next_hour_confidence']:.0f}%",
            f"Tomorrow Same Hour: {r2['tomorrow_same_hour']}",
            f"Confidence: {r2['tomorrow_confidence']:.0f}%",
            "Recommendation:",
            f"{r2['recommendation']}"
        ]
        return "\n".join(lines)


# -------------------------------------------------------------------------
# CLI & APPLICATION DEMONSTRATION
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live Smart Energy Advisory Application Layer")
    parser.add_argument("--power1", type=float, default=None, help="Measured power in Room 1 (Watts)")
    parser.add_argument("--power2", type=float, default=None, help="Measured power in Room 2 (Watts)")
    parser.add_argument("--hour", type=int, default=None, help="Observation hour (0-23)")
    args = parser.parse_args()

    advisor = LiveRoomEnergyAdvisor()

    if args.power1 is not None and args.power2 is not None:
        live_reading = {
            'power1': args.power1,
            'power2': args.power2,
            'hour': args.hour if args.hour is not None else datetime.now().hour
        }
        result = advisor.generate_room_advisory(live_reading)
        print("\n" + advisor.format_advisory_text(result) + "\n")
    else:
        # Run demonstration on sample readings from the dataset
        from energy_advisor import load_and_clean_data, prepare_features_and_targets
        df_raw, _ = load_and_clean_data(CSV_PATH)
        df_features = prepare_features_and_targets(df_raw)

        sample_r1_high = df_features[(df_features['power1'] > 1700) & (df_features['power2'] == 0.0)].iloc[0]

        print("==========================================================================")
        print("          LIVE SMART ROOM ENERGY ADVISORY SYSTEM (ACTIVE TEST)            ")
        print("==========================================================================")
        print(f"Observation Timestamp: {sample_r1_high['timestamp']}\n")

        result = advisor.generate_room_advisory(sample_r1_high)
        print(advisor.format_advisory_text(result))
        print("==========================================================================")

if __name__ == "__main__":
    main()
