import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

FEATURE_COLS = [
    'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
    'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
    'totalPower', 'totalCurrent',
    'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
    'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
    'totalPower_lag1', 'totalPower_roll_mean5'
]

def load_trained_models():
    models = {}
    model_names = ['r1_next_hour', 'r2_next_hour', 'r1_reg_next', 'r2_reg_next', 'r1_tomorrow', 'r2_tomorrow']
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}_rf.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found. Please run energy_advisor.py first.")
        models[name] = joblib.load(path)
    return models

def predict_and_advise(models, power1, power2, voltage1=230.0, voltage2=230.0, hour=None, dayofweek=None):
    if hour is None:
        hour = datetime.now().hour
    if dayofweek is None:
        dayofweek = datetime.now().weekday()
        
    current1 = power1 / max(voltage1, 1.0)
    current2 = power2 / max(voltage2, 1.0)
    total_power = power1 + power2
    total_current = current1 + current2
    is_weekend = 1 if dayofweek in [5, 6] else 0

    feature_dict = {
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
        'power1_lag1': power1,
        'power1_lag3': power1,
        'power1_roll_mean5': power1,
        'power1_roll_max5': power1,
        'power2_lag1': power2,
        'power2_lag3': power2,
        'power2_roll_mean5': power2,
        'power2_roll_max5': power2,
        'totalPower_lag1': total_power,
        'totalPower_roll_mean5': total_power
    }

    input_df = pd.DataFrame([feature_dict])[FEATURE_COLS]

    # Classification Outcomes (Primary Basis)
    r1_next_class = models['r1_next_hour'].predict(input_df)[0]
    r1_next_proba = models['r1_next_hour'].predict_proba(input_df)[0][r1_next_class] * 100
    r1_next_state = "HIGH USAGE" if r1_next_class == 1 else "NORMAL"

    r2_next_class = models['r2_next_hour'].predict(input_df)[0]
    r2_next_proba = models['r2_next_hour'].predict_proba(input_df)[0][r2_next_class] * 100
    r2_next_state = "HIGH USAGE" if r2_next_class == 1 else "NORMAL"

    r1_tmrw_class = models['r1_tomorrow'].predict(input_df)[0]
    r1_tmrw_proba = models['r1_tomorrow'].predict_proba(input_df)[0][r1_tmrw_class] * 100
    r1_tmrw_state = "HIGH USAGE" if r1_tmrw_class == 1 else "NORMAL"

    r2_tmrw_class = models['r2_tomorrow'].predict(input_df)[0]
    r2_tmrw_proba = models['r2_tomorrow'].predict_proba(input_df)[0][r2_tmrw_class] * 100
    r2_tmrw_state = "HIGH USAGE" if r2_tmrw_class == 1 else "NORMAL"

    recommendations = []

    if r1_next_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 1 next hour. Reduce non-essential loads in Room 1.")
    
    if r2_next_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 2 next hour. Reduce non-essential loads in Room 2.")

    if r1_tmrw_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 1 tomorrow around this time. Plan to reduce usage in Room 1.")

    if r2_tmrw_state == "HIGH USAGE":
        recommendations.append("High usage expected in Room 2 tomorrow around this time. Plan to reduce usage in Room 2.")

    if not recommendations:
        recommendations.append("Normal usage expected across all rooms. No load reduction required.")

    return {
        "current": {
            "room1_power": power1,
            "room2_power": power2
        },
        "room1": {
            "next_hour": r1_next_state,
            "next_hour_confidence": round(r1_next_proba, 1),
            "tomorrow_same_hour": r1_tmrw_state,
            "tomorrow_confidence": round(r1_tmrw_proba, 1)
        },
        "room2": {
            "next_hour": r2_next_state,
            "next_hour_confidence": round(r2_next_proba, 1),
            "tomorrow_same_hour": r2_tmrw_state,
            "tomorrow_confidence": round(r2_tmrw_proba, 1)
        },
        "recommendations": recommendations
    }

def main():
    parser = argparse.ArgumentParser(description="Live Energy Usage Predictor & Smart Advisor")
    parser.add_argument("--power1", type=float, default=None, help="Current power in Room 1 (Watts, expected ~80W)")
    parser.add_argument("--power2", type=float, default=None, help="Current power in Room 2 (Watts, expected ~20W)")
    parser.add_argument("--hour", type=int, default=None, help="Current hour of day (0-23)")
    args = parser.parse_args()

    models = load_trained_models()

    if args.power1 is not None and args.power2 is not None:
        p1 = args.power1
        p2 = args.power2
        h = args.hour if args.hour is not None else datetime.now().hour
    else:
        print("\n=== Real-Time Room Energy Advisory Predictor ===")
        try:
            p1_input = input("Enter Room 1 Power in Watts [Default 80.0]: ").strip()
            p1 = float(p1_input) if p1_input else 80.0

            p2_input = input("Enter Room 2 Power in Watts [Default 20.0]: ").strip()
            p2 = float(p2_input) if p2_input else 20.0

            h_input = input(f"Enter Hour of Day (0-23) [Default {datetime.now().hour}]: ").strip()
            h = int(h_input) if h_input else datetime.now().hour
        except ValueError:
            print("Invalid input format. Using default sample: Room 1 = 80W, Room 2 = 20W, Hour = 18")
            p1, p2, h = 80.0, 20.0, 18

    result = predict_and_advise(models, power1=p1, power2=p2, hour=h)

    print("\n" + "=" * 65)
    print("                    PREDICTION RESULTS")
    print("=" * 65)
    print("ROOM 1")
    print(f"Current Power: {result['current']['room1_power']:.1f} W")
    print(f"Next Hour: {result['room1']['next_hour']}")
    print(f"Confidence: {result['room1']['next_hour_confidence']:.1f}%")
    print(f"Tomorrow Same Hour: {result['room1']['tomorrow_same_hour']}")
    print(f"Confidence: {result['room1']['tomorrow_confidence']:.1f}%")
    print("\nROOM 2")
    print(f"Current Power: {result['current']['room2_power']:.1f} W")
    print(f"Next Hour: {result['room2']['next_hour']}")
    print(f"Confidence: {result['room2']['next_hour_confidence']:.1f}%")
    print(f"Tomorrow Same Hour: {result['room2']['tomorrow_same_hour']}")
    print(f"Confidence: {result['room2']['tomorrow_confidence']:.1f}%")
    print("\nACTIONABLE RECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(f"  * \"{rec}\"")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
