import sys
import os
import pandas as pd
import numpy as np

from energy_advisor import load_and_clean_data, prepare_features_and_targets, EnergyPredictionAdvisor, CSV_PATH
from app_advisor import LiveRoomEnergyAdvisor

def main():
    print("=" * 80)
    print("      AI ENERGY OPTIMIZER: RANDOM FOREST MULTI-HORIZON ADVISORY (OCTOBER 2026)")
    print("=" * 80)

    # 1. Load full 60,480-row dataset
    df_raw, counts = load_and_clean_data(CSV_PATH)
    df_features = prepare_features_and_targets(df_raw, threshold_r1=40.0, threshold_r2=10.0)

    # 2. Print Dataset Accounting
    print("\n--- DATASET PIPELINE ROW COUNTS ---")
    print(f"Raw CSV rows                     : {counts['raw_csv_rows']}")
    print(f"Rows after timestamp validation  : {counts['rows_after_timestamp_validation']}")
    print(f"Rows after sorting               : {counts['rows_after_sorting']}")
    print(f"Rows after feature engineering   : {len(df_features)}")
    print(f"Dataset Time Range               : {counts['start_timestamp']} to {counts['end_timestamp']}")

    # 3. Live Advisor Simulation
    advisor = LiveRoomEnergyAdvisor()

    print("\n" + "=" * 80)
    print("     SIMULATING CLASSIFICATION-FIRST ADVISORY OUTCOMES (CALIBRATED HARDWARE)")
    print("=" * 80)

    scenarios = [
        {
            "name": "Scenario 1: Room 1 (80W Incandescent) Active, Room 2 OFF",
            "data": df_features[(df_features['power1'] > 75.0) & (df_features['power2'] == 0.0)].iloc[0].to_dict()
        },
        {
            "name": "Scenario 2: Room 2 (20W LED) Active, Room 1 OFF",
            "data": df_features[(df_features['power1'] == 0.0) & (df_features['power2'] > 18.0)].iloc[0].to_dict()
        },
        {
            "name": "Scenario 3: Both Rooms Active (~100W Total Load)",
            "data": df_features[(df_features['power1'] > 75.0) & (df_features['power2'] > 18.0)].iloc[0].to_dict()
        },
        {
            "name": "Scenario 4: Standby / Zero Load Across Both Rooms",
            "data": df_features[(df_features['power1'] == 0.0) & (df_features['power2'] == 0.0)].iloc[0].to_dict()
        }
    ]

    for sc in scenarios:
        sample = sc['data']
        ts = sample.get('timestamp', 'N/A')

        result = advisor.generate_room_advisory(sample)
        r1 = result["room1"]
        r2 = result["room2"]
        
        print(f"\n>>> [{sc['name']}] (Observation Time: {ts})")
        print("ROOM 1 (Channel 1, 80W Incandescent)")
        print(f"Current Power: {r1['current_power_watts']:.1f} W")
        print(f"Next Hour: {r1['next_hour']}")
        print(f"Confidence: {r1['next_hour_confidence']:.1f}%")
        print(f"Tomorrow Same Hour: {r1['tomorrow_same_hour']}")
        print(f"Confidence: {r1['tomorrow_confidence']:.1f}%")
        print("Recommendation:")
        print(f"{r1['recommendation']}")
        print("")
        print("ROOM 2 (Channel 2, 20W LED)")
        print(f"Current Power: {r2['current_power_watts']:.1f} W")
        print(f"Next Hour: {r2['next_hour']}")
        print(f"Confidence: {r2['next_hour_confidence']:.1f}%")
        print(f"Tomorrow Same Hour: {r2['tomorrow_same_hour']}")
        print(f"Confidence: {r2['tomorrow_confidence']:.1f}%")
        print("Recommendation:")
        print(f"{r2['recommendation']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
