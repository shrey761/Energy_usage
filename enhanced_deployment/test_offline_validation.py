#!/usr/bin/env python3
"""
VOLTIX — Isolated Offline Validation Suite
Script: enhanced_deployment/test_offline_validation.py
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from enhanced_deployment.predict_live import load_enhanced_models, predict_enhanced, THRESHOLD_R1, THRESHOLD_R2

def run_offline_tests():
    print("=" * 80)
    print(" VOLTIX ENHANCED RF ISOLATED OFFLINE VALIDATION")
    print("=" * 80)
    
    # Load dataset
    data_path = os.path.join(BASE_DIR, "VOLTIX_realistic_30day_synthetic_dataset.csv")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Load isolated models
    models = load_enhanced_models()
    
    # Define test scenario search (must have at least 360 prior rows)
    scenarios = [
        ("Scenario 1: Room 1 OFF & Room 2 OFF", lambda r: r['power1'] < 5.0 and r['power2'] < 2.0),
        ("Scenario 2: Room 1 ON & Room 2 OFF", lambda r: r['power1'] >= 40.0 and r['power2'] < 2.0),
        ("Scenario 3: Room 1 OFF & Room 2 ON", lambda r: r['power1'] < 5.0 and r['power2'] >= 10.0),
        ("Scenario 4: Both Rooms ON", lambda r: r['power1'] >= 40.0 and r['power2'] >= 10.0),
    ]
    
    all_passed = True
    
    for desc, cond in scenarios:
        print(f"\n--- Testing {desc} ---")
        # Find matching index with at least 360 prior records
        matches = [i for i in range(360, len(df)) if cond(df.iloc[i])]
        if not matches:
            print(f"[!] Warning: No sample matching condition in dataset.")
            continue
            
        idx = matches[min(50, len(matches)-1)]
        window = df.iloc[idx-359:idx+1].copy().reset_index(drop=True)
        
        cur_p1 = window['power1'].iloc[-1]
        cur_p2 = window['power2'].iloc[-1]
        cur_ts = window['timestamp'].iloc[-1]
        
        res = predict_enhanced(models, window)
        
        r1 = res['room1']
        r2 = res['room2']
        
        print(f"Timestamp       : {cur_ts}")
        print(f"Current Power   : Room 1 = {cur_p1:.2f} W, Room 2 = {cur_p2:.2f} W")
        print(f"Room 1 Pred     : {r1['prediction']} (Normal={r1['probability_normal']*100:.2f}%, High={r1['probability_high']*100:.2f}%)")
        print(f"Room 2 Pred     : {r2['prediction']} (Normal={r2['probability_normal']*100:.2f}%, High={r2['probability_high']*100:.2f}%)")
        print(f"Features Fed    : {res['feature_count']} (Schema match: {res['schema_match']})")
        
        if res['feature_count'] != 126 or not res['schema_match']:
            all_passed = False
            
    # Test error handling on truncated window (<360)
    print("\n--- Testing Failure Handling on Insufficient History (< 360 rows) ---")
    truncated_window = df.iloc[500:550].copy().reset_index(drop=True) # 50 rows
    try:
        predict_enhanced(models, truncated_window)
        print("[!] ERROR: Should have thrown ValueError on truncated window.")
        all_passed = False
    except ValueError as e:
        print(f"[+] PASS: Successfully caught error -> {e}")
        
    print("\n" + "=" * 80)
    print(f" OFFLINE VALIDATION OVERALL STATUS: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 80)
    return all_passed

if __name__ == '__main__':
    run_offline_tests()
