#!/usr/bin/env python3
"""
VOLTIX — Isolated Enhanced Random Forest Firebase ML Bridge
Script: enhanced_deployment/firebase_ml_bridge.py
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
from datetime import datetime

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from enhanced_deployment.predict_live import (
    load_enhanced_models, predict_enhanced, build_126_feature_vector,
    EXPECTED_FEATURES, THRESHOLD_R1, THRESHOLD_R2, MIN_HISTORY_REQUIRED
)

FIREBASE_DB_URL = "https://voltix-1bac0-default-rtdb.asia-southeast1.firebasedatabase.app"
READINGS_PATH = "/sensor_logs/esp32_01/readings"
ENHANCED_PREDICTIONS_PATH = "/sensor_logs/esp32_01/predictions/enhanced_latest"

class FirebaseClient:
    def __init__(self, db_url=FIREBASE_DB_URL):
        self.db_url = db_url.rstrip('/')

    def get(self, path, query_params=""):
        full_path = f"{path}?{query_params}" if query_params else path
        if '?' in full_path:
            base_p, q_p = full_path.split('?', 1)
            url = f"{self.db_url}{base_p}.json?{q_p}"
        else:
            url = f"{self.db_url}{full_path}.json"
            
        try:
            req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=15) as response:
                res_body = response.read().decode('utf-8')
                return json.loads(res_body) if res_body and res_body != 'null' else None
        except Exception as e:
            print(f"[!] Firebase GET error: {e}")
            return None

    def patch(self, path, payload):
        url = f"{self.db_url}{path}.json"
        try:
            data_bytes = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data_bytes, method='PATCH', headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"[!] Firebase PATCH error: {e}")
            return None

class EnhancedFirebaseBridge:
    def __init__(self, models_dir=None, mock_data_path=None):
        if models_dir is None:
            models_dir = os.path.join(BASE_DIR, "realistic_dataset_experiment", "models")
        self.models_dir = models_dir
        self.mock_data_path = mock_data_path
        self.fb = FirebaseClient()
        self.models = None
        self.load_status = self._init_models()

    def _init_models(self):
        try:
            self.models = load_enhanced_models(self.models_dir)
            return "PASS"
        except Exception as e:
            print(f"[!] Model loading failure: {e}")
            return "FAIL"

    def fetch_historical_telemetry(self, limit=400):
        """
        Retrieves at least 400 latest readings from Firebase RTDB (or mock stream) and validates consistency.
        """
        if self.mock_data_path:
            print(f"[*] Using mock telemetry feed: {self.mock_data_path}")
            if not os.path.exists(self.mock_data_path):
                return None, "FAIL", f"Mock file not found: {self.mock_data_path}"
            df_mock = pd.read_csv(self.mock_data_path)
            raw_data = df_mock.tail(limit).to_dict(orient='records')
        else:
            raw_data = self.fb.get(READINGS_PATH, f'orderBy="timestamp"&limitToLast={limit}')
            
        if not raw_data:
            return None, "FAIL", "No data returned from Firebase endpoint."
            
        if isinstance(raw_data, dict):
            records = list(raw_data.values())
        elif isinstance(raw_data, list):
            records = [r for r in raw_data if r is not None]
        else:
            return None, "FAIL", f"Unexpected data format: {type(raw_data)}"
            
        if len(records) == 0:
            return None, "FAIL", "Zero readings available in telemetry stream."
            
        df = pd.DataFrame(records)
        if 'timestamp' not in df.columns:
            return None, "FAIL", "Missing 'timestamp' field in readings."
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        
        # Verify required physical fields
        req_fields = ['power1', 'power2', 'voltage1', 'voltage2', 'current1', 'current2', 'totalPower', 'totalCurrent']
        for f in req_fields:
            if f not in df.columns:
                if f == 'totalPower' and 'power1' in df and 'power2' in df:
                    df['totalPower'] = df['power1'] + df['power2']
                elif f == 'current1' and 'power1' in df and 'voltage1' in df:
                    df['current1'] = df['power1'] / np.maximum(df['voltage1'], 1.0)
                elif f == 'current2' and 'power2' in df and 'voltage2' in df:
                    df['current2'] = df['power2'] / np.maximum(df['voltage2'], 1.0)
                elif f == 'totalCurrent' and 'current1' in df and 'current2' in df:
                    df['totalCurrent'] = df['current1'] + df['current2']
                elif f in ['energy1', 'energy2', 'totalEnergy']:
                    df[f] = 0.0
                else:
                    return None, "FAIL", f"Missing essential field '{f}' in readings."
                    
        if 'energy1' not in df.columns: df['energy1'] = 0.0
        if 'energy2' not in df.columns: df['energy2'] = 0.0
        if 'totalEnergy' not in df.columns: df['totalEnergy'] = 0.0
        
        return df, "PASS", f"Successfully retrieved {len(df)} validated readings."

    def run_cycle(self, publish=False):
        print("=" * 80)
        print(" VOLTIX ENHANCED RF LIVE INFERENCE CYCLE")
        print("=" * 80)
        
        status_report = {
            'MODEL_LOAD': self.load_status,
            'FIREBASE_READ': 'FAIL',
            'HISTORY_REQUIREMENT': 'FAIL',
            'FEATURE_SCHEMA_MATCH': 'FAIL',
            'INFERENCE': 'FAIL'
        }
        
        if self.load_status != "PASS":
            print("[!] Halting: Model load failed.")
            return status_report
            
        # 1. Fetch from Firebase / Telemetry Stream
        history_df, fb_status, fb_msg = self.fetch_historical_telemetry(limit=400)
        status_report['FIREBASE_READ'] = fb_status
        print(f"[*] Firebase Read: {fb_status} ({fb_msg})")
        
        if fb_status != "PASS" or history_df is None:
            print("[!] Halting: Firebase read failed.")
            self._print_summary(status_report)
            return status_report
            
        rec_count = len(history_df)
        print(f"[*] Historical readings retrieved : {rec_count}")
        
        # 2. Check History Requirement
        if rec_count < MIN_HISTORY_REQUIRED:
            print(f"[!] Insufficient historical data for Enhanced RF prediction. Found: {rec_count}, Required: {MIN_HISTORY_REQUIRED}")
            status_report['HISTORY_REQUIREMENT'] = "FAIL"
            self._print_summary(status_report)
            return status_report
            
        status_report['HISTORY_REQUIREMENT'] = "PASS"
        
        # 3. Build & Assert 126 Features
        try:
            X, latest_ts, p1_cur, p2_cur = build_126_feature_vector(history_df)
            feat_count = X.shape[1]
            cols = list(X.columns)
            if feat_count == 126 and cols == EXPECTED_FEATURES:
                status_report['FEATURE_SCHEMA_MATCH'] = "PASS"
            else:
                status_report['FEATURE_SCHEMA_MATCH'] = "FAIL"
                print(f"[!] Feature schema mismatch: {feat_count} features.")
                self._print_summary(status_report)
                return status_report
        except Exception as e:
            print(f"[!] Feature generation error: {e}")
            self._print_summary(status_report)
            return status_report
            
        print(f"[*] Feature count                 : {feat_count} (Schema Match: PASS)")
        print(f"[*] Latest timestamp              : {latest_ts}")
        print(f"[*] Room 1 current power          : {p1_cur:.2f} W")
        print(f"[*] Room 2 current power          : {p2_cur:.2f} W")
        
        # 4. Inference
        try:
            pred_results = predict_enhanced(self.models, history_df)
            status_report['INFERENCE'] = "PASS"
            
            r1 = pred_results['room1']
            r2 = pred_results['room2']
            
            print("\n--- ROOM 1 (Next-Hour Prediction) ---")
            print(f"Prediction  : {r1['prediction']}")
            print(f"Probability : Normal={r1['probability_normal']*100:.2f}%, High={r1['probability_high']*100:.2f}% (Confidence: {r1['confidence']*100:.2f}%)")
            print(f"Threshold   : {r1['threshold_w']} W")
            
            print("\n--- ROOM 2 (Next-Hour Prediction) ---")
            print(f"Prediction  : {r2['prediction']}")
            print(f"Probability : Normal={r2['probability_normal']*100:.2f}%, High={r2['probability_high']*100:.2f}% (Confidence: {r2['confidence']*100:.2f}%)")
            print(f"Threshold   : {r2['threshold_w']} W")
            
            if publish and not self.mock_data_path:
                payload = {
                    'timestamp': str(latest_ts),
                    'room1_prediction': r1['prediction'],
                    'room1_prob_high': r1['probability_high'],
                    'room2_prediction': r2['prediction'],
                    'room2_prob_high': r2['probability_high'],
                    'model_version': 'Enhanced_RF_126Feat_v1.0_Isolated',
                    'updated_at': datetime.now().isoformat()
                }
                self.fb.patch(ENHANCED_PREDICTIONS_PATH, payload)
                print(f"\n[+] Enhanced predictions safely published to {ENHANCED_PREDICTIONS_PATH}")
                
        except Exception as e:
            print(f"[!] Inference execution error: {e}")
            status_report['INFERENCE'] = "FAIL"
            
        self._print_summary(status_report)
        return status_report

    def _print_summary(self, status_report):
        print("\n" + "=" * 80)
        print(" AUDIT VERIFICATION SUMMARY")
        print("=" * 80)
        for check, res in status_report.items():
            print(f" {check:<25} : {res}")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="VOLTIX Isolated Enhanced RF Bridge")
    parser.add_argument("--once", action="store_true", help="Run a single validation cycle and exit")
    parser.add_argument("--publish", action="store_true", help="Publish predictions to /predictions/enhanced_latest")
    parser.add_argument("--mock", action="store_true", help="Use local realistic dataset as mock telemetry stream")
    args = parser.parse_args()
    
    mock_file = os.path.join(BASE_DIR, "VOLTIX_realistic_30day_synthetic_dataset.csv") if args.mock else None
    bridge = EnhancedFirebaseBridge(mock_data_path=mock_file)
    bridge.run_cycle(publish=args.publish)

if __name__ == '__main__':
    main()
