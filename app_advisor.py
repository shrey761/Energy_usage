#!/usr/bin/env python3
"""
VOLTIX — Live Room Energy Advisor (Application Layer)
Script: app_advisor.py
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from predict_live import (
    load_trained_models, build_enhanced_feature_vector, build_baseline_feature_vector,
    THRESHOLD_R1, THRESHOLD_R2, MIN_HISTORY_REQUIRED, MODEL_DIR,
    BASELINE_FEATURE_COLS, ENHANCED_FEATURE_COLS, FEATURE_COLS
)

class LiveRoomEnergyAdvisor:
    """
    Production application layer wrapping the Enhanced Random Forest classifiers.
    Accepts live streaming energy readings, generates 126 causal features, and produces
    classification advisories for Flutter UI.
    """
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = load_trained_models(self.model_dir)
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self):
        thresh_path = os.path.join(self.model_dir, "tomorrow_thresholds.json")
        thresholds = {'r1': 0.40, 'r2': 0.40}
        if os.path.exists(thresh_path):
            with open(thresh_path, 'r') as f:
                thresholds = json.load(f)
        return thresholds

    def process_reading(self, reading, history_df=None):
        """
        Processes reading with optional historical window.
        """
        p1 = float(reading.get('power1', 0.0))
        p2 = float(reading.get('power2', 0.0))
        v1 = float(reading.get('voltage1', 230.0))
        v2 = float(reading.get('voltage2', 230.0))
        
        # Build features
        X_base, dt, _, _, tp = build_baseline_feature_vector(reading, history_df)
        
        if history_df is not None and len(history_df) >= MIN_HISTORY_REQUIRED:
            X_enh, _, _, _ = build_enhanced_feature_vector(history_df)
        else:
            # Steady-state baseline fallback for single reading
            history_df = pd.DataFrame([reading] * MIN_HISTORY_REQUIRED)
            history_df['timestamp'] = [datetime.now() - pd.Timedelta(seconds=(MIN_HISTORY_REQUIRED-1-i)*10) for i in range(MIN_HISTORY_REQUIRED)]
            X_enh, _, _, _ = build_enhanced_feature_vector(history_df)

        # Enhanced RF 1-Hour Predictions
        r1_next_probas = self.models['r1_next_hour'].predict_proba(X_enh)[0]
        r1_next_state = "HIGH USAGE" if r1_next_probas[1] >= 0.5 else "NORMAL"
        r1_next_conf = float(r1_next_probas[1 if r1_next_state == "HIGH USAGE" else 0])

        r2_next_probas = self.models['r2_next_hour'].predict_proba(X_enh)[0]
        r2_next_state = "HIGH USAGE" if r2_next_probas[1] >= 0.5 else "NORMAL"
        r2_next_conf = float(r2_next_probas[1 if r2_next_state == "HIGH USAGE" else 0])

        # Tomorrow Predictions (24-Hour)
        th_r1 = self.thresholds.get('r1', 0.40)
        r1_tmrw_probas = self.models['r1_tomorrow'].predict_proba(X_base)[0]
        r1_tmrw_state = "HIGH USAGE" if r1_tmrw_probas[1] >= th_r1 else "NORMAL"
        r1_tmrw_conf = float(r1_tmrw_probas[1 if r1_tmrw_state == "HIGH USAGE" else 0])

        th_r2 = self.thresholds.get('r2', 0.40)
        r2_tmrw_probas = self.models['r2_tomorrow'].predict_proba(X_base)[0]
        r2_tmrw_state = "HIGH USAGE" if r2_tmrw_probas[1] >= th_r2 else "NORMAL"
        r2_tmrw_conf = float(r2_tmrw_probas[1 if r2_tmrw_state == "HIGH USAGE" else 0])

        # Regression
        r1_pred_power = max(0.0, float(self.models['r1_reg_next'].predict(X_base)[0]))
        r2_pred_power = max(0.0, float(self.models['r2_reg_next'].predict(X_base)[0]))

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

if __name__ == '__main__':
    advisor = LiveRoomEnergyAdvisor()
    sample = {'power1': 80.0, 'power2': 20.0, 'voltage1': 230.0, 'voltage2': 230.0}
    res = advisor.process_reading(sample)
    print(json.dumps(res, indent=2))
