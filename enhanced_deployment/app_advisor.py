#!/usr/bin/env python3
"""
VOLTIX — Isolated Enhanced Energy Advisor Application Layer
Script: enhanced_deployment/app_advisor.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from enhanced_deployment.predict_live import load_enhanced_models, predict_enhanced, MIN_HISTORY_REQUIRED

class EnhancedLiveRoomEnergyAdvisor:
    """
    Isolated application layer wrapping the Enhanced Random Forest classifiers (126 causal features).
    Processes continuous streaming telemetry windows (>= 360 readings) and outputs advance advisories.
    """
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(BASE_DIR, "realistic_dataset_experiment", "models")
        self.models_dir = models_dir
        self.models = load_enhanced_models(self.models_dir)
        print("[*] EnhancedLiveRoomEnergyAdvisor initialized successfully with isolated models.")

    def process_telemetry_window(self, history_df):
        """
        Processes a dataframe containing historical readings up to current time t.
        Returns full prediction dictionary, probabilities, and actionable advisories.
        """
        if history_df is None or len(history_df) < MIN_HISTORY_REQUIRED:
            return {
                'status': 'INSUFFICIENT_HISTORY',
                'message': f'Insufficient historical data for Enhanced RF prediction. Requires {MIN_HISTORY_REQUIRED} readings.',
                'records_available': len(history_df) if history_df is not None else 0
            }
            
        pred_res = predict_enhanced(self.models, history_df)
        
        # Build explainable natural language advisories
        r1_state = pred_res['room1']['prediction']
        r2_state = pred_res['room2']['prediction']
        r1_prob_high = pred_res['room1']['probability_high']
        r2_prob_high = pred_res['room2']['probability_high']
        
        recommendations = []
        if r1_state == "HIGH USAGE":
            recommendations.append(f"Advance Load Warning: Room 1 is predicted to exceed 40W in the next hour ({r1_prob_high*100:.1f}% confidence). Reduce non-essential lighting/appliances.")
        else:
            recommendations.append("Room 1 load is predicted to remain within normal baseline parameters (<40W).")
            
        if r2_state == "HIGH USAGE":
            recommendations.append(f"Advance Load Warning: Room 2 is predicted to exceed 10W in the next hour ({r2_prob_high*100:.1f}% confidence). Optimize secondary room lighting.")
        else:
            recommendations.append("Room 2 load is predicted to remain within normal baseline parameters (<10W).")
            
        pred_res['status'] = 'SUCCESS'
        pred_res['advisories'] = recommendations
        return pred_res

if __name__ == '__main__':
    advisor = EnhancedLiveRoomEnergyAdvisor()
    print("[*] Advisor ready for isolated testing.")
