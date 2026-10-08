#!/usr/bin/env python3
"""
VOLTIX — Full Pipeline Unit Test Suite (Enhanced RF Production Validation)
Script: tests/test_pipeline_full.py
"""

import os
import sys
import unittest
import json
import pandas as pd
import numpy as np

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from firebase_ml_bridge import (
    EnergyMLInferenceEngine,
    EnergyAnalyticsOptimizer,
    FirebaseClient,
    BASELINE_FEATURE_COLS,
    FEATURE_COLS,
    MODEL_DIR,
    FIREBASE_DB_URL
)

class TestVoltixMLPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = EnergyMLInferenceEngine(MODEL_DIR)

    def test_01_models_loaded(self):
        """Verify all 6 Random Forest models and thresholds load properly."""
        expected_models = [
            'r1_next_hour', 'r2_next_hour',
            'r1_tomorrow', 'r2_tomorrow',
            'r1_reg_next', 'r2_reg_next'
        ]
        for name in expected_models:
            self.assertIn(name, self.engine.models)
            self.assertIsNotNone(self.engine.models[name])
        
        self.assertIn('r1', self.engine.thresholds)
        self.assertIn('r2', self.engine.thresholds)
        self.assertEqual(self.engine.thresholds['r1'], 0.40)
        self.assertEqual(self.engine.thresholds['r2'], 0.40)

    def test_02_feature_count_and_order(self):
        """Verify feature schema matches the 23 features expected by the 6 trained models."""
        self.assertEqual(len(FEATURE_COLS), 23)
        self.assertEqual(len(BASELINE_FEATURE_COLS), 23)
        for model_name, model in self.engine.models.items():
            if hasattr(model, 'feature_names_in_'):
                self.assertEqual(list(model.feature_names_in_), FEATURE_COLS, f"Schema mismatch in {model_name}")


    def test_03_inference_predictions(self):
        """Verify inference output structure and predict_proba calculations on calibrated loads."""
        sample_reading = {
            'timestamp': '2026-10-04 18:05:04',
            'power1': 80.0,
            'power2': 20.0,
            'voltage1': 230.0,
            'voltage2': 230.0,
            'current1': 0.35,
            'current2': 0.09,
            'totalPower': 100.0,
            'totalCurrent': 0.44
        }
        res = self.engine.predict(sample_reading)
        self.assertIn('room1', res)
        self.assertIn('room2', res)
        self.assertIn('summary', res)

        # Room 1 assertions
        self.assertIn(res['room1']['nextHour']['state'], ['HIGH USAGE', 'NORMAL'])
        self.assertGreaterEqual(res['room1']['nextHour']['confidencePercent'], 50.0)
        self.assertIn(res['room1']['tomorrow']['state'], ['HIGH USAGE', 'NORMAL'])
        self.assertIsInstance(res['room1']['predictedPower'], float)

        # Room 2 assertions
        self.assertIn(res['room2']['nextHour']['state'], ['HIGH USAGE', 'NORMAL'])
        self.assertIn(res['room2']['tomorrow']['state'], ['HIGH USAGE', 'NORMAL'])
        self.assertIsInstance(res['room2']['predictedPower'], float)

    def test_04_meter_reset_handling(self):
        """Verify reset-safe delta calculation detects rollover and avoids negative values."""
        df_normal = pd.DataFrame([
            {'totalEnergy': 10.0},
            {'totalEnergy': 12.5},
            {'totalEnergy': 15.0}
        ])
        delta_normal = EnergyAnalyticsOptimizer._compute_delta_reset_safe(df_normal, 'totalEnergy')
        self.assertEqual(delta_normal, 5.0)

        # Rollover test
        df_reset = pd.DataFrame([
            {'totalEnergy': 10.0},
            {'totalEnergy': 12.0},
            {'totalEnergy': 0.5},
            {'totalEnergy': 2.0}
        ])
        delta_reset = EnergyAnalyticsOptimizer._compute_delta_reset_safe(df_reset, 'totalEnergy')
        self.assertEqual(delta_reset, 4.0)

    def test_05_enhanced_inference_with_history(self):
        """Verify full 126-feature Enhanced RF prediction using 400-reading history window."""
        sample_reading = {
            'timestamp': '2026-10-30 20:00:00',
            'power1': 80.0,
            'power2': 20.0,
            'voltage1': 230.0,
            'voltage2': 230.0,
            'current1': 0.35,
            'current2': 0.09,
            'totalPower': 100.0,
            'totalCurrent': 0.44
        }
        history_df = pd.DataFrame([sample_reading] * 400)
        history_df['timestamp'] = [pd.to_datetime('2026-10-30 20:00:00') - pd.Timedelta(seconds=(399-i)*10) for i in range(400)]
        
        res = self.engine.predict(sample_reading, history_df)
        self.assertIn('room1', res)
        self.assertIn('room2', res)
        self.assertIn(res['room1']['nextHour']['state'], ['HIGH USAGE', 'NORMAL'])
        self.assertIn(res['room2']['nextHour']['state'], ['HIGH USAGE', 'NORMAL'])
        self.assertGreaterEqual(res['room1']['nextHour']['confidencePercent'], 50.0)

if __name__ == '__main__':
    unittest.main()
