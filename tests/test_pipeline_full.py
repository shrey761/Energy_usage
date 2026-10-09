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

    def test_06_invalid_timestamp_sanitization(self):
        """Verify TIME_ERROR, null, and malformed timestamps are safely sanitized without warnings."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test reading with TIME_ERROR
            reading_err = {
                'timestamp': 'TIME_ERROR',
                'power1': 60.0,
                'power2': 0.0,
                'voltage1': 230.0,
                'voltage2': 230.0,
                'totalPower': 60.0,
                'totalCurrent': 0.26
            }
            feats, dt, p1, p2, tp = self.engine.prepare_features(reading_err)
            self.assertEqual(len(feats.columns), 23)
            self.assertIsNotNone(dt)

            # Test history_df containing TIME_ERROR, None, and valid out-of-order timestamps
            history_data = [
                {'timestamp': '2026-10-08 14:00:10', 'power1': 50.0, 'power2': 10.0, 'totalPower': 60.0, 'totalEnergy': 1.0},
                {'timestamp': 'TIME_ERROR', 'power1': 99.0, 'power2': 99.0, 'totalPower': 198.0, 'totalEnergy': 1.1},
                {'timestamp': '2026-10-08 14:00:00', 'power1': 40.0, 'power2': 10.0, 'totalPower': 50.0, 'totalEnergy': 0.9},
                {'timestamp': None, 'power1': 0.0, 'power2': 0.0, 'totalPower': 0.0, 'totalEnergy': 1.2},
                {'timestamp': '2026-10-08 14:00:20', 'power1': 60.0, 'power2': 10.0, 'totalPower': 70.0, 'totalEnergy': 1.2},
            ]
            df_hist = pd.DataFrame(history_data)
            res = self.engine.predict(reading_err, df_hist)
            self.assertIn('room1', res)
            self.assertIn('room2', res)

            # Check that no UserWarning for datetime format inference was triggered
            dateutil_warnings = [warning for warning in w if 'dateutil' in str(warning.message) or 'Could not infer format' in str(warning.message)]
            self.assertEqual(len(dateutil_warnings), 0, "Found unexpected dateutil fallback warnings!")

if __name__ == '__main__':
    unittest.main()
