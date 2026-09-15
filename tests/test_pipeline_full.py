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
        self.assertEqual(self.engine.thresholds['r1'], 0.70)
        self.assertEqual(self.engine.thresholds['r2'], 0.65)

    def test_02_feature_count_and_order(self):
        """Verify feature count is exactly 23 and matches expected order."""
        self.assertEqual(len(FEATURE_COLS), 23)
        expected = [
            'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
            'power1', 'power2', 'current1', 'current2', 'voltage1', 'voltage2',
            'totalPower', 'totalCurrent',
            'power1_lag1', 'power1_lag3', 'power1_roll_mean5', 'power1_roll_max5',
            'power2_lag1', 'power2_lag3', 'power2_roll_mean5', 'power2_roll_max5',
            'totalPower_lag1', 'totalPower_roll_mean5'
        ]
        self.assertEqual(FEATURE_COLS, expected)

    def test_03_inference_predictions(self):
        """Verify inference output structure and predict_proba calculations."""
        sample_reading = {
            'timestamp': '2026-08-24 18:05:04',
            'power1': 1600.0,
            'power2': 500.0,
            'voltage1': 230.0,
            'voltage2': 230.0,
            'current1': 6.95,
            'current2': 2.17,
            'totalPower': 2100.0,
            'totalCurrent': 9.12
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
        self.assertEqual(delta_normal, 5.0) # 15 - 10

        # With rollover: 10.0 -> 12.0 -> 0.5 (reset!) -> 2.0
        df_reset = pd.DataFrame([
            {'totalEnergy': 10.0},
            {'totalEnergy': 12.0},
            {'totalEnergy': 0.5},
            {'totalEnergy': 2.0}
        ])
        delta_reset = EnergyAnalyticsOptimizer._compute_delta_reset_safe(df_reset, 'totalEnergy')
        # Deltas: (12-10)=2 + (reset: 0.5) + (2-0.5)=1.5 -> Total = 4.0
        self.assertEqual(delta_reset, 4.0)

    def test_05_firebase_connection_and_sync(self):
        """Verify live connection to Firebase Realtime Database and roundtrip data verification."""
        client = FirebaseClient(FIREBASE_DB_URL)
        data = client.get("/sensor_logs/esp32_01/predictions/latest")
        self.assertIsNotNone(data)
        self.assertIn('room1', data)
        self.assertIn('room2', data)
        self.assertIn('timestamp', data)

if __name__ == '__main__':
    unittest.main()
