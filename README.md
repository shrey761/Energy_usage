# Voltix AI Energy Optimizer — Machine Learning Engine

This module provides real-time inference, multi-horizon energy predictions, reset-safe energy analytics, and dynamic optimization advisories for the Voltix Smart Energy platform.

## Architecture

```text
ESP32 Energy Meter
        │ (Streams telemetry every ~11s)
        ▼
Firebase Realtime Database (/sensor_logs/esp32_01/readings)
        ▲
        │ (Real-time sync)
        ▼
Python ML Bridge (firebase_ml_bridge.py)
  ├── 1. Timestamp-sorted Telemetry Ingestion
  ├── 2. 23-Feature Pipeline (Strictly historical/current at time t)
  ├── 3. Multi-Horizon Random Forest Models (Room 1 & Room 2)
  │      ├── 1-Hour Lookahead Classifiers & Power Regressors (t + 1h ± 15m)
  │      └── 24-Hour Lookahead Classifiers (t + 24h ± 15m)
  ├── 4. Reset-Safe Cumulative Energy Aggregator
  ├── 5. Statistical Month-End Projection (96-Unit Quota)
  └── 6. Dynamic Optimization & Explainable Alerts
        │
        ▼ (Writes to /predictions, /analytics, /optimization, /alerts)
Firebase Realtime Database
        │
        ▼ (StreamBuilder real-time listeners)
Flutter Application (smartenergy)
```

## Trained Models (`saved_models/`)

- `r1_next_hour_rf.pkl`: Room 1 1-Hour Ahead High Usage Classifier ($t + 1\text{h} \pm 15\text{m}$, threshold $\ge 1200\text{W}$)
- `r2_next_hour_rf.pkl`: Room 2 1-Hour Ahead High Usage Classifier ($t + 1\text{h} \pm 15\text{m}$, threshold $\ge 1200\text{W}$)
- `r1_tomorrow_rf.pkl`: Room 1 Tomorrow Same-Hour Classifier ($t + 24\text{h} \pm 15\text{m}$, Calibrated Threshold: 0.70)
- `r2_tomorrow_rf.pkl`: Room 2 Tomorrow Same-Hour Classifier ($t + 24\text{h} \pm 15\text{m}$, Calibrated Threshold: 0.65)
- `r1_reg_next_rf.pkl`: Room 1 1-Hour Ahead Power Regressor (Watts, $t + 1\text{h} \pm 15\text{m}$)
- `r2_reg_next_rf.pkl`: Room 2 1-Hour Ahead Power Regressor (Watts, $t + 1\text{h} \pm 15\text{m}$)
- `tomorrow_thresholds.json`: Calibrated probability threshold configuration

### One-Hour Target Construction & Resolution Correction
- **Previous Implementation**: Used `df['power'].shift(-1)`, which on 11-second resolution telemetry corresponded to predicting $t + 11\text{s}$ (the next telemetry packet) rather than 1 hour into the future.
- **Corrected Implementation**: Uses timestamp-based forward lookup via `pd.merge_asof` matching $t + 1\text{h}$ with a 15-minute tolerance window. This ensures genuine one-hour-ahead predictive semantics while maintaining 100% causal separation ($t_{\text{match}} > t_{\text{feature}}$).

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain 1-Hour Horizon Models
```bash
python energy_advisor.py
```

### 3. Run Single Sync Cycle
```bash
python firebase_ml_bridge.py --once
```

### 4. Run Continuous Real-Time Daemon
```bash
python firebase_ml_bridge.py --interval 10
```

### 5. Run Test Suite
```bash
python tests/test_pipeline_full.py
```
