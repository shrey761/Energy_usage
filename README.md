# Voltix AI Energy Optimizer — Machine Learning Engine

This module provides real-time inference, multi-horizon energy predictions, reset-safe energy analytics, and dynamic optimization advisories for the Voltix Smart Energy platform.

## Architecture

```text
ESP32 Energy Meter
        │ (Streams telemetry every 10s)
        ▼
Firebase Realtime Database (/sensor_logs/esp32_01/readings)
        ▲
        │ (Real-time sync)
        ▼
Python ML Bridge (firebase_ml_bridge.py)
  ├── 1. Timestamp-sorted Telemetry Ingestion
  ├── 2. 23-Feature Pipeline (Strictly causal at time t)
  ├── 3. Multi-Horizon Random Forest Models (Room 1 & Room 2)
  │      ├── 1-Hour Lookahead Classifiers & Power Regressors (t + 1h ± 15s)
  │      └── 24-Hour Lookahead Classifiers (t + 24h ± 15s)
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

## Dataset & Physical Hardware Setup

- **Dataset**: `VOLTIX_OCTOBER_2026_calibrated_dataset.csv` (Synthetically extended calibrated VOLTIX dataset based on experimentally obtained calibrated measurements; 259,200 records, 10-second sampling, October 1–31, 2026).
- **Room 1 (Channel 1 / `power1`)**: 2 × 40 W incandescent bulbs ($\approx 80\text{ W}$ active, $0\text{ W}$ standby). Calibrated threshold: **$40.0\text{ W}$**.
- **Room 2 (Channel 2 / `power2`)**: 2 × 10 W LED bulbs ($\approx 20\text{ W}$ active, $0\text{ W}$ standby). Calibrated threshold: **$10.0\text{ W}$**.
- **Combined Peak Load**: $\approx 100\text{ W}$.

## Trained Models (`saved_models/`)

- `r1_next_hour_rf.pkl`: Room 1 1-Hour Ahead High Usage Classifier ($t + 1\text{h}$, threshold $\ge 40.0\text{ W}$)
- `r2_next_hour_rf.pkl`: Room 2 1-Hour Ahead High Usage Classifier ($t + 1\text{h}$, threshold $\ge 10.0\text{ W}$)
- `r1_reg_next_rf.pkl`: Room 1 1-Hour Ahead Power Regressor (Watts, $t + 1\text{h}$)
- `r2_reg_next_rf.pkl`: Room 2 1-Hour Ahead Power Regressor (Watts, $t + 1\text{h}$)
- `r1_tomorrow_rf.pkl`: Room 1 Tomorrow Same-Hour Classifier ($t + 24\text{h}$, Frozen Val Threshold: 0.40)
- `r2_tomorrow_rf.pkl`: Room 2 Tomorrow Same-Hour Classifier ($t + 24\text{h}$, Frozen Val Threshold: 0.40)
- `tomorrow_thresholds.json`: Calibrated probability threshold configuration (`{"r1": 0.4, "r2": 0.4}`)

## Setup & Execution

### 1. Train Models on October Dataset
```bash
python train_and_evaluate_final.py
```

### 2. Run Single Firebase Sync Cycle
```bash
python firebase_ml_bridge.py --once
```

### 3. Run Test Suite
```bash
python tests/test_pipeline_full.py
```
