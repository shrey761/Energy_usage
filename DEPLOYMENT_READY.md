# VOLTIX — Enhanced Random Forest Deployment Readiness Audit

**Status**: **AUDIT COMPLETED — PENDING BACKEND INTEGRATION UPDATES**  
**Audit Target**: Enhanced Random Forest ($126$ Causal Features) from the Realistic Dataset Experiment  
**Experimental Performance**: Room 1: **$86.07\%$** Accuracy ($0.7339$ F1) | Room 2: **$87.38\%$** Accuracy ($0.6377$ F1)  
**Production Status**: Current production models in `saved_models/` remain **untouched and active**.

---

## 1. Executive Summary & Verdict

> [!WARNING]
> **COMPATIBILITY VERDICT: CONDITIONAL (PIPELINE UPDATES REQUIRED)**
> The Enhanced Random Forest models **cannot** be deployed by simply copying `.pkl` files into `saved_models/`. 
> The existing production pipeline (`firebase_ml_bridge.py`, `predict_live.py`, `app_advisor.py`) currently constructs a **23-feature vector** with a short 5-sample historical window. The Enhanced RF requires **126 exact causal features** and a historical telemetry buffer of at least **360 readings (1 hour at 10-second sampling)**.

---

## 2. Model Files & Artifacts

The trained enhanced models are serialized in isolation inside `realistic_dataset_experiment/models/`:
- `realistic_dataset_experiment/models/r1_enhanced_rf.pkl` ($500$ trees, $\text{max\_depth}=15$, $\text{min\_samples\_leaf}=1$, $\text{max\_features}=\text{'sqrt'}$)
- `realistic_dataset_experiment/models/r2_enhanced_rf.pkl` ($300$ trees, $\text{max\_depth}=25$, $\text{min\_samples\_leaf}=2$, $\text{max\_features}=\text{'sqrt'}$)
- `realistic_dataset_experiment/models/enhanced_features.json` (Exact metadata and 126-feature ordered list)

---

## 3. Exact Feature Schema (126 Features in Strict Order)

```
 1. current1
 2. current2
 3. energy1
 4. energy2
 5. power1
 6. power2
 7. totalCurrent
 8. totalEnergy
 9. totalPower
10. voltage1
11. voltage2
12. hour
13. minute
14. day_of_week
15. is_weekend
16. month
17. hour_sin
18. hour_cos
19. day_sin
20. day_cos
21. power_ratio
22. apparent_power_1
23. apparent_power_2
24. power1_lag1
25. power1_lag3
26. power1_lag6
27. power1_lag30
28. power1_lag60
29. power1_lag180
30. power1_lag360
31. power2_lag1
32. power2_lag3
33. power2_lag6
34. power2_lag30
35. power2_lag60
36. power2_lag180
37. power2_lag360
38. totalPower_lag1
39. totalPower_lag3
40. totalPower_lag6
41. totalPower_lag30
42. totalPower_lag60
43. totalPower_lag180
44. totalPower_lag360
45. power1_roll_mean1m
46. power1_roll_min1m
47. power1_roll_max1m
48. power1_roll_std1m
49. power1_roll_mean5m
50. power1_roll_min5m
51. power1_roll_max5m
52. power1_roll_std5m
53. power1_roll_mean15m
54. power1_roll_min15m
55. power1_roll_max15m
56. power1_roll_std15m
57. power1_roll_mean30m
58. power1_roll_min30m
59. power1_roll_max30m
60. power1_roll_std30m
61. power1_roll_mean1h
62. power1_roll_min1h
63. power1_roll_max1h
64. power1_roll_std1h
65. power2_roll_mean1m
66. power2_roll_min1m
67. power2_roll_max1m
68. power2_roll_std1m
69. power2_roll_mean5m
70. power2_roll_min5m
71. power2_roll_max5m
72. power2_roll_std5m
73. power2_roll_mean15m
74. power2_roll_min15m
75. power2_roll_max15m
76. power2_roll_std15m
77. power2_roll_mean30m
78. power2_roll_min30m
79. power2_roll_max30m
80. power2_roll_std30m
81. power2_roll_mean1h
82. power2_roll_min1h
83. power2_roll_max1h
84. power2_roll_std1h
85. totalPower_roll_mean1m
86. totalPower_roll_min1m
87. totalPower_roll_max1m
88. totalPower_roll_std1m
89. totalPower_roll_mean5m
90. totalPower_roll_min5m
91. totalPower_roll_max5m
92. totalPower_roll_std5m
93. totalPower_roll_mean15m
94. totalPower_roll_min15m
95. totalPower_roll_max15m
96. totalPower_roll_std15m
97. totalPower_roll_mean30m
98. totalPower_roll_min30m
99. totalPower_roll_max30m
100. totalPower_roll_std30m
101. totalPower_roll_mean1h
102. totalPower_roll_min1h
103. totalPower_roll_max1h
104. totalPower_roll_std1h
105. power1_diff_1m
106. power1_diff_5m
107. power1_diff_15m
108. power1_diff_1h
109. power2_diff_1m
110. power2_diff_5m
111. power2_diff_15m
112. power2_diff_1h
113. totalPower_diff_1m
114. totalPower_diff_5m
115. totalPower_diff_15m
116. totalPower_diff_1h
117. r1_is_on
118. r1_consec_on_min
119. r1_consec_off_min
120. r1_min_since_trans
121. r2_is_on
122. r2_consec_on_min
123. r2_consec_off_min
124. r2_min_since_trans
125. r1_daily_cum_on_min
126. r2_daily_cum_on_min
```

---

## 4. Required Historical Window & Firebase Telemetry

1. **Required Firebase Realtime DB Fields**:
   - `power1`, `power2`, `voltage1`, `voltage2`, `current1`, `current2`, `totalPower`, `totalCurrent`, `energy1`, `energy2`, `totalEnergy`, `timestamp`.
   - All fields are already emitted by the VOLTIX ESP32 firmware to `/sensor_logs/esp32_01/readings`.
2. **Historical Buffer Depth**:
   - **Minimum Historical Buffer**: **360 consecutive readings** ($360 \times 10\text{s} = 3,600\text{s} = 1\text{ hour}$).
   - **Recommended Query**: `?orderBy="timestamp"&limitToLast=400` from Firebase RTDB.
3. **Daily Cumulative Runtime Tracking**:
   - `r1_daily_cum_on_min` and `r2_daily_cum_on_min` can be maintained by an in-memory accumulator in `firebase_ml_bridge.py` that resets at midnight, or populated by querying the day's readings on bridge startup.

---

## 5. Test Inference Verification

A standalone end-to-end inference test was executed using `realistic_dataset_experiment/test_enhanced_inference.py`:

```
================================================================================
 TEST INFERENCE RESULTS (Enhanced Random Forest)
================================================================================
[*] Test Sample Timestamp : 2026-10-25 02:10:40
[*] Total Features Fed    : 126
[*] Room 1 Prediction     : NORMAL (Probability: Normal=77.75%, High=22.25%)
[*] Room 1 Actual (t+1h)  : NORMAL (0.36 W)
[*] Room 2 Prediction     : NORMAL (Probability: Normal=72.33%, High=27.67%)
[*] Room 2 Actual (t+1h)  : NORMAL (0.00 W)
================================================================================
```

---

## 6. Pre-Deployment Action Plan (When Authorized)

When deployment is approved by the user, the following minimal, clean changes will be required:

1. **`firebase_ml_bridge.py`**:
   - Expand `limitToLast` query parameter from `10` to `400`.
   - Update `prepare_features()` to construct the exact 126-feature vector from the 360-reading window.
   - Maintain daily runtime state accumulator.
2. **`predict_live.py` & `app_advisor.py`**:
   - Update `FEATURE_COLS` to the 126-feature schema.
3. **`saved_models/`**:
   - Copy `realistic_dataset_experiment/models/r1_enhanced_rf.pkl` $\to$ `saved_models/r1_next_hour_rf.pkl`.
   - Copy `realistic_dataset_experiment/models/r2_enhanced_rf.pkl` $\to$ `saved_models/r2_next_hour_rf.pkl`.
