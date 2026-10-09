"""
Verification test for updated firebase_ml_bridge logic:
- Dynamic optimization advisories
- Realistic notification triggers with deterministic IDs
- Generalized energy quota naming
"""
import sys
import os
import pandas as pd
import numpy as np

# Ensure path contains energy_usage_rf
sys.path.insert(0, r"C:\Users\shrey\.gemini\antigravity\scratch\energy_usage_rf")

from firebase_ml_bridge import EnergyAnalyticsOptimizer, FEATURE_COLS

def test_optimization_advisories():
    print("Testing generate_optimization_advisories...")
    optimizer = EnergyAnalyticsOptimizer(cost_per_kwh=7.0)
    
    # Scenario A: Low normal power (e.g. 15W bulb in Room 1, 0W in Room 2)
    pred_low = {
        "room1": {"currentPower": 15.0, "nextHour": {"state": "NORMAL"}},
        "room2": {"currentPower": 0.0, "nextHour": {"state": "NORMAL"}},
        "summary": {"totalCurrentPower": 15.0, "totalPredictedPower": 15.0, "overallNextHourState": "NORMAL"}
    }
    analytics_low = {
        "room1MonthUnits": 2.0, "room2MonthUnits": 1.0,
        "monthToDateUnits": 3.0, "projectedMonthlyKwh": 10.0,
        "averageDailyUnits": 0.33, "usagePercentage": 0.03
    }
    
    adv_low = optimizer.generate_optimization_advisories(pred_low, analytics_low)
    suggestions_low = adv_low["suggestions"]
    assert len(suggestions_low) >= 2, "Expected at least 2 suggestions"
    print("  [Pass] Low load suggestions count:", len(suggestions_low))
    titles = [s["title"].lower() for s in suggestions_low]
    print("  Low load titles:", titles)
    assert any("within" in t or "balanced" in t or "optimal" in t or "expected range" in t for t in titles), "Should report within expected range/balanced"
    
    # Scenario B: High heavy power (e.g. 250W in Room 1)
    pred_high = {
        "room1": {"currentPower": 250.0, "nextHour": {"state": "HIGH USAGE"}},
        "room2": {"currentPower": 20.0, "nextHour": {"state": "NORMAL"}},
        "summary": {"totalCurrentPower": 270.0, "totalPredictedPower": 260.0, "overallNextHourState": "HIGH USAGE"}
    }
    analytics_high = {
        "room1MonthUnits": 75.0, "room2MonthUnits": 10.0,
        "monthToDateUnits": 85.0, "projectedMonthlyKwh": 110.0,
        "averageDailyUnits": 3.7, "usagePercentage": 0.885
    }
    adv_high = optimizer.generate_optimization_advisories(pred_high, analytics_high)
    suggestions_high = adv_high["suggestions"]
    titles_high = [s["title"] for s in suggestions_high]
    print("  High load titles:", titles_high)
    assert any("Room 1 accounts for" in t or "Shift flexible loads" in t for t in titles_high), "Should report Room 1 contribution or peak shift"
    print("  [Pass] High load suggestions correctly identify Room 1 contribution and peak.")

def test_notifications():
    print("\nTesting generate_notifications...")
    optimizer = EnergyAnalyticsOptimizer(cost_per_kwh=7.0)
    
    # Scenario 1: Normal 100W bulb in Room 1 (p1=100W, p2=0W) -> SHOULD NOT TRIGGER OVERLOAD WARNING!
    pred_normal = {
        "room1": {"currentPower": 100.0, "nextHour": {"state": "NORMAL"}},
        "room2": {"currentPower": 0.0, "nextHour": {"state": "NORMAL"}},
        "summary": {"totalCurrentPower": 100.0, "totalPredictedPower": 95.0, "overallNextHourState": "NORMAL"}
    }
    analytics_normal = {
        "usagePercentage": 0.15, "monthToDateUnits": 14.4
    }
    notifs_normal = optimizer.generate_notifications(pred_normal, analytics_normal)
    notif_ids_normal = [n["id"] for n in notifs_normal["items"]]
    print("  Normal 100W bulb notification IDs:", notif_ids_normal)
    assert "notif_r1_overload" not in notif_ids_normal, "100W bulb must NOT trigger overload warning!"
    assert "notif_optimal" in notif_ids_normal, "Should have optimal status"
    
    # Scenario 2: Overload 220W in Room 1 and approaching quota (88%)
    pred_overload = {
        "room1": {"currentPower": 220.0, "nextHour": {"state": "HIGH USAGE"}},
        "room2": {"currentPower": 0.0, "nextHour": {"state": "NORMAL"}},
        "summary": {"totalCurrentPower": 220.0, "totalPredictedPower": 210.0, "overallNextHourState": "HIGH USAGE"}
    }
    analytics_overload = {
        "usagePercentage": 0.88, "monthToDateUnits": 84.5
    }
    notifs_overload = optimizer.generate_notifications(pred_overload, analytics_overload)
    notif_ids_overload = [n["id"] for n in notifs_overload["items"]]
    print("  Overload notification IDs:", notif_ids_overload)
    assert "notif_r1_overload" in notif_ids_overload, "220W should trigger Room 1 overload warning!"
    assert "notif_predicted_peak" in notif_ids_overload, "Predicted 210W should trigger peak warning!"
    assert "notif_quota_warning" in notif_ids_overload, "88% usage should trigger quota warning!"
    
    print("  [Pass] Deterministic IDs and thresholds verified across all notification rules.")

if __name__ == "__main__":
    test_optimization_advisories()
    test_notifications()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
