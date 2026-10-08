#!/usr/bin/env python3
"""
VOLTIX Research Report Graph Generation Suite
Generates publication-quality IEEE-style figures for the research report.
- Graph 1: Random Forest vs Persistence Baseline 1-Hour Classification Accuracy
- Graph 2: Actual vs Predicted 1-Hour Power (Room 1 & Room 2)
"""

import os
import sys
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from train_and_evaluate_final import (
    load_and_preprocess_dataset,
    FEATURE_COLS,
    MODEL_DIR,
    THRESHOLD_R1,
    THRESHOLD_R2
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "report_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ============================================================
# GRAPH 1: RF vs Persistence 1-Hour Classification Accuracy
# ============================================================
def generate_graph1():
    print("\n" + "=" * 80)
    print(" GENERATING GRAPH 1: RF vs Persistence Baseline Classification Accuracy")
    print("=" * 80)

    rooms = ["Room 1", "Room 2"]
    random_forest = [67.16, 72.04]
    persistence = [61.17, 65.99]

    x = np.arange(len(rooms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    bars1 = ax.bar(
        x - width / 2,
        random_forest,
        width,
        label="Random Forest",
        color="#1f77b4"
    )

    bars2 = ax.bar(
        x + width / 2,
        persistence,
        width,
        label="Persistence Baseline",
        color="#aec7e8"
    )

    ax.set_title(
        "Random Forest vs Persistence Baseline\n"
        "1-Hour High-Usage Classification Accuracy",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel("Load Channel", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(rooms, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "graph1_rf_vs_persistence_accuracy.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Graph 1 saved successfully to: {output_file}")
    return output_file


# ============================================================
# GRAPH 2: Actual vs Predicted 1-Hour Power (Room 1 & Room 2)
# ============================================================
def generate_graph2(sample_window=600):
    print("\n" + "=" * 80)
    print(" GENERATING GRAPH 2: Actual vs Predicted 1-Hour Power (Room 1 & Room 2)")
    print("=" * 80)

    # 1. Load dataset & build 20% chronological test split
    df = load_and_preprocess_dataset()
    df_1h = df.dropna(subset=['power1_1h_actual', 'power2_1h_actual']).reset_index(drop=True)
    split_1h = int(len(df_1h) * 0.8)
    test_1h = df_1h.iloc[split_1h:].reset_index(drop=True)
    X_te = test_1h[FEATURE_COLS]

    # 2. Load trained Random Forest regression models
    r1_model_path = os.path.join(MODEL_DIR, "r1_reg_next_rf.pkl")
    r2_model_path = os.path.join(MODEL_DIR, "r2_reg_next_rf.pkl")

    if not os.path.exists(r1_model_path) or not os.path.exists(r2_model_path):
        raise FileNotFoundError(f"Trained regression models not found in {MODEL_DIR}")

    r1_model = joblib.load(r1_model_path)
    r2_model = joblib.load(r2_model_path)

    # 3. Generate predictions on untouched test set
    p1_pred = np.maximum(0.0, r1_model.predict(X_te))
    p2_pred = np.maximum(0.0, r2_model.predict(X_te))

    p1_actual = test_1h['power1_1h_actual'].values
    p2_actual = test_1h['power2_1h_actual'].values

    # 4. Compute exact full-test-set evaluation metrics
    mae1 = mean_absolute_error(p1_actual, p1_pred)
    rmse1 = root_mean_squared_error(p1_actual, p1_pred)
    r2_1 = r2_score(p1_actual, p1_pred)

    mae2 = mean_absolute_error(p2_actual, p2_pred)
    rmse2 = root_mean_squared_error(p2_actual, p2_pred)
    r2_2 = r2_score(p2_actual, p2_pred)

    # Console Verification Output
    print(f"\n--- Model Verification Metrics (Test Set N = {len(test_1h):,}) ---")
    print(f"Channel 1: MAE {mae1:.2f} W, RMSE {rmse1:.2f} W, R² {r2_1:.4f}")
    print(f"Channel 2: MAE {mae2:.2f} W, RMSE {rmse2:.2f} W, R² {r2_2:.4f}")

    sample_n = min(sample_window, len(test_1h))
    x_samples = np.arange(sample_n)

    # -------------------------------------------------------------
    # 5. Plot Room 1 (Channel 1)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.plot(
        x_samples,
        p1_actual[:sample_n],
        label="Actual Power (t + 1h)",
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9
    )
    ax.plot(
        x_samples,
        p1_pred[:sample_n],
        label="Random Forest Predicted Power (t + 1h)",
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        alpha=0.9
    )
    ax.axhline(
        THRESHOLD_R1,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.5,
        label=f"High Usage Threshold ({THRESHOLD_R1:.0f} W)"
    )

    ax.set_title(
        "Room 1 (Channel 1): Actual vs Random Forest Predicted Power\n"
        "(1-Hour Ahead Horizon)",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Test Telemetry Samples (Consecutive 10s intervals)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Power (W)", fontsize=11, fontweight="semibold")
    ax.set_ylim(-5, 95)
    ax.set_xlim(0, sample_n)

    metrics_text_1 = (
        f"Test Evaluation (N = {len(test_1h):,}):\n"
        f"MAE  = {mae1:.2f} W\n"
        f"RMSE = {rmse1:.2f} W\n"
        f"R²   = {r2_1:.4f}"
    )
    ax.text(
        0.02, 0.95,
        metrics_text_1,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95)
    )

    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="#888888")
    plt.tight_layout()

    file_r1 = OUTPUT_DIR / "graph2_room1_actual_vs_predicted.png"
    plt.savefig(file_r1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Graph 2 (Room 1) saved successfully to: {file_r1}")

    # -------------------------------------------------------------
    # 6. Plot Room 2 (Channel 2)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.plot(
        x_samples,
        p2_actual[:sample_n],
        label="Actual Power (t + 1h)",
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9
    )
    ax.plot(
        x_samples,
        p2_pred[:sample_n],
        label="Random Forest Predicted Power (t + 1h)",
        color="#2ca02c",
        linewidth=1.8,
        linestyle="--",
        alpha=0.9
    )
    ax.axhline(
        THRESHOLD_R2,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.5,
        label=f"High Usage Threshold ({THRESHOLD_R2:.0f} W)"
    )

    ax.set_title(
        "Room 2 (Channel 2): Actual vs Random Forest Predicted Power\n"
        "(1-Hour Ahead Horizon)",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Test Telemetry Samples (Consecutive 10s intervals)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Power (W)", fontsize=11, fontweight="semibold")
    ax.set_ylim(-2, 28)
    ax.set_xlim(0, sample_n)

    metrics_text_2 = (
        f"Test Evaluation (N = {len(test_1h):,}):\n"
        f"MAE  = {mae2:.2f} W\n"
        f"RMSE = {rmse2:.2f} W\n"
        f"R²   = {r2_2:.4f}"
    )
    ax.text(
        0.02, 0.95,
        metrics_text_2,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95)
    )

    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="#888888")
    plt.tight_layout()

    file_r2 = OUTPUT_DIR / "graph2_room2_actual_vs_predicted.png"
    plt.savefig(file_r2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Graph 2 (Room 2) saved successfully to: {file_r2}")

    return file_r1, file_r2


def main():
    generate_graph1()
    generate_graph2()


if __name__ == "__main__":
    main()