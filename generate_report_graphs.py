import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# VOLTIX - GRAPH 1
# Random Forest vs Persistence Baseline
# 1-Hour Classification Accuracy
# ============================================================

# Results from the actual model evaluation
rooms = ["Room 1", "Room 2"]

random_forest = [67.16, 72.04]
persistence = [61.17, 65.99]

x = np.arange(len(rooms))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

bars1 = ax.bar(
    x - width / 2,
    random_forest,
    width,
    label="Random Forest"
)

bars2 = ax.bar(
    x + width / 2,
    persistence,
    width,
    label="Persistence Baseline"
)

# Labels and title
ax.set_title(
    "Random Forest vs Persistence Baseline\n"
    "1-Hour High-Usage Classification Accuracy",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Load Channel")
ax.set_ylabel("Accuracy (%)")

ax.set_xticks(x)
ax.set_xticklabels(rooms)

ax.set_ylim(0, 100)

ax.legend()

# Add percentage values above bars
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
            fontsize=10
        )

ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

# Save graph
output_dir = Path("report_graphs")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "graph1_rf_vs_persistence_accuracy.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nGraph saved successfully:")
print(output_file)