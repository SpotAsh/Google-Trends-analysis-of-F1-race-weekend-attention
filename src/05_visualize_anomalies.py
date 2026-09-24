from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Directory Setup
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "f1_anomalies_detected.csv"

print("Loading anomaly detection results...")
df = pd.read_csv(INPUT_FILE)

# Set global plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

# 2. Scatter Plot: Search Intensity vs. Spike Factor
plt.figure(figsize=(10, 6))

# Non-anomalies
sns.scatterplot(
    data=df[~df["is_high_confidence_anomaly"]],
    x="race_week_peak",
    y="spike_factor",
    color="gray",
    alpha=0.6,
    s=70,
    label="Standard Event",
)

# Consensus Anomalies
anomalies = df[df["is_high_confidence_anomaly"]]
sns.scatterplot(
    data=anomalies,
    x="race_week_peak",
    y="spike_factor",
    color="red",
    s=150,
    edgecolor="black",
    linewidth=1.5,
    label="High-Confidence Anomaly",
)

# Label top anomalies cleanly with explicit offsets to prevent overlapping text
y_offsets = [150, 150, -250]  # Stagger vertically for overlapping points
x_offsets = [1.0, 1.0, 1.0]

for idx, (_, row) in enumerate(anomalies.iterrows()):
    plt.text(
        row["race_week_peak"] + x_offsets[idx % len(x_offsets)],
        row["spike_factor"] + y_offsets[idx % len(y_offsets)],
        f"{row['grand_prix']} ({row['season']})",
        fontsize=10,
        fontweight="bold",
        color="darkred",
    )

plt.title(
    "F1 Race Search Spikes vs. Peak Interest (2023–2025)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Race Week Peak Search Volume", fontsize=12)
plt.ylabel("Spike Factor (Peak / Baseline)", fontsize=12)
plt.legend(loc="upper left")
plt.tight_layout()

scatter_path = FIGURES_DIR / "search_spike_anomalies.png"
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved cleanly formatted scatter plot to: {scatter_path.resolve()}")

# 3. Bar Chart: Anomaly Model Breakdown (Fixed Seaborn syntax warning)
model_counts = {
    "Z-Score (>1.5)": df["is_anomaly_candidate"].sum(),
    "Isolation Forest": df["is_iso_anomaly"].sum(),
    "Local Outlier Factor": df["is_lof_anomaly"].sum(),
    "Consensus Anomalies": df["is_high_confidence_anomaly"].sum(),
}

counts_df = pd.DataFrame(
    list(model_counts.items()), columns=["Model", "Detections"]
)

plt.figure(figsize=(8, 5))
palette = ["#4c72b0", "#4c72b0", "#4c72b0", "#c44e52"]
ax = sns.barplot(
    data=counts_df,
    x="Model",
    y="Detections",
    hue="Model",
    palette=palette,
    legend=False,
)

# Annotate values on top of bars
for p in ax.patches:
    height = p.get_height()
    if pd.notna(height) and height > 0:
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 8),
            textcoords="offset points",
            fontweight="bold",
        )

plt.title("Detected Anomalies by Model", fontsize=14, pad=15)
plt.ylabel("Count of Flagged Events", fontsize=12)
plt.xlabel("")
plt.ylim(0, max(model_counts.values()) + 2)
plt.tight_layout()

bar_path = FIGURES_DIR / "model_detection_comparison.png"
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"Saved model comparison plot to: {bar_path.resolve()}")

print("\nVisualization generation complete!")