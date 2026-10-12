"""
05_visualize_anomalies.py

Original scatter + bar chart, plus a NEW plot: championship_gap_entering_race
vs spike_factor, colored by season — this is the direct visual test of the
"does championship closeness drive search interest" hypothesis, with 2022
(blowout season) as a visual contrast point against tighter seasons.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "f1_anomalies_detected.csv"

print("Loading anomaly detection results...")
df = pd.read_csv(INPUT_FILE)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

# --- Plot 1: Search Intensity vs. Spike Factor (unchanged) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df[~df["is_high_confidence_anomaly"]],
    x="race_week_peak", y="spike_factor",
    color="gray", alpha=0.6, s=70, label="Standard Event",
)
anomalies = df[df["is_high_confidence_anomaly"]]
sns.scatterplot(
    data=anomalies, x="race_week_peak", y="spike_factor",
    color="red", s=150, edgecolor="black", linewidth=1.5, label="High-Confidence Anomaly",
)
for idx, (_, row) in enumerate(anomalies.iterrows()):
    plt.text(
        row["race_week_peak"] + 1.0,
        row["spike_factor"] + (150 if idx % 2 == 0 else -250),
        f"{row['grand_prix']} ({row['season']})",
        fontsize=10, fontweight="bold", color="darkred",
    )
plt.title("F1 Race Search Spikes vs. Peak Interest (2016–2025, excl. 2020)", fontsize=14, pad=15)
plt.xlabel("Race Week Peak Search Volume", fontsize=12)
plt.ylabel("Spike Factor (Peak / Baseline)", fontsize=12)
plt.legend(loc="upper left")
plt.tight_layout()
scatter_path = FIGURES_DIR / "search_spike_anomalies.png"
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved scatter plot to: {scatter_path.resolve()}")

# --- Plot 2: Model detection breakdown (unchanged) ---
model_counts = {
    "Z-Score (>1.5)": df["is_anomaly_candidate"].sum(),
    "Isolation Forest": df["is_iso_anomaly"].sum(),
    "Local Outlier Factor": df["is_lof_anomaly"].sum(),
    "Consensus Anomalies": df["is_high_confidence_anomaly"].sum(),
}
counts_df = pd.DataFrame(list(model_counts.items()), columns=["Model", "Detections"])
plt.figure(figsize=(8, 5))
palette = ["#4c72b0", "#4c72b0", "#4c72b0", "#c44e52"]
ax = sns.barplot(data=counts_df, x="Model", y="Detections", hue="Model", palette=palette, legend=False)
for p in ax.patches:
    height = p.get_height()
    if pd.notna(height) and height > 0:
        ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="center", xytext=(0, 8), textcoords="offset points", fontweight="bold")
plt.title("Detected Anomalies by Model", fontsize=14, pad=15)
plt.ylabel("Count of Flagged Events", fontsize=12)
plt.xlabel("")
plt.ylim(0, max(model_counts.values()) + 2)
plt.tight_layout()
bar_path = FIGURES_DIR / "model_detection_comparison.png"
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"Saved model comparison plot to: {bar_path.resolve()}")

# --- Plot 3 (NEW): Championship gap vs spike factor, by season ---
valid = df.dropna(subset=["championship_gap_entering_race"])
if not valid.empty:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=valid, x="championship_gap_entering_race", y="spike_factor",
        hue="season", palette="viridis", s=90, alpha=0.8,
    )
    sns.regplot(
        data=valid, x="championship_gap_entering_race", y="spike_factor",
        scatter=False, color="black", line_kws={"linestyle": "--", "linewidth": 1.5},
    )
    plt.title("Championship Points Gap vs. Search Spike Factor", fontsize=14, pad=15)
    plt.xlabel("Points Gap Between P1/P2 Entering Race (lower = closer title fight)", fontsize=11)
    plt.ylabel("Spike Factor (Peak / Baseline)", fontsize=12)
    plt.legend(title="Season", loc="upper right")
    plt.tight_layout()
    champ_path = FIGURES_DIR / "championship_gap_vs_spike.png"
    plt.savefig(champ_path, dpi=300)
    plt.close()
    print(f"Saved championship gap vs. spike plot to: {champ_path.resolve()}")

    corr = valid["championship_gap_entering_race"].corr(valid["spike_factor"])
    print(f"\nPearson correlation (championship_gap vs spike_factor): {corr:.3f}")
    print("(Negative = closer championships associated with bigger search spikes)")

    print("\nMean spike_factor: close-championship races vs. not:")
    print(valid.groupby("is_close_championship_race")["spike_factor"].agg(["mean", "median", "count"]))
else:
    print("\n[WARNING] No races have a valid championship_gap_entering_race — skipping Plot 3.")

print("\nVisualization generation complete!")