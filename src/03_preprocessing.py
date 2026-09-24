import re
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Directory Paths
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

# Load datasets
races_df = pd.read_csv(RAW_DIR / "races.csv")
trends_df = pd.read_csv(PROCESSED_DIR / "trends_master.csv")

print("Preprocessing and Feature Engineering pipeline initialized...")

# 2. Clean 'races.csv' Footnotes & Metadata Rows
# Filter out Wikipedia source footer rows
races_df = races_df[
    ~races_df["Round"].astype(str).str.contains("Source", case=False, na=False)
].copy()

# Strip Wikipedia footnote brackets (e.g. "[d]", "[e]")
races_df["round_clean"] = (
    races_df["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
)
races_df["date_clean"] = (
    races_df["Race date"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
)

# Convert Round to integer
races_df["round_num"] = pd.to_numeric(
    races_df["round_clean"], errors="coerce"
).astype("Int64")
races_df = races_df.dropna(subset=["round_num"]).copy()

# Construct standard 'race_id' matching trends_master.csv (e.g. '2023_R1')
races_df["race_id"] = (
    races_df["season"].astype(str) + "_R" + races_df["round_num"].astype(str)
)

# Combine Date string + Season year to create datetime timestamp
races_df["full_date_str"] = (
    races_df["date_clean"] + " " + races_df["season"].astype(str)
)
races_df["race_date"] = pd.to_datetime(
    races_df["full_date_str"], format="%d %B %Y", errors="coerce"
)

# Clean dates in trends_master.csv
trends_df["date"] = pd.to_datetime(trends_df["date"], errors="coerce")

# 3. Merge Datasets on 'race_id'
merged_df = pd.merge(
    trends_df,
    races_df[["race_id", "Grand Prix", "Circuit", "race_date"]],
    on="race_id",
    how="inner",
)

if merged_df.empty:
    raise ValueError(
        "[ERROR] Merge resulted in 0 rows! Check race_id formatting."
    )

print(
    f"Successfully merged datasets! ({len(merged_df)} daily trend records matched across {merged_df['race_id'].nunique()} race events)."
)

# 4. Calculate Days Relative to Race Date
merged_df["days_to_race"] = (
    merged_df["date"] - merged_df["race_date"]
).dt.days

# Define analytical windows
merged_df["is_race_week"] = merged_df["days_to_race"].between(-7, 0)
merged_df["is_baseline_period"] = merged_df["days_to_race"].between(-60, -14)

# 5. Extract Features per Race Event
features_list = []

for race_id, group in merged_df.groupby("race_id"):
    season = group["season"].iloc[0]
    grand_prix = group["grand_prix"].iloc[0]
    circuit = group["Circuit"].iloc[0] if "Circuit" in group.columns else "N/A"

    # Baseline & Peak metrics
    baseline_search = group[group["is_baseline_period"]][
        "search_interest"
    ].mean()
    race_week_peak = group[group["is_race_week"]]["search_interest"].max()
    race_week_mean = group[group["is_race_week"]]["search_interest"].mean()

    # Prevent division by zero
    baseline_val = (
        baseline_search
        if pd.notna(baseline_search) and baseline_search > 0
        else 1.0
    )
    spike_factor = (
        race_week_peak / baseline_val if pd.notna(race_week_peak) else 0.0
    )

    features_list.append(
        {
            "race_id": race_id,
            "season": season,
            "grand_prix": grand_prix,
            "circuit": circuit,
            "baseline_search_mean": round(float(baseline_val), 2),
            "race_week_peak": (
                int(race_week_peak) if pd.notna(race_week_peak) else 0
            ),
            "race_week_mean": round(
                float(race_week_mean) if pd.notna(race_week_mean) else 0.0, 2
            ),
            "spike_factor": round(float(spike_factor), 2),
        }
    )

features_df = pd.DataFrame(features_list)

# 6. Statistical Anomaly Normalization (Z-Score)
spike_mean = features_df["spike_factor"].mean()
spike_std = features_df["spike_factor"].std()

features_df["spike_zscore"] = (
    (features_df["spike_factor"] - spike_mean)
    / (spike_std if spike_std > 0 else 1.0)
).round(2)

# Binary Anomaly Flag (Z-score > 1.5)
features_df["is_anomaly_candidate"] = features_df["spike_zscore"] > 1.5

# 7. Export Processed Feature Dataset
output_file = PROCESSED_DIR / "f1_anomalies_features.csv"
features_df.to_csv(output_file, index=False)

print(
    f"\nSUCCESS! Created engineered feature matrix for {len(features_df)} race events."
)
print(f"File saved to: {output_file.resolve()}\n")

print("Top 5 Search Spike Anomalies:")
print(
    features_df.sort_values(by="spike_factor", ascending=False)[
        ["race_id", "grand_prix", "spike_factor", "spike_zscore"]
    ].head(5)
)