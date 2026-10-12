"""
08_descriptive_stats_and_normalization.py

Covers three rubric items:
1. Descriptive statistics of each variable.
2. Visualization of key variables (histograms, scatter, categorical tables)
   — general exploratory visuals, distinct from the anomaly-specific plots
   in 05_visualize_anomalies.py.
3. Normalization/transformation — log-transforms the heavily right-skewed
   spike_factor, and re-checks the championship_gap correlation on the
   transformed variable (closing the loop flagged earlier: raw correlation
   was ~0 but spike_factor's median was 0 across both groups, suggesting
   the raw correlation may be misleading).

Produces the final enriched feature table used downstream for binary
variables + frequent pattern mining.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11})

# --- Load and merge: features + Airbnb (race-level join) ---
features_df = pd.read_csv(PROCESSED_DIR / "f1_anomalies_detected.csv")
airbnb_df = pd.read_csv(RAW_DIR / "airbnb_raw.csv")

df = features_df.merge(
    airbnb_df[["race_id", "has_listings_data", "airbnb_price_snapshot_mean_usd", "airbnb_listing_count"]],
    on="race_id", how="left"
)

# Airbnb data only exists for 2022-2025 (06 was run on those races only), so
# races from earlier seasons come out of the left join as NaN. That is correct:
# a 2025 price snapshot must not be attached to a 2016 race.
df["has_listings_data"] = df["has_listings_data"].fillna(False).astype(bool)

# Optional covariate from 03; synthesize if an older features file is used.
if "is_covid_season" not in df.columns:
    df["is_covid_season"] = df["season"].astype(int) == 2020
else:
    df["is_covid_season"] = df["is_covid_season"].fillna(False).astype(bool)

print(f"Combined dataset: {len(df)} races, {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# PART 1: Descriptive statistics
# ---------------------------------------------------------------------------
numeric_cols = [
    "baseline_search_mean", "race_week_peak", "race_week_mean", "spike_factor",
    "spike_zscore", "championship_gap_entering_race", "airbnb_price_snapshot_mean_usd",
    "airbnb_listing_count",
]
desc_stats = df[numeric_cols].describe().T
desc_stats_out = REPORTS_DIR / "descriptive_stats.csv"
desc_stats.to_csv(desc_stats_out)
print(f"\nDescriptive statistics saved to: {desc_stats_out.resolve()}")
print(desc_stats)

categorical_cols = [
    "season", "is_close_championship_race", "is_high_confidence_anomaly",
    "has_listings_data", "is_covid_season",
]
cat_summaries = {}
for col in categorical_cols:
    cat_summaries[col] = df[col].value_counts(dropna=False)

cat_out = REPORTS_DIR / "categorical_counts.csv"
with open(cat_out, "w") as f:
    for col, counts in cat_summaries.items():
        f.write(f"\n--- {col} ---\n")
        counts.to_csv(f)
print(f"Categorical counts saved to: {cat_out.resolve()}")
for col, counts in cat_summaries.items():
    print(f"\n{col}:\n{counts}")

# ---------------------------------------------------------------------------
# PART 2: Visualization of key variables
# ---------------------------------------------------------------------------

# Histogram: spike_factor (raw) — heavily right-skewed, log-scale x-axis
plt.figure(figsize=(9, 5))
sns.histplot(df["spike_factor"], bins=30, log_scale=(True, False))
plt.title("Distribution of Spike Factor (log-scale x-axis)")
plt.xlabel("Spike Factor (Peak / Baseline search interest)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hist_spike_factor.png", dpi=300)
plt.close()

# Histogram: championship_gap_entering_race
plt.figure(figsize=(9, 5))
sns.histplot(df["championship_gap_entering_race"].dropna(), bins=25)
plt.title("Distribution of Championship Points Gap Entering Race")
plt.xlabel("Points Gap (P1 - P2)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hist_championship_gap.png", dpi=300)
plt.close()

# Scatter: baseline_search_mean vs race_week_peak, colored by season
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="baseline_search_mean", y="race_week_peak", hue="season", palette="viridis", s=70, alpha=0.8)
plt.title("Baseline Search Interest vs. Race-Week Peak")
plt.xlabel("Baseline Search Mean (offseason)")
plt.ylabel("Race-Week Peak Search Interest")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "scatter_baseline_vs_peak.png", dpi=300)
plt.close()

# Table/bar: race counts by season
plt.figure(figsize=(7, 5))
df["season"].value_counts().sort_index().plot(kind="bar", color="#4c72b0")
plt.title("Race Count by Season")
plt.xlabel("Season")
plt.ylabel("Number of Races")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "bar_races_by_season.png", dpi=300)
plt.close()

# Histogram: Airbnb price snapshot (USD), where available
valid_airbnb = df.dropna(subset=["airbnb_price_snapshot_mean_usd"])
if not valid_airbnb.empty:
    plt.figure(figsize=(9, 5))
    sns.histplot(valid_airbnb.drop_duplicates(subset=["circuit"])["airbnb_price_snapshot_mean_usd"], bins=15)
    plt.title("Distribution of Avg. Airbnb Price Snapshot by City (USD)")
    plt.xlabel("Mean Price (USD)")
    plt.ylabel("Count of Cities")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hist_airbnb_price.png", dpi=300)
    plt.close()

print(f"\nSaved 5 exploratory figures to: {FIGURES_DIR.resolve()}")

# ---------------------------------------------------------------------------
# PART 3: Normalization / transformation
# ---------------------------------------------------------------------------
# spike_factor is heavily right-skewed (median=0 in both championship-gap
# groups per the earlier correlation check, despite means in the 80-100
# range) — log1p transform compresses the extreme outliers (Japan 2022 at
# 2350x) so distributional analysis and correlation checks aren't dominated
# by a handful of extreme values.
df["spike_factor_log"] = np.log1p(df["spike_factor"])
df["baseline_search_mean_log"] = np.log1p(df["baseline_search_mean"])
df["race_week_peak_log"] = np.log1p(df["race_week_peak"])

# Min-max scale championship gap (kept as a separate column; raw gap is
# still used for the "is_close" threshold flag already computed in 03)
gap_min = df["championship_gap_entering_race"].min()
gap_max = df["championship_gap_entering_race"].max()
gap_span = gap_max - gap_min
df["championship_gap_minmax"] = (
    (df["championship_gap_entering_race"] - gap_min) / gap_span
    if pd.notna(gap_span) and gap_span != 0
    else np.nan
)

print("\n" + "=" * 70)
print("RE-CHECKING CHAMPIONSHIP GAP CORRELATION ON LOG-TRANSFORMED SPIKE FACTOR")
print("=" * 70)
valid = df.dropna(subset=["championship_gap_entering_race"])
corr_raw = valid["championship_gap_entering_race"].corr(valid["spike_factor"])
corr_log = valid["championship_gap_entering_race"].corr(valid["spike_factor_log"])
print(f"Correlation (raw spike_factor):          r = {corr_raw:.3f}")
print(f"Correlation (log1p-transformed spike_factor): r = {corr_log:.3f}")
print("(Negative = closer championships associated with bigger search spikes, as hypothesized)")

plt.figure(figsize=(9, 6))
sns.scatterplot(data=valid, x="championship_gap_entering_race", y="spike_factor_log", hue="season", palette="viridis", s=80, alpha=0.8)
sns.regplot(data=valid, x="championship_gap_entering_race", y="spike_factor_log", scatter=False, color="black", line_kws={"linestyle": "--"})
plt.title(f"Championship Gap vs. log(Spike Factor) — r = {corr_log:.3f}")
plt.xlabel("Points Gap Entering Race")
plt.ylabel("log(1 + Spike Factor)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "championship_gap_vs_log_spike.png", dpi=300)
plt.close()
print(f"\nSaved updated correlation plot to: {(FIGURES_DIR / 'championship_gap_vs_log_spike.png').resolve()}")

# --- Export final enriched table for the binary-variable / FPM stage ---
output_file = PROCESSED_DIR / "f1_final_features.csv"
df.to_csv(output_file, index=False)
print(f"\nSUCCESS! Final enriched feature table saved to: {output_file.resolve()}")
print(f"Columns: {list(df.columns)}")