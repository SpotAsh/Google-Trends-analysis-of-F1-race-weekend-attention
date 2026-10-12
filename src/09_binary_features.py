"""
09_binary_features.py

Builds the binary variable table for frequent pattern mining:
- One-hot encodes categorical variables (season, circuit_type).
- Discretizes continuous variables into binary flags via thresholds.
"""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")

df = pd.read_csv(PROCESSED_DIR / "f1_final_features.csv")
print(f"Loaded {len(df)} races, {len(df.columns)} columns")

if "is_covid_season" not in df.columns:
    df["is_covid_season"] = df["season"].astype(int) == 2020
df["is_covid_season"] = df["is_covid_season"].fillna(False).astype(bool)

# --- Circuit type (street vs. permanent) ---
# Hardcoded mapping — only ~25 unique circuits across the dataset, derived
# from each circuit's Wikipedia infobox classification.
# Keyword match instead of exact strings: older Wikipedia calendar tables carry
# country-code prefixes ("AZE Baku City Circuit, Baku"), so exact matching would
# silently mislabel 2016-2021 street circuits as permanent. Hand-coded list:
# Baku, Monaco, Singapore (Marina Bay), Jeddah, Miami, Las Vegas.
STREET_KEYWORDS = ["baku", "monaco", "marina bay", "jeddah", "miami", "las vegas", "street circuit"]
df["circuit_type"] = df["circuit"].astype(str).str.lower().apply(
    lambda c: "street" if any(k in c for k in STREET_KEYWORDS) else "permanent"
)
print("Street-circuit races by circuit string (verify this list is right):")
print(df[df["circuit_type"] == "street"]["circuit"].value_counts().to_string())

# --- One-hot encoding ---
one_hot_cols = ["season", "circuit_type"]
df_encoded = pd.get_dummies(df, columns=one_hot_cols, prefix=one_hot_cols)

# --- Discretized binary flags from continuous variables ---
median_spike = df["spike_factor"].median()
median_baseline = df["baseline_search_mean"].median()
median_airbnb_price = df["airbnb_price_snapshot_mean_usd"].median()
median_listing_count = df["airbnb_listing_count"].median()

df_encoded["is_spike_above_median"] = df["spike_factor"] > median_spike
df_encoded["is_high_baseline_city"] = df["baseline_search_mean"] > median_baseline
df_encoded["has_airbnb_data"] = df["has_listings_data"].fillna(False).astype(bool)
df_encoded["is_high_airbnb_price"] = (
    df["airbnb_price_snapshot_mean_usd"] > median_airbnb_price
).fillna(False)
df_encoded["is_high_listing_count"] = (
    df["airbnb_listing_count"] > median_listing_count
).fillna(False)

# Already boolean/derived upstream — carried through as-is:
# is_anomaly_candidate, is_iso_anomaly, is_lof_anomaly,
# is_high_confidence_anomaly, is_close_championship_race, is_covid_season

# --- Assemble final binary table ---
binary_cols = [
    "race_id",  # kept for traceability, dropped before FPM itself
    "is_spike_above_median",
    "is_high_baseline_city",
    "is_anomaly_candidate",
    "is_iso_anomaly",
    "is_lof_anomaly",
    "is_high_confidence_anomaly",
    "is_close_championship_race",
    "is_covid_season",
    "has_airbnb_data",
    "is_high_airbnb_price",
    "is_high_listing_count",
] + [c for c in df_encoded.columns if c.startswith("season_") or c.startswith("circuit_type_")]

binary_df = df_encoded[binary_cols].copy()

# is_close_championship_race has real NaNs (season-openers) — for FPM
# purposes, treat NaN as False (not a close championship, since there's no
# standings yet to indicate tightness). Document this choice explicitly.
binary_df["is_close_championship_race"] = binary_df["is_close_championship_race"].fillna(False)

# Ensure all non-ID columns are proper booleans (0/1), not mixed types
for col in binary_df.columns:
    if col != "race_id":
        binary_df[col] = binary_df[col].astype(bool)

print(f"\nFinal binary table: {len(binary_df)} rows, {len(binary_df.columns) - 1} binary columns")
print(f"Columns: {[c for c in binary_df.columns if c != 'race_id']}")

print("\nColumn value counts (True share):")
for col in binary_df.columns:
    if col != "race_id":
        print(f"  {col}: {binary_df[col].mean():.1%} True")

output_file = PROCESSED_DIR / "f1_binary_features.csv"
binary_df.to_csv(output_file, index=False)
print(f"\nSUCCESS! Saved binary feature table to: {output_file.resolve()}")