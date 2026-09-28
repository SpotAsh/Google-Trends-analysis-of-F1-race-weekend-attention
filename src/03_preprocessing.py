"""
03_preprocessing.py

Merges races + trends, engineers search-interest features (baseline, peak,
spike factor, z-score) — same as before — and ADDS a new feature:
championship_gap_entering_race, the points gap between P1 and P2 in the
Drivers' Championship standings as of the round BEFORE this race.

Rationale: tests whether search-interest spikes correlate with how close
the title fight is. 2022 (Verstappen won 19/22 races, decided by August) is
included specifically as a contrast case against tighter fights like late
2024 (Norris vs Verstappen).

Standings come from the Jolpica-F1 API (https://api.jolpi.ca/ergast/f1/),
a free, actively maintained mirror of the Ergast API's interface (Ergast
itself was retired in 2024). This is far more reliable than trying to
derive cumulative points from Wikipedia's race-by-race finishing-position
grids by hand (which would require correctly modeling the points system,
sprint weekends, fastest-lap bonus points, DNFs, etc.)
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

races_df = pd.read_csv(RAW_DIR / "races.csv")
trends_df = pd.read_csv(PROCESSED_DIR / "trends_master.csv")

print("Preprocessing and Feature Engineering pipeline initialized...")

# --- Clean races.csv footnotes & metadata rows (same cleaning as 02) ---
races_df = races_df[
    ~races_df["Round"].astype(str).str.contains("Source", case=False, na=False)
].copy()
races_df["round_clean"] = races_df["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
races_df["date_clean"] = races_df["Race date"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
races_df["round_num"] = pd.to_numeric(races_df["round_clean"], errors="coerce").astype("Int64")
races_df = races_df.dropna(subset=["round_num"]).copy()
races_df["race_id"] = races_df["season"].astype(str) + "_R" + races_df["round_num"].astype(str)
races_df["full_date_str"] = races_df["date_clean"] + " " + races_df["season"].astype(str)
races_df["race_date"] = pd.to_datetime(races_df["full_date_str"], format="%d %B %Y", errors="coerce")

trends_df["date"] = pd.to_datetime(trends_df["date"], errors="coerce")

# --- Merge datasets on race_id ---
merged_df = pd.merge(
    trends_df,
    races_df[["race_id", "Grand Prix", "Circuit", "race_date", "round_num"]],
    on="race_id",
    how="inner",
)

if merged_df.empty:
    raise ValueError("[ERROR] Merge resulted in 0 rows! Check race_id formatting.")

print(f"Successfully merged datasets! ({len(merged_df)} daily trend records matched across "
      f"{merged_df['race_id'].nunique()} race events).")

# --- Days relative to race date, analytical windows ---
merged_df["days_to_race"] = (merged_df["date"] - merged_df["race_date"]).dt.days
merged_df["is_race_week"] = merged_df["days_to_race"].between(-7, 0)
merged_df["is_baseline_period"] = merged_df["days_to_race"].between(-60, -14)

# --- Extract search-interest features per race event ---
features_list = []

for race_id, group in merged_df.groupby("race_id"):
    season = group["season"].iloc[0]
    round_num = int(group["round_num"].iloc[0])
    grand_prix = group["grand_prix"].iloc[0]
    circuit = group["Circuit"].iloc[0] if "Circuit" in group.columns else "N/A"
    race_date = group["race_date"].iloc[0]

    baseline_search = group[group["is_baseline_period"]]["search_interest"].mean()
    race_week_peak = group[group["is_race_week"]]["search_interest"].max()
    race_week_mean = group[group["is_race_week"]]["search_interest"].mean()

    baseline_val = baseline_search if pd.notna(baseline_search) and baseline_search > 0 else 1.0
    spike_factor = race_week_peak / baseline_val if pd.notna(race_week_peak) else 0.0

    features_list.append({
        "race_id": race_id,
        "season": season,
        "round_num": round_num,
        "grand_prix": grand_prix,
        "circuit": circuit,
        "race_date": race_date,
        "baseline_search_mean": round(float(baseline_val), 2),
        "race_week_peak": int(race_week_peak) if pd.notna(race_week_peak) else 0,
        "race_week_mean": round(float(race_week_mean) if pd.notna(race_week_mean) else 0.0, 2),
        "spike_factor": round(float(spike_factor), 2),
    })

features_df = pd.DataFrame(features_list)

# --- Z-score + anomaly candidate flag (unchanged logic) ---
spike_mean = features_df["spike_factor"].mean()
spike_std = features_df["spike_factor"].std()
features_df["spike_zscore"] = (
    (features_df["spike_factor"] - spike_mean) / (spike_std if spike_std > 0 else 1.0)
).round(2)
features_df["is_anomaly_candidate"] = features_df["spike_zscore"] > 1.5


# --- NEW: championship points-gap feature via Jolpica-F1 API ---
def get_championship_gap(season: int, round_num: int) -> float | None:
    """Points gap between P1 and P2 in the Drivers' Championship as of the
    round BEFORE this race. Round 1 of a season has no gap yet (preseason),
    returns None — document this as expected missingness for season openers,
    not a data quality issue."""
    if round_num <= 1:
        return None

    url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_num - 1}/driverStandings.json"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [WARNING] Standings fetch failed for {season} round {round_num - 1}: {e}")
        return None

    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return None

    standings = lists[0].get("DriverStandings", [])
    if len(standings) < 2:
        return None

    p1_pts = float(standings[0]["points"])
    p2_pts = float(standings[1]["points"])
    return round(p1_pts - p2_pts, 1)


print("\nFetching championship standings gap per race (Jolpica-F1 API)...")
gaps = []
for _, row in features_df.iterrows():
    gap = get_championship_gap(int(row["season"]), int(row["round_num"]))
    gaps.append(gap)
    print(f"  {row['race_id']} ({row['grand_prix']}): gap entering race = {gap}")
    time.sleep(0.5)  # be polite to the API

features_df["championship_gap_entering_race"] = gaps

# Binary flag: "close championship" if gap is under ~25 points (roughly the
# value of a single race win + fastest lap) — i.e. still very plausible for
# the standings to flip on the next race's result. Season-opener races
# (gap=None) are left as NaN, not coerced to True/False, since "close" is
# undefined before any points have been scored.
CLOSE_CHAMPIONSHIP_THRESHOLD = 25
features_df["is_close_championship_race"] = features_df["championship_gap_entering_race"].apply(
    lambda g: (g < CLOSE_CHAMPIONSHIP_THRESHOLD) if pd.notna(g) else np.nan
)

# --- Export ---
output_file = PROCESSED_DIR / "f1_anomalies_features.csv"
features_df.to_csv(output_file, index=False)

print(f"\nSUCCESS! Created engineered feature matrix for {len(features_df)} race events.")
print(f"File saved to: {output_file.resolve()}\n")

print("Top 5 Search Spike Anomalies:")
print(features_df.sort_values(by="spike_factor", ascending=False)[
    ["race_id", "grand_prix", "spike_factor", "spike_zscore"]
].head(5))

print("\nChampionship gap coverage:")
print(f"  Races with a computed gap: {features_df['championship_gap_entering_race'].notna().sum()} / {len(features_df)}")
print(f"  Races flagged as 'close championship' (gap < {CLOSE_CHAMPIONSHIP_THRESHOLD}): "
      f"{(features_df['is_close_championship_race'] == True).sum()}")

# Quick directional check (not a substitute for the formal FPM step later,
# just a sanity check that the feature has real signal worth pursuing)
valid = features_df.dropna(subset=["championship_gap_entering_race"])
if len(valid) > 2:
    corr = valid["championship_gap_entering_race"].corr(valid["spike_factor"])
    print(f"\n  Pearson correlation (championship_gap vs spike_factor): {corr:.3f}")
    print("  (Negative = closer championships associated with bigger search spikes, as hypothesized)")