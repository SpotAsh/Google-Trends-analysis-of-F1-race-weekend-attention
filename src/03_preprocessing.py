"""
03_preprocessing.py

Merges races + trends, engineers search-interest features (baseline, peak,
spike factor, z-score) — same as before — and ADDS:
championship_gap_entering_race (P1-P2 gap entering the race) and
is_covid_season (True for 2020, so models can separate pandemic-era
behind-closed-doors weekends from other years).

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
# Normalize whitespace first (Wikipedia can emit non-breaking spaces, which a
# strict format string silently rejects), then parse strictly, then fall back
# to pandas' lenient parser (the same one 02_pull_trends.py uses).
races_df["full_date_str"] = (
    races_df["full_date_str"]
    .str.replace(" ", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
races_df["race_date"] = pd.to_datetime(races_df["full_date_str"], format="%d %B %Y", errors="coerce")
_lenient = pd.to_datetime(races_df["full_date_str"], errors="coerce")
races_df["race_date"] = races_df["race_date"].fillna(_lenient)

# FAIL LOUDLY: a NaT race_date silently empties that race's race-week and
# baseline windows, which turns its spike_factor into 0 and corrupts everything
# downstream (anomalies, correlations, FPM).
_bad = races_df[races_df["race_date"].isna()]
if not _bad.empty:
    raise ValueError(
        f"{len(_bad)} race(s) have an unparseable race_date:\n"
        + _bad[["race_id", "Race date", "full_date_str"]].to_string(index=False)
    )

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
        "is_covid_season": int(season) == 2020,
        "baseline_search_mean": round(float(baseline_val), 2),
        "race_week_peak": int(race_week_peak) if pd.notna(race_week_peak) else 0,
        "race_week_mean": round(float(race_week_mean) if pd.notna(race_week_mean) else 0.0, 2),
        "spike_factor": round(float(spike_factor), 2),
    })

features_df = pd.DataFrame(features_list)

# SANITY GUARD: every race should have a non-zero race-week peak, because each
# Trends pull is scaled so its own busiest day is 100. If most races show a peak
# of 0, the date windows are broken, not the data.
_zero_share = (features_df["race_week_peak"] == 0).mean()
print(f"Races with race_week_peak == 0: {(features_df['race_week_peak'] == 0).sum()} / {len(features_df)}")
if _zero_share > 0.25:
    raise ValueError(
        f"{_zero_share:.0%} of races have a race-week peak of 0 — the race-week window is empty "
        "for most races. Check race_date parsing and days_to_race before continuing."
    )

# --- Z-score + anomaly candidate flag (unchanged logic) ---
spike_mean = features_df["spike_factor"].mean()
spike_std = features_df["spike_factor"].std()
features_df["spike_zscore"] = (
    (features_df["spike_factor"] - spike_mean) / (spike_std if spike_std > 0 else 1.0)
).round(2)
features_df["is_anomaly_candidate"] = features_df["spike_zscore"] > 1.5
features_df["is_covid_season"] = features_df["season"].astype(int) == 2020


# --- NEW: championship points-gap feature via Jolpica-F1 API ---
# Standings are cached on disk: with ~200 races the API's hourly rate limit is
# easy to hit on a rerun, and failures are NOT cached so a rerun fills them in.
GAP_CACHE_FILE = RAW_DIR / "standings_gap_cache.csv"
if GAP_CACHE_FILE.exists():
    _c = pd.read_csv(GAP_CACHE_FILE)
    GAP_CACHE = {
        (int(r.season), int(r.round)): float(r.gap)
        for r in _c.itertuples()
        if pd.notna(r.gap)
    }
else:
    GAP_CACHE = {}


def _save_gap_cache():
    pd.DataFrame(
        [{"season": s, "round": r, "gap": g} for (s, r), g in sorted(GAP_CACHE.items())]
    ).to_csv(GAP_CACHE_FILE, index=False)


def _parse_gap_from_json(data):
    """Defensive Jolpica/Ergast parser: empty lists and missing points must
    not raise. 2020's rescheduled calendar can yield a 200 with no StandingsLists
    for a round that never ran."""
    if not isinstance(data, dict):
        return None
    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists") or []
    if not lists:
        return None
    standings = lists[0].get("DriverStandings") or []
    if len(standings) < 2:
        return None
    try:
        p1_pts = float(standings[0].get("points"))
        p2_pts = float(standings[1].get("points"))
    except (TypeError, ValueError, KeyError):
        return None
    if pd.isna(p1_pts) or pd.isna(p2_pts):
        return None
    return round(p1_pts - p2_pts, 1)


def _fetch_standings_json(season: int, standing_round: int):
    url = f"https://api.jolpi.ca/ergast/f1/{season}/{standing_round}/driverStandings.json"
    for attempt in range(1, 5):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 429:
                wait = 10 * attempt
                print(f"    [RATE LIMIT] {season} round {standing_round}; waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"    [WARNING] Standings fetch failed for {season} round {standing_round} "
                  f"(attempt {attempt}/4): {e}")
            time.sleep(3 * attempt)
    return None


def get_championship_gap(season: int, round_num: int) -> float | None:
    """Points gap between P1 and P2 in the Drivers' Championship as of the
    round BEFORE this race. Round 1 of a season has no gap yet (preseason),
    returns None — document this as expected missingness for season openers,
    not a data quality issue.

    If round_num-1 has no standings (cancelled/rescheduled 2020 round), walk
    backward until a populated StandingsList is found."""
    if round_num <= 1:
        return None
    key = (season, round_num)
    if key in GAP_CACHE:
        return GAP_CACHE[key]

    gap = None
    for standing_round in range(round_num - 1, 0, -1):
        data = _fetch_standings_json(season, standing_round)
        gap = _parse_gap_from_json(data)
        if gap is not None:
            break
        print(f"    [INFO] No standings at {season} round {standing_round}; "
              "trying an earlier round (possible 2020 cancellation/reschedule).")
        time.sleep(0.5)

    if gap is None:
        return None
    GAP_CACHE[key] = gap
    _save_gap_cache()
    return gap


print("\nFetching championship standings gap per race (Jolpica-F1 API, cached)...")
gaps = []
for _, row in features_df.iterrows():
    cached = (int(row["season"]), int(row["round_num"])) in GAP_CACHE
    gap = get_championship_gap(int(row["season"]), int(row["round_num"]))
    gaps.append(gap)
    print(f"  {row['race_id']} ({row['grand_prix']}): gap entering race = {gap}")
    if not cached and int(row["round_num"]) > 1:
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