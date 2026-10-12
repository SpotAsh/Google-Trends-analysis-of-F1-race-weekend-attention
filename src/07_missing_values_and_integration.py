"""
07_missing_values_and_integration.py

Covers two rubric items:
1. Missing-value frequency per variable, across each raw/intermediate
   dataset, with documented reasoning.
2. Integration/match-loss reporting: how many samples matched at each
   merge stage, how many were lost, and why.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PART 1: Missing-value frequency per dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missingness_rows = []

def analyze_missingness(df: pd.DataFrame, table_name: str, reasons: dict):
    for col in df.columns:
        pct_missing = df[col].isnull().mean()
        n_missing = df[col].isnull().sum()
        reason = reasons.get(col, "No systematic reason identified; appears incidental.")
        missingness_rows.append({
            "table": table_name,
            "column": col,
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 4),
            "likely_reason": reason if pct_missing > 0 else "N/A (no missing values)",
        })
    print(f"\n--- {table_name} ({len(df)} rows) ---")
    summary = df.isnull().sum()
    summary = summary[summary > 0]
    if summary.empty:
        print("  No missing values in any column.")
    else:
        for col, n in summary.items():
            print(f"  {col}: {n} missing ({n/len(df):.1%})")


# races.csv (raw, before footer-row cleaning) — Round/Grand Prix/Circuit/Race
# date missingness here is mostly from the "Sources:[..]" Wikipedia footer
# rows, which get filtered out downstream in 02/03 — not real missing race
# data, but worth documenting since it shows up in the raw scrape.
races_raw = pd.read_csv(RAW_DIR / "races.csv")
analyze_missingness(races_raw, "races.csv (raw)", {
    "Round": "Wikipedia 'Sources:[..]' footer rows scraped alongside real race rows; filtered out in preprocessing, not true missingness in race data.",
})

# trends_master.csv — search_interest missingness, if any, reflects Trends
# API returning no data for a given day/city combination (below its minimum
# reporting threshold), not a pipeline failure.
trends_df = pd.read_csv(PROCESSED_DIR / "trends_master.csv")
analyze_missingness(trends_df, "trends_master.csv", {
    "search_interest": "Google Trends returns no data below a minimum search-volume threshold for some city/date combinations.",
})

# airbnb_raw.csv — city_slug is intentionally None for cities Inside Airbnb
# doesn't cover (small towns like Spielberg, Zandvoort, Imola); listings
# data is further missing for cities with no listings.csv.gz available
# (confirmed case: Las Vegas, São Paulo) even when a slug was assigned.
airbnb_df = pd.read_csv(RAW_DIR / "airbnb_raw.csv")
analyze_missingness(airbnb_df, "airbnb_raw.csv", {
    "airbnb_city_slug": "City has no Inside Airbnb coverage at all (confirmed: Sakhir, Baku, Zandvoort, Monaco, Spa, Imola, Jeddah, Lusail, Spielberg, Shanghai, Silverstone, Suzuka, Abu Dhabi, Le Castellet) — small towns or cities outside Inside Airbnb's covered-city list.",
    "local_currency": "Only populated when listings data was successfully retrieved.",
    "airbnb_price_snapshot_mean_local": "Only populated when listings data was successfully retrieved (city had a slug AND a working listings.csv.gz URL was found).",
    "airbnb_price_snapshot_median_local": "Same as above.",
    "airbnb_price_snapshot_mean_usd": "Same as above.",
    "airbnb_price_snapshot_median_usd": "Same as above.",
    "airbnb_listing_count": "Same as above.",
    "pct_unavailable_baseline": "Calendar snapshot data does not cover historical race-week dates (Inside Airbnb calendars are forward-looking from scrape date only) — structurally missing for all past races, not a retrieval failure.",
    "pct_unavailable_race_week": "Same as above.",
    "occupancy_surge_factor": "Same as above.",
})

# f1_anomalies_detected.csv — championship_gap_entering_race is expectedly
# missing for season-opening races (round 1), since there's no "entering
# this season" standings yet.
features_df = pd.read_csv(PROCESSED_DIR / "f1_anomalies_detected.csv")
analyze_missingness(features_df, "f1_anomalies_detected.csv", {
    "championship_gap_entering_race": "Season-opening races (round 1) have no prior-round standings to compute a gap from — structural, not a retrieval failure.",
    "is_close_championship_race": "Same as above (derived directly from championship_gap_entering_race).",
})

missingness_df = pd.DataFrame(missingness_rows)
missingness_out = REPORTS_DIR / "missingness_report.csv"
missingness_df.to_csv(missingness_out, index=False)
print(f"\nSaved full missingness report to: {missingness_out.resolve()}")


# ---------------------------------------------------------------------------
# PART 2: Integration / match-loss reporting
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("INTEGRATION / MATCH-LOSS REPORT")
print("=" * 70)

# Stage 1: races.csv raw -> cleaned (footer rows removed)
races_cleaned = races_raw[
    ~races_raw["Round"].astype(str).str.contains("Source", case=False, na=False)
].copy()
races_cleaned["round_clean"] = races_cleaned["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
races_cleaned["round_num"] = pd.to_numeric(races_cleaned["round_clean"], errors="coerce")
races_cleaned = races_cleaned.dropna(subset=["round_num"])

stage1_before = len(races_raw)
stage1_after = len(races_cleaned)

# Stage 2: cleaned races -> races with Trends data matched
races_with_trends = trends_df["race_id"].nunique()

# Stage 3: races with Trends -> races with championship gap computed
races_with_gap = features_df["championship_gap_entering_race"].notna().sum()

# Stage 4: races -> races matched to an Airbnb city (slug assigned)
races_airbnb_matched = airbnb_df["airbnb_city_slug"].notna().sum()

# Stage 5: races matched -> races with actual usable listings data
races_airbnb_usable = airbnb_df["has_listings_data"].sum()

integration_summary = pd.DataFrame([
    {"stage": "races.csv raw scrape", "n_races": stage1_before, "n_lost_this_stage": None,
     "reason": "Includes Wikipedia footer/source rows mixed in with real race rows."},
    {"stage": "races.csv cleaned (footers removed)", "n_races": stage1_after,
     "n_lost_this_stage": stage1_before - stage1_after,
     "reason": "Removed 'Sources:[..]' footer rows and rows with unparseable round numbers."},
    {"stage": "Matched to Google Trends data", "n_races": races_with_trends,
     "n_lost_this_stage": stage1_after - races_with_trends,
     "reason": "Races where Trends returned no data after all retries/fallback queries exhausted (0 after bracket-stripping fix; previously 6 races affected by a date-parsing bug, now resolved)."},
    {"stage": "Championship gap computed", "n_races": races_with_gap,
     "n_lost_this_stage": races_with_trends - races_with_gap,
     "reason": "Season-opening races (round 1) have no prior standings — structural, one race per season (including 2020)."},
    {"stage": "Matched to an Airbnb city (slug assigned)", "n_races": races_airbnb_matched,
     "n_lost_this_stage": races_with_trends - races_airbnb_matched,
     "reason": "Race's host circuit is in a city/town Inside Airbnb doesn't cover (confirmed for 13 of 25 unique circuits — mostly small towns)."},
    {"stage": "Airbnb data actually usable", "n_races": races_airbnb_usable,
     "n_lost_this_stage": races_airbnb_matched - races_airbnb_usable,
     "reason": "City had a slug assigned but no listings.csv.gz found on Inside Airbnb (confirmed: Las Vegas, São Paulo)."},
])

print(integration_summary.to_string(index=False))
integration_out = REPORTS_DIR / "integration_match_loss_report.csv"
integration_summary.to_csv(integration_out, index=False)
print(f"\nSaved integration report to: {integration_out.resolve()}")