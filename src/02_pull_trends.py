"""
02_pull_trends.py

Pulls Google Trends daily search-interest data per race.

KEY FIX (root cause of earlier data loss): Wikipedia dates carry footnote
markers (e.g. "18 November[d]"), which broke pd.to_datetime and silently
caused those races to be skipped entirely. Brackets are now stripped from
both Round and Race date BEFORE parsing, same as 03_preprocessing.py does.
This is season-agnostic, so it applies cleanly to the newly-added 2022 data
without any special-casing.

Also includes: retry logic with backoff for transient API failures, and a
query-string fallback list for races whose default "{Grand Prix name}" query
is known to be flaky on Trends (currently just Las Vegas, which failed
outright before the bracket fix was identified — kept as a defensive
measure in case it resurfaces).
"""

import random
import time
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq

RAW_TRENDS_DIR = Path("data/raw/trends")
PROCESSED_DIR = Path("data/processed")
RAW_TRENDS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

races_df = pd.read_csv("data/raw/races.csv", encoding="utf-8-sig")
races_df.columns = races_df.columns.str.strip()

# Filter Wikipedia footer rows, strip footnote brackets from Round + Race date
races_df = races_df[
    ~races_df["Round"].astype(str).str.contains("Source", case=False, na=False)
].copy()
races_df["Round"] = races_df["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
races_df["Race date"] = races_df["Race date"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()

pytrends = TrendReq(hl="en-US", tz=360)
master_trends_list = []

QUERY_OVERRIDES = {
    "Las Vegas Grand Prix": ["Las Vegas Grand Prix", "Las Vegas Grand Prix F1", "F1 Las Vegas"],
}


def fetch_with_retries(query, timeframe, max_retries=3):
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            pytrends.build_payload([query], timeframe=timeframe, geo="")
            interest_df = pytrends.interest_over_time()
            if not interest_df.empty:
                return interest_df, None
            last_exception = "Empty response (no error raised, but no data returned)"
        except Exception as e:
            last_exception = str(e)

        if attempt < max_retries:
            backoff = attempt * 8 + random.uniform(2, 5)
            print(f"    [RETRY {attempt}/{max_retries}] '{query}' failed: {last_exception}. "
                  f"Waiting {backoff:.1f}s before retry...")
            time.sleep(backoff)

    return pd.DataFrame(), last_exception


print(f"Starting Google Trends extraction for {len(races_df)} races...")

for idx, row in races_df.iterrows():
    season = str(row["season"]).strip()
    round_num = str(row["Round"]).strip()
    race_id = f"{season}_R{round_num}"
    city = str(row["Grand Prix"]).strip()
    raw_date_str = str(row["Race date"]).strip()
    full_date_str = f"{raw_date_str} {season}"

    parsed_date = pd.to_datetime(full_date_str, errors="coerce")
    if pd.isna(parsed_date):
        parsed_date = pd.to_datetime(raw_date_str, errors="coerce")
    if pd.isna(parsed_date):
        print(f"  --> [ERROR] Could not parse date '{raw_date_str}' for {race_id}. Skipping.")
        continue

    race_date = parsed_date
    raw_file_path = RAW_TRENDS_DIR / f"trends_{race_id}.csv"

    if raw_file_path.exists():
        print(f"[{idx+1}/{len(races_df)}] Skipping {race_id} ({city}) — already downloaded")
        df_single = pd.read_csv(raw_file_path)
        if not df_single.empty:
            master_trends_list.append(df_single)
        continue

    start_date = (race_date - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = (race_date + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    timeframe = f"{start_date} {end_date}"

    print(f"[{idx+1}/{len(races_df)}] Fetching '{city}' ({race_id}) | Window: {start_date} to {end_date}")

    queries_to_try = QUERY_OVERRIDES.get(city, [city])
    interest_df = pd.DataFrame()
    query_used = None
    final_error = None

    for query in queries_to_try:
        if query != queries_to_try[0]:
            print(f"    [FALLBACK QUERY] Trying alternate query: '{query}'")
        interest_df, final_error = fetch_with_retries(query, timeframe)
        if not interest_df.empty:
            query_used = query
            break
        time.sleep(random.uniform(3, 6))

    if not interest_df.empty:
        interest_df = interest_df.reset_index()
        first_col = interest_df.columns[0]
        interest_df = interest_df.rename(columns={first_col: "date", query_used: "search_interest"})
        if "isPartial" in interest_df.columns:
            interest_df = interest_df.drop(columns=["isPartial"])

        interest_df["race_id"] = race_id
        interest_df["grand_prix"] = city
        interest_df["season"] = season
        interest_df["round"] = round_num
        interest_df["query_used"] = query_used

        interest_df.to_csv(raw_file_path, index=False)
        master_trends_list.append(interest_df)
    else:
        print(f"  --> [FAILED] All queries exhausted for {race_id} ({city}). Last error: {final_error}")
        pd.DataFrame(
            columns=["date", "search_interest", "race_id", "grand_prix", "season", "round", "query_used"]
        ).to_csv(raw_file_path, index=False)

    sleep_time = random.uniform(5, 12)
    print(f"  Sleeping {sleep_time:.1f}s to prevent rate limits...")
    time.sleep(sleep_time)

if master_trends_list:
    master_df = pd.concat(master_trends_list, ignore_index=True)
    master_output = PROCESSED_DIR / "trends_master.csv"
    master_df.to_csv(master_output, index=False)
    print(f"\nSUCCESS! Master Google Trends dataset saved to: {master_output.resolve()}")
    print(f"Unique races with Trends data: {master_df['race_id'].nunique()}")