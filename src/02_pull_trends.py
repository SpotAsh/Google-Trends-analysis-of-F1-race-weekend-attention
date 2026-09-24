import random
import time
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq

# 1. Directory Setup (Creates folders automatically if missing)
RAW_TRENDS_DIR = Path("data/raw/trends")
PROCESSED_DIR = Path("data/processed")

RAW_TRENDS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load Races Spine & Clean Headers
races_df = pd.read_csv("data/raw/races.csv", encoding="utf-8-sig")
races_df.columns = races_df.columns.str.strip()

# --- FIX (root cause): strip Wikipedia footnote brackets (e.g. "[d]", "[e]")
# from both Round and Race date BEFORE parsing. The previous version of this
# script parsed the raw "Race date" column directly, so any date with a
# footnote marker attached (e.g. "18 November[d]") failed pd.to_datetime and
# the whole race was silently skipped — this is what caused Las Vegas (all 3
# seasons) and the 2024 season-openers to drop out, not a Trends API issue.
races_df = races_df[
    ~races_df["Round"].astype(str).str.contains("Source", case=False, na=False)
].copy()
races_df["Round"] = races_df["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
races_df["Race date"] = races_df["Race date"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()

# Initialize PyTrends (omitting retries to prevent urllib3 compatibility error)
pytrends = TrendReq(hl="en-US", tz=360)

master_trends_list = []

# --- Known-bad race_ids from previous runs get their cache files cleared so
# the resumable-skip logic doesn't skip them again this run. Update this list
# based on whatever trends_master.csv currently shows as missing. ---
KNOWN_FAILED_RACE_IDS = [
    "2023_R21",  # Las Vegas GP 2023
    "2024_R1",   # Bahrain GP 2024
    "2024_R2",   # Saudi Arabian GP 2024
    "2024_R22",  # Las Vegas GP 2024
    "2025_R22",  # Las Vegas GP 2025
]
for race_id in KNOWN_FAILED_RACE_IDS:
    stale_file = RAW_TRENDS_DIR / f"trends_{race_id}.csv"
    if stale_file.exists():
        stale_file.unlink()
        print(f"[CLEANUP] Removed stale/empty cache file for {race_id}")

# --- FIX 2: alternate query strings to try, in order, for races whose
# default "{Grand Prix name}" query is known/likely to fail on Trends. ---
QUERY_OVERRIDES = {
    "Las Vegas Grand Prix": ["Las Vegas Grand Prix", "Las Vegas Grand Prix F1", "F1 Las Vegas"],
}

# --- FIX 3: retry wrapper with backoff for transient API failures ---
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
    # Construct a unique race_id (e.g., 2023_R1)
    season = str(row["season"]).strip()
    round_num = str(row["Round"]).strip()
    race_id = f"{season}_R{round_num}"

    # Extract target search term (using Grand Prix name)
    city = str(row["Grand Prix"]).strip()

    # Safely construct the full date using season + race date string
    raw_date_str = str(row["Race date"]).strip()
    full_date_str = f"{raw_date_str} {season}"

    # Parse date safely
    parsed_date = pd.to_datetime(full_date_str, errors="coerce")

    # Fallback if combining season didn't work (e.g., if Race date already contained year)
    if pd.isna(parsed_date):
        parsed_date = pd.to_datetime(raw_date_str, errors="coerce")

    # Skip if date completely failed to parse
    if pd.isna(parsed_date):
        print(f"  --> [ERROR] Could not parse date '{raw_date_str}' for {race_id}. Skipping.")
        continue

    race_date = parsed_date

    # Define raw output file path
    raw_file_path = RAW_TRENDS_DIR / f"trends_{race_id}.csv"

    # Check if file already exists locally (allows safe script resuming)
    if raw_file_path.exists():
        print(f"[{idx+1}/{len(races_df)}] Skipping {race_id} ({city}) — already downloaded")
        df_single = pd.read_csv(raw_file_path)
        if not df_single.empty:
            master_trends_list.append(df_single)
        continue

    # 3. Calculate ~90-day Window (60 days pre-race to 30 days post-race)
    start_date = (race_date - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = (race_date + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    timeframe = f"{start_date} {end_date}"

    print(
        f"[{idx+1}/{len(races_df)}] Fetching '{city}' ({race_id}) | Window: {start_date} to {end_date}"
    )

    # Build the list of query strings to try for this race (override list if
    # one exists for this Grand Prix name, otherwise just the default name).
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
        # brief pause before trying the next fallback query
        time.sleep(random.uniform(3, 6))

    if not interest_df.empty:
        # Clean dataframe
        interest_df = interest_df.reset_index()

        # Rename index to 'date' and the target column to 'search_interest'
        first_col = interest_df.columns[0]
        interest_df = interest_df.rename(
            columns={first_col: "date", query_used: "search_interest"}
        )

        # Drop 'isPartial' column if pytrends included it
        if "isPartial" in interest_df.columns:
            interest_df = interest_df.drop(columns=["isPartial"])

        # Add metadata keys
        interest_df["race_id"] = race_id
        interest_df["grand_prix"] = city
        interest_df["season"] = season
        interest_df["round"] = round_num
        interest_df["query_used"] = query_used  # NEW: document which query worked

        # Save raw individual file immediately
        interest_df.to_csv(raw_file_path, index=False)
        master_trends_list.append(interest_df)
    else:
        print(f"  --> [FAILED] All queries exhausted for {race_id} ({city}). "
              f"Last error: {final_error}")
        # Create an empty placeholder file with headers to prevent re-querying on resume.
        # NOTE: if you need to retry this race again later, delete this file first —
        # same as the KNOWN_FAILED_RACE_IDS cleanup step above.
        pd.DataFrame(
            columns=["date", "search_interest", "race_id", "grand_prix", "season", "round", "query_used"]
        ).to_csv(raw_file_path, index=False)

    # 4. Rate-limiting Delay (random 5 to 12 seconds)
    sleep_time = random.uniform(5, 12)
    print(f"  Sleeping {sleep_time:.1f}s to prevent rate limits...")
    time.sleep(sleep_time)

# 5. Concatenate & Save Master File
if master_trends_list:
    master_df = pd.concat(master_trends_list, ignore_index=True)
    master_output = PROCESSED_DIR / "trends_master.csv"
    master_df.to_csv(master_output, index=False)
    print(
        f"\nSUCCESS! Master Google Trends dataset saved to: {master_output.resolve()}"
    )
    print(f"Unique races with Trends data: {master_df['race_id'].nunique()}")