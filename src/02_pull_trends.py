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

# Initialize PyTrends (omitting retries to prevent urllib3 compatibility error)
pytrends = TrendReq(hl="en-US", tz=360)

master_trends_list = []

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

    try:
        # Build payload & pull data worldwide (geo="" covers global search interest)
        pytrends.build_payload([city], timeframe=timeframe, geo="")
        interest_df = pytrends.interest_over_time()

        if not interest_df.empty:
            # Clean dataframe
            interest_df = interest_df.reset_index()

            # Rename index to 'date' and the target column to 'search_interest'
            first_col = interest_df.columns[0]
            interest_df = interest_df.rename(
                columns={first_col: "date", city: "search_interest"}
            )

            # Drop 'isPartial' column if pytrends included it
            if "isPartial" in interest_df.columns:
                interest_df = interest_df.drop(columns=["isPartial"])

            # Add metadata keys
            interest_df["race_id"] = race_id
            interest_df["grand_prix"] = city
            interest_df["season"] = season
            interest_df["round"] = round_num

            # Save raw individual file immediately
            interest_df.to_csv(raw_file_path, index=False)
            master_trends_list.append(interest_df)
        else:
            print(f"  --> Warning: No data returned for {city}")
            # Create an empty placeholder file with headers to prevent re-querying on resume
            pd.DataFrame(
                columns=["date", "search_interest", "race_id", "grand_prix", "season", "round"]
            ).to_csv(raw_file_path, index=False)

    except Exception as e:
        print(f"  --> Failed to pull {race_id} ({city}): {e}")

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