from pathlib import Path
import pandas as pd
import requests

# 1. Define output directory and file path relative to project root
RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "races.csv"

# Ensure the 'data/raw' directory exists
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 2. Scrape Wikipedia race tables
urls = {
    2023: "https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship",
    2024: "https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship",
    2025: "https://en.wikipedia.org/wiki/2025_Formula_One_World_Championship",
}

# Define headers to emulate a standard web browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

all_tables = []

for season, url in urls.items():
    print(f"Scraping {season} season data...")

    try:
        # Fetch the HTML content using requests with custom headers
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        # Pass the raw HTML text to pandas
        tables = pd.read_html(response.text)

        for df in tables:
            # Check if table is the race calendar
            col_str = " ".join([str(c) for c in df.columns]).lower()
            if "grand prix" in col_str and ("circuit" in col_str or "venue" in col_str):
                df["season"] = season
                all_tables.append(df)
                break

    except Exception as e:
        print(f"Error scraping {season}: {e}")

# Combine scraped tables
if all_tables:
    races_df = pd.concat(all_tables, ignore_index=True)

    # 3. Save directly to data/raw/races.csv
    # index=False prevents pandas from writing an extra unwanted column (0, 1, 2...)
    races_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved raw race data to: {OUTPUT_FILE.resolve()}")
else:
    print("Failed to scrape tables. Check network connection or Wikipedia URLs.")