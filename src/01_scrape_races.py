"""
01_scrape_races.py

Scrapes Wikipedia to build the F1 race calendar for 2022-2025.
2022 added per project decision: it's a deliberate contrast case (Verstappen
won 19/22 races, championship decided by August) to test whether search
interest still spikes on race weekends even when there's no championship
drama — i.e. does venue/event novelty drive interest independent of stakes.
"""

from pathlib import Path
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "races.csv"
RAW_DIR.mkdir(parents=True, exist_ok=True)

urls = {
    2022: "https://en.wikipedia.org/wiki/2022_Formula_One_World_Championship",
    2023: "https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship",
    2024: "https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship",
    2025: "https://en.wikipedia.org/wiki/2025_Formula_One_World_Championship",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

all_tables = []

for season, url in urls.items():
    print(f"Scraping {season} season data...")
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        tables = pd.read_html(response.text)

        for df in tables:
            col_str = " ".join([str(c) for c in df.columns]).lower()
            if "grand prix" in col_str and ("circuit" in col_str or "venue" in col_str):
                df["season"] = season
                all_tables.append(df)
                break

    except Exception as e:
        print(f"Error scraping {season}: {e}")

if all_tables:
    races_df = pd.concat(all_tables, ignore_index=True)
    races_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved raw race data to: {OUTPUT_FILE.resolve()}")
    print(f"Total rows scraped (including footer/junk rows to be cleaned later): {len(races_df)}")
    print(races_df.groupby("season").size())
else:
    print("Failed to scrape tables. Check network connection or Wikipedia URLs.")