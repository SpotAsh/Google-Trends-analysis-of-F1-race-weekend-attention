"""
01_scrape_races.py

Scrapes Wikipedia to build the F1 race calendar for 2016-2025, EXCLUDING 2020.

Why 2016+: gives three seasons (2016-2018) before Netflix's "Drive to Survive"
(first season released March 2019) shifted baseline interest, plus 2021 (the
Hamilton-Verstappen title fight, the closest recent season) and 2022 (a
deliberate blowout contrast case: Verstappen clinched early).

Why not 2020: a COVID-shortened season (17 races, mostly behind closed doors,
several one-off venues). Its search patterns are not comparable, so it is
dropped entirely rather than modelled.

Wikipedia calendar tables are NOT identical across years: 2016/2019 label the
date column "Date" while 2021+ label it "Race date". This script normalizes
the column to "Race date" so 02/03 work for every season unchanged.
"""

from pathlib import Path
import io
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "races.csv"
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [y for y in range(2016, 2026) if y != 2020]
urls = {y: f"https://en.wikipedia.org/wiki/{y}_Formula_One_World_Championship" for y in YEARS}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def find_calendar_table(tables):
    """First table that has Round + Grand Prix + Circuit + a date column."""
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        low = [c.lower() for c in cols]
        has_round = any(c == "round" for c in low)
        has_gp = any("grand prix" in c for c in low)
        has_circuit = any("circuit" in c or "venue" in c for c in low)
        has_date = any("date" in c for c in low)
        if has_round and has_gp and has_circuit and has_date:
            out = df.copy()
            out.columns = cols
            # Normalize the date column name ("Date" in 2016/2019, "Race date" in 2021+)
            date_col = [c for c in cols if "date" in c.lower()][0]
            out = out.rename(columns={date_col: "Race date"})
            # Normalize the other names too, in case of stray whitespace/case
            for c in list(out.columns):
                if c.lower() == "round":
                    out = out.rename(columns={c: "Round"})
                elif "grand prix" in c.lower() and c != "Grand Prix":
                    out = out.rename(columns={c: "Grand Prix"})
                elif ("circuit" in c.lower() or "venue" in c.lower()) and c != "Circuit":
                    out = out.rename(columns={c: "Circuit"})
            return out[["Round", "Grand Prix", "Circuit", "Race date"]]
    return None


all_tables = []

for season, url in urls.items():
    print(f"Scraping {season} season data...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        cal = find_calendar_table(tables)
        if cal is None:
            print(f"  [WARNING] No calendar table found for {season}")
            continue
        cal["season"] = season
        n_numeric = pd.to_numeric(
            cal["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True), errors="coerce"
        ).notna().sum()
        print(f"  found calendar table: {len(cal)} rows ({n_numeric} with numeric round)")
        all_tables.append(cal)
    except Exception as e:
        print(f"Error scraping {season}: {e}")

if all_tables:
    races_df = pd.concat(all_tables, ignore_index=True)
    races_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved raw race data to: {OUTPUT_FILE.resolve()}")
    print(f"Total rows scraped (including footer/junk rows to be cleaned later): {len(races_df)}")
    real = races_df[pd.to_numeric(
        races_df["Round"].astype(str).str.replace(r"\[.*?\]", "", regex=True), errors="coerce").notna()]
    counts = real.groupby("season").size()
    print("\nReal races per season (expected: 2016=21, 2017=20, 2018=21, 2019=21, 2021=22, "
          "2022=22, 2023=22, 2024=24, 2025=24):")
    print(counts)
    expected = {2016: 21, 2017: 20, 2018: 21, 2019: 21, 2021: 22, 2022: 22, 2023: 22, 2024: 24, 2025: 24}
    for s, n in expected.items():
        got = int(counts.get(s, 0))
        if got != n:
            print(f"  [CHECK] {s}: got {got}, expected {n}")
    missing = [y for y in YEARS if y not in counts.index]
    if missing:
        print(f"  [WARNING] Seasons missing entirely: {missing}")
else:
    print("Failed to scrape tables. Check network connection or Wikipedia URLs.")