"""
06_pull_airbnb.py

Pulls Airbnb pricing data from Inside Airbnb for race cities, matched against
the `circuit` column already present in f1_anomalies_features.csv /
f1_anomalies_detected.csv (no need to re-derive city names from races.csv).

Inside Airbnb (insideairbnb.com) publishes free per-city snapshot downloads.
There is no live API — this script hits known static download URLs per city.
Coverage is expected to be PARTIAL: many F1 host cities (small towns like
Spielberg, Zandvoort, Imola, Mogyoród, Sakhir, Suzuka, Lusail) are not
covered by Inside Airbnb at all. This is normal and should be documented,
not treated as a bug.
"""

import time
from pathlib import Path
import gzip
import shutil
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
AIRBNB_DIR = RAW_DIR / "airbnb"
PROCESSED_DIR = Path("data/processed")
AIRBNB_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FEATURES_FILE = PROCESSED_DIR / "f1_anomalies_detected.csv"  # use the 70-race corrected file

# ---------------------------------------------------------------------------
# STEP 1: map each `circuit` string (as it appears in your data) to:
#   - a human city name (for logging/reference)
#   - an Inside Airbnb city slug, if one is known to exist there
#     (None = not expected to have Inside Airbnb coverage; script will still
#     try common slug guesses for these but expect them to fail)
#
# Inside Airbnb slugs correspond to the folder names at:
#   http://insideairbnb.com/get-the-data/
# You can verify/update slugs by checking that page directly if a download
# below returns 404 — city coverage changes periodically.
# ---------------------------------------------------------------------------
CIRCUIT_TO_AIRBNB = {
    "Albert Park Circuit, Melbourne":                          ("Melbourne", "melbourne"),
    "Autódromo Hermanos Rodríguez, Mexico City":                ("Mexico City", "mexico-city"),
    "Bahrain International Circuit, Sakhir":                   ("Sakhir", None),
    "Baku City Circuit, Baku":                                  ("Baku", None),
    "Circuit Gilles Villeneuve, Montreal":                      ("Montreal", "montreal"),
    "Circuit Zandvoort, Zandvoort":                             ("Zandvoort", None),
    "Circuit de Barcelona-Catalunya, Montmeló":                 ("Barcelona", "barcelona"),
    "Circuit de Monaco, Monaco":                                ("Monaco", None),
    "Circuit de Spa-Francorchamps, Stavelot":                   ("Stavelot", None),
    "Circuit of the Americas, Austin, Texas":                   ("Austin", "austin"),
    "Hungaroring, Mogyoród":                                    ("Budapest", "budapest"),  # nearest covered city
    "Imola Circuit, Imola":                                     ("Imola", None),
    "Interlagos Circuit, São Paulo":                            ("São Paulo", "sao-paulo"),
    "Jeddah Corniche Circuit, Jeddah":                          ("Jeddah", None),
    "Lusail International Circuit, Lusail":                     ("Lusail", None),
    "Marina Bay Street Circuit, Singapore":                     ("Singapore", "singapore"),
    "Miami International Autodrome, Miami Gardens, Florida":    ("Miami", "broward-county"),  # Inside Airbnb covers broader Miami/Broward area, not Miami Gardens specifically
    "Monza Circuit, Monza":                                     ("Milan", "milan"),  # nearest major covered city
    "Red Bull Ring, Spielberg":                                 ("Spielberg", None),
    "Shanghai International Circuit, Shanghai":                 ("Shanghai", None),
    "Silverstone Circuit, Silverstone":                         ("Silverstone", None),
    "Suzuka Circuit, Suzuka":                                   ("Suzuka", None),
    "Suzuka International Racing Course, Suzuka":               ("Suzuka", None),
    "Yas Marina Circuit, Abu Dhabi":                             ("Abu Dhabi", None),
    "Las Vegas Strip Circuit, Las Vegas":                       ("Las Vegas", "las-vegas"),
}

# --- Currency conversion ---
# Inside Airbnb reports prices in each listing's LOCAL currency, not
# standardized USD (confirmed: Budapest listings came back with a $54,315
# mean, which is HUF, not USD — an obvious tell since it's ~150-300x every
# other city). Convert everything to USD before using it in any comparison
# or downstream binary flag, or cross-city comparisons will be meaningless.
# Rates below are approximate as of late 2025/2026 — fine for relative
# comparison purposes in this project, but note in the report that these are
# fixed approximate rates, not date-matched historical FX rates.
CITY_SLUG_TO_CURRENCY = {
    "melbourne": ("AUD", 0.65),
    "mexico-city": ("MXN", 0.050),
    "montreal": ("CAD", 0.73),
    "barcelona": ("EUR", 1.08),
    "austin": ("USD", 1.0),
    "budapest": ("HUF", 0.0028),
    "sao-paulo": ("BRL", 0.17),
    "singapore": ("SGD", 0.75),
    "broward-county": ("USD", 1.0),
    "milan": ("EUR", 1.08),
    "las-vegas": ("USD", 1.0),
}

# Inside Airbnb download URL pattern (data.insideairbnb.com hosts the actual
# files; each city+date combination has its own path). Since exact scrape
# dates change over time, we query the city's data page first to discover
# the most recent available file rather than hardcoding a date.
INSIDE_AIRBNB_DATA_PAGE = "https://insideairbnb.com/get-the-data/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def find_latest_listings_url(city_slug: str) -> str | None:
    """
    Scrape the Inside Airbnb data page for the most recent listings.csv.gz
    link matching this city slug. Returns None if not found.

    NOTE: we use listings.csv, not calendar.csv, because Inside Airbnb's
    current calendar.csv releases no longer include a price column (columns
    are just listing_id, date, available, minimum_nights, maximum_nights —
    confirmed by direct inspection). listings.csv has one price per listing
    at scrape time — no date granularity, so this becomes a single
    cross-sectional price snapshot per city rather than a baseline/race-week
    comparison. Document this as a real data-availability limitation, not a
    bug, in the report.
    """
    try:
        resp = requests.get(INSIDE_AIRBNB_DATA_PAGE, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except Exception as e:
        print(f"    [ERROR] Could not load Inside Airbnb data page: {e}")
        return None

    html = resp.text
    if city_slug not in html.lower().replace(" ", "-"):
        return None

    import re
    pattern = rf'(https://data\.insideairbnb\.com/[^"]*{city_slug}[^"]*listings\.csv\.gz)'
    matches = re.findall(pattern, html, flags=re.IGNORECASE)
    if matches:
        return sorted(matches)[-1]
    return None


def download_and_load_listings(url: str, city_slug: str) -> pd.DataFrame:
    gz_path = AIRBNB_DIR / f"{city_slug}_listings.csv.gz"
    csv_path = AIRBNB_DIR / f"{city_slug}_listings.csv"

    if not csv_path.exists():
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()

        if resp.content[:2] != b"\x1f\x8b":
            preview = resp.content[:200]
            raise ValueError(
                f"Downloaded content for {city_slug} is not a valid gzip file. "
                f"First 200 bytes: {preview!r}"
            )

        gz_path.write_bytes(resp.content)
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    df = pd.read_csv(csv_path, low_memory=False)

    if "price" not in df.columns:
        raise ValueError(
            f"'price' column not found for {city_slug} listings file either. "
            f"Actual columns: {list(df.columns)}"
        )

    df["price"] = (
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    )
    return df


def main():
    features_df = pd.read_csv(INPUT_FEATURES_FILE)

    results = []

    for _, row in features_df.iterrows():
        race_id = row["race_id"]
        circuit = row["circuit"]
        race_date = pd.to_datetime(row.get("race_date"), errors="coerce")

        city_name, city_slug = CIRCUIT_TO_AIRBNB.get(circuit, (circuit, None))

        record = {
            "race_id": race_id,
            "circuit": circuit,
            "city_matched_name": city_name,
            "airbnb_city_slug": city_slug,
            "has_listings_data": False,
            "local_currency": None,
            "airbnb_price_snapshot_mean_local": None,
            "airbnb_price_snapshot_median_local": None,
            "airbnb_price_snapshot_mean_usd": None,
            "airbnb_price_snapshot_median_usd": None,
            "airbnb_listing_count": None,
        }

        if city_slug is None:
            print(f"[{race_id}] {city_name}: no Inside Airbnb slug expected — skipping")
            results.append(record)
            continue

        print(f"[{race_id}] {city_name} ({city_slug}): looking up listings data...")
        listings_url = find_latest_listings_url(city_slug)

        if listings_url is None:
            print(
                f"    No listings.csv.gz found for '{city_slug}'. "
                f"Check https://insideairbnb.com/get-the-data/ manually for "
                f"'{city_name}' and hardcode the URL in a MANUAL_URLS dict if found."
            )
            results.append(record)
            continue

        try:
            listings_df = download_and_load_listings(listings_url, city_slug)
        except Exception as e:
            print(f"    [ERROR] Failed to download/parse listings for {city_slug}: {e}")
            results.append(record)
            continue

        record["has_listings_data"] = True
        currency, fx_to_usd = CITY_SLUG_TO_CURRENCY.get(city_slug, ("UNKNOWN", 1.0))
        record["local_currency"] = currency
        record["airbnb_price_snapshot_mean_local"] = round(listings_df["price"].mean(), 2)
        record["airbnb_price_snapshot_median_local"] = round(listings_df["price"].median(), 2)
        record["airbnb_price_snapshot_mean_usd"] = round(listings_df["price"].mean() * fx_to_usd, 2)
        record["airbnb_price_snapshot_median_usd"] = round(listings_df["price"].median() * fx_to_usd, 2)
        record["airbnb_listing_count"] = len(listings_df)

        print(
            f"    Mean: {record['airbnb_price_snapshot_mean_local']} {currency} "
            f"(${record['airbnb_price_snapshot_mean_usd']} USD) | "
            f"Median: {record['airbnb_price_snapshot_median_local']} {currency} "
            f"(${record['airbnb_price_snapshot_median_usd']} USD) | "
            f"n={record['airbnb_listing_count']} listings"
        )

        results.append(record)
        time.sleep(1)  # be polite to insideairbnb.com

    out_df = pd.DataFrame(results)
    out_path = RAW_DIR / "airbnb_raw.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nSUCCESS! Saved Airbnb pull results to {out_path.resolve()}")
    print(f"Races with a matched Airbnb city: {out_df['airbnb_city_slug'].notna().sum()} / {len(out_df)}")
    print(f"Races with usable listings data: {out_df['has_listings_data'].sum()}")
    print(
        "\nNOTE: this is a single cross-sectional price snapshot per city "
        "(current scrape date), not a baseline-vs-race-week comparison — "
        "Inside Airbnb's calendar.csv no longer includes price data. "
        "Document this as a data-availability limitation in the report."
    )


if __name__ == "__main__":
    main()