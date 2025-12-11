# src/download_stats.py

import time
from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Seasons you want
YEARS = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Map your stat names to stat IDs
STAT_IDS = {
    # Strokes gained
    "sg_total": "02675",
    "sg_ott": "02567",
    "sg_app": "02568",
    "sg_arg": "02569",
    "sg_putt": "02564",

    # Traditional stats
    "driving_distance": "101",
    "driving_accuracy": "102",
    "greens_in_regulation": "103",
    "scoring_average": "120",

    # Success metrics
    "money_earned": "109",   # Money Leaders
    "fedex_rank": "02671",   # FedEx / points-style ranking you picked
    # (we can still add events_played later if we find a good ID)
}


# Base endpoint
BASE_URL = (
    "https://www.pgatour.com/api/stats-download"
    "?timePeriod=THROUGH_EVENT"
    "&tourCode=R"
    "&statsId={stat_id}"
    "&year={year}"
)

def download_stat_csv(stat_name: str, stat_id: str, year: int):
    url = BASE_URL.format(stat_id=stat_id, year=year)
    out_path = RAW_DIR / f"{stat_name}_{year}.csv"

    print(f"Downloading {stat_name} ({stat_id}) for {year}")
    print(f"  URL: {url}")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    print(f"  Saved to {out_path}\n")

    time.sleep(1)  # politeness

def main():
    for year in YEARS:
        for stat_name, stat_id in STAT_IDS.items():
            download_stat_csv(stat_name, stat_id, year)

    print("All CSV files downloaded successfully!")

if __name__ == "__main__":
    main()
