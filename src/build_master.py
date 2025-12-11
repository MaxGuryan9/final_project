# src/build_master.py

from pathlib import Path
from functools import reduce

import pandas as pd

INTER_DIR = Path("data/intermediate")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# This must match the stat_name → output_col mapping from parse_stats.py
STAT_COLUMNS = {
    "sg_total": "sg_total",
    "sg_ott": "sg_off_the_tee",
    "sg_app": "sg_approach",
    "sg_arg": "sg_around_green",
    "sg_putt": "sg_putting",
    "driving_distance": "driving_distance",
    "driving_accuracy": "driving_accuracy",
    "greens_in_regulation": "greens_in_regulation",
    "scoring_average": "scoring_average",
    "money_earned": "money_earned",
    "fedex_rank": "final_season_rank",
}


def load_stat_for_year(stat_name: str, year: int, col_name: str) -> pd.DataFrame:
    """
    Load one intermediate CSV like 'sg_total_2025.csv'
    and return columns: year, player_name, <col_name>.
    Returns None if the file is empty (only headers) or has no data rows.
    """
    path = INTER_DIR / f"{stat_name}_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing intermediate file: {path}")

    df = pd.read_csv(path)
    
    # Check if file is empty (only headers, no data rows)
    if len(df) == 0:
        return None
    
    # Ensure expected columns exist
    if "year" not in df.columns or "player_name" not in df.columns:
        raise ValueError(f"File {path} is missing 'year' or 'player_name' columns.")

    if col_name not in df.columns:
        raise ValueError(f"File {path} is missing value column '{col_name}'.")

    # Filter out rows with missing player_name (these shouldn't exist, but just in case)
    df = df[df["player_name"].notna()].copy()
    
    if len(df) == 0:
        return None
        
    return df[["year", "player_name", col_name]]


def build_for_year(year: int) -> pd.DataFrame:
    """
    Merge all stats for a single year into one DataFrame.
    Uses OUTER join to keep all players even if they don't have all stats.
    Missing values will be NaN, which is better than losing players entirely.
    """
    frames = []
    stats_loaded = []
    stats_skipped = []
    stats_empty = []
    
    for stat_name, col_name in STAT_COLUMNS.items():
        try:
            df_stat = load_stat_for_year(stat_name, year, col_name)
            if df_stat is None:
                stats_empty.append(stat_name)
                print(f"[WARN] {stat_name}_{year}.csv is empty (no data rows) — skipping.")
                continue
            
            frames.append(df_stat)
            stats_loaded.append(stat_name)
            
        except FileNotFoundError as e:
            stats_skipped.append(stat_name)
            print(f"[WARN] {e} — skipping this stat/year.")
            continue
        except Exception as e:
            stats_skipped.append(stat_name)
            print(f"[ERROR] Failed to load {stat_name}_{year}.csv: {e} — skipping.")
            continue

    if not frames:
        raise ValueError(
            f"No intermediate files with data found for year {year}. "
            f"Skipped: {stats_skipped}, Empty: {stats_empty}"
        )
    
    print(f"    Loaded {len(stats_loaded)} stats: {', '.join(stats_loaded)}")
    if stats_skipped:
        print(f"    Skipped {len(stats_skipped)} stats (missing files): {', '.join(stats_skipped)}")
    if stats_empty:
        print(f"    Empty {len(stats_empty)} stats (no data): {', '.join(stats_empty)}")

    # Use OUTER join to keep all players, even if they don't have all stats
    # This prevents data loss when different stats have different player sets
    merged = reduce(
        lambda left, right: pd.merge(
            left, right, on=["year", "player_name"], how="outer"
        ),
        frames,
    )
    
    # Count players before and after merge to show what we're getting
    if len(frames) > 1:
        player_counts = [len(df) for df in frames]
        print(f"    Player counts per stat: {dict(zip(stats_loaded, player_counts))}")
        print(f"    Final merged dataset: {len(merged)} players (may be more due to outer join)")

    # Compute tee-to-green
    if all(c in merged.columns for c in ["sg_off_the_tee", "sg_approach", "sg_around_green"]):
        merged["sg_tee_to_green"] = (
            merged["sg_off_the_tee"]
            + merged["sg_approach"]
            + merged["sg_around_green"]
        )
    else:
        print("[WARN] Missing one of the SG components; cannot compute sg_tee_to_green.")

    # Ensure consistent column order across all years
    # Define the expected column order
    expected_cols = [
        "year",
        "player_name",
        "sg_total",
        "sg_off_the_tee",
        "sg_approach",
        "sg_around_green",
        "sg_putting",
        "driving_distance",
        "driving_accuracy",
        "greens_in_regulation",
        "scoring_average",
        "money_earned",
        "final_season_rank",
        "sg_tee_to_green",
    ]
    
    # Reorder columns to match expected order (add any missing columns as NaN)
    for col in expected_cols:
        if col not in merged.columns:
            merged[col] = None
            print(f"[WARN] Added missing column '{col}' with NaN values for year {year}")
    
    # Reorder to match expected order
    merged = merged[expected_cols]
    
    # Sort nicely
    merged = merged.sort_values(["year", "player_name"]).reset_index(drop=True)
    return merged


def main():
    # Infer available years from intermediate filenames like '<stat>_2025.csv'
    years = set()
    for path in INTER_DIR.glob("*.csv"):
        stem = path.stem  # e.g. 'sg_total_2025'
        try:
            _, year_str = stem.rsplit("_", 1)
            years.add(int(year_str))
        except ValueError:
            continue

    if not years:
        print("No intermediate files found in data/intermediate/. Run parse_stats.py first.")
        return

    years = sorted(years)
    print(f"Building master dataset for years: {years}")

    all_years = []
    for y in years:
        print(f"  -> Merging stats for {y}")
        df_year = build_for_year(y)
        out_year = PROC_DIR / f"master_{y}.csv"
        df_year.to_csv(out_year, index=False)
        print(f"     Saved {out_year} ({len(df_year)} rows)")
        all_years.append(df_year)

    master = pd.concat(all_years, ignore_index=True)
    out_master = PROC_DIR / "master_player_seasons.csv"
    master.to_csv(out_master, index=False)
    print(f"\nWrote {out_master} ({len(master)} total rows)")


if __name__ == "__main__":
    main()
