# src/parse_stats.py

from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
INTER_DIR = Path("data/intermediate")
INTER_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Spec for each stat:
#  - output_col: how we want the column named in intermediate CSVs
#  - hints: list of substrings to search for in the raw CSV column names
#           to identify the "value" column for that stat.
# ---------------------------------------------------------------------
STAT_SPEC = {
    "sg_total": {
        "output_col": "sg_total",
        "hints": ["mean", "avg", "average", "sg:total"],
    },
    "sg_ott": {
        "output_col": "sg_off_the_tee",
        "hints": ["mean", "avg", "average"],
    },
    "sg_app": {
        "output_col": "sg_approach",
        "hints": ["mean", "avg", "average"],
    },
    "sg_arg": {
        "output_col": "sg_around_green",
        "hints": ["mean", "avg", "average"],
    },
    "sg_putt": {
        "output_col": "sg_putting",
        "hints": ["mean", "avg", "average"],
    },
    "driving_distance": {
        "output_col": "driving_distance",
        "hints": ["avg", "average", "distance"],
    },
    "driving_accuracy": {
        "output_col": "driving_accuracy",
        "hints": ["%", "pct", "accuracy"],
    },
    "greens_in_regulation": {
        "output_col": "greens_in_regulation",
        "hints": ["%", "pct", "greens in reg"],
    },
    "scoring_average": {
        "output_col": "scoring_average",
        "hints": ["avg", "average", "mean"],
    },
    "money_earned": {
        "output_col": "money_earned",
        "hints": ["money", "earnings", "$", "amount"],
    },
    "fedex_rank": {
        "output_col": "final_season_rank",
        "hints": ["rank", "points", "fedexcup", "strokes", "finish position"],
    },
}


def normalize_col_name(col: str) -> str:
    """Lowercase, strip spaces, collapse internal spaces for comparison."""
    return " ".join(col.strip().lower().split())


def find_player_name_column(df: pd.DataFrame) -> str:
    """
    Try to locate the player name column in the raw CSV.

    Priority:
      1. Columns that contain 'player' but NOT 'id'  (e.g. 'PLAYER')
      2. Columns that contain 'name'                (e.g. 'Player Name')
      3. Fallback: first column
    """
    candidates = []

    for col in df.columns:
        norm = normalize_col_name(col)

        # Prefer 'player' columns that are not IDs
        if "player" in norm and "id" not in norm:
            candidates.append((0, col))
        # Also allow generic 'name' columns
        elif "name" in norm:
            candidates.append((1, col))

    if candidates:
        # Sort so 'player' (priority 0) wins over 'name' (priority 1)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # Fallback: just assume the first column is the player name
    return df.columns[0]


def find_value_column(df: pd.DataFrame, stat_name: str) -> str:
    """
    Use STAT_SPEC hints to identify which column holds the value we care about.
    If hints don't find anything, fall back to a numeric-ish heuristic.
    """
    spec = STAT_SPEC.get(stat_name)
    if spec is None:
        raise ValueError(f"No STAT_SPEC entry for stat_name={stat_name!r}")

    hints = [h.lower() for h in spec["hints"]]

    # Normalize column names once
    norm_map = {col: normalize_col_name(col) for col in df.columns}

    # 1) Try hint-based matching
    for col, norm in norm_map.items():
        for h in hints:
            if h in norm:
                return col

    # 2) Fallback: try to find a numeric column that isn't obviously rank or name
    player_col = find_player_name_column(df)
    bad_keywords = ["rank", "this week", "last week", "event", "tournament", "round", "movement"]

    numeric_candidates = []
    for col in df.columns:
        if col == player_col:
            continue
        norm = norm_map[col]
        if any(b in norm for b in bad_keywords):
            continue

        # Try to coerce to numeric and see how many values survive
        s = pd.to_numeric(
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)  # Handle percentage values
            .str.strip(),
            errors="coerce",
        )
        non_null = s.notna().sum()
        if non_null > 0:
            numeric_candidates.append((col, non_null))

    if not numeric_candidates:
        raise ValueError(
            f"Could not identify a numeric value column for stat {stat_name!r}. "
            f"Available columns: {list(df.columns)}"
        )

    # Pick the column with the most numeric values
    numeric_candidates.sort(key=lambda x: x[1], reverse=True)
    best_col = numeric_candidates[0][0]
    print(
        f"[WARN] Using fallback numeric column '{best_col}' for stat {stat_name!r}. "
        "You may want to adjust STAT_SPEC hints if this looks wrong."
    )
    return best_col


def parse_one_file(path: Path) -> pd.DataFrame:
    """
    Parse a single raw CSV into an intermediate per-stat DataFrame with columns:
      year, player_name, <stat_column>
    """

    # Example filename: sg_total_2025.csv → stat_name='sg_total', year='2025'
    stem = path.stem  # 'sg_total_2025'
    try:
        stat_name, year_str = stem.rsplit("_", 1)
    except ValueError:
        raise ValueError(
            f"Expected filename like '<stat_name>_<year>.csv', got {path.name}"
        )
    year = int(year_str)

    if stat_name not in STAT_SPEC:
        raise ValueError(
            f"stat_name {stat_name!r} not found in STAT_SPEC. "
            f"Please add it there before parsing."
        )

    spec = STAT_SPEC[stat_name]
    output_col = spec["output_col"]

    # Read CSV with error handling for malformed files
    # Some files (like fedex_rank_2023+) have more data columns than header columns
    # Handle fedex_rank files specially since they have inconsistent structure
    if stat_name == "fedex_rank":
        # For fedex_rank, 2023+ files have 8 columns but only 7 header fields
        # Read with 8 columns specified, including the extra FEDEXCUP STROKES column
        fedex_cols_8 = ['RANK', 'MOVEMENT', 'PLAYER_ID', 'PLAYER', 'FINISH POSITION', '# OF WINS', '# OF TOP-10S', 'FEDEXCUP STROKES']
        try:
            # Read with 8 columns, skipping the header row in the file
            df_raw = pd.read_csv(path, names=fedex_cols_8, header=None, skiprows=1, engine='python')
        except Exception as e:
            # Fallback: try reading normally (for 2022 files which have all 8 columns in header)
            try:
                df_raw = pd.read_csv(path, engine='python')
            except Exception:
                # Last resort: skip bad lines
                try:
                    df_raw = pd.read_csv(path, names=fedex_cols_8, header=None, skiprows=1, on_bad_lines='skip', engine='python')
                    print(f"[WARN] File {path.name} had parsing errors - some rows may have been skipped.")
                except TypeError:
                    df_raw = pd.read_csv(path, names=fedex_cols_8, header=None, skiprows=1, error_bad_lines=False, warn_bad_lines=False, engine='python')
                    print(f"[WARN] File {path.name} had parsing errors - some rows may have been skipped.")
    else:
        # For other stats, read normally
        try:
            df_raw = pd.read_csv(path)
        except pd.errors.ParserError:
            # If parsing fails, try with python engine
            try:
                df_raw = pd.read_csv(path, engine='python')
            except Exception:
                try:
                    df_raw = pd.read_csv(path, on_bad_lines='skip', engine='python')
                    print(f"[WARN] File {path.name} had parsing errors - some rows may have been skipped.")
                except TypeError:
                    df_raw = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, engine='python')
                    print(f"[WARN] File {path.name} had parsing errors - some rows may have been skipped.")

    player_col = find_player_name_column(df_raw)
    value_col = find_value_column(df_raw, stat_name)

    # Coerce numeric value - strip common non-numeric characters
    value_series = pd.to_numeric(
        df_raw[value_col].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)  # Handle percentage values like "72.57%"
        .str.strip(),
        errors="coerce",
    )

    out = pd.DataFrame(
        {
            "year": year,
            "player_name": df_raw[player_col].astype(str).str.strip(),
            output_col: value_series,
        }
    )

    # Drop rows with missing player or value, then drop duplicate player rows
    out = out.dropna(subset=["player_name", output_col])
    out = out.drop_duplicates(subset=["player_name"])

    return out


def main():
    csv_paths = sorted(RAW_DIR.glob("*.csv"))

    if not csv_paths:
        print("No CSV files found in data/raw/. Run download_stats.py first.")
        return

    for path in csv_paths:
        try:
            df = parse_one_file(path)
        except Exception as e:
            print(f"[ERROR] Skipping {path.name} due to: {e}")
            continue

        stem = path.stem  # e.g. 'sg_total_2025'
        out_path = INTER_DIR / f"{stem}.csv"

        df.to_csv(out_path, index=False)
        if len(df) == 0:
            print(f"[WARN] Wrote empty intermediate file: {out_path} (0 rows - check raw CSV)")
        else:
            print(f"Wrote intermediate file: {out_path} ({len(df)} rows)")

    print("Parsing complete. Check data/intermediate/ for per-stat CSVs.")


if __name__ == "__main__":
    main()
