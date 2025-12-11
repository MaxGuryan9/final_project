from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/master_player_seasons.csv")


def _load_master():
    """Helper to load the master CSV."""
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


def test_master_file_exists():
    """Master player-season file should exist."""
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"


def test_master_file_not_empty():
    """Master file should have at least one row."""
    df = _load_master()
    assert len(df) > 0, "master_player_seasons.csv is empty"


def test_master_has_minimal_expected_columns():
    """
    Master file should contain the core columns that the app needs.
    We only require a minimal set so the pipeline can evolve without breaking tests.
    """
    df = _load_master()

    expected_min_cols = {
        "year",
        "player_name",
        "sg_total",
        "sg_off_the_tee",
        "sg_approach",
        "sg_around_green",
        "sg_putting",
        "sg_tee_to_green",
        "driving_distance",
        "driving_accuracy",
        "greens_in_regulation",
        "scoring_average",
        "money_earned",
        "final_season_rank",
    }

    missing = expected_min_cols - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_years_covered_2022_to_2025():
    """We expect to cover at least 2022–2025 in the master data."""
    df = _load_master()
    years = set(df["year"].unique())
    for year in [2022, 2023, 2024, 2025]:
        assert year in years, f"Year {year} missing from master_player_seasons.csv"


def test_sg_tee_to_green_consistency():
    """
    Check that sg_tee_to_green is (roughly) the sum of OTT + Approach + ARG.

    We allow a tiny numerical tolerance because of floating-point rounding,
    but the max absolute difference should be very small.
    """
    df = _load_master()

    required_cols = ["sg_off_the_tee", "sg_approach", "sg_around_green", "sg_tee_to_green"]
    for col in required_cols:
        assert col in df.columns, f"Column {col} missing from master_player_seasons.csv"

    # Drop rows with missing components
    sub = df.dropna(subset=required_cols).copy()
    assert len(sub) > 0, "No complete rows to check sg_tee_to_green consistency"

    computed = sub["sg_off_the_tee"] + sub["sg_approach"] + sub["sg_around_green"]
    diff = (sub["sg_tee_to_green"] - computed).abs()

    # Allow for tiny floating-point noise, but not large discrepancies
    max_diff = diff.max()
    assert np.isfinite(max_diff), "Non-finite differences found in sg_tee_to_green check"
    assert max_diff < 1e-6, f"sg_tee_to_green differs from component sum by up to {max_diff}"
