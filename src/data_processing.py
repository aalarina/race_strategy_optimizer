from pathlib import Path

import pandas as pd

from tyre_life_repair import repair_compound_change_tyre_life

PROCESSED_DATA_DIR = Path("data/processed")

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Loading
# ============================================================

def load_enriched_data():
    """
    Load enriched dataset from Parquet files.
    """

    ENRICHED_DATA_PATH = PROCESSED_DATA_DIR / "enriched_laps.parquet"

    enriched_laps = pd.read_parquet(ENRICHED_DATA_PATH)

    return enriched_laps

# ============================================================
# Track status
# ============================================================

def add_track_status_flags(df):
    """
    Add flags describing whether a lap was completed
    under normal track conditions.
    """

    df = df.copy()

    # Normalize TrackStatus values before comparison.
    df["TrackStatus"] = (
        df["TrackStatus"]
        .astype("string")
        .str.strip()
    )

    # TrackStatus == "1" corresponds to normal track conditions.
    df["IsNormalTrack"] = (
        df["TrackStatus"].eq("1")
    )

    return df


# ============================================================
# Lap quality flags
# ============================================================

def add_lap_quality_flags(df):
    """
    Add flags describing lap quality without removing observations.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Pit stop flags
    # --------------------------------------------------------

    df["IsPitInLap"] = (
        df["PitInTime"].notna()
    )

    df["IsPitOutLap"] = (
        df["PitOutTime"].notna()
    )

    df["IsPitLap"] = (
        df["IsPitInLap"]
        | df["IsPitOutLap"]
    )

    # --------------------------------------------------------
    # Compound classification
    # --------------------------------------------------------

    dry_compounds = {
        "SOFT",
        "MEDIUM",
        "HARD",
    }

    df["IsDryCompound"] = (
        df["Compound"].isin(dry_compounds)
    )

    # --------------------------------------------------------
    # Lap time validity
    # --------------------------------------------------------

    df["LapTimeSeconds"] = (
        df["LapTime"]
        .dt.total_seconds()
    )

    df["HasLapTime"] = (
        df["LapTimeSeconds"].notna()
    )

    # --------------------------------------------------------
    # Gross lap-time outlier flag
    # --------------------------------------------------------

    df["IsReasonableLapTime"] = (
        df["LapTimeSeconds"].between(
            50,
            180,
        )
    )

    # --------------------------------------------------------
    # Tyre information
    # --------------------------------------------------------

    df["HasTyreLife"] = (
        df["TyreLife"].notna()
    )

    # --------------------------------------------------------
    # Personal-best information
    # --------------------------------------------------------

    df["IsPersonalBest"] = (
        df["IsPersonalBest"]
        .astype("boolean")
        .fillna(False)
    )

    # --------------------------------------------------------
    # FastF1 accuracy flag
    # --------------------------------------------------------

    df["IsAccurate"] = (
        df["IsAccurate"]
        .astype("boolean")
    )

    # --------------------------------------------------------
    # Track status
    # --------------------------------------------------------

    df = add_track_status_flags(df)

    return df


# ============================================================
# Candidate dataset
# ============================================================

def create_candidate_laps(df):
    """
    Create a candidate dataset for dry tyre degradation modeling.

    Only dry-compound laps with valid lap time and tyre-life
    information are retained. Laps affected by pit entry/exit,
    abnormal track conditions or FastF1 accuracy issues are excluded.
    """

    candidate_laps = df[
        df["IsDryCompound"]
        & df["HasLapTime"]
        & df["HasTyreLife"]
        & df["IsReasonableLapTime"]
        & df["IsNormalTrack"]
        & df["IsAccurate"]
        & ~df["IsPitLap"]
        ].copy()

    return candidate_laps


# ============================================================
# Saving
# ============================================================

def save_processed_data(df):
    """
    Save the processed lap dataset to Parquet.
    """

    output_path = (
        PROCESSED_DATA_DIR
        / "laps_with_quality_flags.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Saved processed data to: "
        f"{output_path}"
    )


# ============================================================
# Pipeline
# ============================================================

def build_processed_dataset():
    """
    Build the processed lap dataset.
    """

    enriched_laps = load_enriched_data()

    print(f"Enriched laps: {len(enriched_laps):,}")
    print("\nColumns:")
    print(enriched_laps.columns.tolist())

    # Repair TyreLife metadata before creating quality flags.
    laps = repair_compound_change_tyre_life(enriched_laps)

    # Add all quality-related flags.
    laps = add_lap_quality_flags(laps)

    print(
        "\nQuality flags:"
    )

    flag_columns = [
        "IsPitLap",
        "IsDryCompound",
        "HasLapTime",
        "IsReasonableLapTime",
        "HasTyreLife",
        "IsNormalTrack",
        "IsAccurate",
    ]

    for column in flag_columns:
        print(
            f"{column}: "
            f"{laps[column].sum():,}"
        )

    # Create the clean candidate/training dataset.
    candidate_laps = create_candidate_laps(
        laps
    )

    print(
        f"\nCandidate dry laps: "
        f"{len(candidate_laps):,}"
    )

    save_processed_data(
        laps
    )

    return laps, candidate_laps


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":

    laps, candidate_laps = (
        build_processed_dataset()
    )

    print(
        "\nTraining dataset shape:",
        candidate_laps.shape,
    )

    print(
        "\nTraining dataset compounds:"
    )

    print(
        candidate_laps["Compound"]
        .value_counts()
    )

