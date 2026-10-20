import pandas as pd

from pathlib import Path


RAW_DATA_DIR = Path("data/raw")


def inspect_lap_data(df):
    """
    Print a basic quality report for lap-level data.
    """

    print("=" * 60)
    print("LAP DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nMissing values:")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nUnique values:")
    for column in [
        "Driver",
        "Compound",
        "TrackStatus",
        "GrandPrix",
    ]:
        if column in df.columns:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False))

    print("\nNumeric ranges:")

    for column in [
        "LapNumber",
        "Stint",
        "TyreLife",
    ]:
        if column in df.columns:
            print(
                f"{column}: "
                f"{df[column].min()} -> "
                f"{df[column].max()}"
            )

    print("\nLap time statistics:")

    if "LapTime" in df.columns:
        lap_seconds = (
            df["LapTime"]
            .dt.total_seconds()
        )

        print(
            lap_seconds.describe()
        )


def inspect_weather_data(df):
    """
    Print a basic quality report for weather data.
    """

    print("=" * 60)
    print("WEATHER DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nMissing values:")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nWeather statistics:")

    numeric_columns = [
        "AirTemp",
        "Humidity",
        "Pressure",
        "TrackTemp",
        "WindDirection",
        "WindSpeed",
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    print(
        df[available_columns].describe()
    )

    print("\nRainfall values:")
    print(
        df["Rainfall"]
        .value_counts(dropna=False)
    )


def load_raw_data():
    """
    Load raw Parquet datasets.
    """

    laps_path = RAW_DATA_DIR / "laps.parquet"
    weather_path = RAW_DATA_DIR / "weather.parquet"

    laps = pd.read_parquet(laps_path)
    weather = pd.read_parquet(weather_path)

    return laps, weather


if __name__ == "__main__":

    laps, weather = load_raw_data()

    inspect_lap_data(laps)
    inspect_weather_data(weather)