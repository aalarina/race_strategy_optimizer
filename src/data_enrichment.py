from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Loading
# ============================================================

def load_raw_data():
    """
    Load raw lap and weather datasets from Parquet files.
    """

    laps_path = RAW_DATA_DIR / "laps.parquet"
    weather_path = RAW_DATA_DIR / "weather.parquet"

    laps = pd.read_parquet(laps_path)
    weather = pd.read_parquet(weather_path)

    return laps, weather


# ============================================================
# Time normalization
# ============================================================

def prepare_time_columns(laps, weather):
    """
    Normalize time columns before performing a time-aware merge.
    """

    laps = laps.copy()
    weather = weather.copy()

    laps["Time"] = pd.to_timedelta(laps["Time"])
    weather["Time"] = pd.to_timedelta(weather["Time"])

    # Normalize categorical merge keys
    laps["GrandPrix"] = laps["GrandPrix"].astype("string")
    weather["GrandPrix"] = weather["GrandPrix"].astype("string")

    # Normalize date columns
    laps["Date"] = pd.to_datetime(laps["Date"])
    weather["Date"] = pd.to_datetime(weather["Date"])

    # Make sure year has the same numeric type
    laps["Year"] = laps["Year"].astype("int64")
    weather["Year"] = weather["Year"].astype("int64")


    return laps, weather


# ============================================================
# Weather enrichment
# ============================================================

def merge_weather_data(laps, weather):
    """
    Attach the nearest weather observation to each lap.
    """

    laps = laps.copy()
    weather = weather.copy()

    # Preserve the original weather observation timestamp
    weather["WeatherTime"] = weather["Time"]

    # merge_asof requires the time key to be sorted
    laps = laps.sort_values(
        ["Time", "Year", "GrandPrix", "Date"]
    ).reset_index(drop=True)

    weather = weather.sort_values(
        ["Time", "Year", "GrandPrix", "Date"]
    ).reset_index(drop=True)

    # Keep only weather columns needed for enrichment
    weather_columns = [
        "Year",
        "GrandPrix",
        "Date",
        "Time",
        "WeatherTime",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindDirection",
        "WindSpeed",
    ]

    weather = weather[weather_columns]

    enriched_laps = pd.merge_asof(
        laps,
        weather,
        on="Time",
        by=["Year", "GrandPrix", "Date"],
        direction="backward",
    )

    # Calculate the age of the matched weather observation.
    enriched_laps["WeatherTimeDelta"] = (
        enriched_laps["Time"] - enriched_laps["WeatherTime"]
    )

    # Flag laps where the matched weather observation is older than one minute.
    enriched_laps["IsWeatherFresh"] = (
        enriched_laps["WeatherTimeDelta"]
        <= pd.Timedelta(minutes=1)
    )

    return enriched_laps



# ============================================================
# Validation
# ============================================================

def validate_weather_merge(original_laps, enriched_laps):
    """
    Check whether weather variables were successfully merged.
    """
    print("\n=== Weather enrichment validation ===")

    print(f"Original laps shape:  {original_laps.shape}")
    print(f"Enriched laps shape:  {enriched_laps.shape}")

    print("\nMissing weather values:")

    weather_columns = [
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindDirection",
        "WindSpeed",
    ]

    print("\nWeather columns after merge:\n")

    print(
        enriched_laps[weather_columns]
        .isna()
        .sum()
    )

    print("\nTrack temperature statistics:\n")

    print(
        enriched_laps["TrackTemp"]
        .describe()
    )

    print("\nWeather freshness:")

    print(
        enriched_laps["IsWeatherFresh"]
        .value_counts()
    )

    print(
    "\nWeather time delta:"
)

    print(
        enriched_laps["WeatherTimeDelta"]
        .describe()
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    laps, weather = load_raw_data()

    print("Raw lap shape:", laps.shape)
    print("Raw weather shape:", weather.shape)

    # Normalize merge keys and time columns
    laps, weather = prepare_time_columns(
        laps,
        weather,
    )

    # Merge weather observations into lap data
    enriched_laps = merge_weather_data(
        laps,
        weather,
    )

    # Validate the result
    validate_weather_merge(
        laps,
        enriched_laps,
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "enriched_laps.parquet"
    )

    enriched_laps.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nSaved enriched data to {output_path}"
    )
