import fastf1
import pandas as pd
import os

# Set the cache folder name
cache_folder_name = 'f1_local_cache'

# Get the absolute path in the Windows system for this folder
cache_path = os.path.abspath(cache_folder_name)

# Create the cache folder if it doesn't exist
os.makedirs(cache_path, exist_ok=True)

# Inform about the cache path
print(f"Connecting to cache at: {cache_path}")

# Enable the cache
fastf1.Cache.enable_cache(cache_path)
print("Cache enabled successfully!")


def load_and_clean_data(year, grnd_prix_list):
    """Retrieves race data, cleans it and return a ready-to-use DataFrame"""
    all_laps = []

    for gp in grnd_prix_list:
        try:
            print(f"Request for {year} - GrandPrix {gp}...")
            session = fastf1.get_session(year, gp, 'R')

            try:
                session.load(telemetry=False, weather=True)
            except Exception as load_err:
                print(f"WARNING: We were unable to fully load all thedata for {gp} ({load_err}). Let's try again without the weather data...")
                session.load(telemetry=False, weather=False)

            laps = session.laps.copy()
            if laps.empty:
                print(f"No lap data for {gp}, skip it")
                continue

            # Safely retrieve the track temperature
            try:
                if hasattr(session, 'weather_data') and not session.weather_data.empty:
                    # Add the average track temperature
                    track_temp = session.weather_data['TrackTemp'].mean()
                else:
                    track_temp = 30.0
            except Exception:
                track_temp = 30.0

            laps['TrackTemp'] = track_temp
            laps['GrandPrix'] = gp

            all_laps.append(laps)
            print(f"{gp} successfully processed ({len(laps)} laps loaded).")

        except Exception as e:
            print(f"Stage loading error {gp}: {e}")

    if not all_laps:
        print("Attention: No stages were loaded")
        return pd.DataFrame()

    df = pd.concat(all_laps, ignore_index=True)

    # Filter the necessary columns and remove anomalies (yellow flags, safety car)
    features= ['GrandPrix', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'TrackTemp', 'TrackStatus']
    df = df[features].dropna()
    df = df[df['TrackStatus'] == '1'] # Only clean racing laps

    # Convert the time of a lap to seconds
    df['LapTime_Seconds'] = df['LapTime'].dt.total_seconds()

    return df.drop(columns=['LapTime', 'TrackStatus'])

if __name__ == "__main__":
    # Execute only if this file is run manually
    os.makedirs('./data', exist_ok=True)
    print("Start collecting data...")
    tracks = ['Bahrain', 'Austria', 'Silverstone']
    cleaned_df = load_and_clean_data(2024, tracks)

    cleaned_df.to_csv('./data/f1_clean_data_from_ml.csv', index=False)
    print("The clean data has been saved in './data/f1_clean_data_from_ml.csv'")