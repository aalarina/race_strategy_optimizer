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
    """Retrieves race data, filters anomalies and return a clean ready-to-use DataFrame"""
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

            laps.loc[:, 'LapTime_Seconds'] = laps['LapTime'].dt.total_seconds()

            # Safely retrieve the track temperature
            try:
                if hasattr(session, 'weather_data') and not session.weather_data.empty:
                    # Add the average track temperature
                    track_temp = session.weather_data['TrackTemp'].mean()
                else:
                    track_temp = 30.0
            except Exception:
                track_temp = 30.0

            laps.loc[:, 'TrackTemp'] = track_temp
            laps.loc[:, 'GrandPrix'] = gp

            # Filtration of anomalies (outliers)
            # Find the best lap time for this race
            best_lap_time = laps['LapTime_Seconds'].min()

            # Find the median tempo of this race  
            # Pitstop laps and accidences have anomalous lap times
            typical_lap_time = laps['LapTime_Seconds'].median()

            # Leave only laps that are slower then median + 5 seconds 
            clean_laps = laps[laps['LapTime_Seconds'] <= (typical_lap_time + 5)].copy() 

            # Create new variable  
            clean_laps.loc[:, 'LapTime_Delta'] = clean_laps['LapTime_Seconds'] - best_lap_time

            print(f"Anomalies removed: {len(laps) - len(clean_laps)} laps. Clean laps: {len(clean_laps)}")

            all_laps.append(clean_laps)
            print(f"{gp} successfully processed ({len(clean_laps)} laps loaded).")

        except Exception as e:
            print(f"Stage loading error {gp}: {e}")

    if not all_laps:
        print("Attention: No stages were loaded")
        return pd.DataFrame()

    df = pd.concat(all_laps, ignore_index=True)

    # Filter the necessary columns and remove anomalies (yellow flags, safety car)
    features= ['GrandPrix', 'Driver', 'LapNumber', 'Compound', 'TyreLife', 'TrackTemp', 'LapTime_Seconds', 'LapTime_Delta']
    df = df[features].dropna()
  
    return df

if __name__ == "__main__":
    os.makedirs('./data', exist_ok=True)
    print("Start collecting data...")

    # tracks = ['Bahrain', 'Austria', 'Silverstone', 'Monza', 'Spa', 'Monaco', 'Barcelona', 'Singapore']
    tracks_to_download = [
        'Bahrain', 'Austria', 'Great Britain', 
        'Italy', 'Belgium', 'Monaco', 
        'Spain', 'Singapore'
    ]

    cleaned_df = load_and_clean_data(2024, tracks_to_download)

    cleaned_df.to_csv('./data/f1_clean_data_for_ml.csv', index=False)
    print("The clean data has been saved in './data/f1_clean_data_for_ml.csv'")