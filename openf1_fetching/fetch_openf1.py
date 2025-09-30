# openf1_fetching/fetch_openf1.py
import os
import requests
import pandas as pd

# Base API URL
BASE_URL = "https://api.openf1.org/v1/"

# Central directory for OpenF1 data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_fetching_openf1")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_data(endpoint: str, params: dict = None, save_as: str = None):
    """
    Fetch data from OpenF1 API endpoint and save as CSV.

    Args:
        endpoint (str): API endpoint, e.g. "laps"
        params (dict): query parameters
        save_as (str): filename for saving CSV inside data_fetching_openf1/
    """
    url = BASE_URL + endpoint
    print(f"📡 Fetching {url} with params={params}...")

    response = requests.get(url, params=params)
    if response.status_code == 422:
        print(f"⚠️ Invalid request for {endpoint}. Skipping.")
        return None

    response.raise_for_status()
    data = response.json()

    if not data:
        print(f"⚠️ No data returned for {endpoint}.")
        return None

    df = pd.DataFrame(data)

    if save_as:
        if not save_as.endswith("_openf1.csv"):
            save_as = save_as.replace(".csv", "") + "_openf1.csv"
        output_path = os.path.join(DATA_DIR, save_as)
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} rows → {output_path}")

    return df


if __name__ == "__main__":
    # Example: Bahrain GP 2024 session key
    session_key = 9158

    # --- Core endpoints ---
    fetch_data("sessions", {"year": 2024}, "sessions_2024")
    fetch_data("laps", {"session_key": session_key}, "bahrain_laps_2024")
    fetch_data("weather", {"session_key": session_key}, "bahrain_weather_2024")
    fetch_data("drivers", {"session_key": session_key}, "bahrain_drivers_2024")

    # --- Extra race dynamics ---
    fetch_data("intervals", {"session_key": session_key}, "bahrain_intervals_2024")
    fetch_data("stints", {"session_key": session_key}, "bahrain_stints_2024")
    fetch_data("overtakes", {"session_key": session_key}, "bahrain_overtakes_2024")

    # --- Location (⚠️ very large; filtering recommended) ---
    fetch_data("location", {"session_key": session_key, "driver_number": 1}, "bahrain_location_driver1_2024")
