import os
import time
import requests
import pandas as pd
from time import sleep
from datetime import datetime

# ===============================================================
# 🌍 OpenF1 Weather Data Fetcher (All Seasons 1950–Present)
# ===============================================================

BASE_URL = "https://api.openf1.org/v1/"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data_fetching_openf1", "openf1_all_weather_data.csv")

CURRENT_YEAR = datetime.now().year
START_YEAR = 1950  # F1 began in 1950

# ---------------------------------------------------------------
# 🧩 Helper function to fetch from API with retry logic
# ---------------------------------------------------------------
def fetch(endpoint, params=None, retries=3):
    """Generic API fetch helper with retries and error handling."""
    url = BASE_URL + endpoint
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code == 422:
                print(f"⚠️ Invalid params for {endpoint}: {params}")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {endpoint} ({params}): {e}")
            time.sleep(2)
    print(f"❌ All retries failed for {endpoint} ({params}).")
    return None


# ---------------------------------------------------------------
# 📅 Fetch meetings for a specific year
# ---------------------------------------------------------------
def fetch_meetings_for_year(year):
    """Fetch all meetings for a given season."""
    data = fetch("meetings", params={"year": year})
    if data:
        df = pd.DataFrame(data)
        print(f"✅ {len(df)} meetings found for {year}.")
        return df
    print(f"⚠️ No meetings for {year}.")
    return pd.DataFrame()


# ---------------------------------------------------------------
# 🌦️ Fetch weather for a meeting
# ---------------------------------------------------------------
def fetch_weather_for_meeting(meeting):
    """Fetch weather for a given meeting row."""
    mk = meeting["meeting_key"]
    data = fetch("weather", params={"meeting_key": mk})
    if not data:
        return None

    w_df = pd.DataFrame(data)
    # add contextual metadata
    w_df["year"] = meeting["year"]
    w_df["meeting_name"] = meeting["meeting_name"]
    w_df["country_name"] = meeting["country_name"]
    w_df["location"] = meeting["location"]
    return w_df


# ---------------------------------------------------------------
# 🚀 Main execution function
# ---------------------------------------------------------------
def main():
    all_weather = []

    for y in range(START_YEAR, CURRENT_YEAR + 1):
        print(f"\n📅 ===== Fetching meetings for year {y} =====")
        meetings_df = fetch_meetings_for_year(y)
        if meetings_df.empty:
            continue

        for idx, (_, meeting) in enumerate(meetings_df.iterrows(), 1):
            print(f"🌦️ [{idx}/{len(meetings_df)}] {meeting['meeting_name']} ({y}) ...")
            w_df = fetch_weather_for_meeting(meeting)
            if w_df is not None and not w_df.empty:
                all_weather.append(w_df)
            sleep(0.5)

        # partial save every 20 seasons
        if (y - START_YEAR) % 20 == 0 and all_weather:
            temp_file = OUTPUT_FILE.replace(".csv", f"_partial_{y}.csv")
            pd.concat(all_weather, ignore_index=True).to_csv(temp_file, index=False)
            print(f"💾 Partial save up to year {y} → {temp_file}")

    if not all_weather:
        print("⚠️ No weather data fetched for any season.")
        return

    weather_df = pd.concat(all_weather, ignore_index=True)
    weather_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Finished! Saved {len(weather_df)} weather records → {OUTPUT_FILE}")


# ---------------------------------------------------------------
# 🏁 Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
