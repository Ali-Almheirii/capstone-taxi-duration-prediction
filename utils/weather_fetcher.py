import requests
import pandas as pd
import time

def fetch_weather_data(retries=3, delay=2):
    params = {
        "latitude": 25.2528,
        "longitude": 55.3644,
        "start_date": "2024-12-31",
        "end_date": "2025-06-01",
        "hourly": ["temperature_2m", "wind_gusts_10m", "relative_humidity_2m", "precipitation", "wind_speed_10m",
                   "weather_code", "cloud_cover", "surface_pressure"]
    }

    url = "https://archive-api.open-meteo.com/v1/archive"

    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt} to fetch weather data...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()
            df_weather = pd.DataFrame(data["hourly"])
            df_weather["time"] = pd.to_datetime(df_weather["time"])
            print("✅ Weather data fetched successfully.")
            return df_weather
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError("Failed to fetch weather data after multiple attempts.")
    return None


# Example usage
if __name__ == "__main__":
    weather_df = fetch_weather_data()
    print(weather_df.head())
