import pandas as pd

def map_weather_to_flights(flights_df, weather_df, time_col="ACTUAL_OFFBLOCK_TIME_AOBT_UTC"):
    flights_df = flights_df.sort_values(time_col)
    weather_df = weather_df.sort_values("time")

    enriched = pd.merge_asof(
        flights_df,
        weather_df,
        left_on=time_col,
        right_on="time",
        direction="backward",
        tolerance=pd.Timedelta("1H")
    )
    return enriched
