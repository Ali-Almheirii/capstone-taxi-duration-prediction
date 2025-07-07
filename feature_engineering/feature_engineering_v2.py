import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def engineer_features(df):
    latest_len = len(df)
    original_len = len(df)
    logger.info(f"Original dataset length: {latest_len} rows")

    df = df[df["FLIGHT_DIRECTION"] == "D"]
    logger.info(f"Dropped {latest_len - len(df)} rows filtering for departures only.)")
    latest_len = len(df)

    df.dropna(subset=["ACTUAL_TAKE_OFF_TIME_ATOT_UTC", "ACTUAL_OFFBLOCK_TIME_AOBT_UTC"], inplace=True)
    logger.info(f"Dropped {latest_len - len(df)} rows due to missing timestamps. (ACTUAL_TAKE_OFF_TIME_ATOT_UTC AND ACTUAL_OFFBLOCK_TIME_AOBT_UTC)")
    latest_len = len(df)

    df["ACTUAL_OFFBLOCK_TIME_AOBT_UTC"] = pd.to_datetime(df["ACTUAL_OFFBLOCK_TIME_AOBT_UTC"])
    df["ACTUAL_TAKE_OFF_TIME_ATOT_UTC"] = pd.to_datetime(df["ACTUAL_TAKE_OFF_TIME_ATOT_UTC"])

    df["TAXI_OUT_DURATION"] = (df["ACTUAL_TAKE_OFF_TIME_ATOT_UTC"] -
                               df["ACTUAL_OFFBLOCK_TIME_AOBT_UTC"]).dt.total_seconds() / 60

    df = df[(df["TAXI_OUT_DURATION"] >= 10) & (df["TAXI_OUT_DURATION"] <= 25)]
    logger.info(f"Dropped {latest_len - len(df)} rows outside 10–25 min taxi duration range.")
    latest_len = len(df)

    df = compute_vectorized_traffic(df, "ACTUAL_OFFBLOCK_TIME_AOBT_UTC", 10)
    df = compute_vectorized_traffic(df, "ACTUAL_OFFBLOCK_TIME_AOBT_UTC", 30)
    df = compute_vectorized_traffic(df, "ACTUAL_OFFBLOCK_TIME_AOBT_UTC", 60)

    df = compute_grouped_traffic(df,
                                 time_col="ACTUAL_OFFBLOCK_TIME_AOBT_UTC",
                                 group_col="TERMINAL",
                                 window_minutes=30,
                                 output_col="TRAFFIC_IN_TERMINAL_LAST_30_MIN")

    df = compute_grouped_traffic(df,
                                 time_col="ACTUAL_OFFBLOCK_TIME_AOBT_UTC",
                                 group_col="RUNWAY",
                                 window_minutes=30,
                                 output_col="TRAFFIC_TO_SAME_RUNWAY_LAST_30_MIN")

    df["HOUR"] = df["ACTUAL_OFFBLOCK_TIME_AOBT_UTC"].dt.hour
    df["DAY_OF_WEEK"] = df["ACTUAL_OFFBLOCK_TIME_AOBT_UTC"].dt.dayofweek
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)
    df["IS_PEAK_HOUR"] = df["HOUR"].apply(lambda h: 1 if (6 <= h <= 9) or (16 <= h <= 20) else 0)

    # Apply mapping
    #df['AIRCRAFT_CATEGORY'] = df['AIRCRAFT_TYPE_ICAO'].map(aircraft_category_map).fillna('M')
    #df['AIRCRAFT_CATEGORY'] = df['AIRCRAFT_CATEGORY'].astype('category')

    logger.info(f"Total dropped rows: {original_len - latest_len} rows.")
    logger.info(f"Final dataset has {latest_len} rows.")

    return df



def compute_vectorized_traffic(df, time_col, window_minutes):
    df = df.sort_values(time_col).reset_index(drop=True).copy()

    # Convert time to integer timestamps (seconds since epoch)
    timestamps = df[time_col].astype('int64') // 1_000_000_000  # convert ns to seconds
    window_size_sec = window_minutes * 60

    # Use searchsorted to find how many earlier events are within the window
    start_times = timestamps - window_size_sec
    traffic_counts = np.searchsorted(timestamps.values, timestamps.values, side='left') - \
                     np.searchsorted(timestamps.values, start_times.values, side='left')

    df[f"TRAFFIC_LAST_{window_minutes}_MIN"] = traffic_counts
    return df


def compute_grouped_traffic(df, time_col, group_col, window_minutes, output_col):
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df[output_col] = 0  # Initialize column

    timestamps = df[time_col].astype("int64") // 1_000_000_000  # seconds
    window_size = window_minutes * 60

    group_values = df[group_col].astype(str).values
    traffic_counts = np.zeros(len(df), dtype=int)

    group_to_indices = {}
    for idx, (gval, t) in enumerate(zip(group_values, timestamps)):
        group_to_indices.setdefault(gval, []).append((idx, t))

    for gval, idx_time_pairs in group_to_indices.items():
        indices, times = zip(*idx_time_pairs)
        times = np.array(times)
        indices = np.array(indices)

        sorted_order = np.argsort(times)
        times = times[sorted_order]
        indices = indices[sorted_order]

        for i in range(len(times)):
            t_now = times[i]
            t_start = t_now - window_size
            start_idx = np.searchsorted(times, t_start, side="left")
            traffic_counts[indices[i]] = i - start_idx

    df[output_col] = traffic_counts
    return df