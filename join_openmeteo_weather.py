# join_openmeteo_weather.py

"""
Join hourly Open-Meteo historical weather to bird radar tracks.

The radar is located at Windpark Eemshaven, Groningen, Netherlands.
No per-track lat/lon columns exist, so fixed coordinates are used:
  latitude  = 53.44
  longitude =  6.83

Open-Meteo hourly timestamps are treated as hour-start timestamps.
The join uses direction="backward" so each track midpoint maps to
the hour bucket containing it (e.g. midpoint 09:40 → 09:00 row).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Eemshaven radar location ──
DEFAULT_LAT = 53.44
DEFAULT_LON = 6.83

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "pressure_msl",
    "surface_pressure",
    "weather_code",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "sunshine_duration",
    "vapour_pressure_deficit",
    "is_day",
]

RENAME_MAP = {
    "temperature_2m": "openmeteo_air_temperature_2m_c",
    "relative_humidity_2m": "openmeteo_relative_humidity_2m_percent",
    "dew_point_2m": "openmeteo_dew_point_2m_c",
    "precipitation": "openmeteo_precipitation_mm",
    "cloud_cover": "openmeteo_cloud_cover_percent",
    "cloud_cover_low": "openmeteo_cloud_cover_low_percent",
    "cloud_cover_mid": "openmeteo_cloud_cover_mid_percent",
    "cloud_cover_high": "openmeteo_cloud_cover_high_percent",
    "pressure_msl": "openmeteo_pressure_msl_hpa",
    "surface_pressure": "openmeteo_surface_pressure_hpa",
    "weather_code": "openmeteo_weather_code",
    "wind_speed_10m": "openmeteo_wind_speed_10m_kmh",
    "wind_direction_10m": "openmeteo_wind_direction_10m_degrees",
    "wind_gusts_10m": "openmeteo_wind_gusts_10m_kmh",
    "shortwave_radiation": "openmeteo_shortwave_radiation_w_m2",
    "direct_radiation": "openmeteo_direct_radiation_w_m2",
    "diffuse_radiation": "openmeteo_diffuse_radiation_w_m2",
    "sunshine_duration": "openmeteo_sunshine_duration_s",
    "vapour_pressure_deficit": "openmeteo_vapour_pressure_deficit_kpa",
    "is_day": "openmeteo_is_day",
}


def fetch_openmeteo_hourly(
    start_date: str,
    end_date: str,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from the Open-Meteo archive API.

    Returns a DataFrame with a UTC-aware 'timestamp_utc' column and one
    column per requested variable.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required: pip install requests")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    print(f"  Requesting Open-Meteo: {start_date} to {end_date} "
          f"at ({latitude}, {longitude})")

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected API response: {list(data.keys())}")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    print(f"  Received {len(df)} hourly rows")
    return df


def build_track_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the temporal midpoint for each radar track."""
    df = df.copy()
    df["timestamp_start_radar_utc"] = pd.to_datetime(
        df["timestamp_start_radar_utc"], utc=True, errors="coerce"
    )
    df["timestamp_end_radar_utc"] = pd.to_datetime(
        df["timestamp_end_radar_utc"], utc=True, errors="coerce"
    )
    df["track_midpoint_utc"] = df["timestamp_start_radar_utc"] + (
        df["timestamp_end_radar_utc"] - df["timestamp_start_radar_utc"]
    ) / 2
    return df


def join_weather_to_tracks(
    tracks_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    tolerance_hours: int = 1,
) -> pd.DataFrame:
    """
    Join Open-Meteo hourly weather to radar tracks using merge_asof
    with direction='backward' (hour-start bucket semantics) on each
    track's temporal midpoint.
    """
    tracks_df = build_track_midpoint(tracks_df)
    weather_df = weather_df.copy()

    # Separate valid/invalid midpoints (merge_asof requires non-NaT keys)
    tracks_sorted = tracks_df.sort_values("track_midpoint_utc").copy()
    valid = tracks_sorted[tracks_sorted["track_midpoint_utc"].notna()].copy()
    invalid = tracks_sorted[tracks_sorted["track_midpoint_utc"].isna()].copy()

    weather_df = weather_df[weather_df["timestamp_utc"].notna()].copy()
    weather_df = weather_df.sort_values("timestamp_utc")

    merged = pd.merge_asof(
        valid,
        weather_df,
        left_on="track_midpoint_utc",
        right_on="timestamp_utc",
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )

    merged = pd.concat([merged, invalid], ignore_index=True, sort=False)

    # Rename weather columns
    present = {k: v for k, v in RENAME_MAP.items() if k in merged.columns}
    merged = merged.rename(columns=present)
    if "timestamp_utc" in merged.columns:
        merged = merged.rename(columns={"timestamp_utc": "openmeteo_matched_timestamp_utc"})

    # Match time difference
    if "openmeteo_matched_timestamp_utc" in merged.columns:
        merged["openmeteo_match_time_difference_minutes"] = (
            merged["track_midpoint_utc"] - merged["openmeteo_matched_timestamp_utc"]
        ).dt.total_seconds().abs() / 60.0

    # Sort by track_id if available
    if "track_id" in merged.columns:
        merged = merged.sort_values("track_id").reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    return merged


def print_diagnostics(df: pd.DataFrame, label: str) -> None:
    """Print null rates and basic coverage checks."""
    om_cols = [c for c in df.columns if c.startswith("openmeteo_")]
    print(f"\n  {label}: {len(df)} rows, {len(df.columns)} columns "
          f"({len(om_cols)} Open-Meteo columns)")

    matched = df["openmeteo_matched_timestamp_utc"].notna().sum() if \
        "openmeteo_matched_timestamp_utc" in df.columns else 0
    print(f"  Matched rows: {matched}/{len(df)} "
          f"({matched/len(df)*100:.1f}%)")

    print("  Null rates:")
    for c in sorted(om_cols):
        n = df[c].isna().sum()
        pct = n / len(df) * 100
        print(f"    {c}: {n}/{len(df)} ({pct:.1f}%)")

    if "openmeteo_match_time_difference_minutes" in df.columns:
        mtd = df["openmeteo_match_time_difference_minutes"].dropna()
        if len(mtd) > 0:
            print(f"  Match gap (min): min={mtd.min():.1f}, "
                  f"median={mtd.median():.1f}, max={mtd.max():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join Open-Meteo hourly weather to bird radar tracks."
    )
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--test_csv", default="dataset/test.csv")
    parser.add_argument("--out_train", default="dataset/train_with_openmeteo.csv")
    parser.add_argument("--out_test", default="dataset/test_with_openmeteo.csv")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LAT)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LON)
    parser.add_argument("--tolerance_hours", type=int, default=1)
    args = parser.parse_args()

    print("Reading train/test CSVs...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # Determine the full date range across both sets
    all_starts = pd.to_datetime(
        pd.concat([train_df["timestamp_start_radar_utc"],
                    test_df["timestamp_start_radar_utc"]]),
        utc=True,
    )
    all_ends = pd.to_datetime(
        pd.concat([train_df["timestamp_end_radar_utc"],
                    test_df["timestamp_end_radar_utc"]]),
        utc=True,
    )
    date_min = all_starts.min().strftime("%Y-%m-%d")
    date_max = all_ends.max().strftime("%Y-%m-%d")

    print("Fetching Open-Meteo historical weather...")
    weather = fetch_openmeteo_hourly(
        start_date=date_min,
        end_date=date_max,
        latitude=args.latitude,
        longitude=args.longitude,
    )

    print("Joining weather to train...")
    train_joined = join_weather_to_tracks(
        train_df, weather, tolerance_hours=args.tolerance_hours
    )
    print_diagnostics(train_joined, "Train")

    print("\nJoining weather to test...")
    test_joined = join_weather_to_tracks(
        test_df, weather, tolerance_hours=args.tolerance_hours
    )
    print_diagnostics(test_joined, "Test")

    print(f"\nSaving: {args.out_train}")
    train_joined.to_csv(args.out_train, index=False)
    print(f"Saving: {args.out_test}")
    test_joined.to_csv(args.out_test, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
