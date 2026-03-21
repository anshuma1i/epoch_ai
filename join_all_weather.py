import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests

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

RENAME_MAP_OPENMETEO = {
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

RENAME_MAP_KNMI = {
    "knmi_timestamp_utc": "knmi_286_hour_end_timestamp_utc",
    "STN": "knmi_286_station_id",
    "DD": "knmi_286_wind_direction_degrees",
    "FH_scaled": "knmi_286_hourly_mean_wind_speed_mps",
    "FF_scaled": "knmi_286_wind_speed_at_observation_mps",
    "FX_scaled": "knmi_286_max_wind_gust_mps",
    "T_scaled": "knmi_286_air_temperature_c",
    "T10N_scaled": "knmi_286_min_air_temperature_last_6h_c",
    "TD_scaled": "knmi_286_dew_point_temperature_c",
    "SQ_scaled": "knmi_286_sunshine_duration_hours",
    "Q": "knmi_286_global_radiation_j_cm2",
    "DR_scaled": "knmi_286_precipitation_duration_hours",
    "RH_scaled": "knmi_286_precipitation_amount_mm",
    "U": "knmi_286_relative_humidity_percent",
    "IX": "knmi_286_weather_indicator_code",
}

def fetch_openmeteo_hourly(start_date: str, end_date: str, latitude: float = DEFAULT_LAT, longitude: float = DEFAULT_LON) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }
    print(f"  Requesting Open-Meteo: {start_date} to {end_date} at ({latitude}, {longitude})")
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

    print(f"  Received {len(df)} Open-Meteo hourly rows")
    return df

def parse_knmi_hourly_file(knmi_txt_path: str, station_id: int = 286) -> pd.DataFrame:
    knmi_txt_path = Path(knmi_txt_path)
    with knmi_txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    header_cols = None
    for i, line in enumerate(lines):
        clean = line.strip()
        if clean.startswith("# STN"):
            header_idx = i
            header_cols = [c.strip().lstrip("#").strip() for c in clean.split(",")]
            break

    data_lines = []
    for line in lines[header_idx + 1 :]:
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        data_lines.append(clean)

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(header_cols):
            parts = parts + [""] * (len(header_cols) - len(parts))
        elif len(parts) > len(header_cols):
            parts = parts[: len(header_cols)]
        rows.append(parts)

    weather = pd.DataFrame(rows, columns=header_cols)
    weather = weather.replace({"": pd.NA, " ": pd.NA, ";": pd.NA})

    for col in weather.columns:
        try:
            weather[col] = pd.to_numeric(weather[col])
        except (ValueError, TypeError):
            pass

    if "STN" in weather.columns:
        weather["STN"] = pd.to_numeric(weather["STN"], errors="coerce")
        weather = weather[weather["STN"] == station_id].copy()

    weather["YYYYMMDD"] = weather["YYYYMMDD"].astype(str).str.strip()
    base_date = pd.to_datetime(weather["YYYYMMDD"], format="%Y%m%d", errors="coerce")
    weather["HH"] = pd.to_numeric(weather["HH"], errors="coerce")

    hh_for_ts = weather["HH"].copy()
    day_offset = (hh_for_ts == 24).fillna(False).astype(int)
    hour_value = hh_for_ts.where(hh_for_ts != 24, 0)

    weather["knmi_timestamp_utc"] = (
        base_date + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(hour_value, unit="h")
    )
    weather["knmi_timestamp_utc"] = pd.to_datetime(weather["knmi_timestamp_utc"], utc=True)

    conversions = {
        "FH": 10.0, "FF": 10.0, "FX": 10.0, "T": 10.0, "T10N": 10.0,
        "TD": 10.0, "SQ": 10.0, "DR": 10.0, "RH": 10.0, "P": 10.0,
    }
    for col, divisor in conversions.items():
        if col in weather.columns:
            weather[f"{col}_scaled"] = pd.to_numeric(weather[col], errors="coerce") / divisor

    weather = weather.sort_values("knmi_timestamp_utc").reset_index(drop=True)
    print(f"  Received {len(weather)} KNMI hourly rows")
    return weather

def join_all_weather(tracks_df: pd.DataFrame, knmi_df: pd.DataFrame, openmeteo_df: pd.DataFrame, tol_hours: int = 1) -> pd.DataFrame:
    tracks_df = tracks_df.copy()
    tracks_df["_old_index"] = np.arange(len(tracks_df))
    
    start = pd.to_datetime(tracks_df["timestamp_start_radar_utc"], utc=True, errors="coerce")
    end = pd.to_datetime(tracks_df["timestamp_end_radar_utc"], utc=True, errors="coerce")
    tracks_df["track_midpoint_utc"] = start + (end - start) / 2
    
    tracks_sorted = tracks_df.sort_values("track_midpoint_utc")
    valid = tracks_sorted[tracks_sorted["track_midpoint_utc"].notna()].copy()
    invalid = tracks_sorted[tracks_sorted["track_midpoint_utc"].isna()].copy()

    # KNMI Join (forward)
    knmi_df = knmi_df[knmi_df["knmi_timestamp_utc"].notna()].sort_values("knmi_timestamp_utc")
    wanted_cols = ["knmi_timestamp_utc", "STN", "DD", "FH_scaled", "FF_scaled", "FX_scaled", "T_scaled", "T10N_scaled", "TD_scaled", "SQ_scaled", "Q", "DR_scaled", "RH_scaled", "U", "IX"]
    knmi_keep = [c for c in wanted_cols if c in knmi_df.columns]
    knmi_small = knmi_df[knmi_keep].copy()

    merged_valid = pd.merge_asof(
        valid, knmi_small,
        left_on="track_midpoint_utc", right_on="knmi_timestamp_utc",
        direction="forward", tolerance=pd.Timedelta(hours=tol_hours)
    )

    # OpenMeteo Join (backward)
    openmeteo_df = openmeteo_df[openmeteo_df["timestamp_utc"].notna()].sort_values("timestamp_utc")
    merged_valid = pd.merge_asof(
        merged_valid, openmeteo_df,
        left_on="track_midpoint_utc", right_on="timestamp_utc",
        direction="backward", tolerance=pd.Timedelta(hours=tol_hours)
    )

    merged = pd.concat([merged_valid, invalid], ignore_index=True, sort=False)
    # Restore exact test.csv / train.csv row order!
    merged = merged.sort_values("_old_index").drop(columns=["_old_index"]).reset_index(drop=True)

    # Knmi encoding (sin/cos)
    dd_col = "DD"
    if dd_col in merged.columns:
        dd = merged[dd_col]
        variable_wind = dd == 990
        dd_rad = np.where(variable_wind | dd.isna(), np.nan, np.deg2rad(dd))
        merged["knmi_286_wind_dir_sin"] = np.where(variable_wind, 0.0, np.sin(dd_rad))
        merged["knmi_286_wind_dir_cos"] = np.where(variable_wind, 0.0, np.cos(dd_rad))
        merged["knmi_286_wind_dir_variable"] = variable_wind.astype(int)

    # Time differences
    if "knmi_timestamp_utc" in merged.columns:
        merged["knmi_286_match_time_difference_minutes"] = (
            merged["track_midpoint_utc"] - merged["knmi_timestamp_utc"]
        ).dt.total_seconds().abs() / 60.0

    if "timestamp_utc" in merged.columns:
        merged["openmeteo_match_time_difference_minutes"] = (
            merged["track_midpoint_utc"] - merged["timestamp_utc"]
        ).dt.total_seconds().abs() / 60.0

    # Renaming
    rename_knmi_present = {k: v for k, v in RENAME_MAP_KNMI.items() if k in merged.columns}
    merged = merged.rename(columns=rename_knmi_present)
    
    rename_om_present = {k: v for k, v in RENAME_MAP_OPENMETEO.items() if k in merged.columns}
    if "timestamp_utc" in merged.columns:
        rename_om_present["timestamp_utc"] = "openmeteo_matched_timestamp_utc"
    merged = merged.rename(columns=rename_om_present)

    return merged

def main() -> None:
    parser = argparse.ArgumentParser(description="Join KNMI and OpenMeteo weather datsets into train and test.")
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--test_csv", default="dataset/test.csv")
    parser.add_argument("--knmi_txt", default="dataset/286_2021-2030.txt")
    parser.add_argument("--out_train", default="dataset/train_with_all_weather.csv")
    parser.add_argument("--out_test", default="dataset/test_with_all_weather.csv")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LAT)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LON)
    parser.add_argument("--station_id", type=int, default=286)
    parser.add_argument("--tolerance_hours", type=int, default=1)
    args = parser.parse_args()

    print("Reading train/test CSVs...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    print("Reading KNMI hourly file...")
    knmi_df = parse_knmi_hourly_file(args.knmi_txt, station_id=args.station_id)

    print("Determining Date Ranges across Train/Test for OpenMeteo...")
    start_train = pd.to_datetime(train_df["timestamp_start_radar_utc"], utc=True, errors="coerce")
    start_test = pd.to_datetime(test_df["timestamp_start_radar_utc"], utc=True, errors="coerce")
    end_train = pd.to_datetime(train_df["timestamp_end_radar_utc"], utc=True, errors="coerce")
    end_test = pd.to_datetime(test_df["timestamp_end_radar_utc"], utc=True, errors="coerce")
    
    all_starts = pd.concat([start_train, start_test])
    all_ends = pd.concat([end_train, end_test])
    
    date_min = all_starts.min().strftime("%Y-%m-%d")
    date_max = all_ends.max().strftime("%Y-%m-%d")

    print(f"Fetching Open-Meteo historical weather ({date_min} to {date_max})...")
    openmeteo_df = fetch_openmeteo_hourly(
        start_date=date_min, end_date=date_max, 
        latitude=args.latitude, longitude=args.longitude
    )

    print("\nJoining all weather to train...")
    train_joined = join_all_weather(train_df, knmi_df, openmeteo_df, tol_hours=args.tolerance_hours)

    print("\nJoining all weather to test...")
    test_joined = join_all_weather(test_df, knmi_df, openmeteo_df, tol_hours=args.tolerance_hours)

    print(f"\nSaving: {args.out_train}")
    train_joined.to_csv(args.out_train, index=False)
    print(f"Saving: {args.out_test}")
    test_joined.to_csv(args.out_test, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
