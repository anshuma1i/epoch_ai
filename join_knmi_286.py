# join_knmi_286.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_knmi_hourly_file(knmi_txt_path: str, station_id: int = 286) -> pd.DataFrame:
    """
    Parse a KNMI hourly weather ASCII/text file.

    Expected format:
    - metadata/comment lines at the top
    - one header line starting with '# STN,YYYYMMDD,HH,...'
    - data rows after that

    Returns a dataframe with:
    - raw KNMI columns
    - a parsed UTC timestamp column: knmi_timestamp_utc
    - some unit-converted weather columns
    """
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

    if header_idx is None or header_cols is None:
        raise ValueError("Could not find KNMI header line starting with '# STN'.")

    data_lines = []
    for line in lines[header_idx + 1 :]:
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("#"):
            continue
        data_lines.append(clean)

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]

        # Pad/truncate to header length if needed
        if len(parts) < len(header_cols):
            parts = parts + [""] * (len(header_cols) - len(parts))
        elif len(parts) > len(header_cols):
            parts = parts[: len(header_cols)]

        rows.append(parts)

    weather = pd.DataFrame(rows, columns=header_cols)

    # Clean missing values
    weather = weather.replace({"": pd.NA, " ": pd.NA, ";": pd.NA})

    # Convert everything that looks numeric
    for col in weather.columns:
        try:
            weather[col] = pd.to_numeric(weather[col])
        except (ValueError, TypeError):
            pass

    # Keep only requested station if mixed content ever appears
    if "STN" in weather.columns:
        weather["STN"] = pd.to_numeric(weather["STN"], errors="coerce")
        weather = weather[weather["STN"] == station_id].copy()

    # Parse date
    weather["YYYYMMDD"] = weather["YYYYMMDD"].astype(str).str.strip()
    base_date = pd.to_datetime(weather["YYYYMMDD"], format="%Y%m%d", errors="coerce")

    # KNMI HH is the end of the hourly division.
    # Example: HH=5 means the interval 04:00-05:00 UTC.
    # HH=24 should become next day 00:00 UTC.
    weather["HH"] = pd.to_numeric(weather["HH"], errors="coerce")

    hh_for_ts = weather["HH"].copy()
    day_offset = (hh_for_ts == 24).fillna(False).astype(int)
    hour_value = hh_for_ts.where(hh_for_ts != 24, 0)

    weather["knmi_timestamp_utc"] = (
        base_date
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour_value, unit="h")
    )

    # Optional: convert the main KNMI units into human-friendly units
    conversions = {
        "FH": 10.0,    # hourly mean wind speed -> m/s
        "FF": 10.0,    # wind speed at observation time -> m/s
        "FX": 10.0,    # max gust -> m/s
        "T": 10.0,     # temperature -> °C
        "T10N": 10.0,  # min temp last 6h -> °C
        "TD": 10.0,    # dew point -> °C
        "SQ": 10.0,    # sunshine duration -> hours
        "DR": 10.0,    # precipitation duration -> hours
        "RH": 10.0,    # precipitation amount -> mm
        "P": 10.0,     # pressure -> hPa
    }

    for col, divisor in conversions.items():
        if col in weather.columns:
            weather[f"{col}_scaled"] = pd.to_numeric(weather[col], errors="coerce") / divisor

    # Sort for merge_asof
    weather = weather.sort_values("knmi_timestamp_utc").reset_index(drop=True)

    return weather


def build_track_join_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create radar timestamps and midpoint timestamp for each track.
    """
    df = df.copy()

    df["timestamp_start_radar_utc"] = pd.to_datetime(
        df["timestamp_start_radar_utc"], utc=True, errors="coerce"
    )
    df["timestamp_end_radar_utc"] = pd.to_datetime(
        df["timestamp_end_radar_utc"], utc=True, errors="coerce"
    )

    df["track_duration_s"] = (
        df["timestamp_end_radar_utc"] - df["timestamp_start_radar_utc"]
    ).dt.total_seconds()

    df["track_midpoint_utc"] = df["timestamp_start_radar_utc"] + (
        df["timestamp_end_radar_utc"] - df["timestamp_start_radar_utc"]
    ) / 2

    return df


def join_weather_to_tracks(
    tracks_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    station_id: int = 286,
    tolerance_hours: int = 2,
) -> pd.DataFrame:
    """
    Join KNMI hourly weather row using hour-end bucket semantics based on track midpoint.
    """
    tracks_df = build_track_join_timestamp(tracks_df)

    weather_df = weather_df.copy()

    # Make weather timestamp explicitly UTC-aware
    weather_df["knmi_timestamp_utc"] = pd.to_datetime(
        weather_df["knmi_timestamp_utc"], utc=True, errors="coerce"
    )

    # Keep a practical subset of weather columns
    wanted_cols = [
        "knmi_timestamp_utc",
        "STN",
        "DD",
        "FH_scaled",
        "FF_scaled",
        "FX_scaled",
        "T_scaled",
        "T10N_scaled",
        "TD_scaled",
        "SQ_scaled",
        "Q",
        "DR_scaled",
        "RH_scaled",
        "P_scaled",
        "VV",
        "N",
        "U",
        "WW",
        "IX",
        "M",
        "R",
        "S",
        "O",
        "Y",
    ]
    weather_keep = [c for c in wanted_cols if c in weather_df.columns]
    weather_small = weather_df[weather_keep].copy()

    # Drop rows with missing keys to avoid merge_asof runtime errors
    tracks_sorted = tracks_df.sort_values("track_midpoint_utc").copy()
    tracks_valid = tracks_sorted[tracks_sorted["track_midpoint_utc"].notna()].copy()
    tracks_invalid = tracks_sorted[tracks_sorted["track_midpoint_utc"].isna()].copy()

    weather_small = weather_small[weather_small["knmi_timestamp_utc"].notna()].copy()
    weather_small = weather_small.sort_values("knmi_timestamp_utc")

    # KNMI HH is the hour-end timestamp (HH=15 covers 14:00–15:00).
    # Use direction="forward" so each midpoint joins to the KNMI hour
    # whose interval contains it (i.e. the next hour-end on or after).
    merged_valid = pd.merge_asof(
        tracks_valid,
        weather_small,
        left_on="track_midpoint_utc",
        right_on="knmi_timestamp_utc",
        direction="forward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )

    merged = pd.concat([merged_valid, tracks_invalid], ignore_index=True, sort=False)

    # Rename joined weather columns clearly
    rename_map = {}
    for col in weather_small.columns:
        if col == "knmi_timestamp_utc":
            rename_map[col] = f"knmi_{station_id}_timestamp_utc"
        elif col != "STN":
            rename_map[col] = f"knmi_{station_id}_{col}"
        else:
            rename_map[col] = f"knmi_{station_id}_station_id"

    merged = merged.rename(columns=rename_map)

    # Add time gap between track midpoint and matched KNMI hour
    ts_col = f"knmi_{station_id}_timestamp_utc"
    if ts_col in merged.columns:
        merged[f"knmi_{station_id}_time_diff_min"] = (
            merged["track_midpoint_utc"] - merged[ts_col]
        ).dt.total_seconds().abs() / 60.0

    if "track_id" in merged.columns:
        merged = merged.sort_values("track_id").reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join KNMI hourly weather from station 286 to bird radar tracks."
    )
    parser.add_argument("--train_csv", type=str, default="dataset/train.csv", help="Path to train.csv")
    parser.add_argument("--test_csv", type=str, default="dataset/test.csv", help="Path to test.csv")
    parser.add_argument("--knmi_txt", type=str, default="dataset/286_2021-2030.txt", help="Path to KNMI hourly .txt file")
    parser.add_argument("--out_train", type=str, default="dataset/train_with_knmi_286.csv")
    parser.add_argument("--out_test", type=str, default="dataset/test_with_knmi_286.csv")
    parser.add_argument("--station_id", type=int, default=286)
    parser.add_argument("--tolerance_hours", type=int, default=1)

    args = parser.parse_args()

    print("Reading KNMI hourly file...")
    weather = parse_knmi_hourly_file(args.knmi_txt, station_id=args.station_id)
    print(f"KNMI rows loaded: {len(weather)}")

    print("Reading train/test CSVs...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    print("Joining weather to train...")
    train_joined = join_weather_to_tracks(
        train_df, weather, station_id=args.station_id, tolerance_hours=args.tolerance_hours
    )

    print("Joining weather to test...")
    test_joined = join_weather_to_tracks(
        test_df, weather, station_id=args.station_id, tolerance_hours=args.tolerance_hours
    )

    print(f"Saving: {args.out_train}")
    train_joined.to_csv(args.out_train, index=False)

    print(f"Saving: {args.out_test}")
    test_joined.to_csv(args.out_test, index=False)

    print("Done.")


if __name__ == "__main__":
    main()