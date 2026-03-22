from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise RuntimeError(f"Could not parse input file: {path}") from last_error


def prepare_tide(tide_df: pd.DataFrame) -> pd.DataFrame:
    required = {"tide_timestamp_utc", "tide_water_level_cm_nap"}
    missing = required - set(tide_df.columns)
    if missing:
        raise KeyError(f"Missing required tide columns: {sorted(missing)}")

    tide = tide_df.copy()
    tide["tide_timestamp_utc"] = pd.to_datetime(
        tide["tide_timestamp_utc"], utc=True, errors="coerce"
    )
    tide["tide_water_level_cm_nap"] = pd.to_numeric(
        tide["tide_water_level_cm_nap"], errors="coerce"
    )

    tide = tide.dropna(subset=["tide_timestamp_utc"]).copy()
    tide = tide.sort_values("tide_timestamp_utc").drop_duplicates(
        subset=["tide_timestamp_utc"], keep="first"
    )

    # Optional compact derived features
    tide["tide_delta_10min"] = tide["tide_water_level_cm_nap"].diff()
    tide["rising_tide_flag"] = (tide["tide_delta_10min"] > 0).astype("float")

    return tide.reset_index(drop=True)


def add_track_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_start_radar_utc"] = pd.to_datetime(
        out["timestamp_start_radar_utc"], utc=True, errors="coerce"
    )
    out["timestamp_end_radar_utc"] = pd.to_datetime(
        out["timestamp_end_radar_utc"], utc=True, errors="coerce"
    )
    out["track_midpoint_utc"] = out["timestamp_start_radar_utc"] + (
        out["timestamp_end_radar_utc"] - out["timestamp_start_radar_utc"]
    ) / 2
    return out


def merge_tide_to_tracks(
    tracks_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    tolerance_minutes: int,
    label: str,
) -> pd.DataFrame:
    before_rows = len(tracks_df)
    before_dups = (
        int(tracks_df["track_id"].duplicated().sum())
        if "track_id" in tracks_df.columns
        else 0
    )

    tracks = add_track_midpoint(tracks_df)
    tracks["_row_order"] = np.arange(len(tracks))

    valid = tracks[tracks["track_midpoint_utc"].notna()].copy()
    invalid = tracks[tracks["track_midpoint_utc"].isna()].copy()

    valid = valid.sort_values("track_midpoint_utc")
    tide_sorted = tide_df.sort_values("tide_timestamp_utc")

    merged_valid = pd.merge_asof(
        valid,
        tide_sorted,
        left_on="track_midpoint_utc",
        right_on="tide_timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )

    merged = pd.concat([merged_valid, invalid], ignore_index=True, sort=False)
    merged = merged.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)

    after_rows = len(merged)
    if after_rows != before_rows:
        raise RuntimeError(
            f"{label}: row count changed after merge ({before_rows} -> {after_rows})"
        )

    after_dups = (
        int(merged["track_id"].duplicated().sum())
        if "track_id" in merged.columns
        else 0
    )
    if after_dups > before_dups:
        raise RuntimeError(
            f"{label}: duplicate track_id count increased ({before_dups} -> {after_dups})"
        )

    tide_nulls = int(merged["tide_water_level_cm_nap"].isna().sum())
    tide_null_pct = (tide_nulls / after_rows * 100.0) if after_rows else 0.0

    print(f"\n[{label}] merge diagnostics")
    print(f"  rows_before={before_rows}")
    print(f"  rows_after={after_rows}")
    print(f"  duplicate_track_id_before={before_dups}")
    print(f"  duplicate_track_id_after={after_dups}")
    print(f"  null_tide_water_level_cm_nap={tide_nulls} ({tide_null_pct:.2f}%)")

    return merged


def resolve_tide_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    fallback = Path("dataset") / path_str
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not find tide file: {path_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge cleaned tide series into Open-Meteo-enriched train/test datasets"
    )
    parser.add_argument("--train_csv", default="dataset/train_with_openmeteo.csv")
    parser.add_argument("--test_csv", default="dataset/test_with_openmeteo.csv")
    parser.add_argument("--tide_csv", default="dataset/waterinfo_tide_clean.csv")
    parser.add_argument("--out_train", default="dataset/train_with_openmeteo_tide.csv")
    parser.add_argument("--out_test", default="dataset/test_with_openmeteo_tide.csv")
    parser.add_argument(
        "--tolerance_minutes",
        type=int,
        default=15,
        help="merge_asof nearest tolerance in minutes (recommended: 10-15)",
    )
    args = parser.parse_args()

    tide_path = resolve_tide_path(args.tide_csv)

    print("Loading input datasets...")
    print(f"  train={args.train_csv}")
    print(f"  test={args.test_csv}")
    print(f"  tide={tide_path}")

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    tide_raw = read_table(tide_path)

    tide_df = prepare_tide(tide_raw)
    print(f"Prepared tide rows: {len(tide_df)}")
    print(f"Using merge tolerance: +/- {args.tolerance_minutes} minutes")

    merged_train = merge_tide_to_tracks(
        tracks_df=train_df,
        tide_df=tide_df,
        tolerance_minutes=args.tolerance_minutes,
        label="train",
    )
    merged_test = merge_tide_to_tracks(
        tracks_df=test_df,
        tide_df=tide_df,
        tolerance_minutes=args.tolerance_minutes,
        label="test",
    )

    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    merged_train.to_csv(out_train, index=False)
    merged_test.to_csv(out_test, index=False)

    print("\nSaved merged datasets:")
    print(f"  {out_train}")
    print(f"  {out_test}")


if __name__ == "__main__":
    main()
