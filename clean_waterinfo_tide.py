from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_POINT_VALUES = {"eemshaven, wadden sea", "eemshaven, waddenzee"}
TARGET_LOCATION_CODE = "eemshaven.waddenzee"
TARGET_QUANTITY_CODE = "wathte"
TARGET_REFERENCE_CODE = "nap"
TARGET_STATUS_VALUES = {"validated", "gecontroleerd"}
TARGET_METHOD_CODE = "other:f007"


def _norm(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _read_table(path: Path) -> pd.DataFrame:
    # Try automatic delimiter detection first; fall back to common delimiters.
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ";"},
        {"sep": ","},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    raise RuntimeError(f"Could not parse input file: {path}") from last_error


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    norm_to_original: dict[str, str] = {}
    for col in df.columns:
        key = _norm(col)
        if key not in norm_to_original:
            norm_to_original[key] = col

    aliases = {
        "measurement_point_identification": [
            "measurement_point_identification",
            "meetpunt_identificatie",
        ],
        "location_code": ["location_code", "locatie_code"],
        "quantity_code": ["quantity_code", "grootheid_code"],
        "reference_code": ["reference_code", "hoedanigheid_code"],
        "status_value": ["status_value", "statuswaarde"],
        "value_determination_method_code": [
            "value_determination_method_code",
            "waardebepalingsmethode_code",
        ],
        "observation_date": ["observation_date", "waarnemingdatum"],
        "observation_time": ["observation_time", "waarnemingtijd"],
        "numeric_value": ["numeric_value", "numeriekewaarde"],
    }

    resolved: dict[str, str] = {}
    missing: list[str] = []

    for canonical, options in aliases.items():
        found = None
        for option in options:
            lookup = _norm(option)
            if lookup in norm_to_original:
                found = norm_to_original[lookup]
                break
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found

    if missing:
        raise KeyError(
            "Missing required columns: " + ", ".join(missing)
        )

    return resolved


def _approx_step(ts: pd.Series) -> pd.Timedelta | None:
    ts_unique = ts.dropna().drop_duplicates().sort_values()
    if len(ts_unique) < 2:
        return None

    diffs = ts_unique.diff().dropna()
    mode = diffs.mode()
    if not mode.empty:
        return mode.iloc[0]
    return diffs.median()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal cleaner for Rijkswaterstaat Waterinfo tide export"
    )
    parser.add_argument(
        "--input_csv",
        default="dataset/waterinfo_data.csv",
        help="Path to raw Waterinfo export (CSV or semicolon-separated CSV)",
    )
    parser.add_argument(
        "--output_csv",
        default="dataset/waterinfo_tide_clean.csv",
        help="Path to cleaned output CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    df = _read_table(input_path)
    col = _resolve_columns(df)

    work = df[[
        col["measurement_point_identification"],
        col["location_code"],
        col["quantity_code"],
        col["reference_code"],
        col["status_value"],
        col["value_determination_method_code"],
        col["observation_date"],
        col["observation_time"],
        col["numeric_value"],
    ]].copy()

    work.columns = [
        "measurement_point_identification",
        "location_code",
        "quantity_code",
        "reference_code",
        "status_value",
        "value_determination_method_code",
        "observation_date",
        "observation_time",
        "numeric_value",
    ]

    for c in work.columns:
        work[c] = work[c].astype(str).str.strip()

    mask = (
        work["measurement_point_identification"].str.casefold().isin(TARGET_POINT_VALUES)
        & work["location_code"].str.casefold().eq(TARGET_LOCATION_CODE)
        & work["quantity_code"].str.casefold().eq(TARGET_QUANTITY_CODE)
        & work["reference_code"].str.casefold().eq(TARGET_REFERENCE_CODE)
        & work["status_value"].str.casefold().isin(TARGET_STATUS_VALUES)
        & work["value_determination_method_code"].str.casefold().eq(TARGET_METHOD_CODE)
    )

    filtered = work.loc[mask].copy()

    tide_ts = pd.to_datetime(
        filtered["observation_date"] + " " + filtered["observation_time"],
        dayfirst=True,
        utc=True,
        errors="coerce",
    )
    tide_level = pd.to_numeric(
        filtered["numeric_value"].str.replace(",", ".", regex=False),
        errors="coerce",
    )

    out = pd.DataFrame(
        {
            "tide_timestamp_utc": tide_ts,
            "tide_water_level_cm_nap": tide_level,
        }
    ).sort_values("tide_timestamp_utc", kind="mergesort")

    null_timestamp_count = int(out["tide_timestamp_utc"].isna().sum())
    null_level_count = int(out["tide_water_level_cm_nap"].isna().sum())
    duplicate_timestamp_count = int(
        out["tide_timestamp_utc"].dropna().duplicated().sum()
    )

    valid_ts = out["tide_timestamp_utc"].dropna()
    min_ts = valid_ts.min() if not valid_ts.empty else pd.NaT
    max_ts = valid_ts.max() if not valid_ts.empty else pd.NaT
    step = _approx_step(out["tide_timestamp_utc"])

    output = out.copy()
    output["tide_timestamp_utc"] = output["tide_timestamp_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output[["tide_timestamp_utc", "tide_water_level_cm_nap"]].to_csv(
        output_path, index=False
    )

    print(f"rows_in={len(df)}")
    print(f"rows_after_filter={len(filtered)}")
    print(f"rows_out={len(output)}")
    print(f"null_tide_timestamp_utc={null_timestamp_count}")
    print(f"null_tide_water_level_cm_nap={null_level_count}")
    print(f"duplicate_tide_timestamp_utc={duplicate_timestamp_count}")
    print(
        "min_tide_timestamp_utc="
        + (
            min_ts.isoformat().replace("+00:00", "Z")
            if pd.notna(min_ts)
            else "NA"
        )
    )
    print(
        "max_tide_timestamp_utc="
        + (
            max_ts.isoformat().replace("+00:00", "Z")
            if pd.notna(max_ts)
            else "NA"
        )
    )

    if step is None:
        print("approx_timestep=NA")
    else:
        print(
            f"approx_timestep={step} (~{step.total_seconds() / 60:.2f} minutes)"
        )

    print(f"saved_to={output_path}")


if __name__ == "__main__":
    main()
