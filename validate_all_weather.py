import pandas as pd
import numpy as np

LOG_PATH = "all_weather_merge_validation.txt"
lines = []

def log(msg):
    print(msg)
    lines.append(msg)

train_orig = pd.read_csv("dataset/train.csv")
train_weather = pd.read_csv("dataset/train_with_all_weather.csv")
test_orig = pd.read_csv("dataset/test.csv")
test_weather = pd.read_csv("dataset/test_with_all_weather.csv")

log("=== Row Count Check ===")
log(f"Train: orig={len(train_orig)}, merged={len(train_weather)}, match={len(train_orig)==len(train_weather)}")
log(f"Test:  orig={len(test_orig)},  merged={len(test_weather)},  match={len(test_orig)==len(test_weather)}")

log("\n=== track_id Integrity ===")
log(f"Train IDs match: {set(train_orig['track_id']) == set(train_weather['track_id'])}")
log(f"Test IDs match:  {set(test_orig['track_id']) == set(test_weather['track_id'])}")
log(f"Train duplicate IDs in merged: {train_weather['track_id'].duplicated().sum()}")
log(f"Test duplicate IDs in merged:  {test_weather['track_id'].duplicated().sum()}")

# To check if row order was actually preserved correctly!
log("\n=== Row Order Integrity ===")
log(f"Train row order matches exactly: {train_orig['track_id'].equals(train_weather['track_id'])}")
log(f"Test row order matches exactly: {test_orig['track_id'].equals(test_weather['track_id'])}")

log("\n=== Time Difference Stats (minutes) ===")
for gap_col, name in [('knmi_286_match_time_difference_minutes', 'KNMI'), ('openmeteo_match_time_difference_minutes', 'OpenMeteo')]:
    log(f"--- {name} ---")
    if gap_col in train_weather.columns:
        log(f"Train gap stats:\n{train_weather[gap_col].describe().to_string()}")
        log(f"Train NaN gap count: {train_weather[gap_col].isna().sum()} / {len(train_weather)}")
        max_val = train_weather[gap_col].max()
        log(f"Train max gap: {max_val:.1f} min" if pd.notna(max_val) else "Train max gap: NaT")
    if gap_col in test_weather.columns:
        log(f"\nTest gap stats:\n{test_weather[gap_col].describe().to_string()}")
        log(f"Test NaN gap count: {test_weather[gap_col].isna().sum()} / {len(test_weather)}")
        max_val = test_weather[gap_col].max()
        log(f"Test max gap: {max_val:.1f} min" if pd.notna(max_val) else "Test max gap: NaT")

log("\n=== Column NaN Rates ===")
weather_cols = [c for c in train_weather.columns if c.startswith('knmi_286') or c.startswith('openmeteo_')]
for c in weather_cols:
    train_pct = train_weather[c].isna().mean() * 100
    test_pct = test_weather[c].isna().mean() * 100
    log(f"  {c}: train={train_pct:.1f}% NaN, test={test_pct:.1f}% NaN")

log("\n=== Spot Check: 5 Random Rows ===")
sample = train_weather.sample(5, random_state=42)
for _, row in sample.iterrows():
    mid = row.get('track_midpoint_utc', 'N/A')
    matched_knmi = row.get('knmi_286_hour_end_timestamp_utc', 'N/A')
    matched_om = row.get('openmeteo_matched_timestamp_utc', 'N/A')
    gap_knmi = row.get('knmi_286_match_time_difference_minutes', float('nan'))
    gap_om = row.get('openmeteo_match_time_difference_minutes', float('nan'))
    
    log(f"  track_id={row['track_id']}: midpoint={mid}, knmi={matched_knmi} ({gap_knmi:.1f}min), om={matched_om} ({gap_om:.1f}min)")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(lines))

log(f"\nLog saved to {LOG_PATH}")
