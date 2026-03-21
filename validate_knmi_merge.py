import pandas as pd

LOG_PATH = "knmi_merge_validation.txt"
lines = []

def log(msg):
    print(msg)
    lines.append(msg)

train_orig = pd.read_csv("dataset/train.csv")
train_knmi = pd.read_csv("dataset/train_with_knmi_286.csv")
test_orig = pd.read_csv("dataset/test.csv")
test_knmi = pd.read_csv("dataset/test_with_knmi_286.csv")

log("=== Row Count Check ===")
log(f"Train: orig={len(train_orig)}, merged={len(train_knmi)}, match={len(train_orig)==len(train_knmi)}")
log(f"Test:  orig={len(test_orig)},  merged={len(test_knmi)},  match={len(test_orig)==len(test_knmi)}")

log("\n=== track_id Integrity ===")
log(f"Train IDs match: {set(train_orig['track_id']) == set(train_knmi['track_id'])}")
log(f"Test IDs match:  {set(test_orig['track_id']) == set(test_knmi['track_id'])}")
log(f"Train duplicate IDs in merged: {train_knmi['track_id'].duplicated().sum()}")
log(f"Test duplicate IDs in merged:  {test_knmi['track_id'].duplicated().sum()}")

log("\n=== Time Difference Stats (minutes) ===")
gap_col = 'knmi_286_match_time_difference_minutes'
if gap_col in train_knmi.columns:
    log(f"Train gap stats:\n{train_knmi[gap_col].describe().to_string()}")
    log(f"Train NaN gap count: {train_knmi[gap_col].isna().sum()} / {len(train_knmi)}")
    log(f"Train max gap: {train_knmi[gap_col].max():.1f} min")
if gap_col in test_knmi.columns:
    log(f"\nTest gap stats:\n{test_knmi[gap_col].describe().to_string()}")
    log(f"Test NaN gap count: {test_knmi[gap_col].isna().sum()} / {len(test_knmi)}")
    log(f"Test max gap: {test_knmi[gap_col].max():.1f} min")

log("\n=== KNMI Column NaN Rates ===")
knmi_cols = [c for c in train_knmi.columns if c.startswith('knmi_286')]
for c in knmi_cols:
    train_pct = train_knmi[c].isna().mean() * 100
    test_pct = test_knmi[c].isna().mean() * 100
    log(f"  {c}: train={train_pct:.1f}% NaN, test={test_pct:.1f}% NaN")

log("\n=== Spot Check: 5 Random Rows ===")
sample = train_knmi.sample(5, random_state=42)
for _, row in sample.iterrows():
    mid = row.get('track_midpoint_utc', 'N/A')
    matched = row.get('knmi_286_hour_end_timestamp_utc', 'N/A')
    gap = row.get(gap_col, float('nan'))
    temp = row.get('knmi_286_air_temperature_c', float('nan'))
    wind = row.get('knmi_286_hourly_mean_wind_speed_mps', float('nan'))
    log(f"  track_id={row['track_id']}: midpoint={mid}, matched_hour={matched}, gap={gap:.1f}min, temp={temp}C, wind={wind}m/s")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(lines))

log(f"\nLog saved to {LOG_PATH}")
