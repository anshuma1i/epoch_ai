"""
Diagnostic script for weak classes: Cormorants, Waders, Geese.
Analyzes: class counts, confusion patterns, feature distributions,
and creates a debug submission CSV for the bird-radar-visualisation tool.
"""

import numpy as np
import pandas as pd
from shapely import wkb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek
import sys
sys.path.insert(0, '.')  # so we can import from solution if needed

LOG_PATH = "weak_class_diagnostic.txt"
lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

# ── Load same data as solution.py ──
train_df = pd.read_csv("dataset/train_with_knmi_286.csv").set_index("track_id")
test_df = pd.read_csv("dataset/test_with_knmi_286.csv").set_index("track_id")

y = train_df['bird_group']

# ═══════════════════════════════════════
# 1. CLASS DISTRIBUTION
# ═══════════════════════════════════════
log("=" * 60)
log("1. CLASS DISTRIBUTION")
log("=" * 60)
counts = y.value_counts().sort_index()
for cls, cnt in counts.items():
    pct = cnt / len(y) * 100
    bar = "█" * int(pct)
    log(f"  {cls:20s}: {cnt:5d} ({pct:5.1f}%) {bar}")
log(f"\n  Total samples: {len(y)}")
log(f"  Smallest class: {counts.idxmin()} ({counts.min()} samples)")
log(f"  Largest class:  {counts.idxmax()} ({counts.max()} samples)")
log(f"  Imbalance ratio: {counts.max() / counts.min():.1f}x")

# ═══════════════════════════════════════
# 2. SPECIES BREAKDOWN FOR WEAK CLASSES
# ═══════════════════════════════════════
log()
log("=" * 60)
log("2. SPECIES BREAKDOWN (bird_species within bird_group)")
log("=" * 60)
weak_classes = ['Cormorants', 'Waders', 'Geese']
if 'bird_species' in train_df.columns:
    for cls in weak_classes:
        subset = train_df[train_df['bird_group'] == cls]
        log(f"\n  {cls} ({len(subset)} tracks):")
        species_counts = subset['bird_species'].value_counts()
        for sp, cnt in species_counts.items():
            log(f"    {sp}: {cnt}")
else:
    log("  bird_species column not found")

# ═══════════════════════════════════════
# 3. RADAR BIRD SIZE DISTRIBUTION PER CLASS
# ═══════════════════════════════════════
log()
log("=" * 60)
log("3. RADAR BIRD SIZE vs BIRD GROUP (confusion source?)")
log("=" * 60)
crosstab = pd.crosstab(train_df['bird_group'], train_df['radar_bird_size'], normalize='index')
crosstab_pct = (crosstab * 100).round(1)
log(f"\n{crosstab_pct.to_string()}")

# ═══════════════════════════════════════
# 4. KEY FEATURE STATISTICS FOR WEAK CLASSES
# ═══════════════════════════════════════
log()
log("=" * 60)
log("4. FEATURE STATISTICS FOR WEAK vs STRONG CLASSES")
log("=" * 60)
key_features = ['airspeed', 'min_z', 'max_z', 'duration_s']
# Add duration_s if not yet computed
if 'duration_s' not in train_df.columns:
    train_df['ts_start'] = pd.to_datetime(train_df['timestamp_start_radar_utc'], utc=True)
    train_df['ts_end'] = pd.to_datetime(train_df['timestamp_end_radar_utc'], utc=True)
    train_df['duration_s'] = (train_df['ts_end'] - train_df['ts_start']).dt.total_seconds()

for feat in key_features:
    if feat not in train_df.columns:
        continue
    log(f"\n  {feat}:")
    stats = train_df.groupby('bird_group')[feat].agg(['mean', 'std', 'median', 'min', 'max'])
    for cls in stats.index:
        s = stats.loc[cls]
        marker = " ◄ WEAK" if cls in weak_classes else ""
        log(f"    {cls:20s}: mean={s['mean']:8.1f}  std={s['std']:8.1f}  median={s['median']:8.1f}{marker}")

# ═══════════════════════════════════════
# 5. CONFUSION MATRIX (from OOF predictions)
# ═══════════════════════════════════════
log()
log("=" * 60)
log("5. RUNNING CV TO GET CONFUSION PATTERNS")
log("=" * 60)

# Re-run feature engineering (minimal — reuse solution.py logic)
for df in [train_df, test_df]:
    df['ts_start'] = pd.to_datetime(df['timestamp_start_radar_utc'], utc=True)
    df['ts_end'] = pd.to_datetime(df['timestamp_end_radar_utc'], utc=True)
    df['duration_s'] = (df['ts_end'] - df['ts_start']).dt.total_seconds()
    df['hour'] = df['ts_start'].dt.hour
    df['month'] = df['ts_start'].dt.month
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['alt_range'] = df['max_z'] - df['min_z']
    df['airspeed_per_m'] = df['airspeed'] / (df['max_z'] + 1)

# Use same features as solution.py but skip trajectory features for speed
# (the confusion patterns will be similar)
features = [
    'airspeed', 'min_z', 'max_z', 'duration_s', 'radar_bird_size',
    'hour', 'month', 'is_daytime',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'alt_range', 'airspeed_per_m',
    'knmi_286_wind_direction_degrees',
    'knmi_286_hourly_mean_wind_speed_mps',
    'knmi_286_air_temperature_c',
    'knmi_286_dew_point_temperature_c',
    'knmi_286_relative_humidity_percent',
    'knmi_286_wind_dir_sin', 'knmi_286_wind_dir_cos',
]

numeric_features = [f for f in features if f != 'radar_bird_size']
categorical_features = ['radar_bird_size']
cat_indices = [len(numeric_features) + i for i in range(len(categorical_features))]

X = train_df[features]
groups = train_df['primary_observation_id']

imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='median'), numeric_features),
    ('cat_imputer', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), categorical_features)
])

lgb_model = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42, n_jobs=-1, device='gpu', verbose=-1
)

pipeline = ImbPipeline([
    ('imputer', imputer),
    ('oversampler', SMOTETomek(
        smote=SMOTENC(categorical_features=cat_indices, random_state=42),
        random_state=42
    )),
    ('model', lgb_model)
])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
split = list(cv.split(X, y, groups))
classes = np.sort(y.unique())

oof_preds = pd.DataFrame(0.0, index=X.index, columns=classes)
oof_labels = pd.Series('', index=X.index)

log("  Training 5-fold quick model (no trajectory features for speed)...")
for i, (train_idx, val_idx) in enumerate(split):
    p = clone(pipeline)
    p.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_preds.iloc[val_idx] = p.predict_proba(X.iloc[val_idx])
    oof_labels.iloc[val_idx] = classes[np.argmax(p.predict_proba(X.iloc[val_idx]), axis=1)]
    log(f"  Fold {i+1}/5 done")

# Confusion matrix
log("\n  CONFUSION MATRIX (rows=true, cols=predicted):")
log("  Which classes do the weak ones get confused with?\n")
cm = confusion_matrix(y, oof_labels, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# Show normalized version (per-row = recall per class)
cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
log(f"  Normalized (% of true class → predicted):\n")
log(cm_norm.round(1).to_string())

log()
for cls in weak_classes:
    row = cm_norm.loc[cls]
    correct = row[cls]
    top_confusions = row.drop(cls).nlargest(3)
    log(f"  {cls} (correct: {correct:.1f}%):")
    for confused_cls, pct in top_confusions.items():
        log(f"    → misclassified as {confused_cls}: {pct:.1f}%")
    log()

# ═══════════════════════════════════════
# 6. GENERATE DEBUG SUBMISSION FOR VISUALISATION TOOL
# ═══════════════════════════════════════
log("=" * 60)
log("6. DEBUG SUBMISSION CSV FOR BIRD-RADAR-VISUALISATION")
log("=" * 60)

# Include OOF predictions for train set with debug columns
debug_df = oof_preds.copy()
debug_df['predicted_class'] = oof_labels.values
debug_df['true_class'] = y.values
debug_df['correct'] = (debug_df['predicted_class'] == debug_df['true_class']).astype(int)
debug_df['max_prob'] = oof_preds.max(axis=1).values

# Mark weak classes
debug_df['is_weak_class'] = y.isin(weak_classes).astype(int)
debug_df['is_misclassified_weak'] = (
    (debug_df['is_weak_class'] == 1) & (debug_df['correct'] == 0)
).astype(int)

debug_path = "debug_weak_classes_submission.csv"
debug_df.index.name = 'track_id'
debug_df.to_csv(debug_path)
log(f"  Saved {debug_path} ({len(debug_df)} rows)")
log(f"  Debug columns: predicted_class, true_class, correct, max_prob, is_weak_class, is_misclassified_weak")

# Save log
with open(LOG_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"\nFull log saved to {LOG_PATH}")
