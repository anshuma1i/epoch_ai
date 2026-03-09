# Code Snippets to Maximize mAP (Optimized from GridSearch)

Based on GridSearch optimization, here are the code snippets you can directly copy into your `test-notebook-aicup2026.ipynb` to engineer advanced trajectories features and upgrade to the LightGBM classifier.

### Snippet 1: Import New Required Libraries
Run this early in your notebook.

```python
import numpy as np
import pandas as pd
from shapely import wkb
import lightgbm as lgb
```

### Snippet 2: Define Advanced Feature Engineering Functions
Replace your basic `split_xyzm` approach with these robust trajectory geometry parsers.

```python
def parse_trajectory(hex_str):
    """Decode EWKB hex string into a list of (lon, lat, alt, rcs) tuples."""
    if not isinstance(hex_str, str) or len(hex_str) == 0:
        return []
    try:
        # Load hex string (support both WKB and EWKB)
        geom = wkb.loads(bytes.fromhex(hex_str) if not hex_str.startswith('\x01') else hex_str, hex=True)
        if geom.geom_type == 'LineString':
            return list(geom.coords)
        elif geom.geom_type == 'Point':
            return [geom.coords[0]]
        elif geom.geom_type in ('MultiPoint', 'GeometryCollection'):
            return [g.coords[0] for g in geom.geoms]
    except Exception:
        pass
    return []

def trajectory_features(row):
    """Extract numerous spatial and numeric features from a trajectory."""
    coords = parse_trajectory(row['trajectory'])
    n = len(coords)
    if n == 0:
        return pd.Series({})

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    alts = [c[2] for c in coords] if len(coords[0]) > 2 else [np.nan]*n
    rcs  = [c[3] for c in coords] if len(coords[0]) > 3 else [np.nan]*n

    # Convert lat/lon distance to approx meters
    dx = np.diff(lons) * 71000   
    dy = np.diff(lats) * 111000  
    step_dist = np.sqrt(dx**2 + dy**2)
    total_dist = step_dist.sum() if len(step_dist) > 0 else 0.0

    # Tortuosity (turning behavior)
    bearings = np.arctan2(dy, dx)
    bearing_changes = np.abs(np.diff(bearings)) if len(bearings) > 1 else np.array([0.0])
    bearing_changes = np.minimum(bearing_changes, 2*np.pi - bearing_changes) 

    return pd.Series({
        'n_points'           : n,
        'total_dist_m'       : total_dist,
        'mean_step_m'        : step_dist.mean() if len(step_dist) > 0 else 0.0,
        'std_step_m'         : step_dist.std()  if len(step_dist) > 0 else 0.0,
        'lon_range'          : max(lons) - min(lons),
        'lat_range'          : max(lats) - min(lats),
        'alt_mean'           : np.nanmean(alts),
        'alt_std'            : np.nanstd(alts),
        'rcs_mean'           : np.nanmean(rcs),
        'rcs_std'            : np.nanstd(rcs),
        'rcs_min'            : np.nanmin(rcs),
        'rcs_max'            : np.nanmax(rcs),
        'tortuosity'         : bearing_changes.mean() if len(bearing_changes) > 0 else 0.0,
        'tortuosity_max'     : bearing_changes.max() if len(bearing_changes) > 0 else 0.0,
    })
```

### Snippet 3: Extract Time Features & Apply Trajectory Engineering
Run this *after* loading your `train_df` and `test_df` but *before* you define `X` and `X_test`. This replaces the `apply(split_xyzm)` block.

```python
for df in [train_df, test_df]:
    # Parse UTC timestamps
    df['ts_start'] = pd.to_datetime(df['timestamp_start_radar_utc'], utc=True)
    df['ts_end']   = pd.to_datetime(df['timestamp_end_radar_utc'],   utc=True)
    df['duration_s'] = (df['ts_end'] - df['ts_start']).dt.total_seconds()
    
    # Extract time-of-day / seasonality features
    df['hour']       = df['ts_start'].dt.hour
    df['month']      = df['ts_start'].dt.month
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)
    
    # Derived features calculation
    df['alt_range']      = df['max_z'] - df['min_z']
    df['airspeed_per_m'] = df['airspeed'] / (df['max_z'] + 1)
    
print("Applying advanced trajectory features for train_df (this may take a minute) ...")
extra_train_cols = train_df.apply(trajectory_features, axis=1)
train_df = train_df.join(extra_train_cols)

print("Applying advanced trajectory features for test_df...")
extra_test_cols = test_df.apply(trajectory_features, axis=1)
test_df = test_df.join(extra_test_cols)
```

### Snippet 4: Update Your Feature List
Modify the variable definition of `features`. The rest of your split logic (`X = train_df[features]`) remains identical.

```python
features = [
    'airspeed', 'min_z', 'max_z', 'duration_s', 'radar_bird_size',
    'hour', 'month', 'is_daytime', 'alt_range', 'airspeed_per_m',
    'n_points', 'total_dist_m', 'mean_step_m', 'std_step_m',
    'lon_range', 'lat_range', 'alt_mean', 'alt_std',
    'rcs_mean', 'rcs_std', 'rcs_min', 'rcs_max', 'tortuosity', 'tortuosity_max'
]

X = train_df[features]
X_test = test_df[features]
```

### Snippet 5: Upgrade your Model Pipeline Based on Grid Search Results
Replace your current classifier with LightGBM and use the optimized parameters found by the gridsearch script.

```python
from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=255,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    device='gpu',
    verbose=-1
)

# Final Pipeline configured with best parameters found
pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', 'passthrough'),
    ('model', lgb_model)
])
```
