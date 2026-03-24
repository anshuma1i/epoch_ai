"""
AI Cup 2026 — Bird Radar Track Classification (Open-Meteo Mainline)
Trains a LightGBM model with group-aware CV using Open-Meteo-enriched datasets,
and generates a submission file.
"""

# ─────────────────────────────────────────────
# 1. Model Hyperparameters
# ─────────────────────────────────────────────
# Primary experiment defaults (Open-Meteo-only Kaggle baseline)
USE_ENSEMBLE = False
USE_TWO_STAGE = False

# Optional model controls
USE_FOCAL_LOSS = False
USE_PASSTHROUGH = False
USE_SMOTETOMEK = False

# Weak-class boost controls
BOOST_WEAK = 0.0
BOOST_CORMORANTS = 0.0
BOOST_WADERS = 0.0
BOOST_GEESE = 0.0

# Removed from Open-Meteo mainline by design:
# - confusion resolver path
# - weak specialist blend path
# - weak OOF reweight path
# - tide-dependent metadata interaction pack

# Two-stage controls
GULL_THRESHOLD = 0.5
UNDERSAMPLE_GULLS = 0

# Feature-pack controls
ENABLE_RCS_PACK_V1 = False
ENABLE_RCS_PACK_V2 = False
ENABLE_TRAJECTORY_PACK_V1 = True
TRAJECTORY_PACK_V1_MODE = 'full'  # options: full, no_turn, no_vertical, drop_vz_p90_abs, drop_climb_descent_ratio

if ENABLE_RCS_PACK_V1 and ENABLE_RCS_PACK_V2:
    raise ValueError("Enable only one RCS pack at a time: V1 or V2.")
if TRAJECTORY_PACK_V1_MODE not in {
    'full',
    'no_turn',
    'no_vertical',
    'drop_vz_p90_abs',
    'drop_climb_descent_ratio',
}:
    raise ValueError(
        "TRAJECTORY_PACK_V1_MODE must be one of: "
        "full, no_turn, no_vertical, drop_vz_p90_abs, drop_climb_descent_ratio"
    )

# Optional tuning/grid-search controls
RUN_GRID_SEARCH = False  # legacy alias for RUN_GRID_SEARCH_LGBM
RUN_GRID_SEARCH_LGBM = False
RUN_GRID_SEARCH_CATBOOST = False
RUN_TUNE_LGBM = False
RUN_TUNE_CATBOOST = False
TUNE_MAX_CONFIGS = 12
GRID_MAX_CONFIGS = 24
SEARCH_SEED = 42

# Output
SUBMISSION_OUT = 'submission.csv'

# Core training hyperparameters
N_ESTIMATORS = 1000
LEARNING_RATE = 0.05
NUM_LEAVES = 63
MIN_CHILD_SAMPLES = 10
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
CLASS_WEIGHT = 'balanced'
RANDOM_STATE = 42
N_SPLITS = 10

COMPETITION_CLASS_ORDER = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds",
]

# ─────────────────────────────────────────────
# 2. Imports
# ─────────────────────────────────────────────
from pathlib import Path
import numpy as np
import pandas as pd
from shapely import wkb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek
from sklearn.model_selection import ParameterGrid
from catboost import CatBoostClassifier

if RUN_GRID_SEARCH:
    RUN_GRID_SEARCH_LGBM = True
    print("Note: RUN_GRID_SEARCH is a legacy alias for RUN_GRID_SEARCH_LGBM.")
if RUN_TUNE_LGBM and RUN_GRID_SEARCH_LGBM:
    print("Note: both RUN_TUNE_LGBM and RUN_GRID_SEARCH_LGBM were set; using RUN_GRID_SEARCH_LGBM.")
if RUN_TUNE_CATBOOST and RUN_GRID_SEARCH_CATBOOST:
    print("Note: both RUN_TUNE_CATBOOST and RUN_GRID_SEARCH_CATBOOST were set; using RUN_GRID_SEARCH_CATBOOST.")

# ─────────────────────────────────────────────
# 3. Load Data (Open-Meteo-only)
# ─────────────────────────────────────────────
# Hard-wired mainline input files (no auto-detection, no fallbacks).
TRAIN_DATA_PATH = 'dataset/train_with_openmeteo.csv'
TEST_DATA_PATH = 'dataset/test_with_openmeteo.csv'

ds = {
    'train': TRAIN_DATA_PATH,
    'test': TEST_DATA_PATH,
    'wind_speed_col': 'openmeteo_wind_speed_10m_kmh',
    'wind_speed_obs_col': 'openmeteo_wind_speed_10m_kmh',
    'wind_unit_factor': 1 / 3.6,
    'weather_features': [
        'openmeteo_air_temperature_2m_c',
        'openmeteo_relative_humidity_2m_percent',
        'openmeteo_dew_point_2m_c',
        'openmeteo_precipitation_mm',
        'openmeteo_cloud_cover_percent',
        'openmeteo_pressure_msl_hpa',
        'openmeteo_weather_code',
        'openmeteo_wind_speed_10m_kmh',
        'openmeteo_wind_direction_10m_degrees',
        'openmeteo_wind_gusts_10m_kmh',
        'openmeteo_shortwave_radiation_w_m2',
        'openmeteo_sunshine_duration_s',
        'openmeteo_vapour_pressure_deficit_kpa',
        'openmeteo_is_day',
        'openmeteo_wind_dir_sin',
        'openmeteo_wind_dir_cos',
    ],
}

print("Loading simplified Open-Meteo-only pipeline...")
print(f"  train file: {ds['train']}")
print(f"  test file:  {ds['test']}")
if RUN_TUNE_CATBOOST and not USE_ENSEMBLE:
    print("Note: RUN_TUNE_CATBOOST requires USE_ENSEMBLE. CatBoost tuning will be skipped.")
if RUN_GRID_SEARCH_CATBOOST and not USE_ENSEMBLE:
    print("Note: RUN_GRID_SEARCH_CATBOOST requires USE_ENSEMBLE. CatBoost grid search will be skipped.")
print("Tide features disabled: simplified mainline does not include tide branches")
print("Removed optional paths: confusion resolver, weak specialist blend, weak reweight")

for required_path in [ds['train'], ds['test']]:
    if not Path(required_path).exists():
        raise FileNotFoundError(
            "Required mainline Open-Meteo file is missing: "
            f"{required_path}. "
            "This script only loads dataset/train_with_openmeteo.csv and "
            "dataset/test_with_openmeteo.csv."
        )

print("\n" + "="*70)
print("CONTROLLED RUN: OPEN-METEO-ONLY")
print("="*70)
print(f"Train path: {ds['train']}")
print(f"Test path:  {ds['test']}")
print(f"Ensemble enabled: {USE_ENSEMBLE}")
print(f"Two-stage enabled: {USE_TWO_STAGE}")
print(f"Gull threshold: {GULL_THRESHOLD}")
print(f"Undersample Gulls: {UNDERSAMPLE_GULLS}")
print(f"RCS pack v1 enabled: {ENABLE_RCS_PACK_V1}")
print(f"RCS pack v2 enabled: {ENABLE_RCS_PACK_V2}")
print(f"Trajectory pack v1 enabled: {ENABLE_TRAJECTORY_PACK_V1}")
print(f"Trajectory pack v1 mode: {TRAJECTORY_PACK_V1_MODE}")
print("="*70 + "\n")

train_df = pd.read_csv(ds['train']).set_index("track_id")
test_df = pd.read_csv(ds['test']).set_index("track_id")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ─────────────────────────────────────────────
# 4. Trajectory Parsing & Feature Extraction
# ─────────────────────────────────────────────

def parse_trajectory(hex_str):
    """Decode EWKB hex string into a list of (lon, lat, alt, rcs) tuples."""
    if not isinstance(hex_str, str) or len(hex_str) == 0:
        return []
    try:
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
    """Extract spatial, RCS, and velocity features from a trajectory."""
    coords = parse_trajectory(row['trajectory'])
    n = len(coords)
    if n == 0:
        return pd.Series({})

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    alts = [c[2] for c in coords] if len(coords[0]) > 2 else [np.nan] * n
    rcs  = [c[3] for c in coords] if len(coords[0]) > 3 else [np.nan] * n

    # Horizontal displacement (degrees → approx metres at ~53°N)
    dx = np.diff(lons) * 71000
    dy = np.diff(lats) * 111000
    step_dist = np.sqrt(dx**2 + dy**2)
    total_dist = step_dist.sum() if len(step_dist) > 0 else 0.0

    # Tortuosity (turning behaviour)
    bearings = np.arctan2(dy, dx)
    bearing_changes = np.abs(np.diff(bearings)) if len(bearings) > 1 else np.array([0.0])
    bearing_changes = np.minimum(bearing_changes, 2 * np.pi - bearing_changes)

    # Straightness index: displacement / total path length
    displacement = np.sqrt(
        ((lons[-1] - lons[0]) * 71000) ** 2 +
        ((lats[-1] - lats[0]) * 111000) ** 2
    )
    straightness = displacement / (total_dist + 1e-6) if total_dist > 0 else 0.0

    # Altitude change features
    alt_arr = np.array(alts, dtype=float)
    if n > 1 and not np.all(np.isnan(alt_arr)):
        alt_changes = np.diff(alt_arr)
        alt_climb_rate = np.nanmean(alt_changes)
        alt_descent_rate = np.nanmin(alt_changes)
        alt_variability = np.nanstd(alt_changes)
    else:
        alt_climb_rate = 0.0
        alt_descent_rate = 0.0
        alt_variability = 0.0

    # Safe all-NaN handling avoids warnings on tracks with missing RCS in all points.
    rcs_arr = np.array(rcs, dtype=float)
    if np.all(np.isnan(rcs_arr)):
        rcs_mean = np.nan
        rcs_std = np.nan
        rcs_min = np.nan
        rcs_max = np.nan
        rcs_range = np.nan
        rcs_p10 = np.nan
        rcs_p50 = np.nan
        rcs_p90 = np.nan
        rcs_iqr = np.nan
        rcs_diff_std = np.nan
        rcs_tv_norm = np.nan
        rcs_lag1_acf = np.nan
        rcs_peakiness = np.nan
    else:
        valid_rcs = rcs_arr[~np.isnan(rcs_arr)]
        rcs_mean = np.nanmean(valid_rcs)
        rcs_std = np.nanstd(valid_rcs)
        rcs_min = np.nanmin(valid_rcs)
        rcs_max = np.nanmax(valid_rcs)
        rcs_range = rcs_max - rcs_min

        # RCS sequence pack v1: compact temporal/distributional signal set.
        rcs_p10 = np.nanpercentile(valid_rcs, 10)
        rcs_p50 = np.nanpercentile(valid_rcs, 50)
        rcs_p90 = np.nanpercentile(valid_rcs, 90)
        rcs_q25 = np.nanpercentile(valid_rcs, 25)
        rcs_q75 = np.nanpercentile(valid_rcs, 75)
        rcs_iqr = rcs_q75 - rcs_q25

        if len(valid_rcs) > 1:
            rcs_diff = np.diff(valid_rcs)
            rcs_diff_std = np.nanstd(rcs_diff)
            rcs_tv_norm = np.nansum(np.abs(rcs_diff)) / (len(valid_rcs) - 1)
        else:
            rcs_diff_std = np.nan
            rcs_tv_norm = np.nan

        if len(valid_rcs) > 2 and np.nanstd(valid_rcs) > 1e-12:
            lag_x = valid_rcs[:-1]
            lag_y = valid_rcs[1:]
            lag_x_std = np.nanstd(lag_x)
            lag_y_std = np.nanstd(lag_y)
            if lag_x_std > 1e-12 and lag_y_std > 1e-12:
                rcs_lag1_acf = np.corrcoef(lag_x, lag_y)[0, 1]
            else:
                rcs_lag1_acf = 0.0
        else:
            rcs_lag1_acf = np.nan

        rcs_peakiness = (rcs_p90 - rcs_p50) / (rcs_iqr + 1e-6)

    # Trajectory behavior pack v1 features.
    if len(bearings) > 0:
        heading_stability_R = np.sqrt(np.mean(np.cos(bearings))**2 + np.mean(np.sin(bearings))**2)
    else:
        heading_stability_R = np.nan

    turn_rate_p50 = np.nan
    turn_rate_p90 = np.nan
    frac_high_turn = np.nan
    speed_p10 = np.nan
    speed_p90 = np.nan
    vz_p90_abs = np.nan
    climb_descent_ratio = np.nan

    feats = {
        'n_points':       n,
        'total_dist_m':   total_dist,
        'mean_step_m':    step_dist.mean() if len(step_dist) > 0 else 0.0,
        'std_step_m':     step_dist.std()  if len(step_dist) > 0 else 0.0,
        'lon_range':      max(lons) - min(lons),
        'lat_range':      max(lats) - min(lats),
        'alt_mean':       np.nanmean(alts),
        'alt_std':        np.nanstd(alts),
        'rcs_mean':       rcs_mean,
        'rcs_std':        rcs_std,
        'rcs_min':        rcs_min,
        'rcs_max':        rcs_max,
        'rcs_range':      rcs_range,
        'rcs_p10':        rcs_p10,
        'rcs_p50':        rcs_p50,
        'rcs_p90':        rcs_p90,
        'rcs_iqr':        rcs_iqr,
        'rcs_diff_std':   rcs_diff_std,
        'rcs_tv_norm':    rcs_tv_norm,
        'rcs_lag1_acf':   rcs_lag1_acf,
        'rcs_peakiness':  rcs_peakiness,
        'tortuosity':     bearing_changes.mean() if len(bearing_changes) > 0 else 0.0,
        'tortuosity_max': bearing_changes.max()  if len(bearing_changes) > 0 else 0.0,
        'straightness':       straightness,
        'alt_climb_rate':     alt_climb_rate,
        'alt_descent_rate':   alt_descent_rate,
        'alt_variability':    alt_variability,
    }

    # ── Speed & acceleration from trajectory_time ──
    times = row.get('trajectory_time', '')
    t_list = []
    if isinstance(times, str) and times.strip():
        try:
            t_list = [float(x) for x in times.strip('[]').split(',')]
        except ValueError:
            pass
    elif isinstance(times, (list, np.ndarray)):
        t_list = list(times)

    if len(t_list) == n and n > 1:
        dt = np.diff(t_list)
        dt = np.where(dt == 0, 1e-6, dt)  # avoid div-by-zero
        speeds = step_dist / dt
        feats['speed_mean'] = np.mean(speeds)
        feats['speed_std']  = np.std(speeds)
        feats['speed_max']  = np.max(speeds)
        speed_p10 = np.nanpercentile(speeds, 10)
        speed_p90 = np.nanpercentile(speeds, 90)

        feats['speed_cv'] = np.std(speeds) / (np.mean(speeds) + 1e-6)

        # Turn rate from wrapped heading change over elapsed time.
        if len(bearings) > 1:
            turn_rate = bearing_changes / dt[1:]
            abs_turn_rate = np.abs(turn_rate)
            turn_rate_p50 = np.nanpercentile(abs_turn_rate, 50)
            turn_rate_p90 = np.nanpercentile(abs_turn_rate, 90)
            tr_q25 = np.nanpercentile(abs_turn_rate, 25)
            tr_q75 = np.nanpercentile(abs_turn_rate, 75)
            tr_iqr = tr_q75 - tr_q25
            # Robust per-track threshold avoids brittle global constants.
            high_turn_threshold = turn_rate_p50 + tr_iqr
            frac_high_turn = np.mean(abs_turn_rate > high_turn_threshold)
        else:
            turn_rate_p50 = np.nan
            turn_rate_p90 = np.nan
            frac_high_turn = np.nan

        if not np.all(np.isnan(alt_arr)) and len(alt_arr) > 1:
            vz = np.diff(alt_arr) / dt
            abs_vz = np.abs(vz)
            vz_p90_abs = np.nanpercentile(abs_vz, 90)
            climb_total = np.nansum(vz[vz > 0])
            descent_total = np.nansum(np.abs(vz[vz < 0]))
            climb_descent_ratio = climb_total / (descent_total + 1e-6)
        else:
            vz_p90_abs = np.nan
            climb_descent_ratio = np.nan

        if len(speeds) > 1:
            accel = np.diff(speeds) / dt[1:]
            feats['accel_mean'] = np.mean(accel)
            feats['accel_std']  = np.std(accel)
        else:
            feats['accel_mean'] = 0.0
            feats['accel_std']  = 0.0
    else:
        feats['speed_mean'] = np.nan
        feats['speed_std']  = np.nan
        feats['speed_max']  = np.nan
        feats['speed_cv']   = np.nan
        feats['accel_mean'] = np.nan
        feats['accel_std']  = np.nan

    feats['turn_rate_p50'] = turn_rate_p50
    feats['turn_rate_p90'] = turn_rate_p90
    feats['frac_high_turn'] = frac_high_turn
    feats['speed_p10'] = speed_p10
    feats['speed_p90'] = speed_p90
    feats['vz_p90_abs'] = vz_p90_abs
    feats['climb_descent_ratio'] = climb_descent_ratio
    feats['heading_stability_R'] = heading_stability_R

    return pd.Series(feats)


# ─────────────────────────────────────────────
# 5. Feature Engineering
# ─────────────────────────────────────────────
for df in [train_df, test_df]:
    # Timestamps
    df['ts_start'] = pd.to_datetime(df['timestamp_start_radar_utc'], utc=True)
    df['ts_end']   = pd.to_datetime(df['timestamp_end_radar_utc'],   utc=True)
    df['duration_s'] = (df['ts_end'] - df['ts_start']).dt.total_seconds()

    # Time-of-day / seasonality
    df['hour']  = df['ts_start'].dt.hour
    df['month'] = df['ts_start'].dt.month
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)

    # Cyclical encoding
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Derived radar features
    df['alt_range']      = df['max_z'] - df['min_z']
    df['airspeed_per_m'] = df['airspeed'] / (df['max_z'] + 1)

    # Wind-relative features (convert to m/s if needed)
    wf = ds['wind_unit_factor']
    df['headwind_component'] = df['airspeed'] - df[ds['wind_speed_obs_col']] * wf
    df['airspeed_wind_ratio'] = df['airspeed'] / (df[ds['wind_speed_col']] * wf + 0.1)

    # Compute wind direction sin/cos for Open-Meteo wind direction.
    wd = df['openmeteo_wind_direction_10m_degrees']
    df['openmeteo_wind_dir_sin'] = np.sin(2 * np.pi * wd / 360)
    df['openmeteo_wind_dir_cos'] = np.cos(2 * np.pi * wd / 360)

# Trajectory features
print("Extracting trajectory features for train_df...")
train_df = train_df.join(train_df.apply(trajectory_features, axis=1))
print("Extracting trajectory features for test_df...")
test_df = test_df.join(test_df.apply(trajectory_features, axis=1))

# ─────────────────────────────────────────────
# 6. Feature List
# ─────────────────────────────────────────────
base_features = [
    'airspeed', 'min_z', 'max_z', 'duration_s', 'radar_bird_size',
    'hour', 'month', 'is_daytime',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'alt_range', 'airspeed_per_m',
    'headwind_component', 'airspeed_wind_ratio',
]

trajectory_feats = [
    'n_points', 'total_dist_m', 'mean_step_m', 'std_step_m',
    'lon_range', 'lat_range', 'alt_mean', 'alt_std',
    'rcs_mean', 'rcs_std', 'rcs_min', 'rcs_max', 'rcs_range',
    'tortuosity', 'tortuosity_max',
    'straightness',
    'alt_climb_rate', 'alt_descent_rate', 'alt_variability',
    'speed_mean', 'speed_std', 'speed_max', 'speed_cv',
    'accel_mean', 'accel_std',
]

rcs_pack_v1_features = [
    'rcs_p10',
    'rcs_p50',
    'rcs_p90',
    'rcs_iqr',
    'rcs_diff_std',
    'rcs_tv_norm',
    'rcs_lag1_acf',
    'rcs_peakiness',
]

rcs_pack_v2_features = [
    'rcs_diff_std',
    'rcs_tv_norm',
    'rcs_lag1_acf',
    'rcs_peakiness',
]

trajectory_pack_v1_features = [
    'turn_rate_p50',
    'turn_rate_p90',
    'frac_high_turn',
    'speed_p10',
    'speed_p90',
    'vz_p90_abs',
    'climb_descent_ratio',
    'heading_stability_R',
]

trajectory_pack_v1_turn_features = [
    'turn_rate_p50',
    'turn_rate_p90',
    'frac_high_turn',
]
trajectory_pack_v1_vertical_features = [
    'vz_p90_abs',
    'climb_descent_ratio',
]

if TRAJECTORY_PACK_V1_MODE == 'full':
    trajectory_pack_v1_active_features = list(trajectory_pack_v1_features)
elif TRAJECTORY_PACK_V1_MODE == 'no_turn':
    trajectory_pack_v1_active_features = [
        f for f in trajectory_pack_v1_features
        if f not in trajectory_pack_v1_turn_features
    ]
elif TRAJECTORY_PACK_V1_MODE == 'no_vertical':
    trajectory_pack_v1_active_features = [
        f for f in trajectory_pack_v1_features
        if f not in trajectory_pack_v1_vertical_features
    ]
elif TRAJECTORY_PACK_V1_MODE == 'drop_vz_p90_abs':
    trajectory_pack_v1_active_features = [
        f for f in trajectory_pack_v1_features
        if f != 'vz_p90_abs'
    ]
else:  # drop_climb_descent_ratio
    trajectory_pack_v1_active_features = [
        f for f in trajectory_pack_v1_features
        if f != 'climb_descent_ratio'
    ]

if ENABLE_RCS_PACK_V1:
    trajectory_feats += rcs_pack_v1_features
elif ENABLE_RCS_PACK_V2:
    trajectory_feats += rcs_pack_v2_features

if ENABLE_TRAJECTORY_PACK_V1:
    trajectory_feats += trajectory_pack_v1_active_features

weather_features = ds['weather_features']

metadata_interaction_features = []
features = base_features + trajectory_feats + weather_features

print("Active feature groups summary:")
print(f"  base features: {len(base_features)}")
print(f"  trajectory features: {len(trajectory_feats)}")
print(f"  weather features: {len(weather_features)}")
print(f"  rcs pack v1 features: {rcs_pack_v1_features if ENABLE_RCS_PACK_V1 else 'disabled'}")
print(f"  rcs pack v2 features: {rcs_pack_v2_features if ENABLE_RCS_PACK_V2 else 'disabled'}")
print(f"  trajectory pack v1 features: {trajectory_pack_v1_active_features if ENABLE_TRAJECTORY_PACK_V1 else 'disabled'}")
print("  metadata interaction features: 0 (disabled in simplified mainline)")
print(f"  total selected features: {len(features)}")

missing_train = [f for f in features if f not in train_df.columns]
missing_test = [f for f in features if f not in test_df.columns]
if missing_train or missing_test:
    raise KeyError(
        "Missing feature columns after selection. "
        f"Missing in train: {missing_train}. Missing in test: {missing_test}"
    )

X = train_df[features]
X_test = test_df[features]
y = train_df['bird_group']

print(f"Feature matrix: X={X.shape}, X_test={X_test.shape}, classes={y.nunique()}")

# ─────────────────────────────────────────────
# 7. Model & Pipeline
# ─────────────────────────────────────────────
N_CLASSES = 9
FOCAL_GAMMA = 2.0

def focal_loss_lgb(y_true, y_pred, *_args, **_kwargs):
    """Multi-class focal loss custom objective for LightGBM."""
    y_pred = y_pred.reshape(-1, N_CLASSES, order='F')
    # Softmax
    p = np.exp(y_pred - y_pred.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    # One-hot
    y_onehot = np.eye(N_CLASSES)[y_true.astype(int)]
    pt = (p * y_onehot).sum(axis=1, keepdims=True)
    focal_weight = (1 - pt) ** FOCAL_GAMMA
    # Gradient & Hessian
    grad = (focal_weight * (p - y_onehot)).flatten(order='F')
    hess = (focal_weight * p * (1 - p)).flatten(order='F')
    return grad, hess

# Model and preprocessing setup
numeric_features = [f for f in features if f != 'radar_bird_size']
categorical_features = ['radar_bird_size']
cat_indices = [len(numeric_features) + i for i in range(len(categorical_features))]

imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='median', keep_empty_features=True), numeric_features),
    ('cat_imputer', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), categorical_features)
])

# Oversampler selection
if USE_PASSTHROUGH:
    oversampler = 'passthrough'
    print("Oversampler: passthrough (no oversampling)")
elif USE_SMOTETOMEK:
    oversampler = SMOTETomek(
        smote=SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
    print("Oversampler: SMOTETomek (SMOTENC + TomekLinks)")
else:
    oversampler = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)
    print("Oversampler: SMOTENC (default)")

# CPU-safe defaults: do not assume CUDA/MPS backend availability for
# LightGBM/CatBoost on this Apple Silicon Mac.
lgb_params = dict(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    num_leaves=NUM_LEAVES,
    min_child_samples=MIN_CHILD_SAMPLES,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_STATE,
    device_type='cpu',
    n_jobs=-1,
    verbose=-1,
)

if USE_FOCAL_LOSS:
    print("Using focal loss objective")
    lgb_params['objective'] = focal_loss_lgb

lgb_model = LGBMClassifier(**lgb_params)

# Per-class boost multipliers (fall back to BOOST_WEAK if per-class not set)
CLASS_BOOST = {}
for cls, cli_val in [('Cormorants', BOOST_CORMORANTS),
                     ('Waders', BOOST_WADERS),
                     ('Geese', BOOST_GEESE)]:
    mult = cli_val if cli_val > 0 else BOOST_WEAK
    if mult > 0:
        CLASS_BOOST[cls] = mult

if CLASS_BOOST:
    print("Per-class boost multipliers:")
    for cls, mult in CLASS_BOOST.items():
        print(f"  {cls}: {mult}x")
# Keep a single flag for backward compat in run_cv
boost_weak_mult = 1 if CLASS_BOOST else 0

pipeline = ImbPipeline([
    ('imputer', imputer),
    ('oversampler', oversampler),
    ('model', lgb_model)
])

# ─────────────────────────────────────────────
# 8. Group-Aware Cross-Validation + Training
# ─────────────────────────────────────────────
n_splits = N_SPLITS
groups = train_df['primary_observation_id']
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
split = list(cv.split(X, y, groups))

classes = np.sort(y.unique())
oof_preds = pd.DataFrame(0.0, index=X.index, columns=classes)
test_preds = np.zeros((len(X_test), len(classes)))

def run_cv(pipeline, X, y, split, classes, X_test, boost_weak_mult=0, label=""):
    """Run group-aware CV, return OOF predictions and averaged test predictions."""
    oof_preds = pd.DataFrame(0.0, index=X.index, columns=classes)
    test_preds = np.zeros((len(X_test), len(classes)))

    if label:
        print(f"\n{label}")
    print(f"Training {len(split)}-fold StratifiedGroupKFold...")

    for i, (train_idx, val_idx) in enumerate(split):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold   = X.iloc[val_idx]
        y_val_fold   = y.iloc[val_idx]

        pipeline_fold = clone(pipeline)

        if boost_weak_mult > 0:
            # Manually run imputer + oversampler
            preprocessor = Pipeline([
                (pipeline_fold.steps[0][0], pipeline_fold.steps[0][1]),  # imputer
            ])
            oversampler_step = pipeline_fold.steps[1][1]
            model_step = pipeline_fold.steps[2][1]

            X_transformed = preprocessor.fit_transform(X_train_fold, y_train_fold)
            X_val_transformed = preprocessor.transform(X_val_fold)
            X_test_transformed = preprocessor.transform(X_test)

            if oversampler_step != 'passthrough':
                X_resampled, y_resampled = oversampler_step.fit_resample(X_transformed, y_train_fold)
            else:
                X_resampled, y_resampled = X_transformed, y_train_fold

            # Safe boosting: duplicate rows per class with individual multipliers
            # This avoids LightGBM C++ GPU crashes with custom float weights
            X_resampled_np = X_resampled if isinstance(X_resampled, np.ndarray) else X_resampled.values
            y_resampled_np = y_resampled if isinstance(y_resampled, np.ndarray) else y_resampled.values

            extra_X, extra_y = [], []
            for cls, mult in CLASS_BOOST.items():
                repeats = int(mult) - 1
                if repeats <= 0:
                    continue
                if hasattr(y_resampled, 'values'):
                    cls_mask = (y_resampled == cls).values
                else:
                    cls_mask = (y_resampled_np == cls)
                X_cls = X_resampled_np[cls_mask]
                y_cls = y_resampled_np[cls_mask]
                if len(X_cls) > 0:
                    extra_X.extend([X_cls] * repeats)
                    extra_y.extend([y_cls] * repeats)

            if extra_X:
                X_resampled_np = np.vstack([X_resampled_np] + extra_X)
                y_resampled_np = np.concatenate([y_resampled_np] + extra_y)

            # Increase min_child_samples to avoid degenerate splits from duplicated rows
            model_step.set_params(min_child_samples=max(10, int(len(X_resampled_np) * 0.01)))
            model_step.fit(X_resampled_np, y_resampled_np)
            val_proba = model_step.predict_proba(X_val_transformed)
            test_proba = model_step.predict_proba(X_test_transformed)
        else:
            pipeline_fold.fit(X_train_fold, y_train_fold)
            val_proba = pipeline_fold.predict_proba(X_val_fold)
            test_proba = pipeline_fold.predict_proba(X_test)

        oof_preds.iloc[val_idx] = val_proba
        test_preds += test_proba

        fold_ap = average_precision_score(
            pd.get_dummies(y_val_fold).reindex(columns=classes, fill_value=0),
            val_proba, average='macro'
        )
        print(f"  Fold {i+1}/{len(split)} — train: {len(train_idx)}, val: {len(val_idx)}, val mAP: {fold_ap:.4f}")

    test_preds /= len(split)
    return oof_preds, test_preds


def compute_macro_map(oof_pred_df: pd.DataFrame) -> float:
    """Compute macro-averaged AP on OOF predictions using competition class order."""
    needed_cols = COMPETITION_CLASS_ORDER
    solution_df_local = (
        train_df.reset_index()
        .groupby(["track_id", "bird_group"]).size()
        .unstack(fill_value=0)
    )
    oof_aligned_local = oof_pred_df.loc[solution_df_local.index, solution_df_local.columns]
    return average_precision_score(
        solution_df_local[needed_cols],
        oof_aligned_local[needed_cols],
        average='macro'
    )


def compute_class_ap(oof_pred_df: pd.DataFrame, class_name: str) -> float:
    """Compute AP for a single class using the same OOF alignment as final evaluation."""
    solution_df_local = (
        train_df.reset_index()
        .groupby(["track_id", "bird_group"]).size()
        .unstack(fill_value=0)
    )
    if class_name not in solution_df_local.columns or class_name not in oof_pred_df.columns:
        return np.nan
    oof_aligned_local = oof_pred_df.loc[solution_df_local.index, solution_df_local.columns]
    return average_precision_score(solution_df_local[class_name], oof_aligned_local[class_name])


def build_sampled_grid(search_space, max_configs, seed):
    """Build a shuffled parameter grid and keep only a practical number of configs."""
    full_grid = list(ParameterGrid(search_space))
    rng = np.random.RandomState(seed)
    rng.shuffle(full_grid)
    if max_configs and max_configs > 0:
        return full_grid[:max_configs], len(full_grid)
    return full_grid, len(full_grid)


def prepend_baseline_if_missing(configs, baseline):
    """Ensure baseline configuration is evaluated for direct tuned-vs-untuned comparison."""
    baseline_key = tuple(sorted(baseline.items()))
    seen = {tuple(sorted(cfg.items())) for cfg in configs}
    if baseline_key in seen:
        return configs
    return [baseline] + configs


# ─────────────────────────────────────────────
# 8b. Optional LightGBM Tuning / Grid Search
# ─────────────────────────────────────────────
run_lgbm_grid = RUN_GRID_SEARCH_LGBM
run_lgbm_compact = RUN_TUNE_LGBM and not run_lgbm_grid

if run_lgbm_compact or run_lgbm_grid:
    print("\n" + "=" * 50)
    if run_lgbm_grid:
        print("LIGHTGBM SAMPLED GRID SEARCH")
    else:
        print("LIGHTGBM COMPACT TUNING")
    print("=" * 50)

    compact_space = {
        'n_estimators': [800, 1000, 1300],
        'learning_rate': [0.03, 0.05],
        'num_leaves': [31, 63, 95],
        'min_child_samples': [8, 15, 25],
        'max_depth': [-1, 10],
        'subsample': [0.75, 0.9],
        'colsample_bytree': [0.75, 0.9],
        'reg_alpha': [0.0, 0.2],
        'reg_lambda': [0.0, 1.0],
    }
    grid_space = {
        'n_estimators': [700, 1000, 1300, 1600],
        'learning_rate': [0.02, 0.03, 0.05],
        'num_leaves': [31, 63, 95, 127],
        'min_child_samples': [8, 15, 25, 40],
        'max_depth': [-1, 8, 12],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
        'reg_alpha': [0.0, 0.2, 0.5],
        'reg_lambda': [0.0, 1.0, 2.0],
    }

    search_space = grid_space if run_lgbm_grid else compact_space
    max_configs = GRID_MAX_CONFIGS if run_lgbm_grid else TUNE_MAX_CONFIGS
    search_label = "LightGBM sampled grid search" if run_lgbm_grid else "LightGBM compact tuning"

    param_grid, total_configs = build_sampled_grid(search_space, max_configs, SEARCH_SEED)
    baseline_lgbm = {
        'n_estimators': lgb_params.get('n_estimators', 1000),
        'learning_rate': lgb_params.get('learning_rate', 0.05),
        'num_leaves': lgb_params.get('num_leaves', 63),
        'min_child_samples': lgb_params.get('min_child_samples', 10),
        'max_depth': lgb_params.get('max_depth', -1),
        'subsample': lgb_params.get('subsample', 0.8),
        'colsample_bytree': lgb_params.get('colsample_bytree', 0.8),
        'reg_alpha': lgb_params.get('reg_alpha', 0.0),
        'reg_lambda': lgb_params.get('reg_lambda', 0.0),
    }
    param_grid = prepend_baseline_if_missing(param_grid, baseline_lgbm)

    print(f"Total candidate configurations before cap: {total_configs}")
    print(f"Evaluating configurations: {len(param_grid)}")

    best_score = -1.0
    best_params = None
    best_oof = None
    best_test = None

    for i, model_params in enumerate(param_grid):
        print(f"\n[{i+1}/{len(param_grid)}] model params: {model_params}")
        candidate = clone(pipeline)
        candidate.set_params(**{f'model__{k}': v for k, v in model_params.items()})
        oof, t_preds = run_cv(
            candidate,
            X,
            y,
            split,
            classes,
            X_test,
            boost_weak_mult,
            label=search_label
        )
        score = compute_macro_map(oof)
        print(f"  -> mAP: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = model_params
            best_oof = oof
            best_test = t_preds

    print(f"\n{'=' * 50}")
    print(f"Best LightGBM mAP: {best_score:.4f}")
    print(f"Best LightGBM params: {best_params}")
    print(f"{'=' * 50}")

    oof_preds = best_oof
    test_preds = best_test
    pipeline.set_params(**{f'model__{k}': v for k, v in best_params.items()})

else:
    # ── Single run (default) ──
    oof_preds, test_preds = run_cv(pipeline, X, y, split, classes, X_test, boost_weak_mult)

# ─────────────────────────────────────────────
# 8c. CatBoost Ensemble (optional)
# ─────────────────────────────────────────────
if USE_ENSEMBLE:
    print("\n" + "=" * 50)
    print("CATBOOST ENSEMBLE")
    print("=" * 50)

    # After ColumnTransformer, numeric cols come first, then cat cols
    cb_cat_idx = [len(numeric_features) + i for i in range(len(categorical_features))]

    default_cb_params = dict(
        iterations=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        depth=8,
        l2_leaf_reg=3,
        bagging_temperature=0.0,
        random_strength=1.0,
        bootstrap_type='Bayesian',
        auto_class_weights='Balanced',
        task_type='CPU',
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    run_cb_grid = RUN_GRID_SEARCH_CATBOOST
    run_cb_compact = RUN_TUNE_CATBOOST and not run_cb_grid

    cb_param_grid = [default_cb_params]
    if run_cb_compact or run_cb_grid:
        compact_cb_space = {
            'iterations': [700, 1000, 1300],
            'learning_rate': [0.03, 0.05],
            'depth': [6, 8],
            'l2_leaf_reg': [3, 6],
            'bagging_temperature': [0.0, 0.5],
            'random_strength': [1.0, 2.0],
        }
        grid_cb_space = {
            'iterations': [700, 1000, 1300, 1600],
            'learning_rate': [0.02, 0.03, 0.05],
            'depth': [6, 8, 10],
            'l2_leaf_reg': [3, 6, 9],
            'bagging_temperature': [0.0, 0.5, 1.0],
            'random_strength': [1.0, 2.0, 4.0],
        }
        cb_space = grid_cb_space if run_cb_grid else compact_cb_space
        cb_max_configs = GRID_MAX_CONFIGS if run_cb_grid else TUNE_MAX_CONFIGS
        sampled_cb_params, cb_total = build_sampled_grid(
            cb_space,
            cb_max_configs,
            SEARCH_SEED + 1,
        )
        baseline_cb = {
            'iterations': default_cb_params['iterations'],
            'learning_rate': default_cb_params['learning_rate'],
            'depth': default_cb_params['depth'],
            'l2_leaf_reg': default_cb_params['l2_leaf_reg'],
            'bagging_temperature': default_cb_params['bagging_temperature'],
            'random_strength': default_cb_params['random_strength'],
        }
        sampled_cb_params = prepend_baseline_if_missing(sampled_cb_params, baseline_cb)
        cb_param_grid = [{**default_cb_params, **p} for p in sampled_cb_params]

        if run_cb_grid:
            print(f"CatBoost grid-search candidates before cap: {cb_total}")
            print(f"CatBoost grid-search configs to evaluate: {len(cb_param_grid)}")
        else:
            print(f"CatBoost compact-tuning candidates before cap: {cb_total}")
            print(f"CatBoost compact-tuning configs to evaluate: {len(cb_param_grid)}")

    # CatBoost needs categoricals as strings, not ordinal-encoded floats
    cb_imputer = ColumnTransformer([
        ('num_imputer', SimpleImputer(strategy='median', keep_empty_features=True), numeric_features),
        ('cat_imputer', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ]), categorical_features)
    ])

    best_cb_map = -1.0
    best_cb_params = None
    best_cb_oof = None
    best_cb_test = None

    for cfg_i, cb_params in enumerate(cb_param_grid, start=1):
        if run_cb_compact or run_cb_grid:
            print(f"\nCatBoost config [{cfg_i}/{len(cb_param_grid)}]: {cb_params}")
        else:
            print(f"Training {len(split)}-fold CatBoost...")

        cb_oof = pd.DataFrame(0.0, index=X.index, columns=classes)
        cb_test = np.zeros((len(X_test), len(classes)))

        for i, (train_idx, val_idx) in enumerate(split):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            cb_imp = clone(cb_imputer)
            X_tr = cb_imp.fit_transform(X_train_fold, y_train_fold)
            X_va = cb_imp.transform(X_val_fold)
            X_te = cb_imp.transform(X_test)

            # Cast categorical columns to string for CatBoost
            for ci in cb_cat_idx:
                X_tr[:, ci] = X_tr[:, ci].astype(str)
                X_va[:, ci] = X_va[:, ci].astype(str)
                X_te[:, ci] = X_te[:, ci].astype(str)

            cb_params_cpu = dict(cb_params)
            # Hard-enforce CPU in every CatBoost training path.
            cb_params_cpu['task_type'] = 'CPU'
            cb_params_cpu.pop('devices', None)
            cb_params_cpu.pop('gpu_ram_part', None)
            cb_params_cpu.pop('gpu_cat_features_storage', None)

            cb = CatBoostClassifier(**cb_params_cpu)
            cb.fit(
                X_tr,
                y_train_fold,
                eval_set=(X_va, y.iloc[val_idx]),
                early_stopping_rounds=50,
                cat_features=cb_cat_idx,
            )

            val_proba = cb.predict_proba(X_va)
            test_proba = cb.predict_proba(X_te)

            cb_oof.iloc[val_idx] = val_proba
            cb_test += test_proba

            fold_ap = average_precision_score(
                pd.get_dummies(y.iloc[val_idx]).reindex(columns=classes, fill_value=0),
                val_proba,
                average='macro'
            )
            print(f"  Fold {i+1}/{len(split)} — val mAP: {fold_ap:.4f}")

        cb_test /= len(split)
        cb_map = compute_macro_map(cb_oof)
        print(f"  CatBoost standalone mAP: {cb_map:.4f}")

        if cb_map > best_cb_map:
            best_cb_map = cb_map
            best_cb_params = cb_params
            best_cb_oof = cb_oof
            best_cb_test = cb_test

    if run_cb_compact or run_cb_grid:
        print(f"\n{'=' * 50}")
        print(f"Best CatBoost mAP: {best_cb_map:.4f}")
        print(f"Best CatBoost params: {best_cb_params}")
        print(f"{'=' * 50}")

    # Simple average ensemble
    oof_preds = (oof_preds + best_cb_oof) / 2
    test_preds = (test_preds + best_cb_test) / 2
    print("  Ensembled LightGBM + CatBoost (simple average)")

# ─────────────────────────────────────────────
# 8d. Two-Stage Gull Detector (optional)
# ─────────────────────────────────────────────
if USE_TWO_STAGE:
    print("\n" + "=" * 50)
    print("TWO-STAGE GULL DETECTOR")
    print("=" * 50)

    non_gull_classes = np.sort([c for c in classes if c != 'Gulls'])

    # Stage 1: Binary Gull vs non-Gull
    y_binary = (y == 'Gulls').astype(int)

    lgb_binary = LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE_BYTREE,
        class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1,
        device_type='cpu',
        verbose=-1,
    )
    binary_pipeline = ImbPipeline([
        ('imputer', clone(imputer)),
        ('oversampler', 'passthrough'),
        ('model', lgb_binary)
    ])

    # Stage 2: 8-class on non-Gull samples only
    lgb_multi = LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE_BYTREE,
        class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE, n_jobs=-1,
        device_type='cpu',
        verbose=-1,
    )
    multi_pipeline = ImbPipeline([
        ('imputer', clone(imputer)),
        ('oversampler', 'passthrough'),
        ('model', lgb_multi)
    ])

    ts_oof = pd.DataFrame(0.0, index=X.index, columns=classes)
    ts_test = np.zeros((len(X_test), len(classes)))

    gull_thresh = GULL_THRESHOLD
    us_gulls = UNDERSAMPLE_GULLS
    if gull_thresh != 0.5:
        print(f"  Gull threshold: {gull_thresh} (raw p(Gull) must exceed this to count as Gull)")
    if us_gulls > 0:
        print(f"  Undersampling Gulls to {us_gulls} tracks per fold")

    print(f"Training {len(split)}-fold two-stage...")
    for i, (train_idx, val_idx) in enumerate(split):
        X_tr = X.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        # ── Optional: undersample Gulls in training set ──
        if us_gulls > 0:
            gull_mask = y_tr == 'Gulls'
            gull_indices = y_tr.index[gull_mask]
            non_gull_indices = y_tr.index[~gull_mask]
            n_gulls = len(gull_indices)
            if n_gulls > us_gulls:
                rng = np.random.RandomState(RANDOM_STATE + i)
                keep = rng.choice(gull_indices, size=us_gulls, replace=False)
                keep_idx = np.concatenate([keep, non_gull_indices.values])
                X_tr = X_tr.loc[keep_idx]
                y_tr = y_tr.loc[keep_idx]

        # ── Stage 1: Gull binary ──
        y_tr_bin = (y_tr == 'Gulls').astype(int)
        pipe1 = clone(binary_pipeline)
        pipe1.fit(X_tr, y_tr_bin)
        # p(Gull) is column index 1
        val_p_gull_raw = pipe1.predict_proba(X_va)[:, 1]
        test_p_gull_raw = pipe1.predict_proba(X_test)[:, 1]

        # ── Asymmetric threshold: re-calibrate p(Gull) ──
        if gull_thresh != 0.5:
            # We want to maintain strict monotonic ranking. 
            # If threshold is high (e.g. 0.8), we heavily suppress anything below it.
            # Using exponentiation smoothly squashes probabilities without breaking rankings.
            # Example: penalty maps threshold -> 0.5
            gamma = np.log(0.5) / np.log(gull_thresh)
            val_p_gull = np.power(val_p_gull_raw, gamma)
            test_p_gull = np.power(test_p_gull_raw, gamma)
        else:
            val_p_gull = val_p_gull_raw
            test_p_gull = test_p_gull_raw

        # ── Stage 2: non-Gull multi-class ──
        non_gull_mask = y_tr != 'Gulls'
        X_tr_ng = X_tr[non_gull_mask]
        y_tr_ng = y_tr[non_gull_mask]

        pipe2 = clone(multi_pipeline)
        pipe2.fit(X_tr_ng, y_tr_ng)
        # Get class order from fitted model
        s2_classes = pipe2.classes_

        val_p_rest = pipe2.predict_proba(X_va)
        test_p_rest = pipe2.predict_proba(X_test)

        # ── Combine: p(class) = (1 - p_gull) * p(class | non-gull) ──
        val_combined = np.zeros((len(X_va), len(classes)))
        test_combined = np.zeros((len(X_test), len(classes)))

        gull_idx = list(classes).index('Gulls')
        val_combined[:, gull_idx] = val_p_gull
        test_combined[:, gull_idx] = test_p_gull

        for j, cls in enumerate(s2_classes):
            cls_idx = list(classes).index(cls)
            val_combined[:, cls_idx] = (1 - val_p_gull) * val_p_rest[:, j]
            test_combined[:, cls_idx] = (1 - test_p_gull) * test_p_rest[:, j]

        ts_oof.iloc[val_idx] = val_combined
        ts_test += test_combined

        fold_ap = average_precision_score(
            pd.get_dummies(y_va).reindex(columns=classes, fill_value=0),
            val_combined, average='macro'
        )
        print(f"  Fold {i+1}/{len(split)} — val mAP: {fold_ap:.4f}")

    ts_test /= len(split)

    # Print two-stage standalone score
    solution_df_ts = (
        train_df.reset_index()
        .groupby(["track_id", "bird_group"]).size()
        .unstack(fill_value=0)
    )
    needed_ts = COMPETITION_CLASS_ORDER
    ts_oof_aligned = ts_oof.loc[solution_df_ts.index, solution_df_ts.columns]
    ts_map = average_precision_score(
        solution_df_ts[needed_ts], ts_oof_aligned[needed_ts], average='macro'
    )
    print(f"\n  Two-stage standalone mAP: {ts_map:.4f}")

    # Print per-class for comparison
    print("  Per-class AP (two-stage):")
    for cls in needed_ts:
        if cls in solution_df_ts.columns:
            ap = average_precision_score(solution_df_ts[cls], ts_oof_aligned[cls])
            print(f"    {cls:20s}: {ap:.4f}")

    # Average two-stage with existing predictions
    oof_preds = (oof_preds + ts_oof) / 2
    test_preds = (test_preds + ts_test) / 2
    print("\n  Ensembled with base model (simple average)")

# ─────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────
needed_columns = COMPETITION_CLASS_ORDER

# Build ground-truth OOF solution
solution_df = (
    train_df
    .reset_index()
    .groupby(["track_id", "bird_group"])
    .size()
    .unstack(fill_value=0)
)

# Align OOF predictions to solution_df
oof_aligned = oof_preds.loc[solution_df.index, solution_df.columns]

overall_map = average_precision_score(
    solution_df[needed_columns],
    oof_aligned[needed_columns],
    average='macro'
)

print(f"\n{'='*50}")
print(f" OOF Macro-Averaged AP (mAP): {overall_map:.4f}")
print(f"{'='*50}")
print("\n Per-Class Average Precision:")
for cls in needed_columns:
    if cls in solution_df.columns and cls in oof_aligned.columns:
        ap = average_precision_score(solution_df[cls], oof_aligned[cls])
        print(f"   {cls:20s}: {ap:.4f}")

# ─────────────────────────────────────────────
# 9b. WEAK-CLASS DIAGNOSTICS (Cormorants & Waders)
# ─────────────────────────────────────────────

# Create diagnostics directory
diag_dir = Path('diagnostics')
diag_dir.mkdir(exist_ok=True)

def analyze_false_negatives(solution_df, oof_aligned, target_class, classes_list):
    """
    For rows where true label = target_class but prediction is wrong,
    count which predicted classes absorbed them most often.
    Returns: DataFrame with top 5 predicted classes, counts, and percentages.
    """
    # Get true positives for target class
    solution_target = solution_df[target_class]
    pred_target = oof_aligned[target_class]
    
    # Rows where true label is target class
    true_mask = solution_target == 1
    if true_mask.sum() == 0:
        return pd.DataFrame({'predicted_class': [], 'count': [], 'percentage': []})
    
    # Among those rows, find where the target class is NOT the top prediction
    # (i.e., false negatives)
    predicted_classes = oof_aligned.loc[true_mask].idxmax(axis=1)
    true_predictions = predicted_classes == target_class
    false_negatives = predicted_classes[~true_predictions]
    
    if len(false_negatives) == 0:
        return pd.DataFrame({'predicted_class': [], 'count': [], 'percentage': []})
    
    # Count which classes absorbed the false negatives
    absorption = false_negatives.value_counts()
    absorption_df = pd.DataFrame({
        'predicted_class': absorption.index,
        'count': absorption.values,
        'percentage': (100 * absorption.values / len(false_negatives)).round(1)
    }).reset_index(drop=True)
    
    return absorption_df.head(5)


def analyze_false_positives(solution_df, oof_aligned, target_class, classes_list):
    """
    For rows where predicted label = target_class but true label is different,
    count which true classes are most often being mistaken as the target.
    Returns: DataFrame with top 5 source classes, counts, and percentages.
    """
    # Rows where target class is the top prediction
    predicted_top = oof_aligned.idxmax(axis=1)
    pred_mask = predicted_top == target_class
    
    if pred_mask.sum() == 0:
        return pd.DataFrame({'true_class': [], 'count': [], 'percentage': []})
    
    # Among those, find where the true label is NOT target class (false positives)
    true_classes = solution_df.loc[pred_mask].idxmax(axis=1)
    true_match = true_classes == target_class
    false_positives = true_classes[~true_match]
    
    if len(false_positives) == 0:
        return pd.DataFrame({'true_class': [], 'count': [], 'percentage': []})
    
    # Count which true classes were mistaken for target
    sources = false_positives.value_counts()
    sources_df = pd.DataFrame({
        'true_class': sources.index,
        'count': sources.values,
        'percentage': (100 * sources.values / len(false_positives)).round(1)
    }).reset_index(drop=True)
    
    return sources_df.head(5)


def analyze_topk_probability(solution_df, oof_aligned, target_class):
    """
    For rows where true label = target_class:
    - Compute how often target class appears in top 1, top 2, top 3
    - Report average probability assigned to target class on correct/incorrect predictions
    Returns: dict with diagnostics.
    """
    solution_target = solution_df[target_class]
    pred_target = oof_aligned[target_class]
    
    # Rows where true label is target class
    true_mask = solution_target == 1
    if true_mask.sum() == 0:
        return {}
    
    # Get rankings and probabilities for those rows
    pred_proba_subset = oof_aligned.loc[true_mask]
    pred_target_subset = pred_target[true_mask]
    
    # Top-k rankings
    ranked = pred_proba_subset.rank(axis=1, method='min', ascending=False)
    
    top1_count = (ranked[target_class] == 1).sum()
    top2_count = (ranked[target_class] <= 2).sum()
    top3_count = (ranked[target_class] <= 3).sum()
    total = len(pred_proba_subset)
    
    # Predictions correctness
    predicted_classes = pred_proba_subset.idxmax(axis=1)
    is_correct = (predicted_classes == target_class)
    
    # Average probability
    if is_correct.sum() > 0:
        avg_prob_correct = pred_target_subset[is_correct].mean()
    else:
        avg_prob_correct = 0.0
    
    if (~is_correct).sum() > 0:
        avg_prob_incorrect = pred_target_subset[~is_correct].mean()
    else:
        avg_prob_incorrect = 0.0
    
    return {
        'total_true_instances': total,
        'top1_count': top1_count,
        'top1_pct': round(100 * top1_count / total, 1),
        'top2_count': top2_count,
        'top2_pct': round(100 * top2_count / total, 1),
        'top3_count': top3_count,
        'top3_pct': round(100 * top3_count / total, 1),
        'avg_prob_on_correct': round(avg_prob_correct, 4) if is_correct.sum() > 0 else None,
        'avg_prob_on_incorrect': round(avg_prob_incorrect, 4) if (~is_correct).sum() > 0 else None,
        'n_correct': is_correct.sum(),
        'n_incorrect': (~is_correct).sum(),
    }


def analyze_gull_overlap(solution_df, oof_aligned, target_class='Cormorants', classes_list=None):
    """
    For target class (default Cormorants):
    - Among false negatives for target, percentage predicted as Gulls
    For Gulls:
    - Among true Gull instances, how often target class appears in top 2 or top 3
    Returns: dict with diagnostics.
    """
    if classes_list is None:
        classes_list = list(oof_aligned.columns)
    
    result = {}
    
    # Target class false negatives → Gulls
    solution_target = solution_df[target_class]
    true_mask = solution_target == 1
    if true_mask.sum() > 0:
        predicted_classes = oof_aligned.loc[true_mask].idxmax(axis=1)
        true_predictions = predicted_classes == target_class
        false_negatives_mask = ~true_predictions
        
        if false_negatives_mask.sum() > 0:
            fn_as_gulls = (predicted_classes[false_negatives_mask] == 'Gulls').sum()
            fn_pct_as_gulls = round(100 * fn_as_gulls / false_negatives_mask.sum(), 1)
            result[f'{target_class}_fn_as_gulls_count'] = fn_as_gulls
            result[f'{target_class}_fn_as_gulls_pct'] = fn_pct_as_gulls
            result[f'{target_class}_total_fn'] = false_negatives_mask.sum()
    
    # Gulls true instances → target in top-2 or top-3
    solution_gull = solution_df['Gulls']
    gull_true_mask = solution_gull == 1
    if gull_true_mask.sum() > 0:
        pred_gull_subset = oof_aligned.loc[gull_true_mask]
        ranked = pred_gull_subset.rank(axis=1, method='min', ascending=False)
        
        top2_count = (ranked[target_class] <= 2).sum()
        top3_count = (ranked[target_class] <= 3).sum()
        
        top2_pct = round(100 * top2_count / gull_true_mask.sum(), 1)
        top3_pct = round(100 * top3_count / gull_true_mask.sum(), 1)
        
        result[f'{target_class}_in_gull_top2_count'] = top2_count
        result[f'{target_class}_in_gull_top2_pct'] = top2_pct
        result[f'{target_class}_in_gull_top3_count'] = top3_count
        result[f'{target_class}_in_gull_top3_pct'] = top3_pct
        result['gull_total_true_instances'] = gull_true_mask.sum()
    
    return result


def analyze_waders_posthoc(solution_df, oof_aligned, needed_columns):
    """
    Baseline-only post-hoc diagnostics for Waders:
    - Ranking gap: top-2/top-3 but not top-1 for true Waders rows.
    - Plausibility of small score adjustments via simple multipliers.
    - Impact on Waders AP and overall macro AP without retraining.
    """
    target = 'Waders'
    if target not in oof_aligned.columns or target not in solution_df.columns:
        return None, pd.DataFrame()

    true_mask = solution_df[target] == 1
    if true_mask.sum() == 0:
        return None, pd.DataFrame()

    ranked_true = oof_aligned.loc[true_mask].rank(axis=1, method='min', ascending=False)
    top1 = (ranked_true[target] == 1).sum()
    top2 = (ranked_true[target] <= 2).sum()
    top3 = (ranked_true[target] <= 3).sum()
    total = int(true_mask.sum())

    # These are the key ranking-gap diagnostics requested.
    top2_not_top1 = top2 - top1
    top3_not_top1 = top3 - top1

    summary = {
        'total_true_waders': total,
        'top1_count': int(top1),
        'top2_count': int(top2),
        'top3_count': int(top3),
        'top2_not_top1_count': int(top2_not_top1),
        'top2_not_top1_pct': round(100 * top2_not_top1 / total, 1),
        'top3_not_top1_count': int(top3_not_top1),
        'top3_not_top1_pct': round(100 * top3_not_top1 / total, 1),
    }

    # Tiny post-hoc multiplier check: small Waders score adjustments only.
    base_waders_ap = average_precision_score(solution_df[target], oof_aligned[target])
    base_map = average_precision_score(
        solution_df[needed_columns],
        oof_aligned[needed_columns],
        average='macro'
    )

    rows = []
    for mult in [1.00, 1.05, 1.10, 1.15]:
        adjusted = oof_aligned.copy()
        adjusted[target] = adjusted[target] * mult

        w_ap = average_precision_score(solution_df[target], adjusted[target])
        m_ap = average_precision_score(
            solution_df[needed_columns],
            adjusted[needed_columns],
            average='macro'
        )
        rows.append({
            'waders_multiplier': mult,
            'waders_ap': round(w_ap, 4),
            'delta_waders_ap': round(w_ap - base_waders_ap, 4),
            'macro_map': round(m_ap, 4),
            'delta_macro_map': round(m_ap - base_map, 4),
        })

    return summary, pd.DataFrame(rows)


def analyze_waders_vs_gulls_boundary(solution_df, oof_aligned):
    """
    Focused diagnostics for Waders-vs-Gulls boundary behavior.

    Returns:
      - summary dict
      - true_waders_pred_gulls_top23_df: row-level cases where true=Waders, pred=Gulls, Waders in top2/top3
      - pred_gulls_waders_rank23_df: row-level cases where pred=Gulls, Waders rank is 2 or 3
      - pred_gulls_rank23_true_class_df: true-class composition for pred=Gulls with Waders rank 2/3
      - gap_quantiles_df: gap quantiles for key subsets
    """
    target = 'Waders'
    blocker = 'Gulls'
    if target not in oof_aligned.columns or blocker not in oof_aligned.columns:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pred_top1 = oof_aligned.idxmax(axis=1)
    true_class = solution_df.idxmax(axis=1)
    ranks = oof_aligned.rank(axis=1, method='min', ascending=False)

    waders_prob = oof_aligned[target]
    gulls_prob = oof_aligned[blocker]
    gap_gull_minus_waders = gulls_prob - waders_prob

    # A) true=Waders, pred=Gulls, and Waders in top2/top3
    mask_true_waders_pred_gulls = (true_class == target) & (pred_top1 == blocker)
    mask_true_waders_pred_gulls_top23 = mask_true_waders_pred_gulls & (ranks[target] <= 3)

    true_waders_pred_gulls_top23_df = pd.DataFrame({
        'true_class': true_class[mask_true_waders_pred_gulls_top23],
        'pred_top1': pred_top1[mask_true_waders_pred_gulls_top23],
        'waders_rank': ranks[target][mask_true_waders_pred_gulls_top23].astype(int),
        'gulls_prob': gulls_prob[mask_true_waders_pred_gulls_top23].round(6),
        'waders_prob': waders_prob[mask_true_waders_pred_gulls_top23].round(6),
        'gap_gull_minus_waders': gap_gull_minus_waders[mask_true_waders_pred_gulls_top23].round(6),
    })

    # B) pred=Gulls and Waders is rank2 or rank3
    mask_pred_gulls_waders_rank23 = (pred_top1 == blocker) & (ranks[target].isin([2, 3]))
    pred_gulls_waders_rank23_df = pd.DataFrame({
        'true_class': true_class[mask_pred_gulls_waders_rank23],
        'pred_top1': pred_top1[mask_pred_gulls_waders_rank23],
        'waders_rank': ranks[target][mask_pred_gulls_waders_rank23].astype(int),
        'gulls_prob': gulls_prob[mask_pred_gulls_waders_rank23].round(6),
        'waders_prob': waders_prob[mask_pred_gulls_waders_rank23].round(6),
        'gap_gull_minus_waders': gap_gull_minus_waders[mask_pred_gulls_waders_rank23].round(6),
    })

    if not pred_gulls_waders_rank23_df.empty:
        pred_gulls_rank23_true_class_df = (
            pred_gulls_waders_rank23_df['true_class']
            .value_counts()
            .rename_axis('true_class')
            .reset_index(name='count')
        )
        pred_gulls_rank23_true_class_df['pct'] = (
            100.0 * pred_gulls_rank23_true_class_df['count'] / len(pred_gulls_waders_rank23_df)
        ).round(1)
    else:
        pred_gulls_rank23_true_class_df = pd.DataFrame({'true_class': [], 'count': [], 'pct': []})

    def _quantiles(series):
        if len(series) == 0:
            return {'q10': np.nan, 'q25': np.nan, 'q50': np.nan, 'q75': np.nan, 'q90': np.nan}
        return {
            'q10': float(np.quantile(series, 0.10)),
            'q25': float(np.quantile(series, 0.25)),
            'q50': float(np.quantile(series, 0.50)),
            'q75': float(np.quantile(series, 0.75)),
            'q90': float(np.quantile(series, 0.90)),
        }

    q_a = _quantiles(true_waders_pred_gulls_top23_df['gap_gull_minus_waders'].values)
    q_b = _quantiles(pred_gulls_waders_rank23_df['gap_gull_minus_waders'].values)
    gap_quantiles_df = pd.DataFrame([
        {'subset': 'true_waders_pred_gulls_top23', **{k: round(v, 6) if pd.notna(v) else np.nan for k, v in q_a.items()}},
        {'subset': 'pred_gulls_waders_rank23', **{k: round(v, 6) if pd.notna(v) else np.nan for k, v in q_b.items()}},
    ])

    # Plausibility counters for narrow reranking rules on small margins.
    thresholds = [0.01, 0.02, 0.05, 0.10]
    summary = {
        'count_true_waders_pred_gulls': int(mask_true_waders_pred_gulls.sum()),
        'count_true_waders_pred_gulls_top23': int(mask_true_waders_pred_gulls_top23.sum()),
        'count_pred_gulls_waders_rank23': int(mask_pred_gulls_waders_rank23.sum()),
        'count_pred_gulls_waders_rank2': int(((pred_top1 == blocker) & (ranks[target] == 2)).sum()),
        'count_pred_gulls_waders_rank3': int(((pred_top1 == blocker) & (ranks[target] == 3)).sum()),
        'avg_gap_true_waders_pred_gulls_top23': round(
            float(true_waders_pred_gulls_top23_df['gap_gull_minus_waders'].mean()), 6
        ) if not true_waders_pred_gulls_top23_df.empty else np.nan,
        'avg_gap_pred_gulls_waders_rank23': round(
            float(pred_gulls_waders_rank23_df['gap_gull_minus_waders'].mean()), 6
        ) if not pred_gulls_waders_rank23_df.empty else np.nan,
    }

    for th in thresholds:
        key_a = f'count_true_waders_pred_gulls_top23_gap_le_{th:.2f}'
        key_b = f'count_pred_gulls_waders_rank23_gap_le_{th:.2f}'
        summary[key_a] = int((true_waders_pred_gulls_top23_df['gap_gull_minus_waders'] <= th).sum())
        summary[key_b] = int((pred_gulls_waders_rank23_df['gap_gull_minus_waders'] <= th).sum())

    if not pred_gulls_waders_rank23_df.empty:
        summary['pct_true_waders_within_pred_gulls_waders_rank23'] = round(
            100.0 * (pred_gulls_waders_rank23_df['true_class'] == target).sum() / len(pred_gulls_waders_rank23_df),
            1,
        )
    else:
        summary['pct_true_waders_within_pred_gulls_waders_rank23'] = np.nan

    return (
        summary,
        true_waders_pred_gulls_top23_df,
        pred_gulls_waders_rank23_df,
        pred_gulls_rank23_true_class_df,
        gap_quantiles_df,
    )


# ─── Generate diagnostics for Cormorants ───
print(f"\n{'='*70}")
print("WEAK-CLASS DIAGNOSTICS: CORMORANTS")
print('='*70)

target = 'Cormorants'
print(f"\n1. False-Negative Absorption (where true={target} but prediction is wrong):")
fn_absorption_cormorants = analyze_false_negatives(
    solution_df, oof_aligned, target, needed_columns
)
print(fn_absorption_cormorants.to_string(index=False))
fn_absorption_cormorants.to_csv(
    diag_dir / f'{target.lower()}_false_negative_absorption.csv',
    index=False
)
print(f"   → saved to diagnostics/{target.lower()}_false_negative_absorption.csv")

print(f"\n2. False-Positive Sources (where predicted={target} but true label is different):")
fp_sources_cormorants = analyze_false_positives(
    solution_df, oof_aligned, target, needed_columns
)
print(fp_sources_cormorants.to_string(index=False))
fp_sources_cormorants.to_csv(
    diag_dir / f'{target.lower()}_false_positive_sources.csv',
    index=False
)
print(f"   → saved to diagnostics/{target.lower()}_false_positive_sources.csv")

print(f"\n3. Top-K Probability Diagnostics:")
topk_cormorants = analyze_topk_probability(solution_df, oof_aligned, target)
for key, val in topk_cormorants.items():
    print(f"   {key:30s}: {val}")

print(f"\n4. Gull Overlap Check:")
gull_overlap_cormorants = analyze_gull_overlap(solution_df, oof_aligned, target, needed_columns)
for key, val in gull_overlap_cormorants.items():
    print(f"   {key:40s}: {val}")

# ─── Generate diagnostics for Waders ───
print(f"\n{'='*70}")
print("WEAK-CLASS DIAGNOSTICS: WADERS")
print('='*70)

target = 'Waders'
print(f"\n1. False-Negative Absorption (where true={target} but prediction is wrong):")
fn_absorption_waders = analyze_false_negatives(
    solution_df, oof_aligned, target, needed_columns
)
print(fn_absorption_waders.to_string(index=False))
fn_absorption_waders.to_csv(
    diag_dir / f'{target.lower()}_false_negative_absorption.csv',
    index=False
)
print(f"   → saved to diagnostics/{target.lower()}_false_negative_absorption.csv")

print(f"\n2. False-Positive Sources (where predicted={target} but true label is different):")
fp_sources_waders = analyze_false_positives(
    solution_df, oof_aligned, target, needed_columns
)
print(fp_sources_waders.to_string(index=False))
fp_sources_waders.to_csv(
    diag_dir / f'{target.lower()}_false_positive_sources.csv',
    index=False
)
print(f"   → saved to diagnostics/{target.lower()}_false_positive_sources.csv")

print(f"\n3. Top-K Probability Diagnostics:")
topk_waders = analyze_topk_probability(solution_df, oof_aligned, target)
for key, val in topk_waders.items():
    print(f"   {key:30s}: {val}")

print(f"\n4. Gull Overlap Check:")
gull_overlap_waders = analyze_gull_overlap(solution_df, oof_aligned, target, needed_columns)
for key, val in gull_overlap_waders.items():
    print(f"   {key:40s}: {val}")

print(f"\n5. Waders Post-Hoc Ranking/Calibration Check:")
waders_posthoc_summary, waders_posthoc_table = analyze_waders_posthoc(
    solution_df,
    oof_aligned,
    needed_columns,
)
if waders_posthoc_summary is not None:
    for key, val in waders_posthoc_summary.items():
        print(f"   {key:40s}: {val}")
    print("\n   Small Waders multiplier test (no retraining):")
    print(waders_posthoc_table.to_string(index=False))
    waders_posthoc_table.to_csv(
        diag_dir / 'waders_posthoc_multiplier_check.csv',
        index=False,
    )
    print("   -> saved to diagnostics/waders_posthoc_multiplier_check.csv")

print(f"\n6. Waders-vs-Gulls Boundary Diagnostics:")
(
    waders_gulls_summary,
    waders_gulls_truew_predg_top23_df,
    waders_gulls_predg_rank23_df,
    waders_gulls_predg_rank23_trueclass_df,
    waders_gulls_gap_quantiles_df,
) = analyze_waders_vs_gulls_boundary(solution_df, oof_aligned)

if waders_gulls_summary is not None:
    print("   A) true=Waders, pred=Gulls, Waders in top2/top3")
    for key, val in waders_gulls_summary.items():
        print(f"   {key:52s}: {val}")

    print("\n   B) true-class mix when pred=Gulls and Waders rank is 2 or 3:")
    if not waders_gulls_predg_rank23_trueclass_df.empty:
        print(waders_gulls_predg_rank23_trueclass_df.to_string(index=False))
    else:
        print("   (no rows)")

    print("\n   C) gap quantiles (gulls_prob - waders_prob):")
    print(waders_gulls_gap_quantiles_df.to_string(index=False))

    waders_gulls_truew_predg_top23_df.to_csv(
        diag_dir / 'waders_vs_gulls_true_waders_pred_gulls_top23_rows.csv',
        index=False,
    )
    waders_gulls_predg_rank23_df.to_csv(
        diag_dir / 'waders_vs_gulls_pred_gulls_waders_rank23_rows.csv',
        index=False,
    )
    waders_gulls_predg_rank23_trueclass_df.to_csv(
        diag_dir / 'waders_vs_gulls_pred_gulls_waders_rank23_true_class_mix.csv',
        index=False,
    )
    waders_gulls_gap_quantiles_df.to_csv(
        diag_dir / 'waders_vs_gulls_gap_quantiles.csv',
        index=False,
    )
    pd.DataFrame([waders_gulls_summary]).to_csv(
        diag_dir / 'waders_vs_gulls_summary.csv',
        index=False,
    )

    print("\n   -> saved to diagnostics/waders_vs_gulls_summary.csv")
    print("   -> saved to diagnostics/waders_vs_gulls_true_waders_pred_gulls_top23_rows.csv")
    print("   -> saved to diagnostics/waders_vs_gulls_pred_gulls_waders_rank23_rows.csv")
    print("   -> saved to diagnostics/waders_vs_gulls_pred_gulls_waders_rank23_true_class_mix.csv")
    print("   -> saved to diagnostics/waders_vs_gulls_gap_quantiles.csv")

print(f"\n{'='*70}")
print("Weak-class diagnostics complete. CSV files saved to ./diagnostics/")
print('='*70)

# ─────────────────────────────────────────────
# 10. Generate Submission
# ─────────────────────────────────────────────
submission_df = pd.DataFrame(
    test_preds,
    index=X_test.index,
    columns=classes
)
missing_submission_cols = [c for c in COMPETITION_CLASS_ORDER if c not in submission_df.columns]
if missing_submission_cols:
    raise KeyError(f"Missing submission columns: {missing_submission_cols}")
submission_df = submission_df[COMPETITION_CLASS_ORDER]
submission_df.index.name = 'track_id'
submission_df.to_csv(SUBMISSION_OUT)
print(f"\nSaved {SUBMISSION_OUT} ({len(submission_df)} rows)")
