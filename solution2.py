"""
AI Cup 2026 — Bird Radar Track Classification (Experimental Solution 2)
Trains a LightGBM model with group-aware CV using tide-enriched datasets,
and generates an experimental submission file.
"""

# ─────────────────────────────────────────────
# 1. Model Hyperparameters
# ─────────────────────────────────────────────
# Primary experiment defaults (best-known branch behavior)
DATASET_VARIANT = 'openmeteo_tide'  # one of: knmi, openmeteo, openmeteo_tide, all
TIDE_ABLATION = 'tide_all'  # one of: tide_all, openmeteo_only, tide_level, tide_level_rising
USE_ENSEMBLE = True
USE_TWO_STAGE = True

# Optional model controls
USE_FOCAL_LOSS = False
USE_PASSTHROUGH = False
USE_SMOTETOMEK = False

# Weak-class boost controls
BOOST_WEAK = 0.0
BOOST_CORMORANTS = 0.0
BOOST_WADERS = 0.0
BOOST_GEESE = 0.0

# Optional OOF-only weak-class probability reweighting
# This runs after model training and tunes small multipliers on OOF predictions.
USE_OOF_WEAK_REWEIGHT = False
WEAK_REWEIGHT_CLASSES = ['Cormorants', 'Waders', 'Geese']
WEAK_REWEIGHT_GRID = [0.9, 1.0, 1.1, 1.2, 1.3]

# Optional weak-class specialist blending
# Trains binary specialist heads for weak classes and blends with OOF-tuned alphas.
USE_WEAK_SPECIALIST_BLEND = False
WEAK_SPECIALIST_CLASSES = ['Cormorants', 'Waders']
WEAK_SPECIALIST_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
WEAK_SPECIALIST_POS_WEIGHT = 4.0
WEAK_SPECIALIST_N_ESTIMATORS = 700

# Optional confusion-set resolver (Experiment A)
# Applies a conservative second-stage resolver only on uncertain confusion subsets.
ENABLE_CONFUSION_RESOLVER = False
CONFUSION_TARGET_CLASSES = ['Cormorants', 'Waders']
CONFUSION_TOPK = 2
RESOLVER_MARGIN_THRESHOLD = 0.08
RESOLVER_ALPHA = 0.35
RESOLVER_MODEL_TYPE = 'logreg'

# Optional compact metadata interaction pack (Stage-1 features only)
ENABLE_METADATA_INTERACTION_PACK = True
METADATA_INTERACTION_PACK_NAME = 'pack_v1'
METADATA_INTERACTION_DROP = ['rising_night_interaction']  # ablation helper: explicit interaction names to exclude
METADATA_INTERACTION_FEATURE_SPECS = [
    ('tide_hour_sin', ['tide_water_level_cm_nap', 'hour_sin']),
    ('tide_hour_cos', ['tide_water_level_cm_nap', 'hour_cos']),
    ('tide_delta_hour_sin', ['tide_delta_10min', 'hour_sin']),
    ('tide_delta_hour_cos', ['tide_delta_10min', 'hour_cos']),
    ('headwind_tide_delta', ['headwind_component', 'tide_delta_10min']),
    ('gustiness_tide_delta', ['openmeteo_wind_gusts_10m_kmh', 'openmeteo_wind_speed_10m_kmh', 'tide_delta_10min']),
    ('rising_night_interaction', ['rising_tide_flag', 'is_daytime']),
    ('precip_tide_motion', ['openmeteo_precipitation_mm', 'tide_delta_10min']),
]

# Two-stage controls
GULL_THRESHOLD = 0.5
UNDERSAMPLE_GULLS = 0

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
from itertools import product
import numpy as np
import pandas as pd
from shapely import wkb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
# 3. Load Data
# ─────────────────────────────────────────────
DATASET_CONFIG = {
    'knmi': {
        'train': 'dataset/train_with_knmi_286.csv',
        'test': 'dataset/test_with_knmi_286.csv',
        'wind_speed_col': 'knmi_286_hourly_mean_wind_speed_mps',
        'wind_speed_obs_col': 'knmi_286_wind_speed_at_observation_mps',
        'wind_unit_factor': 1.0,  # already m/s
        'weather_features': [
            'knmi_286_wind_direction_degrees',
            'knmi_286_hourly_mean_wind_speed_mps',
            'knmi_286_wind_speed_at_observation_mps',
            'knmi_286_max_wind_gust_mps',
            'knmi_286_air_temperature_c',
            'knmi_286_dew_point_temperature_c',
            'knmi_286_sunshine_duration_hours',
            'knmi_286_global_radiation_j_cm2',
            'knmi_286_precipitation_duration_hours',
            'knmi_286_precipitation_amount_mm',
            'knmi_286_relative_humidity_percent',
            'knmi_286_weather_indicator_code',
            'knmi_286_wind_dir_sin',
            'knmi_286_wind_dir_cos',
            'knmi_286_wind_dir_variable',
        ],
    },
    'openmeteo': {
        'train': 'dataset/train_with_openmeteo.csv',
        'test': 'dataset/test_with_openmeteo.csv',
        'wind_speed_col': 'openmeteo_wind_speed_10m_kmh',
        'wind_speed_obs_col': 'openmeteo_wind_speed_10m_kmh',
        'wind_unit_factor': 1 / 3.6,  # km/h → m/s
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
    },
    'openmeteo_tide': {
        'train': 'dataset/train_with_openmeteo_tide.csv',
        'test': 'dataset/test_with_openmeteo_tide.csv',
        'wind_speed_col': 'openmeteo_wind_speed_10m_kmh',
        'wind_speed_obs_col': 'openmeteo_wind_speed_10m_kmh',
        'wind_unit_factor': 1 / 3.6,  # km/h → m/s
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
            'tide_water_level_cm_nap',
            'tide_delta_10min',
            'rising_tide_flag',
        ],
    },
    'all': {
        'train': 'dataset/train_with_all_weather.csv',
        'test': 'dataset/test_with_all_weather.csv',
        'wind_speed_col': 'knmi_286_hourly_mean_wind_speed_mps',
        'wind_speed_obs_col': 'knmi_286_wind_speed_at_observation_mps',
        'wind_unit_factor': 1.0,
        'weather_features': [
            # KNMI features
            'knmi_286_wind_direction_degrees',
            'knmi_286_hourly_mean_wind_speed_mps',
            'knmi_286_wind_speed_at_observation_mps',
            'knmi_286_max_wind_gust_mps',
            'knmi_286_air_temperature_c',
            'knmi_286_dew_point_temperature_c',
            'knmi_286_sunshine_duration_hours',
            'knmi_286_global_radiation_j_cm2',
            'knmi_286_precipitation_duration_hours',
            'knmi_286_precipitation_amount_mm',
            'knmi_286_relative_humidity_percent',
            'knmi_286_weather_indicator_code',
            'knmi_286_wind_dir_sin',
            'knmi_286_wind_dir_cos',
            'knmi_286_wind_dir_variable',
            # OpenMeteo features
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
    },
}

ds = DATASET_CONFIG[DATASET_VARIANT]
print(f"Loading {DATASET_VARIANT} dataset...")
print(f"  train file: {ds['train']}")
print(f"  test file:  {ds['test']}")
if RUN_TUNE_CATBOOST and not USE_ENSEMBLE:
    print("Note: RUN_TUNE_CATBOOST requires USE_ENSEMBLE. CatBoost tuning will be skipped.")
if RUN_GRID_SEARCH_CATBOOST and not USE_ENSEMBLE:
    print("Note: RUN_GRID_SEARCH_CATBOOST requires USE_ENSEMBLE. CatBoost grid search will be skipped.")
if TIDE_ABLATION != 'tide_all' and DATASET_VARIANT != 'openmeteo_tide':
    print("Note: TIDE_ABLATION applies only when DATASET_VARIANT=openmeteo_tide. Ignoring ablation choice.")

for required_path in [ds['train'], ds['test']]:
    if not Path(required_path).exists():
        raise FileNotFoundError(
            f"Required dataset file not found: {required_path}. "
            "Run join_openmeteo_tide.py first for openmeteo_tide experiments."
        )

train_df = pd.read_csv(ds['train']).set_index("track_id")
test_df = pd.read_csv(ds['test']).set_index("track_id")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")
if DATASET_VARIANT == 'openmeteo_tide':
    print("Assumption: tide features were pre-merged using nearest merge_asof with ~10-15 min tolerance.")
    print("Assumption: tide columns (tide_water_level_cm_nap, tide_delta_10min, rising_tide_flag) are present.")

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
    else:
        rcs_mean = np.nanmean(rcs_arr)
        rcs_std = np.nanstd(rcs_arr)
        rcs_min = np.nanmin(rcs_arr)
        rcs_max = np.nanmax(rcs_arr)
        rcs_range = rcs_max - rcs_min

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

        feats['speed_cv'] = np.std(speeds) / (np.mean(speeds) + 1e-6)

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

    # Compute wind direction sin/cos for openmeteo (KNMI has them pre-computed)
    if DATASET_VARIANT in ('openmeteo', 'openmeteo_tide', 'all'):
        wd = df['openmeteo_wind_direction_10m_degrees']
        df['openmeteo_wind_dir_sin'] = np.sin(2 * np.pi * wd / 360)
        df['openmeteo_wind_dir_cos'] = np.cos(2 * np.pi * wd / 360)

    # Optional compact metadata interaction pack (row-wise only)
    if ENABLE_METADATA_INTERACTION_PACK:
        if {'tide_water_level_cm_nap', 'hour_sin'}.issubset(df.columns):
            df['tide_hour_sin'] = df['tide_water_level_cm_nap'] * df['hour_sin']
        if {'tide_water_level_cm_nap', 'hour_cos'}.issubset(df.columns):
            df['tide_hour_cos'] = df['tide_water_level_cm_nap'] * df['hour_cos']
        if {'tide_delta_10min', 'hour_sin'}.issubset(df.columns):
            df['tide_delta_hour_sin'] = df['tide_delta_10min'] * df['hour_sin']
        if {'tide_delta_10min', 'hour_cos'}.issubset(df.columns):
            df['tide_delta_hour_cos'] = df['tide_delta_10min'] * df['hour_cos']
        if {'headwind_component', 'tide_delta_10min'}.issubset(df.columns):
            df['headwind_tide_delta'] = df['headwind_component'] * df['tide_delta_10min']
        if {'openmeteo_wind_gusts_10m_kmh', 'openmeteo_wind_speed_10m_kmh', 'tide_delta_10min'}.issubset(df.columns):
            df['gustiness_tide_delta'] = (
                (df['openmeteo_wind_gusts_10m_kmh'] - df['openmeteo_wind_speed_10m_kmh'])
                * df['tide_delta_10min']
            )
        if {'rising_tide_flag', 'is_daytime'}.issubset(df.columns):
            df['rising_night_interaction'] = df['rising_tide_flag'] * (1 - df['is_daytime'])
        if {'openmeteo_precipitation_mm', 'tide_delta_10min'}.issubset(df.columns):
            df['precip_tide_motion'] = df['openmeteo_precipitation_mm'] * np.abs(df['tide_delta_10min'])

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

weather_features = ds['weather_features']

TIDE_FEATURES = ['tide_water_level_cm_nap', 'tide_delta_10min', 'rising_tide_flag']
if DATASET_VARIANT == 'openmeteo_tide':
    openmeteo_only_weather = [f for f in weather_features if f not in TIDE_FEATURES]
    tide_feature_sets = {
        'openmeteo_only': openmeteo_only_weather,
        'tide_level': openmeteo_only_weather + ['tide_water_level_cm_nap'],
        'tide_level_rising': openmeteo_only_weather + ['tide_water_level_cm_nap', 'rising_tide_flag'],
        'tide_all': openmeteo_only_weather + TIDE_FEATURES,
    }
    weather_features = tide_feature_sets[TIDE_ABLATION]
    print(f"Tide ablation mode: {TIDE_ABLATION}")
    print(f"Selected weather features: {len(weather_features)}")

metadata_interaction_features = []
if ENABLE_METADATA_INTERACTION_PACK:
    metadata_interaction_drop_set = set(METADATA_INTERACTION_DROP)
    if metadata_interaction_drop_set:
        print(
            f"Metadata interaction pack [{METADATA_INTERACTION_PACK_NAME}] drop list active: "
            f"{sorted(metadata_interaction_drop_set)}"
        )

    active_feature_pool = set(base_features + trajectory_feats + weather_features)
    for interaction_name, deps in METADATA_INTERACTION_FEATURE_SPECS:
        if interaction_name in metadata_interaction_drop_set:
            print(
                f"Metadata interaction pack [{METADATA_INTERACTION_PACK_NAME}] skipped {interaction_name}: "
                "explicitly dropped by METADATA_INTERACTION_DROP"
            )
            continue

        missing_in_data = [dep for dep in deps if dep not in train_df.columns or dep not in test_df.columns]
        if missing_in_data:
            print(
                f"Metadata interaction pack [{METADATA_INTERACTION_PACK_NAME}] skipped {interaction_name}: "
                f"missing columns in data {missing_in_data}"
            )
            continue

        inactive_deps = [dep for dep in deps if dep not in active_feature_pool]
        if inactive_deps:
            print(
                f"Metadata interaction pack [{METADATA_INTERACTION_PACK_NAME}] skipped {interaction_name}: "
                f"dependencies not active in current feature setup {inactive_deps}"
            )
            continue

        if interaction_name not in train_df.columns or interaction_name not in test_df.columns:
            print(
                f"Metadata interaction pack [{METADATA_INTERACTION_PACK_NAME}] skipped {interaction_name}: "
                "interaction not computed due to unavailable dependencies"
            )
            continue

        metadata_interaction_features.append(interaction_name)

    if metadata_interaction_features:
        print(
            f"Metadata interaction pack [{METADATA_INTERACTION_PACK_NAME}] active features "
            f"({len(metadata_interaction_features)}): {metadata_interaction_features}"
        )
    else:
        print(f"Metadata interaction pack [{METADATA_INTERACTION_PACK_NAME}] enabled but no features are active")
else:
    print("Metadata interaction pack disabled")

features = base_features + trajectory_feats + weather_features + metadata_interaction_features

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


def apply_probability_reweight(prob_values, class_labels, class_weights):
    """Apply class-wise multipliers to probabilities and renormalize rows."""
    adjusted = np.array(prob_values, dtype=float, copy=True)
    class_index = {cls: i for i, cls in enumerate(class_labels)}
    for cls, mult in class_weights.items():
        idx = class_index.get(cls)
        if idx is not None:
            adjusted[:, idx] *= mult

    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return adjusted / row_sums


def tune_weak_class_reweight(oof_pred_df, target_classes, candidate_multipliers):
    """Tune weak-class multipliers on OOF predictions to improve macro AP."""
    classes_local = list(oof_pred_df.columns)
    valid_targets = [c for c in target_classes if c in classes_local]
    baseline_map = compute_macro_map(oof_pred_df)

    if not valid_targets:
        return {}, baseline_map

    print("\nOOF weak-class reweight search")
    print(f"  target classes: {valid_targets}")
    print(f"  candidate multipliers: {candidate_multipliers}")
    print(f"  baseline mAP before reweight: {baseline_map:.4f}")

    best_map = baseline_map
    best_weights = {cls: 1.0 for cls in valid_targets}

    for values in product(candidate_multipliers, repeat=len(valid_targets)):
        weight_map = dict(zip(valid_targets, values))
        adjusted = apply_probability_reweight(oof_pred_df.values, classes_local, weight_map)
        adjusted_df = pd.DataFrame(adjusted, index=oof_pred_df.index, columns=oof_pred_df.columns)
        score = compute_macro_map(adjusted_df)
        if score > best_map + 1e-9:
            best_map = score
            best_weights = weight_map

    if best_map <= baseline_map + 1e-9:
        print("  no mAP gain found; skipping weak-class reweight")
        return {}, baseline_map

    print(f"  best reweighted mAP: {best_map:.4f}")
    print(f"  chosen multipliers: {best_weights}")
    return best_weights, best_map


def run_weak_specialist_cv(X, y, split, X_test, target_classes, imputer_template):
    """Train weak-class one-vs-rest specialists and return OOF/test probabilities."""
    available = set(np.unique(y))
    valid_targets = [c for c in target_classes if c in available]
    if not valid_targets:
        return pd.DataFrame(index=X.index), pd.DataFrame(index=X_test.index)

    print("\nWeak-class specialist CV")
    print(f"  target classes: {valid_targets}")

    specialist_oof = pd.DataFrame(0.0, index=X.index, columns=valid_targets)
    specialist_test = pd.DataFrame(0.0, index=X_test.index, columns=valid_targets)

    for cls in valid_targets:
        cls_oof = np.zeros(len(X), dtype=float)
        cls_test = np.zeros(len(X_test), dtype=float)
        fold_binary_ap = []

        for i, (train_idx, val_idx) in enumerate(split):
            y_train_bin = (y.iloc[train_idx] == cls).astype(int)
            y_val_bin = (y.iloc[val_idx] == cls).astype(int)

            # Safety fallback in case a fold has only one class.
            if y_train_bin.nunique() < 2:
                prior = float(y_train_bin.mean())
                val_prob = np.full(len(val_idx), prior, dtype=float)
                test_prob = np.full(len(X_test), prior, dtype=float)
            else:
                specialist_model = LGBMClassifier(
                    n_estimators=WEAK_SPECIALIST_N_ESTIMATORS,
                    learning_rate=LEARNING_RATE,
                    num_leaves=max(31, NUM_LEAVES // 2),
                    min_child_samples=max(10, MIN_CHILD_SAMPLES),
                    subsample=SUBSAMPLE,
                    colsample_bytree=COLSAMPLE_BYTREE,
                    class_weight={0: 1.0, 1: WEAK_SPECIALIST_POS_WEIGHT},
                    random_state=RANDOM_STATE + i,
                    device_type='cpu',
                    n_jobs=-1,
                    verbose=-1,
                )
                specialist_pipe = ImbPipeline([
                    ('imputer', clone(imputer_template)),
                    ('oversampler', 'passthrough'),
                    ('model', specialist_model),
                ])
                specialist_pipe.fit(X.iloc[train_idx], y_train_bin)
                val_prob = specialist_pipe.predict_proba(X.iloc[val_idx])[:, 1]
                test_prob = specialist_pipe.predict_proba(X_test)[:, 1]

            cls_oof[val_idx] = val_prob
            cls_test += test_prob

            if y_val_bin.sum() > 0:
                fold_binary_ap.append(average_precision_score(y_val_bin, val_prob))

        cls_test /= len(split)
        specialist_oof[cls] = cls_oof
        specialist_test[cls] = cls_test

        if fold_binary_ap:
            print(f"  {cls:20s}: mean binary AP {np.mean(fold_binary_ap):.4f}")
        else:
            print(f"  {cls:20s}: no positive validation folds")

    return specialist_oof, specialist_test


def apply_specialist_blend(prob_values, class_labels, specialist_values, blend_weights):
    """Blend specialist class probabilities into multiclass predictions and renormalize."""
    adjusted = np.array(prob_values, dtype=float, copy=True)
    class_index = {cls: i for i, cls in enumerate(class_labels)}

    for cls, alpha in blend_weights.items():
        idx = class_index.get(cls)
        if idx is None:
            continue
        if cls not in specialist_values.columns:
            continue
        alpha = float(np.clip(alpha, 0.0, 1.0))
        adjusted[:, idx] = (1.0 - alpha) * adjusted[:, idx] + alpha * specialist_values[cls].values

    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return adjusted / row_sums


def tune_specialist_blend(oof_pred_df, specialist_oof_df, target_classes, alpha_grid):
    """Tune specialist blend weights on OOF predictions for macro AP."""
    valid_targets = [
        c for c in target_classes
        if c in oof_pred_df.columns and c in specialist_oof_df.columns
    ]
    baseline_map = compute_macro_map(oof_pred_df)

    if not valid_targets:
        return {}, baseline_map

    print("\nOOF specialist blend search")
    print(f"  target classes: {valid_targets}")
    print(f"  alpha grid: {alpha_grid}")
    print(f"  baseline mAP before specialist blend: {baseline_map:.4f}")

    best_map = baseline_map
    best_weights = {cls: 0.0 for cls in valid_targets}

    for values in product(alpha_grid, repeat=len(valid_targets)):
        alpha_map = dict(zip(valid_targets, values))
        blended = apply_specialist_blend(
            oof_pred_df.values,
            list(oof_pred_df.columns),
            specialist_oof_df,
            alpha_map,
        )
        blended_df = pd.DataFrame(blended, index=oof_pred_df.index, columns=oof_pred_df.columns)
        score = compute_macro_map(blended_df)
        if score > best_map + 1e-9:
            best_map = score
            best_weights = alpha_map

    if best_map <= baseline_map + 1e-9:
        print("  no mAP gain found; skipping specialist blend")
        return {}, baseline_map

    print(f"  best specialist-blended mAP: {best_map:.4f}")
    print(f"  chosen blend weights: {best_weights}")
    return best_weights, best_map


def build_confusion_sets_from_oof(oof_pred_df, y_true, target_classes, topk=2):
    """Build small confusion sets from false-negative absorption on OOF predictions."""
    confusion_sets = {}
    for weak_cls in target_classes:
        if weak_cls not in oof_pred_df.columns:
            continue

        weak_rows = oof_pred_df.loc[y_true == weak_cls]
        if weak_rows.empty:
            continue

        pred_top1 = weak_rows.idxmax(axis=1)
        fn_rows = weak_rows.loc[pred_top1 != weak_cls]
        if fn_rows.empty:
            fn_rows = weak_rows

        absorption = (
            fn_rows.drop(columns=[weak_cls], errors='ignore')
            .sum(axis=0)
            .sort_values(ascending=False)
        )
        top_confusions = [c for c in absorption.index if c != weak_cls][:topk]
        set_classes = [weak_cls] + top_confusions

        if len(set_classes) >= 2:
            confusion_sets[weak_cls] = set_classes

    return confusion_sets


def compute_trigger_mask(prob_values, class_labels, confusion_set, margin_threshold):
    """Trigger when top-2 classes are both inside set and top1-top2 margin is small."""
    n_rows = len(prob_values)
    if n_rows == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    top2_idx = np.argpartition(prob_values, -2, axis=1)[:, -2:]
    top2_vals = np.take_along_axis(prob_values, top2_idx, axis=1)
    order = np.argsort(top2_vals, axis=1)
    top1_idx = top2_idx[np.arange(n_rows), order[:, 1]]
    top2_idx_ordered = top2_idx[np.arange(n_rows), order[:, 0]]

    top1_vals = prob_values[np.arange(n_rows), top1_idx]
    top2_vals = prob_values[np.arange(n_rows), top2_idx_ordered]
    margins = top1_vals - top2_vals

    class_to_idx = {cls: i for i, cls in enumerate(class_labels)}
    set_indices = [class_to_idx[c] for c in confusion_set if c in class_to_idx]
    if not set_indices:
        return np.zeros(n_rows, dtype=bool), margins

    trigger_mask = (
        np.isin(top1_idx, set_indices)
        & np.isin(top2_idx_ordered, set_indices)
        & (margins < margin_threshold)
    )
    return trigger_mask, margins


def apply_confusion_resolver_blend(prob_values, class_labels, confusion_set, resolver_probs, trigger_mask, alpha):
    """Preserve set mass and partially blend resolver distribution inside confusion set."""
    adjusted = np.array(prob_values, dtype=float, copy=True)
    if adjusted.shape[0] == 0 or not np.any(trigger_mask):
        return adjusted

    class_to_idx = {cls: i for i, cls in enumerate(class_labels)}
    set_indices = [class_to_idx[c] for c in confusion_set if c in class_to_idx]
    if len(set_indices) < 2:
        return adjusted

    trigger_rows = np.where(trigger_mask)[0]
    base_set = adjusted[np.ix_(trigger_rows, set_indices)]
    set_mass = base_set.sum(axis=1, keepdims=True)

    resolver_set = np.clip(resolver_probs[trigger_rows], 1e-12, None)
    resolver_set = resolver_set / resolver_set.sum(axis=1, keepdims=True)
    target_set = resolver_set * set_mass

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended_set = (1.0 - alpha) * base_set + alpha * target_set
    adjusted[np.ix_(trigger_rows, set_indices)] = blended_set
    return adjusted


def train_confusion_resolver_cv(
    X,
    y,
    split,
    X_test,
    oof_pred_df,
    test_pred_values,
    confusion_sets,
    class_labels,
    imputer_template,
    model_type='logreg',
):
    """Train one resolver per confusion set and return OOF/test set-wise probabilities."""
    if model_type != 'logreg':
        raise ValueError(f"Unsupported resolver_model_type: {model_type}")

    class_labels = list(class_labels)
    stage1_oof = oof_pred_df[class_labels].values
    stage1_test = np.array(test_pred_values, dtype=float, copy=True)

    # Stage-1 uncertainty signals used by resolver features.
    oof_sorted = np.sort(stage1_oof, axis=1)
    test_sorted = np.sort(stage1_test, axis=1)
    oof_margin = oof_sorted[:, -1] - oof_sorted[:, -2]
    test_margin = test_sorted[:, -1] - test_sorted[:, -2]
    oof_entropy = -(stage1_oof * np.log(np.clip(stage1_oof, 1e-12, 1.0))).sum(axis=1)
    test_entropy = -(stage1_test * np.log(np.clip(stage1_test, 1e-12, 1.0))).sum(axis=1)

    resolver_outputs = {}

    for weak_cls, set_classes in confusion_sets.items():
        set_indices = [class_labels.index(c) for c in set_classes if c in class_labels]
        if len(set_indices) < 2:
            continue

        label_map = {cls: i for i, cls in enumerate(set_classes)}
        y_in_set = y.isin(set_classes)
        if y_in_set.sum() < max(20, len(set_classes) * 6):
            print(f"  resolver [{weak_cls}] skipped: too few in-set samples")
            continue

        set_oof_stage1 = stage1_oof[:, set_indices]
        set_test_stage1 = stage1_test[:, set_indices]

        # Neutral fallback: if a fold cannot train a resolver, keep set distribution close to Stage-1.
        set_oof_pred = np.clip(set_oof_stage1, 1e-12, None)
        oof_set_sum = set_oof_pred.sum(axis=1, keepdims=True)
        bad_oof_sum = (oof_set_sum[:, 0] <= 0)
        if np.any(bad_oof_sum):
            set_oof_pred[bad_oof_sum] = 1.0 / len(set_classes)
            oof_set_sum = set_oof_pred.sum(axis=1, keepdims=True)
        set_oof_pred = set_oof_pred / oof_set_sum

        set_test_pred_accum = np.zeros((len(X_test), len(set_classes)), dtype=float)
        valid_folds = 0

        for i, (train_idx, val_idx) in enumerate(split):
            train_idx = np.asarray(train_idx)
            val_idx = np.asarray(val_idx)

            # Fold-safe preprocessing to avoid validation-fold leakage.
            fold_preprocessor = clone(imputer_template)
            X_train_fold_base = fold_preprocessor.fit_transform(X.iloc[train_idx], y.iloc[train_idx])
            X_val_fold_base = fold_preprocessor.transform(X.iloc[val_idx])
            X_test_fold_base = fold_preprocessor.transform(X_test)

            if hasattr(X_train_fold_base, 'toarray'):
                X_train_fold_base = X_train_fold_base.toarray()
            if hasattr(X_val_fold_base, 'toarray'):
                X_val_fold_base = X_val_fold_base.toarray()
            if hasattr(X_test_fold_base, 'toarray'):
                X_test_fold_base = X_test_fold_base.toarray()

            train_set_mask = y.iloc[train_idx].isin(set_classes).values
            if train_set_mask.sum() < len(set_classes) * 2:
                continue

            y_train_set = y.iloc[train_idx][train_set_mask].map(label_map).to_numpy()
            prior = np.bincount(y_train_set, minlength=len(set_classes)).astype(float)
            if prior.sum() <= 0:
                prior = np.ones(len(set_classes), dtype=float)
            prior = prior / prior.sum()

            X_train_set = np.hstack([
                X_train_fold_base[train_set_mask],
                set_oof_stage1[train_idx][train_set_mask],
                oof_margin[train_idx][train_set_mask].reshape(-1, 1),
                oof_entropy[train_idx][train_set_mask].reshape(-1, 1),
            ])
            X_val_all = np.hstack([
                X_val_fold_base,
                set_oof_stage1[val_idx],
                oof_margin[val_idx].reshape(-1, 1),
                oof_entropy[val_idx].reshape(-1, 1),
            ])
            X_test_all = np.hstack([
                X_test_fold_base,
                set_test_stage1,
                test_margin.reshape(-1, 1),
                test_entropy.reshape(-1, 1),
            ])

            if np.unique(y_train_set).size < 2:
                val_local = np.tile(prior, (len(val_idx), 1))
                test_local = np.tile(prior, (len(X_test), 1))
            else:
                model = LogisticRegression(
                    C=0.5,
                    penalty='l2',
                    solver='lbfgs',
                    multi_class='multinomial',
                    class_weight='balanced',
                    max_iter=500,
                    random_state=RANDOM_STATE + i,
                )
                model.fit(X_train_set, y_train_set)

                val_raw = model.predict_proba(X_val_all)
                test_raw = model.predict_proba(X_test_all)
                val_local = np.zeros((len(val_idx), len(set_classes)), dtype=float)
                test_local = np.zeros((len(X_test), len(set_classes)), dtype=float)
                val_local[:, model.classes_.astype(int)] = val_raw
                test_local[:, model.classes_.astype(int)] = test_raw

                val_sum = val_local.sum(axis=1, keepdims=True)
                bad_val = (val_sum[:, 0] <= 0)
                if np.any(bad_val):
                    val_local[bad_val] = prior
                    val_sum = val_local.sum(axis=1, keepdims=True)
                val_local = val_local / val_sum

                test_sum = test_local.sum(axis=1, keepdims=True)
                bad_test = (test_sum[:, 0] <= 0)
                if np.any(bad_test):
                    test_local[bad_test] = prior
                    test_sum = test_local.sum(axis=1, keepdims=True)
                test_local = test_local / test_sum

            set_oof_pred[val_idx] = val_local
            set_test_pred_accum += test_local
            valid_folds += 1

        if valid_folds == 0:
            print(f"  resolver [{weak_cls}] skipped: no valid folds")
            continue

        set_test_pred = set_test_pred_accum / valid_folds
        resolver_outputs[weak_cls] = {
            'classes': set_classes,
            'oof_proba': set_oof_pred,
            'test_proba': set_test_pred,
        }

    return resolver_outputs


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
# 8e. Optional Confusion-Set Resolver (Experiment A)
# ─────────────────────────────────────────────
if ENABLE_CONFUSION_RESOLVER:
    print("\n" + "=" * 50)
    print("CONFUSION-SET RESOLVER")
    print("=" * 50)

    resolver_baseline_map = compute_macro_map(oof_preds)
    resolver_baseline_ap = {
        cls: compute_class_ap(oof_preds, cls)
        for cls in CONFUSION_TARGET_CLASSES
        if cls in oof_preds.columns
    }

    confusion_sets = build_confusion_sets_from_oof(
        oof_preds,
        y,
        CONFUSION_TARGET_CLASSES,
        topk=CONFUSION_TOPK,
    )

    if not confusion_sets:
        print("  no confusion sets found from OOF; skipping resolver")
    else:
        print("  confusion sets from OOF false-negative absorption:")
        for weak_cls, set_classes in confusion_sets.items():
            print(f"    {weak_cls}: {set_classes}")

        resolver_outputs = train_confusion_resolver_cv(
            X,
            y,
            split,
            X_test,
            oof_preds,
            test_preds,
            confusion_sets,
            classes,
            imputer,
            model_type=RESOLVER_MODEL_TYPE,
        )

        if not resolver_outputs:
            print("  resolver produced no valid models; skipping blend")
        else:
            stage1_oof_values = oof_preds.values.copy()
            stage1_test_values = np.array(test_preds, dtype=float, copy=True)
            blended_oof_values = stage1_oof_values.copy()
            blended_test_values = stage1_test_values.copy()

            claimed_oof_triggers = np.zeros(len(oof_preds), dtype=bool)
            claimed_test_triggers = np.zeros(len(stage1_test_values), dtype=bool)

            for weak_cls, payload in resolver_outputs.items():
                set_classes = payload['classes']

                oof_trigger_raw, _ = compute_trigger_mask(
                    stage1_oof_values,
                    list(classes),
                    set_classes,
                    RESOLVER_MARGIN_THRESHOLD,
                )
                test_trigger_raw, _ = compute_trigger_mask(
                    stage1_test_values,
                    list(classes),
                    set_classes,
                    RESOLVER_MARGIN_THRESHOLD,
                )

                # Deterministic guardrail: each row can be handled by at most one resolver set.
                oof_trigger_mask = oof_trigger_raw & ~claimed_oof_triggers
                test_trigger_mask = test_trigger_raw & ~claimed_test_triggers
                claimed_oof_triggers |= oof_trigger_mask
                claimed_test_triggers |= test_trigger_mask

                oof_raw_count = int(oof_trigger_raw.sum())
                test_raw_count = int(test_trigger_raw.sum())
                oof_trigger_count = int(oof_trigger_mask.sum())
                test_trigger_count = int(test_trigger_mask.sum())
                print(
                    f"  resolver set [{weak_cls}] trigger count: "
                    f"OOF applied {oof_trigger_count}/{len(oof_trigger_mask)} "
                    f"({oof_trigger_count / max(1, len(oof_trigger_mask)):.2%}), "
                    f"test applied {test_trigger_count}/{len(test_trigger_mask)} "
                    f"({test_trigger_count / max(1, len(test_trigger_mask)):.2%})"
                )
                print(
                    f"    raw trigger count before overlap guard: "
                    f"OOF {oof_raw_count}, test {test_raw_count}"
                )

                blended_oof_values = apply_confusion_resolver_blend(
                    blended_oof_values,
                    list(classes),
                    set_classes,
                    payload['oof_proba'],
                    oof_trigger_mask,
                    RESOLVER_ALPHA,
                )
                blended_test_values = apply_confusion_resolver_blend(
                    blended_test_values,
                    list(classes),
                    set_classes,
                    payload['test_proba'],
                    test_trigger_mask,
                    RESOLVER_ALPHA,
                )

            total_trigger_count = int(claimed_oof_triggers.sum())
            print(
                f"  overall trigger count: {total_trigger_count}/{len(claimed_oof_triggers)} "
                f"({total_trigger_count / max(1, len(claimed_oof_triggers)):.2%})"
            )

            oof_row_sums = blended_oof_values.sum(axis=1)
            test_row_sums = blended_test_values.sum(axis=1)
            print(
                "  probability-sum check "
                f"OOF min/max: {oof_row_sums.min():.6f}/{oof_row_sums.max():.6f}, "
                f"test min/max: {test_row_sums.min():.6f}/{test_row_sums.max():.6f}"
            )
            print(
                "  probability-min check "
                f"OOF min: {blended_oof_values.min():.6f}, "
                f"test min: {blended_test_values.min():.6f}"
            )

            oof_preds = pd.DataFrame(blended_oof_values, index=oof_preds.index, columns=oof_preds.columns)
            test_preds = blended_test_values

            resolver_new_map = compute_macro_map(oof_preds)
            resolver_delta = resolver_new_map - resolver_baseline_map
            print(
                f"  CV mAP delta after resolver: {resolver_new_map:.4f} "
                f"({resolver_delta:+.4f} vs stage-1 baseline {resolver_baseline_map:.4f})"
            )

            for cls in CONFUSION_TARGET_CLASSES:
                if cls in resolver_baseline_ap and not np.isnan(resolver_baseline_ap[cls]):
                    new_ap = compute_class_ap(oof_preds, cls)
                    print(
                        f"  AP delta {cls}: {new_ap:.4f} "
                        f"({new_ap - resolver_baseline_ap[cls]:+.4f})"
                    )

# ─────────────────────────────────────────────
# 8f. Optional Weak-Class Specialist Blending
# ─────────────────────────────────────────────
if USE_WEAK_SPECIALIST_BLEND:
    specialist_oof, specialist_test = run_weak_specialist_cv(
        X,
        y,
        split,
        X_test,
        WEAK_SPECIALIST_CLASSES,
        imputer,
    )
    specialist_weights, _ = tune_specialist_blend(
        oof_preds,
        specialist_oof,
        WEAK_SPECIALIST_CLASSES,
        WEAK_SPECIALIST_ALPHA_GRID,
    )
    if specialist_weights:
        oof_specialist = apply_specialist_blend(
            oof_preds.values,
            list(oof_preds.columns),
            specialist_oof,
            specialist_weights,
        )
        test_specialist = apply_specialist_blend(
            test_preds,
            list(classes),
            specialist_test,
            specialist_weights,
        )
        oof_preds = pd.DataFrame(oof_specialist, index=oof_preds.index, columns=oof_preds.columns)
        test_preds = test_specialist
        print("  Applied weak-class specialist blending to OOF and test predictions")

# ─────────────────────────────────────────────
# 8g. Optional OOF Weak-Class Reweighting
# ─────────────────────────────────────────────
if USE_OOF_WEAK_REWEIGHT:
    weak_weights, _ = tune_weak_class_reweight(
        oof_preds,
        WEAK_REWEIGHT_CLASSES,
        WEAK_REWEIGHT_GRID,
    )
    if weak_weights:
        oof_adjusted = apply_probability_reweight(oof_preds.values, list(oof_preds.columns), weak_weights)
        test_adjusted = apply_probability_reweight(test_preds, list(classes), weak_weights)
        oof_preds = pd.DataFrame(oof_adjusted, index=oof_preds.index, columns=oof_preds.columns)
        test_preds = test_adjusted
        print("  Applied weak-class probability reweighting to OOF and test predictions")

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
