"""
AI Cup 2026 — Bird Radar Track Classification
Two-stage classifier: binary Gull detector + ensembled 8-class non-Gull classifier.
"""

import argparse
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
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, RandomOverSampler, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.model_selection import ParameterGrid
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import mlflow

# ─────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='AI Cup 2026 Bird Radar Classification')
parser.add_argument('--focal-loss', action='store_true',
                    help='Use multi-class focal loss instead of standard multiclass objective')
parser.add_argument('--passthrough', action='store_true',
                    help='Disable oversampling entirely')
parser.add_argument('--smotetomek', action='store_true',
                    help='Use SMOTETomek (SMOTENC + TomekLinks) instead of SMOTENC-only')
parser.add_argument('--oversampler', choices=['smotenc', 'borderline', 'adasyn', 'trajectory', 'adasyn+trajectory'],
                    default='smotenc',
                    help='Oversampling method: smotenc (default), borderline, adasyn, trajectory, adasyn+trajectory')
parser.add_argument('--boost-weak', type=float, default=3, metavar='MULT',
                    help='Default sample weight multiplier for weak classes (default: 3)')
parser.add_argument('--boost-cormorants', type=float, default=0, metavar='MULT',
                    help='Override boost multiplier for Cormorants (default: use --boost-weak)')
parser.add_argument('--boost-waders', type=float, default=0, metavar='MULT',
                    help='Override boost multiplier for Waders (default: use --boost-weak)')
parser.add_argument('--boost-geese', type=float, default=0, metavar='MULT',
                    help='Override boost multiplier for Geese (default: use --boost-weak)')
parser.add_argument('--grid-search', action='store_true',
                    help='Run grid search over hyperparameters (slow)')
parser.add_argument('--stage1-catboost', action='store_true',
                    help='Use CatBoost instead of LightGBM for Stage 1 binary Gull classifier')
parser.add_argument('--no-mlflow', action='store_true',
                    help='Disable MLflow experiment tracking')
parser.add_argument('--dataset', choices=['knmi', 'openmeteo', 'all'], default='openmeteo',
                    help='Weather dataset to use (default: openmeteo)')
parser.add_argument('--gull-threshold', type=float, default=0.5, metavar='T',
                    help='Stage-1 threshold for calling Gull (default: 0.5). '
                         'Higher values (e.g. 0.75) penalise Gull predictions.')
parser.add_argument('--undersample-gulls', type=int, default=0, metavar='N',
                    help='Undersample Gulls to N tracks before two-stage training '
                         '(e.g. --undersample-gulls 500). 0 = no undersampling.')
parser.add_argument('--jitter-scale', type=float, default=1.0, metavar='S',
                    help='Scale factor for trajectory jitter magnitudes (default: 1.0)')
parser.add_argument('--n-seeds', type=int, default=1, metavar='N',
                    help='Number of random seeds for seed averaging (default: 1)')
parser.add_argument('--pseudo-label', action='store_true',
                    help='Pseudo-labeling: train once, add high-confidence test predictions, retrain')
parser.add_argument('--pseudo-threshold', type=float, default=0.95, metavar='T',
                    help='Min probability for pseudo-label acceptance (default: 0.95)')
args = parser.parse_args()

# ─────────────────────────────────────────────
# 1. Dataset Configuration & Loading
# ─────────────────────────────────────────────
DATASET_CONFIG = {
    'knmi': {
        'train': 'dataset/train_with_knmi_286.csv',
        'test': 'dataset/test_with_knmi_286.csv',
        'wind_speed_col': 'knmi_286_hourly_mean_wind_speed_mps',
        'wind_speed_obs_col': 'knmi_286_wind_speed_at_observation_mps',
        'wind_dir_col': 'knmi_286_wind_direction_degrees',
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
        'wind_dir_col': 'openmeteo_wind_direction_10m_degrees',
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
    'all': {
        'train': 'dataset/train_with_all_weather.csv',
        'test': 'dataset/test_with_all_weather.csv',
        'wind_speed_col': 'knmi_286_hourly_mean_wind_speed_mps',
        'wind_speed_obs_col': 'knmi_286_wind_speed_at_observation_mps',
        'wind_dir_col': 'knmi_286_wind_direction_degrees',
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

ds = DATASET_CONFIG[args.dataset]
print(f"Loading {args.dataset} dataset...")
train_df = pd.read_csv(ds['train']).set_index("track_id")
test_df = pd.read_csv(ds['test']).set_index("track_id")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ─────────────────────────────────────────────
# 2. Trajectory Parsing & Feature Extraction
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


def augment_trajectory_coords(coords, times_list, rng, jitter_scale=1.0):
    """Augment trajectory coordinates with jitter, time warp, and rotation."""
    coords = np.array(coords)
    lons, lats = coords[:, 0].copy(), coords[:, 1].copy()
    alts = coords[:, 2].copy() if coords.shape[1] > 2 else None
    rcs = coords[:, 3].copy() if coords.shape[1] > 3 else None

    # 1. Spatial jitter (σ calibrated to dataset: ~3m for lon/lat, 2m alt, 0.3dB RCS)
    s = jitter_scale
    lons += rng.normal(0, 0.00003 * s, len(lons))
    lats += rng.normal(0, 0.00003 * s, len(lats))
    if alts is not None:
        alts += rng.normal(0, 2.0 * s, len(alts))
    if rcs is not None:
        rcs += rng.normal(0, 0.3 * s, len(rcs))

    # 2. Rotation — rotate displacement vectors by random angle
    angle = rng.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lon_c, lat_c = lons.mean(), lats.mean()
    dlon, dlat = lons - lon_c, lats - lat_c
    lons = lon_c + dlon * cos_a - dlat * sin_a
    lats = lat_c + dlon * sin_a + dlat * cos_a

    # 3. Time warp — scale times by ±5%
    new_times = times_list
    if times_list is not None and len(times_list) > 0:
        warp = rng.uniform(0.95, 1.05)
        new_times = [t * warp for t in times_list]

    new_coords = []
    for j in range(len(lons)):
        pt = [lons[j], lats[j]]
        if alts is not None:
            pt.append(alts[j])
        if rcs is not None:
            pt.append(rcs[j])
        new_coords.append(tuple(pt))

    return new_coords, new_times


def augment_weak_classes(train_df, weak_classes, multiplier, rng, jitter_scale=1.0):
    """Augment weak classes by creating trajectory-augmented copies before feature extraction."""
    from shapely.geometry import LineString
    aug_rows = []
    for _, row in train_df.iterrows():
        if row['bird_group'] not in weak_classes:
            continue
        for _ in range(int(multiplier) - 1):
            coords = parse_trajectory(row['trajectory'])
            if len(coords) < 2:
                continue
            times = row.get('trajectory_time', '')
            t_list = []
            if isinstance(times, str) and times.strip():
                try:
                    t_list = [float(x) for x in times.strip('[]').split(',')]
                except ValueError:
                    pass
            elif isinstance(times, (list, np.ndarray)):
                t_list = list(times)

            new_coords, new_times = augment_trajectory_coords(coords, t_list, rng, jitter_scale)

            # Shapely LineString supports max 3D — store as (lon, lat, alt),
            # encode RCS into alt dimension isn't viable, so use original WKB format
            # by building a 4D-aware WKB manually via struct
            import struct
            new_row = row.copy()
            # Build WKB for 4D LineString (EWKB with Z and M)
            n_pts = len(new_coords)
            # Use little-endian, geometry type = LineString with XYZM (0xC0000002)
            wkb_bytes = struct.pack('<BII', 1, 0xC0000002, n_pts)
            for pt in new_coords:
                wkb_bytes += struct.pack('<dddd', pt[0], pt[1],
                                        pt[2] if len(pt) > 2 else 0.0,
                                        pt[3] if len(pt) > 3 else 0.0)
            new_row['trajectory'] = wkb_bytes.hex()
            if new_times:
                new_row['trajectory_time'] = str(new_times)
            for col in ['airspeed', 'min_z', 'max_z']:
                if col in new_row and pd.notna(new_row[col]) and new_row[col] != 0:
                    new_row[col] *= (1 + rng.normal(0, 0.01))
            aug_rows.append(new_row)

    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        aug_df.index = [f"aug_{i}" for i in range(len(aug_df))]
        aug_df.index.name = train_df.index.name
        result = pd.concat([train_df, aug_df])
        print(f"  Trajectory augmentation: {len(train_df)} → {len(result)} samples "
              f"(+{len(aug_rows)} augmented from {weak_classes})")
        return result
    return train_df


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

    # Trajectory smoothing (rolling window=3) to reduce radar noise
    lons_s = pd.Series(lons).rolling(window=3, min_periods=1, center=True).mean().values
    lats_s = pd.Series(lats).rolling(window=3, min_periods=1, center=True).mean().values

    # Horizontal displacement (degrees → approx metres at ~53°N)
    dx = np.diff(lons_s) * 71000
    dy = np.diff(lats_s) * 111000
    step_dist = np.sqrt(dx**2 + dy**2)
    total_dist = step_dist.sum() if len(step_dist) > 0 else 0.0

    # Tortuosity (turning behaviour)
    bearings = np.arctan2(dy, dx)
    bearing_changes = np.abs(np.diff(bearings)) if len(bearings) > 1 else np.array([0.0])
    bearing_changes = np.minimum(bearing_changes, 2 * np.pi - bearing_changes)

    # Sharp turn ratio: fraction of turns > 45 degrees
    sharp_turn_ratio = (bearing_changes > np.pi / 4).mean() if len(bearing_changes) > 0 else 0.0

    # Curvature via cross product of consecutive displacement vectors
    vecs = np.column_stack([dx, dy])
    if len(vecs) > 1:
        curvatures = []
        for k in range(len(vecs) - 1):
            v1, v2 = vecs[k], vecs[k + 1]
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            curvatures.append(cross / norm)
        curvatures = np.array(curvatures)
    else:
        curvatures = np.array([0.0])

    # Straightness index: displacement / total path length
    displacement = np.sqrt(
        ((lons[-1] - lons[0]) * 71000) ** 2 +
        ((lats[-1] - lats[0]) * 111000) ** 2
    )
    straightness = displacement / (total_dist + 1e-6) if total_dist > 0 else 0.0

    # Sinuosity: total path / displacement (inverse of straightness, penalizes wound tracks)
    sinuosity = total_dist / (displacement + 1e-6)

    # Track heading (overall direction from start to end)
    track_heading_rad = np.arctan2(
        (lats[-1] - lats[0]) * 111000,
        (lons[-1] - lons[0]) * 71000
    )

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

    # RCS range
    rcs_range = np.nanmax(rcs) - np.nanmin(rcs)

    # (No extra RCS features — baseline stats sufficient for dataset size)

    feats = {
        'n_points':       n,
        'total_dist_m':   total_dist,
        'mean_step_m':    step_dist.mean() if len(step_dist) > 0 else 0.0,
        'std_step_m':     step_dist.std()  if len(step_dist) > 0 else 0.0,
        'lon_range':      max(lons) - min(lons),
        'lat_range':      max(lats) - min(lats),
        'alt_mean':       np.nanmean(alts),
        'alt_std':        np.nanstd(alts),
        'rcs_mean':       np.nanmean(rcs),
        'rcs_std':        np.nanstd(rcs),
        'rcs_min':        np.nanmin(rcs),
        'rcs_max':        np.nanmax(rcs),
        'rcs_range':      rcs_range,
        'tortuosity':     bearing_changes.mean() if len(bearing_changes) > 0 else 0.0,
        'tortuosity_max': bearing_changes.max()  if len(bearing_changes) > 0 else 0.0,
        'sharp_turn_ratio':   sharp_turn_ratio,
        'straightness':       straightness,
        'sinuosity':          sinuosity,
        'lon_mean':           np.mean(lons),
        'lat_mean':           np.mean(lats),
        'track_heading_rad':  track_heading_rad,
        'alt_climb_rate':     alt_climb_rate,
        'alt_descent_rate':   alt_descent_rate,
        'alt_variability':    alt_variability,
        'curvature_mean':     curvatures.mean(),
        'curvature_std':      curvatures.std() if len(curvatures) > 1 else 0.0,
        'log_path_length':    np.log1p(total_dist),
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
# 3. Feature Engineering
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
    if args.dataset in ('openmeteo', 'all'):
        wd = df['openmeteo_wind_direction_10m_degrees']
        df['openmeteo_wind_dir_sin'] = np.sin(2 * np.pi * wd / 360)
        df['openmeteo_wind_dir_cos'] = np.cos(2 * np.pi * wd / 360)

# Trajectory-level augmentation (before feature extraction)
if args.oversampler in ('trajectory', 'adasyn+trajectory'):
    weak_classes = ['Cormorants', 'Waders', 'Geese']
    aug_rng = np.random.RandomState(42)
    train_df = augment_weak_classes(train_df, weak_classes, args.boost_weak, aug_rng, args.jitter_scale)

# Trajectory features
print("Extracting trajectory features for train_df...")
train_df = train_df.join(train_df.apply(trajectory_features, axis=1))
print("Extracting trajectory features for test_df...")
test_df = test_df.join(test_df.apply(trajectory_features, axis=1))

# Post-trajectory interaction features (depend on trajectory-derived columns)
for df in [train_df, test_df]:
    # RCS-to-speed ratio: large slow blob (flock) vs fast large bird
    df['rcs_speed_ratio'] = df['rcs_mean'] / (df['airspeed'] + 1e-6)

    # Altitude-adjusted wind speed (Hellmann power law, exponent 0.143 for open terrain)
    wf = ds['wind_unit_factor']
    df['alt_adjusted_wind_speed'] = (
        df[ds['wind_speed_col']] * wf
        * np.power(np.maximum(df['alt_mean'], 10) / 10.0, 0.143)
    )

    # True tailwind & crosswind from track heading vs wind direction
    wind_dir_rad = np.deg2rad(df[ds['wind_dir_col']])
    heading = df['track_heading_rad']
    angle_diff = wind_dir_rad - heading
    df['true_tailwind_component'] = df['alt_adjusted_wind_speed'] * np.cos(angle_diff)
    df['true_crosswind_component'] = np.abs(df['alt_adjusted_wind_speed'] * np.sin(angle_diff))

# ─────────────────────────────────────────────
# 4. Feature List
# ─────────────────────────────────────────────
base_features = [
    'airspeed', 'min_z', 'max_z', 'duration_s', 'radar_bird_size',
    'hour', 'month', 'is_daytime',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'alt_range', 'airspeed_per_m',
    'headwind_component', 'airspeed_wind_ratio',
    'rcs_speed_ratio', 'alt_adjusted_wind_speed',
    'true_tailwind_component', 'true_crosswind_component',
]

trajectory_feats = [
    'n_points', 'total_dist_m', 'mean_step_m', 'std_step_m',
    'lon_range', 'lat_range', 'alt_mean', 'alt_std',
    'rcs_mean', 'rcs_std', 'rcs_min', 'rcs_max', 'rcs_range',
    'tortuosity', 'tortuosity_max', 'sharp_turn_ratio',
    'straightness', 'sinuosity',
    'lon_mean', 'lat_mean', 'track_heading_rad',
    'alt_climb_rate', 'alt_descent_rate', 'alt_variability',
    'curvature_mean', 'curvature_std', 'log_path_length',
    'speed_mean', 'speed_std', 'speed_max', 'speed_cv',
    'accel_mean', 'accel_std',
]

weather_features = ds['weather_features']

features = base_features + trajectory_feats + weather_features

X = train_df[features]
X_test = test_df[features]
y = train_df['bird_group']

print(f"Feature matrix: X={X.shape}, X_test={X_test.shape}, classes={y.nunique()}")

# ─────────────────────────────────────────────
# 5. Focal Loss (optional)
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

# ─────────────────────────────────────────────
# 6. Model & Pipeline
# ─────────────────────────────────────────────
numeric_features = [f for f in features if f != 'radar_bird_size']
categorical_features = ['radar_bird_size']
cat_indices = [len(numeric_features) + i for i in range(len(categorical_features))]

imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='median'), numeric_features),
    ('cat_imputer', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), categorical_features)
])

# Oversampler selection
if args.passthrough or args.oversampler == 'trajectory':
    oversampler = 'passthrough'
    if args.passthrough:
        print("Oversampler: passthrough (no oversampling)")
    else:
        print("Oversampler: trajectory augmentation (applied before feature extraction)")
elif args.oversampler == 'adasyn+trajectory':
    oversampler = ADASYN(random_state=42)
    print("Oversampler: ADASYN + trajectory augmentation")
elif args.smotetomek:
    oversampler = SMOTETomek(
        smote=SMOTENC(categorical_features=cat_indices, random_state=42),
        random_state=42
    )
    print("Oversampler: SMOTETomek (SMOTENC + TomekLinks)")
elif args.oversampler == 'borderline':
    oversampler = BorderlineSMOTE(random_state=42, kind='borderline-1')
    print("Oversampler: BorderlineSMOTE")
elif args.oversampler == 'adasyn':
    oversampler = ADASYN(random_state=42)
    print("Oversampler: ADASYN")
else:
    oversampler = SMOTENC(categorical_features=cat_indices, random_state=42)
    print("Oversampler: SMOTENC (default)")

lgb_params = dict(
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    device='gpu',
    verbose=-1,
)

if args.focal_loss:
    print("Using focal loss objective")
    lgb_params['objective'] = focal_loss_lgb

lgb_model = LGBMClassifier(**lgb_params)

# Per-class boost multipliers (fall back to --boost-weak if per-class not set)
CLASS_BOOST = {}
for cls, cli_val in [('Cormorants', args.boost_cormorants),
                     ('Waders', args.boost_waders),
                     ('Geese', args.boost_geese)]:
    mult = cli_val if cli_val > 0 else args.boost_weak
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
# 7. Helper: apply boost-weak row duplication
# ─────────────────────────────────────────────
def apply_boost(X_arr, y_arr, class_boost):
    """Duplicate rows for weak classes. Returns (X_boosted, y_boosted)."""
    if not class_boost:
        return X_arr, y_arr
    X_np = X_arr if isinstance(X_arr, np.ndarray) else np.asarray(X_arr)
    y_np = y_arr if isinstance(y_arr, np.ndarray) else np.asarray(y_arr)
    extra_X, extra_y = [], []
    for cls, mult in class_boost.items():
        repeats = int(mult) - 1
        if repeats <= 0:
            continue
        cls_mask = (y_np == cls)
        X_cls = X_np[cls_mask]
        y_cls = y_np[cls_mask]
        if len(X_cls) > 0:
            extra_X.extend([X_cls] * repeats)
            extra_y.extend([y_cls] * repeats)
    if extra_X:
        X_np = np.vstack([X_np] + extra_X)
        y_np = np.concatenate([y_np] + extra_y)
    return X_np, y_np

# ─────────────────────────────────────────────
# 8. Two-Stage Cross-Validation + Training
# ─────────────────────────────────────────────
n_splits = 10
groups = train_df['primary_observation_id']
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
split = list(cv.split(X, y, groups))

classes = np.sort(y.unique())
needed_columns = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds",
]

# CatBoost setup (used in Stage 2 always, and Stage 1 with --stage1-catboost)
cb_cat_idx = [len(numeric_features) + i for i in range(len(categorical_features))]
cb_params = dict(
    iterations=2500, learning_rate=0.03, depth=8, l2_leaf_reg=3,
    auto_class_weights='Balanced', random_seed=42, task_type='GPU', verbose=0,
)
cb_imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='median'), numeric_features),
    ('cat_imputer', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
    ]), categorical_features)
])

# Stage 1 model setup
if args.stage1_catboost:
    print("Stage 1 (Gull binary): CatBoost")
else:
    print("Stage 1 (Gull binary): LightGBM")
print("Stage 2 (non-Gull 8-class): LightGBM + CatBoost + XGBoost ensemble")

gull_thresh = args.gull_threshold
us_gulls = args.undersample_gulls
if gull_thresh != 0.5:
    print(f"  Gull threshold: {gull_thresh}")
if us_gulls > 0:
    print(f"  Undersampling Gulls to {us_gulls} tracks per fold")

# MLflow setup
use_mlflow = not args.no_mlflow
if use_mlflow:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("bird-radar-classification")
    run = mlflow.start_run()
    mlflow.log_params({
        'dataset': args.dataset,
        'stage1_model': 'catboost' if args.stage1_catboost else 'lightgbm',
        'gull_threshold': gull_thresh,
        'undersample_gulls': us_gulls,
        'boost_weak': args.boost_weak,
        'n_splits': n_splits,
        'n_features': X.shape[1],
        'focal_loss': args.focal_loss,
        'oversampler': 'smotetomek' if args.smotetomek else ('passthrough' if args.passthrough else 'smotenc'),
    })

oof_preds = pd.DataFrame(0.0, index=X.index, columns=classes)
test_preds = np.zeros((len(X_test), len(classes)))
# For weighted ensemble optimization: store per-fold LGB/CB predictions
oof_lgb_rest = {}   # fold_idx -> (val_idx, lgb_proba, s2_classes)
oof_cb_rest = {}    # fold_idx -> (val_idx, cb_proba)
oof_xgb_rest = {}   # fold_idx -> (val_idx, xgb_proba)
oof_gull = {}       # fold_idx -> (val_idx, p_gull)
test_lgb_rest_acc = np.zeros((len(X_test), 1))  # placeholder, sized later
test_cb_rest_acc = np.zeros((len(X_test), 1))
test_xgb_rest_acc = np.zeros((len(X_test), 1))
test_gull_acc = np.zeros(len(X_test))

print(f"\nTraining {len(split)}-fold two-stage pipeline...")
for i, (train_idx, val_idx) in enumerate(split):
    X_tr = X.iloc[train_idx]
    X_va = X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_va = y.iloc[val_idx]

    # Exclude augmented samples from validation (they're copies of training data)
    if args.oversampler in ('trajectory', 'adasyn+trajectory'):
        va_real = ~X_va.index.astype(str).str.startswith('aug_')
        X_va = X_va[va_real]
        y_va = y_va[va_real]

    # ── Optional: undersample Gulls in training set ──
    if us_gulls > 0:
        gull_mask = y_tr == 'Gulls'
        gull_indices = y_tr.index[gull_mask]
        non_gull_indices = y_tr.index[~gull_mask]
        if len(gull_indices) > us_gulls:
            rng = np.random.RandomState(42 + i)
            keep = rng.choice(gull_indices, size=us_gulls, replace=False)
            keep_idx = np.concatenate([keep, non_gull_indices.values])
            X_tr = X_tr.loc[keep_idx]
            y_tr = y_tr.loc[keep_idx]

    # ════════════════════════════════════════════
    # STAGE 1: Binary Gull vs Non-Gull
    # ════════════════════════════════════════════
    y_tr_bin = (y_tr == 'Gulls').astype(int)

    if args.stage1_catboost:
        cb_imp1 = clone(cb_imputer)
        X_tr_cb = cb_imp1.fit_transform(X_tr, y_tr_bin)
        X_va_cb = cb_imp1.transform(X_va)
        X_te_cb = cb_imp1.transform(X_test)
        for ci in cb_cat_idx:
            X_tr_cb[:, ci] = X_tr_cb[:, ci].astype(str)
            X_va_cb[:, ci] = X_va_cb[:, ci].astype(str)
            X_te_cb[:, ci] = X_te_cb[:, ci].astype(str)
        cb1_params = {**cb_params, 'loss_function': 'Logloss'}
        cb1 = CatBoostClassifier(**cb1_params)
        cb1.fit(X_tr_cb, y_tr_bin, eval_set=(X_va_cb, y_va.map(lambda x: 1 if x == 'Gulls' else 0)),
                early_stopping_rounds=50, cat_features=cb_cat_idx)
        val_p_gull_raw = cb1.predict_proba(X_va_cb)[:, 1]
        test_p_gull_raw = cb1.predict_proba(X_te_cb)[:, 1]
    else:
        lgb_binary = LGBMClassifier(
            n_estimators=2500, learning_rate=0.03, num_leaves=63,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1,
            device='gpu', verbose=-1,
        )
        pipe1 = ImbPipeline([
            ('imputer', clone(imputer)),
            ('oversampler', 'passthrough'),
            ('model', lgb_binary)
        ])
        pipe1.fit(X_tr, y_tr_bin)
        val_p_gull_raw = pipe1.predict_proba(X_va)[:, 1]
        test_p_gull_raw = pipe1.predict_proba(X_test)[:, 1]

    # Asymmetric threshold recalibration (monotonic power transform)
    if gull_thresh != 0.5:
        gamma = np.log(0.5) / np.log(gull_thresh)
        val_p_gull = np.power(val_p_gull_raw, gamma)
        test_p_gull = np.power(test_p_gull_raw, gamma)
    else:
        val_p_gull = val_p_gull_raw
        test_p_gull = test_p_gull_raw

    # ════════════════════════════════════════════
    # STAGE 2: 8-class Non-Gull (LightGBM + CatBoost ensemble)
    # ════════════════════════════════════════════
    non_gull_mask = y_tr != 'Gulls'
    X_tr_ng = X_tr[non_gull_mask]
    y_tr_ng = y_tr[non_gull_mask]

    # ── Stage 2a: LightGBM with oversampling + boost-weak ──
    lgb_imp = clone(imputer)
    X_tr_ng_imp = lgb_imp.fit_transform(X_tr_ng, y_tr_ng)
    X_va_imp = lgb_imp.transform(X_va)
    X_te_imp = lgb_imp.transform(X_test)

    if oversampler != 'passthrough':
        os_step = clone(oversampler)
        X_ng_res, y_ng_res = os_step.fit_resample(X_tr_ng_imp, y_tr_ng)
    else:
        X_ng_res, y_ng_res = X_tr_ng_imp, y_tr_ng

    if args.oversampler not in ('trajectory', 'adasyn+trajectory'):
        X_ng_res, y_ng_res = apply_boost(X_ng_res, y_ng_res, CLASS_BOOST)

    # ── Stage 2: Seed-averaged LightGBM + CatBoost ──
    n_seeds = args.n_seeds
    seed_list = [42 + s for s in range(n_seeds)]

    lgb_val_proba_acc = None
    lgb_test_proba_acc = None
    cb_val_proba_acc = None
    cb_test_proba_acc = None
    xgb_val_proba_acc = None
    xgb_test_proba_acc = None
    s2_classes = None

    for seed in seed_list:
        # LightGBM with this seed
        lgb_s2 = LGBMClassifier(**lgb_params)
        lgb_s2.set_params(
            min_child_samples=max(10, int(len(X_ng_res) * 0.01)),
            random_state=seed,
        )
        lgb_s2.fit(X_ng_res, y_ng_res)
        _lgb_val = lgb_s2.predict_proba(X_va_imp)
        _lgb_test = lgb_s2.predict_proba(X_te_imp)
        if s2_classes is None:
            s2_classes = lgb_s2.classes_
        if lgb_val_proba_acc is None:
            lgb_val_proba_acc = _lgb_val
            lgb_test_proba_acc = _lgb_test
        else:
            lgb_val_proba_acc += _lgb_val
            lgb_test_proba_acc += _lgb_test

        # CatBoost with this seed
        cb_imp2 = clone(cb_imputer)
        X_tr_ng_cb = cb_imp2.fit_transform(X_tr_ng, y_tr_ng)
        X_va_cb2 = cb_imp2.transform(X_va)
        X_te_cb2 = cb_imp2.transform(X_test)
        for ci in cb_cat_idx:
            X_tr_ng_cb[:, ci] = X_tr_ng_cb[:, ci].astype(str)
            X_va_cb2[:, ci] = X_va_cb2[:, ci].astype(str)
            X_te_cb2[:, ci] = X_te_cb2[:, ci].astype(str)

        cb_s2 = CatBoostClassifier(**{**cb_params, 'random_seed': seed})
        cb_s2.fit(X_tr_ng_cb, y_tr_ng, cat_features=cb_cat_idx)
        _cb_val = cb_s2.predict_proba(X_va_cb2)
        _cb_test = cb_s2.predict_proba(X_te_cb2)

        # Align CatBoost classes to LightGBM class order
        cb_s2_classes = cb_s2.classes_
        if not np.array_equal(cb_s2_classes, s2_classes):
            cb_reorder = [list(cb_s2_classes).index(c) for c in s2_classes]
            _cb_val = _cb_val[:, cb_reorder]
            _cb_test = _cb_test[:, cb_reorder]

        if cb_val_proba_acc is None:
            cb_val_proba_acc = _cb_val
            cb_test_proba_acc = _cb_test
        else:
            cb_val_proba_acc += _cb_val
            cb_test_proba_acc += _cb_test

        # XGBoost with this seed
        from sklearn.preprocessing import LabelEncoder as _LE
        _le = _LE()
        y_xgb_enc = _le.fit_transform(y_ng_res)
        # Compute sample weights for class balance
        _cls_counts = np.bincount(y_xgb_enc)
        _n_total = len(y_xgb_enc)
        _n_cls = len(_cls_counts)
        _sw = np.array([_n_total / (_n_cls * _cls_counts[c]) for c in y_xgb_enc])
        xgb_s2 = XGBClassifier(
            n_estimators=2500, learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', device='cuda',
            random_state=seed, verbosity=0, n_jobs=-1,
            num_class=_n_cls, objective='multi:softprob',
        )
        xgb_s2.fit(X_ng_res, y_xgb_enc, sample_weight=_sw)
        _xgb_val = xgb_s2.predict_proba(X_va_imp)
        _xgb_test = xgb_s2.predict_proba(X_te_imp)
        # Align XGBoost classes to LightGBM class order
        xgb_classes = _le.classes_
        if not np.array_equal(xgb_classes, s2_classes):
            xgb_reorder = [list(xgb_classes).index(c) for c in s2_classes]
            _xgb_val = _xgb_val[:, xgb_reorder]
            _xgb_test = _xgb_test[:, xgb_reorder]

        if xgb_val_proba_acc is None:
            xgb_val_proba_acc = _xgb_val
            xgb_test_proba_acc = _xgb_test
        else:
            xgb_val_proba_acc += _xgb_val
            xgb_test_proba_acc += _xgb_test

    lgb_val_proba = lgb_val_proba_acc / n_seeds
    lgb_test_proba = lgb_test_proba_acc / n_seeds
    cb_val_proba = cb_val_proba_acc / n_seeds
    cb_test_proba = cb_test_proba_acc / n_seeds
    xgb_val_proba = xgb_val_proba_acc / n_seeds
    xgb_test_proba = xgb_test_proba_acc / n_seeds

    # Store per-model predictions for weight optimization
    oof_lgb_rest[i] = (X_va.index, lgb_val_proba, s2_classes)
    oof_cb_rest[i] = (X_va.index, cb_val_proba)
    oof_xgb_rest[i] = (X_va.index, xgb_val_proba)
    oof_gull[i] = (X_va.index, val_p_gull)
    # Accumulate test predictions per model
    if test_lgb_rest_acc.shape[1] == 1:
        test_lgb_rest_acc = np.zeros((len(X_test), lgb_test_proba.shape[1]))
        test_cb_rest_acc = np.zeros((len(X_test), cb_test_proba.shape[1]))
        test_xgb_rest_acc = np.zeros((len(X_test), xgb_test_proba.shape[1]))
    test_lgb_rest_acc += lgb_test_proba
    test_cb_rest_acc += cb_test_proba
    test_xgb_rest_acc += xgb_test_proba
    test_gull_acc += test_p_gull

    # ── Stage 2 ensemble: simple average for per-fold mAP reporting ──
    val_p_rest = (lgb_val_proba + cb_val_proba + xgb_val_proba) / 3
    test_p_rest = (lgb_test_proba + cb_test_proba + xgb_test_proba) / 3

    # ════════════════════════════════════════════
    # COMBINE: p(Gull) from Stage 1, p(other) from Stage 2
    # ════════════════════════════════════════════
    val_combined = np.zeros((len(X_va), len(classes)))
    test_combined = np.zeros((len(X_test), len(classes)))

    gull_idx = list(classes).index('Gulls')
    val_combined[:, gull_idx] = val_p_gull
    test_combined[:, gull_idx] = test_p_gull

    for j, cls in enumerate(s2_classes):
        cls_idx = list(classes).index(cls)
        val_combined[:, cls_idx] = (1 - val_p_gull) * val_p_rest[:, j]
        test_combined[:, cls_idx] = (1 - test_p_gull) * test_p_rest[:, j]

    # Use X_va.index to handle augmented-sample filtering
    oof_preds.loc[X_va.index] = val_combined
    test_preds += test_combined

    fold_ap = average_precision_score(
        pd.get_dummies(y_va).reindex(columns=classes, fill_value=0),
        val_combined, average='macro'
    )
    print(f"  Fold {i+1}/{len(split)} — val mAP: {fold_ap:.4f}")
    if use_mlflow:
        mlflow.log_metric(f"fold_mAP", fold_ap, step=i)

test_preds /= len(split)
test_lgb_rest_acc /= len(split)
test_cb_rest_acc /= len(split)
test_xgb_rest_acc /= len(split)
test_gull_acc /= len(split)

# ─────────────────────────────────────────────
# 8b. Weighted ensemble optimization (3 models: LGB, CB, XGB)
# ─────────────────────────────────────────────
# Grid search over w_lgb, w_cb (w_xgb = 1 - w_lgb - w_cb) with step 0.05
eval_mask_we = ~train_df.index.astype(str).str.startswith('aug_')
best_weights = (1/3, 1/3, 1/3)
best_map_w = 0.0
step = 5  # percent
for w_lgb_int in range(0, 101, step):
    for w_cb_int in range(0, 101 - w_lgb_int, step):
        w_xgb_int = 100 - w_lgb_int - w_cb_int
        w_lgb = w_lgb_int / 100.0
        w_cb = w_cb_int / 100.0
        w_xgb = w_xgb_int / 100.0
        oof_w = pd.DataFrame(0.0, index=X.index, columns=classes)
        for fi in oof_lgb_rest:
            va_idx, lgb_p, s2_cls = oof_lgb_rest[fi]
            _, cb_p = oof_cb_rest[fi]
            _, xgb_p = oof_xgb_rest[fi]
            _, p_gull = oof_gull[fi]
            rest_p = w_lgb * lgb_p + w_cb * cb_p + w_xgb * xgb_p
            combined = np.zeros((len(va_idx), len(classes)))
            gull_ci = list(classes).index('Gulls')
            combined[:, gull_ci] = p_gull
            for j, cls in enumerate(s2_cls):
                ci = list(classes).index(cls)
                combined[:, ci] = (1 - p_gull) * rest_p[:, j]
            oof_w.loc[va_idx] = combined
        # Evaluate
        eval_df_w = train_df[eval_mask_we]
        sol_w = eval_df_w.reset_index().groupby(["track_id", "bird_group"]).size().unstack(fill_value=0)
        oof_w_al = oof_w.reindex(sol_w.index).loc[sol_w.index, sol_w.columns]
        map_w = average_precision_score(sol_w[needed_columns], oof_w_al[needed_columns], average='macro')
        if map_w > best_map_w:
            best_map_w = map_w
            best_weights = (w_lgb, w_cb, w_xgb)

w_lgb_opt, w_cb_opt, w_xgb_opt = best_weights
print(f"\n  Ensemble weight optimization: w(LGB)={w_lgb_opt:.2f}, w(CB)={w_cb_opt:.2f}, w(XGB)={w_xgb_opt:.2f} → mAP={best_map_w:.4f}")

# Also try rank averaging (convert to percentile ranks per class, then average)
from scipy.stats import rankdata
oof_rank = pd.DataFrame(0.0, index=X.index, columns=classes)
for fi in oof_lgb_rest:
    va_idx, lgb_p, s2_cls = oof_lgb_rest[fi]
    _, cb_p = oof_cb_rest[fi]
    _, xgb_p = oof_xgb_rest[fi]
    _, p_gull = oof_gull[fi]
    # Rank-average the 3 models per class
    n = len(va_idx)
    rest_rank = np.zeros_like(lgb_p)
    for j in range(lgb_p.shape[1]):
        r_lgb = rankdata(lgb_p[:, j]) / n
        r_cb = rankdata(cb_p[:, j]) / n
        r_xgb = rankdata(xgb_p[:, j]) / n
        rest_rank[:, j] = (r_lgb + r_cb + r_xgb) / 3
    combined = np.zeros((n, len(classes)))
    gull_ci = list(classes).index('Gulls')
    combined[:, gull_ci] = p_gull
    for j, cls in enumerate(s2_cls):
        ci = list(classes).index(cls)
        combined[:, ci] = (1 - p_gull) * rest_rank[:, j]
    oof_rank.loc[va_idx] = combined
# Evaluate rank averaging
eval_df_r = train_df[eval_mask_we]
sol_r = eval_df_r.reset_index().groupby(["track_id", "bird_group"]).size().unstack(fill_value=0)
oof_r_al = oof_rank.reindex(sol_r.index).loc[sol_r.index, sol_r.columns]
map_rank = average_precision_score(sol_r[needed_columns], oof_r_al[needed_columns], average='macro')
print(f"  Rank averaging (equal weights): mAP={map_rank:.4f}")

# Use rank averaging if better
use_rank_avg = map_rank > best_map_w
if use_rank_avg:
    print(f"  → Rank averaging is better, using it")
    best_map_w = map_rank

# Rebuild final OOF and test predictions
if use_rank_avg:
    oof_preds = oof_rank.copy()
    # Rank-average test predictions
    n_test = len(X_test)
    test_rest_rank = np.zeros_like(test_lgb_rest_acc)
    for j in range(test_lgb_rest_acc.shape[1]):
        r_lgb = rankdata(test_lgb_rest_acc[:, j]) / n_test
        r_cb = rankdata(test_cb_rest_acc[:, j]) / n_test
        r_xgb = rankdata(test_xgb_rest_acc[:, j]) / n_test
        test_rest_rank[:, j] = (r_lgb + r_cb + r_xgb) / 3
    test_preds = np.zeros((n_test, len(classes)))
    gull_ci = list(classes).index('Gulls')
    test_preds[:, gull_ci] = test_gull_acc
    for j, cls in enumerate(s2_classes):
        ci = list(classes).index(cls)
        test_preds[:, ci] = (1 - test_gull_acc) * test_rest_rank[:, j]
else:
    oof_preds = pd.DataFrame(0.0, index=X.index, columns=classes)
    for fi in oof_lgb_rest:
        va_idx, lgb_p, s2_cls = oof_lgb_rest[fi]
        _, cb_p = oof_cb_rest[fi]
        _, xgb_p = oof_xgb_rest[fi]
        _, p_gull = oof_gull[fi]
        rest_p = w_lgb_opt * lgb_p + w_cb_opt * cb_p + w_xgb_opt * xgb_p
        combined = np.zeros((len(va_idx), len(classes)))
        gull_ci = list(classes).index('Gulls')
        combined[:, gull_ci] = p_gull
        for j, cls in enumerate(s2_cls):
            ci = list(classes).index(cls)
            combined[:, ci] = (1 - p_gull) * rest_p[:, j]
        oof_preds.loc[va_idx] = combined

    # Rebuild test predictions with optimal weights
    test_p_rest_opt = w_lgb_opt * test_lgb_rest_acc + w_cb_opt * test_cb_rest_acc + w_xgb_opt * test_xgb_rest_acc
    test_preds = np.zeros((len(X_test), len(classes)))
    gull_ci = list(classes).index('Gulls')
    test_preds[:, gull_ci] = test_gull_acc
    for j, cls in enumerate(s2_classes):
        ci = list(classes).index(cls)
        test_preds[:, ci] = (1 - test_gull_acc) * test_p_rest_opt[:, j]

# ─────────────────────────────────────────────
# 8c. Per-class probability calibration
# ─────────────────────────────────────────────
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LR_cal

eval_mask_cal = ~train_df.index.astype(str).str.startswith('aug_')
y_cal = train_df.loc[eval_mask_cal, 'bird_group']
oof_cal = oof_preds.loc[eval_mask_cal]

calibrated_oof = oof_cal.copy()
calibrators = {}
for cls in needed_columns:
    y_binary = (y_cal == cls).astype(int).values
    p_raw = oof_cal[cls].values
    n_pos = y_binary.sum()
    if n_pos < 100:
        # Platt scaling for small classes
        cal = LR_cal(C=1.0, solver='lbfgs', max_iter=1000)
        cal.fit(p_raw.reshape(-1, 1), y_binary)
        calibrators[cls] = ('platt', cal)
        calibrated_oof[cls] = cal.predict_proba(p_raw.reshape(-1, 1))[:, 1]
    else:
        # Isotonic regression for large classes
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(p_raw, y_binary)
        calibrators[cls] = ('isotonic', cal)
        calibrated_oof[cls] = cal.predict(p_raw)

# Renormalize rows to sum to 1
row_sums = calibrated_oof.sum(axis=1)
calibrated_oof = calibrated_oof.div(row_sums, axis=0)
oof_preds.loc[eval_mask_cal] = calibrated_oof

# Calibrate test predictions
test_preds_df = pd.DataFrame(test_preds, columns=classes)
for cls in needed_columns:
    cal_type, cal = calibrators[cls]
    if cal_type == 'platt':
        test_preds_df[cls] = cal.predict_proba(test_preds_df[cls].values.reshape(-1, 1))[:, 1]
    else:
        test_preds_df[cls] = cal.predict(test_preds_df[cls].values)
# Renormalize
test_row_sums = test_preds_df.sum(axis=1)
test_preds_df = test_preds_df.div(test_row_sums, axis=0)
test_preds = test_preds_df.values

print("  Probability calibration applied (Platt for small classes, Isotonic for large)")

# ─────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────
# For evaluation, only use original (non-augmented) samples
eval_mask = ~train_df.index.astype(str).str.startswith('aug_')
eval_df = train_df[eval_mask]
solution_df = (
    eval_df.reset_index()
    .groupby(["track_id", "bird_group"]).size()
    .unstack(fill_value=0)
)
oof_aligned = oof_preds.reindex(solution_df.index).loc[solution_df.index, solution_df.columns]

overall_map = average_precision_score(
    solution_df[needed_columns],
    oof_aligned[needed_columns],
    average='macro'
)

print(f"\n{'='*50}")
print(f" OOF Macro-Averaged AP (mAP): {overall_map:.4f}")
print(f"{'='*50}")
print("\n Per-Class Average Precision:")
per_class_ap = {}
for cls in needed_columns:
    if cls in solution_df.columns and cls in oof_aligned.columns:
        ap = average_precision_score(solution_df[cls], oof_aligned[cls])
        per_class_ap[cls] = ap
        print(f"   {cls:20s}: {ap:.4f}")

if use_mlflow:
    mlflow.log_metric("overall_mAP", overall_map)
    for cls, ap in per_class_ap.items():
        mlflow.log_metric(f"AP_{cls.replace(' ', '_')}", ap)

# ─────────────────────────────────────────────
# 10. Generate Submission
# ─────────────────────────────────────────────
submission_df = pd.DataFrame(
    test_preds,
    index=X_test.index,
    columns=classes
)
submission_df.index.name = 'track_id'
submission_df.to_csv('submission.csv')
print(f"\nSaved submission.csv ({len(submission_df)} rows)")

if use_mlflow:
    mlflow.log_artifact('submission.csv')
    mlflow.end_run()
    print(f"MLflow run logged. Run: {run.info.run_name} (ID: {run.info.run_id})")

# ─────────────────────────────────────────────
# 11. Pseudo-Labeling (optional)
# ─────────────────────────────────────────────
if args.pseudo_label:
    # Use test predictions from first round to create pseudo-labeled samples
    pseudo_probs = submission_df[needed_columns]
    max_prob = pseudo_probs.max(axis=1)
    pseudo_mask = max_prob >= args.pseudo_threshold
    pseudo_labels = pseudo_probs.columns[pseudo_probs.values.argmax(axis=1)]

    n_pseudo = pseudo_mask.sum()
    print(f"\n{'='*50}")
    print(f" Pseudo-labeling: {n_pseudo}/{len(pseudo_probs)} test samples above {args.pseudo_threshold} threshold")
    print(f"{'='*50}")

    if n_pseudo > 0:
        # Build pseudo-labeled dataframe from test features
        pseudo_X = X_test[pseudo_mask].copy()
        pseudo_y = pd.Series(pseudo_labels[pseudo_mask], index=pseudo_X.index)

        # Print class distribution of pseudo-labels
        print(" Pseudo-label distribution:")
        for cls in sorted(pseudo_y.value_counts().index):
            print(f"   {cls:20s}: {pseudo_y.value_counts()[cls]}")

        # Combine with original training data
        X_pl = pd.concat([X, pseudo_X])
        y_pl = pd.concat([y, pseudo_y])

        # Need groups for StratifiedGroupKFold — pseudo samples get unique group IDs
        # Use large negative ints for pseudo groups to avoid type mismatch
        max_group = int(groups.max()) + 1
        groups_pl = pd.concat([
            groups,
            pd.Series([max_group + i for i in range(n_pseudo)], index=pseudo_X.index)
        ])

        # Rerun CV with pseudo-labeled data
        cv_pl = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_pl = list(cv_pl.split(X_pl, y_pl, groups_pl))

        oof_preds_pl = pd.DataFrame(0.0, index=X_pl.index, columns=classes)
        test_preds_pl = np.zeros((len(X_test), len(classes)))

        print(f"\nRetraining {len(split_pl)}-fold with pseudo-labels...")
        for i, (train_idx, val_idx) in enumerate(split_pl):
            X_tr = X_pl.iloc[train_idx]
            X_va = X_pl.iloc[val_idx]
            y_tr = y_pl.iloc[train_idx]
            y_va = y_pl.iloc[val_idx]

            # Exclude augmented samples from validation
            if args.oversampler in ('trajectory', 'adasyn+trajectory'):
                va_real = ~X_va.index.astype(str).str.startswith('aug_')
                X_va = X_va[va_real]
                y_va = y_va[va_real]

            if us_gulls > 0:
                gull_mask = y_tr == 'Gulls'
                gull_indices = y_tr.index[gull_mask]
                non_gull_indices = y_tr.index[~gull_mask]
                if len(gull_indices) > us_gulls:
                    rng = np.random.RandomState(42 + i)
                    keep = rng.choice(gull_indices, size=us_gulls, replace=False)
                    keep_idx = np.concatenate([non_gull_indices, keep])
                    X_tr, y_tr = X_tr.loc[keep_idx], y_tr.loc[keep_idx]

            # Stage 1: binary Gull vs non-Gull
            y_binary = (y_tr == 'Gulls').astype(int)

            if args.stage1_catboost:
                s1_imp = clone(cb_imputer)
                X_tr_s1 = s1_imp.fit_transform(X_tr, y_binary)
                X_va_s1 = s1_imp.transform(X_va)
                X_te_s1 = s1_imp.transform(X_test)
                cb_s1 = CatBoostClassifier(**cb_params)
                cb_s1.set_params(loss_function='Logloss')
                pool_tr = __import__('catboost').Pool(X_tr_s1, y_binary, cat_features=cb_cat_idx)
                cb_s1.fit(pool_tr)
                val_p_gull = cb_s1.predict_proba(X_va_s1)[:, 1]
                test_p_gull = cb_s1.predict_proba(X_te_s1)[:, 1]
            else:
                s1_imp = clone(imputer)
                X_tr_s1 = s1_imp.fit_transform(X_tr, y_binary)
                X_va_s1 = s1_imp.transform(X_va)
                X_te_s1 = s1_imp.transform(X_test)
                lgb_s1 = LGBMClassifier(
                    n_estimators=1500, learning_rate=0.03, num_leaves=63,
                    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                    class_weight='balanced', random_state=42, verbose=-1,
                )
                lgb_s1.fit(X_tr_s1, y_binary)
                val_p_gull = lgb_s1.predict_proba(X_va_s1)[:, 1]
                test_p_gull = lgb_s1.predict_proba(X_te_s1)[:, 1]

            val_p_gull = np.where(val_p_gull >= gull_thresh, val_p_gull, val_p_gull * (gull_thresh / 0.5) if gull_thresh < 0.5 else val_p_gull)
            test_p_gull = np.where(test_p_gull >= gull_thresh, test_p_gull, test_p_gull * (gull_thresh / 0.5) if gull_thresh < 0.5 else test_p_gull)

            # Stage 2: non-Gull 8-class
            ng_mask = y_tr != 'Gulls'
            X_tr_ng = X_tr[ng_mask]
            y_tr_ng = y_tr[ng_mask]

            lgb_imp = clone(imputer)
            X_tr_ng_imp = lgb_imp.fit_transform(X_tr_ng, y_tr_ng)
            X_va_imp = lgb_imp.transform(X_va)
            X_te_imp = lgb_imp.transform(X_test)

            if oversampler != 'passthrough':
                os_step = clone(oversampler)
                X_ng_res, y_ng_res = os_step.fit_resample(X_tr_ng_imp, y_tr_ng)
            else:
                X_ng_res, y_ng_res = X_tr_ng_imp, y_tr_ng

            if args.oversampler not in ('trajectory', 'adasyn+trajectory'):
                X_ng_res, y_ng_res = apply_boost(X_ng_res, y_ng_res, CLASS_BOOST)

            lgb_s2 = LGBMClassifier(**lgb_params)
            lgb_s2.set_params(min_child_samples=max(10, int(len(X_ng_res) * 0.01)))
            lgb_s2.fit(X_ng_res, y_ng_res)
            lgb_val_proba = lgb_s2.predict_proba(X_va_imp)
            lgb_test_proba = lgb_s2.predict_proba(X_te_imp)
            s2_classes = lgb_s2.classes_

            # CatBoost Stage 2
            cb_s2_imp = clone(cb_imputer)
            X_tr_ng_cb = cb_s2_imp.fit_transform(X_tr_ng, y_tr_ng)
            X_va_cb = cb_s2_imp.transform(X_va)
            X_te_cb = cb_s2_imp.transform(X_test)
            cb_s2 = CatBoostClassifier(**cb_params)
            pool_ng = __import__('catboost').Pool(X_tr_ng_cb, y_tr_ng, cat_features=cb_cat_idx)
            cb_s2.fit(pool_ng)
            cb_val_proba = cb_s2.predict_proba(X_va_cb)
            cb_test_proba = cb_s2.predict_proba(X_te_cb)

            # Ensemble
            val_p_rest = 0.5 * lgb_val_proba + 0.5 * cb_val_proba
            test_p_rest = 0.5 * lgb_test_proba + 0.5 * cb_test_proba

            # Combine Stage 1 + Stage 2
            val_combined = np.zeros((len(X_va), len(classes)))
            test_combined = np.zeros((len(X_test), len(classes)))
            gull_idx = list(classes).index('Gulls')
            val_combined[:, gull_idx] = val_p_gull
            test_combined[:, gull_idx] = test_p_gull
            for j, cls in enumerate(s2_classes):
                cls_idx = list(classes).index(cls)
                val_combined[:, cls_idx] = (1 - val_p_gull) * val_p_rest[:, j]
                test_combined[:, cls_idx] = (1 - test_p_gull) * test_p_rest[:, j]

            oof_preds_pl.loc[X_va.index] = val_combined
            test_preds_pl += test_combined

            fold_ap = average_precision_score(
                pd.get_dummies(y_va).reindex(columns=classes, fill_value=0),
                val_combined, average='macro'
            )
            print(f"  Fold {i+1}/{len(split_pl)} — val mAP: {fold_ap:.4f}")

        test_preds_pl /= len(split_pl)

        # Evaluate pseudo-label round (only on original training data)
        eval_mask_pl = train_df.index.isin(X_pl.index[:len(X.index)])
        oof_preds_orig = oof_preds_pl.loc[X.index]

        eval_mask2 = ~train_df.index.astype(str).str.startswith('aug_')
        eval_df2 = train_df[eval_mask2]
        solution_df2 = (
            eval_df2.reset_index()
            .groupby(["track_id", "bird_group"]).size()
            .unstack(fill_value=0)
        )
        oof_aligned2 = oof_preds_orig.reindex(solution_df2.index).loc[solution_df2.index, solution_df2.columns]

        overall_map_pl = average_precision_score(
            solution_df2[needed_columns],
            oof_aligned2[needed_columns],
            average='macro'
        )

        print(f"\n{'='*50}")
        print(f" Pseudo-Label OOF mAP: {overall_map_pl:.4f}")
        print(f"{'='*50}")
        print("\n Per-Class Average Precision:")
        for cls in needed_columns:
            if cls in solution_df2.columns and cls in oof_aligned2.columns:
                ap = average_precision_score(solution_df2[cls], oof_aligned2[cls])
                print(f"   {cls:20s}: {ap:.4f}")

        # Save pseudo-label submission
        submission_pl = pd.DataFrame(
            test_preds_pl, index=X_test.index, columns=classes
        )
        submission_pl.index.name = 'track_id'
        submission_pl.to_csv('submission.csv')
        print(f"\nSaved submission.csv with pseudo-labels ({len(submission_pl)} rows)")
    else:
        print("No test samples above threshold — skipping pseudo-label round.")
