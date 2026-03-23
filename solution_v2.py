"""
AI Cup 2026 — Bird Radar Track Classification (Version 2)
Hierarchical Ensemble Architecture:
Stage 1: Binary LightGBM (Gull vs. Non-Gull)
Stage 2: Expert Ensemble (LightGBM + CatBoost) for 8-class Non-Gull classification
Integrates 32-dimensional CNN embeddings.
"""

import argparse
import os
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
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# ─────────────────────────────────────────────
# 0. Configuration & Setup
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='AI Cup 2026 Bird Radar Classification v2')
parser.add_argument('--gull-threshold', type=float, default=0.5, metavar='T',
                    help='Stage-1 threshold for calling Gull (default: 0.5).')
parser.add_argument('--fast-dev-run', action='store_true',
                    help='Run 2 folds with tiny data for speed.')
args = parser.parse_args()

DATASET_CONFIG = {
    'openmeteo': {
        'wind_speed_col': 'openmeteo_wind_speed_10m_kmh',
        'wind_dir_col': 'openmeteo_wind_direction_10m_degrees',
        'wind_unit_factor': 1 / 3.6,
    }
}
ds = DATASET_CONFIG['openmeteo']

# ─────────────────────────────────────────────
# 1. Trajectory Parsing & Feature Engineering
# ─────────────────────────────────────────────

def parse_trajectory(hex_str):
    """Decode EWKB hex string."""
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
    """Extract spatial, RCS, and velocity features."""
    coords = parse_trajectory(row['trajectory'])
    n = len(coords)
    
    # Initialize with NaNs to ensure column consistency
    default_feats = {
        'n_points': n, 'total_dist_m': 0.0, 'mean_step_m': 0.0, 'std_step_m': 0.0,
        'lon_range': 0.0, 'lat_range': 0.0, 'alt_mean': np.nan, 'alt_std': np.nan,
        'rcs_mean': np.nan, 'rcs_std': np.nan, 'rcs_min': np.nan, 'rcs_max': np.nan, 'rcs_range': np.nan,
        'tortuosity': 0.0, 'tortuosity_max': 0.0, 'sharp_turn_ratio': 0.0,
        'straightness': 0.0, 'sinuosity': 0.0, 'lon_mean': np.nan, 'lat_mean': np.nan,
        'track_heading_rad': 0.0, 'alt_climb_rate': 0.0, 'alt_descent_rate': 0.0, 'alt_variability': 0.0,
        'speed_mean': np.nan, 'speed_std': np.nan, 'speed_max': np.nan, 'speed_cv': np.nan,
        'accel_mean': np.nan, 'accel_std': np.nan
    }
    
    if n == 0:
        return pd.Series(default_feats)

    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    alts = [c[2] for c in coords] if len(coords[0]) > 2 else [np.nan] * n
    rcs  = [c[3] for c in coords] if len(coords[0]) > 3 else [np.nan] * n

    lons_s = pd.Series(lons).rolling(window=3, min_periods=1, center=True).mean().values
    lats_s = pd.Series(lats).rolling(window=3, min_periods=1, center=True).mean().values
    dx, dy = np.diff(lons_s) * 71000, np.diff(lats_s) * 111000
    step_dist = np.sqrt(dx**2 + dy**2)
    total_dist = step_dist.sum() if len(step_dist) > 0 else 0.0
    
    bearings = np.arctan2(dy, dx)
    bearing_changes = np.abs(np.diff(bearings)) if len(bearings) > 1 else np.array([0.0])
    bearing_changes = np.minimum(bearing_changes, 2 * np.pi - bearing_changes)
    
    displacement = np.sqrt(((lons[-1] - lons[0]) * 71000) ** 2 + ((lats[-1] - lats[0]) * 111000) ** 2)
    
    feats = default_feats.copy()
    feats.update({
        'total_dist_m':   total_dist,
        'mean_step_m':    step_dist.mean() if len(step_dist) > 0 else 0.0,
        'std_step_m':     step_dist.std()  if len(step_dist) > 0 else 0.0,
        'lon_range':      max(lons) - min(lons),
        'lat_range':      max(lats) - min(lats),
        'alt_mean':       np.nanmean(alts), 'alt_std': np.nanstd(alts),
        'rcs_mean':       np.nanmean(rcs), 'rcs_std': np.nanstd(rcs),
        'rcs_min':        np.nanmin(rcs), 'rcs_max': np.nanmax(rcs), 'rcs_range': np.nanmax(rcs) - np.nanmin(rcs),
        'tortuosity':     bearing_changes.mean() if len(bearing_changes) > 0 else 0.0,
        'tortuosity_max': bearing_changes.max()  if len(bearing_changes) > 0 else 0.0,
        'sharp_turn_ratio': (bearing_changes > np.pi / 4).mean() if len(bearing_changes) > 0 else 0.0,
        'straightness':   displacement / (total_dist + 1e-6),
        'sinuosity':      total_dist / (displacement + 1e-6),
        'lon_mean':       np.mean(lons), 'lat_mean': np.mean(lats),
        'track_heading_rad': np.arctan2((lats[-1] - lats[0]) * 111000, (lons[-1] - lons[0]) * 71000),
    })

    if n > 1 and not np.all(np.isnan(alts)):
        alt_changes = np.diff(np.array(alts, dtype=float))
        feats.update({'alt_climb_rate': np.nanmean(alt_changes), 'alt_descent_rate': np.nanmin(alt_changes), 'alt_variability': np.nanstd(alt_changes)})

    times = row.get('trajectory_time', '')
    if isinstance(times, str) and times.strip():
        try:
            t_list = [float(x) for x in times.strip('[]').split(',')]
            if len(t_list) == n and n > 1:
                dt = np.where(np.diff(t_list) == 0, 1e-6, np.diff(t_list))
                speeds = step_dist / dt
                feats.update({'speed_mean': np.mean(speeds), 'speed_std': np.std(speeds), 'speed_max': np.max(speeds),
                             'speed_cv': np.std(speeds) / (np.mean(speeds) + 1e-6)})
                if len(speeds) > 1:
                    accel = np.diff(speeds) / dt[1:]
                    feats.update({'accel_mean': np.mean(accel), 'accel_std': np.std(accel)})
        except: pass
    return pd.Series(feats)

# ─────────────────────────────────────────────
# 2. Loading & Processing
# ─────────────────────────────────────────────
TRAIN_PATH = 'dataset/train_with_openmeteo_cnn.csv'
TEST_PATH = 'dataset/test_with_openmeteo_cnn.csv'

print(f"Loading datasets...")
train_df = pd.read_csv(TRAIN_PATH).set_index("track_id")
test_df = pd.read_csv(TEST_PATH).set_index("track_id")

if args.fast_dev_run:
    train_df, test_df = train_df.iloc[:500], test_df.iloc[:100]
    n_splits = 2
else:
    n_splits = 10

print("Engineering features...")
for df in [train_df, test_df]:
    df['ts_start'] = pd.to_datetime(df['timestamp_start_radar_utc'], utc=True)
    df['ts_end'] = pd.to_datetime(df['timestamp_end_radar_utc'], utc=True)
    df['duration_s'] = (df['ts_end'] - df['ts_start']).dt.total_seconds()
    df['hour'], df['month'] = df['ts_start'].dt.hour, df['ts_start'].dt.month
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)
    df['hour_sin'], df['hour_cos'] = np.sin(2*np.pi*df['hour']/24), np.cos(2*np.pi*df['hour']/24)
    df['month_sin'], df['month_cos'] = np.sin(2*np.pi*df['month']/12), np.cos(2*np.pi*df['month']/12)
    df['alt_range'] = df['max_z'] - df['min_z']
    df['airspeed_per_m'] = df['airspeed'] / (df['max_z'] + 1)
    wf = ds['wind_unit_factor']
    df['headwind_component'] = df['airspeed'] - df[ds['wind_speed_col']] * wf
    df['airspeed_wind_ratio'] = df['airspeed'] / (df[ds['wind_speed_col']] * wf + 0.1)
    wd = df['openmeteo_wind_direction_10m_degrees']
    df['openmeteo_wind_dir_sin'], df['openmeteo_wind_dir_cos'] = np.sin(2*np.pi*wd/360), np.cos(2*np.pi*wd/360)

train_df = train_df.join(train_df.apply(trajectory_features, axis=1))
test_df = test_df.join(test_df.apply(trajectory_features, axis=1))

for df in [train_df, test_df]:
    df['rcs_speed_ratio'] = df['rcs_mean'] / (df['airspeed'] + 1e-6)
    wf = ds['wind_unit_factor']
    df['alt_adjusted_wind_speed'] = df[ds['wind_speed_col']] * wf * np.power(np.maximum(df['alt_mean'], 10) / 10.0, 0.143)
    wind_dir_rad, heading = np.deg2rad(df[ds['wind_dir_col']]), df['track_heading_rad']
    angle_diff = wind_dir_rad - heading
    df['true_tailwind_component'] = df['alt_adjusted_wind_speed'] * np.cos(angle_diff)
    df['true_crosswind_component'] = np.abs(df['alt_adjusted_wind_speed'] * np.sin(angle_diff))

# ─────────────────────────────────────────────
# 3. Feature Initialization & Validation
# ─────────────────────────────────────────────
base_features = [
    'airspeed', 'min_z', 'max_z', 'duration_s', 'radar_bird_size',
    'hour', 'month', 'is_daytime', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'alt_range', 'airspeed_per_m', 'headwind_component', 'airspeed_wind_ratio',
    'rcs_speed_ratio', 'alt_adjusted_wind_speed', 'true_tailwind_component', 'true_crosswind_component',
]
trajectory_feats = [
    'n_points', 'total_dist_m', 'mean_step_m', 'std_step_m', 'lon_range', 'lat_range',
    'alt_mean', 'alt_std', 'rcs_mean', 'rcs_std', 'rcs_min', 'rcs_max', 'rcs_range',
    'tortuosity', 'tortuosity_max', 'sharp_turn_ratio', 'straightness', 'sinuosity',
    'lon_mean', 'lat_mean', 'track_heading_rad', 'alt_climb_rate', 'alt_descent_rate', 'alt_variability',
    'speed_mean', 'speed_std', 'speed_max', 'speed_cv', 'accel_mean', 'accel_std',
]
weather_features = [
    'openmeteo_air_temperature_2m_c', 'openmeteo_relative_humidity_2m_percent', 'openmeteo_dew_point_2m_c',
    'openmeteo_precipitation_mm', 'openmeteo_cloud_cover_percent', 'openmeteo_pressure_msl_hpa',
    'openmeteo_weather_code', 'openmeteo_wind_speed_10m_kmh', 'openmeteo_wind_direction_10m_degrees',
    'openmeteo_wind_gusts_10m_kmh', 'openmeteo_shortwave_radiation_w_m2', 'openmeteo_sunshine_duration_s',
    'openmeteo_vapour_pressure_deficit_kpa', 'openmeteo_is_day', 'openmeteo_wind_dir_sin', 'openmeteo_wind_dir_cos',
]
cnn_features = [f'cnn_emb_{i}' for i in range(32)]

all_features = base_features + trajectory_feats + weather_features + cnn_features
s1_features = base_features + weather_features
s2_features = all_features

# Defensive Validation
missing = [f for f in all_features if f not in train_df.columns]
if missing:
    raise ValueError(f"Missing features in train_df: {missing}")

X = train_df[all_features]
X_test = test_df[all_features]
y = train_df['bird_group']
classes = np.sort(y.unique())

# ─────────────────────────────────────────────
# 4. Pipelines & Inference
# ─────────────────────────────────────────────
def get_imputer(feat_list):
    num_feats = [f for f in feat_list if f != 'radar_bird_size']
    cat_feats = ['radar_bird_size'] if 'radar_bird_size' in feat_list else []
    transformers = [('num_imputer', SimpleImputer(strategy='median'), num_feats)]
    if cat_feats:
        transformers.append(('cat_imputer', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_feats))
    return ColumnTransformer(transformers)

lgb_s1 = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=63, class_weight='balanced', random_state=42, n_jobs=-1, device='gpu', verbose=-1)
lgb_s2 = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=63, class_weight='balanced', random_state=42, n_jobs=-1, device='gpu', verbose=-1)
cb_s2 = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, auto_class_weights='Balanced', random_seed=42, task_type='GPU', verbose=0)

cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
split = list(cv.split(X, y, train_df['primary_observation_id']))
oof_preds = pd.DataFrame(0.0, index=X.index, columns=classes)
test_preds = np.zeros((len(X_test), len(classes)))

for i, (train_idx, val_idx) in enumerate(split):
    X_tr_f, y_tr_f = X.iloc[train_idx], y.iloc[train_idx]
    X_va_f, y_va_f = X.iloc[val_idx], y.iloc[val_idx]
    
    # Stage 1: Gull Binary
    p1 = Pipeline([('imputer', get_imputer(s1_features)), ('model', clone(lgb_s1))])
    p1.fit(X_tr_f[s1_features], (y_tr_f == 'Gulls').astype(int))
    p_gull_va = p1.predict_proba(X_va_f[s1_features])[:, 1]
    p_gull_te = p1.predict_proba(X_test[s1_features])[:, 1]
    
    if args.gull_threshold != 0.5:
        g = np.log(0.5)/np.log(args.gull_threshold)
        p_gull_va, p_gull_te = np.power(p_gull_va, g), np.power(p_gull_te, g)

    # Stage 2: 8-Class Ensemble
    mask_ng = y_tr_f != 'Gulls'
    X_tr_s2, y_tr_s2 = X_tr_f[mask_ng][s2_features], y_tr_f[mask_ng]
    
    # LGBM
    p_l2 = Pipeline([('imputer', get_imputer(s2_features)), ('model', clone(lgb_s2))])
    p_l2.fit(X_tr_s2, y_tr_s2)
    p_va_l, p_te_l = p_l2.predict_proba(X_va_f[s2_features]), p_l2.predict_proba(X_test[s2_features])
    
    # CatBoost
    imp_cb = get_imputer(s2_features)
    X_tr_cb = imp_cb.fit_transform(X_tr_s2)
    m_cb = clone(cb_s2).fit(X_tr_cb, y_tr_s2)
    p_va_c, p_te_c = m_cb.predict_proba(imp_cb.transform(X_va_f[s2_features])), m_cb.predict_proba(imp_cb.transform(X_test[s2_features]))
    
    p_ens_va, p_ens_te = (p_va_l + p_va_c)/2, (p_te_l + p_te_c)/2
    
    comb_va, comb_te = np.zeros((len(X_va_f), len(classes))), np.zeros((len(X_test), len(classes)))
    g_idx = list(classes).index('Gulls')
    comb_va[:, g_idx], comb_te[:, g_idx] = p_gull_va, p_gull_te
    
    for j, cls_name in enumerate(p_l2.steps[-1][1].classes_):
        c_idx = list(classes).index(cls_name)
        comb_va[:, c_idx] = (1 - p_gull_va) * p_ens_va[:, j]
        comb_te[:, c_idx] = (1 - p_gull_te) * p_ens_te[:, j]
    
    oof_preds.iloc[val_idx] = comb_va
    test_preds += comb_te
    print(f"Fold {i+1} mAP: {average_precision_score(pd.get_dummies(y_va_f).reindex(columns=classes, fill_value=0), comb_va, average='macro'):.4f}")

# ─────────────────────────────────────────────
# 5. Output
# ─────────────────────────────────────────────
overall_map = average_precision_score(pd.get_dummies(y).reindex(columns=classes, fill_value=0), oof_preds, average='macro')
print(f"\nFINAL OOF mAP: {overall_map:.4f}")
pd.DataFrame(test_preds/len(split), index=X_test.index, columns=classes).to_csv('submission_v2.csv')
