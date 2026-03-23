"""
Overfitting diagnostic — tests LightGBM and CatBoost individually on fold 0.
Outputs: train vs val loss curves, train/val mAP gap, n_estimators sweep.
"""

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
from imblearn.over_sampling import SMOTENC
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Reuse data loading & feature extraction from solution.py ───

def parse_trajectory(hex_str):
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
    coords = parse_trajectory(row['trajectory'])
    n = len(coords)
    if n == 0:
        return pd.Series({})

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    alts = [c[2] for c in coords] if len(coords[0]) > 2 else [np.nan] * n
    rcs  = [c[3] for c in coords] if len(coords[0]) > 3 else [np.nan] * n

    lons_s = pd.Series(lons).rolling(window=3, min_periods=1, center=True).mean().values
    lats_s = pd.Series(lats).rolling(window=3, min_periods=1, center=True).mean().values

    dx = np.diff(lons_s) * 71000
    dy = np.diff(lats_s) * 111000
    step_dist = np.sqrt(dx**2 + dy**2)
    total_dist = step_dist.sum() if len(step_dist) > 0 else 0.0

    bearings = np.arctan2(dy, dx)
    bearing_changes = np.abs(np.diff(bearings)) if len(bearings) > 1 else np.array([0.0])
    bearing_changes = np.minimum(bearing_changes, 2 * np.pi - bearing_changes)

    sharp_turn_ratio = (bearing_changes > np.pi / 4).mean() if len(bearing_changes) > 0 else 0.0

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

    displacement = np.sqrt(
        ((lons[-1] - lons[0]) * 71000) ** 2 +
        ((lats[-1] - lats[0]) * 111000) ** 2
    )
    straightness = displacement / (total_dist + 1e-6) if total_dist > 0 else 0.0
    sinuosity = total_dist / (displacement + 1e-6)

    track_heading_rad = np.arctan2(
        (lats[-1] - lats[0]) * 111000,
        (lons[-1] - lons[0]) * 71000
    )

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

    rcs_range = np.nanmax(rcs) - np.nanmin(rcs)

    feats = {
        'n_points': n, 'total_dist_m': total_dist,
        'mean_step_m': step_dist.mean() if len(step_dist) > 0 else 0.0,
        'std_step_m': step_dist.std() if len(step_dist) > 0 else 0.0,
        'lon_range': max(lons) - min(lons), 'lat_range': max(lats) - min(lats),
        'alt_mean': np.nanmean(alts), 'alt_std': np.nanstd(alts),
        'rcs_mean': np.nanmean(rcs), 'rcs_std': np.nanstd(rcs),
        'rcs_min': np.nanmin(rcs), 'rcs_max': np.nanmax(rcs), 'rcs_range': rcs_range,
        'tortuosity': bearing_changes.mean() if len(bearing_changes) > 0 else 0.0,
        'tortuosity_max': bearing_changes.max() if len(bearing_changes) > 0 else 0.0,
        'sharp_turn_ratio': sharp_turn_ratio,
        'straightness': straightness, 'sinuosity': sinuosity,
        'lon_mean': np.mean(lons), 'lat_mean': np.mean(lats),
        'track_heading_rad': track_heading_rad,
        'alt_climb_rate': alt_climb_rate, 'alt_descent_rate': alt_descent_rate,
        'alt_variability': alt_variability,
        'curvature_mean': curvatures.mean(),
        'curvature_std': curvatures.std() if len(curvatures) > 1 else 0.0,
        'log_path_length': np.log1p(total_dist),
    }

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
        dt = np.where(dt == 0, 1e-6, dt)
        speeds = step_dist / dt
        feats['speed_mean'] = np.mean(speeds)
        feats['speed_std'] = np.std(speeds)
        feats['speed_max'] = np.max(speeds)
        feats['speed_cv'] = np.std(speeds) / (np.mean(speeds) + 1e-6)
        if len(speeds) > 1:
            accel = np.diff(speeds) / dt[1:]
            feats['accel_mean'] = np.mean(accel)
            feats['accel_std'] = np.std(accel)
        else:
            feats['accel_mean'] = 0.0
            feats['accel_std'] = 0.0
    else:
        for k in ['speed_mean', 'speed_std', 'speed_max', 'speed_cv', 'accel_mean', 'accel_std']:
            feats[k] = np.nan

    return pd.Series(feats)


def apply_boost(X_arr, y_arr, class_boost):
    if not class_boost:
        return X_arr, y_arr
    X_np = X_arr if isinstance(X_arr, np.ndarray) else np.asarray(X_arr)
    y_np = y_arr if isinstance(y_arr, np.ndarray) else np.asarray(y_arr)
    extra_X, extra_y = [], []
    for cls, mult in class_boost.items():
        repeats = int(mult) - 1
        if repeats <= 0:
            continue
        mask = (y_np == cls)
        if mask.sum() > 0:
            extra_X.extend([X_np[mask]] * repeats)
            extra_y.extend([y_np[mask]] * repeats)
    if extra_X:
        X_np = np.vstack([X_np] + extra_X)
        y_np = np.concatenate([y_np] + extra_y)
    return X_np, y_np


# ─── Load & prepare data ───
print("Loading data...")
train_df = pd.read_csv('dataset/train_with_openmeteo.csv').set_index('track_id')

WIND_SPEED_COL = 'openmeteo_wind_speed_10m_kmh'
WIND_DIR_COL = 'openmeteo_wind_direction_10m_degrees'
WF = 1 / 3.6

for df in [train_df]:
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
    df['headwind_component'] = df['airspeed'] - df[WIND_SPEED_COL] * WF
    df['airspeed_wind_ratio'] = df['airspeed'] / (df[WIND_SPEED_COL] * WF + 0.1)
    wd = df[WIND_DIR_COL]
    df['openmeteo_wind_dir_sin'] = np.sin(2 * np.pi * wd / 360)
    df['openmeteo_wind_dir_cos'] = np.cos(2 * np.pi * wd / 360)

print("Extracting trajectory features...")
train_df = train_df.join(train_df.apply(trajectory_features, axis=1))

for df in [train_df]:
    df['rcs_speed_ratio'] = df['rcs_mean'] / (df['airspeed'] + 1e-6)
    df['alt_adjusted_wind_speed'] = (
        df[WIND_SPEED_COL] * WF
        * np.power(np.maximum(df['alt_mean'], 10) / 10.0, 0.143)
    )
    wind_dir_rad = np.deg2rad(df[WIND_DIR_COL])
    heading = df['track_heading_rad']
    angle_diff = wind_dir_rad - heading
    df['true_tailwind_component'] = df['alt_adjusted_wind_speed'] * np.cos(angle_diff)
    df['true_crosswind_component'] = np.abs(df['alt_adjusted_wind_speed'] * np.sin(angle_diff))

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

openmeteo_features = [
    'openmeteo_air_temperature_2m_c', 'openmeteo_relative_humidity_2m_percent',
    'openmeteo_dew_point_2m_c', 'openmeteo_precipitation_mm',
    'openmeteo_cloud_cover_percent', 'openmeteo_pressure_msl_hpa',
    'openmeteo_weather_code', 'openmeteo_wind_speed_10m_kmh',
    'openmeteo_wind_direction_10m_degrees', 'openmeteo_wind_gusts_10m_kmh',
    'openmeteo_shortwave_radiation_w_m2', 'openmeteo_sunshine_duration_s',
    'openmeteo_vapour_pressure_deficit_kpa', 'openmeteo_is_day',
    'openmeteo_wind_dir_sin', 'openmeteo_wind_dir_cos',
]

features = base_features + trajectory_feats + openmeteo_features
X = train_df[features]
y = train_df['bird_group']
classes = np.sort(y.unique())
print(f"Features: {len(features)}, Samples: {len(X)}, Classes: {len(classes)}")

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

cb_imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='median'), numeric_features),
    ('cat_imputer', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
    ]), categorical_features)
])
cb_cat_idx = [len(numeric_features) + i for i in range(len(categorical_features))]

oversampler = SMOTENC(categorical_features=cat_indices, random_state=42)
CLASS_BOOST = {'Cormorants': 3, 'Waders': 3, 'Geese': 3}

# ─── CV split — fold 0 only ───
groups = train_df['primary_observation_id']
cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
split = list(cv.split(X, y, groups))
train_idx, val_idx = split[0]

X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

print(f"\nFold 0: train={len(X_tr)}, val={len(X_va)}")
print(f"Train class dist: {y_tr.value_counts().to_dict()}")
print(f"Val class dist:   {y_va.value_counts().to_dict()}")


def compute_map(y_true, y_pred_proba, model_classes):
    """Compute macro mAP aligning model classes to global classes."""
    y_onehot = pd.get_dummies(y_true).reindex(columns=classes, fill_value=0)
    pred_df = pd.DataFrame(0.0, index=y_true.index, columns=classes)
    for j, cls in enumerate(model_classes):
        pred_df[cls] = y_pred_proba[:, j]
    return average_precision_score(y_onehot, pred_df, average='macro')


# ─── Prepare data ───
imp = clone(imputer)
X_tr_imp = imp.fit_transform(X_tr, y_tr)
X_va_imp = imp.transform(X_va)

os_step = clone(oversampler)
X_tr_res, y_tr_res = os_step.fit_resample(X_tr_imp, y_tr)
X_tr_res, y_tr_res = apply_boost(X_tr_res, y_tr_res, CLASS_BOOST)

print(f"After SMOTENC + boost: {len(X_tr_res)} samples")

# ═══════════════════════════════════════════
# 1. LightGBM — train vs val loss + mAP gap
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print(" LightGBM Overfit Diagnostic")
print("=" * 60)

lgb_model = LGBMClassifier(
    n_estimators=2500, learning_rate=0.03, num_leaves=63,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42, n_jobs=-1,
    device='gpu', verbose=-1,
)

lgb_model.fit(
    X_tr_res, y_tr_res,
    eval_set=[(X_tr_res, y_tr_res), (X_va_imp, y_va)],
    callbacks=[lgb.log_evaluation(500)],
)

lgb_results = lgb_model.evals_result_
# Keys depend on LightGBM version — find them dynamically
lgb_keys = list(lgb_results.keys())
print(f"LGB eval keys: {lgb_keys}")
lgb_train_loss = list(lgb_results[lgb_keys[0]].values())[0]
lgb_val_loss = list(lgb_results[lgb_keys[1]].values())[0]

# Find best iteration
best_lgb_iter = np.argmin(lgb_val_loss) + 1
print(f"\nBest iteration (by val loss): {best_lgb_iter} / {len(lgb_val_loss)}")
print(f"Train loss at best: {lgb_train_loss[best_lgb_iter - 1]:.4f}")
print(f"Val loss at best:   {lgb_val_loss[best_lgb_iter - 1]:.4f}")
print(f"Final train loss:   {lgb_train_loss[-1]:.4f}")
print(f"Final val loss:     {lgb_val_loss[-1]:.4f}")

# mAP gap
lgb_train_proba = lgb_model.predict_proba(X_tr_res)
lgb_val_proba = lgb_model.predict_proba(X_va_imp)

# For train mAP, need to handle boosted y labels
y_tr_res_series = pd.Series(y_tr_res, index=range(len(y_tr_res)))
lgb_train_map = compute_map(y_tr_res_series, lgb_train_proba, lgb_model.classes_)
lgb_val_map = compute_map(y_va, lgb_val_proba, lgb_model.classes_)

print(f"\nTrain mAP: {lgb_train_map:.4f}")
print(f"Val mAP:   {lgb_val_map:.4f}")
print(f"Gap:       {lgb_train_map - lgb_val_map:.4f}")

# ═══════════════════════════════════════════
# 2. CatBoost — train vs val loss + mAP gap
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print(" CatBoost Overfit Diagnostic")
print("=" * 60)

cb_imp = clone(cb_imputer)
X_tr_cb = cb_imp.fit_transform(X_tr, y_tr)
X_va_cb = cb_imp.transform(X_va)
for ci in cb_cat_idx:
    X_tr_cb[:, ci] = X_tr_cb[:, ci].astype(str)
    X_va_cb[:, ci] = X_va_cb[:, ci].astype(str)

cb_model = CatBoostClassifier(
    iterations=2500, learning_rate=0.03, depth=8, l2_leaf_reg=3,
    auto_class_weights='Balanced', random_seed=42, task_type='GPU', verbose=0,
)

cb_model.fit(
    X_tr_cb, y_tr,
    eval_set=(X_va_cb, y_va),
    use_best_model=False,
    cat_features=cb_cat_idx,
)

cb_evals = cb_model.get_evals_result()
cb_train_loss = cb_evals['learn']['MultiClass']
cb_val_loss = cb_evals['validation']['MultiClass']

best_cb_iter = np.argmin(cb_val_loss) + 1
print(f"\nBest iteration (by val loss): {best_cb_iter} / {len(cb_val_loss)}")
print(f"Train loss at best: {cb_train_loss[best_cb_iter - 1]:.4f}")
print(f"Val loss at best:   {cb_val_loss[best_cb_iter - 1]:.4f}")
print(f"Final train loss:   {cb_train_loss[-1]:.4f}")
print(f"Final val loss:     {cb_val_loss[-1]:.4f}")

cb_train_proba = cb_model.predict_proba(X_tr_cb)
cb_val_proba = cb_model.predict_proba(X_va_cb)
cb_train_map = compute_map(y_tr, cb_train_proba, cb_model.classes_)
cb_val_map = compute_map(y_va, cb_val_proba, cb_model.classes_)

print(f"\nTrain mAP: {cb_train_map:.4f}")
print(f"Val mAP:   {cb_val_map:.4f}")
print(f"Gap:       {cb_train_map - cb_val_map:.4f}")

# ═══════════════════════════════════════════
# 3. Plot train vs val loss curves
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(lgb_train_loss, label='Train', alpha=0.8)
ax.plot(lgb_val_loss, label='Val', alpha=0.8)
ax.axvline(best_lgb_iter - 1, color='red', linestyle='--', alpha=0.5, label=f'Best iter={best_lgb_iter}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Multi-Logloss')
ax.set_title('LightGBM: Train vs Val Loss')
ax.legend()

ax = axes[1]
ax.plot(cb_train_loss, label='Train', alpha=0.8)
ax.plot(cb_val_loss, label='Val', alpha=0.8)
ax.axvline(best_cb_iter - 1, color='red', linestyle='--', alpha=0.5, label=f'Best iter={best_cb_iter}')
ax.set_xlabel('Iteration')
ax.set_ylabel('MultiClass Loss')
ax.set_title('CatBoost: Train vs Val Loss')
ax.legend()

plt.tight_layout()
plt.savefig('overfit_loss_curves.png', dpi=150)
print(f"\nSaved overfit_loss_curves.png")

# ═══════════════════════════════════════════
# 4. N_estimators sweep (LightGBM)
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print(" N_estimators Sweep (LightGBM)")
print("=" * 60)

sweep_values = [250, 500, 1000, 1500, 2000, 2500, 3000, 4000]
sweep_results = []

for n_est in sweep_values:
    lgb_sweep = LGBMClassifier(
        n_estimators=n_est, learning_rate=0.03, num_leaves=63,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, n_jobs=-1,
        device='gpu', verbose=-1,
    )
    lgb_sweep.fit(X_tr_res, y_tr_res)

    tr_proba = lgb_sweep.predict_proba(X_tr_res)
    va_proba = lgb_sweep.predict_proba(X_va_imp)

    tr_map = compute_map(y_tr_res_series, tr_proba, lgb_sweep.classes_)
    va_map = compute_map(y_va, va_proba, lgb_sweep.classes_)

    sweep_results.append((n_est, tr_map, va_map))
    print(f"  n_estimators={n_est:5d}: train mAP={tr_map:.4f}, val mAP={va_map:.4f}, gap={tr_map - va_map:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
n_vals = [r[0] for r in sweep_results]
tr_maps = [r[1] for r in sweep_results]
va_maps = [r[2] for r in sweep_results]

ax.plot(n_vals, tr_maps, 'o-', label='Train mAP')
ax.plot(n_vals, va_maps, 's-', label='Val mAP')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Macro mAP')
ax.set_title('LightGBM: n_estimators vs mAP (Train vs Val)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfit_estimator_sweep.png', dpi=150)
print(f"\nSaved overfit_estimator_sweep.png")
