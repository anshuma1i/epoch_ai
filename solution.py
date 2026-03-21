"""
AI Cup 2026 — Bird Radar Track Classification
Trains a LightGBM model with group-aware CV, generates submission.csv.
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
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import ParameterGrid
from catboost import CatBoostClassifier

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
parser.add_argument('--boost-weak', type=float, default=0, metavar='MULT',
                    help='Default sample weight multiplier for weak classes. E.g. --boost-weak 3')
parser.add_argument('--boost-cormorants', type=float, default=0, metavar='MULT',
                    help='Override boost multiplier for Cormorants (default: use --boost-weak)')
parser.add_argument('--boost-waders', type=float, default=0, metavar='MULT',
                    help='Override boost multiplier for Waders (default: use --boost-weak)')
parser.add_argument('--boost-geese', type=float, default=0, metavar='MULT',
                    help='Override boost multiplier for Geese (default: use --boost-weak)')
parser.add_argument('--grid-search', action='store_true',
                    help='Run grid search over hyperparameters (slow)')
parser.add_argument('--ensemble', action='store_true',
                    help='Ensemble LightGBM + CatBoost (simple average)')
parser.add_argument('--two-stage', action='store_true',
                    help='Two-stage: binary Gull detector + 8-class non-Gull classifier')
args = parser.parse_args()

# ─────────────────────────────────────────────
# 1. Load Data (KNMI-enriched)
# ─────────────────────────────────────────────
print("Loading KNMI-enriched datasets...")
train_df = pd.read_csv("dataset/train_with_knmi_286.csv").set_index("track_id")
test_df = pd.read_csv("dataset/test_with_knmi_286.csv").set_index("track_id")
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

    # RCS range
    rcs_range = np.nanmax(rcs) - np.nanmin(rcs)

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

    # Wind-relative features
    df['headwind_component'] = df['airspeed'] - df['knmi_286_wind_speed_at_observation_mps']
    df['airspeed_wind_ratio'] = df['airspeed'] / (df['knmi_286_hourly_mean_wind_speed_mps'] + 0.1)

# Trajectory features
print("Extracting trajectory features for train_df...")
train_df = train_df.join(train_df.apply(trajectory_features, axis=1))
print("Extracting trajectory features for test_df...")
test_df = test_df.join(test_df.apply(trajectory_features, axis=1))

# ─────────────────────────────────────────────
# 4. Feature List
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

knmi_features = [
    'knmi_286_wind_direction_degrees',
    'knmi_286_hourly_mean_wind_speed_mps',
    'knmi_286_wind_speed_at_observation_mps',
    'knmi_286_max_wind_gust_mps',
    'knmi_286_air_temperature_c',
    # knmi_286_min_air_temperature_last_6h_c excluded — 87% NaN
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
]

features = base_features + trajectory_feats + knmi_features

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
if args.passthrough:
    oversampler = 'passthrough'
    print("Oversampler: passthrough (no oversampling)")
elif args.smotetomek:
    oversampler = SMOTETomek(
        smote=SMOTENC(categorical_features=cat_indices, random_state=42),
        random_state=42
    )
    print("Oversampler: SMOTETomek (SMOTENC + TomekLinks)")
else:
    oversampler = SMOTENC(categorical_features=cat_indices, random_state=42)
    print("Oversampler: SMOTENC (default)")

lgb_params = dict(
    n_estimators=1000,
    learning_rate=0.05,
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
# 7. Group-Aware Cross-Validation + Training
# ─────────────────────────────────────────────
n_splits = 10
groups = train_df['primary_observation_id']
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
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


# ─────────────────────────────────────────────
# 7b. Grid Search (optional)
# ─────────────────────────────────────────────
if args.grid_search:
    print("\n" + "=" * 50)
    print("GRID SEARCH MODE")
    print("=" * 50)

    # Oversampler options for grid search
    oversampler_options = {
        'smotenc': SMOTENC(categorical_features=cat_indices, random_state=42),
        'smotetomek': SMOTETomek(
            smote=SMOTENC(categorical_features=cat_indices, random_state=42),
            random_state=42
        ),
        'passthrough': 'passthrough',
    }

    param_grid = {
        'model__n_estimators': [300, 500, 1000],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__num_leaves': [31, 63, 127],
        'oversampler': list(oversampler_options.values()),
    }

    # Optionally include focal loss in the search
    if args.focal_loss:
        param_grid['model__objective'] = ['multiclass', focal_loss_lgb]

    grid = list(ParameterGrid(param_grid))
    print(f"Total configurations: {len(grid)}")

    best_score = -1
    best_params = None
    best_oof = None
    best_test = None

    for i, params in enumerate(grid):
        print(f"\n[{i+1}/{len(grid)}] {params}")
        pipeline.set_params(**params)
        oof, t_preds = run_cv(pipeline, X, y, split, classes, X_test, boost_weak_mult)

        # Evaluate
        solution_df = (
            train_df.reset_index()
            .groupby(["track_id", "bird_group"]).size()
            .unstack(fill_value=0)
        )
        oof_aligned = oof.loc[solution_df.index, solution_df.columns]
        needed_columns = [
            "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
            "Gulls", "Birds of Prey", "Waders", "Songbirds",
        ]
        s = average_precision_score(solution_df[needed_columns], oof_aligned[needed_columns], average='macro')
        print(f"  → mAP: {s:.4f}")

        if s > best_score:
            best_score = s
            best_params = params
            best_oof = oof
            best_test = t_preds

    print(f"\n{'='*50}")
    print(f"Best mAP: {best_score:.4f}")
    print(f"Best params: {best_params}")
    print(f"{'='*50}")

    oof_preds = best_oof
    test_preds = best_test

else:
    # ── Single run ──
    oof_preds, test_preds = run_cv(pipeline, X, y, split, classes, X_test, boost_weak_mult)

# ─────────────────────────────────────────────
# 7c. CatBoost Ensemble (optional)
# ─────────────────────────────────────────────
if args.ensemble:
    print("\n" + "=" * 50)
    print("CATBOOST ENSEMBLE")
    print("=" * 50)

    # After ColumnTransformer, numeric cols come first, then cat cols
    cb_cat_idx = [len(numeric_features) + i for i in range(len(categorical_features))]

    cb_params = dict(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        auto_class_weights='Balanced',
        random_seed=42,
        task_type='GPU',
        verbose=0,
    )

    cb_oof = pd.DataFrame(0.0, index=X.index, columns=classes)
    cb_test = np.zeros((len(X_test), len(classes)))

    # CatBoost needs categoricals as strings, not ordinal-encoded floats
    cb_imputer = ColumnTransformer([
        ('num_imputer', SimpleImputer(strategy='median'), numeric_features),
        ('cat_imputer', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
        ]), categorical_features)
    ])

    print(f"Training {len(split)}-fold CatBoost...")
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

        cb = CatBoostClassifier(**cb_params)
        cb.fit(X_tr, y_train_fold, eval_set=(X_va, y.iloc[val_idx]),
               early_stopping_rounds=50, cat_features=cb_cat_idx)

        val_proba = cb.predict_proba(X_va)
        test_proba = cb.predict_proba(X_te)

        cb_oof.iloc[val_idx] = val_proba
        cb_test += test_proba

        fold_ap = average_precision_score(
            pd.get_dummies(y.iloc[val_idx]).reindex(columns=classes, fill_value=0),
            val_proba, average='macro'
        )
        print(f"  Fold {i+1}/{len(split)} — val mAP: {fold_ap:.4f}")

    cb_test /= len(split)

    # Print CatBoost standalone score
    solution_df_tmp = (
        train_df.reset_index()
        .groupby(["track_id", "bird_group"]).size()
        .unstack(fill_value=0)
    )
    needed_columns_tmp = [
        "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
        "Gulls", "Birds of Prey", "Waders", "Songbirds",
    ]
    cb_oof_aligned = cb_oof.loc[solution_df_tmp.index, solution_df_tmp.columns]
    cb_map = average_precision_score(
        solution_df_tmp[needed_columns_tmp],
        cb_oof_aligned[needed_columns_tmp],
        average='macro'
    )
    print(f"\n  CatBoost standalone mAP: {cb_map:.4f}")

    # Simple average ensemble
    oof_preds = (oof_preds + cb_oof) / 2
    test_preds = (test_preds + cb_test) / 2
    print("  Ensembled LightGBM + CatBoost (simple average)")

# ─────────────────────────────────────────────
# 7d. Two-Stage Gull Detector (optional)
# ─────────────────────────────────────────────
if args.two_stage:
    print("\n" + "=" * 50)
    print("TWO-STAGE GULL DETECTOR")
    print("=" * 50)

    non_gull_classes = np.sort([c for c in classes if c != 'Gulls'])

    # Stage 1: Binary Gull vs non-Gull
    y_binary = (y == 'Gulls').astype(int)

    lgb_binary = LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, n_jobs=-1,
        device='gpu', verbose=-1,
    )
    binary_pipeline = ImbPipeline([
        ('imputer', clone(imputer)),
        ('oversampler', 'passthrough'),
        ('model', lgb_binary)
    ])

    # Stage 2: 8-class on non-Gull samples only
    lgb_multi = LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, n_jobs=-1,
        device='gpu', verbose=-1,
    )
    multi_pipeline = ImbPipeline([
        ('imputer', clone(imputer)),
        ('oversampler', 'passthrough'),
        ('model', lgb_multi)
    ])

    ts_oof = pd.DataFrame(0.0, index=X.index, columns=classes)
    ts_test = np.zeros((len(X_test), len(classes)))

    print(f"Training {len(split)}-fold two-stage...")
    for i, (train_idx, val_idx) in enumerate(split):
        X_tr = X.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        # ── Stage 1: Gull binary ──
        y_tr_bin = (y_tr == 'Gulls').astype(int)
        pipe1 = clone(binary_pipeline)
        pipe1.fit(X_tr, y_tr_bin)
        # p(Gull) is column index 1
        val_p_gull = pipe1.predict_proba(X_va)[:, 1]
        test_p_gull = pipe1.predict_proba(X_test)[:, 1]

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
    needed_ts = [
        "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
        "Gulls", "Birds of Prey", "Waders", "Songbirds",
    ]
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
# 8. Evaluation
# ─────────────────────────────────────────────
needed_columns = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds",
]

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
# 9. Generate Submission
# ─────────────────────────────────────────────
submission_df = pd.DataFrame(
    test_preds,
    index=X_test.index,
    columns=classes
)
submission_df.index.name = 'track_id'
submission_df.to_csv('submission.csv')
print(f"\nSaved submission.csv ({len(submission_df)} rows)")
