print("Hello from AI Cup")
#---

!echo hello
#---

import numpy as np
import pandas as pd
from shapely import wkb
import lightgbm as lgb
#---


#---

# import skyfield
#---

import pandas as pd

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

train_df = train_df.set_index("track_id")
test_df = test_df.set_index("track_id")

train_df
#---

import matplotlib.pyplot as plt

train_df['bird_group'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()
#---

from shapely import wkb

wkb_hex = train_df['trajectory'].iloc[0]
 
geom = wkb.loads(bytes.fromhex(wkb_hex))
coords = list(geom.coords)

coords
#---

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
#---

def split_xyzm(wkb_hex):
    geom = wkb.loads(bytes.fromhex(wkb_hex))
    coords = list(geom.coords)

    xs, ys, zs, RCSs = zip(*coords)

    return pd.Series({
        "x": xs,
        "y":ys,
        "z":zs,
        "RCS":RCSs
    })





# extra_train_cols = train_df['trajectory'].apply(split_xyzm)
# extra_test_cols = test_df['trajectory'].apply(split_xyzm)

# train_df = train_df.join(extra_train_cols)

# test_df = test_df.join(extra_test_cols)

# train_df['mean_RCS'] = train_df['RCS'].apply(np.mean)
# test_df['mean_RCS'] = test_df['RCS'].apply(np.mean)

test_df
#---

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
#---

# import seaborn as sns

# num_cols_to_plot = ['airspeed', 'max_z', 'mean_RCS']

# for feat in num_cols_to_plot:
#     plt.figure(figsize=(12,6))
#     sns.boxplot(data=train_df, x='bird_group', y=feat, palette='viridis', hue='bird_group')
#     plt.title(f"{feat} per bird_group")
#     plt.xticks(rotation=45)
#     plt.show()
#     plt.close()
#---

#model input X
#model targets y

# features = ['airspeed', 'min_z', 'max_z', 'mean_RCS', 'radar_bird_size']

# X = train_df[features]
# X_test = test_df[features]

features = [
    'airspeed', 'min_z', 'max_z', 'duration_s', 'radar_bird_size',
    'hour', 'month', 'is_daytime', 'alt_range', 'airspeed_per_m',
    'n_points', 'total_dist_m', 'mean_step_m', 'std_step_m',
    'lon_range', 'lat_range', 'alt_mean', 'alt_std',
    'rcs_mean', 'rcs_std', 'rcs_min', 'rcs_max', 'tortuosity', 'tortuosity_max'
]

X = train_df[features]
X_test = test_df[features]

y = train_df['bird_group']

X

print(X_test)
#---

y
#---

from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
#---

from sklearn.pipeline import Pipeline 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer  
from imblearn.pipeline import Pipeline as ImbPipeline  
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_selector
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])


# preprocessor = ColumnTransformer(
#     [('cat', OneHotEncoder(), ['radar_bird_size'])],
#     remainder='passthrough'
# )

# pipeline = Pipeline([
#     ('preprocess', preprocessor),
#     ('model', HistGradientBoostingClassifier(random_state=42))
# ])

# Preprocessing for numeric columns: impute median then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical columns: impute most frequent then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include='number')),
        ('cat', categorical_transformer, make_column_selector(dtype_include='object'))
    ])

# Final pipeline: preprocessing -> SMOTE -> ensemble classifier
# pipeline = ImbPipeline([
#     ('preprocess', preprocessor),
#     ('smote', SMOTE(random_state=42)),                     # generates synthetic samples for minority class
#     ('model', HistGradientBoostingClassifier(random_state=42))
# ])
pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', lgb_model)
])

#---

from sklearn.model_selection import StratifiedKFold

n_splits = 5

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

split = list(cv.split(X, y))

for i, (train_idx, val_idx) in enumerate(split):
    print(f"fold {i+1} train: {len(train_idx)}, val:  {len(val_idx)}" )
#---

from sklearn.base import clone

classes = np.unique(y)

oof_preds = pd.DataFrame(0.0, index=X.index, columns=classes)

test_preds = np.zeros((len(X_test), len(classes)))

for i, (train_idx, val_idx) in enumerate(split):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    pipeline_fold = clone(pipeline)

    pipeline_fold.fit(X_train, y_train)

    val_preds = pipeline_fold.predict_proba(X_val)
    oof_preds.iloc[val_idx] = val_preds

    test_preds_fold = pipeline_fold.predict_proba(X_test)

    test_preds += test_preds_fold

    print(f"Trained fold{i+1}/{n_splits}!")
    
#---

oof_preds
#---

import pandas as pd
import sklearn.metrics

def score(
    solution: pd.DataFrame, 
    submission: pd.DataFrame, 
) -> float:
    #Takes in two pandas dataframe and computes the Macro-averaged Average Precision Score
   
    # Ensure all required columns are present
    needed_columns = [
        "Clutter", 
        "Cormorants", 
        "Pigeons", 
        "Ducks", 
        "Geese", 
        "Gulls", 
        "Birds of Prey", 
        "Waders", 
        "Songbirds",
    ]
    
    # Reorder solution and submission columns/rows to match exactly
    solution = solution.loc[solution.index, needed_columns]
    submission = submission.loc[solution.index, needed_columns]

    
    # Compute the Average Precision score for all required columns
    bird_score = sklearn.metrics.average_precision_score(
        solution[needed_columns],
        submission[needed_columns],
        average='macro'
    )

    return bird_score
#---

# Get local OOF CV score
solution_df = (
    train_df
    .groupby(["track_id", "bird_group"])
    .size()
    .unstack(fill_value=0)
)

oof_score = score(solution_df, oof_preds)

print(f"OOF score: {oof_score}")
#---

from sklearn.metrics import ConfusionMatrixDisplay

# Convert OOF probabilities to hard class labels
y_pred_oof = classes[np.argmax(oof_preds, axis=1)]

# Plot Normalized Confusion Matrix
fig, ax = plt.subplots(figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y, 
    y_pred_oof, 
    display_labels=classes,
    cmap="Blues",
    normalize='true',
    values_format=".2f",
    xticks_rotation=45,
    ax=ax
)

plt.title("Normalized Confusion Matrix (OOF)")
plt.tight_layout()
plt.show()
plt.close()

#---

submission_df = pd.DataFrame(
    test_preds / n_splits,
    index=X_test.index,
    columns=classes
)

submission_df.to_csv('submission.csv')

print("Saved to submission.csv")

submission_df.head()
#---

