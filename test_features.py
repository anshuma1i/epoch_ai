import pandas as pd
import numpy as np
from shapely import wkb

def parse_trajectory(hex_str):
    if not isinstance(hex_str, str) or len(hex_str) == 0: return []
    try:
        geom = wkb.loads(bytes.fromhex(hex_str) if not hex_str.startswith('\x01') else hex_str, hex=True)
        return list(geom.coords) if geom.geom_type == 'LineString' else []
    except: return []

def trajectory_features(row):
    coords = parse_trajectory(row['trajectory'])
    n = len(coords)
    if n == 0:
        return pd.Series({
            'n_points': 0, 'total_dist_m': 0.0, 'mean_step_m': 0.0, 'std_step_m': 0.0,
            'lon_range': 0.0, 'lat_range': 0.0, 'alt_mean': np.nan, 'alt_std': np.nan,
            'rcs_mean': np.nan, 'rcs_std': np.nan, 'rcs_min': np.nan, 'rcs_max': np.nan, 'rcs_range': np.nan,
            'tortuosity': 0.0, 'tortuosity_max': 0.0, 'sharp_turn_ratio': 0.0,
            'straightness': 0.0, 'sinuosity': 0.0, 'lon_mean': np.nan, 'lat_mean': np.nan,
            'track_heading_rad': 0.0, 'alt_climb_rate': 0.0, 'alt_descent_rate': 0.0, 'alt_variability': 0.0,
            'speed_mean': np.nan, 'speed_std': np.nan, 'speed_max': np.nan, 'speed_cv': np.nan,
            'accel_mean': np.nan, 'accel_std': np.nan
        })

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
    displacement = np.sqrt(((lons[-1] - lons[0]) * 71000) ** 2 + ((lats[-1] - lats[0]) * 111000) ** 2)
    straightness = displacement / (total_dist + 1e-6) if total_dist > 0 else 0.0
    sinuosity = total_dist / (displacement + 1e-6)
    track_heading_rad = np.arctan2((lats[-1] - lats[0]) * 111000, (lons[-1] - lons[0]) * 71000)

    alt_arr = np.array(alts, dtype=float)
    if n > 1 and not np.all(np.isnan(alt_arr)):
        alt_changes = np.diff(alt_arr)
        alt_climb_rate, alt_descent_rate, alt_variability = np.nanmean(alt_changes), np.nanmin(alt_changes), np.nanstd(alt_changes)
    else:
        alt_climb_rate, alt_descent_rate, alt_variability = 0.0, 0.0, 0.0

    feats = {
        'n_points': n, 'total_dist_m': total_dist,
        'mean_step_m': step_dist.mean() if len(step_dist) > 0 else 0.0,
        'std_step_m': step_dist.std() if len(step_dist) > 0 else 0.0,
        'lon_range': max(lons) - min(lons), 'lat_range': max(lats) - min(lats),
        'alt_mean': np.nanmean(alts), 'alt_std': np.nanstd(alts),
        'rcs_mean': np.nanmean(rcs), 'rcs_std': np.nanstd(rcs), 'rcs_min': np.nanmin(rcs), 'rcs_max': np.nanmax(rcs),
        'rcs_range': np.nanmax(rcs) - np.nanmin(rcs),
        'tortuosity': bearing_changes.mean() if len(bearing_changes) > 0 else 0.0,
        'tortuosity_max': bearing_changes.max() if len(bearing_changes) > 0 else 0.0,
        'sharp_turn_ratio': sharp_turn_ratio, 'straightness': straightness, 'sinuosity': sinuosity,
        'lon_mean': np.mean(lons), 'lat_mean': np.mean(lats), 'track_heading_rad': track_heading_rad,
        'alt_climb_rate': alt_climb_rate, 'alt_descent_rate': alt_descent_rate, 'alt_variability': alt_variability,
    }

    # Time speed feats
    times = row.get('trajectory_time', '')
    if isinstance(times, str) and times.strip():
        t_list = [float(x) for x in times.strip('[]').split(',')]
        if len(t_list) == n and n > 1:
            dt = np.where(np.diff(t_list) == 0, 1e-6, np.diff(t_list))
            speeds = step_dist / dt
            feats.update({'speed_mean': np.mean(speeds), 'speed_std': np.std(speeds), 'speed_max': np.max(speeds),
                        'speed_cv': np.std(speeds) / (np.mean(speeds) + 1e-6)})
            if len(speeds) > 1:
                accel = np.diff(speeds) / dt[1:]
                feats.update({'accel_mean': np.mean(accel), 'accel_std': np.std(accel)})
            else:
                feats.update({'accel_mean': 0.0, 'accel_std': 0.0})
    
    # Fill Nans for missing keys
    all_keys = [
        'n_points', 'total_dist_m', 'mean_step_m', 'std_step_m',
        'lon_range', 'lat_range', 'alt_mean', 'alt_std',
        'rcs_mean', 'rcs_std', 'rcs_min', 'rcs_max', 'rcs_range',
        'tortuosity', 'tortuosity_max', 'sharp_turn_ratio',
        'straightness', 'sinuosity',
        'lon_mean', 'lat_mean', 'track_heading_rad',
        'alt_climb_rate', 'alt_descent_rate', 'alt_variability',
        'speed_mean', 'speed_std', 'speed_max', 'speed_cv',
        'accel_mean', 'accel_std',
    ]
    for k in all_keys:
        if k not in feats: feats[k] = np.nan
    return pd.Series(feats)

# Dummy test
dummy_row = {
    'trajectory': '0102000020E61000000200000000000000000000000000000000000000000000000000594000000000000000000000000000000000', # dummy linestring
    'trajectory_time': '[0.0, 1.0]'
}
results = trajectory_features(dummy_row)
print("--- Unit Test Results ---")
print(f"Generated {len(results)} features.")
print(results)
if len(results) == 30:
    print("\nSUCCESS: All 30 features generated.")
else:
    print(f"\nFAILURE: Expected 30 features, got {len(results)}.")
