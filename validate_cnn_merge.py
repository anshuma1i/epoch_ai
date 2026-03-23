import pandas as pd

LOG_PATH = "cnn_merge_validation.txt"
lines = []

def log(msg):
    print(msg)
    lines.append(msg)

def validate():
    log("=== Loading Datasets ===")
    try:
        train_orig = pd.read_csv("dataset/train.csv")
        test_orig = pd.read_csv("dataset/test.csv")
        
        train_merged = pd.read_csv("dataset/train_with_openmeteo_cnn.csv")
        test_merged = pd.read_csv("dataset/test_with_openmeteo_cnn.csv")
    except Exception as e:
        log(f"Failed to load datasets: {e}")
        return
        
    log("\n=== Row Count Check ===")
    log(f"Train: orig={len(train_orig)}, merged={len(train_merged)}, match={len(train_orig)==len(train_merged)}")
    log(f"Test:  orig={len(test_orig)},  merged={len(test_merged)},  match={len(test_orig)==len(test_merged)}")

    log("\n=== track_id Integrity ===")
    log(f"Train IDs match: {set(train_orig['track_id']) == set(train_merged['track_id'])}")
    log(f"Test IDs match:  {set(test_orig['track_id']) == set(test_merged['track_id'])}")
    log(f"Train duplicate IDs in merged: {train_merged['track_id'].duplicated().sum()}")
    log(f"Test duplicate IDs in merged:  {test_merged['track_id'].duplicated().sum()}")

    log("\n=== Row Order Integrity ===")
    log(f"Train row order matches exactly: {train_orig['track_id'].equals(train_merged['track_id'])}")
    log(f"Test row order matches exactly: {test_orig['track_id'].equals(test_merged['track_id'])}")

    log("\n=== Feature Null Rates (CNN Embeddings) ===")
    cnn_cols_train = [c for c in train_merged.columns if c.startswith('cnn_')]
    cnn_cols_test = [c for c in test_merged.columns if c.startswith('cnn_')]
    
    log(f"Found {len(cnn_cols_train)} CNN embedding columns in train.")
    
    if len(cnn_cols_train) == 0:
        log("ERROR: No CNN columns found!")
    else:
        # Check overall NaN across all CNN columns
        train_nulls = train_merged[cnn_cols_train].isna().sum().sum()
        test_nulls = test_merged[cnn_cols_test].isna().sum().sum()
        
        total_train_cells = len(train_merged) * len(cnn_cols_train)
        total_test_cells = len(test_merged) * len(cnn_cols_test)
        
        log(f"Train total CNN nulls: {train_nulls} / {total_train_cells} ({train_nulls/total_train_cells*100:.2f}%)")
        log(f"Test total CNN nulls:  {test_nulls} / {total_test_cells} ({test_nulls/total_test_cells*100:.2f}%)")

if __name__ == '__main__':
    validate()
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLog saved to {LOG_PATH}")
