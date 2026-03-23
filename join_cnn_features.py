import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description='Merge CNN features into weather dataset.')
    parser.add_argument('--train-weather', type=str, default='dataset/train_with_openmeteo.csv')
    parser.add_argument('--test-weather', type=str, default='dataset/test_with_openmeteo.csv')
    parser.add_argument('--train-cnn', type=str, default='dataset/train_cnn_features.csv')
    parser.add_argument('--test-cnn', type=str, default='dataset/test_cnn_features.csv')
    parser.add_argument('--out-train', type=str, default='dataset/train_with_openmeteo_cnn.csv')
    parser.add_argument('--out-test', type=str, default='dataset/test_with_openmeteo_cnn.csv')
    return parser.parse_args()

def safe_merge(base_df, features_df, name):
    logger.info(f"[{name}] Base shape: {base_df.shape}, Features shape: {features_df.shape}")
    
    # Ensure track_id is the index for both
    if 'track_id' in base_df.columns:
        base_df = base_df.set_index('track_id')
    if 'track_id' in features_df.columns:
        features_df = features_df.set_index('track_id')
        
    merged_df = base_df.join(features_df, how='left')
    logger.info(f"[{name}] Merged shape: {merged_df.shape}")
    
    # Reset index so track_id becomes a column again for downstream
    merged_df = merged_df.reset_index()
    return merged_df

def main(args):
    logger.info("Loading existing OpenMeteo datasets...")
    train_w = pd.read_csv(args.train_weather)
    test_w = pd.read_csv(args.test_weather)
    
    logger.info("Loading new CNN feature datasets...")
    train_cnn = pd.read_csv(args.train_cnn)
    test_cnn = pd.read_csv(args.test_cnn)
    
    logger.info("Merging...")
    train_out = safe_merge(train_w, train_cnn, "Train")
    test_out = safe_merge(test_w, test_cnn, "Test")
    
    logger.info(f"Saving merged output to {args.out_train}")
    train_out.to_csv(args.out_train, index=False)
    
    logger.info(f"Saving merged output to {args.out_test}")
    test_out.to_csv(args.out_test, index=False)
    
    logger.info("Merge complete.")

if __name__ == '__main__':
    args = setup_args()
    main(args)
