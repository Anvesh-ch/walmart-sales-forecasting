#!/usr/bin/env python3
"""
Data preparation pipeline for Walmart Sales Forecasting.

This pipeline:
1. Loads raw CSV files
2. Merges train, features, and stores data
3. Cleans and validates the data
4. Engineers features for modeling
5. Saves processed data
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logging_utils import setup_logging, get_logger
from src.paths import DATA_PROCESSED, DATA_INTERIM
from src.data.ingest import load_raw_data, check_data_quality
from src.data.merge import merge_train_features_stores, validate_merged_data
from src.data.clean import create_clean_features, validate_cleaned_data
from src.features.build_features import prepare_features_for_modeling
from src.utils.io_utils import safe_write_parquet

def main():
    """Main data preparation pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare data for Walmart sales forecasting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('prepare_data', 'INFO')
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    try:
        logger.info("Starting data preparation pipeline")
        
        # Step 1: Load raw data
        logger.info("Step 1: Loading raw data")
        train_df, features_df, stores_df = load_raw_data()
        
        # Step 2: Check data quality
        logger.info("Step 2: Checking data quality")
        quality_report = check_data_quality(train_df, features_df, stores_df)
        
        # Step 3: Merge data sources
        logger.info("Step 3: Merging data sources")
        merged_df = merge_train_features_stores(train_df, features_df, stores_df)
        
        # Validate merged data
        if not validate_merged_data(merged_df):
            raise ValueError("Merged data validation failed")
        
        # Step 4: Clean data
        logger.info("Step 4: Cleaning data")
        cleaned_df = create_clean_features(merged_df)
        
        # Validate cleaned data
        if not validate_cleaned_data(cleaned_df):
            raise ValueError("Cleaned data validation failed")
        
        # Step 5: Engineer features
        logger.info("Step 5: Engineering features")
        featured_df = prepare_features_for_modeling(cleaned_df)
        
        # Step 6: Save processed data
        logger.info("Step 6: Saving processed data")
        
        # Save global dataset
        global_output_path = DATA_PROCESSED / "global.parquet"
        safe_write_parquet(featured_df, global_output_path)
        logger.info(f"Saved global dataset to {global_output_path}")
        
        # Save intermediate datasets
        interim_output_path = DATA_INTERIM / "cleaned_data.parquet"
        safe_write_parquet(cleaned_df, interim_output_path)
        logger.info(f"Saved cleaned data to {interim_output_path}")
        
        # Save data summary
        summary = {
            'original_shape': merged_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'featured_shape': featured_df.shape,
            'n_stores': featured_df['Store'].nunique(),
            'n_departments': featured_df['Dept'].nunique(),
            'date_range': (featured_df['Date'].min(), featured_df['Date'].max()),
            'total_observations': len(featured_df),
            'feature_columns': len([col for col in featured_df.columns 
                                  if col not in ['Store', 'Dept', 'Date', 'Weekly_Sales']])
        }
        
        summary_path = DATA_INTERIM / "data_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Data preparation pipeline completed successfully")
        logger.info(f"Final dataset shape: {featured_df.shape}")
        logger.info(f"Features created: {summary['feature_columns']}")
        
        return featured_df
        
    except Exception as e:
        logger.error(f"Data preparation pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
