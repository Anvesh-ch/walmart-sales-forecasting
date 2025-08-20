import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.logging_utils import get_logger
from src.utils.io_utils import safe_write_parquet
from src.paths import DATA_INTERIM, GLOBAL_PARQUET

logger = get_logger(__name__)

def merge_train_features_stores(train_df: pd.DataFrame, features_df: pd.DataFrame, 
                              stores_df: pd.DataFrame) -> pd.DataFrame:
    """Merge train, features, and stores data on Store and Date."""
    
    logger.info("Starting data merge process")
    
    # First merge: train with features on Store and Date
    logger.info("Merging train with features data")
    merged_df = train_df.merge(
        features_df, 
        on=['Store', 'Date'], 
        how='left',
        suffixes=('', '_features')
    )
    
    # Check for missing features data
    missing_features = merged_df[merged_df['Temperature'].isnull()].shape[0]
    if missing_features > 0:
        logger.warning(f"Found {missing_features} rows with missing features data")
    
    # Second merge: add stores information
    logger.info("Merging with stores data")
    merged_df = merged_df.merge(
        stores_df,
        on='Store',
        how='left'
    )
    
    # Check for missing stores data
    missing_stores = merged_df[merged_df['Type'].isnull()].shape[0]
    if missing_stores > 0:
        logger.warning(f"Found {missing_stores} rows with missing stores data")
    
    # Remove duplicate columns from features merge
    if 'IsHoliday_features' in merged_df.columns:
        merged_df = merged_df.drop('IsHoliday_features', axis=1)
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    logger.info(f"Columns: {list(merged_df.columns)}")
    
    return merged_df

def validate_merged_data(merged_df: pd.DataFrame) -> bool:
    """Validate the merged dataset."""
    
    logger.info("Validating merged dataset")
    
    # Check for required columns
    required_cols = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday', 
                    'Temperature', 'Fuel_Price', 'Type', 'Size']
    missing_cols = set(required_cols) - set(merged_df.columns)
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for missing values in key columns
    key_cols = ['Store', 'Dept', 'Date', 'Weekly_Sales']
    missing_counts = merged_df[key_cols].isnull().sum()
    
    if missing_counts.sum() > 0:
        logger.warning(f"Missing values in key columns:\n{missing_counts}")
    
    # Check for duplicates
    duplicates = merged_df.duplicated(subset=['Store', 'Dept', 'Date']).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate Store-Dept-Date combinations")
    
    # Check data integrity
    store_dept_combinations = merged_df.groupby(['Store', 'Dept']).size()
    logger.info(f"Store-Dept combinations: {len(store_dept_combinations)}")
    
    # Check date range
    date_range = merged_df['Date'].max() - merged_df['Date'].min()
    logger.info(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()} ({date_range.days} days)")
    
    logger.info("Merged data validation completed successfully")
    return True

def save_merged_data(merged_df: pd.DataFrame, output_path: str = None) -> None:
    """Save merged data to parquet format."""
    
    if output_path is None:
        output_path = GLOBAL_PARQUET
    
    logger.info(f"Saving merged data to {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    safe_write_parquet(merged_df, output_path)
    
    logger.info(f"Merged data saved successfully to {output_path}")

def create_store_dept_mapping(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create a mapping of Store-Dept combinations with metadata."""
    
    logger.info("Creating Store-Dept mapping")
    
    mapping = merged_df.groupby(['Store', 'Dept']).agg({
        'Date': ['min', 'max', 'count'],
        'Weekly_Sales': ['mean', 'std', 'sum'],
        'Type': 'first',
        'Size': 'first'
    }).reset_index()
    
    # Flatten column names
    mapping.columns = ['Store', 'Dept', 'first_date', 'last_date', 'periods', 
                      'avg_sales', 'std_sales', 'total_sales', 'Type', 'Size']
    
    # Calculate additional metrics
    mapping['date_range_days'] = (mapping['last_date'] - mapping['first_date']).dt.days
    mapping['avg_weekly_sales'] = mapping['total_sales'] / mapping['periods']
    
    # Sort by total sales
    mapping = mapping.sort_values('total_sales', ascending=False).reset_index(drop=True)
    
    logger.info(f"Created mapping for {len(mapping)} Store-Dept combinations")
    return mapping

def main(train_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """Main function to merge all data sources."""
    
    logger.info("Starting data merge pipeline")
    
    try:
        # Merge all data sources
        merged_df = merge_train_features_stores(train_df, features_df, stores_df)
        
        # Validate merged data
        if not validate_merged_data(merged_df):
            raise ValueError("Merged data validation failed")
        
        # Create Store-Dept mapping
        mapping = create_store_dept_mapping(merged_df)
        
        # Save merged data
        save_merged_data(merged_df)
        
        # Save mapping to interim directory
        mapping_path = DATA_INTERIM / "store_dept_mapping.parquet"
        safe_write_parquet(mapping, mapping_path)
        
        logger.info("Data merge pipeline completed successfully")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Data merge pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # This module is typically called from other pipelines
    pass
