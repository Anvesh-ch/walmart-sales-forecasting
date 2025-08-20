import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from src.logging_utils import get_logger
from src.utils.io_utils import safe_read_csv, check_file_exists
from src.schema import validate_dataframe, TRAIN_SCHEMA, FEATURES_SCHEMA, STORES_SCHEMA
from src.paths import TRAIN_CSV, FEATURES_CSV, STORES_CSV

logger = get_logger(__name__)

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files and validate schemas."""
    
    logger.info("Loading raw data files")
    
    # Check if files exist
    for file_path, file_name in [(TRAIN_CSV, "train.csv"), 
                                 (FEATURES_CSV, "features.csv"), 
                                 (STORES_CSV, "stores.csv")]:
        if not check_file_exists(file_path):
            raise FileNotFoundError(f"Required file {file_name} not found in data/raw/")
    
    # Load train data
    logger.info("Loading train.csv")
    train_df = safe_read_csv(TRAIN_CSV)
    
    # Load features data
    logger.info("Loading features.csv")
    features_df = safe_read_csv(FEATURES_CSV)
    
    # Load stores data
    logger.info("Loading stores.csv")
    stores_df = safe_read_csv(STORES_CSV)
    
    # Validate schemas
    logger.info("Validating data schemas")
    validate_dataframe(train_df, TRAIN_SCHEMA, "train")
    validate_dataframe(features_df, FEATURES_SCHEMA, "features")
    validate_dataframe(stores_df, STORES_SCHEMA, "stores")
    
    logger.info(f"Raw data loaded successfully:")
    logger.info(f"  Train: {train_df.shape}")
    logger.info(f"  Features: {features_df.shape}")
    logger.info(f"  Stores: {stores_df.shape}")
    
    return train_df, features_df, stores_df

def check_data_quality(train_df: pd.DataFrame, features_df: pd.DataFrame, 
                      stores_df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality and report issues."""
    
    logger.info("Checking data quality")
    
    quality_report = {}
    
    # Check for missing values
    quality_report['train_missing'] = train_df.isnull().sum().to_dict()
    quality_report['features_missing'] = features_df.isnull().sum().to_dict()
    quality_report['stores_missing'] = stores_df.isnull().sum().to_dict()
    
    # Check for duplicates
    quality_report['train_duplicates'] = train_df.duplicated().sum()
    quality_report['features_duplicates'] = features_df.duplicated().sum()
    quality_report['stores_duplicates'] = stores_df.duplicated().sum()
    
    # Check date ranges
    quality_report['train_date_range'] = {
        'min': train_df['Date'].min(),
        'max': train_df['Date'].max()
    }
    quality_report['features_date_range'] = {
        'min': features_df['Date'].min(),
        'max': features_df['Date'].max()
    }
    
    # Check store coverage
    quality_report['train_stores'] = train_df['Store'].nunique()
    quality_report['features_stores'] = features_df['Store'].nunique()
    quality_report['stores_total'] = stores_df['Store'].nunique()
    
    # Check department coverage
    quality_report['train_depts'] = train_df['Dept'].nunique()
    
    # Check for data consistency issues
    train_stores = set(train_df['Store'].unique())
    features_stores = set(features_df['Store'].unique())
    stores_list = set(stores_df['Store'].unique())
    
    quality_report['missing_stores_in_features'] = train_stores - features_stores
    quality_report['missing_stores_in_stores'] = train_stores - stores_list
    
    logger.info("Data quality check completed")
    logger.info(f"Quality report: {quality_report}")
    
    return quality_report

def prepare_data_for_merge(train_df: pd.DataFrame, features_df: pd.DataFrame, 
                          stores_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare data for merging by ensuring consistent data types and handling missing values."""
    
    logger.info("Preparing data for merging")
    
    # Convert dates to datetime
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    # Ensure consistent data types
    train_df['Store'] = train_df['Store'].astype(int)
    train_df['Dept'] = train_df['Dept'].astype(int)
    features_df['Store'] = features_df['Store'].astype(int)
    stores_df['Store'] = stores_df['Store'].astype(int)
    
    # Sort by date for time series operations
    train_df = train_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    features_df = features_df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    logger.info("Data preparation completed")
    return train_df, features_df, stores_df

def main():
    """Main function to load and validate raw data."""
    
    logger.info("Starting data ingestion process")
    
    try:
        # Load raw data
        train_df, features_df, stores_df = load_raw_data()
        
        # Check data quality
        quality_report = check_data_quality(train_df, features_df, stores_df)
        
        # Prepare data for merging
        train_df, features_df, stores_df = prepare_data_for_merge(train_df, features_df, stores_df)
        
        logger.info("Data ingestion completed successfully")
        
        return train_df, features_df, stores_df
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
