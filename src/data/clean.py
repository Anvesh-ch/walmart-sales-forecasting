import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.logging_utils import get_logger
from src.utils.ts_utils import parse_dates, get_holiday_flags, create_time_features

logger = get_logger(__name__)

def handle_markdown_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle MarkDown columns according to specifications."""
    
    logger.info("Handling MarkDown columns")
    
    df = df.copy()
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    
    # Check if MarkDown columns exist
    existing_markdown_cols = [col for col in markdown_cols if col in df.columns]
    if not existing_markdown_cols:
        logger.warning("No MarkDown columns found in dataset")
        return df
    
    logger.info(f"Processing MarkDown columns: {existing_markdown_cols}")
    
    # For each store, handle MarkDown forward-filling
    for store in df['Store'].unique():
        store_mask = df['Store'] == store
        
        for markdown_col in existing_markdown_cols:
            # Get store-specific MarkDown data
            store_markdown = df.loc[store_mask, markdown_col]
            
            # If not a promo week (MarkDown > 0), set to 0
            # Otherwise, forward-fill up to 8 weeks, remaining NaNs to 0
            promo_mask = store_markdown > 0
            
            if promo_mask.any():
                # Forward-fill promo values up to 8 weeks
                promo_series = store_markdown.copy()
                promo_series = promo_series.fillna(method='ffill', limit=8)
                
                # Fill remaining NaNs with 0
                promo_series = promo_series.fillna(0)
                
                # Update the dataframe
                df.loc[store_mask, markdown_col] = promo_series
            else:
                # No promo weeks, set all to 0
                df.loc[store_mask, markdown_col] = 0
    
    # Create MarkDown sum column
    df['MarkDownSum'] = df[existing_markdown_cols].sum(axis=1)
    
    logger.info("MarkDown columns processed successfully")
    return df

def standardize_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize CPI and Unemployment within each store over time."""
    
    logger.info("Standardizing economic features")
    
    df = df.copy()
    
    if 'CPI' in df.columns:
        logger.info("Standardizing CPI by store")
        df['CPI_std'] = df.groupby('Store')['CPI'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        df['CPI_std'] = df['CPI_std'].fillna(0)
    
    if 'Unemployment' in df.columns:
        logger.info("Standardizing Unemployment by store")
        df['Unemployment_std'] = df.groupby('Store')['Unemployment'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        df['Unemployment_std'] = df['Unemployment_std'].fillna(0)
    
    logger.info("Economic features standardized successfully")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    
    logger.info("Handling missing values")
    
    df = df.copy()
    
    # Report missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Missing values before cleaning:\n{missing_counts[missing_counts > 0]}")
    
    # Handle specific columns
    if 'Temperature' in df.columns:
        # Forward-fill temperature within store
        df['Temperature'] = df.groupby('Store')['Temperature'].fillna(method='ffill')
        df['Temperature'] = df.groupby('Store')['Temperature'].fillna(method='bfill')
    
    if 'Fuel_Price' in df.columns:
        # Forward-fill fuel price within store
        df['Fuel_Price'] = df.groupby('Store')['Fuel_Price'].fillna(method='ffill')
        df['Fuel_Price'] = df.groupby('Store')['Fuel_Price'].fillna(method='bfill')
    
    if 'Size' in df.columns:
        # Size is constant per store, forward-fill
        df['Size'] = df.groupby('Store')['Size'].fillna(method='ffill')
    
    if 'Type' in df.columns:
        # Type is constant per store, forward-fill
        df['Type'] = df.groupby('Store')['Type'].fillna(method='ffill')
    
    # For remaining numeric columns, fill with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # For categorical columns, fill with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # Report final missing values
    final_missing = df.isnull().sum()
    if final_missing.sum() > 0:
        logger.warning(f"Missing values after cleaning:\n{final_missing[final_missing > 0]}")
    else:
        logger.info("All missing values handled successfully")
    
    return df

def handle_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """Handle outliers in numeric columns."""
    
    logger.info(f"Handling outliers using {method} method")
    
    df = df.copy()
    
    # Select numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Store', 'Dept']]
    
    outlier_counts = {}
    
    for col in numeric_cols:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    # Cap outliers at bounds
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    outlier_counts[col] = outliers
                    
                    logger.info(f"Column {col}: {outliers} outliers capped using IQR method")
    
    if outlier_counts:
        logger.info(f"Outlier handling completed. Columns affected: {list(outlier_counts.keys())}")
    else:
        logger.info("No outliers detected or handled")
    
    return df

def create_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean features for modeling."""
    
    logger.info("Creating clean features")
    
    df = df.copy()
    
    # Parse dates and add time features
    df = parse_dates(df, 'Date')
    df = create_time_features(df, 'Date')
    df = get_holiday_flags(df, 'Date')
    
    # Handle MarkDown columns
    df = handle_markdown_columns(df)
    
    # Standardize economic features
    df = standardize_economic_features(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Ensure data types are correct
    df['Store'] = df['Store'].astype(int)
    df['Dept'] = df['Dept'].astype(int)
    df['IsHoliday'] = df['IsHoliday'].astype(bool)
    
    # Sort by Store, Dept, Date
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    logger.info(f"Clean features created. Final shape: {df.shape}")
    return df

def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """Validate the cleaned dataset."""
    
    logger.info("Validating cleaned dataset")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.error(f"Found missing values after cleaning:\n{missing_counts[missing_counts > 0]}")
        return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = df[numeric_cols].isin([np.inf, -np.inf]).sum()
    if inf_counts.sum() > 0:
        logger.error(f"Found infinite values after cleaning:\n{inf_counts[inf_counts > 0]}")
        return False
    
    # Check for negative sales
    if 'Weekly_Sales' in df.columns:
        negative_sales = (df['Weekly_Sales'] < 0).sum()
        if negative_sales > 0:
            logger.warning(f"Found {negative_sales} negative sales values")
    
    # Check data types
    expected_types = {
        'Store': 'int64',
        'Dept': 'int64',
        'Date': 'datetime64[ns]',
        'Weekly_Sales': 'float64',
        'IsHoliday': 'bool'
    }
    
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if expected_type not in actual_type:
                logger.warning(f"Column {col}: expected {expected_type}, got {actual_type}")
    
    logger.info("Cleaned data validation completed successfully")
    return True

def main(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to clean the merged dataset."""
    
    logger.info("Starting data cleaning pipeline")
    
    try:
        # Create clean features
        cleaned_df = create_clean_features(df)
        
        # Validate cleaned data
        if not validate_cleaned_data(cleaned_df):
            raise ValueError("Cleaned data validation failed")
        
        logger.info("Data cleaning pipeline completed successfully")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Data cleaning pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # This module is typically called from other pipelines
    pass
