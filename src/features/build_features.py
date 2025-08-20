import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.logging_utils import get_logger
from src.utils.ts_utils import create_lag_features, create_rolling_windows

logger = get_logger(__name__)

def create_lag_features_for_modeling(df: pd.DataFrame, 
                                   lags: List[int] = [1, 2, 3, 4, 6, 8, 13]) -> pd.DataFrame:
    """Create lag features for Weekly_Sales."""
    
    logger.info(f"Creating lag features: {lags}")
    
    df = df.copy()
    
    # Ensure data is sorted by Store, Dept, Date
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    # Create lag features for Weekly_Sales
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
    
    # Fill NaN values in lag features with 0
    lag_cols = [f'sales_lag_{lag}' for lag in lags]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    logger.info(f"Created {len(lag_cols)} lag features")
    return df

def create_rolling_statistics(df: pd.DataFrame, 
                            windows: List[int] = [3, 4, 6, 8, 13]) -> pd.DataFrame:
    """Create rolling mean and standard deviation features."""
    
    logger.info(f"Creating rolling statistics for windows: {windows}")
    
    df = df.copy()
    
    # Ensure data is sorted by Store, Dept, Date
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    # Create rolling features for each window
    for window in windows:
        # Rolling mean
        df[f'sales_rolling_mean_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation
        df[f'sales_rolling_std_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Rolling median
        df[f'sales_rolling_median_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(window=window, min_periods=1).median()
        )
    
    # Fill NaN values with 0
    rolling_cols = []
    for window in windows:
        rolling_cols.extend([
            f'sales_rolling_mean_{window}',
            f'sales_rolling_std_{window}',
            f'sales_rolling_median_{window}'
        ])
    
    df[rolling_cols] = df[rolling_cols].fillna(0)
    
    logger.info(f"Created {len(rolling_cols)} rolling statistics features")
    return df

def create_store_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encodings for store Type and include Size."""
    
    logger.info("Creating store encodings")
    
    df = df.copy()
    
    # One-hot encode store Type
    if 'Type' in df.columns:
        type_dummies = pd.get_dummies(df['Type'], prefix='store_type')
        df = pd.concat([df, type_dummies], axis=1)
        
        # Drop original Type column
        df = df.drop('Type', axis=1)
        
        logger.info(f"Created store type encodings: {list(type_dummies.columns)}")
    
    # Keep Size column as is (already numeric)
    if 'Size' in df.columns:
        logger.info("Size column included in features")
    
    return df

def create_department_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Create department encodings."""
    
    logger.info("Creating department encodings")
    
    df = df.copy()
    
    # Create department size categories
    dept_sales = df.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False)
    
    # Categorize departments by sales volume
    dept_categories = pd.cut(dept_sales, bins=3, labels=['Low', 'Medium', 'High'])
    dept_cat_dict = dept_categories.to_dict()
    
    # Add department category to main dataframe
    df['dept_category'] = df['Dept'].map(dept_cat_dict)
    
    # One-hot encode department category
    dept_dummies = pd.get_dummies(df['dept_category'], prefix='dept_cat')
    df = pd.concat([df, dept_dummies], axis=1)
    
    # Drop original dept_category column
    df = df.drop('dept_category', axis=1)
    
    logger.info(f"Created department encodings: {list(dept_dummies.columns)}")
    return df

def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced holiday features."""
    
    logger.info("Creating enhanced holiday features")
    
    df = df.copy()
    
    # Create holiday week features
    if 'is_holiday_week' in df.columns:
        df['holiday_week_lag1'] = df.groupby(['Store', 'Dept'])['is_holiday_week'].shift(1)
        df['holiday_week_lag2'] = df.groupby(['Store', 'Dept'])['is_holiday_week'].shift(2)
        df['holiday_week_lag3'] = df.groupby(['Store', 'Dept'])['is_holiday_week'].shift(3)
        
        # Fill NaN values
        holiday_lag_cols = ['holiday_week_lag1', 'holiday_week_lag2', 'holiday_week_lag3']
        df[holiday_lag_cols] = df[holiday_lag_cols].fillna(False)
        
        logger.info(f"Created holiday lag features: {holiday_lag_cols}")
    
    # Create holiday interaction features
    if 'IsHoliday' in df.columns and 'is_federal_holiday' in df.columns:
        df['holiday_interaction'] = df['IsHoliday'] & df['is_federal_holiday']
        df['holiday_interaction'] = df['holiday_interaction'].astype(int)
        
        logger.info("Created holiday interaction feature")
    
    return df

def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create trend and seasonality features."""
    
    logger.info("Creating trend and seasonality features")
    
    df = df.copy()
    
    # Create trend features
    df['sales_trend'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
        lambda x: x.rolling(window=13, min_periods=1).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
        )
    )
    
    # Create seasonality features
    if 'month' in df.columns:
        # Monthly seasonality
        monthly_avg = df.groupby(['Store', 'Dept', 'month'])['Weekly_Sales'].transform('mean')
        df['monthly_seasonality'] = monthly_avg / df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform('mean')
        df['monthly_seasonality'] = df['monthly_seasonality'].fillna(1)
    
    if 'week' in df.columns:
        # Weekly seasonality
        weekly_avg = df.groupby(['Store', 'Dept', 'week'])['Weekly_Sales'].transform('mean')
        df['weekly_seasonality'] = weekly_avg / df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform('mean')
        df['weekly_seasonality'] = df['weekly_seasonality'].fillna(1)
    
    # Fill NaN values
    trend_cols = ['sales_trend', 'monthly_seasonality', 'weekly_seasonality']
    existing_trend_cols = [col for col in trend_cols if col in df.columns]
    df[existing_trend_cols] = df[existing_trend_cols].fillna(0)
    
    logger.info(f"Created trend and seasonality features: {existing_trend_cols}")
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between variables."""
    
    logger.info("Creating interaction features")
    
    df = df.copy()
    
    # Store size interactions
    if 'Size' in df.columns:
        df['size_temperature'] = df['Size'] * df['Temperature']
        df['size_fuel_price'] = df['Size'] * df['Fuel_Price']
        
        logger.info("Created store size interaction features")
    
    # MarkDown interactions
    if 'MarkDownSum' in df.columns:
        df['markdown_temperature'] = df['MarkDownSum'] * df['Temperature']
        df['markdown_holiday'] = df['MarkDownSum'] * df['IsHoliday'].astype(int)
        
        logger.info("Created MarkDown interaction features")
    
    # Holiday interactions
    if 'IsHoliday' in df.columns:
        df['holiday_temperature'] = df['IsHoliday'].astype(int) * df['Temperature']
        df['holiday_fuel_price'] = df['IsHoliday'].astype(int) * df['Fuel_Price']
        
        logger.info("Created holiday interaction features")
    
    # Fill NaN values in interaction features
    interaction_cols = [col for col in df.columns if any(x in col for x in ['size_', 'markdown_', 'holiday_'])]
    df[interaction_cols] = df[interaction_cols].fillna(0)
    
    return df

def prepare_features_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare all features for modeling."""
    
    logger.info("Preparing features for modeling")
    
    df = df.copy()
    
    # Create lag features
    df = create_lag_features_for_modeling(df)
    
    # Create rolling statistics
    df = create_rolling_statistics(df)
    
    # Create store encodings
    df = create_store_encodings(df)
    
    # Create department encodings
    df = create_department_encodings(df)
    
    # Create holiday features
    df = create_holiday_features(df)
    
    # Create trend features
    df = create_trend_features(df)
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Remove any remaining NaN values
    df = df.fillna(0)
    
    # Ensure all features are numeric
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric columns found: {list(non_numeric_cols)}")
        # Convert to numeric where possible
        for col in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                logger.warning(f"Could not convert column {col} to numeric")
    
    # Fill any new NaN values created during conversion
    df = df.fillna(0)
    
    logger.info(f"Feature preparation completed. Final shape: {df.shape}")
    logger.info(f"Total features: {len(df.columns)}")
    
    return df

def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Get list of feature columns for modeling."""
    
    if exclude_cols is None:
        exclude_cols = ['Store', 'Dept', 'Date', 'Weekly_Sales']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Identified {len(feature_cols)} feature columns")
    return feature_cols

def main(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to build features for modeling."""
    
    logger.info("Starting feature engineering pipeline")
    
    try:
        # Prepare all features
        featured_df = prepare_features_for_modeling(df)
        
        # Get feature columns
        feature_cols = get_feature_columns(featured_df)
        
        logger.info(f"Feature engineering completed. Features: {len(feature_cols)}")
        return featured_df
        
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # This module is typically called from other pipelines
    pass
