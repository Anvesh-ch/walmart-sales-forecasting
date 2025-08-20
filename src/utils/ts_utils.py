import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from datetime import datetime, timedelta
import holidays
from src.logging_utils import get_logger

logger = get_logger(__name__)

def parse_dates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Parse date column and ensure proper datetime format."""
    
    logger.info(f"Parsing dates in column: {date_col}")
    
    df = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Check for missing dates
    missing_dates = df[date_col].isnull().sum()
    if missing_dates > 0:
        logger.warning(f"Found {missing_dates} missing dates")
    
    logger.info(f"Date parsing completed. Range: {df[date_col].min()} to {df[date_col].max()}")
    return df

def get_holiday_flags(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add US federal holiday flags to dataframe."""
    
    logger.info("Adding US federal holiday flags")
    
    df = df.copy()
    
    # Get US federal holidays
    us_holidays = holidays.US()
    
    # Add holiday flags
    df['is_federal_holiday'] = df[date_col].apply(lambda x: x in us_holidays)
    df['holiday_name'] = df[date_col].apply(lambda x: us_holidays.get(x, ''))
    
    # Add holiday week flag (week containing holiday)
    df['is_holiday_week'] = df[date_col].dt.isocalendar().week.isin(
        df[df['is_federal_holiday']][date_col].dt.isocalendar().week
    )
    
    logger.info(f"Added holiday flags. Found {df['is_federal_holiday'].sum()} federal holidays")
    return df

def create_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Create time-based features from date column."""
    
    logger.info("Creating time-based features")
    
    df = df.copy()
    
    # Extract time components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week'] = df[date_col].dt.isocalendar().week
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['quarter'] = df[date_col].dt.quarter
    
    # Add seasonality features
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_month_start'] = df[date_col].dt.is_month_start
    df['is_month_end'] = df[date_col].dt.is_month_end
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end
    
    # Add cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    
    logger.info("Time features created successfully")
    return df

def get_forecast_horizon(df: pd.DataFrame, date_col: str = "Date", 
                        horizon_weeks: int = 13) -> pd.DataFrame:
    """Generate future dates for forecasting."""
    
    logger.info(f"Generating forecast horizon: {horizon_weeks} weeks")
    
    # Get the latest date in the data
    max_date = df[date_col].max()
    
    # Generate future dates
    future_dates = pd.date_range(
        start=max_date + timedelta(days=1),
        periods=horizon_weeks * 7,
        freq='D'
    )
    
    # Convert to weekly (end of week)
    future_weeks = future_dates.to_period('W').unique().to_timestamp()
    
    # Create future dataframe
    future_df = pd.DataFrame({date_col: future_weeks})
    
    # Add time features
    future_df = create_time_features(future_df, date_col)
    future_df = get_holiday_flags(future_df, date_col)
    
    logger.info(f"Generated {len(future_df)} future weeks for forecasting")
    return future_df

def create_rolling_windows(df: pd.DataFrame, date_col: str = "Date", 
                          windows: List[int] = [3, 4, 6, 8, 13]) -> pd.DataFrame:
    """Create rolling window features for time series."""
    
    logger.info(f"Creating rolling windows: {windows}")
    
    df = df.copy()
    
    # Ensure sorted by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create rolling features for each window
    for window in windows:
        if 'Weekly_Sales' in df.columns:
            df[f'sales_rolling_mean_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'sales_rolling_std_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
    
    logger.info("Rolling window features created successfully")
    return df

def create_lag_features(df: pd.DataFrame, date_col: str = "Date",
                       lags: List[int] = [1, 2, 3, 4, 6, 8, 13]) -> pd.DataFrame:
    """Create lag features for time series."""
    
    logger.info(f"Creating lag features: {lags}")
    
    df = df.copy()
    
    # Ensure sorted by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create lag features for each lag
    for lag in lags:
        if 'Weekly_Sales' in df.columns:
            df[f'sales_lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
    
    logger.info("Lag features created successfully")
    return df

def get_train_test_split_dates(df: pd.DataFrame, date_col: str = "Date",
                              test_size: float = 0.2) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get date boundaries for train-test split."""
    
    logger.info(f"Calculating train-test split dates (test_size: {test_size})")
    
    # Get unique dates
    unique_dates = df[date_col].unique()
    unique_dates = np.sort(unique_dates)
    
    # Calculate split index
    split_idx = int(len(unique_dates) * (1 - test_size))
    split_date = unique_dates[split_idx]
    
    train_end = unique_dates[split_idx - 1]
    test_start = split_date
    
    logger.info(f"Train end: {train_end}, Test start: {test_start}")
    return train_end, test_start

def validate_time_series_data(df: pd.DataFrame, date_col: str = "Date",
                             group_cols: List[str] = None) -> bool:
    """Validate time series data structure."""
    
    logger.info("Validating time series data structure")
    
    if group_cols is None:
        group_cols = ['Store', 'Dept']
    
    # Check for missing dates
    missing_dates = df[date_col].isnull().sum()
    if missing_dates > 0:
        logger.warning(f"Found {missing_dates} missing dates")
    
    # Check for duplicate dates within groups
    duplicates = df.groupby(group_cols + [date_col]).size()
    if (duplicates > 1).any():
        logger.warning("Found duplicate dates within groups")
    
    # Check date range consistency
    date_ranges = df.groupby(group_cols)[date_col].agg(['min', 'max'])
    inconsistent_ranges = date_ranges['max'] - date_ranges['min']
    if inconsistent_ranges.std() > timedelta(days=7):
        logger.warning("Inconsistent date ranges across groups")
    
    logger.info("Time series validation completed")
    return True
